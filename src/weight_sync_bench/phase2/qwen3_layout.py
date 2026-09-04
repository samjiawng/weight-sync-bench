"""Qwen3-0.6B `LayoutTable` at TP in {1, 2, 4}. SPEC.md 2b's deliverable.

UNVERIFIED against a running vLLM -- see `geometry.py` and
`param_layout_inspection.py` for what has and hasn't been checked so far.
This module builds the table and the two prediction checks the box run
verifies; it does not itself touch vLLM or a GPU.

--------------------------------------------------------------------------
REUSES PHASE 1's TYPES UNCHANGED
--------------------------------------------------------------------------
`HeadPartitioned`, `FusedPaired`, `Replicated`, `Shard`, `ShardSpec`,
`LayoutTable`, and `reshard.split_tensor` are all imported from `shardspec.py`
/ `reshard.py` as-is -- SPEC.md phase 2's "What carries over." Nothing here
edits phase 1. The only new code is the mapping from Qwen3's real parameter
names to those placements, driven by `CheckpointGeometry` instead of the
toy's `ModelConfig`.

--------------------------------------------------------------------------
THE FUSION-ORDER ASSUMPTION IS CHECKED, NOT ASSUMED
--------------------------------------------------------------------------
`HeadPartitioned`/`reshard._head_rows` hardcode a q-then-k-then-v row order
in Python control flow; `FusedPaired` hardcodes gate-then-up. Building a
`HeadPartitioned`/`FusedPaired` from `QWEN3_0_6B` without checking that its
`fusion` field actually says `("q_proj","k_proj","v_proj")` /
`("gate_proj","up_proj")` would be exactly the silent assumption
`geometry.py`'s docstring warns against -- so `_placements` calls
`_check_fusion_matches_placement_assumptions` first, every time, and raises
`UnsupportedLayout` (not a warning) if a geometry's stated fusion order ever
disagrees with what these placements are hardcoded to assume. For
`QWEN3_0_6B` this currently passes, because both v0.28.0's `QKVParallelLinear`
and `MergedColumnParallelLinear` loaders were read and confirmed to fuse in
that order (`param_layout_inspection.py`'s SOURCE READING section, points 1
and 2) -- but the check runs unconditionally, not only for checkpoints where
someone remembered it might matter.

--------------------------------------------------------------------------
TWO SEPARATE CHECKS -- deliberately not one
--------------------------------------------------------------------------
`check_shape_predictions` and `check_content_predictions` answer different
questions and can fail independently: a `LayoutTable` could correctly
predict every parameter's SHAPE (dimensions match what vLLM reports) while
still being wrong about which BYTES go where (e.g. a transposed q/k
assignment that happens to preserve shape) -- shape agreement alone would
not catch that. Conversely a shape MISMATCH says the geometry itself is
wrong (a real number, like `head_dim`, is off) before content is even worth
checking.

- `check_shape_predictions(geo, tp, per_rank_modules)`: for every parameter
  `param_layout_inspection.report_parameters` observed at TP degree `tp`,
  compares vLLM's REPORTED shape against `predicted_shapes(geo, tp)`. Exact
  equality; a completely absent predicted-but-not-observed name is reported
  too (`missing_from_inspection`), not silently skipped.

- `check_content_predictions(geo, tp_degree, tp1_full_tensors,
  tp_rank_tensors)`: for `qkv_proj` and `gate_up_proj` (layer 0,
  representative -- every layer shares the same geometry and placement),
  calls THIS REPO'S OWN `reshard.split_tensor` on the REAL TP=1 tensor with
  the predicted `ShardSpec` at `tp_degree` (2 or 4 -- nothing about this
  function is specific to 2; `tp1_full_tensors` is always TP=1's, since
  that is always the reference every other degree reshards against), and
  compares the result to the REAL `tp_degree`-rank-local tensor with
  `torch.equal`, once per rank. This is not a parallel, hand-derived
  formula re-checked against inspection's hashes (an earlier revision of
  `param_layout_inspection.py` did that, with `_qkv_head_ranges`/
  `_gate_up_ranges`/`slice_hashes` computing per-head ranges independently
  and comparing digests) -- it is literally the code phase 2 would use to
  reshard real weights, run once here as a check. SPEC.md's phase 2 "carries
  over" line ("the phase 1 harness is what certifies phase 2's
  measurements") is why this matters: proving `HeadPartitioned`'s row math
  agrees with vLLM's real loader on real weights is a stronger, more
  directly useful fact than proving a second implementation of the same
  arithmetic agrees with a third.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..reshard import split_tensor
from ..shardspec import (
    FusedPaired,
    HeadPartitioned,
    LayoutTable,
    Placement,
    Replicated,
    Shard,
    ShardSpec,
    UnsupportedLayout,
)
from .geometry import CheckpointGeometry

if TYPE_CHECKING:
    import torch

GLOBAL_ROLES = ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight")
LAYER_ROLES = (
    "self_attn.qkv_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "mlp.gate_up_proj.weight",
    "mlp.down_proj.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
)


def _check_fusion_matches_placement_assumptions(geo: CheckpointGeometry) -> None:
    expected_qkv = ("q_proj", "k_proj", "v_proj")
    actual_qkv = geo.fusion.get("qkv_proj")
    if actual_qkv != expected_qkv:
        raise UnsupportedLayout(
            f"HeadPartitioned/reshard._head_rows assume qkv fuses as "
            f"{expected_qkv!r}, but this geometry's fusion mapping says "
            f"{actual_qkv!r}. HeadPartitioned cannot express this "
            "checkpoint's actual fusion order as-is -- extend the "
            "placement (SPEC.md 2b: 'say so explicitly ... rather than "
            "working around it in the resharder'), do not silently reorder "
            "rows here."
        )

    expected_gu = ("gate_proj", "up_proj")
    actual_gu = geo.fusion.get("gate_up_proj")
    if actual_gu != expected_gu:
        raise UnsupportedLayout(
            f"FusedPaired assumes gate_up fuses as {expected_gu!r}, but "
            f"this geometry's fusion mapping says {actual_gu!r}."
        )


def _placements(geo: CheckpointGeometry) -> dict[str, Placement]:
    """Correct placement per real parameter role. Single source of truth
    for `build_qwen3_layout_table` and `check_content_predictions`, mirroring
    shardspec.py's `_placements(config)` for the toy model."""
    _check_fusion_matches_placement_assumptions(geo)
    return {
        "model.embed_tokens.weight": Shard(0),
        "lm_head.weight": Shard(0),
        "model.norm.weight": Replicated(),
        "self_attn.qkv_proj.weight": HeadPartitioned(geo.n_heads, geo.n_kv_heads, geo.head_dim),
        "self_attn.o_proj.weight": Shard(1),
        "self_attn.q_norm.weight": Replicated(),
        "self_attn.k_norm.weight": Replicated(),
        "mlp.gate_up_proj.weight": FusedPaired((0, geo.ffn), (geo.ffn, 2 * geo.ffn)),
        "mlp.down_proj.weight": Shard(1),
        "input_layernorm.weight": Replicated(),
        "post_attention_layernorm.weight": Replicated(),
    }


def build_qwen3_layout_table(geo: CheckpointGeometry, tp_degree: int) -> LayoutTable:
    """Real per-(layer, role) LayoutTable plus the three global roles -- same
    expansion `shardspec.build_layout_table` uses for the toy model
    (`layers.{i}.{role}` there; real vLLM parameter names here,
    `model.layers.{i}.{role}`)."""
    placements = _placements(geo)
    specs: dict[str, ShardSpec] = {
        role: ShardSpec(placements[role], tp_degree) for role in GLOBAL_ROLES
    }
    for i in range(geo.n_layers):
        for role in LAYER_ROLES:
            specs[f"model.layers.{i}.{role}"] = ShardSpec(placements[role], tp_degree)
    return LayoutTable(tp_degree=tp_degree, specs=specs)


def predicted_shapes(geo: CheckpointGeometry, tp_degree: int) -> dict[str, tuple[int, ...]]:
    """Every real parameter name's predicted per-rank shape at `tp_degree`.
    Shapes are identical across layers by construction (same geometry, same
    placement, every layer) -- expanding to all 28 lets this be checked
    against every layer inspection reports, not just layer 0."""
    layer_shapes = geo.fused_shapes(tp_degree)
    shapes: dict[str, tuple[int, ...]] = dict(geo.global_shapes(tp_degree))
    for i in range(geo.n_layers):
        for role, shape in layer_shapes.items():
            shapes[f"model.layers.{i}.{role}"] = shape
    return shapes


def check_shape_predictions(
    geo: CheckpointGeometry, tp_degree: int, per_rank_modules: list[dict[str, Any]]
) -> dict[str, Any]:
    """Check 1: shape prediction. Every parameter name inspection reported
    at `tp_degree`, exact-equality against `predicted_shapes`. A module with
    no `.weight` (none expected here) is skipped rather than counted as a
    match or a mismatch."""
    predicted = predicted_shapes(geo, tp_degree)
    mismatches: list[dict[str, Any]] = []
    seen: set[str] = set()

    for rank_report in per_rank_modules:
        rank = rank_report["rank"]
        for m in rank_report["modules"]:
            if m["weight"] is None:
                continue
            param_name = m["name"] + ".weight"
            if param_name not in predicted:
                mismatches.append(
                    {
                        "rank": rank,
                        "param": param_name,
                        "predicted": None,
                        "actual": list(m["weight"]["shape"]),
                        "reason": "no prediction for this parameter name",
                    }
                )
                continue
            seen.add(param_name)
            actual_shape = tuple(m["weight"]["shape"])
            expected_shape = predicted[param_name]
            if actual_shape != expected_shape:
                mismatches.append(
                    {
                        "rank": rank,
                        "param": param_name,
                        "predicted": list(expected_shape),
                        "actual": list(actual_shape),
                        "reason": "shape mismatch",
                    }
                )

    missing = sorted(set(predicted) - seen)
    return {
        "tp_degree": tp_degree,
        "predicted_total": len(predicted),
        "checked": len(seen),
        "shape_predictions_match": not mismatches and not missing,
        "mismatches": mismatches,
        "missing_from_inspection": missing,
    }


def check_content_predictions(
    geo: CheckpointGeometry,
    tp_degree: int,
    tp1_full_tensors: dict[str, "torch.Tensor"],
    tp_rank_tensors: dict[str, list["torch.Tensor"]],
) -> dict[str, Any]:
    """Check 2: content prediction. reshard.split_tensor(TP=1's real tensor,
    the predicted ShardSpec at `tp_degree`, rank) vs the real rank-local
    tensor at `tp_degree`, torch.equal, for qkv_proj and gate_up_proj
    (layer 0). `tp_degree` is whichever non-1 degree is being checked (2 or
    4) -- TP=1 is always the reference (`tp1_full_tensors`), never the
    thing being verified, since every other degree is defined as a reshard
    of it."""
    import torch

    placements = _placements(geo)
    results: dict[str, list[dict[str, Any]]] = {}

    for role, key in (
        ("self_attn.qkv_proj.weight", "qkv_proj"),
        ("mlp.gate_up_proj.weight", "gate_up_proj"),
    ):
        full = tp1_full_tensors[key]
        spec = ShardSpec(placements[role], tp_degree)
        per_rank = []
        for rank, actual in enumerate(tp_rank_tensors[key]):
            predicted = split_tensor(full, spec, rank)
            shapes_match = predicted.shape == actual.shape
            equal = shapes_match and torch.equal(predicted, actual)
            max_abs_diff = (
                (predicted.float() - actual.float()).abs().max().item()
                if shapes_match
                else None
            )
            per_rank.append(
                {
                    "rank": rank,
                    "predicted_shape": list(predicted.shape),
                    "actual_shape": list(actual.shape),
                    "torch_equal": equal,
                    "max_abs_diff": max_abs_diff,
                }
            )
        results[key] = per_rank

    return results
