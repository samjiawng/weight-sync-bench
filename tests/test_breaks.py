"""Adversarial cases.

Cases 1-3 are layout bugs. Each asserts the correctness invariant **fails**. If any
of them passes, the harness does not work and nothing else in this repo means
anything.

Case 4 is not a break test. At t=4 with n_kv_heads=2 the layout is unrepresentable,
so there is no mismatch to catch; it asserts explicit handling instead.

Every break here is **shape-preserving**: execution completes and the logits come
out wrong. That is the point. A break that raises a shape error is caught by torch
rather than by the invariant, so it would pass without exercising the harness at
all -- see `test_lower_value_dim0_shape_error` for the one case kept in that form,
and why it is worth less.

Breaks only run at t >= 2. At t=1 every one of them is the identity: chunk(1) is
the whole tensor, reversing a one-element rank assignment changes nothing, and a
1-way "shard" of a norm weight is the whole weight.

Observed deviations are recorded to tolerance/break_separation.json for the README
table. Recording is opt-in so a normal test run never writes to the working tree:

    RESHARD_RECORD_BREAKS=1 uv run pytest tests/test_breaks.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from helpers import CONFIGS, DEGREES, reference, tokens_for
from reshard_bench.reshard import split_params
from reshard_bench.sharded import InProcessCollective, ShardedModel
from reshard_bench.shardspec import (
    TOY,
    FusedPaired,
    HeadPartitioned,
    Replicated,
    Shard,
    ShardSpec,
    UnsupportedLayout,
    build_layout_table,
    unrepresentable_roles,
)
from reshard_bench.tolerance import ARTIFACT, load, load_threshold

THRESHOLD = load_threshold()

# A break must clear the threshold by this factor, not merely exceed it. The
# threshold is already 100x the measured floor, so this puts a real layout bug at
# >= 1e4 x the floor -- the "three to six orders of magnitude" separation the spec
# expects. A break that only just exceeds the threshold would be indistinguishable
# from a tolerance that was set slightly too tight.
MIN_SEPARATION = 100.0

# Degrees at which a break is meaningful. t=1 is excluded: see the module docstring.
BREAK_DEGREES = [
    (name, t) for name, degrees in DEGREES.items() for t in degrees if t > 1
]

_OBSERVED: list[dict[str, object]] = []


# --- machinery ----------------------------------------------------------------


def run_break(name: str, t: int, mutate) -> float:
    """Apply `mutate` to a correct shard set, run it, return max abs logit deviation."""
    config = CONFIGS[name]
    model = reference(name)
    tokens = tokens_for(name)

    full = model.full_params()
    params = split_params(full, build_layout_table(config, t))
    mutate(params, full, config, t)

    actual = ShardedModel(config, params, InProcessCollective(t))(tokens)
    return (actual - model(tokens)).abs().max().item()


def assert_invariant_violated(case: str, name: str, t: int, mutate) -> float:
    deviation = run_break(name, t, mutate)
    _OBSERVED.append(
        {
            "case": case,
            "config": name,
            "tp_degree": t,
            "deviation": float(f"{deviation:.4g}"),
            "x_threshold": int(deviation / THRESHOLD),
        }
    )
    assert deviation > THRESHOLD * MIN_SEPARATION, (
        f"{case} at {name}/t={t} deviated by {deviation:.3e}, which does not clear "
        f"the threshold {THRESHOLD:.3e} by {MIN_SEPARATION:g}x. Either the break is "
        "not biting or the harness has stopped detecting layout errors."
    )
    return deviation


def _for_role(params, role: str):
    suffix = f".{role}"
    return [name for name in params if name == role or name.endswith(suffix)]


# --- case 1: fused qkv sliced contiguously ------------------------------------


def break_qkv_contiguous(params, full, config, t) -> None:
    """Slice fused qkv contiguously on dim 0 instead of per head.

    Shape-preserving: a contiguous 1/t slice of [(n_heads + 2*n_kv_heads)*head_dim,
    d_model] has exactly the row count the correct per-head shard has. Every rank
    gets valid-looking weights that are the wrong rows -- rank 0 gets only Q rows and
    no K/V at all.
    """
    for name in _for_role(params, "qkv"):
        params[name] = [chunk.contiguous() for chunk in full[name].chunk(t, dim=0)]


@pytest.mark.parametrize("name,t", BREAK_DEGREES)
def test_case1_contiguous_qkv_slice_violates_invariant(name, t):
    assert_invariant_violated("1: contiguous qkv slice", name, t, break_qkv_contiguous)


# --- case 2: row-parallel tensor with the wrong rank assignment ---------------


def break_row_parallel_rank_swap(role: str):
    """Shard a row-parallel tensor on dim 1 correctly, then reverse the
    rank-to-slice assignment so rank 0 receives rank 1's columns.

    This is the shape-preserving form of break case 2. Every shard has the right
    shape and dtype, every matmul is valid, execution completes, and the row-parallel
    all-reduce sums products of mismatched column blocks. Only the logits are wrong.
    """

    def mutate(params, full, config, t) -> None:
        for name in _for_role(params, role):
            params[name] = list(reversed(params[name]))

    return mutate


@pytest.mark.parametrize("role", ["o_proj", "down"])
@pytest.mark.parametrize("name,t", BREAK_DEGREES)
def test_case2_row_parallel_rank_misassignment_violates_invariant(name, t, role):
    assert_invariant_violated(
        f"2: {role} rank misassignment", name, t, break_row_parallel_rank_swap(role)
    )


# --- case 3: RMSNorm weights sharded instead of replicated --------------------


def break_norm_sharded(params, full, config, t) -> None:
    """Shard the RMSNorm weights instead of replicating them.

    Shape-preserving AND information-preserving, matching case 2's discipline: rank i
    receives all d_model entries, rotated by i shard-widths, so the blocks sit at the
    wrong offsets and the ranks disagree with each other. Under correct replication
    every rank holds an identical full weight; that disagreement is the signature of
    sharding something that should be replicated.

    Nothing is zeroed, so the whole deviation is attributable to the misassignment
    rather than to destroyed information. The raw form -- handing each rank a
    [d_model/t] vector -- is a broadcast error in `rms_norm`, caught by torch rather
    than by the invariant, which is why the injection keeps full length.
    """
    for name in [n for n in params if n.endswith("_norm")]:
        weight = full[name]
        size = weight.numel() // t
        params[name] = [
            torch.roll(weight, shifts=rank * size, dims=0).contiguous()
            for rank in range(t)
        ]


@pytest.mark.parametrize("name,t", BREAK_DEGREES)
def test_case3_sharded_norm_weights_violate_invariant(name, t):
    assert_invariant_violated("3: permuted norm weights", name, t, break_norm_sharded)


# --- the lower-value shape-error variant --------------------------------------


@pytest.mark.parametrize("name,t", BREAK_DEGREES)
def test_lower_value_dim0_shape_error(name, t):
    """Sharding a row-parallel tensor on dim 0 raises instead of computing.

    Kept for completeness, but worth much less than case 2 above: torch catches this
    in the matmul, so the test would pass even if ShardedModel computed nothing at
    all and the correctness invariant were entirely broken. It demonstrates that the
    wrong dimension is caught somehow -- not that the harness detects layout errors.
    Case 2's rank-misassignment form is the one that actually exercises the invariant.
    """
    config = CONFIGS[name]
    full = reference(name).full_params()
    params = split_params(full, build_layout_table(config, t))
    for pname in _for_role(params, "o_proj"):
        params[pname] = [c.contiguous() for c in full[pname].chunk(t, dim=0)]

    with pytest.raises(RuntimeError, match="shapes cannot be multiplied"):
        ShardedModel(config, params, InProcessCollective(t))(tokens_for(name))


# --- separation summary --------------------------------------------------------


BREAKS = {
    "1: contiguous qkv slice": break_qkv_contiguous,
    "2: o_proj rank misassignment": break_row_parallel_rank_swap("o_proj"),
    "2: down rank misassignment": break_row_parallel_rank_swap("down"),
    "3: permuted norm weights": break_norm_sharded,
}


def test_every_break_clears_the_floor_by_orders_of_magnitude():
    """Aggregate view: the weakest break must still sit far above the largest
    correct-path deviation ever measured.

    Recomputes rather than reading `_OBSERVED`, so it does not depend on the
    parametrized cases having run first and holds up under `-k` selection.
    """
    floor = load()["max_deviation"]
    weakest = min(
        run_break(name, t, mutate)
        for mutate in BREAKS.values()
        for name, t in BREAK_DEGREES
    )
    assert weakest / floor > 1e5, (
        f"weakest break deviation {weakest:.3e} is only {weakest / floor:.0f}x the "
        f"measured floor {floor:.3e}; separation has collapsed"
    )


@pytest.fixture(scope="session", autouse=True)
def record_break_separation():
    """Write the README table's numbers, opt-in via RESHARD_RECORD_BREAKS=1."""
    yield
    if not os.environ.get("RESHARD_RECORD_BREAKS") or not _OBSERVED:
        return
    report = {
        "phase": "1a",
        "note": [
            "Observed max absolute logit deviation for each injected layout bug, "
            "against the phase 1a tolerance threshold. Every break here is "
            "shape-preserving: execution completes and only the numbers are wrong.",
            "Compare against tolerance/phase1a.json, whose max_deviation is the "
            "correct-path floor. Regenerate with: "
            "RESHARD_RECORD_BREAKS=1 uv run pytest tests/test_breaks.py",
        ],
        "threshold": THRESHOLD,
        "measured_floor": load()["max_deviation"],
        "min_separation_factor": MIN_SEPARATION,
        "breaks": sorted(
            _OBSERVED, key=lambda row: (row["case"], row["config"], row["tp_degree"])
        ),
    }
    path = Path(ARTIFACT).parent / "break_separation.json"
    path.write_text(json.dumps(report, indent=2) + "\n")


# --- case 4: explicit handling of an unrepresentable layout -------------------


def test_case4_tp4_raises_unsupported_layout():
    with pytest.raises(UnsupportedLayout, match="n_kv_heads"):
        build_layout_table(TOY, 4)


def test_case4_raise_lives_in_the_layout_type_not_the_reshard_path():
    """The decision must be discoverable from the layout data structure.

    Constructing the placement at an unsupported degree is enough to raise; no
    reshard call is required.
    """
    with pytest.raises(UnsupportedLayout, match="n_kv_heads"):
        ShardSpec(HeadPartitioned(n_heads=8, n_kv_heads=2, head_dim=32), group_size=4)


def test_case4_kv_replication_is_not_silently_substituted():
    """Guards the failure mode the raise exists to prevent: a third option that
    quietly pads or reinterprets the shard so tp=4 appears representable."""
    with pytest.raises(UnsupportedLayout):
        build_layout_table(TOY, 4)
    for degree in (3, 5, 8):
        with pytest.raises(UnsupportedLayout):
            build_layout_table(TOY, degree)


def test_only_qkv_is_unrepresentable_at_tp4():
    """The exclusion is per-parameter, not per-TP-degree. Everything that is not
    GQA-dependent must still be exercised at a 4-way split."""
    assert unrepresentable_roles(TOY, 4) == frozenset({"qkv"})
    assert unrepresentable_roles(TOY, 2) == frozenset()
    assert unrepresentable_roles(TOY, 1) == frozenset()


def test_tp4_table_without_qkv_covers_every_other_parameter():
    table = build_layout_table(TOY, 4, omit=unrepresentable_roles(TOY, 4))

    assert table.omitted == frozenset({"qkv"})
    assert not any(name.endswith(".qkv") for name in table)
    for role in ("o_proj", "down", "gate_up", "attn_norm", "ffn_norm"):
        assert table[f"layers.0.{role}"].group_size == 4
    for name in ("embed", "lm_head", "final_norm"):
        assert table[name].group_size == 4


# --- supported degrees --------------------------------------------------------


@pytest.mark.parametrize("tp", [1, 2])
def test_supported_degrees_build_expected_placements(tp):
    table = build_layout_table(TOY, tp)

    assert table["layers.0.qkv"].placement == HeadPartitioned(8, 2, 32)
    assert table["layers.0.gate_up"].placement == FusedPaired((0, 704), (704, 1408))
    assert table["layers.0.o_proj"].placement == Shard(1)
    assert table["layers.0.down"].placement == Shard(1)
    assert table["layers.0.attn_norm"].placement == Replicated()
    assert table["layers.0.ffn_norm"].placement == Replicated()
    assert table["embed"].placement == Shard(0)
    assert table["lm_head"].placement == Shard(0)
    assert table["final_norm"].placement == Replicated()
    assert all(table[name].group_size == tp for name in table)


def test_layout_table_covers_every_parameter():
    table = build_layout_table(TOY, 2)
    expected = {"embed", "lm_head", "final_norm"} | {
        f"layers.{i}.{p}"
        for i in range(TOY.n_layers)
        for p in ("qkv", "o_proj", "gate_up", "down", "attn_norm", "ffn_norm")
    }
    assert set(table) == expected


def test_row_parallel_tensors_are_not_dim0():
    """The correct table must not already be doing what break case 2 injects."""
    table = build_layout_table(TOY, 2)
    assert table["layers.0.o_proj"].placement != Shard(0)
    assert table["layers.0.down"].placement != Shard(0)
