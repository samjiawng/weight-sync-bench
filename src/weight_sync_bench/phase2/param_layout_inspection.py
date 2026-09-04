"""Phase 2b step 0 (SPEC.md "2b. Real layout tables"): inspect what vLLM
actually holds in memory for Qwen3-0.6B at TP=1 and TP=2, before any
`LayoutTable` is written. Inspection, not design -- this module reports; it
does not decide what `HeadPartitioned` / `FusedPaired` / `Replicate()` should
look like for Qwen3.

--------------------------------------------------------------------------
THIS WAS WRITTEN WITHOUT A GPU OR VLLM. TREAT IT AS UNVERIFIED UNTIL RUN.
--------------------------------------------------------------------------
The machine this module was written on has no CUDA device and no `vllm`
install (`import vllm` raises `ModuleNotFoundError` there), so nothing in
this module has been executed. Everything below the SOURCE READING heading
is read from `github.com/vllm-project/vllm` at tag `v0.28.0`
(`vllm/model_executor/layers/linear.py`, `vllm/model_executor/models/qwen3.py`,
`vllm/model_executor/layers/layernorm.py`) and from `Qwen/Qwen3-0.6B`'s
published `config.json`, not from a running process. `collective_logits.py`'s
own docstring documents five separate cases, in this exact repo, where
reading vLLM source to a plausible-looking stopping point produced a
confident and wrong conclusion, caught only by running on the box. Nothing
here is exempt from that risk. This module's whole job is to turn the
predictions below into it-actually-ran findings; run it (`--worker` mode
needs a GPU) before trusting any claim past this point in a report.

--------------------------------------------------------------------------
WHAT TO RUN
--------------------------------------------------------------------------
TP=1 needs one GPU, TP=2 needs two. Both legs need a real `vllm` install
(see `bf16_floor.py`'s docstring for the pinned version / separate-env
install command); there is no CPU path exercised or intended here, since the
question under investigation is what vLLM's CUDA weight loader does, not
what a CPU-only build would do instead -- a CPU run would not be inspecting
the thing this module exists to inspect.

    uv run python -m weight_sync_bench.phase2.param_layout_inspection

runs both legs (each as its own subprocess, one GPU-process per TP degree,
same pattern as `bf16_floor.measure_differential_floor`) and writes
`tolerance/phase2b_param_layout.json`. `--tp {1,2} --worker` runs one leg
standalone (used internally by the subprocess spawn; exposed for debugging a
single leg without paying for both).

--------------------------------------------------------------------------
SOURCE READING: WHAT EACH PARAMETER IS PREDICTED TO LOOK LIKE
--------------------------------------------------------------------------
Qwen3-0.6B config.json (fetched from the model card, not assumed):
`hidden_size=1024`, `num_attention_heads=16`, `num_key_value_heads=8`,
`head_dim=128` (NOT `hidden_size // num_attention_heads == 64` -- Qwen3 sets
`head_dim` explicitly, so `q_size = 16*128 = 2048`, not 1024), `n_layers=28`,
`intermediate_size=3072`, `vocab_size=151936`, `tie_word_embeddings=True`.
Matches `bf16_floor.QWEN3_0_6B` exactly; reused from there rather than
re-declared.

1. **`qkv_proj` fusion order and per-rank construction**
   (`vllm/model_executor/models/qwen3.py:105-112`, `Qwen3Attention.__init__`
   constructs one `QKVParallelLinear`; `linear.py:1072-1163`,
   `QKVParallelLinear._load_fused_module_from_checkpoint` /
   `weight_loader_v2`/`weight_loader`).

   The checkpoint stores `q_proj`/`k_proj`/`v_proj` as three separate
   tensors (SPEC.md 2b, already confirmed against the checkpoint's
   safetensors header). `_load_fused_module_from_checkpoint` narrows each
   loaded checkpoint tensor at `shard_offset`/`shard_size` computed from
   `(total_num_heads, total_num_kv_heads, head_size)` -- q rows
   `[0, total_num_heads*head_size)`, k `[that, +total_num_kv_heads*head_size)`,
   v after that -- and dispatches each narrowed piece to `weight_loader`
   with `loaded_shard_id in {"q","k","v"}`. That per-shard call (lines
   1234-1274) narrows the LOCAL param buffer's OUTPUT dim to
   `[q_size_local)`, `[q_size_local, q_size_local+kv_size_local)`,
   `[+kv_size_local, +2*kv_size_local)` for q/k/v respectively -- i.e. **the
   fusion order in vLLM's in-memory buffer is q-then-k-then-v, at every TP
   degree**, matching the checkpoint's own q/k/v naming order (not some
   Megatron-interleaved q0k0v0q1k1v1... order -- nothing in this loader
   interleaves per head across q/k/v).

   For the head selection within each of q/k/v: `start_idx = shard_rank *
   shard_size` where `shard_rank = tp_rank` for q, and `shard_rank = tp_rank
   // num_kv_head_replicas` for k/v (`num_kv_head_replicas =
   tp_size // total_num_kv_heads` when `tp_size >= total_num_kv_heads`, else
   1). **For Qwen3-0.6B, `total_num_kv_heads=8` divides both TP=1 and TP=2
   evenly (and TP=4: 8/4=2), so `num_kv_head_replicas=1` at every TP degree
   this repo runs -- the GQA-replication branch (`shard_rank = tp_rank //
   num_kv_head_replicas` collapsing multiple ranks onto the same KV shard)
   never actually triggers for real Qwen3-0.6B at TP in {1,2,4}.** It would
   first trigger at TP=8 (`8 // 16`... concretely TP > 8). This is a
   materially different situation from phase 1's `TOY` config, which was
   deliberately built so `n_kv_heads=2` breaks at TP=4 (SPEC.md "GQA at
   TP=4"); real Qwen3-0.6B's `n_kv_heads=8` does not hit that wall until a
   TP degree this repo has no plan to run. **Prediction: `HeadPartitioned`'s
   `UnsupportedLayout` raise is real and correctly motivated by phase 1's
   toy config, but is not something 2b's actual `LayoutTable` for Qwen3 at
   TP in {1,2,4} will ever need to raise** -- worth stating explicitly
   rather than leaving as an unstated assumption, since it changes what "2b
   validates the invariant at TP in {1,2,4}" actually has to cover.

   Each of q/k/v's contiguous-in-the-source-tensor head range is what
   `shard_rank * shard_size` selects -- rank `r`'s q shard is checkpoint
   q_proj rows `[r*q_local, (r+1)*q_local)`, a CONTIGUOUS range of that one
   source tensor, and likewise for k/v against their own source tensors.
   **Prediction, the crux the task asked about:** relative to the checkpoint
   (three separate tensors), each rank's shard of each of q/k/v is
   contiguous. Relative to vLLM's OWN fused representation -- which is
   exactly what TP=1's `qkv_proj.weight` already materializes, a single
   `[q+k+v, hidden]` tensor with q rows first, k next, v last -- rank r's
   TP=2 shard is q's rows `[r*q_local, (r+1)*q_local)` PLUS k's rows
   `[r*kv_local, (r+1)*kv_local)` (offset by `q` in the fused tensor) PLUS
   v's rows similarly (offset by `q+k`): three separately-contiguous,
   non-adjacent row ranges concatenated into one local buffer. **This is
   non-contiguous relative to the fused tensor, in exactly the structured,
   per-head-block way `HeadPartitioned` was built to describe** -- it is
   NOT the "vLLM builds the fused tensor per rank from pieces in some
   arrangement `HeadPartitioned` can't see" failure mode the task worried
   about. The fusion happens per-rank (each rank's local q/k/v pieces are
   loaded and concatenated independently -- there is never a full `[q+k+v,
   hidden]` buffer materialized anywhere at TP=2, unlike at TP=1), but the
   SHAPE of that per-rank fusion is exactly "select this rank's contiguous
   head-block from each of q, k, v, then concatenate in q/k/v order," which
   is precisely what `HeadPartitioned` + a q/k/v-boundary-aware fused
   layout already models. `slice_hashes` below exists to turn this
   paragraph from a source-reading claim into a measured one: it hashes
   each individual head's row-block on both the TP=1 buffer and each TP=2
   rank's local buffer, so `_compare` can assert per-head bit-identity
   across TP degrees without ever transporting a multi-MB tensor over
   `collective_rpc` (contrast `collective_logits.py`'s Q7 byte-buffer
   workaround, needed there only because a full logits tensor had to cross
   the wire -- a 32-byte hex digest needs no such handling).

2. **`gate_up_proj` fusion** (`linear.py:639-833`,
   `MergedColumnParallelLinear`). Same shape of finding: checkpoint stores
   `gate_proj`/`up_proj` separately (SPEC.md 2b); the loader narrows each
   into the local buffer's output dim at `[0, ffn_local)` (gate) and
   `[ffn_local, 2*ffn_local)` (up), `ffn_local = intermediate_size //
   tp_size` -- gate-then-up order, matching phase 1's `FusedPaired`
   assumption. Relative to the TP=1 fused tensor, each TP=2 rank's shard is
   the gate-half paired with the up-half at the SAME rank index, two
   disjoint contiguous ranges -- exactly `FusedPaired`, not a slice.

3. **QK-norm** (`qwen3.py:150-151`, `q_norm = RMSNorm(head_dim, ...)`;
   `layernorm.py:37-65`, `RMSNorm.__init__` sets `self.weight =
   nn.Parameter(torch.ones(hidden_size, ...))` with no `weight_loader`
   override anywhere on the class, and `qwen3.py` wires no TP-aware
   attribute onto it either -- contrast `QKVParallelLinear`, whose
   `weight_loader` reads `param.output_dim`, an attribute the linear layers'
   parameter classes set but plain `nn.Parameter` never does). No
   `weight_loader` on a parameter means vLLM's generic loader falls back to
   `default_weight_loader`, a full `param.data.copy_(loaded_weight)` with a
   shape assert -- no narrowing, no `tp_rank`-dependent offset. **Prediction:
   `q_norm.weight` and `k_norm.weight` are `[128]`, IDENTICAL bytes on every
   rank, at every TP degree** -- genuinely `Replicate()`, not merely
   "effectively" so. `report_parameters` checks `hasattr(weight,
   "output_dim")` / `hasattr(weight, "input_dim")` on every parameter
   (present on the parallel-linear layers' weights, predicted absent on
   every `RMSNorm.weight`) as the direct, structural version of this claim,
   and `slice_hashes` on `q_norm.weight`/`k_norm.weight` with a single
   whole-tensor range at TP=1 vs. each TP=2 rank turns "identical bytes on
   every rank" into a checked hash-equality rather than a re-assertion of
   the same source-reading claim.

   **On the open question in SPEC.md 2b** ("does QK-norm need a new
   placement, or degenerate `HeadPartitioned`, given it's replicated within
   a head-sharded computation"): if this prediction holds, `q_norm`/`k_norm`
   need nothing new. The resharder (`reshard.py`) dispatches purely on
   `Placement` and moves BYTES between TP degrees; a `Replicate()` tensor
   whose content never depends on TP degree (same `[128]` weight vector
   regardless of how many heads a rank owns) reshards as a no-op between any
   two degrees -- gather-then-rescatter of a value already identical
   everywhere. The "applied per head, same vector reused across heads"
   fact lives entirely in `Qwen3Attention.forward` (`qwen3.py:161-165`,
   reshape-then-normalize over the head axis), which is CONSUMER code, not
   layout data -- the resharder never needs to know a `Replicate()` tensor
   is "head-associated" because nothing about moving its bytes between TP
   degrees changes when the number of heads per rank changes. This narrows
   the open question to: is there ever a `LayoutTable` consumer other than
   the resharder that would need "this replicated tensor is conceptually
   per-head" recorded explicitly? Nothing in phase 1's `ShardSpec`/
   `LayoutTable` reads placements for anything except resharding, so
   the answer, ON CURRENT EVIDENCE, is no new placement needed -- but this
   is exactly the kind of claim this module exists to check before writing
   it into a `LayoutTable`, not to assert from source alone.

4. **`o_proj` / `down_proj`** (`RowParallelLinear`, standard phase-1-shaped
   `Shard(1)` / row-parallel: each rank narrows the INPUT dim contiguously,
   `[r*shard, (r+1)*shard)`, no fusion, no q/k/v-style reordering). Included
   in `report_parameters`'s module walk for completeness but not a focus of
   `slice_hashes` -- there is no fused-tensor non-contiguity question here,
   phase 1 already covers this shape (`HeadPartitioned` on the matching
   input columns), and re-deriving it from Qwen3 specifically would not be
   new information.

5. **`embed_tokens` / `lm_head`** (`VocabParallelEmbedding` /
   `ParallelLMHead`, `qwen3.py:298-307`). `tie_word_embeddings=True` for
   Qwen3-0.6B, so `lm_head` is `.tie_weights(embed_tokens)` and the
   checkpoint's `lm_head.*` keys are skipped entirely
   (`qwen3.py:340`) -- **prediction: only one vocab-parallel weight tensor
   exists in memory per rank, referenced by both names**, standard
   vocab-parallel `Shard(0)`, contiguous per rank, no phase-1 analogue
   needed beyond what phase 1 already has (`embed`/`lm_head` are already
   `Shard(0)` there).

--------------------------------------------------------------------------
DESIGN
--------------------------------------------------------------------------
`ParamLayoutInspectionWorkerExtension` -- same `worker_extension_cls`
mechanism `collective_logits.py` established (see that module's Q2 for why
a plain function can't cross `collective_rpc` and what the supported
mechanism is; not re-derived here).

- `report_parameters(self) -> dict`: walks `model.named_modules()`, and for
  every module that is one of `QKVParallelLinear`, `MergedColumnParallelLinear`,
  `RowParallelLinear`, `VocabParallelEmbedding`/`ParallelLMHead`, or `RMSNorm`,
  records its class name, the loader-relevant attributes vLLM's own
  `weight_loader` reads (`output_sizes`, `num_heads`, `num_kv_heads`,
  `num_kv_head_replicas`, `tp_rank`, `tp_size`, ...), and the actual
  `.weight` tensor's shape/dtype/`is_contiguous()`/stride/`hasattr(...,
  "output_dim"/"input_dim")`. Plain dicts of `str`/`int`/`bool`/`list` --
  msgpack-native, no Q7-style byte-buffer handling needed anywhere in this
  module (contrast `collective_logits.py`, which had to move a full logits
  tensor across `collective_rpc` and therefore did).

- `slice_hashes(self, param_name, dim, ranges) -> list[dict]`: given a
  dotted parameter path (e.g. `"model.layers.0.self_attn.qkv_proj.weight"`)
  resolved via `getattr` chains, and a list of `(start, size)` ranges along
  `dim`, returns one `{"start", "size", "sha256", "narrow_was_contiguous"}`
  dict per range. Casts to float32 before hashing (bf16 has no reliable
  native numpy dtype across environments; casting both sides of every
  comparison identically preserves equality since bf16->fp32 is lossless
  and deterministic -- this hashes "the value," not "the checkpoint's exact
  bytes," which is what the contiguity question actually needs).
  `narrow_was_contiguous` records whether the slice was already a
  contiguous view before the forced `.contiguous()` call -- a direct
  per-range answer to "was this rank-local range already contiguous in the
  underlying storage," independent of the cross-TP hash comparison.

- `_run_worker`: loads one TP degree, calls `report_parameters`, hashes
  q_norm/k_norm (`slice_hashes`), pulls the real layer-0 `qkv_proj`/
  `gate_up_proj` tensors (`raw_param_bytes`), checks rank0-vs-rank1 identity
  at TP=2, and dumps a JSON file (facts + hashes) plus a sidecar `.pt` file
  (the real tensors -- too large and not msgpack-native to put in the JSON,
  and not needed there since the orchestrator loads `.pt` files directly
  off the shared filesystem, no `collective_rpc` byte-buffer trick needed
  for a same-machine subprocess handoff).

- `inspect_layouts` / `main`: spawns one subprocess per TP degree (mirrors
  `bf16_floor._spawn_worker` exactly -- isolates CUDA context between
  successive `LLM()` constructions), loads both JSON and `.pt` files, then
  runs `qwen3_layout.py`'s two prediction checks -- `check_shape_predictions`
  (every reported shape vs. `CheckpointGeometry`-derived predictions, exact
  equality) and `check_content_predictions` (`reshard.split_tensor` on the
  real TP=1 tensor vs. the real TP=2 rank tensor, `torch.equal`) -- plus the
  q_norm/k_norm replication-identity check, and writes
  `tolerance/phase2b_param_layout.json`. Prints the per-parameter report
  described in the task (name, shape per rank, fused-or-not and fusion
  order) plus both prediction checks and the QK-norm replication finding.
  An earlier revision of this module computed its own q/k/v and gate/up
  head-block ranges by hand (`_qkv_head_ranges`/`_gate_up_ranges`) and
  compared per-head hashes across TP degrees -- a second, independently
  coded implementation of the same row arithmetic `HeadPartitioned`/
  `reshard.py` already encode. Removed in favor of calling
  `reshard.split_tensor` itself once `qwen3_layout.py` existed to provide
  it: a mismatch between `split_tensor`'s prediction and vLLM's real
  tensor is a direct finding about this repo's own resharder, not about
  whether two independently-written formulas happen to agree.

`main()`'s no-arg path does NOT write a `LayoutTable` and does not touch
`shardspec.py` -- SPEC.md 2b's instruction to inspect before designing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .bf16_floor import MODEL_ID, QWEN3_0_6B

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = REPO_ROOT / "tolerance" / "phase2b_param_layout.json"

WORKER_EXTENSION_QUALNAME = (
    "weight_sync_bench.phase2.param_layout_inspection.ParamLayoutInspectionWorkerExtension"
)

# --------------------------------------------------------------------------- #
# Worker extension. Injected as a base class of Worker by vLLM itself
# (worker_base.py:265-291, see collective_logits.py's Q2) when
# `worker_extension_cls` names this class's dotted path.
# --------------------------------------------------------------------------- #


class ParamLayoutInspectionWorkerExtension:
    """Mixed into `Worker` via `worker_extension_cls`. `self` in every method
    below is the real Worker instance -- `self.model_runner`, `self.rank`,
    `self.parallel_config` are all real attributes set by vLLM's own
    `WorkerBase`/`Worker.__init__`, not stubs this module defines.
    """

    def report_parameters(self) -> dict[str, Any]:
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.model_executor.layers.linear import (
            MergedColumnParallelLinear,
            QKVParallelLinear,
            RowParallelLinear,
        )
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
            VocabParallelEmbedding,
        )

        model = self.model_runner.model
        modules: list[dict[str, Any]] = []

        for name, module in model.named_modules():
            kind: str | None = None
            extra: dict[str, Any] = {}

            if isinstance(module, QKVParallelLinear):
                kind = "QKVParallelLinear"
                extra = {
                    "output_sizes": list(module.output_sizes),
                    "num_heads": module.num_heads,
                    "num_kv_heads": module.num_kv_heads,
                    "num_kv_head_replicas": module.num_kv_head_replicas,
                    "head_size": module.head_size,
                    "total_num_heads": module.total_num_heads,
                    "total_num_kv_heads": module.total_num_kv_heads,
                    "tp_rank": module.tp_rank,
                    "tp_size": module.tp_size,
                }
            elif isinstance(module, MergedColumnParallelLinear):
                kind = "MergedColumnParallelLinear"
                extra = {
                    "output_sizes": list(module.output_sizes),
                    "tp_rank": module.tp_rank,
                    "tp_size": module.tp_size,
                }
            elif isinstance(module, RowParallelLinear):
                kind = "RowParallelLinear"
                extra = {
                    "tp_rank": getattr(module, "tp_rank", None),
                    "tp_size": getattr(module, "tp_size", None),
                }
            elif isinstance(module, ParallelLMHead):
                kind = "ParallelLMHead"
                extra = {
                    "num_embeddings_per_partition": getattr(
                        module, "num_embeddings_per_partition", None
                    ),
                    "org_vocab_size": getattr(module, "org_vocab_size", None),
                }
            elif isinstance(module, VocabParallelEmbedding):
                kind = "VocabParallelEmbedding"
                extra = {
                    "num_embeddings_per_partition": getattr(
                        module, "num_embeddings_per_partition", None
                    ),
                    "org_vocab_size": getattr(module, "org_vocab_size", None),
                }
            elif isinstance(module, RMSNorm):
                kind = "RMSNorm"

            if kind is None:
                continue

            weight = getattr(module, "weight", None)
            weight_info = None
            if weight is not None:
                weight_info = {
                    "shape": list(weight.shape),
                    "dtype": str(weight.dtype),
                    "is_contiguous": weight.is_contiguous(),
                    "stride": list(weight.stride()),
                    "has_output_dim_attr": hasattr(weight, "output_dim"),
                    "output_dim": getattr(weight, "output_dim", None),
                    "has_input_dim_attr": hasattr(weight, "input_dim"),
                    "input_dim": getattr(weight, "input_dim", None),
                }

            modules.append(
                {"name": name, "kind": kind, "weight": weight_info, **extra}
            )

        return {
            "rank": self.rank,
            "tp_size": self.parallel_config.tensor_parallel_size,
            "modules": modules,
        }

    def slice_hashes(
        self, param_name: str, dim: int, ranges: list[list[int]]
    ) -> list[dict[str, Any]]:
        import hashlib

        import torch

        obj: Any = self.model_runner.model
        *parents, leaf = param_name.split(".")
        for p in parents:
            obj = getattr(obj, p)
        param = getattr(obj, leaf)

        out = []
        for start, size in ranges:
            chunk = param.data.narrow(dim, start, size)
            narrow_was_contiguous = chunk.is_contiguous()
            raw = chunk.contiguous().to(torch.float32).cpu().numpy().tobytes()
            out.append(
                {
                    "start": start,
                    "size": size,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "narrow_was_contiguous": narrow_was_contiguous,
                }
            )
        return out

    def raw_param_bytes(self, param_name: str) -> tuple[str, list[int], bytes]:
        """Full local tensor, bit-exact, as (dtype_str, shape, raw_bytes) --
        same msgpack-native shape as collective_logits.py's Q7 tensor
        transport, for the same reason (a bare torch.Tensor return value
        can't cross collective_rpc, see that module's Q7). `.view(torch.uint8)`
        reinterprets the tensor's own bytes directly (works for bf16 with no
        numpy dtype dependency, unlike slice_hashes's float32 cast -- this
        method exists specifically to preserve the original bits for an
        exact torch.equal check, not to hash a value)."""
        import torch

        obj: Any = self.model_runner.model
        *parents, leaf = param_name.split(".")
        for p in parents:
            obj = getattr(obj, p)
        tensor = getattr(obj, leaf).data.contiguous().cpu()
        raw = tensor.view(torch.uint8).numpy().tobytes()
        return (str(tensor.dtype).removeprefix("torch."), list(tensor.shape), raw)


# --------------------------------------------------------------------------- #
# Driver. One subprocess per TP degree -- same reason as
# bf16_floor._spawn_worker (isolate CUDA context between successive LLM()
# constructions within one measurement run).
# --------------------------------------------------------------------------- #

_QKV_PARAM = "model.layers.0.self_attn.qkv_proj.weight"
_GATE_UP_PARAM = "model.layers.0.mlp.gate_up_proj.weight"
_Q_NORM_PARAM = "model.layers.0.self_attn.q_norm.weight"
_K_NORM_PARAM = "model.layers.0.self_attn.k_norm.weight"


def _reconstruct_tensor(dtype_str: str, shape: list[int], raw_bytes: bytes) -> Any:
    """Inverse of raw_param_bytes: reinterpret the raw bytes back to the
    original dtype (uint8 -> dtype is a bit-exact reinterpret, not a cast --
    same reasoning as the encode side) and reshape."""
    import torch

    flat_u8 = torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint8)
    return flat_u8.view(getattr(torch, dtype_str)).view(*shape).clone()


def _run_worker(model_dir: str, tp: int, out_path: Path) -> None:
    """Runs inside its own process (see module docstring's WHAT TO RUN)."""
    import torch
    from vllm import LLM

    llm = LLM(
        model=model_dir,
        tensor_parallel_size=tp,
        dtype="bfloat16",
        enforce_eager=True,
        worker_extension_cls=WORKER_EXTENSION_QUALNAME,
    )

    per_rank_modules = llm.collective_rpc("report_parameters")
    q_norm_hashes = llm.collective_rpc(
        "slice_hashes", args=(_Q_NORM_PARAM, 0, [[0, QWEN3_0_6B.head_dim]])
    )
    k_norm_hashes = llm.collective_rpc(
        "slice_hashes", args=(_K_NORM_PARAM, 0, [[0, QWEN3_0_6B.head_dim]])
    )

    # Full real tensors (layer 0), not hashes -- these back BOTH the rank-
    # identity check below AND qwen3_layout.check_content_predictions, which
    # needs the actual bytes to run reshard.split_tensor on, not a digest of
    # them.
    qkv_raw = llm.collective_rpc("raw_param_bytes", args=(_QKV_PARAM,))
    gate_up_raw = llm.collective_rpc("raw_param_bytes", args=(_GATE_UP_PARAM,))
    qkv_tensors = [_reconstruct_tensor(*r) for r in qkv_raw]
    gate_up_tensors = [_reconstruct_tensor(*r) for r in gate_up_raw]

    rank_identity_check = None
    if tp > 1:
        # Task ask (originally at tp=2 only): confirm ranks hold DIFFERENT
        # contents (sharded), not just same-shaped tensors that happen to be
        # replicated. Full bytes, not a hash -- torch.equal and a real
        # max-abs-diff, not a probabilistic hash-collision argument.
        #
        # Generalized to every pair of ranks, not just (0, 1): at tp=4 there
        # are 4 ranks and C(4,2)=6 pairs; checking only rank0-vs-rank1 would
        # miss a bug where, say, ranks 0 and 2 were accidentally identical
        # while 0-vs-1 correctly differed.
        pairs = []
        for i in range(len(qkv_tensors)):
            for j in range(i + 1, len(qkv_tensors)):
                a, b = qkv_tensors[i], qkv_tensors[j]
                same_shape = a.shape == b.shape
                pairs.append(
                    {
                        "ranks": [i, j],
                        "torch_equal": torch.equal(a, b) if same_shape else False,
                        "max_abs_diff": (a.float() - b.float()).abs().max().item()
                        if same_shape
                        else None,
                        "shape_mismatch": None if same_shape else [list(a.shape), list(b.shape)],
                    }
                )
        rank_identity_check = {
            "param": _QKV_PARAM,
            "shape": list(qkv_tensors[0].shape),
            "all_pairs_distinct": all(not p["torch_equal"] for p in pairs),
            "pairs": pairs,
        }

    out_path.write_text(
        json.dumps(
            {
                "tp": tp,
                "per_rank_modules": per_rank_modules,
                "q_norm_hashes": q_norm_hashes,
                "k_norm_hashes": k_norm_hashes,
                "rank_identity_check_qkv_proj": rank_identity_check,
            },
            indent=2,
        )
    )

    # Sidecar .pt file: the real tensors qwen3_layout.check_content_predictions
    # runs reshard.split_tensor against. Plain torch.save on the local
    # filesystem, not collective_rpc -- both ranks' tensors are already
    # gathered into this (the LLM driver) process's memory by the
    # collective_rpc calls above, so there is no second IPC hop to reuse
    # Q7's byte-buffer trick for; that trick was for crossing collective_rpc
    # itself, not for writing a file.
    torch.save(
        {
            "qkv_proj": qkv_tensors[0] if tp == 1 else qkv_tensors,
            "gate_up_proj": gate_up_tensors[0] if tp == 1 else gate_up_tensors,
        },
        out_path.with_suffix(".pt"),
    )


def _spawn_worker(python: str, model_dir: str, tp: int, out_path: Path) -> None:
    subprocess.run(
        [
            python,
            "-m",
            "weight_sync_bench.phase2.param_layout_inspection",
            "--worker",
            "--model-dir",
            str(model_dir),
            "--tp",
            str(tp),
            "--out",
            str(out_path),
        ],
        check=True,
    )


# Every TP degree the differential covers. TP=1 is always the reference leg
# (never itself "checked" against another degree); 2 and 4 are each checked
# against it independently. Adding a degree here is the only change needed
# to sweep it -- everything below reads this tuple rather than assuming two
# legs. n_kv_heads=8 divides every one of these evenly (no GQA replication;
# see geometry.py/qwen3_layout.py), so none of them should raise
# UnsupportedLayout for Qwen3-0.6B specifically -- if one does, that is
# itself the finding disproving the source-reading prediction.
DEGREES = (1, 2, 4)


def _check_norm_replication(tp1: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    """q_norm/k_norm are Replicated() -- no split_tensor call is meaningful
    for a placement that never changes bytes across TP degree, so this stays
    a direct hash-equality check rather than routing through
    qwen3_layout.check_content_predictions. `other` can be any non-1 degree's
    leg (its q_norm_hashes/k_norm_hashes lists are as long as that degree has
    ranks) -- nothing here is specific to tp=2."""
    tp1_q_norm = tp1["q_norm_hashes"][0][0]["sha256"]
    tp1_k_norm = tp1["k_norm_hashes"][0][0]["sha256"]
    return {
        "q_norm_identical_across_tp1_and_every_rank": all(
            r[0]["sha256"] == tp1_q_norm for r in other["q_norm_hashes"]
        ),
        "k_norm_identical_across_tp1_and_every_rank": all(
            r[0]["sha256"] == tp1_k_norm for r in other["k_norm_hashes"]
        ),
    }


def inspect_layouts(
    model_dir: str | None = None, python: str = sys.executable, degrees: tuple[int, ...] = DEGREES
) -> dict[str, Any]:
    import tempfile

    import torch
    from huggingface_hub import snapshot_download

    from .qwen3_layout import check_content_predictions, check_shape_predictions

    if 1 not in degrees:
        raise ValueError(f"degrees must include 1 (the reference leg every other degree is checked against), got {degrees}")

    checkpoint_dir = model_dir or snapshot_download(MODEL_ID)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        results: dict[int, dict[str, Any]] = {}
        tensors: dict[int, dict[str, Any]] = {}
        for tp in degrees:
            out_path = tmp_path / f"tp{tp}.json"
            _spawn_worker(python, checkpoint_dir, tp, out_path)
            results[tp] = json.loads(out_path.read_text())
            tensors[tp] = torch.load(out_path.with_suffix(".pt"))

    shape_checks = {
        tp: check_shape_predictions(QWEN3_0_6B, tp, results[tp]["per_rank_modules"]) for tp in degrees
    }
    content_checks = {
        tp: check_content_predictions(QWEN3_0_6B, tp, tensors[1], tensors[tp])
        for tp in degrees
        if tp != 1
    }
    norm_checks = {
        tp: _check_norm_replication(results[1], results[tp]) for tp in degrees if tp != 1
    }

    report = {
        "phase": "2b",
        "step": "inspection + prediction checks (SPEC.md 2b's actual object of study: the loader boundary)",
        "model_id": MODEL_ID,
        "checkpoint_dir": checkpoint_dir,
        "geometry": asdict(QWEN3_0_6B),
        "degrees": list(degrees),
        "legs": results,
        "shape_prediction_check": shape_checks,
        "content_prediction_check": content_checks,
        "norm_replication_check": norm_checks,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(report, indent=2))
    return report


def _print_report(report: dict[str, Any]) -> None:
    print(f"phase 2b param layout inspection -> {ARTIFACT}")
    degrees = report["degrees"]
    for tp in degrees:
        leg = report["legs"][tp]
        print(f"\n  TP={tp}, {len(leg['per_rank_modules'])} rank(s):")
        for rank_report in leg["per_rank_modules"]:
            print(f"    rank {rank_report['rank']}:")
            for m in rank_report["modules"]:
                w = m["weight"]
                shape = tuple(w["shape"]) if w else None
                contiguous = w["is_contiguous"] if w else None
                print(f"      {m['name']:<45} {m['kind']:<26} shape={shape} contiguous={contiguous}")

    print("\n  check 1 -- shape prediction (LayoutTable's predicted shape vs inspection):")
    for tp_key, check in report["shape_prediction_check"].items():
        print(
            f"    TP={tp_key}: {check['checked']}/{check['predicted_total']} checked, "
            f"match={check['shape_predictions_match']}"
        )
        for mm in check["mismatches"]:
            print(f"      MISMATCH rank={mm['rank']} {mm['param']}: predicted={mm['predicted']} actual={mm['actual']}")

    print("\n  check 2 -- content prediction (reshard.split_tensor(TP1 real tensor) vs real tensor at each other degree):")
    for tp_key, per_param in report["content_prediction_check"].items():
        for param, per_rank in per_param.items():
            for r in per_rank:
                print(f"    TP={tp_key} {param} rank={r['rank']}: torch.equal={r['torch_equal']} max_abs_diff={r['max_abs_diff']}")

    for tp_key, n in report["norm_replication_check"].items():
        print(f"\n  TP={tp_key} q_norm identical vs TP1 :", n["q_norm_identical_across_tp1_and_every_rank"])
        print(f"  TP={tp_key} k_norm identical vs TP1 :", n["k_norm_identical_across_tp1_and_every_rank"])

    for tp in degrees:
        if tp == 1:
            continue
        leg = report["legs"][tp]
        identity = leg["rank_identity_check_qkv_proj"]
        print(f"\n  TP={tp} rank-pairwise identity, {identity['param']} {tuple(identity['shape'])}:")
        print(f"    all pairs distinct (sharded, not replicated): {identity['all_pairs_distinct']}")
        for p in identity["pairs"]:
            print(f"      ranks {p['ranks']}: torch.equal={p['torch_equal']} max_abs_diff={p['max_abs_diff']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model-dir", type=str, default=None)
    # No top-level default: None is how the check below tells "the user
    # explicitly passed --tp" apart from "they didn't." A numeric default
    # here would make that indistinguishable and silently defeat the check.
    parser.add_argument("--tp", type=int, default=None, choices=DEGREES, help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.worker:
        assert args.model_dir and args.out and args.tp is not None
        _run_worker(args.model_dir, args.tp, args.out)
        return

    # `--tp` is accepted by argparse at the top level only because it is the
    # SAME parser `_spawn_worker` re-invokes with `--worker` for each internal
    # leg (main() -> --worker branch above). Without this check, a user
    # passing `--tp 1` at the top level would have it silently ignored:
    # inspect_layouts() always runs every degree in DEGREES regardless, so
    # the flag would look like it selected one leg while actually doing
    # nothing -- accepted, defaulted, validated by `choices`, and never read.
    # That is the same failure shape as a validation check that passes
    # everything it should have caught: it looks like protection and provides
    # none. FAIL LOUDLY here instead.
    #
    # Do NOT "fix" this by making the top-level path honor `--tp` and run a
    # single leg -- that would let someone silently compute half the
    # differential (e.g. only ever run TP=1) and believe they'd validated the
    # cross-degree invariant this module exists to check. If a single-leg
    # top-level mode is ever genuinely wanted, it needs a differently-named
    # flag that makes clear it is NOT running the full comparison, not a
    # repurposing of this one.
    if args.tp is not None:
        parser.error(
            "--tp has no effect at the top level (without --worker): every "
            f"degree in {DEGREES} is always run, since the differential this "
            "module computes requires all of them. Pass --tp only together "
            "with --worker (internal single-leg use)."
        )

    report = inspect_layouts()
    _print_report(report)


if __name__ == "__main__":
    main()
