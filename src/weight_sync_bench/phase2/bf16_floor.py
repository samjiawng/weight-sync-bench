"""bf16 floor measurement against real Qwen3-0.6B / vLLM. SPEC.md phase 2, section 2a.

UNTESTED. Written on a CPU-only dev machine against vLLM v0.28.0 source read on
GitHub (see citations below), per SPEC.md's "read the weight-update entry point in
the source of the version you pinned rather than trusting a tutorial." It has not
been run against a real GPU or a real vLLM install. Before trusting any number out
of it: run `--tp 1` and `--tp 2` with small `--repetitions/--batch/--seq-len` first
as a smoke test, confirm the kwargs below are still accepted by the pinned version,
and confirm `--tp 2` actually gets two ranks onto two devices rather than erroring
or silently falling back. Fix forward from there; do not assume this file is right
because it reads plausibly.

Pinned: vllm==0.28.0. Record any change to this pin here, since the API surface
this file depends on "has churned repeatedly" (SPEC.md) and a version bump
invalidates the citations below.

Install, in its OWN environment on the GPU box -- not this repo's synced venv.
vllm==0.28.0 pulls in mistral-common, which caps numpy below phase 1's pinned
2.4.6; `uv sync` resolves one universal lock across every extra even when only
some are installed, so declaring vllm as a `[project.optional-dependencies]`
extra here makes the existing phase-1 lock unsatisfiable (discovered by trying
exactly that; reverted). A separate environment sidesteps it and is arguably the
more honest shape anyway -- phase 1 is CPU-only by design and should stay that
way regardless of what phase 2 needs:

    uv venv --python 3.12 .venv-phase2
    uv pip install --python .venv-phase2 -e . vllm==0.28.0 safetensors huggingface-hub
    .venv-phase2/bin/python -m weight_sync_bench.phase2.bf16_floor --repetitions 5 --batch 2 --seq-len 8

Sourced facts this module depends on (vLLM v0.28.0, github.com/vllm-project/vllm):

- Raw pre-softmax logits are obtainable through the public request API, not just
  post-softmax log-probabilities: `ModelConfig.logprobs_mode` (vllm/config/model.py)
  is `Literal["raw_logits", "raw_logprobs", "processed_logits",
  "processed_logprobs"]`. We set it to `"raw_logits"`.
- `SamplingParams.prompt_logprobs = -1` (vllm/sampling_params.py) requests "all
  `vocab_size` log probabilities" for every prompt token position -- a full logits
  tensor per position, matching phase 1's "full logits tensor" comparison.
- For `prompt_logprobs` specifically, raw and processed are documented to be
  identical ("processed_* and raw_* yield identical results" because prompt tokens
  never pass through a sampling processor -- vllm/config/model.py). So
  `logprobs_mode="raw_logits"` only needs stating once; there is no separate
  "processed" concern for the prompt-side numbers we read.
- `ModelConfig.max_logprobs` defaults to 20 and must be set to `-1` (uncapped) or a
  `prompt_logprobs=-1` request is rejected as exceeding the cap
  (`_validate_logprobs`, vllm/sampling_params.py).
- A running engine can be hot-reloaded from a modified on-disk checkpoint via
  `llm.collective_rpc("update_config", args=({"load_config": {"load_format":
  "auto"}},))` then `llm.collective_rpc("reload_weights")`
  (examples/rl/skip_loading_weights_in_engine_init.py). This module does NOT use
  that path -- it constructs a fresh `LLM(model=<dir>)` per checkpoint instead,
  since nothing here needs a survives-a-reload engine. `reload_weights` is the
  right primitive for 2d's filesystem-checkpoint transport (hot reload without
  restart is the point there); reusing it here would just add risk for no benefit.
- vLLM v0.28.0 separately ships `vllm.distributed.weight_transfer`
  (`WeightTransferEngine` / `TrainerWeightTransferEngine`, backends in
  `vllm/distributed/weight_transfer/{nccl,ipc,sparse_nccl}_engine.py`) for a live
  trainer process pushing tensors directly into a running engine over NCCL or CUDA
  IPC. That is 2d's subject, not 2a's -- 2a only needs *a* real load path, and the
  from-disk one is simplest to get right first.
- Qwen3-0.6B checkpoint geometry (huggingface.co/Qwen/Qwen3-0.6B config.json and
  the model.safetensors header, fetched directly rather than assumed): hidden_size
  1024, num_hidden_layers 28, num_attention_heads 16, num_key_value_heads 8,
  head_dim 128 (note head_dim * n_heads = 2048 != hidden_size -- Qwen3 sizes q/k/v
  independently of hidden_size, unlike the phase 1 toy model), vocab_size 151936,
  tie_word_embeddings true. Q and K each carry their own per-head RMSNorm
  (`self_attn.q_norm.weight`, `self_attn.k_norm.weight`, shape [128], applied
  identically to every head) -- a real mechanism the phase 1 toy model has no
  analogue for. The checkpoint stores `q_proj` / `k_proj` / `v_proj` and
  `gate_proj` / `up_proj` as SEPARATE tensors, not fused -- vLLM's own model class
  fuses them into internal `qkv_proj` / `gate_up_proj` buffers at load time. This
  is the exact "which of the three conventions" question SPEC.md 2b has to
  resolve; 2a does not attempt to resolve it (see "What the break cases mean here"
  below), but the fact was confirmed against the real checkpoint rather than
  assumed, since it directly bears on whether `HeadPartitioned` as-is can express
  Qwen3-as-loaded-by-vLLM.
- Continuous batching in vLLM is not guaranteed elementwise-reproducible across
  different batch compositions (see examples/rl/rlhf_async_new_apis.py's use of
  `VLLM_BATCH_INVARIANT=1`, which is only fully supported on compute capability
  >= 9.0 -- H100/H200/B100/B200). To keep that source of nondeterminism out of the
  TP1-vs-TP2 floor measurement, every request here is submitted and generated
  **one prompt at a time** (batch size 1 at the engine level), even though the
  measurement groups them into (batch, seq_len) cells for reporting. This trades
  throughput for isolating the one variable phase 2a is supposed to measure:
  reduction order and dtype under TP, not scheduler batch-composition effects.

What the break cases mean here, and why this is deliberately not 2b:

Phase 1's break cases are bugs in `reshard.py` / `ShardedModel` -- code this repo
owns and tests. In phase 2, vLLM's internal TP sharding is not under test; the
`ReferenceModel(full_params, x) == vLLM(x)` invariant treats vLLM as correct by
construction, and what phase 2 actually validates is *our* `LayoutTable` for the
trainer-to-vLLM handoff, which SPEC.md 2b has not been built yet. So a "break
case" here cannot be "inject a bug into our resharder" -- there is no resharder
yet. Instead, each case hand-corrupts a real Qwen3-0.6B checkpoint tensor in a way
that is shape-preserving and structurally analogous to one of the phase 1 cases
(permuted per-head assignment, permuted row-parallel input columns, permuted norm
weights), then loads the corrupted checkpoint through vLLM's real loader
(`LLM(model=<corrupted_dir>)`) and compares its output to the correct checkpoint's
output. This tests whether the mean-deviation statistic and the derived threshold
can detect a corruption of this shape and magnitude on the real bf16 Qwen3
architecture at all -- which is what 2a exists to gate -- independent of whether
vLLM's own sharding is involved, so it runs identically regardless of `--tp`. It
is not a test of any particular resharder, because none exists yet; once 2b builds
one, break-case injection should move to feeding it a deliberately wrong
`LayoutTable` and asserting on *its* output, the way phase 1 does, and this
checkpoint-hand-editing approach should be retired rather than kept as a parallel
path.

Statistic: MEAN absolute deviation over the full logits tensor, not max (see
SPEC.md's "Change the primary statistic"). max/median are still recorded.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Everything below that touches torch/vllm/safetensors is imported lazily inside
# functions, not at module scope, so `import weight_sync_bench.phase2.bf16_floor` for
# its constants/CLI does not require the phase2 extra to be installed.

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = REPO_ROOT / "tolerance" / "phase2a_bf16_floor.json"

PINNED_VLLM_VERSION = "0.28.0"
MODEL_ID = "Qwen/Qwen3-0.6B"

# bf16 mantissa step at magnitude 1.0: 7 explicit mantissa bits, one fewer than
# fp32's 23. Mirrors tolerance.py's ULP constant, scaled for the dtype in play here.
ULP_BF16 = 2.0**-7

# Deliberately small and CLI-overridable: this costs real rented-GPU money.
# Widen once the plumbing is confirmed working (see module docstring).
DEFAULT_REPETITIONS = 5
DEFAULT_BATCH = 2
DEFAULT_SEQ_LEN = 8

# Measured, not phase-1-inherited. Two runs (see tolerance/phase2a_bf16_floor.json):
# 5 reps at batch=2/seq_len=8 gave floor mean 4.296e-02, weakest break 1.757, a
# 40.9x separation; 20 reps at the same shape gave floor mean 3.898e-02, weakest
# break 1.570, a 40.3x separation. The mean held within 9% across a 4x increase
# in repetitions -- phase 1's earlier finding that sample-count drift is small on
# the mean and large on the max reproduces here (max moved 96 -> 104 ULP over the
# same two runs). Treat ~40x as the real budget bf16 has to spend, against phase
# 1's fp32 budget of roughly 10^5 (floor 1.67e-6 against breaks of 0.34-2.5).
#
# SAFETY_FACTOR x GATE_MARGIN must stay strictly below the measured separation
# ratio, or the gate cannot fail an injected bug no matter how the two factors
# are split -- that is what carrying over phase 1's 100 x 10 = 1000 unchanged
# did here: 1000 > 40x, so no break case could ever clear, and the resulting
# "fail" verdict was an artifact of the unchanged constants, not a finding about
# bf16. 15 x 2 = 30 stays under 40x with headroom for the ratio to move on a
# future re-measurement without flipping the gate. The split is asymmetric on
# purpose: floor noise (dtype rounding, reduction order) is the larger, better-
# characterized source of variance here, so it gets the larger factor; the
# 2x GATE_MARGIN is a thin floor under real-bug detection, not a comfortable
# one, and that thinness is the honest cost of a ~40x total budget rather than
# fp32's ~10^5x one. A future session seeing 2x next to phase 1's thousands-of-x
# GATE_MARGIN should read this comment, not assume the number is a mistake.
SAFETY_FACTOR = 15
MIN_THRESHOLD = 1e-9

# Gate criteria threshold (SPEC.md 2a): a break case must clear the derived bf16
# threshold by at least this multiple on mean deviation to count as "clears."
# See the SAFETY_FACTOR comment above for why this is 2, not phase 1's 10.
GATE_MARGIN = 2


@dataclass(frozen=True)
class Qwen3Geometry:
    """Qwen/Qwen3-0.6B, as read from its published config.json and safetensors
    header rather than assumed -- see the module docstring for the fetch."""

    hidden_size: int = 1024
    n_layers: int = 28
    n_heads: int = 16
    n_kv_heads: int = 8
    head_dim: int = 128
    ffn: int = 3072
    vocab: int = 151936
    tie_word_embeddings: bool = True


QWEN3_0_6B = Qwen3Geometry()


def derive_threshold(mean_deviation: float, safety_factor: int = SAFETY_FACTOR) -> float:
    """threshold = safety_factor * worst observed MEAN deviation. A direct
    multiple, not phase 1's smallest-power-of-ten -- deliberately different rule
    shape from tolerance.derive_threshold, scoped to phase 2 only (phase 1's
    rounded rule in tolerance.py is untouched). Phase 1 could afford power-of-ten
    rounding because it had ~5 orders of magnitude of headroom between floor and
    break; here the whole budget is ~40x (see the SAFETY_FACTOR comment), and
    rounding up to the next power of ten can by itself consume a factor of up to
    10x of that -- e.g. 100 * 4.3e-2 = 4.3 rounds to 10, a free 2.3x tax that a
    ~40x budget cannot absorb. Stating the threshold as a direct multiple removes
    that tax; the remaining conservatism is entirely in the chosen constant.
    Kept as a free function, not inlined, so a reader can re-derive the threshold
    from the artifact themselves -- same discipline as phase 1.
    """
    if mean_deviation <= 0.0:
        return MIN_THRESHOLD
    return float(safety_factor * mean_deviation)


def environment() -> dict[str, Any]:
    """Provenance block. Import errors here mean the phase2 extra isn't installed --
    that's fine when just inspecting the module, but `main()` needs it for real."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "model": MODEL_ID,
        "pinned_vllm_version": PINNED_VLLM_VERSION,
        "dtype": "torch.bfloat16",
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
            # NOT torch.cuda.driver_version. The previous line here was
            # `torch.cuda.driver_version if hasattr(torch.cuda,
            # "driver_version") else None` -- a hasattr guard around an
            # attribute that does not exist on torch 2.13.0+cu130, so it
            # silently fell through to None on every real run. That is the
            # SAME failure class as collective_logits.py's old `shape[0] >=
            # expected_min_positions` check (fixed to an exact `==`): a loose
            # guard around instrumentation that degrades silently instead of
            # raising when its assumption is wrong, so a missing value reads
            # as "not applicable here" instead of "this code is broken."
            # Concretely: every phase-2 artifact through
            # tolerance/phase2a_bf16_floor_v2.json recorded driver_version as
            # null while the box was reporting 580.126.16 throughout, and
            # nothing in this pipeline ever surfaced the mismatch -- it took
            # a human reading the artifact next to the box's own output to
            # notice. The NVIDIA driver version is a system-level property
            # nvidia-smi reports, not something torch's Python API exposes
            # directly, so there was never a torch attribute to guard in the
            # first place.
            info["driver_version"] = None
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if out.returncode == 0:
                    lines = [line.strip() for line in out.stdout.splitlines() if line.strip()]
                    if lines:
                        info["driver_version"] = lines[0]
            except (OSError, subprocess.SubprocessError):
                pass
    except ImportError:
        pass
    try:
        import vllm

        info["vllm_installed_version"] = vllm.__version__
        if info["vllm_installed_version"] != PINNED_VLLM_VERSION:
            info["vllm_version_mismatch_warning"] = (
                f"installed {info['vllm_installed_version']!r} != pinned "
                f"{PINNED_VLLM_VERSION!r}; the sourced-API citations in this "
                "module's docstring were read against the pinned version and may "
                "not hold for what's actually installed."
            )
    except ImportError:
        info["vllm_installed_version"] = None
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        info["git_commit"] = out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        info["git_commit"] = None
    return info


# --------------------------------------------------------------------------- #
# Worker: runs exactly one vLLM engine, one TP degree, in its own process.
#
# Two separate LLM() instances (e.g. TP=1 then TP=2) in a single Python process is
# a known-flaky pattern for vLLM -- leftover NCCL/Ray process-group state from the
# first engine is not guaranteed to be cleanly torn down before the second one
# initializes, especially once TP > 1 is involved. Rather than write teardown code
# whose correctness I cannot verify without a GPU, each TP degree gets its own
# subprocess (`--worker`), and the orchestrator (`measure_differential_floor`)
# just shells out and reads back a saved tensor. Simplify this later on real
# hardware if in-process reuse turns out to be reliable after all.
# --------------------------------------------------------------------------- #


def _seeded_token_batches(
    repetitions: int, batch: int, seq_len: int, vocab: int, seed_base: int = 0
) -> list["torch.Tensor"]:  # noqa: F821 -- torch imported lazily by caller
    """Token seed for repetition `rep` is `10_000 + seed_base + rep`. `seed_base`
    defaults to 0, so the default call (no `seed_base` passed) is byte-for-byte
    the same as before this parameter existed -- every artifact recorded under
    the old, unparameterized version of this function remains reproducible by
    its own stated command. See `--seed-base` (weight_sync_bench.phase2.collective_logits._run_worker)
    for why a caller would ever pass a nonzero value: to draw a genuinely
    different sample (different tokens, different model-seed state) for a
    discriminating rerun, without disturbing what a bare rerun at the default
    reproduces.
    """
    import torch

    return [
        torch.randint(
            0,
            vocab,
            (batch, seq_len),
            generator=torch.Generator().manual_seed(10_000 + seed_base + rep),
        )
        for rep in range(repetitions)
    ]


def _run_worker(
    model_dir: str,
    tp: int,
    repetitions: int,
    batch: int,
    seq_len: int,
    out_path: Path,
) -> None:
    """Runs inside its own process. Loads `model_dir` at TP degree `tp`, computes
    full-vocab raw-logit prompt tensors for `repetitions` seeded token batches, and
    torch.saves them (a list of [batch, seq_len - 1, vocab] float32 tensors -- the
    first position is dropped, see below) to `out_path`.
    """
    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_dir,
        tensor_parallel_size=tp,
        dtype="bfloat16",
        enforce_eager=True,  # determinism over throughput; this is a correctness gate
        max_logprobs=-1,
        logprobs_mode="raw_logits",
        gpu_memory_utilization=0.85,
    )

    token_batches = _seeded_token_batches(repetitions, batch, seq_len, QWEN3_0_6B.vocab)
    all_reps: list["torch.Tensor"] = []

    for tokens in token_batches:
        rep_positions: list["torch.Tensor"] = []
        for row in tokens.tolist():
            # One request at a time -- see module docstring on batch invariance.
            # max_tokens=1 because we only want prompt_logprobs (teacher-forced
            # logits over the given tokens); the one generated token is unused.
            sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=-1)
            [output] = llm.generate([{"prompt_token_ids": row}], sp)
            # output.prompt_logprobs: list[dict[int, Logprob] | None], one entry
            # per prompt position. Position 0 is always None (no preceding
            # context to condition on), so it carries no comparable logits and is
            # dropped -- consistently on both the TP1 and TP2 side.
            positions = output.prompt_logprobs[1:]
            dense = torch.empty(len(positions), QWEN3_0_6B.vocab, dtype=torch.float32)
            for i, pos in enumerate(positions):
                for token_id, logprob_obj in pos.items():
                    dense[i, token_id] = logprob_obj.logprob
            rep_positions.append(dense)
        all_reps.append(torch.stack(rep_positions, dim=0))  # [batch, seq_len-1, vocab]

    torch.save(all_reps, out_path)


def measure_differential_floor(
    repetitions: int, batch: int, seq_len: int, python: str = sys.executable
) -> dict[str, Any]:
    """SPEC.md 2a step 4: vLLM-TP1 vs vLLM-TP2 on the correct, unmutated checkpoint.

    Downloads the real checkpoint once, runs each TP degree as its own subprocess
    (see the note above `_run_worker`), and diffs the two saved tensor sets.
    """
    import tempfile

    import torch
    from huggingface_hub import snapshot_download

    checkpoint_dir = snapshot_download(MODEL_ID)

    cells: list[dict[str, float]] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tp1_path, tp2_path = tmp / "tp1.pt", tmp / "tp2.pt"
        for tp, out_path in ((1, tp1_path), (2, tp2_path)):
            _spawn_worker(python, checkpoint_dir, tp, repetitions, batch, seq_len, out_path)

        tp1_reps = torch.load(tp1_path)
        tp2_reps = torch.load(tp2_path)

        for tp1_logits, tp2_logits in zip(tp1_reps, tp2_reps):
            diff = (tp2_logits - tp1_logits).abs()
            cells.append(
                {
                    "max": diff.max().item(),
                    "median": diff.median().item(),
                    "mean": diff.mean().item(),
                }
            )

    mean_deviation = sum(cell["mean"] for cell in cells) / len(cells)
    max_deviation = max(cell["max"] for cell in cells)
    return {
        "cells": cells,
        "mean_deviation": mean_deviation,
        "max_deviation": max_deviation,
        "mean_ulp": mean_deviation / ULP_BF16,
        "max_ulp": max_deviation / ULP_BF16,
        "threshold": derive_threshold(mean_deviation),
        "checkpoint_dir": checkpoint_dir,
        # Kept (not just its stats) so callers can reuse it as the break-case
        # reference without re-spawning a TP=1 engine a second time -- that run
        # already cost real rented-GPU minutes once.
        "tp1_reference": tp1_reps,
    }


def _spawn_worker(
    python: str,
    model_dir: str,
    tp: int,
    repetitions: int,
    batch: int,
    seq_len: int,
    out_path: Path,
) -> None:
    subprocess.run(
        [
            python,
            "-m",
            "weight_sync_bench.phase2.bf16_floor",
            "--worker",
            "--model-dir",
            str(model_dir),
            "--tp",
            str(tp),
            "--repetitions",
            str(repetitions),
            "--batch",
            str(batch),
            "--seq-len",
            str(seq_len),
            "--out",
            str(out_path),
        ],
        check=True,
    )


# --------------------------------------------------------------------------- #
# Break-case checkpoint corruption. See the module docstring's "What the break
# cases mean here" section before touching this.
# --------------------------------------------------------------------------- #

BREAK_CASES = ("case1_qkv_head_permute", "case2_oproj_col_permute", "case3_norm_permute")


def corrupt_checkpoint(src_dir: str, dst_dir: Path, case: str, layer: int = 0) -> None:
    """Copy `src_dir` to `dst_dir`, then apply one shape-preserving corruption to
    layer `layer`'s tensors and re-save. `dst_dir` is a full, loadable HF-format
    checkpoint directory afterward -- `LLM(model=dst_dir)` goes through vLLM's real
    loader exactly as it would for the original checkpoint.
    """
    import torch
    from safetensors.torch import load_file, save_file

    if case not in BREAK_CASES:
        raise ValueError(f"unknown break case {case!r}, expected one of {BREAK_CASES}")

    shutil.copytree(src_dir, dst_dir)

    st_path = _find_safetensors_file(dst_dir)
    tensors = load_file(st_path)
    prefix = f"model.layers.{layer}.self_attn."
    g = QWEN3_0_6B

    if case == "case1_qkv_head_permute":
        # Swap adjacent head-pairs (0<->1, 2<->3, ...) in q_proj's row blocks, and
        # correspondingly in k_proj (fewer KV heads, GQA groups n_heads //
        # n_kv_heads query heads per KV head -- swapping adjacent Q heads within
        # the same KV group keeps K/V head boundaries meaningful, so also permute
        # k_proj's own adjacent KV-head pairs to keep the corruption analogous to
        # phase 1 case 1's "wrong per-head assignment" rather than degenerating
        # into a q-only bug). Shape-preserving: same tensor shape in and out.
        tensors[prefix + "q_proj.weight"] = _swap_adjacent_blocks(
            tensors[prefix + "q_proj.weight"], g.head_dim, g.n_heads
        )
        tensors[prefix + "k_proj.weight"] = _swap_adjacent_blocks(
            tensors[prefix + "k_proj.weight"], g.head_dim, g.n_kv_heads
        )
    elif case == "case2_oproj_col_permute":
        # o_proj.weight is [hidden_size, n_heads * head_dim]; permuting its INPUT
        # (dim 1) columns by head is the phase-2 analogue of phase 1 case 2's
        # reversed rank-to-slice assignment on a row-parallel tensor's dim-1 split.
        tensors[prefix + "o_proj.weight"] = _swap_adjacent_blocks(
            tensors[prefix + "o_proj.weight"].T, g.head_dim, g.n_heads
        ).T.contiguous()
    elif case == "case3_norm_permute":
        # Rotate by half the width -- information-preserving (a permutation, not a
        # zeroing), same discipline as phase 1 case 3.
        w = tensors[f"model.layers.{layer}.input_layernorm.weight"]
        tensors[f"model.layers.{layer}.input_layernorm.weight"] = torch.roll(
            w, shifts=w.shape[0] // 2
        )

    save_file(tensors, st_path)


def _swap_adjacent_blocks(tensor: "torch.Tensor", block_size: int, n_blocks: int) -> "torch.Tensor":
    """Swap block 2i and block 2i+1 along dim 0, where dim 0 has n_blocks blocks of
    block_size rows each. Shape-preserving, information-preserving.
    """
    blocks = list(tensor.split(block_size, dim=0))
    for i in range(0, n_blocks - n_blocks % 2, 2):
        blocks[i], blocks[i + 1] = blocks[i + 1], blocks[i]
    import torch

    return torch.cat(blocks, dim=0)


def _find_safetensors_file(model_dir: Path) -> Path:
    """Qwen3-0.6B ships as a single model.safetensors (confirmed against the
    published repo -- no model.safetensors.index.json shard map). A larger model
    would need to consult the index to find which shard holds a given layer's
    tensors; this is deliberately not handled since 2a only targets Qwen3-0.6B.
    """
    single = model_dir / "model.safetensors"
    if single.exists():
        return single
    raise FileNotFoundError(
        f"{model_dir} has no single model.safetensors -- this checkpoint is "
        "sharded and corrupt_checkpoint() needs the index-map case added"
    )


def run_break_case(
    case: str,
    checkpoint_dir: str,
    tp1_reference: list["torch.Tensor"],
    repetitions: int,
    batch: int,
    seq_len: int,
    python: str = sys.executable,
) -> dict[str, Any]:
    """SPEC.md 2a step 5. Corrupts a fresh copy of the checkpoint, loads it at
    TP=2 through the real loader, and diffs against the TP1 reference computed
    from the correct checkpoint by `measure_differential_floor`. TP=2 is arbitrary
    here (see module docstring: the corruption doesn't depend on vLLM's own
    sharding), chosen only to also exercise the two-GPU path.
    """
    import tempfile

    import torch

    with tempfile.TemporaryDirectory() as tmp:
        corrupted_dir = Path(tmp) / case
        corrupt_checkpoint(checkpoint_dir, corrupted_dir, case)

        out_path = Path(tmp) / f"{case}.pt"
        _spawn_worker(python, str(corrupted_dir), 2, repetitions, batch, seq_len, out_path)
        broken_reps = torch.load(out_path)

    cells = []
    for ref_logits, broken_logits in zip(tp1_reference, broken_reps):
        diff = (broken_logits - ref_logits).abs()
        cells.append(
            {
                "max": diff.max().item(),
                "median": diff.median().item(),
                "mean": diff.mean().item(),
            }
        )
    mean_deviation = sum(cell["mean"] for cell in cells) / len(cells)
    return {"case": case, "mean_deviation": mean_deviation, "cells": cells}


def gate_decision(floor: dict[str, Any], break_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply SPEC.md 2a's pass / marginal / fail rule. Pure function of the
    measurements so the decision is auditable against the artifact.
    """
    threshold = floor["threshold"]
    clears = {
        r["case"]: r["mean_deviation"] >= GATE_MARGIN * threshold for r in break_results
    }
    n_clear = sum(clears.values())
    case3 = "case3_norm_permute"
    if n_clear == len(clears):
        verdict = "pass"
    elif n_clear >= len(clears) - 1 and not clears.get(case3, True):
        verdict = "marginal"
    else:
        verdict = "fail"
    return {"threshold": threshold, "gate_margin": GATE_MARGIN, "clears": clears, "verdict": verdict}


def write(report: dict[str, Any], path: Path = ARTIFACT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--tp", type=int, default=1, choices=(1, 2))
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.worker:
        assert args.model_dir and args.out
        _run_worker(args.model_dir, args.tp, args.repetitions, args.batch, args.seq_len, args.out)
        return

    floor = measure_differential_floor(args.repetitions, args.batch, args.seq_len)
    tp1_reference = floor.pop("tp1_reference")

    break_results = [
        run_break_case(
            case,
            floor["checkpoint_dir"],
            tp1_reference,
            args.repetitions,
            args.batch,
            args.seq_len,
        )
        for case in BREAK_CASES
    ]

    gate = gate_decision(floor, break_results)

    report = {
        "phase": "2a",
        "environment": environment(),
        "measurement": {
            "repetitions": args.repetitions,
            "batch": args.batch,
            "seq_len": args.seq_len,
        },
        "floor": floor,
        "break_cases": break_results,
        "gate": gate,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = write(report)

    print(f"phase 2a bf16 floor -> {path}")
    print(f"  mean deviation : {floor['mean_deviation']:.3e} ({floor['mean_ulp']:.2f} ULP)")
    print(f"  max  deviation : {floor['max_deviation']:.3e} ({floor['max_ulp']:.2f} ULP)")
    print(f"  threshold      : {floor['threshold']:.3e}  (gate margin {GATE_MARGIN}x)")
    for r in break_results:
        clears = gate["clears"][r["case"]]
        print(f"  {r['case']:<24} mean {r['mean_deviation']:.3e}  {'CLEARS' if clears else 'does not clear'}")
    print(f"\n  verdict: {gate['verdict']}")


if __name__ == "__main__":
    main()
