"""bf16 floor measurement, ported onto collective_logits.run_one_prompt for
extraction. SPEC.md phase 2, section 2a, extraction-path revision.

UNTESTED AS A WHOLE, though built entirely from verified and unchanged parts.
What IS verified on the real GPU box: the extraction primitive this module
is built on, `collective_logits.run_one_prompt` -- bit-identical output
against `bf16_floor.py`'s `prompt_logprobs=-1` path, 272.7x-932.5x faster
(`tolerance/phase2b_extraction.json`). What is NOT verified is the
COMPOSITION built on top of that primitive in this file: the repetitions
loop, the TP=2 leg, break-case checkpoint loading, and the gate-decision
assembly, run end to end. That composition is new relative to what was
executed on the box, even though every individual piece it's made of is
either unchanged (imported directly from `bf16_floor.py`, not re-derived)
or already verified (`run_one_prompt` itself). Smoke-test with
`--repetitions 1 --batch 1 --seq-len 8` before trusting a full run, same
discipline as both modules this one is built from.

`bf16_floor.py` is left completely unmodified and remains runnable as the
reference implementation -- this module is additive, writes to a different
artifact path, and the two can be run side by side.

--------------------------------------------------------------------------
WHAT'S UNCHANGED, AND HOW
--------------------------------------------------------------------------
Imported directly from `bf16_floor.py`, not copied or re-derived -- a call
to any of these always uses bf16_floor.py's own code and its own module
globals, so there is no possibility of the two implementations drifting
apart on these specific pieces:
  - `BREAK_CASES`, `corrupt_checkpoint` -- every break-case injection.
    Checkpoint corruption is pure CPU-side safetensors file manipulation; it
    has no dependency on how logits get extracted afterward, so it did not
    need porting at all.
  - `derive_threshold`, `SAFETY_FACTOR` (=15), `GATE_MARGIN` (=2),
    `MIN_THRESHOLD` -- the threshold rule and its constants. `derive_threshold`
    is called with no explicit `safety_factor` override here, so it always
    resolves `SAFETY_FACTOR` from bf16_floor.py's own module namespace.
  - `gate_decision` -- the pass/marginal/fail rule, same reason: its
    internal reference to `GATE_MARGIN` resolves in bf16_floor.py's
    namespace regardless of what this module does.
  - `environment`, `QWEN3_0_6B`, `MODEL_ID`, `PINNED_VLLM_VERSION`,
    `ULP_BF16`, `_seeded_token_batches`, `DEFAULT_REPETITIONS`,
    `DEFAULT_BATCH`, `DEFAULT_SEQ_LEN`, `write` -- provenance, geometry, and
    plumbing with no extraction-method dependency.
  - `collective_logits._run_worker`, `collective_logits.WORKER_EXTENSION_QUALNAME`
    -- the one piece of collective_logits.py this module needs is its
    already-verified worker process body (construct `LLM(...)` with
    `worker_extension_cls` and the cache flags below, then call
    `run_one_prompt` once per prompt). Reused as-is rather than duplicated a
    third time.

The differential TP1-vs-TP2 design, the subprocess-per-TP-leg pattern (own
process per `LLM()` -- see bf16_floor.py's module docstring on why: leftover
NCCL/Ray process-group state from one engine is not guaranteed to tear down
cleanly before a second one initializes in the same process), and MEAN
absolute deviation as the primary statistic are all unchanged in shape --
this module's `measure_differential_floor` / `run_break_case` / `_spawn_worker`
are structurally identical to bf16_floor.py's functions of the same name,
just pointed at this module's own `--worker` entry point
(`-m weight_sync_bench.phase2.bf16_floor_v2 --worker`, not bf16_floor.py's)
and keeping `tp1_reference` around for break-case comparison, which
`collective_logits.measure_differential_floor` (a narrower,
extraction-method-comparison-only function, no break cases) does not need
and does not do.

--------------------------------------------------------------------------
WHAT'S DIFFERENT
--------------------------------------------------------------------------
- Extraction: `collective_logits.run_one_prompt` (collective_rpc + a worker
  extension hook on `compute_logits`) instead of
  `SamplingParams(prompt_logprobs=-1)` plus a Python-object scatter loop.
  See `collective_logits.py`'s own docstring (Q1-Q7) for the full mechanism
  and `tolerance/phase2b_extraction.json` for the verification this port
  relies on.
- `enable_chunked_prefill=False, enable_prefix_caching=False` are now baked
  into the `LLM(...)` call `collective_logits._run_worker` makes -- this is
  a MEANINGFUL difference, not just a config change riding along with the
  extraction swap. `bf16_floor.py`'s own `LLM(...)` call leaves both at
  vLLM's defaults (on), and `tolerance/phase2a_bf16_floor.json` records an
  explicit, unresolved caveat that repetition independence under prefix
  caching was never verified there. In this module's artifact, that caveat
  does not apply: independence is guaranteed by construction (prefix
  caching is off), not merely assumed.
- `max_logprobs=-1` and `logprobs_mode="raw_logits"` are gone from the
  `LLM(...)` call. Both were specifically required by the OLD
  `prompt_logprobs=-1` extraction path (`max_logprobs=-1` to lift the
  20-logprob default cap; `logprobs_mode="raw_logits"` to make vLLM's own
  prompt-logprobs machinery report pre-softmax values). Neither has any
  effect on what `run_one_prompt` reads: its hook captures `compute_logits`'s
  raw return value directly, upstream of anything either setting governs,
  and the `SamplingParams(prompt_logprobs=1, ...)` it issues exists only to
  keep the scheduler computing logits at every prompt position -- the actual
  `prompt_logprobs` field on the returned `RequestOutput` is never read.
- New: `--sweep-seq-len` (see below).
- New artifact path: `tolerance/phase2a_bf16_floor_v2.json`, not
  `tolerance/phase2a_bf16_floor.json` -- the old artifact is left exactly as
  it was measured and is not overwritten.

--------------------------------------------------------------------------
SEQ_LEN SWEEP
--------------------------------------------------------------------------
`--sweep-seq-len` (default when passed with no value: `8,32,128,512`) runs
the full floor+break-case+gate pipeline once per seq_len in the list, at
fixed `--repetitions`/`--batch`, and writes one combined report. This is the
question the killed `--repetitions 20 --batch 4 --seq-len 64` run
(`tolerance/phase2a_bf16_floor.json`'s own "SEQ_LEN DEPENDENCE IS UNTESTED"
note) was trying to answer, made affordable by the extraction speedup: the
old path's ~932x-slower-at-seq_len=128 cost would have made a four-point
sweep prohibitive; the new path's near-fixed cost does not scale with it.
Cost and wall-clock duration of an actual sweep run were not verified here
-- no GPU on this machine -- only that the code constructs and writes a
sensible report shape.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .bf16_floor import (
    BREAK_CASES,
    DEFAULT_BATCH,
    DEFAULT_REPETITIONS,
    DEFAULT_SEQ_LEN,
    GATE_MARGIN,
    MODEL_ID,
    QWEN3_0_6B,
    ULP_BF16,
    corrupt_checkpoint,
    derive_threshold,
    environment,
    gate_decision,
    write,
)
from .collective_logits import WORKER_EXTENSION_QUALNAME, _run_worker

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = REPO_ROOT / "tolerance" / "phase2a_bf16_floor_v2.json"

CONFIGURATION = {
    "enforce_eager": True,
    "enable_chunked_prefill": False,
    "enable_prefix_caching": False,
    "worker_extension_cls": WORKER_EXTENSION_QUALNAME,
    "extraction_method": "collective_rpc",
}

SUPERSEDES_NOTE = (
    "Supersedes tolerance/phase2a_bf16_floor.json on extraction path and cache "
    "flags only, while measuring the same quantity (TP1-vs-TP2 mean absolute "
    "logit deviation under bf16, and the same three break-case injections "
    "against it). Extraction: collective_rpc "
    "(weight_sync_bench.phase2.collective_logits.run_one_prompt) instead of "
    "SamplingParams(prompt_logprobs=-1) plus a Python-object scatter loop -- "
    "see tolerance/phase2b_extraction.json for the bit-identity and timing "
    "verification this port relies on. Cache flags: enable_chunked_prefill=False "
    "and enable_prefix_caching=False are now set explicitly (bf16_floor.py "
    "leaves both at vLLM's defaults, i.e. on); this resolves "
    "tolerance/phase2a_bf16_floor.json's own recorded caveat that repetition "
    "independence under prefix caching was unverified there -- in this "
    "artifact independence is guaranteed by construction, not merely assumed. "
    "SAFETY_FACTOR, GATE_MARGIN, the mean-as-primary-statistic rule, the "
    "differential TP1-vs-TP2 design, and every break-case injection are "
    "unchanged -- imported directly from bf16_floor.py, not re-derived. This "
    "artifact's own measurement composition (the full repetitions/TP2/"
    "break-case/gate pipeline run end to end with the new extraction path) "
    "was NOT verified by execution when this module was written -- only the "
    "underlying run_one_prompt primitive was (tolerance/phase2b_extraction.json)."
)


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
            "weight_sync_bench.phase2.bf16_floor_v2",
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


def measure_differential_floor(
    repetitions: int, batch: int, seq_len: int, python: str = sys.executable
) -> dict[str, Any]:
    """Same as bf16_floor.measure_differential_floor: downloads the real
    checkpoint once, runs each TP degree as its own subprocess (this
    module's `--worker`, which is `collective_logits._run_worker` --
    unchanged), and diffs the two saved tensor sets.
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
        "tp1_reference": tp1_reps,
    }


def run_break_case(
    case: str,
    checkpoint_dir: str,
    tp1_reference: list["torch.Tensor"],  # noqa: F821 -- torch imported lazily
    repetitions: int,
    batch: int,
    seq_len: int,
    python: str = sys.executable,
) -> dict[str, Any]:
    """Same as bf16_floor.run_break_case: corrupts a fresh copy of the
    checkpoint (corrupt_checkpoint, unchanged, imported), loads it at TP=2
    through this module's worker, and diffs against the TP1 reference.
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


def _measure_one_config(
    repetitions: int, batch: int, seq_len: int, python: str = sys.executable
) -> dict[str, Any]:
    """Floor + all break cases + gate decision, for one (repetitions, batch,
    seq_len). Factored out so both the single-config CLI path and
    --sweep-seq-len share it.
    """
    floor = measure_differential_floor(repetitions, batch, seq_len, python)
    tp1_reference = floor.pop("tp1_reference")

    break_results = [
        run_break_case(case, floor["checkpoint_dir"], tp1_reference, repetitions, batch, seq_len, python)
        for case in BREAK_CASES
    ]
    gate = gate_decision(floor, break_results)

    return {
        "measurement": {"repetitions": repetitions, "batch": batch, "seq_len": seq_len},
        "floor": floor,
        "break_cases": break_results,
        "gate": gate,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--tp", type=int, default=1, choices=(1, 2))
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--sweep-seq-len",
        type=str,
        nargs="?",
        const="8,32,128,512",
        default=None,
        help=(
            "Comma-separated seq_len values to sweep in one run (default when "
            "passed with no value: 8,32,128,512). Overrides --seq-len. Runs the "
            "full floor+break-case+gate pipeline once per value and writes one "
            "combined report -- see the module docstring's SEQ_LEN SWEEP section."
        ),
    )
    args = parser.parse_args()

    if args.worker:
        assert args.model_dir and args.out
        _run_worker(args.model_dir, args.tp, args.repetitions, args.batch, args.seq_len, args.out)
        return

    if args.sweep_seq_len:
        seq_lens = [int(s) for s in args.sweep_seq_len.split(",")]
        results = [
            _measure_one_config(args.repetitions, args.batch, seq_len) for seq_len in seq_lens
        ]
        report = {
            "phase": "2a",
            "supersedes": SUPERSEDES_NOTE,
            "configuration": CONFIGURATION,
            "environment": environment(),
            "sweep": {
                "repetitions": args.repetitions,
                "batch": args.batch,
                "seq_lens": seq_lens,
            },
            "results": results,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        path = write(report, ARTIFACT)

        print(f"phase 2a bf16 floor sweep (collective_rpc extraction) -> {path}")
        for r in results:
            f, g = r["floor"], r["gate"]
            print(
                f"  seq_len={r['measurement']['seq_len']:<5} "
                f"mean {f['mean_deviation']:.3e} ({f['mean_ulp']:.2f} ULP)  "
                f"threshold {f['threshold']:.3e}  verdict {g['verdict']}"
            )
        return

    result = _measure_one_config(args.repetitions, args.batch, args.seq_len)
    report = {
        "phase": "2a",
        "supersedes": SUPERSEDES_NOTE,
        "configuration": CONFIGURATION,
        "environment": environment(),
        **result,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = write(report, ARTIFACT)

    floor, gate = result["floor"], result["gate"]
    print(f"phase 2a bf16 floor (collective_rpc extraction) -> {path}")
    print(f"  mean deviation : {floor['mean_deviation']:.3e} ({floor['mean_ulp']:.2f} ULP)")
    print(f"  max  deviation : {floor['max_deviation']:.3e} ({floor['max_ulp']:.2f} ULP)")
    print(f"  threshold      : {floor['threshold']:.3e}  (gate margin {GATE_MARGIN}x)")
    for r in result["break_cases"]:
        clears = gate["clears"][r["case"]]
        print(f"  {r['case']:<24} mean {r['mean_deviation']:.3e}  {'CLEARS' if clears else 'does not clear'}")
    print(f"\n  verdict: {gate['verdict']}")


if __name__ == "__main__":
    main()
