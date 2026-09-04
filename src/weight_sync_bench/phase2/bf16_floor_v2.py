"""bf16 floor measurement, ported onto collective_logits.run_one_prompt for
extraction. SPEC.md phase 2, section 2a, extraction-path revision.

VERIFIED END TO END on the real GPU box -- see `tolerance/phase2a_bf16_floor_v2.json`
for the full record. Three runs: a smoke test (`--repetitions 1 --batch 1
--seq-len 8`, mean_deviation 3.038e-02), a reproduction of
`tolerance/phase2a_bf16_floor.json`'s exact recorded configuration
(`--repetitions 20 --batch 2 --seq-len 8`), and a seq_len sweep
(`--repetitions 20 --batch 4 --sweep-seq-len 8,32,128,512`). The
reproduction is the load-bearing result: mean_deviation 3.898e-02,
max_deviation 8.125e-01 (104.00 ULP), and break-case means 1.772 / 1.570 /
2.984 -- IDENTICAL to `tolerance/phase2a_bf16_floor.json`'s recorded floor
to four significant figures on every one of those six numbers, with the
full repetitions loop, both TP legs, and all three break-case checkpoint
loads exercised, not just the single-prompt primitive. That reproduction is
itself the confirmation that TP=2's all-gather bit-identity assertion in
`collective_logits.run_one_prompt` works correctly: it fires on every TP=2
call this run made (the floor's TP=2 leg and all three break-case loads),
and a mismatch there raises immediately by design, so completing cleanly
with matching numbers means it passed every time. The sweep additionally
establishes that the floor's mean is invariant across a 64x seq_len range
(8 to 512) -- see `tolerance/phase2a_bf16_floor_v2.json`'s
`seq_len_dependence` block, which supersedes the seq_len-dependence caveat
`tolerance/phase2a_bf16_floor.json` previously carried as open.

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
- New: `--tp {1,2,4}` (was `{1,2}`, and was silently unread at the top level
  -- every prior run of this module used TP=2 unconditionally regardless of
  what `--tp` was passed, since only `--worker` mode ever consulted it) and
  `--layer` (new; forwards to `corrupt_checkpoint`, which has always
  accepted a `layer` argument -- no change there). `measure_differential_
  floor`/`run_break_case`/`_measure_one_config` all take `tp`/`layer` now,
  default `tp=2, layer=0` so every prior recorded run reproduces unchanged.
  `--tp 4` checks SAFETY_FACTOR=15/GATE_MARGIN=2 at a third configuration
  axis the two recorded separation-ratio bands (batch=2/seq_len=8 and
  batch=4/seq_len=128, both TP=2 -- see the SAFETY_FACTOR comment in
  bf16_floor.py) never covered; `--layer N` (N != 0) checks whether the
  break cases' layer-0 hardcoding was a scope choice or a hidden dependency
  (every layer shares the same geometry/placement, so a materially
  different magnitude at another layer would be a finding, not expected).
  Non-default `(tp, layer)` combinations write to a suffixed artifact path
  (`_artifact_path`) rather than overwriting the canonical
  `tolerance/phase2a_bf16_floor_v2.json` record.

--------------------------------------------------------------------------
SEQ_LEN SWEEP
--------------------------------------------------------------------------
`--sweep-seq-len` (default when passed with no value: `8,32,128,512`) runs
the full floor+break-case+gate pipeline once per seq_len in the list, at
fixed `--repetitions`/`--batch`, and writes one combined report. This is the
question the killed `--repetitions 20 --batch 4 --seq-len 64` run
(`tolerance/phase2a_bf16_floor.json`'s own former "SEQ_LEN DEPENDENCE IS
UNTESTED" note, since retired) was trying to answer, made affordable by the
extraction speedup: the old path's ~932x-slower-at-seq_len=128 cost would
have made a four-point sweep prohibitive; the new path's near-fixed cost
does not scale with it. RUN ON THE BOX: `--repetitions 20 --batch 4
--sweep-seq-len 8,32,128,512` completed and passed the gate at every point
-- see `tolerance/phase2a_bf16_floor_v2.json`'s `sweep` and
`seq_len_dependence` blocks for the full numbers, including that the mean is
measured invariant across the whole 64x range (design's predicted
behavior). Exact per-seq_len wall-clock timing for the sweep itself was not
captured in what was reported into this session, so is not recorded here,
though it plainly completed in a practical amount of time (unlike the
killed old-path run at a single seq_len this sweep's largest point exceeds
by 8x).
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
    "leaves both at vLLM's defaults, i.e. on); this resolved "
    "tolerance/phase2a_bf16_floor.json's own recorded caveat that repetition "
    "independence under prefix caching was unverified there -- in this "
    "artifact independence is guaranteed by construction, not merely assumed, "
    "AND a same-configuration reproduction run (tolerance/phase2a_bf16_floor_v2.json's "
    "'reproduction' block) confirmed the two are numerically identical to four "
    "significant figures, so caching being on during the original measurement "
    "demonstrably did not affect the floor. SAFETY_FACTOR, GATE_MARGIN, the "
    "mean-as-primary-statistic rule, the differential TP1-vs-TP2 design, and "
    "every break-case injection are unchanged -- imported directly from "
    "bf16_floor.py, not re-derived. This module's own measurement composition "
    "(the full repetitions/TP2/break-case/gate pipeline run end to end with "
    "the new extraction path) has been verified by execution: a reproduction "
    "run reproduced tolerance/phase2a_bf16_floor.json's recorded floor and "
    "break-case means to four significant figures, and a seq_len sweep "
    "(8/32/128/512) passed the gate at every point -- see "
    "tolerance/phase2a_bf16_floor_v2.json. A run at OTHER parameters than "
    "those two (a different model, dtype, or far outside the swept seq_len "
    "range) is not covered by that verification and should be treated with "
    "the same caution as any first run."
)


def _spawn_worker(
    python: str,
    model_dir: str,
    tp: int,
    repetitions: int,
    batch: int,
    seq_len: int,
    out_path: Path,
    seed_base: int = 0,
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
            "--seed-base",
            str(seed_base),
        ],
        check=True,
    )


def measure_differential_floor(
    repetitions: int,
    batch: int,
    seq_len: int,
    python: str = sys.executable,
    seed_base: int = 0,
    tp: int = 2,
) -> dict[str, Any]:
    """Same as bf16_floor.measure_differential_floor: downloads the real
    checkpoint once, runs each TP degree as its own subprocess (this
    module's `--worker`, which is `collective_logits._run_worker` --
    unchanged), and diffs the two saved tensor sets. `seed_base` is forwarded
    to both TP legs unchanged -- see `collective_logits._run_worker`'s
    docstring for what it shifts and why `seed_base=0` (the default)
    reproduces every artifact recorded before this parameter existed.

    `tp` is the non-1 degree the TP=1 reference is diffed against -- default
    2 reproduces every artifact recorded before this parameter existed
    (`tolerance/phase2a_bf16_floor_v2.json`'s runs all implicitly used TP=2,
    hardcoded). Passing `tp=4` checks the SAME threshold/statistic at a third
    configuration axis the two recorded separation-ratio bands (batch=2/
    seq_len=8 and batch=4/seq_len=128, both at TP=2) never covered.
    """
    import tempfile

    import torch
    from huggingface_hub import snapshot_download

    checkpoint_dir = snapshot_download(MODEL_ID)

    cells: list[dict[str, float]] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tp1_path, tpN_path = tmp / "tp1.pt", tmp / f"tp{tp}.pt"
        for degree, out_path in ((1, tp1_path), (tp, tpN_path)):
            _spawn_worker(python, checkpoint_dir, degree, repetitions, batch, seq_len, out_path, seed_base)

        tp1_reps = torch.load(tp1_path)
        tpN_reps = torch.load(tpN_path)

        for tp1_logits, tpN_logits in zip(tp1_reps, tpN_reps):
            diff = (tpN_logits - tp1_logits).abs()
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
    seed_base: int = 0,
    tp: int = 2,
    layer: int = 0,
) -> dict[str, Any]:
    """Same as bf16_floor.run_break_case: corrupts a fresh copy of the
    checkpoint (corrupt_checkpoint, unchanged, imported), loads it at TP
    degree `tp` through this module's worker, and diffs against the TP1
    reference. `seed_base` must match the value `tp1_reference` was generated
    with, or this is comparing draws from different tokens/model-seed state
    rather than the same one under a genuine layout corruption --
    `_measure_one_config` passes the same `seed_base` to both, so this only
    matters if calling `run_break_case` directly.

    `tp` used to be hardcoded to 2 -- generalized so the break-case leg runs
    at the SAME degree `measure_differential_floor` was called with, rather
    than silently always TP=2 regardless of what the floor measurement used
    (that mismatch would compare a break case injected at one degree against
    a floor measured at another, which is not the differential this module
    is for).

    `layer` forwards to `corrupt_checkpoint` (which has always accepted it,
    default 0 -- no change needed there). Every layer shares the same
    geometry and placement (SPEC.md 2b), so a non-zero layer is expected to
    produce a materially similar break magnitude to layer 0; this parameter
    exists to check that expectation rather than assume it.
    """
    import tempfile

    import torch

    with tempfile.TemporaryDirectory() as tmp:
        corrupted_dir = Path(tmp) / case
        corrupt_checkpoint(checkpoint_dir, corrupted_dir, case, layer=layer)

        out_path = Path(tmp) / f"{case}.pt"
        _spawn_worker(python, str(corrupted_dir), tp, repetitions, batch, seq_len, out_path, seed_base)
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
    repetitions: int,
    batch: int,
    seq_len: int,
    python: str = sys.executable,
    seed_base: int = 0,
    tp: int = 2,
    layer: int = 0,
) -> dict[str, Any]:
    """Floor + all break cases + gate decision, for one (repetitions, batch,
    seq_len, tp, layer). Factored out so both the single-config CLI path and
    --sweep-seq-len share it. `seed_base=0` reproduces every artifact
    recorded before this parameter existed -- see
    `collective_logits._run_worker`'s docstring. `tp=2, layer=0` reproduces
    every artifact recorded before those two parameters existed.
    """
    floor = measure_differential_floor(repetitions, batch, seq_len, python, seed_base, tp)
    tp1_reference = floor.pop("tp1_reference")

    break_results = [
        run_break_case(
            case,
            floor["checkpoint_dir"],
            tp1_reference,
            repetitions,
            batch,
            seq_len,
            python,
            seed_base,
            tp,
            layer,
        )
        for case in BREAK_CASES
    ]
    gate = gate_decision(floor, break_results)

    return {
        "measurement": {
            "repetitions": repetitions,
            "batch": batch,
            "seq_len": seq_len,
            "seed_base": seed_base,
            "tp": tp,
            "layer": layer,
        },
        "floor": floor,
        "break_cases": break_results,
        "gate": gate,
    }


def _artifact_path(tp: int, layer: int) -> Path:
    """`ARTIFACT` (`tolerance/phase2a_bf16_floor_v2.json`) is the canonical
    recorded file for the tp=2/layer=0 configuration every prior run of this
    module used -- returned unchanged here so nothing about that file's path
    or the meaning of a bare `write(report, ARTIFACT)` call changes. Any
    OTHER (tp, layer) combination gets its own suffixed path instead of
    silently overwriting that canonical record; `--tp 4` and `--layer 13`
    runs are exploratory checks against the recorded bands, not replacements
    for them.
    """
    if tp == 2 and layer == 0:
        return ARTIFACT
    suffix = "".join(
        (f"_tp{tp}" if tp != 2 else "", f"_layer{layer}" if layer != 0 else "")
    )
    return ARTIFACT.with_name(ARTIFACT.stem + suffix + ARTIFACT.suffix)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model-dir", type=str, default=None)
    # Default 2, not 1: unlike param_layout_inspection.py's --tp (which is
    # genuinely dead at that module's top level, because that module always
    # sweeps every degree together), THIS --tp is meaningful at the top
    # level too -- it selects the one non-1 degree the differential and
    # break cases run against, a single-degree-at-a-time granularity this
    # module already has (one seq_len/batch per invocation, externally
    # looped via --sweep-seq-len). The fix here is to wire it up, not
    # reject it -- see the module docstring's TP degree note.
    parser.add_argument("--tp", type=int, default=2, choices=(1, 2, 4))
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help=(
            "Layer index the break-case corruption targets (forwarded to "
            "bf16_floor.corrupt_checkpoint, which has always accepted this -- "
            "no change needed there). Default 0 reproduces every artifact "
            "recorded before this flag existed. Every layer shares the same "
            "geometry and placement, so a non-zero layer is expected to "
            "produce a materially similar break magnitude to layer 0; this "
            "flag exists to check that, not assume it."
        ),
    )
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
    parser.add_argument(
        "--seed-base",
        type=int,
        default=0,
        help=(
            "Shifts both seed axes for a discriminating rerun: token seed "
            "becomes 10_000 + seed_base + rep, model seed (torch.manual_seed, "
            "collective_logits._run_worker) becomes seed_base + rep. Default 0 "
            "reproduces every artifact recorded before this flag existed -- see "
            "collective_logits._run_worker's docstring for exactly what this "
            "does and does not change under this harness's greedy-decoding "
            "configuration."
        ),
    )
    args = parser.parse_args()

    if args.worker:
        assert args.model_dir and args.out
        _run_worker(
            args.model_dir, args.tp, args.repetitions, args.batch, args.seq_len, args.out, args.seed_base
        )
        return

    if args.sweep_seq_len:
        seq_lens = [int(s) for s in args.sweep_seq_len.split(",")]
        results = [
            _measure_one_config(
                args.repetitions,
                args.batch,
                seq_len,
                seed_base=args.seed_base,
                tp=args.tp,
                layer=args.layer,
            )
            for seq_len in seq_lens
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
                "seed_base": args.seed_base,
                "tp": args.tp,
                "layer": args.layer,
            },
            "results": results,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        path = write(report, _artifact_path(args.tp, args.layer))

        print(f"phase 2a bf16 floor sweep (collective_rpc extraction) -> {path}")
        for r in results:
            f, g = r["floor"], r["gate"]
            print(
                f"  seq_len={r['measurement']['seq_len']:<5} "
                f"mean {f['mean_deviation']:.3e} ({f['mean_ulp']:.2f} ULP)  "
                f"threshold {f['threshold']:.3e}  verdict {g['verdict']}"
            )
        return

    result = _measure_one_config(
        args.repetitions,
        args.batch,
        args.seq_len,
        seed_base=args.seed_base,
        tp=args.tp,
        layer=args.layer,
    )
    report = {
        "phase": "2a",
        "supersedes": SUPERSEDES_NOTE,
        "configuration": CONFIGURATION,
        "environment": environment(),
        **result,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = write(report, _artifact_path(args.tp, args.layer))

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
