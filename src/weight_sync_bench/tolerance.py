"""Tolerance floor measurement and the acceptance threshold derived from it.

PHASE 1a ONLY. The in-process collective reduces in sequential rank order; gloo
need not, so phase 1b requires its own measurement written to its own artifact
(`tolerance/phase1b.json`). Do not reuse this threshold for 1b.

Why this exists: sharded execution changes the numerics. A row-parallel matmul
computes partial products per rank and reduces them, summing in a different order
than the full matmul, so the *correct* sharded result differs from the reference in
the low bits. The threshold must admit that reordering and still catch a real
layout bug. The spec is explicit that guessing gives either flaky passes or break
cases that fail for the wrong reason, so the number is measured and derived, never
hand-picked.

Regenerate with:

    uv run python -m weight_sync_bench.tolerance

The artifact is committed. Regenerate it after a torch or numpy upgrade, since the
recorded environment is the threshold's provenance.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy
import torch

from .model import ReferenceModel
from .reshard import reshard, split_params
from .sharded import InProcessCollective, ShardedModel
from .shardspec import TOY, TOY_KV4, ModelConfig, build_layout_table, supported_degrees

# Assumes a source checkout (the harness is installed editable). src/weight_sync_bench
# -> src -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = REPO_ROOT / "tolerance" / "phase1a.json"
ARTIFACT_1B = REPO_ROOT / "tolerance" / "phase1b.json"

CONFIGS: dict[str, ModelConfig] = {"kv2": TOY, "kv4": TOY_KV4}

REPETITIONS = 20

# LOAD-BEARING. The floor is the maximum over BATCH * SEQ_LEN * vocab elements, and
# the maximum of ~1e6 draws is a saturated tail statistic. More elements means a
# higher maximum. Changing either of these invalidates the threshold and requires
# regeneration; tests assert their token shape against the values recorded here.
BATCH, SEQ_LEN = 2, 16

# Float32 mantissa step at magnitude 1.0. Deviations land on multiples of this, so
# it is the natural unit for reporting the error distribution.
ULP = 2.0**-23

NOTE = [
    "PHASE 1a ONLY. Collectives are simulated in-process and reduce in sequential "
    "rank order. Gloo need not, so phase 1b requires its own measurement written to "
    "tolerance/phase1b.json. Reusing this threshold for 1b is not valid.",
    "THIS IS NOT A GENERAL BOUND. The floor is an input-dependent quantity: the "
    "maximum over batch * seq_len * vocab elements of a per-element error "
    "distribution. The maximum of ~1e6 draws is a saturated tail statistic on a "
    "quantized (2^-24) grid, not a magnitude-driven bound -- the max-deviating "
    "element is typically NOT the largest logit. This is why independently seeded "
    "repetitions agree to several significant figures.",
    "IT DEPENDS ON FOUR INPUTS: d_model and n_layers (which set reduction length and "
    "accumulation depth) and batch and seq_len (which set how many draws the maximum "
    "is taken over). Measured sensitivity: deviation scales ~linearly with d_model "
    "(11.5 -> 25 -> 50 ULP for d_model 256 -> 512 -> 1024), sub-linearly with "
    "n_layers, and only weakly with vocab.",
    "CHANGING d_model, n_layers, batch OR seq_len INVALIDATES THIS THRESHOLD and "
    "requires regeneration. Raising seq_len in a test fixture looks like a harmless "
    "change and is not. tests/test_reshard.py asserts the running config geometry "
    "and token shape against the values recorded here, so such a change fails loudly "
    "instead of silently widening the floor.",
    "median_ulp and mean_ulp characterize the per-element error distribution and do "
    "not saturate, so a reader comparing across changes gets more signal from them "
    "than from the max. The threshold is still derived from the max, which is the "
    "conservative choice.",
    "PHASE 1b HAS ITS OWN ARTIFACT (tolerance/phase1b.json) and its own threshold. "
    "Gloo's reduction order need not match this one; do not reuse this number. Both "
    "phases currently derive 1e-3, but that is the rounding rule mapping two "
    "independent measurements onto the same power of ten, not agreement: kv4 at t=4 "
    "measures 13 ULP here against 12 under gloo.",
    "SAMPLE-COUNT SENSITIVITY, measured directly: raising repetitions from 5 to 20 "
    "(4x the draws) left the threshold unchanged at 1e-3 but raised max_deviation "
    "from 11.5 to 14 ULP, while median_ulp (~1.15) and mean_ulp (~1.39) did not "
    "move. The maximum is an extreme order statistic; it drifts upward with sample "
    "count and is not converged, by nature rather than by defect. That is both the "
    "reason to compare median_ulp/mean_ulp across changes rather than the max, and "
    "the reason the threshold carries a 100x safety factor over it.",
]

# The rule. Stated here, recorded in the artifact, and applied by `derive_threshold`
# so a reader can check the number against the measurement themselves.
SAFETY_FACTOR = 100
RULE = (
    "threshold = the smallest power of ten >= SAFETY_FACTOR * max observed deviation, "
    "across every measured cell and every repetition"
)
# Floor for the degenerate all-zeros case, which would make log10 undefined.
MIN_THRESHOLD = 1e-9

# How far a deviation must clear the threshold before it counts as a detected
# layout error rather than a tolerance set slightly too tight. The threshold is
# already SAFETY_FACTOR x the measured floor, so this puts a real bug at >= 1e4 x
# the floor -- the "three to six orders of magnitude" separation the spec expects.
#
# It lives here, beside the threshold it multiplies, because two consumers need
# the same number: the break cases assert their injected layout errors clear it,
# and the gate-resolution sweep asks how many optimizer steps a missed sync must
# skip before it clears the same bar. `src` importing from `tests` is not an
# option, and a second copy of a required separation is a second thing to drift.
MIN_SEPARATION = 100.0


class EnvironmentMismatch(UserWarning):
    """The artifact was measured under a different environment than the one running.

    A warning, never an error. A stale environment makes the threshold's provenance
    weaker, not the threshold wrong, and hard-failing on a routine version bump is a
    bad trade for a repo strangers clone.
    """


# Versions that plausibly change floating-point results. `platform` and `machine`
# are recorded for provenance but not compared: they differ across every
# contributor's machine and would warn constantly.
COMPARED_ENVIRONMENT_KEYS = ("torch", "numpy", "python")


def environment_mismatches(report: dict[str, object] | None = None) -> list[str]:
    """Human-readable differences between the recorded and running environments."""
    recorded = (report or load())["environment"]
    current = environment()
    return [
        f"{key}: measured under {recorded.get(key)!r}, running {current[key]!r}"
        for key in COMPARED_ENVIRONMENT_KEYS
        if recorded.get(key) != current[key]
    ]


def derive_threshold(max_deviation: float, safety_factor: int = SAFETY_FACTOR) -> float:
    """Apply RULE. Pure function of the measurement, so it is auditable."""
    if max_deviation <= 0.0:
        return MIN_THRESHOLD
    return float(10 ** math.ceil(math.log10(safety_factor * max_deviation)))


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def environment() -> dict[str, object]:
    return {
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dtype": "torch.float32",
        "device": "cpu",
        "git_commit": _git_commit(),
    }


def measure(repetitions: int = REPETITIONS) -> dict[str, object]:
    """Run the correct sharded path over every config and ordered pair."""
    cells: dict[tuple[str, int, int], list[dict[str, float]]] = {}

    for name, config in CONFIGS.items():
        degrees = supported_degrees(config)
        for rep in range(repetitions):
            reference = ReferenceModel(config, seed=rep)
            tokens = torch.randint(
                0,
                config.vocab,
                (BATCH, SEQ_LEN),
                generator=torch.Generator().manual_seed(10_000 + rep),
            )
            expected = reference(tokens)
            full = reference.full_params()

            for src, dst in itertools.product(degrees, repeat=2):
                src_table = build_layout_table(config, src)
                dst_table = build_layout_table(config, dst)
                moved = reshard(split_params(full, src_table), src_table, dst_table)
                actual = ShardedModel(config, moved, InProcessCollective(dst))(tokens)
                diff = (actual - expected).abs()
                cells.setdefault((name, src, dst), []).append(
                    {
                        "max": diff.max().item(),
                        "median": diff.median().item(),
                        "mean": diff.mean().item(),
                    }
                )

    results = []
    for (name, src, dst), reps in sorted(cells.items()):
        maxima = [rep["max"] for rep in reps]
        results.append(
            {
                "config": name,
                "n_kv_heads": CONFIGS[name].n_kv_heads,
                "src": src,
                "dst": dst,
                # The threshold derives from this. Saturated tail statistic.
                "max_deviation": max(maxima),
                "max_ulp": max(maxima) / ULP,
                # These characterize the distribution and do not saturate. Averaged
                # over repetitions.
                "median_ulp": sum(rep["median"] for rep in reps) / len(reps) / ULP,
                "mean_ulp": sum(rep["mean"] for rep in reps) / len(reps) / ULP,
                "per_seed": maxima,
            }
        )
    max_deviation = max(row["max_deviation"] for row in results)

    return {
        "phase": "1a",
        "note": NOTE,
        "ulp": ULP,
        # Geometry the floor depends on. Asserted against the running configs by
        # tests/test_reshard.py.
        "configs": {name: asdict(config) for name, config in CONFIGS.items()},
        "rule": {
            "description": RULE,
            "safety_factor": SAFETY_FACTOR,
            "expression": "10 ** ceil(log10(safety_factor * max_deviation))",
        },
        "threshold": derive_threshold(max_deviation),
        "max_deviation": max_deviation,
        "measurement": {
            "repetitions": repetitions,
            "model_seeds": list(range(repetitions)),
            "token_seeds": [10_000 + rep for rep in range(repetitions)],
            "batch": BATCH,
            "seq_len": SEQ_LEN,
        },
        "environment": environment(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results": results,
    }


NOTE_1B = [
    "PHASE 1b. One process per rank, real torch.distributed all_reduce / all_gather "
    "over gloo. Measured separately from phase 1a because gloo's reduction order need "
    "not match the in-process simulation's; the 1a threshold is not valid here and "
    "vice versa.",
    "Measured per (config, tp_degree) rather than per ordered pair. The src layout "
    "cannot affect the result: gather(split(full, src)) == full byte-exactly, so "
    "logits under the dst layout depend only on dst. Ordered pairs are covered in "
    "phase 1a, where the reshard path itself runs.",
    "Every rank computes its own statistics and they are asserted equal before being "
    "recorded -- after the final all_gather each rank holds the same logits, so "
    "disagreement would mean the collective is not replicating the result.",
    "THE 1a AND 1b THRESHOLDS BOTH LAND ON 1e-3. That is NOT agreement between the "
    "phases -- it is an artifact of the rounding rule mapping two independent "
    "measurements to the same power of ten. The underlying measurements differ: kv4 "
    "at t=4 is 12 ULP under gloo against 13 in-process. That difference is the "
    "justification for measuring separately, and it is why neither threshold may be "
    "reused for the other phase even while the numbers coincide. A future change to "
    "either backend can move them apart without warning.",
    "The same input-dependence applies as in phase 1a: the floor depends on d_model, "
    "n_layers, batch and seq_len, and the max is a saturated tail statistic rather "
    "than a magnitude-driven bound. Compare median_ulp and mean_ulp across changes.",
]


def measure_1b(repetitions: int = REPETITIONS) -> dict[str, object]:
    """Run the correct sharded path under gloo, one process per rank."""
    from .distributed import run_cell

    seeds = list(range(repetitions))
    results = []

    for name, config in CONFIGS.items():
        tokens = [
            torch.randint(
                0,
                config.vocab,
                (BATCH, SEQ_LEN),
                generator=torch.Generator().manual_seed(10_000 + rep),
            )
            for rep in seeds
        ]
        for degree in supported_degrees(config):
            rows = run_cell(config, degree, seeds, tokens)
            maxima = [row["max"] for row in rows]
            results.append(
                {
                    "config": name,
                    "n_kv_heads": config.n_kv_heads,
                    "tp_degree": degree,
                    "max_deviation": max(maxima),
                    "max_ulp": max(maxima) / ULP,
                    "median_ulp": sum(r["median"] for r in rows) / len(rows) / ULP,
                    "mean_ulp": sum(r["mean"] for r in rows) / len(rows) / ULP,
                    "per_seed": maxima,
                }
            )

    max_deviation = max(row["max_deviation"] for row in results)
    return {
        "phase": "1b",
        "note": NOTE_1B,
        "ulp": ULP,
        "configs": {name: asdict(config) for name, config in CONFIGS.items()},
        "rule": {
            "description": RULE,
            "safety_factor": SAFETY_FACTOR,
            "expression": "10 ** ceil(log10(safety_factor * max_deviation))",
        },
        "threshold": derive_threshold(max_deviation),
        "max_deviation": max_deviation,
        "measurement": {
            "repetitions": repetitions,
            "model_seeds": seeds,
            "token_seeds": [10_000 + rep for rep in seeds],
            "batch": BATCH,
            "seq_len": SEQ_LEN,
            "backend": "gloo",
            "processes_per_cell": "one per rank",
        },
        "environment": environment(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results": results,
    }


def write(report: dict[str, object], path: Path = ARTIFACT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path


def load(path: Path = ARTIFACT) -> dict[str, object]:
    """Read the committed measurement."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Regenerate with: "
            "uv run python -m weight_sync_bench.tolerance"
        )
    return json.loads(path.read_text())


def load_threshold(path: Path = ARTIFACT) -> float:
    """The acceptance threshold. Measured and derived -- never edit it by hand."""
    return float(load(path)["threshold"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", choices=("1a", "1b"), default="1a")
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.phase == "1b":
        report = measure_1b(args.repetitions)
        path = write(report, args.out or ARTIFACT_1B)
    else:
        report = measure(args.repetitions)
        path = write(report, args.out or ARTIFACT)

    print(f"phase {args.phase} tolerance floor -> {path}")
    print(f"  {'cell':>12}   {'max':>10}  {'max':>7} {'median':>7} {'mean':>7}   (ULP)")
    for row in report["results"]:
        cell = (
            f"{row['config']:>4} {row['src']} -> {row['dst']}"
            if "src" in row
            else f"{row['config']:>4}      t={row['tp_degree']}"
        )
        print(
            f"  {cell}   {row['max_deviation']:.3e}  {row['max_ulp']:7.2f} "
            f"{row['median_ulp']:7.2f} {row['mean_ulp']:7.2f}"
        )
    print(f"\n  worst observed : {report['max_deviation']:.3e}")
    print(f"  rule           : {RULE}")
    print(f"  threshold      : {report['threshold']:.3e}")


if __name__ == "__main__":
    main()
