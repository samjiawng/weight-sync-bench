"""How stale a sampler's weights must be before the correctness gate can tell.

CPU, toy model, phase 1a's threshold. Regenerate with:

    uv run python -m weight_sync_bench.phase2.gate_resolution

WHAT THIS ANSWERS, AND WHY IT IS NOT A CONSTANT
------------------------------------------------
SPEC 2e runs the invariant at every configuration and says a timing measurement
of a sync that delivered wrong weights is worthless. That is only true if a sync
which delivered NOTHING would fail the gate, and nothing checked it. With a
source moving by a small per-step update, a transport that silently re-delivers
the previous step's weights can pass, because one step moves the logits less than
the floor admits -- and then every timing 2e records is gated by a check that
cannot fail. That is the same shape as the sabotage result already recorded for
phase 1, where zeroing `o_proj`'s output left all three break cases green.

The useful answer is not a magic update magnitude. It is a RESOLUTION IN STEPS:
the smallest number of skipped steps at which the gate fires, in the units 2e and
phase 3 actually work in. Both outcomes are results and the artifact says which
one it got. If the gate fires at k=1 at a realistic magnitude, 2e's per-sync gate
resolves a single missed sync. If it does not, 2e has to state that its gate
catches a transport failing for k or more CONSECUTIVE syncs, which is a real
limitation of the correctness story and worth more than a number that flatters it.

THE REFERENCE IS THE STATE THE SYNC WAS INTENDED TO DELIVER
------------------------------------------------------------
Not the source's current state. Deliberately stale weights and a broken transport
are the same bytes on the sampler, and the two have to come out differently:
phase 3 varies staleness on purpose and needs the gate quiet, 2e needs it to fire
on a transport that delivered nothing. Both hold only under this choice of
reference. A correct stale sampler is compared against the state it was meant to
reach and passes; a sampler that received nothing holds an earlier state than
intended and fails. Getting this backwards produces a measurement that looks fine
and is wrong.

THE STATISTIC IS THE MAX, BECAUSE THE THRESHOLD WAS DERIVED UNDER THE MAX
--------------------------------------------------------------------------
`tolerance.load_threshold()` is read, never re-derived and never inlined. Phase
1a's threshold is the smallest power of ten at or above 100x the worst observed
MAXIMUM absolute logit error; comparing a mean against it would be the same error
the flag-profile rule exists to prevent.

WHY THE SOURCE LAYOUT IS A CONTROL AND NOT A SWEPT AXIS
--------------------------------------------------------
A step adds noise elementwise, and `split`/`gather` is a byte-exact round trip, so
the state a source intends to deliver provably cannot depend on how that source
held it. Sweeping the layout across the whole grid would multiply the cells
without adding an axis. It is measured instead at the reference magnitude, as a
control that RECORDS the independence rather than assuming it -- including a
storage source at a degree where no execution table exists at all, which is the
case the layout role was added for.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from ..model import ReferenceModel
from ..reshard import split_params
from ..sharded import InProcessCollective, ShardedModel
from ..shardspec import (
    TOY,
    TOY_KV4,
    ModelConfig,
    build_layout_table,
    build_storage_table,
    supported_degrees,
)
from ..tolerance import (
    ARTIFACT as PHASE1A_ARTIFACT,
    BATCH,
    MIN_SEPARATION,
    REPO_ROOT,
    SEQ_LEN,
    environment,
    load_threshold,
)
from .param_source import ShardedParamSource

ARTIFACT = REPO_ROOT / "tolerance" / "phase2c_gate_resolution.json"

CONFIGS: dict[str, ModelConfig] = {"kv2": TOY, "kv4": TOY_KV4}

MODEL_SEED = 0
TOKEN_SEED = 20_000
SOURCE_SEED = 7

# How many skipped steps to sweep. The answer is the smallest k in this range at
# which the gate fires; "did not fire within the range" is a result, and the
# artifact records the range so that result is readable rather than ambiguous.
K_MAX = 8

# Pinned for the duration of the measurement, then restored. Two reasons, and
# neither is only speed. Determinism: single-threaded matmuls reduce in a fixed
# order, so this artifact reproduces on a machine with a different core count.
# Speed: the same cell takes ~15s at 128 ambient threads and ~0.2s at one, since
# the toy tensors are far too small to amortize the fork/join, measured on the
# box this was generated on.
#
# It does not put the correct-sync leg at risk of reading differently from phase
# 1a, whose floor was measured at ambient threads: thread count moves the last
# bits, and the correct leg sits about three orders of magnitude under the
# threshold it is checked against.
MEASUREMENT_THREADS = 1

# Straddles the reference magnitude by two orders in each direction.
MAGNITUDES = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)

# Geometric, at the reference magnitude only, because the boundary there is far
# outside any grid worth sweeping linearly. Staleness accumulates as a random
# walk while the gate is a fixed absolute bar, so k buys a decade of deviation
# only per hundredfold increase -- this measures that exponent instead of
# asserting it, and the extrapolation to the crossing point is marked as one.
GROWTH_K = (1, 2, 4, 8, 16, 32, 64, 128, 256)

# EXTRAPOLATED, not measured. See REFERENCE_MAGNITUDE_REASONING; there is no
# training run behind this number and the artifact marks it as such. The
# deliverable is where the resolution boundary sits RELATIVE to it.
REFERENCE_MAGNITUDE = 1e-4
REFERENCE_MAGNITUDE_REASONING = (
    "An Adam update has magnitude roughly the learning rate per element, since "
    "the update is the normalized moment ratio scaled by lr. At lr 1e-5 against "
    "toy weights whose RMS is about 1/sqrt(d_model), the per-step change relative "
    "to the weight scale is order 1e-4. EXTRAPOLATED from that arithmetic, not "
    "measured from a training run: no optimizer, gradient or loss exists in this "
    "harness, and inventing one to produce this number would be a larger change "
    "than the question warrants. Treat it as the place on the swept axis a real "
    "run is expected to sit, not as a measurement."
)

NOTE = [
    "WHAT IS MEASURED: the smallest number of optimizer steps a null sync must "
    "skip before phase 1a's correctness gate fires on it. A null sync is a "
    "transport that delivered nothing, so the sampler still holds params_0 while "
    "the source has advanced to params_k.",
    "THE REFERENCE IS THE INTENDED STATE, never the source's current state. "
    "Deliberately stale weights and a broken transport are the same bytes; phase "
    "3 needs the gate quiet on the first and 2e needs it loud on the second, and "
    "only this choice of reference gives both.",
    "THE STATISTIC IS THE MAXIMUM absolute logit error, because phase 1a's "
    "threshold was derived under the maximum. The threshold is READ from "
    "tolerance/phase1a.json and is neither re-derived nor inlined here.",
    "THE GATE FIRES when the null deviation exceeds MIN_SEPARATION * threshold, "
    "which is the same bar the break cases must clear. A null sync that merely "
    "exceeded the threshold would be indistinguishable from a tolerance set "
    "slightly too tight.",
    "SAME INPUT-DEPENDENCE AS PHASE 1a. The deviations are maxima over "
    "batch * seq_len * vocab elements, so d_model, n_layers, batch and seq_len "
    "all move them. The geometry and token shape are recorded and checked against "
    "the running values, and this artifact must be regenerated if any of them "
    "changes.",
    "THE SOURCE LAYOUT IS A CONTROL, not a swept axis: a step adds noise "
    "elementwise and split/gather is byte-exact, so the intended state cannot "
    "depend on how the source held it. Recorded at the reference magnitude "
    "instead, including a storage source at a degree where no execution table "
    "exists.",
    "NOT AN RL RESULT. There is no optimizer, gradient or loss here. A step is "
    "modelled as noise scaled relative to each parameter's RMS, because the only "
    "property of a training step this question depends on is that the weights "
    "move by a small amount per step.",
]


def _tokens(config: ModelConfig) -> torch.Tensor:
    return torch.randint(
        0,
        config.vocab,
        (BATCH, SEQ_LEN),
        generator=torch.Generator().manual_seed(TOKEN_SEED),
    )


def _source_layouts(config: ModelConfig) -> list[tuple[str, Any]]:
    """The source layouts the control compares, canonical one first.

    The storage table at 4 is the case T002a's role field exists for: at `kv2` no
    execution table can be built there at all, so before the role a source simply
    could not shard that way.
    """
    return [
        ("storage@t4", build_storage_table(config, 4)),
        ("storage@t2", build_storage_table(config, 2)),
        ("execution@t2", build_layout_table(config, 2)),
    ]


def _deviation(expected: torch.Tensor, actual: torch.Tensor) -> float:
    return float((actual - expected).abs().max())


def _sharded_logits(
    config: ModelConfig, full: dict[str, torch.Tensor], dst: int, tokens: torch.Tensor
) -> torch.Tensor:
    table = build_layout_table(config, dst)
    return ShardedModel(config, split_params(full, table), InProcessCollective(dst))(
        tokens
    )


def measure(
    k_max: int = K_MAX, magnitudes: tuple[float, ...] = MAGNITUDES
) -> dict[str, Any]:
    """Sweep skipped steps against update magnitude, per config and dst degree."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(MEASUREMENT_THREADS)
    try:
        return _measure(k_max, magnitudes)
    finally:
        # Restored rather than left set: a measurement that silently reconfigured
        # the process it was called from would be a side effect a caller cannot
        # see, and the tests call this in the same process as everything else.
        torch.set_num_threads(previous_threads)


def _measure(k_max: int, magnitudes: tuple[float, ...]) -> dict[str, Any]:
    threshold = load_threshold()
    gate = MIN_SEPARATION * threshold

    cells: list[dict[str, Any]] = []
    control: list[dict[str, Any]] = []
    growth: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    for name, config in CONFIGS.items():
        tokens = _tokens(config)
        degrees = supported_degrees(config)
        model = ReferenceModel(config, seed=MODEL_SEED)
        # Cloned. `full_params()` hands back the live parameter tensors, and this
        # same model is reloaded with params_k below; without the clone the
        # null leg's "state the sampler still holds" would advance with the
        # source and the measurement would compare params_k against itself.
        params_0 = {name_: t.detach().clone() for name_, t in model.full_params().items()}

        # The null sampler holds params_0 under the destination layout at every
        # cell, so its logits are computed once per (config, dst) rather than per
        # magnitude and k.
        null_logits = {
            dst: _sharded_logits(config, params_0, dst, tokens) for dst in degrees
        }

        canonical_label, canonical_table = _source_layouts(config)[0]
        for magnitude in magnitudes:
            source = ShardedParamSource(params_0, canonical_table, seed=SOURCE_SEED)
            for k in range(1, k_max + 1):
                source.step(magnitude)
                params_k = source.intended_state()
                model.load_state_dict(params_k)
                expected = model(tokens)

                for dst in degrees:
                    correct = _deviation(
                        expected, _sharded_logits(config, params_k, dst, tokens)
                    )
                    null = _deviation(expected, null_logits[dst])
                    row = {
                        "config": name,
                        "source_layout": canonical_label,
                        "dst_degree": dst,
                        "magnitude": magnitude,
                        "skipped_steps": k,
                        "correct_max_deviation": correct,
                        "null_max_deviation": null,
                        "correct_sync_passes": correct < threshold,
                        "gate_fires": null > gate,
                    }
                    # A cell whose CORRECT leg fails is a broken sync path, and
                    # the null result beside it is then not attributable to the
                    # missed update. Reported and excluded rather than averaged.
                    if not row["correct_sync_passes"]:
                        excluded.append(row)
                    else:
                        cells.append(row)

        # How the null deviation grows with k at the reference magnitude. The
        # linear grid above establishes that the gate does not fire there within
        # k_max; this says how far outside the boundary actually is, which is the
        # number 2e needs and "did not fire" does not give.
        dst = max(degrees)
        source = ShardedParamSource(params_0, canonical_table, seed=SOURCE_SEED)
        applied = 0
        for target in GROWTH_K:
            while applied < target:
                source.step(REFERENCE_MAGNITUDE)
                applied += 1
            model.load_state_dict(source.intended_state())
            deviation = _deviation(model(tokens), null_logits[dst])
            growth.append(
                {
                    "config": name,
                    "dst_degree": dst,
                    "magnitude": REFERENCE_MAGNITUDE,
                    "skipped_steps": applied,
                    "null_max_deviation": deviation,
                    "exceeds_threshold": deviation > threshold,
                    "gate_fires": deviation > gate,
                }
            )

        # The control: the same measurement under every source layout, at the
        # reference magnitude, at the largest destination degree.
        dst = max(degrees)
        for label, table in _source_layouts(config):
            source = ShardedParamSource(params_0, table, seed=SOURCE_SEED)
            for k in range(1, k_max + 1):
                source.step(REFERENCE_MAGNITUDE)
            params_k = source.intended_state()
            model.load_state_dict(params_k)
            expected = model(tokens)
            control.append(
                {
                    "config": name,
                    "source_layout": label,
                    "source_role": table.role,
                    "source_tp_degree": table.tp_degree,
                    "dst_degree": dst,
                    "magnitude": REFERENCE_MAGNITUDE,
                    "skipped_steps": k_max,
                    "null_max_deviation": _deviation(expected, null_logits[dst]),
                    "intended_state_digest": _digest(params_k),
                }
            )
        model.load_state_dict(params_0)

    return _report(cells, control, growth, excluded, threshold, gate, k_max, magnitudes)


def measure_cell(
    config_name: str, dst_degree: int, magnitude: float, skipped_steps: int
) -> dict[str, Any]:
    """One cell, rebuilt from the seeds alone.

    The sweep steps a source incrementally across k, which is the only affordable
    way to run the grid; this recomputes a single cell from scratch. A test
    asserts the two agree on a recorded cell, which is what turns "the loop does
    not leak state between cells" from an assumption into a check -- and it is
    the path the null-transport control runs, so the control exercises the
    measurement code rather than a second copy of it.
    """
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(MEASUREMENT_THREADS)
    try:
        config = CONFIGS[config_name]
        threshold = load_threshold()
        tokens = _tokens(config)
        model = ReferenceModel(config, seed=MODEL_SEED)
        params_0 = {
            name: t.detach().clone() for name, t in model.full_params().items()
        }

        source = ShardedParamSource(
            params_0, _source_layouts(config)[0][1], seed=SOURCE_SEED
        )
        for _ in range(skipped_steps):
            source.step(magnitude)
        params_k = source.intended_state()

        null = _sharded_logits(config, params_0, dst_degree, tokens)
        correct = _sharded_logits(config, params_k, dst_degree, tokens)
        model.load_state_dict(params_k)
        expected = model(tokens)

        correct_deviation = _deviation(expected, correct)
        null_deviation = _deviation(expected, null)
        return {
            "config": config_name,
            "source_layout": _source_layouts(config)[0][0],
            "dst_degree": dst_degree,
            "magnitude": magnitude,
            "skipped_steps": skipped_steps,
            "correct_max_deviation": correct_deviation,
            "null_max_deviation": null_deviation,
            "correct_sync_passes": correct_deviation < threshold,
            "gate_fires": null_deviation > MIN_SEPARATION * threshold,
        }
    finally:
        torch.set_num_threads(previous_threads)


def smallest_detectable_cell(
    config: str, dst_degree: int, report: dict[str, Any] | None = None
) -> tuple[float, int]:
    """The recorded (magnitude, k) the gate first fires at, smallest magnitude
    first.

    The control reads its cell from here rather than hardcoding one, so a
    re-measurement that moves the boundary moves the test with it instead of
    leaving a stale constant that happens to still pass.
    """
    report = report or load()
    firing = [
        (row["magnitude"], row["smallest_detectable_k"])
        for row in report["resolution"]
        if row["config"] == config
        and row["dst_degree"] == dst_degree
        and row["smallest_detectable_k"] is not None
    ]
    if not firing:
        raise LookupError(
            f"no magnitude in {ARTIFACT.name} fires the gate for config={config!r} "
            f"at dst_degree={dst_degree}. That is itself a result, but it leaves "
            "the null-transport control with no cell to reproduce. Regenerate "
            f"with a wider grid: {REGENERATE}"
        )
    return min(firing)


def _digest(params: dict[str, torch.Tensor]) -> str:
    """Order-independent-per-name fingerprint of a full parameter state.

    Compared across source layouts to show, rather than assert, that the intended
    state does not depend on how the source held it.
    """
    import hashlib

    hasher = hashlib.sha256()
    for name in sorted(params):
        hasher.update(name.encode())
        hasher.update(params[name].detach().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def smallest_detectable_k(
    cells: list[dict[str, Any]], config: str, dst_degree: int, magnitude: float
) -> int | None:
    """The smallest k at which the gate fires, or None if it never did.

    None is a RESULT, not a gap: it says the gate cannot resolve a null sync at
    this magnitude within the swept range, which is what 2e would then have to
    state about its correctness gate.
    """
    firing = [
        row["skipped_steps"]
        for row in cells
        if row["config"] == config
        and row["dst_degree"] == dst_degree
        and row["magnitude"] == magnitude
        and row["gate_fires"]
    ]
    return min(firing) if firing else None


def _resolution(cells: list[dict[str, Any]], magnitudes: tuple[float, ...]) -> list[dict[str, Any]]:
    rows = []
    for name, config in CONFIGS.items():
        for dst in supported_degrees(config):
            for magnitude in magnitudes:
                rows.append(
                    {
                        "config": name,
                        "dst_degree": dst,
                        "magnitude": magnitude,
                        "smallest_detectable_k": smallest_detectable_k(
                            cells, name, dst, magnitude
                        ),
                    }
                )
    return rows


def _growth_summary(
    growth: list[dict[str, Any]], gate: float
) -> list[dict[str, Any]]:
    """Fit the observed growth in k and extrapolate to the gate.

    Two points, first and last, fitted in log-log. A real fit over nine points
    would report a tighter exponent and would not change the conclusion, which is
    an order of magnitude on k rather than a digit; the two-point slope is stated
    for what it is rather than dressed up.
    """
    import math

    rows = []
    for name in CONFIGS:
        points = [row for row in growth if row["config"] == name]
        if len(points) < 2:
            continue
        first, last = points[0], points[-1]
        exponent = math.log(
            last["null_max_deviation"] / first["null_max_deviation"]
        ) / math.log(last["skipped_steps"] / first["skipped_steps"])
        fired = next((row["skipped_steps"] for row in points if row["gate_fires"]), None)
        needed = (
            None
            if fired
            else last["skipped_steps"]
            * (gate / last["null_max_deviation"]) ** (1.0 / exponent)
        )
        rows.append(
            {
                "config": name,
                "measured_exponent_in_k": exponent,
                "exponent_reading": (
                    "the null deviation grows roughly as k ** exponent. Near 0.5 "
                    "is a random walk: k skipped steps accumulate independent "
                    "updates, so the displacement grows as sqrt(k) while the gate "
                    "is a fixed absolute bar. Buying one decade of deviation "
                    "therefore costs a hundredfold increase in k."
                ),
                "largest_k_measured": last["skipped_steps"],
                "null_max_deviation_at_largest_k": last["null_max_deviation"],
                "gate_fired_within_measured_range": fired,
                "extrapolated_k_to_fire": needed,
                "extrapolated": needed is not None,
                "extrapolation_is": (
                    "the crossing point implied by the measured exponent, NOT a "
                    "measurement. It is reported because sweeping k that far is "
                    "not worth the compute and because the order of magnitude is "
                    "what 2e needs; treat the digits as indicative."
                ),
            }
        )
    return rows


def _report(
    cells: list[dict[str, Any]],
    control: list[dict[str, Any]],
    growth: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    threshold: float,
    gate: float,
    k_max: int,
    magnitudes: tuple[float, ...],
) -> dict[str, Any]:
    resolution = _resolution(cells, magnitudes)
    growth_summary = _growth_summary(growth, gate)
    at_reference = [
        row for row in resolution if row["magnitude"] == REFERENCE_MAGNITUDE
    ]
    resolves_one_step = bool(at_reference) and all(
        row["smallest_detectable_k"] == 1 for row in at_reference
    )
    detected = [row["smallest_detectable_k"] for row in at_reference]
    within_range = bool(detected) and all(k is not None for k in detected)
    # When nothing fires within the linear grid, the honest number is the
    # extrapolated crossing rather than `k_max + 1`: "9 or more" and "34000 or
    # more" are different claims about 2e's gate, and only one of them is true.
    extrapolated_k = [
        row["extrapolated_k_to_fire"]
        for row in growth_summary
        if row["extrapolated_k_to_fire"] is not None
    ]
    if resolves_one_step:
        consequence = (
            "2e's per-sync correctness gate resolves a single missed sync at the "
            "reference magnitude."
        )
    elif within_range:
        consequence = (
            "2e must state that its correctness gate catches a transport failing "
            f"for {max(detected)} or more CONSECUTIVE syncs at the reference "
            "magnitude, not a single one. A transport missing fewer consecutive "
            "syncs than that passes the gate, so a timing recorded beside it is "
            "gated by a check that cannot fire on it."
        )
    else:
        consequence = (
            "AT THE REFERENCE MAGNITUDE THE GATE DOES NOT RESOLVE A NULL SYNC "
            f"within any practical number of steps. It does not fire by k={k_max} "
            "in the linear sweep, and the measured growth (deviation ~ k ** ~0.5, "
            f"a random walk) puts the crossing near k~{min(extrapolated_k):.0f} to "
            f"{max(extrapolated_k):.0f} skipped steps, EXTRAPOLATED. A single "
            "missed sync at this magnitude does not even exceed the bare phase 1a "
            "threshold, let alone MIN_SEPARATION times it. 2e therefore cannot "
            "claim its per-sync gate protects a recorded timing against a "
            "transport that delivered nothing: the gate catches gross corruption, "
            "which is what the break cases inject, and not a silent no-op. "
            "Catching a no-op needs a different check -- comparing the delivered "
            "bytes against the intended state directly rather than through the "
            "logits -- which is a 2d/2e design consequence and not a number to "
            "tune here."
        )

    return {
        "phase": "2c",
        "probe": "how many skipped optimizer steps the correctness gate resolves",
        "note": NOTE,
        "outcome": (
            "gate_resolves_a_single_missed_sync"
            if resolves_one_step
            else "gate_resolves_k_or_more_consecutive_missed_syncs"
        ),
        "consequence_for_2e": consequence,
        "reference_magnitude": {
            "value": REFERENCE_MAGNITUDE,
            "extrapolated": True,
            "reasoning": REFERENCE_MAGNITUDE_REASONING,
            "smallest_detectable_k": at_reference,
        },
        "gate": {
            "threshold": threshold,
            "threshold_source": str(PHASE1A_ARTIFACT.relative_to(REPO_ROOT)),
            "threshold_is": (
                "read from phase 1a, never re-derived here. It was derived under "
                "the MAXIMUM absolute logit error, which is why the statistic "
                "below is the maximum."
            ),
            "min_separation": MIN_SEPARATION,
            "fires_above": gate,
            "statistic": "max absolute logit error",
        },
        "reference_state_rule": (
            "Every deviation is measured against the state the sync was INTENDED "
            "to deliver (the source's state after k steps), never against the "
            "source's current state and never against what the sampler holds. "
            "Deliberately stale weights and a broken transport are the same bytes; "
            "this is the only reference under which phase 3's staleness passes and "
            "2e's null transport fails."
        ),
        "sweep": {
            "configs": {name: asdict(config) for name, config in CONFIGS.items()},
            "magnitudes": list(magnitudes),
            "k_max": k_max,
            "batch": BATCH,
            "seq_len": SEQ_LEN,
            "model_seed": MODEL_SEED,
            "token_seed": TOKEN_SEED,
            "source_seed": SOURCE_SEED,
            "torch_num_threads": MEASUREMENT_THREADS,
            "step_model": (
                "noise per parameter, drawn at the parameter's full shape and "
                "split under the source's own layout, scaled by magnitude times "
                "that parameter's RMS at construction. Relative scaling so one "
                "magnitude means the same thing for an embedding and a norm "
                "weight; full-shape draw so a replicated parameter receives "
                "identical noise on every rank."
            ),
        },
        "resolution": resolution,
        "source_layout_control": {
            "why": (
                "A step adds noise elementwise and split/gather is byte-exact, so "
                "the intended state cannot depend on the source's layout. "
                "Recorded rather than assumed: identical digests across layouts "
                "are the evidence. Includes a storage source at degree 4, where "
                "kv2 has no execution table at all."
            ),
            "rows": control,
        },
        "growth_at_reference_magnitude": {
            "why": (
                "The linear grid shows the gate does not fire at the reference "
                "magnitude within k_max. This says how far outside the boundary "
                "is, which is the number 2e needs; 'did not fire' alone does not "
                "distinguish a boundary at k=10 from one at k=10000."
            ),
            "k_values": list(GROWTH_K),
            "summary": growth_summary,
            "rows": growth,
        },
        "excluded_cells": excluded,
        "excluded_cells_are": (
            "cells whose CORRECT sync leg exceeded the threshold. The null result "
            "beside such a cell is not attributable to the missed update, so it is "
            "reported and left out of the resolution rather than averaged in."
        ),
        "cells": cells,
        "environment": environment(),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# --------------------------------------------------------------------------- #
# The artifact, in the shape tolerance.py sets.
# --------------------------------------------------------------------------- #

REGENERATE = "uv run python -m weight_sync_bench.phase2.gate_resolution"


def write(report: dict[str, Any], path: Path = ARTIFACT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path


def load(path: Path = ARTIFACT) -> dict[str, Any]:
    """Read the committed measurement."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Regenerate with: {REGENERATE}")
    return json.loads(path.read_text())


def detectable_k(
    config: str, dst_degree: int, magnitude: float = REFERENCE_MAGNITUDE,
    report: dict[str, Any] | None = None,
) -> int | None:
    """The recorded resolution at one cell. Read, never hardcoded by a caller."""
    report = report or load()
    for row in report["resolution"]:
        if (
            row["config"] == config
            and row["dst_degree"] == dst_degree
            and row["magnitude"] == magnitude
        ):
            return row["smallest_detectable_k"]
    raise KeyError(
        f"no recorded resolution for config={config!r}, dst_degree={dst_degree}, "
        f"magnitude={magnitude}. Regenerate with: {REGENERATE}"
    )


def geometry_mismatches(report: dict[str, Any] | None = None) -> list[str]:
    """Differences between the recorded inputs and the running ones.

    These are under the repo's control and they MOVE the numbers -- the same
    input-dependence phase 1a's floor has -- so a caller is expected to fail on a
    mismatch rather than warn, unlike the environment block.
    """
    report = report or load()
    sweep = report["sweep"]
    problems = [
        f"{key}: recorded {sweep[key]}, running {value}"
        for key, value in (("batch", BATCH), ("seq_len", SEQ_LEN))
        if sweep[key] != value
    ]
    for name, config in CONFIGS.items():
        recorded = sweep["configs"].get(name)
        if recorded != asdict(config):
            problems.append(f"config {name}: recorded {recorded}, running {asdict(config)}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k-max", type=int, default=K_MAX)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = measure(args.k_max)
    path = write(report, args.out or ARTIFACT)

    print(f"phase 2c gate resolution -> {path}")
    print(f"  threshold      : {report['gate']['threshold']:.3e} (phase 1a)")
    print(f"  gate fires above: {report['gate']['fires_above']:.3e}")
    print(f"  outcome        : {report['outcome']}")
    print(f"\n  {'config':>6} {'dst':>4} {'magnitude':>10}  smallest detectable k")
    for row in report["resolution"]:
        k = row["smallest_detectable_k"]
        print(
            f"  {row['config']:>6} {row['dst_degree']:>4} {row['magnitude']:>10.0e}  "
            f"{'not within k<=' + str(args.k_max) if k is None else k}"
        )
    if report["excluded_cells"]:
        print(f"\n  excluded cells : {len(report['excluded_cells'])}")
    print(f"\n  at the reference magnitude {REFERENCE_MAGNITUDE:.0e}:")
    for row in report["growth_at_reference_magnitude"]["summary"]:
        fired = row["gate_fired_within_measured_range"]
        needed = row["extrapolated_k_to_fire"]
        print(
            f"    {row['config']:>4}  deviation ~ k**{row['measured_exponent_in_k']:.2f}, "
            f"{row['null_max_deviation_at_largest_k']:.3e} at k="
            f"{row['largest_k_measured']}; "
            + (f"gate fires at k={fired}" if fired else f"gate would fire near k~{needed:.0f} (extrapolated)")
        )


if __name__ == "__main__":
    main()
