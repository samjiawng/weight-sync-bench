"""Phase 1b: the correctness invariant under real gloo collectives.

One process per rank, real torch.distributed all_reduce / all_gather. The
ShardedModel forward body is identical to phase 1a -- only the Collective changes,
which is the property the phase 1a/1b contract in sharded.py exists to preserve.

These tests spawn processes, so they are slower than the rest of the suite. Two
seeds per cell here; the tolerance measurement is what runs the full sweep.

The threshold comes from tolerance/phase1b.json, measured separately from 1a
because gloo's reduction order need not match the in-process simulation's.
"""

from __future__ import annotations

import pytest

from helpers import CONFIGS, DEGREES, tokens_for
from reshard_bench.distributed import run_cell
from reshard_bench.tolerance import (
    ARTIFACT,
    ARTIFACT_1B,
    derive_threshold,
    load,
    load_threshold,
)

THRESHOLD = load_threshold(ARTIFACT_1B)

CELLS = [(name, t) for name, degrees in DEGREES.items() for t in degrees]


@pytest.mark.parametrize("name,t", CELLS)
def test_gloo_sharded_matches_reference(name, t):
    """The spec's acceptance criterion for 1b. `run_cell` also asserts internally
    that every rank agrees after the final all_gather."""
    rows = run_cell(CONFIGS[name], t, seeds=[0, 1], tokens=[tokens_for(name)] * 2)

    assert len(rows) == 2
    for row in rows:
        assert row["max"] < THRESHOLD, (
            f"{name}/t={t} deviated by {row['max']:.3e}, over the phase 1b threshold "
            f"{THRESHOLD:.3e}"
        )


def test_phase_1b_threshold_is_measured_not_borrowed_from_1a():
    """The two phases must have independent artifacts. Reusing 1a's threshold for 1b
    is explicitly not valid -- gloo need not reduce in the same order."""
    one_a, one_b = load(ARTIFACT), load(ARTIFACT_1B)

    assert one_a["phase"] == "1a"
    assert one_b["phase"] == "1b"
    assert one_b["measurement"]["backend"] == "gloo"
    # Same rule applied to an independent measurement, so the numbers are comparable.
    assert one_b["rule"] == one_a["rule"]
    assert one_b["threshold"] == derive_threshold(
        one_b["max_deviation"], one_b["rule"]["safety_factor"]
    )


def test_phase_1b_artifact_covers_every_cell():
    measured = {(row["config"], row["tp_degree"]) for row in load(ARTIFACT_1B)["results"]}
    assert measured == set(CELLS)


def test_phase_1b_records_distribution_statistics():
    for row in load(ARTIFACT_1B)["results"]:
        if row["max_deviation"] == 0.0:
            continue
        assert 0 < row["median_ulp"] <= row["mean_ulp"] * 2
        assert row["mean_ulp"] < row["max_ulp"], row
