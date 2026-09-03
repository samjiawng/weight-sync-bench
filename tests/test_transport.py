"""GlooTransport: parameters actually crossing a process boundary.

Rank 0 holds the full parameters and scatters each rank its shard; no other rank
ever sees a full tensor. Correctness is the *existing* invariant -- reference
logits versus the logits computed from transported shards, under the phase 1b
threshold -- not a separate notion of transport correctness.
"""

from __future__ import annotations

import pytest

from helpers import CONFIGS, DEGREES, tokens_for
from weight_sync_bench.distributed import run_transport_cell
from weight_sync_bench.tolerance import ARTIFACT_1B, load_threshold

THRESHOLD = load_threshold(ARTIFACT_1B)

CELLS = [(name, t) for name, degrees in DEGREES.items() for t in degrees]


@pytest.fixture(scope="module")
def transported():
    """One spawn per cell, reused across the assertions below."""
    return {
        (name, t): run_transport_cell(
            CONFIGS[name], t, seeds=[0], tokens=[tokens_for(name)]
        )[0]
        for name, t in CELLS
    }


@pytest.mark.parametrize("name,t", CELLS)
def test_transported_parameters_satisfy_the_invariant(name, t, transported):
    row = transported[(name, t)]
    assert row["max"] < THRESHOLD, (
        f"{name}/t={t} deviated by {row['max']:.3e} after transport, over the phase "
        f"1b threshold {THRESHOLD:.3e}"
    )


@pytest.mark.parametrize("name,t", CELLS)
def test_sync_record_is_populated(name, t, transported):
    record = transported[(name, t)]["record"]

    assert record["transport"] == "gloo"
    assert record["src_layout"] == "full"
    assert record["dst_layout"] == f"tp{t}"
    # Rank 0 does the splitting, so all three stages ran here.
    for stage in ("t_reshard", "t_transfer", "t_load"):
        assert record[stage] is not None and record[stage] > 0.0, stage
    for key in ("torch", "numpy", "python", "platform", "dtype"):
        assert record["environment"][key], key


@pytest.mark.parametrize("name,t", CELLS)
def test_param_count_is_this_ranks_share(name, t, transported):
    """param_count counts elements delivered to this rank, not the model total.

    Replicated tensors (the three norms) arrive whole on every rank, so the count is
    the sharded parameters' 1/t share plus the full norms.
    """
    config = CONFIGS[name]
    total = sum(p.numel() for p in _reference_params(name).values())
    norms = config.d_model * (2 * config.n_layers + 1)
    expected = (total - norms) // t + norms

    assert transported[(name, t)]["record"]["param_count"] == expected


def _reference_params(name: str):
    from helpers import reference

    return reference(name).full_params()
