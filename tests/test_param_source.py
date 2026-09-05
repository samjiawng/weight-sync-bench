"""The sharded parameter source: what it holds, how it moves, what it refuses.

The properties here are the ones the gate measurement depends on being true. The
sharp one is the replicated-rank check: if a step drew noise per shard, every
replicated parameter would diverge across ranks, and the fixture built to measure
layout-error detection would be manufacturing a layout error of its own.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from weight_sync_bench.phase2.param_source import (
    ParamSourceError,
    ShardedParamSource,
)
from weight_sync_bench.reshard import gather_params
from weight_sync_bench.shardspec import (
    TOY,
    build_layout_table,
    build_storage_table,
)

from helpers import reference

MAGNITUDE = 1e-2


def _params():
    return {name: t.detach().clone() for name, t in reference("kv2").full_params().items()}


def test_the_source_does_not_write_through_into_the_callers_parameters():
    """`split_tensor` returns views of the input, and under `Replicated()` the
    input itself. An in-place step on those would advance the caller's dict --
    and the null-sync leg compares against exactly that dict, so the two legs
    would become the same measurement and the result would be vacuous."""
    full = _params()
    before = {name: t.clone() for name, t in full.items()}

    source = ShardedParamSource(full, build_storage_table(TOY, 2), seed=1)
    source.step(MAGNITUDE)

    for name, want in before.items():
        assert torch.equal(full[name], want), name


def test_replicated_parameters_stay_identical_across_ranks_after_a_step():
    """THE REASON THE NOISE IS DRAWN AT FULL SHAPE. Independent per-shard draws
    would leave the ranks disagreeing about a replicated parameter, which is a
    layout error injected by the fixture rather than by a break case."""
    source = ShardedParamSource(_params(), build_layout_table(TOY, 2), seed=1)
    source.step(MAGNITUDE)

    replicated = [n for n in source.shards if n.endswith(("attn_norm", "ffn_norm"))]
    assert replicated
    for name in replicated:
        first, *rest = source.shards[name]
        for other in rest:
            assert torch.equal(first, other), name


def test_the_step_moves_every_parameter_by_about_the_relative_magnitude():
    """Relative to each parameter's RMS, so one magnitude means the same thing
    for an embedding and a norm weight. Absolute noise would make a single number
    a large perturbation of one tensor and a rounding error on another."""
    full = _params()
    source = ShardedParamSource(full, build_storage_table(TOY, 2), seed=1)
    source.step(MAGNITUDE)
    moved = source.intended_state()

    for name, before in full.items():
        delta = (moved[name] - before).pow(2).mean().sqrt()
        scale = before.pow(2).mean().sqrt()
        assert delta / scale == pytest.approx(MAGNITUDE, rel=0.1), name


def test_the_intended_state_does_not_depend_on_how_the_source_held_it():
    """A step adds noise elementwise and split/gather is byte-exact, so the state
    a source intends to deliver cannot depend on its layout. This is why the
    artifact treats the source layout as a control rather than an axis -- and the
    storage table at degree 4 is a layout `TOY` has no execution table for at
    all."""
    states = []
    for table in (
        build_storage_table(TOY, 2),
        build_storage_table(TOY, 4),
        build_layout_table(TOY, 2),
    ):
        source = ShardedParamSource(_params(), table, seed=1)
        source.step(MAGNITUDE)
        source.step(MAGNITUDE)
        states.append(source.intended_state())

    first, *rest = states
    for other in rest:
        assert first.keys() == other.keys()
        for name in first:
            assert torch.equal(first[name], other[name]), name


def test_intended_state_round_trips_the_shards_it_holds():
    source = ShardedParamSource(_params(), build_storage_table(TOY, 4), seed=1)
    assert torch.equal(
        source.intended_state()["layers.0.qkv"],
        gather_params(source.shards, source.table)["layers.0.qkv"],
    )


def test_steps_are_counted():
    source = ShardedParamSource(_params(), build_storage_table(TOY, 2), seed=1)
    assert source.steps == 0
    source.step(MAGNITUDE)
    source.step(MAGNITUDE)
    assert source.steps == 2


def test_a_degree_larger_than_a_parameter_has_rows_is_refused_by_name():
    """THE T002a CARRY-OVER. A storage table has no unrepresentable degree, so
    nothing upstream constrains this; left to `split_tensor` it surfaces as
    `IndexError: list index out of range` from inside a chunk, which names
    neither the parameter nor the degree nor the row count."""
    degree = TOY.d_model + 1  # more ranks than a norm weight has rows
    with pytest.raises(ParamSourceError) as excinfo:
        ShardedParamSource(_params(), build_storage_table(TOY, degree), seed=1)

    message = str(excinfo.value)
    assert "final_norm" in message
    assert str(degree) in message
    assert str(TOY.d_model) in message


def test_a_degree_that_fits_every_parameter_is_accepted():
    """The refusal is about rows available, not about divisibility: an uneven
    storage split still gathers back exactly, which is a storage layout's whole
    contract."""
    source = ShardedParamSource(_params(), build_storage_table(TOY, 3), seed=1)
    source.step(MAGNITUDE)
    recovered = source.intended_state()
    assert recovered["final_norm"].shape == (TOY.d_model,)
