"""A layout table's role, and the one thing that actually differs between the two.

FSDP-style full sharding is `Shard(0)` on every parameter, fused `qkv` included.
That placement on that tensor is correct as STORAGE and is break case 1 as
EXECUTION -- same placement, same tensor, opposite verdicts -- so the table has
to say which question it is answering.

These tests cover the two halves of that. The permissive half is as load-bearing
as the strict one: `test_a_wrong_execution_table_still_constructs` exists to fail
if anyone adds a check that an execution placement is the CORRECT one, because
such a check would make a wrong execution table unconstructible and the break
cases depend on a wrong layout producing wrong NUMBERS rather than a
construction error.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from weight_sync_bench.reshard import gather_params, split_params
from weight_sync_bench.sharded import InProcessCollective, ShardedModel
from weight_sync_bench.shardspec import (
    EXECUTION,
    STORAGE,
    FusedPaired,
    HeadPartitioned,
    LayoutTable,
    Shard,
    ShardSpec,
    TOY,
    TOY_KV4,
    UnsupportedLayout,
    build_layout_table,
    build_storage_table,
    supported_degrees,
    unrepresentable_roles,
)

from helpers import reference

DEGREES = (1, 2, 4)


# --- the default, which is what keeps every existing table unchanged ---------


def test_a_table_with_no_role_is_an_execution_table():
    """The default is not a convenience. Every table this repo builds outside an
    FSDP-style source is an execution table, so defaulting this way leaves the
    existing behavior identical and makes the less-constrained thing opt in by
    name; defaulting the other way would silently drop validation from the
    tables that have it."""
    assert LayoutTable(tp_degree=2).role == EXECUTION
    assert build_layout_table(TOY, 2).role == EXECUTION
    assert build_storage_table(TOY, 2).role == STORAGE


def test_an_unrecognized_role_is_refused_rather_than_read_as_the_loose_one():
    with pytest.raises(ValueError, match="role must be one of"):
        LayoutTable(tp_degree=2, role="stroage")


# --- the case the whole change is for ----------------------------------------


def test_storage_constructs_at_a_degree_where_execution_cannot():
    """The contrast IS the assertion, so both halves are stated here together.

    At TOY and degree 4, `n_kv_heads=2` leaves no per-rank KV partition to run
    attention over, so no execution table exists. A storage table at the same
    degree is fine: its shards are only ever gathered, and the gather is exact.
    A sharded source has to be able to shard where no execution layout exists,
    and before the role field it could not.
    """
    with pytest.raises(UnsupportedLayout, match="n_kv_heads"):
        build_layout_table(TOY, 4)

    table = build_storage_table(TOY, 4)
    assert table.role == STORAGE
    assert table.tp_degree == 4
    assert "layers.0.qkv" in table
    assert all(spec.placement == Shard(0) for spec in table.specs.values())


def test_storage_and_execution_tables_name_the_same_parameters():
    """Two builders that expanded parameter names even slightly differently would
    produce tables that cannot be resharded into each other."""
    assert set(build_storage_table(TOY, 2)) == set(build_layout_table(TOY, 2))


def test_supported_degrees_and_unrepresentable_roles_are_role_aware():
    assert 4 in supported_degrees(TOY, role=STORAGE)
    assert 4 not in supported_degrees(TOY)
    assert supported_degrees(TOY, role=STORAGE) == DEGREES
    assert supported_degrees(TOY) == (1, 2)
    # The execution answer is unchanged, and only qkv is affected there.
    assert unrepresentable_roles(TOY, 4) == frozenset({"qkv"})
    assert unrepresentable_roles(TOY, 4, role=STORAGE) == frozenset()
    # TOY_KV4 has no unrepresentable degree under either role, so the two agree.
    assert supported_degrees(TOY_KV4) == supported_degrees(TOY_KV4, role=STORAGE)


# --- the strict direction ----------------------------------------------------


@pytest.mark.parametrize(
    "placement",
    [
        HeadPartitioned(n_heads=8, n_kv_heads=2, head_dim=32),
        FusedPaired((0, 704), (704, 1408)),
    ],
    ids=["head_partitioned", "fused_paired"],
)
def test_a_storage_table_refuses_the_execution_only_placements(placement):
    """Both exist to select non-contiguous source rows so a rank can compute over
    its own slice. A source that only holds parameters shards flat on dim 0, so a
    storage table carrying either is mislabeled."""
    with pytest.raises(UnsupportedLayout) as excinfo:
        LayoutTable(
            tp_degree=2,
            specs={"layers.0.qkv": ShardSpec(placement, 2)},
            role=STORAGE,
        )
    message = str(excinfo.value)
    assert "layers.0.qkv" in message
    assert type(placement).__name__ in message


def test_the_refusal_is_not_triggered_by_a_replicated_parameter():
    """Norms are `Replicated()` in an execution table and that placement is not
    execution-only, so this must not become a blanket "storage is Shard(0) only"
    check by accident."""
    table = LayoutTable(
        tp_degree=2,
        specs={"final_norm": ShardSpec(Shard(0), 2)},
        role=STORAGE,
    )
    assert table.role == STORAGE


# --- the permissive direction, which is equally deliberate -------------------


def test_a_wrong_execution_table_still_constructs():
    """GUARD AGAINST A VALIDATOR THAT SHOULD NOT EXIST.

    `Shard(0)` on fused `qkv` labeled as execution is precisely break case 1. It
    has to remain constructible: the break cases depend on a wrong layout
    producing wrong numbers rather than a construction error, and handing the
    resharder a wrong table is the injection route this repo's own notes name.

    A check that rejected it would pass the suite as it stands today, because
    `tests/test_breaks.py` injects by corrupting parameters rather than by
    building a wrong table. This test is what would fail instead.
    """
    wrong = LayoutTable(
        tp_degree=2,
        specs={"layers.0.qkv": ShardSpec(Shard(0), 2)},
    )
    assert wrong.role == EXECUTION
    assert wrong["layers.0.qkv"].placement == Shard(0)


# --- storage's whole contract ------------------------------------------------


@pytest.mark.parametrize("t", DEGREES)
def test_storage_split_gathers_back_byte_exactly(t):
    """SELF-CONSISTENCY, NOT CORRECTNESS -- and for storage that is the point.

    For any dim, `cat(chunk(X, n, d), d) == X`, so this passes for a consistently
    wrong layout too; that is why the repo refuses it as evidence for an
    EXECUTION table, where the consumer's arithmetic is what exposes a bad
    layout. A storage layout has no consumer that computes: its only job is to
    gather back exactly. So the same check that proves nothing there is the whole
    contract here.

    Degree 4 is included, and it is the reason this test exists at TOY: no
    execution table can be built there at all.
    """
    full = reference("kv2").full_params()
    table = build_storage_table(TOY, t)

    recovered = gather_params(split_params(full, table), table)

    assert recovered.keys() == full.keys()
    for key, want in full.items():
        assert torch.equal(recovered[key], want), key


def test_storage_shards_of_fused_qkv_are_the_contiguous_slice():
    """The other side of `test_qkv_shard_is_not_a_contiguous_slice`. The same
    tensor, the same degree, and the opposite expectation, because the question
    is a different one."""
    params = reference("kv2").full_params()
    qkv = params["layers.0.qkv"]
    shards = split_params(params, build_storage_table(TOY, 2))["layers.0.qkv"]

    assert torch.equal(shards[0], qkv.chunk(2, dim=0)[0])
    assert torch.equal(shards[1], qkv.chunk(2, dim=0)[1])


# --- the second line, which already exists -----------------------------------


def test_the_execution_consumer_still_refuses_degree_4_but_only_by_divisibility():
    """Storage shards handed to an execution consumer are refused -- by the GQA
    divisibility check, NOT by the role.

    `ShardedModel` takes a bare `ShardedParams` and never sees a table, so at a
    degree where BOTH tables exist nothing structurally prevents storage shards
    reaching a forward. That limit is known and accepted, and is recorded here
    rather than closed: closing it would mean changing `ShardedModel`'s
    signature, and the defense that does fire is the same defense-in-depth the
    GQA decision already runs at two levels.
    """
    storage_shards = split_params(
        reference("kv2").full_params(), build_storage_table(TOY, 4)
    )
    with pytest.raises(UnsupportedLayout, match="n_kv_heads"):
        ShardedModel(TOY, storage_shards, InProcessCollective(4))
