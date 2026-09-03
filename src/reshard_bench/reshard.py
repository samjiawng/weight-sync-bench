"""split / gather / reshard.

Everything here dispatches on `Placement` and reads its geometry from the placement
itself, so a `LayoutTable` fully determines the split. That is deliberate: it makes
the layout table load-bearing, which is how break cases 1-3 get injected (hand the
resharder a wrong table and the shards come out wrong, and the consuming matmul in
ShardedModel then produces wrong logits).

The reshard path
----------------
`reshard()` is all-gather-then-rescatter: materialize the full tensor from the
source layout, then split it into the destination layout. The two halves are the
separately timeable stages and are kept as separate public functions --
`gather_params` (all-gather) and `split_params` (re-scatter) -- because phase 2
decomposes T_sync and needs to attribute cost to each.

This is the naive path. A production resharder can exchange shards directly
between source and destination ranks without ever materializing the full tensor,
which is a different cost profile entirely. Such an implementation goes behind the
same signature as an alternative to `reshard`, not as an edit to it, so the two
stay comparable when phase 2 measures them.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from .sharded import ShardedParams
from .shardspec import (
    FusedPaired,
    HeadPartitioned,
    LayoutTable,
    Replicated,
    Shard,
    ShardSpec,
)


def split_tensor(full: Tensor, spec: ShardSpec, rank: int) -> Tensor:
    """Rank-local shard of `full` under `spec`."""
    placement, t = spec.placement, spec.group_size

    match placement:
        case Replicated():
            return full
        case Shard(dim=dim):
            return full.chunk(t, dim=dim)[rank].contiguous()
        case HeadPartitioned():
            q_rows, kv_rows = _head_rows(placement)
            local_q, local_kv = q_rows // t, kv_rows // t
            return torch.cat(
                [
                    full[rank * local_q : (rank + 1) * local_q],
                    full[q_rows + rank * local_kv : q_rows + (rank + 1) * local_kv],
                    full[
                        q_rows + kv_rows + rank * local_kv : q_rows
                        + kv_rows
                        + (rank + 1) * local_kv
                    ],
                ],
                dim=0,
            ).contiguous()
        case FusedPaired(first=first, second=second):
            per_rank = placement.rows // t
            return torch.cat(
                [
                    full[first[0] + rank * per_rank : first[0] + (rank + 1) * per_rank],
                    full[second[0] + rank * per_rank : second[0] + (rank + 1) * per_rank],
                ],
                dim=0,
            ).contiguous()

    raise AssertionError(f"unhandled placement: {placement!r}")


def gather_tensor(shards: Sequence[Tensor], spec: ShardSpec) -> Tensor:
    """Inverse of `split_tensor`: reassemble the full tensor from all ranks."""
    placement, t = spec.placement, spec.group_size

    match placement:
        case Replicated():
            # Every rank holds the whole tensor; rank 0's copy is the answer.
            return shards[0]
        case Shard(dim=dim):
            return torch.cat(list(shards), dim=dim)
        case HeadPartitioned():
            q_rows, kv_rows = _head_rows(placement)
            local_q, local_kv = q_rows // t, kv_rows // t
            # Regroup: all ranks' Q, then all ranks' K, then all ranks' V.
            parts = [s[:local_q] for s in shards]
            parts += [s[local_q : local_q + local_kv] for s in shards]
            parts += [s[local_q + local_kv :] for s in shards]
            return torch.cat(parts, dim=0)
        case FusedPaired():
            per_rank = placement.rows // t
            parts = [s[:per_rank] for s in shards]
            parts += [s[per_rank:] for s in shards]
            return torch.cat(parts, dim=0)

    raise AssertionError(f"unhandled placement: {placement!r}")


def _head_rows(placement: HeadPartitioned) -> tuple[int, int]:
    """Row counts of the Q block and of one of the K/V blocks in fused qkv."""
    return (
        placement.n_heads * placement.head_dim,
        placement.n_kv_heads * placement.head_dim,
    )


def split_params(full: dict[str, Tensor], table: LayoutTable) -> ShardedParams:
    """Full parameters -> per-rank shards for every rank in `table`.

    The re-scatter half of a reshard; timed separately in phase 2.
    """
    if missing := table.specs.keys() - full.keys():
        raise ValueError(f"layout table names absent from parameters: {sorted(missing)}")
    return {
        name: [split_tensor(full[name], spec, rank) for rank in range(table.tp_degree)]
        for name, spec in table.specs.items()
    }


def gather_params(sharded: ShardedParams, table: LayoutTable) -> dict[str, Tensor]:
    """Per-rank shards -> full parameters.

    The all-gather half of a reshard; timed separately in phase 2.
    """
    if missing := table.specs.keys() - sharded.keys():
        raise ValueError(f"layout table names absent from shards: {sorted(missing)}")
    for name in table.specs:
        if len(sharded[name]) != table.tp_degree:
            raise ValueError(
                f"{name}: got {len(sharded[name])} shards, "
                f"expected {table.tp_degree} for tp_degree={table.tp_degree}"
            )
    return {name: gather_tensor(sharded[name], spec) for name, spec in table.specs.items()}


def reshard(sharded: ShardedParams, src: LayoutTable, dst: LayoutTable) -> ShardedParams:
    """Move parameters from the `src` layout to the `dst` layout.

    All-gather then re-scatter. See the module docstring on why the two stages stay
    separately callable.
    """
    if src.specs.keys() != dst.specs.keys():
        only_src = sorted(src.specs.keys() - dst.specs.keys())
        only_dst = sorted(dst.specs.keys() - src.specs.keys())
        raise ValueError(
            f"src and dst layouts cover different parameters "
            f"(src only: {only_src}, dst only: {only_dst})"
        )
    return split_params(gather_params(sharded, src), dst)
