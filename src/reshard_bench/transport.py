"""Moving parameters from a trainer-side holder into rank-local sinks.

The minimum phase 2 needs, and no more. Phase 2 measures

    T_sync = T_reshard + T_transfer + T_load

so this module exists to give those three stages somewhere to be measured and a
record to report them in. It is not a distributed systems framework.

What GlooTransport does: rank 0 holds the full parameters, splits them under the
destination layout, and sends each rank its shard. Every rank ends up holding
exactly the shards `ShardedModel` expects, and correctness is asserted with the
existing invariant -- reference logits versus sharded logits under the phase 1b
threshold -- rather than with a new notion of "transport correctness".

Deliberately absent
-------------------
**Cross-process resharding**, where the source layout is itself spread across
processes and shards move directly between source and destination ranks without a
full tensor ever existing. That is the interesting case and the expensive one, and
it is phase 2 work. Its absence is why `sync_weights` takes full parameters on rank
0 rather than the `ShardedParams` the spec's sketch named: with no full tensor
anywhere, the src argument would have to be this rank's slice of a source layout,
and moving between two distributed layouts is exactly the piece not built here.

**InProcessTransport.** The spec suggested one, but in phase 1 it would move
nothing between nowhere -- the 1b workers already reconstruct parameters locally
from a seed. It would exist only to make the interface look symmetrical, and a
timing harness whose baseline implementation does no work invites exactly the
wrong comparison in phase 2.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Protocol

import torch
import torch.distributed as dist
from torch import Tensor

from .reshard import split_params
from .shardspec import LayoutTable
from .tolerance import environment

# Trainer-side parameters: the full, unsharded tensors.
FullParams = dict[str, Tensor]


@dataclass(frozen=True)
class SyncRecord:
    """One weight-synchronization event, decomposed.

    Timing fields are None where the stage did not run on this rank -- t_reshard is
    None off rank 0, because only rank 0 splits. That is a real distinction and not
    a zero: a zero would say "took no time", None says "did not happen here".
    """

    t_reshard: float | None
    t_transfer: float | None
    t_load: float | None
    src_layout: str
    dst_layout: str
    # Elements delivered to *this* rank, not the model's total.
    param_count: int
    transport: str
    environment: dict[str, object] = field(default_factory=environment)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class ParamSink(Protocol):
    """Where parameters land. A rank-local holder in phase 1, a vLLM engine in 2."""

    def load(self, params: dict[str, Tensor]) -> None: ...


class Transport(Protocol):
    def sync_weights(self, src: FullParams | None, dst: ParamSink) -> SyncRecord: ...


class LocalParamSink:
    """Rank-local parameter holder. Phase 2 swaps this for a vLLM engine."""

    def __init__(self) -> None:
        self.params: dict[str, Tensor] = {}

    def load(self, params: dict[str, Tensor]) -> None:
        self.params = dict(params)

    def sharded_params(self) -> dict[str, list[Tensor]]:
        """In the shape ShardedModel wants: one tensor per local rank, and this
        process owns one rank."""
        return {name: [tensor] for name, tensor in self.params.items()}


class GlooTransport:
    """Scatter shards from rank 0 to every rank over a gloo process group.

    Requires an initialized process group. One `dist.scatter` per parameter, which
    is the minimal primitive for "each rank gets a different piece": every shard of
    a given parameter has the same shape under every placement in this repo, which
    is what scatter requires.
    """

    name = "gloo"

    def __init__(self, rank: int, world_size: int, dst_table: LayoutTable) -> None:
        if dst_table.tp_degree != world_size:
            raise ValueError(
                f"layout table is for tp_degree={dst_table.tp_degree}, "
                f"world_size={world_size}"
            )
        self.rank = rank
        self.world_size = world_size
        self.dst_table = dst_table

    def sync_weights(self, src: FullParams | None, dst: ParamSink) -> SyncRecord:
        if self.rank == 0 and src is None:
            raise ValueError("rank 0 holds the full parameters and must be given them")

        # --- reshard: rank 0 splits under the destination layout ---
        shards = None
        t_reshard = None
        if self.rank == 0:
            start = perf_counter()
            shards = split_params(src, self.dst_table)
            t_reshard = perf_counter() - start

        # --- transfer: metadata, then one scatter per parameter ---
        start = perf_counter()
        names = sorted(self.dst_table.specs)
        meta: list[object] = [None]
        if self.rank == 0:
            meta = [
                {
                    name: (tuple(shards[name][0].shape), str(shards[name][0].dtype))
                    for name in names
                }
            ]
        # Receivers cannot allocate without knowing shard shapes, and deriving them
        # would duplicate the placement geometry here. One small broadcast instead.
        dist.broadcast_object_list(meta, src=0)
        layout: dict[str, tuple[tuple[int, ...], str]] = meta[0]

        received: dict[str, Tensor] = {}
        for name in names:
            shape, dtype = layout[name]
            out = torch.empty(shape, dtype=getattr(torch, dtype.split(".")[-1]))
            scatter_list = None
            if self.rank == 0:
                scatter_list = [shards[name][r].contiguous() for r in range(self.world_size)]
            dist.scatter(out, scatter_list, src=0)
            received[name] = out
        t_transfer = perf_counter() - start

        # --- load ---
        start = perf_counter()
        dst.load(received)
        t_load = perf_counter() - start

        return SyncRecord(
            t_reshard=t_reshard,
            t_transfer=t_transfer,
            t_load=t_load,
            src_layout="full",
            dst_layout=f"tp{self.world_size}",
            param_count=sum(tensor.numel() for tensor in received.values()),
            transport=self.name,
        )
