"""Phase 1b: run ShardedModel with one process per rank over a gloo process group.

Topology is deliberately minimal, per the spec: every rank participates in the same
logical operation, `gloo` over a file-backed rendezvous, no production topology being
simulated. The only thing being demonstrated is that the collective version matches
the single-process reference algorithm.

What is NOT distributed here: the reshard itself. Each worker rebuilds the full
parameters deterministically from the seed and takes its own destination shard, so
ShardedModel receives already-sharded parameters exactly as its contract says.
That is sufficient because the src layout provably cannot affect the result --
`gather(split(full, src)) == full` byte-exactly (test_gather_inverts_split_exactly),
so logits under the dst layout depend only on dst. Ordered pairs are therefore
redundant for 1b numerics and 1b measures per (config, degree). Distributed
parameter *movement* is phase 2's subject, not phase 1's.
"""

from __future__ import annotations

import datetime
import json
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor

from .model import ReferenceModel
from .reshard import split_tensor
from .sharded import GlooCollective, ShardedModel
from .shardspec import ModelConfig, build_layout_table
from .transport import GlooTransport, LocalParamSink

# Gloo rendezvous timeout. Generous: these are tiny models, so exceeding this means
# something is wedged rather than slow.
TIMEOUT_SECONDS = 300


def _worker(
    rank: int,
    world_size: int,
    init_file: str,
    out_dir: str,
    config: ModelConfig,
    seeds: list[int],
    tokens: list[Tensor],
) -> None:
    """One rank. Runs every seed for this (config, degree) cell in one process group.

    Spawning once per cell rather than once per repetition keeps the measurement to a
    handful of process launches.
    """
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=TIMEOUT_SECONDS),
    )
    try:
        collective = GlooCollective(rank, world_size)
        table = build_layout_table(config, world_size)
        rows = []

        for seed, token_batch in zip(seeds, tokens):
            model = ReferenceModel(config, seed=seed)
            full = model.full_params()
            # This rank's shard only -- one tensor per parameter, matching
            # collective.local_ranks of length 1.
            params = {
                name: [split_tensor(full[name], spec, rank)]
                for name, spec in table.specs.items()
            }

            actual = ShardedModel(config, params, collective)(token_batch)
            diff = (actual - model(token_batch)).abs()
            rows.append(
                {
                    "max": diff.max().item(),
                    "median": diff.median().item(),
                    "mean": diff.mean().item(),
                }
            )

        Path(out_dir, f"rank{rank}.json").write_text(json.dumps(rows))
    finally:
        dist.destroy_process_group()


def _transport_worker(
    rank: int,
    world_size: int,
    init_file: str,
    out_dir: str,
    config: ModelConfig,
    seeds: list[int],
    tokens: list[Tensor],
) -> None:
    """One rank, parameters arriving over GlooTransport rather than built locally.

    Only rank 0 constructs the full parameters; every other rank receives its shard
    and never sees a full tensor. Rank 0 also builds the reference logits, so the
    result is checked with the existing invariant. Other ranks report a checksum
    only, which is enough to confirm every rank agrees after the final all_gather.
    """
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=TIMEOUT_SECONDS),
    )
    try:
        collective = GlooCollective(rank, world_size)
        table = build_layout_table(config, world_size)
        transport = GlooTransport(rank, world_size, table)
        rows = []

        for seed, token_batch in zip(seeds, tokens):
            model = ReferenceModel(config, seed=seed) if rank == 0 else None
            sink = LocalParamSink()
            record = transport.sync_weights(
                model.full_params() if rank == 0 else None, sink
            )

            actual = ShardedModel(config, sink.sharded_params(), collective)(token_batch)
            row: dict[str, object] = {
                "checksum": [float(actual.sum()), float(actual.abs().max())]
            }
            if rank == 0:
                diff = (actual - model(token_batch)).abs()
                row |= {
                    "max": diff.max().item(),
                    "median": diff.median().item(),
                    "mean": diff.mean().item(),
                    "record": record.as_dict(),
                }
            rows.append(row)

        Path(out_dir, f"rank{rank}.json").write_text(json.dumps(rows))
    finally:
        dist.destroy_process_group()


def run_transport_cell(
    config: ModelConfig, tp_degree: int, seeds: list[int], tokens: list[Tensor]
) -> list[dict[str, object]]:
    """Run the transport path. Returns rank 0's rows, which carry the deviation and
    the SyncRecord; cross-rank agreement is asserted on the checksums."""
    per_rank = _spawn(_transport_worker, config, tp_degree, seeds, tokens)
    for rank, rows in enumerate(per_rank[1:], start=1):
        mine = [row["checksum"] for row in rows]
        theirs = [row["checksum"] for row in per_rank[0]]
        if mine != theirs:
            raise AssertionError(
                f"rank {rank} disagrees with rank 0 after all_gather at "
                f"tp_degree={tp_degree}; the collective is not replicating the result"
            )
    return per_rank[0]


def _spawn(
    worker, config: ModelConfig, tp_degree: int, seeds: list[int], tokens: list[Tensor]
) -> list[list[dict]]:
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "rendezvous")
        mp.spawn(
            worker,
            args=(tp_degree, init_file, tmp, config, seeds, tokens),
            nprocs=tp_degree,
            join=True,
        )
        return [
            json.loads(Path(tmp, f"rank{rank}.json").read_text())
            for rank in range(tp_degree)
        ]


def run_cell(
    config: ModelConfig, tp_degree: int, seeds: list[int], tokens: list[Tensor]
) -> list[dict[str, float]]:
    """Spawn `tp_degree` processes, run every seed, return the per-seed statistics.

    Every rank computes its own statistics. They must agree exactly: after the final
    all_gather each rank holds the same logits, so a disagreement means the collective
    is not doing what it claims. That is checked here rather than left to a test,
    because it is a property of this launcher's contract.
    """
    per_rank = _spawn(_worker, config, tp_degree, seeds, tokens)
    for rank, rows in enumerate(per_rank[1:], start=1):
        if rows != per_rank[0]:
            raise AssertionError(
                f"rank {rank} disagrees with rank 0 after all_gather at "
                f"tp_degree={tp_degree}; the collective is not replicating the result"
            )
    return per_rank[0]
