"""A parameter source that holds shards and moves them the way a step would.

CPU, toy model. No optimizer, no gradient, no loss, no RL loop: the scope
boundary is unchanged by this module. What a training step contributes to the
question being asked here is only that the weights MOVE by a small amount per
step, so that is the whole of what is modelled.

WHY A SOURCE THAT HOLDS SHARDS
-------------------------------
Phase 1's `Transport.sync_weights` takes full tensors on rank 0, so reshard cost
is a function of the destination plus a constant gather. The asymmetry 2e is
after is only measurable when the source is genuinely sharded, and a source that
is genuinely sharded has to mutate the shards it holds rather than a gathered
copy -- otherwise the gather this exists to avoid is back, hidden in the step.

EITHER ROLE IS A LEGITIMATE SOURCE
-----------------------------------
FSDP-style is `build_storage_table`: flat `Shard(0)` everywhere, gathered before
anything executes. A Megatron-style trainer holds its parameters in the layout it
computes in, which is an execution table. This class privileges neither, which is
why it takes a `LayoutTable` and reads the role off it rather than assuming one.

THE NOISE IS DRAWN AT FULL SHAPE AND THEN SPLIT
------------------------------------------------
Drawing independently per shard would be the obvious thing and is wrong: under a
`Replicated()` placement every rank holds the same tensor, and independent draws
would make the ranks disagree. That is a layout violation, not a training step --
the source would be manufacturing exactly the class of error this harness exists
to detect, inside the fixture meant to measure detection.

So the noise is drawn once at the parameter's full shape and split under the same
spec the parameter itself is split by, and each rank adds only its own slice in
place. `split_tensor` dispatches on the placement, so a replicated parameter gets
the same noise on every rank and a sharded one gets disjoint noise, without this
module knowing anything about placements. It also makes the step provably
identical to mutating the full tensor and re-splitting, which is what lets
`intended_state()` mean what it says -- and which is asserted by a test rather
than left as an argument.

Note the shape of what is avoided: the PARAMETERS are never gathered. A
full-shaped noise tensor is allocated per parameter per step, which is a real
cost at model scale and is the price of keeping replicated ranks in agreement.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ..reshard import gather_params, split_params, split_tensor
from ..shardspec import LayoutTable

ShardedParams = dict[str, list[Tensor]]


class ParamSourceError(RuntimeError):
    """The source cannot hold these parameters under this table."""


class ShardedParamSource:
    """Holds `ShardedParams` under a stated layout and moves them per step.

    `full` is consumed at construction and never referenced afterwards: the
    shards are cloned out of it, so a caller that keeps the dict around sees an
    unmutated copy of the initial state. That matters here, because the null-sync
    leg of the gate measurement compares against exactly that initial state, and
    a source that wrote through into it would make the two legs identical and the
    measurement vacuous.
    """

    def __init__(
        self, full: dict[str, Tensor], table: LayoutTable, seed: int = 0
    ) -> None:
        _check_degree_fits(full, table)
        self.table = table
        self.role = table.role
        self.tp_degree = table.tp_degree
        # `split_tensor` returns views of `full` (and, under `Replicated`, `full`
        # itself), so an in-place step would write through into the caller's
        # dict. Cloned per rank instead. The clone under `Replicated` is what
        # breaks the aliasing between ranks, which is precisely why the noise is
        # drawn at full shape below rather than per rank.
        self.shards: ShardedParams = {
            name: [shard.clone() for shard in shards]
            for name, shards in split_params(full, table).items()
        }
        # Fixed at construction, not recomputed per step. A step's scale is meant
        # to stand for a learning rate against the weight scale, and a learning
        # rate does not chase the weights as they drift; recomputing it would also
        # need the full tensor, which is the gather this class exists to avoid.
        self._rms = {name: _rms(tensor) for name, tensor in full.items()}
        self._shapes = {name: tuple(tensor.shape) for name, tensor in full.items()}
        self._generator = torch.Generator().manual_seed(seed)
        self.seed = seed
        self.steps = 0

    def step(self, magnitude: float) -> None:
        """One step's worth of movement on every parameter.

        `magnitude` is RELATIVE to each parameter's RMS, so one number means the
        same thing across an embedding and a norm weight. Absolute noise would
        make a single magnitude a large perturbation of one tensor and a rounding
        error on another, and the sweep's axis would then be uninterpretable.
        """
        for name, shards in self.shards.items():
            noise = torch.randn(
                self._shapes[name], generator=self._generator, dtype=torch.float32
            ).mul_(magnitude * self._rms[name])
            spec = self.table[name]
            for rank, shard in enumerate(shards):
                shard.add_(split_tensor(noise, spec, rank))
        self.steps += 1

    def intended_state(self) -> dict[str, Tensor]:
        """The full parameters a sync running now would be expected to deliver.

        THE GATE'S REFERENCE IS BUILT FROM THIS, and not from whatever the
        sampler currently holds. Deliberately stale weights and a broken transport
        are the same bytes on the sampler, and the two must come out differently:
        phase 3 varies staleness on purpose and needs the gate to stay quiet,
        while 2e needs it to fire on a transport that delivered nothing. Both hold
        only when the comparison is against the state the sync was supposed to
        deliver -- a correct stale sampler is then compared against the state it
        was meant to reach, and passes; a sampler that received nothing holds an
        earlier state than intended, and fails.
        """
        return gather_params(self.shards, self.table)

    def sharded_params(self) -> ShardedParams:
        """The shards themselves, for a transport that reads them directly."""
        return self.shards


def _rms(tensor: Tensor) -> float:
    return float(tensor.float().pow(2).mean().sqrt())


def _check_degree_fits(full: dict[str, Tensor], table: LayoutTable) -> None:
    """Every parameter has at least `tp_degree` rows to give.

    THE T002a CARRY-OVER, decided here rather than in `shardspec`. A storage table
    has no unrepresentable degree, so nothing upstream constrains the degree, and
    this class is the first thing that shards at an arbitrary one. Left to
    `split_tensor`, an oversized degree surfaces as `IndexError: list index out of
    range` from inside a `chunk(...)[rank]` -- an error that names neither the
    parameter nor the degree nor the row count, so a reader cannot tell whether
    the layout, the model or the caller is wrong.

    The shapes come from the tensors in hand, which is why this does not need the
    shape table `shardspec` deliberately does not have: a layout vocabulary that
    knew tensor shapes would be a second source of truth for the model's geometry.
    """
    too_small = [
        (name, tuple(tensor.shape))
        for name, tensor in full.items()
        if name in table and tensor.shape[0] < table.tp_degree
    ]
    if too_small:
        detail = ", ".join(
            f"{name} has {shape[0]} rows (shape {shape})" for name, shape in too_small
        )
        raise ParamSourceError(
            f"tp_degree={table.tp_degree} exceeds the rows available on "
            f"{len(too_small)} parameter(s): {detail}. A shard per rank is not "
            "possible, so this table cannot describe this model at this degree."
        )


def source_description(source: ShardedParamSource) -> dict[str, Any]:
    """What an artifact should record about a source it measured through."""
    return {
        "role": source.role,
        "tp_degree": source.tp_degree,
        "seed": source.seed,
        "steps_applied": source.steps,
        "parameters": len(source.shards),
    }
