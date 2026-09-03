"""ShardedModel: rank-local tensor-parallel forward.

This is the half of the correctness invariant that makes the harness work. A wrong
shard dimension changes the result of the consuming matmul, which a byte-level
round trip could never detect.

Phase 1a/1b contract
--------------------
Every activation in the forward is a `list[Tensor]` holding one tensor per rank
that *this process* is responsible for, aligned with `collective.local_ranks`.
Collectives take and return those lists.

* Phase 1a (`InProcessCollective`): `local_ranks` is every rank, and a collective
  is a direct sum or cat over the list.
* Phase 1b (gloo): `local_ranks` is a single rank, the lists have length 1, and
  the collective calls `dist.all_reduce` / `dist.all_gather` on the one entry.

The forward body is identical in both cases -- only the `Collective` changes. Keep
it that way: do not rewrite the forward to operate on bare tensors, and do not let
a collective's semantics leak into the layer code.

ShardedModel consumes already-sharded parameters. Producing them is reshard.py's
job and is deliberately not done here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from .model import rms_norm
from .shardspec import GLOBAL_ROLES, LAYER_ROLES, ModelConfig, UnsupportedLayout

# name -> one shard per local rank slot, indexed to match collective.local_ranks.
ShardedParams = dict[str, list[Tensor]]


class Collective(Protocol):
    """Collectives over per-rank tensor lists. See the phase 1a/1b contract above."""

    world_size: int
    local_ranks: tuple[int, ...]

    def all_reduce(self, shards: Sequence[Tensor]) -> list[Tensor]:
        """Sum across ranks; every rank receives the total."""
        ...

    def all_gather(self, shards: Sequence[Tensor], dim: int) -> list[Tensor]:
        """Concatenate across ranks along `dim`; every rank receives the whole."""
        ...


class InProcessCollective:
    """Phase 1a. All ranks live in this process; collectives are direct sum/cat.

    Reduction order is sequential in rank order. Gloo need not reduce in the same
    order, which is why the spec requires measuring the tolerance floor separately
    for 1a and 1b.

    The returned list repeats one tensor object across slots rather than copying.
    Nothing in the forward mutates activations in place, so this is safe; if that
    ever changes, copy here.
    """

    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self.local_ranks = tuple(range(world_size))

    def all_reduce(self, shards: Sequence[Tensor]) -> list[Tensor]:
        total = shards[0]
        for shard in shards[1:]:
            total = total + shard
        return [total] * self.world_size

    def all_gather(self, shards: Sequence[Tensor], dim: int) -> list[Tensor]:
        return [torch.cat(list(shards), dim=dim)] * self.world_size


class GlooCollective:
    """Phase 1b. One rank per process, real collectives over a gloo process group.

    Same interface as InProcessCollective: lists in, lists out. Here the lists always
    have length 1, because this process owns exactly one rank -- which is precisely
    why the ShardedModel forward body does not change between phases.

    Requires an initialized process group; see `reshard_bench.distributed`.
    """

    def __init__(self, rank: int, world_size: int) -> None:
        self.world_size = world_size
        self.local_ranks = (rank,)

    def _only(self, shards: Sequence[Tensor]) -> Tensor:
        if len(shards) != 1:
            raise ValueError(
                f"GlooCollective owns one rank, got {len(shards)} shards. The forward "
                "must pass one tensor per local rank."
            )
        return shards[0].contiguous()

    def all_reduce(self, shards: Sequence[Tensor]) -> list[Tensor]:
        # dist.all_reduce is in-place; clone so activations stay immutable, matching
        # InProcessCollective's contract.
        out = self._only(shards).clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return [out]

    def all_gather(self, shards: Sequence[Tensor], dim: int) -> list[Tensor]:
        local = self._only(shards)
        parts = [torch.empty_like(local) for _ in range(self.world_size)]
        dist.all_gather(parts, local)
        return [torch.cat(parts, dim=dim)]


class ShardedModel:
    """Tensor-parallel forward over already-sharded parameters."""

    def __init__(
        self,
        config: ModelConfig,
        params: ShardedParams,
        collective: Collective,
    ) -> None:
        self.config = config
        self.params = params
        self.collective = collective
        self._validate()

    # -- setup ----------------------------------------------------------------

    def _validate(self) -> None:
        cfg, t = self.config, self.collective.world_size

        if cfg.n_kv_heads % t:
            # Same decision as shardspec.HeadPartitioned.validate: no clean per-rank
            # KV partition, so there is no qkv shard to run attention over. See
            # README.md, "GQA at TP=4".
            raise UnsupportedLayout(
                f"n_kv_heads={cfg.n_kv_heads} is not divisible by TP degree {t}; "
                "ShardedModel cannot run a forward pass at this degree"
            )
        for value, label in ((cfg.n_heads, "n_heads"), (cfg.ffn, "ffn"), (cfg.vocab, "vocab")):
            if value % t:
                raise UnsupportedLayout(f"{label}={value} is not divisible by TP degree {t}")

        expected = set(GLOBAL_ROLES) | {
            f"layers.{i}.{role}" for i in range(cfg.n_layers) for role in LAYER_ROLES
        }
        if missing := expected - self.params.keys():
            raise ValueError(f"missing sharded parameters: {sorted(missing)}")

        slots = len(self.collective.local_ranks)
        for name, shards in self.params.items():
            if len(shards) != slots:
                raise ValueError(
                    f"{name}: got {len(shards)} shards, expected {slots} "
                    "(one per local rank)"
                )

    def _p(self, name: str, slot: int) -> Tensor:
        return self.params[name][slot]

    # -- layers ---------------------------------------------------------------

    def _embed(self, tokens: Tensor) -> list[Tensor]:
        """Vocab-parallel embedding: mask out-of-range ids, look up locally with an
        offset, zero the out-of-range rows, all-reduce. Output is replicated."""
        per_rank = self.config.vocab // self.collective.world_size
        partials = []
        for slot, rank in enumerate(self.collective.local_ranks):
            local_ids = tokens - rank * per_rank
            out_of_range = (local_ids < 0) | (local_ids >= per_rank)
            rows = self._p("embed", slot)[local_ids.masked_fill(out_of_range, 0)]
            partials.append(rows.masked_fill(out_of_range.unsqueeze(-1), 0.0))
        return self.collective.all_reduce(partials)

    def _attention(self, layer: int, slot: int, x: Tensor) -> Tensor:
        """Column-parallel qkv, local attention over this rank's heads, row-parallel
        o_proj. Returns a partial sum; the caller all-reduces."""
        cfg, t = self.config, self.collective.world_size
        b, seq, _ = x.shape
        local_heads = cfg.n_heads // t
        local_kv = cfg.n_kv_heads // t
        q_rows = local_heads * cfg.head_dim
        kv_rows = local_kv * cfg.head_dim

        # No collective before qkv: input is replicated and the rank owns its heads.
        fused = F.linear(x, self._p(f"layers.{layer}.qkv", slot))
        q = fused[..., :q_rows].view(b, seq, local_heads, cfg.head_dim).transpose(1, 2)
        k = fused[..., q_rows : q_rows + kv_rows]
        v = fused[..., q_rows + kv_rows :]
        k = k.view(b, seq, local_kv, cfg.head_dim).transpose(1, 2)
        v = v.view(b, seq, local_kv, cfg.head_dim).transpose(1, 2)

        # Same grouping as ReferenceModel: Q head j uses KV head j // group. The
        # ratio is preserved under an even head split, so the local group size
        # equals the global one.
        group = local_heads // local_kv
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)

        scores = (q @ k.transpose(-2, -1)) * (cfg.head_dim**-0.5)
        causal = torch.ones(seq, seq, dtype=torch.bool, device=x.device).triu(diagonal=1)
        out = F.softmax(scores.masked_fill(causal, float("-inf")), dim=-1) @ v
        out = out.transpose(1, 2).reshape(b, seq, q_rows)

        return F.linear(out, self._p(f"layers.{layer}.o_proj", slot))

    def _ffn(self, layer: int, slot: int, x: Tensor) -> Tensor:
        """Column-parallel fused gate_up with SwiGLU applied locally, row-parallel
        down. Returns a partial sum; the caller all-reduces."""
        local_ffn = self.config.ffn // self.collective.world_size
        fused = F.linear(x, self._p(f"layers.{layer}.gate_up", slot))
        gate = fused[..., :local_ffn]
        up = fused[..., local_ffn:]
        return F.linear(F.silu(gate) * up, self._p(f"layers.{layer}.down", slot))

    def _block(self, layer: int, xs: list[Tensor]) -> list[Tensor]:
        eps = self.config.norm_eps
        # RMSNorm: replicated weight, replicated input, no communication.
        normed = [rms_norm(x, self._p(f"layers.{layer}.attn_norm", s), eps) for s, x in enumerate(xs)]
        attn = self.collective.all_reduce(
            [self._attention(layer, s, h) for s, h in enumerate(normed)]
        )
        xs = [x + a for x, a in zip(xs, attn)]

        normed = [rms_norm(x, self._p(f"layers.{layer}.ffn_norm", s), eps) for s, x in enumerate(xs)]
        ffn = self.collective.all_reduce([self._ffn(layer, s, h) for s, h in enumerate(normed)])
        return [x + f for x, f in zip(xs, ffn)]

    # -- forward --------------------------------------------------------------

    def forward(self, tokens: Tensor) -> Tensor:
        """tokens: [batch, seq] int64 -> logits: [batch, seq, vocab] fp32.

        Returns this process's first local rank's copy. The value is replicated
        across ranks by construction: the final all_gather gives every rank the
        whole vocab dimension.
        """
        xs = self._embed(tokens)
        for layer in range(self.config.n_layers):
            xs = self._block(layer, xs)

        eps = self.config.norm_eps
        xs = [rms_norm(x, self._p("final_norm", s), eps) for s, x in enumerate(xs)]
        # Vocab-parallel lm_head: local logits, then all-gather along vocab.
        local_logits = [F.linear(x, self._p("lm_head", s)) for s, x in enumerate(xs)]
        return self.collective.all_gather(local_logits, dim=-1)[0]

    __call__ = forward
