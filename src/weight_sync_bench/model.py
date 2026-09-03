"""ReferenceModel: the unsharded toy transformer.

Its only job is to contain one instance of every layout that can silently corrupt
weights. It is never trained; parameters are random and fixed by a seed, because
the correctness invariant compares ShardedModel's logits against this model's.

Two structural constraints that must not be "cleaned up":

* `qkv` is ONE fused [384, 256] parameter, not separate q/k/v.
* `gate_up` is ONE fused [1408, 256] parameter, not separate gate/up.

Splitting either into separate parameters is the standard HF-Llama idiom and will
look like a tidy-up, but it deletes what the harness tests. With a separate `q` of
[256, 256], the per-head split is a contiguous `Shard(0)` and nothing is
non-contiguous any more; with separate `gate` and `up`, the pairing is enforced by
construction rather than by the layout. `HeadPartitioned` and `FusedPaired` then
have nothing to express, and break case 1 has nothing to break. The fused storage
is the test fixture.

Positional encoding: the spec defines no positional parameters and never mentions
RoPE, so there is none. Attention is causal.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .shardspec import ModelConfig


def rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """RMSNorm, weight only, no bias."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def _randn(shape: tuple[int, ...], generator: torch.Generator, std: float = 0.02) -> nn.Parameter:
    t = torch.empty(shape, dtype=torch.float32)
    t.normal_(mean=0.0, std=std, generator=generator)
    return nn.Parameter(t, requires_grad=False)


def _norm_weight(size: int, generator: torch.Generator) -> nn.Parameter:
    """Random norm weights, deliberately NOT ones.

    All-ones norm weights would make break case 3 (sharding RMSNorm weights instead
    of replicating them) undetectable: if every entry is identical, mis-slicing or
    permuting the weight vector produces the same numbers. The spec also says
    parameters are random. Do not "fix" this to torch.ones.
    """
    t = torch.empty(size, dtype=torch.float32)
    t.normal_(mean=1.0, std=0.1, generator=generator)
    return nn.Parameter(t, requires_grad=False)


class ReferenceBlock(nn.Module):
    """One transformer layer. Parameter names match the LayoutTable roles."""

    def __init__(self, config: ModelConfig, generator: torch.Generator) -> None:
        super().__init__()
        self.config = config
        q_rows = config.n_heads * config.head_dim
        kv_rows = config.n_kv_heads * config.head_dim

        # Fused QKV: Q rows [0, q_rows), K rows [q_rows, q_rows + kv_rows),
        # V rows [q_rows + kv_rows, q_rows + 2 * kv_rows). One tensor, by design.
        self.qkv = _randn((q_rows + 2 * kv_rows, config.d_model), generator)
        self.o_proj = _randn((config.d_model, q_rows), generator)
        # Fused gate/up: gate rows [0, ffn), up rows [ffn, 2 * ffn). One tensor.
        self.gate_up = _randn((2 * config.ffn, config.d_model), generator)
        self.down = _randn((config.d_model, config.ffn), generator)
        self.attn_norm = _norm_weight(config.d_model, generator)
        self.ffn_norm = _norm_weight(config.d_model, generator)

    def _attention(self, x: Tensor) -> Tensor:
        cfg = self.config
        b, t, _ = x.shape
        q_rows = cfg.n_heads * cfg.head_dim
        kv_rows = cfg.n_kv_heads * cfg.head_dim

        fused = F.linear(x, self.qkv)
        q = fused[..., :q_rows]
        k = fused[..., q_rows : q_rows + kv_rows]
        v = fused[..., q_rows + kv_rows :]

        q = q.view(b, t, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k = k.view(b, t, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        v = v.view(b, t, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)

        # GQA: Q head j uses KV head j // group. repeat_interleave (not repeat/tile)
        # gives that contiguous grouping, which is the same mapping HeadPartitioned
        # assumes when it hands rank i its Q heads and the corresponding KV heads.
        group = cfg.n_heads // cfg.n_kv_heads
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)

        scores = (q @ k.transpose(-2, -1)) * (cfg.head_dim**-0.5)
        causal = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        out = F.softmax(scores, dim=-1) @ v

        out = out.transpose(1, 2).reshape(b, t, q_rows)
        return F.linear(out, self.o_proj)

    def _ffn(self, x: Tensor) -> Tensor:
        fused = F.linear(x, self.gate_up)
        gate = fused[..., : self.config.ffn]
        up = fused[..., self.config.ffn :]
        return F.linear(F.silu(gate) * up, self.down)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self._attention(rms_norm(x, self.attn_norm, self.config.norm_eps))
        return x + self._ffn(rms_norm(x, self.ffn_norm, self.config.norm_eps))


class ReferenceModel(nn.Module):
    """Unsharded fp32 forward. `state_dict()` keys match LayoutTable parameter names."""

    def __init__(self, config: ModelConfig | None = None, seed: int = 0) -> None:
        super().__init__()
        self.config = config = config or ModelConfig()
        if config.n_heads * config.head_dim != config.d_model:
            raise ValueError(
                f"n_heads * head_dim ({config.n_heads * config.head_dim}) "
                f"!= d_model ({config.d_model})"
            )
        if config.n_heads % config.n_kv_heads:
            raise ValueError(
                f"n_heads ({config.n_heads}) must be divisible by "
                f"n_kv_heads ({config.n_kv_heads})"
            )

        # Explicit generator, not the global RNG: init must be reproducible
        # regardless of what else has drawn random numbers in the process.
        generator = torch.Generator().manual_seed(seed)

        self.embed = _randn((config.vocab, config.d_model), generator)
        self.layers = nn.ModuleList(
            ReferenceBlock(config, generator) for _ in range(config.n_layers)
        )
        self.final_norm = _norm_weight(config.d_model, generator)
        # Untied: a separate draw, never a view or transpose of embed.
        self.lm_head = _randn((config.vocab, config.d_model), generator)

    def forward(self, tokens: Tensor) -> Tensor:
        """tokens: [batch, seq] int64 -> logits: [batch, seq, vocab] fp32."""
        x = self.embed[tokens]
        for layer in self.layers:
            x = layer(x)
        x = rms_norm(x, self.final_norm, self.config.norm_eps)
        return F.linear(x, self.lm_head)

    def full_params(self) -> dict[str, Tensor]:
        """Flat name -> tensor, keyed to match LayoutTable."""
        return dict(self.state_dict())
