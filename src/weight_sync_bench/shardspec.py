"""Layout vocabulary for tensor-parallel resharding.

DTensor's placement names (`Shard`, `Replicated`) plus exactly two extensions.
Each extension exists because the corresponding rank-local shard is a
non-contiguous selection of source rows and therefore cannot be spelled as
`Shard(0)`; see README.md, "The two fused tensors".

A placement validates itself against the TP degree it is being used at, so an
unrepresentable layout fails at construction rather than at reshard time.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class UnsupportedLayout(Exception):
    """A (parameter, TP degree) pair has no valid rank-local shard representation."""


@dataclass(frozen=True)
class Replicated:
    """Every rank holds the full tensor."""

    def validate(self, group_size: int) -> None:
        return


@dataclass(frozen=True)
class Shard:
    """Contiguous equal split along `dim`. DTensor semantics."""

    dim: int

    def validate(self, group_size: int) -> None:
        if self.dim < 0:
            raise UnsupportedLayout(f"Shard dim must be non-negative, got {self.dim}")


@dataclass(frozen=True)
class HeadPartitioned:
    """Fused QKV split by attention head.

    Rank `i` receives Q heads [i * n_heads/t, (i+1) * n_heads/t) together with the
    corresponding K and V heads, reassembled into a fused local tensor. The rows
    are not contiguous in the source, so this is not `Shard(0)`.
    """

    n_heads: int
    n_kv_heads: int
    head_dim: int

    def validate(self, group_size: int) -> None:
        if self.n_heads % group_size:
            raise UnsupportedLayout(
                f"n_heads={self.n_heads} is not divisible by TP degree {group_size}; "
                "Q heads cannot be split evenly across ranks"
            )
        if self.n_kv_heads % group_size:
            # The other branch is legitimate and is what production takes: when
            # n_kv_heads < TP degree, Megatron-Core and vLLM replicate KV heads so
            # every rank holds a full copy of each K/V head and only Q is split.
            # Phase 2 may need that path to interoperate with a real vLLM engine.
            #
            # We raise instead. This harness exists to make unrepresentable layouts
            # loud, and silently falling back to replication would be a layout claim
            # the LayoutTable has no way to express -- the table would say "split by
            # head" while the bytes on each rank say "replicated". See README.md,
            # "GQA at TP=4", for the full reasoning before changing this.
            raise UnsupportedLayout(
                f"n_kv_heads={self.n_kv_heads} is not divisible by TP degree "
                f"{group_size}; no clean per-rank KV partition exists. Replicating "
                "KV is a valid alternative design but is deliberately not implemented"
            )


@dataclass(frozen=True)
class FusedPaired:
    """Two equal row ranges fused into one tensor, split in lockstep.

    `gate_up` stores gate at rows `first` and up at rows `second`. Rank `i` receives
    slice `i` of each, concatenated. Not contiguous in the source, so not `Shard(0)`.
    """

    first: tuple[int, int]
    second: tuple[int, int]

    @property
    def rows(self) -> int:
        return self.first[1] - self.first[0]

    def validate(self, group_size: int) -> None:
        if self.rows != self.second[1] - self.second[0]:
            raise UnsupportedLayout(
                f"FusedPaired ranges must be equal length, got {self.first} and {self.second}"
            )
        if self.rows % group_size:
            raise UnsupportedLayout(
                f"paired range of {self.rows} rows is not divisible by TP degree {group_size}"
            )


Placement = Replicated | Shard | HeadPartitioned | FusedPaired


@dataclass(frozen=True)
class ShardSpec:
    placement: Placement
    group_size: int

    def __post_init__(self) -> None:
        if self.group_size < 1:
            raise UnsupportedLayout(f"group_size must be >= 1, got {self.group_size}")
        self.placement.validate(self.group_size)


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 2
    head_dim: int = 32
    ffn: int = 704
    vocab: int = 32000
    norm_eps: float = 1e-6


@dataclass(frozen=True)
class LayoutTable:
    """Parameter name -> ShardSpec at one TP degree.

    `omitted` records roles deliberately left out, so a partial table states its own
    incompleteness rather than looking like a full one.
    """

    tp_degree: int
    specs: dict[str, ShardSpec] = field(default_factory=dict)
    omitted: frozenset[str] = frozenset()

    def __getitem__(self, name: str) -> ShardSpec:
        return self.specs[name]

    def __contains__(self, name: str) -> bool:
        return name in self.specs

    def __iter__(self):
        return iter(self.specs)


TOY = ModelConfig()
"""The spec's toy model. n_kv_heads=2 is unrepresentable at t=4 by design."""

TOY_KV4 = ModelConfig(n_kv_heads=4)
"""Variant with n_kv_heads=4, representable at every t in {1, 2, 4}.

Exists so the forward invariant gets real t=4 coverage of every parameter, `qkv`
included -- a 4-way head split, a 4-way vocab split and 4-way row-parallel
reductions all execute here. The two configs cover different things and neither
substitutes for the other: TOY is the only one that exercises the unrepresentable
-layout handling, TOY_KV4 is the only one that exercises a 4-way forward. Its qkv
is [512, 256] rather than [384, 256], since qkv rows track n_kv_heads.
"""

GLOBAL_ROLES = ("embed", "lm_head", "final_norm")
LAYER_ROLES = ("qkv", "o_proj", "gate_up", "down", "attn_norm", "ffn_norm")


def _placements(config: ModelConfig) -> dict[str, Placement]:
    """Correct placement per parameter role. Single source of truth for both
    `build_layout_table` and `unrepresentable_roles`."""
    return {
        "embed": Shard(0),
        "lm_head": Shard(0),
        "final_norm": Replicated(),
        "qkv": HeadPartitioned(config.n_heads, config.n_kv_heads, config.head_dim),
        "o_proj": Shard(1),
        "gate_up": FusedPaired((0, config.ffn), (config.ffn, 2 * config.ffn)),
        "down": Shard(1),
        "attn_norm": Replicated(),
        "ffn_norm": Replicated(),
    }


def unrepresentable_roles(config: ModelConfig, tp_degree: int) -> frozenset[str]:
    """Roles with no valid shard representation at `tp_degree`.

    Unrepresentability is a property of a *parameter*, not of a TP degree. At
    tp_degree=4 only `qkv` is affected (n_kv_heads=2 does not divide 4); `o_proj`,
    `down`, `embed`, `lm_head` and `gate_up` have nothing to do with GQA and split
    4 ways cleanly. Derived by probing each placement, so a new placement with its
    own divisibility constraint is picked up without editing this function.
    """
    bad = set()
    for role, placement in _placements(config).items():
        try:
            placement.validate(tp_degree)
        except UnsupportedLayout:
            bad.add(role)
    return frozenset(bad)


def supported_degrees(
    config: ModelConfig, candidates: tuple[int, ...] = (1, 2, 4)
) -> tuple[int, ...]:
    """TP degrees at which every parameter is representable, so a full forward runs.

    Derived from `unrepresentable_roles` rather than hardcoded, so the sweep follows
    the config instead of a list someone has to remember to update.
    """
    return tuple(t for t in candidates if not unrepresentable_roles(config, t))


def build_layout_table(
    config: ModelConfig, tp_degree: int, omit: frozenset[str] | set[str] | tuple[str, ...] = ()
) -> LayoutTable:
    """Correct layout for the toy model at `tp_degree`.

    Raises `UnsupportedLayout` if any included parameter has no valid representation
    at this degree. Pass `omit=unrepresentable_roles(config, tp_degree)` to build the
    largest valid table at a degree where some role cannot be represented -- this is
    how the invariant sweep still exercises a 4-way split of everything except `qkv`.
    """
    omit = frozenset(omit)
    placements = _placements(config)
    if unknown := omit - placements.keys():
        raise ValueError(f"unknown roles in omit: {sorted(unknown)}")

    specs: dict[str, ShardSpec] = {
        role: ShardSpec(placements[role], tp_degree)
        for role in GLOBAL_ROLES
        if role not in omit
    }
    for i in range(config.n_layers):
        for role in LAYER_ROLES:
            if role not in omit:
                specs[f"layers.{i}.{role}"] = ShardSpec(placements[role], tp_degree)

    return LayoutTable(tp_degree=tp_degree, specs=specs, omitted=omit)
