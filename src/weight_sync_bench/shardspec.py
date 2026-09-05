"""Layout vocabulary for tensor-parallel resharding.

DTensor's placement names (`Shard`, `Replicated`) plus exactly two extensions.
Each extension exists because the corresponding rank-local shard is a
non-contiguous selection of source rows and therefore cannot be spelled as
`Shard(0)`; see README.md, "The two fused tensors".

A placement validates itself against the TP degree it is being used at, so an
unrepresentable layout fails at construction rather than at reshard time.

STORAGE AND EXECUTION ARE DIFFERENT QUESTIONS ABOUT THE SAME PLACEMENT
----------------------------------------------------------------------
FSDP-style full sharding is `Shard(0)` on every parameter, fused `qkv` included.
On `qkv` that placement is correct as STORAGE -- nothing executes attention from
an FSDP shard, it all-gathers first, and the gather is exact -- and it is a
layout bug as an EXECUTION layout, the one break case 1 injects. Same placement,
same tensor, opposite verdicts. So a table carries its `role`, because no
property of the placement alone can answer which one is being asked.

The role changes what is legal, and it does so in ONE DIRECTION only. A storage
table refuses `HeadPartitioned` and `FusedPaired`, which exist only to describe
execution. An execution table refuses nothing new, and must not grow a check
that a placement is the CORRECT one for its role: the break cases depend on a
wrong layout producing wrong numbers, so a wrong execution table has to stay
constructible. A layout error a validator can catch is not the kind of layout
error this harness exists to catch.

The one consequence beyond documentation is GQA divisibility, which is an
execution constraint and not a storage one. At a degree where `n_kv_heads` does
not divide the degree there is no per-rank KV partition to run attention over,
so the execution table raises; the storage table at that degree is fine, because
its shards are only ever gathered.
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


EXECUTION = "execution"
"""A table describing how ranks COMPUTE: each rank runs its own matmuls."""

STORAGE = "storage"
"""A table describing how ranks HOLD parameters: shards are gathered, not run."""

ROLES = (EXECUTION, STORAGE)

# Placements that exist only to describe execution. Both name a non-contiguous
# selection of source rows made so a rank can run attention or an FFN over its
# own slice; an FSDP-style source shards flat on dim 0 and has no use for either,
# so a storage table carrying one is mislabeled rather than merely unusual.
EXECUTION_ONLY_PLACEMENTS = (HeadPartitioned, FusedPaired)


def _check_role(role: str) -> str:
    """A mistyped role must not read as the permissive one.

    ValueError, not `UnsupportedLayout`: a role that is not a role is a caller
    bug, and reporting it as a layout problem would put it in the same class as
    the GQA raise, which is a genuine statement about a model and a degree.
    """
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    return role


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
    """Parameter name -> ShardSpec at one TP degree, for one `role`.

    `omitted` records roles deliberately left out, so a partial table states its own
    incompleteness rather than looking like a full one.

    `role` says whether these placements describe COMPUTATION or STORAGE, and it
    defaults to `EXECUTION` for a reason that is not convenience: every table
    this repo builds outside an FSDP-style source is an execution table, so
    defaulting that way leaves existing behavior identical and makes the new,
    less-constrained thing opt in by name. Defaulting the other way would
    silently drop validation from the tables that have it.
    """

    tp_degree: int
    specs: dict[str, ShardSpec] = field(default_factory=dict)
    omitted: frozenset[str] = frozenset()
    role: str = EXECUTION

    def __post_init__(self) -> None:
        """One-directional validation. Storage is constrained; execution is not.

        A storage table refuses the two placements that exist only to describe
        execution: an FSDP-style source shards flat on dim 0 and has no use for
        either, so a table carrying one under this role is mislabeled.

        AN EXECUTION TABLE REFUSES NOTHING HERE, and that asymmetry is the whole
        shape of this class. A check that a placement is the CORRECT one for
        execution would make a WRONG execution table unconstructible, and the
        break cases depend on a wrong layout producing wrong numbers rather than
        a construction error. It would also pass the suite today, since
        `tests/test_breaks.py` injects by corrupting parameters rather than by
        building a wrong table -- so its being wrong is not something the tests
        would tell you. Do not add one.
        """
        _check_role(self.role)
        if self.role != STORAGE:
            return
        for name, spec in self.specs.items():
            if isinstance(spec.placement, EXECUTION_ONLY_PLACEMENTS):
                raise UnsupportedLayout(
                    f"storage table at tp_degree={self.tp_degree} carries "
                    f"{type(spec.placement).__name__} on {name!r}. That placement "
                    "describes execution: it selects non-contiguous source rows so "
                    "a rank can compute over its own slice. A source that only "
                    "holds parameters shards flat on dim 0, so a storage table "
                    "carrying it is mislabeled."
                )

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


def _named_roles(config: ModelConfig) -> list[tuple[str, str]]:
    """(parameter name, role) for every parameter, in table order.

    Shared by both builders. Two places that construct parameter names is two
    places to drift, and a storage table that named its parameters even slightly
    differently from an execution table could not be reshareded into one.
    """
    named = [(role, role) for role in GLOBAL_ROLES]
    named += [
        (f"layers.{i}.{role}", role)
        for i in range(config.n_layers)
        for role in LAYER_ROLES
    ]
    return named


def unrepresentable_roles(
    config: ModelConfig, tp_degree: int, *, role: str = EXECUTION
) -> frozenset[str]:
    """Roles with no valid shard representation at `tp_degree`, for this table role.

    Two senses of "role" meet in this signature and they are unrelated: the
    RETURNED roles are parameter roles ("qkv", "o_proj"), while the `role`
    argument is the table's, execution or storage.

    Unrepresentability is a property of a *parameter*, not of a TP degree. At
    tp_degree=4 only `qkv` is affected (n_kv_heads=2 does not divide 4); `o_proj`,
    `down`, `embed`, `lm_head` and `gate_up` have nothing to do with GQA and split
    4 ways cleanly. Derived by probing each placement, so a new placement with its
    own divisibility constraint is picked up without editing this function.

    UNDER STORAGE THE SET IS ALWAYS EMPTY, and that is the point of the role
    rather than a shortcut. Every constraint the execution answer reports comes
    from a placement that exists to make a rank's own computation possible --
    GQA divisibility above all -- and a source that only holds shards never runs
    that computation. A storage table is flat `Shard(0)`, which validates at any
    degree, so there is nothing left to probe.
    """
    _check_role(role)
    if role == STORAGE:
        return frozenset()
    bad = set()
    for param_role, placement in _placements(config).items():
        try:
            placement.validate(tp_degree)
        except UnsupportedLayout:
            bad.add(param_role)
    return frozenset(bad)


def supported_degrees(
    config: ModelConfig,
    candidates: tuple[int, ...] = (1, 2, 4),
    *,
    role: str = EXECUTION,
) -> tuple[int, ...]:
    """TP degrees at which every parameter is representable under `role`.

    Derived from `unrepresentable_roles` rather than hardcoded, so the sweep follows
    the config instead of a list someone has to remember to update. Under
    `EXECUTION` that means a full forward runs; under `STORAGE` it means every
    candidate, since a storage table imposes no divisibility.

    The storage answer claims exactly that and no more. It does NOT claim the
    shards come out equal: `Shard` splits with `chunk`, which is happy to return
    uneven pieces, and `gather_tensor` concatenates them back exactly either way,
    which is a storage layout's whole contract. The toy geometry does divide by
    every candidate degree here, so the distinction does not bite at these
    numbers; a degree exceeding a parameter's row count would fail in
    `split_tensor` rather than being reported here.
    """
    return tuple(t for t in candidates if not unrepresentable_roles(config, t, role=role))


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
        name: ShardSpec(placements[role], tp_degree)
        for name, role in _named_roles(config)
        if role not in omit
    }
    return LayoutTable(tp_degree=tp_degree, specs=specs, omitted=omit, role=EXECUTION)


def build_storage_table(config: ModelConfig, tp_degree: int) -> LayoutTable:
    """FSDP-style full sharding: flat `Shard(0)` on every parameter, at any degree.

    This is what a genuinely sharded SOURCE holds, and it is the case the role
    field exists for. On fused `qkv` this same placement is break case 1 as an
    execution layout and is correct here, because these shards are only ever
    gathered and the gather is exact.

    Constructs at degrees where `build_layout_table` raises -- `TOY` at 4 above
    all, where `n_kv_heads=2` leaves no per-rank KV partition to run attention
    over. That is not a relaxation of the GQA decision: it is the observation
    that the decision is about executing attention, and nothing here executes.

    No `omit` parameter, deliberately. `omit` exists because an execution table
    can be partial at a degree where some parameter is unrepresentable; a storage
    table has no such degree, so a partial one would be a caller's choice rather
    than a property of the layout, and `LayoutTable.omitted` would then record a
    gap that the geometry does not explain.
    """
    return LayoutTable(
        tp_degree=tp_degree,
        specs={
            name: ShardSpec(Shard(0), tp_degree) for name, _ in _named_roles(config)
        },
        role=STORAGE,
    )
