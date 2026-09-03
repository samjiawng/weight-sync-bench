"""Phase 1a: the correctness invariant.

    ReferenceModel(full, x) == ShardedModel(reshard(full, src -> dst), x)

Swept over two configs. TOY (n_kv_heads=2) is the spec's model and runs at
t in {1, 2}; TOY_KV4 (n_kv_heads=4) is representable at t=4 and is what gives the
sweep a real 4-way forward, qkv included. Degrees come from `supported_degrees`,
so the sweep follows the config rather than a hardcoded list.

The tolerance threshold is read from `tolerance/phase1a.json`, which is produced by
`weight_sync_bench.tolerance` and committed. It is measured and derived, never
hand-picked -- see that module.
"""

import dataclasses
import itertools
import warnings

import pytest
import torch

from helpers import CONFIGS, DEGREES, reference, tokens_for
from weight_sync_bench.reshard import gather_params, reshard, split_params
from weight_sync_bench.sharded import InProcessCollective, ShardedModel
from weight_sync_bench.shardspec import (
    TOY,
    TOY_KV4,
    ModelConfig,
    UnsupportedLayout,
    build_layout_table,
)
from weight_sync_bench.tolerance import (
    EnvironmentMismatch,
    derive_threshold,
    environment_mismatches,
    load,
    load_threshold,
)

# Phase 1a acceptance threshold, measured. Regenerate with:
#   uv run python -m weight_sync_bench.tolerance
THRESHOLD = load_threshold()

ORDERED_PAIRS = [
    (name, src, dst)
    for name, degrees in DEGREES.items()
    for src, dst in itertools.product(degrees, repeat=2)
]


@pytest.fixture(scope="module")
def tokens():
    return tokens_for("kv2")


def sharded_at(name: str, t: int) -> ShardedModel:
    cfg = CONFIGS[name]
    params = split_params(reference(name).full_params(), build_layout_table(cfg, t))
    return ShardedModel(cfg, params, InProcessCollective(t))


def test_configs_cover_what_they_are_meant_to():
    """Guards the reason two configs exist: one must be unrepresentable at t=4 and
    the other must not be. If this drifts, the sweep silently loses coverage."""
    assert DEGREES["kv2"] == (1, 2)
    assert DEGREES["kv4"] == (1, 2, 4)


@pytest.mark.parametrize("name,t", [(n, t) for n, ds in DEGREES.items() for t in ds])
def test_sharded_matches_reference(tokens, name, t):
    expected = reference(name)(tokens)
    actual = sharded_at(name, t)(tokens)

    assert actual.shape == expected.shape
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    assert (actual - expected).abs().max().item() < THRESHOLD


@pytest.mark.parametrize("name,src,dst", ORDERED_PAIRS)
def test_invariant_across_ordered_pairs(tokens, name, src, dst):
    """The spec's acceptance criterion: reshard src -> dst, then run at dst."""
    cfg = CONFIGS[name]
    src_table, dst_table = build_layout_table(cfg, src), build_layout_table(cfg, dst)

    start = split_params(reference(name).full_params(), src_table)
    moved = reshard(start, src_table, dst_table)
    actual = ShardedModel(cfg, moved, InProcessCollective(dst))(tokens)

    expected = reference(name)(tokens)
    assert (actual - expected).abs().max().item() < THRESHOLD


@pytest.mark.parametrize("name,t", [(n, t) for n, ds in DEGREES.items() for t in ds])
def test_gather_inverts_split_exactly(name, t):
    """split/gather is a byte-exact round trip. Necessary but nowhere near
    sufficient -- revision 1 of the spec was rejected for treating this as the
    correctness test, since it holds for a consistently wrong layout too."""
    cfg = CONFIGS[name]
    table = build_layout_table(cfg, t)
    full = reference(name).full_params()

    recovered = gather_params(split_params(full, table), table)

    assert recovered.keys() == full.keys()
    for key, want in full.items():
        assert torch.equal(recovered[key], want), key


@pytest.mark.parametrize("name,t", [(n, t) for n, ds in DEGREES.items() for t in ds])
def test_split_accounts_for_every_element(name, t):
    cfg = CONFIGS[name]
    replicated = {"attn_norm", "ffn_norm", "final_norm"}
    full = reference(name).full_params()
    shards = split_params(full, build_layout_table(cfg, t))

    for key, want in full.items():
        parts = shards[key]
        if key.rsplit(".", 1)[-1] in replicated:
            assert all(p.shape == want.shape for p in parts)
        else:
            assert sum(p.numel() for p in parts) == want.numel()


def test_qkv_shard_is_not_a_contiguous_slice():
    """The property break case 1 attacks: the correct qkv shard must differ from
    naive chunk(dim=0). Asserted directly so a regression in split_tensor shows up
    here rather than only as a logits mismatch."""
    table = build_layout_table(TOY, 2)
    qkv = reference("kv2").full_params()["layers.0.qkv"]

    correct = split_params({"layers.0.qkv": qkv}, _only(table, "layers.0.qkv"))
    naive = list(qkv.chunk(2, dim=0))

    assert correct["layers.0.qkv"][0].shape == naive[0].shape
    assert not torch.equal(correct["layers.0.qkv"][0], naive[0])


def _only(table, name):
    from dataclasses import replace

    return replace(table, specs={name: table.specs[name]})


def test_reshard_rejects_mismatched_layouts():
    src = build_layout_table(TOY, 2)
    dst = build_layout_table(TOY, 4, omit={"qkv"})
    with pytest.raises(ValueError, match="different parameters"):
        reshard(split_params(reference("kv2").full_params(), src), src, dst)


def test_tp4_forward_unavailable_for_kv2():
    """TOY keeps the unrepresentable-layout handling; TOY_KV4 does not replace it."""
    with pytest.raises(UnsupportedLayout, match="n_kv_heads"):
        ShardedModel(TOY, {}, InProcessCollective(4))
    assert 4 not in DEGREES["kv2"]


def test_missing_parameters_are_rejected():
    params = split_params(reference("kv2").full_params(), build_layout_table(TOY, 2))
    del params["layers.0.qkv"]
    with pytest.raises(ValueError, match="missing sharded parameters"):
        ShardedModel(TOY, params, InProcessCollective(2))


def test_threshold_follows_the_stated_rule():
    """The threshold must be derivable from the recorded measurement. If someone
    edits the number by hand, this fails."""
    report = load()
    assert report["threshold"] == derive_threshold(
        report["max_deviation"], report["rule"]["safety_factor"]
    )
    assert report["max_deviation"] == max(r["max_deviation"] for r in report["results"])
    assert report["threshold"] > report["max_deviation"]


def test_recorded_environment_matches_running_environment():
    """Warns, never fails. The threshold's provenance is the environment it was
    measured under; if that has moved, regenerate. But a torch bump should not
    break the suite for someone who just cloned the repo."""
    mismatches = environment_mismatches()
    if mismatches:
        warnings.warn(
            EnvironmentMismatch(
                "tolerance/phase1a.json was measured under a different environment:\n  "
                + "\n  ".join(mismatches)
                + "\nThe threshold may no longer reflect this machine. Regenerate with:"
                "\n  uv run python -m weight_sync_bench.tolerance"
            ),
            stacklevel=2,
        )


def test_token_shape_matches_what_the_floor_was_measured_at(tokens):
    """FAILS, deliberately. The floor is the max over batch*seq_len*vocab elements,
    and the max of ~1e6 draws is a saturated tail statistic -- more elements means a
    higher floor. Raising seq_len here looks harmless and would silently invalidate
    the threshold, so it is asserted rather than trusted."""
    measurement = load()["measurement"]
    assert tuple(tokens.shape) == (measurement["batch"], measurement["seq_len"]), (
        "the invariant tests run on a different token shape than the tolerance floor "
        "was measured at. Regenerate: uv run python -m weight_sync_bench.tolerance"
    )


def test_config_geometry_matches_what_the_floor_was_measured_at():
    """Same reasoning for d_model and n_layers, which set reduction length and
    accumulation depth."""
    recorded = load()["configs"]
    assert recorded.keys() == CONFIGS.keys()
    for name, config in CONFIGS.items():
        assert recorded[name] == dataclasses.asdict(config), (
            f"config {name!r} has changed since the floor was measured. "
            "Regenerate: uv run python -m weight_sync_bench.tolerance"
        )


def test_distribution_statistics_are_recorded_and_unsaturated():
    """median/mean are the comparable-across-changes numbers; max is the conservative
    one the threshold derives from. All three are recorded per cell."""
    for row in load()["results"]:
        if row["max_deviation"] == 0.0:
            continue
        assert 0 < row["median_ulp"] <= row["mean_ulp"] * 2
        assert row["mean_ulp"] < row["max_ulp"], row


def test_artifact_is_labelled_phase_1a_and_carries_provenance():
    report = load()
    assert report["phase"] == "1a"
    for key in ("torch", "numpy", "python", "platform", "dtype", "device"):
        assert report["environment"][key], key


def test_artifact_covers_every_measured_cell():
    """The measurement must span exactly the sweep the tests assert over."""
    measured = {(r["config"], r["src"], r["dst"]) for r in load()["results"]}
    assert measured == set(ORDERED_PAIRS)


def test_every_cell_is_below_the_threshold_it_produced():
    report = load()
    for row in report["results"]:
        assert row["max_deviation"] < report["threshold"], row
        assert len(row["per_seed"]) == report["measurement"]["repetitions"]


def test_kv4_config_shapes():
    """TOY_KV4's qkv is wider, since qkv rows track n_kv_heads."""
    assert isinstance(TOY_KV4, ModelConfig)
    qkv = reference("kv4").full_params()["layers.0.qkv"]
    assert qkv.shape == (512, 256)
    assert reference("kv2").full_params()["layers.0.qkv"].shape == (384, 256)
