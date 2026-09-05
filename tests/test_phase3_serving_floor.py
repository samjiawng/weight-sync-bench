"""Assembly of the serving-boundary floor, on synthetic legs.

The measurement legs need a GPU, a checkpoint and a running server. The
ASSEMBLY does not: once legs exist it is arithmetic plus phase 2's imported gate
rule, and that is the part where a mistake would silently produce a plausible
artifact. So the legs are synthesized here with known deviations and the
assembly is checked against numbers computed by hand.

What these tests are really guarding is that phase 3 did not grow its own copy
of the gate. Every threshold and verdict below has to come out of phase 2's
`derive_threshold` and `gate_decision`.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from weight_sync_bench.phase2 import bf16_floor
from weight_sync_bench.phase3 import serving_floor as sf

REPS, PROMPTS, ROWS, VOCAB = 3, 2, 4, 8


def _logits(offset: float):
    """Repetitions of prompts, each a fixed tensor plus a constant offset, so
    every element's absolute deviation from the zero-offset reference is exactly
    `offset` and the expected means are known without recomputing them."""
    base = torch.arange(ROWS * VOCAB, dtype=torch.float32).reshape(ROWS, VOCAB)
    return [[base + offset for _ in range(PROMPTS)] for _ in range(REPS)]


BASE_ARGV = ["rl-inference", "@", "cfg.toml", "--model", "/w/clean", "--tp", "1"]


def _launch(model="/w/clean"):
    return {"command": [t if t != "/w/clean" else model for t in BASE_ARGV],
            "cwd": "/opt/wsb/prime-rl"}


def _corruption(case, layer):
    names = sf.expected_corrupted_tensors(case, layer)
    return {"case": case, "layer": layer, "verified": True,
            "expected_tensors": list(names), "changed_tensors": sorted(names),
            "max_abs_delta": {n: 0.5 for n in names}, "checks": "..."}


def _leg(name, case, layer, offset, side="served", self_consistent=True,
         launch=None, served_model=None):
    launch = launch if launch is not None else _launch()
    return {
        "leg": name, "case": case, "layer": layer, "side": side,
        "repetitions": REPS, "prompts": PROMPTS, "seq_len": ROWS, "seed": 0,
        "self_consistent": self_consistent,
        "served_model": served_model or sf._weights_path(launch),
        "corruption": None if case == "clean" else _corruption(case, layer),
        "launch": launch,
        "resolved": {"max_num_batched_tokens": 40960, "chunked_prefill_enabled": False},
        "logits": _logits(offset),
    }


def _write_legs(tmp_path, floor_offset=1e-3, break_offsets=None, **kw):
    break_offsets = break_offsets or {0: 2.0, 13: 0.5}
    legs = [
        {**_leg("direct_reference", "clean", None, 0.0, side="direct"),
         "model_dir": "/w/clean"},
        _leg("clean", "clean", None, floor_offset, **kw),
    ]
    for layer, offset in break_offsets.items():
        for case in bf16_floor.BREAK_CASES:
            legs.append(
                _leg(sf.leg_name(case, layer), case, layer, offset,
                     launch=_launch(f"/w/{case}-{layer}"))
            )
    for leg in legs:
        torch.save(leg, tmp_path / f"{leg['leg']}.pt")
    return tmp_path


def test_floor_mean_is_the_offset_it_was_built_from(tmp_path):
    out = tmp_path / "artifact.json"
    report = sf.assemble(_write_legs(tmp_path), out)
    # rel=1e-3, not tighter: the offset is added to float32 values up to 31, so
    # the stored deviation is the offset after float32 rounding, not the offset.
    assert report["floor"]["mean_deviation"] == pytest.approx(1e-3, rel=1e-3)
    assert report["repetitions"] == REPS


def test_threshold_is_phase_2s_rule_not_a_new_one(tmp_path):
    report = sf.assemble(_write_legs(tmp_path), tmp_path / "a.json")
    floor_mean = report["floor"]["mean_deviation"]
    assert report["floor"]["threshold"] == bf16_floor.derive_threshold(floor_mean)
    assert report["floor"]["safety_factor"] == bf16_floor.SAFETY_FACTOR
    assert report["floor"]["threshold"] == pytest.approx(
        bf16_floor.SAFETY_FACTOR * floor_mean
    )


def test_each_layer_gets_its_own_verdict(tmp_path):
    """A break far above the floor passes; one below the gate margin fails. Both
    verdicts have to appear, because a single-layer artifact is what would let
    the known depth limitation pass unstated."""
    report = sf.assemble(
        _write_legs(tmp_path, floor_offset=1e-3, break_offsets={0: 2.0, 13: 1e-3}),
        tmp_path / "a.json",
    )
    assert report["gate"]["layer_0"]["verdict"] == "pass"
    assert report["gate"]["layer_13"]["verdict"] == "fail"
    assert report["gate_carried_forward"]["layer"] == 0
    assert report["gate_carried_forward"]["verdict"] == "pass"


def test_verdict_matches_phase_2s_gate_decision_exactly(tmp_path):
    report = sf.assemble(_write_legs(tmp_path), tmp_path / "a.json")
    at_zero = [b for b in report["breaks"] if b["layer"] == 0]
    expected = bf16_floor.gate_decision(report["floor"], at_zero)
    assert report["gate"]["layer_0"]["clears"] == expected["clears"]
    assert report["gate"]["layer_0"]["verdict"] == expected["verdict"]


def test_weakest_case_ratio_is_against_the_gate_not_the_threshold(tmp_path):
    """The gate is GATE_MARGIN * threshold, so a ratio against the bare
    threshold would read as twice the headroom actually held."""
    report = sf.assemble(_write_legs(tmp_path), tmp_path / "a.json")
    gate = report["gate"]["layer_0"]
    weakest = min(gate["cases"].values())
    assert gate["weakest_case_ratio_to_gate"] == pytest.approx(
        weakest / (bf16_floor.GATE_MARGIN * report["floor"]["threshold"])
    )


def test_a_server_that_is_not_self_consistent_cannot_be_the_floor(tmp_path):
    with pytest.raises(sf.ServingFloorError, match="self-consistent"):
        sf.assemble(
            _write_legs(tmp_path, self_consistent=False), tmp_path / "a.json"
        )


def test_an_inconsistent_break_leg_is_excluded_rather_than_averaged(tmp_path):
    _write_legs(tmp_path)
    bad = torch.load(tmp_path / "case1_qkv_head_permute@layer0.pt", weights_only=False)
    bad["self_consistent"] = False
    torch.save(bad, tmp_path / "case1_qkv_head_permute@layer0.pt")
    report = sf.assemble(tmp_path, tmp_path / "a.json")
    assert "case1_qkv_head_permute@layer0" in report["excluded_legs"]
    assert "case1_qkv_head_permute" not in report["gate"]["layer_0"]["cases"]


def test_missing_reference_is_refused(tmp_path):
    _write_legs(tmp_path)
    (tmp_path / "direct_reference.pt").unlink()
    with pytest.raises(sf.ServingFloorError, match="direct_reference"):
        sf.assemble(tmp_path, tmp_path / "a.json")


def test_missing_clean_leg_is_refused(tmp_path):
    _write_legs(tmp_path)
    (tmp_path / "clean.pt").unlink()
    with pytest.raises(sf.ServingFloorError, match="floor"):
        sf.assemble(tmp_path, tmp_path / "a.json")


def test_repetition_mismatch_is_refused_rather_than_zipped_short(tmp_path):
    """zip() would silently truncate to the shorter leg and produce a floor over
    fewer repetitions than the artifact claims."""
    _write_legs(tmp_path)
    short = torch.load(tmp_path / "clean.pt", weights_only=False)
    short["logits"] = short["logits"][:1]
    torch.save(short, tmp_path / "clean.pt")
    with pytest.raises(sf.ServingFloorError, match="repetition count differs"):
        sf.assemble(tmp_path, tmp_path / "a.json")


def test_disagreeing_resolved_configs_are_recorded_not_swallowed(tmp_path):
    _write_legs(tmp_path)
    leg = torch.load(tmp_path / "clean.pt", weights_only=False)
    leg["resolved"] = {**leg["resolved"], "max_num_batched_tokens": 2048}
    torch.save(leg, tmp_path / "clean.pt")
    report = sf.assemble(tmp_path, tmp_path / "a.json")
    assert report["resolved_configs"]["agree"] is False
    assert "max_num_batched_tokens" in report["resolved_configs"]["differing"]


def test_artifact_is_written_and_is_json(tmp_path):
    out = tmp_path / "phase3_serving_floor.json"
    sf.assemble(_write_legs(tmp_path), out)
    written = json.loads(out.read_text())
    for key in ("floor", "gate", "breaks", "broadcast_type", "repetitions",
                "resolved_configs", "environment", "prime_rl_pin"):
        assert key in written


def test_both_layers_are_declared_so_a_one_layer_run_is_visible(tmp_path):
    report = sf.assemble(_write_legs(tmp_path), tmp_path / "a.json")
    assert report["layers"] == [0, 13]
    assert set(report["gate"]) == {"layer_0", "layer_13"}


# --- the launch command, which is what makes "started the same way" checkable ---


def test_legs_that_differ_only_in_the_weights_path_are_combined(tmp_path):
    """Break legs are SUPPOSED to point at a corrupted copy. That difference
    alone must not abort the assembly."""
    report = sf.assemble(_write_legs(tmp_path), tmp_path / "a.json")
    assert report["launch"]["normalized_argv_matches_across_legs"] is True
    assert report["launch"]["command"] == BASE_ARGV


def test_a_leg_started_by_a_different_command_aborts_naming_the_difference(tmp_path):
    _write_legs(tmp_path)
    path = tmp_path / "case1_qkv_head_permute@layer0.pt"
    leg = torch.load(path, weights_only=False)
    leg["launch"] = {"command": ["rl-inference", "@", "cfg.toml", "--model",
                                 "/w/x", "--tp", "2"], "cwd": "/opt/wsb/prime-rl"}
    torch.save(leg, path)
    with pytest.raises(sf.ServingFloorError, match="different command"):
        sf.assemble(tmp_path, tmp_path / "a.json")


def test_the_config_argument_is_not_normalized_away(tmp_path):
    """`@ cfg.toml` names the config, not the weights. A leg pointed at a
    different config is a different server, and must not slip through."""
    _write_legs(tmp_path)
    path = tmp_path / "case2_oproj_col_permute@layer0.pt"
    leg = torch.load(path, weights_only=False)
    leg["launch"] = {"command": ["rl-inference", "@", "other.toml", "--model",
                                 "/w/y", "--tp", "1"], "cwd": "/opt/wsb/prime-rl"}
    torch.save(leg, path)
    with pytest.raises(sf.ServingFloorError, match="different command"):
        sf.assemble(tmp_path, tmp_path / "a.json")


def test_a_different_working_directory_aborts(tmp_path):
    _write_legs(tmp_path)
    path = tmp_path / "case3_norm_permute@layer13.pt"
    leg = torch.load(path, weights_only=False)
    leg["launch"] = {**_launch("/w/z"), "cwd": "/somewhere/else"}
    torch.save(leg, path)
    with pytest.raises(sf.ServingFloorError, match="working"):
        sf.assemble(tmp_path, tmp_path / "a.json")


def test_a_leg_with_no_launch_record_is_refused(tmp_path):
    _write_legs(tmp_path)
    path = tmp_path / "clean.pt"
    leg = torch.load(path, weights_only=False)
    leg["launch"] = None
    torch.save(leg, path)
    with pytest.raises(sf.ServingFloorError, match="no launch command"):
        sf.assemble(tmp_path, tmp_path / "a.json")


def test_served_weights_are_checked_against_the_direct_reference(tmp_path):
    """The attachment probe never recorded which model the server served, so
    that the two sides matched was inferable only from the deviation being
    small. Here it is recorded and checked."""
    report = sf.assemble(_write_legs(tmp_path), tmp_path / "a.json")
    assert report["served_weights"]["clean_leg_matches_direct_reference"] is True
    assert report["served_weights"]["per_leg"]["clean"]["loaded_what_its_launch_named"]


def test_a_server_not_running_the_weights_it_was_told_to_aborts(tmp_path):
    _write_legs(tmp_path)
    path = tmp_path / "clean.pt"
    leg = torch.load(path, weights_only=False)
    leg["served_model"] = "/w/something-else"
    torch.save(leg, path)
    with pytest.raises(sf.ServingFloorError, match="not running the weights"):
        sf.assemble(tmp_path, tmp_path / "b.json")


def test_a_break_leg_that_came_up_clean_aborts_rather_than_reading_as_no_separation(tmp_path):
    """The asymmetric failure: a break leg on the clean checkpoint deviates by
    about the floor and reads as this case failing to separate, which is a false
    statement about the harness rather than about the case."""
    _write_legs(tmp_path)
    path = tmp_path / "case1_qkv_head_permute@layer0.pt"
    leg = torch.load(path, weights_only=False)
    leg["launch"] = _launch("/w/clean")
    leg["served_model"] = "/w/clean"
    torch.save(leg, path)
    with pytest.raises(sf.ServingFloorError, match="clean checkpoint"):
        sf.assemble(tmp_path, tmp_path / "b.json")


def test_every_break_leg_is_recorded_as_distinct_from_clean(tmp_path):
    report = sf.assemble(_write_legs(tmp_path), tmp_path / "a.json")
    breaks = {k: v for k, v in report["served_weights"]["per_leg"].items() if k != "clean"}
    assert len(breaks) == len(bf16_floor.BREAK_CASES) * 2
    assert all(v["distinct_from_clean_checkpoint"] for v in breaks.values())


def test_the_artifact_says_what_the_identity_check_covers(tmp_path):
    """Recorded as a bare match it would read as a content check."""
    report = sf.assemble(_write_legs(tmp_path), tmp_path / "a.json")
    identity = report["served_weights"]["identity_is"]
    assert "not a hash" in identity and "DIRECTORY" in identity


# --- the corruption, verified where it was written -------------------------


def _tensors(layer=0, d=8):
    """A stand-in checkpoint carrying the names the break cases touch."""
    attn = f"model.layers.{layer}.self_attn."
    return {
        attn + "q_proj.weight": torch.arange(d * d, dtype=torch.float32).reshape(d, d),
        attn + "k_proj.weight": torch.arange(d * d, dtype=torch.float32).reshape(d, d) + 1,
        attn + "o_proj.weight": torch.arange(d * d, dtype=torch.float32).reshape(d, d) + 2,
        f"model.layers.{layer}.input_layernorm.weight": torch.arange(d, dtype=torch.float32),
        "model.embed_tokens.weight": torch.ones(d, d),
    }


def test_expected_tensors_match_what_corrupt_checkpoint_actually_touches():
    """Derived from corrupt_checkpoint's branches and asserted against its source.

    ONE-DIRECTIONAL, deliberately noted rather than left to look symmetric. This
    catches `expected_corrupted_tensors` naming a tensor `corrupt_checkpoint`
    never touches. It does NOT catch `corrupt_checkpoint` growing a fourth
    tensor this function does not know about -- and that is the direction that
    bites, because `compare_tensor_sets` would then raise "also changed" on a
    legitimate corruption, an error naming the wrong thing entirely. Closing it
    means parsing corrupt_checkpoint's branches rather than substring-matching
    them.
    """
    import inspect

    source = inspect.getsource(bf16_floor.corrupt_checkpoint)
    for case in bf16_floor.BREAK_CASES:
        for name in sf.expected_corrupted_tensors(case, 0):
            suffix = name.split(".", 3)[-1]        # e.g. self_attn.q_proj.weight
            assert suffix.split(".")[-2] in source, (case, name)


def test_a_landed_corruption_records_the_names_and_deltas():
    src = _tensors()
    corrupted = dict(src)
    attn = "model.layers.0.self_attn."
    corrupted[attn + "o_proj.weight"] = corrupted[attn + "o_proj.weight"] + 3.0
    result = sf.compare_tensor_sets(src, corrupted, "case2_oproj_col_permute", 0)
    assert result["verified"] is True
    assert result["changed_tensors"] == [attn + "o_proj.weight"]
    assert result["max_abs_delta"][attn + "o_proj.weight"] == pytest.approx(3.0)


def test_a_corruption_that_changed_nothing_raises_naming_the_case():
    """The identity-permutation hole: shape-preserving, written, and inert."""
    src = _tensors()
    with pytest.raises(sf.ServingFloorError, match="changed nothing"):
        sf.compare_tensor_sets(src, dict(src), "case2_oproj_col_permute", 0)


def test_a_corruption_that_changed_more_than_it_should_raises():
    src = _tensors()
    corrupted = dict(src)
    attn = "model.layers.0.self_attn."
    corrupted[attn + "o_proj.weight"] = corrupted[attn + "o_proj.weight"] + 3.0
    corrupted["model.embed_tokens.weight"] = corrupted["model.embed_tokens.weight"] + 1.0
    with pytest.raises(sf.ServingFloorError, match="also changed"):
        sf.compare_tensor_sets(src, corrupted, "case2_oproj_col_permute", 0)


def test_case1_requires_both_of_its_tensors_to_move():
    """q alone is the q-only bug corrupt_checkpoint deliberately avoids."""
    src = _tensors()
    corrupted = dict(src)
    attn = "model.layers.0.self_attn."
    corrupted[attn + "q_proj.weight"] = corrupted[attn + "q_proj.weight"] + 1.0
    with pytest.raises(sf.ServingFloorError, match="changed nothing"):
        sf.compare_tensor_sets(src, corrupted, "case1_qkv_head_permute", 0)


def test_a_missing_tensor_name_raises_rather_than_reading_as_unchanged():
    src = _tensors()
    corrupted = {k: v for k, v in src.items() if "o_proj" not in k}
    with pytest.raises(sf.ServingFloorError, match="absent from the checkpoint"):
        sf.compare_tensor_sets(src, corrupted, "case2_oproj_col_permute", 0)


def test_assembly_refuses_a_break_leg_with_no_verified_corruption(tmp_path):
    _write_legs(tmp_path)
    path = tmp_path / "case3_norm_permute@layer0.pt"
    leg = torch.load(path, weights_only=False)
    leg["corruption"] = None
    torch.save(leg, path)
    with pytest.raises(sf.ServingFloorError, match="did not verify their corruption"):
        sf.assemble(tmp_path, tmp_path / "a.json")


def test_the_artifact_carries_the_verification_per_leg(tmp_path):
    report = sf.assemble(_write_legs(tmp_path), tmp_path / "a.json")
    verification = report["corruption_verification"]
    assert len(verification) == len(bf16_floor.BREAK_CASES) * 2
    assert all(v["verified"] for v in verification.values())
