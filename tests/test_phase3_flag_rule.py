"""Phase 3: the engine-flag acceptance rule, and the controls around it.

CPU-only. Everything here exercises pure functions and artifact reads -- no
engine is constructed, so the rule that decides which threshold phase 3 gates on
is testable long before a GPU is rented. That is the point of it being a
function rather than a paragraph: the decision can be checked without the run.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from weight_sync_bench.phase3 import engine_probe
from weight_sync_bench.phase3.engine_probe import (
    BRANCH_NO_READING,
    BRANCH_REMEASURE,
    BRANCH_TRANSFERS,
    FLOOR_ARTIFACT,
    FloorLookupError,
    apply_rule,
    differing_flags,
    floor_mean,
    scheduler_evidence,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _recorded_rows() -> list[tuple[tuple[int, int, int], float]]:
    """Every (repetitions, batch, seq_len) -> mean the 2a artifact records.

    Read from the artifact rather than transcribed, so these tests never become
    a second copy of a measured number -- the failure the artifact convention
    exists to prevent.
    """
    data = json.loads(FLOOR_ARTIFACT.read_text())
    rows = []
    for result in data["results"]:
        m = result["measurement"]
        rows.append(
            ((m["repetitions"], m["batch"], m["seq_len"]), float(result["floor"]["mean_deviation"]))
        )
    return rows


# --- the rule ---------------------------------------------------------------


def test_deviation_below_the_floor_mean_transfers():
    result = apply_rule(0.01, 0.04)
    assert result["branch"] == BRANCH_TRANSFERS
    assert result["threshold_transfers"] is True
    assert result["multiple_of_floor_mean"] == pytest.approx(0.25)


def test_deviation_above_the_floor_mean_requires_remeasurement():
    result = apply_rule(0.08, 0.04)
    assert result["branch"] == BRANCH_REMEASURE
    assert result["threshold_transfers"] is False
    assert result["multiple_of_floor_mean"] == pytest.approx(2.0)


def test_deviation_exactly_at_the_floor_mean_transfers():
    """The rule says "at or below", and the implementation uses `<=`. Asserted
    explicitly because it is exactly the edge a later refactor flips silently,
    and flipping it would send phase 3 off to re-measure a floor it already has.
    """
    result = apply_rule(0.04, 0.04)
    assert result["branch"] == BRANCH_TRANSFERS
    assert result["threshold_transfers"] is True
    assert result["multiple_of_floor_mean"] == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [0.0, -1e-9, -1.0])
def test_rule_rejects_a_non_positive_floor_mean(bad):
    with pytest.raises(ValueError, match="floor mean must be positive"):
        apply_rule(0.01, bad)


def test_rule_is_a_pure_function_of_its_arguments():
    """No hidden state: the same inputs give the same branch and multiple."""
    a = apply_rule(0.05, 0.04)
    b = apply_rule(0.05, 0.04)
    assert a == b


# --- the third branch -------------------------------------------------------


def test_extraction_raising_is_its_own_branch():
    """Chunked prefill breaks the extraction rather than perturbing it, so
    there is no deviation to divide. That is not `remeasure_required`: a reader
    who saw that branch would go re-measure something that cannot be measured
    this way while the profiles disagree.
    """
    result = apply_rule(None, 0.04, extraction_error="expected at most one ... got 2")
    assert result["branch"] == BRANCH_NO_READING
    assert result["branch"] not in (BRANCH_TRANSFERS, BRANCH_REMEASURE)
    assert result["threshold_transfers"] is False
    assert result["deviation_mean"] is None
    assert result["multiple_of_floor_mean"] is None


def test_the_raise_is_kept_as_runtime_evidence():
    """The error text is the proof that chunking actually occurred, so it has to
    survive into the artifact rather than being reduced to a boolean."""
    message = "expected at most one compute_logits capture ... got 2: shapes [(16, V), (16, V)]"
    result = apply_rule(None, 0.04, extraction_error=message)
    assert result["extraction_error"] == message


def test_an_extraction_error_wins_over_a_deviation():
    """If both are somehow supplied, the absent reading is the stronger fact."""
    result = apply_rule(0.01, 0.04, extraction_error="boom")
    assert result["branch"] == BRANCH_NO_READING


def test_a_missing_deviation_without_an_error_is_a_programming_mistake():
    with pytest.raises(ValueError, match="deviation_mean is required"):
        apply_rule(None, 0.04)


def test_the_three_branches_are_distinct():
    assert len({BRANCH_TRANSFERS, BRANCH_REMEASURE, BRANCH_NO_READING}) == 3


# --- the floor lookup -------------------------------------------------------


def test_floor_mean_returns_the_recorded_value_at_every_covered_row():
    for (repetitions, batch, seq_len), expected in _recorded_rows():
        assert floor_mean(seq_len=seq_len, batch=batch, repetitions=repetitions) == expected


def test_floor_mean_at_the_module_defaults_resolves():
    """The guard that the module's default configuration and the artifact stay
    in agreement. If someone retunes DEFAULT_SEQ_LEN to a value the sweep never
    covered, the probe would have no floor to divide by -- and would find that
    out on the GPU box rather than here."""
    assert floor_mean() in {mean for _, mean in _recorded_rows()}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seq_len": 33},
        {"seq_len": 31},
        {"seq_len": 1024},
        {"batch": 5},
        {"repetitions": 19},
    ],
    ids=["near-miss-high", "near-miss-low", "far", "wrong-batch", "wrong-repetitions"],
)
def test_floor_mean_raises_rather_than_returning_a_neighbour(kwargs):
    """The behavioural claim in `floor_mean`'s docstring. A nearest-neighbour
    regression would silently answer a different question than the rule asks,
    and the near misses are the cases where that would look plausible."""
    with pytest.raises(FloorLookupError):
        floor_mean(**kwargs)


def test_floor_lookup_error_names_what_is_covered():
    with pytest.raises(FloorLookupError, match=r"covers"):
        floor_mean(seq_len=33)


def test_floor_mean_is_not_inlined_anywhere_in_the_source():
    """Phase 1's rule applies unchanged: a measured number lives in its artifact
    and nowhere else."""
    recorded = [str(mean)[:12] for _, mean in _recorded_rows()]
    source = (REPO_ROOT / "src" / "weight_sync_bench" / "phase3" / "engine_probe.py").read_text()
    assert not [value for value in recorded if value in source]


# --- the control flags ------------------------------------------------------


def test_differing_flags_excludes_the_controls():
    """`max_num_batched_tokens` and `worker_extension_cls` differ by
    construction -- they are the instrument, not the subject -- so neither may
    ever be reported as a suspect explanation for a deviation."""
    reported = differing_flags()
    for name in engine_probe.NOT_UNDER_TEST:
        assert name not in reported


def test_every_control_actually_differs_between_the_profiles():
    """Otherwise NOT_UNDER_TEST would be excluding something that was never a
    candidate, which would make the exclusion misleading rather than load-bearing."""
    for name in engine_probe.NOT_UNDER_TEST:
        row = engine_probe.ENGINE_FLAGS[name]
        assert row["prime_rl"] != row["floor"], name


def test_the_forced_budget_is_below_the_probes_prompt_length():
    """The whole point of the control: at the default budget chunked prefill is
    enabled and never fires, and the probe would pass while testing nothing."""
    assert engine_probe.DEFAULT_MAX_NUM_BATCHED_TOKENS < engine_probe.DEFAULT_SEQ_LEN


def test_only_the_prime_rl_profile_carries_the_budget():
    prime_rl = engine_probe.engine_kwargs("prime_rl", model_dir="/nonexistent")
    floor = engine_probe.engine_kwargs("floor", model_dir="/nonexistent")
    assert prime_rl["max_num_batched_tokens"] == engine_probe.DEFAULT_MAX_NUM_BATCHED_TOKENS
    assert "max_num_batched_tokens" not in floor


# --- chunking prediction ----------------------------------------------------


def _stub_engine(max_num_batched_tokens, chunked_prefill_enabled=True):
    """Enough of an engine for `scheduler_evidence` to read a resolved config."""
    scheduler = SimpleNamespace(
        max_num_batched_tokens=max_num_batched_tokens,
        chunked_prefill_enabled=chunked_prefill_enabled,
    )
    return SimpleNamespace(
        llm_engine=SimpleNamespace(vllm_config=SimpleNamespace(scheduler_config=scheduler))
    )


def test_forced_budget_predicts_chunking_into_two_chunks():
    evidence = scheduler_evidence(_stub_engine(16), prompt_len=32)
    assert evidence["config_predicts_chunking"] is True
    assert evidence["expected_chunks"] == 2
    assert evidence["budget_below_prompt_len"] is True


def test_vllm_default_budget_predicts_no_chunking():
    """The condition the control exists to escape: nominally enabled, never fires."""
    evidence = scheduler_evidence(_stub_engine(8192), prompt_len=32)
    assert evidence["config_predicts_chunking"] is False
    assert evidence["budget_below_prompt_len"] is False
    assert evidence["expected_chunks"] == 1


def test_chunked_prefill_disabled_predicts_no_chunking_whatever_the_budget():
    evidence = scheduler_evidence(_stub_engine(16, chunked_prefill_enabled=False), prompt_len=32)
    assert evidence["config_predicts_chunking"] is False


def test_a_budget_that_did_not_take_is_visible():
    """A budget clamped or ignored by vLLM must not read as a successful control.
    The resolved value is read back off the constructed engine for exactly this."""
    evidence = scheduler_evidence(_stub_engine(2048), prompt_len=32)
    assert evidence["resolved_max_num_batched_tokens"] == 2048
    assert evidence["config_predicts_chunking"] is False


def test_scheduler_evidence_survives_a_missing_budget():
    evidence = scheduler_evidence(_stub_engine(None), prompt_len=32)
    assert evidence["expected_chunks"] is None
    assert evidence["config_predicts_chunking"] is False
