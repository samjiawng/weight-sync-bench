"""Phase 3: the prime-rl pin, and the lazy-import discipline around it.

CPU-only, and deliberately so: everything here runs on a box with neither vLLM
nor prime-rl installed. That is not a limitation being worked around, it is the
property under test -- the phase 3 modules have to import on a development
machine so they can be read, reviewed and unit-tested before any GPU time is
spent executing them.
"""

from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

import pytest

from weight_sync_bench.phase3 import engine_probe, step_runner
from weight_sync_bench.phase3.pin import PinError, Pin, pin, provenance

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE3_DIR = REPO_ROOT / "src" / "weight_sync_bench" / "phase3"
PHASE3_SOURCES = sorted(PHASE3_DIR.glob("*.py"))

RECORDED_COMMIT = "26b3131d2716a4f8210b165df584b83b4bc54f61"
RECORDED_TAG = "v0.9.1.dev41"
RECORDED_VLLM = "0.28.0"


def _write_pyproject(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "pyproject.toml"
    path.write_text(textwrap.dedent(body))
    return path


def test_pin_reads_the_recorded_revision():
    p = pin()
    assert p.prime_rl_commit == RECORDED_COMMIT
    assert p.prime_rl_tag == RECORDED_TAG
    assert p.vllm_version == RECORDED_VLLM
    assert p.short_commit == RECORDED_COMMIT[:12]


def test_pinned_vllm_version_is_the_one_the_floor_was_measured_at():
    """The reason this commit was chosen over a release tag.

    A pin whose lock resolved a different vLLM would invalidate phase 2's bf16
    floor rather than reuse it, so the two versions are asserted equal here
    rather than left as a comment. `PINNED_VLLM_VERSION` may carry a local
    suffix (e.g. a +cuXXX build tag); the release part is what has to agree.
    """
    from weight_sync_bench.phase2.bf16_floor import PINNED_VLLM_VERSION

    assert PINNED_VLLM_VERSION.split("+")[0] == pin().vllm_version


def test_commit_is_a_full_sha():
    assert re.fullmatch(r"[0-9a-f]{40}", pin().prime_rl_commit)


def test_provenance_carries_the_pin_and_a_readable_source():
    block = provenance()
    assert block["prime_rl_commit"] == RECORDED_COMMIT
    assert RECORDED_COMMIT in block["source"]


def test_the_sha_is_written_down_exactly_once():
    """`pin.py` is the single source of truth, so no other module may hardcode
    the commit. A second copy is the failure this table exists to prevent: it
    goes stale silently and every artifact stamped from it lies."""
    offenders = [
        path.relative_to(REPO_ROOT)
        for path in (REPO_ROOT / "src").rglob("*.py")
        if RECORDED_COMMIT in path.read_text()
    ]
    assert offenders == [], f"commit hardcoded outside pyproject.toml: {offenders}"


@pytest.mark.parametrize(
    "body, expected",
    [
        ("[tool.weight_sync_bench]\nother = 1\n", "missing"),
        (
            '[tool.weight_sync_bench.phase3]\nprime_rl_commit = "abc123"\n'
            'prime_rl_tag = "t"\nprime_rl_pinned_date = "d"\nvllm_version = "0"\n',
            "40-character",
        ),
        (
            '[tool.weight_sync_bench.phase3]\nprime_rl_tag = "t"\n',
            "missing",
        ),
        (
            f'[tool.weight_sync_bench.phase3]\nprime_rl_commit = "{RECORDED_COMMIT}"\n'
            'prime_rl_tag = "t"\nprime_rl_pinned_date = "d"\nvllm_version = "0"\n'
            'surprise = "x"\n',
            "unrecognized",
        ),
    ],
    ids=["no-table", "short-sha", "incomplete", "unknown-key"],
)
def test_malformed_pin_raises(tmp_path, body, expected):
    with pytest.raises(PinError, match=expected):
        pin(_write_pyproject(tmp_path, body))


def test_missing_pyproject_raises_rather_than_defaulting(tmp_path):
    with pytest.raises(PinError, match="not found"):
        pin(tmp_path / "absent.toml")


# --- lazy-import discipline -------------------------------------------------


def _module_level_imports(path: Path) -> set[str]:
    """Top-level import names only -- imports nested inside a function or class
    body are exactly what this package is supposed to use, so they must not
    count as violations."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize("path", PHASE3_SOURCES, ids=lambda p: p.name)
def test_no_module_level_gpu_imports(path):
    """The check that the acceptance command's `ok` is not an accident: it would
    still print on a box that happened to have vLLM installed."""
    forbidden = {"vllm", "prime_rl", "safetensors", "huggingface_hub"}
    assert not (_module_level_imports(path) & forbidden)


@pytest.mark.parametrize("path", PHASE3_SOURCES, ids=lambda p: p.name)
def test_layout_machinery_is_not_imported(path):
    """`LayoutTable` and the Qwen3 geometry are deliberately unused at this
    stage; importing either would quietly widen phase 3's surface."""
    source = path.read_text()
    assert "qwen3_layout" not in source
    assert "geometry" not in source


def test_probe_modules_import_without_vllm_or_prime_rl():
    # Both are already imported at module scope above; this asserts the thing
    # that matters about them rather than restating the import.
    assert engine_probe.ARTIFACT.name == "phase3_engine_probe.json"
    assert step_runner.ARTIFACT.name == "phase3_step_runner.json"


# --- engine flag table ------------------------------------------------------


def test_flag_table_records_both_profiles_with_sources():
    for name, row in engine_probe.ENGINE_FLAGS.items():
        assert "prime_rl" in row and "floor" in row, name
        assert row["prime_rl_source"] and row["floor_source"], name


def test_floor_profile_matches_what_the_floor_actually_passes():
    """Anti-drift: the table's `floor` column is a transcription of
    `collective_logits._run_worker`'s `LLM(...)` call, and a transcription can
    go stale. Assert against the constructed kwargs, not against the source
    text, so the two cannot disagree."""
    kwargs = engine_probe.engine_kwargs("floor", model_dir="/nonexistent", tp=1)
    for flag in ("enforce_eager", "enable_chunked_prefill", "enable_prefix_caching", "dtype"):
        assert kwargs[flag] == engine_probe.ENGINE_FLAGS[flag]["floor"], flag


def test_extraction_requires_the_phase2_worker_extension_on_both_profiles():
    """Both legs must keep the phase 2 hook: without it there is nothing to read
    logits out of, so a profile that dropped it would compare nothing."""
    from weight_sync_bench.phase2.collective_logits import WORKER_EXTENSION_QUALNAME

    for profile in ("prime_rl", "floor"):
        kwargs = engine_probe.engine_kwargs(profile, model_dir="/nonexistent")
        assert kwargs["worker_extension_cls"] == WORKER_EXTENSION_QUALNAME


def test_the_two_cache_flags_are_the_named_suspects():
    """prime-rl leaves both at vLLM's defaults (on) and `collective_logits`
    documents both as silent failure modes for this extraction path, so they
    lead the attribution order a bit-identity failure is read against."""
    assert engine_probe.SUSPECT_FLAGS[:2] == ("enable_chunked_prefill", "enable_prefix_caching")
    differing = engine_probe.differing_flags()
    assert differing[:2] == ["enable_chunked_prefill", "enable_prefix_caching"]


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="unknown profile"):
        engine_probe.engine_kwargs("whatever", model_dir="/nonexistent")


def test_unknown_broadcast_transport_is_rejected():
    with pytest.raises(ValueError, match="unknown weight-broadcast transport"):
        engine_probe.compose_worker_extension("carrier-pigeon")


def test_composed_worker_qualname_points_at_this_module():
    """vLLM resolves `worker_extension_cls` by qualname inside each worker
    process, so the advertised name has to be the one that resolves."""
    module, _, attr = engine_probe.COMPOSED_WORKER_QUALNAME.rpartition(".")
    assert module == engine_probe.__name__
    assert attr == engine_probe.COMPOSED_WORKER_NAME


def test_unknown_attribute_still_raises_attribute_error():
    """The PEP 562 hook must not turn every typo into an import of prime-rl."""
    with pytest.raises(AttributeError):
        engine_probe.NoSuchThing


# --- step runner ------------------------------------------------------------


def test_step_runner_targets_the_two_gpu_config():
    """prime-rl splits trainer and inference across separate GPUs, so its
    smallest end-to-end loop is two GPUs, not one."""
    assert step_runner.REQUIRED_GPUS == 2
    assert step_runner.RL_CONFIG_RELPATH == "configs/basic/reverse-text/rl.toml"


def test_published_steps_ignores_incomplete_steps(tmp_path):
    """A step dir without the trainer's own completion marker is half-written;
    comparing against it would read torn weights as a parameter change."""
    broadcasts = tmp_path / step_runner.BROADCAST_SUBDIR
    (broadcasts / "step_0").mkdir(parents=True)
    (broadcasts / "step_0" / step_runner.SENDER_READY_MARKER).touch()
    (broadcasts / "step_1").mkdir()  # no marker: still being written
    (broadcasts / "not_a_step").mkdir()
    assert step_runner.published_steps(tmp_path) == [0]


def test_published_steps_sorts_numerically_not_lexically(tmp_path):
    broadcasts = tmp_path / step_runner.BROADCAST_SUBDIR
    for n in (0, 2, 10):
        d = broadcasts / f"step_{n}"
        d.mkdir(parents=True)
        (d / step_runner.SENDER_READY_MARKER).touch()
    assert step_runner.published_steps(tmp_path) == [0, 2, 10]


def test_published_steps_on_a_run_that_produced_nothing(tmp_path):
    assert step_runner.published_steps(tmp_path) == []


def test_compare_steps_detects_movement_and_stillness():
    torch = pytest.importorskip("torch")
    before = {"a": torch.zeros(4), "b": torch.ones(2)}
    after = {"a": torch.tensor([0.0, 0.5, 0.0, 0.0]), "b": torch.ones(2)}
    result = step_runner.compare_steps(before, after)
    assert result["any_parameter_changed"] is True
    assert result["num_changed"] == 1
    assert result["unchanged"] == ["b"]
    assert result["changed"][0]["param"] == "a"
    assert result["changed"][0]["max_abs_delta"] == pytest.approx(0.5)


def test_compare_steps_reports_an_unchanged_policy():
    """The failure this whole module exists to catch: a step that runs cleanly
    and moves nothing."""
    torch = pytest.importorskip("torch")
    weights = {"a": torch.ones(3)}
    result = step_runner.compare_steps(weights, {"a": torch.ones(3)})
    assert result["any_parameter_changed"] is False
    assert result["num_unchanged"] == 1


def test_compare_steps_requires_overlapping_names():
    torch = pytest.importorskip("torch")
    with pytest.raises(step_runner.StepRunnerError, match="share no parameter"):
        step_runner.compare_steps({"a": torch.ones(1)}, {"b": torch.ones(1)})


def test_rl_command_names_the_config_and_output_dir(tmp_path):
    cmd = step_runner.rl_command(tmp_path / "rl.toml", tmp_path / "out", max_steps=1)
    assert cmd[0] == "rl"
    assert str(tmp_path / "rl.toml") in cmd
    assert "1" in cmd


def test_run_one_step_refuses_a_directory_without_the_config(tmp_path):
    with pytest.raises(step_runner.StepRunnerError, match="not found"):
        step_runner.run_one_step(tmp_path, tmp_path / "out")


def test_load_step_weights_without_safetensors_raises(tmp_path):
    with pytest.raises(step_runner.StepRunnerError, match="no safetensors"):
        step_runner.load_step_weights(tmp_path)
