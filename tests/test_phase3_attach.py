"""Phase 3: the live-server attachment, as far as a CPU-only box reaches.

What is testable here is everything except the attachment itself: that the two
patches apply cleanly to a fresh extraction of the pin, that the launcher and
client import with neither vLLM nor prime-rl installed, that the wire encoding
round-trips, and that the precondition check agrees with the in-process one.
Whether the extraction actually attaches to a running server is a GPU question
and is not answered here.

The patch-apply test needs network access to fetch the pinned tarball and is
skipped when that is unavailable, so a clone with no network still runs green.
"""

from __future__ import annotations

import ast
import base64
import json
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

import pytest

from weight_sync_bench.phase3 import attach, engine_probe
from weight_sync_bench.phase3.attach import (
    AttachError,
    HttpEngineAdapter,
    decode_rpc_result,
    extraction_preconditions,
    patch_digests,
    patch_paths,
)
from weight_sync_bench.phase3.pin import pin

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE3_DIR = REPO_ROOT / "src" / "weight_sync_bench" / "phase3"


# --- the patches as files ---------------------------------------------------


def test_both_patches_exist_and_are_ordered():
    paths = patch_paths()
    assert [p.name for p in paths] == list(attach.PATCHES)
    for path in paths:
        assert path.is_file(), path


def test_exactly_two_patches():
    """The two-patch boundary is the point. An accumulating set of diffs against
    a pinned third party is a fork, and a fork of the system under measurement
    invalidates the measurement."""
    on_disk = sorted(p.name for p in (PHASE3_DIR / "patches").glob("*.patch"))
    assert on_disk == sorted(attach.PATCHES)


@pytest.mark.parametrize("path", patch_paths(), ids=lambda p: p.name)
def test_patch_header_names_the_pin_it_applies_to(path):
    """A diff with no recorded target is unauditable: a reader cannot tell what
    tree it was generated against."""
    text = path.read_text()
    assert pin().prime_rl_commit in text
    assert "Applies to:" in text


@pytest.mark.parametrize("path", patch_paths(), ids=lambda p: p.name)
def test_patch_targets_only_the_inference_server(path):
    """Both patches touch one prime-rl file. A patch reaching further than that
    is a different kind of change than this task authorized."""
    targets = {
        line.split("\t")[0][len("+++ b/") :]
        for line in path.read_text().splitlines()
        if line.startswith("+++ b/")
    }
    assert targets == {"src/prime_rl/inference/vllm/server.py"}


def test_patch_digests_cover_every_patch():
    digests = patch_digests()
    assert set(digests) == set(attach.PATCHES)
    assert all(len(d) == 64 for d in digests.values())


# --- the patches actually applying -----------------------------------------


def _fetch_pinned_tree(dest: Path) -> Path:
    url = (
        "https://api.github.com/repos/PrimeIntellect-ai/prime-rl/tarball/"
        + pin().prime_rl_commit
    )
    tarball = dest / "prime-rl.tar.gz"
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            tarball.write_bytes(response.read())
    except Exception as exc:  # noqa: BLE001 - offline is a skip, not a failure
        pytest.skip(f"cannot fetch the pinned tree: {exc}")
    with tarfile.open(tarball) as archive:
        # `filter` became required to avoid a deprecation warning in 3.12+ and
        # is the safe extraction mode regardless.
        archive.extractall(dest, filter="data")
    roots = [p for p in dest.iterdir() if p.is_dir()]
    assert len(roots) == 1, roots
    return roots[0]


@pytest.mark.slow
def test_patches_apply_cleanly_to_a_fresh_extraction(tmp_path):
    """The claim the patch headers make. Verified against a tree fetched at the
    pinned SHA, not against a working copy that may already carry them."""
    if shutil.which("patch") is None:
        pytest.skip("no `patch` binary")
    tree = _fetch_pinned_tree(tmp_path)
    for path in patch_paths():
        result = subprocess.run(
            ["patch", "-p1", "--forward", "-i", str(path)],
            cwd=tree,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{path.name}: {result.stdout}{result.stderr}"
        assert "FAILED" not in result.stdout

    patched = tree / "src/prime_rl/inference/vllm/server.py"
    ast.parse(patched.read_text())  # the patched file still parses
    text = patched.read_text()
    assert "/collective_rpc" in text
    assert 'getattr(args, "worker_extension_cls", None) is None' in text


# --- lazy-import discipline -------------------------------------------------


def test_attach_imports_without_vllm_or_prime_rl():
    assert attach.RPC_ROUTE == "/collective_rpc"
    assert attach.PATCH_DIR.is_dir()


def test_no_module_level_gpu_imports_in_attach():
    tree = ast.parse((PHASE3_DIR / "attach.py").read_text())
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    assert not (names & {"vllm", "prime_rl", "torch"})


# --- the composed class -----------------------------------------------------


def test_composed_qualname_is_what_the_launcher_binds():
    """The launcher binds a qualname string, not a class: vLLM resolves this
    field by qualname inside each worker, and spawn-based workers never see this
    process's dict mutation."""
    assert attach.COMPOSED_WORKER_QUALNAME == engine_probe.COMPOSED_WORKER_QUALNAME
    module, _, name = attach.COMPOSED_WORKER_QUALNAME.rpartition(".")
    assert module == engine_probe.__name__
    assert name == engine_probe.COMPOSED_WORKER_NAME


def test_bind_rejects_an_unknown_transport():
    with pytest.raises(ValueError, match="unknown weight-broadcast transport"):
        attach.bind_worker_extension("carrier-pigeon")


def test_scheduler_summary_is_added_without_a_third_patch():
    """Check 0 needs a resolved-config readback from inside a worker. That is a
    method on the composed class, which this project owns -- deliberately not a
    third patch to prime-rl."""
    source = (PHASE3_DIR / "engine_probe.py").read_text()
    assert "def get_scheduler_config_summary" in source
    for path in patch_paths():
        # Naming the method in the route's allowlist is expected and is not what
        # this guards. What must not happen is prime-rl gaining the
        # IMPLEMENTATION, which is what would make it a third patch.
        assert "def get_scheduler_config_summary" not in path.read_text()


# --- the method allowlist ---------------------------------------------------

EXPECTED_METHODS = {
    "install_logits_hook",
    "retrieve_and_clear_logits",
    "uninstall_logits_hook",
    "get_scheduler_config_summary",
}


def _allowlist_in_patch() -> set[str]:
    """The allowlist as the patch actually declares it.

    Parsed out of the added lines rather than imported, because the patch is the
    thing that runs on the server and this test exists to check the client's
    copy against it."""
    text = (PHASE3_DIR / "patches" / "02-collective-rpc-route.patch").read_text()
    added = "\n".join(
        line[1:] for line in text.splitlines() if line.startswith("+")
    )
    body = added.split("FORWARDABLE_RPC_METHODS = frozenset(", 1)[1].split(")", 1)[0]
    return set(ast.literal_eval(body.strip().rstrip(",")))


def test_the_route_forwards_exactly_four_methods():
    assert _allowlist_in_patch() == EXPECTED_METHODS


def test_client_allowlist_matches_the_patch():
    """Two copies of one list is a drift hazard; this is what makes it safe.
    The client's copy exists so a bad method fails locally rather than as an
    HTTP 400 from a rented box."""
    assert set(attach.FORWARDABLE_RPC_METHODS) == _allowlist_in_patch()


def test_the_route_refuses_anything_outside_the_allowlist():
    """The reason for narrowing: without this the route is general worker
    access, which is a wider capability than logit extraction needs."""
    patch_text = (PHASE3_DIR / "patches" / "02-collective-rpc-route.patch").read_text()
    assert "+    if method not in FORWARDABLE_RPC_METHODS:" in patch_text
    assert "+            status_code=400," in patch_text


@pytest.mark.parametrize(
    "method", ["update_weights_from_path", "init_broadcaster", "__init__", "shutdown"]
)
def test_client_refuses_a_method_outside_the_allowlist(method):
    with pytest.raises(AttachError, match="not forwardable"):
        attach.collective_rpc("http://stub", method)


def test_refusal_names_what_is_allowed():
    with pytest.raises(AttachError, match="install_logits_hook"):
        attach.collective_rpc("http://stub", "something_else")


def test_every_method_the_extraction_path_calls_is_forwardable():
    """The allowlist has to actually cover the flow it exists for: the hook
    install/retrieve pair `run_one_prompt` calls, and the config readback check
    0 needs."""
    for method in ("install_logits_hook", "retrieve_and_clear_logits"):
        assert method in attach.FORWARDABLE_RPC_METHODS
    assert "get_scheduler_config_summary" in attach.FORWARDABLE_RPC_METHODS


def test_a_server_side_refusal_is_reported_as_a_revision_mismatch(monkeypatch):
    """A 400 for a method this client allows means the running server carries a
    different patch revision, which is a specific and fixable condition."""
    import urllib.error

    def raise_400(*args, **kwargs):
        raise urllib.error.HTTPError(
            "http://stub/collective_rpc", 400, "Bad Request", {}, None
        )

    monkeypatch.setattr(urllib.request, "urlopen", raise_400)
    with pytest.raises(AttachError, match="allowlist differs"):
        attach.collective_rpc("http://stub", "install_logits_hook")


# --- the wire encoding ------------------------------------------------------


def test_bytes_round_trip_through_the_wire_encoding():
    """The reason the route needs an encoder at all: the extraction's return
    value is a (dtype, shape, raw bytes) triple and JSON cannot carry the third."""
    raw = b"\x00\x01\xfe\xff" * 8
    encoded = {attach.BYTES_KEY: base64.b64encode(raw).decode("ascii")}
    assert decode_rpc_result(encoded) == raw


def test_decoding_a_full_triple():
    raw = b"abcd"
    wire = ["float32", [2, 3], {attach.BYTES_KEY: base64.b64encode(raw).decode("ascii")}]
    assert decode_rpc_result(wire) == ["float32", [2, 3], raw]


def test_decoding_leaves_ordinary_values_alone():
    assert decode_rpc_result({"a": 1, "b": [2, "x", None]}) == {"a": 1, "b": [2, "x", None]}


def test_a_dict_that_merely_resembles_the_marker_is_not_decoded():
    """The marker is only honoured as the sole key, so a payload that happens to
    carry a similarly named field is not silently reinterpreted as bytes."""
    value = {attach.BYTES_KEY: "AAAA", "other": 1}
    assert decode_rpc_result(value) == {attach.BYTES_KEY: "AAAA", "other": 1}


# --- the adapter ------------------------------------------------------------


class _StubAdapter(HttpEngineAdapter):
    def __init__(self, model="m"):
        self.base_url = "http://stub"
        self.request_timeout = 1.0
        self.model = model
        self.sent = []

    def _post(self, body):
        self.sent.append(body)


def test_adapter_refuses_more_than_one_prompt():
    """`run_one_prompt` sends exactly one; batching would silently change what
    the hook captures."""
    adapter = _StubAdapter()
    with pytest.raises(AttachError, match="one prompt at a time"):
        adapter.generate([{"prompt_token_ids": [1]}, {"prompt_token_ids": [2]}], object())


def test_adapter_only_implements_what_run_one_prompt_calls():
    """Deliberately not a general client. `run_one_prompt` calls exactly these
    two, and anything it grew to need should fail loudly rather than be
    approximated here."""
    for name in ("collective_rpc", "generate"):
        assert callable(getattr(HttpEngineAdapter, name))


# --- preconditions ----------------------------------------------------------


def _preconditions_with(monkeypatch, summary):
    monkeypatch.setattr(attach, "scheduler_config_over_http", lambda base_url: summary)
    return extraction_preconditions(prompt_len=32, base_url="http://stub")


def test_preconditions_pass_when_both_flags_are_off(monkeypatch):
    result = _preconditions_with(
        monkeypatch,
        {
            "max_num_batched_tokens": 8192,
            "chunked_prefill_enabled": False,
            "enable_prefix_caching": False,
        },
    )
    assert result["extraction_can_work"] is True
    assert result["blockers"] == []


def test_preconditions_name_chunked_prefill_as_a_blocker(monkeypatch):
    """Chunking breaks the extraction rather than perturbing it, so this has to
    be caught before a run rather than diagnosed from the raise afterwards."""
    result = _preconditions_with(
        monkeypatch,
        {
            "max_num_batched_tokens": 16,
            "chunked_prefill_enabled": True,
            "enable_prefix_caching": False,
        },
    )
    assert result["extraction_can_work"] is False
    assert result["blockers"] == ["chunked_prefill"]


def test_preconditions_name_prefix_caching_as_a_blocker(monkeypatch):
    result = _preconditions_with(
        monkeypatch,
        {
            "max_num_batched_tokens": 8192,
            "chunked_prefill_enabled": False,
            "enable_prefix_caching": True,
        },
    )
    assert result["blockers"] == ["prefix_caching"]


def test_preconditions_report_both_blockers(monkeypatch):
    result = _preconditions_with(
        monkeypatch,
        {
            "max_num_batched_tokens": 16,
            "chunked_prefill_enabled": True,
            "enable_prefix_caching": True,
        },
    )
    assert result["blockers"] == ["chunked_prefill", "prefix_caching"]


def test_chunked_prefill_enabled_but_budget_above_prompt_is_not_a_blocker(monkeypatch):
    """Enabled and never firing is the condition B1 had to force past; it does
    not block extraction, and calling it a blocker would be wrong."""
    result = _preconditions_with(
        monkeypatch,
        {
            "max_num_batched_tokens": 8192,
            "chunked_prefill_enabled": True,
            "enable_prefix_caching": False,
        },
    )
    assert result["extraction_can_work"] is True


def test_preconditions_share_one_implementation_with_the_in_process_check():
    """Reuse, asserted: the served and in-process paths must not grow two
    definitions of what counts as chunking."""
    from weight_sync_bench.phase3.engine_probe import evidence_from_scheduler

    assert evidence_from_scheduler(16, True, 32)["expected_chunks"] == 2
    assert "evidence_from_scheduler" in (PHASE3_DIR / "attach.py").read_text()


# --- the client's error surface --------------------------------------------


def test_missing_route_is_reported_as_a_missing_patch(monkeypatch):
    """A 404 here means the running server does not carry patch 02, which is a
    setup error with a specific fix -- not a mysterious HTTP failure."""
    import urllib.error

    def raise_404(*args, **kwargs):
        raise urllib.error.HTTPError(
            "http://stub/collective_rpc", 404, "Not Found", {}, None
        )

    monkeypatch.setattr(urllib.request, "urlopen", raise_404)
    with pytest.raises(AttachError, match="02-collective-rpc-route.patch"):
        attach.collective_rpc("http://stub", "install_logits_hook")


def test_rpc_payload_shape(monkeypatch):
    """The route reads `method`, `args` and `timeout`; the client must send
    exactly those names."""
    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({"results": [None]}).encode()

    def fake_urlopen(request, timeout=None):
        captured["body"] = json.loads(request.data)
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    attach.collective_rpc("http://stub", "retrieve_and_clear_logits", args=(32,))
    assert captured["body"] == {
        "method": "retrieve_and_clear_logits",
        "args": [32],
        "timeout": None,
    }
