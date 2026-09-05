"""Attach the phase 2 logit-extraction path to a running prime-rl server.

Runs on a GPU box only; imports anywhere.

THE SEAM, AND WHY IT IS TWO HALVES
-----------------------------------
prime-rl's sampler is behind its OpenAI API server: there is no in-process `LLM`
object, and `phase2.collective_logits.run_one_prompt` calls `collective_rpc` and
`generate` on a local one. Attaching therefore needs two things, and they are
independent:

1. **The hook has to be inside the workers.** `worker_extension_cls` names
   exactly one class, so the logits hook cannot be supplied alongside prime-rl's
   weight-update worker -- it has to be mixed into it. That composed class is
   built per transport by `engine_probe.compose_worker_extension` and named
   by `engine_probe.composed_worker_qualname`. Getting prime-rl to use it
   has two routes, and which one applies depends on who starts the server:

   - `serve()` below rebinds `WORKER_EXTENSION_CLS` in-process before calling
     prime-rl's `server()`, which reads the dict at call time. This needs NO
     patch. It works only when this process is the one starting the server.
   - When prime-rl's own orchestration spawns the inference process, an
     in-process rebind cannot reach it, and the class has to arrive through
     prime-rl's vLLM config passthrough instead -- which needs
     `patches/01-worker-extension-passthrough.patch`, because that assignment is
     the one field the passthrough does not survive.

2. **Something has to be able to call the hook.** All three of prime-rl's
   collective_rpc routes hardcode a method name, so
   `patches/02-collective-rpc-route.patch` adds a general forwarder. This half
   has no patch-free route; it is the reason the attachment is not purely a
   launcher.

`HttpEngineAdapter` then presents that HTTP surface as the small object
`run_one_prompt` expects, so the extraction logic itself is REUSED rather than
reimplemented against HTTP. That matters: a second copy of the multi-rank gather
handling would be free to drift from the one phase 2 verified.

The flags are deliberately not set here. Chunked prefill in particular has to be
off for the extraction to work at all -- it breaks `retrieve_and_clear_logits`
rather than perturbing it -- but it reaches the engine through prime-rl's own
config passthrough, and hardcoding it here would hide whether that passthrough
actually works.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .engine_probe import (
    DEFAULT_BROADCAST_TYPE,
    PRIME_RL_WORKER_EXTENSIONS,
    composed_worker_qualname,
)

PATCH_DIR = Path(__file__).resolve().parent / "patches"
PATCHES = (
    "01-worker-extension-passthrough.patch",
    "02-collective-rpc-route.patch",
)

RPC_ROUTE = "/collective_rpc"
BYTES_KEY = "__bytes_b64__"
DEFAULT_BASE_URL = "http://localhost:8000"

# Must match FORWARDABLE_RPC_METHODS in patch 02. Declared here too so a
# mistyped method fails locally with a useful message instead of as an HTTP 400
# from a server that may be a rented box away; a test asserts the two agree, so
# the copy cannot drift from the patch it mirrors.
FORWARDABLE_RPC_METHODS = frozenset(
    {
        "install_logits_hook",
        "retrieve_and_clear_logits",
        "uninstall_logits_hook",
        "get_scheduler_config_summary",
    }
)


class AttachError(RuntimeError):
    """The extraction path could not be attached, or the server disagreed."""


def patch_paths() -> list[Path]:
    """The recorded diffs, in apply order."""
    return [PATCH_DIR / name for name in PATCHES]


def patch_digests() -> dict[str, str]:
    """sha256 of each patch, for the artifact.

    A patch is a change to the system under measurement, so which exact bytes
    were applied is part of the provenance of any number produced afterwards.
    """
    import hashlib

    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in patch_paths()
    }


# --------------------------------------------------------------------------- #
# Server side: bind the composed extension, then hand off to prime-rl.
# --------------------------------------------------------------------------- #


def bind_worker_extension(broadcast_type: str = DEFAULT_BROADCAST_TYPE) -> str:
    """Rebind prime-rl's worker-extension dict to the composed class.

    Returns the qualname now bound, so a caller can record what it installed.

    A qualname STRING is bound, not a class object: vLLM resolves this field by
    qualname inside each worker process, and with spawn-based workers the child
    never sees this process's dict mutation. The string is what travels.

    The name bound is the TRANSPORT'S name, not the bare one. Binding the bare
    name here would install the filesystem composition under every key,
    including NCCL -- which `rl` auto-selects for any run with an inference
    server and no LoRA -- and that mismatch does not fail at bind time. It fails
    later inside prime-rl's weight-update path with
    "init_broadcaster() takes 1 positional argument but 8 were given", an error
    that names the method rather than the transport it was built for. The key
    and the value have to agree about the transport, and only
    `composed_worker_qualname` makes them.
    """
    if broadcast_type not in PRIME_RL_WORKER_EXTENSIONS:
        raise ValueError(
            f"unknown weight-broadcast transport {broadcast_type!r}; "
            f"expected one of {sorted(PRIME_RL_WORKER_EXTENSIONS)}"
        )
    from prime_rl.inference.vllm import server as prime_rl_server

    qualname = composed_worker_qualname(broadcast_type)
    prime_rl_server.WORKER_EXTENSION_CLS[broadcast_type] = qualname
    return qualname


def serve(config: Any, broadcast_type: str | None = None) -> None:
    """Bind the composed extension, then start prime-rl's server.

    `config` is a prime-rl `InferenceConfig`. The rebind happens before
    `server()` is called, which is what makes it take: `server()` reads
    WORKER_EXTENSION_CLS at call time, whereas the `vllm.general_plugins` entry
    point prime-rl registers loads later, inside `EngineArgs.__post_init__`.
    """
    from prime_rl.inference.vllm.server import server

    resolved = broadcast_type or getattr(
        getattr(config, "weight_broadcast", None), "type", DEFAULT_BROADCAST_TYPE
    )
    bind_worker_extension(resolved)
    server(config)


# --------------------------------------------------------------------------- #
# Client side: the added route, and the adapter that reuses run_one_prompt.
# --------------------------------------------------------------------------- #


def decode_rpc_result(value: Any) -> Any:
    """Inverse of the patch's `_encode_rpc_result`."""
    import base64

    if isinstance(value, dict) and set(value) == {BYTES_KEY}:
        return base64.b64decode(value[BYTES_KEY])
    if isinstance(value, list):
        return [decode_rpc_result(item) for item in value]
    if isinstance(value, dict):
        return {key: decode_rpc_result(item) for key, item in value.items()}
    return value


def collective_rpc(
    base_url: str,
    method: str,
    args: tuple = (),
    timeout: float | None = None,
    request_timeout: float = 600.0,
) -> list[Any]:
    """Call one worker RPC through the added route. Returns every rank's result.

    `retrieve_and_clear_logits` returns a `(dtype, shape, bytes)` triple, and the
    triple survives the round trip as a list whose third element was base64
    encoded -- so it comes back as a list, not a tuple. Rebuilt as a tuple here
    because that is what `_reconstruct_tensor` expects.
    """
    import urllib.error
    import urllib.request

    if method not in FORWARDABLE_RPC_METHODS:
        raise AttachError(
            f"method {method!r} is not forwardable; the route allows "
            f"{sorted(FORWARDABLE_RPC_METHODS)}"
        )

    payload = json.dumps(
        {"method": method, "args": list(args), "timeout": timeout}
    ).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + RPC_ROUTE,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=request_timeout) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:2000]
        if exc.code == 400:
            # The server's allowlist refused it while ours did not, which means
            # the running server carries a different patch revision than this
            # client -- worth saying, since the two are versioned separately.
            raise AttachError(
                f"{RPC_ROUTE} refused method {method!r}. This client allows "
                f"{sorted(FORWARDABLE_RPC_METHODS)}, so the running server's "
                f"allowlist differs from this patch revision.\n{detail}"
            ) from exc
        raise AttachError(
            f"{RPC_ROUTE} returned {exc.code} for method {method!r}. If this is "
            f"404, patches/02-collective-rpc-route.patch is not applied to the "
            f"running server.\n{detail}"
        ) from exc

    results = [decode_rpc_result(item) for item in body["results"]]
    return [tuple(item) if isinstance(item, list) else item for item in results]


class HttpEngineAdapter:
    """Presents a served engine as the object `run_one_prompt` expects.

    Only the two methods that function actually calls are implemented, and
    deliberately no more: this is an adapter over one extraction path, not a
    general client. Anything else `run_one_prompt` grew to need should fail
    loudly here rather than be quietly approximated.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str | None = None,
        request_timeout: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.request_timeout = request_timeout
        self.model = model or self._first_served_model()

    def _get(self, route: str) -> Any:
        import urllib.request

        with urllib.request.urlopen(
            self.base_url + route, timeout=self.request_timeout
        ) as response:
            return json.loads(response.read())

    def _first_served_model(self) -> str:
        data = self._get("/v1/models")["data"]
        if not data:
            raise AttachError(f"{self.base_url}/v1/models served no models")
        return data[0]["id"]

    def collective_rpc(self, method: str, args: tuple = (), timeout: float | None = None):
        return collective_rpc(
            self.base_url, method, args, timeout, request_timeout=self.request_timeout
        )

    def generate(self, prompts, sampling_params, use_tqdm: bool = True):
        """Teacher-forced completion over /v1/completions.

        The return value is intentionally unused by `run_one_prompt`, which
        reads its logits out of the worker hook rather than out of the response;
        the call exists to make the engine run a prefill over the prompt.
        """
        import urllib.error
        import urllib.request

        if len(prompts) != 1:
            raise AttachError(f"one prompt at a time; got {len(prompts)}")
        token_ids = prompts[0]["prompt_token_ids"]

        body: dict[str, Any] = {
            "model": self.model,
            "prompt": token_ids,
            "max_tokens": getattr(sampling_params, "max_tokens", 1),
            "temperature": getattr(sampling_params, "temperature", 0.0),
        }
        prompt_logprobs = getattr(sampling_params, "prompt_logprobs", None)
        if prompt_logprobs is not None:
            # What makes the engine compute logits over the whole prompt rather
            # than only the sampled position -- the same reason run_one_prompt
            # sets it on SamplingParams.
            body["prompt_logprobs"] = prompt_logprobs

        request = urllib.request.Request(
            self.base_url + "/v1/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:2000]
            raise AttachError(f"/v1/completions returned {exc.code}\n{detail}") from exc


def run_one_prompt_over_http(
    token_ids: list[int],
    base_url: str = DEFAULT_BASE_URL,
    model: str | None = None,
):
    """Extract one prompt's logits from a running prime-rl server.

    Delegates to `phase2.collective_logits.run_one_prompt` unchanged. The
    multi-rank gather handling, the bit-identity check across ranks and the
    trim are all phase 2's, verified there; this module only supplies the
    transport.
    """
    from ..phase2.collective_logits import run_one_prompt

    return run_one_prompt(HttpEngineAdapter(base_url, model), token_ids)


def scheduler_config_over_http(base_url: str = DEFAULT_BASE_URL) -> dict[str, Any]:
    """Read the resolved scheduler config back off a RUNNING server.

    A flag that silently failed to apply looks exactly like one that applied,
    right up until the extraction raises, so the check has to read the resolved
    value rather than trust the request. Routed through the worker RPC because
    the workers are where the resolved config lives.
    """
    results = collective_rpc(base_url, "get_scheduler_config_summary")
    non_none = [r for r in results if r is not None]
    if not non_none:
        raise AttachError(
            "no rank returned a scheduler config summary; the composed worker "
            "extension is probably not bound on this server"
        )
    return non_none[0]


def extraction_preconditions(
    prompt_len: int, base_url: str = DEFAULT_BASE_URL
) -> dict[str, Any]:
    """Whether a running server is configured such that extraction can work.

    Both flags have to be off. Chunked prefill breaks `retrieve_and_clear_logits`
    rather than perturbing it -- it asserts a single multi-position capture and
    chunking produces one per chunk -- and prefix caching can skip recomputation
    for a repeated prompt, which silently makes repetitions non-independent.

    Reuses `engine_probe`'s evidence check rather than restating it, so the
    served and in-process paths cannot disagree about what counts as chunking.
    """
    from .engine_probe import evidence_from_scheduler

    summary = scheduler_config_over_http(base_url)
    evidence = evidence_from_scheduler(
        summary["max_num_batched_tokens"], summary["chunked_prefill_enabled"], prompt_len
    )
    prefix_caching = summary["enable_prefix_caching"]
    return {
        "resolved": summary,
        "chunking_evidence": evidence,
        "prefix_caching_enabled": prefix_caching,
        "extraction_can_work": (
            not evidence["config_predicts_chunking"] and not prefix_caching
        ),
        "blockers": [
            name
            for name, blocked in (
                ("chunked_prefill", evidence["config_predicts_chunking"]),
                ("prefix_caching", bool(prefix_caching)),
            )
            if blocked
        ],
    }
