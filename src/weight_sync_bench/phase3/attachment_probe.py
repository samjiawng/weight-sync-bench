"""Does the phase 2 extraction path attach to a RUNNING prime-rl server?

Runs on a GPU box only; imports anywhere.

This is the execution stage of the question `attach.py` describes and
`engine_probe.py` could not reach. `engine_probe` compared two in-process
engines at differing FLAG PROFILES because prime-rl's sampler is behind its
OpenAI API server and no in-process `LLM` exists; this module drives that
server for real, with both recorded patches applied, and reads logits out of it
through the added route.

FIVE CHECKS, AND WHY CHECK 0 IS NOT ONE OF THE FIVE
----------------------------------------------------
Check 0 is a PRECONDITION, not a peer of the others. The flag differential
measured what happens to this extraction path once chunked prefill actually
chunks: `retrieve_and_clear_logits` does not return a perturbed tensor, it
raises, because it asserts a single multi-position capture and chunking
produces one per chunk (`tolerance/phase3_engine_probe.json`, outcome
`extraction_raised_under_chunking`). prime-rl leaves chunked prefill at vLLM's
default, which is on.

So if the readback says chunking is on, checks 1 through 4 cannot produce a
reading, and a failed extraction under chunking is not evidence about the
attachment -- it is evidence about the flags. Running them anyway would put
numbers in an artifact that look like attachment results and are not. This
module therefore STOPS after check 0 in that case, and says so in the outcome.

WHAT THE READBACK IS, AND WHY IT IS NOT THE REQUEST
-----------------------------------------------------
A flag that silently failed to apply looks exactly like one that applied, right
up until the extraction raises. So check 0 reads the RESOLVED scheduler config
off the running server's workers, through the same route the extraction uses --
not from what was passed on the command line, and not from a locally
constructed engine that merely received the same arguments. Reaching the
workers at all is itself part of the result: `get_scheduler_config_summary`
exists only on the composed class, so a summary coming back is evidence that
patch 01 delivered the composed extension AND that patch 02's route reaches it.

The evidence is evaluated by `engine_probe.evidence_from_scheduler`, reused
rather than restated, so the served and in-process paths cannot disagree about
what counts as chunking.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..phase2.bf16_floor import MODEL_ID
from . import attach
from .engine_probe import (
    COMPOSED_WORKER_QUALNAME,
    MULTI_CAPTURE_MARKER,
    DEFAULT_BROADCAST_TYPE,
    probe_environment,
)
from .pin import provenance

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = REPO_ROOT / "tolerance" / "phase3_attachment.json"

DEFAULT_BASE_URL = attach.DEFAULT_BASE_URL
DEFAULT_SEQ_LEN = 32
DEFAULT_PROMPTS = 4
DEFAULT_SEED = 0

# The flags the direct engine is built with, matched to what the server
# resolves. Named here rather than inline so check 1's flag table and the engine
# it describes cannot drift apart.
#
# `max_num_batched_tokens` is explicit and is not decoration. With chunked
# prefill off, vLLM's SERVER path defaults the budget to 2048 and then refuses
# to start because that is below `max_model_len` (40960 for this checkpoint) --
# while the in-process `LLM` path resolves it to 40960 and starts fine. Setting
# it explicitly is what makes the two paths comparable, and it matches the value
# the flag differential's floor leg resolved.
MATCHED_ENGINE_FLAGS: dict[str, Any] = {
    "dtype": "bfloat16",
    "enforce_eager": True,
    "enable_chunked_prefill": False,
    "enable_prefix_caching": False,
    "max_num_batched_tokens": 40960,
    "gpu_memory_utilization": 0.85,
    "logprobs_mode": "raw_logits",
    "seed": 0,
    "tensor_parallel_size": 1,
}

# Differences between the two sides of check 1 that are structural rather than
# configurable. Named in the artifact rather than left for a reader to notice:
# an unmatched flag that goes unmentioned is indistinguishable from one that
# matched.
UNMATCHABLE = {
    "serving_path": (
        "The served side runs vLLM's OpenAI API server over an async engine "
        "reached by HTTP; the direct side is a synchronous in-process LLM. That "
        "difference IS what check 1 measures, so it cannot be matched away."
    ),
}


class AttachmentProbeError(RuntimeError):
    """The probe could not carry out a check it was asked to carry out."""


def is_chunking_failure(exc: BaseException) -> bool:
    """Whether an exception is the extraction refusing to read under chunking.

    Matched on the MESSAGE, never on the type. vLLM's executor re-raises a
    worker-side exception across the process boundary as a bare `Exception`
    with the original type lost and the text preserved, and prime-rl's HTTP
    route adds a boundary of its own on top of that. An `except RuntimeError`
    written against `retrieve_and_clear_logits`'s own raise statement never
    fires here -- that was measured, not predicted.
    """
    return MULTI_CAPTURE_MARKER in str(exc)


# --------------------------------------------------------------------------- #
# Check 4: the composition, against the real base class.
# --------------------------------------------------------------------------- #


def check_composition(broadcast_type: str = DEFAULT_BROADCAST_TYPE) -> dict[str, Any]:
    """Assert the composed class carries both parents' methods.

    Asserted against prime-rl's REAL worker, not a stub. A stub would prove only
    that `type()` composes what it is given, which was never in doubt; what is
    in doubt is whether prime-rl's actual worker composes with the hook without
    either shadowing the other.
    """
    from vllm.utils.import_utils import resolve_obj_by_qualname

    from ..phase2.collective_logits import LogitsHookWorkerExtension
    from .engine_probe import PRIME_RL_WORKER_EXTENSIONS, compose_worker_extension

    composed = compose_worker_extension(broadcast_type)
    base = resolve_obj_by_qualname(PRIME_RL_WORKER_EXTENSIONS[broadcast_type])

    def public(cls: type) -> list[str]:
        return sorted(n for n in dir(cls) if not n.startswith("_"))

    prime_rl_methods = public(base)
    hook_methods = public(LogitsHookWorkerExtension)
    missing = [m for m in prime_rl_methods + hook_methods if not hasattr(composed, m)]

    return {
        "resolved_by_qualname": resolve_obj_by_qualname(COMPOSED_WORKER_QUALNAME) is composed,
        "qualname": COMPOSED_WORKER_QUALNAME,
        "mro": [f"{k.__module__}.{k.__qualname__}" for k in composed.__mro__],
        "prime_rl_base": f"{base.__module__}.{base.__qualname__}",
        "prime_rl_base_in_mro": base in composed.__mro__,
        "logits_hook_in_mro": LogitsHookWorkerExtension in composed.__mro__,
        "prime_rl_methods": prime_rl_methods,
        "logits_hook_methods": hook_methods,
        "missing_methods": missing,
        "passed": not missing
        and base in composed.__mro__
        and LogitsHookWorkerExtension in composed.__mro__,
    }


# --------------------------------------------------------------------------- #
# Check 0: the precondition.
# --------------------------------------------------------------------------- #


def check_preconditions(
    base_url: str = DEFAULT_BASE_URL, prompt_len: int = DEFAULT_SEQ_LEN
) -> dict[str, Any]:
    """Read the resolved scheduler config back off the RUNNING server."""
    preconditions = attach.extraction_preconditions(prompt_len, base_url)
    return {
        **preconditions,
        "source": (
            "collective_rpc get_scheduler_config_summary, forwarded by the added "
            "route to the server's workers -- the resolved config inside the "
            "worker processes, not the request that configured them and not a "
            "locally constructed engine"
        ),
        "passed": preconditions["extraction_can_work"],
    }


# --------------------------------------------------------------------------- #
# Check 1: attachment, against a direct engine at matched flags.
# --------------------------------------------------------------------------- #


def _direct_engine_worker(
    model_dir: str, token_batches: list[list[int]], out_path: Path
) -> None:
    """Extract the same prompts from a directly-constructed in-process engine.

    A subprocess, as in phase 2 and in the flag differential: one `LLM` per
    process, because leftover engine state between constructions is not
    something a bit-identity comparison can afford to reason about.
    """
    import torch
    from vllm import LLM

    from ..phase2.collective_logits import run_one_prompt

    llm = LLM(
        model=model_dir,
        worker_extension_cls=COMPOSED_WORKER_QUALNAME,
        **MATCHED_ENGINE_FLAGS,
    )
    # The same readback check 0 makes against the server, made here against the
    # direct engine. Recording both is what turns "the flags were matched" from
    # a claim about the arguments into a claim about the RESOLVED configs, which
    # is the only version of it worth anything -- an argument that was accepted
    # and then clamped looks identical to one that took.
    scheduler = llm.collective_rpc("get_scheduler_config_summary")[0]
    logits = [run_one_prompt(llm, tokens) for tokens in token_batches]
    torch.save({"logits": logits, "scheduler_summary": scheduler}, out_path)


def check_attachment(
    base_url: str,
    model_dir: str,
    python: str,
    prompts: int = DEFAULT_PROMPTS,
    seq_len: int = DEFAULT_SEQ_LEN,
    seed: int = DEFAULT_SEED,
    direct_engine_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Extract through the route, extract directly, compare with `torch.equal`.

    Bit-identity is the right acceptance here, unlike in the flag differential,
    precisely BECAUSE the flags match: this compares two paths to the same
    computation rather than two computations. That is what phase 2b's extraction
    check compared when it earned `torch.equal`.
    """
    import os
    import tempfile

    import torch

    from ..phase2.bf16_floor import QWEN3_0_6B, _seeded_token_batches

    batch = _seeded_token_batches(1, prompts, seq_len, QWEN3_0_6B.vocab, seed)[0]
    token_batches = [row for row in batch.tolist()]

    served: list[Any] = []
    for tokens in token_batches:
        try:
            served.append(attach.run_one_prompt_over_http(tokens, base_url))
        except Exception as exc:  # noqa: BLE001 - classified, then re-raised
            if is_chunking_failure(exc):
                # Should be unreachable: check 0 gates this call. Kept because
                # "unreachable" and "not checked" look the same in an artifact.
                raise AttachmentProbeError(
                    "extraction raised under chunking during check 1, which "
                    "check 0 should have prevented. The server's flags changed "
                    f"between the two checks.\n{exc}"
                ) from exc
            raise

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "direct.pt"
        env = {**os.environ, **(direct_engine_env or {})}
        subprocess.run(
            [
                python,
                "-m",
                "weight_sync_bench.phase3.attachment_probe",
                "--worker",
                "--model-dir", model_dir,
                "--tokens", json.dumps(token_batches),
                "--out", str(out_path),
            ],
            check=True,
            env=env,
        )
        loaded = torch.load(out_path)
        direct = loaded["logits"]
        direct_scheduler = loaded["scheduler_summary"]

    # Extracting one prompt twice from the SAME server separates two very
    # different failures that a bare `torch.equal` false cannot tell apart:
    # a serving path that is nondeterministic, and one that is deterministic
    # but does not reproduce the direct engine's bits. Only the second is a
    # statement about the attachment.
    repeat = attach.run_one_prompt_over_http(token_batches[0], base_url)
    served_self_consistent = torch.equal(served[0], repeat)

    per_prompt = []
    for index, (a, b) in enumerate(zip(served, direct)):
        identical = torch.equal(a, b)
        delta = (a.float() - b.float()).abs()
        per_prompt.append(
            {
                "prompt": index,
                "bit_identical": identical,
                "shape": list(a.shape),
                "max_abs_diff": float(delta.max()),
                "mean_abs_diff": float(delta.mean()),
            }
        )

    from .engine_probe import floor_mean

    mean_deviation = sum(r["mean_abs_diff"] for r in per_prompt) / len(per_prompt)
    reference = floor_mean(seq_len=32, batch=4, repetitions=20)

    return {
        "bit_identical": all(row["bit_identical"] for row in per_prompt),
        "served_self_consistent": served_self_consistent,
        "server_scheduler_summary": attach.scheduler_config_over_http(base_url),
        "direct_scheduler_summary": direct_scheduler,
        "resolved_configs_agree": (
            attach.scheduler_config_over_http(base_url) == direct_scheduler
        ),
        # Context, not acceptance. Bit-identity is the acceptance; this says how
        # far off a failure is, in the one unit this project already has for
        # "small" -- the dtype-and-reduction-order floor phase 2a measured.
        "mean_deviation": mean_deviation,
        "floor_mean_reference": {
            "artifact": "phase2a_bf16_floor_v2.json",
            "repetitions": 20, "batch": 4, "seq_len": 32,
            "mean_deviation": reference,
        },
        "multiple_of_floor_mean": mean_deviation / reference,
        "prompts": len(per_prompt),
        "seq_len": seq_len,
        "seed": seed,
        "per_prompt": per_prompt,
        "max_abs_diff": max(row["max_abs_diff"] for row in per_prompt),
        "matched_flags": dict(MATCHED_ENGINE_FLAGS),
        "unmatchable": UNMATCHABLE,
        "served_side": "prime-rl server, logits read through the added route",
        "direct_side": "in-process vLLM LLM in a subprocess, same flags",
        "passed": all(row["bit_identical"] for row in per_prompt),
    }


# --------------------------------------------------------------------------- #
# Check 3: prime-rl's own RPCs still work through the composed class.
# --------------------------------------------------------------------------- #


def check_prime_rl_rpcs(base_url: str, weight_dir: str) -> dict[str, Any]:
    """Exercise prime-rl's own weight-update path with the composition bound.

    Both calls reach methods that exist ONLY on prime-rl's base worker, so they
    fail if the composition shadowed the parent. `/update_weights` is the load-
    bearing one: `/liveness` is a no-op probe, while `update_weights_from_path`
    is the RPC the RL loop actually depends on. Reloading the same checkpoint is
    a real exercise of that path whose expected effect on the weights is none,
    which is what lets check 1's comparison stay valid afterwards.
    """
    import urllib.error
    import urllib.request

    results: dict[str, Any] = {}
    for name, route, payload in (
        ("liveness", "/liveness", None),
        ("update_weights", "/update_weights", {"weight_dir": weight_dir}),
    ):
        request = urllib.request.Request(
            base_url.rstrip("/") + route,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={} if payload is None else {"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                results[name] = {
                    "status_code": response.status,
                    "body": json.loads(response.read()),
                }
        except urllib.error.HTTPError as exc:
            results[name] = {
                "status_code": exc.code,
                "error": exc.read().decode(errors="replace")[:2000],
            }

    passed = all(
        row.get("body", {}).get("status") == "ok" for row in results.values()
    )
    return {
        "routes": results,
        "weight_dir": weight_dir,
        "note": (
            "liveness_probe and update_weights_from_path are prime-rl's own "
            "worker methods, present only through the composed class's "
            "prime-rl parent; reaching them is what shows the subclassing did "
            "not break what it subclasses"
        ),
        "passed": passed,
    }


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #

OUTCOME_STOPPED = "stopped_at_precondition"
OUTCOME_RAN = "checks_ran"


def run_probe(
    base_url: str = DEFAULT_BASE_URL,
    model_dir: str | None = None,
    python: str = sys.executable,
    prompts: int = DEFAULT_PROMPTS,
    seq_len: int = DEFAULT_SEQ_LEN,
    seed: int = DEFAULT_SEED,
    step_runner_result: dict[str, Any] | None = None,
    direct_engine_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run check 0, and the rest only if check 0 permits it."""
    if model_dir is None:
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(MODEL_ID)

    common = {
        "probe": "live prime-rl server attachment",
        "question": (
            "Can phase 2's logit-extraction path be attached to a running "
            "prime-rl server, and does a policy update complete with it in place?"
        ),
        "base_url": base_url,
        "model_dir": model_dir,
        "patches": {
            "names": list(attach.PATCHES),
            "sha256": attach.patch_digests(),
            "count": len(attach.PATCHES),
            "boundary": (
                "The attachment is exactly these two. Anything further is "
                "reported rather than built, because an accumulating patch set "
                "against a pinned third party is a fork, and a fork of the "
                "system under measurement invalidates the measurement."
            ),
        },
        "forwardable_rpc_methods": sorted(attach.FORWARDABLE_RPC_METHODS),
        "prime_rl_pin": provenance(),
        "environment": probe_environment(),
    }

    check_0 = check_preconditions(base_url, seq_len)
    if not check_0["passed"]:
        # The hard stop. Checks 1-4 cannot produce a reading under chunking, and
        # a failed extraction there would say nothing about the attachment.
        return {
            **common,
            "outcome": OUTCOME_STOPPED,
            "check_0_chunked_prefill_off": check_0,
            "stopped_because": (
                "check 0 failed: the running server's resolved config blocks "
                f"extraction ({', '.join(check_0['blockers'])}). Checks 1-4 were "
                "NOT run. Under chunking the extraction raises rather than "
                "returning perturbed logits, so those checks cannot produce a "
                "reading, and their failure would be evidence about the flags "
                "rather than about the attachment."
            ),
            "evidence_for_that_rule": "tolerance/phase3_engine_probe.json",
        }

    check_1 = check_attachment(
        base_url, model_dir, python, prompts, seq_len, seed, direct_engine_env
    )
    check_3 = check_prime_rl_rpcs(base_url, model_dir)
    check_4 = check_composition()

    checks = {
        "check_0_chunked_prefill_off": check_0,
        "check_1_attachment": check_1,
        "check_2_policy_update": step_runner_result
        or {"passed": None, "note": "not run in this invocation"},
        "check_3_prime_rl_rpcs": check_3,
        "check_4_composition": check_4,
    }
    return {
        **common,
        "outcome": OUTCOME_RAN,
        **checks,
        "summary": {
            name: checks[name]["passed"] for name in checks
        },
    }


def write(report: dict[str, Any], path: Path = ARTIFACT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--tokens", type=str, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--prompts", type=int, default=DEFAULT_PROMPTS)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--step-runner-artifact",
        type=Path,
        default=None,
        help="a phase3_step_runner.json to fold in as check 2",
    )
    args = parser.parse_args()

    if args.worker:
        assert args.model_dir and args.tokens and args.out
        _direct_engine_worker(args.model_dir, json.loads(args.tokens), args.out)
        return

    step_runner_result = None
    if args.step_runner_artifact is not None and args.step_runner_artifact.is_file():
        step = json.loads(args.step_runner_artifact.read_text())
        comparison = step["comparison"]
        step_runner_result = {
            "passed": comparison["any_parameter_changed"],
            "compared_steps": step["compared_steps"],
            "all_published_steps": step["all_published_steps"],
            "num_changed": comparison["num_changed"],
            "num_unchanged": comparison["num_unchanged"],
            "num_parameters_compared": comparison["num_parameters_compared"],
            "largest_deltas": comparison["changed"][:5],
            "artifact": args.step_runner_artifact.name,
        }

    report = run_probe(
        args.base_url,
        args.model_dir,
        prompts=args.prompts,
        seq_len=args.seq_len,
        seed=args.seed,
        step_runner_result=step_runner_result,
    )
    path = write(report, args.out or ARTIFACT)
    if report["outcome"] == OUTCOME_STOPPED:
        print(f"{OUTCOME_STOPPED}: {report['check_0_chunked_prefill_off']['blockers']} -> {path}")
    else:
        print(f"{report['summary']} -> {path}")


if __name__ == "__main__":
    main()
