"""Does the phase 2 logit-extraction path attach to a prime-rl-hosted engine?

Runs on a GPU box only; imports anywhere.

WHAT THE SOURCE READ SETTLED, AND HOW IT CHANGED THIS MODULE'S SHAPE
--------------------------------------------------------------------
The original sketch for this probe was "construct a prime-rl-hosted sampler and
a direct vLLM engine at matched flags, extract logits from each via
`collective_logits.run_one_prompt`, compare with `torch.equal`". Two facts
about prime-rl at the pinned commit make the first half of that impossible as
written, and both are findings about what this phase needs rather than defects
to route around quietly:

1. **prime-rl has no in-process `LLM` object.** Its only engine construction is
   `server()` in `src/prime_rl/inference/vllm/server.py`, which ends in vLLM's
   OpenAI API server (`run_server` / `run_headless` / `run_multi_api_server`).
   The engine is reached as `request.app.state.engine_client`, an
   `EngineClient`, from inside FastAPI request handlers. `run_one_prompt` calls
   `llm.collective_rpc(...)` and `llm.generate(...)` on a local object, so it
   does not attach to prime-rl's sampler at all. prime-rl does expose
   `collective_rpc` over HTTP, but only through routes that each hardcode one
   RPC method name (`/update_weights`, `/liveness`, `/init_broadcaster`); there
   is no generic pass-through route and no `generate` equivalent.

2. **A caller-supplied `worker_extension_cls` is dropped.** prime-rl's config
   forwards it -- the vLLM config section is `extra="allow"` and `to_namespace`
   copies every field and extra onto the argparse `Namespace` -- but `server()`
   then overwrites it unconditionally from a module-level dict keyed by the
   weight-broadcast transport, with no check for an existing value. The
   `vllm.general_plugins` entry point prime-rl registers does not help: plugins
   load in `EngineArgs.__post_init__`, which happens inside `run_server(args)`,
   i.e. after the overwrite has already happened.

So this module does NOT compare "prime-rl's engine" against "our engine" --
that comparison needs a serving-path change that does not exist yet. It
compares the two ENGINE FLAG PROFILES, both constructed in-process, which is
the part of the question that is answerable now and the part the GPU run of this
differential actually spends its time on: whether prime-rl's flag choices change
the logits the correctness gate reads.

`ENGINE_FLAGS` records both profiles side by side with the source citation for
every value, and the artifact carries it, so a bit-identity failure on the GPU
box can be attributed to a specific flag without a second source read.

THE SEAM, FOR WHEN THE SERVING PATH IS ADDRESSED
-------------------------------------------------
prime-rl resolves its worker extension by qualname out of a mutable
module-level dict, read at `server()` call time. A launcher in this repo can
import `prime_rl.inference.vllm.server`, rebind the entry for the configured
transport to a class that subclasses BOTH prime-rl's worker extension and
`LogitsHookWorkerExtension`, and then call `server(config)` -- and the ordering
works, because the dict read happens when `server()` runs, after the rebind.
`PrimeRlLogitsHookWorker` below is that composed class. Composition by
subclassing is forced: `worker_extension_cls` names exactly one class, so the
logits hook cannot be supplied alongside prime-rl's weight-update worker, only
mixed into it. That keeps prime-rl's weight-update RPCs working, which the RL
loop needs.

This is the smaller of the two options. The other is a one-line change in
prime-rl making the overwrite conditional on the value being unset -- smaller
to write, but it lands in prime-rl rather than here, so it would have to be
carried as a patch against a pinned third-party commit.

Nothing above is verified by execution. The composed class is exercised only as
far as a CPU-only box allows (that it is constructible and carries both sets of
methods); attaching it to a live prime-rl server is GPU work.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..phase2.bf16_floor import MODEL_ID, environment
from ..phase2.collective_logits import WORKER_EXTENSION_QUALNAME
from .pin import provenance

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = REPO_ROOT / "tolerance" / "phase3_engine_probe.json"
FLOOR_ARTIFACT = REPO_ROOT / "tolerance" / "phase2a_bf16_floor_v2.json"

# Matched to a point the recorded 2a sweep actually covers (repetitions=20,
# batch=4, seq_len in {8, 32, 128, 512}), so the deviation this probe measures
# can be divided by a floor mean measured at the SAME configuration rather than
# by whichever one is nearest to hand.
DEFAULT_REPETITIONS = 20
DEFAULT_BATCH = 4
DEFAULT_SEQ_LEN = 32
DEFAULT_SEED = 0

# Forced onto the prime-rl leg only. vLLM's default batched-token budget is far
# above a 32-token prompt, so leaving it at the default would run a single
# unchunked prefill: the probe would report agreement while leaving its own
# first-suspect flag entirely unexercised. 16 against a 32-token prompt splits
# it into two chunks. Lowering the budget rather than raising the prompt is
# deliberate -- it isolates the mechanism instead of also changing the input.
DEFAULT_MAX_NUM_BATCHED_TOKENS = 16

# How the GPU box's environment for this probe is built, recorded in every
# artifact this module writes so the numbers carry the install that produced
# them.
#
# The editable install carries `--no-deps` deliberately, and the reason is a
# fact about the measurement rather than a packaging convenience. This package
# pins numpy==2.4.6 for phase 1; vllm==0.28.0 caps numpy below 2.4 through
# mistral-common; the two do not resolve together, which is exactly the
# conflict pyproject.toml's own comment predicts. Nothing under phase2/ or
# phase3/ imports numpy -- the only uses are `.numpy()` on torch tensors -- so
# the pin is not load-bearing for the code this probe runs. Dropping it leaves
# the environment at the numpy vLLM resolves, which is the environment the
# committed 2a floor was measured in, and that is the property the rule's
# division by that floor mean depends on. `--no-deps` puts this package on the
# path without its dependency set; it changes no pin in pyproject.toml.
INSTALL_COMMANDS = (
    "uv venv --python 3.12 .venv-phase3",
    "uv pip install --python .venv-phase3 vllm==0.28.0 safetensors huggingface-hub",
    "uv pip install --python .venv-phase3 -e . --no-deps",
)

# Substring of the error `retrieve_and_clear_logits` raises when more than one
# multi-position capture arrives. Matched on rather than parsed: it is the
# runtime signature of chunking having actually happened.
MULTI_CAPTURE_MARKER = "expected at most one compute_logits capture"

# The two flag profiles, with the source each value was read from. prime-rl
# paths are at the pinned commit; this repo's are at the working tree.
#
# The two cache flags are the load-bearing rows. `collective_logits` documents
# both as SILENT failure modes for this extraction method -- chunked prefill
# makes `compute_logits` fire once per scheduling chunk instead of once for the
# whole prompt, and prefix caching can skip recomputation for a repeated prompt
# entirely -- and prime-rl leaves both at vLLM's defaults, which are on. Neither
# raises or changes a tensor shape on its own.
ENGINE_FLAGS: dict[str, dict[str, Any]] = {
    "enforce_eager": {
        "prime_rl": False,
        "prime_rl_source": "packages/prime-rl-configs/src/prime_rl/configs/inference.py:66",
        "floor": True,
        "floor_source": "src/weight_sync_bench/phase2/collective_logits.py:1257",
    },
    "enable_chunked_prefill": {
        "prime_rl": None,
        "prime_rl_note": "not a prime-rl config field at all; vLLM's default applies, i.e. on",
        "prime_rl_source": "packages/prime-rl-configs/src/prime_rl/configs/inference.py (absent)",
        "floor": False,
        "floor_source": "src/weight_sync_bench/phase2/collective_logits.py:1273",
    },
    "enable_prefix_caching": {
        "prime_rl": None,
        "prime_rl_note": "declared but defaults to None, which omits it; vLLM's default applies, i.e. on",
        "prime_rl_source": "packages/prime-rl-configs/src/prime_rl/configs/inference.py:105",
        "floor": False,
        "floor_source": "src/weight_sync_bench/phase2/collective_logits.py:1274",
    },
    "dtype": {
        "prime_rl": "auto",
        "prime_rl_note": "'auto' resolves to bfloat16 for a bf16 checkpoint, so this row is expected to agree in practice",
        "prime_rl_source": "packages/prime-rl-configs/src/prime_rl/configs/inference.py:60",
        "floor": "bfloat16",
        "floor_source": "src/weight_sync_bench/phase2/collective_logits.py:1256",
    },
    "attention_backend": {
        "prime_rl": None,
        "prime_rl_note": "prime-rl sets no vLLM attention backend; its attention-backend selection is trainer-side (ring attention) and does not reach the inference engine",
        "prime_rl_source": "no inference-side setting at the pinned commit",
        "floor": None,
        "floor_note": "also unset; vLLM's default applies on both sides",
        "floor_source": "src/weight_sync_bench/phase2/collective_logits.py:1253-1279",
    },
    "max_num_batched_tokens": {
        "prime_rl": DEFAULT_MAX_NUM_BATCHED_TOKENS,
        "prime_rl_note": (
            "forced down by this probe, not a prime-rl setting: prime-rl leaves "
            "vLLM's default, which is far above the probe's prompt length, so "
            "chunked prefill would be nominally enabled and never actually chunk"
        ),
        "prime_rl_source": "this module (DEFAULT_MAX_NUM_BATCHED_TOKENS)",
        "floor": None,
        "floor_note": "unset; irrelevant with chunked prefill disabled",
        "floor_source": "src/weight_sync_bench/phase2/collective_logits.py:1253-1279",
    },
    "worker_extension_cls": {
        "prime_rl": "prime_rl.inference.vllm.worker.filesystem.FileSystemWeightUpdateWorker",
        "prime_rl_note": (
            "set unconditionally from WORKER_EXTENSION_CLS keyed by "
            "weight_broadcast.type, whose default is 'filesystem'; any "
            "caller-supplied value reaching the namespace is overwritten"
        ),
        "prime_rl_source": "src/prime_rl/inference/vllm/server.py:233 (dict at :59-63)",
        "floor": WORKER_EXTENSION_QUALNAME,
        "floor_source": "src/weight_sync_bench/phase2/collective_logits.py:1278",
    },
}

# Flags that differ and are known to affect this extraction path. Ordered
# worst-first: a bit-identity failure on the GPU box should be attributed
# against this list from the top.
SUSPECT_FLAGS = ("enable_chunked_prefill", "enable_prefix_caching", "enforce_eager")

# Rows that are instrument rather than subject: they differ by construction, so
# listing them as candidate explanations for a deviation would be noise.
NOT_UNDER_TEST = ("worker_extension_cls", "max_num_batched_tokens")

PRIME_RL_WORKER_EXTENSIONS = {
    "nccl": "prime_rl.inference.vllm.worker.nccl.NCCLWeightUpdateWorker",
    "filesystem": "prime_rl.inference.vllm.worker.filesystem.FileSystemWeightUpdateWorker",
    "nixl": "prime_rl.inference.vllm.worker.nixl.NIXLWeightUpdateWorker",
}
DEFAULT_BROADCAST_TYPE = "filesystem"

# vLLM's resolved SchedulerConfig names this field `enable_chunked_prefill`.
# An earlier version of this module read `chunked_prefill_enabled` through a
# `getattr(..., None)` default, and on a real engine that silently produced
# None on BOTH legs: `config_predicts_chunking` came out False, and with it
# `chunking_evidence.confirmed`, while the run was in fact chunking a 32-token
# prompt into two 16-token pieces and the extraction was raising over it. That
# is the same failure class as the `hasattr(torch.cuda, "driver_version")`
# guard backfilled in tolerance/phase2a_bf16_floor_v2.json and the old
# `shape[0] >= expected_min_positions` check in collective_logits: a loose
# guard around instrumentation that degrades to a plausible-looking value
# instead of saying its assumption no longer holds. It is also precisely the
# error this probe is supposed to be immune to -- a probe that silently fails
# to record the condition it is testing.
SCHEDULER_BUDGET_ATTR = "max_num_batched_tokens"
SCHEDULER_CHUNKED_PREFILL_ATTR = "enable_chunked_prefill"


class SchedulerIntrospectionError(RuntimeError):
    """A resolved vLLM config does not carry an attribute this probe reads."""


def read_resolved(config: Any, attr: str) -> Any:
    """Reads one attribute off a resolved vLLM config object, strictly.

    No default. A missing attribute means vLLM renamed or removed it, which
    invalidates what this probe claims to observe; raising says so at the read
    site instead of writing a None that a reader cannot distinguish from a
    genuine "chunking is off".
    """
    if not hasattr(config, attr):
        available = sorted(n for n in dir(config) if not n.startswith("_"))
        raise SchedulerIntrospectionError(
            f"{type(config).__name__} has no attribute {attr!r}; this probe reads "
            f"it to report whether chunking actually occurred. vLLM may have "
            f"renamed it. Available: {available}"
        )
    return getattr(config, attr)


COMPOSED_WORKER_NAME = "PrimeRlLogitsHookWorker"
COMPOSED_WORKER_QUALNAME = f"{__name__}.{COMPOSED_WORKER_NAME}"

_composed_cache: dict[str, type] = {}


def compose_worker_extension(broadcast_type: str = DEFAULT_BROADCAST_TYPE) -> type:
    """Builds the class that carries BOTH prime-rl's weight-update RPCs and the
    phase 2 logits hook.

    Imports prime-rl, so this is GPU-box-only in practice; it is a function
    rather than a module-level class for exactly that reason.
    """
    if broadcast_type not in PRIME_RL_WORKER_EXTENSIONS:
        raise ValueError(
            f"unknown weight-broadcast transport {broadcast_type!r}; "
            f"expected one of {sorted(PRIME_RL_WORKER_EXTENSIONS)}"
        )
    if broadcast_type in _composed_cache:
        return _composed_cache[broadcast_type]

    from vllm.utils.import_utils import resolve_obj_by_qualname

    from ..phase2.collective_logits import LogitsHookWorkerExtension

    base = resolve_obj_by_qualname(PRIME_RL_WORKER_EXTENSIONS[broadcast_type])

    def get_scheduler_config_summary(self) -> dict[str, Any]:
        """The engine flags that decide whether extraction can work, read from
        inside a worker.

        Added here rather than as a third patch to prime-rl: it is a method on
        the composed class, which this project already owns. It exists because
        check 0 has to confirm chunked prefill and prefix caching are actually
        off by reading the RESOLVED config -- a flag that silently failed to
        apply is indistinguishable from one that applied until extraction
        raises.
        """
        from vllm.config import get_current_vllm_config

        config = getattr(self, "vllm_config", None) or get_current_vllm_config()
        scheduler = config.scheduler_config
        cache = getattr(config, "cache_config", None)
        # Key names are this repo's vocabulary and stay stable for `attach.py`;
        # what changed is the vLLM attribute each is read FROM. Strict reads --
        # see SCHEDULER_CHUNKED_PREFILL_ATTR.
        return {
            "max_num_batched_tokens": read_resolved(scheduler, SCHEDULER_BUDGET_ATTR),
            "chunked_prefill_enabled": read_resolved(
                scheduler, SCHEDULER_CHUNKED_PREFILL_ATTR
            ),
            "enable_prefix_caching": getattr(cache, "enable_prefix_caching", None),
        }

    # LogitsHookWorkerExtension first: its methods are the ones being added, and
    # neither class defines a name the other does, so the MRO order is about
    # intent rather than resolution.
    composed = type(
        COMPOSED_WORKER_NAME,
        (LogitsHookWorkerExtension, base),
        {"get_scheduler_config_summary": get_scheduler_config_summary},
    )
    # vLLM resolves the extension by qualname in each worker process, which
    # means `getattr(this_module, COMPOSED_WORKER_NAME)` has to return this
    # class. Bind it so the module-level __getattr__ below finds it.
    composed.__module__ = __name__
    composed.__qualname__ = COMPOSED_WORKER_NAME
    _composed_cache[broadcast_type] = composed
    return composed


def __getattr__(name: str) -> Any:
    """PEP 562 hook so `PrimeRlLogitsHookWorker` resolves by qualname without a
    module-level prime-rl import.

    vLLM's `resolve_obj_by_qualname` does `getattr(module, name)` inside each
    worker process. Building the class here keeps this module importable on a
    CPU-only box with neither prime-rl nor vLLM installed, which is the
    discipline the whole package follows.
    """
    if name == COMPOSED_WORKER_NAME:
        return compose_worker_extension()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def engine_kwargs(
    profile: str,
    model_dir: str,
    tp: int = 1,
    max_num_batched_tokens: int = DEFAULT_MAX_NUM_BATCHED_TOKENS,
) -> dict[str, Any]:
    """The `LLM(...)` kwargs for one flag profile.

    `prime_rl` reproduces the flags prime-rl's `server()` would hand vLLM, with
    the worker extension replaced by the phase 2 hook -- without that
    substitution there is no way to read logits out, and substituting it is
    precisely the change whose numerical effect this probe is measuring.
    `floor` reproduces `collective_logits._run_worker` exactly.
    """
    if profile not in ("prime_rl", "floor"):
        raise ValueError(f"unknown profile {profile!r}; expected 'prime_rl' or 'floor'")
    common = {
        "model": model_dir,
        "tensor_parallel_size": tp,
        "gpu_memory_utilization": 0.85,
        "worker_extension_cls": WORKER_EXTENSION_QUALNAME,
    }
    if profile == "floor":
        return {
            **common,
            "dtype": "bfloat16",
            "enforce_eager": True,
            "enable_chunked_prefill": False,
            "enable_prefix_caching": False,
        }
    # prime-rl leaves the two cache flags unset (vLLM's defaults, both on) and
    # enforce_eager False. They are passed explicitly here rather than omitted
    # so the constructed engine states its own configuration, and so a vLLM
    # default flip between versions shows up as an artifact diff instead of
    # silently changing what "prime-rl's profile" meant.
    return {
        **common,
        "dtype": "bfloat16",  # what prime-rl's "auto" resolves to for this checkpoint
        "enforce_eager": False,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
        # Not a prime-rl setting -- see DEFAULT_MAX_NUM_BATCHED_TOKENS. Without
        # it "chunked prefill enabled" is a flag that never fires at this
        # prompt length, and the probe would pass while testing nothing.
        "max_num_batched_tokens": max_num_batched_tokens,
    }


def scheduler_evidence(llm: Any, prompt_len: int) -> dict[str, Any]:
    """Reads the RESOLVED scheduler config back off a constructed engine.

    The point is to catch a control that did not take. A budget passed to the
    constructor and then clamped, ignored, or overridden by vLLM would leave the
    probe reporting agreement from an unchunked run -- the same class of error
    as a flag that is accepted and then never read.
    """
    scheduler = llm.llm_engine.vllm_config.scheduler_config
    return evidence_from_scheduler(
        read_resolved(scheduler, SCHEDULER_BUDGET_ATTR),
        read_resolved(scheduler, SCHEDULER_CHUNKED_PREFILL_ATTR),
        prompt_len,
    )


def evidence_from_scheduler(
    budget: int | None, chunked_prefill_enabled: Any, prompt_len: int
) -> dict[str, Any]:
    """The chunking prediction, as a pure function of resolved config values.

    Split out from `scheduler_evidence` so a server reached over HTTP and an
    in-process engine share ONE implementation of the check rather than growing
    a second one that can disagree with it.
    """
    import math

    will_chunk = bool(chunked_prefill_enabled) and budget is not None and budget < prompt_len
    return {
        "resolved_max_num_batched_tokens": budget,
        "resolved_chunked_prefill_enabled": chunked_prefill_enabled,
        "prompt_len": prompt_len,
        "budget_below_prompt_len": budget is not None and budget < prompt_len,
        "expected_chunks": math.ceil(prompt_len / budget) if budget else None,
        "config_predicts_chunking": will_chunk,
    }


def _run_worker(
    profile: str,
    model_dir: str,
    tp: int,
    repetitions: int,
    batch: int,
    seq_len: int,
    seed: int,
    out_path: Path,
    max_num_batched_tokens: int = DEFAULT_MAX_NUM_BATCHED_TOKENS,
) -> None:
    """One `LLM` per process, as in phase 2: leftover engine state between two
    constructions in one interpreter is not something this comparison can
    afford to reason about.

    Mirrors `bf16_floor_v2`'s repetition structure so the deviation computed
    from the saved tensors is the same statistic the 2a floor mean is, and the
    two are therefore divisible.

    A chunked prefill breaks the extraction rather than perturbing it:
    `retrieve_and_clear_logits` asserts a single multi-position capture and
    chunking produces one per chunk. That failure is caught and recorded, not
    swallowed -- it is a stronger answer than a number would have been, and it
    is itself runtime proof that chunking occurred.
    """
    import torch
    from vllm import LLM

    from ..phase2.bf16_floor import QWEN3_0_6B, _seeded_token_batches
    from ..phase2.collective_logits import run_one_prompt

    llm = LLM(**engine_kwargs(profile, model_dir, tp, max_num_batched_tokens))
    evidence = scheduler_evidence(llm, seq_len)

    token_batches = _seeded_token_batches(repetitions, batch, seq_len, QWEN3_0_6B.vocab, seed)
    all_reps: list[Any] = []
    extraction_error: str | None = None
    try:
        for rep, tokens in enumerate(token_batches):
            torch.manual_seed(seed + rep)
            all_reps.append(torch.stack([run_one_prompt(llm, row) for row in tokens.tolist()], dim=0))
    except Exception as exc:
        # Caught on the MESSAGE, not the type. `retrieve_and_clear_logits`
        # raises a RuntimeError, but it raises it inside a worker, and vLLM's
        # executor re-raises across that boundary as a bare
        # `Exception("Call to collective_rpc method failed: ...")` with the
        # original text carried in the message and the original type lost. An
        # `except RuntimeError` here therefore never fires on a real engine:
        # it lets the one outcome this probe most needs to record escape as a
        # crash. The marker check is what keeps this narrow -- anything whose
        # text does not carry it is re-raised untouched, so an unrelated
        # failure still fails the run instead of being recorded as a chunking
        # result.
        if MULTI_CAPTURE_MARKER not in str(exc):
            raise
        extraction_error = str(exc)

    torch.save(
        {
            "profile": profile,
            "logits": all_reps,
            "scheduler_evidence": evidence,
            "extraction_error": extraction_error,
            "multi_capture_observed": extraction_error is not None,
        },
        out_path,
    )


def _spawn_worker(
    python: str,
    profile: str,
    model_dir: str,
    tp: int,
    repetitions: int,
    batch: int,
    seq_len: int,
    seed: int,
    out_path: Path,
    max_num_batched_tokens: int = DEFAULT_MAX_NUM_BATCHED_TOKENS,
) -> None:
    subprocess.run(
        [
            python,
            "-m",
            "weight_sync_bench.phase3.engine_probe",
            "--worker",
            "--profile",
            profile,
            "--model-dir",
            str(model_dir),
            "--tp",
            str(tp),
            "--repetitions",
            str(repetitions),
            "--batch",
            str(batch),
            "--seq-len",
            str(seq_len),
            "--seed",
            str(seed),
            "--max-num-batched-tokens",
            str(max_num_batched_tokens),
            "--out",
            str(out_path),
        ],
        check=True,
    )


def differing_flags() -> list[str]:
    """Flag names whose two profiles disagree, suspects first."""
    differing = [
        name
        for name, row in ENGINE_FLAGS.items()
        if row["prime_rl"] != row["floor"] and name not in NOT_UNDER_TEST
    ]
    return sorted(
        differing,
        key=lambda n: (n not in SUSPECT_FLAGS, SUSPECT_FLAGS.index(n) if n in SUSPECT_FLAGS else 0),
    )


# --------------------------------------------------------------------------- #
# The acceptance rule.
# --------------------------------------------------------------------------- #

RULE = (
    "Report the flag deviation as a multiple of the phase 2a floor mean at the "
    "MATCHING configuration. At or below 1x, prime-rl's flags move the logits no "
    "more than the dtype-and-reduction-order floor already does, and phase 3 "
    "keeps the 2a threshold. Above 1x, the flag profile is a source of deviation "
    "the existing threshold was never measured against, so the threshold does "
    "not transfer and the floor is re-measured under prime-rl's flag profile "
    "(SAFETY_FACTOR unchanged) into its own artifact."
)


class FloorLookupError(LookupError):
    """No recorded floor measurement at the requested configuration."""


def floor_mean(
    seq_len: int = DEFAULT_SEQ_LEN,
    batch: int = DEFAULT_BATCH,
    repetitions: int = DEFAULT_REPETITIONS,
    artifact: Path = FLOOR_ARTIFACT,
) -> float:
    """The 2a floor mean at one configuration, read from the committed artifact.

    Read, never inlined: phase 1's rule that a measured number lives in its
    artifact and nowhere else applies here unchanged. The match is exact on all
    three axes rather than nearest-neighbour, because "compare against the floor
    at the matching configuration" is the whole point of the lookup -- silently
    substituting a neighbouring seq_len would answer a different question than
    the one the rule asks.
    """
    data = json.loads(Path(artifact).read_text())
    covered = []
    for result in data["results"]:
        m = result["measurement"]
        covered.append((m["repetitions"], m["batch"], m["seq_len"]))
        if (m["repetitions"], m["batch"], m["seq_len"]) == (repetitions, batch, seq_len):
            return float(result["floor"]["mean_deviation"])
    raise FloorLookupError(
        f"no recorded 2a floor at (repetitions={repetitions}, batch={batch}, "
        f"seq_len={seq_len}); {Path(artifact).name} covers "
        f"(repetitions, batch, seq_len) = {sorted(covered)}"
    )


BRANCH_TRANSFERS = "threshold_transfers"
BRANCH_REMEASURE = "remeasure_required"
BRANCH_NO_READING = "extraction_raised_under_chunking"


def apply_rule(
    deviation_mean: float | None,
    floor_mean_value: float,
    extraction_error: str | None = None,
) -> dict[str, Any]:
    """Apply RULE. Pure function of the measurement, so it is auditable.

    Deliberately in the shape of `tolerance.derive_threshold`: the decision is
    code that runs on the number, not a paragraph a reader applies by hand after
    seeing it.

    Three branches, not two. The rule as stated has two, but it presumes a
    deviation exists to divide by the floor mean, and under prime-rl's profile
    that presumption can fail: once chunked prefill actually chunks,
    `retrieve_and_clear_logits` raises rather than returning a perturbed tensor,
    because it asserts a single multi-position capture and chunking produces one
    per chunk. That is not the `remeasure_required` branch and must not be
    recorded as it -- there is no number, and a reader who saw
    `remeasure_required` would go and re-measure something that cannot be
    measured this way. It gets its own branch, and the raise is kept as the
    runtime evidence that chunking occurred.
    """
    if floor_mean_value <= 0.0:
        raise ValueError(f"floor mean must be positive, got {floor_mean_value!r}")

    if extraction_error is not None:
        return {
            "rule": RULE,
            "deviation_mean": None,
            "floor_mean": floor_mean_value,
            "multiple_of_floor_mean": None,
            "branch": BRANCH_NO_READING,
            "threshold_transfers": False,
            "extraction_error": extraction_error,
            "implication": (
                "No deviation exists, so the rule's two branches do not apply. "
                "The phase 2 extraction path cannot read logits under prime-rl's "
                "flag profile once chunked prefill actually chunks; the profiles "
                "have to be reconciled before any threshold question can be "
                "asked. The raise is itself runtime proof that chunking occurred."
            ),
        }

    if deviation_mean is None:
        raise ValueError("deviation_mean is required unless extraction_error is given")

    multiple = deviation_mean / floor_mean_value
    transfers = deviation_mean <= floor_mean_value
    return {
        "rule": RULE,
        "deviation_mean": deviation_mean,
        "floor_mean": floor_mean_value,
        "multiple_of_floor_mean": multiple,
        "branch": BRANCH_TRANSFERS if transfers else BRANCH_REMEASURE,
        "threshold_transfers": transfers,
        "implication": (
            "phase 3 keeps the phase 2a threshold unchanged"
            if transfers
            else "re-measure the floor under prime-rl's flag profile into "
            "tolerance/phase3_bf16_floor_prime_rl_flags.json and gate on that"
        ),
    }


def probe_environment() -> dict[str, Any]:
    """Phase 2's provenance block, plus what this phase's environment adds.

    `numpy` is recorded here because this environment is built by an install
    that deliberately does not resolve this package's numpy pin (see
    INSTALL_COMMANDS), so which numpy actually landed is a property of the
    measurement rather than an incidental version. Phase 2's `environment()`
    records no numpy field; it is read, never edited, from here.

    Imported inside the function, like every other heavyweight import in this
    package, so the module stays importable where the dependency is absent.
    """
    block = dict(environment())
    try:
        import numpy

        block["numpy"] = numpy.__version__
    except ImportError:
        # Not silently None: an absent numpy is a fact about the environment,
        # and stating it beats a field that reads as "not recorded".
        block["numpy"] = "not installed"
    block["install"] = list(INSTALL_COMMANDS)
    return block


def run_probe(
    seq_len: int = DEFAULT_SEQ_LEN,
    seed: int = DEFAULT_SEED,
    tp: int = 1,
    repetitions: int = DEFAULT_REPETITIONS,
    batch: int = DEFAULT_BATCH,
    max_num_batched_tokens: int = DEFAULT_MAX_NUM_BATCHED_TOKENS,
    python: str = sys.executable,
) -> dict[str, Any]:
    """Extracts logits under both flag profiles and measures the deviation.

    The verdict is `apply_rule`, not bit-identity. Bit-identity would be the
    wrong acceptance across these two profiles: chunked prefill changes how
    prefill attention accumulates across chunk boundaries, and cuda-graph
    capture can change kernel selection and batch padding, so a CORRECT engine
    is under no obligation to return identical bits under differing values of
    either. `bit_identical` is still computed and recorded -- it is free and
    informative -- but it decides nothing.

    What decides is the mean absolute deviation expressed as a multiple of the
    2a floor mean at the same configuration: at or below it, prime-rl's flags
    move the logits no more than the dtype-and-reduction-order floor already
    does and the existing threshold transfers; above it, the threshold was
    never measured against this source of deviation and the floor is
    re-measured under prime-rl's profile.
    """
    import tempfile

    import torch
    from huggingface_hub import snapshot_download

    checkpoint_dir = snapshot_download(MODEL_ID)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        paths = {p: tmp / f"{p}.pt" for p in ("prime_rl", "floor")}
        for profile, out_path in paths.items():
            _spawn_worker(
                python, profile, checkpoint_dir, tp, repetitions, batch,
                seq_len, seed, out_path, max_num_batched_tokens,
            )
        prime_rl = torch.load(paths["prime_rl"])
        floor = torch.load(paths["floor"])

    chunking = {
        "config_predicts_chunking": prime_rl["scheduler_evidence"]["config_predicts_chunking"],
        "multi_capture_observed_at_runtime": prime_rl["multi_capture_observed"],
        "confirmed": bool(
            prime_rl["scheduler_evidence"]["config_predicts_chunking"]
            and prime_rl["multi_capture_observed"]
        ),
        "prime_rl_scheduler": prime_rl["scheduler_evidence"],
        "floor_scheduler": floor["scheduler_evidence"],
        "extraction_error": prime_rl["extraction_error"],
    }

    common = {
        "probe": "engine flag profile differential",
        "question": (
            "Do prime-rl's engine flags change the logits the correctness gate "
            "reads, relative to the profile the bf16 floor was measured under?"
        ),
        "measurement": {
            "repetitions": repetitions,
            "batch": batch,
            "seq_len": seq_len,
            "seed": seed,
            "tensor_parallel_size": tp,
        },
        "chunking_evidence": chunking,
        "engine_flags": ENGINE_FLAGS,
        "differing_flags_worst_first": differing_flags(),
        "attribution_order": SUSPECT_FLAGS,
        "serving_path_limitation": (
            "Both legs are in-process LLM objects. prime-rl's actual sampler is "
            "behind its OpenAI API server and was NOT exercised: at the pinned "
            "commit prime-rl constructs no in-process LLM, and overwrites any "
            "caller-supplied worker_extension_cls at "
            "src/prime_rl/inference/vllm/server.py:233. This probe therefore "
            "isolates the flag difference only. It is not evidence that the "
            "extraction path attaches to a running prime-rl server."
        ),
        "prime_rl_pin": provenance(),
        "environment": probe_environment(),
    }

    # Looked up in both branches: even when no deviation exists, recording what
    # the comparison would have been made against keeps the artifact readable
    # without a second trip to the floor artifact.
    reference_mean = floor_mean(seq_len=seq_len, batch=batch, repetitions=repetitions)
    floor_reference = {
        "artifact": FLOOR_ARTIFACT.name,
        "repetitions": repetitions,
        "batch": batch,
        "seq_len": seq_len,
        "mean_deviation": reference_mean,
    }

    if prime_rl["extraction_error"] is not None:
        # Not a crash to debug and not a number to divide: under prime-rl's
        # flags the extraction path does not produce a reading at all. That is
        # a stronger result than a deviation would have been, and it is the
        # answer the attachment stage has to design against.
        return {
            **common,
            "outcome": BRANCH_NO_READING,
            "bit_identical": None,
            "rule_applied": apply_rule(
                None, reference_mean, extraction_error=prime_rl["extraction_error"]
            ),
            "floor_reference": floor_reference,
        }

    prime_rl_logits = torch.stack(prime_rl["logits"])
    floor_logits = torch.stack(floor["logits"])
    identical = torch.equal(prime_rl_logits, floor_logits)

    # Per-repetition cell means, then the mean of those -- the construction
    # `bf16_floor_v2.measure_differential_floor` uses for `mean_deviation`. A
    # single global mean over the stacked tensor happens to give the same number
    # while every cell is the same size, but the rule divides this by that
    # recorded mean, so the two are built the same way rather than left equal by
    # coincidence.
    cells = []
    for prime_rl_rep, floor_rep in zip(prime_rl["logits"], floor["logits"]):
        cell = (prime_rl_rep.float() - floor_rep.float()).abs()
        cells.append({"max": float(cell.max()), "mean": float(cell.mean())})
    deviation_mean = sum(c["mean"] for c in cells) / len(cells)
    deviation_max = max(c["max"] for c in cells)

    return {
        **common,
        "outcome": "measured",
        # Recorded, deliberately not the verdict -- see run_probe's docstring.
        "bit_identical": identical,
        "max_abs_diff": deviation_max,
        "mean_abs_diff": deviation_mean,
        "cells": cells,
        "shape": list(prime_rl_logits.shape),
        "rule_applied": apply_rule(deviation_mean, reference_mean),
        "floor_reference": floor_reference,
    }


def write(report: dict[str, Any], path: Path = ARTIFACT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--profile", choices=("prime_rl", "floor"), default="floor")
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-num-batched-tokens", type=int, default=DEFAULT_MAX_NUM_BATCHED_TOKENS
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.worker:
        assert args.model_dir and args.out
        _run_worker(
            args.profile, args.model_dir, args.tp, args.repetitions, args.batch,
            args.seq_len, args.seed, args.out, args.max_num_batched_tokens,
        )
        return

    report = run_probe(
        args.seq_len, args.seed, args.tp, args.repetitions, args.batch,
        args.max_num_batched_tokens,
    )
    path = write(report, args.out or ARTIFACT)
    if report["outcome"] == "measured":
        rule = report["rule_applied"]
        print(
            f"{rule['multiple_of_floor_mean']:.3f}x floor mean -> "
            f"{rule['branch']} (bit_identical={report['bit_identical']}) -> {path}"
        )
    else:
        print(f"{report['outcome']} -> {path}")


if __name__ == "__main__":
    main()
