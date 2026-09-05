"""The floor for the served-against-direct comparison, and its break separation.

Runs on a GPU box only; imports anywhere.

WHY THIS FLOOR EXISTS AT ALL
-----------------------------
The attachment probe compared logits extracted through a running server against
logits from a directly constructed engine whose resolved config agrees with the
server's field for field, and expected them to be bit-identical: same flags, so
two paths to one computation rather than two computations. They are not. The
measurement found 2.3e-3 mean and 6.2e-2 worst element, while the same prompt
extracted twice off one server is bit-identical -- a deterministic offset, which
is reduction order across the serving boundary and not divergence.

Bit-identity is therefore withdrawn as the acceptance, and it is NOT replaced by
phase 2a's floor mean. That number is TP1 against TP2 inside one process. This
is a different quantity across a different boundary, and expressing one as a
multiple of the other is exactly the mistake the flag-profile rule was written
to prevent. A deviation is only interpretable against a floor measured for the
same comparison, so this module measures that floor and shows that an injected
layout break still clears it.

THE SHAPE IS PHASE 2a's, WITH THE AXIS CHANGED
-----------------------------------------------
2a measured a floor between two TP degrees and then corrupted the checkpoint to
show the break clears it. Here the axis is the serving boundary instead of the
TP degree, and everything else is deliberately the same:

- `derive_threshold`, `gate_decision`, `SAFETY_FACTOR`, `GATE_MARGIN`,
  `corrupt_checkpoint` and `BREAK_CASES` are IMPORTED from phase 2, never
  reimplemented. A phase 3 copy of the gate rule that drifts from phase 2's is
  worse than no phase 3 gate, because two rules that disagree cannot both be
  the project's rule.
- Break legs diff against the CLEAN direct-path reference, not against a
  corrupted direct engine, mirroring 2a where every break leg diffs against the
  clean TP1 reference. A corrupted-against-corrupted diff would cancel most of
  the corruption and measure almost nothing.

TWO LAYERS, BECAUSE ONE LAYER IS THE KNOWN LIMITATION
-------------------------------------------------------
`tolerance/phase2a_layer_depth_finding.json` records that 2a's gate passes at
layer 0 and fails at layers 7, 13, 20 and 27, as a step rather than a gradient.
A served floor injected only at layer 0 would inherit that limitation silently.
Injecting at layer 0 AND layer 13 makes it visible across the new boundary. A
failing verdict at layer 13 is a RESULT here, not a defect: it reproduces a
known limitation in a new place, which is worth more than a second pass at the
most favorable position. The gate the served path carries forward is layer 0's,
and the artifact says so rather than leaving a reader to assume the better
number is the general one.

WHY THE LEGS ARE SEPARATE INVOCATIONS
--------------------------------------
Each leg runs against an ALREADY-RUNNING server and writes a file; `--assemble`
reads the files. That keeps process management out of this module: a leg is a
measurement against whatever server is up, and which server is up is the
driver's business. It also means a leg that fails costs one leg rather than the
whole sweep, which matters when the sweep is seven server starts on rented
hardware.

The break magnitude contains the serving-boundary offset as well as the
corruption. At layer 0 the corruption is ~1.5 to 3.0 and at other layers ~0.16
to 1.05, against an offset of 2.3e-3, so the offset is a rounding effect on the
break means rather than a confound. This is stated in the artifact rather than
left for a reader to work out.

THE FLOOR IS SCOPED TO ONE GPU, AND THAT IS CHECKED THREE TIMES
----------------------------------------------------------------
`MATCHED_ENGINE_FLAGS` pins every flag this comparison depends on except the
attention backend, which vLLM selects by compute capability. The device name
therefore stands proxy for the kernel and so for the accumulation order this
floor measures, which makes it a scope CONDITION rather than provenance. Every
leg observes it (`running_device_name`), `assemble` refuses to combine legs that
disagree (`_check_devices`), and `load_serving_floor` raises when the running
device is not the recorded one. The last is the one with teeth: without it the
artifact merely states its scope and nothing enforces it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ..phase2.bf16_floor import (
    BREAK_CASES,
    GATE_MARGIN,
    SAFETY_FACTOR,
    derive_threshold,
    gate_decision,
)
from . import attach
from .attachment_probe import (
    MATCHED_ENGINE_FLAGS,
    REPO_ROOT,
)
from .engine_probe import DEFAULT_BROADCAST_TYPE, probe_environment
from .pin import provenance

ARTIFACT = REPO_ROOT / "tolerance" / "phase3_serving_floor.json"

DEFAULT_REPETITIONS = 20
DEFAULT_PROMPTS = 4
DEFAULT_SEQ_LEN = 32
DEFAULT_SEED = 0

# Layer 0 is the position 2a's gate was calibrated at and the only one it passes
# at; layer 13 is one of the four it fails at. Both are run so the artifact
# carries the limitation rather than inheriting it silently.
GATE_LAYER = 0
LAYERS = (0, 13)

CLEAN_LABEL = "clean"

# The floor is valid for ONE GPU architecture, and this is where that is said.
#
# `MATCHED_ENGINE_FLAGS` pins dtype, eager execution, chunked prefill, prefix
# caching, the batched-token budget, logprobs_mode, seed and TP degree. It does
# not pin the attention backend, and prime-rl sets none either: vLLM selects it
# by compute capability. So the kernel this floor is a floor for is chosen by
# the GPU, and the kernel sets the accumulation order -- which is the entire
# quantity being measured. The device name therefore stands proxy for the
# kernel, and a floor read on another architecture is wrong rather than stale.
#
# Capacity is not the reason. Qwen3-0.6B at gpu_memory_utilization 0.85 fits on
# any modern card, so the 80 GB has nothing to do with it.
SCOPE_CONDITION = (
    "This floor is valid only on the recorded device_name. MATCHED_ENGINE_FLAGS "
    "pins every engine flag this comparison depends on EXCEPT the attention "
    "backend, which neither this project nor prime-rl sets and which vLLM "
    "selects by compute capability. The device name therefore stands proxy for "
    "the attention kernel, and the kernel sets the reduction order this floor "
    "measures. Capacity is not involved: the model fits on any modern card."
)

# Why the device name is a scope condition and gpu_name in `environment` is not
# the same statement. Recorded because a reader seeing the same string twice
# would otherwise take the second for a duplicate.
SCOPE_VS_PROVENANCE = (
    "`environment.gpu_name` is provenance: it says where this ran. `scope."
    "device_name` is a condition: it says where the number may be used. They "
    "carry the same string and answer different questions, and only the second "
    "one raises -- see `load_serving_floor`."
)

REGENERATE_COMMAND = (
    "bash scripts/phase3_floor_legs.sh, then .venv-phase3/bin/python -m "
    "weight_sync_bench.phase3.serving_floor --assemble --leg-dir <leg dir>"
)


# Launch arguments naming the weights. These are the ONLY tokens allowed to
# differ between the clean leg and a break leg, because pointing at a corrupted
# copy is what a break leg is.
WEIGHTS_ARGS = frozenset({"--model", "--model-dir", "--inference.model.name"})
# Deliberately NOT in that set: prime-rl's `@ <config.toml>` argument. It names
# the config, not the weights, and normalizing it out would drop the one token
# most in need of matching across legs.


class ServingFloorError(RuntimeError):
    """A leg could not be measured, or the legs do not compose into a floor."""


def leg_name(case: str, layer: int | None) -> str:
    """File stem for one leg. The clean leg has no layer because nothing was
    injected into it; a break leg carries its layer because the same case at a
    different depth is a different measurement, not a repeat of one."""
    return CLEAN_LABEL if case == CLEAN_LABEL else f"{case}@layer{layer}"


def running_device_name() -> str:
    """The name of the CUDA device this process would run on.

    Read per leg rather than once into the artifact's environment block. A
    per-leg invariant asserted from a single global is exactly the move this
    project exists to catch: seven legs are seven measurements, and whether they
    ran on one device is something to observe, not to assume from the fact that
    one driver started them.
    """
    import torch

    if not torch.cuda.is_available():
        raise ServingFloorError(
            "no CUDA device is present, so there is no device name to record "
            "or to check the floor's scope against. This floor is scoped to the "
            "GPU it was measured on; see SCOPE_CONDITION."
        )
    return torch.cuda.get_device_name()


def _token_batches(
    repetitions: int, prompts: int, seq_len: int, seed: int
) -> list[list[list[int]]]:
    """Repetition r's tokens, as plain lists of ids.

    Reuses phase 2's seeding so a repetition index means the same tokens here as
    it does in every 2a artifact. Only the TOKEN seed varies across repetitions.
    The model seed that 2a's `seed_base` also moves is inert in this measurement:
    the weights come from a real checkpoint on disk rather than from an
    initializer, so there is no model-seed state for it to disturb.
    """
    from ..phase2.bf16_floor import QWEN3_0_6B, _seeded_token_batches

    return [
        [row for row in batch.tolist()]
        for batch in _seeded_token_batches(
            repetitions, prompts, seq_len, QWEN3_0_6B.vocab, seed
        )
    ]


# --------------------------------------------------------------------------- #
# Legs.
# --------------------------------------------------------------------------- #


def measure_served_leg(
    case: str,
    layer: int | None,
    base_url: str = attach.DEFAULT_BASE_URL,
    repetitions: int = DEFAULT_REPETITIONS,
    prompts: int = DEFAULT_PROMPTS,
    seq_len: int = DEFAULT_SEQ_LEN,
    seed: int = DEFAULT_SEED,
    launch: dict[str, Any] | None = None,
    source_dir: str | None = None,
) -> dict[str, Any]:
    """Extract every repetition through a RUNNING server.

    `source_dir` is the clean checkpoint a break leg's corruption was written
    from. Given it, the leg verifies its own corruption against the directory
    the server is actually serving, before any deviation is interpreted.

    `launch` is the argv and cwd that started the server answering `base_url`,
    recorded verbatim. "Started the same way" is prose, and prose cannot be
    checked after the fact; argv can. `assemble` compares them.

    Records the device this leg ran on, read here rather than copied from a
    global. `assemble` refuses to combine legs that disagree.

    Records the resolved-config readback and a self-consistency check. Self
    consistency is not decoration: a server that does not reproduce its own bits
    is not measuring the boundary, it is measuring its own nondeterminism, and
    averaging such a leg into the floor would inflate the floor with something
    the floor is not about. `--assemble` refuses those legs.
    """
    import torch

    from ..phase2.collective_logits import run_one_prompt

    batches = _token_batches(repetitions, prompts, seq_len, seed)

    # One adapter for the whole leg. `run_one_prompt_over_http` builds a fresh
    # one per call, which re-reads /v1/models on every prompt; more to the point
    # the served model id is the thing this leg has to RECORD, and an adapter
    # that is discarded cannot report it.
    engine = attach.HttpEngineAdapter(base_url)

    logits: list[Any] = []
    for tokens_per_rep in batches:
        logits.append([run_one_prompt(engine, t) for t in tokens_per_rep])

    repeat = run_one_prompt(engine, batches[0][0])
    self_consistent = bool(torch.equal(logits[0][0], repeat))

    return {
        "leg": leg_name(case, layer),
        "case": case,
        "layer": layer,
        "side": "served",
        "base_url": base_url,
        "repetitions": repetitions,
        "prompts": prompts,
        "seq_len": seq_len,
        "seed": seed,
        "self_consistent": self_consistent,
        "served_model": engine.model,
        "launch": launch,
        # Beside the launch command, and for the same reason: "the same GPU"
        # is prose until a leg records which one it ran on.
        "device_name": running_device_name(),
        "corruption": (
            verify_corruption(source_dir, engine.model, case, layer)
            if case != CLEAN_LABEL and source_dir
            else None
        ),
        "resolved": attach.scheduler_config_over_http(base_url),
        "logits": logits,
    }


def measure_direct_reference(
    model_dir: str,
    repetitions: int = DEFAULT_REPETITIONS,
    prompts: int = DEFAULT_PROMPTS,
    seq_len: int = DEFAULT_SEQ_LEN,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """The clean direct-path reference, in this process.

    Built once and reused by the floor leg and by every break leg, exactly as
    2a's `tp1_reference` is: it already cost GPU minutes once, and recomputing
    it per leg would also let it drift between legs.
    """
    from vllm import LLM

    from ..phase2.collective_logits import run_one_prompt

    batches = _token_batches(repetitions, prompts, seq_len, seed)
    llm = LLM(
        model=model_dir,
        # The bare composed name: this engine is standalone, with no prime-rl
        # weight broadcast in play, which is what the bare name means.
        worker_extension_cls=attach.composed_worker_qualname(DEFAULT_BROADCAST_TYPE),
        **MATCHED_ENGINE_FLAGS,
    )
    resolved = llm.collective_rpc("get_scheduler_config_summary")[0]
    logits = [[run_one_prompt(llm, t) for t in tokens_per_rep] for tokens_per_rep in batches]

    return {
        "leg": "direct_reference",
        "case": CLEAN_LABEL,
        "layer": None,
        "side": "direct",
        "model_dir": model_dir,
        "repetitions": repetitions,
        "prompts": prompts,
        "seq_len": seq_len,
        "seed": seed,
        "device_name": running_device_name(),
        "resolved": resolved,
        "logits": logits,
    }


# --------------------------------------------------------------------------- #
# Assembly: pure once the legs exist.
# --------------------------------------------------------------------------- #


def _cells(reference_logits, leg_logits) -> list[dict[str, float]]:
    """One cell per repetition, aggregated over that repetition's prompts.

    `mean` is the mean over every element of every prompt, which is what
    `measure_differential_floor` takes the mean of; taking a per-prompt mean and
    then averaging those would weight a short prompt like a long one.
    """
    import torch

    if len(reference_logits) != len(leg_logits):
        raise ServingFloorError(
            f"repetition count differs: reference has {len(reference_logits)}, "
            f"leg has {len(leg_logits)}. Legs measured at different repetition "
            "counts cannot be averaged into one floor."
        )

    cells = []
    for ref_rep, leg_rep in zip(reference_logits, leg_logits):
        diffs = [
            (a.float() - b.float()).abs().reshape(-1) for a, b in zip(leg_rep, ref_rep)
        ]
        diff = torch.cat(diffs)
        cells.append(
            {
                "max": diff.max().item(),
                "median": diff.median().item(),
                "mean": diff.mean().item(),
            }
        )
    return cells


def _leg_stats(reference: dict[str, Any], leg: dict[str, Any]) -> dict[str, Any]:
    cells = _cells(reference["logits"], leg["logits"])
    return {
        "leg": leg["leg"],
        "case": leg["case"],
        "layer": leg["layer"],
        "self_consistent": leg.get("self_consistent"),
        "cells": cells,
        "mean_deviation": sum(c["mean"] for c in cells) / len(cells),
        "max_deviation": max(c["max"] for c in cells),
    }


def _configs_agree(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(set(a) | set(b))
    differing = {k: [a.get(k), b.get(k)] for k in keys if a.get(k) != b.get(k)}
    return {"agree": not differing, "differing": differing}


def _normalized_argv(launch: dict[str, Any]) -> list[str]:
    """The launch argv with the weights path removed.

    A break leg is SUPPOSED to differ in its weights path -- that is the
    injection. Everything else differing means the two servers were not brought
    up the same way, and a difference between legs would then be a difference in
    how the server started rather than in what was injected.

    The weights path is not simply ignored: `_check_launches` asserts each leg
    points at the directory that leg's corruption was written to, so removing it
    here narrows the comparison rather than dropping a constraint.
    """
    argv = list(launch.get("command", []))
    drop_next = False
    out = []
    for token in argv:
        if drop_next:
            drop_next = False
            continue
        if token in WEIGHTS_ARGS:
            drop_next = True
            continue
        if any(token.startswith(f"{name}=") for name in WEIGHTS_ARGS):
            continue
        out.append(token)
    return out


# --------------------------------------------------------------------------- #
# The corruption, verified where it was written.
# --------------------------------------------------------------------------- #

# What each break case is DEFINED to touch, at a given layer. Derived from
# `corrupt_checkpoint`'s own branches and asserted against them by a test, so
# this cannot drift into a second, disagreeing statement of the same thing.
def expected_corrupted_tensors(case: str, layer: int) -> tuple[str, ...]:
    attn = f"model.layers.{layer}.self_attn."
    if case == "case1_qkv_head_permute":
        return (attn + "q_proj.weight", attn + "k_proj.weight")
    if case == "case2_oproj_col_permute":
        return (attn + "o_proj.weight",)
    if case == "case3_norm_permute":
        return (f"model.layers.{layer}.input_layernorm.weight",)
    raise ValueError(f"unknown break case {case!r}, expected one of {BREAK_CASES}")


def compare_tensor_sets(
    source: dict, corrupted: dict, case: str, layer: int
) -> dict[str, Any]:
    """The corrupted tensors are exactly the intended ones, changed by more than
    nothing.

    A HASH WOULD NOT ANSWER THIS. It proves the directory is what corruption
    wrote; comparing a clean hash against a corrupted one proves only that the
    two differ somewhere. What is worth knowing is narrower and stronger: the
    intended tensors changed, and no others did.

    VERIFIED AGAINST THE WEIGHTS THAT PRODUCED THE LOGITS, not against the
    arguments that were supposed to produce them: the caller passes the model
    the server reports serving, not the path its launch named. This is the same
    rule as reading the resolved scheduler config off the workers rather than
    off the request -- a directory that silently failed to load looks exactly
    like one that loaded, right up until the number is interpreted.

    The failure this closes only bites when the deviation comes out SMALL, which
    is exactly when the conclusion drawn is "this break case does not separate".
    Where the deviation is large, its being large already proves the corruption
    landed. So this earns its place in the one branch where nothing else is
    evidence.

    The reachable way to write a corruption that changes nothing is a
    permutation that degenerates to the identity -- `_swap_adjacent_blocks` with
    `n_blocks == 1`, whose loop range is empty. Not reachable at this model's
    head counts, but that is the shape of the hole. A layer index that does not
    exist is NOT that hole: `corrupt_checkpoint` subscripts the tensor names
    directly and raises KeyError.
    """
    import torch

    expected = expected_corrupted_tensors(case, layer)

    missing = [name for name in expected if name not in source or name not in corrupted]
    if missing:
        raise ServingFloorError(
            f"{case} at layer {layer}: {missing} absent from the checkpoint, so "
            "the corruption cannot be verified where it was written."
        )

    changed, deltas = [], {}
    for name, tensor in corrupted.items():
        other = source.get(name)
        if other is None or other.shape != tensor.shape:
            changed.append(name)
            deltas[name] = None
            continue
        delta = (tensor.float() - other.float()).abs().max().item()
        if delta > 0.0:
            changed.append(name)
            deltas[name] = delta

    unchanged_but_intended = [n for n in expected if n not in changed]
    if unchanged_but_intended:
        raise ServingFloorError(
            f"{case} at layer {layer} changed nothing in {unchanged_but_intended}. "
            "The corruption did not land, so this leg would deviate by about the "
            "floor and read as this break case failing to separate."
        )

    unintended = sorted(set(changed) - set(expected))
    if unintended:
        raise ServingFloorError(
            f"{case} at layer {layer} also changed {unintended}, which it is not "
            "defined to touch. A break leg has to isolate one layout error."
        )

    return {
        "case": case,
        "layer": layer,
        "verified": True,
        "expected_tensors": list(expected),
        "changed_tensors": sorted(changed),
        "max_abs_delta": {name: deltas[name] for name in sorted(changed)},
        "checks": (
            "exactly the tensors this case is defined to touch differ from the "
            "source, each by a non-zero amount, and no others differ"
        ),
    }


def verify_corruption(
    source_dir: str, corrupted_dir: str, case: str, layer: int
) -> dict[str, Any]:
    """`compare_tensor_sets` against the two checkpoints on disk.

    Two safetensors reads, no server, no GPU. `corrupt_checkpoint` is not
    touched and its phase 2 callers are unaffected.
    """
    from safetensors.torch import load_file

    from ..phase2.bf16_floor import _find_safetensors_file

    source = load_file(_find_safetensors_file(Path(source_dir)))
    corrupted = load_file(_find_safetensors_file(Path(corrupted_dir)))
    return compare_tensor_sets(source, corrupted, case, layer)


def _weights_path(launch: dict[str, Any]) -> str | None:
    """The weights argument out of a launch argv, or None if it names none."""
    argv = list(launch.get("command", []))
    for index, token in enumerate(argv):
        if token in WEIGHTS_ARGS and index + 1 < len(argv):
            return argv[index + 1]
        for name in WEIGHTS_ARGS:
            if token.startswith(f"{name}="):
                return token.split("=", 1)[1]
    return None


def _check_served_weights(
    reference: dict[str, Any], floor_leg: dict[str, Any], break_legs: list[dict[str, Any]]
) -> dict[str, Any]:
    """Every leg served the weights its own launch named, and each break leg
    served something other than the clean checkpoint.

    THE SECOND HALF IS THE ONE WITH TEETH, and the failure it catches is
    asymmetric. A clean leg on the wrong weights produces a large deviation and
    announces itself. A BREAK leg that silently comes up on the clean checkpoint
    produces a floor-sized deviation and reads as "this break case does not
    separate" -- a false statement about the harness's sensitivity, dressed as a
    legitimate result about the case. It is the one wrong answer this
    measurement can produce that looks right, so it is checked rather than
    assumed.

    WHAT THIS IDENTITY DOES AND DOES NOT COVER: /v1/models returns a model id,
    which for a locally loaded checkpoint is its path. So this catches a server
    pointed at the wrong DIRECTORY. It does not catch two directories with the
    same path or name and different contents, and it is not a hash of the
    weights. Stated in the artifact because "served_model matches" reads as a
    content check to anyone who did not write it.
    """
    clean_path = _weights_path(floor_leg["launch"])
    per_leg = {}

    def declared(leg):
        return _weights_path(leg["launch"])

    for leg in [floor_leg, *break_legs]:
        served, path = leg.get("served_model"), declared(leg)
        loaded_what_it_was_told = bool(served and path and Path(served).name == Path(path).name)
        entry = {
            "served_model": served,
            "launch_weights_path": path,
            "loaded_what_its_launch_named": loaded_what_it_was_told,
        }
        if leg is not floor_leg:
            entry["distinct_from_clean_checkpoint"] = bool(
                path and clean_path and Path(path).name != Path(clean_path).name
            )
            if not entry["distinct_from_clean_checkpoint"]:
                raise ServingFloorError(
                    f"break leg {leg['leg']!r} was served the clean checkpoint "
                    f"({path!r}). Its corruption was never loaded, so its "
                    "deviation would be floor-sized and would read as this "
                    "break case failing to separate."
                )
        if not loaded_what_it_was_told:
            raise ServingFloorError(
                f"leg {leg['leg']!r} served {served!r} but its launch named "
                f"{path!r}; the server is not running the weights it was told to."
            )
        per_leg[leg["leg"]] = entry

    reference_model = reference.get("model_dir")
    return {
        "per_leg": per_leg,
        "direct_reference_model_dir": reference_model,
        "clean_leg_matches_direct_reference": bool(
            clean_path and reference_model
            and Path(clean_path).name == Path(reference_model).name
        ),
        "identity_is": (
            "the model id /v1/models reports, which for a local checkpoint is "
            "its path. This catches a server pointed at the wrong DIRECTORY. It "
            "is not a hash: two directories with the same name and different "
            "contents are indistinguishable to it."
        ),
    }


def _check_launches(floor_leg: dict[str, Any], break_legs: list[dict[str, Any]]) -> dict[str, Any]:
    """Every served leg has to have been started by the same command."""
    reference = floor_leg.get("launch")
    if not reference:
        raise ServingFloorError(
            "the clean served leg records no launch command. Legs that cannot "
            "say how their server was started cannot be shown to have been "
            "started the same way, which is the only thing keeping a difference "
            "between legs from being a difference in the launch."
        )
    expected = _normalized_argv(reference)
    for leg in break_legs:
        if not leg.get("launch"):
            raise ServingFloorError(f"leg {leg['leg']!r} records no launch command")
        actual = _normalized_argv(leg["launch"])
        if actual != expected:
            first = next(
                (
                    f"position {i}: {expected[i] if i < len(expected) else '<missing>'!r} "
                    f"vs {actual[i] if i < len(actual) else '<missing>'!r}"
                    for i in range(max(len(expected), len(actual)))
                    if expected[i : i + 1] != actual[i : i + 1]
                ),
                "lengths differ",
            )
            raise ServingFloorError(
                f"leg {leg['leg']!r} was started by a different command than the "
                f"clean leg, first difference at {first}. Only the weights path "
                "may differ between legs."
            )
        if leg["launch"].get("cwd") != reference.get("cwd"):
            raise ServingFloorError(
                f"leg {leg['leg']!r} was started from a different working "
                f"directory than the clean leg"
            )
    return {
        "command": reference["command"],
        "cwd": reference.get("cwd"),
        "normalized_argv_matches_across_legs": True,
        "weights_args_normalized_out": sorted(WEIGHTS_ARGS),
    }


DEPTH_FINDING = REPO_ROOT / "tolerance" / "phase2a_layer_depth_finding.json"

# The clean leg misread during this sweep's first attempt, before the launcher
# derived every flag from MATCHED_ENGINE_FLAGS. Kept as a literal because it is
# a historical observation that no longer reproduces: the launcher it came from
# no longer exists. See scripts/phase3_serve.py's `build_args`.
MISREAD_CLEAN_LEG_MEAN = 3.71e-2


def _flag_near_miss() -> dict[str, Any]:
    """The near-miss worth recording: a broken measurement's wrong answer agreed
    with a number already in the repo.

    A launcher that passed only some of `MATCHED_ENGINE_FLAGS` left the server
    compiling and capturing CUDA graphs while the direct engine ran eager. The
    clean leg then read 3.71e-2 instead of 2.3e-3 -- and 3.71e-2 lands within
    half a percent of phase 2a's TP1-vs-TP2 floor mean at the same repetitions,
    batch and sequence length. A reader who had that number in mind would have
    found the wrong one plausible, which is exactly why the coincidence is
    written down rather than left as a fixed bug.

    2a's mean is READ from its artifact at the matching configuration, never
    retyped, so the coincidence cannot become a stale claim about a number that
    has since moved.
    """
    from .engine_probe import floor_mean

    reference = floor_mean(
        seq_len=DEFAULT_SEQ_LEN, batch=DEFAULT_PROMPTS, repetitions=DEFAULT_REPETITIONS
    )
    return {
        "what": (
            "A launcher that passed only part of MATCHED_ENGINE_FLAGS left the "
            "served side compiling and capturing CUDA graphs while the direct "
            "side ran under enforce_eager, so the clean leg measured a launcher "
            "difference rather than the serving boundary."
        ),
        "misread_clean_leg_mean": MISREAD_CLEAN_LEG_MEAN,
        "phase2a_floor_mean": reference,
        "phase2a_floor_artifact": "phase2a_bf16_floor_v2.json",
        "phase2a_configuration": {
            "repetitions": DEFAULT_REPETITIONS,
            "batch": DEFAULT_PROMPTS,
            "seq_len": DEFAULT_SEQ_LEN,
        },
        "relative_difference": abs(MISREAD_CLEAN_LEG_MEAN - reference) / reference,
        "why_it_was_worth_catching": (
            "The wrong answer agreed with a number already in the repo: the "
            "misread clean leg landed within half a percent of phase 2a's floor "
            "mean, a quantity measured across a DIFFERENT boundary. A broken "
            "measurement that lands on a familiar value is the one a reader "
            "checks least."
        ),
        "fix": (
            "scripts/phase3_serve.py derives every argument from "
            "MATCHED_ENGINE_FLAGS rather than listing a subset, so the two "
            "sides read one definition of the matched set."
        ),
    }


def _depth_reference(deep_layer: int) -> dict[str, Any] | None:
    """2a's numbers at the same layer pair, read from its committed artifact.

    Read rather than quoted: the note below compares this run against 2a, and a
    comparison built from retyped numbers goes stale the moment 2a is
    re-measured, silently and in the one field a reader consults for the summary.
    """
    if not DEPTH_FINDING.is_file():
        return None
    data = json.loads(DEPTH_FINDING.read_text())
    rows = {row["layer"]: row for row in data["sweep"]["results"]}
    if deep_layer not in rows or GATE_LAYER not in rows:
        return None
    floor_mean_2a = data["sweep"]["floor_mean_deviation"]
    deep = rows[deep_layer]
    weakest = min(
        value for case, value in deep.items() if case in set(BREAK_CASES)
    )
    return {
        "artifact": DEPTH_FINDING.name,
        "layer0_over_deep_ratios": {
            case: ratios.get(str(deep_layer))
            for case, ratios in data["layer0_over_layerN_ratios"].items()
        },
        "verdicts": {
            f"layer_{GATE_LAYER}": rows[GATE_LAYER]["verdict"].lower(),
            f"layer_{deep_layer}": deep["verdict"].lower(),
        },
        "floor_mean_deviation": floor_mean_2a,
        "weakest_over_floor_at_deep_layer": weakest / floor_mean_2a,
    }


def _gate_carried_forward(
    gates: dict[str, Any], break_stats: list[dict[str, Any]], floor: dict[str, Any]
) -> dict[str, Any]:
    """The summary field, written FROM the measurement.

    Its previous version was written from an expectation -- that layer 13 would
    fail -- and then contradicted `gate.layer_13.verdict` in the same artifact.
    Generating it closes that by construction: every clause below is computed
    from the legs this run produced, so the note cannot disagree with the
    numbers beside it.
    """
    gate_key = f"layer_{GATE_LAYER}"
    verdicts = {name: block["verdict"] for name, block in sorted(gates.items())}
    deep_layers = sorted(
        block["layer"] for block in gates.values() if block["layer"] != GATE_LAYER
    )
    required = SAFETY_FACTOR * GATE_MARGIN
    floor_mean = floor["mean_deviation"]

    means = {}
    for stat in break_stats:
        means.setdefault(stat["layer"], {})[stat["case"]] = stat["mean_deviation"]

    sentences = []
    if verdicts and len(set(verdicts.values())) == 1:
        only = next(iter(verdicts.values()))
        sentences.append(
            f"All measured layers ({', '.join(sorted(gates))}) {only}: "
            + ", ".join(f"{name} {verdict}" for name, verdict in verdicts.items())
            + "."
        )
    else:
        sentences.append(
            "Verdicts differ by layer: "
            + ", ".join(f"{name} {verdict}" for name, verdict in verdicts.items())
            + "."
        )
    sentences.append(
        f"The gate the served path carries forward is layer {GATE_LAYER}'s, "
        f"which is the position phase 2a calibrated at; it is not a general gate."
    )

    depth_ratios: dict[str, dict[str, float]] = {}
    reference = None
    for deep in deep_layers:
        ratios = {
            case: means[GATE_LAYER][case] / means[deep][case]
            for case in sorted(means.get(GATE_LAYER, {}))
            if case in means.get(deep, {}) and means[deep][case]
        }
        depth_ratios[f"layer_{GATE_LAYER}_over_layer_{deep}"] = ratios
        reference = _depth_reference(deep)
        here = " / ".join(f"{value:.1f}x" for _, value in sorted(ratios.items()))
        # Whether the step is reproduced is read off the ratios, not asserted:
        # every case shallower-than-deeper is the step 2a recorded, and a run
        # where it did not hold has to be able to say so in this same field.
        step = "is reproduced" if ratios and all(v > 1 for v in ratios.values()) else (
            "is NOT reproduced"
        )
        if reference and all(reference["layer0_over_deep_ratios"].values()):
            there = " / ".join(
                f"{value:.1f}x"
                for _, value in sorted(reference["layer0_over_deep_ratios"].items())
            )
            sentences.append(
                f"The depth step in break MAGNITUDE {step}: layer "
                f"{GATE_LAYER} over layer {deep} is {here} by case here, against "
                f"phase 2a's {there} at the same pair."
            )
        else:
            sentences.append(
                f"The depth step in break MAGNITUDE {step}: layer "
                f"{GATE_LAYER} over layer {deep} is {here} by case here."
            )

        weakest_here = min(means[deep].values()) / floor_mean
        if reference:
            there_verdict = reference["verdicts"][f"layer_{deep}"]
            carried = (
                "did NOT carry over from"
                if verdicts.get(f"layer_{deep}") != there_verdict
                else "agrees with"
            )
            sentences.append(
                f"The VERDICT is floor-relative and {carried} "
                f"phase 2a, where layer {deep} was "
                f"{there_verdict}. Against its own "
                f"floor the weakest case at layer {deep} separates by "
                f"{weakest_here:.0f}x here and by "
                f"{reference['weakest_over_floor_at_deep_layer']:.1f}x in 2a, "
                f"against the same required {required}x. The break magnitudes "
                "are the smaller part of that difference; the served floor "
                "being tighter is the larger."
            )
        else:
            sentences.append(
                f"The VERDICT is floor-relative: at layer {deep} the weakest "
                f"case separates from this floor by {weakest_here:.0f}x against "
                f"a required {required}x."
            )

    ratios_to_gate = {
        name: block["weakest_case_ratio_to_gate"]
        for name, block in sorted(gates.items())
        if block["weakest_case_ratio_to_gate"] is not None
    }
    if len(ratios_to_gate) > 1:
        ordered = sorted(ratios_to_gate.items(), key=lambda item: item[1])
        tightest, widest = ordered[0], ordered[-1]
        sentences.append(
            "The layers are not interchangeable: "
            + ", ".join(f"{name} clears the gate by {value:.2f}x"
                        for name, value in ordered)
            + f". A floor regression flips {tightest[0]} first and leaves "
            f"{widest[0]} untouched, so the better number is not the general one."
        )

    return {
        "layer": GATE_LAYER,
        "verdict": gates.get(gate_key, {}).get("verdict"),
        "verdicts": verdicts,
        "weakest_case_ratio_to_gate": ratios_to_gate,
        "break_magnitude_depth_ratios": depth_ratios,
        "phase2a_reference": reference,
        "required_separation_over_floor": required,
        "note": " ".join(sentences),
        "note_is": (
            "generated from this run's legs, not written from an expectation. "
            "An earlier hand-written version of this field stated a failing "
            "verdict at the deeper layer while the measurement beside it said "
            "pass."
        ),
    }


def _check_devices(legs: list[dict[str, Any]]) -> dict[str, Any]:
    """Every leg was measured on the same GPU, observed per leg.

    On the same footing as `_check_launches`, and for the same reason. The
    attention backend is the one thing `MATCHED_ENGINE_FLAGS` does not pin --
    vLLM picks it by compute capability -- so two legs on two architectures ran
    two different kernels, and their difference would be a kernel difference
    wearing the serving boundary's name.

    A leg with no device name is refused rather than skipped. Skipping it would
    let a leg measured somewhere else average into a floor that then states one
    device in its scope block, which is the failure this check exists for.
    """
    missing = [leg["leg"] for leg in legs if not leg.get("device_name")]
    if missing:
        raise ServingFloorError(
            f"legs {missing} record no device name, so they cannot be shown to "
            "have run on the device this floor is scoped to. Re-measure them, or "
            f"they are not legs of this floor. {REGENERATE_COMMAND}"
        )

    first = legs[0]
    for leg in legs[1:]:
        if leg["device_name"] != first["device_name"]:
            raise ServingFloorError(
                f"leg {leg['leg']!r} was measured on {leg['device_name']!r} but "
                f"leg {first['leg']!r} was measured on {first['device_name']!r}. "
                "These legs do not compose into one floor: vLLM selects the "
                "attention backend by compute capability, so a difference "
                "between them is a kernel difference rather than the serving "
                "boundary this floor measures."
            )
    return {
        "device_name": first["device_name"],
        "per_leg": {leg["leg"]: leg["device_name"] for leg in legs},
        "matches_across_legs": True,
    }


def assemble(
    leg_dir: Path,
    artifact_path: Path = ARTIFACT,
    environment_note: str | None = None,
    snapshot: dict[str, Any] | None = None,
    attention_backend: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read the leg files, build the floor, apply phase 2's gate at each layer."""
    import torch

    legs = {}
    for path in sorted(leg_dir.glob("*.pt")):
        loaded = torch.load(path, weights_only=False)
        legs[loaded["leg"]] = loaded

    reference = legs.pop("direct_reference", None)
    if reference is None:
        raise ServingFloorError(
            f"no direct_reference leg in {leg_dir}. Every deviation here is "
            "measured against the clean direct path, so there is nothing to "
            "measure without it."
        )
    if CLEAN_LABEL not in legs:
        raise ServingFloorError(
            f"no {CLEAN_LABEL!r} served leg in {leg_dir}. That leg IS the floor; "
            "break legs without it have no scale to be read against."
        )

    floor_leg = legs.pop(CLEAN_LABEL)
    if not floor_leg["self_consistent"]:
        raise ServingFloorError(
            "the clean server was not self-consistent: one prompt extracted "
            "twice off it did not reproduce its own bits. That measures the "
            "server's nondeterminism rather than the serving boundary, and a "
            "floor built on it would be neither."
        )

    launches = _check_launches(floor_leg, [legs[n] for n in sorted(legs)])

    # The reference is included: it is the side every deviation is measured
    # against, so a reference built on another device would put the kernel
    # difference into every number here rather than into one leg.
    devices = _check_devices([reference, floor_leg, *(legs[n] for n in sorted(legs))])

    # The served side has to be serving the weights the direct reference was
    # built from, or the comparison is about checkpoints rather than about the
    # boundary. The attachment probe did not record the served model id at all,
    # so this was inferable only from the deviation being small; recording it
    # makes it checkable.
    weights = _check_served_weights(
        reference, floor_leg, [legs[n] for n in sorted(legs)]
    )

    unverified = [
        name
        for name, leg in sorted(legs.items())
        if not (leg.get("corruption") or {}).get("verified")
    ]
    if unverified:
        raise ServingFloorError(
            f"break legs {unverified} did not verify their corruption against "
            "the checkpoint the server was serving. An unverified corruption "
            "only bites when the deviation comes out small, which is exactly "
            "when the conclusion drawn would be that the case does not separate."
        )

    floor_stats = _leg_stats(reference, floor_leg)
    threshold = derive_threshold(floor_stats["mean_deviation"])
    floor = {
        **floor_stats,
        "threshold": threshold,
        "safety_factor": SAFETY_FACTOR,
        "rule": (
            "threshold = safety_factor * mean deviation of the served-against-"
            "direct comparison on the clean checkpoint. Imported from phase 2 "
            "rather than restated, so the two phases cannot drift apart."
        ),
    }

    inconsistent = [
        name for name, leg in legs.items() if leg.get("self_consistent") is False
    ]
    break_stats = [
        _leg_stats(reference, leg)
        for name, leg in sorted(legs.items())
        if name not in inconsistent
    ]

    gates = {}
    for layer in sorted({s["layer"] for s in break_stats if s["layer"] is not None}):
        at_layer = [s for s in break_stats if s["layer"] == layer]
        gates[f"layer_{layer}"] = {
            **gate_decision(floor, at_layer),
            "layer": layer,
            "cases": {s["case"]: s["mean_deviation"] for s in at_layer},
            "weakest_case_ratio_to_gate": min(
                (s["mean_deviation"] / (GATE_MARGIN * threshold) for s in at_layer),
                default=None,
            ),
        }

    served_resolved = floor_leg["resolved"]
    report = {
        "probe": "serving-boundary floor and break separation",
        "question": (
            "How far apart are logits extracted through a running server and "
            "logits from a directly constructed engine on the same checkpoint, "
            "and does an injected layout break still clear that distance?"
        ),
        "broadcast_type": floor_leg.get("broadcast_type", DEFAULT_BROADCAST_TYPE),
        "repetitions": floor_leg["repetitions"],
        "prompts": floor_leg["prompts"],
        "seq_len": floor_leg["seq_len"],
        "seed": floor_leg["seed"],
        "seed_note": (
            "Only the token seed varies across repetitions. The model seed that "
            "phase 2a's seed_base also moves is inert here: the weights come "
            "from a real checkpoint rather than from an initializer."
        ),
        "matched_flags": dict(MATCHED_ENGINE_FLAGS),
        "launch": launches,
        "served_weights": {
            **weights,
            "note": (
                "The clean served leg and the direct reference have to be the "
                "same weights, or the floor is a checkpoint difference wearing "
                "a boundary's name. Each break leg has to be on its own "
                "corrupted copy, or it reads as a case that does not separate."
            ),
        },
        "resolved_configs": {
            "served": served_resolved,
            "direct": reference["resolved"],
            **_configs_agree(served_resolved, reference["resolved"]),
        },
        "floor": floor,
        "break_cases": list(BREAK_CASES),
        "layers": list(LAYERS),
        "breaks": break_stats,
        "corruption_verification": {
            name: legs[name]["corruption"] for name in sorted(legs)
        },
        "excluded_legs": inconsistent,
        "gate": gates,
        "gate_carried_forward": _gate_carried_forward(gates, break_stats, floor),
        "scope": {
            "device_name": devices["device_name"],
            "condition": SCOPE_CONDITION,
            "why_not_provenance": SCOPE_VS_PROVENANCE,
            "checked_by": (
                "`load_serving_floor` raises when the recorded device name "
                "differs from the running one, and `assemble` refuses to "
                "combine legs that disagree. Fail, not warn: this follows the "
                "precedent phase 1 sets for a value under the repo's own "
                "control, where a mismatch makes the number wrong rather than "
                "merely stale, not the environment block's precedent of warning "
                "on a version bump."
            ),
            "per_leg": devices["per_leg"],
            "matches_across_legs": devices["matches_across_legs"],
            "attention_backend": attention_backend,
            "regenerate": REGENERATE_COMMAND,
        },
        "flag_near_miss": _flag_near_miss(),
        "offset_vs_corruption": (
            "Break means contain the serving-boundary offset as well as the "
            "corruption. The offset is the floor mean above; the corruption is "
            "orders larger at both layers, so the offset rounds the break means "
            "rather than confounding them."
        ),
        "patches": {
            "names": list(attach.PATCHES),
            "sha256": attach.patch_digests(),
            "count": len(attach.PATCHES),
        },
        "prime_rl_pin": provenance(),
        "environment": probe_environment(),
        # Whether this run was made on the same environment as an earlier one is
        # not derivable from the version block: the same versions reinstalled
        # are a different build. The runner states it, and a later run can then
        # say whether it is on this environment or another re-creation.
        "environment_provenance": environment_note,
        "environment_snapshot": snapshot,
    }

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


# --------------------------------------------------------------------------- #
# The point of use.
# --------------------------------------------------------------------------- #


def load_serving_floor(
    path: Path = ARTIFACT, device_name: str | None = None
) -> dict[str, Any]:
    """The committed floor, refused if the running GPU is not the one it is for.

    Shaped after phase 1's `load_threshold`: the measured number lives in its
    artifact and is read from it, never inlined by a caller. The whole report is
    returned rather than the bare threshold because this floor is a threshold, a
    gate at two layers and a scope condition, and a caller handed only the number
    would have to reopen the file to learn which layer it holds at.

    RAISES ON A MISMATCH, and does not warn. The environment block's precedent
    is the other way -- a torch or numpy bump warns, because it weakens
    provenance without making the number wrong -- but the device belongs with
    the values phase 1 hard-fails on instead: it is chosen deliberately at rental
    time, it is under this repo's control, and a floor read on another
    architecture is wrong rather than stale. vLLM picks the attention backend by
    compute capability, so another device is another kernel and another reduction
    order, which is the whole quantity this floor measures.

    `device_name` is an argument so the raise path is testable without a GPU.
    Left unset it probes the running device, and having no CUDA device at all is
    itself a refusal rather than a pass: nothing can be shown to be in scope.
    """
    report = json.loads(Path(path).read_text())
    scope = report.get("scope")
    if not scope or not scope.get("device_name"):
        raise ServingFloorError(
            f"{Path(path).name} carries no scope block, so the device it was "
            "measured on is not recorded and nothing can be checked against it. "
            f"Regenerate: {REGENERATE_COMMAND}"
        )

    recorded = scope["device_name"]
    running = device_name if device_name is not None else running_device_name()
    if running != recorded:
        raise ServingFloorError(
            f"{Path(path).name} was measured on {recorded!r} and this process "
            f"is running on {running!r}. The floor does not transfer: vLLM "
            "selects the attention backend by compute capability, so the "
            "reduction order this floor measures is not the one running here. "
            f"Measure a floor on this device instead: {REGENERATE_COMMAND}"
        )
    return report


# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--leg-dir", type=Path, required=True)
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--served-leg", action="store_true")
    parser.add_argument("--direct-reference", action="store_true")
    parser.add_argument("--case", type=str, default=CLEAN_LABEL)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--base-url", type=str, default=attach.DEFAULT_BASE_URL)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="clean checkpoint a break leg's corruption was written from",
    )
    parser.add_argument("--broadcast-type", type=str, default=DEFAULT_BROADCAST_TYPE)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--prompts", type=int, default=DEFAULT_PROMPTS)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--launch-json",
        type=str,
        default=None,
        help="JSON {command: [...], cwd: ...} that started the server for this leg",
    )
    parser.add_argument("--environment-note", type=str, default=None,
                        help="how the environment this ran on came to exist")
    parser.add_argument("--snapshot-json", type=str, default=None,
                        help="JSON record of the environment snapshot attempt")
    parser.add_argument("--attention-backend-json", type=str, default=None,
                        help="JSON evidence of the attention backend a server "
                             "log named, recorded in `scope`")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.assemble:
        report = assemble(
            args.leg_dir,
            args.out or ARTIFACT,
            args.environment_note,
            json.loads(args.snapshot_json) if args.snapshot_json else None,
            json.loads(args.attention_backend_json) if args.attention_backend_json else None,
        )
        print(json.dumps({"floor": report["floor"]["mean_deviation"],
                          "threshold": report["floor"]["threshold"],
                          "gate": {k: v["verdict"] for k, v in report["gate"].items()},
                          "scope": {"device_name": report["scope"]["device_name"]}},
                         indent=2))
        return

    import torch

    args.leg_dir.mkdir(parents=True, exist_ok=True)
    if args.direct_reference:
        assert args.model_dir, "--direct-reference needs --model-dir"
        leg = measure_direct_reference(
            args.model_dir, args.repetitions, args.prompts, args.seq_len, args.seed
        )
    elif args.served_leg:
        leg = measure_served_leg(
            args.case, args.layer, args.base_url, args.repetitions,
            args.prompts, args.seq_len, args.seed,
            json.loads(args.launch_json) if args.launch_json else None,
            args.source_dir,
        )
        leg["broadcast_type"] = args.broadcast_type
    else:
        parser.error("one of --assemble, --served-leg, --direct-reference is required")

    path = args.leg_dir / f"{leg['leg']}.pt"
    torch.save(leg, path)
    print(f"{leg['leg']} -> {path}")


if __name__ == "__main__":
    main()
