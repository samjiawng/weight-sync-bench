"""Run one prime-rl RL step and record whether any trainer parameter moved.

Runs on a GPU box only; imports anywhere.

The question is narrow and prior to any correctness gate: does a single step of
prime-rl's own smallest loop actually update the policy? A gate that compares
logits across a weight sync is meaningless if the sync carries an unchanged
tensor, and "unchanged" is exactly the failure a passing RL run does not
surface -- loss goes down on paper, weights never move, the gate compares a
model against itself and reports agreement. So this is measured before anything
is built on top of it.

HARDWARE
--------
prime-rl splits trainer and inference into separate processes on separate GPUs.
Its smallest end-to-end loop is therefore TWO GPUs, not one:
`configs/basic/reverse-text/rl.toml` at the pinned commit sets
`num_train_gpus = 1` and `num_infer_gpus = 1`. A one-GPU box cannot run this
module at all.

WHAT IS READ, AND WHY THAT SIGNAL
----------------------------------
The trainer publishes each policy version through its weight-broadcast
transport. Under the default transport (`filesystem`) that is a directory tree
the trainer writes and the inference server reads:

    <output_dir>/broadcasts/step_<N>/     sharded safetensors + index
    <output_dir>/broadcasts/step_<N>/.sender_ready    written last

Comparing two consecutive published steps reads the trainer's own parameters as
the trainer itself hands them to inference -- the same bytes the weight sync
moves, rather than a reconstruction of them. `.sender_ready` is the trainer's
own completion marker, so a step is only compared once the trainer has declared
it whole; the index is written after a barrier, so a complete index implies
complete shards.

The steps present are DISCOVERED, never assumed. How many versions a given
`max_steps` publishes depends on prime-rl's own scheduling, which this module
does not model; asserting a count here would be a guess dressed as a check.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..phase2.bf16_floor import environment
from .pin import provenance

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = REPO_ROOT / "tolerance" / "phase3_step_runner.json"

# Relative to a prime-rl checkout at the pinned commit.
RL_CONFIG_RELPATH = "configs/basic/reverse-text/rl.toml"

# prime-rl's own names, at the pinned commit:
#   src/prime_rl/utils/pathing.py:221-226  (broadcasts/, step_<N>)
#   src/prime_rl/transports/weights/base.py:23  (.sender_ready)
BROADCAST_SUBDIR = "broadcasts"
STEP_PREFIX = "step_"
SENDER_READY_MARKER = ".sender_ready"
SAFETENSORS_INDEX = "model.safetensors.index.json"

REQUIRED_GPUS = 2


class StepRunnerError(RuntimeError):
    """The run did not produce something comparable."""


def rl_command(
    config_path: Path,
    output_dir: Path,
    max_steps: int = 1,
    extra_args: tuple[str, ...] = (),
) -> list[str]:
    """The command that runs the loop.

    `rl` is a console script prime-rl declares (`pyproject.toml` at the pinned
    commit, `[project.scripts]`), so this assumes prime-rl is installed in the
    environment the command runs in -- which on the GPU box it is, and on this
    development box it deliberately is not.

    The two override flags are the assumption most likely to need correcting on
    first contact: prime-rl's CLI is derived from its pydantic config tree, and
    while `max_steps` and `output_dir` are both top-level fields, the exact
    dashed spelling has not been verified by execution here. `extra_args` is
    the escape hatch, and the artifact records the command verbatim so a
    mismatch is visible in the artifact rather than only in a shell.
    """
    return [
        "rl",
        "--config",
        str(config_path),
        "--max-steps",
        str(max_steps),
        "--output-dir",
        str(output_dir),
        *extra_args,
    ]


def published_steps(output_dir: Path) -> list[int]:
    """Step numbers the trainer has declared complete, ascending."""
    broadcast_dir = Path(output_dir) / BROADCAST_SUBDIR
    if not broadcast_dir.is_dir():
        return []
    steps = []
    for child in broadcast_dir.iterdir():
        if not child.is_dir() or not child.name.startswith(STEP_PREFIX):
            continue
        suffix = child.name[len(STEP_PREFIX) :]
        if not suffix.isdigit():
            continue
        if (child / SENDER_READY_MARKER).exists():
            steps.append(int(suffix))
    return sorted(steps)


def step_dir(output_dir: Path, step: int) -> Path:
    return Path(output_dir) / BROADCAST_SUBDIR / f"{STEP_PREFIX}{step}"


def load_step_weights(directory: Path) -> dict[str, "Any"]:
    """Reads one published step's sharded safetensors into a name -> tensor dict."""
    directory = Path(directory)
    index_path = directory / SAFETENSORS_INDEX
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        # A model small enough not to be sharded gets no index. Qwen3-0.6B is
        # in that range, so this is the branch the reverse-text config is most
        # likely to take, not a fallback for an exotic case.
        shards = sorted(p.name for p in directory.glob("*.safetensors"))
        if not shards:
            raise StepRunnerError(f"no safetensors and no index under {directory}")

    # Imported after discovery: finding that a step published nothing is a
    # complete answer on its own and should not need the reader library to be
    # installed to reach it.
    from safetensors.torch import load_file

    tensors: dict[str, Any] = {}
    for shard in shards:
        tensors.update(load_file(directory / shard))
    return tensors


def compare_steps(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Which parameters moved between two published versions, and by how much.

    Reported per-parameter rather than as a single boolean: "some parameter
    changed" and "the parameters you expected to change changed" are different
    claims, and a partial update -- say, everything but the embedding -- is a
    real failure mode that a bare boolean would hide.
    """
    import torch

    missing = sorted(set(before) - set(after))
    added = sorted(set(after) - set(before))
    shared = sorted(set(before) & set(after))
    if not shared:
        raise StepRunnerError("the two published steps share no parameter names")

    changed: list[dict[str, Any]] = []
    unchanged: list[str] = []
    for name in shared:
        a, b = before[name], after[name]
        if a.shape != b.shape:
            changed.append({"param": name, "shape_changed": [list(a.shape), list(b.shape)]})
            continue
        if torch.equal(a, b):
            unchanged.append(name)
            continue
        delta = (b.float() - a.float()).abs()
        changed.append(
            {
                "param": name,
                "max_abs_delta": float(delta.max()),
                "mean_abs_delta": float(delta.mean()),
            }
        )

    changed.sort(key=lambda row: row.get("max_abs_delta", float("inf")), reverse=True)
    return {
        "any_parameter_changed": bool(changed),
        "num_parameters_compared": len(shared),
        "num_changed": len(changed),
        "num_unchanged": len(unchanged),
        "changed": changed,
        "unchanged": unchanged,
        "only_in_before": missing,
        "only_in_after": added,
    }


def check_hardware() -> dict[str, Any]:
    """Refuses early on a box that cannot host the loop, rather than after a
    model download and a partial launch."""
    import torch

    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if count < REQUIRED_GPUS:
        raise StepRunnerError(
            f"prime-rl runs trainer and inference on separate GPUs; "
            f"{RL_CONFIG_RELPATH} needs {REQUIRED_GPUS} (1 trainer + 1 inference), found {count}"
        )
    return {
        "gpu_count": count,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(count)],
    }


def run_one_step(
    prime_rl_dir: Path,
    output_dir: Path,
    max_steps: int = 1,
    extra_args: tuple[str, ...] = (),
    timeout: int | None = None,
) -> dict[str, Any]:
    """Launches the loop and waits for it. Returns the launch record."""
    prime_rl_dir = Path(prime_rl_dir)
    config_path = prime_rl_dir / RL_CONFIG_RELPATH
    if not config_path.is_file():
        raise StepRunnerError(f"{config_path} not found; expected a prime-rl checkout at {prime_rl_dir}")

    cmd = rl_command(config_path, Path(output_dir), max_steps, extra_args)
    completed = subprocess.run(
        cmd, cwd=prime_rl_dir, capture_output=True, text=True, timeout=timeout
    )
    record = {
        "command": cmd,
        "cwd": str(prime_rl_dir),
        "returncode": completed.returncode,
        # Tails only: a full RL log is large and the artifact is meant to be
        # read. The failure branch below carries the tail that explains it.
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    if completed.returncode != 0:
        raise StepRunnerError(
            f"`{' '.join(cmd)}` exited {completed.returncode}\n"
            f"--- stderr tail ---\n{completed.stderr[-4000:]}"
        )
    return record


def run_probe(
    prime_rl_dir: Path,
    output_dir: Path,
    max_steps: int = 1,
    extra_args: tuple[str, ...] = (),
    timeout: int | None = None,
) -> dict[str, Any]:
    hardware = check_hardware()
    launch = run_one_step(prime_rl_dir, output_dir, max_steps, extra_args, timeout)

    steps = published_steps(output_dir)
    if len(steps) < 2:
        raise StepRunnerError(
            f"need two published steps to compare, found {steps or 'none'} under "
            f"{Path(output_dir) / BROADCAST_SUBDIR}. The run itself succeeded, so this "
            f"is about how many versions max_steps={max_steps} publishes, not a crash."
        )
    before_step, after_step = steps[0], steps[1]
    comparison = compare_steps(
        load_step_weights(step_dir(output_dir, before_step)),
        load_step_weights(step_dir(output_dir, after_step)),
    )

    return {
        "probe": "one prime-rl step, trainer parameter movement",
        "question": "Does one step of prime-rl's smallest loop actually update the policy?",
        "config": RL_CONFIG_RELPATH,
        "max_steps": max_steps,
        "compared_steps": [before_step, after_step],
        "all_published_steps": steps,
        "comparison": comparison,
        "launch": launch,
        "hardware": hardware,
        "prime_rl_pin": provenance(),
        "environment": environment(),
    }


def write(report: dict[str, Any], path: Path = ARTIFACT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prime-rl-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("extra", nargs="*", help="extra arguments forwarded to `rl`")
    args = parser.parse_args()

    report = run_probe(
        args.prime_rl_dir, args.output_dir, args.max_steps, tuple(args.extra), args.timeout
    )
    path = write(report, args.out or ARTIFACT)
    changed = report["comparison"]["any_parameter_changed"]
    print(f"any_parameter_changed={changed} -> {path}")


if __name__ == "__main__":
    main()
