"""Start one bare prime-rl inference server for a serving-boundary leg.

GPU box only. Every leg of the floor measurement is started by THIS script, so
that a difference between legs cannot be a difference in how the server was
brought up. The only argument that varies between legs is `--model`.

WHY A SCRIPT AND NOT prime-rl's `inference` ENTRY POINT
-------------------------------------------------------
The entry point is prime-rl's own and would be the better thing to call, but it
cannot express the configuration this measurement needs. `InferenceConfig.router`
defaults to a vllm-router fronting the engine on `server.port`, with the engine
moved to `backend_port`; the router is a separate `vllm-router` process. The
added `/collective_rpc` route lives on the ENGINE, so the extraction has to
reach the engine, and the documented way to get a bare engine on `server.port`
is `router = None` -- which the CLI cannot spell, because the field is a
discriminated union with no null tag (`--router none`, `--router null` and
`--router.type null` are all rejected, measured).

So the config is still built by prime-rl's OWN loader, `cli(InferenceConfig,
args=[...])`, with exactly one field set afterwards -- the one the CLI has no
syntax for. Nothing here hand-assembles a config object; a hand-built config
would be a fourth thing to keep in sync with the pin.

THE WORKER EXTENSION IS PASSED EXPLICITLY, and that is forced rather than
preferred. `patches/01` made prime-rl's assignment conditional on
`getattr(args, "worker_extension_cls", None) is None`, but vLLM 0.28's default
for that field is the empty STRING, not None. The guard therefore never fires,
the fallback to `WORKER_EXTENSION_CLS[transport]` never runs, and a server
started without an explicit value binds no worker extension at all -- the
`/collective_rpc` route reaches the worker and the worker answers
`NotImplementedError: Method 'install_logits_hook' is not implemented`.
Measured here, on the first leg of this sweep.

Passing the qualname through the passthrough is exactly what the recorded
attachment run did, so this measures the configuration that run validated
rather than a new one. The name comes from `composed_worker_qualname`, so it is
still the per-transport name. `bind_worker_extension` is exercised by the
composition tests at both transports instead of by this launcher.
"""

from __future__ import annotations

import argparse
import json
import sys

from weight_sync_bench.phase3.engine_probe import composed_worker_qualname


def build_args(model: str, port: int, broadcast_type: str) -> list[str]:
    """The prime-rl CLI arguments, as `cli()` will receive them.

    EVERY entry of `MATCHED_ENGINE_FLAGS` is passed, derived from that dict
    rather than listed here. Listing them by hand is how the two sides drift:
    a first version of this script passed only the two cache flags, the budget
    and the TP degree, which left the server compiling and capturing CUDA graphs
    while the direct engine ran under `enforce_eager=True`. The clean leg then
    measured 3.7e-2 against the recorded 2.3e-3 for the same tokens -- a
    sixteenfold inflation that is a launcher difference, not a boundary.

    The floor is only a floor if the two sides differ in the serving path and
    nothing else, so the matched set has exactly one definition and both sides
    read it.
    """
    from weight_sync_bench.phase3.attachment_probe import MATCHED_ENGINE_FLAGS

    args = ["--vllm.model", model, "--server.port", str(port),
            "--weight-broadcast.type", broadcast_type]
    for key, value in MATCHED_ENGINE_FLAGS.items():
        args += [f"--vllm.{key.replace('_', '-')}", str(value)]
    # See the module docstring: without this the patched guard leaves the field
    # at vLLM's empty-string default and no extension is bound.
    args += ["--vllm.worker-extension-cls", composed_worker_qualname(broadcast_type)]
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--broadcast-type", default="filesystem")
    parser.add_argument("--print-argv", action="store_true",
                        help="print the launch record and exit, starting nothing")
    args = parser.parse_args()

    if args.print_argv:
        print(json.dumps({"command": sys.argv[:1] + [
            "--model", args.model, "--port", str(args.port),
            "--broadcast-type", args.broadcast_type]}))
        return

    from prime_rl.configs.inference import InferenceConfig
    from prime_rl.utils.config import cli

    from weight_sync_bench.phase3 import attach

    config = cli(InferenceConfig, args=build_args(args.model, args.port, args.broadcast_type))
    # The one field the CLI cannot express. See the module docstring.
    config.router = None
    attach.serve(config, args.broadcast_type)


if __name__ == "__main__":
    main()
