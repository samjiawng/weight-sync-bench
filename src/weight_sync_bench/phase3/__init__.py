"""Phase 3 (SPEC.md): the correctness gate run inside a prime-rl RL loop.

Two of the three modules here (`engine_probe`, `step_runner`) only do useful
work on a GPU box with vLLM and prime-rl installed. All three import on a
CPU-only machine with neither package present: every vLLM, torch-CUDA and
prime-rl import lives inside the function that needs it, following
`phase2/collective_logits.py`. `pin` is pure `tomllib` and always works.

prime-rl is pinned but never installed as a dependency -- see the
`[tool.weight_sync_bench.phase3]` table in `pyproject.toml` for the reason and
`pin.py` for the reader.
"""
