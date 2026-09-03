"""Phase 2 (SPEC.md): weight-sync latency against vLLM.

Requires the `phase2` extra (`uv sync --extra phase2`) and a CUDA GPU. Nothing
under `src/reshard_bench` outside this package imports vLLM, so phase 1's
default install and test suite are unaffected.
"""
