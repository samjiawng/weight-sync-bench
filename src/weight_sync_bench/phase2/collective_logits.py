"""Design + implementation of a collective_rpc-based replacement for
bf16_floor.py's prompt_logprobs full-vocab extraction path. SPEC.md phase 2,
section 2a follow-up.

UNTESTED. Written on a CPU-only dev machine against vLLM v0.28.0 source read
directly from github.com/vllm-project/vllm at tag v0.28.0 (citations below
give exact files/line ranges as they stood at that tag). It has not been run
against a real GPU or a real vLLM install. Every vLLM API call in this module
is unverified by execution -- read the citations, do not trust this file
because it reads plausibly, and smoke-test with tiny repetitions/batch/seq_len
before trusting any number out of it, same discipline as bf16_floor.py.

`bf16_floor.py` is left completely unmodified. This module is additive, so the
two extraction paths can be run side by side on the box and compared -- both
for correctness (do they measure the same floor?) and for wall-clock cost.

--------------------------------------------------------------------------
PROBLEM
--------------------------------------------------------------------------
bf16_floor.py._run_worker requests `SamplingParams(prompt_logprobs=-1)` (all
vocab_size=151936 logprobs, uncapped) and then, in the SAME process, walks
`output.prompt_logprobs` -- a `list[dict[int, Logprob] | None]`, one dict per
prompt position, each with up to 151936 entries -- with a nested Python loop
to scatter it into a dense tensor. A `--batch 4 --seq-len 64` run was killed
after an hour without completing.

There are actually TWO Python-object-construction costs stacked here, both
scaling with vocab_size * num_positions, not with the ~4.9MB logits tensor
itself:

  1. INSIDE vLLM, driver-process side: `LogprobsProcessor._update_prompt_logprobs`
     (vllm/v1/engine/logprobs.py:121-180) calls `.tolist()` on the returned
     `LogprobsTensors` and then loops per (position, vocab-entry) building a
     `Logprob` container object per entry via `append_logprobs_for_next_position`
     -- this is what actually builds the "one Python object per (position,
     vocab-entry)" bf16_floor.py's docstring already diagnosed.
  2. IN THIS SCRIPT, after `generate()` returns: the second nested loop over
     `output.prompt_logprobs` (`_run_worker`'s `for i, pos in enumerate(positions):
     for token_id, logprob_obj in pos.items(): dense[i, token_id] = ...`) --
     this repeats the same O(positions * vocab) Python-level work a second
     time just to get the values back into a tensor.

--------------------------------------------------------------------------
FOUR QUESTIONS, ANSWERED FROM SOURCE (v0.28.0, everything below is read from
GitHub, not executed)
--------------------------------------------------------------------------

Q1. Entry point for a raw forward pass + logits; does it need hand-built
    input batch metadata?

    `Qwen3ForCausalLM.compute_logits(self, hidden_states) -> torch.Tensor | None`
    (vllm/model_executor/models/qwen3.py:330-334):

        def compute_logits(self, hidden_states):
            logits = self.logits_processor(self.lm_head, hidden_states)
            return logits

    For Qwen3-0.6B at 0.28.0 with default settings, the worker's model runner
    is `GPUModelRunnerV1` (vllm/v1/worker/gpu_model_runner.py), not the newer
    `GPUModelRunnerV2` (vllm/v1/worker/gpu/model_runner.py). Verified by
    tracing: `gpu_worker.py:425-438` branches
    `GPUModelRunnerV2 if self.use_v2_model_runner else GPUModelRunnerV1`, and
    `use_v2_model_runner` (vllm/config/vllm.py:615-666) falls through to
    `_is_default_v2_model_runner_model()` (vllm/config/vllm.py:691-703), which
    checks `model_config.architectures` against
    `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES` (vllm/config/vllm.py:69-79) --
    `{"DeepseekV2ForCausalLM", "DeepseekV4ForCausalLM", "GraniteMoeForCausalLM",
    "InklingForCausalLM", "InklingForConditionalGeneration",
    "KimiK3ForConditionalGeneration", "LongcatFlashNgramForCausalLM",
    "Qwen2MoeForCausalLM"}` -- "Qwen3ForCausalLM" is not in that set, so V1 is
    used absent a `VLLM_USE_V2_MODEL_RUNNER=1` override. THIS ASSUMPTION IS
    CHECKED AT RUNTIME below (`_install_logits_hook` asserts the runner's
    class name) rather than silently trusted, since it is a config-dependent
    branch this analysis cannot exercise.

    `compute_logits` is already called from exactly this raw-forward-pass
    entry point inside the real engine loop, no synthetic construction needed:
      - gpu_model_runner.py:5802, inside `_get_prompt_logprobs_dict`, once per
        prompt-logprobs-enabled request, over every requested prompt position
        in one call.
      - gpu_model_runner.py:4600 / :4619, the main per-step path, over
        whichever positions `logits_indices` selects (normally just the next
        token to sample).

    Hand-building batch metadata (a `SchedulerOutput`, KV-cache block tables,
    attention-backend metadata) to call `execute_model()` standalone was
    considered and rejected: that state is deeply `Scheduler`-internal and not
    designed for standalone construction. Instead this design rides a REAL
    `LLM.generate()` call -- the real `Scheduler` builds all of that exactly
    as production does -- and intercepts the value at the point it is already
    computed, by monkeypatching `compute_logits` on the worker's model
    instance before calling `generate()`.

Q2. `collective_rpc` signature, worker-side registration, return values.

    Signature (unchanged across the whole call chain -- LLM.collective_rpc
    (vllm/entrypoints/llm.py:567) -> LLMEngine.collective_rpc
    (vllm/v1/engine/llm_engine.py:421) -> EngineCore.collective_rpc
    (vllm/v1/engine/core.py:952) -> MultiprocExecutor.collective_rpc
    (vllm/v1/executor/multiproc_executor.py:372)):

        def collective_rpc(
            self,
            method: str | Callable[..., _R],
            timeout: float | None = None,
            args: tuple = (),
            kwargs: dict[str, Any] | None = None,
        ) -> list[_R]

    Worker-side dispatch (multiproc_executor.py:1026-1037,
    `WorkerProc.worker_busy_loop`):

        if isinstance(method, str):
            func = getattr(self.worker, method)
        elif isinstance(method, bytes):
            func = partial(cloudpickle.loads(method), self.worker)
        output = func(*args, **kwargs)

    A string names an existing bound method on the `Worker` object. A
    callable is `cloudpickle.dumps`'d by the driver (multiproc_executor.py:416),
    sent to every worker process, `cloudpickle.loads`'d and bound via
    `functools.partial(fn, self.worker)` -- so a callable's first positional
    argument is the `Worker` instance itself, matching `LLM.collective_rpc`'s
    own docstring ("If the method is a callable, it should accept an
    additional `self` argument... The `self` argument will be the worker
    object.").

    Every rank runs the call and returns a result; `MultiprocExecutor`
    collects one entry per rank into `list[_R]` (multiproc_executor.py:419-441,
    `get_response` appends one result per `response_mqs` entry, in rank
    order). `MultiprocExecutor.collective_rpc` also accepts a
    `unique_reply_rank` parameter for a single-rank response, but that
    parameter is NOT threaded through `EngineCore.collective_rpc` (core.py:952,
    only forwards `method, timeout, args, kwargs`) or `LLMEngine`/`LLM` above
    it -- so from a script holding an `LLM` instance, `unique_reply_rank` is
    unreachable, and "only rank 0 has real data" has to be handled inside the
    callable itself, not requested from the driver side.

Q3. Under TP=2, does the worker already gather logits, or does each rank hold
    a vocab slice? all-gather-on-worker vs concatenate-on-driver?

    Already gathered, INSIDE `compute_logits` itself, and it is a gather to
    one rank, not an all-gather. `vllm/model_executor/layers/logits_processor.py:
    118-129` (`LogitsProcessor._gather_logits`):

        if self.use_all_gather:
            logits = tensor_model_parallel_all_gather(logits)
        else:
            # None may be returned for rank > 0
            logits = tensor_model_parallel_gather(logits)

    `current_platform.use_all_gather()` is False on CUDA (the comment says
    the all-gather branch exists for platforms like TPU that require strict
    SPMD). So on the target platform, every call to `compute_logits` already
    performs a gather-to-rank-0 as part of computing that step's logits --
    this is not something the design needs to add.

    DECISION: neither all-gather-on-worker nor concatenate-on-driver, in the
    sense the question poses them -- there is nothing to gather or
    concatenate ourselves. The hook simply captures whatever `compute_logits`
    returns on each rank: a real tensor on rank 0, `None` on every other rank,
    for free, as a consequence of vLLM's own internal gather. The retrieval
    side (`_retrieve_and_clear_logits`) picks the one non-None result out of
    the list `collective_rpc` returns.

Q4. How do return values cross the boundary? Must tensors move to CPU first?

    Yes. Results are pickled through
    `vllm.distributed.device_communicators.shm_broadcast.MessageQueue`, a
    shared-memory ring buffer (`max_chunk_bytes` defaults to `1024*1024*24` =
    24 MiB, shm_broadcast.py:472). Its `enqueue()` (shm_broadcast.py:823-860)
    has a code path specifically for CPU tensors -- the comment there:
    "CPU tensors are routed through `_reduce_tensor` so that their bytes are
    emitted as out-of-band buffers instead of being copied into the pickle
    stream by torch's default reducer" -- and a payload exceeding the chunk
    size automatically overflows to a local ZeroMQ socket
    (`self.local_socket.send_multipart(...)`) rather than failing. GPU
    tensors get no such dedicated path in this code, and `collective_rpc`'s
    own docstring says: "It is recommended to use this API to only pass
    control messages, and set up data-plane communication to pass data."
    Both point the same way: `.detach().cpu()` before returning from the
    worker callable, done in `_install_logits_hook`'s wrapper below.

Q5. Does the CPU float32 cast in the hook happen after the bf16 computation,
    or does it change what dtype the computation itself runs in?

    After -- confirmed by reading the full call chain the hook wraps, from
    `compute_logits` down to the actual GEMM, with nothing in between that
    would force fp32:
      - `Qwen3ForCausalLM.compute_logits` (qwen3.py:330-334) calls
        `self.logits_processor(self.lm_head, hidden_states)`, i.e.
        `LogitsProcessor.forward` (logits_processor.py:97-115), which calls
        `_get_logits` (logits_processor.py:171-186).
      - `_get_logits` calls `_apply_head` (logits_processor.py:130-169) --
        `self.head_dtype = model_config.head_dtype`, `None` unless a
        `--hf-overrides '{"head_dtype": ...}'` is passed (not set anywhere in
        this repo). With `head_dtype is None`, `_apply_head` takes its first
        branch: `lm_head.quant_method.apply(lm_head, hidden_states,
        bias=embedding_bias)` -- the LM-head GEMM runs in whatever dtype
        `hidden_states` already is, i.e. the model's compute dtype (bf16,
        from `LLM(dtype="bfloat16", ...)`). The fp32-accumulation branch in
        `_apply_head` only triggers when `head_dtype == torch.float32`
        explicitly, which is not this configuration.
      - `_get_logits` then gathers across TP (`_gather_logits`, Q3) and
        slices off vocab padding (`logits[..., :org_vocab_size]`) -- a
        gather/concatenation and a slice, neither of which promotes dtype;
        `logits` is still bf16 when `compute_logits` returns it.
      - Only then does this module's `hooked_compute_logits` wrapper call
        `.to(dtype=torch.float32, device="cpu")` on the already-returned bf16
        tensor -- strictly a post-hoc upcast of a value bf16 arithmetic
        already produced, not a change to the arithmetic itself. Upcasting a
        bf16 result to fp32 is lossless (every bf16 value is exactly
        representable in fp32) and matches what
        gpu_model_runner.py:5812-5813 already does in the existing
        prompt_logprobs path (`scores = logits.to(torch.float32)` under
        `logprobs_mode="raw_logits"`) -- this hook measures the same
        quantity, cast at the same point in the pipeline.
      - One nearby method, `LogitsProcessor.get_top_tokens`
        (logits_processor.py:189-234), does upcast to float32 mid-computation
        ("Use float32 to avoid bf16 precision loss on large vocab indices")
        -- but that is a different method, an alternative argmax-only
        optimization path, and is not on `compute_logits`'s call graph, so it
        does not apply here. Noted only so a future reader who greps for
        `.float()` in this file does not mistake it for evidence against the
        above.

--------------------------------------------------------------------------
DESIGN
--------------------------------------------------------------------------
1. `_install_logits_hook(worker)` -- a collective_rpc callable. Wraps
   `worker.model_runner.model.compute_logits` with a closure that calls the
   original, and if the result is not None (i.e. this is rank 0, or TP=1),
   clones it to CPU float32 and appends it to a list stashed on the worker
   (`worker._collective_logits_captures`). Returns the original output
   unchanged, so normal engine behavior (sampling, the existing
   prompt_logprobs machinery) is untouched -- this hook only *observes*.

2. Drive a real `LLM.generate()` call exactly as bf16_floor.py does, EXCEPT
   `prompt_logprobs=1` instead of `prompt_logprobs=-1`. Per
   gpu_model_runner.py:5747-5802 and vllm/v1/sample/sampler.py:310-339
   (`Sampler.gather_logprobs`, `torch.topk(logprobs, num_logprobs, dim=-1)`),
   the code path that computes logits at every prompt position is gated by
   `sampling_params.prompt_logprobs is not None` -- not by its magnitude.
   Everything downstream that scales with vocab_size (the `topk` call, the
   `LogprobsTensors` CPU buffers sized `num_prompt_logprobs + 1` wide, and
   both Python-object-construction loops described in PROBLEM above) scales
   with the REQUESTED count instead once the hook has already captured the
   full untruncated tensor straight off `compute_logits`. `prompt_logprobs=1`
   keeps that whole downstream machinery (which we no longer need, and don't
   disable) cheap rather than trying to disable it.

3. `_retrieve_and_clear_logits(worker, expected_min_positions)` -- a second
   collective_rpc callable, called after `generate()` returns. Reads and
   clears `worker._collective_logits_captures`. A single `generate()` call
   with `max_tokens=1` calls `compute_logits` twice per request (once for
   prompt logprobs over every position, once for the single sampled token,
   per Q1's two call sites), so with chunked prefill and prefix caching both
   disabled (design point 3a below) exactly one captured tensor should have
   `shape[0] > 1` (the prompt-logprobs one) and exactly one should have
   `shape[0] == 1` (the sampled token). **This is asserted, not heuristically
   selected**: if more than one captured tensor has `shape[0] > 1`, retrieval
   raises with every such shape listed, rather than silently picking the
   largest. A silent "pick the biggest one" would have quietly returned a
   partial chunk instead of the full prompt tensor exactly when chunked
   prefill fires -- the same failure mode this design otherwise disables at
   the source (3a) is not allowed to also hide behind a heuristic here.

3a. **Chunked prefill and prefix caching are explicitly disabled** in the
   `LLM(...)` constructor in `_run_worker` (`enable_chunked_prefill=False,
   enable_prefix_caching=False`), not left at their defaults. Both default to
   `True` for a standard generate model (`ModelConfig.is_chunked_prefill_supported`
   / `is_prefix_caching_supported`, vllm/config/model.py:1988-2044, resolved
   in `EngineArgs._set_default_chunked_prefill_and_prefix_caching_args`,
   vllm/engine/arg_utils.py:2645-2704) -- bf16_floor.py's own `LLM(...)` call
   does not set either, so it runs with both on by default (confirmed
   independently: its engine log from the real 20-repetition run shows
   `enable_prefix_caching=True`; see `tolerance/phase2a_bf16_floor.json` for
   the resulting caveat on that measurement).

   Both are silent failure modes for this specific extraction method, and
   both get worse exactly as the sweep grows (longer seq_len, more
   repetitions -- the two things phase 2a's roadmap wants more of):
     - Chunked prefill splits one long prefill across multiple forward
       passes, so `compute_logits` fires once per chunk instead of once for
       the whole prompt (gpu_model_runner.py:5778-5786's `num_logits =
       min(num_tokens, num_remaining_tokens)` is exactly the per-chunk
       slice). Retrieval would then see several `shape[0] > 1` captures, none
       of which is the full tensor -- design point 3's assertion is what
       catches this instead of silently returning a chunk.
     - Prefix caching means a repeated prompt (or a shared prefix across the
       one-seed-per-repetition prompts `_seeded_token_batches` generates) can
       skip recomputation for the cached portion entirely, serving cached KV
       (and, upstream of that, the cached run's already-computed hidden
       states) instead of running the forward pass again. That would make
       repeated measurements non-independent -- the entire premise of the
       floor being an average over independently-seeded repetitions (see
       tolerance.py's docstring on why 20 reps at different seeds is the
       basis for `mean_deviation`) breaks silently, with no error and no
       shape mismatch to catch it.

4. `_uninstall_logits_hook(worker)` -- restores the original
   `compute_logits`, called after every retrieval so the hook does not leak
   into whatever the box does next (e.g. SPEC.md 2d's weight-transfer work,
   which also touches these worker processes).

5. `run_one_prompt(llm, token_ids) -> torch.Tensor` -- ties 1-4 together for
   one prompt: install, generate, retrieve, uninstall. Returns a
   `[seq_len - 1, vocab]` float32 CPU tensor -- position 0 dropped, same
   convention as bf16_floor.py's `_run_worker` (there is no preceding context
   to condition position 0's logits on, so it is not comparable and both
   paths drop it). This drop happens automatically here, not as an explicit
   slice: gpu_model_runner.py:5776-5786 computes `num_logits =
   num_prompt_tokens - (start_idx + 1)` for a non-chunked request, i.e.
   exactly `seq_len - 1` positions, the same set bf16_floor.py's
   `output.prompt_logprobs[1:]` selects.

6. `_run_worker` / `measure_differential_floor` -- structurally identical to
   bf16_floor.py's functions of the same name (same one-prompt-at-a-time
   submission discipline and its rationale, same TP1-vs-TP2
   subprocess-per-degree pattern, same torch.save/torch.load handoff to the
   parent process), with `run_one_prompt` substituted for the
   prompt_logprobs-and-Python-loop extraction. The LLM constructor args are
   the same EXCEPT `enable_chunked_prefill=False, enable_prefix_caching=False`
   (design point 3a) -- bf16_floor.py's own `LLM(...)` call leaves both at
   their (on) defaults, so this is a deliberate divergence, not an oversight;
   see the SMOKE TEST section below for how to hold the computation identical
   across both modules despite it.

--------------------------------------------------------------------------
WHAT THIS DOES NOT ADDRESS -- flagged, not solved
--------------------------------------------------------------------------
- `torch.compile` / CUDA graph interaction: bf16_floor.py's LLM construction
  already passes `enforce_eager=True`, which should mean `compute_logits` is
  called as a plain Python method (not captured inside a compiled graph or a
  CUDA-graph replay region) -- but this was not verified by tracing the
  compile/capture boundary in gpu_model_runner.py, and cudagraph interaction
  is exactly the kind of thing that silently breaks a monkeypatch. If this
  module is adapted to run without `enforce_eager=True`, re-verify that
  `compute_logits` is still a live Python call at the point the hook patches
  it.
- The V1/V2 model runner branch is model- and config-dependent (Q1). The
  `RuntimeError` in `_install_logits_hook` below turns a wrong assumption
  into a loud failure rather than a silent wrong measurement, but a V2
  implementation of this hook was not written -- V2's prompt-logprobs path
  lives in vllm/v1/worker/gpu/sample/prompt_logprob.py and calls
  `compute_logits` (there called `logits_fn`) through a different chunking
  loop (`compute_prompt_logprobs_with_chunking`, CHUNK_SIZE=1024); the same
  hook on `model.compute_logits` should still work there since the call site
  is the same underlying model method, but this was not traced as carefully
  as the V1 path and is unverified.
- Monkeypatching an instance attribute (`model.compute_logits = hooked`)
  shadows the class method via normal Python instance-`__dict__` lookup, a
  standard technique -- but was not verified against whatever `nn.Module`
  machinery (hooks, `__setattr__` overrides) vLLM's model base classes might
  define that could interfere with a plain attribute assignment.
- No timing comparison exists yet between this path and bf16_floor.py's,
  since neither has been run. That comparison is the actual point of writing
  this as a second, side-by-side module rather than editing the original.
- Disabling chunked prefill and prefix caching (design point 3a) changes the
  configuration bf16_floor.py's own measurement was taken under (which runs
  with both left at their defaults, i.e. on). The two paths are therefore
  not a pure extraction-method A/B test as configured -- see the smoke test
  below for what IS held constant across them, and
  `tolerance/phase2a_bf16_floor.json` for the resulting prefix-caching
  caveat on the committed floor.

--------------------------------------------------------------------------
SMOKE TEST -- run before trusting any number out of this module
--------------------------------------------------------------------------
Same model, same input, same kernels; only the extraction method differs
between this module and bf16_floor.py. That means the two paths must produce
BIT-IDENTICAL logits for the same prompt, not merely approximately equal --
any difference at all is a bug in one of the two extraction paths, not
floating-point noise (there is no dtype conversion, no reduction-order
change, and no additional forward pass between them; both read out the same
`compute_logits` return value for the same tokens on the same loaded model).

    import torch
    from vllm import LLM, SamplingParams
    from weight_sync_bench.phase2 import bf16_floor, collective_logits

    llm = LLM(
        model=checkpoint_dir, tensor_parallel_size=1, dtype="bfloat16",
        enforce_eager=True, max_logprobs=-1, logprobs_mode="raw_logits",
        enable_chunked_prefill=False, enable_prefix_caching=False,
    )
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]  # any short prompt

    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=-1)
    [output] = llm.generate([{"prompt_token_ids": tokens}], sp, use_tqdm=False)
    old = torch.empty(len(tokens) - 1, collective_logits.QWEN3_0_6B.vocab)
    for i, pos in enumerate(output.prompt_logprobs[1:]):
        for token_id, logprob_obj in pos.items():
            old[i, token_id] = logprob_obj.logprob

    new = collective_logits.run_one_prompt(llm, tokens)

    if not torch.equal(old, new):
        max_diff = (old - new).abs().max().item()
        raise AssertionError(
            f"extraction paths disagree: max abs diff = {max_diff:.3e} "
            f"(expected exact equality -- see module docstring's SMOKE TEST)"
        )

Both paths must run against the SAME `llm` instance -- the LLM constructor
kwargs must match on every argument that affects computation
(`enable_chunked_prefill`/`enable_prefix_caching` included, since
bf16_floor.py does not set them but this smoke test must, to hold the
computation identical while isolating the extraction method as the only
variable). `torch.equal`, not `torch.allclose`: if this ever needs relaxing
to tolerate real floating-point noise, that in itself is a finding (it would
mean the two paths are not actually reading the same computed value), not a
reason to loosen the assertion first.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .bf16_floor import (
    MODEL_ID,
    QWEN3_0_6B,
    ULP_BF16,
    derive_threshold,
)

if TYPE_CHECKING:
    import torch

# --------------------------------------------------------------------------- #
# Worker-side collective_rpc callables. Each is sent via cloudpickle to every
# worker process (Q2) and called with the Worker instance as its first
# argument. Everything that touches torch/vllm internals is imported lazily
# inside the function bodies, matching bf16_floor.py's convention, so that
# `import weight_sync_bench.phase2.collective_logits` for its constants does
# not require the phase2 extra to be installed.
# --------------------------------------------------------------------------- #

_EXPECTED_RUNNER_CLASS = "GPUModelRunnerV1"


def _install_logits_hook(worker: Any) -> None:
    """collective_rpc callable. See Q1/Q3/design point 1 in the module
    docstring for what this wraps and why. Idempotent per worker process:
    re-installing after `_uninstall_logits_hook` is safe; installing twice
    without uninstalling raises rather than silently double-wrapping.
    """
    runner = worker.model_runner
    runner_class = type(runner).__name__
    if runner_class != _EXPECTED_RUNNER_CLASS:
        raise RuntimeError(
            f"expected {_EXPECTED_RUNNER_CLASS} (verified for Qwen3ForCausalLM "
            f"at vLLM v0.28.0 default settings -- see the module docstring's "
            f"Q1 answer), got {runner_class!r}. VLLM_USE_V2_MODEL_RUNNER may "
            "be set, or a different model/version combination is in use that "
            "this hook was not verified against."
        )

    model = runner.model
    if getattr(model, "_collective_logits_hook_installed", False):
        raise RuntimeError(
            "hook already installed on this worker; call "
            "_uninstall_logits_hook before installing again"
        )

    original_compute_logits = model.compute_logits

    def hooked_compute_logits(hidden_states, *args, **kwargs):
        import torch

        out = original_compute_logits(hidden_states, *args, **kwargs)
        # Non-None only on the rank `tensor_model_parallel_gather` gathered
        # to (Q3) -- rank 0 on CUDA. Cast to float32 to match
        # gpu_model_runner.py:5812-5813's own `logits_mode="raw_logits"`
        # handling, so this path measures the same quantity bf16_floor.py's
        # prompt_logprobs path does.
        if out is not None:
            worker._collective_logits_captures.append(
                out.detach().to(dtype=torch.float32, device="cpu")
            )
        return out

    worker._collective_logits_captures = []
    worker._collective_logits_original_compute_logits = original_compute_logits
    model.compute_logits = hooked_compute_logits
    model._collective_logits_hook_installed = True


def _retrieve_and_clear_logits(
    worker: Any, expected_min_positions: int
) -> "torch.Tensor | None":
    """collective_rpc callable. Returns the CPU float32 [positions, vocab]
    tensor captured since the last call (or since install), and clears the
    buffer. Returns None on every rank whose `compute_logits` never produced
    a non-None result (Q3) -- i.e. every rank except the one
    tensor_model_parallel_gather gathered to.

    With chunked prefill and prefix caching disabled (see `_run_worker`'s
    `LLM(...)` call and the module docstring's design point 3a), a single
    `generate()` call with `max_tokens=1` should produce exactly one capture
    with `shape[0] > 1` (the prompt-logprobs tensor, one call site) and
    exactly one with `shape[0] == 1` (the sampled token, the other call
    site) -- see Q1. This is ASSERTED, not heuristically selected: if
    chunking or caching is silently active despite the constructor flags (or
    if the compute_logits call graph doesn't match what was traced), more
    than one multi-position capture will show up here, and that is exactly
    the failure this function exists to surface loudly instead of
    quietly returning a partial chunk.
    """
    captures = getattr(worker, "_collective_logits_captures", [])
    worker._collective_logits_captures = []
    if not captures:
        return None

    candidates = [t for t in captures if t.shape[0] > 1]
    if len(candidates) > 1:
        shapes = [tuple(t.shape) for t in candidates]
        raise RuntimeError(
            f"expected at most one compute_logits capture with more than one "
            f"position (chunked prefill and prefix caching should both be "
            f"disabled -- see design point 3a), got {len(candidates)}: "
            f"shapes {shapes}. Either the constructor flags did not take "
            "effect, or an unexpected extra compute_logits call happened."
        )
    if not candidates:
        # Every capture had exactly 1 position: either the prompt genuinely
        # has 1 comparable position (expected_min_positions <= 1), or no
        # prompt-logprobs tensor was captured at all.
        if expected_min_positions <= 1 and captures:
            return captures[0]
        raise RuntimeError(
            f"expected a compute_logits capture with >= {expected_min_positions} "
            f"positions, but every capture had exactly 1 position: "
            f"{[tuple(t.shape) for t in captures]}"
        )

    best = candidates[0]
    if best.shape[0] < expected_min_positions:
        raise RuntimeError(
            f"captured logits have {best.shape[0]} positions, expected at "
            f"least {expected_min_positions} -- the compute_logits call "
            "graph for this request may not match what this hook assumed "
            "(see the module docstring's 'what this does not address')"
        )
    return best


def _uninstall_logits_hook(worker: Any) -> None:
    """collective_rpc callable. Restores the original compute_logits so the
    hook does not linger for whatever the box does with this worker next.
    """
    runner = worker.model_runner
    model = runner.model
    original = getattr(worker, "_collective_logits_original_compute_logits", None)
    if original is None:
        return
    model.compute_logits = original
    model._collective_logits_hook_installed = False
    worker._collective_logits_original_compute_logits = None
    worker._collective_logits_captures = []


# --------------------------------------------------------------------------- #
# Driver-side orchestration. Mirrors bf16_floor.py's _run_worker /
# measure_differential_floor structure and rationale (one prompt at a time,
# one subprocess per TP degree) with the extraction method swapped.
# --------------------------------------------------------------------------- #


def run_one_prompt(llm: Any, token_ids: list[int]) -> "torch.Tensor":
    """Runs one teacher-forced prompt through `llm` and returns a
    [len(token_ids) - 1, vocab] float32 CPU tensor of raw logits -- position
    0 dropped, matching bf16_floor.py's `output.prompt_logprobs[1:]`
    convention (see design point 5 for why this drop happens automatically).
    """
    from vllm import SamplingParams

    llm.collective_rpc(_install_logits_hook)
    try:
        sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
        llm.generate([{"prompt_token_ids": token_ids}], sp, use_tqdm=False)

        expected_min_positions = len(token_ids) - 1
        results = llm.collective_rpc(
            _retrieve_and_clear_logits, args=(expected_min_positions,)
        )
        non_none = [r for r in results if r is not None]
        if len(non_none) != 1:
            raise RuntimeError(
                f"expected exactly one rank to return captured logits (Q3: "
                f"gather-to-rank-0), got {len(non_none)} of {len(results)} "
                "ranks -- current_platform.use_all_gather() may be True on "
                "this box (see the module docstring's Q3 answer), which "
                "would mean every rank returns the same tensor instead."
            )
        return non_none[0]
    finally:
        llm.collective_rpc(_uninstall_logits_hook)


def _run_worker(
    model_dir: str,
    tp: int,
    repetitions: int,
    batch: int,
    seq_len: int,
    out_path: Path,
) -> None:
    """Runs inside its own process -- same reasoning as bf16_floor.py's
    function of the same name (two LLM() instances in one process is flaky).
    Loads `model_dir` at TP degree `tp`, computes full-vocab raw-logit
    prompt tensors for `repetitions` seeded token batches via the
    collective_rpc hook instead of prompt_logprobs=-1, and torch.saves them
    (a list of [batch, seq_len - 1, vocab] float32 tensors) to `out_path`.
    """
    import torch
    from vllm import LLM

    from .bf16_floor import _seeded_token_batches

    llm = LLM(
        model=model_dir,
        tensor_parallel_size=tp,
        dtype="bfloat16",
        enforce_eager=True,  # see module docstring's compute_logits/compile caveat
        gpu_memory_utilization=0.85,
        # Both default to True for a standard generate model (see design
        # point 3a) and both are silent failure modes for this extraction
        # method: chunked prefill makes compute_logits fire once per chunk
        # instead of once for the whole prompt (retrieval would see several
        # multi-position captures instead of one -- see
        # _retrieve_and_clear_logits's assertion), and prefix caching can
        # skip recomputation for a repeated prompt/prefix entirely, which
        # would make the repetitions this floor is averaged over
        # non-independent. Neither failure raises or changes a tensor shape
        # on its own, so both are disabled explicitly rather than left at
        # their defaults.
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
    )

    token_batches = _seeded_token_batches(repetitions, batch, seq_len, QWEN3_0_6B.vocab)
    all_reps: list["torch.Tensor"] = []

    for tokens in token_batches:
        rep_positions = [run_one_prompt(llm, row) for row in tokens.tolist()]
        all_reps.append(torch.stack(rep_positions, dim=0))  # [batch, seq_len-1, vocab]

    torch.save(all_reps, out_path)


def _spawn_worker(
    python: str,
    model_dir: str,
    tp: int,
    repetitions: int,
    batch: int,
    seq_len: int,
    out_path: Path,
) -> None:
    import subprocess

    subprocess.run(
        [
            python,
            "-m",
            "weight_sync_bench.phase2.collective_logits",
            "--worker",
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
            "--out",
            str(out_path),
        ],
        check=True,
    )


def measure_differential_floor(
    repetitions: int, batch: int, seq_len: int, python: str = sys.executable
) -> dict[str, Any]:
    """Same measurement as bf16_floor.measure_differential_floor, same
    inputs, extraction path swapped -- for a direct side-by-side comparison
    of both the resulting numbers and the wall-clock cost.
    """
    import tempfile

    import torch
    from huggingface_hub import snapshot_download

    checkpoint_dir = snapshot_download(MODEL_ID)

    cells: list[dict[str, float]] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tp1_path, tp2_path = tmp / "tp1.pt", tmp / "tp2.pt"
        for tp, out_path in ((1, tp1_path), (2, tp2_path)):
            _spawn_worker(python, checkpoint_dir, tp, repetitions, batch, seq_len, out_path)

        tp1_reps = torch.load(tp1_path)
        tp2_reps = torch.load(tp2_path)

        for tp1_logits, tp2_logits in zip(tp1_reps, tp2_reps):
            diff = (tp2_logits - tp1_logits).abs()
            cells.append(
                {
                    "max": diff.max().item(),
                    "median": diff.median().item(),
                    "mean": diff.mean().item(),
                }
            )

    mean_deviation = sum(cell["mean"] for cell in cells) / len(cells)
    max_deviation = max(cell["max"] for cell in cells)
    return {
        "cells": cells,
        "mean_deviation": mean_deviation,
        "max_deviation": max_deviation,
        "mean_ulp": mean_deviation / ULP_BF16,
        "max_ulp": max_deviation / ULP_BF16,
        "threshold": derive_threshold(mean_deviation),
        "checkpoint_dir": checkpoint_dir,
        "extraction_method": "collective_rpc",
    }


def main() -> None:
    import argparse
    import time

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--tp", type=int, default=1, choices=(1, 2))
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.worker:
        assert args.model_dir and args.out
        _run_worker(
            args.model_dir, args.tp, args.repetitions, args.batch, args.seq_len, args.out
        )
        return

    start = time.monotonic()
    floor = measure_differential_floor(args.repetitions, args.batch, args.seq_len)
    elapsed = time.monotonic() - start

    print(f"collective_rpc extraction path -- elapsed {elapsed:.1f}s")
    print(f"  mean deviation : {floor['mean_deviation']:.3e} ({floor['mean_ulp']:.2f} ULP)")
    print(f"  max  deviation : {floor['max_deviation']:.3e} ({floor['max_ulp']:.2f} ULP)")
    print(f"  threshold      : {floor['threshold']:.3e}")
    print(
        "\nCompare against: .venv-phase2/bin/python -m weight_sync_bench.phase2.bf16_floor "
        f"--repetitions {args.repetitions} --batch {args.batch} --seq-len {args.seq_len}"
    )


if __name__ == "__main__":
    main()
