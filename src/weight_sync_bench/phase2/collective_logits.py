"""Design + implementation of a collective_rpc-based replacement for
bf16_floor.py's prompt_logprobs full-vocab extraction path. SPEC.md phase 2,
section 2a follow-up.

UNTESTED (still, as of this revision). Written on a CPU-only dev machine
against vLLM v0.28.0 source read directly from github.com/vllm-project/vllm
at tag v0.28.0 (citations below give exact files/line ranges as they stood at
that tag). It has not been run against a real GPU or a real vLLM install.
Every vLLM API call in this module is unverified by execution -- read the
citations, do not trust this file because it reads plausibly, and smoke-test
with tiny repetitions/batch/seq_len before trusting any number out of it,
same discipline as bf16_floor.py.

`bf16_floor.py` is left completely unmodified. This module is additive, so the
two extraction paths can be run side by side on the box and compared -- both
for correctness (do they measure the same floor?) and for wall-clock cost.

--------------------------------------------------------------------------
HOW THIS REVISION CAME ABOUT, AND THE ONE MISTAKE BEHIND TWO OF ITS BUGS
--------------------------------------------------------------------------
The first version of this module failed on the box in two independent ways:
`collective_rpc` rejected a raw function (TypeError, not serializable), and
the runner-identity assertion below raised because the box was actually
running the V2 model runner, not V1 as this module assumed.

The second failure traces back to one root cause worth naming explicitly,
because it produced a SECOND bug too: `vllm/config/vllm.py`'s
`_is_default_v2_model_runner_model` was read up through its architecture
allowlist check and treated as if that check were the whole answer, without
reading down to its actual `return` statement -- which turns out to be
`is_default_v2_architecture or not model_config.is_moe`, a materially
different rule (see Q1). Reading a config-resolution function to a partial
conclusion instead of to its return statement is the specific failure mode.
It produced two bugs, not one: the wrong V1/V2 assumption itself, AND the
runner-identity check meant to catch a wrong assumption
(`type(runner).__name__ != "GPUModelRunnerV1"`) turned out to be broken by
construction -- both V1's and V2's runner classes are literally named
`GPUModelRunner` at their `class` statement (the "V1"/"V2" suffixes are local
import aliases inside `gpu_worker.py` only), so that check would raise
unconditionally regardless of which runner was actually active, for the wrong
reason. Neither bug was caught by re-reading harder; both were caught by the
box actually running the code. A source citation that stops before a
function's return statement is not a citation of what the function does.

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
QUESTIONS, ANSWERED FROM SOURCE (v0.28.0, everything below is read from
GitHub, not executed)
--------------------------------------------------------------------------

Q1. Entry point for a raw forward pass + logits; which model runner is
    actually active for Qwen3-0.6B; does it need hand-built input batch
    metadata?

    `Qwen3ForCausalLM.compute_logits(self, hidden_states) -> torch.Tensor | None`
    (vllm/model_executor/models/qwen3.py:330-334):

        def compute_logits(self, hidden_states):
            logits = self.logits_processor(self.lm_head, hidden_states)
            return logits

    This is shared code, identical regardless of which model runner drives
    it -- the runner is an orchestration layer around the same model object,
    not a different model. The model runner IS, however, load-bearing for
    which call sites exist and how many times they call `compute_logits`
    (design points 2-3 below), and it is CONFIG- AND MODEL-DEPENDENT which
    one is active. This module got it wrong once (see the section above) and
    now states the corrected trace in full, plus a runtime assertion instead
    of a silently trusted assumption.

    `gpu_worker.py:423-438` branches on `self.use_v2_model_runner`:

        if self.use_v2_model_runner:
            from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
            self.model_runner: GPUModelRunner = GPUModelRunnerV2(self.vllm_config, self.device)
        else:
            from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunnerV1
            self.model_runner = GPUModelRunnerV1(self.vllm_config, self.device)

    `use_v2_model_runner` (vllm/config/vllm.py:615-666) falls through, absent
    an env override or a speculative-decoding/PCP special case (none apply
    here), to `_is_default_v2_model_runner_model()`
    (vllm/config/vllm.py:691-711):

        architectures = getattr(model_config, "architectures", [])
        default_architectures = default_v2_model_runner_architectures()
        is_default_v2_architecture = any(arch in default_architectures for arch in architectures)
        if getattr(model_config, "is_hybrid", False) and not is_default_v2_architecture:
            return False
        if getattr(model_config, "is_attention_free", False):
            return False
        return is_default_v2_architecture or not model_config.is_moe

    `default_v2_model_runner_architectures()` (vllm/config/vllm.py:69-93) is
    `{"DeepseekV2ForCausalLM", "DeepseekV4ForCausalLM", "GraniteMoeForCausalLM",
    "InklingForCausalLM", "InklingForConditionalGeneration",
    "KimiK3ForConditionalGeneration", "LongcatFlashNgramForCausalLM",
    "Qwen2MoeForCausalLM"}` -- "Qwen3ForCausalLM" is not in it, so
    `is_default_v2_architecture` is False for Qwen3-0.6B. THE ALLOWLIST ONLY
    DECIDES THE OUTCOME FOR MOE MODELS, though: the function's actual return
    value is `is_default_v2_architecture or not model_config.is_moe`.
    Qwen3-0.6B is a plain dense model (`model_config.is_moe` is False), so
    `not model_config.is_moe` is True, and the whole expression is True --
    V2 IS THE DEFAULT for Qwen3-0.6B, precisely because it is NOT MoE, not
    despite it. The allowlist exists to let specific MoE architectures opt
    IN to V2 despite being MoE; it says nothing about the default for
    non-MoE models, which is V2 unconditionally (subject only to the
    `is_hybrid`/`is_attention_free`/`runner_type` guards above it, none of
    which Qwen3-0.6B trips). `use_v2_model_runner` then checks `HAS_TRITON`
    and `_get_v2_model_runner_unsupported_features()`
    (vllm/config/vllm.py:648-666) before finally returning True; nothing in
    bf16_floor.py's or this module's configuration (no LoRA, no speculative
    decoding, no unusual quantization) is expected to trip either of those,
    consistent with the engine log actually observed on the box: `Using V2
    Model Runner`.

    ASSERTED AT RUNTIME, not just reasoned about: `install_logits_hook`
    below checks `type(self.model_runner).__module__ ==
    "vllm.v1.worker.gpu.model_runner"` and raises if not. Module, not
    `__name__` -- both `vllm/v1/worker/gpu_model_runner.py` (V1) and
    `vllm/v1/worker/gpu/model_runner.py` (V2) declare `class GPUModelRunner`
    verbatim (gpu_model_runner.py:500, gpu/model_runner.py:158); `V1`/`V2`
    are local import aliases inside `gpu_worker.py` only and never appear in
    either class's real `__name__`. `__module__` is the only reliable
    discriminator available at runtime, and it is what a future run of this
    module on a config where V1 (or some future V3) is active will report in
    the resulting RuntimeError, rather than silently mismeasuring.

    `compute_logits` is already called from exactly this raw-forward-pass
    entry point inside the real V2 engine loop (vllm/v1/worker/gpu/model_runner.py,
    class `GPUModelRunner`), no synthetic construction needed:
      - gpu/model_runner.py:1340, inside `sample()` -- the main per-step
        path, over whichever positions `input_batch.logits_indices` selects
        (normally just the next token to sample).
      - gpu/model_runner.py:1764, passed BY REFERENCE (not called directly
        at that line) as the `logits_fn` argument to
        `self.prompt_logprobs_worker.compute_prompt_logprobs(self.model.compute_logits,
        hidden_states, input_batch, ...)` -- the prompt-logprobs path, see
        design points 2-3 and Q6 for what actually happens inside
        `compute_prompt_logprobs`.
      - gpu/model_runner.py:773, inside `_dummy_sampler_run` -- called ONLY
        during memory-profiling at `LLM()` construction time (its own
        comment: "During the initial memory profiling..."), before this
        hook is ever installed (design point 4 installs it via
        `collective_rpc` after `LLM(...)` returns). Harmless: any captures
        from this call site cannot exist in `self._collective_logits_captures`
        because that list doesn't exist yet when it runs, so there is
        nothing for it to pollute.

    Hand-building batch metadata (a `SchedulerOutput`, KV-cache block tables,
    attention-backend metadata) to call `execute_model()` standalone was
    considered and rejected: that state is deeply `Scheduler`-internal and
    not designed for standalone construction. Instead this design rides a
    REAL `LLM.generate()` call -- the real `Scheduler` builds all of that
    exactly as production does -- and intercepts the value at the point it
    is already computed, by monkeypatching `compute_logits` on the worker's
    model instance before calling `generate()`.

Q2. `collective_rpc` mechanics: what actually failed, and the supported fix.

    There are TWO IPC hops, not one, and the original version of this module
    only accounted for the second:

      Hop A: the `LLM`'s own process <-> a separate `EngineCore` process,
      over ZMQ, encoded with `MsgpackEncoder`/`MsgpackDecoder`
      (vllm/v1/serial_utils.py). `LLM.collective_rpc`
      (vllm/entrypoints/llm.py:567) -> `LLMEngine.collective_rpc`
      (vllm/v1/engine/llm_engine.py:421) -> `EngineCoreClient` ->
      `SyncMPClient.collective_rpc` (vllm/v1/engine/core_client.py:962-968),
      which is `self.call_utility("collective_rpc", method, timeout, args,
      kwargs)`. `call_utility` -> `_send_input` (core_client.py:884-890):

          msg = (self.core_engine, request_type.value, *self.encoder.encode(request))

      `self.encoder` is a `MsgpackEncoder` (core_client.py:632). THIS is the
      line the actual traceback cited (`core_client.py:890`).

      Hop B: the `EngineCore` process <-> each GPU worker process, over a
      shared-memory `MessageQueue`, using cloudpickle for a callable method
      or plain `getattr` for a string one (vllm/v1/executor/multiproc_executor.py,
      `worker_busy_loop`, :1026-1037). This is the hop the original version
      of this module analyzed and got right; it was never the problem.

    THE ACTUAL FAILURE IS AT HOP A. `MsgpackEncoder.enc_hook`
    (serial_utils.py:189-232) has explicit native handling for a fixed set
    of types -- `torch.Tensor` (`_encode_tensor`), `numpy.ndarray`, `slice`,
    `MultiModalKwargsItem(s)`, `UtilityResult` -- and ONLY for anything
    outside that set does it fall through to:

          if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
              raise TypeError(f"Object of type {type(obj)} is not serializable"
                               "Set VLLM_ALLOW_INSECURE_SERIALIZATION=1 to allow "
                               "fallback to pickle-based serialization.")

      A plain Python function is not in that set, so passing one as `method`
      raises exactly this, matching the observed error text precisely.
      `VLLM_ALLOW_INSECURE_SERIALIZATION` gates BOTH `MsgpackEncoder`
      (serial_utils.py:163, 214, 221-224) and `MsgpackDecoder`
      (serial_utils.py:337, 370-372, 477) -- it has to be set on whatever
      unpickles too, and it exists specifically to allow arbitrary
      pickle/cloudpickle blobs to cross this boundary, which is a remote
      code-execution surface if anything untrusted can reach it (hence
      "insecure" in the name and the log line `_log_insecure_serialization_warning`
      literally warning about it). NOT USED HERE, per instruction: the
      env var was never the fix for the actual problem, only a workaround
      for feeding it a shape (a raw function as `method`) it was never meant
      to carry.

      Return values are NOT the problem, and needed no change: `_encode_tensor`
      is one of the natively-handled types (serial_utils.py:257-273) --
      msgpack already carries CPU tensors (and lists/None of them, via
      ordinary recursive encoding) across Hop A with no insecure flag
      required. Only the `method` argument -- previously a raw function --
      needed to change shape.

    THE SUPPORTED MECHANISM: `worker_extension_cls`, an `EngineArgs`/
    `ParallelConfig` string field (a dotted import path) that vLLM was built
    for exactly this. `vllm/v1/worker/worker_base.py:265-291`:

          if parallel_config.worker_extension_cls:
              worker_extension_cls = resolve_obj_by_qualname(parallel_config.worker_extension_cls)
              if worker_extension_cls not in worker_class.__bases__:
                  for attr in dir(worker_extension_cls):
                      if attr.startswith("__"):
                          continue
                      assert not hasattr(worker_class, attr), (...)
                      ...
                  worker_class.__bases__ = worker_class.__bases__ + (worker_extension_cls,)

      The extension class is dynamically added as a BASE CLASS of `Worker`
      at construction time, in each worker process. Every method it defines
      becomes a real bound method on the worker instance -- dispatchable by
      the plain STRING path (`getattr(self.worker, method)`,
      multiproc_executor.py:1033), which is just a `str` and msgpack-encodes
      trivially at Hop A. There is a collision guard
      (`assert not hasattr(worker_class, attr)`) for every non-dunder
      extension attribute, so a name clash with an existing `Worker`/
      `WorkerBase` method fails loudly at `LLM()` construction, not
      silently. This module's hook methods are now defined on
      `LogitsHookWorkerExtension` below, and `LLM(...)` is passed
      `worker_extension_cls="weight_sync_bench.phase2.collective_logits.LogitsHookWorkerExtension"`
      in `_run_worker`. Calls are `llm.collective_rpc("install_logits_hook")`
      etc. -- string names throughout, both hops.

Q3. Under TP=2, does the worker already gather logits, or does each rank hold
    a vocab slice? all-gather-on-worker vs concatenate-on-driver?

    Already gathered, INSIDE `compute_logits` itself, and it is a gather to
    one rank, not an all-gather. This is `LogitsProcessor` code, unrelated to
    the model runner, so V1/V2 doesn't change it. `vllm/model_executor/layers/
    logits_processor.py:118-129` (`LogitsProcessor._gather_logits`):

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
    side (`retrieve_and_clear_logits`) picks the one non-None result out of
    the list `collective_rpc` returns.

Q4. How do return values cross the boundary? Must tensors move to CPU first?

    Yes, still, and this is unaffected by the worker_extension_cls change --
    Hop B (EngineCore <-> worker process) is unchanged. Results are pickled
    through `vllm.distributed.device_communicators.shm_broadcast.MessageQueue`,
    a shared-memory ring buffer (`max_chunk_bytes` defaults to `1024*1024*24`
    = 24 MiB, shm_broadcast.py:472). Its `enqueue()` (shm_broadcast.py:823-860)
    has a code path specifically for CPU tensors -- the comment there:
    "CPU tensors are routed through `_reduce_tensor` so that their bytes are
    emitted as out-of-band buffers instead of being copied into the pickle
    stream by torch's default reducer" -- and a payload exceeding the chunk
    size automatically overflows to a local ZeroMQ socket
    (`self.local_socket.send_multipart(...)`) rather than failing. GPU
    tensors get no such dedicated path in this code, and `collective_rpc`'s
    own docstring says: "It is recommended to use this API to only pass
    control messages, and set up data-plane communication to pass data."
    After crossing Hop B, the value also crosses Hop A (EngineCore -> `LLM`
    process) via `MsgpackEncoder`. CORRECTED IN THIS REVISION: an earlier
    version of this docstring claimed Hop A had "native, secure support for
    CPU tensors -- no additional handling needed there." That is true of
    `MsgpackEncoder.enc_hook` IN ISOLATION (`_encode_tensor` always fires for
    any `torch.Tensor`, insecure flag or not), but false of what actually
    happens to a `collective_rpc` return value once it is wrapped in
    `UtilityResult` for the trip back -- see Q7, found by running on the box
    after this claim turned out wrong. Both hops still point the same way on
    the ENCODE side regardless -- `.detach().cpu()` before returning from
    the worker extension method, done in `install_logits_hook`'s wrapper
    below -- but getting the tensor back out on the DECODE side needed the
    fix Q7 and design point 2a describe.

Q5. Does the CPU float32 cast in the hook happen after the bf16 computation,
    or does it change what dtype the computation itself runs in?

    After -- confirmed by reading the full call chain the hook wraps, from
    `compute_logits` down to the actual GEMM, with nothing in between that
    would force fp32. This is shared `LogitsProcessor`/model code, unrelated
    to the model runner, so V1/V2 doesn't change it either:
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
        representable in fp32) and matches what the V1 runner's
        `_get_prompt_logprobs_dict` (gpu_model_runner.py:5812-5813) already
        does in the same situation (`scores = logits.to(torch.float32)`
        under `logprobs_mode="raw_logits"`) -- V2's `PromptLogprobsWorker`
        does the analogous thing (Q6). This hook measures the same
        quantity, cast at the same point in the pipeline, on either runner.
      - One nearby method, `LogitsProcessor.get_top_tokens`
        (logits_processor.py:189-234), does upcast to float32 mid-computation
        ("Use float32 to avoid bf16 precision loss on large vocab indices")
        -- but that is a different method, an alternative argmax-only
        optimization path, and is not on `compute_logits`'s call graph, so it
        does not apply here. Noted only so a future reader who greps for
        `.float()` in this file does not mistake it for evidence against the
        above.

Q6. Why does the raw capture have `len(token_ids)` rows, and which one has
    to be dropped to match bf16_floor.py's `len(token_ids) - 1`?

    CORRECTED IN THIS REVISION. The previous version of this docstring
    concluded the hook's captured tensor already had `len(token_ids) - 1`
    rows, with position 0 dropped, by analogy with V1's structure (where the
    hidden-state slice passed to `compute_logits` was already trimmed before
    the call). Running on the box showed the actual captured shape for an
    8-token prompt is `[8, vocab]`, not `[7, vocab]` -- the conclusion about
    WHICH ROWS ARE SEMANTICALLY MEANINGFUL was right, but the reasoning about
    where the trim happens, and therefore whether the hook needs to do it
    itself, was wrong. This is a second instance of the failure named at the
    top of this docstring: reasoning by analogy from V1's structure instead
    of tracing V2's actual call sequence through to where the shape is
    decided. It was caught the same way the first instance was -- by running
    on the box, not by rereading harder -- and the fix is the same: read the
    full function, in order, to where it actually produces the value in
    question.

    The real mechanism, in `vllm/v1/worker/gpu/sample/prompt_logprob.py`,
    `PromptLogprobsWorker.compute_prompt_logprobs`:

        prompt_token_ids, prompt_logprobs, prompt_ranks = compute_prompt_logprobs_with_chunking(
            prompt_logprobs_token_ids,
            hidden_states[: input_batch.num_tokens],   # ALL scheduled positions, untrimmed
            logits_fn,                                  # our hook fires HERE, on all of them
            max_num_prompt_logprobs,
            self.logprobs_mode,
        )
        ...
        for i, req_id in enumerate(input_batch.req_ids):
            start_idx = query_start_loc_np[i]
            end_idx = query_start_loc_np[i + 1]
            if not req_is_prompt_chunked:
                end_idx -= 1                             # trim happens HERE, AFTER logits_fn returned
            logprobs = LogprobsTensors(
                logprob_token_ids=prompt_token_ids[start_idx:end_idx, :width], ...
            )

    `logits_fn` (this hook's wrapped `compute_logits`) is called on
    `hidden_states[: input_batch.num_tokens]` -- every scheduled position,
    0 through `len(token_ids) - 1`, with no exclusion. The trim to
    `len(token_ids) - 1` rows happens AFTERWARD, in Python, on the arrays
    `compute_prompt_logprobs_with_chunking` returned -- code the hook never
    sees, since it only intercepts `compute_logits`'s return value, not what
    the caller does with it next.

    The trimmed row is the LAST one, not the first. For an unchunked,
    complete request, the row at index `len(token_ids) - 1` (hidden state at
    the last prompt position) would need a target token at
    `num_computed_tokens + 1 + (len(token_ids) - 1)` -- one position past the
    end of the prompt, i.e. whatever the model goes on to sample, not a
    prompt token (`_prompt_logprobs_token_ids_kernel`'s own comment: "the
    logprob is computed for the next token"). That row is dropped
    (`end_idx -= 1`) because it has no valid prompt-token target, not
    because of any "no preceding context" reasoning about position 0 --
    position 0's row IS kept; it predicts the token at position 1.
    Semantically this lands on the same 7 rows bf16_floor.py's
    `output.prompt_logprobs[1:]` selects (predict-from-positions 0 through
    `len(token_ids) - 2`) -- the earlier conclusion about which rows matter
    was correct -- but getting there now requires an explicit `[:-1]` slice
    on the hook's raw capture (design point 3a / code below), not an
    assumption that the capture already arrives pre-trimmed.

Q7. Why can `collective_rpc` not carry a `torch.Tensor` back as a return
    value at all, and what does this module do instead?

    Diagnosed on the box: `retrieve_and_clear_logits` was returning a real
    `torch.Tensor`, but what `llm.collective_rpc(...)` handed back to the
    caller was `['float32', [8, 151936], 1]` -- `_encode_tensor`'s raw
    `(dtype, shape, aux_buffer_index)` triple, never reconstructed.

    The reconstruction step, `_decode_tensor` (serial_utils.py:398-419), is
    only ever invoked by msgspec's `dec_hook(self, t: type, obj: Any)`, and
    `dec_hook` only fires when msgspec is decoding into a STATICALLY TYPED
    position -- either the whole decoder was constructed with a known
    message type (`MsgpackDecoder(t=EngineCoreOutputs)`, used for the normal
    engine-output path, where `torch.Tensor`-typed fields are declared on
    real msgspec Structs), or a value is explicitly re-converted via
    `msgspec.convert(value, KnownType, dec_hook=...)`. `collective_rpc`'s
    return value gets neither. It travels wrapped in `UtilityResult`
    (`core_client.py:792`, `_process_utility_output` does
    `future.set_result(output.result.result)` where `output.result` is a
    `UtilityResult`). On encode, `enc_hook`'s `UtilityResult` branch
    (serial_utils.py:207-217):

        if isinstance(obj, UtilityResult):
            result = obj.result
            if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
                return None, result          # result_type discarded
            result_type_info = _encode_type_info_recursive(result)
            return result_type_info, result

    Without the insecure flag, `result_type` is `None` -- the information
    that would let the decoder know "this payload is `list[torch.Tensor |
    None]`" is thrown away before it ever crosses the wire. On decode,
    `_decode_utility_result` (serial_utils.py:365-379) checks that same
    `result_type`, finds `None`, and skips the only branch
    (`_decode_type_info_recursive` -> `_convert_result` ->
    `msgspec.convert(result, result_type, dec_hook=self.dec_hook)`) that
    would ever call `dec_hook` with `t=torch.Tensor` for this payload. The
    generic, untyped result comes back exactly as its raw native structure
    -- a 3-element list, because that is what `(str, list[int], int)`
    decodes to with no type annotation telling msgspec otherwise.

    The tensor's bytes are not lost -- `self.aux_buffers.append(tensor_data(obj))`
    happened on encode, and the resulting index (`1`) is sitting right there
    in the decoded triple. What is missing is purely the reconstruction call
    (`self.aux_buffers[1]` -> `torch.frombuffer(...).view(dtype).view(shape)`),
    which only `_decode_tensor` performs, and `_decode_tensor` never runs for
    an untyped `UtilityResult` without the insecure flag.

    THIS IS ALSO WHY IT HAS NEVER BEEN HIT UPSTREAM: every `collective_rpc`
    call site actually shipped in vLLM returns a plain msgpack-native type --
    `get_model_inspection` returns `str`, `is_sleeping`/`add_lora` return
    `bool`, `list_loras` returns `set[int]`, `save_sharded_state` returns
    `None`. None of them need `UtilityResult`'s type-directed reconstruction
    branch, so nobody upstream has needed to turn on the flag that gates it.

    NOT FIXED WITH `VLLM_ALLOW_INSECURE_SERIALIZATION`. It would work (it
    re-enables exactly the `_convert_result`/`dec_hook` branch above), but
    `_convert_result` does `importlib.import_module(mod_name);
    getattr(mod, name)` on a module/attribute name carried in the message
    and converts arbitrary structure into whatever that names -- a
    materially different, narrower risk than unpickling a function (Q2), but
    gated behind the identical global switch, and the instruction not to use
    it stands.

    FIX: reconstruct the tensor by hand, using only always-native msgpack
    types -- `bytes` needs no type annotation and no insecure flag (it takes
    the same `CUSTOM_TYPE_RAW_VIEW` ext-type path small inline tensors
    already use, and `ext_hook`'s handling of that code
    (serial_utils.py:477-478, `if code == CUSTOM_TYPE_RAW_VIEW: return data`)
    has no `VLLM_ALLOW_INSECURE_SERIALIZATION` guard at all). Do the encoding
    and decoding vLLM's own `_encode_tensor`/`_decode_tensor` would have done,
    explicitly, at the call site: `retrieve_and_clear_logits` returns
    `(dtype_str, shape_list, raw_bytes)` -- a tuple of a `str`, a `list[int]`,
    and `bytes`, all natively msgpack-safe -- and `run_one_prompt`
    reconstructs with `torch.frombuffer(raw_bytes, dtype=getattr(torch,
    dtype_str)).view(shape).clone()`. `.clone()` because `frombuffer` returns
    a tensor backed by the (soon to be garbage-collected) decoded bytes
    object, same reasoning `_decode_tensor` itself applies to its own
    non-aux buffers (serial_utils.py:414, "Clone ensures tensor is backed by
    pytorch-owned memory").

    SIZE, so far unexercised: at this repo's current parameters the payload
    is small -- ~4.9MB fp32 at `seq_len=8`, ~39MB at `seq_len=64` (one
    request's tensor at a time, not batched, per bf16_floor.py's one-prompt-
    at-a-time discipline). Comfortable for a single ZMQ message either way.
    If a later sweep pushes `seq_len` far enough that a single collective_rpc
    payload becomes unwieldy, reconsider a temp-file handoff (worker writes,
    returns a path -- a plain `str`, trivially safe over this same
    transport) instead of growing the in-memory byte buffer further; this is
    the same `seq_len`-scaling boundary design point 3b already names for
    the chunked-prefill no-op finding, and both should be re-examined
    together if `seq_len` grows substantially.

--------------------------------------------------------------------------
DESIGN
--------------------------------------------------------------------------
1. `LogitsHookWorkerExtension` -- a plain class, defined at module level so
   it has a stable dotted import path (Q2's `worker_extension_cls`
   requirement). Injected as a base class of `Worker` at `LLM()` construction
   time; its methods run with `self` bound to the actual worker instance, so
   they read exactly like the original design's `worker`-parameter callables
   did, just via inheritance instead of `functools.partial`.

2. `install_logits_hook(self)` -- asserts the V2 runner is active (Q1),
   then wraps `self.model_runner.model.compute_logits` with a closure that
   calls the original, and if the result is not None (i.e. this is rank 0,
   or TP=1), clones it to CPU float32 (Q4, Q5) and appends it to a list
   stashed on `self` (`self._collective_logits_captures`). Returns the
   original output unchanged, so normal engine behavior (sampling, the
   existing prompt_logprobs machinery) is untouched -- this hook only
   *observes*. The list stores real `torch.Tensor` objects, not the wire
   format -- the byte-buffer conversion (Q7) happens only at
   `retrieve_and_clear_logits`, the point where a value actually has to
   cross back over `collective_rpc`.

3. Drive a real `LLM.generate()` call exactly as bf16_floor.py does, EXCEPT
   `prompt_logprobs=1` instead of `prompt_logprobs=-1`. The code path that
   computes logits at every prompt position is gated by
   `sampling_params.prompt_logprobs is not None` -- not by its magnitude.
   Everything downstream that scales with vocab_size (the top-k call inside
   `Sampler.gather_logprobs`, the `LogprobsTensors` CPU buffers, and both
   Python-object-construction loops described in PROBLEM above) scales with
   the REQUESTED count instead once the hook has already captured the full
   untruncated tensor straight off `compute_logits`. `prompt_logprobs=1`
   keeps that whole downstream machinery (which we no longer need, and don't
   disable) cheap rather than trying to disable it.

3a. `retrieve_and_clear_logits(self, expected_num_tokens)` -- a second
   collective_rpc call, called after `generate()` returns. Reads and clears
   `self._collective_logits_captures`. A single `generate()` call with
   `max_tokens=1` calls `compute_logits` twice per request under V2's real
   call graph (Q1: once via `sample()` for the sampled token, once via
   `compute_prompt_logprobs`'s `logits_fn` for the prompt, on ALL
   `expected_num_tokens` positions untrimmed -- Q6), so with chunked prefill
   and prefix caching both disabled (design point 3b below) exactly one
   captured tensor should have `shape[0] > 1` (the prompt-logprobs one, and
   its `shape[0]` should be EXACTLY `expected_num_tokens`, not merely `>=`
   it) and exactly one should have `shape[0] == 1` (the sampled token).
   **Both are asserted exactly, not bounded loosely and not heuristically
   selected**: if more than one captured tensor has `shape[0] > 1`,
   retrieval raises with every such shape listed; if the one multi-position
   capture's row count isn't exactly `expected_num_tokens`, retrieval raises
   naming both numbers. An earlier version of this check used `shape[0] <
   expected_min_positions` -- a `>=` bound -- against a name
   (`expected_min_positions`) that was actually `len(token_ids) - 1`,
   already-trimmed. That bound is exactly the class of failure this whole
   repository exists to catch elsewhere: a loose tolerance in the
   instrumentation passed silently on `[8, vocab]` when `7` was expected
   (`8 >= 7`), the same shape as a genuinely correct capture would have
   looked wrong by, so instrumentation meant to catch a shape mismatch
   cannot itself use a loose bound to check one. `retrieve_and_clear_logits`
   now asserts `best.shape[0] == expected_num_tokens` (the raw, untrimmed
   count) and returns `best[:-1]` (Q6's `[:-1]` slice, applied here so
   `run_one_prompt` never sees the untrimmed row), which is separately
   checked to be exactly `expected_num_tokens - 1` before it is converted to
   the wire tuple (Q7) and returned.

   THIS ASSERTION IS ALSO WHAT CATCHES A SECOND, DISTINCT CHUNKING HAZARD,
   independent of `enable_chunked_prefill`: `compute_prompt_logprobs`'s
   actual logits computation happens inside `compute_prompt_logprobs_with_chunking`
   (prompt_logprob.py), which has its OWN internal chunk loop --

       CHUNK_SIZE = 1024
       for start_idx in range(0, prompt_token_ids.shape[0], CHUNK_SIZE):
           ...
           prompt_logits = logits_fn(prompt_hidden_states[start_idx:end_idx])

   -- sized in PROMPT-LOGPROB POSITIONS, not scheduler token-budget units,
   and NOT read from or controlled by `scheduler_config.enable_chunked_prefill`
   at all; it is a fixed memory-management chunk size for materializing
   full-vocab logits, unconditional on the scheduler flag. For any prompt
   whose prompt-logprobs computation exceeds 1024 positions, `logits_fn`
   (i.e. `compute_logits`, i.e. this hook) fires more than once for the
   prompt tensor, and `retrieve_and_clear_logits` would see more than one
   `shape[0] > 1` capture and raise, exactly as it does for the
   scheduler-level chunking case -- the same assertion covers both hazards
   without needing to know which one caused it. At this repo's current sweep
   parameters (`seq_len` up to 64), 1024 is never approached, so this has
   not been observed to trigger; it would first become relevant if `seq_len`
   grows past roughly 1025.

3b. **Chunked prefill and prefix caching are explicitly disabled** in the
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
     - Chunked prefill (the SCHEDULER-level mechanism, distinct from design
       point 3a's CHUNK_SIZE=1024 hazard) splits one long prefill across
       multiple forward-pass STEPS, so `compute_logits` fires once per
       scheduling chunk instead of once for the whole prompt. Retrieval
       would then see several `shape[0] > 1` captures, none of which is the
       full tensor -- the same 3a assertion catches this too.
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

   THE WARNING THIS TRIGGERS, READ FROM SOURCE, DOES NOT CHANGE THE ABOVE.
   Setting `enable_chunked_prefill=False` here logs (arg_utils.py:2657-2666):
   "This model does not officially support disabling chunked prefill.
   Disabling this manually may cause the engine to crash or produce
   incorrect outputs." This fires for essentially any "generate" model with
   `enable_chunked_prefill` manually set False while the model's own default
   (`is_chunked_prefill_supported`) is True -- it is a blanket
   deviation-from-default notice, not a model-specific diagnostic. Every
   source location that actually branches on `enable_chunked_prefill` was
   checked for a concrete mechanism:
     - `vllm/v1/core/sched/scheduler.py:956-964` -- the real effect for a
       plain dense model: a request whose prefill exceeds the step's token
       budget is deferred (not scheduled this step) rather than split.
       Scheduling delay, not corruption.
     - `vllm/model_executor/models/config.py:611-622` -- a hard assert tied
       to Mamba `cache_config.mamba_cache_mode == "align"`. Qwen3-0.6B has
       no Mamba layers; does not apply.
     - `vllm/v1/engine/core.py:265-277` -- the engine itself force-disables
       chunked prefill AND prefix caching for models with non-causal
       attention layers, its own comment stating both "would corrupt
       non-causal prefill." Qwen3 is fully causal; this is evidence FOR
       causal models being safe (it is the non-causal case being protected
       against), not against.
     - `vllm/v1/attention/backends/turboquant_attn.py:278-283` -- workspace
       memory sizing for the TurboQuant quantized-attention backend, not the
       plain bf16/eager path this repo uses.
   No source-confirmed corruption mechanism was found for a causal,
   non-Mamba, non-quantized-attention bf16 dense model. Separately: on the
   A100 box, `EngineArgs`' `LLM_CLASS`-context default `max_num_batched_tokens`
   is 8192 (arg_utils.py:2602-2611, A100 explicitly routed to the
   non-H100/H200 branch). Every prompt this repo submits (`seq_len` up to
   64, one request at a time) is trivially under that budget, meaning
   scheduler-level chunking would never fire even with the flag left on.
   **Disabling it is a provable no-op at current parameters, not a live
   correctness risk being traded for a live scheduling hazard.** THIS STOPS
   HOLDING if a later sweep pushes `seq_len` toward the thousands -- at that
   point re-examine whether the token budget is still comfortably clear
   before continuing to disable the flag.

4. `uninstall_logits_hook(self)` -- restores the original `compute_logits`,
   called after every retrieval so the hook does not leak into whatever the
   box does next (e.g. SPEC.md 2d's weight-transfer work, which also touches
   these worker processes).

5. `run_one_prompt(llm, token_ids) -> torch.Tensor` -- ties 2-4 together for
   one prompt: install, generate, retrieve, uninstall. Reconstructs the
   tensor from the `(dtype_str, shape_list, raw_bytes)` triple
   `retrieve_and_clear_logits` returns (Q7) via
   `torch.frombuffer(raw_bytes, dtype=...).view(shape).clone()`. Returns a
   `[len(token_ids) - 1, vocab]` float32 CPU tensor -- the last row dropped,
   not the first, per Q6 -- already sliced by `retrieve_and_clear_logits`
   before it crossed the wire.

6. `_run_worker` / `measure_differential_floor` -- structurally identical to
   bf16_floor.py's functions of the same name (same one-prompt-at-a-time
   submission discipline and its rationale, same TP1-vs-TP2
   subprocess-per-degree pattern, same torch.save/torch.load handoff to the
   parent process), with `run_one_prompt` substituted for the
   prompt_logprobs-and-Python-loop extraction. The LLM constructor args are
   the same EXCEPT `enable_chunked_prefill=False, enable_prefix_caching=False`
   (design point 3b) and `worker_extension_cls=...` (Q2, new in this
   revision) -- bf16_floor.py's own `LLM(...)` call leaves the first two at
   their (on) defaults and has no need of the third; see the SMOKE TEST
   section below for how to hold the computation identical across both
   modules despite it.

--------------------------------------------------------------------------
WHAT THIS DOES NOT ADDRESS -- flagged, not solved
--------------------------------------------------------------------------
- `torch.compile` / CUDA graph interaction: bf16_floor.py's LLM construction
  already passes `enforce_eager=True`, which should mean `compute_logits` is
  called as a plain Python method (not captured inside a compiled graph or a
  CUDA-graph replay region) -- but this was not verified by tracing the
  compile/capture boundary in the V2 runner, and cudagraph interaction is
  exactly the kind of thing that silently breaks a monkeypatch. If this
  module is adapted to run without `enforce_eager=True`, re-verify that
  `compute_logits` is still a live Python call at the point the hook patches
  it.
- Monkeypatching an instance attribute (`model.compute_logits = hooked`)
  shadows the class method via normal Python instance-`__dict__` lookup, a
  standard technique -- but was not verified against whatever `nn.Module`
  machinery (hooks, `__setattr__` overrides) vLLM's model base classes might
  define that could interfere with a plain attribute assignment.
- No timing comparison exists yet between this path and bf16_floor.py's,
  since neither has been run to completion. That comparison is the actual
  point of writing this as a second, side-by-side module rather than editing
  the original.
- Disabling chunked prefill and prefix caching (design point 3b) changes the
  configuration bf16_floor.py's own measurement was taken under (which runs
  with both left at their defaults, i.e. on). The two paths are therefore
  not a pure extraction-method A/B test as configured -- see the smoke test
  below for what IS held constant across them, and
  `tolerance/phase2a_bf16_floor.json` for the resulting prefix-caching
  caveat on the committed floor.
- Whether `worker_extension_cls`'s dynamic `__bases__` mutation
  (worker_base.py:283-285) interacts correctly with repeated `LLM()`
  construction within one long-lived worker PROCESS (e.g. across the two
  `_spawn_worker` subprocess invocations this module makes for TP=1 then
  TP=2 -- each is its own fresh process, so this should not matter here, but
  was not traced for the general case of constructing more than one `LLM`
  per process).

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
        worker_extension_cls=collective_logits.WORKER_EXTENSION_QUALNAME,
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
variable; `worker_extension_cls` does not affect computation, only which
worker methods exist, so it is safe to add even though bf16_floor.py has no
equivalent). `torch.equal`, not `torch.allclose`: if this ever needs relaxing
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
# Worker extension (Q2). Injected as a base class of Worker by vLLM itself
# (worker_base.py:265-291) when `worker_extension_cls` names this class's
# dotted path. Methods run with `self` bound to the actual Worker instance.
# Everything that touches torch/vllm internals is imported lazily inside
# method bodies, matching bf16_floor.py's convention, so that
# `import weight_sync_bench.phase2.collective_logits` for its constants does
# not require the phase2 extra to be installed.
# --------------------------------------------------------------------------- #

WORKER_EXTENSION_QUALNAME = (
    "weight_sync_bench.phase2.collective_logits.LogitsHookWorkerExtension"
)

_EXPECTED_RUNNER_MODULE = "vllm.v1.worker.gpu.model_runner"


class LogitsHookWorkerExtension:
    """Mixed into `Worker` via `worker_extension_cls` (Q2). Do not
    instantiate directly -- vLLM adds this as a base class of the real
    Worker class in each worker process; `self` in every method below is
    the Worker instance, with normal access to `self.model_runner` etc.
    """

    def install_logits_hook(self) -> None:
        """See Q1 (runner check)/Q3 (why only rank 0 captures anything)/
        design point 2 for what this wraps and why. Idempotent per worker
        process: re-installing after `uninstall_logits_hook` is safe;
        installing twice without uninstalling raises rather than silently
        double-wrapping.
        """
        runner = self.model_runner
        runner_module = type(runner).__module__
        if runner_module != _EXPECTED_RUNNER_MODULE:
            raise RuntimeError(
                f"expected the V2 model runner (module "
                f"{_EXPECTED_RUNNER_MODULE!r}, verified as the default for "
                f"Qwen3ForCausalLM at vLLM v0.28.0 -- see the module "
                f"docstring's Q1 answer), got module {runner_module!r}. "
                "VLLM_USE_V2_MODEL_RUNNER may be set to 0, or a different "
                "model/version combination is in use that this hook was "
                "not verified against."
            )

        model = runner.model
        if getattr(model, "_collective_logits_hook_installed", False):
            raise RuntimeError(
                "hook already installed on this worker; call "
                "uninstall_logits_hook before installing again"
            )

        original_compute_logits = model.compute_logits

        def hooked_compute_logits(hidden_states, *args, **kwargs):
            import torch

            out = original_compute_logits(hidden_states, *args, **kwargs)
            # Non-None only on the rank `tensor_model_parallel_gather` gathered
            # to (Q3) -- rank 0 on CUDA. Cast to float32 post-hoc (Q5), on an
            # already-bf16-computed tensor, matching what the engine's own
            # prompt_logprobs path does under logprobs_mode="raw_logits" so
            # this hook measures the same quantity bf16_floor.py's does.
            if out is not None:
                self._collective_logits_captures.append(
                    out.detach().to(dtype=torch.float32, device="cpu")
                )
            return out

        self._collective_logits_captures = []
        self._collective_logits_original_compute_logits = original_compute_logits
        model.compute_logits = hooked_compute_logits
        model._collective_logits_hook_installed = True

    def retrieve_and_clear_logits(
        self, expected_num_tokens: int
    ) -> tuple[str, list[int], bytes] | None:
        """Returns the captured prompt-logprobs tensor since the last call
        (or since install), and clears the buffer -- as a
        `(dtype_str, shape_list, raw_bytes)` triple, NOT a `torch.Tensor`.
        `collective_rpc` cannot carry a `torch.Tensor` back as a return
        value at this vLLM version (Q7); this triple uses only
        msgpack-native types (`str`, `list[int]`, `bytes`) so it round-trips
        intact, and `run_one_prompt` reconstructs the tensor on the other
        side. Returns None on every rank whose `compute_logits` never
        produced a non-None result (Q3) -- i.e. every rank except the one
        `tensor_model_parallel_gather` gathered to.

        `expected_num_tokens` is `len(token_ids)` -- the RAW, UNTRIMMED
        position count `compute_logits` is actually called with (Q6), not
        the `len(token_ids) - 1` comparable-row count bf16_floor.py's
        `output.prompt_logprobs[1:]` produces. This function does the
        `[:-1]` trim (Q6) itself before returning, so callers never see the
        untrimmed row.

        With chunked prefill and prefix caching disabled (see `_run_worker`'s
        `LLM(...)` call and design point 3b), a single `generate()` call
        with `max_tokens=1` should produce exactly one capture with
        `shape[0] > 1` (the prompt-logprobs tensor, and its `shape[0]`
        should be EXACTLY `expected_num_tokens`) and exactly one with
        `shape[0] == 1` (the sampled token) -- see Q1. Both are ASSERTED
        EXACTLY, not bounded loosely and not heuristically selected: a
        loose `>=` bound here would be exactly the kind of instrumentation
        failure this repository exists to catch elsewhere in what it
        measures -- see design point 3a for the concrete case this caught.
        The multiple-capture check also doubles as the catch for design
        point 3a's independent CHUNK_SIZE=1024 hazard: if either
        scheduler-level chunking or the fixed-size prompt-logprobs chunk
        loop fires despite the constructor flags, more than one
        multi-position capture shows up here, and that is exactly the
        failure this function exists to surface loudly instead of quietly
        returning a partial chunk.
        """
        captures = getattr(self, "_collective_logits_captures", [])
        self._collective_logits_captures = []
        if not captures:
            return None

        candidates = [t for t in captures if t.shape[0] > 1]
        if len(candidates) > 1:
            shapes = [tuple(t.shape) for t in candidates]
            raise RuntimeError(
                f"expected at most one compute_logits capture with more than "
                f"one position (chunked prefill and prefix caching should "
                f"both be disabled, and prompt length should be under "
                f"compute_prompt_logprobs_with_chunking's CHUNK_SIZE=1024 -- "
                f"see design points 3a/3b), got {len(candidates)}: shapes "
                f"{shapes}. Either the constructor flags did not take "
                "effect, the prompt exceeded 1024 positions, or an "
                "unexpected extra compute_logits call happened."
            )
        if not candidates:
            raise RuntimeError(
                f"expected one compute_logits capture with shape[0] == "
                f"{expected_num_tokens} (the untrimmed prompt-logprobs "
                f"tensor -- Q6), but every capture had exactly 1 position: "
                f"{[tuple(t.shape) for t in captures]}. The prompt-logprobs "
                "call site (compute_prompt_logprobs's logits_fn) never fired."
            )

        best = candidates[0]
        if best.shape[0] != expected_num_tokens:
            raise RuntimeError(
                f"captured prompt-logprobs tensor has {best.shape[0]} rows, "
                f"expected EXACTLY {expected_num_tokens} (Q6: compute_logits "
                "is called on every scheduled position, untrimmed -- not "
                "already reduced by one) -- the compute_logits call graph "
                "for this request does not match what this hook assumed "
                "(see the module docstring's 'what this does not address')"
            )

        trimmed = best[:-1].contiguous()  # drop the LAST row, not the first (Q6)
        if trimmed.shape[0] != expected_num_tokens - 1:
            raise RuntimeError(
                f"internal error: trimmed shape {tuple(trimmed.shape)} does "
                f"not have {expected_num_tokens - 1} rows"
            )

        # Wire format: msgpack-native types only (Q7). dtype string matches
        # vLLM's own _encode_tensor convention (str(dtype).removeprefix("torch.")).
        dtype_str = str(trimmed.dtype).removeprefix("torch.")
        raw_bytes = trimmed.numpy().tobytes()
        return dtype_str, list(trimmed.shape), raw_bytes

    def uninstall_logits_hook(self) -> None:
        """Restores the original compute_logits so the hook does not leak
        into whatever the box does with this worker next.
        """
        runner = self.model_runner
        model = runner.model
        original = getattr(self, "_collective_logits_original_compute_logits", None)
        if original is None:
            return
        model.compute_logits = original
        model._collective_logits_hook_installed = False
        self._collective_logits_original_compute_logits = None
        self._collective_logits_captures = []


# --------------------------------------------------------------------------- #
# Driver-side orchestration. Mirrors bf16_floor.py's _run_worker /
# measure_differential_floor structure and rationale (one prompt at a time,
# one subprocess per TP degree) with the extraction method swapped.
# --------------------------------------------------------------------------- #


def run_one_prompt(llm: Any, token_ids: list[int]) -> "torch.Tensor":
    """Runs one teacher-forced prompt through `llm` and returns a
    [len(token_ids) - 1, vocab] float32 CPU tensor of raw logits -- the last
    row dropped, not the first, per Q6. Requires `llm` to have been
    constructed with `worker_extension_cls=WORKER_EXTENSION_QUALNAME`.
    """
    import torch
    from vllm import SamplingParams

    llm.collective_rpc("install_logits_hook")
    try:
        sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
        llm.generate([{"prompt_token_ids": token_ids}], sp, use_tqdm=False)

        # The RAW, untrimmed position count compute_logits is called with
        # (Q6) -- not len(token_ids) - 1. retrieve_and_clear_logits asserts
        # against this exact number and does the [:-1] trim itself.
        expected_num_tokens = len(token_ids)
        results = llm.collective_rpc(
            "retrieve_and_clear_logits", args=(expected_num_tokens,)
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
        # collective_rpc cannot carry a torch.Tensor back as a return value
        # at this vLLM version (Q7) -- reconstruct from the wire triple.
        dtype_str, shape, raw_bytes = non_none[0]
        return torch.frombuffer(raw_bytes, dtype=getattr(torch, dtype_str)).view(shape).clone()
    finally:
        llm.collective_rpc("uninstall_logits_hook")


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
    collective_rpc worker extension instead of prompt_logprobs=-1, and
    torch.saves them (a list of [batch, seq_len - 1, vocab] float32 tensors)
    to `out_path`.
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
        # point 3b) and both are silent failure modes for this extraction
        # method: chunked prefill makes compute_logits fire once per
        # scheduling chunk instead of once for the whole prompt (retrieval
        # would see several multi-position captures instead of one --
        # LogitsHookWorkerExtension.retrieve_and_clear_logits's assertion),
        # and prefix caching can skip recomputation for a repeated
        # prompt/prefix entirely, which would make the repetitions this
        # floor is averaged over non-independent. Neither failure raises or
        # changes a tensor shape on its own, so both are disabled explicitly
        # rather than left at their defaults. See design point 3b for why
        # the resulting "does not officially support disabling chunked
        # prefill" warning does not indicate a correctness risk here, and
        # why it is a no-op at this module's current seq_len.
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        # Q2: the supported way to add worker-side methods reachable by
        # collective_rpc's string dispatch path, without needing
        # VLLM_ALLOW_INSECURE_SERIALIZATION.
        worker_extension_cls=WORKER_EXTENSION_QUALNAME,
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
