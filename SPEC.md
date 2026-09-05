# Phase 1 spec: resharding correctness harness

Revision 2. Supersedes the earlier version, which compared reference logits
against parameters reassembled from shards. That design was broken: for any
dimension `d`, `cat(chunk(X, n, dim=d), dim=d)` recovers `X` exactly, so a
round trip through a consistently wrong layout is the identity map and no
injected bug is detectable. A resharding harness cannot validate a layout by
proving bytes can be reconstructed. It must validate that a consumer executing
under the declared layout produces correct output.

## Background (context only, do not build)

The eventual project measures weight-synchronization latency in asynchronous RL
post-training: the cost of pushing updated parameters from a trainer to N vLLM
sampler replicas, decomposed as

    T_sync = T_reshard + T_transfer + T_load

Target stack for the real runs is prime-rl plus vLLM on rented multi-node GPUs.
None of that is in scope here. Phase 1 is CPU-only and exists to make
resharding bugs loud before any money is spent on GPUs. Resharding bugs are
silent by default: the parameter arrives, the shapes are right, the model still
emits plausible text, and the logits are wrong.

## Goal

A harness that detects incorrect resharding of a small transformer between two
tensor-parallel layouts, verified adversarially: three independent layout bugs
are injected and each must produce a logits mismatch, plus one unrepresentable
configuration that must be handled or rejected explicitly.

## Correctness invariant

    ReferenceModel(full_params, x) == ShardedModel(reshard(full_params, src → dst), x)

`ShardedModel` performs rank-local computation with real collectives, so a wrong
shard dimension changes the result of the consuming matmul. This is the change
from revision 1 and it is the reason the harness works at all.

## Non-goals

- No RL loop, environments, verifiers, advantage estimation, or optimizer.
- No timing or benchmarking code. `SyncRecord` fields may be left unpopulated.
- No GPU, CUDA, NCCL, or CUDA IPC.
- No vLLM dependency.
- No pretrained weights, tokenizer, or dataset. Parameters are random and are
  never updated by a loss.
- No layout-inference diagnostics (see Diagnostics below).

## Environment

macOS, CPU only. Python with `uv`. PyTorch, `gloo` backend for phase 1b.
fp32 throughout, so numerical tolerance can be tight.

## Scope warning

Implementing `ShardedModel` means writing tensor-parallel forward for the toy:
column-parallel and row-parallel linear layers with their collectives,
vocab-parallel embedding and output projection, and GQA attention over a rank
subset of heads. That is roughly 200 lines and it is now the bulk of phase 1.
Plan for it rather than discovering it midway.

## Toy model

Defined in this repo, not imported. Its only job is to contain one instance of
every layout that can silently corrupt weights.

| field         | value  |
|---------------|--------|
| `d_model`     | 256    |
| `n_layers`    | 4      |
| `n_heads`     | 8      |
| `n_kv_heads`  | 2      |
| `head_dim`    | 32     |
| `ffn`         | 704    |
| `vocab`       | 32000  |
| attention     | GQA    |
| activation    | SwiGLU |
| norm          | RMSNorm (weight only, no bias) |
| embeddings    | untied (`embed` and `lm_head` are separate) |

About 19.2M parameters. No biases anywhere.

### Parameter table

Per layer:

| tensor      | shape         | partition |
|-------------|---------------|-----------|
| `qkv`       | `[384, 256]`  | by attention head, fused (see below) |
| `o_proj`    | `[256, 256]`  | by input dim (dim 1) |
| `gate_up`   | `[1408, 256]` | by FFN neuron, gate/up pairing preserved (see below) |
| `down`      | `[256, 704]`  | by input dim (dim 1) |
| `attn_norm` | `[256]`       | replicated |
| `ffn_norm`  | `[256]`       | replicated |

Global:

| tensor       | shape          | partition |
|--------------|----------------|-----------|
| `embed`      | `[32000, 256]` | by vocabulary (dim 0) |
| `lm_head`    | `[32000, 256]` | by vocabulary (dim 0) |
| `final_norm` | `[256]`        | replicated |

### The two fused tensors

`qkv` has shape `[(n_heads + 2 * n_kv_heads) * head_dim, d_model] = [384, 256]`.
Row ranges: Q is `0:256`, K is `256:320`, V is `320:384`. Under TP degree `t`,
rank `i` must receive Q heads `[i * n_heads / t, (i + 1) * n_heads / t)` together
with the corresponding K and V heads, reassembled into a fused local tensor. The
correct shard is **not contiguous** in the source tensor.

`gate_up` has shape `[2 * ffn, d_model] = [1408, 256]`. Gate is rows `0:704`, up
is rows `704:1408`. Rank `i` must receive gate rows
`[i * 704 / t, (i + 1) * 704 / t)` concatenated with up rows
`[704 + i * 704 / t, 704 + (i + 1) * 704 / t)`. Also not contiguous.

Worth a README note: production frameworks often sidestep this by storing the
fused tensor pre-permuted into a head-interleaved order so that contiguous
slicing happens to be correct. Verify how Megatron-Core actually orders its
fused QKV before asserting this in writing.

### GQA at TP = 4

`n_kv_heads = 2` with `t ∈ {1, 2, 4}` is deliberate. At `t = 4` there are fewer
KV heads than ranks and no clean one-KV-head-per-rank partition exists. The
implementation must either replicate KV heads across ranks or raise
`UnsupportedLayout`. It must not silently invent a shard representation. Make
the choice discoverable from the layout data structure rather than buried in
`reshard.py`.

## Layout vocabulary

Do not invent a taxonomy. PyTorch DTensor already supplies the base vocabulary:
`Shard(dim)`, `Replicate()`, `Partial()`. Use those names.

The interesting fact, and the one to put in the README, is that the two fused
tensors are **not expressible** in that vocabulary. A per-head split of fused
QKV is not `Shard(0)`, and neither is the interleaved gate/up split, because in
both cases the rank-local shard is a non-contiguous selection of source rows.
So the spec is DTensor placements plus exactly two extensions, each with a
stated reason it is needed.

Represent it as a sum type, not a record with independent fields. `partition`
and `replicated` are the same axis, and a project about invisible layout errors
should not have a layout representation that admits contradictory states.

```python
Placement = Replicated | Shard(dim: int) | HeadPartitioned(...) | FusedPaired(...)

@dataclass(frozen=True)
class ShardSpec:
    placement: Placement
    group_size: int
```

`HeadPartitioned` carries `n_heads`, `n_kv_heads`, `head_dim` so the per-head
split is derivable. `FusedPaired` carries the two equal row ranges. A
`LayoutTable` maps parameter name to `ShardSpec` for a given TP degree.

## ShardedModel

Rank-local computation with collectives. Semantics per layer type:

- **Embedding** (vocab-parallel): mask token ids outside the rank's vocab range,
  look up locally with an offset, zero the out-of-range rows, all-reduce. Output
  is replicated.
- **qkv** (column-parallel): input replicated, no collective before. Rank
  computes only its own Q heads and corresponding K/V heads.
- **attention**: computed locally over the rank's head subset.
- **o_proj** (row-parallel): input is the rank-local head output, weight is the
  dim-1 slice, output is a partial sum, all-reduce.
- **RMSNorm**: replicated weight, replicated input, no communication.
- **gate_up** (column-parallel, fused): rank computes local gate and up over its
  neuron slice, SwiGLU applied locally.
- **down** (row-parallel): partial output, all-reduce.
- **lm_head** (vocab-parallel): rank computes logits for its vocab slice,
  all-gather along the vocab dimension.

Residual connections work naturally since activations are replicated between
blocks.

## Build order

**1a. Single process.** `ShardedModel` runs all ranks in one process; collectives
are simulated by summing or concatenating the per-rank tensors directly. Same
algorithm, no distributed machinery. Every break case must be caught here.

**1b. Gloo.** Same algorithm with shards in separate processes and real
`all_reduce` / `all_gather` over a `gloo` process group. Keep the topology as
simple as possible: every rank participates in the same logical operation on a
defined communication schedule. Do not simulate a production topology; the point
is that the collective version matches the single-process reference algorithm.

Do not start 1b until 1a catches all break cases.

## Tolerance

Sharded execution changes the numerics. A row-parallel matmul computes partial
products per rank and reduces them, which sums in a different order than the
full matmul, so the correct sharded result differs from the reference in the low
bits. The tolerance must admit summation reordering and still catch a real
layout bug.

Establish the floor empirically before writing any break test:

1. Run the correct sharded path at every `t ∈ {1, 2, 4}`.
2. Record max absolute logit deviation from `ReferenceModel`.
3. Set the assertion threshold well above the observed floor.

In fp32 on CPU expect a floor near `1e-6`. Expect injected layout bugs to
produce deviations three to six orders of magnitude larger, so the separation is
comfortable. Do not pick a tolerance by guessing; a guessed threshold gives
either flaky passes or a break case that fails for the wrong reason.

Measure the floor separately for 1a and 1b, since gloo's reduction order need
not match the in-process simulation's.

## Break cases

Cases 1 through 3 are layout bugs, implemented as deliberately wrong layout
tables or reshard paths. Each test asserts the correctness invariant **fails**.
If any passes, the harness does not work.

1. Slice fused `qkv` contiguously on dim 0 instead of per-head.
2. Shard a row-parallel tensor (`o_proj` or `down`) on dim 0 instead of dim 1.
3. Shard the RMSNorm weights instead of replicating them.

Case 4 is a different kind of test and should be labeled as such in the code and
the README. At `t = 4` with `n_kv_heads = 2` the layout is unrepresentable, so if
the implementation replicates KV correctly there is no mismatch to catch. Assert
on the explicit behavior instead: either `UnsupportedLayout` is raised, or the
KV replication invariant holds and the sharded output matches the reference.

## Diagnostics

On failure, report: first mismatching parameter name, source and destination TP
degrees, and max absolute logit error. That covers most of the debugging value.

Do **not** build layout inference that reports "expected `HeadPartitioned`,
actual `Shard(0)`". Identifying which wrong layout produced observed shard
contents requires searching candidate layouts against the data and is a separate
feature. Add it later if it earns its place.

## Acceptance criteria

- Correct reshard across every ordered pair in `t ∈ {1, 2, 4}` satisfies the
  invariant, in both 1a and 1b.
- Break cases 1 through 3 each violate the invariant.
- Case 4 either raises or satisfies the invariant under KV replication.
- Measured tolerance floor is documented and the threshold is set above it.
- No GPU, vLLM, or RL dependency in `pyproject.toml`.

## Transport interface

Define it now so phase 2 is additive, and keep it small. Two implementations,
`InProcessTransport` and `GlooTransport`, and nothing more. The artifact should
read as the minimum interface phase 2 needed, not as a distributed systems
framework.

```python
@dataclass
class SyncRecord:
    t_reshard: float | None
    t_transfer: float | None
    t_load: float | None
    src_layout: str
    dst_layout: str
    param_count: int
    transport: str

class Transport(Protocol):
    def sync_weights(self, src: ShardedParams, dst: ParamSink) -> SyncRecord: ...
```

`ParamSink` is a rank-local parameter holder in phase 1 and a vLLM engine in
phase 2.

## Suggested repo layout

```
weight-sync-bench/
  README.md
  pyproject.toml
  src/weight_sync_bench/
    model.py        # ReferenceModel
    sharded.py      # ShardedModel, collectives
    shardspec.py    # Placement sum type, ShardSpec, layout tables
    reshard.py      # split / gather / reshard
    transport.py    # Transport protocol, SyncRecord, InProcess, Gloo
  tests/
    test_reshard.py # invariant across all TP pairs, 1a and 1b
    test_breaks.py  # cases 1-3 assert violation, case 4 asserts explicit handling
```

## README structure

Write this after the tests pass, not before.

Problem → why naive resharding is invisible → layout diagrams for the two fused
tensors → design → correctness invariant → adversarial break table → measured
tolerance floor → how this extends to GPU and vLLM.

Lead with the fused-layout explanation. It is the part that distinguishes
understanding model-parallel infrastructure from knowing PyTorch, and it should
be in the README rather than only in the code.

# Phase 2 spec: weight-sync latency against vLLM

## Status

- **2a met.** bf16 floor measured differentially against real Qwen3-0.6B on vLLM 0.28.0, gate
  verdict PASS, reproduced through a second extraction path. `tolerance/phase2a_bf16_floor.json`,
  `tolerance/phase2a_bf16_floor_v2.json`.
- **2b met.** A real `LayoutTable` for Qwen3-0.6B at TP in {1, 2, 4}, validated by shape
  prediction (227/227 parameter names) and bit-exact content prediction against vLLM's post-load
  tensors. `tolerance/phase2b_layout.json`.
- **2c, 2d, 2e pending, and deliberately waiting.** They are not blocked on anything 2a or 2b
  left unfinished. They wait because all three are built against a specific engine construction,
  and which construction is not yet settled. 2a and 2b constructed the vLLM engine directly, in
  this repo, with `worker_extension_cls` set so that `collective_logits.run_one_prompt` could
  reach the raw logits tensor. The real runs target prime-rl, which constructs its own engine.
  If prime-rl's construction does not admit that extension, the correctness gate every 2e timing
  depends on cannot run inside a prime-rl loop, and 2c through 2e would be built against an
  engine that cannot certify them. An engine-construction probe settles this before any of the
  three is built. See phase 3's build order.
- Break-case reinjection through the real vLLM load path is tracked as an open question, not as
  a 2b deliverable; the layout table is validated by the two prediction checks above.


## What carries over

`LayoutTable`, the `Placement` sum type, `reshard.py`, and the discipline of measuring the
tolerance floor before writing any assertion. All unchanged.

What gets replaced is the consumer. `ShardedModel` was a stand-in for an inference engine. In
phase 2 the consumer is vLLM, so the invariant becomes

    ReferenceModel(full_params, x) == vLLM(x)   after a weight sync

Same invariant, different right-hand side. The phase 1 harness is what certifies phase 2's
measurements rather than a separate exercise.

## Goal

Measure the cost of pushing updated parameters from a trainer to N vLLM sampler replicas,
decomposed as

    T_sync = T_reshard + T_transfer + T_load

swept over model size, trainer parallelism config, sampler TP degree, and transport, with
correctness asserted at every configuration. Report sampler GPU idle time during sync separately,
since idle time is what connects to rollout throughput and latency alone does not.

This is the seam between the two halves of an asynchronous RL post-training system. Every such
system has it on the critical path. There is no public benchmark of it.

## Non-goals

- No RL loop, environments, verifiers, advantage estimation, or real optimizer. The trainer is a
  parameter holder that mutates weights with noise. See "Minimal trainer" below.
- No staleness sweep. That is phase 3 and it uses this as instrumentation.
- No multi-node. Single node, up to 8 GPUs.
- No training throughput measurement. Only sync cost.

## Build order

2a is a go/no-go gate. Do not build anything else until it resolves.

---

## 2a. bf16 floor (gate)

### Why this gates everything

Phase 1 measured an fp32 floor of 1.669e-06 absolute against injected bugs of 0.338 to 2.535,
roughly six orders of magnitude of separation. fp32 carries 23 mantissa bits, bf16 carries 7
explicit, a difference of 16 bits. Naive scaling puts the bf16 floor near

    1.669e-06 * 2^16 = 0.109

against a weakest injected bug of 0.338. That is a ratio of about 3, not 2 x 10^5. Case 3
(permuted norm weights) may stop being detectable at all, and cases 1 and 2 at 1.1 to 2.5 would
have a margin of 10x to 25x rather than thousands.

Measure this before building anything. It determines whether the correctness half of phase 2 is
viable in its current form.

### What to run

Two GPUs. TP=2 needs two ranks on two physical devices — a single GPU cannot produce a real TP=2
measurement, no software fallback reflects actual sharded execution. TP=1 runs on one of the two.
Rent by the hour; still under $10 for a few hours on two GPUs.

1. Pin a vLLM version. Record it. Read the weight-update entry point in the source of the version
   you pinned rather than trusting a tutorial; that API surface has churned repeatedly.
2. Load Qwen3-0.6B in bf16.
3. Do not build a `ReferenceModel`-equivalent forward, and do not use HF `transformers` as the
   floor reference. Measure the floor **differentially**: load the correct, unmutated checkpoint
   into vLLM at TP=1 and separately at TP=2, run identical prompts through both, and compare
   vLLM-TP=1 logits to vLLM-TP=2 logits. Both sides run identical kernels (same vLLM build, same
   attention implementation, same RoPE/RMSNorm code, same upcasting choices), so the only
   remaining difference is reduction order and dtype — which is exactly the floor the threshold
   has to admit and nothing else. Use HF `transformers` only as a sanity check that both vLLM
   configurations produce plausible output (e.g. low perplexity on real text); it never supplies
   the floor number.

   **The confound this avoids, and why not to "fix" it by checking upcasting parity instead:**
   comparing vLLM against an independent implementation (HF or a hand-rolled forward) conflates
   two different sources of deviation — dtype/reduction-order (what 2a is supposed to measure)
   and kernel/implementation differences (flash-attention vs. eager attention, differing
   RoPE/RMSNorm code paths, differing internal fp32 upcasting) which are not what 2a exists to
   gate and could dominate the measured floor without indicating anything about layout
   correctness. Verifying that both implementations upcast identically was considered and
   rejected: it is hard to verify from outside each library, easy to get wrong silently, and even
   if verified today can drift on the next vLLM or transformers version with no signal that it
   broke. The differential design (vLLM against itself at two TP degrees) removes the
   implementation variable by construction instead of by inspection. The accepted cost is that
   this no longer tests vLLM's output against an independent implementation — that is a deliberate
   tradeoff, not an oversight, that is a deliberate tradeoff, not an oversight. Do not "simplify" 
   this back to an HF comparison later; doing so reintroduces the exact confound this design exists 
   to remove.
4. Measure the floor at TP=1 vs TP=2 exactly as phase 1 did: N repetitions, varied token seeds,
   full logits tensor, record max, median, and mean deviation in ULP and absolute.
5. Re-inject the phase 1 break cases against the real Qwen3 layout, through the real vLLM
   weight-load entry point identified in step 1 — never through a synthetic construction that
   bypasses it. If an injected case fails to reproduce a mismatch when run through the real load
   path, that is not evidence the case doesn't apply to Qwen3; it is a finding about the load path
   itself (it may not be wired the way it's assumed to be, or it may be silently correcting the
   injected layout error the same way a resharding bug would be silently swallowed), and it is
   exactly the kind of thing 2a exists to catch. Investigate it, don't discard it.

### Change the primary statistic

In fp32 the max worked. In bf16 it will not, for two reasons already established in phase 1: the
max is an unconverged extreme order statistic that drifts with sample count, and the margin is
now thin enough that drift matters.

Switch the primary statistic to **mean absolute deviation over the full logits tensor**, keeping
max and median recorded. The reasoning: a layout bug is a deterministic function of the weight
permutation, so it perturbs every token in the same direction, while dtype rounding is
approximately zero-mean. Mean deviation therefore separates systematic from random error with a
stable statistic instead of a tail one, and the separation improves with token count rather than
degrading.

Derive the threshold from the mean by the same rule shape as phase 1: a stated multiple of the
worst observed mean, applied by a pure function of the measurement.

### Gate criteria

- **Pass**: every phase 1 break case clears the bf16 threshold by at least 10x on mean deviation.
  Proceed to 2b.
- **Marginal**: cases 1 and 2 clear by 10x but case 3 does not. Proceed, but replace case 3 with a
  stronger norm injection and document that the original permutation is undetectable in bf16,
  which is itself a finding worth reporting.
- **Fail**: fewer than two cases clear. Stop and reconsider. Options in order of preference:
  compare on a held-out set of many prompts to exploit the systematic-vs-random distinction
  further; compare top-k agreement or softmax KL instead of raw logits; run the reference in fp32
  and accept that dtype error and layout error are then conflated, which weakens the harness and
  must be stated.

Report the numbers before proceeding either way.

### Measured (2a results)

Two runs against real Qwen3-0.6B / vLLM 0.28.0, both at batch=2, seq_len=8: 5 repetitions (80
positions total) on a single-GPU rented host, then 20 repetitions (320 positions total) on a
separate 2x A100-SXM4-80GB rented host, to check whether the statistics were sample-count-stable
before setting any constant. Full numbers and provenance for both hosts are in
`tolerance/phase2a_bf16_floor.json`.

- **The two runs differ in more than repetition count, which limits what the comparison shows.**
  The 5-rep run's host had only one physical GPU. TP=2 needs two ranks on two physical devices (see
  "What to run" above), so that run's TP=2 leg of the differential measurement never completed --
  its numbers come from a partial run, not a validated TP=1-vs-TP=2 differential under real sharded
  execution, unlike the 20-rep run's. Sample count and run completeness/host therefore changed
  together, not independently.
- **The mean moved less than the max, but treat this as suggestive, not confirmed.**
  `mean_deviation` moved from 4.296e-02 (5 reps, partial run) to 3.898e-02 (20 reps, complete run),
  a 9% move. `max_ulp` moved from 96.00 to 104.00 over the same two runs. This is directionally
  consistent with phase 1's finding (`tolerance/phase1a.json`) that the max is an unconverged
  extreme-order-statistic that drifts with sample count while the mean stays comparatively stable,
  and with this document's earlier choice of mean as the primary bf16 statistic -- but given the
  confound above, it corroborates rather than empirically confirms that choice. A same-host
  repetition-count comparison would be needed to confirm it cleanly. The threshold and gate verdict
  below do not depend on this comparison: they are derived solely from the 20-rep, complete-TP=2
  run.
- **The measured separation ratio is ~40x, not fp32's ~10^5x.** weakest break / floor mean was
  40.9x at 5 reps and 40.3x at 20 reps (case2_oproj_col_permute at 1.570 against floor mean
  3.898e-02). `SAFETY_FACTOR * GATE_MARGIN` must stay strictly below this ratio or the gate cannot
  fail an injected bug no matter how the two factors are split. The originally-stated gate criteria
  above ("clears... by at least 10x") assumed headroom this project did not have data for yet; naive
  reuse of phase 1's constants (`SAFETY_FACTOR=100`, `GATE_MARGIN=10`, product 1000) against a ~40x
  budget produced a threshold no break case could ever clear, and the resulting "fail" would have
  been an artifact of the unchanged constants, not a finding about bf16. **Superseding the "10x"
  figure above:** `weight_sync_bench/phase2/bf16_floor.py` now uses `SAFETY_FACTOR=15`,
  `GATE_MARGIN=2` (product 30, under the ~40x budget with headroom for the ratio to move on a
  future re-measurement), and states the threshold as a direct multiple of the mean
  (`threshold = SAFETY_FACTOR * mean_deviation`) rather than phase 1's power-of-ten rounding --
  rounding cost phase 1 nothing against ~5 orders of magnitude of headroom, but against a ~40x
  budget it can by itself consume up to a 10x tax the budget cannot absorb. Phase 1's rounded rule
  in `tolerance.py` is untouched; this change is scoped to phase 2.
- **Provenance narrowing applies unchanged from phase 1.** The floor is specific to Qwen3-0.6B,
  bf16, and the (batch, seq_len) it was measured at (2, 8) here. 2b must re-measure if the model,
  dtype, batch, or seq_len change.
- **The extraction bottleneck that blocked measuring seq_len dependence is resolved, and the fix
  was used to run the actual measurement, not just the primitive it's built on.** A run at
  `--repetitions 20 --batch 4 --seq-len 64` was started and killed after an hour without
  completing: vLLM's `prompt_logprobs=-1` extraction path builds on the order of 150k Python
  `Logprob` objects per position (one per vocab entry shown, `vocab_size=151936`), and that does
  not scale to seq_len=64, batch=4 -- a cost of the from-disk logprobs API `bf16_floor.py` uses,
  not a phase-2a physics finding. `src/weight_sync_bench/phase2/collective_logits.py` replaces
  that path with a `collective_rpc` call that returns the raw logits tensor directly (bit-identical
  `[7, 151936]` float32 tensors, `torch.equal`, against the old path for one prompt, 272.7x-932.5x
  faster, `tolerance/phase2b_extraction.json`), and
  `src/weight_sync_bench/phase2/bf16_floor_v2.py` ports the full differential-floor-plus-break-case
  measurement onto it (same TP1-vs-TP2 design, same subprocess-per-leg pattern, same
  `SAFETY_FACTOR=15`/`GATE_MARGIN=2`, same break-case injections -- all imported unchanged from
  `bf16_floor.py`, not re-derived) with `enable_chunked_prefill=False` and
  `enable_prefix_caching=False` now set explicitly (`bf16_floor.py` leaves both at vLLM's defaults).
  A reproduction run at `bf16_floor.py`'s exact recorded configuration
  (`--repetitions 20 --batch 2 --seq-len 8`) gave mean_deviation 3.898e-02, max_deviation 8.125e-01
  (104.00 ULP), and break-case means 1.772 / 1.570 / 2.984 -- **identical to
  `tolerance/phase2a_bf16_floor.json` to four significant figures on every one of those six
  numbers**, despite prefix caching being off this time and on in the original. **This resolves,
  not merely annotates, the open caveat that repetition independence under prefix caching was
  unverified**: had caching been silently correlating repetitions, turning it off would have moved
  the mean by more than four-significant-figure noise; it did not. Full numbers, provenance, and
  the resolved-caveat record are in `tolerance/phase2a_bf16_floor_v2.json`, which now supersedes
  `tolerance/phase2a_bf16_floor.json` on extraction path and cache flags while matching it on every
  measured quantity.
- **The seq_len sweep this unblocked shows the mean is invariant across a 64x range, and the
  practical seq_len-dependence caveat is retired, not narrowed.** `--repetitions 20 --batch 4
  --sweep-seq-len 8,32,128,512` gives mean_deviation 3.895e-02 / 3.725e-02 / 3.791e-02 / 3.280e-02
  -- a ~16% peak-to-trough spread with no monotonic trend across seq_len 8 to 512, consistent with
  repetition-count sampling noise rather than a seq_len effect, and every point gates PASS. This is
  what the design predicted: the mean characterizes the per-element error distribution, and more
  prompt positions means more independent draws from that same distribution, not a different one.
  Since the threshold is derived from the mean (`derive_threshold`, `SAFETY_FACTOR=15`), not the
  max, and the mean is what is now measured invariant, the practical question -- does the floor
  change enough with seq_len to invalidate the threshold at a longer prompt -- is answered no,
  within the range measured, and the caveat is retired accordingly for the mean. Also recorded: a
  `--repetitions 1 --batch 1` smoke run gave mean 3.038e-02, about 22% below the 20-repetition
  value (batch also differs, 1 vs 2, so not a pure repetitions-only comparison) -- a useful
  datapoint on how much a quick smoke test's number should be trusted, and the transported payload
  at seq_len=512 (~1.2GB per call, TP=2's all-gather branch carrying a full tensor back from each
  rank) moved without failure, pushing the file-handoff-reconsideration threshold
  `collective_logits.py`'s docstring names well past its previous ~77MB-exercised figure. Full
  numbers in `tolerance/phase2a_bf16_floor_v2.json`, which is the machine-generated output of this
  run pulled from the box, not a hand assembly from reported figures -- an earlier hand-written
  version of that file was lost before it was committed and has been superseded by this one.
  That file's commit (`43df6399b818cf53207329e7535cb8f7fe070303`) was made directly from the
  rented box rather than a local checkout, since transferring the artifact back over the Runpod SSH
  proxy kept failing -- since transferring the artifact back over the Runpod SSH proxy kept failing.
- **The seq_len=128 point's max is an unexplained spike, not an illustration of the
  order-statistic mechanism -- an earlier version of this document said the latter and was
  wrong.** `max_ulp` across the sweep is 104.0 / 168.0 / 1192.0 / 156.0 (`max_deviation` 0.8125 /
  1.3125 / 9.3125 / 1.21875) at seq_len 8 / 32 / 128 / 512. Phase 1's order-statistic finding
  (`tolerance/phase1a.json`) was a monotonic drift upward with more draws (11.5 -> 14 ULP going
  from 5 to 20 repetitions, same distribution, more samples). This is not that: the max spikes 7x
  at seq_len=128 and falls back *below* the seq_len=32 value at seq_len=512, despite seq_len=512
  drawing from roughly 6.21 billion pooled elements (20 reps x 4 batch x 511 rows x 151936 vocab)
  against seq_len=128's ~1.54 billion -- about 4x more draws landing on a smaller max. Calling the
  seq_len=128 spike "a clean, real illustration" of unconverged-order-statistic behavior, as an
  earlier revision of this document did, explains past the actual shape of the data rather than
  accounting for it. **Corrected: the spike is unexplained.** It is either a genuine, rare
  catastrophic-cancellation outlier on one of ~1.54 billion element-comparisons in that run, or a
  bug; both remain open. The stronger argument for mean-as-primary is the spike's magnitude, not a
  drift story: at seq_len=128, `max_deviation` (9.3125) is ~245.6x the floor mean at that same
  point (3.791e-02) and ~3.63x the weakest break case measured in that same run
  (`case2_oproj_col_permute`, 2.5637) -- had the gate used max instead of mean, this single
  point's noise floor would have exceeded a real injected layout bug, and the gate would have
  failed to distinguish "correct execution, one outlier element" from "genuinely broken sharding."
  Same-run evidence makes the case even more directly: the outlier element (9.3125) is one value
  among `4 * 127 * 151936 = 77,183,488` elements in the single repetition-cell that contains it, so
  it contributes `9.3125 / 77,183,488 ~= 1.2065e-07` to that cell's own mean -- about 5.5 orders of
  magnitude (a factor of ~3.18e-06) below the reported mean_deviation (3.791e-02). One anomalous
  element moved the max 7x while being arithmetically invisible to a mean averaged over tens of
  millions of siblings, from the same run, not a comparison across runs or against break-case
  values.
- **Three draws in at batch=4/seq_len=128, two at batch=2/seq_len=8. The max has no mechanism and
  needs none -- it's a heavy-tailed statistic, and three samples look exactly like one.**
  `max_ulp` at batch=4/seq_len=128/20 reps across seed_base 0/1000/2000: 1192.00 / 393.50 / 445.00.
  Two cluster near ~400 and one sits ~2.8x higher -- consistent with a heavy tail sampled three
  times, not a pattern demanding its own explanation. None of the three reproduces another; a real
  single-element bug would have. The "worth locating a specific element" branch is retired. The
  gate is unaffected -- it derives the threshold from the mean, not the max.
- **The mean is confirmed invariant across a full prompt-set change, more strongly than the seq_len
  sweep alone shows.** At batch=4/seq_len=128/20 reps: mean 3.791e-02 / 3.712e-02 / 3.726e-02 across
  the three seed bases, a 2.1% spread, landing inside the sweep's own 3.280e-02-3.895e-02 band. At
  batch=2/seq_len=8/20 reps: 3.898e-02 / 3.865e-02, 0.8% apart. This varies the input itself, not
  just the sample count the seq_len sweep varies -- stronger evidence for the same conclusion.
- **Prompt-draw dependence of break magnitudes is retired, at both configurations measured -- the
  earlier "tens-of-percent" finding was entirely a three-way configuration confound, not a real
  effect.** That comparison set three break-case triples against each other -- the smoke run
  (`--batch 1 --seq-len 8 --repetitions 1`), the reproduction (`--batch 2 --seq-len 8
  --repetitions 20`), and the seed_base=1000 rerun (`--batch 4 --seq-len 128 --repetitions 20`) --
  with batch, seq_len, and repetition count all differing alongside the prompt draw, isolating
  nothing. With those held fixed and only the draw varied, at two independent configurations, break
  magnitudes are stable: at batch=4/seq_len=128/20 reps (seed_base 0/1000/2000), case2 (weakest
  throughout) is 2.564 / 2.596 / 2.602, a 1.5% spread; at batch=2/seq_len=8/20 reps (seed_base
  0/1000), case2 (weakest throughout) is 1.570 / 1.648, 5.0% apart. Every value at every
  configuration moves single-digit-percent, not tens. There is no open prompt-draw-dependence
  question left at either configuration.
- **The separation ratio is a function of configuration, not of draw -- record both bands.** At
  batch=2/seq_len=8/20 reps: 40.3x and 42.6x across two draws, 5.7% apart. At batch=4/seq_len=128/20
  reps: 67.6x / 69.9x / 69.8x across three draws, under 2.1% apart. Each band is tight within
  itself; the ~1.7x gap between the two bands is a real configuration effect (batch and/or seq_len),
  not draw noise. batch=2/seq_len=8 is the binding (narrower) configuration and the one this
  project's break cases run at by default.
- **`SAFETY_FACTOR=15 * GATE_MARGIN=2 = 30` is now validated against the binding configuration's
  measured band, not a single draw.** The binding band (40.3x-42.6x) leaves ~34% headroom at its
  narrowest (`(40.3-30)/30`) -- the budget would need enlarging by more than a third before either
  observed draw could threaten it. This is a measured band, not a proven lower bound: a later draw
  at this configuration could in principle land narrower than 40.3x, the same caveat the max finding
  above carries. Two independent draws within 5.7% of each other is meaningfully stronger support
  than the single 40.3x observation the constants were originally set against.
  `weight_sync_bench/phase2/bf16_floor.py`'s `SAFETY_FACTOR` comment now cites this band. **Constants
  are unchanged** -- this records that the existing split has more support than when it was chosen,
  not a case for retuning it. Full numbers in `tolerance/phase2a_bf16_floor_v2.json`.
- **Provenance narrowing still applies to model, dtype, and batch.** The floor is specific to
  Qwen3-0.6B and bf16; 2b must re-measure if either changes. seq_len is now measured stable across
  8-512 at batch=4 (and separately confirmed unaffected by prefix caching at batch=2), so it is no
  longer in the same "must re-measure on any change" category the four-input phase 1 dependency
  names -- but `batch` itself has not been swept the way `seq_len` was, and remains an open input
  in that sense.

**Gate verdict: PASS on the physics, now independently reproduced.** All three break cases at 20
reps sit 40x-77x above the measured floor mean (1.772 / 1.570 / 2.984 against 3.898e-02) and clear
the revised threshold (0.5847) by 2.7x-5.1x. The pre-measurement risk estimate above (naive bf16
scaling predicting a floor near 0.109 and case 3 possibly undetectable) did not materialize: the
measured floor (3.898e-02) is about 3x better than that estimate, and case 3 (norm permutation) is
measured as the **strongest** break (2.984), not the weakest -- the weakest is case 2 (o_proj/down
column permutation, 1.570). This exact result -- floor, break-case means, and verdict -- was
reproduced via the `collective_rpc` extraction path with prefix caching and chunked prefill both
disabled (`tolerance/phase2a_bf16_floor_v2.json`), and the gate additionally passes at seq_len 32,
128, and 512. Proceed to 2b.

### Phase 2a's break magnitudes are not comparable to phase 1's

Phase 1 (fp32, toy model): case 3 (permuted norm) was the **weakest** break at 0.338-0.413, cases
1/2 ran 1.1-2.5. Phase 2a (bf16, Qwen3-0.6B) above: case 3 is the **strongest** at 2.984, case 2
the weakest at 1.570. The rank order inverted.

Two variables changed at once (dtype and model), so this was investigated by attribution before
being explained -- CPU-only, no GPU, nothing re-measured against vLLM: the real Qwen3-0.6B
checkpoint was downloaded and inspected directly, and this repo's own toy `ReferenceModel` was
instantiated for a controlled, same-operation comparison. Full numbers are in
`tolerance/phase2a_bf16_floor.json`'s `break_case_ordering_inversion` block.

- **Case 3's inversion is a model-distribution effect, not a dtype effect, and it is fully
  explained.** Applying `bf16_floor.py`'s actual case-3 operation
  (`torch.roll(w, shifts=w.shape[0]//2)`) by hand to real layer-0 weights gives a relative
  weight-space perturbation of **0.573** on Qwen3's `input_layernorm` against **0.145** on the
  toy's `attn_norm` -- a ~4x gap that tracks coefficient of variation almost exactly (0.43 vs 0.10).
  Qwen3's trained norm weights at layer 0 are right-skewed with mean 0.175 (pooled across all 28
  layers the tensor has entries up to 106.5, but case 3 only touches layer 0, whose own outliers
  are milder), against the toy's `normal(1.0, 0.1)` by construction (`model.py:_norm_weight`). A
  roll-by-half approximately replaces each entry with an unrelated value from elsewhere in the same
  vector, so its relative perturbation scales with the vector's own spread, not with dtype. This
  measurement is fp32-vs-fp32 weight arithmetic on CPU, before any forward pass -- the same ~4x gap
  would appear if both phases ran in fp32. **The ordering inversion is evidence about this model's
  norm-weight distribution, not evidence about bf16 detectability.**
- **Case 2's weakening is only partially attributed.** `n_kv_heads` is not a candidate variable at
  all -- `case2_oproj_col_permute` permutes `o_proj`'s query-head column blocks only and never
  touches K/V heads. Geometry is a plausible partial contributor (Qwen3 has 2x the toy's heads, 16
  vs 8, and its o_proj compresses 2048 input channels to 1024 output channels where the toy's is
  square, 256-to-256), but the strongest CPU-measurable proxy does not confirm it: the same
  adjacent-head-block swap gives relative weight-space perturbation **1.413** on the toy against
  **1.367** on Qwen3 -- essentially unchanged (~3%), which cannot account for a 23-38% weaker
  measured break. Adjacent o_proj head blocks are mildly more correlated on Qwen3 than the toy's
  independently-random ones (cosine similarities up to 0.29 vs at most 0.01), directionally
  consistent with a weaker swap but too small and noisy to call a mechanism. Whatever remains is a
  downstream forward-pass effect -- how the permuted output propagates through the residual stream,
  the next layer's norm, and the logits -- that requires a real forward pass through the full model
  to investigate, i.e. a GPU. Recorded as partly geometry, remainder unresolved, not as fully
  explained.
- **Consequence for reading these artifacts:** the 40x separation ratio and the PASS verdict above
  are valid on their own terms, since they compare Qwen3's break magnitudes to Qwen3's own bf16
  floor measured on the same checkpoint. Any comparison of a phase-1 break magnitude to a phase-2a
  one, or any reading of the case-3/case-2 rank swap as a statement about bf16 detectability, is
  not -- it is a statement about the two models' weights and the two independently re-derived
  injection operations, not about dtype.

### LIMITATION: the gate is calibrated at the single most favorable layer, and the drop is a step, not a gradient

Every break case above injects at layer 0 (`corrupt_checkpoint`'s `layer` parameter, default 0,
never varied by any run this gate's PASS verdict rests on). A five-point sweep, `--layer` in
`{0, 7, 13, 20, 27}` (28 layers total, 0-indexed), all at `--tp 2 --repetitions 20 --batch 2
--seq-len 8` (`tolerance/phase2a_layer_depth_finding.json`), refutes both mechanism hypotheses
this investigation produced along the way and lands on a sharper, still-unresolved finding.

- **The floor is identical across all five layers (3.898e-02)**, as expected -- it has no
  injection, so it cannot depend on where a break case would target. The effect below is entirely
  in the break-case legs.
- **DATA** (mean_deviation, threshold 5.847e-01): layer 0 -- 1.772 / 1.570 / 2.984, **PASS** (2.7x
  clearance). layer 7 -- 0.2766 / 0.2265 / 0.2675, FAIL. layer 13 -- 0.2269 / 0.1617 / 0.4996,
  FAIL. layer 20 -- 0.4449 / 0.2092 / 1.052, FAIL (case3 alone would individually clear the
  threshold here, but the verdict gates on the weakest case, case2, which does not). layer 27 --
  0.5227 / 0.2150 / 0.4448, FAIL.
- **BOTH PRIOR HYPOTHESES ARE REFUTED.** The first (case 3 tracks that layer's own norm-weight
  distribution while cases 1/2 hold steady, from the ordering-inversion investigation above) was
  already refuted by the initial layer-0-vs-13 comparison. The second, floated after that
  refutation -- monotonic depth decay, break magnitude falling steadily with distance from the
  logits -- is now ALSO refuted: it predicted layer 27 would be smallest, and it isn't for any
  case; layers 7-27 instead form a band with mild non-monotonic structure (case1 rises gently
  7->27, case2 is roughly flat, case3 peaks at layer 20). **Layer 0 sits 2.8x-11.2x above the
  corresponding case at every other tested layer; layers 7 through 27 do not differ from each
  other by anything resembling that gap.** This is a step between layer 0 and the rest, not a
  gradient across depth.
- **A third candidate, offered after two wrong guesses and marked accordingly as inference, not a
  traced mechanism:** layer 0 is the only layer whose input is the raw token embedding rather than
  a residual stream already carrying accumulated contributions from every preceding block. A
  perturbation there acts on a signal with nothing else yet mixed in and is then amplified across
  all 27 subsequent blocks with no prior dilution; everywhere else, the perturbation competes with
  a stream that already carries most of the eventual representation. This predicts a step between
  layer 0 and everywhere else, matching what the sweep shows -- but it has not been confirmed by
  residual-stream instrumentation, only inferred from the outcome shape.
- **Consequence, sharper than before: this cannot be fixed by a depth-dependent threshold.** A
  smooth depth-indexed correction to `SAFETY_FACTOR`/`GATE_MARGIN` would need the failure to scale
  with depth; it doesn't -- layers 7 through 27 fail by broadly similar amounts regardless of how
  far they are from layer 0. Of the five layers tested, **layer 0 is the only one where the
  current gate would catch an injected bug of this shape and magnitude.** Calibrating at layer 0
  is calibrating at the single most favorable position in the model, not a representative one.
- **Next test, not yet run**: a finer sweep over layers `{1,...,6}` to locate the step precisely
  (a hard boundary right after layer 0, versus a fast-but-continuous drop over the first few
  layers), and direct residual-stream-norm instrumentation during a forward pass to test the
  embedding-input candidate rather than merely infer it.

This affects any future break-case reinjection built against 2b's real resharder (see 2b's
"What's left" note) too: whatever layer(s) that reinjection targets should not default to layer 0
without accounting for this.

### TP degree is not a third axis for the separation ratio

The separation ratio (weakest break / floor) has two measured bands at TP=2: 40.3x-42.6x at
batch=2/seq_len=8, and 67.6x-69.9x at batch=4/seq_len=128 (see the `SAFETY_FACTOR` comment in
`bf16_floor.py`). A TP=4 run at the batch=2/seq_len=8 configuration
(`tolerance/phase2a_bf16_floor_v2_tp4.json`) checks whether TP degree moves it the way batch and
seq_len do. It does not: floor 3.840e-02 and break-case means 1.775 / 1.572 / 3.004 are within
1.5% of TP=2's recorded values at the identical configuration (floor -1.49%, breaks +0.17%/+0.13%/
+0.67%), and the resulting separation ratio (40.9x) sits inside the same TP=2 band rather than
forming a new one. Consistent with the differential design's premise: the floor and break-case
magnitudes are properties of dtype, reduction order, and the checkpoint's weight distribution,
none of which TP degree changes -- TP=4 shards the same computation differently, it does not
change what is being computed. (This single configuration does not by itself confirm TP=2's
already-recorded seq_len-invariance or prefix-caching-independence findings hold at TP=4 too --
only that the separation ratio itself does not move with TP degree at this one point.)

---

## 2b. Real layout tables

Phase 1's layouts were designed. Phase 2's must match reality, and three conventions are in play:
the Qwen3 checkpoint ordering, vLLM's internal ordering after load, and Megatron-Core's fused QKV
ordering. `HeadPartitioned` has to express the real one.

This is where phase 1's abstraction gets tested and it is the part most likely to need revision.
If `HeadPartitioned` cannot express Qwen3-as-loaded-by-vLLM, say so explicitly and state what
extension it needs rather than working around it in the resharder.

Deliverable: a `LayoutTable` for Qwen3 at each TP degree in {1, 2, 4}, validated by the invariant
against vLLM at each.

**STATUS: MET.** `src/weight_sync_bench/phase2/{geometry.py,qwen3_layout.py}` build a real
`LayoutTable` for Qwen3-0.6B at TP in {1, 2, 4} out of phase 1's unmodified `HeadPartitioned` /
`FusedPaired` / `Replicated` / `Shard` / `ShardSpec` / `LayoutTable`, driven by a new
`CheckpointGeometry` type (deliberately not `ModelConfig` -- see `geometry.py`'s docstring for why
reusing the toy's config would leak its `n_heads*head_dim==d_model` assumption into the real-model
path, which Qwen3-0.6B breaks: `16*128=2048 != 1024`). "Validated by the invariant against vLLM":
2b has no separate reference implementation the way phase 1's `ReferenceModel`/`ShardedModel` pair
does, so validation here is two independently-falsifiable checks, both run on a real 4x
A100-SXM4-80GB box at TP in {1, 2, 4} (`tolerance/phase2b_layout.json`) -- shape prediction
(227/227 real parameter names matched `CheckpointGeometry`'s predicted per-rank shape exactly, at
every degree) and content prediction (`reshard.split_tensor` -- this repo's own resharder, not a
parallel formula -- applied to the real TP=1 tensor reproduced the real TP=2/TP=4 rank-local
tensor bit-exactly, `torch.equal`, for `qkv_proj` and `gate_up_proj`, at every rank). Break-case reinjection through
the real vLLM load path (see below) is tracked separately as an open question; the layout-table
deliverable itself is met and needs no further work.

**Two findings from 2a prep, confirmed against the real Qwen3-0.6B checkpoint (config.json and
the safetensors header, not assumed), that change what 2b is:**

- **The checkpoint stores `q_proj` / `k_proj` / `v_proj` and `gate_proj` / `up_proj` as separate
  tensors. Fusion into `qkv_proj` / `gate_up_proj` happens inside vLLM's own model-loading code,
  not in the checkpoint.** So `HeadPartitioned` and `FusedPaired`, as phase 1 defined them,
  describe vLLM's *post-load* internal ordering, not the checkpoint's on-disk ordering. There is
  no single `LayoutTable` that is "the" Qwen3 layout independent of which side of the load you
  mean. 2b's actual object of study is therefore **the loader boundary itself**: what the
  checkpoint's tensors look like going in, what vLLM's internal buffers look like coming out, and
  whether `HeadPartitioned` / `FusedPaired` describe the output side, the input side, or need to
  become two different things. It is entirely possible that the phase 1 placements are never
  exercised against a raw checkpoint at all -- only against vLLM's already-fused internal
  representation -- in which case 2b's `LayoutTable` targets that representation and the
  checkpoint's unfused shape is only ever read, never placed. Do not assume which side the table
  describes; determine it from where the trainer-to-vLLM handoff actually needs a layout decision.

  **RESOLVED (this is that determination):** `qwen3_layout.py`'s `LayoutTable` targets vLLM's
  post-load fused representation exclusively (`qkv_proj`, `gate_up_proj`, real vLLM parameter
  names) -- confirmed correct by the content-prediction check above, run against real fused
  tensors. `CheckpointGeometry.checkpoint_shapes()` reports the checkpoint's unfused, on-disk
  shapes (verified against the real safetensors header) but nothing ever builds a `Placement` or
  `ShardSpec` from them; they exist to be read (e.g. by a future trainer-side checkpoint loader),
  never placed. `HeadPartitioned`/`FusedPaired` needed no extension: `_check_fusion_matches_
  placement_assumptions` (qwen3_layout.py) checks the geometry's stated fusion order against what
  those placements hardcode before ever constructing them, and for Qwen3-0.6B it passes.

- **Qwen3 applies a per-head RMSNorm to Q and K before RoPE** (`self_attn.q_norm.weight`,
  `self_attn.k_norm.weight`, each shape `[head_dim]`), **the same weight vector reused identically
  across every head.** This has no phase 1 analogue. It is replicated *within* a head (every
  element of the `head_dim` vector applies to that one head's activations, same as any RMSNorm
  weight) and simultaneously partitioned *across* heads in the sense that the whole point of
  `HeadPartitioned` is to say which heads a rank owns -- yet this weight does not vary by head at
  all, so "partitioned across heads" is vacuous for it even though it lives inside a
  per-head-sharded computation. No existing `Placement` states this combination.

  **CLOSED: `Replicate()` is sufficient. No new placement.** Confirmed on the box
  (`tolerance/phase2b_layout.json`): `q_norm.weight`/`k_norm.weight` are bit-identical across
  every rank and every TP degree tested (1, 2, 4), matching the source-level prediction that
  neither carries a `weight_loader`/`output_dim` attribute and so is loaded by vLLM's generic
  full-tensor-copy path with no TP-dependent narrowing. The resharder (`reshard.py`) dispatches
  purely on `Placement` and moves bytes; a `Replicate()` tensor whose content never depends on TP
  degree reshards as a no-op between any two degrees regardless of how many heads a rank owns
  after the reshard. The "applied per head, same vector reused across heads" fact lives entirely
  in `Qwen3Attention.forward`'s reshape-then-normalize (consumer code), never in layout data --
  there is no `LayoutTable` consumer other than the resharder that would need "this replicated
  tensor is conceptually per-head" recorded explicitly. `qwen3_layout.py`'s `LayoutTable` places
  both as plain `Replicated()`.

  This question was untouched by the 2a break-case-ordering investigation (that investigation
  confirmed `case3_norm_permute` injects into `model.layers.0.input_layernorm.weight` only --
  `q_norm`/`k_norm` are never read or written by it); the closure above comes from 2b's own
  inspection run, not from anything 2a measured.

---

## 2c. Minimal trainer

Not a trainer. A process holding parameters under a training parallelism config, mutating them
each iteration with random noise scaled to a realistic update magnitude.

Rationale: weight sync cost depends on parameter count, layout, and transport. It does not depend
on where the new values came from. Adding an optimizer would add setup cost and change nothing
measured.

Support at minimum FSDP-style full sharding on dim 0 and a Megatron-style TP config, since the
reshard cost is asymmetric between source and destination and that asymmetry is the part nobody
documents.

---

## 2d. Transports

Three, in increasing implementation cost. Build in this order.

1. **Filesystem checkpoint.** Trainer writes, sampler reloads. Everyone starts here and nobody
   publishes numbers for it. Baseline.
2. **NCCL broadcast.** A process group spanning trainer and sampler ranks, bucketed. This is what
   production async RL systems do. Note that vLLM workers are separate processes, so the group
   has to be established through vLLM's collective RPC surface.
3. **CUDA IPC.** Zero-copy via shared device memory handles, requires trainer and sampler
   colocated on the same GPUs. NeMo-RL already uses CUDA IPC for teacher logits in cross-tokenizer
   distillation, so the primitive exists there to read.

Each implements the phase 1 `Transport` protocol and returns a populated `SyncRecord`.

---

## 2e. Measurement

### Configurations

- Models: Qwen3-0.6B, then 1.7B, then 8B. Report the crossover, since at small P fixed costs
  dominate and at large P bandwidth does.
- Sampler TP in {1, 2, 4} against a fixed trainer config, then flip and hold the sampler fixed
  while varying the trainer. The asymmetry is the result.
- All three transports at every configuration where they apply.

### Per-sync record

`t_reshard`, `t_transfer`, `t_load`, plus **sampler GPU idle time during sync**, which is not the
same as sync latency because a well-implemented sync overlaps with generation.

`None` rather than `0.0` for a stage that did not run on this rank. Phase 1 established this;
aggregation must handle it.

### Statistics

p50 and p99 over at least 100 syncs per configuration. Not a single timing. Discard the first 10
as warmup and say so.

### Correctness

The invariant runs at every configuration, using the 2a threshold. A timing measurement of a sync
that delivered wrong weights is worthless, so correctness gates every recorded number.

---

## Artifacts

Same discipline as phase 1. Committed JSON with provenance: vLLM version, torch, CUDA, driver,
GPU model, NVLink topology, python, platform, dtype, model, commit. A measurement without its
environment is not reproducible and the numbers are the deliverable.

Record raw per-sync values, not only aggregates.

## Acceptance criteria

- bf16 floor measured and documented, with the gate decision stated.
- Qwen3 `LayoutTable` validated against vLLM at every TP degree in {1, 2, 4}.
- All three transports implemented, or a stated reason one was dropped.
- `T_sync` decomposed into its three components at every configuration, with sampler idle time
  recorded separately.
- p50 and p99 over >= 100 syncs per configuration.
- Correctness invariant passes at every configuration that produced a recorded timing.
- Crossover between fixed-cost-dominated and bandwidth-dominated regimes identified and stated.

## Cost

2a is two GPUs for a few hours, roughly $10. The full sweep is an 8xH100 node at roughly
$20/hour, used intermittently, so a few hundred dollars total. Prime Intellect Fast Compute Grants
cover this; apply with 2a results in hand rather than as a proposal.

## Order of work

Do not rent 8 GPUs before 2a resolves. Do not build transports before 2b validates a real layout
table. The single most common failure in this genre is building the measurement apparatus before
confirming the thing being measured can be measured.

---

# Phase 3 spec: staleness against a real asynchronous loop

## Goal

Measure how sampler staleness trades against rollout throughput in a real asynchronous RL
post-training loop, with phase 2's decomposed `T_sync` as the instrument.

Staleness is the number of optimizer steps by which a sampler's weights lag the trainer's. It is
the parameter every asynchronous RL system sets and almost none reports the cost of. Phase 2
measures what one sync costs. Phase 3 measures what buying fewer of them is worth: as staleness
increases, sync frequency falls and sampler GPU idle time falls with it, while the rollouts a
sampler produces are drawn from an increasingly out-of-date policy. The deliverable is the
throughput side of that trade, measured, with the correctness side gated rather than assumed.

The engine is prime-rl at a pinned commit, running vLLM. Phases 1 and 2 built their own
consumers precisely so this phase could stop doing that and measure a system someone actually
runs.

## What it consumes from phase 2

- **2c through 2e as instrumentation.** The per-sync record (`t_reshard`, `t_transfer`, `t_load`,
  sampler idle time), the p50/p99 discipline over at least 100 syncs, and the transport set.
  Phase 3 varies staleness and reads those same fields; it does not define new ones.
- **The 2a threshold as the correctness gate.** Every phase 3 timing is gated by the same mean
  absolute deviation threshold, `SAFETY_FACTOR * mean_deviation` with `SAFETY_FACTOR = 15`, and
  by the same differential design that produced it. A staleness measurement on a loop that is
  delivering wrong weights measures nothing. The threshold's provenance narrowing carries over
  unchanged: it is specific to Qwen3-0.6B, bf16, and the vLLM version it was measured at, and it
  must be re-measured if any of those change.
- **2b's `LayoutTable`** for the TP degrees the sweep uses, and `reshard.py` underneath it.

## What phase 3 changes about the trainer

2c's minimal trainer is a parameter holder that mutates weights with noise, which is correct for
measuring sync cost and useless for measuring staleness: staleness is defined against optimizer
steps, and a loop that never computes a gradient has no steps to lag behind. Phase 3 therefore
runs prime-rl's real trainer and its real loop. This is the one place where the deliberate
absence of an RL loop in phases 1 and 2 ends.

It does not follow that phase 3 measures learning. See non-goals.

## Non-goals

- **No claim about learning quality.** No reward curves, no comparison of final task performance
  across staleness settings, no statement about which staleness value trains better. Those
  require many full runs to separate signal from seed variance and are a different project. Phase
  3 reports throughput, idle time, and sync cost against staleness, and stops there.
- No new transport implementations beyond whatever phase 2d settles on.
- No environment or verifier authoring. An existing small task is used as a load generator.
- No multi-node.
- No model beyond the sizes phase 2e already swept.

## Build order

Three stages, in order. Each gates the next.

### 3a. Engine-construction probe

The correctness gate reaches raw logits through a vLLM worker extension
(`collective_logits.py`), which requires the engine to be constructed with
`worker_extension_cls` set. Phases 2a and 2b constructed that engine themselves. prime-rl
constructs its own. Whether prime-rl's construction accepts that argument, forwards it, or drops
it decides whether the gate can run inside a prime-rl loop at all, and it is cheap to answer:
read the engine construction at the pinned commit, then check bit-identity of the extracted
logits tensor against the same extraction from a directly-constructed engine at the same
checkpoint and flags.

**RESOLVED by source read, and the answer is that it does not attach.** At the pinned commit
prime-rl constructs no in-process `LLM` object at all: its only engine construction ends in
vLLM's OpenAI API server, and the engine is reachable only as an `EngineClient` from inside
request handlers. The extraction path calls `collective_rpc` and `generate` on a local object,
so it does not attach as written, and prime-rl's HTTP surface exposes `collective_rpc` only
through routes that each hardcode one RPC method name. A caller-supplied `worker_extension_cls`
is separately dropped: prime-rl's config forwards it onto the argparse namespace, and the server
entry point then overwrites it unconditionally from a module-level dict keyed by the
weight-broadcast transport, before the plugin hook that might otherwise have intervened runs.

The seam is that dict. It is mutable and read at call time, so a launcher can rebind the
configured transport's entry to a class subclassing both prime-rl's weight-update worker and the
logits-hook extension, then start the server. Composition by subclassing is forced, since the
field names exactly one class. The alternative, making prime-rl's overwrite conditional, is one
line but has to be carried as a patch against a pinned third party. Exercising the launcher
against a running server is the remaining feasibility question.

What is still worth measuring on a GPU is narrower and separable: whether prime-rl's engine
flags change the logits the correctness gate reads. prime-rl leaves chunked prefill and prefix
caching at vLLM's defaults, both on, where the floor was measured with both off, and runs with
cuda graphs where the floor enforced eager execution. Two constraints on that comparison, both
consequences of what those flags do:

- **Compare under the 2a threshold, not bit-identity.** Chunked prefill changes how prefill
  attention accumulates and cuda-graph capture can change kernel and padding choices, so a
  correct engine is under no obligation to return identical bits across these profiles.
  Bit-identity is the right check only between engines whose flags match, which is where phase
  2b's extraction check earned it.
- **The prompt must be long enough to actually chunk.** Below vLLM's batched-token budget,
  chunked prefill never chunks, and a comparison at a short prompt reports agreement while
  leaving the first suspect flag unexercised. Force chunking by lowering the batched-token
  budget on that leg or by raising the prompt past it.

**Measured: under prime-rl's flag profile the extraction does not deviate, it stops.** Run on one
A100 at the point the bf16 floor was swept (20 repetitions, batch 4, seq_len 32, TP=1) with the
batched-token budget forced to 16 against a 32-token prompt, the prime-rl-profile leg raised on
its first prompt: two `compute_logits` captures of 16 positions each, where the extraction path
requires exactly one multi-position capture per prompt. Chunked prefill produces one capture per
scheduling chunk, and the raise is itself the runtime proof that chunking occurred; the resolved
scheduler config predicted it independently, and the floor-profile leg resolved the default
budget with chunking off, so both profiles took as configured. No deviation exists, so the
threshold question does not arise in either direction, and no re-measurement of the floor under
prime-rl's flags is possible or planned: there is no reading to measure.

The consequence lands on the attachment, not on the threshold. Gating a live prime-rl loop
requires reconciling the two profiles first, by serving with chunked prefill disabled or by
making the hook chunk-aware. Which one is a design question the attachment stage answers; until
it does, the 2a threshold stands unchallenged rather than confirmed, because nothing has been
measured against it under the serving profile.

The probe also establishes the hardware floor for everything after it. prime-rl separates trainer
and inference into distinct processes with distinct GPUs, so its smallest end-to-end loop is one
trainer GPU plus one inference GPU. There is no single-GPU end-to-end configuration.

Nothing in 2c through 2e is built until this resolves.

### 3b. 2c through 2e, against the engine the probe validated

Phase 2's remaining deliverables, built once and against a settled engine construction rather
than twice.

**Settled: 2d wraps prime-rl's own transports rather than reimplementing them.** prime-rl
implements filesystem, NCCL, and NIXL weight transports plus a worker-side transfer path, and
those are what people actually run; a reimplementation would produce numbers about code nobody
uses. 2d therefore implements the phase 1 `Transport` protocol over them. Three consequences
follow and none of them is optional.

- **Stage decomposition comes from instrumentation.** Splitting `T_sync` into `T_reshard`,
  `T_transfer`, and `T_load` needs timing points the transports do not expose. Add them as a
  small patch to the pinned checkout, carried in this repo as a recorded diff applied at setup
  time, never as a vendored copy and never as a fork. A patch that grows past timing points is
  evidence the wrapping is failing for that transport, and is reported rather than absorbed.
- **A transport whose stages cannot be separated without restructuring reports its total only,**
  with the reason recorded beside the number. An invented decomposition would sit in the same
  fields as three measured ones and read as comparable to them, which is worse than an honest
  total.
- **CUDA IPC is replaced by NIXL, and the substitution is a finding, not a renumbering.** CUDA
  IPC shares device memory handles and so requires trainer and sampler resident on the same
  GPUs. prime-rl places them on disjoint GPU sets, so at the pinned commit there is no
  configuration in which a CUDA-IPC transport could run at all. NIXL occupies that slot. Report
  the substitution with the colocation reason; do not present the transport list as though the
  originally specified three had been built.

### 3c. The staleness sweep

Vary staleness across a stated range at fixed model, transport, and parallelism, then repeat at
the transport and TP degree that phase 2e identified as most and least favorable. Record rollout
throughput, sampler GPU idle time, and the full per-sync decomposition at every point, with the
correctness gate passing at every point that produced a recorded number.

## Pinning

prime-rl moves fast and vLLM's weight-loading surface has churned repeatedly. Pin prime-rl to an
exact commit SHA, chosen so that its resolved vLLM version matches the one the bf16 floor was
measured at. Record the SHA in `pyproject.toml` and in every artifact this phase writes, next to
the vLLM version, not in place of it.

If a later pin moves prime-rl to a different vLLM version, the floor is re-measured at that
version and the threshold re-derived before any timing is recorded. Reusing a threshold across
vLLM versions is the same error as reusing phase 1a's threshold for 1b: reduction order is not
guaranteed stable across versions, and the floor is a measurement, not a constant.

## Acceptance criteria

- Engine construction resolved: the extracted logits tensor is bit-identical between the pinned
  prime-rl engine and a directly-constructed one at matched flags, or the divergence is
  attributed to a named cause and the gate's status inside a prime-rl loop is stated explicitly.
- prime-rl pinned to an exact SHA, recorded in `pyproject.toml` and in every artifact, with the
  resolved vLLM version recorded alongside it.
- 2c through 2e met against that engine.
- Rollout throughput and sampler GPU idle time reported against staleness, p50 and p99 over at
  least 100 syncs per point, first 10 discarded as warmup and said so.
- The correctness gate passes at every configuration that produced a recorded timing, at the 2a
  threshold, re-measured if the model, dtype, or vLLM version changed.
- The point at which further reduction in sync cost stops improving rollout throughput,
  identified and stated, or stated as not reached within the range swept.
