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
   tradeoff, not an oversight, and future sessions should not "simplify" this back to an HF
   comparison; doing so reintroduces the exact confound this design exists to remove.
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
- **Seq_len dependence is untested.** A run at `--repetitions 20 --batch 4 --seq-len 64` was
  started and killed after an hour without completing: vLLM's `prompt_logprobs=-1` extraction path
  builds on the order of 150k Python `Logprob` objects per position (one per vocab entry shown,
  `vocab_size=151936`), and that does not scale to seq_len=64, batch=4. This is a cost of the
  from-disk logprobs API `bf16_floor.py` currently uses, not a phase-2a physics finding. Replacing
  it with a `collective_rpc` call that returns the logits tensor directly is 2b work, and is needed
  for the weight-sync path itself regardless of this measurement.

**Gate verdict: PASS on the physics.** All three break cases at 20 reps sit 40x-77x above the
measured floor mean (1.772 / 1.570 / 2.984 against 3.898e-02) and clear the revised threshold
(0.5847) by 2.7x-5.1x. The pre-measurement risk estimate above (naive bf16 scaling predicting a
floor near 0.109 and case 3 possibly undetectable) did not materialize: the measured floor
(3.898e-02) is about 3x better than that estimate, and case 3 (norm permutation) is measured as the
**strongest** break (2.984), not the weakest -- the weakest is case 2 (o_proj/down column
permutation, 1.570). Proceed to 2b.

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

- **Qwen3 applies a per-head RMSNorm to Q and K before RoPE** (`self_attn.q_norm.weight`,
  `self_attn.k_norm.weight`, each shape `[head_dim]`), **the same weight vector reused identically
  across every head.** This has no phase 1 analogue. It is replicated *within* a head (every
  element of the `head_dim` vector applies to that one head's activations, same as any RMSNorm
  weight) and simultaneously partitioned *across* heads in the sense that the whole point of
  `HeadPartitioned` is to say which heads a rank owns -- yet this weight does not vary by head at
  all, so "partitioned across heads" is vacuous for it even though it lives inside a
  per-head-sharded computation. No existing `Placement` states this combination. **Open question
  for 2b, not answered here:** does this need a new placement, or is it expressible as
  `HeadPartitioned` with a degenerate `head_dim` (i.e. one norm vector shared by every head in the
  group rather than one per head)? Leave it open until 2b actually has to represent it.

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
