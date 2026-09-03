# weight-sync-bench

A CPU-only harness for validating tensor-parallel resharding, by comparing sharded execution
against an unsharded reference.

```
uv sync
uv run pytest    # 90 tests, ~8s
```

A small transformer is resharded between tensor-parallel layouts and its logits are compared
against an unsharded reference. Three injected layout bugs each break the comparison by 338x to
2535x the threshold, which is itself 100x above the measured numerical floor of 1.669e-06.

## Why byte comparison doesn't work

The obvious check, splitting a parameter and reassembling it to confirm the bytes match, passes
regardless of whether the layout is correct. For any dimension `d`

```python
cat(chunk(X, n, dim=d), dim=d) == X
```

so the round trip returns the original tensor as long as the split and the reassembly use the
same layout. An implementation that shards `o_proj` on the wrong axis passes it.

A layout bug only becomes visible when a consumer runs under the declared layout, since the shard
dimension determines the result of the consuming matmul. That is why the repo contains a
tensor-parallel forward pass with real collectives rather than a set of tensor comparisons.

## The two fused tensors

Two of the parameters cannot be sharded by slicing along a dimension, and mis-handling either one
produces the largest deviations measured here.

### qkv

`[(n_heads + 2 * n_kv_heads) * head_dim, d_model]`, which at `n_heads=8`, `n_kv_heads=2`,
`head_dim=32` is `[384, 256]`:

```
qkv [384, 256]                    correct at t=2              naive chunk(2, dim=0)
                 rows
              +---------+
  Q  8 heads  |   0-255 |         rank0: 0-127                rank0: 0-191
              |         |         rank1: 128-255                     Q heads 0-5,
  K  2 heads  | 256-319 |         rank0: 256-287                     no K, no V
              |         |         rank1: 288-319
  V  2 heads  | 320-383 |         rank0: 320-351              rank1: 192-383
              +---------+         rank1: 352-383                     Q heads 6-7,
                                                                     all K, all V
```

Rank 0's correct shard is rows `0-127`, `256-287` and `320-351`, three disjoint ranges
reassembled into a fused local tensor. Slicing contiguously instead produces a tensor of the
right shape and dtype in which rank 0 holds Q heads with no matching K or V.

### gate_up

`[2 * ffn, d_model] = [1408, 256]`, gate stacked on up:

```
gate_up [1408, 256]               correct at t=2              naive chunk(2, dim=0)
                 rows
              +-----------+
  gate        |    0-703  |       rank0: 0-351                rank0: 0-703
              |           |       rank1: 352-703                     (all gate,
  up          |  704-1407 |       rank0: 704-1055                     no up)
              |           |       rank1: 1056-1407
              +-----------+                                   rank1: 704-1407
                                                                     (all up,
                                                                      no gate)
```

Rank 0 needs its slice of gate paired with the corresponding slice of up so that
`silu(gate) * up` multiplies matching neurons, which again means two disjoint ranges rather than
one contiguous block.

Some frameworks avoid this by storing the fused tensor pre-permuted into head-interleaved order,
so that contiguous slicing happens to produce the right shard. That depends on how the tensor was
written, so an implementation relying on it breaks against a differently ordered checkpoint.

### Placements

DTensor supplies `Shard(dim)`, `Replicate()` and `Partial()`. Neither fused split is expressible
in that vocabulary, since in both cases the rank-local shard is a non-contiguous selection of
source rows. The harness adds two placements:

| placement | covers |
|---|---|
| `HeadPartitioned(n_heads, n_kv_heads, head_dim)` | fused qkv, per-head split across three row ranges |
| `FusedPaired(first, second)` | fused gate/up, paired split across two row ranges |

`Placement` is a sum type rather than a record with independent `partition` and `replicated`
fields, since those describe the same axis and a record would let both be set at once.

## Invariant

```
ReferenceModel(full_params, x) == ShardedModel(reshard(full_params, src -> dst), x)
```

Checked across every ordered pair of TP degrees in phase 1a, which runs all ranks in one process
with simulated collectives, and per degree in phase 1b, which runs one process per rank over
gloo. Phase 1b needs only per-degree coverage because `gather(split(full, src)) == full`
byte-exactly, so logits under the destination layout cannot depend on the source.

Both phases run the same forward pass. Activations are `list[Tensor]` indexed by local rank;
`InProcessCollective` holds every rank and reduces by sum or cat, while `GlooCollective` holds one
rank and calls `dist.all_reduce`. The forward body is byte-identical between them.

The toy model has 19.2M parameters (`d_model` 256, 4 layers, 8 heads, 2 KV heads, SwiGLU,
RMSNorm, GQA, untied embeddings, no biases) and exists only to contain one instance of each
layout.

## Injected bugs

| break | injection | deviation | vs threshold |
|---|---|---|---|
| 1. contiguous qkv slice | slice fused qkv on dim 0 instead of per head | 2.098 - 2.314 | 2097x - 2313x |
| 2. o_proj misassignment | shard dim 1 correctly, reverse the rank-to-slice assignment | 2.053 - 2.535 | 2052x - 2535x |
| 2. down misassignment | same, on the other row-parallel tensor | 1.143 - 1.305 | 1142x - 1304x |
| 3. permuted norm weights | rotate each rank's norm weight by rank x shard width | 0.338 - 0.413 | 338x - 412x |

Ranges span both configs at every TP degree of 2 or more. At t=1 each injection is the identity,
so none are run there.

Every injection preserves shapes, because a shape error gets caught by torch rather than by the
invariant and would pass even if `ShardedModel` computed nothing at all. The obvious form of case
2, sharding a row-parallel tensor on dim 0, is a shape error for that reason, so the version here
shards dim 1 correctly and reverses the rank-to-slice assignment, leaving the all-reduce to sum
products of mismatched column blocks. Case 3 originally zeroed the entries a rank would not own,
which measured 1.65 to 1.83 against the permutation's 0.34 to 0.41, meaning four to five times
the deviation came from destroyed information rather than from the layout error.

### Break tests are not sufficient on their own

They assert that deviation is large, and a broken `ShardedModel` also produces large deviation.
Zeroing `o_proj`'s output leaves all three break cases passing while 18 invariant tests fail.
Both suites are needed, since the invariant tests establish that deviation stays small when the
layout is correct and the break tests establish that it grows when the layout is wrong.

## Tolerance floor

A row-parallel matmul reduces partial products across ranks, summing in a different order than
the full matmul, so correct sharded output differs from the reference in the low bits. The
threshold has to admit that reordering while still catching a layout bug, so it is measured
rather than guessed.

20 repetitions, varied model and token seeds, in units of 2^-23:

| destination cell | 1a max | 1b max | median | mean |
|---|---|---|---|---|
| kv2, t=2 | 14 | 14 | 1.15 | 1.39 |
| kv4, t=2 | 13 | 13 | 1.14 | 1.38 |
| kv4, t=4 | 13 | 12 | 1.14 | 1.38 |

Destination t=1 deviates by exactly 0, since a 1-way split reorders nothing. The worst observed
value is 1.669e-06 in both phases, and the threshold is 1e-3, derived by the rule *smallest power
of ten at or above 100x the worst observation* and applied by a pure function of the measurement.
Both phases land on the same threshold because of that rounding rule rather than because the
measurements agree; kv4 at t=4 measures 13 ULP in-process against 12 under gloo, so neither
threshold may be reused for the other phase.

The floor is specific to the model geometry and the logits tensor it was measured on. It is the
maximum over `batch * seq_len * vocab` elements of a per-element error distribution, and the
maximum of roughly 1e6 draws is a tail statistic on a quantized grid, with the max-deviating
element usually not being the largest logit. It depends on `d_model` and `n_layers`, which set
reduction length and accumulation depth, and on `batch` and `seq_len`, which set how many draws
the maximum is taken over. Deviation is roughly linear in `d_model`, at 11.5, 25 and 50 ULP for
256, 512 and 1024. Going from 5 to 20 repetitions moved the maximum from 11.5 to 14 ULP while
median and mean held steady.

Raising `seq_len` in a fixture would widen the floor without looking like a change to it, so the
running config geometry and token shape are asserted against the recorded values and fail on
mismatch. `median_ulp` and `mean_ulp` are recorded per cell because they do not saturate.

```
uv run python -m weight_sync_bench.tolerance              # tolerance/phase1a.json
uv run python -m weight_sync_bench.tolerance --phase 1b   # tolerance/phase1b.json
```

Artifacts record torch, numpy, python, platform, dtype and commit.

## Two configs

| config | `n_kv_heads` | degrees | covers |
|---|---|---|---|
| `TOY` | 2 | 1, 2 | the unrepresentable layout at t=4 |
| `TOY_KV4` | 4 | 1, 2, 4 | a real 4-way forward, `qkv` included |

`TOY`'s `n_kv_heads=2` is unrepresentable at t=4, so it never exercises a 4-way split of
anything. `TOY_KV4` supplies that, and 4-way splits of the row-parallel and vocab-parallel
tensors are where case 2 has the most room to hide. Supported degrees are derived by probing each
placement rather than hardcoded, so the sweep follows the config.

## GQA at TP=4

With `n_kv_heads=2` and `t=4` there are fewer KV heads than ranks, so either the KV heads get
replicated across ranks, which is what Megatron-Core and vLLM do, or the layout gets rejected,
which is what this harness does.

Replication is an execution answer and this is a layout validation harness, so implementing it
would leave the `LayoutTable` saying "split by head" while the rank-local bytes say "replicated",
a discrepancy the table has no vocabulary to express. The rejection lives in
`HeadPartitioned.validate`, so the placement cannot be constructed at an unsupported degree at
all. If phase 2 needs replication against a real vLLM engine, it should arrive as a new placement
that states the replication rather than as a deletion of the check.

## Limits

`GlooTransport` scatters shards outward from rank 0, which holds the full tensors, so nothing
here reshards between two layouts that both live across processes with no full tensor anywhere,
which is the expensive case and phase 2 work.

There is no GPU, CUDA, NCCL or CUDA IPC path, and `ParamSink` is a dict rather than an inference
engine. The timing fields on `SyncRecord` exist so that phase 2 has somewhere to write them; at
19M parameters over loopback gloo they measure process startup and memory bandwidth.

On failure the harness reports the first mismatching parameter, the source and destination
degrees, and the max absolute logit error. It does not try to infer which wrong layout produced
the observed shards.

## Modules

```
src/weight_sync_bench/
  shardspec.py    placements, ShardSpec, LayoutTable, UnsupportedLayout
  model.py        ReferenceModel
  sharded.py      ShardedModel, InProcessCollective, GlooCollective
  reshard.py      split / gather / reshard
  distributed.py  phase 1b launcher
  transport.py    SyncRecord, Transport, ParamSink, GlooTransport
  tolerance.py    floor measurement and threshold derivation
```

## Phase 2

`HeadPartitioned` and `FusedPaired` are where a framework's storage convention gets encoded, so
Megatron-Core's fused qkv ordering belongs in the `LayoutTable` rather than scattered through the
resharder. The invariant is unchanged on GPU, though the floor has to be re-measured since bf16
and GPU reduction order both move it. The one assumption that does not survive is that a full
tensor fits on one rank, which `GlooTransport` currently relies on.
