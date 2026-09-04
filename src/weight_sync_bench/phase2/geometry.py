"""`CheckpointGeometry`: real-checkpoint attention/FFN geometry.

Deliberately not `shardspec.ModelConfig`. `ModelConfig` hard-asserts
`n_heads * head_dim == d_model` (`ReferenceModel.__init__`, model.py:124) --
a real, load-bearing property of the phase 1 toy model (`TOY`/`TOY_KV4` both
hold it by construction) that must stay enforced there, not relaxed to make
room for a case it was never meant to cover. Qwen3-0.6B breaks it:
`d_model=1024, n_heads=16, head_dim=128` gives `16*128=2048 != 1024` --
Qwen3 sizes `head_dim` independently of `d_model`/`n_heads`, a real headroom
Megatron/Llama-style toy configs don't exercise (phase 1's toy has
`d_model=256, n_heads=8, head_dim=32`, where `8*32==256` holds and would
never surface this). Giving `ModelConfig` an escape hatch for Qwen3 is
exactly how the toy's assumption leaks into the real-model path; a separate
type is how it doesn't.

Three things `CheckpointGeometry` carries that `ModelConfig` has no field for
at all:

1. `head_dim` as its own field, with no equality ever asserted against
   `d_model`/`n_heads`. `q_size`/`kv_size` are `n_heads*head_dim`/
   `n_kv_heads*head_dim` -- multiplication only, exactly what
   `reshard.py`'s `_head_rows` already does, never a division back to
   `head_dim`.

2. `fusion`: which checkpoint tensors vLLM's loader concatenates into which
   in-memory tensor, AND IN WHAT ROW ORDER, stated as data rather than
   implied by a placement's code. This is the fact SPEC.md 2b exists to pin
   down. `HeadPartitioned`/`_head_rows` (reshard.py) and `FusedPaired`
   hardcode a q-then-k-then-v / gate-then-up row order in Python control
   flow -- correct for vLLM's `QKVParallelLinear`/`MergedColumnParallelLinear`
   at v0.28.0 (confirmed by reading `linear.py`'s `weight_loader`/
   `weight_loader_v2`; see `qwen3_layout.py`'s citations), but that's
   Megatron-Core's convention as vLLM happens to have adopted it, not a law
   every loader obeys. A loader that fused k-then-q-then-v, or interleaved
   per head, would make the exact same `HeadPartitioned(n_heads, n_kv_heads,
   head_dim)` placement silently produce wrong shards, because the row
   order lives in `reshard.py`'s code, not in any data `HeadPartitioned`
   carries. `fusion` makes the assumption an explicit, checkable value:
   `qwen3_layout.py` asserts `geometry.fusion["qkv_proj"] ==
   ("q_proj", "k_proj", "v_proj")` before it ever builds a `HeadPartitioned`
   from this geometry, so a future checkpoint whose loader fuses in a
   different order fails loudly at table-construction time instead of
   quietly mis-sharding. Vocabulary matches vLLM's own
   `Qwen3ForCausalLM.packed_modules_mapping`
   (vllm/model_executor/models/qwen3.py) -- same fact, made an explicit
   constructor argument here so a table-builder can read and check it
   without importing vllm.

3. Checkpoint-side (on-disk, unfused) shapes kept separate from loader-side
   (fused, in vLLM's memory, per-rank) shapes -- SPEC.md 2b's finding that
   "there is no single LayoutTable that is 'the' Qwen3 layout independent of
   which side of the load you mean." `checkpoint_shapes()` (TP-independent;
   the checkpoint has no TP degree) and `fused_shapes(tp)` /
   `global_shapes(tp)` compute each side from the same seven numbers, under
   different names, so code that needs one can never silently read the
   other.

`checkpoint_shapes()`'s numbers were checked against the real
`Qwen/Qwen3-0.6B` checkpoint -- not assumed, not read from vLLM source, but
fetched directly from the published `model.safetensors` file's own header
(a plain HTTP byte-range GET for the first 8 bytes to get the header
length, then that many bytes as JSON -- no GPU, no vllm, no `safetensors`
package needed, since the safetensors format's header is self-describing
stdlib-readable JSON). All eleven layer-0 tensor shapes and all three global
tensor shapes matched this module's formulas exactly, and the checkpoint
has exactly `28*11 + 3 = 311` tensors, matching `n_layers=28` times 11
per-layer tensors (no q/k/v/o bias tensors -- Qwen3-0.6B's
`attention_bias=false`) plus the 3 globals, confirming no hidden tensor this
geometry doesn't account for. One finding from that fetch with no phase-1
analogue: `lm_head.weight` (`[151936, 1024]`) is PHYSICALLY PRESENT on disk
as its own tensor, byte-identical in shape to `model.embed_tokens.weight` --
`tie_word_embeddings=True` does not mean the checkpoint omits a redundant
copy, only that vLLM's loader (`qwen3.py`'s `skip_prefixes=["lm_head."]`)
discards it at load time and ties the in-memory module to
`embed_tokens.weight` instead. `checkpoint_shapes()` reports the on-disk
fact (`lm_head.weight` present); `global_shapes(tp)` reports the in-memory
fact (one tensor, referenced by two names) -- another instance of point 3
above, checked rather than assumed.

`fused_shapes()`/`global_shapes()`'s loader-side predictions were, at the
time the paragraph above was written, checked against nothing but vLLM
source (linear.py's `weight_loader`/`weight_loader_v2`) -- unlike
`checkpoint_shapes()`, which already had the safetensors-header check.
**Since confirmed**: a real GPU run (`tolerance/phase2b_layout.json`) matched
every `fused_shapes(tp)`/`global_shapes(tp)` prediction against vLLM's
actual per-rank tensors at TP in {1, 2, 4} -- 227/227 parameter names,
exact shape equality, no mismatches -- AND confirmed the predicted BYTES,
not just shapes: `reshard.split_tensor` applied to the real TP=1 tensor
reproduced the real TP=2/TP=4 rank-local tensor bit-exactly
(`torch.equal`, `max_abs_diff=0.0`) for both `qkv_proj` and `gate_up_proj`,
at every rank. See `qwen3_layout.py`'s module docstring and
`tolerance/phase2b_layout.json` for the full record; SPEC.md 2b's
deliverable is met.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CheckpointGeometry:
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int  # independent of d_model; never d_model // n_heads
    ffn: int
    vocab: int
    tie_word_embeddings: bool = False
    # fused in-memory tensor name -> ordered tuple of the checkpoint tensor
    # names concatenated (in that row order) to build it. See module
    # docstring point 2.
    fusion: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_kv_heads <= 0 or self.n_heads % self.n_kv_heads:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads "
                f"({self.n_kv_heads}) -- GQA requires an integer number of "
                "query heads per KV head. Unlike ModelConfig's "
                "n_heads*head_dim == d_model, this IS a load-bearing "
                "structural fact for any real GQA checkpoint, not a "
                "toy-model convenience, and stays enforced here."
            )

    @property
    def q_size(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.n_kv_heads * self.head_dim

    def checkpoint_shapes(self) -> dict[str, tuple[int, ...]]:
        """On-disk, per-tensor shapes for one representative layer, plus the
        three global tensors -- the checkpoint convention (separate
        q_proj/k_proj/v_proj, gate_proj/up_proj), before vLLM's loader
        fuses anything. TP-independent: the checkpoint itself has no TP
        degree. Verified against the real Qwen3-0.6B safetensors header --
        see module docstring."""
        return {
            "self_attn.q_proj.weight": (self.q_size, self.d_model),
            "self_attn.k_proj.weight": (self.kv_size, self.d_model),
            "self_attn.v_proj.weight": (self.kv_size, self.d_model),
            "self_attn.o_proj.weight": (self.d_model, self.q_size),
            "self_attn.q_norm.weight": (self.head_dim,),
            "self_attn.k_norm.weight": (self.head_dim,),
            "mlp.gate_proj.weight": (self.ffn, self.d_model),
            "mlp.up_proj.weight": (self.ffn, self.d_model),
            "mlp.down_proj.weight": (self.d_model, self.ffn),
            "input_layernorm.weight": (self.d_model,),
            "post_attention_layernorm.weight": (self.d_model,),
            "model.embed_tokens.weight": (self.vocab, self.d_model),
            "lm_head.weight": (self.vocab, self.d_model),
            "model.norm.weight": (self.d_model,),
        }

    def fused_shapes(self, tp: int) -> dict[str, tuple[int, ...]]:
        """Post-loader, in-memory PER-RANK shapes at TP degree `tp`, for one
        representative layer -- what HeadPartitioned/FusedPaired actually
        describe. CONFIRMED against a running vLLM at tp in {1,2,4}
        (`tolerance/phase2b_layout.json`): every shape here matched exactly,
        and reshard.split_tensor's output at these shapes matched the real
        tensor bit-exactly too (see qwen3_layout.py).

        GQA replication (KV head count < tp) is NOT modeled -- raises if
        `n_kv_heads` doesn't divide `tp` evenly, the same condition
        `HeadPartitioned.validate` checks. For Qwen3-0.6B (n_kv_heads=8)
        this never triggers at tp in {1,2,4}; CONFIRMED by execution at
        tp=4 specifically (4 Q heads + 2 KV heads per rank, bit-exact
        against the real loader) -- it would first matter at tp=16."""
        if self.n_kv_heads % tp:
            raise ValueError(
                f"n_kv_heads={self.n_kv_heads} not divisible by tp={tp}; "
                "KV replication is not modeled here (see HeadPartitioned)"
            )
        num_heads_local = self.n_heads // tp
        num_kv_heads_local = self.n_kv_heads // tp
        q_local = num_heads_local * self.head_dim
        kv_local = num_kv_heads_local * self.head_dim
        ffn_local = self.ffn // tp
        return {
            "self_attn.qkv_proj.weight": (q_local + 2 * kv_local, self.d_model),
            "self_attn.o_proj.weight": (self.d_model, q_local),
            "self_attn.q_norm.weight": (self.head_dim,),
            "self_attn.k_norm.weight": (self.head_dim,),
            "mlp.gate_up_proj.weight": (2 * ffn_local, self.d_model),
            "mlp.down_proj.weight": (self.d_model, ffn_local),
            "input_layernorm.weight": (self.d_model,),
            "post_attention_layernorm.weight": (self.d_model,),
        }

    def global_shapes(self, tp: int) -> dict[str, tuple[int, ...]]:
        """Post-loader, in-memory per-rank shapes of the three global
        (non-per-layer) tensors. CONFIRMED against a running vLLM at
        tp in {1,2,4} -- see `fused_shapes`'s docstring and
        `tolerance/phase2b_layout.json`.

        `vocab // tp` is exact only because `vocab=151936` is already a
        multiple of both `tp` (for tp in {1,2,4}) and vLLM's own vocab
        padding granularity (`DEFAULT_VOCAB_PADDING_SIZE=64` in
        vocab_parallel_embedding.py -- 151936 / 64 = 2374 exactly, so
        `pad_vocab_size` is a no-op for this checkpoint). This is a checked
        fact about THIS vocab size, not a general law -- a checkpoint whose
        vocab isn't already 64-aligned would get padded before the `// tp`
        split and this formula would be wrong for it.

        `lm_head.weight` and `model.embed_tokens.weight` get the SAME
        predicted shape here because `tie_word_embeddings=True` means they
        are the same in-memory tensor (module docstring point 3) -- this
        method states that as one shared prediction, not two coincidentally
        equal ones.
        """
        if self.vocab % tp:
            raise ValueError(f"vocab={self.vocab} not divisible by tp={tp}")
        vocab_local = self.vocab // tp
        embed_shape = (vocab_local, self.d_model)
        return {
            "model.embed_tokens.weight": embed_shape,
            "lm_head.weight": embed_shape,
            "model.norm.weight": (self.d_model,),
        }


QWEN3_0_6B = CheckpointGeometry(
    d_model=1024,
    n_layers=28,
    n_heads=16,
    n_kv_heads=8,
    head_dim=128,
    ffn=3072,
    vocab=151936,
    tie_word_embeddings=True,
    fusion={
        "qkv_proj": ("q_proj", "k_proj", "v_proj"),
        "gate_up_proj": ("gate_proj", "up_proj"),
    },
)
