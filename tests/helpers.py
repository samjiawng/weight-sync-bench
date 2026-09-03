"""Shared fixtures for the phase 1a test modules."""

from __future__ import annotations

import functools

import torch

from reshard_bench.model import ReferenceModel
from reshard_bench.shardspec import TOY, TOY_KV4, ModelConfig, supported_degrees

CONFIGS: dict[str, ModelConfig] = {"kv2": TOY, "kv4": TOY_KV4}
DEGREES = {name: supported_degrees(config) for name, config in CONFIGS.items()}

# Must match the shape the tolerance floor was measured at; asserted by
# test_reshard.test_token_shape_matches_what_the_floor_was_measured_at.
BATCH, SEQ_LEN = 2, 16


@functools.cache
def reference(name: str) -> ReferenceModel:
    return ReferenceModel(CONFIGS[name], seed=0)


@functools.cache
def tokens_for(name: str) -> torch.Tensor:
    return torch.randint(
        0,
        CONFIGS[name].vocab,
        (BATCH, SEQ_LEN),
        generator=torch.Generator().manual_seed(7),
    )
