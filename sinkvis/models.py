"""Data models for SinkVis."""

from dataclasses import dataclass
from enum import Enum
from typing import List

import numpy as np


class EvictionPolicy(str, Enum):
    """KV cache eviction strategies."""

    FULL = "full"
    LRU = "lru"
    SLIDING_WINDOW = "sliding_window"
    STREAMING_LLM = "streaming_llm"
    H2O = "h2o"


class MemoryTier(str, Enum):
    """Memory hierarchy tiers."""

    GPU_HBM = "gpu_hbm"
    GPU_L2 = "gpu_l2"
    SYSTEM_RAM = "system_ram"
    DISK = "disk"


@dataclass
class SimulationConfig:
    """Configuration for eviction simulation."""

    policy: EvictionPolicy
    cache_size: int
    sink_count: int = 4
    window_size: int = 1024
    heavy_hitter_ratio: float = 0.1


@dataclass
class SimulationResult:
    """Results from eviction simulation."""

    policy: EvictionPolicy
    total_tokens_processed: int
    cache_hits: int
    cache_misses: int
    evictions: int
    retained_sinks: int
    retained_heavy_hitters: int
    final_cache: "CacheState"


@dataclass
class CacheState:
    """Current cache state."""

    total_tokens: int
    token_ids: List[int]
    is_sink: List[bool]
    is_heavy_hitter: List[bool]


@dataclass
class CacheBlock:
    """KV cache block metadata."""

    block_id: int
    start_token: int
    end_token: int
    size_bytes: int
    memory_tier: MemoryTier
    is_sink: bool
    is_heavy_hitter: bool
    access_count: int


@dataclass
class AttentionFrame:
    """Single frame of attention data."""

    layer: int
    head: int
    seq_len: int
    attention_weights: np.ndarray
    token_labels: List[str]
    sink_indices: List[int]
    heavy_hitter_indices: List[int]
    timestamp: float
