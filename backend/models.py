"""Data models for SinkVis API."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EvictionPolicy(str, Enum):
    """Supported KV cache eviction policies."""
    
    LRU = "lru"
    SLIDING_WINDOW = "sliding_window"
    STREAMING_LLM = "streaming_llm"
    H2O = "h2o"
    FULL = "full"


class MemoryTier(str, Enum):
    """Memory hierarchy tiers for cache blocks."""
    
    GPU_HBM = "gpu_hbm"
    GPU_L2 = "gpu_l2"
    SYSTEM_RAM = "system_ram"
    DISK = "disk"


class AttentionFrame(BaseModel):
    """Single frame of attention data for streaming."""
    
    layer: int
    head: int
    seq_len: int
    attention_weights: list[list[float]]
    sink_indices: list[int] = Field(default_factory=list)
    heavy_hitter_indices: list[int] = Field(default_factory=list)
    token_labels: list[str] = Field(default_factory=list)
    timestamp: float


class CacheBlock(BaseModel):
    """Represents a block in the KV cache."""
    
    block_id: int
    token_range: tuple[int, int]
    memory_tier: MemoryTier
    size_bytes: int
    last_access: float
    access_count: int
    is_sink: bool = False
    is_heavy_hitter: bool = False


class CacheProfile(BaseModel):
    """Hierarchical cache profile snapshot."""
    
    total_tokens: int
    blocks: list[CacheBlock]
    memory_usage: dict[str, int]
    eviction_policy: EvictionPolicy
    timestamp: float


class SimulationConfig(BaseModel):
    """Configuration for eviction simulation."""
    
    policy: EvictionPolicy
    cache_size: int = Field(default=2048, ge=1, le=131072)
    sink_count: int = Field(default=4, ge=0, le=64)
    window_size: int = Field(default=1024, ge=1, le=65536)
    heavy_hitter_ratio: float = Field(default=0.1, ge=0.0, le=1.0)


class SimulationResult(BaseModel):
    """Result from running an eviction simulation."""
    
    policy: EvictionPolicy
    total_tokens_processed: int
    cache_hits: int
    cache_misses: int
    evictions: int
    retained_sinks: int
    retained_heavy_hitters: int
    attention_frames: list[AttentionFrame]
    final_cache: CacheProfile


class PromptRequest(BaseModel):
    """Request to simulate a prompt through the cache."""
    
    prompt: str
    config: SimulationConfig
    generate_tokens: int = Field(default=128, ge=1, le=4096)


class StreamConfig(BaseModel):
    """Configuration for live attention streaming."""
    
    layers: Optional[list[int]] = None
    heads: Optional[list[int]] = None
    update_interval_ms: int = Field(default=100, ge=16, le=5000)
    highlight_sinks: bool = True
    highlight_heavy_hitters: bool = True
    sink_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    heavy_hitter_threshold: float = Field(default=0.05, ge=0.0, le=1.0)

