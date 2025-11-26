"""Attention pattern generation and analysis."""

from typing import List

import numpy as np

from .models import AttentionFrame, CacheBlock, MemoryTier


def generate_attention_pattern(seq_len: int, num_sinks: int = 4) -> np.ndarray:
    """Generate realistic attention pattern with sinks."""
    attention = np.random.rand(seq_len, seq_len)
    attention = np.tril(attention)
    for i in range(num_sinks):
        attention[:, i] += 0.5
    attention = np.tril(attention)  # Re-apply causal mask after sink weights
    attention = attention / attention.sum(axis=1, keepdims=True)
    return attention


def identify_sinks(attention: np.ndarray, threshold: float = 0.1) -> List[int]:
    """Identify attention sink positions."""
    avg_attention = attention.mean(axis=0)
    return np.where(avg_attention > threshold)[0].tolist()


def identify_heavy_hitters(
    attention: np.ndarray,
    threshold: float = 0.05,
    exclude_sinks: List[int] = None,
) -> List[int]:
    """Identify heavy hitter positions."""
    exclude_sinks = exclude_sinks or []
    avg_attention = attention.mean(axis=0)
    heavy = np.where(avg_attention > threshold)[0].tolist()
    return [h for h in heavy if h not in exclude_sinks]


def create_attention_frame(
    attention: np.ndarray,
    tokens: List[str],
    layer: int = 0,
    head: int = 0,
    timestamp: float = 0.0,
) -> AttentionFrame:
    """Create attention frame with analysis."""
    sinks = identify_sinks(attention)
    heavy = identify_heavy_hitters(attention, exclude_sinks=sinks)
    return AttentionFrame(
        layer=layer,
        head=head,
        seq_len=len(tokens),
        attention_weights=attention,
        token_labels=tokens,
        sink_indices=sinks,
        heavy_hitter_indices=heavy,
        timestamp=timestamp,
    )


def generate_cache_blocks(
    seq_len: int,
    block_size: int = 16,
    sink_indices: List[int] = None,
    heavy_hitter_indices: List[int] = None,
) -> List[CacheBlock]:
    """Generate cache blocks with memory tier assignments."""
    sink_indices = sink_indices or []
    heavy_hitter_indices = heavy_hitter_indices or []
    blocks = []
    num_blocks = (seq_len + block_size - 1) // block_size
    for i in range(num_blocks):
        start = i * block_size
        end = min(start + block_size, seq_len)
        is_sink = any(s >= start and s < end for s in sink_indices)
        is_heavy = any(h >= start and h < end for h in heavy_hitter_indices)
        if is_sink or is_heavy:
            tier = MemoryTier.GPU_HBM
        elif i < num_blocks * 0.7:
            tier = MemoryTier.GPU_L2
        elif i < num_blocks * 0.9:
            tier = MemoryTier.SYSTEM_RAM
        else:
            tier = MemoryTier.DISK
        blocks.append(
            CacheBlock(
                block_id=i,
                start_token=start,
                end_token=end,
                size_bytes=(end - start) * 128,
                memory_tier=tier,
                is_sink=is_sink,
                is_heavy_hitter=is_heavy,
                access_count=10 if is_sink else 5 if is_heavy else 1,
            )
        )
    return blocks
