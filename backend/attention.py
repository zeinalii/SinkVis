"""Attention pattern generation and analysis."""

import math
import random
from typing import Optional

import numpy as np

from .models import AttentionFrame, MemoryTier, CacheBlock


def generate_attention_pattern(
    seq_len: int,
    num_sinks: int = 4,
    pattern_type: str = "realistic"
) -> np.ndarray:
    """
    Generate synthetic attention patterns that mimic real LLM behavior.
    
    Args:
        seq_len: Sequence length for attention matrix
        num_sinks: Number of attention sink positions
        pattern_type: Type of pattern ("realistic", "uniform", "local")
    
    Returns:
        Attention weight matrix of shape (seq_len, seq_len)
    """
    attention = np.zeros((seq_len, seq_len))
    
    if pattern_type == "uniform":
        for i in range(seq_len):
            attention[i, :i + 1] = 1.0 / (i + 1)
        return attention
    
    for i in range(seq_len):
        if i == 0:
            attention[i, 0] = 1.0
            continue
        
        weights = np.zeros(i + 1)
        
        # Attention sinks - first few tokens get disproportionate attention
        sink_weight = 0.3 + random.uniform(0, 0.2)
        for s in range(min(num_sinks, i + 1)):
            weights[s] = sink_weight / num_sinks * (1 - s * 0.1)
        
        # Local attention - recent tokens
        local_window = min(64, i + 1)
        local_start = max(0, i - local_window + 1)
        for j in range(local_start, i + 1):
            distance = i - j
            local_weight = 0.4 * math.exp(-distance / (local_window / 3))
            weights[j] += local_weight
        
        # Heavy hitters - semantically important tokens
        num_heavy = max(1, int((i + 1) * 0.05))
        available_positions = list(range(num_sinks, i + 1))
        if available_positions and num_heavy > 0:
            heavy_positions = random.sample(available_positions, min(num_heavy, len(available_positions)))
        else:
            heavy_positions = []
        for h in heavy_positions:
            weights[h] += 0.1 + random.uniform(0, 0.1)
        
        # Add noise
        weights += np.random.exponential(0.01, i + 1)
        
        # Normalize
        weights = weights / weights.sum()
        attention[i, :i + 1] = weights
    
    return attention


def identify_sinks(
    attention: np.ndarray,
    threshold: float = 0.1
) -> list[int]:
    """
    Identify attention sink positions.
    
    Sinks are tokens that receive high attention across many query positions.
    """
    seq_len = attention.shape[0]
    if seq_len < 2:
        return [0] if seq_len == 1 else []
    
    # Average attention received by each position
    avg_attention = np.zeros(seq_len)
    for i in range(seq_len):
        avg_attention[i] = attention[i:, i].mean() if i < seq_len else 0
    
    # Find positions with above-threshold average attention
    sinks = []
    for i in range(min(16, seq_len)):  # Sinks typically in first few positions
        if avg_attention[i] > threshold:
            sinks.append(i)
    
    return sinks


def identify_heavy_hitters(
    attention: np.ndarray,
    threshold: float = 0.05,
    exclude_sinks: Optional[list[int]] = None
) -> list[int]:
    """
    Identify heavy hitter positions.
    
    Heavy hitters are semantically important tokens that receive
    consistently high attention but aren't attention sinks.
    """
    exclude_sinks = exclude_sinks or []
    seq_len = attention.shape[0]
    
    if seq_len < 2:
        return []
    
    # Compute attention received, excluding self-attention
    received = np.zeros(seq_len)
    for i in range(seq_len):
        col = attention[i + 1:, i] if i < seq_len - 1 else np.array([])
        received[i] = col.mean() if len(col) > 0 else 0
    
    # Find heavy hitters
    heavy_hitters = []
    for i in range(seq_len):
        if i not in exclude_sinks and received[i] > threshold:
            heavy_hitters.append(i)
    
    return heavy_hitters


def create_attention_frame(
    attention: np.ndarray,
    layer: int,
    head: int,
    tokens: list[str],
    timestamp: float,
    sink_threshold: float = 0.1,
    heavy_hitter_threshold: float = 0.05
) -> AttentionFrame:
    """Create an AttentionFrame from raw attention weights."""
    sinks = identify_sinks(attention, sink_threshold)
    heavy_hitters = identify_heavy_hitters(attention, heavy_hitter_threshold, sinks)
    
    return AttentionFrame(
        layer=layer,
        head=head,
        seq_len=attention.shape[0],
        attention_weights=attention.tolist(),
        sink_indices=sinks,
        heavy_hitter_indices=heavy_hitters,
        token_labels=tokens[:attention.shape[0]],
        timestamp=timestamp
    )


def generate_cache_blocks(
    seq_len: int,
    block_size: int = 16,
    sink_indices: Optional[list[int]] = None,
    heavy_hitter_indices: Optional[list[int]] = None,
    base_timestamp: float = 0.0
) -> list[CacheBlock]:
    """Generate cache block representations for visualization."""
    sink_indices = set(sink_indices or [])
    heavy_hitter_indices = set(heavy_hitter_indices or [])
    
    blocks = []
    num_blocks = (seq_len + block_size - 1) // block_size
    
    for i in range(num_blocks):
        start = i * block_size
        end = min(start + block_size, seq_len)
        
        # Determine if block contains sinks or heavy hitters
        block_tokens = set(range(start, end))
        has_sink = bool(block_tokens & sink_indices)
        has_heavy = bool(block_tokens & heavy_hitter_indices)
        
        # Assign memory tier based on recency and importance
        if has_sink:
            tier = MemoryTier.GPU_HBM
        elif i >= num_blocks - 4:  # Recent blocks
            tier = MemoryTier.GPU_HBM
        elif has_heavy:
            tier = MemoryTier.GPU_L2
        elif i >= num_blocks - 16:
            tier = MemoryTier.SYSTEM_RAM
        else:
            tier = MemoryTier.DISK
        
        blocks.append(CacheBlock(
            block_id=i,
            token_range=(start, end),
            memory_tier=tier,
            size_bytes=(end - start) * 2 * 128 * 4,  # 2 (K+V) * hidden_dim * float32
            last_access=base_timestamp - (num_blocks - i - 1) * 0.01,
            access_count=random.randint(1, 100) if has_sink or has_heavy else random.randint(1, 10),
            is_sink=has_sink,
            is_heavy_hitter=has_heavy
        ))
    
    return blocks

