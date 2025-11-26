"""Vectorized eviction simulation logic."""

from typing import Tuple

import numpy as np
import torch


def simulate_lru(attention: np.ndarray, budget: int) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate LRU eviction policy."""
    scores = attention.sum(axis=0)
    indices = np.argsort(scores)[-budget:]
    mask = np.zeros(attention.shape[1], dtype=bool)
    mask[indices] = True
    return mask, scores


def simulate_streaming_llm(
    seq_len: int, budget: int, sink_count: int = 4
) -> np.ndarray:
    """Simulate StreamingLLM (sink + window) policy."""
    mask = np.zeros(seq_len, dtype=bool)
    mask[:sink_count] = True
    window_start = max(sink_count, seq_len - (budget - sink_count))
    mask[window_start:] = True
    return mask


def simulate_h2o(
    attention: np.ndarray, budget: int, sink_count: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate H2O (Heavy-Hitter Oracle) policy."""
    scores = attention.sum(axis=0)
    mask = np.zeros(len(scores), dtype=bool)
    mask[:sink_count] = True
    remaining = budget - sink_count
    if remaining > 0:
        heavy_scores = scores[sink_count:]
        top_k = np.argsort(heavy_scores)[-remaining:]
        mask[sink_count + top_k] = True
    return mask, scores


def simulate_sliding_window(seq_len: int, window_size: int) -> np.ndarray:
    """Simulate sliding window policy."""
    mask = np.zeros(seq_len, dtype=bool)
    start = max(0, seq_len - window_size)
    mask[start:] = True
    return mask


def apply_eviction_mask(attention: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
    """Apply eviction mask to attention tensor."""
    device = attention.device
    mask_tensor = torch.from_numpy(mask).to(device)
    return attention[:, :, :, mask_tensor]
