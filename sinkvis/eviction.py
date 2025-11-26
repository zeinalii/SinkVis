"""KV cache eviction policy implementations."""

from typing import List

import numpy as np

from .models import CacheState, EvictionPolicy, SimulationConfig, SimulationResult


def tokenize_simple(text: str) -> List[int]:
    """Simple tokenization for simulation."""
    words = text.split()
    return [hash(w) % 50000 for w in words]


def _evict_lru(
    cache: List[int], last_used: List[int], capacity: int
) -> tuple[List[int], List[int]]:
    """LRU eviction: remove least recently used."""
    if len(cache) <= capacity:
        return cache, last_used
    indices = np.argsort(last_used)
    keep = indices[-capacity:]
    return [cache[i] for i in keep], [last_used[i] for i in keep]


def _evict_sliding_window(cache: List[int], window_size: int) -> List[int]:
    """Keep only last N tokens."""
    return cache[-window_size:]


def _evict_streaming_llm(
    cache: List[int], sink_count: int, window_size: int
) -> List[int]:
    """Preserve sinks + sliding window."""
    if len(cache) <= sink_count + window_size:
        return cache
    sinks = cache[:sink_count]
    window = cache[-(window_size):]
    return sinks + window


def _evict_h2o(
    cache: List[int],
    attention_scores: np.ndarray,
    sink_count: int,
    capacity: int,
) -> List[int]:
    """H2O: preserve sinks + heavy hitters."""
    if len(cache) <= capacity:
        return cache
    sinks = cache[:sink_count]
    remaining = capacity - sink_count
    scores = attention_scores[sink_count:]
    top_k = np.argsort(scores)[-remaining:]
    heavy = [cache[sink_count + i] for i in sorted(top_k)]
    return sinks + heavy


def run_simulation(
    prompt: str, config: SimulationConfig, generate_tokens: int = 32
) -> SimulationResult:
    """Run eviction simulation."""
    tokens = tokenize_simple(prompt)
    total_tokens = len(tokens) + generate_tokens
    cache = []
    hits = 0
    misses = 0
    evictions = 0
    last_used = []
    attention = np.random.rand(total_tokens)
    for i in range(total_tokens):
        if i < len(tokens):
            token = tokens[i]
        else:
            token = hash(f"gen_{i}") % 50000
        if token in cache:
            hits += 1
            idx = cache.index(token)
            last_used[idx] = i
        else:
            misses += 1
            cache.append(token)
            last_used.append(i)
        if len(cache) > config.cache_size:
            evictions += len(cache) - config.cache_size
            if config.policy == EvictionPolicy.LRU:
                cache, last_used = _evict_lru(cache, last_used, config.cache_size)
            elif config.policy == EvictionPolicy.SLIDING_WINDOW:
                cache = _evict_sliding_window(cache, config.window_size)
                last_used = last_used[-len(cache) :]
            elif config.policy == EvictionPolicy.STREAMING_LLM:
                cache = _evict_streaming_llm(
                    cache, config.sink_count, config.window_size
                )
                last_used = last_used[-len(cache) :]
            elif config.policy == EvictionPolicy.H2O:
                cache = _evict_h2o(
                    cache, attention, config.sink_count, config.cache_size
                )
                last_used = last_used[-len(cache) :]
    sinks = min(config.sink_count, len(cache))
    heavy = int(len(cache) * config.heavy_hitter_ratio)
    final_state = CacheState(
        total_tokens=len(cache),
        token_ids=cache,
        is_sink=[i < sinks for i in range(len(cache))],
        is_heavy_hitter=[False] * len(cache),
    )
    return SimulationResult(
        policy=config.policy,
        total_tokens_processed=total_tokens,
        cache_hits=hits,
        cache_misses=misses,
        evictions=evictions,
        retained_sinks=sinks,
        retained_heavy_hitters=heavy,
        final_cache=final_state,
    )

