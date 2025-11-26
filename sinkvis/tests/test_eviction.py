"""Tests for eviction.py - Cache eviction policy implementations.

Tests the simulation runner and individual eviction helper functions.
"""

import numpy as np
import pytest

from sinkvis.eviction import (
    _evict_h2o,
    _evict_lru,
    _evict_sliding_window,
    _evict_streaming_llm,
    run_simulation,
    tokenize_simple,
)
from sinkvis.models import EvictionPolicy, SimulationConfig


class TestTokenizeSimple:
    """Tests for simple tokenization."""

    def test_splits_on_whitespace(self):
        """Tokenization splits text on whitespace."""
        tokens = tokenize_simple("hello world test")
        assert len(tokens) == 3

    def test_empty_string(self):
        """Empty string produces empty token list."""
        tokens = tokenize_simple("")
        assert tokens == []

    def test_single_word(self):
        """Single word produces single token."""
        tokens = tokenize_simple("hello")
        assert len(tokens) == 1

    def test_tokens_are_integers(self):
        """All tokens are integers."""
        tokens = tokenize_simple("hello world test")
        assert all(isinstance(t, int) for t in tokens)

    def test_deterministic(self):
        """Same input produces same tokens."""
        tokens1 = tokenize_simple("hello world")
        tokens2 = tokenize_simple("hello world")
        assert tokens1 == tokens2

    def test_different_words_different_tokens(self):
        """Different words produce different tokens (usually)."""
        tokens = tokenize_simple("apple banana cherry")
        # Very unlikely all three hash to same value
        assert len(set(tokens)) >= 2


class TestEvictLRU:
    """Tests for LRU eviction helper."""

    def test_no_eviction_under_capacity(self):
        """No eviction when cache is under capacity."""
        cache = [1, 2, 3]
        last_used = [0, 1, 2]
        capacity = 5

        new_cache, new_used = _evict_lru(cache, last_used, capacity)

        assert new_cache == [1, 2, 3]
        assert new_used == [0, 1, 2]

    def test_evicts_least_recent(self):
        """Evicts least recently used tokens."""
        cache = [10, 20, 30, 40, 50]
        last_used = [5, 1, 4, 2, 3]  # 20 was used longest ago
        capacity = 3

        new_cache, new_used = _evict_lru(cache, last_used, capacity)

        assert len(new_cache) == 3
        assert 20 not in new_cache  # Least recent evicted
        assert 10 in new_cache  # Most recent kept

    def test_respects_capacity(self):
        """Result has exactly capacity items."""
        cache = list(range(10))
        last_used = list(range(10))
        capacity = 4

        new_cache, _ = _evict_lru(cache, last_used, capacity)

        assert len(new_cache) == capacity


class TestEvictSlidingWindow:
    """Tests for sliding window eviction helper."""

    def test_keeps_last_n_tokens(self):
        """Keeps exactly the last window_size tokens."""
        cache = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        window_size = 4

        result = _evict_sliding_window(cache, window_size)

        assert result == [7, 8, 9, 10]

    def test_window_larger_than_cache(self):
        """Returns all tokens when window >= cache size."""
        cache = [1, 2, 3]
        window_size = 10

        result = _evict_sliding_window(cache, window_size)

        assert result == [1, 2, 3]

    def test_window_equals_cache(self):
        """Returns all tokens when window == cache size."""
        cache = [1, 2, 3, 4, 5]
        window_size = 5

        result = _evict_sliding_window(cache, window_size)

        assert result == cache


class TestEvictStreamingLLM:
    """Tests for StreamingLLM eviction helper."""

    def test_preserves_sinks_and_window(self):
        """Keeps sink tokens and sliding window."""
        cache = list(range(1, 11))  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        sink_count = 2
        window_size = 3

        result = _evict_streaming_llm(cache, sink_count, window_size)

        # Should be sinks [1, 2] + window [8, 9, 10]
        assert result[:sink_count] == [1, 2]  # Sinks preserved
        assert result[-window_size:] == [8, 9, 10]  # Window preserved

    def test_no_eviction_small_cache(self):
        """No eviction when cache fits in sink + window."""
        cache = [1, 2, 3, 4, 5]
        sink_count = 2
        window_size = 4

        result = _evict_streaming_llm(cache, sink_count, window_size)

        assert result == cache

    def test_middle_tokens_evicted(self):
        """Middle tokens (between sinks and window) are evicted."""
        cache = list(range(1, 21))  # 1-20
        sink_count = 4
        window_size = 4

        result = _evict_streaming_llm(cache, sink_count, window_size)

        # Tokens 5-16 should be evicted
        assert len(result) == sink_count + window_size
        assert 5 not in result
        assert 15 not in result


class TestEvictH2O:
    """Tests for H2O eviction helper."""

    def test_preserves_sinks(self):
        """Always preserves sink tokens."""
        cache = list(range(10))
        attention_scores = np.random.rand(10)
        sink_count = 4
        capacity = 6

        result = _evict_h2o(cache, attention_scores, sink_count, capacity)

        # First 4 tokens (sinks) should be preserved
        assert result[:sink_count] == list(range(4))

    def test_keeps_heavy_hitters(self):
        """Keeps tokens with highest attention scores."""
        cache = list(range(10))
        # Token 7 has highest score outside sinks
        attention_scores = np.array([0.1] * 10)
        attention_scores[7] = 0.9
        sink_count = 2
        capacity = 4

        result = _evict_h2o(cache, attention_scores, sink_count, capacity)

        assert 7 in result  # Heavy hitter preserved

    def test_respects_capacity(self):
        """Result has at most capacity items."""
        cache = list(range(20))
        attention_scores = np.random.rand(20)
        sink_count = 4
        capacity = 10

        result = _evict_h2o(cache, attention_scores, sink_count, capacity)

        assert len(result) == capacity

    def test_no_eviction_small_cache(self):
        """No eviction when cache <= capacity."""
        cache = [1, 2, 3]
        attention_scores = np.random.rand(3)

        result = _evict_h2o(cache, attention_scores, sink_count=1, capacity=10)

        assert result == cache


class TestRunSimulation:
    """Tests for the full simulation runner."""

    def test_lru_policy(self):
        """Simulation with LRU policy runs successfully."""
        config = SimulationConfig(
            policy=EvictionPolicy.LRU,
            cache_size=10,
        )

        result = run_simulation("hello world test", config, generate_tokens=5)

        assert result.policy == EvictionPolicy.LRU
        assert result.total_tokens_processed > 0

    def test_sliding_window_policy(self):
        """Simulation with sliding window policy runs successfully."""
        config = SimulationConfig(
            policy=EvictionPolicy.SLIDING_WINDOW,
            cache_size=10,
            window_size=5,
        )

        result = run_simulation("hello world test", config, generate_tokens=5)

        assert result.policy == EvictionPolicy.SLIDING_WINDOW

    def test_streaming_llm_policy(self):
        """Simulation with StreamingLLM policy runs successfully."""
        config = SimulationConfig(
            policy=EvictionPolicy.STREAMING_LLM,
            cache_size=20,
            sink_count=4,
            window_size=8,
        )

        result = run_simulation("hello world test example", config, generate_tokens=10)

        assert result.policy == EvictionPolicy.STREAMING_LLM
        assert result.retained_sinks <= config.sink_count

    def test_h2o_policy(self):
        """Simulation with H2O policy runs successfully."""
        config = SimulationConfig(
            policy=EvictionPolicy.H2O,
            cache_size=15,
            sink_count=4,
        )

        result = run_simulation("hello world test", config, generate_tokens=10)

        assert result.policy == EvictionPolicy.H2O

    def test_result_has_valid_fields(self):
        """SimulationResult has all expected fields populated."""
        config = SimulationConfig(
            policy=EvictionPolicy.LRU,
            cache_size=10,
        )

        result = run_simulation("test prompt", config)

        assert result.total_tokens_processed > 0
        assert result.cache_hits >= 0
        assert result.cache_misses >= 0
        assert result.evictions >= 0
        assert result.final_cache is not None

    def test_cache_hits_misses_sum(self):
        """Cache hits + misses should equal total tokens processed."""
        config = SimulationConfig(
            policy=EvictionPolicy.LRU,
            cache_size=50,
        )

        result = run_simulation("the quick brown fox", config, generate_tokens=10)

        assert result.cache_hits + result.cache_misses == result.total_tokens_processed

    def test_evictions_triggered_when_over_capacity(self):
        """Evictions should occur when cache exceeds capacity."""
        config = SimulationConfig(
            policy=EvictionPolicy.LRU,
            cache_size=3,
        )

        result = run_simulation("one two three four five six", config)

        assert result.evictions > 0, "Should have evicted tokens"

    def test_final_cache_size_within_capacity(self):
        """Final cache should not exceed configured size."""
        config = SimulationConfig(
            policy=EvictionPolicy.LRU,
            cache_size=5,
        )

        result = run_simulation("many words here to process", config)

        assert result.final_cache.total_tokens <= config.cache_size

