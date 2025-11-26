"""Tests for eviction policy simulation."""

import pytest

from backend.models import EvictionPolicy, SimulationConfig
from backend.eviction import (
    LRUEviction,
    SlidingWindowEviction,
    StreamingLLMEviction,
    H2OEviction,
    FullCacheEviction,
    get_eviction_strategy,
    tokenize_simple,
    run_simulation,
    CacheEntry,
)


class TestTokenizeSimple:
    """Tests for tokenize_simple - basic text tokenization."""

    def test_tokenizes_real_sentence(self):
        tokens = tokenize_simple("The quick brown fox jumps.")
        assert tokens[0] == "<s>"
        assert tokens[1] == "<bos>"
        assert "quick" in tokens
        assert "." in tokens
        assert tokens[-1] == "<eos>"

    def test_handles_punctuation_separately(self):
        tokens = tokenize_simple("Hello, world!")
        assert "," in tokens
        assert "!" in tokens
        assert "Hello" in tokens

    def test_empty_string_has_special_tokens(self):
        tokens = tokenize_simple("")
        assert tokens == ["<s>", "<bos>", "<eos>"]


class TestLRUEviction:
    """Tests for LRU eviction - evicts least recently accessed."""

    def test_evicts_oldest_when_full(self):
        config = SimulationConfig(policy=EvictionPolicy.LRU, cache_size=3)
        lru = LRUEviction(config)
        
        lru.access(0, "The")
        lru.access(1, "cat")
        lru.access(2, "sat")
        lru.access(3, "on")  # Evicts position 0
        
        assert 0 not in lru.cache
        assert 3 in lru.cache

    def test_reaccessing_prevents_eviction(self):
        config = SimulationConfig(policy=EvictionPolicy.LRU, cache_size=3)
        lru = LRUEviction(config)
        
        lru.access(0, "A")
        lru.access(1, "B")
        lru.access(2, "C")
        lru.access(0, "A")  # Re-access makes 0 most recent
        lru.access(3, "D")  # Should evict 1, not 0
        
        assert 0 in lru.cache
        assert 1 not in lru.cache


class TestSlidingWindowEviction:
    """Tests for sliding window - keeps only recent N tokens."""

    def test_keeps_only_recent_window(self):
        config = SimulationConfig(
            policy=EvictionPolicy.SLIDING_WINDOW,
            cache_size=8, window_size=5
        )
        sw = SlidingWindowEviction(config)
        
        for i in range(20):
            sw.access(i, f"tok{i}")
        
        # Only last ~5 positions should remain
        assert 19 in sw.cache
        assert 0 not in sw.cache

    def test_window_respects_size(self):
        config = SimulationConfig(
            policy=EvictionPolicy.SLIDING_WINDOW,
            cache_size=50, window_size=10
        )
        sw = SlidingWindowEviction(config)
        
        for i in range(30):
            sw.access(i, f"t{i}")
        
        # Positions 20-29 should be in cache (window of 10 from position 29)
        for i in range(20, 30):
            assert i in sw.cache


class TestStreamingLLMEviction:
    """Tests for StreamingLLM - preserves sinks + sliding window."""

    def test_preserves_marked_sinks(self):
        config = SimulationConfig(
            policy=EvictionPolicy.STREAMING_LLM,
            cache_size=10, window_size=5
        )
        streaming = StreamingLLMEviction(config)
        
        streaming.access(0, "<s>")
        streaming.mark_sink(0)
        
        for i in range(1, 50):
            streaming.access(i, f"tok{i}")
        
        assert 0 in streaming.cache, "Sink should never be evicted"

    def test_evicts_non_sinks_normally(self):
        config = SimulationConfig(
            policy=EvictionPolicy.STREAMING_LLM,
            cache_size=8, window_size=4
        )
        streaming = StreamingLLMEviction(config)
        
        streaming.access(0, "<s>")
        streaming.mark_sink(0)
        for i in range(1, 20):
            streaming.access(i, f"t{i}")
        
        # Position 5 (not a sink, not recent) should be evicted
        assert 5 not in streaming.cache
        assert 0 in streaming.cache  # Sink remains


class TestH2OEviction:
    """Tests for H2O - preserves sinks and heavy hitters."""

    def test_preserves_heavy_hitters(self):
        config = SimulationConfig(policy=EvictionPolicy.H2O, cache_size=5)
        h2o = H2OEviction(config)
        
        for i in range(5):
            h2o.access(i, f"tok{i}")
        h2o.mark_heavy_hitter(2)  # "important" token
        
        for i in range(5, 15):
            h2o.access(i, f"tok{i}")
        
        assert 2 in h2o.cache, "Heavy hitter should be preserved"

    def test_evicts_low_importance_first(self):
        config = SimulationConfig(policy=EvictionPolicy.H2O, cache_size=4)
        h2o = H2OEviction(config)
        
        h2o.access(0, "a", importance=0.1)
        h2o.access(1, "b", importance=0.9)  # High importance
        h2o.access(2, "c", importance=0.2)
        h2o.access(3, "d", importance=0.3)
        h2o.access(4, "e", importance=0.5)  # Triggers eviction
        
        # Position 0 (lowest importance) should be evicted
        assert 0 not in h2o.cache
        assert 1 in h2o.cache


class TestFullCacheEviction:
    """Tests for full cache - no eviction baseline."""

    def test_never_evicts(self):
        config = SimulationConfig(policy=EvictionPolicy.FULL, cache_size=100)
        full = FullCacheEviction(config)
        
        for i in range(500):
            full.access(i, f"tok{i}")
        
        assert len(full.cache) == 500
        assert full.stats.evictions == 0


class TestGetEvictionStrategy:
    """Tests for get_eviction_strategy factory function."""

    @pytest.mark.parametrize("policy,expected_type", [
        (EvictionPolicy.LRU, LRUEviction),
        (EvictionPolicy.SLIDING_WINDOW, SlidingWindowEviction),
        (EvictionPolicy.STREAMING_LLM, StreamingLLMEviction),
        (EvictionPolicy.H2O, H2OEviction),
        (EvictionPolicy.FULL, FullCacheEviction),
    ])
    def test_returns_correct_strategy(self, policy, expected_type):
        config = SimulationConfig(policy=policy)
        strategy = get_eviction_strategy(config)
        assert isinstance(strategy, expected_type)


class TestRunSimulation:
    """Tests for run_simulation - full end-to-end simulation."""

    def test_processes_real_prompt(self):
        config = SimulationConfig(
            policy=EvictionPolicy.STREAMING_LLM,
            cache_size=64, sink_count=4
        )
        result = run_simulation(
            prompt="The transformer model uses self-attention.",
            config=config, generate_tokens=16
        )
        
        assert result.total_tokens_processed > 10
        assert result.cache_misses > 0
        assert len(result.attention_frames) > 0

    def test_tracks_eviction_statistics(self):
        config = SimulationConfig(
            policy=EvictionPolicy.LRU, cache_size=8
        )
        result = run_simulation(
            prompt="A B C D E F G H I J K L M N O P",
            config=config, generate_tokens=32
        )
        
        assert result.evictions > 0, "Small cache should cause evictions"
        assert result.cache_hits + result.cache_misses == result.total_tokens_processed

    def test_streaming_llm_retains_sinks(self):
        config = SimulationConfig(
            policy=EvictionPolicy.STREAMING_LLM,
            cache_size=32, sink_count=4
        )
        result = run_simulation(
            prompt="Long context processing example.",
            config=config, generate_tokens=64
        )
        
        assert result.retained_sinks > 0, "StreamingLLM should retain sinks"

    def test_returns_final_cache_profile(self):
        config = SimulationConfig(policy=EvictionPolicy.H2O, cache_size=32)
        result = run_simulation(prompt="Test", config=config, generate_tokens=20)
        
        assert result.final_cache is not None
        assert result.final_cache.total_tokens > 0
        assert len(result.final_cache.blocks) > 0


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_default_values(self):
        entry = CacheEntry(position=5, token="hello")
        assert entry.importance == 0.0
        assert entry.access_count == 0
        assert entry.is_sink is False
        assert entry.is_heavy_hitter is False
