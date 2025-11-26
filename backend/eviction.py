"""Eviction policy simulation for KV cache."""

import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .models import (
    EvictionPolicy,
    SimulationConfig,
    SimulationResult,
    AttentionFrame,
    CacheProfile,
    MemoryTier,
    CacheBlock,
)
from .attention import (
    generate_attention_pattern,
    identify_sinks,
    identify_heavy_hitters,
    create_attention_frame,
)


@dataclass
class CacheEntry:
    """Single entry in the KV cache."""
    
    position: int
    token: str
    importance: float = 0.0
    access_count: int = 0
    last_access: float = 0.0
    is_sink: bool = False
    is_heavy_hitter: bool = False


@dataclass
class CacheStats:
    """Statistics for cache simulation."""
    
    hits: int = 0
    misses: int = 0
    evictions: int = 0


class EvictionStrategy(ABC):
    """Base class for eviction strategies."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.cache: OrderedDict[int, CacheEntry] = OrderedDict()
        self.stats = CacheStats()
        self.current_time = 0.0
    
    @abstractmethod
    def should_evict(self, position: int, entry: CacheEntry) -> bool:
        """Determine if an entry should be evicted."""
        pass
    
    @abstractmethod
    def select_victim(self) -> Optional[int]:
        """Select a cache entry to evict. Returns position or None."""
        pass
    
    def access(self, position: int, token: str, importance: float = 0.0):
        """Record an access to a cache position."""
        self.current_time += 0.001
        
        if position in self.cache:
            self.stats.hits += 1
            entry = self.cache[position]
            entry.access_count += 1
            entry.last_access = self.current_time
            entry.importance = max(entry.importance, importance)
            # Move to end for LRU tracking
            self.cache.move_to_end(position)
        else:
            self.stats.misses += 1
            # Evict if necessary
            while len(self.cache) >= self.config.cache_size:
                victim = self.select_victim()
                if victim is not None:
                    del self.cache[victim]
                    self.stats.evictions += 1
                else:
                    break
            
            # Add new entry
            self.cache[position] = CacheEntry(
                position=position,
                token=token,
                importance=importance,
                access_count=1,
                last_access=self.current_time,
            )
    
    def mark_sink(self, position: int):
        """Mark a position as an attention sink."""
        if position in self.cache:
            self.cache[position].is_sink = True
    
    def mark_heavy_hitter(self, position: int):
        """Mark a position as a heavy hitter."""
        if position in self.cache:
            self.cache[position].is_heavy_hitter = True
    
    def get_retained_counts(self) -> tuple[int, int]:
        """Get counts of retained sinks and heavy hitters."""
        sinks = sum(1 for e in self.cache.values() if e.is_sink)
        heavy = sum(1 for e in self.cache.values() if e.is_heavy_hitter)
        return sinks, heavy


class LRUEviction(EvictionStrategy):
    """Least Recently Used eviction policy."""
    
    def should_evict(self, position: int, entry: CacheEntry) -> bool:
        return True  # LRU evicts based on access time
    
    def select_victim(self) -> Optional[int]:
        if not self.cache:
            return None
        # First item is least recently used
        return next(iter(self.cache))


class SlidingWindowEviction(EvictionStrategy):
    """Sliding window eviction - keeps only recent tokens."""
    
    def should_evict(self, position: int, entry: CacheEntry) -> bool:
        max_position = max(self.cache.keys()) if self.cache else 0
        return position < max_position - self.config.window_size
    
    def select_victim(self) -> Optional[int]:
        if not self.cache:
            return None
        
        max_pos = max(self.cache.keys())
        # Evict oldest position outside window
        for pos in list(self.cache.keys()):
            if pos < max_pos - self.config.window_size:
                return pos
        
        # If all in window, evict oldest
        return min(self.cache.keys())


class StreamingLLMEviction(EvictionStrategy):
    """StreamingLLM-style eviction - preserves sinks + sliding window."""
    
    def should_evict(self, position: int, entry: CacheEntry) -> bool:
        # Never evict sinks
        if entry.is_sink:
            return False
        
        max_position = max(self.cache.keys()) if self.cache else 0
        return position < max_position - self.config.window_size
    
    def select_victim(self) -> Optional[int]:
        if not self.cache:
            return None
        
        max_pos = max(self.cache.keys())
        
        # Find oldest non-sink outside window
        candidates = [
            pos for pos, entry in self.cache.items()
            if not entry.is_sink and pos < max_pos - self.config.window_size
        ]
        
        if candidates:
            return min(candidates)
        
        # Fall back to oldest non-sink
        non_sinks = [pos for pos, e in self.cache.items() if not e.is_sink]
        return min(non_sinks) if non_sinks else None


class H2OEviction(EvictionStrategy):
    """H2O (Heavy-Hitter Oracle) eviction - preserves sinks + heavy hitters."""
    
    def should_evict(self, position: int, entry: CacheEntry) -> bool:
        if entry.is_sink or entry.is_heavy_hitter:
            return False
        return True
    
    def select_victim(self) -> Optional[int]:
        if not self.cache:
            return None
        
        # Find entry with lowest importance that isn't protected
        candidates = [ 
            (pos, entry) for pos, entry in self.cache.items()
            if not entry.is_sink and not entry.is_heavy_hitter
        ]
        
        if not candidates:
            # All entries are protected, evict least important heavy hitter
            candidates = [
                (pos, entry) for pos, entry in self.cache.items()
                if not entry.is_sink
            ]
        
        if not candidates:
            return None
        
        # Sort by importance, then by access time
        candidates.sort(key=lambda x: (x[1].importance, x[1].last_access))
        return candidates[0][0]


class FullCacheEviction(EvictionStrategy):
    """No eviction - keeps everything (for comparison)."""
    
    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        # Override with very large cache
        self.config.cache_size = 131072
    
    def should_evict(self, position: int, entry: CacheEntry) -> bool:
        return False
    
    def select_victim(self) -> Optional[int]:
        return None


def get_eviction_strategy(config: SimulationConfig) -> EvictionStrategy:
    """Factory function to get the appropriate eviction strategy."""
    strategies = {
        EvictionPolicy.LRU: LRUEviction,
        EvictionPolicy.SLIDING_WINDOW: SlidingWindowEviction,
        EvictionPolicy.STREAMING_LLM: StreamingLLMEviction,
        EvictionPolicy.H2O: H2OEviction,
        EvictionPolicy.FULL: FullCacheEviction,
    }
    return strategies[config.policy](config)


def tokenize_simple(text: str) -> list[str]:
    """Simple word-based tokenization for demonstration."""
    import re
    
    # Split on whitespace and punctuation, keeping tokens
    tokens = re.findall(r'\w+|[^\w\s]', text)
    
    # Add special tokens
    return ["<s>", "<bos>"] + tokens + ["<eos>"]


def run_simulation(
    prompt: str,
    config: SimulationConfig,
    generate_tokens: int = 128
) -> SimulationResult:
    """
    Run a complete eviction simulation on a prompt.
    
    Args:
        prompt: Input text to process
        config: Simulation configuration
        generate_tokens: Number of tokens to simulate generating
    
    Returns:
        SimulationResult with detailed metrics and attention frames
    """
    strategy = get_eviction_strategy(config)
    tokens = tokenize_simple(prompt)
    
    # Add generated tokens
    for i in range(generate_tokens):
        tokens.append(f"<gen_{i}>")
    
    attention_frames = []
    base_time = time.time()
    
    # Process each token
    for i, token in enumerate(tokens):
        # Generate attention pattern for current position
        attention = generate_attention_pattern(
            seq_len=i + 1,
            num_sinks=config.sink_count
        )
        
        # Identify sinks and heavy hitters
        sinks = identify_sinks(attention, threshold=0.1)
        heavy_hitters = identify_heavy_hitters(
            attention,
            threshold=config.heavy_hitter_ratio,
            exclude_sinks=sinks
        )
        
        # Calculate importance for current token
        importance = attention[-1, :].mean() if i > 0 else 1.0
        
        # Access cache
        strategy.access(i, token, importance)
        
        # Mark special tokens
        for s in sinks:
            strategy.mark_sink(s)
        for h in heavy_hitters:
            strategy.mark_heavy_hitter(h)
        
        # Sample attention frames (not every step)
        if i % max(1, len(tokens) // 20) == 0 or i == len(tokens) - 1:
            frame = create_attention_frame(
                attention=attention,
                layer=0,
                head=0,
                tokens=tokens[:i + 1],
                timestamp=base_time + i * 0.01,
            )
            attention_frames.append(frame)
    
    # Build final cache profile
    retained_sinks, retained_heavy = strategy.get_retained_counts()
    
    cache_blocks = []
    block_size = 16
    positions = sorted(strategy.cache.keys())
    
    # Group into blocks
    if positions:
        current_block_start = positions[0]
        for i, pos in enumerate(positions):
            if i == len(positions) - 1 or positions[i + 1] - pos > block_size:
                # End current block
                entries_in_block = [
                    e for p, e in strategy.cache.items()
                    if current_block_start <= p <= pos
                ]
                
                has_sink = any(e.is_sink for e in entries_in_block)
                has_heavy = any(e.is_heavy_hitter for e in entries_in_block)
                
                if has_sink:
                    tier = MemoryTier.GPU_HBM
                elif has_heavy:
                    tier = MemoryTier.GPU_L2
                else:
                    tier = MemoryTier.SYSTEM_RAM
                
                cache_blocks.append(CacheBlock(
                    block_id=len(cache_blocks),
                    token_range=(current_block_start, pos + 1),
                    memory_tier=tier,
                    size_bytes=(pos - current_block_start + 1) * 256 * 4,
                    last_access=max(e.last_access for e in entries_in_block),
                    access_count=sum(e.access_count for e in entries_in_block),
                    is_sink=has_sink,
                    is_heavy_hitter=has_heavy,
                ))
                
                if i < len(positions) - 1:
                    current_block_start = positions[i + 1]
    
    # Calculate memory usage by tier
    memory_usage = {tier.value: 0 for tier in MemoryTier}
    for block in cache_blocks:
        memory_usage[block.memory_tier.value] += block.size_bytes
    
    final_cache = CacheProfile(
        total_tokens=len(strategy.cache),
        blocks=cache_blocks,
        memory_usage=memory_usage,
        eviction_policy=config.policy,
        timestamp=time.time(),
    )
    
    return SimulationResult(
        policy=config.policy,
        total_tokens_processed=len(tokens),
        cache_hits=strategy.stats.hits,
        cache_misses=strategy.stats.misses,
        evictions=strategy.stats.evictions,
        retained_sinks=retained_sinks,
        retained_heavy_hitters=retained_heavy,
        attention_frames=attention_frames,
        final_cache=final_cache,
    )

