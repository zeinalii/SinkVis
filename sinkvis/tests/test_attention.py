"""Tests for attention.py - Attention pattern generation and analysis.

Tests attention matrix generation, sink/heavy-hitter identification,
and cache block generation.
"""

import numpy as np
import pytest

from sinkvis.attention import (
    create_attention_frame,
    generate_attention_pattern,
    generate_cache_blocks,
    identify_heavy_hitters,
    identify_sinks,
)
from sinkvis.models import MemoryTier


class TestGenerateAttentionPattern:
    """Tests for attention pattern generation."""

    def test_shape(self):
        """Output has shape (seq_len, seq_len)."""
        seq_len = 50
        attention = generate_attention_pattern(seq_len)

        assert attention.shape == (seq_len, seq_len)

    def test_causal_mask(self):
        """Upper triangle (above diagonal) should be zero."""
        seq_len = 20
        attention = generate_attention_pattern(seq_len)

        upper_tri = np.triu(attention, k=1)
        assert np.allclose(upper_tri, 0), "Should be causal (lower triangular)"

    def test_rows_sum_to_one(self):
        """Each row should sum to 1 (normalized probabilities)."""
        seq_len = 30
        attention = generate_attention_pattern(seq_len)

        row_sums = attention.sum(axis=1)
        assert np.allclose(row_sums, 1.0), "Rows should sum to 1"

    def test_sinks_have_higher_attention(self):
        """Sink positions should have higher average attention."""
        seq_len = 50
        num_sinks = 4
        attention = generate_attention_pattern(seq_len, num_sinks=num_sinks)

        avg_attention = attention.mean(axis=0)
        sink_avg = avg_attention[:num_sinks].mean()
        non_sink_avg = avg_attention[num_sinks:].mean()

        assert sink_avg > non_sink_avg, "Sinks should have higher avg attention"

    def test_non_negative(self):
        """All attention values should be non-negative."""
        attention = generate_attention_pattern(40)

        assert (attention >= 0).all(), "Attention should be non-negative"

    def test_custom_num_sinks(self):
        """Can specify custom number of sinks."""
        attention = generate_attention_pattern(50, num_sinks=8)

        # Just verify it runs without error
        assert attention.shape == (50, 50)


class TestIdentifySinks:
    """Tests for sink identification."""

    def test_finds_high_attention_positions(self):
        """Identifies positions with above-threshold average attention."""
        attention = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.2, 0.0],
                [0.6, 0.3, 0.1],
            ]
        )

        sinks = identify_sinks(attention, threshold=0.5)

        assert 0 in sinks, "First position should be identified as sink"

    def test_empty_with_low_threshold(self):
        """Returns empty list if nothing exceeds threshold."""
        attention = np.array(
            [
                [0.1, 0.0, 0.0],
                [0.05, 0.05, 0.0],
                [0.03, 0.03, 0.04],
            ]
        )

        sinks = identify_sinks(attention, threshold=0.5)

        assert len(sinks) == 0

    def test_returns_list_of_ints(self):
        """Returns list of integer indices."""
        attention = generate_attention_pattern(20)
        sinks = identify_sinks(attention, threshold=0.05)

        assert isinstance(sinks, list)
        assert all(isinstance(s, (int, np.integer)) for s in sinks)


class TestIdentifyHeavyHitters:
    """Tests for heavy hitter identification."""

    def test_excludes_sinks(self):
        """Heavy hitters should exclude specified sink positions."""
        attention = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.7, 0.3, 0.0, 0.0],
                [0.5, 0.1, 0.4, 0.0],
                [0.4, 0.1, 0.3, 0.2],
            ]
        )

        heavy = identify_heavy_hitters(attention, threshold=0.1, exclude_sinks=[0])

        assert 0 not in heavy, "Position 0 is a sink and should be excluded"

    def test_finds_above_threshold(self):
        """Identifies non-sink positions above threshold."""
        attention = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.3, 0.7, 0.0],
                [0.2, 0.6, 0.2],
            ]
        )

        # Position 1 has high avg attention
        heavy = identify_heavy_hitters(attention, threshold=0.3, exclude_sinks=[0])

        assert 1 in heavy

    def test_no_exclusions_by_default(self):
        """Without exclude_sinks, no positions are excluded."""
        attention = generate_attention_pattern(20)
        heavy = identify_heavy_hitters(attention, threshold=0.05)

        # Should be able to include any position
        assert isinstance(heavy, list)


class TestCreateAttentionFrame:
    """Tests for attention frame creation."""

    def test_creates_valid_frame(self):
        """Creates AttentionFrame with all required fields."""
        attention = generate_attention_pattern(10)
        tokens = [f"token_{i}" for i in range(10)]

        frame = create_attention_frame(attention, tokens, layer=2, head=3)

        assert frame.layer == 2
        assert frame.head == 3
        assert frame.seq_len == 10
        assert np.array_equal(frame.attention_weights, attention)
        assert frame.token_labels == tokens

    def test_identifies_sinks_and_heavy_hitters(self):
        """Frame includes identified sinks and heavy hitters."""
        attention = generate_attention_pattern(20, num_sinks=4)
        tokens = [f"t{i}" for i in range(20)]

        frame = create_attention_frame(attention, tokens)

        assert isinstance(frame.sink_indices, list)
        assert isinstance(frame.heavy_hitter_indices, list)

    def test_default_layer_head(self):
        """Defaults to layer 0 and head 0."""
        attention = generate_attention_pattern(5)
        tokens = ["a", "b", "c", "d", "e"]

        frame = create_attention_frame(attention, tokens)

        assert frame.layer == 0
        assert frame.head == 0

    def test_timestamp(self):
        """Can set custom timestamp."""
        attention = generate_attention_pattern(5)
        tokens = list("abcde")

        frame = create_attention_frame(attention, tokens, timestamp=1234.5)

        assert frame.timestamp == 1234.5


class TestGenerateCacheBlocks:
    """Tests for cache block generation."""

    def test_covers_full_sequence(self):
        """Blocks cover the entire sequence length."""
        seq_len = 100
        block_size = 16

        blocks = generate_cache_blocks(seq_len, block_size)

        # Check coverage
        covered = set()
        for block in blocks:
            for i in range(block.start_token, block.end_token):
                covered.add(i)

        assert covered == set(range(seq_len))

    def test_block_count(self):
        """Correct number of blocks generated."""
        seq_len = 100
        block_size = 16
        expected_blocks = (seq_len + block_size - 1) // block_size

        blocks = generate_cache_blocks(seq_len, block_size)

        assert len(blocks) == expected_blocks

    def test_sink_blocks_in_hbm(self):
        """Blocks containing sinks are assigned to GPU HBM."""
        seq_len = 64
        block_size = 16
        sink_indices = [0, 1, 2, 3]  # First block contains sinks

        blocks = generate_cache_blocks(seq_len, block_size, sink_indices=sink_indices)

        first_block = blocks[0]
        assert first_block.is_sink
        assert first_block.memory_tier == MemoryTier.GPU_HBM

    def test_heavy_hitter_blocks_in_hbm(self):
        """Blocks containing heavy hitters are assigned to GPU HBM."""
        seq_len = 64
        block_size = 16
        heavy_indices = [32, 33]  # Third block

        blocks = generate_cache_blocks(
            seq_len, block_size, heavy_hitter_indices=heavy_indices
        )

        block_with_heavy = blocks[2]  # index 32-47
        assert block_with_heavy.is_heavy_hitter
        assert block_with_heavy.memory_tier == MemoryTier.GPU_HBM

    def test_block_ids_sequential(self):
        """Block IDs are sequential starting from 0."""
        blocks = generate_cache_blocks(50, 10)

        block_ids = [b.block_id for b in blocks]
        assert block_ids == list(range(len(blocks)))

    def test_size_bytes_calculation(self):
        """Size bytes calculated correctly."""
        seq_len = 32
        block_size = 16

        blocks = generate_cache_blocks(seq_len, block_size)

        for block in blocks:
            tokens_in_block = block.end_token - block.start_token
            expected_size = tokens_in_block * 128
            assert block.size_bytes == expected_size

    def test_access_count_priority(self):
        """Sink blocks have highest access count, then heavy hitters."""
        blocks = generate_cache_blocks(
            64, 16, sink_indices=[0, 1], heavy_hitter_indices=[32]
        )

        sink_block = blocks[0]
        heavy_block = blocks[2]
        regular_block = blocks[3]

        assert sink_block.access_count > regular_block.access_count
        assert heavy_block.access_count > regular_block.access_count
