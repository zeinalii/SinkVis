"""Tests for attention pattern generation and analysis."""

import numpy as np
import pytest

from backend.attention import (
    generate_attention_pattern,
    identify_sinks,
    identify_heavy_hitters,
    create_attention_frame,
    generate_cache_blocks,
)
from backend.models import MemoryTier


class TestGenerateAttentionPattern:
    """Tests for generate_attention_pattern function."""

    def test_shape_matches_sequence_length(self):
        attention = generate_attention_pattern(seq_len=10)
        assert attention.shape == (10, 10)

    def test_causal_mask_upper_triangle_is_zero(self):
        attention = generate_attention_pattern(seq_len=8)
        for i in range(8):
            for j in range(i + 1, 8):
                assert attention[i, j] == 0, "Future tokens should not attend"

    def test_each_row_sums_to_one(self):
        attention = generate_attention_pattern(seq_len=16)
        for i in range(16):
            assert np.isclose(attention[i, :i + 1].sum(), 1.0, atol=1e-6)

    def test_uniform_pattern_equal_weights(self):
        attention = generate_attention_pattern(seq_len=5, pattern_type="uniform")
        # Row 4 should attend equally to positions 0-4: each = 0.2
        assert np.allclose(attention[4, :5], 0.2)

    def test_single_token_self_attention(self):
        attention = generate_attention_pattern(seq_len=1)
        assert attention[0, 0] == 1.0


class TestIdentifySinks:
    """Tests for identify_sinks function - finds tokens with high avg attention."""

    def test_bos_token_is_typically_a_sink(self):
        # Simulate attention where position 0 (BOS) gets high attention
        attention = np.array([
            [1.0, 0.0, 0.0],
            [0.6, 0.4, 0.0],
            [0.5, 0.2, 0.3],
        ])
        sinks = identify_sinks(attention, threshold=0.3)
        assert 0 in sinks, "BOS token should be identified as sink"

    def test_returns_empty_for_empty_input(self):
        attention = np.array([]).reshape(0, 0)
        assert identify_sinks(attention) == []

    def test_single_token_is_sink(self):
        attention = np.array([[1.0]])
        assert identify_sinks(attention) == [0]

    def test_higher_threshold_finds_fewer_sinks(self):
        attention = generate_attention_pattern(seq_len=32)
        low = identify_sinks(attention, threshold=0.05)
        high = identify_sinks(attention, threshold=0.3)
        assert len(low) >= len(high)


class TestIdentifyHeavyHitters:
    """Tests for identify_heavy_hitters - finds semantically important tokens."""

    def test_excludes_specified_sinks(self):
        attention = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.4, 0.6, 0.0, 0.0],
            [0.3, 0.3, 0.4, 0.0],
            [0.2, 0.4, 0.2, 0.2],  # Position 1 gets high attention
        ])
        heavy = identify_heavy_hitters(attention, threshold=0.2, exclude_sinks=[0])
        assert 0 not in heavy, "Sinks should be excluded"

    def test_finds_tokens_with_consistent_attention(self):
        # Position 1 consistently receives attention from later positions
        attention = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.3, 0.7, 0.0, 0.0],
            [0.2, 0.5, 0.3, 0.0],
            [0.1, 0.6, 0.2, 0.1],
        ])
        heavy = identify_heavy_hitters(attention, threshold=0.4)
        assert 1 in heavy

    def test_empty_for_single_token(self):
        attention = np.array([[1.0]])
        assert identify_heavy_hitters(attention) == []


class TestCreateAttentionFrame:
    """Tests for create_attention_frame - bundles attention data."""

    def test_creates_frame_with_correct_metadata(self):
        attention = np.array([[1.0, 0.0], [0.5, 0.5]])
        tokens = ["<s>", "Hello"]
        frame = create_attention_frame(
            attention=attention, layer=2, head=5,
            tokens=tokens, timestamp=1000.0
        )
        assert frame.layer == 2
        assert frame.head == 5
        assert frame.seq_len == 2
        assert frame.timestamp == 1000.0
        assert frame.token_labels == ["<s>", "Hello"]

    def test_identifies_sinks_in_frame(self):
        attention = np.array([
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.7, 0.2, 0.1],
        ])
        frame = create_attention_frame(
            attention=attention, layer=0, head=0,
            tokens=["<s>", "The", "cat"], timestamp=0.0,
            sink_threshold=0.5
        )
        assert 0 in frame.sink_indices


class TestGenerateCacheBlocks:
    """Tests for generate_cache_blocks - creates KV cache block layout."""

    def test_blocks_cover_entire_sequence(self):
        blocks = generate_cache_blocks(seq_len=50, block_size=16)
        covered = set()
        for b in blocks:
            covered.update(range(b.token_range[0], b.token_range[1]))
        assert covered == set(range(50))

    def test_sink_blocks_go_to_gpu_hbm(self):
        blocks = generate_cache_blocks(
            seq_len=32, block_size=16, sink_indices=[0, 1]
        )
        sink_block = blocks[0]  # Block containing position 0
        assert sink_block.is_sink
        assert sink_block.memory_tier == MemoryTier.GPU_HBM

    def test_heavy_hitter_blocks_marked_correctly(self):
        blocks = generate_cache_blocks(
            seq_len=64, block_size=16, heavy_hitter_indices=[20]
        )
        # Position 20 is in block 1 (positions 16-31)
        assert blocks[1].is_heavy_hitter

    def test_block_size_calculation(self):
        blocks = generate_cache_blocks(seq_len=100, block_size=32)
        assert len(blocks) == 4  # ceil(100/32) = 4
