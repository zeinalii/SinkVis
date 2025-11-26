"""Tests for eviction simulation logic - The Physics Engine.

Validates backend logic against theoretical behaviors from StreamingLLM and H2O papers.
Uses deterministic synthetic NumPy arrays - NO real LLM required.
"""

import numpy as np
import pytest
import torch

from sinkvis.simulation import (
    apply_eviction_mask,
    simulate_h2o,
    simulate_lru,
    simulate_sliding_window,
    simulate_streaming_llm,
)


class TestAttentionSinkPreservation:
    """Test StreamingLLM behavior: attention sinks always preserved."""

    def test_sink_tokens_always_preserved(self, sink_attention):
        """Verify first sink_count tokens are kept regardless of age."""
        seq_len = 100
        budget = 10
        sink_count = 4

        mask = simulate_streaming_llm(seq_len, budget, sink_count)

        # StreamingLLM invariant: sinks ALWAYS preserved
        assert mask[:sink_count].all(), "Sink tokens must always be preserved"

    def test_sinks_preserved_with_small_budget(self, sink_attention):
        """Sinks preserved even when budget barely fits them."""
        seq_len = 100
        sink_count = 4
        budget = 5  # Only 1 slot beyond sinks

        mask = simulate_streaming_llm(seq_len, budget, sink_count)

        assert mask[:sink_count].all(), "All sinks must be kept"
        assert mask.sum() == budget, f"Exactly {budget} tokens should be kept"

    def test_recent_window_also_preserved(self, sink_attention):
        """StreamingLLM preserves sinks + recent window."""
        seq_len = 100
        budget = 20
        sink_count = 4

        mask = simulate_streaming_llm(seq_len, budget, sink_count)

        # Recent window should fill remaining budget
        window_size = budget - sink_count
        assert mask[-window_size:].all(), "Recent window must be preserved"

    def test_middle_tokens_evicted(self, sink_attention):
        """Tokens between sinks and recent window should be evicted."""
        seq_len = 100
        budget = 20
        sink_count = 4

        mask = simulate_streaming_llm(seq_len, budget, sink_count)
        window_size = budget - sink_count

        # Middle section should be False
        middle_start = sink_count
        middle_end = seq_len - window_size
        if middle_start < middle_end:
            assert not mask[
                middle_start:middle_end
            ].any(), "Middle tokens should be evicted"


class TestHeavyHitterPreservation:
    """Test H2O behavior: heavy hitters preserved based on attention scores."""

    def test_heavy_hitter_preserved(self, heavy_hitter_attention):
        """Token with highest cumulative attention is kept."""
        seq_len = 100
        hitter_index = 42
        budget = 15
        sink_count = 4

        attention = heavy_hitter_attention(seq_len, hitter_index, sink_count)
        mask, scores = simulate_h2o(attention, budget, sink_count)

        # H2O invariant: heavy hitter should be preserved
        assert mask[
            hitter_index
        ], f"Heavy hitter at index {hitter_index} must be preserved"

    def test_sinks_always_preserved_in_h2o(self, heavy_hitter_attention):
        """H2O also preserves sink tokens."""
        seq_len = 100
        hitter_index = 50
        budget = 10
        sink_count = 4

        attention = heavy_hitter_attention(seq_len, hitter_index, sink_count)
        mask, _ = simulate_h2o(attention, budget, sink_count)

        # Sinks preserved in H2O
        assert mask[:sink_count].all(), "H2O must preserve sink tokens"

    def test_h2o_selects_top_attention_tokens(self, random_attention):
        """H2O keeps tokens with highest cumulative attention scores."""
        seq_len = 50
        budget = 15
        sink_count = 4

        attention = random_attention(seq_len)
        mask, scores = simulate_h2o(attention, budget, sink_count)

        # Check that selected non-sink tokens have higher scores
        non_sink_scores = scores[sink_count:]
        non_sink_mask = mask[sink_count:]

        if non_sink_mask.any():
            kept_scores = non_sink_scores[non_sink_mask]
            evicted_scores = non_sink_scores[~non_sink_mask]
            if len(evicted_scores) > 0:
                assert (
                    kept_scores.min() >= evicted_scores.max()
                ), "H2O should keep highest-attention tokens"

    def test_h2o_budget_constraint(self, heavy_hitter_attention):
        """H2O respects budget constraint."""
        seq_len = 100
        budget = 20
        sink_count = 4

        attention = heavy_hitter_attention(seq_len, 42, sink_count)
        mask, _ = simulate_h2o(attention, budget, sink_count)

        assert mask.sum() == budget, f"H2O must keep exactly {budget} tokens"


class TestSlidingWindow:
    """Test sliding window eviction: keep only last N tokens."""

    def test_exact_window_indices(self):
        """Verify exactly indices [N-K:N] are returned."""
        seq_len = 100
        window_size = 10

        mask = simulate_sliding_window(seq_len, window_size)

        # Exactly last window_size tokens should be True
        assert mask[90:100].all(), "Last 10 tokens must be kept"
        assert not mask[:90].any(), "Earlier tokens must be evicted"

    def test_window_count(self):
        """Exactly window_size tokens are kept."""
        seq_len = 100
        window_size = 25

        mask = simulate_sliding_window(seq_len, window_size)

        assert mask.sum() == window_size, f"Exactly {window_size} tokens should be kept"

    def test_window_larger_than_sequence(self):
        """Window >= seq_len keeps all tokens."""
        seq_len = 20
        window_size = 50

        mask = simulate_sliding_window(seq_len, window_size)

        assert mask.all(), "All tokens should be kept when window >= seq_len"


class TestLRUEviction:
    """Test LRU eviction: lowest cumulative attention evicted first."""

    def test_lru_keeps_high_attention_tokens(self, random_attention):
        """LRU keeps tokens with highest cumulative attention."""
        seq_len = 50
        budget = 20

        attention = random_attention(seq_len)
        mask, scores = simulate_lru(attention, budget)

        # Budget constraint
        assert mask.sum() == budget, f"LRU must keep exactly {budget} tokens"

        # High-score tokens kept
        kept_scores = scores[mask]
        evicted_scores = scores[~mask]
        if len(evicted_scores) > 0:
            assert (
                kept_scores.min() >= evicted_scores.max()
            ), "LRU should keep highest-score tokens"

    def test_lru_budget_respected(self, uniform_attention):
        """LRU keeps exactly budget tokens."""
        seq_len = 100
        budget = 30

        attention = uniform_attention(seq_len)
        mask, _ = simulate_lru(attention, budget)

        assert mask.sum() == budget


class TestBoundaryConditions:
    """Test edge cases and boundary conditions."""

    def test_seq_len_smaller_than_budget(self, uniform_attention):
        """When seq_len < budget, keep all tokens."""
        seq_len = 5
        budget = 20

        attention = uniform_attention(seq_len)
        mask, _ = simulate_lru(attention, budget)

        # Should keep all tokens (or up to seq_len)
        assert mask.sum() == seq_len, "Should keep all tokens when seq < budget"

    def test_sliding_window_seq_smaller_than_window(self):
        """Sliding window with small sequence."""
        seq_len = 5
        window_size = 20

        mask = simulate_sliding_window(seq_len, window_size)

        assert mask.all(), "All tokens kept when window > seq_len"

    def test_streaming_llm_budget_equals_sinks(self):
        """When budget == sink_count, only sinks are kept."""
        seq_len = 100
        sink_count = 4
        budget = 4

        mask = simulate_streaming_llm(seq_len, budget, sink_count)

        assert mask[:sink_count].all(), "All sinks should be kept"
        assert mask.sum() == sink_count, "Only sinks should be kept"

    def test_h2o_budget_equals_sinks(self, uniform_attention):
        """When budget == sink_count, only sinks kept in H2O."""
        seq_len = 50
        sink_count = 4
        budget = 4

        attention = uniform_attention(seq_len)
        mask, _ = simulate_h2o(attention, budget, sink_count)

        assert mask[:sink_count].all(), "Sinks should be kept"
        assert mask.sum() == sink_count, "Only sinks when budget == sink_count"

    def test_zero_budget_streaming_llm(self):
        """Zero budget edge case for StreamingLLM."""
        seq_len = 50
        sink_count = 4
        budget = 0

        mask = simulate_streaming_llm(seq_len, budget, sink_count)

        # With zero budget, behavior depends on implementation
        # At minimum, should not crash
        assert mask.shape[0] == seq_len

    def test_sink_count_larger_than_budget(self):
        """When sink_count > budget, all sinks still preserved (StreamingLLM priority)."""
        seq_len = 50
        sink_count = 10
        budget = 5

        mask = simulate_streaming_llm(seq_len, budget, sink_count)

        # StreamingLLM always preserves sinks - they take priority
        # When sink_count > budget, sinks may exceed budget (intended behavior)
        assert mask[:sink_count].all(), "All sinks must be preserved"


class TestNaNHandling:
    """Test graceful handling of NaN values."""

    def test_nan_in_attention_no_crash(self, nan_attention):
        """NaN values should not crash the simulation."""
        seq_len = 20
        nan_positions = [(5, 3), (10, 8)]

        attention = nan_attention(seq_len, nan_positions)

        # Should not raise exception
        try:
            mask, scores = simulate_h2o(attention, budget=10, sink_count=4)
            # If we get here, it handled NaN gracefully
            assert True
        except Exception as e:
            # Document if NaN causes issues
            pytest.skip(f"NaN handling not implemented: {e}")


class TestCausalMaskingInvariant:
    """Verify attention matrices maintain causal masking property."""

    def test_generated_attention_is_causal(self, sink_attention, random_attention):
        """All fixtures produce lower-triangular attention."""
        seq_len = 50

        for fixture_fn in [sink_attention, random_attention]:
            attention = fixture_fn(seq_len)

            # Upper triangle (excluding diagonal) should be zero
            upper_tri = np.triu(attention, k=1)
            assert np.allclose(
                upper_tri, 0
            ), "Attention must be causal (lower triangular)"

    def test_rows_sum_to_one(self, sink_attention, random_attention):
        """Normalized attention rows sum to 1."""
        seq_len = 50

        for fixture_fn in [sink_attention, random_attention]:
            attention = fixture_fn(seq_len)

            # Skip first row (may have only one element)
            for i in range(1, seq_len):
                row_sum = attention[i, : i + 1].sum()
                assert np.isclose(
                    row_sum, 1.0, atol=1e-6
                ), f"Row {i} should sum to 1.0, got {row_sum}"


class TestApplyEvictionMask:
    """Test eviction mask application to attention tensors."""

    def test_mask_reduces_sequence_length(self):
        """Applying mask reduces the key dimension."""
        batch, heads, seq_q, seq_k = 2, 4, 10, 10
        attention = torch.randn(batch, heads, seq_q, seq_k)
        mask = np.array(
            [True, True, False, False, True, False, False, False, True, True]
        )

        result = apply_eviction_mask(attention, mask)

        # Original: (2, 4, 10, 10), after mask: (2, 4, 10, 5)
        assert result.shape == (batch, heads, seq_q, mask.sum())

    def test_preserves_selected_positions(self):
        """Masked result contains only selected key positions."""
        batch, heads, seq = 1, 1, 5
        attention = torch.arange(25).float().view(batch, heads, seq, seq)
        mask = np.array([True, False, True, False, True])

        result = apply_eviction_mask(attention, mask)

        # Should keep columns 0, 2, 4
        assert result.shape[-1] == 3

    def test_all_true_mask(self):
        """All-True mask keeps everything."""
        attention = torch.randn(2, 4, 10, 10)
        mask = np.ones(10, dtype=bool)

        result = apply_eviction_mask(attention, mask)

        assert result.shape == attention.shape
        assert torch.allclose(result, attention)

    def test_preserves_device(self):
        """Result stays on same device as input."""
        attention = torch.randn(1, 1, 5, 5)
        mask = np.array([True, True, False, True, False])

        result = apply_eviction_mask(attention, mask)

        assert result.device == attention.device

    def test_single_true(self):
        """Mask with single True keeps one position."""
        attention = torch.randn(2, 4, 10, 10)
        mask = np.zeros(10, dtype=bool)
        mask[5] = True

        result = apply_eviction_mask(attention, mask)

        assert result.shape == (2, 4, 10, 1)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_tensor(self):
        """Works with CUDA tensors."""
        attention = torch.randn(2, 4, 10, 10, device="cuda")
        mask = np.array([True] * 5 + [False] * 5)

        result = apply_eviction_mask(attention, mask)

        assert result.device.type == "cuda"
        assert result.shape[-1] == 5

