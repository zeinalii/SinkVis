"""Shared pytest fixtures for SinkVis tests."""

import numpy as np
import pytest


@pytest.fixture
def uniform_attention():
    """Create uniform attention matrix (baseline)."""

    def _create(seq_len: int) -> np.ndarray:
        """All positions have equal attention weight."""
        attention = np.ones((seq_len, seq_len))
        attention = np.tril(attention)  # Causal mask
        row_sums = attention.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        return attention / row_sums

    return _create


@pytest.fixture
def sink_attention():
    """Create attention matrix with strong sink tokens (StreamingLLM pattern)."""

    def _create(
        seq_len: int, num_sinks: int = 4, sink_weight: float = 100.0
    ) -> np.ndarray:
        """First num_sinks tokens receive disproportionately high attention."""
        attention = np.full((seq_len, seq_len), 0.01)
        attention = np.tril(attention)  # Causal mask

        # Set sink columns to high values
        for i in range(num_sinks):
            attention[:, i] = sink_weight

        # Re-apply causal mask (sinks only visible to later tokens)
        attention = np.tril(attention)

        # Normalize rows to sum to 1
        row_sums = attention.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return attention / row_sums

    return _create


@pytest.fixture
def heavy_hitter_attention():
    """Create attention matrix with a specific heavy hitter token (H2O pattern)."""

    def _create(
        seq_len: int,
        hitter_index: int,
        num_sinks: int = 4,
        hitter_weight: float = 50.0,
    ) -> np.ndarray:
        """Token at hitter_index accumulates high attention from all later tokens."""
        attention = np.full((seq_len, seq_len), 0.01)
        attention = np.tril(attention)  # Causal mask

        # Set sink columns
        for i in range(num_sinks):
            attention[:, i] = 10.0

        # Set heavy hitter column (high weight from all positions after it)
        attention[hitter_index:, hitter_index] = hitter_weight

        # Re-apply causal mask
        attention = np.tril(attention)

        # Normalize rows
        row_sums = attention.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return attention / row_sums

    return _create


@pytest.fixture
def random_attention():
    """Create random but properly normalized causal attention matrix."""

    def _create(seq_len: int, seed: int = 42) -> np.ndarray:
        """Random values with causal masking and row normalization."""
        rng = np.random.default_rng(seed)
        attention = rng.random((seq_len, seq_len))
        attention = np.tril(attention)  # Causal mask

        # Normalize rows
        row_sums = attention.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return attention / row_sums

    return _create


@pytest.fixture
def nan_attention():
    """Create attention matrix with NaN values for edge case testing."""

    def _create(seq_len: int, nan_positions: list = None) -> np.ndarray:
        """Matrix with NaN at specified positions."""
        attention = np.ones((seq_len, seq_len))
        attention = np.tril(attention)

        if nan_positions:
            for row, col in nan_positions:
                if row >= col:  # Only in lower triangle
                    attention[row, col] = np.nan

        row_sums = attention.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return attention / row_sums

    return _create

