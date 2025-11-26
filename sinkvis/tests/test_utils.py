"""Tests for utils.py - Utility functions.

Tests tensor conversion, token formatting, and safe division.
"""

import numpy as np
import pytest
import torch

from sinkvis.utils import format_tokens, safe_divide, tensor_to_json


class TestTensorToJson:
    """Tests for tensor to JSON conversion."""

    def test_torch_tensor(self):
        """Converts PyTorch tensor to list."""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = tensor_to_json(tensor)

        assert isinstance(result, list)
        assert result == [[1.0, 2.0], [3.0, 4.0]]

    def test_numpy_array(self):
        """Converts NumPy array to list."""
        arr = np.array([1.5, 2.5, 3.5])
        result = tensor_to_json(arr)

        assert isinstance(result, list)
        assert len(result) == 3

    def test_decimals_rounding(self):
        """Rounds to specified decimal places."""
        tensor = torch.tensor([1.123456789])
        result = tensor_to_json(tensor, decimals=2)

        # Use tolerance for float comparison (float32 precision limits)
        assert abs(result[0] - 1.12) < 1e-5

    def test_default_decimals(self):
        """Default is 4 decimal places."""
        tensor = torch.tensor([1.123456789])
        result = tensor_to_json(tensor, decimals=4)

        # Use tolerance for float comparison
        assert abs(result[0] - 1.1235) < 1e-5

    def test_multidimensional(self):
        """Handles multi-dimensional tensors."""
        tensor = torch.zeros(2, 3, 4)
        result = tensor_to_json(tensor)

        assert len(result) == 2
        assert len(result[0]) == 3
        assert len(result[0][0]) == 4

    def test_detaches_gradient(self):
        """Detaches gradient before conversion."""
        tensor = torch.tensor([1.0, 2.0], requires_grad=True)
        result = tensor_to_json(tensor)

        # Should not raise, gradient is detached
        assert result == [1.0, 2.0]

    def test_passthrough_non_tensor(self):
        """Non-tensor inputs are passed through."""
        result = tensor_to_json([1, 2, 3])

        assert result == [1, 2, 3]

    def test_cuda_tensor(self):
        """Handles CUDA tensors (if available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        tensor = torch.tensor([1.0, 2.0], device="cuda")
        result = tensor_to_json(tensor)

        assert result == [1.0, 2.0]


class TestFormatTokens:
    """Tests for token formatting."""

    def test_basic_formatting(self):
        """Formats token IDs and strings into list of dicts."""
        token_ids = [1, 2, 3]
        token_strings = ["hello", "world", "test"]

        result = format_tokens(token_ids, token_strings)

        assert len(result) == 3
        assert result[0] == {"id": 1, "text": "hello"}
        assert result[1] == {"id": 2, "text": "world"}
        assert result[2] == {"id": 3, "text": "test"}

    def test_empty_lists(self):
        """Handles empty input lists."""
        result = format_tokens([], [])

        assert result == []

    def test_single_token(self):
        """Handles single token."""
        result = format_tokens([42], ["answer"])

        assert result == [{"id": 42, "text": "answer"}]

    def test_preserves_order(self):
        """Preserves input order."""
        ids = [5, 3, 1, 4, 2]
        texts = ["a", "b", "c", "d", "e"]

        result = format_tokens(ids, texts)

        assert [r["id"] for r in result] == ids
        assert [r["text"] for r in result] == texts


class TestSafeDivide:
    """Tests for safe division utility."""

    def test_normal_division(self):
        """Normal division works correctly."""
        result = safe_divide(10.0, 2.0)

        assert result == 5.0

    def test_zero_denominator_returns_default(self):
        """Zero denominator returns default value."""
        result = safe_divide(10.0, 0.0)

        assert result == 0.0

    def test_custom_default(self):
        """Can specify custom default value."""
        result = safe_divide(10.0, 0.0, default=-1.0)

        assert result == -1.0

    def test_integer_division(self):
        """Handles integer inputs."""
        result = safe_divide(10, 3)

        assert abs(result - 3.333333) < 0.001

    def test_float_precision(self):
        """Maintains float precision."""
        result = safe_divide(1.0, 3.0)

        assert abs(result - 0.333333) < 0.001

    def test_negative_numbers(self):
        """Handles negative numbers."""
        assert safe_divide(-10.0, 2.0) == -5.0
        assert safe_divide(10.0, -2.0) == -5.0
        assert safe_divide(-10.0, -2.0) == 5.0
