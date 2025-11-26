"""Tests for PyTorch attention capture hooks - The Sensor.

Validates hook registration, cleanup, and data capture behavior.
Uses a tiny real model to test actual PyTorch hook mechanics.
"""

import gc
import weakref

import pytest
import torch
import torch.nn as nn

from sinkvis.hooks import AttentionCapture


class MockAttentionLayer(nn.Module):
    """Minimal attention layer that outputs attention weights."""

    def __init__(self, hidden_size: int = 64, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention: (batch, heads, seq, dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention weights
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply attention
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Return output and attention weights (like HuggingFace models)
        return output, attn_weights


class MockTransformerModel(nn.Module):
    """Minimal transformer-like model for testing hooks."""

    def __init__(self, hidden_size: int = 64, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.embed = nn.Linear(32, hidden_size)
        # Use ModuleDict with 'attn' in names so pattern matching works
        self.attn_layers = nn.ModuleDict(
            {
                f"self_attn_{i}": MockAttentionLayer(hidden_size, num_heads)
                for i in range(num_layers)
            }
        )

    def forward(self, x):
        x = self.embed(x)
        attentions = []
        for name in sorted(self.attn_layers.keys()):
            layer = self.attn_layers[name]
            x, attn = layer(x)
            attentions.append(attn)
        return x, attentions


@pytest.fixture
def mock_model():
    """Create a small mock transformer model."""
    return MockTransformerModel(hidden_size=64, num_heads=4, num_layers=2)


@pytest.fixture
def sample_input():
    """Create sample input tensor."""
    batch_size = 2
    seq_len = 10
    input_dim = 32
    return torch.randn(batch_size, seq_len, input_dim)


class TestHookRegistration:
    """Test hook registration and cleanup lifecycle."""

    def test_no_hooks_before_start(self, mock_model):
        """Model should have no forward hooks initially."""
        capture = AttentionCapture(mock_model)

        # Count hooks on attention layers
        hook_count = sum(
            len(m._forward_hooks)
            for name, m in mock_model.named_modules()
            if "attn" in name.lower()
        )
        assert hook_count == 0, "No hooks should be registered before start()"

    def test_hooks_registered_after_start(self, mock_model):
        """Hooks should be registered after calling start()."""
        capture = AttentionCapture(mock_model)
        capture.start(pattern=r".*attn.*")

        # Should have hooks registered
        assert len(capture._hooks) > 0, "Hooks should be registered after start()"

    def test_hooks_removed_after_stop(self, mock_model):
        """All hooks should be removed after stop()."""
        capture = AttentionCapture(mock_model)
        capture.start(pattern=r".*attn.*")
        initial_hooks = len(capture._hooks)

        capture.stop()

        assert len(capture._hooks) == 0, "All hooks should be removed after stop()"

    def test_context_manager_cleanup(self, mock_model):
        """Context manager should properly clean up hooks on exit."""
        # Note: __enter__ calls start() with default pattern
        # Don't call start() again to avoid duplicate hooks
        capture = AttentionCapture(mock_model)
        with capture:
            # Manually start with custom pattern (after stopping default)
            capture.stop()
            capture.start(pattern=r".*attn.*")
            assert len(capture._hooks) > 0, "Hooks should exist inside context"

        # After context exit, hooks should be removed
        assert len(capture._hooks) == 0, "Hooks must be removed after context exit"

    def test_context_manager_default_pattern(self, mock_model, sample_input):
        """Context manager with default pattern should work."""
        with AttentionCapture(mock_model) as capture:
            # __enter__ calls start() with default pattern r".*attn.*"
            assert len(capture._hooks) > 0, "Default pattern should register hooks"

            with torch.no_grad():
                _ = mock_model(sample_input)

            assert capture.get_latest() is not None

        assert len(capture._hooks) == 0, "Hooks removed after exit"

    def test_multiple_start_stop_cycles(self, mock_model):
        """Can start and stop hooks multiple times."""
        capture = AttentionCapture(mock_model)

        for _ in range(3):
            capture.start(pattern=r".*attn.*")
            assert len(capture._hooks) > 0
            capture.stop()
            assert len(capture._hooks) == 0


class TestDataCapture:
    """Test attention data capture during forward pass."""

    def test_captures_attention_tensor(self, mock_model, sample_input):
        """Should capture attention tensor during forward pass."""
        with AttentionCapture(mock_model) as capture:
            capture.start(pattern=r".*attn.*")

            with torch.no_grad():
                _ = mock_model(sample_input)

            attn = capture.get_latest()

            assert attn is not None, "Should capture attention tensor"

    def test_attention_shape(self, mock_model, sample_input):
        """Captured attention should have shape (batch, heads, seq, seq)."""
        batch_size, seq_len, _ = sample_input.shape
        num_heads = 4

        with AttentionCapture(mock_model) as capture:
            capture.start(pattern=r".*attn.*")

            with torch.no_grad():
                _ = mock_model(sample_input)

            attn = capture.get_latest()

            assert attn is not None
            assert attn.dim() == 4, "Attention should be 4D tensor"
            assert attn.shape[0] == batch_size, "Batch dimension mismatch"
            assert attn.shape[1] == num_heads, "Heads dimension mismatch"
            assert attn.shape[2] == seq_len, "Query seq dimension mismatch"
            assert attn.shape[3] == seq_len, "Key seq dimension mismatch"

    def test_gradients_detached(self, mock_model, sample_input):
        """Captured tensors should have gradients detached for memory safety."""
        with AttentionCapture(mock_model) as capture:
            capture.start(pattern=r".*attn.*")

            # Run with gradients enabled
            _ = mock_model(sample_input)

            attn = capture.get_latest()

            assert attn is not None
            assert not attn.requires_grad, "Captured attention must be detached"

    def test_get_latest_returns_none_when_empty(self, mock_model):
        """get_latest() returns None when buffer is empty."""
        capture = AttentionCapture(mock_model)

        assert capture.get_latest() is None


class TestBufferManagement:
    """Test attention buffer management."""

    def test_buffer_respects_maxlen(self, mock_model, sample_input):
        """Buffer should not exceed maxlen."""
        buffer_size = 5
        capture = AttentionCapture(mock_model, buffer_size=buffer_size)
        capture.start(pattern=r".*attn.*")

        # Run more forward passes than buffer size
        with torch.no_grad():
            for _ in range(buffer_size * 2):
                _ = mock_model(sample_input)

        # Buffer should not exceed maxlen
        assert len(capture.buffer) <= buffer_size, "Buffer exceeded maxlen"

        capture.stop()

    def test_clear_empties_buffer(self, mock_model, sample_input):
        """clear() should empty the buffer."""
        with AttentionCapture(mock_model) as capture:
            capture.start(pattern=r".*attn.*")

            with torch.no_grad():
                _ = mock_model(sample_input)

            assert len(capture.buffer) > 0, "Buffer should have data"

            capture.clear()

            assert len(capture.buffer) == 0, "Buffer should be empty after clear()"

    def test_buffer_fifo_order(self, mock_model, sample_input):
        """Buffer should maintain FIFO order (oldest evicted first)."""
        buffer_size = 3
        capture = AttentionCapture(mock_model, buffer_size=buffer_size)
        capture.start(pattern=r".*attn.*")

        with torch.no_grad():
            for _ in range(buffer_size + 2):
                _ = mock_model(sample_input)

        # get_latest should return most recent
        latest = capture.get_latest()
        assert latest is capture.buffer[-1], "get_latest should return newest"

        capture.stop()


class TestWeakReference:
    """Test weak reference handling for model lifecycle."""

    def test_weak_ref_raises_on_deleted_model(self):
        """start() should raise RuntimeError if model is garbage collected."""
        model = MockTransformerModel()
        capture = AttentionCapture(model)

        # Delete the model
        del model
        gc.collect()

        # Attempting to start should raise
        with pytest.raises(RuntimeError, match="garbage collected"):
            capture.start()

    def test_weak_ref_allows_normal_operation(self, mock_model, sample_input):
        """Normal operation works when model exists."""
        capture = AttentionCapture(mock_model)

        # Should work fine
        capture.start(pattern=r".*attn.*")
        with torch.no_grad():
            _ = mock_model(sample_input)

        assert capture.get_latest() is not None
        capture.stop()


class TestPatternMatching:
    """Test hook pattern matching on layer names."""

    def test_pattern_matches_attn_layers(self, mock_model):
        """Pattern r'.*attn.*' should match attention layers."""
        capture = AttentionCapture(mock_model)
        capture.start(pattern=r".*attn.*")

        # Should have registered hooks on attention layers
        assert len(capture._hooks) > 0, "Should match layers with 'attn' in name"

        capture.stop()

    def test_non_matching_pattern(self, mock_model):
        """Non-matching pattern should register no hooks."""
        capture = AttentionCapture(mock_model)
        capture.start(pattern=r".*nonexistent.*")

        assert len(capture._hooks) == 0, "Should not match any layers"

        capture.stop()

    def test_case_insensitive_matching(self, mock_model):
        """Pattern matching should be case-insensitive."""
        capture = AttentionCapture(mock_model)
        capture.start(pattern=r".*ATTN.*")  # Uppercase

        # Should still match due to re.IGNORECASE
        assert len(capture._hooks) > 0, "Should match case-insensitively"

        capture.stop()

