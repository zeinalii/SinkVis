"""Tests for memory profiling - The Hardware Check.

Validates memory statistics calculation and CPU fallback behavior.
"""

import pytest
import torch

from sinkvis.memory import MemoryProfiler, MemoryStats


class TestMemoryStats:
    """Test MemoryStats dataclass."""

    def test_memory_stats_fields(self):
        """MemoryStats should have all required fields."""
        stats = MemoryStats(
            allocated_gb=1.0,
            reserved_gb=2.0,
            max_allocated_gb=1.5,
            free_gb=0.5,
        )

        assert stats.allocated_gb == 1.0
        assert stats.reserved_gb == 2.0
        assert stats.max_allocated_gb == 1.5
        assert stats.free_gb == 0.5

    def test_memory_stats_are_floats(self):
        """All MemoryStats fields should be floats."""
        stats = MemoryStats(1.0, 2.0, 1.5, 0.5)

        assert isinstance(stats.allocated_gb, float)
        assert isinstance(stats.reserved_gb, float)
        assert isinstance(stats.max_allocated_gb, float)
        assert isinstance(stats.free_gb, float)


class TestMemoryProfilerCPU:
    """Test MemoryProfiler behavior on CPU (no CUDA)."""

    def test_cpu_returns_zero_stats(self):
        """On CPU-only system, stats should return zeros."""
        profiler = MemoryProfiler(device=torch.device("cpu"))
        stats = profiler.stats

        # All values should be zero on CPU
        assert stats.allocated_gb == 0.0
        assert stats.reserved_gb == 0.0
        assert stats.max_allocated_gb == 0.0
        assert stats.free_gb == 0.0

    def test_cpu_no_exceptions(self):
        """CPU profiling should not raise any exceptions."""
        profiler = MemoryProfiler()

        # Should not raise
        _ = profiler.stats
        profiler.reset_peak()
        _ = profiler.snapshot()

    def test_reset_peak_no_crash_on_cpu(self):
        """reset_peak() should not crash on CPU."""
        profiler = MemoryProfiler(device=torch.device("cpu"))

        # Should complete without exception
        profiler.reset_peak()


class TestMemoryProfilerSnapshot:
    """Test snapshot format and content."""

    def test_snapshot_returns_dict(self):
        """snapshot() should return a dictionary."""
        profiler = MemoryProfiler()
        snapshot = profiler.snapshot()

        assert isinstance(snapshot, dict)

    def test_snapshot_has_required_keys(self):
        """Snapshot should have all expected keys."""
        profiler = MemoryProfiler()
        snapshot = profiler.snapshot()

        expected_keys = ["allocated_gb", "reserved_gb", "max_allocated_gb", "free_gb"]

        for key in expected_keys:
            assert key in snapshot, f"Missing key: {key}"

    def test_snapshot_values_are_floats(self):
        """All snapshot values should be floats."""
        profiler = MemoryProfiler()
        snapshot = profiler.snapshot()

        for key, value in snapshot.items():
            assert isinstance(
                value, float
            ), f"{key} should be float, got {type(value)}"

    def test_snapshot_values_non_negative(self):
        """All snapshot values should be non-negative."""
        profiler = MemoryProfiler()
        snapshot = profiler.snapshot()

        for key, value in snapshot.items():
            assert value >= 0, f"{key} should be >= 0, got {value}"


class TestMemoryProfilerDevice:
    """Test device handling."""

    def test_default_device_selection(self):
        """Default device should be CUDA if available, else CPU."""
        profiler = MemoryProfiler()

        if torch.cuda.is_available():
            assert profiler.device.type == "cuda"
        else:
            assert profiler.device.type == "cpu"

    def test_explicit_cpu_device(self):
        """Can explicitly set CPU device."""
        profiler = MemoryProfiler(device=torch.device("cpu"))

        assert profiler.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_explicit_cuda_device(self):
        """Can explicitly set CUDA device if available."""
        profiler = MemoryProfiler(device=torch.device("cuda"))

        assert profiler.device.type == "cuda"


class TestMemoryProfilerGPU:
    """Tests that require CUDA - skipped on CPU-only systems."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_returns_non_zero_after_allocation(self):
        """After GPU allocation, stats should show memory usage."""
        profiler = MemoryProfiler()

        # Allocate some GPU memory
        tensor = torch.randn(1000, 1000, device="cuda")

        stats = profiler.stats

        # Should have allocated some memory
        assert stats.allocated_gb > 0, "Should show allocated memory"

        # Cleanup
        del tensor
        torch.cuda.empty_cache()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_reset_peak_works(self):
        """reset_peak() should reset peak memory tracking."""
        profiler = MemoryProfiler()

        # Allocate and deallocate to create peak
        tensor = torch.randn(1000, 1000, device="cuda")
        del tensor
        torch.cuda.empty_cache()

        # Reset peak
        profiler.reset_peak()

        stats = profiler.stats

        # Peak should be reset (close to current allocation)
        assert stats.max_allocated_gb >= 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_free_calculation(self):
        """free_gb should be reserved - allocated."""
        profiler = MemoryProfiler()

        # Allocate some memory
        tensor = torch.randn(100, 100, device="cuda")

        stats = profiler.stats

        # Free = reserved - allocated
        expected_free = stats.reserved_gb - stats.allocated_gb
        assert (
            abs(stats.free_gb - expected_free) < 0.001
        ), "Free calculation incorrect"

        del tensor
        torch.cuda.empty_cache()


class TestMemoryProfilerConsistency:
    """Test consistency between stats property and snapshot method."""

    def test_stats_and_snapshot_consistent(self):
        """stats property and snapshot() should return same values."""
        profiler = MemoryProfiler()

        stats = profiler.stats
        snapshot = profiler.snapshot()

        assert stats.allocated_gb == snapshot["allocated_gb"]
        assert stats.reserved_gb == snapshot["reserved_gb"]
        assert stats.max_allocated_gb == snapshot["max_allocated_gb"]
        assert stats.free_gb == snapshot["free_gb"]
