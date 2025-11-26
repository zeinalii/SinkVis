"""Memory profiling for KV cache."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    allocated_gb: float
    reserved_gb: float
    max_allocated_gb: float
    free_gb: float


class MemoryProfiler:
    """Profile GPU memory usage."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    @property
    def stats(self) -> MemoryStats:
        """Get current memory statistics."""
        if not torch.cuda.is_available():
            return MemoryStats(0.0, 0.0, 0.0, 0.0)
        allocated = torch.cuda.memory_allocated(self.device) / 1e9
        reserved = torch.cuda.memory_reserved(self.device) / 1e9
        max_allocated = torch.cuda.max_memory_allocated(self.device) / 1e9
        free = reserved - allocated
        return MemoryStats(allocated, reserved, max_allocated, free)

    def reset_peak(self) -> None:
        """Reset peak memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def snapshot(self) -> dict:
        """Get detailed memory snapshot."""
        stats = self.stats
        return {
            "allocated_gb": stats.allocated_gb,
            "reserved_gb": stats.reserved_gb,
            "max_allocated_gb": stats.max_allocated_gb,
            "free_gb": stats.free_gb,
        }
