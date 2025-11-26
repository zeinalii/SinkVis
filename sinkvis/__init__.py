"""SinkVis - Attention sink and KV cache visualization."""

from typing import List, Optional

import numpy as np
import torch.nn as nn

from .attention import create_attention_frame
from .hooks import AttentionCapture
from .memory import MemoryProfiler
from .models import AttentionFrame
from .simulation import (
    simulate_h2o,
    simulate_lru,
    simulate_sliding_window,
    simulate_streaming_llm,
)
from .utils import tensor_to_json

__all__ = ["SinkVis"]


class SinkVis:
    """Main API for SinkVis instrumentation."""

    def __init__(self, model: nn.Module, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self._capture = AttentionCapture(model)
        self._profiler = MemoryProfiler()
        self._original_output_attentions = None
        self._original_attn_implementation = None

    def __enter__(self):
        # Enable attention output in model config (required for hooks to capture)
        if hasattr(self.model, "config"):
            self._original_output_attentions = getattr(
                self.model.config, "output_attentions", None
            )
            self._original_attn_implementation = getattr(
                self.model.config, "_attn_implementation", None
            )
            # SDPA doesn't support attention output, switch to eager
            if self._original_attn_implementation == "sdpa":
                self.model.config._attn_implementation = "eager"
            self.model.config.output_attentions = True
        self._capture.start()
        return self

    def __exit__(self, *args):
        self._capture.stop()
        # Restore original config values
        if hasattr(self.model, "config"):
            if self._original_output_attentions is not None:
                self.model.config.output_attentions = self._original_output_attentions
            if self._original_attn_implementation is not None:
                self.model.config._attn_implementation = (
                    self._original_attn_implementation
                )

    def get_attention_data(
        self, layer: int = -1, head: int = 0
    ) -> Optional[List[List[float]]]:
        """Get attention matrix as JSON-ready list."""
        attn = self._capture.get_latest()
        if attn is None:
            return None
        if layer < 0:
            layer = attn.shape[1] + layer
        attention_2d = attn[0, head].cpu().numpy()
        return tensor_to_json(attention_2d)

    def simulate_policy(
        self, policy: str, budget: int, sink_count: int = 4
    ) -> List[bool]:
        """Simulate eviction policy and return keep mask."""
        attn = self._capture.get_latest()
        if attn is None:
            return []
        attention_2d = attn[0, 0].cpu().numpy()
        seq_len = attention_2d.shape[0]
        if policy == "lru":
            mask, _ = simulate_lru(attention_2d, budget)
        elif policy == "streaming_llm":
            mask = simulate_streaming_llm(seq_len, budget, sink_count)
        elif policy == "h2o":
            mask, _ = simulate_h2o(attention_2d, budget, sink_count)
        elif policy == "sliding_window":
            mask = simulate_sliding_window(seq_len, budget)
        else:
            mask = np.ones(seq_len, dtype=bool)
        return mask.tolist()

    def get_memory_stats(self) -> dict:
        """Get current memory statistics."""
        return self._profiler.snapshot()

    def analyze_attention(
        self, tokens: Optional[List[str]] = None
    ) -> Optional[AttentionFrame]:
        """Analyze current attention pattern."""
        attn = self._capture.get_latest()
        if attn is None:
            return None
        attention_2d = attn[0, 0].cpu().numpy()
        if tokens is None:
            tokens = [f"tok_{i}" for i in range(attention_2d.shape[0])]
        return create_attention_frame(attention_2d, tokens)
