"""PyTorch hooks for capturing attention patterns."""

import weakref
from collections import deque
from typing import Any, Deque, Optional

import torch
import torch.nn as nn


class AttentionCapture:
    """Captures attention matrices using forward hooks."""

    def __init__(self, model: nn.Module, buffer_size: int = 100):
        self._model_ref = weakref.ref(model)
        self.buffer: Deque[torch.Tensor] = deque(maxlen=buffer_size)
        self._hooks = []

    def start(self, pattern: str = r".*attn.*") -> None:
        """Register hooks on attention layers."""
        import re

        model = self._model_ref()
        if model is None:
            raise RuntimeError("Model has been garbage collected")
        regex = re.compile(pattern, re.IGNORECASE)
        for name, module in model.named_modules():
            if regex.search(name):
                hook = module.register_forward_hook(self._capture_hook)
                self._hooks.append(hook)

    def _capture_hook(self, module: nn.Module, inputs: tuple, output: Any) -> None:
        """Hook function to capture attention."""
        if isinstance(output, tuple):
            # Search for attention weights (4D tensor) in output tuple
            # Different models put attention at different indices:
            # - GPT-2: (attn_output, present, attn_weights) -> index 2
            # - Some models: (attn_output, attn_weights) -> index 1
            for item in output:
                if isinstance(item, torch.Tensor) and item.dim() == 4:
                    self.buffer.append(item.detach())
                    break

    def get_latest(self) -> Optional[torch.Tensor]:
        """Get most recent attention matrix."""
        return self.buffer[-1] if self.buffer else None

    def clear(self) -> None:
        """Clear buffer."""
        self.buffer.clear()

    def stop(self) -> None:
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
