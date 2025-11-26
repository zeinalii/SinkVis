"""Utility functions for SinkVis."""

from typing import List, Union

import numpy as np
import torch


def tensor_to_json(
    tensor: Union[torch.Tensor, np.ndarray],
    decimals: int = 4,
) -> List:
    """Convert tensor to JSON-serializable list."""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    if isinstance(tensor, np.ndarray):
        tensor = np.round(tensor, decimals)
        return tensor.tolist()
    return tensor


def format_tokens(token_ids: List[int], token_strings: List[str]) -> List:
    """Format tokens for frontend display."""
    return [{"id": tid, "text": tstr} for tid, tstr in zip(token_ids, token_strings)]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide, returning default if denominator is zero."""
    return numerator / denominator if denominator != 0 else default
