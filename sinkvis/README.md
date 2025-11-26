# SinkVis

**KV Cache Visualizer** — A Python library for understanding attention sinks and KV cache eviction policies in large language models.

## Installation

```bash
# Basic installation
pip install .

# With HuggingFace transformers support
pip install ".[transformers]"

# With development dependencies
pip install ".[dev]"

# Everything
pip install ".[all]"
```

### From GitHub

```bash
pip install git+https://github.com/yourusername/SinkVis.git#subdirectory=py-package/sinkvis
```

## Quick Start

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sinkvis import SinkVis

# Load model
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Capture attention patterns
with SinkVis(model, tokenizer) as sv:
    inputs = tokenizer("Hello world", return_tensors="pt")
    outputs = model(**inputs, output_attentions=True)
    
    # Get attention data (layer -1 = last layer, head 0)
    attention_data = sv.get_attention_data(layer=-1, head=0)
    
    # Simulate eviction policy
    keep_mask = sv.simulate_policy("streaming_llm", budget=20, sink_count=4)
```

## Features

- **Attention Capture**: Hook into any PyTorch transformer model to capture attention patterns
- **Sink Detection**: Automatically identify attention sink tokens
- **Heavy Hitter Detection**: Find semantically important tokens with high attention
- **Eviction Simulation**: Compare policies (LRU, Sliding Window, StreamingLLM, H2O)
- **Memory Profiling**: Track GPU memory usage

## Eviction Policies

| Policy | Description |
|--------|-------------|
| `full` | No eviction (baseline) |
| `lru` | Least Recently Used |
| `sliding_window` | Keep only recent tokens |
| `streaming_llm` | Preserve sinks + sliding window |
| `h2o` | Heavy-Hitter Oracle |

## API Reference

### SinkVis

```python
class SinkVis:
    def __init__(self, model: nn.Module, tokenizer=None): ...
    def get_attention_data(self, layer: int = -1, head: int = 0) -> List[List[float]]: ...
    def simulate_policy(self, policy: str, budget: int, sink_count: int = 4) -> List[bool]: ...
    def get_memory_stats(self) -> dict: ...
    def analyze_attention(self, tokens: List[str] = None) -> AttentionFrame: ...
```

### Simulation Functions

```python
from sinkvis.simulation import (
    simulate_lru,
    simulate_sliding_window,
    simulate_streaming_llm,
    simulate_h2o,
)

# All return a boolean mask indicating which tokens to keep
mask = simulate_streaming_llm(seq_len=100, budget=20, sink_count=4)
```

## License

MIT License

