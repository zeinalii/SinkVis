# SinkVis

**KV Cache Visualizer** — A Python library for understanding attention sinks and KV cache eviction policies in large language models.

[![PyPI version](https://badge.fury.io/py/sinkvis.svg)](https://pypi.org/project/sinkvis/)

## Installation

Install SinkVis from PyPI:

```bash
pip install sinkvis
```

For full functionality with Hugging Face Transformers models:

```bash
pip install sinkvis[transformers]
```

## Quick Start

### 1. Load a Model and Capture Attention

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sinkvis import SinkVis

# Load model
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval()

# Tokenize input
prompt = "The capital of France is Paris. The Eiffel Tower is a famous landmark."
inputs = tokenizer(prompt, return_tensors="pt")

# Capture attention patterns using SinkVis context manager
with SinkVis(model, tokenizer) as sv:
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Get attention data for the last layer, head 0
    attention_data = sv.get_attention_data(layer=-1, head=0)
```

### 2. Visualize Attention Sinks

```python
import numpy as np
import matplotlib.pyplot as plt

if attention_data is not None:
    attention_matrix = np.array(attention_data)
    tokens = [tokenizer.decode([tok]) for tok in inputs["input_ids"][0]]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(attention_matrix, cmap='viridis', aspect='auto')
    
    # Identify attention sinks (high average attention)
    avg_attention = attention_matrix.mean(axis=0)
    sink_threshold = 0.1
    sink_indices = np.where(avg_attention > sink_threshold)[0]
    
    # Highlight sinks
    for idx in sink_indices:
        ax.add_patch(plt.Rectangle((idx-0.5, -0.5), 1, len(tokens), 
                                   fill=False, edgecolor='red', linewidth=2))
    
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, ha='right')
    ax.set_title('Attention Pattern with Sinks Highlighted')
    plt.show()
```

### 3. Compare Eviction Policies

SinkVis supports multiple KV cache eviction policies:

- **Full Cache**: Keep all tokens (baseline)
- **Sliding Window**: Keep only the most recent N tokens
- **StreamingLLM**: Preserve attention sinks + sliding window
- **H2O**: Preserve sinks + heavy hitters (tokens with high attention scores)
- **LRU**: Keep tokens based on recency and attention scores

```python
from sinkvis.simulation import (
    simulate_sliding_window,
    simulate_streaming_llm,
    simulate_h2o,
    simulate_lru,
)

# Define cache budget
cache_budget = 20
sink_count = 4
seq_len = len(tokens)

# Run different policies
sliding_window_mask = simulate_sliding_window(seq_len, cache_budget)
streaming_llm_mask = simulate_streaming_llm(seq_len, cache_budget, sink_count)
h2o_mask, scores = simulate_h2o(attention_matrix, cache_budget, sink_count)
lru_mask, lru_scores = simulate_lru(attention_matrix, cache_budget)

# Compare results
print(f"Sliding Window: {sliding_window_mask.sum()} tokens kept")
print(f"StreamingLLM: {streaming_llm_mask.sum()} tokens kept")
print(f"H2O: {h2o_mask.sum()} tokens kept")
print(f"LRU: {lru_mask.sum()} tokens kept")
```

### 4. Needle in a Haystack Test

Test whether critical entities survive cache eviction:

```python
# Define prompt with specific entity
needle_prompt = (
    "Alice went to the market and bought fresh vegetables. "
    "The weather was nice and sunny. Many people were shopping. "
    "She met her friend Bob near the fruit stand."
)

entity = "Alice"

# Tokenize and find entity position
inputs = tokenizer(needle_prompt, return_tensors="pt")
tokens = [tokenizer.decode([tok]) for tok in inputs["input_ids"][0]]
entity_token_idx = next(i for i, t in enumerate(tokens) if entity.lower() in t.lower())

# Capture attention
with SinkVis(model, tokenizer) as sv:
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    attention_matrix = np.array(sv.get_attention_data(layer=-1, head=0))

# Test if entity is preserved
cache_budget = 20
h2o_mask, _ = simulate_h2o(attention_matrix, cache_budget, sink_count=4)
entity_preserved = h2o_mask[entity_token_idx]

print(f"Entity '{entity}' preserved: {'✓ YES' if entity_preserved else '✗ NO'}")
```

## API Reference

### SinkVis Class

Main API for capturing and analyzing attention patterns.

```python
from sinkvis import SinkVis

sv = SinkVis(model, tokenizer)

# Context manager usage
with sv:
    outputs = model(**inputs, output_attentions=True)
    attention_data = sv.get_attention_data(layer=-1, head=0)

# Methods
attention_data = sv.get_attention_data(layer=-1, head=0)  # Get attention matrix
mask = sv.simulate_policy("h2o", budget=20, sink_count=4)  # Simulate eviction
stats = sv.get_memory_stats()  # Get memory statistics
frame = sv.analyze_attention(tokens=tokens)  # Analyze attention pattern
```

### Eviction Policies

#### `simulate_sliding_window(seq_len, window_size)`

Keep only the most recent `window_size` tokens.

#### `simulate_streaming_llm(seq_len, budget, sink_count=4)`

Preserve first `sink_count` tokens (sinks) + most recent tokens up to `budget`.

#### `simulate_h2o(attention, budget, sink_count=4)`

Preserve sinks + tokens with highest attention scores (heavy hitters).

#### `simulate_lru(attention, budget)`

Keep tokens based on attention scores (sum across query positions).

### Models

```python
from sinkvis.models import EvictionPolicy, SimulationConfig

# Available policies
EvictionPolicy.FULL
EvictionPolicy.SLIDING_WINDOW
EvictionPolicy.STREAMING_LLM
EvictionPolicy.H2O
EvictionPolicy.LRU

# Configuration
config = SimulationConfig(
    policy=EvictionPolicy.H2O,
    cache_size=1024,
    sink_count=4,
    window_size=512,
    heavy_hitter_ratio=0.1
)
```

## Examples

See the complete demo notebook at `examples/sinkvis_demo.ipynb` for:

- Loading GPT-2 and capturing attention patterns
- Visualizing attention sinks with heatmaps
- Comparing all eviction policies side-by-side
- Needle in a haystack entity preservation test
- VRAM usage calculations
- Color-coded visualization of token retention

## Key Concepts

### Attention Sinks

Early tokens in transformer models often receive disproportionately high attention weights across all query positions. These "attention sinks" are critical for maintaining model performance and should be preserved during cache eviction.

### Eviction Policy Details

**Sliding Window**: Simple but may lose important early context.

```python
mask = simulate_sliding_window(seq_len=100, window_size=20)
```

**StreamingLLM**: Preserves sinks + recent window, balancing memory and performance.

```python
mask = simulate_streaming_llm(seq_len=100, budget=20, sink_count=4)
```

**H2O**: Attention-aware policy that preserves sinks + heavy hitters.

```python
mask, scores = simulate_h2o(attention_matrix, budget=20, sink_count=4)
```

## Requirements

- Python >= 3.10
- numpy >= 1.24.0
- torch >= 2.0.0

Optional:

- transformers >= 4.36.0 (for Hugging Face models)
- matplotlib (for visualization)

## License

MIT License

## Links

- **PyPI**: <https://pypi.org/project/sinkvis/>
- **Repository**: <https://github.com/zeinalii/SinkVis>
- **Issues**: <https://github.com/zeinalii/SinkVis/issues>
