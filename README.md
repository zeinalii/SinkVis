# SinkVis

**KV Cache Visualizer** — An interactive tool for understanding KV cache behavior and eviction policies in large language models.

![SinkVis](https://img.shields.io/badge/version-1.0.0-00ffcc?style=flat-square)
![Python](https://img.shields.io/badge/python-3.10+-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## Overview

The key-value cache stores keys and values for attention over past tokens and is the primary memory cost for long context inference. At large context lengths, KV cache memory can reach many gigabytes, forcing developers to truncate history or reduce batch sizes.

Modern techniques like **StreamingLLM** and **H2O** rely on **attention sinks** (tokens that attract large attention scores but carry little semantic content) and **heavy hitters** (semantically crucial tokens). Understanding these patterns is vital for stability and performance.

SinkVis makes these mechanisms visible through interactive visualizations.

## Repository Structure

```
SinkVis/
├── py-package/           # Python package (sinkvis)
│   └── sinkvis/
│       ├── __init__.py   # Main SinkVis API
│       ├── attention.py  # Attention pattern generation & analysis
│       ├── eviction.py   # Eviction policy implementations
│       ├── hooks.py      # PyTorch hooks for capturing attention
│       ├── memory.py     # Memory profiling
│       ├── models.py     # Data models
│       ├── simulation.py # Vectorized simulation logic
│       ├── utils.py      # Utility functions
│       └── tests/        # Test suite
├── apps/                 # Example applications & notebooks
│   ├── README.md
│   └── sinkvis_demo.ipynb
└── basefiles/            # Configuration files
    ├── README.md         # This file
    ├── requirements.txt
    ├── .flake8
    └── .isort.cfg
```

## Features

### 🔴 Attention Capture
- Real-time attention pattern capture via PyTorch hooks
- Automatic sink and heavy hitter detection
- Support for HuggingFace transformers

### ⚙️ Eviction Policy Simulation
- Compare different cache eviction strategies:
  - **LRU** (Least Recently Used)
  - **Sliding Window**
  - **StreamingLLM** (preserves attention sinks)
  - **H2O** (Heavy-Hitter Oracle)
  - **Full Cache** (no eviction, for baseline comparison)
- Detailed metrics: hits, misses, evictions, retained sinks

### 📊 Memory Profiling
- GPU memory usage tracking
- VRAM estimation per policy

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/SinkVis.git
cd SinkVis

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r basefiles/requirements.txt

# Add sinkvis to your Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/py-package"
```

## Usage

### Basic Usage

```python
import sys
sys.path.insert(0, "path/to/SinkVis/py-package")

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sinkvis import SinkVis

# Load model
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Capture attention
with SinkVis(model, tokenizer) as sv:
    inputs = tokenizer("Hello world", return_tensors="pt")
    outputs = model(**inputs, output_attentions=True)
    
    # Get attention data
    attention_data = sv.get_attention_data(layer=-1, head=0)
    
    # Simulate eviction policy
    keep_mask = sv.simulate_policy("streaming_llm", budget=20, sink_count=4)
```

### Running the Demo Notebook

```bash
cd apps
jupyter notebook sinkvis_demo.ipynb
```

## Running Tests

```bash
cd py-package
pytest sinkvis/tests/ -v

# Run with coverage
pytest sinkvis/tests/ --cov=sinkvis
```

## Background

### Attention Sinks

Research has shown that the first few tokens in a sequence often receive disproportionately high attention scores, regardless of their semantic content. These "attention sinks" appear to serve as aggregation points for residual attention.

### Heavy Hitters

Beyond sinks, certain tokens are semantically important and receive consistently high attention across query positions. The H2O algorithm identifies and preserves these heavy hitters during cache eviction.

### References

- [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) (StreamingLLM)
- [H2O: Heavy-Hitter Oracle for Efficient Generative Inference](https://arxiv.org/abs/2306.14048)
- [PagedAttention](https://arxiv.org/abs/2309.06180) (vLLM)

## License

MIT License — see LICENSE file for details.

---

Built with ❤️ for understanding transformer attention

