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

## Features

### 🔴 Live Attention Streaming
- Real-time attention heatmaps over the context window
- Automatic sink and heavy hitter detection and highlighting
- WebSocket-based streaming for low-latency updates
- Configurable thresholds and update intervals

### ⚙️ Eviction Policy Simulation
- Compare different cache eviction strategies:
  - **LRU** (Least Recently Used)
  - **Sliding Window**
  - **StreamingLLM** (preserves attention sinks)
  - **H2O** (Heavy-Hitter Oracle)
  - **Full Cache** (no eviction, for baseline comparison)
- Detailed metrics: hits, misses, evictions, retained sinks
- Side-by-side policy comparison

### 📊 Hierarchical Cache Profiling
- Visualize where KV blocks reside in memory hierarchy:
  - GPU HBM
  - GPU L2 Cache
  - System RAM
  - Disk (offloaded)
- Block-level details: size, access patterns, importance markers

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
pip install -r requirements.txt

# Run the server
python run.py
```

Open your browser to **http://localhost:8765**

## Usage

### Live Attention Stream

1. Navigate to the **Live Stream** tab
2. Click **Start** to begin streaming attention patterns
3. Watch the heatmap update in real-time as tokens are processed
4. Tokens highlighted in red are attention sinks, blue are heavy hitters
5. Use **Step** for frame-by-frame analysis

### Eviction Simulation

1. Go to the **Eviction Sim** tab
2. Enter a prompt in the text area
3. Configure cache parameters:
   - **Cache Size**: Maximum tokens to retain
   - **Sink Count**: Number of initial sink tokens to preserve
   - **Window Size**: For sliding window policies
4. Click **Run Simulation** or **Compare All Policies**

### Cache Profile

1. Select the **Cache Profile** tab
2. Set the sequence length to analyze
3. Click **Refresh** to generate the memory hierarchy visualization
4. Hover over blocks to see detailed information

## Project Structure

```
SinkVis/
├── backend/
│   ├── __init__.py
│   ├── attention.py      # Attention pattern generation & analysis
│   ├── eviction.py       # Eviction policy implementations
│   ├── hf_loader.py      # HuggingFace model loading
│   ├── models.py         # Pydantic data models
│   ├── server.py         # FastAPI server
│   └── tests/
│       ├── test_attention.py
│       ├── test_eviction.py
│       └── test_server.py
├── frontend/
│   ├── index.html        # Main HTML page
│   ├── styles.css        # Styling
│   └── app.js            # Frontend application
├── requirements.txt
├── run.py                # Entry point
└── README.md
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve main visualization page |
| `/api/health` | GET | Health check |
| `/api/simulate` | POST | Run eviction simulation |
| `/api/compare` | POST | Compare all eviction policies |
| `/api/cache-profile` | GET | Get cache profile snapshot |
| `/api/models/search` | GET | Search HuggingFace models |
| `/api/models/load` | POST | Load a model from HuggingFace |
| `/api/models/generate` | POST | Generate text with loaded model |
| `/ws/attention` | WebSocket | Live attention streaming |

## Configuration

### Stream Configuration

```javascript
{
    "update_interval_ms": 200,    // Update frequency
    "sink_threshold": 0.1,        // Threshold for sink detection
    "heavy_hitter_threshold": 0.05 // Threshold for heavy hitter detection
}
```

### Simulation Configuration

```json
{
    "policy": "streaming_llm",
    "cache_size": 2048,
    "sink_count": 4,
    "window_size": 1024,
    "heavy_hitter_ratio": 0.1
}
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=backend

# Run specific test file
pytest backend/tests/test_attention.py -v
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
