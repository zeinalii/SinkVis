# SinkVis - How to Use Guide

A step-by-step guide to using all features of SinkVis, the Attention Sink & KV Cache Visualizer.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Model Hub - Loading Models](#model-hub---loading-models)
3. [Live Attention Stream](#live-attention-stream)
4. [Eviction Policy Simulation](#eviction-policy-simulation)
5. [Cache Profile View](#cache-profile-view)
6. [Tips & Troubleshooting](#tips--troubleshooting)

---

## Getting Started

### Prerequisites

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the server:**
   ```bash
   python run.py
   ```

3. **Open in browser:**
   Navigate to `http://localhost:8765`

### First Steps

When you open SinkVis, you'll land on the **Model Hub** tab. This is intentional - you need to load a model before using other features.

---

## Model Hub - Loading Models

The Model Hub is your starting point for loading transformer models from Hugging Face.

### Step 1: Search for a Model

1. In the **search bar**, type a model name:
   - `gpt2` - Small, fast, good for testing
   - `distilgpt2` - Even smaller and faster
   - `microsoft/phi-2` - More capable but larger
   - `meta-llama/Llama-2-7b-hf` - Requires HF token (gated)

2. Click **Search** or press Enter

3. Browse the results showing:
   - Model name
   - Download count
   - Likes
   - Tags (e.g., "text-generation")

### Step 2: Select a Model

1. Click on any model card to select it
2. The **Model Info Card** will show details:
   - Author
   - Download/like counts
   - Pipeline type
   - Tags
   - Gated status (if authentication required)

### Step 3: Configure Loading Options

**Device:**
- `Auto` - Automatically selects best available (CUDA > MPS > CPU)
- `CUDA GPU` - NVIDIA GPU acceleration
- `Apple Silicon` - M1/M2/M3 Metal acceleration
- `CPU` - Slower but always works

**Precision:**
- `Auto` - Float16 for GPU, Float32 for CPU
- `Float16` - Faster, less memory, slightly less accurate
- `BFloat16` - Good balance (requires compatible hardware)
- `Float32` - Most accurate, uses more memory

**HuggingFace Token:**
- If you see a green checkmark, your token was auto-detected from:
  - `~/.cache/huggingface/token` (from `huggingface-cli login`)
  - Environment variable `HF_TOKEN`
  - `.env` file in project root
- For gated models (like Llama), you need a valid token

**Trust Remote Code:**
- Enable for models that require custom code (e.g., some Phi models)
- Only enable for models you trust

### Step 4: Load the Model

1. Click **Load Model**
2. Watch the progress:
   - "Downloading..." - Fetching model weights
   - "Loading tokenizer..." - Preparing tokenizer
   - "Loading model weights..." - Loading into memory
3. Once loaded, you'll see:
   - Model specs (layers, heads, hidden size, vocab size)
   - Device and precision info
   - Memory usage

### Step 5: Quick Attention Test

After loading, test the model immediately:

1. Enter text in the **Quick Attention Test** panel
2. Set **Layer** (-1 for last layer, 0 for first)
3. Optionally set a specific **Head** number
4. Click **Get Attention**
5. View the attention heatmap showing token relationships

### Step 6: Navigate to Features

Use the **Quick Actions** buttons to jump to:
- **Live Stream** - Watch attention in real-time
- **Eviction Sim** - Compare cache policies
- **Cache Profile** - View memory hierarchy

---

## Live Attention Stream

Visualize attention patterns as tokens are processed one by one.

### How to Use

1. **Enter your prompt** in the text area at the top
   ```
   The quick brown fox jumps over the lazy dog.
   ```

2. **Click Start** to begin streaming
   - Tokens appear one by one
   - Heatmap updates in real-time
   - Sink tokens highlighted in pink
   - Heavy hitter tokens highlighted in blue

3. **Controls:**
   - **Start** - Begin streaming from the beginning
   - **Stop** - Pause the stream
   - **Step** - Advance one token at a time
   - **Reset** - Clear and prepare for new stream

### Configuration Options

**Update Interval (ms):**
- Controls speed of token processing
- Lower = faster (50ms minimum)
- Higher = slower, easier to observe (up to 1000ms)

**Sink Threshold:**
- Adjusts sensitivity for detecting attention sinks
- Higher value = fewer tokens marked as sinks
- Default: 0.10 (10%)

### Understanding the Heatmap

- **X-axis**: Key positions (tokens being attended to)
- **Y-axis**: Query positions (tokens doing the attending)
- **Color intensity**: Attention weight strength
  - Dark = low attention
  - Bright yellow = high attention
- **Pink vertical lines**: Attention sink positions
- **Blue vertical lines**: Heavy hitter positions

### What to Look For

1. **Attention Sinks**: First few tokens often receive high attention regardless of content
2. **Diagonal Pattern**: Self-attention (token attends to itself)
3. **Vertical Stripes**: Tokens that many others attend to
4. **Local Patterns**: Nearby token relationships

---

## Eviction Policy Simulation

Compare how different KV cache eviction policies handle long sequences.

### How to Use

1. **Enter a prompt** in the text area
   - Longer prompts show more interesting eviction behavior
   - Try 100+ tokens for meaningful results

2. **Configure settings:**

   **Eviction Policy:**
   - `StreamingLLM` - Keeps sink tokens + recent window
   - `H2O (Heavy-Hitter Oracle)` - Keeps important tokens based on attention
   - `LRU (Least Recently Used)` - Evicts oldest accessed tokens
   - `Sliding Window` - Fixed window of recent tokens
   - `Full Cache` - No eviction (baseline comparison)

   **Cache Size:** Maximum tokens in cache (default: 256)
   
   **Sink Count:** Number of initial tokens to always keep (default: 4)
   
   **Window Size:** Size of recent token window (default: 128)

3. **Run Simulation:**
   - Click **Run Simulation** for single policy
   - Click **Compare All Policies** to see all side-by-side

### Understanding Results

**Metrics explained:**
- **Tokens Processed**: Total tokens handled
- **Cache Hits**: Tokens found in cache (good)
- **Cache Misses**: Tokens not in cache (requires recomputation)
- **Evictions**: Tokens removed from cache
- **Retained Sinks**: Sink tokens still in cache
- **Retained Heavy Hitters**: Important tokens still in cache
- **Hit Rate**: Percentage of hits (higher is better)

### Comparing Policies

When you click **Compare All Policies**, you'll see all five policies side by side:

| Policy | Best For |
|--------|----------|
| StreamingLLM | Long conversations, infinite context |
| H2O | Tasks requiring semantic memory |
| LRU | General purpose, simple |
| Sliding Window | Recent context focus |
| Full Cache | Maximum accuracy (memory permitting) |

---

## Cache Profile View

Visualize the hierarchical memory layout of the KV cache.

### How to Use

1. Set the **Sequence Length** (how many tokens to simulate)
2. Click **Refresh** to generate the profile

### Memory Hierarchy

The visualization shows four memory tiers:

1. **GPU HBM** (Green) - Fastest, most frequently accessed
2. **GPU L2 Cache** (Blue) - Fast, hot data
3. **System RAM** (Yellow) - Larger, slower
4. **Disk** (Pink) - Offloaded, slowest

### Block Map

- Each small square represents a cache block
- **Color**: Indicates memory tier
- **Pink border**: Contains attention sink
- **Blue border**: Contains heavy hitter

### Block Details

Hover over any block to see:
- Block ID
- Token range (which tokens it contains)
- Memory tier location
- Size in bytes
- Access count
- Status (Sink/Heavy Hitter/Normal)

---

## Tips & Troubleshooting

### Performance Tips

1. **Use smaller models for testing**: `distilgpt2` loads in seconds
2. **Use Float16**: Halves memory usage with minimal accuracy loss
3. **Start with short prompts**: Debug with 10-20 tokens first

### Common Issues

**"Model Required" overlay won't go away:**
- Make sure you've loaded a model in the Model Hub tab
- Check for loading errors in the status panel

**Model loading fails:**
- Check your internet connection
- For gated models, ensure HF token is valid
- Try a smaller model first
- Check available memory (some models need 16GB+ RAM)

**Slow streaming:**
- Increase the update interval
- Use a smaller model
- Switch to CPU if GPU memory is exhausted

**"Address already in use" error:**
- Kill existing server: `lsof -ti:8765 | xargs kill -9`
- Or use a different port in `run.py`

### Keyboard Shortcuts

- **Enter** in search box: Search models
- **Enter** in prompt fields: Submit

### Recommended Models by Hardware

| Hardware | Recommended Models |
|----------|-------------------|
| CPU only | `distilgpt2`, `gpt2` |
| 8GB GPU | `gpt2-medium`, `microsoft/phi-2` |
| 16GB+ GPU | `gpt2-large`, `meta-llama/Llama-2-7b-hf` |
| Apple Silicon | `gpt2`, `microsoft/phi-2` (use MPS) |

---

## Quick Reference

### Typical Workflow

```
1. Model Hub → Search "distilgpt2" → Load
2. Live Stream → Enter prompt → Start → Observe patterns
3. Eviction Sim → Compare policies → Analyze results
4. Cache Profile → Refresh → Explore memory layout
```

### What Each Tab Shows

| Tab | Purpose | Key Insight |
|-----|---------|-------------|
| Model Hub | Load models | Model architecture details |
| Live Stream | Real-time attention | Sink/heavy hitter patterns |
| Eviction Sim | Policy comparison | Cache efficiency metrics |
| Cache Profile | Memory layout | Hierarchical data placement |

---

## Need Help?

- Check the [README.md](README.md) for project overview
- Review error messages in browser console (F12)
- Check server terminal for backend errors

