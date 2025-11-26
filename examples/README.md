# SinkVis Examples

This folder contains interactive Jupyter notebooks demonstrating SinkVis features.

## Notebooks

### 1. `sinkvis_demo.ipynb` ⭐ **START HERE**
**Comprehensive demonstration with GPT-2 and Needle in a Haystack test**

Features demonstrated:
- ✅ Load GPT-2 and capture real attention patterns
- ✅ Use SinkVis context manager for instrumentation
- ✅ Visualize attention sinks with heatmaps
- ✅ Compare 4 eviction policies (Full, Sliding Window, StreamingLLM, H2O)
- ✅ **Needle in a Haystack**: Track if critical entities survive eviction
- ✅ Calculate VRAM usage for each policy
- ✅ Color-coded visualization (Green = preserved, Red = evicted)

**Perfect for:** Understanding cache eviction and entity preservation

---

## Quick Start

```bash
# Install dependencies
pip install -r ../basefiles/requirements.txt

# Launch Jupyter
jupyter notebook

# Open sinkvis_demo.ipynb and run all cells
```

## Running in Google Colab

All notebooks are Colab-compatible! Just:

1. Upload the notebook to Colab
2. Run the setup cells (they'll install dependencies)
3. The notebooks will automatically clone SinkVis

## Example Output

### Attention Heatmap
The notebooks generate visualizations like:
- **Attention heatmaps** showing which tokens attend to which
- **Sink detection** highlighting tokens receiving high attention
- **Heavy hitter identification** showing semantically important tokens
- **Policy comparisons** with bar charts and metrics

### Policy Performance
Compare eviction strategies:
- **FULL** - Baseline (no eviction)
- **LRU** - Least Recently Used
- **Sliding Window** - Keep recent tokens
- **StreamingLLM** - Sinks + window
- **H2O** - Heavy-Hitter Oracle

## Key Insights from Examples

From running the notebooks, you'll learn:

1. **Attention sinks appear consistently** at the beginning of sequences
2. **StreamingLLM and H2O** significantly outperform naive policies
3. **Heavy hitters** are often content words with high semantic value
4. **Memory profiling** helps optimize deployment strategies

## Troubleshooting

### "No module named 'sinkvis'"
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "py-package"))
```

### "Model not found"
The first run will download the model (~500MB for GPT-2). Subsequent runs use cached versions.

### Memory issues
If you encounter OOM errors:
- Use smaller models (gpt2 instead of gpt2-large)
- Reduce sequence length
- Clear GPU memory: `torch.cuda.empty_cache()`

## Next Steps

After running the examples:

1. **Experiment** with different models
2. **Modify** prompts and sequences
3. **Implement** custom eviction policies
4. **Integrate** SinkVis into your projects

## Support

- See main `README.md` in basefiles for project overview

---

**Built with SinkVis - Making attention patterns visible** 🔍

