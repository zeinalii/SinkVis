"""Model architecture analysis and visualization data."""

from dataclasses import dataclass
from typing import Optional, Any
from pydantic import BaseModel, Field


class LayerInfo(BaseModel):
    """Information about a single layer in the model."""
    
    layer_idx: int
    layer_type: str
    num_heads: int
    head_dim: int
    hidden_size: int
    intermediate_size: Optional[int] = None
    has_mlp: bool = True
    attention_type: str = "self"  # self, cross, grouped-query
    num_kv_heads: Optional[int] = None  # For GQA models
    

class AttentionConfig(BaseModel):
    """Attention mechanism configuration."""
    
    num_heads: int
    head_dim: int
    num_kv_heads: Optional[int] = None  # For GQA
    is_gqa: bool = False  # Grouped Query Attention
    is_mqa: bool = False  # Multi-Query Attention
    rotary_embedding: bool = False
    max_position_embeddings: int = 2048
    attention_dropout: float = 0.0


class KVCacheMetrics(BaseModel):
    """KV Cache metrics and projections."""
    
    # Per-layer cache size
    kv_cache_per_layer_bytes: int
    total_kv_cache_bytes: int
    
    # Memory projections at different sequence lengths
    memory_at_seq_lengths: dict[int, int] = Field(default_factory=dict)
    
    # Efficiency metrics
    bytes_per_token: int
    theoretical_max_context: int
    
    # For comparison
    model_params_bytes: int
    kv_cache_ratio: float  # kv_cache / model_params at max context


class ModelArchitecture(BaseModel):
    """Complete model architecture information."""
    
    model_id: str
    model_type: str
    
    # Core dimensions
    num_layers: int
    num_heads: int
    hidden_size: int
    intermediate_size: int = 0  # Default if not available
    vocab_size: int
    
    # Attention details
    attention_config: AttentionConfig
    
    # Per-layer information
    layers: list[LayerInfo]
    
    # Memory and cache info
    kv_cache_metrics: KVCacheMetrics
    
    # Additional architecture features
    tie_word_embeddings: bool = True
    use_cache: bool = True
    rope_theta: Optional[float] = None
    sliding_window: Optional[int] = None
    
    # Computed properties
    total_params: int = 0
    total_params_billions: float = 0.0
    dtype: str = "float16"
    device: str = "cpu"


def analyze_model_architecture(model, tokenizer, model_id: str) -> ModelArchitecture:
    """
    Analyze a loaded model and extract architecture details.
    
    Args:
        model: The loaded transformer model
        tokenizer: The model's tokenizer
        model_id: HuggingFace model identifier
    
    Returns:
        ModelArchitecture with complete analysis
    """
    config = model.config
    
    # Extract basic dimensions
    num_layers = getattr(config, 'num_hidden_layers', 
                 getattr(config, 'n_layer',
                 getattr(config, 'num_layers', 12)))
    
    num_heads = getattr(config, 'num_attention_heads',
                getattr(config, 'n_head',
                getattr(config, 'num_heads', 12)))
    
    hidden_size = getattr(config, 'hidden_size',
                  getattr(config, 'n_embd', 768))
    
    intermediate_size = getattr(config, 'intermediate_size',
                        getattr(config, 'n_inner', None))
    if intermediate_size is None:
        intermediate_size = hidden_size * 4  # Default FFN size
    
    vocab_size = getattr(config, 'vocab_size', 50257)
    
    # Attention configuration
    num_kv_heads = getattr(config, 'num_key_value_heads', 
                   getattr(config, 'num_kv_heads', None))
    
    head_dim = hidden_size // num_heads
    
    is_gqa = num_kv_heads is not None and num_kv_heads < num_heads and num_kv_heads > 1
    is_mqa = num_kv_heads == 1
    
    rotary = getattr(config, 'rope_scaling', None) is not None or \
             getattr(config, 'rotary_emb_base', None) is not None or \
             hasattr(config, 'rope_theta')
    
    max_position = getattr(config, 'max_position_embeddings',
                   getattr(config, 'n_positions', 2048))
    
    attention_dropout = getattr(config, 'attention_dropout',
                        getattr(config, 'attn_pdrop', 0.0))
    
    attention_config = AttentionConfig(
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        is_gqa=is_gqa,
        is_mqa=is_mqa,
        rotary_embedding=rotary,
        max_position_embeddings=max_position,
        attention_dropout=attention_dropout,
    )
    
    # Build layer information
    layers = []
    for i in range(num_layers):
        layer_info = LayerInfo(
            layer_idx=i,
            layer_type="TransformerBlock",
            num_heads=num_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            has_mlp=True,
            attention_type="grouped-query" if is_gqa else ("multi-query" if is_mqa else "self"),
            num_kv_heads=num_kv_heads,
        )
        layers.append(layer_info)
    
    # Calculate KV cache metrics
    # KV cache size per layer = 2 (K+V) * seq_len * num_kv_heads * head_dim * dtype_size
    effective_kv_heads = num_kv_heads if num_kv_heads else num_heads
    dtype_size = 2  # float16 = 2 bytes
    
    # KV cache per layer at sequence length 1
    kv_per_token_per_layer = 2 * effective_kv_heads * head_dim * dtype_size
    kv_per_layer_at_2048 = kv_per_token_per_layer * 2048
    total_kv_at_2048 = kv_per_layer_at_2048 * num_layers
    
    # Memory projections at different sequence lengths
    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    memory_projections = {}
    for seq_len in seq_lengths:
        if seq_len <= max_position:
            kv_size = kv_per_token_per_layer * seq_len * num_layers
            memory_projections[seq_len] = kv_size
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    total_params_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Theoretical max context (assuming 80GB GPU)
    available_memory = 80 * 1024 * 1024 * 1024  # 80GB in bytes
    memory_after_model = available_memory - total_params_bytes
    bytes_per_token_total = kv_per_token_per_layer * num_layers
    theoretical_max = memory_after_model // bytes_per_token_total if bytes_per_token_total > 0 else 0
    
    kv_cache_metrics = KVCacheMetrics(
        kv_cache_per_layer_bytes=kv_per_layer_at_2048,
        total_kv_cache_bytes=total_kv_at_2048,
        memory_at_seq_lengths=memory_projections,
        bytes_per_token=bytes_per_token_total,
        theoretical_max_context=min(theoretical_max, max_position),
        model_params_bytes=total_params_bytes,
        kv_cache_ratio=total_kv_at_2048 / total_params_bytes if total_params_bytes > 0 else 0,
    )
    
    # Additional config
    tie_embeddings = getattr(config, 'tie_word_embeddings', True)
    use_cache = getattr(config, 'use_cache', True)
    rope_theta = getattr(config, 'rope_theta', None)
    sliding_window = getattr(config, 'sliding_window', None)
    
    # Get dtype and device
    dtype = str(next(model.parameters()).dtype).split('.')[-1]
    device = str(next(model.parameters()).device)
    
    # Model type
    model_type = getattr(config, 'model_type', 'unknown')
    
    return ModelArchitecture(
        model_id=model_id,
        model_type=model_type,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size,
        attention_config=attention_config,
        layers=layers,
        kv_cache_metrics=kv_cache_metrics,
        tie_word_embeddings=tie_embeddings,
        use_cache=use_cache,
        rope_theta=rope_theta,
        sliding_window=sliding_window,
        total_params=total_params,
        total_params_billions=total_params / 1e9,
        dtype=dtype,
        device=device,
    )


class LayerActivationStats(BaseModel):
    """Statistics about activations at a specific layer."""
    
    layer_idx: int
    attention_entropy: float  # Measure of attention distribution
    attention_sparsity: float  # % of near-zero attention weights
    top_k_concentration: float  # Attention concentrated in top-k positions
    sink_attention_ratio: float  # Ratio of attention going to sink tokens
    local_attention_ratio: float  # Ratio of attention in local window


class PerLayerAnalysis(BaseModel):
    """Per-layer analysis of attention patterns."""
    
    layer_stats: list[LayerActivationStats]
    sink_positions: list[int]
    heavy_hitter_positions: list[int]
    average_entropy: float
    average_sparsity: float


def analyze_layer_attention(
    attention_weights: list,  # List of [batch, heads, seq, seq] per layer
    sink_threshold: float = 0.1,
    heavy_hitter_threshold: float = 0.05,
) -> PerLayerAnalysis:
    """
    Analyze attention patterns across all layers.
    
    Args:
        attention_weights: Attention weights from all layers
        sink_threshold: Threshold for identifying sinks
        heavy_hitter_threshold: Threshold for heavy hitters
    
    Returns:
        PerLayerAnalysis with detailed per-layer statistics
    """
    import numpy as np
    
    layer_stats = []
    all_sink_scores = []
    all_heavy_scores = []
    
    for layer_idx, layer_attn in enumerate(attention_weights):
        # Convert to numpy and average over batch and heads
        attn = np.array(layer_attn)
        if attn.ndim == 4:
            attn = attn[0].mean(axis=0)  # [seq, seq]
        elif attn.ndim == 3:
            attn = attn.mean(axis=0)
        
        seq_len = attn.shape[0]
        
        # Calculate entropy (measure of attention distribution)
        # Higher entropy = more uniform attention
        entropy = 0.0
        for i in range(seq_len):
            row = attn[i, :i+1]
            row = row / (row.sum() + 1e-9)
            row = row[row > 0]
            if len(row) > 0:
                entropy -= (row * np.log(row + 1e-9)).sum()
        entropy /= seq_len
        
        # Calculate sparsity (% of weights below threshold)
        sparsity = (attn < 0.01).sum() / attn.size
        
        # Top-k concentration
        k = min(5, seq_len)
        top_k_ratio = 0.0
        for i in range(seq_len):
            row = attn[i, :i+1]
            if len(row) > 0:
                top_k = np.sort(row)[-k:]
                top_k_ratio += top_k.sum() / (row.sum() + 1e-9)
        top_k_ratio /= seq_len
        
        # Sink attention ratio (attention to first 4 positions)
        sink_count = min(4, seq_len)
        sink_ratio = 0.0
        for i in range(sink_count, seq_len):
            sink_ratio += attn[i, :sink_count].sum() / (attn[i, :i+1].sum() + 1e-9)
        sink_ratio /= max(1, seq_len - sink_count)
        
        # Local attention ratio (within window of 64)
        window = 64
        local_ratio = 0.0
        for i in range(seq_len):
            start = max(0, i - window)
            local_attn = attn[i, start:i+1].sum()
            total_attn = attn[i, :i+1].sum() + 1e-9
            local_ratio += local_attn / total_attn
        local_ratio /= seq_len
        
        layer_stats.append(LayerActivationStats(
            layer_idx=layer_idx,
            attention_entropy=float(entropy),
            attention_sparsity=float(sparsity),
            top_k_concentration=float(top_k_ratio),
            sink_attention_ratio=float(sink_ratio),
            local_attention_ratio=float(local_ratio),
        ))
        
        # Track sink/heavy hitter candidates across layers
        col_sums = attn.sum(axis=0)
        for pos in range(min(16, seq_len)):
            if col_sums[pos] / (col_sums.sum() + 1e-9) > sink_threshold:
                all_sink_scores.append(pos)
        
        for pos in range(seq_len):
            if col_sums[pos] / (col_sums.sum() + 1e-9) > heavy_hitter_threshold:
                if pos >= 4:  # Exclude initial positions
                    all_heavy_scores.append(pos)
    
    # Find consistent sinks and heavy hitters
    from collections import Counter
    sink_counter = Counter(all_sink_scores)
    heavy_counter = Counter(all_heavy_scores)
    
    # Positions that appear as sinks in more than half the layers
    num_layers = len(attention_weights)
    sink_positions = [pos for pos, count in sink_counter.items() if count > num_layers // 2]
    heavy_positions = [pos for pos, count in heavy_counter.items() if count > num_layers // 3]
    
    avg_entropy = sum(s.attention_entropy for s in layer_stats) / len(layer_stats)
    avg_sparsity = sum(s.attention_sparsity for s in layer_stats) / len(layer_stats)
    
    return PerLayerAnalysis(
        layer_stats=layer_stats,
        sink_positions=sorted(set(sink_positions)),
        heavy_hitter_positions=sorted(set(heavy_positions)),
        average_entropy=avg_entropy,
        average_sparsity=avg_sparsity,
    )

