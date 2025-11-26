"""Tests for model architecture analysis and visualization."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from backend.architecture import (
    LayerInfo,
    AttentionConfig,
    KVCacheMetrics,
    ModelArchitecture,
    analyze_model_architecture,
    LayerActivationStats,
    PerLayerAnalysis,
    analyze_layer_attention,
)


class TestLayerInfo:
    """Tests for LayerInfo model."""

    def test_basic_layer_info(self):
        layer = LayerInfo(
            layer_idx=0,
            layer_type="TransformerBlock",
            num_heads=12,
            head_dim=64,
            hidden_size=768,
        )
        assert layer.layer_idx == 0
        assert layer.num_heads == 12
        assert layer.head_dim == 64
        assert layer.has_mlp is True  # Default
        assert layer.attention_type == "self"  # Default

    def test_gqa_layer_info(self):
        layer = LayerInfo(
            layer_idx=5,
            layer_type="TransformerBlock",
            num_heads=32,
            head_dim=128,
            hidden_size=4096,
            num_kv_heads=8,
            attention_type="grouped-query",
        )
        assert layer.num_kv_heads == 8
        assert layer.attention_type == "grouped-query"


class TestAttentionConfig:
    """Tests for AttentionConfig model."""

    def test_basic_attention_config(self):
        config = AttentionConfig(
            num_heads=12,
            head_dim=64,
        )
        assert config.num_heads == 12
        assert config.head_dim == 64
        assert config.is_gqa is False
        assert config.is_mqa is False
        assert config.rotary_embedding is False

    def test_gqa_config(self):
        config = AttentionConfig(
            num_heads=32,
            head_dim=128,
            num_kv_heads=8,
            is_gqa=True,
            rotary_embedding=True,
            max_position_embeddings=4096,
        )
        assert config.is_gqa is True
        assert config.num_kv_heads == 8
        assert config.rotary_embedding is True
        assert config.max_position_embeddings == 4096


class TestKVCacheMetrics:
    """Tests for KVCacheMetrics model."""

    def test_kv_cache_metrics(self):
        metrics = KVCacheMetrics(
            kv_cache_per_layer_bytes=1024 * 1024,  # 1MB per layer
            total_kv_cache_bytes=12 * 1024 * 1024,  # 12MB total
            bytes_per_token=1536,
            theoretical_max_context=32768,
            model_params_bytes=500 * 1024 * 1024,  # 500MB
            kv_cache_ratio=0.024,
        )
        assert metrics.kv_cache_per_layer_bytes == 1024 * 1024
        assert metrics.bytes_per_token == 1536
        assert metrics.theoretical_max_context == 32768

    def test_memory_projections(self):
        metrics = KVCacheMetrics(
            kv_cache_per_layer_bytes=0,
            total_kv_cache_bytes=0,
            memory_at_seq_lengths={512: 1000, 1024: 2000, 2048: 4000},
            bytes_per_token=100,
            theoretical_max_context=2048,
            model_params_bytes=1000000,
            kv_cache_ratio=0.1,
        )
        assert 512 in metrics.memory_at_seq_lengths
        assert metrics.memory_at_seq_lengths[1024] == 2000


class TestModelArchitecture:
    """Tests for ModelArchitecture model."""

    def test_basic_architecture(self):
        arch = ModelArchitecture(
            model_id="test/model",
            model_type="gpt2",
            num_layers=6,
            num_heads=12,
            hidden_size=768,
            intermediate_size=3072,
            vocab_size=50257,
            attention_config=AttentionConfig(num_heads=12, head_dim=64),
            layers=[],
            kv_cache_metrics=KVCacheMetrics(
                kv_cache_per_layer_bytes=0,
                total_kv_cache_bytes=0,
                bytes_per_token=0,
                theoretical_max_context=1024,
                model_params_bytes=0,
                kv_cache_ratio=0,
            ),
        )
        assert arch.model_id == "test/model"
        assert arch.num_layers == 6
        assert arch.intermediate_size == 3072

    def test_intermediate_size_default(self):
        """Test that intermediate_size defaults to 0 when not provided."""
        arch = ModelArchitecture(
            model_id="test/model",
            model_type="gpt2",
            num_layers=6,
            num_heads=12,
            hidden_size=768,
            vocab_size=50257,
            attention_config=AttentionConfig(num_heads=12, head_dim=64),
            layers=[],
            kv_cache_metrics=KVCacheMetrics(
                kv_cache_per_layer_bytes=0,
                total_kv_cache_bytes=0,
                bytes_per_token=0,
                theoretical_max_context=1024,
                model_params_bytes=0,
                kv_cache_ratio=0,
            ),
        )
        assert arch.intermediate_size == 0


class TestAnalyzeModelArchitecture:
    """Tests for analyze_model_architecture function."""

    def create_mock_model(
        self,
        num_layers=6,
        num_heads=12,
        hidden_size=768,
        vocab_size=50257,
        intermediate_size=3072,
        max_position_embeddings=1024,
        model_type="gpt2",
    ):
        """Create a mock model for testing."""
        mock_model = MagicMock()
        mock_config = MagicMock()
        
        mock_config.num_hidden_layers = num_layers
        mock_config.num_attention_heads = num_heads
        mock_config.hidden_size = hidden_size
        mock_config.vocab_size = vocab_size
        mock_config.intermediate_size = intermediate_size
        mock_config.max_position_embeddings = max_position_embeddings
        mock_config.model_type = model_type
        mock_config.num_key_value_heads = None
        mock_config.rope_scaling = None
        mock_config.rotary_emb_base = None
        mock_config.attention_dropout = 0.0
        mock_config.tie_word_embeddings = True
        mock_config.use_cache = True
        mock_config.rope_theta = None
        mock_config.sliding_window = None
        
        mock_model.config = mock_config
        
        # Mock parameters for memory calculation - use a function to return fresh iterator
        mock_param = MagicMock()
        mock_param.numel.return_value = 100000000  # 100M params
        mock_param.element_size.return_value = 2  # float16
        mock_param.dtype = "torch.float16"
        mock_param.device = "cpu"
        
        # Return a new iterator each time parameters() is called
        mock_model.parameters = lambda: iter([mock_param])
        
        return mock_model

    def test_basic_architecture_extraction(self):
        mock_model = self.create_mock_model()
        mock_tokenizer = MagicMock()
        
        arch = analyze_model_architecture(mock_model, mock_tokenizer, "test/gpt2")
        
        assert arch.model_id == "test/gpt2"
        assert arch.model_type == "gpt2"
        assert arch.num_layers == 6
        assert arch.num_heads == 12
        assert arch.hidden_size == 768

    def test_layer_info_generation(self):
        mock_model = self.create_mock_model(num_layers=4)
        mock_tokenizer = MagicMock()
        
        arch = analyze_model_architecture(mock_model, mock_tokenizer, "test/model")
        
        assert len(arch.layers) == 4
        for i, layer in enumerate(arch.layers):
            assert layer.layer_idx == i
            assert layer.num_heads == 12
            assert layer.hidden_size == 768

    def test_kv_cache_metrics_calculation(self):
        mock_model = self.create_mock_model(num_layers=6, num_heads=12, hidden_size=768)
        mock_tokenizer = MagicMock()
        
        arch = analyze_model_architecture(mock_model, mock_tokenizer, "test/model")
        
        assert arch.kv_cache_metrics.bytes_per_token > 0
        assert arch.kv_cache_metrics.theoretical_max_context > 0
        assert 512 in arch.kv_cache_metrics.memory_at_seq_lengths or \
               1024 in arch.kv_cache_metrics.memory_at_seq_lengths

    def test_missing_intermediate_size_uses_default(self):
        mock_model = self.create_mock_model()
        mock_model.config.intermediate_size = None
        delattr(mock_model.config, 'intermediate_size')
        mock_model.config.n_inner = None
        mock_tokenizer = MagicMock()
        
        arch = analyze_model_architecture(mock_model, mock_tokenizer, "test/model")
        
        # Should default to hidden_size * 4
        assert arch.intermediate_size == 768 * 4

    def test_gqa_detection(self):
        mock_model = self.create_mock_model()
        mock_model.config.num_key_value_heads = 4  # Less than num_heads=12
        mock_tokenizer = MagicMock()
        
        arch = analyze_model_architecture(mock_model, mock_tokenizer, "test/gqa-model")
        
        assert arch.attention_config.is_gqa is True
        assert arch.attention_config.num_kv_heads == 4

    def test_mqa_detection(self):
        mock_model = self.create_mock_model()
        mock_model.config.num_key_value_heads = 1  # Multi-query attention
        mock_tokenizer = MagicMock()
        
        arch = analyze_model_architecture(mock_model, mock_tokenizer, "test/mqa-model")
        
        assert arch.attention_config.is_mqa is True


class TestLayerActivationStats:
    """Tests for LayerActivationStats model."""

    def test_layer_stats(self):
        stats = LayerActivationStats(
            layer_idx=0,
            attention_entropy=2.5,
            attention_sparsity=0.7,
            top_k_concentration=0.8,
            sink_attention_ratio=0.3,
            local_attention_ratio=0.6,
        )
        assert stats.layer_idx == 0
        assert stats.attention_entropy == 2.5
        assert stats.attention_sparsity == 0.7


class TestAnalyzeLayerAttention:
    """Tests for analyze_layer_attention function."""

    def test_single_layer_analysis(self):
        # Create simple attention pattern with clear sink
        attention = np.array([
            [[
                [1.0, 0.0, 0.0, 0.0],
                [0.7, 0.3, 0.0, 0.0],
                [0.5, 0.2, 0.3, 0.0],
                [0.4, 0.2, 0.2, 0.2],
            ]]
        ])  # [batch=1, heads=1, seq=4, seq=4]
        
        result = analyze_layer_attention([attention.tolist()])
        
        assert isinstance(result, PerLayerAnalysis)
        assert len(result.layer_stats) == 1
        assert result.layer_stats[0].layer_idx == 0

    def test_multiple_layers_analysis(self):
        # Create attention for 3 layers
        attention = np.random.rand(1, 4, 8, 8)
        # Apply causal mask
        for i in range(8):
            attention[:, :, i, i+1:] = 0
            attention[:, :, i, :i+1] /= attention[:, :, i, :i+1].sum() + 1e-9
        
        attention_list = [attention.tolist() for _ in range(3)]
        result = analyze_layer_attention(attention_list)
        
        assert len(result.layer_stats) == 3
        for i, stat in enumerate(result.layer_stats):
            assert stat.layer_idx == i

    def test_entropy_calculation(self):
        # Uniform attention = high entropy
        uniform_attn = np.array([
            [[
                [1.0, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0, 0.0],
                [0.33, 0.33, 0.34, 0.0],
                [0.25, 0.25, 0.25, 0.25],
            ]]
        ])
        
        # Peaked attention = low entropy  
        peaked_attn = np.array([
            [[
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [0.9, 0.05, 0.05, 0.0],
                [0.9, 0.05, 0.03, 0.02],
            ]]
        ])
        
        uniform_result = analyze_layer_attention([uniform_attn.tolist()])
        peaked_result = analyze_layer_attention([peaked_attn.tolist()])
        
        assert uniform_result.layer_stats[0].attention_entropy > \
               peaked_result.layer_stats[0].attention_entropy

    def test_sparsity_calculation(self):
        # Dense attention
        dense_attn = np.ones((1, 1, 4, 4)) * 0.1
        for i in range(4):
            dense_attn[:, :, i, i+1:] = 0
        
        # Sparse attention
        sparse_attn = np.zeros((1, 1, 4, 4))
        for i in range(4):
            sparse_attn[:, :, i, 0] = 0.9
            sparse_attn[:, :, i, i] = 0.1
        
        dense_result = analyze_layer_attention([dense_attn.tolist()])
        sparse_result = analyze_layer_attention([sparse_attn.tolist()])
        
        # Sparse attention should have higher sparsity
        assert sparse_result.layer_stats[0].attention_sparsity >= \
               dense_result.layer_stats[0].attention_sparsity

    def test_sink_detection(self):
        # Attention pattern where position 0 is a strong sink
        sink_attn = np.array([
            [[
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.8, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.7, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
                [0.6, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0],
                [0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0],
                [0.5, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.0],
                [0.5, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05],
            ]]
        ])
        
        result = analyze_layer_attention([sink_attn.tolist()], sink_threshold=0.3)
        
        assert 0 in result.sink_positions

    def test_average_metrics(self):
        attention = np.random.rand(1, 4, 8, 8)
        for i in range(8):
            attention[:, :, i, i+1:] = 0
            attention[:, :, i, :i+1] /= attention[:, :, i, :i+1].sum() + 1e-9
        
        attention_list = [attention.tolist() for _ in range(3)]
        result = analyze_layer_attention(attention_list)
        
        # Average should be calculated correctly
        expected_avg_entropy = sum(s.attention_entropy for s in result.layer_stats) / 3
        expected_avg_sparsity = sum(s.attention_sparsity for s in result.layer_stats) / 3
        
        assert abs(result.average_entropy - expected_avg_entropy) < 1e-6
        assert abs(result.average_sparsity - expected_avg_sparsity) < 1e-6

    def test_empty_attention_handling(self):
        """Test handling of minimal attention patterns."""
        small_attn = np.array([[[[1.0]]]])
        result = analyze_layer_attention([small_attn.tolist()])
        
        assert len(result.layer_stats) == 1
        assert result.layer_stats[0].layer_idx == 0


class TestPerLayerAnalysis:
    """Tests for PerLayerAnalysis model."""

    def test_per_layer_analysis(self):
        stats = [
            LayerActivationStats(
                layer_idx=i,
                attention_entropy=1.0 + i * 0.1,
                attention_sparsity=0.5,
                top_k_concentration=0.7,
                sink_attention_ratio=0.2,
                local_attention_ratio=0.6,
            )
            for i in range(3)
        ]
        
        analysis = PerLayerAnalysis(
            layer_stats=stats,
            sink_positions=[0, 1],
            heavy_hitter_positions=[5, 10],
            average_entropy=1.1,
            average_sparsity=0.5,
        )
        
        assert len(analysis.layer_stats) == 3
        assert 0 in analysis.sink_positions
        assert 5 in analysis.heavy_hitter_positions

