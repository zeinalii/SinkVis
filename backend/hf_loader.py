"""Hugging Face model loader for SinkVis."""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Callable
import threading

import torch
from huggingface_hub import HfApi, model_info
from pydantic import BaseModel, Field


class TokenSource(str, Enum):
    """Source of the HuggingFace token."""
    
    NONE = "none"
    HF_CACHE = "hf_cache"
    ENV_FILE = "env_file"
    ENV_VAR = "env_var"
    MANUAL = "manual"


class TokenInfo(BaseModel):
    """Information about the HuggingFace token."""
    
    available: bool = False
    source: TokenSource = TokenSource.NONE
    source_path: Optional[str] = None
    masked_token: Optional[str] = None


def get_hf_token() -> tuple[Optional[str], TokenSource, Optional[str]]:
    """
    Get HuggingFace token from available sources.
    
    Returns:
        Tuple of (token, source, source_path)
    
    Priority:
    1. HuggingFace cache (~/.cache/huggingface/token or HF_HOME)
    2. Environment variable HF_TOKEN or HUGGING_FACE_HUB_TOKEN
    3. .env file in project root
    """
    # 1. Check HuggingFace cache (from `huggingface-cli login`)
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    token_path = Path(hf_home) / "token"
    
    if token_path.exists():
        try:
            token = token_path.read_text().strip()
            if token:
                return token, TokenSource.HF_CACHE, str(token_path)
        except Exception:
            pass
    
    # Also check the older location
    old_token_path = Path.home() / ".huggingface" / "token"
    if old_token_path.exists():
        try:
            token = old_token_path.read_text().strip()
            if token:
                return token, TokenSource.HF_CACHE, str(old_token_path)
        except Exception:
            pass
    
    # 2. Check environment variables
    for env_var in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]:
        token = os.environ.get(env_var)
        if token:
            return token, TokenSource.ENV_VAR, env_var
    
    # 3. Check .env file in project root
    try:
        from dotenv import dotenv_values
        
        # Try multiple possible .env locations
        possible_paths = [
            Path(__file__).parent.parent / ".env",  # Project root
            Path.cwd() / ".env",  # Current working directory
        ]
        
        for env_path in possible_paths:
            if env_path.exists():
                env_values = dotenv_values(env_path)
                for key in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]:
                    if key in env_values and env_values[key]:
                        return env_values[key], TokenSource.ENV_FILE, str(env_path)
    except ImportError:
        pass
    
    return None, TokenSource.NONE, None


def get_token_info() -> TokenInfo:
    """Get information about available HuggingFace token."""
    token, source, source_path = get_hf_token()
    
    if token:
        # Mask token for display (show first 4 and last 4 chars)
        if len(token) > 10:
            masked = f"{token[:4]}...{token[-4:]}"
        else:
            masked = "****"
        
        return TokenInfo(
            available=True,
            source=source,
            source_path=source_path,
            masked_token=masked,
        )
    
    return TokenInfo(
        available=False,
        source=TokenSource.NONE,
        source_path=None,
        masked_token=None,
    )


class ModelStatus(str, Enum):
    """Status of a model loading operation."""
    
    IDLE = "idle"
    SEARCHING = "searching"
    DOWNLOADING = "downloading"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


class ModelInfo(BaseModel):
    """Information about a Hugging Face model."""
    
    model_id: str
    author: Optional[str] = None
    downloads: int = 0
    likes: int = 0
    pipeline_tag: Optional[str] = None
    library_name: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    is_gated: bool = False


class LoadProgress(BaseModel):
    """Progress of model loading operation."""
    
    status: ModelStatus
    progress: float = 0.0
    message: str = ""
    model_id: Optional[str] = None
    error: Optional[str] = None


class LoadedModel(BaseModel):
    """Information about a loaded model."""
    
    model_id: str
    num_layers: int
    num_heads: int
    hidden_size: int
    vocab_size: int
    dtype: str
    device: str
    memory_mb: float


@dataclass
class HFModelLoader:
    """Handles loading models from Hugging Face Hub."""
    
    model: Optional[object] = None
    tokenizer: Optional[object] = None
    model_id: Optional[str] = None
    status: ModelStatus = ModelStatus.IDLE
    progress: float = 0.0
    message: str = ""
    error: Optional[str] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _progress_callbacks: list[Callable] = field(default_factory=list)
    
    def add_progress_callback(self, callback: Callable[[LoadProgress], None]):
        """Register a callback for progress updates."""
        self._progress_callbacks.append(callback)
    
    def remove_progress_callback(self, callback: Callable):
        """Remove a progress callback."""
        if callback in self._progress_callbacks:
            self._progress_callbacks.remove(callback)
    
    def _notify_progress(self):
        """Notify all callbacks of progress update."""
        progress = LoadProgress(
            status=self.status,
            progress=self.progress,
            message=self.message,
            model_id=self.model_id,
            error=self.error,
        )
        for callback in self._progress_callbacks:
            try:
                callback(progress)
            except Exception:
                pass
    
    def _update_progress(
        self,
        status: ModelStatus,
        progress: float = 0.0,
        message: str = "",
        error: Optional[str] = None,
    ):
        """Update and broadcast progress."""
        with self._lock:
            self.status = status
            self.progress = progress
            self.message = message
            self.error = error
            self._notify_progress()
    
    def search_models(
        self,
        query: str,
        limit: int = 20,
        filter_text_generation: bool = True,
    ) -> list[ModelInfo]:
        """
        Search for models on Hugging Face Hub.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            filter_text_generation: Only return text generation models
        
        Returns:
            List of matching models
        """
        self._update_progress(ModelStatus.SEARCHING, 0.0, f"Searching for '{query}'...")
        
        try:
            api = HfApi()
            
            filter_str = None
            if filter_text_generation:
                filter_str = "text-generation"
            
            models = api.list_models(
                search=query,
                limit=limit,
                sort="downloads",
                direction=-1,
                filter=filter_str,
            )
            
            results = []
            for m in models:
                results.append(ModelInfo(
                    model_id=m.id,
                    author=m.author,
                    downloads=m.downloads or 0,
                    likes=m.likes or 0,
                    pipeline_tag=m.pipeline_tag,
                    library_name=m.library_name,
                    tags=list(m.tags) if m.tags else [],
                    is_gated=m.gated if hasattr(m, 'gated') and m.gated else False,
                ))
            
            self._update_progress(
                ModelStatus.IDLE,
                1.0,
                f"Found {len(results)} models",
            )
            return results
            
        except Exception as e:
            self._update_progress(ModelStatus.ERROR, 0.0, "", str(e))
            raise
    
    def get_model_info(self, model_id: str) -> ModelInfo:
        """Get detailed information about a specific model."""
        try:
            info = model_info(model_id)
            return ModelInfo(
                model_id=info.id,
                author=info.author,
                downloads=info.downloads or 0,
                likes=info.likes or 0,
                pipeline_tag=info.pipeline_tag,
                library_name=info.library_name,
                tags=list(info.tags) if info.tags else [],
                is_gated=info.gated if hasattr(info, 'gated') and info.gated else False,
            )
        except Exception as e:
            raise ValueError(f"Failed to get model info: {e}")
    
    def load_model(
        self,
        model_id: str,
        device: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = False,
        token: Optional[str] = None,
    ) -> LoadedModel:
        """
        Load a model from Hugging Face Hub.
        
        Args:
            model_id: HuggingFace model ID (e.g., 'meta-llama/Llama-2-7b-hf')
            device: Device to load model on ('auto', 'cpu', 'cuda', 'mps')
            dtype: Data type ('auto', 'float16', 'float32', 'bfloat16')
            trust_remote_code: Whether to trust remote code in the model
            token: HuggingFace API token for gated models (if None, auto-detect)
        
        Returns:
            LoadedModel with model configuration info
        """
        self._update_progress(
            ModelStatus.DOWNLOADING,
            0.1,
            f"Downloading {model_id}...",
        )
        self.model_id = model_id
        
        # Auto-detect token if not provided
        if token is None:
            auto_token, source, _ = get_hf_token()
            if auto_token:
                token = auto_token
                self._update_progress(
                    ModelStatus.DOWNLOADING,
                    0.1,
                    f"Using token from {source.value}...",
                )
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
            
            # Determine device
            if device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            
            # Determine dtype
            torch_dtype = None
            if dtype == "auto":
                if device == "cpu":
                    torch_dtype = torch.float32
                else:
                    torch_dtype = torch.float16
            elif dtype == "float16":
                torch_dtype = torch.float16
            elif dtype == "float32":
                torch_dtype = torch.float32
            elif dtype == "bfloat16":
                torch_dtype = torch.bfloat16
            
            # Load config first to get model info
            self._update_progress(
                ModelStatus.DOWNLOADING,
                0.2,
                "Loading model configuration...",
            )
            config = AutoConfig.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
                token=token,
            )
            
            # Load tokenizer
            self._update_progress(
                ModelStatus.DOWNLOADING,
                0.3,
                "Loading tokenizer...",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
                token=token,
            )
            
            # Load model
            self._update_progress(
                ModelStatus.LOADING,
                0.5,
                f"Loading model weights to {device}...",
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map=device if device != "cpu" else None,
                trust_remote_code=trust_remote_code,
                token=token,
                low_cpu_mem_usage=True,
                attn_implementation="eager",  # Required for attention weight extraction
            )
            
            if device == "cpu":
                self.model = self.model.to(device)
            
            self.model.eval()
            
            # Calculate memory usage
            param_bytes = sum(
                p.numel() * p.element_size() for p in self.model.parameters()
            )
            memory_mb = param_bytes / (1024 * 1024)
            
            # Extract model info
            num_layers = getattr(config, 'num_hidden_layers', 
                         getattr(config, 'n_layer', 
                         getattr(config, 'num_layers', 0)))
            num_heads = getattr(config, 'num_attention_heads',
                        getattr(config, 'n_head',
                        getattr(config, 'num_heads', 0)))
            hidden_size = getattr(config, 'hidden_size',
                          getattr(config, 'n_embd', 0))
            vocab_size = getattr(config, 'vocab_size', 0)
            
            self._update_progress(
                ModelStatus.READY,
                1.0,
                f"Model loaded successfully",
            )
            
            return LoadedModel(
                model_id=model_id,
                num_layers=num_layers,
                num_heads=num_heads,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                dtype=str(torch_dtype).split('.')[-1] if torch_dtype else "unknown",
                device=device,
                memory_mb=round(memory_mb, 2),
            )
            
        except Exception as e:
            self._update_progress(
                ModelStatus.ERROR,
                0.0,
                "",
                str(e),
            )
            self.model = None
            self.tokenizer = None
            raise
    
    def unload_model(self):
        """Unload the current model and free memory."""
        with self._lock:
            if self.model is not None:
                del self.model
                self.model = None
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            self.model_id = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self._update_progress(ModelStatus.IDLE, 0.0, "Model unloaded")
    
    def get_attention_weights(
        self,
        text: str,
        layer: int = -1,
        head: Optional[int] = None,
    ) -> dict:
        """
        Get attention weights from the loaded model for given text.
        
        Args:
            text: Input text to process
            layer: Layer index (-1 for last layer)
            head: Head index (None for all heads)
        
        Returns:
            Dictionary with attention weights and token information
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("No model loaded")
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Get attention weights
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
                return_dict=True,
            )
        
        attentions = outputs.attentions
        
        # Select layer
        if layer < 0:
            layer = len(attentions) + layer
        layer_attention = attentions[layer]  # [batch, heads, seq, seq]
        
        # Select head if specified
        if head is not None:
            attention_weights = layer_attention[0, head].cpu().numpy().tolist()
        else:
            attention_weights = layer_attention[0].cpu().numpy().tolist()
        
        # Get token labels
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        return {
            "attention_weights": attention_weights,
            "tokens": tokens,
            "layer": layer,
            "head": head,
            "seq_len": len(tokens),
        }
    
    def get_status(self) -> LoadProgress:
        """Get current loading status."""
        return LoadProgress(
            status=self.status,
            progress=self.progress,
            message=self.message,
            model_id=self.model_id,
            error=self.error,
        )
    
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.model is not None and self.tokenizer is not None


# Global loader instance
_loader: Optional[HFModelLoader] = None


def get_loader() -> HFModelLoader:
    """Get or create the global model loader instance."""
    global _loader
    if _loader is None:
        _loader = HFModelLoader()
    return _loader

