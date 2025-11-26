"""Hugging Face model loader for SinkVis."""

import os
import io
import base64
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, Union
import threading

import torch
from huggingface_hub import HfApi, model_info
from pydantic import BaseModel, Field


class ModelType(str, Enum):
    """Type of model loaded."""
    
    CAUSAL_LM = "causal_lm"
    DIFFUSION = "diffusion"
    UNKNOWN = "unknown"


def detect_model_type(model_id: str) -> ModelType:
    """Detect the type of model from its ID or info."""
    try:
        info = model_info(model_id)
        pipeline_tag = info.pipeline_tag or ""
        tags = info.tags or []
        library = info.library_name or ""
        
        # Check for diffusion models
        diffusion_indicators = [
            "text-to-image",
            "image-to-image", 
            "diffusers",
            "stable-diffusion",
            "sdxl",
        ]
        
        for indicator in diffusion_indicators:
            if indicator in pipeline_tag.lower():
                return ModelType.DIFFUSION
            if indicator in library.lower():
                return ModelType.DIFFUSION
            if any(indicator in tag.lower() for tag in tags):
                return ModelType.DIFFUSION
        
        # Check for causal LM
        if pipeline_tag in ["text-generation", "text2text-generation"]:
            return ModelType.CAUSAL_LM
        if library in ["transformers"]:
            return ModelType.CAUSAL_LM
        
        return ModelType.UNKNOWN
    except Exception:
        return ModelType.UNKNOWN


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
    model_type: ModelType = ModelType.CAUSAL_LM
    num_layers: int = 0
    num_heads: int = 0
    hidden_size: int = 0
    vocab_size: int = 0
    dtype: str = "unknown"
    device: str = "cpu"
    memory_mb: float = 0.0
    # Diffusion-specific fields
    image_size: Optional[int] = None
    num_inference_steps: int = 20


@dataclass
class HFModelLoader:
    """Handles loading models from Hugging Face Hub."""
    
    model: Optional[object] = None
    tokenizer: Optional[object] = None
    pipeline: Optional[object] = None  # For diffusion models
    model_id: Optional[str] = None
    model_type: ModelType = ModelType.UNKNOWN
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
            f"Detecting model type for {model_id}...",
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
        
        # Detect model type
        self.model_type = detect_model_type(model_id)
        
        if self.model_type == ModelType.DIFFUSION:
            return self._load_diffusion_model(model_id, device, dtype, token)
        else:
            return self._load_causal_lm(model_id, device, dtype, trust_remote_code, token)
    
    def _load_diffusion_model(
        self,
        model_id: str,
        device: str,
        dtype: str,
        token: Optional[str],
    ) -> LoadedModel:
        """Load a diffusion model (Stable Diffusion, etc.)."""
        try:
            from diffusers import DiffusionPipeline, AutoPipelineForText2Image
            
            self._update_progress(
                ModelStatus.DOWNLOADING,
                0.2,
                "Loading diffusion pipeline...",
            )
            
            # Determine device
            if device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            
            # Determine dtype
            torch_dtype = torch.float16 if device != "cpu" else torch.float32
            if dtype == "float32":
                torch_dtype = torch.float32
            elif dtype == "bfloat16":
                torch_dtype = torch.bfloat16
            
            self._update_progress(
                ModelStatus.LOADING,
                0.5,
                f"Loading pipeline to {device}...",
            )
            
            # Try AutoPipelineForText2Image first, then fallback to DiffusionPipeline
            try:
                self.pipeline = AutoPipelineForText2Image.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                    token=token,
                    safety_checker=None,  # Disable safety checker for faster loading
                )
            except Exception:
                self.pipeline = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                    token=token,
                )
            
            self.pipeline = self.pipeline.to(device)
            
            # Try to enable memory optimizations
            if hasattr(self.pipeline, 'enable_attention_slicing'):
                self.pipeline.enable_attention_slicing()
            
            # Get image size from config
            image_size = 512
            if hasattr(self.pipeline, 'unet') and hasattr(self.pipeline.unet, 'config'):
                unet_config = self.pipeline.unet.config
                image_size = getattr(unet_config, 'sample_size', 64) * 8
            
            # Calculate memory (rough estimate)
            memory_mb = 0.0
            if hasattr(self.pipeline, 'unet'):
                memory_mb += sum(p.numel() * p.element_size() for p in self.pipeline.unet.parameters()) / (1024 * 1024)
            if hasattr(self.pipeline, 'vae'):
                memory_mb += sum(p.numel() * p.element_size() for p in self.pipeline.vae.parameters()) / (1024 * 1024)
            if hasattr(self.pipeline, 'text_encoder'):
                memory_mb += sum(p.numel() * p.element_size() for p in self.pipeline.text_encoder.parameters()) / (1024 * 1024)
            
            self._update_progress(
                ModelStatus.READY,
                1.0,
                f"Diffusion model loaded successfully",
            )
            
            return LoadedModel(
                model_id=model_id,
                model_type=ModelType.DIFFUSION,
                dtype=str(torch_dtype).split('.')[-1],
                device=device,
                memory_mb=round(memory_mb, 2),
                image_size=image_size,
            )
            
        except Exception as e:
            self._update_progress(
                ModelStatus.ERROR,
                0.0,
                "",
                str(e),
            )
            self.pipeline = None
            raise
    
    def _load_causal_lm(
        self,
        model_id: str,
        device: str,
        dtype: str,
        trust_remote_code: bool,
        token: Optional[str],
    ) -> LoadedModel:
        """Load a causal language model."""
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
            self.model_type = ModelType.CAUSAL_LM
            
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
                model_type=ModelType.CAUSAL_LM,
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
            if self.pipeline is not None:
                del self.pipeline
                self.pipeline = None
            self.model_id = None
            self.model_type = ModelType.UNKNOWN
            
            # Force garbage collection
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self._update_progress(ModelStatus.IDLE, 0.0, "Model unloaded")
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None,
    ) -> str:
        """
        Generate an image using the loaded diffusion model.
        
        Args:
            prompt: Text prompt for image generation
            negative_prompt: Negative prompt to avoid certain features
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            width: Output image width
            height: Output image height
            seed: Random seed for reproducibility
        
        Returns:
            Base64-encoded PNG image
        """
        if self.pipeline is None:
            raise ValueError("No diffusion model loaded")
        
        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.pipeline.device).manual_seed(seed)
        
        # Generate image
        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        )
        
        image = result.images[0]
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return image_base64
    
    def get_cross_attention(
        self,
        prompt: str,
        num_inference_steps: int = 10,
    ) -> dict:
        """
        Get cross-attention maps from the diffusion model.
        
        This is useful for visualizing which parts of the prompt
        influence which parts of the generated image.
        
        Args:
            prompt: Text prompt
            num_inference_steps: Number of steps (fewer = faster)
        
        Returns:
            Dictionary with cross-attention data
        """
        if self.pipeline is None:
            raise ValueError("No diffusion model loaded")
        
        # This is a simplified version - full implementation would require
        # hooking into the UNet's cross-attention layers
        # For now, return basic info
        return {
            "prompt": prompt,
            "model_id": self.model_id,
            "note": "Cross-attention visualization requires model-specific hooks",
        }
    
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
        if self.model_type == ModelType.DIFFUSION:
            return self.pipeline is not None
        return self.model is not None and self.tokenizer is not None
    
    def is_diffusion_model(self) -> bool:
        """Check if the loaded model is a diffusion model."""
        return self.model_type == ModelType.DIFFUSION and self.pipeline is not None


# Global loader instance
_loader: Optional[HFModelLoader] = None


def get_loader() -> HFModelLoader:
    """Get or create the global model loader instance."""
    global _loader
    if _loader is None:
        _loader = HFModelLoader()
    return _loader

