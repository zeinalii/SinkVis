"""FastAPI server for SinkVis."""

import asyncio
import json
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .models import (
    PromptRequest,
    SimulationConfig,
    SimulationResult,
    StreamConfig,
    AttentionFrame,
    CacheProfile,
    EvictionPolicy,
)
from .attention import (
    generate_attention_pattern,
    identify_sinks,
    identify_heavy_hitters,
    create_attention_frame,
    generate_cache_blocks,
)
from .eviction import run_simulation, tokenize_simple
from .hf_loader import (
    get_loader,
    ModelInfo,
    LoadProgress,
    LoadedModel,
    ModelStatus,
    TokenInfo,
    get_token_info,
)
from .architecture import (
    analyze_model_architecture,
    analyze_layer_attention,
    ModelArchitecture,
    PerLayerAnalysis,
)

app = FastAPI(
    title="SinkVis",
    description="Attention Sink and KV Cache Visualizer",
    version="1.0.0",
)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent


# Active streaming connections
class ConnectionManager:
    """Manage WebSocket connections for live streaming."""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.streaming_task: Optional[asyncio.Task] = None
        self.is_streaming = False
        self.stream_config = StreamConfig()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()


# Demo state for live streaming
class DemoState:
    """State for the demo attention stream."""
    
    DEFAULT_TEXT = (
        "The transformer architecture revolutionized natural language processing. "
        "Attention mechanisms allow models to focus on relevant parts of the input. "
        "Key-value caches store intermediate computations for efficient generation. "
        "Attention sinks are tokens that receive disproportionate attention scores. "
        "Heavy hitters are semantically important tokens in the sequence."
    )
    
    def __init__(self):
        self.reset()
    
    def reset(self, prompt: Optional[str] = None):
        self.tokens = ["<s>", "<bos>"]
        self.current_position = 2
        self.demo_text = prompt if prompt else self.DEFAULT_TEXT
        self.demo_tokens = tokenize_simple(self.demo_text)


demo_state = DemoState()


@app.get("/")
async def root():
    """Serve the main visualization page."""
    return FileResponse(PROJECT_ROOT / "frontend" / "index.html")


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/api/simulate", response_model=SimulationResult)
async def simulate_eviction(request: PromptRequest):
    """
    Run an eviction simulation on the provided prompt.
    
    Returns detailed metrics and attention frames for visualization.
    """
    result = run_simulation(
        prompt=request.prompt,
        config=request.config,
        generate_tokens=request.generate_tokens,
    )
    return result


@app.post("/api/compare")
async def compare_policies(request: PromptRequest):
    """
    Run simulation with all eviction policies for comparison.
    
    Returns results for each policy side by side.
    """
    policies = [
        EvictionPolicy.FULL,
        EvictionPolicy.LRU,
        EvictionPolicy.SLIDING_WINDOW,
        EvictionPolicy.STREAMING_LLM,
        EvictionPolicy.H2O,
    ]
    
    results = {}
    for policy in policies:
        config = SimulationConfig(
            policy=policy,
            cache_size=request.config.cache_size,
            sink_count=request.config.sink_count,
            window_size=request.config.window_size,
            heavy_hitter_ratio=request.config.heavy_hitter_ratio,
        )
        result = run_simulation(
            prompt=request.prompt,
            config=config,
            generate_tokens=request.generate_tokens,
        )
        results[policy.value] = result.model_dump()
    
    return results


@app.get("/api/cache-profile")
async def get_cache_profile(
    seq_len: int = 512,
    policy: EvictionPolicy = EvictionPolicy.STREAMING_LLM,
):
    """Get a sample cache profile for visualization."""
    attention = generate_attention_pattern(seq_len)
    sinks = identify_sinks(attention)
    heavy_hitters = identify_heavy_hitters(attention, exclude_sinks=sinks)
    
    blocks = generate_cache_blocks(
        seq_len=seq_len,
        sink_indices=sinks,
        heavy_hitter_indices=heavy_hitters,
        base_timestamp=time.time(),
    )
    
    from .models import MemoryTier
    memory_usage = {tier.value: 0 for tier in MemoryTier}
    for block in blocks:
        memory_usage[block.memory_tier.value] += block.size_bytes
    
    return CacheProfile(
        total_tokens=seq_len,
        blocks=blocks,
        memory_usage=memory_usage,
        eviction_policy=policy,
        timestamp=time.time(),
    )


# ============================================
# Model Hub Endpoints
# ============================================

@app.get("/api/models/token-info")
async def get_hf_token_info() -> TokenInfo:
    """Get information about available HuggingFace token."""
    return get_token_info()


@app.get("/api/models/search")
async def search_models(
    query: str,
    limit: int = 20,
    filter_text_generation: bool = True,
) -> list[ModelInfo]:
    """Search for models on Hugging Face Hub."""
    loader = get_loader()
    try:
        results = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: loader.search_models(query, limit, filter_text_generation),
        )
        return results
    except Exception as e:
        raise ValueError(f"Search failed: {e}")


@app.get("/api/models/info/{model_id:path}")
async def get_model_info(model_id: str) -> ModelInfo:
    """Get information about a specific model."""
    loader = get_loader()
    try:
        info = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: loader.get_model_info(model_id),
        )
        return info
    except Exception as e:
        raise ValueError(f"Failed to get model info: {e}")


from pydantic import BaseModel as PydanticBaseModel


class LoadModelRequest(PydanticBaseModel):
    """Request body for loading a model."""
    model_id: str
    device: str = "auto"
    dtype: str = "auto"
    trust_remote_code: bool = False
    token: Optional[str] = None


@app.post("/api/models/load")
async def load_model(request: LoadModelRequest) -> LoadedModel:
    """Load a model from Hugging Face Hub."""
    loader = get_loader()
    
    # Unload existing model first
    if loader.is_model_loaded():
        loader.unload_model()
    
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: loader.load_model(
                model_id=request.model_id,
                device=request.device,
                dtype=request.dtype,
                trust_remote_code=request.trust_remote_code,
                token=request.token,
            ),
        )
        return result
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}")


@app.post("/api/models/unload")
async def unload_model():
    """Unload the current model."""
    loader = get_loader()
    loader.unload_model()
    return {"status": "ok", "message": "Model unloaded"}


@app.get("/api/models/status")
async def get_model_status() -> LoadProgress:
    """Get the current model loading status."""
    loader = get_loader()
    return loader.get_status()


class AttentionRequest(PydanticBaseModel):
    """Request body for getting attention weights."""
    text: str
    layer: int = -1
    head: Optional[int] = None


@app.post("/api/models/attention")
async def get_model_attention(request: AttentionRequest):
    """Get attention weights from the loaded model."""
    loader = get_loader()
    
    if not loader.is_model_loaded():
        raise ValueError("No model loaded")
    
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: loader.get_attention_weights(
                text=request.text,
                layer=request.layer,
                head=request.head,
            ),
        )
        return result
    except Exception as e:
        raise ValueError(f"Failed to get attention: {e}")


@app.get("/api/models/architecture")
async def get_model_architecture() -> ModelArchitecture:
    """Get detailed architecture information about the loaded model."""
    loader = get_loader()
    
    if not loader.is_model_loaded():
        raise ValueError("No model loaded. Please load a model first.")
    
    if loader.model is None or loader.tokenizer is None or loader.model_id is None:
        raise ValueError("Model not properly loaded. Please reload the model.")
    
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: analyze_model_architecture(
                model=loader.model,
                tokenizer=loader.tokenizer,
                model_id=loader.model_id,
            ),
        )
        return result
    except Exception as e:
        raise ValueError(f"Failed to analyze architecture: {e}")


class LayerAnalysisRequest(PydanticBaseModel):
    """Request for per-layer attention analysis."""
    text: str
    sink_threshold: float = 0.1
    heavy_hitter_threshold: float = 0.05


@app.post("/api/models/layer-analysis")
async def get_layer_analysis(request: LayerAnalysisRequest) -> PerLayerAnalysis:
    """Analyze attention patterns across all layers."""
    loader = get_loader()
    
    if not loader.is_model_loaded():
        raise ValueError("No model loaded. Please load a model first.")
    
    if loader.model is None or loader.tokenizer is None:
        raise ValueError("Model not properly loaded. Please reload the model.")
    
    try:
        # Get attention from all layers
        def run_analysis():
            import torch
            
            if loader.model is None or loader.tokenizer is None:
                raise ValueError("Model was unloaded during processing")
            
            inputs = loader.tokenizer(request.text, return_tensors="pt")
            device = next(loader.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = loader.model(
                    **inputs,
                    output_attentions=True,
                    return_dict=True,
                )
            
            if outputs.attentions is None or len(outputs.attentions) == 0:
                raise ValueError("Model did not return attention weights")
            
            # Convert attention to list format
            attentions = [attn.cpu().numpy().tolist() for attn in outputs.attentions]
            
            return analyze_layer_attention(
                attention_weights=attentions,
                sink_threshold=request.sink_threshold,
                heavy_hitter_threshold=request.heavy_hitter_threshold,
            )
        
        result = await asyncio.get_event_loop().run_in_executor(None, run_analysis)
        return result
    except Exception as e:
        raise ValueError(f"Failed to analyze layers: {e}")


class AllLayersAttentionRequest(PydanticBaseModel):
    """Request for attention from all layers."""
    text: str
    head: int | None = None


@app.post("/api/models/all-layers-attention")
async def get_all_layers_attention(request: AllLayersAttentionRequest):
    """Get attention weights from all layers for visualization."""
    loader = get_loader()
    
    if not loader.is_model_loaded():
        raise ValueError("No model loaded. Please load a model first.")
    
    if loader.model is None or loader.tokenizer is None:
        raise ValueError("Model or tokenizer not available. Please reload the model.")
    
    try:
        def run_attention():
            import torch
            
            # Double-check model is still loaded
            if loader.model is None or loader.tokenizer is None:
                raise ValueError("Model was unloaded during processing")
            
            inputs = loader.tokenizer(request.text, return_tensors="pt")
            device = next(loader.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Some models use SDPA or flash attention which don't return attention weights
            # Try to temporarily disable it if possible
            original_attn_impl = None
            if hasattr(loader.model.config, '_attn_implementation'):
                original_attn_impl = loader.model.config._attn_implementation
                loader.model.config._attn_implementation = "eager"
            
            try:
                with torch.no_grad():
                    outputs = loader.model(
                        **inputs,
                        output_attentions=True,
                        return_dict=True,
                    )
            finally:
                # Restore original attention implementation
                if original_attn_impl is not None:
                    loader.model.config._attn_implementation = original_attn_impl
            
            tokens = loader.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            # Check if attentions are available
            if outputs.attentions is None or len(outputs.attentions) == 0:
                raise ValueError("Model did not return attention weights. This model may use an attention implementation (like Flash Attention) that doesn't support attention output.")
            
            # Filter out None attentions - also check for tensor validity
            valid_attentions = []
            for a in outputs.attentions:
                if a is not None and hasattr(a, 'shape'):
                    valid_attentions.append(a)
            
            if len(valid_attentions) == 0:
                raise ValueError("Model returned empty attention weights. Try reloading the model or using a different model.")
            
            # Process attention for each layer
            layer_attentions = []
            for layer_idx, attn in enumerate(outputs.attentions):
                # Skip if attention is None for this layer
                if attn is None:
                    continue
                    
                # attn is [batch, heads, seq, seq]
                if request.head is not None:
                    layer_attn = attn[0, request.head].cpu().numpy().tolist()
                else:
                    # Average across heads
                    layer_attn = attn[0].mean(dim=0).cpu().numpy().tolist()
                
                layer_attentions.append({
                    "layer": layer_idx,
                    "attention": layer_attn,
                })
            
            # Get num_heads from first valid attention
            num_heads = valid_attentions[0].shape[1] if valid_attentions else 0
            
            return {
                "tokens": tokens,
                "seq_len": len(tokens),
                "num_layers": len(layer_attentions),
                "num_heads": num_heads,
                "layers": layer_attentions,
            }
        
        result = await asyncio.get_event_loop().run_in_executor(None, run_attention)
        return result
    except Exception as e:
        raise ValueError(f"Failed to get attention: {e}")


class GenerateRequest(PydanticBaseModel):
    """Request for text generation."""
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True


@app.post("/api/models/generate")
async def generate_text(request: GenerateRequest):
    """Generate text using the loaded model."""
    loader = get_loader()
    
    if not loader.is_model_loaded():
        raise ValueError("No model loaded. Please load a model first.")
    
    if loader.model is None or loader.tokenizer is None:
        raise ValueError("Model not properly loaded. Please reload the model.")
    
    try:
        def run_generation():
            import torch
            
            if loader.model is None or loader.tokenizer is None:
                raise ValueError("Model was unloaded during processing")
            
            inputs = loader.tokenizer(request.prompt, return_tensors="pt")
            device = next(loader.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Set pad token if not set
            if loader.tokenizer.pad_token_id is None:
                loader.tokenizer.pad_token_id = loader.tokenizer.eos_token_id
            
            with torch.no_grad():
                outputs = loader.model.generate(
                    **inputs,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature if request.do_sample else 1.0,
                    top_p=request.top_p if request.do_sample else 1.0,
                    top_k=request.top_k if request.do_sample else 0,
                    do_sample=request.do_sample,
                    pad_token_id=loader.tokenizer.pad_token_id,
                )
            
            generated_text = loader.tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_text = generated_text[len(request.prompt):]
            
            return {
                "prompt": request.prompt,
                "generated": new_text,
                "full_text": generated_text,
                "tokens_generated": len(outputs[0]) - len(inputs["input_ids"][0]),
            }
        
        result = await asyncio.get_event_loop().run_in_executor(None, run_generation)
        return result
    except Exception as e:
        raise ValueError(f"Failed to generate text: {e}")


@app.websocket("/ws/attention")
async def attention_stream(websocket: WebSocket):
    """
    WebSocket endpoint for live attention streaming.
    
    Streams attention frames as they're generated.
    """
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive configuration updates
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "config":
                manager.stream_config = StreamConfig(**message.get("config", {}))
            
            elif message.get("type") == "start":
                prompt = message.get("prompt")
                demo_state.reset(prompt)
                manager.is_streaming = True
                
                # Start streaming in background
                asyncio.create_task(stream_attention_frames(websocket))
            
            elif message.get("type") == "stop":
                manager.is_streaming = False
            
            elif message.get("type") == "reset":
                prompt = message.get("prompt")
                demo_state.reset(prompt)
                manager.is_streaming = False
                await websocket.send_json({"type": "reset", "status": "ok"})
            
            elif message.get("type") == "step":
                # Single step forward
                frame = generate_next_frame()
                if frame:
                    await websocket.send_json({
                        "type": "frame",
                        "data": frame.model_dump(),
                    })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)


async def stream_attention_frames(websocket: WebSocket):
    """Stream attention frames to the connected client."""
    while manager.is_streaming and demo_state.current_position < len(demo_state.demo_tokens):
        frame = generate_next_frame()
        if frame:
            try:
                await websocket.send_json({
                    "type": "frame",
                    "data": frame.model_dump(),
                })
            except Exception:
                break
        
        await asyncio.sleep(manager.stream_config.update_interval_ms / 1000)
    
    if manager.is_streaming:
        try:
            await websocket.send_json({"type": "complete"})
        except Exception:
            pass
    
    manager.is_streaming = False


def generate_next_frame() -> Optional[AttentionFrame]:
    """Generate the next attention frame in the demo sequence."""
    if demo_state.current_position >= len(demo_state.demo_tokens):
        return None
    
    # Add next token
    demo_state.tokens.append(demo_state.demo_tokens[demo_state.current_position])
    demo_state.current_position += 1
    
    seq_len = len(demo_state.tokens)
    attention = generate_attention_pattern(seq_len)
    
    frame = create_attention_frame(
        attention=attention,
        layer=0,
        head=0,
        tokens=demo_state.tokens,
        timestamp=time.time(),
        sink_threshold=manager.stream_config.sink_threshold,
        heavy_hitter_threshold=manager.stream_config.heavy_hitter_threshold,
    )
    
    return frame


# Mount static files after routes
@app.on_event("startup")
async def startup():
    """Mount static files on startup."""
    frontend_path = PROJECT_ROOT / "frontend"
    if frontend_path.exists():
        app.mount("/static", StaticFiles(directory=frontend_path), name="static")


def main():
    """Run the server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)


if __name__ == "__main__":
    main()

