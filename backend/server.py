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
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.tokens = ["<s>", "<bos>"]
        self.current_position = 2
        self.demo_text = (
            "The transformer architecture revolutionized natural language processing. "
            "Attention mechanisms allow models to focus on relevant parts of the input. "
            "Key-value caches store intermediate computations for efficient generation. "
            "Attention sinks are tokens that receive disproportionate attention scores. "
            "Heavy hitters are semantically important tokens in the sequence."
        )
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
                demo_state.reset()
                manager.is_streaming = True
                
                # Start streaming in background
                asyncio.create_task(stream_attention_frames(websocket))
            
            elif message.get("type") == "stop":
                manager.is_streaming = False
            
            elif message.get("type") == "reset":
                demo_state.reset()
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

