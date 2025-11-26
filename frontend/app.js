/**
 * SinkVis - Attention Sink & KV Cache Visualizer
 * Frontend Application
 */

// ============================================
// Constants & Configuration
// ============================================

const API_BASE = '';
const WS_URL = `ws://${window.location.host}/ws/attention`;

const HEATMAP_CONFIG = {
    cellSize: 4,
    maxSize: 600,
    colorScale: [
        { stop: 0.0, color: [26, 26, 46] },
        { stop: 0.2, color: [74, 25, 66] },
        { stop: 0.5, color: [255, 51, 102] },
        { stop: 0.8, color: [255, 180, 100] },
        { stop: 1.0, color: [255, 204, 0] }
    ]
};

// ============================================
// State Management
// ============================================

const state = {
    currentView: 'stream',
    ws: null,
    isConnected: false,
    isStreaming: false,
    currentFrame: null,
    streamConfig: {
        update_interval_ms: 200,
        sink_threshold: 0.1,
        heavy_hitter_threshold: 0.05
    }
};

// ============================================
// Initialization
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initStreamView();
    initSimulateView();
    initProfileView();
    connectWebSocket();
});

// ============================================
// Navigation
// ============================================

function initNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const view = btn.dataset.view;
            switchView(view);
        });
    });
}

function switchView(viewName) {
    // Update nav buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.view === viewName);
    });
    
    // Update views
    document.querySelectorAll('.view').forEach(view => {
        view.classList.toggle('active', view.id === `${viewName}-view`);
    });
    
    state.currentView = viewName;
}

// ============================================
// WebSocket Connection
// ============================================

function connectWebSocket() {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        return;
    }
    
    state.ws = new WebSocket(WS_URL);
    
    state.ws.onopen = () => {
        state.isConnected = true;
        updateConnectionStatus(true);
        console.log('WebSocket connected');
    };
    
    state.ws.onclose = () => {
        state.isConnected = false;
        state.isStreaming = false;
        updateConnectionStatus(false);
        updateStreamButtons();
        
        // Reconnect after delay
        setTimeout(connectWebSocket, 3000);
    };
    
    state.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
    
    state.ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        handleWebSocketMessage(message);
    };
}

function handleWebSocketMessage(message) {
    switch (message.type) {
        case 'frame':
            state.currentFrame = message.data;
            renderAttentionFrame(message.data);
            break;
        case 'complete':
            state.isStreaming = false;
            updateStreamButtons();
            break;
        case 'reset':
            clearHeatmap();
            break;
    }
}

function updateConnectionStatus(connected) {
    const statusEl = document.getElementById('connection-status');
    const textEl = statusEl.querySelector('.status-text');
    
    statusEl.classList.toggle('connected', connected);
    textEl.textContent = connected ? 'Connected' : 'Disconnected';
}

function sendWSMessage(message) {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify(message));
    }
}

// ============================================
// Stream View
// ============================================

function initStreamView() {
    // Control buttons
    document.getElementById('stream-start').addEventListener('click', startStreaming);
    document.getElementById('stream-stop').addEventListener('click', stopStreaming);
    document.getElementById('stream-step').addEventListener('click', stepStreaming);
    document.getElementById('stream-reset').addEventListener('click', resetStreaming);
    
    // Configuration sliders
    const intervalSlider = document.getElementById('config-interval');
    const intervalValue = document.getElementById('config-interval-value');
    intervalSlider.addEventListener('input', () => {
        intervalValue.textContent = intervalSlider.value;
        state.streamConfig.update_interval_ms = parseInt(intervalSlider.value);
        sendConfigUpdate();
    });
    
    const sinkSlider = document.getElementById('config-sink-threshold');
    const sinkValue = document.getElementById('config-sink-threshold-value');
    sinkSlider.addEventListener('input', () => {
        const value = parseInt(sinkSlider.value) / 100;
        sinkValue.textContent = value.toFixed(2);
        state.streamConfig.sink_threshold = value;
        sendConfigUpdate();
    });
}

function startStreaming() {
    if (!state.isConnected) {
        alert('Not connected to server. Please wait...');
        return;
    }
    
    state.isStreaming = true;
    sendWSMessage({ type: 'start' });
    updateStreamButtons();
}

function stopStreaming() {
    state.isStreaming = false;
    sendWSMessage({ type: 'stop' });
    updateStreamButtons();
}

function stepStreaming() {
    sendWSMessage({ type: 'step' });
}

function resetStreaming() {
    state.isStreaming = false;
    sendWSMessage({ type: 'reset' });
    clearHeatmap();
    clearTokens();
    updateStats(0, 0, 0);
    updateStreamButtons();
}

function sendConfigUpdate() {
    sendWSMessage({
        type: 'config',
        config: state.streamConfig
    });
}

function updateStreamButtons() {
    document.getElementById('stream-start').disabled = state.isStreaming;
    document.getElementById('stream-stop').disabled = !state.isStreaming;
}

// ============================================
// Attention Heatmap Rendering
// ============================================

function renderAttentionFrame(frame) {
    const container = document.getElementById('attention-heatmap');
    const seqLen = frame.seq_len;
    
    // Calculate cell size based on sequence length
    const maxDim = Math.min(container.clientWidth - 40, container.clientHeight - 40, HEATMAP_CONFIG.maxSize);
    const cellSize = Math.max(2, Math.floor(maxDim / seqLen));
    const size = cellSize * seqLen;
    
    // Get or create canvas
    let canvas = container.querySelector('.heatmap-canvas');
    if (!canvas) {
        container.innerHTML = '';
        canvas = document.createElement('canvas');
        canvas.className = 'heatmap-canvas';
        container.appendChild(canvas);
    }
    
    canvas.width = size;
    canvas.height = size;
    
    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(size, size);
    
    // Render attention weights
    const weights = frame.attention_weights;
    const sinks = new Set(frame.sink_indices);
    const heavyHitters = new Set(frame.heavy_hitter_indices);
    
    for (let i = 0; i < seqLen; i++) {
        for (let j = 0; j < seqLen; j++) {
            const value = j <= i ? weights[i][j] : 0;
            const color = getHeatmapColor(value);
            
            // Draw cell
            for (let cy = 0; cy < cellSize; cy++) {
                for (let cx = 0; cx < cellSize; cx++) {
                    const px = j * cellSize + cx;
                    const py = i * cellSize + cy;
                    const idx = (py * size + px) * 4;
                    
                    imageData.data[idx] = color[0];
                    imageData.data[idx + 1] = color[1];
                    imageData.data[idx + 2] = color[2];
                    imageData.data[idx + 3] = 255;
                }
            }
        }
    }
    
    ctx.putImageData(imageData, 0, 0);
    
    // Draw sink and heavy hitter markers
    ctx.lineWidth = 2;
    
    // Highlight sink columns
    sinks.forEach(idx => {
        ctx.strokeStyle = '#ff3366';
        ctx.beginPath();
        ctx.moveTo(idx * cellSize + cellSize / 2, 0);
        ctx.lineTo(idx * cellSize + cellSize / 2, size);
        ctx.stroke();
    });
    
    // Highlight heavy hitter columns
    heavyHitters.forEach(idx => {
        ctx.strokeStyle = '#00ccff';
        ctx.beginPath();
        ctx.moveTo(idx * cellSize + cellSize / 2, 0);
        ctx.lineTo(idx * cellSize + cellSize / 2, size);
        ctx.stroke();
    });
    
    // Update tokens display
    renderTokens(frame.token_labels, sinks, heavyHitters);
    
    // Update stats
    updateStats(seqLen, sinks.size, heavyHitters.size);
}

function getHeatmapColor(value) {
    const stops = HEATMAP_CONFIG.colorScale;
    
    // Find the two stops to interpolate between
    let lower = stops[0];
    let upper = stops[stops.length - 1];
    
    for (let i = 0; i < stops.length - 1; i++) {
        if (value >= stops[i].stop && value <= stops[i + 1].stop) {
            lower = stops[i];
            upper = stops[i + 1];
            break;
        }
    }
    
    // Interpolate
    const range = upper.stop - lower.stop;
    const t = range > 0 ? (value - lower.stop) / range : 0;
    
    return [
        Math.round(lower.color[0] + t * (upper.color[0] - lower.color[0])),
        Math.round(lower.color[1] + t * (upper.color[1] - lower.color[1])),
        Math.round(lower.color[2] + t * (upper.color[2] - lower.color[2]))
    ];
}

function clearHeatmap() {
    const container = document.getElementById('attention-heatmap');
    container.innerHTML = `
        <div class="heatmap-placeholder">
            <p>Start streaming to visualize attention patterns</p>
        </div>
    `;
}

function renderTokens(tokens, sinks, heavyHitters) {
    const container = document.getElementById('token-container');
    container.innerHTML = '';
    
    tokens.forEach((token, idx) => {
        const el = document.createElement('span');
        el.className = 'token';
        el.textContent = token.length > 10 ? token.slice(0, 10) + '…' : token;
        
        if (sinks.has(idx)) {
            el.classList.add('sink');
        } else if (heavyHitters.has(idx)) {
            el.classList.add('heavy');
        }
        
        container.appendChild(el);
    });
    
    // Scroll to bottom
    container.scrollTop = container.scrollHeight;
}

function clearTokens() {
    const container = document.getElementById('token-container');
    container.innerHTML = '<span class="token-placeholder">Tokens will appear here...</span>';
}

function updateStats(seqLen, sinks, heavy) {
    document.getElementById('stat-seq-len').textContent = seqLen;
    document.getElementById('stat-sinks').textContent = sinks;
    document.getElementById('stat-heavy').textContent = heavy;
}

// ============================================
// Simulation View
// ============================================

function initSimulateView() {
    document.getElementById('sim-run').addEventListener('click', runSimulation);
    document.getElementById('sim-compare').addEventListener('click', runComparison);
}

async function runSimulation() {
    const prompt = document.getElementById('sim-prompt').value;
    const policy = document.getElementById('sim-policy').value;
    const cacheSize = parseInt(document.getElementById('sim-cache-size').value);
    const sinkCount = parseInt(document.getElementById('sim-sink-count').value);
    const windowSize = parseInt(document.getElementById('sim-window-size').value);
    
    const resultsContainer = document.getElementById('sim-results');
    resultsContainer.innerHTML = '<div class="results-placeholder"><div class="placeholder-icon">⏳</div><p>Running simulation...</p></div>';
    
    try {
        const response = await fetch(`${API_BASE}/api/simulate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt,
                config: {
                    policy,
                    cache_size: cacheSize,
                    sink_count: sinkCount,
                    window_size: windowSize,
                    heavy_hitter_ratio: 0.1
                },
                generate_tokens: 128
            })
        });
        
        const result = await response.json();
        renderSimulationResult(result);
    } catch (error) {
        console.error('Simulation error:', error);
        resultsContainer.innerHTML = `<div class="results-placeholder"><div class="placeholder-icon">⚠</div><p>Error: ${error.message}</p></div>`;
    }
}

async function runComparison() {
    const prompt = document.getElementById('sim-prompt').value;
    const cacheSize = parseInt(document.getElementById('sim-cache-size').value);
    const sinkCount = parseInt(document.getElementById('sim-sink-count').value);
    const windowSize = parseInt(document.getElementById('sim-window-size').value);
    
    const resultsContainer = document.getElementById('sim-results');
    resultsContainer.innerHTML = '<div class="results-placeholder"><div class="placeholder-icon">⏳</div><p>Comparing all policies...</p></div>';
    
    try {
        const response = await fetch(`${API_BASE}/api/compare`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt,
                config: {
                    policy: 'streaming_llm',
                    cache_size: cacheSize,
                    sink_count: sinkCount,
                    window_size: windowSize,
                    heavy_hitter_ratio: 0.1
                },
                generate_tokens: 128
            })
        });
        
        const results = await response.json();
        renderComparisonResults(results);
    } catch (error) {
        console.error('Comparison error:', error);
        resultsContainer.innerHTML = `<div class="results-placeholder"><div class="placeholder-icon">⚠</div><p>Error: ${error.message}</p></div>`;
    }
}

function renderSimulationResult(result) {
    const container = document.getElementById('sim-results');
    const policyName = formatPolicyName(result.policy);
    
    const hitRate = result.cache_hits / (result.cache_hits + result.cache_misses) * 100 || 0;
    
    container.innerHTML = `
        <div class="sim-result-card">
            <h4>${policyName}</h4>
            <div class="metrics-grid">
                <div class="metric">
                    <span class="metric-value">${result.total_tokens_processed}</span>
                    <span class="metric-label">Tokens Processed</span>
                </div>
                <div class="metric">
                    <span class="metric-value good">${result.cache_hits}</span>
                    <span class="metric-label">Cache Hits</span>
                </div>
                <div class="metric">
                    <span class="metric-value warning">${result.cache_misses}</span>
                    <span class="metric-label">Cache Misses</span>
                </div>
                <div class="metric">
                    <span class="metric-value bad">${result.evictions}</span>
                    <span class="metric-label">Evictions</span>
                </div>
                <div class="metric">
                    <span class="metric-value">${result.retained_sinks}</span>
                    <span class="metric-label">Retained Sinks</span>
                </div>
                <div class="metric">
                    <span class="metric-value">${result.retained_heavy_hitters}</span>
                    <span class="metric-label">Retained Heavy</span>
                </div>
            </div>
        </div>
        <div class="sim-result-card">
            <h4>Final Cache State</h4>
            <div class="metrics-grid">
                <div class="metric">
                    <span class="metric-value">${result.final_cache.total_tokens}</span>
                    <span class="metric-label">Cached Tokens</span>
                </div>
                <div class="metric">
                    <span class="metric-value">${result.final_cache.blocks.length}</span>
                    <span class="metric-label">Cache Blocks</span>
                </div>
                <div class="metric">
                    <span class="metric-value good">${hitRate.toFixed(1)}%</span>
                    <span class="metric-label">Hit Rate</span>
                </div>
            </div>
        </div>
    `;
}

function renderComparisonResults(results) {
    const container = document.getElementById('sim-results');
    
    let html = '<div class="comparison-grid">';
    
    for (const [policy, result] of Object.entries(results)) {
        const policyName = formatPolicyName(policy);
        const hitRate = result.cache_hits / (result.cache_hits + result.cache_misses) * 100 || 0;
        
        html += `
            <div class="comparison-card">
                <h4>${policyName}</h4>
                <div class="comparison-metrics">
                    <div class="comparison-metric">
                        <div class="value">${result.cache_hits}</div>
                        <div class="label">Hits</div>
                    </div>
                    <div class="comparison-metric">
                        <div class="value">${result.evictions}</div>
                        <div class="label">Evictions</div>
                    </div>
                    <div class="comparison-metric">
                        <div class="value">${hitRate.toFixed(0)}%</div>
                        <div class="label">Hit Rate</div>
                    </div>
                    <div class="comparison-metric">
                        <div class="value">${result.retained_sinks}</div>
                        <div class="label">Sinks</div>
                    </div>
                    <div class="comparison-metric">
                        <div class="value">${result.retained_heavy_hitters}</div>
                        <div class="label">Heavy</div>
                    </div>
                    <div class="comparison-metric">
                        <div class="value">${result.final_cache.total_tokens}</div>
                        <div class="label">Cached</div>
                    </div>
                </div>
            </div>
        `;
    }
    
    html += '</div>';
    container.innerHTML = html;
}

function formatPolicyName(policy) {
    const names = {
        'lru': 'Least Recently Used (LRU)',
        'sliding_window': 'Sliding Window',
        'streaming_llm': 'StreamingLLM',
        'h2o': 'H2O (Heavy-Hitter Oracle)',
        'full': 'Full Cache (No Eviction)'
    };
    return names[policy] || policy;
}

// ============================================
// Cache Profile View
// ============================================

function initProfileView() {
    document.getElementById('profile-refresh').addEventListener('click', loadCacheProfile);
}

async function loadCacheProfile() {
    const seqLen = parseInt(document.getElementById('profile-seq-len').value);
    
    try {
        const response = await fetch(`${API_BASE}/api/cache-profile?seq_len=${seqLen}`);
        const profile = await response.json();
        renderCacheProfile(profile);
    } catch (error) {
        console.error('Profile error:', error);
    }
}

function renderCacheProfile(profile) {
    // Calculate total memory
    const totalMemory = Object.values(profile.memory_usage).reduce((a, b) => a + b, 0);
    
    // Update tier sizes and bars
    const tiers = ['gpu_hbm', 'gpu_l2', 'system_ram', 'disk'];
    
    tiers.forEach(tier => {
        const size = profile.memory_usage[tier] || 0;
        const percentage = totalMemory > 0 ? (size / totalMemory) * 100 : 0;
        
        // Update size text
        const sizeEl = document.getElementById(`tier-${tier.replace('_', '-')}-size`);
        if (sizeEl) {
            sizeEl.textContent = formatBytes(size);
        }
        
        // Update bar
        const barEl = document.getElementById(`tier-${tier.replace('_', '-')}-bar`);
        if (barEl) {
            barEl.style.width = `${percentage}%`;
        }
        
        // Update blocks
        const blocksEl = document.getElementById(`tier-${tier.replace('_', '-')}-blocks`);
        if (blocksEl) {
            blocksEl.innerHTML = '';
            const tierBlocks = profile.blocks.filter(b => b.memory_tier === tier);
            tierBlocks.forEach(block => {
                const blockEl = document.createElement('div');
                blockEl.className = 'tier-block';
                if (block.is_sink) blockEl.classList.add('sink');
                if (block.is_heavy_hitter) blockEl.classList.add('heavy');
                blockEl.addEventListener('mouseenter', () => showBlockDetails(block));
                blocksEl.appendChild(blockEl);
            });
        }
    });
    
    // Render block map
    renderBlockMap(profile.blocks);
}

function renderBlockMap(blocks) {
    const container = document.getElementById('block-map');
    container.innerHTML = '';
    
    blocks.forEach(block => {
        const blockEl = document.createElement('div');
        blockEl.className = 'cache-block';
        
        // Add tier class
        const tierClass = block.memory_tier.replace('_', '-');
        blockEl.classList.add(tierClass);
        
        // Add special markers
        if (block.is_sink) blockEl.classList.add('sink');
        if (block.is_heavy_hitter) blockEl.classList.add('heavy');
        
        // Add tooltip behavior
        blockEl.addEventListener('mouseenter', () => showBlockDetails(block));
        
        container.appendChild(blockEl);
    });
}

function showBlockDetails(block) {
    const container = document.getElementById('block-info');
    
    const tierNames = {
        'gpu_hbm': 'GPU HBM',
        'gpu_l2': 'GPU L2 Cache',
        'system_ram': 'System RAM',
        'disk': 'Disk'
    };
    
    let statusClass = '';
    let statusText = 'Normal';
    if (block.is_sink) {
        statusClass = 'sink';
        statusText = 'Attention Sink';
    } else if (block.is_heavy_hitter) {
        statusClass = 'heavy';
        statusText = 'Heavy Hitter';
    }
    
    container.innerHTML = `
        <div class="block-info-item">
            <span class="block-info-label">Block ID</span>
            <span class="block-info-value">${block.block_id}</span>
        </div>
        <div class="block-info-item">
            <span class="block-info-label">Token Range</span>
            <span class="block-info-value">${block.token_range[0]} - ${block.token_range[1]}</span>
        </div>
        <div class="block-info-item">
            <span class="block-info-label">Memory Tier</span>
            <span class="block-info-value">${tierNames[block.memory_tier]}</span>
        </div>
        <div class="block-info-item">
            <span class="block-info-label">Size</span>
            <span class="block-info-value">${formatBytes(block.size_bytes)}</span>
        </div>
        <div class="block-info-item">
            <span class="block-info-label">Access Count</span>
            <span class="block-info-value">${block.access_count}</span>
        </div>
        <div class="block-info-item">
            <span class="block-info-label">Status</span>
            <span class="block-info-value ${statusClass}">${statusText}</span>
        </div>
    `;
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

