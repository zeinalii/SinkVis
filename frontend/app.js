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
    currentView: 'modelhub',
    ws: null,
    isConnected: false,
    isStreaming: false,
    currentFrame: null,
    streamConfig: {
        update_interval_ms: 200,
        sink_threshold: 0.1,
        heavy_hitter_threshold: 0.05
    },
    // Model Hub state
    modelHub: {
        selectedModel: null,
        loadedModel: null,
        isLoading: false,
        searchResults: []
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
    initModelHubView();
    initArchitectureView();
    initPlaygroundView();
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
    
    // Handle "Go to Model Hub" buttons in overlays
    document.querySelectorAll('.goto-modelhub-btn').forEach(btn => {
        btn.addEventListener('click', () => switchView('modelhub'));
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
    
    // Auto-load architecture when switching to architecture tab with a loaded model
    if (viewName === 'architecture' && state.modelHub.loadedModel && !archState.architecture) {
        loadArchitecture();
    }
    
    // Check model type when switching to playground
    if (viewName === 'playground' && state.modelHub.loadedModel) {
        checkModelType();
    }
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
    
    const prompt = document.getElementById('stream-prompt').value.trim();
    if (!prompt) {
        alert('Please enter a prompt to stream');
        return;
    }
    
    state.isStreaming = true;
    sendWSMessage({ type: 'start', prompt: prompt });
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
    const prompt = document.getElementById('stream-prompt').value.trim();
    sendWSMessage({ type: 'reset', prompt: prompt || null });
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

// Store current heatmap data for tooltip
let currentHeatmapData = null;

function renderAttentionFrame(frame) {
    const container = document.getElementById('attention-heatmap');
    const seqLen = frame.seq_len;
    
    // Calculate cell size based on sequence length
    const maxDim = Math.min(container.clientWidth - 40, container.clientHeight - 40, HEATMAP_CONFIG.maxSize);
    const cellSize = Math.max(2, Math.floor(maxDim / seqLen));
    const size = cellSize * seqLen;
    
    // Store data for tooltip
    currentHeatmapData = {
        weights: frame.attention_weights,
        tokens: frame.token_labels,
        sinks: new Set(frame.sink_indices),
        heavyHitters: new Set(frame.heavy_hitter_indices),
        cellSize: cellSize,
        seqLen: seqLen
    };
    
    // Get or create canvas
    let canvas = container.querySelector('.heatmap-canvas');
    if (!canvas) {
        container.innerHTML = '';
        canvas = document.createElement('canvas');
        canvas.className = 'heatmap-canvas';
        container.appendChild(canvas);
        
        // Add hover event listeners
        canvas.addEventListener('mousemove', handleHeatmapHover);
        canvas.addEventListener('mouseleave', hideHeatmapTooltip);
    }
    
    canvas.width = size;
    canvas.height = size;
    
    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(size, size);
    
    // Render attention weights
    const weights = frame.attention_weights;
    const sinks = currentHeatmapData.sinks;
    const heavyHitters = currentHeatmapData.heavyHitters;
    
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

function handleHeatmapHover(event) {
    if (!currentHeatmapData) return;
    
    const canvas = event.target;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    const { cellSize, seqLen, weights, tokens, sinks, heavyHitters } = currentHeatmapData;
    
    // Calculate which cell is being hovered
    const keyIdx = Math.floor(x / cellSize);
    const queryIdx = Math.floor(y / cellSize);
    
    // Check bounds
    if (keyIdx < 0 || keyIdx >= seqLen || queryIdx < 0 || queryIdx >= seqLen) {
        hideHeatmapTooltip();
        return;
    }
    
    // Only show for valid attention (lower triangle)
    if (keyIdx > queryIdx) {
        hideHeatmapTooltip();
        return;
    }
    
    const attentionValue = weights[queryIdx][keyIdx];
    const queryToken = tokens[queryIdx] || `[${queryIdx}]`;
    const keyToken = tokens[keyIdx] || `[${keyIdx}]`;
    
    const isSinkKey = sinks.has(keyIdx);
    const isHeavyKey = heavyHitters.has(keyIdx);
    const isSinkQuery = sinks.has(queryIdx);
    const isHeavyQuery = heavyHitters.has(queryIdx);
    
    showHeatmapTooltip(event, {
        queryIdx,
        keyIdx,
        queryToken,
        keyToken,
        attentionValue,
        isSinkKey,
        isHeavyKey,
        isSinkQuery,
        isHeavyQuery
    });
}

function showHeatmapTooltip(event, data) {
    const tooltip = document.getElementById('heatmap-tooltip');
    
    const keyClass = data.isSinkKey ? 'sink' : (data.isHeavyKey ? 'heavy' : '');
    const queryClass = data.isSinkQuery ? 'sink' : (data.isHeavyQuery ? 'heavy' : '');
    
    tooltip.innerHTML = `
        <div class="tooltip-header">
            <span>Attention</span>
            <span class="tooltip-value">${(data.attentionValue * 100).toFixed(1)}%</span>
        </div>
        <div class="tooltip-tokens">
            <div class="token-row">
                <span class="token-label">Query [${data.queryIdx}]</span>
                <span class="token-text ${queryClass}">${escapeHtml(data.queryToken)}</span>
            </div>
            <div class="token-row">
                <span class="token-label">→ Key [${data.keyIdx}]</span>
                <span class="token-text ${keyClass}">${escapeHtml(data.keyToken)}</span>
            </div>
        </div>
        <div class="attention-bar">
            <div class="attention-fill" style="width: ${data.attentionValue * 100}%"></div>
        </div>
    `;
    
    // Position tooltip
    const tooltipRect = tooltip.getBoundingClientRect();
    let left = event.clientX + 15;
    let top = event.clientY + 15;
    
    // Keep tooltip in viewport
    if (left + 300 > window.innerWidth) {
        left = event.clientX - 315;
    }
    if (top + 150 > window.innerHeight) {
        top = event.clientY - 155;
    }
    
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
    tooltip.classList.add('visible');
}

function hideHeatmapTooltip() {
    const tooltip = document.getElementById('heatmap-tooltip');
    if (tooltip) {
        tooltip.classList.remove('visible');
    }
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
    currentHeatmapData = null;
    hideHeatmapTooltip();
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
    // Auto-refresh when seq length changes
    document.getElementById('profile-seq-len').addEventListener('change', loadCacheProfile);
    
    // Handle prompt input - update seq length estimate as user types
    const promptInput = document.getElementById('profile-prompt');
    promptInput.addEventListener('input', () => {
        const prompt = promptInput.value.trim();
        if (prompt) {
            // Rough token estimate: ~4 chars per token on average
            const estimatedTokens = Math.max(64, Math.ceil(prompt.length / 4) + 10);
            document.getElementById('profile-seq-len').value = estimatedTokens;
        }
    });
}

async function loadCacheProfile() {
    const promptText = document.getElementById('profile-prompt').value.trim();
    const seqLen = parseInt(document.getElementById('profile-seq-len').value);
    
    try {
        // Build the URL with parameters
        let profileUrl = `${API_BASE}/api/cache-profile?seq_len=${seqLen}`;
        if (promptText) {
            profileUrl += `&prompt=${encodeURIComponent(promptText)}`;
        }
        
        // Fetch both cache profile and architecture data in parallel
        const [profileRes, archRes] = await Promise.all([
            fetch(profileUrl),
            state.modelHub.loadedModel ? fetch(`${API_BASE}/api/models/architecture`) : Promise.resolve(null)
        ]);
        
        const profile = await profileRes.json();
        const architecture = archRes ? await archRes.json().catch(() => null) : null;
        
        // Use actual token count from profile if available
        const actualSeqLen = profile.total_tokens || seqLen;
        
        renderCacheProfileEnhanced(profile, architecture, actualSeqLen);
    } catch (error) {
        console.error('Profile error:', error);
    }
}

function renderCacheProfileEnhanced(profile, architecture, seqLen) {
    // Calculate total memory from tier breakdown
    const totalTierMemory = Object.values(profile.memory_usage).reduce((a, b) => a + b, 0);
    
    // Get architecture metrics if available
    const kvMetrics = architecture?.kv_cache_metrics;
    const numLayers = architecture?.num_layers || 6;
    const numHeads = architecture?.num_heads || 12;
    const hiddenSize = architecture?.hidden_size || 768;
    const modelParamsBytes = kvMetrics?.model_params_bytes || 0;
    const maxContext = architecture?.attention_config?.max_position_embeddings || 2048;
    
    // Calculate KV cache size for current sequence length
    const bytesPerToken = kvMetrics?.bytes_per_token || (2 * 2 * numLayers * hiddenSize);
    const kvCacheSize = bytesPerToken * seqLen;
    const kvPerLayer = kvCacheSize / numLayers;
    
    // Update summary metrics
    document.getElementById('profile-total-kv').textContent = formatBytes(kvCacheSize);
    document.getElementById('profile-total-kv-detail').textContent = `${seqLen.toLocaleString()} tokens`;
    
    document.getElementById('profile-per-token').textContent = formatBytes(bytesPerToken);
    
    document.getElementById('profile-per-layer').textContent = formatBytes(kvPerLayer);
    document.getElementById('profile-per-layer-detail').textContent = `${numLayers} layers`;
    
    document.getElementById('profile-model-mem').textContent = formatBytesCompact(modelParamsBytes);
    document.getElementById('profile-model-mem-detail').textContent = architecture ? 
        `${(architecture.total_params_billions || 0).toFixed(2)}B params` : 'No model loaded';
    
    // Cache to model ratio
    const ratio = modelParamsBytes > 0 ? (kvCacheSize / modelParamsBytes * 100).toFixed(1) : '--';
    document.getElementById('profile-ratio').textContent = ratio !== '--' ? `${ratio}%` : '--';
    
    // Update memory breakdown
    const kSize = kvCacheSize / 2;  // Keys
    const vSize = kvCacheSize / 2;  // Values
    const activationsEst = hiddenSize * seqLen * numLayers * 2 * 0.5;  // Rough estimate
    const totalRuntime = kvCacheSize + modelParamsBytes + activationsEst;
    
    document.getElementById('breakdown-k-size').textContent = formatBytes(kSize);
    document.getElementById('breakdown-k-pct').textContent = 
        totalRuntime > 0 ? `${(kSize / totalRuntime * 100).toFixed(1)}%` : '--';
    
    document.getElementById('breakdown-v-size').textContent = formatBytes(vSize);
    document.getElementById('breakdown-v-pct').textContent = 
        totalRuntime > 0 ? `${(vSize / totalRuntime * 100).toFixed(1)}%` : '--';
    
    document.getElementById('breakdown-params-size').textContent = formatBytes(modelParamsBytes);
    document.getElementById('breakdown-params-pct').textContent = 
        totalRuntime > 0 ? `${(modelParamsBytes / totalRuntime * 100).toFixed(1)}%` : '--';
    
    document.getElementById('breakdown-act-size').textContent = formatBytes(activationsEst);
    document.getElementById('breakdown-act-pct').textContent = 
        totalRuntime > 0 ? `${(activationsEst / totalRuntime * 100).toFixed(1)}%` : '--';
    
    document.getElementById('breakdown-total-size').textContent = formatBytes(totalRuntime);
    
    // Render per-layer KV cache visualization
    renderLayerCacheViz(numLayers, kvPerLayer);
    
    // Update context analysis
    document.getElementById('ctx-max').textContent = maxContext.toLocaleString();
    document.getElementById('ctx-current').textContent = seqLen.toLocaleString();
    const utilPct = (seqLen / maxContext * 100).toFixed(1);
    document.getElementById('ctx-util').textContent = `${utilPct}%`;
    document.getElementById('ctx-remaining').textContent = (maxContext - seqLen).toLocaleString();
    
    // Update context meter
    document.getElementById('ctx-meter-bar').style.width = `${Math.min(100, utilPct)}%`;
    document.getElementById('ctx-meter-max').textContent = maxContext.toLocaleString();
    
    // Render scaling projections
    renderScalingProjections(bytesPerToken, modelParamsBytes, seqLen, maxContext);
    
    // Update tier sizes and bars (original functionality)
    const tiers = ['gpu_hbm', 'gpu_l2', 'system_ram', 'disk'];
    
    tiers.forEach(tier => {
        const size = profile.memory_usage[tier] || 0;
        const percentage = totalTierMemory > 0 ? (size / totalTierMemory) * 100 : 0;
        
        const sizeEl = document.getElementById(`tier-${tier.replace('_', '-')}-size`);
        if (sizeEl) {
            sizeEl.textContent = formatBytes(size);
        }
        
        const barEl = document.getElementById(`tier-${tier.replace('_', '-')}-bar`);
        if (barEl) {
            barEl.style.width = `${percentage}%`;
        }
        
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

function renderLayerCacheViz(numLayers, sizePerLayer) {
    const container = document.getElementById('layer-cache-viz');
    
    container.innerHTML = `
        <div class="layer-cache-bars">
            ${Array.from({length: numLayers}, (_, i) => 
                `<div class="layer-cache-bar" data-layer="L${i}: ${formatBytes(sizePerLayer)}" style="height: 100%;"></div>`
            ).join('')}
        </div>
        <div class="layer-cache-labels">
            <span>Layer 0</span>
            <span>Layer ${numLayers - 1}</span>
        </div>
    `;
}

function renderScalingProjections(bytesPerToken, modelParamsBytes, currentSeqLen, maxContext) {
    const container = document.getElementById('scaling-rows');
    
    const seqLengths = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
        .filter(s => s <= maxContext);
    
    container.innerHTML = seqLengths.map(seqLen => {
        const kvSize = bytesPerToken * seqLen;
        const total = kvSize + modelParamsBytes;
        const isCurrent = seqLen === currentSeqLen;
        
        return `
            <div class="scaling-row${isCurrent ? ' current' : ''}">
                <span class="ctx-len">${formatNumber(seqLen)}</span>
                <span class="kv-size">${formatBytes(kvSize)}</span>
                <span class="total-size">${formatBytes(total)}</span>
            </div>
        `;
    }).join('');
}

function formatNumber(num) {
    if (num >= 1000000) return (num / 1000000).toFixed(0) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(0) + 'K';
    return num.toString();
}

function formatBytesCompact(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    const val = bytes / Math.pow(k, i);
    return val >= 10 ? val.toFixed(0) + ' ' + sizes[i] : val.toFixed(1) + ' ' + sizes[i];
}

// Old renderCacheProfile for compatibility
function renderCacheProfile(profile) {
    renderCacheProfileEnhanced(profile, null, parseInt(document.getElementById('profile-seq-len').value));
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

// ============================================
// Model Hub View
// ============================================

function initModelHubView() {
    // Search functionality
    const searchBtn = document.getElementById('model-search-btn');
    const searchInput = document.getElementById('model-search-input');
    
    searchBtn.addEventListener('click', searchModels);
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') searchModels();
    });
    
    // Load/Unload buttons
    document.getElementById('model-load-btn').addEventListener('click', loadSelectedModel);
    document.getElementById('model-unload-btn').addEventListener('click', unloadModel);
    
    // Attention test
    document.getElementById('attention-test-btn').addEventListener('click', testAttention);
    
    // Quick action buttons
    document.querySelectorAll('.quick-action-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const action = btn.dataset.action;
            if (action) switchView(action);
        });
    });
    
    // Check initial model status and token
    checkModelStatus();
    checkTokenStatus();
}

async function checkTokenStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/models/token-info`);
        if (!response.ok) return;
        
        const tokenInfo = await response.json();
        renderTokenStatus(tokenInfo);
    } catch (error) {
        // Server not available, show empty input
        renderTokenStatus({ available: false, source: 'none' });
    }
}

function renderTokenStatus(tokenInfo) {
    const statusEl = document.getElementById('token-status');
    const inputEl = document.getElementById('hf-token');
    
    if (tokenInfo.available) {
        const sourceLabels = {
            'hf_cache': 'HuggingFace CLI',
            'env_var': 'Environment Variable',
            'env_file': '.env file'
        };
        
        const sourceLabel = sourceLabels[tokenInfo.source] || tokenInfo.source;
        const sourcePath = tokenInfo.source_path || '';
        
        statusEl.innerHTML = `
            <div class="token-found">
                <span class="token-icon">✓</span>
                <span>Token loaded from <strong>${sourceLabel}</strong></span>
                ${sourcePath ? `<span class="token-path" title="${sourcePath}">(${truncatePath(sourcePath)})</span>` : ''}
                <span class="token-masked">${tokenInfo.masked_token}</span>
            </div>
        `;
        statusEl.classList.add('has-token');
        
        // Hide the manual input since we have a token
        inputEl.style.display = 'none';
        inputEl.placeholder = 'Token auto-detected';
    } else {
        statusEl.innerHTML = `
            <div class="token-missing">
                <span class="token-icon">○</span>
                <span>No token found. Enter manually for gated models.</span>
            </div>
        `;
        statusEl.classList.remove('has-token');
        inputEl.style.display = 'block';
        inputEl.placeholder = 'hf_...';
    }
}

function truncatePath(path) {
    if (path.length <= 40) return path;
    const parts = path.split('/');
    if (parts.length <= 3) return path;
    return `.../${parts.slice(-2).join('/')}`;
}

async function searchModels() {
    const query = document.getElementById('model-search-input').value.trim();
    if (!query) return;
    
    const filterTextGen = document.getElementById('filter-text-gen').checked;
    const resultsContainer = document.getElementById('model-search-results');
    
    resultsContainer.innerHTML = `
        <div class="model-list-placeholder">
            <div class="placeholder-icon loading">⟳</div>
            <p>Searching...</p>
        </div>
    `;
    
    try {
        const response = await fetch(
            `${API_BASE}/api/models/search?query=${encodeURIComponent(query)}&limit=20&filter_text_generation=${filterTextGen}`
        );
        
        if (!response.ok) throw new Error('Search failed');
        
        const models = await response.json();
        state.modelHub.searchResults = models;
        renderSearchResults(models);
    } catch (error) {
        console.error('Search error:', error);
        resultsContainer.innerHTML = `
            <div class="model-list-placeholder error">
                <div class="placeholder-icon">⚠</div>
                <p>Search failed: ${error.message}</p>
            </div>
        `;
    }
}

function renderSearchResults(models) {
    const container = document.getElementById('model-search-results');
    
    if (models.length === 0) {
        container.innerHTML = `
            <div class="model-list-placeholder">
                <div class="placeholder-icon">∅</div>
                <p>No models found</p>
            </div>
        `;
        return;
    }
    
    container.innerHTML = models.map(model => `
        <div class="model-card" data-model-id="${model.model_id}">
            <div class="model-card-header">
                <span class="model-name">${model.model_id}</span>
                ${model.is_gated ? '<span class="model-badge gated">Gated</span>' : ''}
            </div>
            <div class="model-card-meta">
                <span class="meta-item">
                    <span class="meta-icon">↓</span> ${formatNumber(model.downloads)}
                </span>
                <span class="meta-item">
                    <span class="meta-icon">♥</span> ${formatNumber(model.likes)}
                </span>
                ${model.pipeline_tag ? `<span class="meta-item tag">${model.pipeline_tag}</span>` : ''}
            </div>
        </div>
    `).join('');
    
    // Add click handlers
    container.querySelectorAll('.model-card').forEach(card => {
        card.addEventListener('click', () => selectModel(card.dataset.modelId));
    });
}

function formatNumber(num) {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toString();
}

function selectModel(modelId) {
    // Update UI selection
    document.querySelectorAll('.model-card').forEach(card => {
        card.classList.toggle('selected', card.dataset.modelId === modelId);
    });
    
    const model = state.modelHub.searchResults.find(m => m.model_id === modelId);
    state.modelHub.selectedModel = model;
    
    renderModelInfo(model);
    document.getElementById('model-load-btn').disabled = false;
}

function renderModelInfo(model) {
    const container = document.getElementById('model-info-card');
    
    if (!model) {
        container.innerHTML = `
            <div class="model-info-placeholder">
                <p>Select a model from the search results</p>
            </div>
        `;
        return;
    }
    
    container.innerHTML = `
        <div class="model-info-content">
            <h3>${model.model_id}</h3>
            ${model.author ? `<p class="model-author">by ${model.author}</p>` : ''}
            <div class="model-stats">
                <div class="stat-item">
                    <span class="stat-value">${formatNumber(model.downloads)}</span>
                    <span class="stat-label">Downloads</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">${formatNumber(model.likes)}</span>
                    <span class="stat-label">Likes</span>
                </div>
            </div>
            ${model.pipeline_tag ? `<div class="model-pipeline">${model.pipeline_tag}</div>` : ''}
            ${model.tags.length > 0 ? `
                <div class="model-tags">
                    ${model.tags.slice(0, 8).map(tag => `<span class="tag">${tag}</span>`).join('')}
                </div>
            ` : ''}
            ${model.is_gated ? `
                <div class="gated-warning">
                    <span class="icon">🔒</span>
                    This model requires authentication. Please provide your HuggingFace token.
                </div>
            ` : ''}
        </div>
    `;
}

async function loadSelectedModel() {
    const model = state.modelHub.selectedModel;
    if (!model) return;
    
    const device = document.getElementById('load-device').value;
    const dtype = document.getElementById('load-dtype').value;
    const tokenInput = document.getElementById('hf-token');
    // Only send manual token if the input is visible and has a value
    const token = (tokenInput.style.display !== 'none' && tokenInput.value) ? tokenInput.value : null;
    const trustRemoteCode = document.getElementById('trust-remote-code').checked;
    
    state.modelHub.isLoading = true;
    updateLoadingUI(true);
    updateModelStatus('downloading', 0, `Downloading ${model.model_id}...`);
    
    try {
        const response = await fetch(`${API_BASE}/api/models/load`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_id: model.model_id,
                device,
                dtype,
                trust_remote_code: trustRemoteCode,
                token,
            })
        });
        
        if (!response.ok) {
            const error = await response.text();
            throw new Error(error);
        }
        
        const loadedModel = await response.json();
        state.modelHub.loadedModel = loadedModel;
        
        updateModelStatus('ready', 1.0, 'Model loaded successfully');
        renderLoadedModelInfo(loadedModel);
        updateLoadingUI(false);
        
        // Check model type and update playground UI
        await checkModelType();
        
        // Enable attention test
        document.getElementById('attention-test-btn').disabled = false;
        document.getElementById('model-unload-btn').disabled = false;
        
    } catch (error) {
        console.error('Load error:', error);
        updateModelStatus('error', 0, '', error.message);
        updateLoadingUI(false);
    }
    
    state.modelHub.isLoading = false;
}

async function unloadModel() {
    try {
        await fetch(`${API_BASE}/api/models/unload`, { method: 'POST' });
        
        state.modelHub.loadedModel = null;
        updateModelStatus('idle', 0, 'Model unloaded');
        
        document.getElementById('loaded-model-info').innerHTML = `
            <div class="no-model-loaded">
                <div class="placeholder-icon">⬡</div>
                <p>Load a model to see its details</p>
            </div>
        `;
        
        // Hide quick actions
        const quickActions = document.getElementById('quick-actions');
        if (quickActions) {
            quickActions.style.display = 'none';
        }
        
        document.getElementById('attention-test-btn').disabled = true;
        document.getElementById('model-unload-btn').disabled = true;
        document.getElementById('attention-test-results').innerHTML = `
            <p class="results-placeholder">Load a model to test attention</p>
        `;
        
        // Update header indicator
        updateModelIndicator(null);
        
        // Show model-required overlays
        updateModelRequiredOverlays(false);
        
    } catch (error) {
        console.error('Unload error:', error);
    }
}

function updateLoadingUI(isLoading) {
    document.getElementById('model-load-btn').disabled = isLoading || !state.modelHub.selectedModel;
    document.getElementById('model-search-btn').disabled = isLoading;
}

function updateModelStatus(status, progress, message, error = null) {
    const panel = document.getElementById('model-status-panel');
    
    const statusIcons = {
        idle: '○',
        searching: '⟳',
        downloading: '↓',
        loading: '⟳',
        ready: '●',
        error: '✕'
    };
    
    const statusClasses = {
        idle: '',
        searching: 'loading',
        downloading: 'loading',
        loading: 'loading',
        ready: 'ready',
        error: 'error'
    };
    
    let html = `
        <div class="status-${status} ${statusClasses[status]}">
            <span class="status-icon">${statusIcons[status]}</span>
            <span>${error || message || 'No model loaded'}</span>
        </div>
    `;
    
    if (progress > 0 && progress < 1 && (status === 'downloading' || status === 'loading')) {
        html += `
            <div class="progress-bar">
                <div class="progress-fill" style="width: ${progress * 100}%"></div>
            </div>
        `;
    }
    
    panel.innerHTML = html;
}

function renderLoadedModelInfo(model) {
    const container = document.getElementById('loaded-model-info');
    
    container.innerHTML = `
        <div class="loaded-model-content">
            <div class="loaded-header">
                <span class="status-dot active"></span>
                <h4>${model.model_id}</h4>
            </div>
            <div class="model-specs">
                <div class="spec-item">
                    <span class="spec-label">Layers</span>
                    <span class="spec-value">${model.num_layers}</span>
                </div>
                <div class="spec-item">
                    <span class="spec-label">Heads</span>
                    <span class="spec-value">${model.num_heads}</span>
                </div>
                <div class="spec-item">
                    <span class="spec-label">Hidden Size</span>
                    <span class="spec-value">${model.hidden_size}</span>
                </div>
                <div class="spec-item">
                    <span class="spec-label">Vocab Size</span>
                    <span class="spec-value">${formatNumber(model.vocab_size)}</span>
                </div>
                <div class="spec-item">
                    <span class="spec-label">Device</span>
                    <span class="spec-value device">${model.device.toUpperCase()}</span>
                </div>
                <div class="spec-item">
                    <span class="spec-label">Precision</span>
                    <span class="spec-value">${model.dtype}</span>
                </div>
                <div class="spec-item full-width">
                    <span class="spec-label">Memory Usage</span>
                    <span class="spec-value memory">${model.memory_mb.toFixed(1)} MB</span>
                </div>
            </div>
        </div>
    `;
    
    // Show quick actions
    const quickActions = document.getElementById('quick-actions');
    if (quickActions) {
        quickActions.style.display = 'block';
    }
    
    // Update model indicator in header
    updateModelIndicator(model);
    
    // Hide model-required overlays
    updateModelRequiredOverlays(true);
}

function updateModelIndicator(model) {
    const indicator = document.getElementById('model-indicator');
    if (!indicator) return;
    
    if (model) {
        indicator.classList.add('active');
        indicator.innerHTML = `
            <span class="indicator-dot active"></span>
            <span class="indicator-text">${model.model_id}</span>
        `;
    } else {
        indicator.classList.remove('active');
        indicator.innerHTML = `
            <span class="indicator-dot"></span>
            <span class="indicator-text">No model loaded</span>
        `;
    }
}

function updateModelRequiredOverlays(modelLoaded) {
    const overlays = document.querySelectorAll('.model-required-overlay');
    overlays.forEach(overlay => {
        overlay.style.display = modelLoaded ? 'none' : 'flex';
    });
}

async function testAttention() {
    if (!state.modelHub.loadedModel) return;
    
    const text = document.getElementById('attention-test-input').value;
    const layer = parseInt(document.getElementById('attention-layer').value) || -1;
    const headInput = document.getElementById('attention-head').value;
    const head = headInput ? parseInt(headInput) : null;
    
    const resultsContainer = document.getElementById('attention-test-results');
    resultsContainer.innerHTML = '<p class="loading">Computing attention...</p>';
    
    try {
        const response = await fetch(`${API_BASE}/api/models/attention`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, layer, head })
        });
        
        if (!response.ok) {
            const error = await response.text();
            throw new Error(error);
        }
        
        const result = await response.json();
        renderAttentionResults(result);
        
    } catch (error) {
        console.error('Attention error:', error);
        resultsContainer.innerHTML = `<p class="error">Error: ${error.message}</p>`;
    }
}

function renderAttentionResults(result) {
    const container = document.getElementById('attention-test-results');
    
    const tokens = result.tokens;
    const seqLen = result.seq_len;
    const isMultiHead = Array.isArray(result.attention_weights[0][0]);
    
    let html = `
        <div class="attention-meta">
            <span>Layer: ${result.layer}</span>
            <span>Tokens: ${seqLen}</span>
            ${result.head !== null ? `<span>Head: ${result.head}</span>` : `<span>Heads: ${result.attention_weights.length}</span>`}
        </div>
        <div class="attention-tokens">
            ${tokens.map((t, i) => `<span class="attn-token" data-idx="${i}">${escapeHtml(t)}</span>`).join('')}
        </div>
    `;
    
    // Simple heatmap visualization for single head
    if (!isMultiHead) {
        const weights = result.attention_weights;
        html += renderMiniHeatmap(weights, tokens);
    }
    
    container.innerHTML = html;
}

function renderMiniHeatmap(weights, tokens) {
    const size = Math.min(weights.length, 32);
    const cellSize = Math.floor(200 / size);
    
    let html = '<div class="mini-heatmap" style="display: grid; grid-template-columns: repeat(' + size + ', ' + cellSize + 'px); gap: 1px;">';
    
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            const value = j <= i ? weights[i][j] : 0;
            const intensity = Math.floor(value * 255);
            const color = value > 0.1 
                ? `rgb(${intensity}, ${Math.floor(intensity * 0.2)}, ${Math.floor(intensity * 0.4)})`
                : `rgb(${Math.floor(intensity * 0.5)}, ${Math.floor(intensity * 0.5)}, ${Math.floor(intensity * 0.7)})`;
            html += `<div class="heatmap-cell" style="width: ${cellSize}px; height: ${cellSize}px; background: ${color};" title="${tokens[j]} → ${tokens[i]}: ${value.toFixed(3)}"></div>`;
        }
    }
    
    html += '</div>';
    return html;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function checkModelStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/models/status`);
        if (!response.ok) return;
        
        const status = await response.json();
        
        if (status.status === 'ready' && status.model_id) {
            // Model is already loaded, update UI
            updateModelStatus('ready', 1.0, 'Model loaded');
            document.getElementById('attention-test-btn').disabled = false;
            document.getElementById('model-unload-btn').disabled = false;
            
            // Show quick actions
            const quickActions = document.getElementById('quick-actions');
            if (quickActions) {
                quickActions.style.display = 'block';
            }
            
            // Update header indicator
            updateModelIndicator({ model_id: status.model_id });
            
            // Hide model-required overlays
            updateModelRequiredOverlays(true);
            
            // Show model info
            document.getElementById('loaded-model-info').innerHTML = `
                <div class="loaded-model-content">
                    <div class="loaded-header">
                        <span class="status-dot active"></span>
                        <h4>${status.model_id}</h4>
                    </div>
                    <p class="note">Model loaded from previous session</p>
                </div>
            `;
        } else {
            updateModelStatus(status.status, status.progress, status.message, status.error);
            // Show model-required overlays since no model is loaded
            updateModelRequiredOverlays(false);
        }
    } catch (error) {
        // Server not available, show overlays
        updateModelRequiredOverlays(false);
    }
}

// ============================================
// Architecture View
// ============================================

// State for architecture view
const archState = {
    architecture: null,
    layerAnalysis: null,
    allLayersAttention: null,
    selectedLayer: null,
};

function initArchitectureView() {
    document.getElementById('arch-refresh').addEventListener('click', loadArchitecture);
    document.getElementById('arch-analyze-btn').addEventListener('click', analyzeAllLayers);
    
    document.getElementById('layer-view-mode').addEventListener('change', (e) => {
        if (archState.architecture) {
            renderLayerVisualization(archState.architecture, e.target.value);
        }
    });
}

async function loadArchitecture() {
    const modelInfo = document.getElementById('arch-model-info');
    modelInfo.innerHTML = '<div class="arch-placeholder"><p>Loading architecture...</p></div>';
    
    try {
        const response = await fetch(`${API_BASE}/api/models/architecture`);
        if (!response.ok) throw new Error('Failed to load architecture');
        
        const arch = await response.json();
        archState.architecture = arch;
        
        renderArchitectureOverview(arch);
        renderLayerVisualization(arch, 'stack');
        renderKVCacheMetrics(arch);
        
    } catch (error) {
        console.error('Architecture load error:', error);
        modelInfo.innerHTML = `<div class="arch-placeholder"><p>Error: ${error.message}</p></div>`;
    }
}

function renderArchitectureOverview(arch) {
    const container = document.getElementById('arch-model-info');
    
    const attentionType = arch.attention_config.is_gqa ? 'GQA' : 
                          (arch.attention_config.is_mqa ? 'MQA' : 'MHA');
    
    container.innerHTML = `
        <div class="arch-model-header">
            <span class="model-type">${arch.model_type}</span>
            <div class="model-name">${arch.model_id}</div>
            <div class="model-params">${arch.total_params_billions.toFixed(2)}B parameters</div>
        </div>
        
        <div class="arch-specs">
            <div class="arch-spec-item">
                <div class="arch-spec-value">${arch.num_layers}</div>
                <div class="arch-spec-label">Layers</div>
            </div>
            <div class="arch-spec-item">
                <div class="arch-spec-value">${arch.num_heads}</div>
                <div class="arch-spec-label">Attention Heads</div>
            </div>
            <div class="arch-spec-item">
                <div class="arch-spec-value">${arch.hidden_size}</div>
                <div class="arch-spec-label">Hidden Size</div>
            </div>
            <div class="arch-spec-item">
                <div class="arch-spec-value">${arch.attention_config.head_dim}</div>
                <div class="arch-spec-label">Head Dimension</div>
            </div>
            <div class="arch-spec-item">
                <div class="arch-spec-value highlight">${formatNumber(arch.vocab_size)}</div>
                <div class="arch-spec-label">Vocab Size</div>
            </div>
            <div class="arch-spec-item">
                <div class="arch-spec-value">${arch.intermediate_size}</div>
                <div class="arch-spec-label">FFN Size</div>
            </div>
            <div class="arch-spec-item full">
                <div class="arch-spec-value">${arch.dtype} on ${arch.device.toUpperCase()}</div>
                <div class="arch-spec-label">Precision & Device</div>
            </div>
        </div>
        
        <div class="arch-attention-info">
            <h4>Attention Configuration</h4>
            <div class="attention-features">
                <div class="attention-feature">
                    <span class="feature-name">Attention Type</span>
                    <span class="feature-value">${attentionType}</span>
                </div>
                ${arch.attention_config.num_kv_heads ? `
                <div class="attention-feature">
                    <span class="feature-name">KV Heads</span>
                    <span class="feature-value">${arch.attention_config.num_kv_heads}</span>
                </div>
                ` : ''}
                <div class="attention-feature">
                    <span class="feature-name">Max Context</span>
                    <span class="feature-value">${formatNumber(arch.attention_config.max_position_embeddings)}</span>
                </div>
                <div class="attention-feature">
                    <span class="feature-name">RoPE Embeddings</span>
                    <span class="feature-value ${arch.attention_config.rotary_embedding ? 'yes' : 'no'}">${arch.attention_config.rotary_embedding ? 'Yes' : 'No'}</span>
                </div>
                ${arch.sliding_window ? `
                <div class="attention-feature">
                    <span class="feature-name">Sliding Window</span>
                    <span class="feature-value">${arch.sliding_window}</span>
                </div>
                ` : ''}
                ${arch.rope_theta ? `
                <div class="attention-feature">
                    <span class="feature-name">RoPE Theta</span>
                    <span class="feature-value">${arch.rope_theta.toLocaleString()}</span>
                </div>
                ` : ''}
            </div>
        </div>
    `;
}

function renderLayerVisualization(arch, mode) {
    const container = document.getElementById('layer-visualization');
    
    if (mode === 'stack') {
        renderStackView(container, arch);
    } else if (mode === 'grid') {
        renderGridView(container, arch);
    } else {
        renderDetailedView(container, arch);
    }
}

function renderStackView(container, arch) {
    let html = '<div class="layer-stack">';
    
    // Embedding layer
    html += `
        <div class="layer-block embedding">
            <span class="layer-index">EMB</span>
            <div class="layer-components">
                <div class="layer-component norm">Token Embedding</div>
                <div class="layer-component norm">Position Embedding</div>
            </div>
            <span class="layer-dims">${arch.vocab_size} → ${arch.hidden_size}</span>
        </div>
    `;
    
    // Transformer layers
    for (let i = 0; i < arch.num_layers; i++) {
        const layer = arch.layers[i];
        html += `
            <div class="layer-block" data-layer="${i}">
                <span class="layer-index">L${i}</span>
                <div class="layer-components">
                    <div class="layer-component attention">${layer.num_heads}H Attn</div>
                    <div class="layer-component mlp">MLP</div>
                    <div class="layer-component norm">LN</div>
                </div>
                <span class="layer-dims">${layer.hidden_size}d</span>
            </div>
        `;
    }
    
    // Output layer
    html += `
        <div class="layer-block output">
            <span class="layer-index">OUT</span>
            <div class="layer-components">
                <div class="layer-component norm">LM Head</div>
            </div>
            <span class="layer-dims">${arch.hidden_size} → ${arch.vocab_size}</span>
        </div>
    `;
    
    html += '</div>';
    container.innerHTML = html;
    
    // Add click handlers
    container.querySelectorAll('.layer-block[data-layer]').forEach(block => {
        block.addEventListener('click', () => {
            const layerIdx = parseInt(block.dataset.layer);
            selectLayer(layerIdx);
        });
    });
}

function renderGridView(container, arch) {
    let html = '<div class="layer-grid">';
    
    for (let i = 0; i < arch.num_layers; i++) {
        html += `
            <div class="layer-grid-item" data-layer="${i}">
                <span class="layer-num">${i}</span>
                <span class="layer-type">Block</span>
            </div>
        `;
    }
    
    html += '</div>';
    container.innerHTML = html;
    
    container.querySelectorAll('.layer-grid-item').forEach(item => {
        item.addEventListener('click', () => {
            const layerIdx = parseInt(item.dataset.layer);
            selectLayer(layerIdx);
        });
    });
}

function renderDetailedView(container, arch) {
    // Show detailed view with expandable sections
    let html = '<div class="layer-detailed">';
    
    for (let i = 0; i < Math.min(arch.num_layers, 12); i++) {
        const layer = arch.layers[i];
        html += `
            <div class="layer-detailed-item" data-layer="${i}">
                <div class="layer-detailed-header">
                    <span class="layer-num">Layer ${i}</span>
                    <span class="layer-type">${layer.attention_type}</span>
                </div>
                <div class="layer-detailed-body">
                    <div class="detail-row">
                        <span>Attention</span>
                        <span>${layer.num_heads} heads × ${layer.head_dim} dim</span>
                    </div>
                    <div class="detail-row">
                        <span>FFN</span>
                        <span>${layer.hidden_size} → ${layer.intermediate_size} → ${layer.hidden_size}</span>
                    </div>
                </div>
            </div>
        `;
    }
    
    if (arch.num_layers > 12) {
        html += `<div class="layer-more">... and ${arch.num_layers - 12} more layers</div>`;
    }
    
    html += '</div>';
    container.innerHTML = html;
}

function selectLayer(layerIdx) {
    archState.selectedLayer = layerIdx;
    
    // Update UI selection
    document.querySelectorAll('.layer-block, .layer-grid-item').forEach(el => {
        el.classList.toggle('selected', parseInt(el.dataset.layer) === layerIdx);
    });
    
    // Could show layer-specific details here
}

function renderKVCacheMetrics(arch) {
    const container = document.getElementById('kv-cache-metrics');
    const projectionContainer = document.getElementById('memory-projection');
    
    const metrics = arch.kv_cache_metrics;
    
    container.innerHTML = `
        <div class="kv-metric-card">
            <h4>KV Cache per Token</h4>
            <div class="kv-metric-value">${formatBytes(metrics.bytes_per_token)}</div>
            <div class="kv-metric-sub">per token across all layers</div>
        </div>
        
        <div class="kv-metric-card">
            <h4>Cache at 2K Context</h4>
            <div class="kv-metric-value">${formatBytes(metrics.total_kv_cache_bytes)}</div>
            <div class="kv-metric-sub">2048 tokens × ${arch.num_layers} layers</div>
        </div>
        
        <div class="kv-metric-card">
            <h4>Model Parameters</h4>
            <div class="kv-metric-value">${formatBytes(metrics.model_params_bytes)}</div>
            <div class="kv-metric-sub">${arch.total_params_billions.toFixed(2)}B params</div>
        </div>
        
        <div class="kv-metric-card">
            <h4>Cache / Model Ratio</h4>
            <div class="kv-metric-value ${metrics.kv_cache_ratio > 0.5 ? 'warning' : ''}">${(metrics.kv_cache_ratio * 100).toFixed(1)}%</div>
            <div class="kv-metric-sub">at 2K context</div>
        </div>
        
        <div class="kv-metric-card">
            <h4>Theoretical Max Context</h4>
            <div class="kv-metric-value">${formatNumber(metrics.theoretical_max_context)}</div>
            <div class="kv-metric-sub">on 80GB GPU</div>
        </div>
    `;
    
    // Render memory projection chart
    const seqLengths = Object.keys(metrics.memory_at_seq_lengths).map(Number).sort((a, b) => a - b);
    const maxMemory = Math.max(...Object.values(metrics.memory_at_seq_lengths));
    
    let chartHtml = '<div class="projection-bars">';
    
    for (const seqLen of seqLengths.slice(0, 7)) {
        const memory = metrics.memory_at_seq_lengths[seqLen];
        const heightPercent = (memory / maxMemory) * 100;
        
        chartHtml += `
            <div class="projection-bar">
                <div class="projection-bar-fill" style="height: ${heightPercent}%"></div>
                <div class="projection-bar-label">${formatSeqLen(seqLen)}</div>
            </div>
        `;
    }
    
    chartHtml += '</div>';
    chartHtml += `
        <div class="projection-legend">
            <span>KV Cache memory at different sequence lengths</span>
        </div>
    `;
    
    projectionContainer.innerHTML = chartHtml;
}

function formatSeqLen(len) {
    if (len >= 1024) return `${len / 1024}K`;
    return len.toString();
}

async function analyzeAllLayers() {
    const text = document.getElementById('arch-sample-text').value.trim();
    if (!text) {
        alert('Please enter sample text to analyze');
        return;
    }
    
    const gridContainer = document.getElementById('layer-attention-grid');
    const statsContainer = document.getElementById('layer-stats-grid');
    
    gridContainer.innerHTML = '<div class="analysis-placeholder"><p>Analyzing attention across layers...</p></div>';
    statsContainer.innerHTML = '';
    
    try {
        // Get attention from all layers
        const response = await fetch(`${API_BASE}/api/models/all-layers-attention`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        
        if (!response.ok) throw new Error('Failed to get attention');
        
        const result = await response.json();
        archState.allLayersAttention = result;
        
        renderLayerAttentionGrid(result);
        
        // Get layer statistics
        const statsResponse = await fetch(`${API_BASE}/api/models/layer-analysis`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        
        if (statsResponse.ok) {
            const stats = await statsResponse.json();
            archState.layerAnalysis = stats;
            renderLayerStats(stats);
        }
        
    } catch (error) {
        console.error('Analysis error:', error);
        gridContainer.innerHTML = `<div class="analysis-placeholder"><p>Error: ${error.message}</p></div>`;
    }
}

function renderLayerAttentionGrid(result) {
    const container = document.getElementById('layer-attention-grid');
    
    let html = '';
    
    for (const layer of result.layers) {
        html += `
            <div class="layer-attention-item" data-layer="${layer.layer}">
                <span class="layer-label">Layer ${layer.layer}</span>
                <div class="layer-heatmap-mini">
                    <canvas id="layer-mini-${layer.layer}" width="80" height="80"></canvas>
                </div>
            </div>
        `;
    }
    
    container.innerHTML = html;
    
    // Render mini heatmaps
    for (const layer of result.layers) {
        renderMiniHeatmap(`layer-mini-${layer.layer}`, layer.attention, result.seq_len);
    }
    
    // Add click handlers
    container.querySelectorAll('.layer-attention-item').forEach(item => {
        item.addEventListener('click', () => {
            const layerIdx = parseInt(item.dataset.layer);
            showLayerDetail(layerIdx);
        });
    });
}

function renderMiniHeatmap(canvasId, attention, seqLen) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const size = 80;
    const cellSize = size / seqLen;
    
    for (let i = 0; i < seqLen; i++) {
        for (let j = 0; j < seqLen; j++) {
            const value = j <= i ? attention[i][j] : 0;
            const color = getHeatmapColor(value);
            
            ctx.fillStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
            ctx.fillRect(j * cellSize, i * cellSize, cellSize + 1, cellSize + 1);
        }
    }
}

function showLayerDetail(layerIdx) {
    // Select the layer
    document.querySelectorAll('.layer-attention-item').forEach(item => {
        item.classList.toggle('selected', parseInt(item.dataset.layer) === layerIdx);
    });
    
    // Could show detailed view in a modal or side panel
    console.log('Selected layer:', layerIdx);
}

function renderLayerStats(stats) {
    const container = document.getElementById('layer-stats-grid');
    
    const avgEntropy = stats.average_entropy;
    const avgSparsity = stats.average_sparsity;
    
    const sinkPositions = stats.sink_positions.slice(0, 5).join(', ');
    const heavyPositions = stats.heavy_hitter_positions.slice(0, 5).join(', ') || 'None';
    
    container.innerHTML = `
        <div class="layer-stat-item">
            <div class="layer-stat-value good">${avgEntropy.toFixed(2)}</div>
            <div class="layer-stat-label">Avg Entropy</div>
        </div>
        <div class="layer-stat-item">
            <div class="layer-stat-value medium">${(avgSparsity * 100).toFixed(1)}%</div>
            <div class="layer-stat-label">Avg Sparsity</div>
        </div>
        <div class="layer-stat-item">
            <div class="layer-stat-value high">${stats.sink_positions.length}</div>
            <div class="layer-stat-label">Sink Positions</div>
        </div>
        <div class="layer-stat-item">
            <div class="layer-stat-value">${stats.heavy_hitter_positions.length}</div>
            <div class="layer-stat-label">Heavy Hitters</div>
        </div>
        <div class="layer-stat-item">
            <div class="layer-stat-value">${stats.layer_stats.length}</div>
            <div class="layer-stat-label">Total Layers</div>
        </div>
    `;
}

// ============================================
// Playground View
// ============================================

// Playground state
const playgroundState = {
    modelType: 'causal_lm',  // 'causal_lm' or 'diffusion'
};

function initPlaygroundView() {
    document.getElementById('playground-generate').addEventListener('click', handleGenerate);
    document.getElementById('playground-clear').addEventListener('click', clearPlayground);
    
    // Temperature slider
    const tempSlider = document.getElementById('gen-temperature');
    const tempValue = document.getElementById('gen-temperature-value');
    tempSlider.addEventListener('input', () => {
        tempValue.textContent = (parseInt(tempSlider.value) / 100).toFixed(2);
    });
    
    // Top-P slider
    const topPSlider = document.getElementById('gen-top-p');
    const topPValue = document.getElementById('gen-top-p-value');
    topPSlider.addEventListener('input', () => {
        topPValue.textContent = (parseInt(topPSlider.value) / 100).toFixed(2);
    });
    
    // Image guidance slider
    const guidanceSlider = document.getElementById('img-guidance');
    const guidanceValue = document.getElementById('img-guidance-value');
    if (guidanceSlider && guidanceValue) {
        guidanceSlider.addEventListener('input', () => {
            guidanceValue.textContent = (parseInt(guidanceSlider.value) / 10).toFixed(1);
        });
    }
}

async function checkModelType() {
    try {
        const response = await fetch(`${API_BASE}/api/models/type`);
        const data = await response.json();
        
        if (data.loaded) {
            playgroundState.modelType = data.type || 'causal_lm';
            updatePlaygroundUI(data.is_diffusion);
        }
    } catch (error) {
        console.error('Failed to check model type:', error);
    }
}

function updatePlaygroundUI(isDiffusion) {
    const textConfig = document.getElementById('text-gen-config');
    const imageConfig = document.getElementById('image-gen-config');
    const textTips = document.getElementById('text-gen-tips');
    const imageTips = document.getElementById('image-gen-tips');
    const textOutput = document.getElementById('playground-output');
    const imageOutput = document.getElementById('playground-image-output');
    const negativePrompt = document.getElementById('negative-prompt-group');
    const subtitle = document.getElementById('playground-subtitle');
    const indicator = document.getElementById('model-type-indicator');
    const promptLabel = document.getElementById('playground-prompt-label');
    const outputHeader = document.getElementById('output-header-text');
    const promptTextarea = document.getElementById('playground-prompt');
    
    if (isDiffusion) {
        // Show image generation UI
        textConfig.style.display = 'none';
        imageConfig.style.display = 'block';
        textTips.style.display = 'none';
        imageTips.style.display = 'block';
        textOutput.style.display = 'none';
        imageOutput.style.display = 'flex';
        negativePrompt.style.display = 'block';
        subtitle.textContent = 'Generate images with Stable Diffusion';
        indicator.style.display = 'block';
        indicator.innerHTML = '<span class="type-badge diffusion-model">Diffusion Model</span>';
        promptLabel.textContent = 'Image Prompt';
        outputHeader.textContent = 'Generated Image';
        promptTextarea.placeholder = 'A majestic mountain landscape at sunset, highly detailed, 8k...';
        promptTextarea.value = 'A beautiful sunset over mountains, highly detailed, digital art';
    } else {
        // Show text generation UI
        textConfig.style.display = 'block';
        imageConfig.style.display = 'none';
        textTips.style.display = 'block';
        imageTips.style.display = 'none';
        textOutput.style.display = 'block';
        imageOutput.style.display = 'none';
        negativePrompt.style.display = 'none';
        subtitle.textContent = 'Generate text and explore model outputs';
        indicator.style.display = 'block';
        indicator.innerHTML = '<span class="type-badge text-model">Text Model</span>';
        promptLabel.textContent = 'Prompt';
        outputHeader.textContent = 'Generated Output';
        promptTextarea.placeholder = 'Enter your prompt here...';
    }
}

async function handleGenerate() {
    if (playgroundState.modelType === 'diffusion') {
        await generateImage();
    } else {
        await generateText();
    }
}

async function generateText() {
    const prompt = document.getElementById('playground-prompt').value.trim();
    if (!prompt) {
        alert('Please enter a prompt');
        return;
    }
    
    const outputContainer = document.getElementById('playground-output');
    const statsContainer = document.getElementById('generation-stats');
    
    // Show loading
    outputContainer.innerHTML = `
        <div class="output-loading">
            <div class="spinner"></div>
            <span>Generating...</span>
        </div>
    `;
    statsContainer.innerHTML = '';
    
    // Get generation settings
    const maxTokens = parseInt(document.getElementById('gen-max-tokens').value) || 50;
    const temperature = parseInt(document.getElementById('gen-temperature').value) / 100;
    const topP = parseInt(document.getElementById('gen-top-p').value) / 100;
    const topK = parseInt(document.getElementById('gen-top-k').value) || 50;
    const doSample = document.getElementById('gen-do-sample').checked;
    
    const startTime = Date.now();
    
    try {
        const response = await fetch(`${API_BASE}/api/models/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt,
                max_new_tokens: maxTokens,
                temperature,
                top_p: topP,
                top_k: topK,
                do_sample: doSample,
            })
        });
        
        if (!response.ok) {
            const error = await response.text();
            throw new Error(error);
        }
        
        const result = await response.json();
        const elapsed = Date.now() - startTime;
        
        // Display result
        outputContainer.innerHTML = `
            <div class="output-text">
                <span class="prompt-part">${escapeHtml(result.prompt)}</span><span class="generated-part">${escapeHtml(result.generated)}</span>
            </div>
        `;
        
        // Show stats
        const tokensPerSec = (result.tokens_generated / (elapsed / 1000)).toFixed(1);
        statsContainer.innerHTML = `
            <span><span class="stat-label">Tokens:</span> ${result.tokens_generated}</span>
            <span><span class="stat-label">Time:</span> ${(elapsed / 1000).toFixed(2)}s</span>
            <span><span class="stat-label">Speed:</span> ${tokensPerSec} tok/s</span>
        `;
        
    } catch (error) {
        console.error('Generation error:', error);
        outputContainer.innerHTML = `
            <div class="output-error" style="color: var(--sink-color);">
                Error: ${error.message}
            </div>
        `;
    }
}

async function generateImage() {
    const prompt = document.getElementById('playground-prompt').value.trim();
    if (!prompt) {
        alert('Please enter a prompt');
        return;
    }
    
    const negativePrompt = document.getElementById('playground-negative-prompt')?.value.trim() || '';
    const imageOutput = document.getElementById('playground-image-output');
    const statsContainer = document.getElementById('generation-stats');
    
    // Show loading
    imageOutput.innerHTML = `
        <div class="image-loading">
            <div class="spinner"></div>
            <span class="progress-text">Generating image...</span>
        </div>
    `;
    statsContainer.innerHTML = '';
    
    // Get generation settings
    const steps = parseInt(document.getElementById('img-steps')?.value) || 20;
    const guidance = parseInt(document.getElementById('img-guidance')?.value) / 10 || 7.5;
    const width = parseInt(document.getElementById('img-width')?.value) || 512;
    const height = parseInt(document.getElementById('img-height')?.value) || 512;
    const seedInput = document.getElementById('img-seed')?.value;
    const seed = seedInput ? parseInt(seedInput) : null;
    
    const startTime = Date.now();
    
    try {
        const response = await fetch(`${API_BASE}/api/models/generate-image`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt,
                negative_prompt: negativePrompt,
                num_inference_steps: steps,
                guidance_scale: guidance,
                width,
                height,
                seed,
            })
        });
        
        if (!response.ok) {
            const error = await response.text();
            throw new Error(error);
        }
        
        const result = await response.json();
        const elapsed = Date.now() - startTime;
        
        // Display image
        imageOutput.innerHTML = `
            <div class="image-container">
                <img id="generated-image" src="data:image/png;base64,${result.image}" alt="Generated image">
            </div>
        `;
        
        // Show stats
        statsContainer.innerHTML = `
            <span><span class="stat-label">Size:</span> ${result.width}×${result.height}</span>
            <span><span class="stat-label">Steps:</span> ${result.steps}</span>
            <span><span class="stat-label">Time:</span> ${(elapsed / 1000).toFixed(1)}s</span>
        `;
        
    } catch (error) {
        console.error('Image generation error:', error);
        imageOutput.innerHTML = `
            <div class="image-container">
                <p class="image-placeholder" style="color: var(--sink-color);">
                    Error: ${error.message}
                </p>
            </div>
        `;
    }
}

function clearPlayground() {
    document.getElementById('playground-prompt').value = '';
    
    // Clear text output
    document.getElementById('playground-output').innerHTML = `
        <p class="output-placeholder">Generated content will appear here...</p>
    `;
    
    // Clear image output
    const imageOutput = document.getElementById('playground-image-output');
    if (imageOutput) {
        imageOutput.innerHTML = `
            <div class="image-container">
                <p class="image-placeholder">Generated image will appear here...</p>
            </div>
        `;
    }
    
    // Clear negative prompt
    const negPrompt = document.getElementById('playground-negative-prompt');
    if (negPrompt) negPrompt.value = '';
    
    document.getElementById('generation-stats').innerHTML = '';
}

