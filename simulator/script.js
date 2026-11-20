// Chart Configurations
Chart.defaults.color = '#8b8f94';
Chart.defaults.font.family = "'Outfit', sans-serif";

// THRESHOLD must match CRASH_THRESHOLD in CrashDetector.cpp and build/threshold.txt.
// The firmware computes MSE as sum((input[i] - output[i])^2) / 300, producing values
// in the 0.0 – 1.0 range for normal riding. 8.5 was wrong (28x too high).
const THRESHOLD = 0.30;
const MAX_DATA_POINTS = 50;

// State
let currentMode = 'normal'; // normal, pothole, crash
let isSosActive = false;
// Require 2 consecutive above-threshold readings before triggering SOS.
// This mirrors real firmware debounce behaviour and prevents instant SOS
// on the very first tick after clicking "Simulate Crash".
let aboveThresholdCount = 0;
const SOS_DEBOUNCE_COUNT = 2;

// DOM Elements
const btnNormal = document.getElementById('btn-normal');
const btnPothole = document.getElementById('btn-pothole');
const btnCrash = document.getElementById('btn-crash');
const statusIndicator = document.getElementById('status-indicator');
const sosOverlay = document.getElementById('sos-overlay');
const btnDismiss = document.getElementById('btn-dismiss');
const sosDispatchText = document.getElementById('sos-dispatch-text');
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const chatSend = document.getElementById('chat-send');
const chatMessagesElement = document.getElementById('chat-messages');

// The simulator is served separately from FastAPI. Override this from the
// browser console with `window.RIDERLINK_API_URL = "https://..."` if needed.
const SOS_DISPATCH_URL = `${window.RIDERLINK_API_URL || 'http://127.0.0.1:8000'}/api/v1/sos-dispatch`;
const DEMO_COORDINATES = [12.9716, 77.5946]; // Simulator-only example location (Bengaluru).
let dispatchController = null;
const chatHistory = [];

// Setup UI Listeners
const updateActiveBtn = (activeBtn) => {
    [btnNormal, btnPothole, btnCrash].forEach(b => {
        b.classList.remove('active');
        b.setAttribute('aria-pressed', 'false');
    });
    activeBtn.classList.add('active');
    activeBtn.setAttribute('aria-pressed', 'true');
};

btnNormal.addEventListener('click', () => { currentMode = 'normal'; updateActiveBtn(btnNormal); });
btnPothole.addEventListener('click', () => { currentMode = 'pothole'; updateActiveBtn(btnPothole); });
btnCrash.addEventListener('click', () => { currentMode = 'crash'; updateActiveBtn(btnCrash); });

btnDismiss.addEventListener('click', () => {
    if (dispatchController) dispatchController.abort();
    dispatchController = null;
    sosOverlay.classList.add('hidden');
    isSosActive = false;
    aboveThresholdCount = 0;
    currentMode = 'normal';
    updateActiveBtn(btnNormal);
    statusIndicator.className = 'status-safe';
    statusIndicator.innerText = 'SAFE (NORMAL)';
    sosDispatchText.innerText = 'High-G Anomaly Detected above Threshold';
});

async function requestSosDispatch({ ax, ay, az }) {
    if (dispatchController) dispatchController.abort();
    dispatchController = new AbortController();
    sosDispatchText.innerText = 'Crash detected. Generating emergency dispatch…';

    const impactGForce = Math.sqrt(ax ** 2 + ay ** 2 + az ** 2);
    const payload = {
        rider_id: 'simulator-rider',
        bike_model: 'Triumph Speed 400',
        speed_kmh: 48,
        impact_g_force: Number(impactGForce.toFixed(2)),
        pre_crash_lean_angle: Number((Math.atan2(ay, az) * 180 / Math.PI).toFixed(1)),
        coordinates: DEMO_COORDINATES,
        tumbles_detected: false,
        time_of_impact: new Date().toISOString()
    };

    try {
        const response = await fetch(SOS_DISPATCH_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: dispatchController.signal
        });
        if (!response.ok) {
            throw new Error(`Dispatcher returned HTTP ${response.status}`);
        }
        const result = await response.json();
        sosDispatchText.innerText = result.dispatch_text || 'Dispatch service returned an empty response.';
    } catch (error) {
        if (error.name === 'AbortError') return;
        console.error('SOS dispatch request failed:', error);
        sosDispatchText.innerText = 'SOS alert is active, but the LLM dispatcher is unavailable. Check the API connection and model credentials.';
    }
}

function appendChatMessage(role, content) {
    const messageElement = document.createElement('div');
    messageElement.className = `chat-message ${role === 'user' ? 'user-message' : 'assistant-message'}`;
    messageElement.innerText = content;
    chatMessagesElement.appendChild(messageElement);
    chatMessagesElement.scrollTop = chatMessagesElement.scrollHeight;
    return messageElement;
}

chatForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const content = chatInput.value.trim();
    if (!content || chatSend.disabled) return;

    chatHistory.push({ role: 'user', content });
    while (chatHistory.length > 20) chatHistory.shift();
    appendChatMessage('user', content);
    chatInput.value = '';
    chatSend.disabled = true;
    const pendingMessage = appendChatMessage('assistant', 'Thinking…');

    try {
        const response = await fetch(`${window.RIDERLINK_API_URL || 'http://127.0.0.1:8000'}/api/v1/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ messages: chatHistory })
        });
        if (!response.ok) throw new Error(`Chat service returned HTTP ${response.status}`);
        const result = await response.json();
        const reply = result.reply || 'The assistant returned an empty response.';
        pendingMessage.innerText = reply;
        chatHistory.push({ role: 'assistant', content: reply });
        while (chatHistory.length > 20) chatHistory.shift();
    } catch (error) {
        console.error('Chat request failed:', error);
        pendingMessage.innerText = 'The assistant is unavailable. Check the API connection and model credentials, then try again.';
        // Keep failed turns out of the model context so a retry starts cleanly.
        chatHistory.pop();
    } finally {
        chatSend.disabled = false;
        chatInput.focus();
    }
});

// Initialize Charts
const ctxImu = document.getElementById('imuChart').getContext('2d');
const ctxError = document.getElementById('errorChart').getContext('2d');

const initialLabels = Array.from({length: MAX_DATA_POINTS}, (_, i) => '');

const imuChart = new Chart(ctxImu, {
    type: 'line',
    data: {
        labels: initialLabels,
        datasets: [
            { label: 'Accel X', borderColor: '#ff0055', data: [], borderWidth: 2, tension: 0.3, pointRadius: 0 },
            { label: 'Accel Y', borderColor: '#00ffaa', data: [], borderWidth: 2, tension: 0.3, pointRadius: 0 },
            { label: 'Accel Z', borderColor: '#00aaff', data: [], borderWidth: 2, tension: 0.3, pointRadius: 0 }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        scales: {
            y: { min: -12, max: 12, grid: { color: 'rgba(255,255,255,0.05)' } },
            x: { grid: { display: false } }
        },
        plugins: { legend: { position: 'top', labels: { usePointStyle: true, boxWidth: 6 } } }
    }
});

const errorChart = new Chart(ctxError, {
    type: 'line',
    data: {
        labels: initialLabels,
        datasets: [
            {
                label: 'Reconstruction MSE',
                borderColor: '#c026d3', // Neon purple
                backgroundColor: 'rgba(192, 38, 211, 0.2)',
                fill: true,
                data: [],
                borderWidth: 3,
                tension: 0.4,
                pointRadius: 0
            },
            {
                label: 'Anomaly Threshold',
                borderColor: '#ffb800', // Warning Yellow
                borderDash: [5, 5],
                data: Array(MAX_DATA_POINTS).fill(THRESHOLD),
                borderWidth: 2,
                pointRadius: 0,
                fill: false
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        scales: {
            y: { min: 0, max: 1.0, grid: { color: 'rgba(255,255,255,0.05)' } },
            x: { grid: { display: false } }
        },
        plugins: { legend: { position: 'top', labels: { usePointStyle: true, boxWidth: 6 } } }
    }
});

// Fill initial empty data
for(let i=0; i<MAX_DATA_POINTS; i++) {
    imuChart.data.datasets.forEach(d => d.data.push(0));
    errorChart.data.datasets[0].data.push(0);
}

// Data Simulator Loop
setInterval(() => {
    if (isSosActive) return; // Freeze charts during SOS

    // 1. Generate Fake Telemetry Data
    // All IMU values are in g-force units (gravity = 1.0 g).
    // Normal riding: small noise around 0 on all axes.
    let ax = (Math.random() - 0.5) * 0.4;
    let ay = (Math.random() - 0.5) * 0.4;
    let az = (Math.random() - 0.5) * 0.4;

    // MSE values mirror the firmware's 0.0–1.0 range for normal data.
    let errorMse = Math.random() * 0.08 + 0.02; // Normal: ~0.02 – 0.10

    if (currentMode === 'pothole') {
        ax += (Math.random() - 0.5) * 2.0;
        ay += (Math.random() - 0.5) * 2.0;
        // Pothole: error rises but stays below threshold (0.30)
        errorMse = Math.random() * 0.12 + 0.14; // ~0.14 – 0.26
        statusIndicator.className = 'status-warning';
        statusIndicator.innerText = 'BUMP DETECTED';
    } 
    else if (currentMode === 'crash') {
        ax += (Math.random() - 0.5) * 8.0;
        ay += (Math.random() - 0.5) * 10.0;
        az += (Math.random() - 0.5) * 8.0;
        // Crash: error spikes well above threshold
        errorMse = Math.random() * 0.5 + 0.35; // ~0.35 – 0.85
    }
    else {
        // Normal mode — always reset status indicator
        statusIndicator.className = 'status-safe';
        statusIndicator.innerText = 'SAFE (NORMAL)';
    }

    // 2. Threshold Check Logic (Phase 3 Replication)
    // Debounce: only trigger SOS after SOS_DEBOUNCE_COUNT consecutive above-threshold
    // readings, so a single noisy spike doesn't immediately fire the overlay.
    if (errorMse >= THRESHOLD) {
        aboveThresholdCount++;
    } else {
        aboveThresholdCount = 0; // Reset on any below-threshold reading
    }

    if (aboveThresholdCount >= SOS_DEBOUNCE_COUNT) {
        statusIndicator.className = 'status-danger';
        statusIndicator.innerText = 'CRITICAL ANOMALY';
        
        isSosActive = true;
        aboveThresholdCount = 0;
        void requestSosDispatch({ ax, ay, az });
        setTimeout(() => {
            sosOverlay.classList.remove('hidden');
        }, 300); // 300ms delay to see the graph spike
    }

    // 3. Update Charts
    imuChart.data.datasets[0].data.push(ax);
    imuChart.data.datasets[1].data.push(ay); // Already zero-centred — no gravity offset needed
    imuChart.data.datasets[2].data.push(az);
    
    errorChart.data.datasets[0].data.push(errorMse);

    // Keep arrays at max length
    imuChart.data.datasets.forEach(d => {
        if(d.data.length > MAX_DATA_POINTS) d.data.shift();
    });
    if(errorChart.data.datasets[0].data.length > MAX_DATA_POINTS) {
        errorChart.data.datasets[0].data.shift();
    }

    imuChart.update();
    errorChart.update();

}, 100); // Super fast 100ms updates
