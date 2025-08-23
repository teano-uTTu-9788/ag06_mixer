"""
Production AG06 Flask app with AI-powered autonomous mixing
Integrates studio DSP chain and intelligent mixing decisions
"""

from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import os
import sys
import signal
import logging
from datetime import datetime
import psutil
import numpy as np
import threading
import time
from typing import Optional, Dict, Any

# Import our modules
from sse_streaming import sse_bp, audio_state
from audio_rt_callback import RealTimeAudioProcessor, AudioConfig
from ai_mixing_brain import AutonomousMixingEngine
from studio_dsp_chain import StudioDSPChain, GateParams, CompressorParams, EQBand, LimiterParams

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# CORS configuration - restrict to LAN
CORS(app, origins=[
    "http://localhost:*",
    "http://127.0.0.1:*", 
    "http://192.168.1.*:*",
    "http://10.0.0.*:*"
])

# Register SSE blueprint
app.register_blueprint(sse_bp)

# Global components
audio_processor: Optional[RealTimeAudioProcessor] = None
mixing_engine: Optional[AutonomousMixingEngine] = None
dsp_chain: Optional[StudioDSPChain] = None

# Mixing state
mixing_state = {
    "enabled": True,
    "mode": "auto",  # auto, manual, bypass
    "current_genre": "Unknown",
    "confidence": 0.0,
    "gate_reduction": 0.0,
    "comp_reduction": 0.0,
    "limiter_reduction": 0.0,
    "headroom_db": 0.0,
    "manual_params": None,
    "processing_time_ms": 0.0
}
mixing_lock = threading.Lock()

# Process start time for uptime
START_TIME = datetime.utcnow()

@app.route("/healthz")
def healthz():
    """Kubernetes-style health check endpoint"""
    return {
        "ok": True,
        "pid": os.getpid(),
        "uptime_seconds": (datetime.utcnow() - START_TIME).total_seconds(),
        "timestamp": datetime.utcnow().isoformat(),
        "mixing_enabled": mixing_state["enabled"]
    }

@app.route("/api/status")
def api_status():
    """Detailed status endpoint with mixing info"""
    
    # Get audio processor stats
    audio_stats = {}
    if audio_processor:
        audio_stats = audio_processor.get_stats()
    
    # Get system metrics
    process = psutil.Process()
    
    with mixing_lock:
        mixing_info = mixing_state.copy()
    
    return jsonify({
        "status": "healthy",
        "audio": audio_stats,
        "mixing": mixing_info,
        "system": {
            "cpu_percent": process.cpu_percent(),
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "threads": process.num_threads(),
            "connections": len(process.connections()),
        },
        "uptime": {
            "seconds": (datetime.utcnow() - START_TIME).total_seconds(),
            "since": START_TIME.isoformat()
        }
    })

@app.route("/api/mixing/enable", methods=["POST"])
def enable_mixing():
    """Enable/disable AI mixing"""
    data = request.get_json() or {}
    enabled = data.get("enabled", True)
    
    with mixing_lock:
        mixing_state["enabled"] = enabled
    
    logger.info(f"Mixing {'enabled' if enabled else 'disabled'}")
    return jsonify({"enabled": enabled})

@app.route("/api/mixing/mode", methods=["POST"])
def set_mixing_mode():
    """Set mixing mode: auto, manual, bypass"""
    data = request.get_json() or {}
    mode = data.get("mode", "auto")
    
    if mode not in ["auto", "manual", "bypass"]:
        return jsonify({"error": "Invalid mode"}), 400
    
    with mixing_lock:
        mixing_state["mode"] = mode
        if mode == "manual":
            # Parse manual parameters
            mixing_state["manual_params"] = {
                "gate": GateParams(**data.get("gate", {})),
                "compressor": CompressorParams(**data.get("compressor", {})),
                "eq_bands": [EQBand(**b) for b in data.get("eq_bands", [])],
                "limiter": LimiterParams(**data.get("limiter", {}))
            }
    
    logger.info(f"Mixing mode set to {mode}")
    return jsonify({"mode": mode})

@app.route("/api/mixing/profiles")
def get_profiles():
    """Get available mixing profiles"""
    if not mixing_engine:
        return jsonify({"error": "Mixing engine not initialized"}), 500
    
    profiles = {}
    for genre in ["Speech", "Rock", "Jazz", "Electronic", "Classical"]:
        profile = mixing_engine.profiles.get(genre)
        if profile:
            profiles[genre] = {
                "gate": {
                    "threshold_db": profile["gate"].threshold_db,
                    "ratio": profile["gate"].ratio,
                    "attack_ms": profile["gate"].attack_ms,
                    "release_ms": profile["gate"].release_ms
                },
                "compressor": {
                    "threshold_db": profile["compressor"].threshold_db,
                    "ratio": profile["compressor"].ratio,
                    "attack_ms": profile["compressor"].attack_ms,
                    "release_ms": profile["compressor"].release_ms,
                    "knee_db": profile["compressor"].knee_db,
                    "makeup_gain_db": profile["compressor"].makeup_gain_db
                },
                "eq_bands": [
                    {
                        "freq_hz": band.freq_hz,
                        "gain_db": band.gain_db,
                        "q": band.q,
                        "type": band.type
                    } for band in profile["eq_bands"]
                ],
                "limiter": {
                    "ceiling_db": profile["limiter"].ceiling_db,
                    "release_ms": profile["limiter"].release_ms,
                    "lookahead_ms": profile["limiter"].lookahead_ms
                }
            }
    
    return jsonify(profiles)

@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Enhanced analyze endpoint with mixing info"""
    snapshot = audio_state.get_snapshot()
    
    with mixing_lock:
        mixing_info = mixing_state.copy()
    
    return jsonify({
        "rms_db": snapshot["rms_db"],
        "peak_db": snapshot["peak_db"],
        "dominant_frequency_hz": snapshot["peak_hz"],
        "classification": snapshot["classification"],
        "spectrum_64_bands": snapshot["spectrum"],
        "mixing": {
            "genre": mixing_info["current_genre"],
            "confidence": mixing_info["confidence"],
            "gate_reduction_db": mixing_info["gate_reduction"],
            "comp_reduction_db": mixing_info["comp_reduction"],
            "limiter_reduction_db": mixing_info["limiter_reduction"],
            "headroom_db": mixing_info["headroom_db"]
        }
    })

@app.route("/")
def index():
    """Enhanced HTML page with mixing controls"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AG06 AI Mixing Studio</title>
        <style>
            body { 
                font-family: 'SF Mono', monospace; 
                background: linear-gradient(135deg, #1a1a2e, #0f0f1e); 
                color: #e0e0e0; 
                padding: 20px; 
                margin: 0;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            h1 { 
                color: #00ff88; 
                text-shadow: 0 0 10px rgba(0,255,136,0.5);
                margin-bottom: 30px;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .card {
                background: rgba(255,255,255,0.05);
                border: 1px solid rgba(0,255,136,0.3);
                border-radius: 8px;
                padding: 15px;
                backdrop-filter: blur(10px);
            }
            .card h2 {
                color: #00ff88;
                margin-top: 0;
                font-size: 1.2em;
            }
            .metric { 
                margin: 10px 0; 
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .value { 
                color: #00ffff; 
                font-weight: bold;
                font-size: 1.1em;
            }
            .reduction {
                color: #ff6b6b;
            }
            .genre {
                color: #ffd93d;
                font-size: 1.3em;
            }
            #spectrum { 
                width: 100%;
                height: 150px; 
                background: #000; 
                border-radius: 4px;
                margin: 20px 0; 
            }
            .controls {
                display: flex;
                gap: 10px;
                margin: 20px 0;
            }
            button {
                background: rgba(0,255,136,0.2);
                border: 1px solid #00ff88;
                color: #00ff88;
                padding: 10px 20px;
                border-radius: 4px;
                cursor: pointer;
                font-family: inherit;
                transition: all 0.3s;
            }
            button:hover {
                background: rgba(0,255,136,0.3);
                box-shadow: 0 0 10px rgba(0,255,136,0.5);
            }
            button.active {
                background: #00ff88;
                color: #1a1a2e;
            }
            .meter-bar {
                height: 8px;
                background: #333;
                border-radius: 4px;
                overflow: hidden;
                margin-top: 5px;
            }
            .meter-fill {
                height: 100%;
                background: linear-gradient(90deg, #00ff88, #00ffff);
                transition: width 0.1s;
            }
            .reduction-meter .meter-fill {
                background: linear-gradient(90deg, #ff6b6b, #ff3838);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎚️ AG06 AI Mixing Studio</h1>
            
            <div class="controls">
                <button id="btn-auto" class="active" onclick="setMode('auto')">Auto Mix</button>
                <button id="btn-manual" onclick="setMode('manual')">Manual</button>
                <button id="btn-bypass" onclick="setMode('bypass')">Bypass</button>
                <button id="btn-toggle" onclick="toggleMixing()">Disable Mixing</button>
            </div>
            
            <div class="grid">
                <div class="card">
                    <h2>Audio Levels</h2>
                    <div class="metric">
                        <span>RMS:</span>
                        <span id="rms" class="value">--</span>
                    </div>
                    <div class="meter-bar">
                        <div id="rms-meter" class="meter-fill" style="width: 0%"></div>
                    </div>
                    <div class="metric">
                        <span>Peak:</span>
                        <span id="peak" class="value">--</span>
                    </div>
                    <div class="meter-bar">
                        <div id="peak-meter" class="meter-fill" style="width: 0%"></div>
                    </div>
                    <div class="metric">
                        <span>Frequency:</span>
                        <span id="freq" class="value">--</span>
                    </div>
                    <div class="metric">
                        <span>Headroom:</span>
                        <span id="headroom" class="value">--</span>
                    </div>
                </div>
                
                <div class="card">
                    <h2>AI Detection</h2>
                    <div class="metric">
                        <span>Genre:</span>
                        <span id="genre" class="genre">--</span>
                    </div>
                    <div class="metric">
                        <span>Confidence:</span>
                        <span id="confidence" class="value">--</span>
                    </div>
                    <div class="metric">
                        <span>Processing:</span>
                        <span id="processing" class="value">--</span>
                    </div>
                </div>
                
                <div class="card">
                    <h2>DSP Processing</h2>
                    <div class="metric">
                        <span>Gate:</span>
                        <span id="gate" class="reduction">--</span>
                    </div>
                    <div class="meter-bar reduction-meter">
                        <div id="gate-meter" class="meter-fill" style="width: 0%"></div>
                    </div>
                    <div class="metric">
                        <span>Compressor:</span>
                        <span id="comp" class="reduction">--</span>
                    </div>
                    <div class="meter-bar reduction-meter">
                        <div id="comp-meter" class="meter-fill" style="width: 0%"></div>
                    </div>
                    <div class="metric">
                        <span>Limiter:</span>
                        <span id="limiter" class="reduction">--</span>
                    </div>
                    <div class="meter-bar reduction-meter">
                        <div id="limiter-meter" class="meter-fill" style="width: 0%"></div>
                    </div>
                </div>
            </div>
            
            <canvas id="spectrum"></canvas>
        </div>
        
        <script>
            const es = new EventSource('/api/stream');
            const canvas = document.getElementById('spectrum');
            const ctx = canvas.getContext('2d');
            
            // Resize canvas
            function resizeCanvas() {
                canvas.width = canvas.offsetWidth;
                canvas.height = 150;
            }
            resizeCanvas();
            window.addEventListener('resize', resizeCanvas);
            
            let mixingEnabled = true;
            let currentMode = 'auto';
            
            es.onmessage = (e) => {
                const data = JSON.parse(e.data);
                
                // Update levels
                document.getElementById('rms').textContent = data.rms_db.toFixed(1) + ' dB';
                document.getElementById('peak').textContent = data.peak_db.toFixed(1) + ' dB';
                document.getElementById('freq').textContent = data.peak_hz.toFixed(0) + ' Hz';
                
                // Update meters
                const rmsPercent = Math.max(0, Math.min(100, (data.rms_db + 60) * 1.67));
                const peakPercent = Math.max(0, Math.min(100, (data.peak_db + 60) * 1.67));
                document.getElementById('rms-meter').style.width = rmsPercent + '%';
                document.getElementById('peak-meter').style.width = peakPercent + '%';
                
                // Update mixing info if available
                if (data.mixing) {
                    document.getElementById('genre').textContent = data.mixing.genre || 'Unknown';
                    document.getElementById('confidence').textContent = 
                        ((data.mixing.confidence || 0) * 100).toFixed(0) + '%';
                    document.getElementById('headroom').textContent = 
                        (data.mixing.headroom_db || 0).toFixed(1) + ' dB';
                    
                    // Update reduction meters
                    document.getElementById('gate').textContent = 
                        (data.mixing.gate_reduction_db || 0).toFixed(1) + ' dB';
                    document.getElementById('comp').textContent = 
                        (data.mixing.comp_reduction_db || 0).toFixed(1) + ' dB';
                    document.getElementById('limiter').textContent = 
                        (data.mixing.limiter_reduction_db || 0).toFixed(1) + ' dB';
                    
                    const gatePercent = Math.min(100, Math.abs(data.mixing.gate_reduction_db || 0) * 2);
                    const compPercent = Math.min(100, Math.abs(data.mixing.comp_reduction_db || 0) * 5);
                    const limiterPercent = Math.min(100, Math.abs(data.mixing.limiter_reduction_db || 0) * 10);
                    
                    document.getElementById('gate-meter').style.width = gatePercent + '%';
                    document.getElementById('comp-meter').style.width = compPercent + '%';
                    document.getElementById('limiter-meter').style.width = limiterPercent + '%';
                }
                
                if (data.processing_ms) {
                    document.getElementById('processing').textContent = 
                        data.processing_ms.toFixed(1) + ' ms';
                }
                
                // Draw spectrum
                ctx.fillStyle = '#000';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                if (data.spectrum && data.spectrum.length > 0) {
                    const barWidth = canvas.width / data.spectrum.length;
                    const gradient = ctx.createLinearGradient(0, canvas.height, 0, 0);
                    gradient.addColorStop(0, '#00ff88');
                    gradient.addColorStop(0.5, '#00ffff');
                    gradient.addColorStop(1, '#ff00ff');
                    ctx.fillStyle = gradient;
                    
                    data.spectrum.forEach((val, i) => {
                        const height = Math.max(0, (val + 60) * 2.5);
                        ctx.fillRect(i * barWidth, canvas.height - height, barWidth - 1, height);
                    });
                }
            };
            
            function setMode(mode) {
                currentMode = mode;
                ['auto', 'manual', 'bypass'].forEach(m => {
                    const btn = document.getElementById('btn-' + m);
                    if (m === mode) {
                        btn.classList.add('active');
                    } else {
                        btn.classList.remove('active');
                    }
                });
                
                fetch('/api/mixing/mode', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({mode: mode})
                });
            }
            
            function toggleMixing() {
                mixingEnabled = !mixingEnabled;
                const btn = document.getElementById('btn-toggle');
                btn.textContent = mixingEnabled ? 'Disable Mixing' : 'Enable Mixing';
                
                fetch('/api/mixing/enable', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({enabled: mixingEnabled})
                });
            }
            
            es.onerror = (e) => {
                console.error('SSE error:', e);
            };
        </script>
    </body>
    </html>
    """

def process_audio_with_ai(audio_data: np.ndarray) -> Dict[str, Any]:
    """Process audio through AI mixing engine"""
    if not mixing_engine or not dsp_chain:
        return {}
    
    start_time = time.perf_counter()
    
    with mixing_lock:
        if not mixing_state["enabled"] or mixing_state["mode"] == "bypass":
            # Bypass processing
            return {
                "processed": audio_data,
                "genre": "Bypass",
                "confidence": 0.0,
                "gate_reduction": 0.0,
                "comp_reduction": 0.0,
                "limiter_reduction": 0.0,
                "headroom_db": 0.0,
                "processing_ms": 0.0
            }
        
        if mixing_state["mode"] == "manual" and mixing_state["manual_params"]:
            # Manual mode with user parameters
            params = mixing_state["manual_params"]
            processed, metrics = dsp_chain.process(
                audio_data,
                params["gate"],
                params["compressor"],
                params["eq_bands"],
                params["limiter"]
            )
            genre = "Manual"
            confidence = 1.0
        else:
            # Auto mode with AI
            processed, ai_results = mixing_engine.process(audio_data)
            metrics = ai_results.get("metrics", {})
            genre = ai_results.get("genre", "Unknown")
            confidence = ai_results.get("confidence", 0.0)
    
    processing_time = (time.perf_counter() - start_time) * 1000
    
    # Update state
    with mixing_lock:
        mixing_state["current_genre"] = genre
        mixing_state["confidence"] = confidence
        mixing_state["gate_reduction"] = metrics.get("gate_reduction_db", 0.0)
        mixing_state["comp_reduction"] = metrics.get("comp_reduction_db", 0.0)
        mixing_state["limiter_reduction"] = metrics.get("limiter_reduction_db", 0.0)
        mixing_state["headroom_db"] = metrics.get("headroom_db", 0.0)
        mixing_state["processing_time_ms"] = processing_time
    
    return {
        "processed": processed,
        "genre": genre,
        "confidence": confidence,
        "gate_reduction": metrics.get("gate_reduction_db", 0.0),
        "comp_reduction": metrics.get("comp_reduction_db", 0.0),
        "limiter_reduction": metrics.get("limiter_reduction_db", 0.0),
        "headroom_db": metrics.get("headroom_db", 0.0),
        "processing_ms": processing_time
    }

def shutdown_handler(signum, frame):
    """Graceful shutdown handler"""
    logger.info(f"Received signal {signum}, shutting down...")
    if audio_processor:
        audio_processor.stop()
    sys.exit(0)

def initialize_audio():
    """Initialize audio processor with AI mixing"""
    global audio_processor, mixing_engine, dsp_chain
    
    # Get config from environment
    config = AudioConfig(
        rate=int(os.getenv("AUDIO_RATE", "44100")),
        block_size=int(os.getenv("AUDIO_BLOCK", "512")),
        channels=1
    )
    
    # Initialize AI components
    sample_rate = config.rate
    mixing_engine = AutonomousMixingEngine(sample_rate)
    dsp_chain = StudioDSPChain(sample_rate)
    logger.info("AI mixing engine initialized")
    
    # Create processor
    audio_processor = RealTimeAudioProcessor(config)
    
    # Bridge to SSE state with AI processing
    def update_state(results):
        # Get raw audio if available
        if "audio_buffer" in results:
            audio = results["audio_buffer"]
            ai_results = process_audio_with_ai(audio)
            
            # Merge AI results
            results.update({
                "mixing": {
                    "genre": ai_results.get("genre"),
                    "confidence": ai_results.get("confidence"),
                    "gate_reduction_db": ai_results.get("gate_reduction"),
                    "comp_reduction_db": ai_results.get("comp_reduction"),
                    "limiter_reduction_db": ai_results.get("limiter_reduction"),
                    "headroom_db": ai_results.get("headroom_db")
                },
                "processing_ms": ai_results.get("processing_ms", 0.0)
            })
        
        audio_state.update(**results)
        if results.get("drops", 0) > 0:
            logger.warning(f"Audio drops detected: {results['drops']}")
    
    # Start processing
    audio_processor.start(update_state)
    logger.info("Audio processor started with AI mixing")

# Initialize on import (for Gunicorn)
if __name__ != "__main__":
    initialize_audio()

# Development server (use Gunicorn in production!)
if __name__ == "__main__":
    # Register shutdown handlers
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    # Initialize audio
    initialize_audio()
    
    # Run dev server
    app.run(host="0.0.0.0", port=5001, debug=False)