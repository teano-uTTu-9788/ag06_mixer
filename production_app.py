"""
Production-ready AG06 Flask app with all improvements
Gunicorn-compatible, SSE streaming, health checks, proper error handling
"""

from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import os
import sys
import signal
import logging
from datetime import datetime
import psutil

# Import our modules
from sse_streaming import sse_bp, audio_state
from audio_rt_callback import RealTimeAudioProcessor, AudioConfig

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

# Global audio processor
audio_processor = None

# Process start time for uptime
START_TIME = datetime.utcnow()

@app.route("/healthz")
def healthz():
    """Kubernetes-style health check endpoint"""
    return {
        "ok": True,
        "pid": os.getpid(),
        "uptime_seconds": (datetime.utcnow() - START_TIME).total_seconds(),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.route("/api/status")
def api_status():
    """Detailed status endpoint"""
    
    # Get audio processor stats
    audio_stats = {}
    if audio_processor:
        audio_stats = audio_processor.get_stats()
    
    # Get system metrics
    process = psutil.Process()
    
    return jsonify({
        "status": "healthy",
        "audio": audio_stats,
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

@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Legacy analyze endpoint - returns latest state"""
    snapshot = audio_state.get_snapshot()
    
    return jsonify({
        "rms_db": snapshot["rms_db"],
        "peak_db": snapshot["peak_db"],
        "dominant_frequency_hz": snapshot["peak_hz"],
        "classification": snapshot["classification"],
        "spectrum_64_bands": snapshot["spectrum"]
    })

@app.route("/")
def index():
    """Simple HTML page with SSE client"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AG06 Real-Time Monitor</title>
        <style>
            body { font-family: monospace; background: #1a1a1a; color: #0f0; padding: 20px; }
            .metric { margin: 10px 0; }
            .value { color: #0ff; }
            #spectrum { height: 100px; background: #000; margin: 20px 0; }
        </style>
    </head>
    <body>
        <h1>AG06 Real-Time Audio Monitor</h1>
        <div class="metric">RMS: <span id="rms" class="value">--</span> dB</div>
        <div class="metric">Peak: <span id="peak" class="value">--</span> dB</div>
        <div class="metric">Frequency: <span id="freq" class="value">--</span> Hz</div>
        <div class="metric">Class: <span id="class" class="value">--</span></div>
        <canvas id="spectrum" width="800" height="100"></canvas>
        
        <script>
            const es = new EventSource('/api/stream');
            const canvas = document.getElementById('spectrum');
            const ctx = canvas.getContext('2d');
            
            es.onmessage = (e) => {
                const data = JSON.parse(e.data);
                document.getElementById('rms').textContent = data.rms.toFixed(1);
                document.getElementById('peak').textContent = data.peak.toFixed(1);
                document.getElementById('freq').textContent = data.peak_hz.toFixed(1);
                document.getElementById('class').textContent = data.class;
                
                // Draw spectrum
                ctx.fillStyle = '#000';
                ctx.fillRect(0, 0, 800, 100);
                ctx.fillStyle = '#0f0';
                const barWidth = 800 / data.spectrum.length;
                data.spectrum.forEach((val, i) => {
                    const height = Math.max(0, (val + 60) * 2);  // Scale for display
                    ctx.fillRect(i * barWidth, 100 - height, barWidth - 1, height);
                });
            };
            
            es.onerror = (e) => {
                console.error('SSE error:', e);
            };
        </script>
    </body>
    </html>
    """

def shutdown_handler(signum, frame):
    """Graceful shutdown handler"""
    logger.info(f"Received signal {signum}, shutting down...")
    if audio_processor:
        audio_processor.stop()
    sys.exit(0)

def initialize_audio():
    """Initialize audio processor"""
    global audio_processor
    
    # Get config from environment
    config = AudioConfig(
        rate=int(os.getenv("AUDIO_RATE", "44100")),
        block_size=int(os.getenv("AUDIO_BLOCK", "512")),
        channels=1
    )
    
    # Create processor
    audio_processor = RealTimeAudioProcessor(config)
    
    # Bridge to SSE state
    def update_state(results):
        audio_state.update(**results)
        if results["drops"] > 0:
            logger.warning(f"Audio drops detected: {results['drops']}")
    
    # Start processing
    audio_processor.start(update_state)
    logger.info("Audio processor started")

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