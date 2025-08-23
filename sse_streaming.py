"""
Server-Sent Events (SSE) streaming for real-time audio metrics
Following big-tech patterns: simple one-way push for telemetry
"""

from flask import Blueprint, Response, stream_with_context
import json
import time
import threading
from typing import Dict, Any, Optional

sse_bp = Blueprint("sse", __name__)

class SharedAudioState:
    """Thread-safe shared state for audio metrics"""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._state = {
            "rms_db": -60.0,
            "peak_db": -60.0,
            "classification": "silence",
            "peak_hz": 0.0,
            "spectrum": [0.0] * 64,
            "timestamp": time.time()
        }
    
    def update(self, **kwargs):
        """Update state with thread safety"""
        with self._lock:
            self._state.update(kwargs)
            self._state["timestamp"] = time.time()
    
    def get_snapshot(self) -> Dict[str, Any]:
        """Get atomic snapshot of current state"""
        with self._lock:
            return self._state.copy()

# Global state instance
audio_state = SharedAudioState()

@sse_bp.route("/api/stream")
def stream():
    """SSE endpoint for real-time audio metrics streaming"""
    
    @stream_with_context
    def event_stream():
        while True:
            try:
                # Get current state snapshot
                snapshot = audio_state.get_snapshot()
                
                # Format as SSE message
                data = json.dumps({
                    "rms": snapshot["rms_db"],
                    "peak": snapshot["peak_db"],
                    "class": snapshot["classification"],
                    "peak_hz": snapshot["peak_hz"],
                    "spectrum": snapshot["spectrum"],
                    "ts": snapshot["timestamp"]
                })
                
                yield f"data: {data}\n\n"
                
                # Stream at ~10 Hz (configurable via env)
                time.sleep(0.1)
                
            except GeneratorExit:
                # Client disconnected
                break
            except Exception as e:
                # Log error but keep stream alive
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
                time.sleep(1)  # Back off on error
    
    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "Connection": "keep-alive"
        }
    )

@sse_bp.route("/api/stream/health")
def stream_health():
    """Health check for SSE streaming"""
    snapshot = audio_state.get_snapshot()
    age = time.time() - snapshot["timestamp"]
    
    return {
        "healthy": age < 5.0,  # Consider stale if > 5s old
        "age_seconds": age,
        "last_update": snapshot["timestamp"]
    }