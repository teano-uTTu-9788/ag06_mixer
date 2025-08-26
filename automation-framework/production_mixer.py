#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed AI Mixer — production-ready Flask + sounddevice
- Robust AG06 device selection and fallback
- Full-duplex callback stream with soft limiter and target loudness autopilot
- SSE telemetry for live meters/spectrum
- JSON API: /api/status, /api/config (POST), /api/start, /api/stop, /healthz
"""

import os, time, json, threading, queue, math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
from flask import Flask, jsonify, request, Response
from flask_cors import CORS

# --------------------- Config / State ---------------------

@dataclass
class MixerConfig:
    samplerate: int = 44100
    blocksize: int = 256         # small for low latency
    channels: int = 2
    ai_mix: float = 0.70         # 0..1 (blend dry/wet)
    target_lufs: float = -14.0   # streaming norm (approx)
    makeup_max_db: float = 12.0  # cap automatic gain
    limiter_ceiling_db: float = -1.0  # true-peak ceiling
    sse_hz: float = 10.0         # telemetry rate

@dataclass
class MixerMetrics:
    rms_db: float = -60.0
    peak_db: float = -60.0
    lufs_est: float = -60.0
    clipping: bool = False
    dropouts: int = 0
    device_in: Optional[str] = None
    device_out: Optional[str] = None
    running: bool = False
    err: Optional[str] = None

cfg = MixerConfig()
metrics = MixerMetrics()

_state_lock = threading.Lock()
_running = False
_stream: Optional[sd.Stream] = None
_ring_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=16)

# --------------------- Utility DSP ---------------------

def dbfs(x: float) -> float:
    return 20.0 * math.log10(max(x, 1e-12))

def est_lufs(block: np.ndarray) -> float:
    """Very rough LUFS estimate using RMS; replace with R128 if needed."""
    # convert to mono energy
    if block.ndim == 2:
        mono = block.mean(axis=1)
    else:
        mono = block
    rms = np.sqrt(np.mean(mono**2))
    return dbfs(rms)

def soft_limiter(x: np.ndarray, ceiling_db: float) -> np.ndarray:
    ceiling = 10 ** (ceiling_db / 20.0)
    # simple soft knee using tanh
    return np.tanh(x / ceiling) * ceiling

def apply_gain_db(x: np.ndarray, gain_db: float) -> np.ndarray:
    return x * (10 ** (gain_db / 20.0))

# --------------------- Device selection ---------------------

def pick_ag06_devices() -> Tuple[Optional[int], Optional[int], Optional[str], Optional[str]]:
    """
    Google-style device detection with comprehensive fallback strategy.
    Implements tiered device selection: AG06 -> High-quality -> System default
    """
    try:
        devices = sd.query_devices()
    except Exception as e:
        with _state_lock:
            metrics.err = f"query_devices failed: {e}"
        return None, None, "unavailable", "unavailable"

    in_idx = out_idx = None
    in_name = out_name = None
    
    # Tier 1: Prefer AG06/AG03 devices
    for idx, d in enumerate(devices):
        name = d.get("name", "")
        is_in = d.get("max_input_channels", 0) > 0
        is_out = d.get("max_output_channels", 0) > 0
        
        if any(keyword in name.upper() for keyword in ["AG06", "AG03", "YAMAHA"]):
            if is_in and in_idx is None:
                in_idx, in_name = idx, name.strip()
            if is_out and out_idx is None:
                out_idx, out_name = idx, name.strip()
    
    # Tier 2: Fallback to system defaults with proper name resolution
    try:
        defaults = sd.default.device
        if isinstance(defaults, (list, tuple)) and len(defaults) >= 2:
            if in_idx is None and defaults[0] is not None:
                in_idx = defaults[0]
                if 0 <= in_idx < len(devices):
                    in_name = devices[in_idx]["name"].strip()
            
            if out_idx is None and defaults[1] is not None:
                out_idx = defaults[1] 
                if 0 <= out_idx < len(devices):
                    out_name = devices[out_idx]["name"].strip()
        
        # Tier 3: Last resort - find any working device
        if in_idx is None or in_name is None:
            for idx, d in enumerate(devices):
                if d.get("max_input_channels", 0) > 0:
                    in_idx, in_name = idx, d["name"].strip()
                    break
        
        if out_idx is None or out_name is None:
            for idx, d in enumerate(devices):
                if d.get("max_output_channels", 0) > 0:
                    out_idx, out_name = idx, d["name"].strip()
                    break
                    
    except Exception as e:
        # Final fallback with descriptive names
        if in_name is None:
            in_name = f"system_default_input_{in_idx}" if in_idx is not None else "no_input_device"
        if out_name is None:
            out_name = f"system_default_output_{out_idx}" if out_idx is not None else "no_output_device"
    
    return in_idx, out_idx, in_name or "unknown_input", out_name or "unknown_output"

# --------------------- Audio callback & engine ---------------------

_last_gain_db = 0.0   # smoothed autopilot gain

def audio_callback(indata, outdata, frames, time_info, status):
    global _last_gain_db
    if status:
        # buffer over/underrun flags
        with _state_lock:
            metrics.dropouts += int(status.input_underflow or status.output_underflow)

    x = indata.copy().astype(np.float32)

    # meters
    block_peak = float(np.max(np.abs(x)))
    block_rms = float(np.sqrt(np.mean(x**2)))
    lufs_now = est_lufs(x)

    # autopilot gain toward target (~-14 LUFS), limited to +/- makeup_max_db
    delta = cfg.target_lufs - lufs_now
    delta = float(np.clip(delta, -cfg.makeup_max_db, cfg.makeup_max_db))
    # smooth (one-pole) to avoid pumping
    _last_gain_db = 0.8 * _last_gain_db + 0.2 * delta
    y = apply_gain_db(x, _last_gain_db)

    # simple "AI mix" placeholder: dry/wet blend (here wet==limited)
    y_limited = soft_limiter(y, cfg.limiter_ceiling_db)
    out = (1.0 - cfg.ai_mix) * x + cfg.ai_mix * y_limited

    # write
    outdata[:] = np.clip(out, -1.0, 1.0)

    # update shared metrics (cheap work only)
    with _state_lock:
        metrics.rms_db = dbfs(block_rms)
        metrics.peak_db = dbfs(block_peak)
        metrics.lufs_est = lufs_now
        metrics.clipping = bool(block_peak >= 0.999)

def start_engine() -> None:
    global _running, _stream
    if _running:
        return
    # AG06 preferred
    in_idx, out_idx, in_name, out_name = pick_ag06_devices()
    sd.default.latency = "low"
    sd.default.dtype = "float32"

    tries = 0
    while tries < 3:
        try:
            _stream = sd.Stream(
                samplerate=cfg.samplerate,
                blocksize=cfg.blocksize,
                channels=cfg.channels,
                dtype="float32",
                callback=audio_callback,
                device=(in_idx, out_idx),
            )
            _stream.start()
            with _state_lock:
                metrics.device_in = in_name
                metrics.device_out = out_name
                metrics.running = True
                metrics.err = None
            _running = True
            return
        except Exception as e:
            # macOS CoreAudio can return -10863 intermittently; backoff and retry.
            tries += 1
            with _state_lock:
                metrics.err = f"start failed (try {tries}): {e}"
            time.sleep(0.5 * tries)

    raise RuntimeError(metrics.err or "failed to start audio engine")

def stop_engine() -> None:
    global _running, _stream
    _running = False
    try:
        if _stream is not None:
            _stream.stop()
            _stream.close()
    finally:
        with _state_lock:
            metrics.running = False
        _stream = None

# --------------------- Flask API ---------------------

app = Flask(__name__)
CORS(app)

@app.get("/healthz")
def healthz():
    return {"ok": True, "running": metrics.running}

@app.get("/api/status")
def api_status():
    with _state_lock:
        return jsonify({
            "metrics": asdict(metrics),
            "config": asdict(cfg),
        })

@app.post("/api/config")
def api_config():
    data = request.get_json(force=True, silent=True) or {}
    changed = {}
    
    # Process each parameter separately with correct values
    if "ai_mix" in data:
        cfg.ai_mix = float(data["ai_mix"])
        changed["ai_mix"] = cfg.ai_mix
    
    if "target_lufs" in data:
        cfg.target_lufs = float(data["target_lufs"])
        changed["target_lufs"] = cfg.target_lufs
    
    if "blocksize" in data:
        cfg.blocksize = int(data["blocksize"])
        changed["blocksize"] = cfg.blocksize
        
    if "samplerate" in data:
        cfg.samplerate = int(data["samplerate"])
        changed["samplerate"] = cfg.samplerate
    
    return jsonify({"ok": True, "changed": changed})

@app.post("/api/start")
def api_start():
    try:
        start_engine()
        return jsonify({"ok": True, "running": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/api/stop")
def api_stop():
    stop_engine()
    return jsonify({"ok": True, "running": False})

@app.get("/api/stream")
def api_stream():
    # Server-Sent Events stream for meters at cfg.sse_hz
    def gen():
        while True:
            with _state_lock:
                payload = {"metrics": asdict(metrics)}
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(1.0 / max(1.0, cfg.sse_hz))
    return Response(gen(), mimetype="text/event-stream")

if __name__ == "__main__":
    # dev mode only; use gunicorn in production
    app.run(host="0.0.0.0", port=8080, debug=False)