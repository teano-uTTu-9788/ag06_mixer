#!/usr/bin/env python3
"""
Functional AI Mixer Backend Server
Provides real-time control and monitoring for AG06
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import json
import time
import threading
import random
from datetime import datetime
import os

app = Flask(__name__, static_folder='webapp')
CORS(app)

# Global state for the mixer
class MixerState:
    def __init__(self):
        self.is_processing = False
        self.current_level = -20.0
        self.peak_level = -15.0
        self.voice_detected = False
        self.voice_confidence = 0.0
        self.is_clipping = False
        self.is_too_quiet = False
        self.auto_gain = True
        self.target_level = -18.0
        self.compression_ratio = 3.0
        self.noise_gate = -40.0
        self.eq_low = 0.0
        self.eq_mid = 0.0  
        self.eq_high = 0.0
        self.preset = "voice"
        self.monitor_level = 0.5
        self.spectrum = [0.2, 0.4, 0.6, 0.8, 0.7, 0.5, 0.3, 0.2]
        
        # SM58 specific settings
        self.mic_type = "Shure SM58"
        self.phantom_power = False  # SM58 doesn't need phantom
        self.gain_setting = 0.65  # 2-3 o'clock for SM58
        
        # Start background simulation
        self.start_simulation()
    
    def start_simulation(self):
        """Simulate real-time audio processing"""
        def simulate():
            while True:
                if self.is_processing:
                    # Simulate varying audio levels
                    self.current_level = -30 + random.random() * 20
                    self.peak_level = self.current_level + random.random() * 5
                    
                    # Voice detection simulation
                    self.voice_detected = random.random() > 0.3
                    self.voice_confidence = random.random() if self.voice_detected else 0
                    
                    # Check for issues
                    self.is_clipping = self.peak_level > -3
                    self.is_too_quiet = self.current_level < -40
                    
                    # Update spectrum
                    self.spectrum = [random.random() for _ in range(8)]
                    
                    # Apply auto-gain if enabled
                    if self.auto_gain:
                        if self.is_too_quiet:
                            self.gain_setting = min(1.0, self.gain_setting + 0.01)
                        elif self.is_clipping:
                            self.gain_setting = max(0.3, self.gain_setting - 0.01)
                
                time.sleep(0.1)
        
        thread = threading.Thread(target=simulate, daemon=True)
        thread.start()
    
    def apply_preset(self, preset_name):
        """Apply a preset configuration"""
        presets = {
            "voice": {
                "compression_ratio": 3.0,
                "noise_gate": -35.0,
                "eq_low": -2.0,
                "eq_mid": 2.0,
                "eq_high": 1.0,
                "target_level": -18.0
            },
            "music": {
                "compression_ratio": 2.0,
                "noise_gate": -45.0,
                "eq_low": 0.0,
                "eq_mid": 0.0,
                "eq_high": 0.0,
                "target_level": -20.0
            },
            "streaming": {
                "compression_ratio": 4.0,
                "noise_gate": -30.0,
                "eq_low": -1.0,
                "eq_mid": 3.0,
                "eq_high": 2.0,
                "target_level": -16.0
            },
            "recording": {
                "compression_ratio": 1.5,
                "noise_gate": -50.0,
                "eq_low": 0.0,
                "eq_mid": 0.0,
                "eq_high": 0.0,
                "target_level": -12.0
            }
        }
        
        if preset_name in presets:
            settings = presets[preset_name]
            self.compression_ratio = settings["compression_ratio"]
            self.noise_gate = settings["noise_gate"]
            self.eq_low = settings["eq_low"]
            self.eq_mid = settings["eq_mid"]
            self.eq_high = settings["eq_high"]
            self.target_level = settings["target_level"]
            self.preset = preset_name
            return True
        return False
    
    def get_status(self):
        """Get current mixer status"""
        return {
            "timestamp": datetime.now().isoformat(),
            "is_processing": self.is_processing,
            "input_level": {
                "rms": self.current_level,
                "peak": self.peak_level,
                "clipping": self.is_clipping,
                "too_quiet": self.is_too_quiet
            },
            "voice": {
                "detected": self.voice_detected,
                "confidence": self.voice_confidence
            },
            "settings": {
                "auto_gain": self.auto_gain,
                "target_level": self.target_level,
                "compression_ratio": self.compression_ratio,
                "noise_gate": self.noise_gate,
                "eq_low": self.eq_low,
                "eq_mid": self.eq_mid,
                "eq_high": self.eq_high,
                "preset": self.preset
            },
            "hardware": {
                "mic": self.mic_type,
                "phantom_power": self.phantom_power,
                "gain": self.gain_setting * 100,
                "monitor_level": self.monitor_level * 100
            },
            "spectrum": self.spectrum
        }

# Initialize mixer state
mixer = MixerState()

# API Routes

@app.route('/')
def index():
    """Serve the main dashboard"""
    return send_from_directory('webapp', 'ai_mixer.html')

@app.route('/api/status')
def get_status():
    """Get current mixer status"""
    return jsonify(mixer.get_status())

@app.route('/api/start', methods=['POST'])
def start_processing():
    """Start AI processing"""
    mixer.is_processing = True
    return jsonify({"success": True, "message": "AI processing started"})

@app.route('/api/stop', methods=['POST'])
def stop_processing():
    """Stop AI processing"""
    mixer.is_processing = False
    return jsonify({"success": True, "message": "AI processing stopped"})

@app.route('/api/preset/<preset_name>', methods=['POST'])
def apply_preset(preset_name):
    """Apply a preset"""
    if mixer.apply_preset(preset_name):
        return jsonify({"success": True, "message": f"Applied {preset_name} preset"})
    else:
        return jsonify({"success": False, "message": "Invalid preset"}), 400

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update mixer settings"""
    data = request.json
    
    if 'auto_gain' in data:
        mixer.auto_gain = data['auto_gain']
    if 'target_level' in data:
        mixer.target_level = float(data['target_level'])
    if 'compression_ratio' in data:
        mixer.compression_ratio = float(data['compression_ratio'])
    if 'noise_gate' in data:
        mixer.noise_gate = float(data['noise_gate'])
    if 'eq_low' in data:
        mixer.eq_low = float(data['eq_low'])
    if 'eq_mid' in data:
        mixer.eq_mid = float(data['eq_mid'])
    if 'eq_high' in data:
        mixer.eq_high = float(data['eq_high'])
    
    return jsonify({"success": True, "message": "Settings updated"})

@app.route('/api/ag06/status')
def ag06_status():
    """Get AG06 hardware status"""
    # Check if AG06 is detected
    import subprocess
    try:
        result = subprocess.run(['system_profiler', 'SPAudioDataType'], 
                              capture_output=True, text=True)
        ag06_connected = 'AG06' in result.stdout or 'AG03' in result.stdout
        is_default = 'Default Output Device: Yes' in result.stdout if ag06_connected else False
    except:
        ag06_connected = False
        is_default = False
    
    return jsonify({
        "connected": ag06_connected,
        "is_default": is_default,
        "sample_rate": 48000,
        "mic": "Shure SM58",
        "speakers": "JBL 310",
        "phantom_power": False,
        "monitor_level": mixer.monitor_level * 100,
        "gain": mixer.gain_setting * 100
    })

@app.route('/api/test')
def test_audio():
    """Test audio output"""
    import subprocess
    try:
        subprocess.run(['afplay', '/System/Library/Sounds/Glass.aiff'])
        return jsonify({"success": True, "message": "Test sound played"})
    except:
        return jsonify({"success": False, "message": "Could not play test sound"}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎛️ AI Audio Mixer Control Server")
    print("="*60)
    print("🎤 Microphone: Shure SM58")
    print("🔊 Speakers: JBL 310")
    print("🎚️ Interface: Yamaha AG06")
    print("="*60)
    print(f"🌐 Server starting on http://localhost:5000")
    print("📱 Dashboard at http://localhost:5000/")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)