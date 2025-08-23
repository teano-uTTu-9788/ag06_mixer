#!/usr/bin/env python3
"""
Optimized AG06 Audio Processing Flask API
Integrates real-time audio processing with AG06 hardware
"""

from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO, emit
import sounddevice as sd
import numpy as np
from scipy import signal
from collections import deque
import threading
import time
import json
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ag06_audio_key'
socketio = SocketIO(app, cors_allowed_origins="*")

class RealTimeAG06Processor:
    def __init__(self):
        self.is_running = False
        self.sample_rate = 48000
        self.block_size = 256
        self.spectrum_bands = 64
        self.buffer = deque(maxlen=4096)
        
        # Detect AG06 device
        self.device_index = self.detect_ag06_device()
        
        # Initialize frequency bands (64 bands, logarithmic spacing)
        self.freq_bands = np.logspace(np.log10(20), np.log10(20000), self.spectrum_bands)
        
        # Real-time data storage
        self.current_data = {
            'spectrum': [0] * self.spectrum_bands,
            'level_db': -60,
            'classification': 'ambient',
            'peak_frequency': 0,
            'timestamp': time.time()
        }
        
    def detect_ag06_device(self):
        """Detect AG06 audio device"""
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                device_name = device['name'].lower()
                if 'ag06' in device_name or 'ag03' in device_name or 'yamaha' in device_name:
                    print(f'✅ Found AG06: {device["name"]} (Device {i})')
                    return i
            print('❌ AG06 not found, using default input device')
            return None
        except Exception as e:
            print(f'❌ Device detection error: {e}')
            return None
    
    def audio_callback(self, indata, frames, time, status):
        """Real-time audio processing callback"""
        if status:
            print(f'Audio status: {status}')
        
        try:
            # Convert to mono
            mono_data = np.mean(indata, axis=1) if indata.shape[1] > 1 else indata[:, 0]
            self.buffer.extend(mono_data)
            
            # Process if buffer has enough data
            if len(self.buffer) >= self.block_size:
                self.process_audio_block()
                
        except Exception as e:
            print(f'Audio callback error: {e}')
    
    def process_audio_block(self):
        """Process audio block with spectrum analysis"""
        try:
            # Get latest block from buffer
            audio_block = np.array(list(self.buffer)[-self.block_size:])
            
            # Apply Hann window (Google best practice)
            windowed = audio_block * signal.windows.hann(len(audio_block))
            
            # FFT analysis
            fft = np.fft.fft(windowed, n=2048)
            magnitude = np.abs(fft[:1024])
            
            # Convert to frequency bands
            freqs = np.fft.fftfreq(2048, 1/self.sample_rate)[:1024]
            band_values = []
            
            for i in range(self.spectrum_bands):
                if i < len(self.freq_bands) - 1:
                    band_mask = (freqs >= self.freq_bands[i]) & (freqs < self.freq_bands[i+1])
                    band_energy = np.sum(magnitude[band_mask]) if np.any(band_mask) else 0
                    band_values.append(band_energy)
                else:
                    band_values.append(0)
            
            # Normalize to 0-100 range and convert to Python float
            if max(band_values) > 0:
                normalized_bands = [float((val/max(band_values)) * 100) for val in band_values]
            else:
                normalized_bands = [0.0] * self.spectrum_bands
            
            # Find peak frequency
            peak_idx = np.argmax(magnitude)
            peak_freq = freqs[peak_idx] if peak_idx < len(freqs) else 0
            
            # Music vs Voice classification (Spotify-inspired algorithm)
            low_freq_energy = sum(normalized_bands[:8])     # 20-200 Hz (bass)
            mid_freq_energy = sum(normalized_bands[8:32])   # 200-2000 Hz (vocals)
            high_freq_energy = sum(normalized_bands[32:])   # 2000+ Hz (harmonics)
            
            # Classification logic
            total_energy = sum(normalized_bands)
            if total_energy < 1:
                classification = 'silent'
            elif mid_freq_energy > max(low_freq_energy, high_freq_energy) and 80 <= peak_freq <= 300:
                classification = 'voice'
            elif high_freq_energy > 20 and total_energy > 10:
                classification = 'music'
            else:
                classification = 'ambient'
            
            # Calculate RMS level in dB
            rms_level = np.sqrt(np.mean(audio_block**2))
            level_db = 20 * np.log10(max(rms_level, 1e-10)) + 60  # Adjust for display
            level_db = max(-60, min(level_db, 0))  # Clamp to -60dB to 0dB range
            
            # Update current data - ensure JSON serializable types
            self.current_data = {
                'spectrum': normalized_bands,
                'level_db': float(level_db),
                'classification': classification,
                'peak_frequency': float(abs(peak_freq)),
                'timestamp': float(time.time())
            }
            
            # Emit real-time data via WebSocket
            socketio.emit('audio_data', self.current_data)
            
        except Exception as e:
            print(f'Processing error: {e}')
    
    def start_monitoring(self):
        """Start real-time audio monitoring"""
        if self.is_running:
            return
            
        try:
            self.stream = sd.InputStream(
                device=self.device_index,
                channels=2,
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                callback=self.audio_callback
            )
            
            self.stream.start()
            self.is_running = True
            print('✅ Real-time audio monitoring started')
            
        except Exception as e:
            print(f'❌ Failed to start audio monitoring: {e}')
    
    def stop_monitoring(self):
        """Stop audio monitoring"""
        if hasattr(self, 'stream') and self.is_running:
            self.stream.stop()
            self.stream.close()
            self.is_running = False
            print('✅ Audio monitoring stopped')

# Initialize global processor
processor = RealTimeAG06Processor()

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')

@app.route('/api/spectrum')
def get_spectrum():
    """Get current spectrum data"""
    return jsonify(processor.current_data)

@app.route('/api/start')
def start_monitoring():
    """Start audio monitoring"""
    processor.start_monitoring()
    return jsonify({'status': 'started', 'message': 'Real-time monitoring active'})

@app.route('/api/stop')
def stop_monitoring():
    """Stop audio monitoring"""
    processor.stop_monitoring()
    return jsonify({'status': 'stopped', 'message': 'Monitoring stopped'})

@app.route('/api/status')
def get_status():
    """Get system status"""
    return jsonify({
        'monitoring': processor.is_running,
        'device_detected': processor.device_index is not None,
        'sample_rate': processor.sample_rate,
        'bands': processor.spectrum_bands,
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    print('Client connected to real-time audio stream')
    emit('status', {'message': 'Connected to AG06 real-time processor'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    print('Client disconnected from real-time audio stream')

if __name__ == '__main__':
    print('🎛️ OPTIMIZED AG06 AUDIO PROCESSOR STARTING')
    print('=' * 50)
    
    # Auto-start monitoring
    processor.start_monitoring()
    
    # Start Flask app
    socketio.run(app, host='0.0.0.0', port=5001, debug=False, allow_unsafe_werkzeug=True)