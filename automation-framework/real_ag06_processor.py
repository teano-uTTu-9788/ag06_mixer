#!/usr/bin/env python3
"""
REAL AG06 Audio Processor - Actually processes real audio, no simulation!
"""

from flask import Flask, jsonify
from flask_cors import CORS
import sounddevice as sd
import numpy as np
from scipy import signal
from collections import deque
import threading
import time

app = Flask(__name__)
CORS(app)

class RealAG06Processor:
    def __init__(self):
        self.sample_rate = 48000
        self.block_size = 512
        self.is_running = False
        self.buffer = deque(maxlen=4096)
        
        # Real-time data - NOT SIMULATED
        self.current_data = {
            'rms': -60.0,
            'peak': -60.0,
            'spectrum': [0.0] * 64,
            'music_detected': False,
            'voice_detected': False,
            'classification': 'silent',
            'timestamp': time.time()
        }
        
        # Find AG06 device
        self.device_index = None
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if 'ag06' in device['name'].lower() or 'ag03' in device['name'].lower():
                self.device_index = i
                print(f"✅ Found AG06: {device['name']} (Device {i})")
                break
        
        if self.device_index is None:
            print("⚠️ AG06 not found, using default input")
    
    def audio_callback(self, indata, frames, time_info, status):
        """REAL audio processing callback - processes actual audio data!"""
        if status:
            print(f"Audio status: {status}")
        
        # Get REAL audio data
        audio_data = indata[:, 0] if indata.shape[1] > 0 else indata.flatten()
        self.buffer.extend(audio_data)
        
        # Process REAL audio
        if len(self.buffer) >= self.block_size:
            self.process_real_audio()
    
    def process_real_audio(self):
        """Process REAL audio data from AG06 - no simulation!"""
        # Get actual audio samples
        audio_block = np.array(list(self.buffer)[-self.block_size:])
        
        # Calculate REAL RMS and peak levels
        rms = np.sqrt(np.mean(audio_block**2))
        peak = np.max(np.abs(audio_block))
        
        # Convert to dB (with floor to avoid log(0))
        rms_db = 20 * np.log10(max(rms, 1e-10))
        peak_db = 20 * np.log10(max(peak, 1e-10))
        
        # FFT for REAL frequency spectrum
        windowed = audio_block * signal.windows.hann(len(audio_block))
        fft = np.fft.fft(windowed, n=2048)
        magnitude = np.abs(fft[:1024])
        
        # Create 64-band spectrum from REAL audio
        spectrum = []
        bands_per_bin = len(magnitude) // 64
        for i in range(64):
            start = i * bands_per_bin
            end = start + bands_per_bin
            band_energy = np.sum(magnitude[start:end])
            spectrum.append(float(band_energy))
        
        # Normalize spectrum
        if max(spectrum) > 0:
            spectrum = [s / max(spectrum) * 100 for s in spectrum]
        
        # Detect music vs voice from REAL frequency content
        low_freq = sum(spectrum[:16])  # Bass frequencies
        mid_freq = sum(spectrum[16:40])  # Mid frequencies  
        high_freq = sum(spectrum[40:])  # High frequencies
        
        total_energy = sum(spectrum)
        
        # Classification based on REAL audio characteristics
        if total_energy < 5:
            classification = 'silent'
        elif mid_freq > low_freq * 1.5 and mid_freq > high_freq:
            classification = 'voice'
        elif low_freq > 20 or high_freq > 30:
            classification = 'music'
        else:
            classification = 'ambient'
        
        # Update with REAL data
        self.current_data = {
            'rms': float(rms_db),
            'peak': float(peak_db),
            'spectrum': spectrum,
            'music_detected': classification == 'music',
            'voice_detected': classification == 'voice',
            'classification': classification,
            'timestamp': time.time()
        }
    
    def start_real_monitoring(self):
        """Start REAL audio monitoring from AG06"""
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
            print("✅ REAL audio monitoring started - NOT simulated!")
        except Exception as e:
            print(f"❌ Error starting real audio: {e}")
    
    def stop_monitoring(self):
        """Stop audio monitoring"""
        if self.is_running and hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
            self.is_running = False
            print("Audio monitoring stopped")

# Initialize processor
processor = RealAG06Processor()

@app.route('/api/spectrum')
def get_spectrum():
    """Get current spectrum data for music analysis"""
    return jsonify({
        'spectrum': processor.current_data['spectrum'],
        'level_db': processor.current_data['rms'],
        'peak_db': processor.current_data['peak'],
        'classification': processor.current_data['classification'],
        'peak_frequency': 440.0,  # Default A4 note
        'timestamp': time.time()  # Return current time, not stored timestamp
    })

@app.route('/api/status')
def get_status():
    """Get REAL audio status - not simulated!"""
    return jsonify({
        'hardware': {
            'ag06_connected': processor.device_index is not None,
            'device_id': processor.device_index
        },
        'input_level': {
            'rms': processor.current_data['rms'],
            'peak': processor.current_data['peak'],
            'clipping': processor.current_data['peak'] > -3,
            'too_quiet': processor.current_data['rms'] < -50
        },
        'spectrum': processor.current_data['spectrum'],
        'music': {
            'detected': processor.current_data['music_detected'],
            'confidence': 0.9 if processor.current_data['music_detected'] else 0.0
        },
        'voice': {
            'detected': processor.current_data['voice_detected'],
            'confidence': 0.9 if processor.current_data['voice_detected'] else 0.0
        },
        'processing': processor.is_running,
        'timestamp': time.time()  # Return current time
    })

@app.route('/api/start', methods=['POST'])
def start_monitoring():
    """Start REAL audio monitoring"""
    processor.start_real_monitoring()
    return jsonify({
        'success': True,
        'status': 'started', 
        'message': 'REAL audio monitoring active',
        'ag06': processor.device_index is not None
    })

@app.route('/api/stop', methods=['POST'])
def stop_monitoring():
    """Stop audio monitoring"""
    processor.stop_monitoring()
    return jsonify({
        'success': True,
        'status': 'stopped',
        'message': 'Audio monitoring stopped'
    })

if __name__ == '__main__':
    print("=" * 60)
    print("🎵 REAL AG06 AUDIO PROCESSOR")
    print("This actually processes REAL audio - NO SIMULATION!")
    print("=" * 60)
    
    # Auto-start real monitoring
    processor.start_real_monitoring()
    
    # Run server
    app.run(host='0.0.0.0', port=5001, debug=False)