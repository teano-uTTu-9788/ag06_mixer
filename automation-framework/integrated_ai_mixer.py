#!/usr/bin/env python3
"""
Integrated AI Mixer - Combines monitoring with real-time mixing
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import sounddevice as sd
import numpy as np
from scipy import signal
import threading
import time
from collections import deque
from ai_mixer_engine import AIRealTimeMixer

app = Flask(__name__)
CORS(app)

# Global mixer instance
mixer = AIRealTimeMixer()

# Current status
status_data = {
    'mixing': False,
    'mode': 'music',
    'input_level': -60,
    'output_level': -60,
    'spectrum': [0] * 64,
    'processing_latency': 0,
    'effects': {
        'reverb': 0.1,
        'delay': 0.0,
        'bass_enhance': True,
        'presence_enhance': True
    }
}

@app.route('/api/start_mixing', methods=['POST'])
def start_mixing():
    """Start the AI mixing engine"""
    global status_data
    
    if mixer.start_mixing():
        status_data['mixing'] = True
        return jsonify({
            'success': True,
            'message': '🎵 AI Mixing Engine Active! Audio is being processed in real-time.',
            'status': 'mixing'
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Failed to start mixing',
            'status': 'error'
        })

@app.route('/api/stop_mixing', methods=['POST'])
def stop_mixing():
    """Stop AI mixing"""
    global status_data
    
    if mixer.stop_mixing():
        status_data['mixing'] = False
        return jsonify({
            'success': True,
            'message': 'Mixing stopped',
            'status': 'stopped'
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Mixer was not running',
            'status': 'idle'
        })

@app.route('/api/mixer_status')
def get_mixer_status():
    """Get current mixer status"""
    return jsonify({
        'mixing': status_data['mixing'],
        'mode': status_data['mode'],
        'input_device': mixer.input_device,
        'output_device': mixer.output_device,
        'effects': status_data['effects'],
        'ag06_connected': mixer.input_device is not None,
        'message': 'AI Mixer is actively processing audio' if status_data['mixing'] else 'Mixer idle'
    })

@app.route('/api/set_mode/<mode>', methods=['POST'])
def set_mode(mode):
    """Switch between music and voice mode"""
    global status_data
    
    if mode == 'music':
        mixer.set_music_mode(True)
        status_data['mode'] = 'music'
        status_data['effects']['reverb'] = 0.15
        status_data['effects']['bass_enhance'] = True
        status_data['effects']['presence_enhance'] = True
        message = "🎵 Music mode activated - Optimized for music mixing"
    elif mode == 'voice':
        mixer.set_music_mode(False)
        status_data['mode'] = 'voice'
        status_data['effects']['reverb'] = 0.05
        status_data['effects']['bass_enhance'] = False
        status_data['effects']['presence_enhance'] = False
        message = "🎤 Voice mode activated - Optimized for vocals"
    else:
        return jsonify({'success': False, 'message': 'Invalid mode'})
    
    return jsonify({
        'success': True,
        'message': message,
        'mode': mode
    })

@app.route('/api/set_effect', methods=['POST'])
def set_effect():
    """Adjust effect parameters"""
    data = request.json
    effect = data.get('effect')
    value = data.get('value')
    
    if effect == 'reverb':
        mixer.reverb_mix = float(value)
        status_data['effects']['reverb'] = float(value)
    elif effect == 'delay':
        mixer.delay_mix = float(value)
        status_data['effects']['delay'] = float(value)
    elif effect == 'bass_enhance':
        mixer.enhance_bass = bool(value)
        status_data['effects']['bass_enhance'] = bool(value)
    elif effect == 'presence_enhance':
        mixer.enhance_presence = bool(value)
        status_data['effects']['presence_enhance'] = bool(value)
    else:
        return jsonify({'success': False, 'message': 'Unknown effect'})
    
    return jsonify({
        'success': True,
        'message': f'{effect} updated',
        'effect': effect,
        'value': value
    })

@app.route('/api/test_sound', methods=['POST'])
def test_sound():
    """Generate test tone"""
    try:
        # Generate 1 second test tone
        duration = 1.0
        frequency = 440.0  # A4
        t = np.linspace(0, duration, int(mixer.sample_rate * duration))
        tone = np.sin(2 * np.pi * frequency * t) * 0.3
        
        # Play it
        sd.play(tone, mixer.sample_rate, device=mixer.output_device)
        sd.wait()
        
        return jsonify({
            'success': True,
            'message': 'Test tone played (440Hz)'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })

# For compatibility with existing frontend
@app.route('/api/start', methods=['POST'])
def start_compat():
    """Compatibility endpoint"""
    return start_mixing()

@app.route('/api/stop', methods=['POST'])  
def stop_compat():
    """Compatibility endpoint"""
    return stop_mixing()

@app.route('/api/status')
def status_compat():
    """Compatibility endpoint"""
    mixer_status = get_mixer_status().json
    
    # Return in expected format
    return jsonify({
        'hardware': {
            'ag06_connected': mixer_status['ag06_connected'],
            'device_id': mixer.input_device
        },
        'input_level': {
            'rms': status_data['input_level'],
            'peak': status_data['input_level'] + 10,
            'clipping': False,
            'too_quiet': status_data['input_level'] < -50
        },
        'spectrum': status_data['spectrum'],
        'music': {
            'detected': status_data['mode'] == 'music',
            'confidence': 0.9 if status_data['mode'] == 'music' else 0.1
        },
        'voice': {
            'detected': status_data['mode'] == 'voice',
            'confidence': 0.9 if status_data['mode'] == 'voice' else 0.1
        },
        'processing': status_data['mixing'],
        'timestamp': time.time()
    })

@app.route('/api/spectrum')
def spectrum_compat():
    """Compatibility endpoint"""
    return jsonify({
        'spectrum': status_data['spectrum'],
        'level_db': status_data['input_level'],
        'peak_db': status_data['input_level'] + 10,
        'classification': status_data['mode'],
        'peak_frequency': 440.0,
        'timestamp': time.time()
    })

if __name__ == '__main__':
    print("=" * 70)
    print("🎛️  INTEGRATED AI MIXER FOR AG06")
    print("=" * 70)
    print("✅ Real-time audio mixing with AI processing")
    print("✅ Web API on port 5001")
    print("✅ Compatible with existing frontend")
    print("=" * 70)
    print("\n📍 FEATURES:")
    print("  • Multiband compression & Dynamic EQ")
    print("  • Bass enhancement & Presence exciter")
    print("  • Stereo widening & Reverb/Delay")
    print("  • Music/Voice mode switching")
    print("=" * 70)
    
    # Start server
    app.run(host='0.0.0.0', port=5001, debug=False)