#!/usr/bin/env python3
"""
Production AI Mixer - Cloud-Ready SSE Backend
Following Google SRE & Meta Engineering best practices
"""

from flask import Flask, jsonify, Response, request, send_from_directory, g
from flask_cors import CORS
from auth_system import require_auth, require_api_key, optional_auth, auth_system
import time
import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional, Generator, Dict, Any
import threading
import queue
import random
import numpy as np
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

@dataclass
class AudioState:
    """Thread-safe audio state for cloud deployment"""
    input_rms: float = -60.0
    input_peak: float = -60.0
    output_rms: float = -60.0
    output_peak: float = -60.0
    processing: bool = False
    spectrum: list = None
    genre: str = "unknown"
    timestamp: float = 0.0
    stream_active: bool = False
    client_count: int = 0
    total_processed_samples: int = 0
    latency_ms: float = 0.0
    
    def __post_init__(self):
        if self.spectrum is None:
            self.spectrum = [0.0] * 64

class CloudAIMixer:
    """Cloud-ready AI mixer with SSE streaming"""
    
    def __init__(self):
        self.state = AudioState()
        self.state_lock = threading.Lock()
        
        # Audio simulation config
        self.sample_rate = 48000
        self.blocksize = 1024
        self.channels = 2
        
        # Processing parameters
        self.ai_mix = 0.7
        self.bass_boost = 1.3
        self.presence_boost = 1.2
        self.compression_ratio = 2.5
        self.stereo_width = 1.1
        
        # SSE streaming
        self.event_queue = queue.Queue()
        self.is_running = False
        self.simulation_thread = None
        
        # Performance metrics
        self.metrics = {
            'total_events': 0,
            'error_count': 0,
            'uptime_start': time.time()
        }
    
    def simulate_audio_processing(self):
        """Simulate audio processing for cloud deployment"""
        while self.is_running:
            try:
                # Simulate audio data generation
                timestamp = time.time()
                
                # Generate simulated audio metrics
                input_rms = -40 + random.gauss(0, 5)
                input_peak = max(input_rms + 6, -3)
                
                # Simulate processing
                output_rms = input_rms + 3  # Simulated gain
                output_peak = min(input_peak + 3, 0)  # Limited
                
                # Generate spectrum data
                spectrum = self._generate_spectrum()
                
                # Detect genre based on spectrum
                genre = self._detect_genre(spectrum)
                
                # Update state
                with self.state_lock:
                    self.state.input_rms = input_rms
                    self.state.input_peak = input_peak
                    self.state.output_rms = output_rms
                    self.state.output_peak = output_peak
                    self.state.spectrum = spectrum
                    self.state.genre = genre
                    self.state.timestamp = timestamp
                    self.state.total_processed_samples += self.blocksize
                    self.state.latency_ms = random.uniform(5, 15)
                
                # Create SSE event
                event_data = {
                    'type': 'audio_update',
                    'timestamp': timestamp,
                    'metrics': {
                        'input_rms': input_rms,
                        'input_peak': input_peak,
                        'output_rms': output_rms,
                        'output_peak': output_peak,
                        'latency_ms': self.state.latency_ms
                    },
                    'spectrum': spectrum[:16],  # Send reduced spectrum
                    'genre': genre
                }
                
                # Queue event for SSE
                self.event_queue.put(json.dumps(event_data))
                self.metrics['total_events'] += 1
                
                # Simulate processing rate
                time.sleep(0.1)  # 10Hz update rate
                
            except Exception as e:
                logger.error(f"Simulation error: {e}")
                self.metrics['error_count'] += 1
                time.sleep(1)
    
    def _generate_spectrum(self) -> list:
        """Generate simulated spectrum data"""
        try:
            # Generate realistic spectrum based on genre simulation
            spectrum = []
            for i in range(64):
                # Simulate frequency response curve
                freq = (i + 1) * (self.sample_rate / 2) / 64
                
                # Generate band energy with some randomness
                if freq < 200:  # Bass
                    energy = 80 + random.gauss(0, 10) * self.bass_boost
                elif freq < 2000:  # Mids
                    energy = 60 + random.gauss(0, 8)
                elif freq < 8000:  # Presence
                    energy = 40 + random.gauss(0, 6) * self.presence_boost
                else:  # Highs
                    energy = 20 + random.gauss(0, 5)
                
                spectrum.append(max(0, min(100, energy)))
            
            return spectrum
        except Exception as e:
            logger.error(f"Spectrum generation error: {e}")
            return [0.0] * 64
    
    def _detect_genre(self, spectrum: list) -> str:
        """Detect genre from spectrum data"""
        try:
            if not spectrum or sum(spectrum) < 100:
                return "silent"
            
            # Simple genre detection based on frequency distribution
            bass = np.mean(spectrum[:16])
            mids = np.mean(spectrum[16:48])
            highs = np.mean(spectrum[48:])
            
            if bass > mids * 1.5:
                return "electronic"
            elif highs > mids * 1.2:
                return "rock"
            elif mids > (bass + highs) / 2:
                return "vocal"
            else:
                return "pop"
        except:
            return "unknown"
    
    def start(self) -> bool:
        """Start cloud audio simulation"""
        if self.is_running:
            return True
            
        try:
            self.is_running = True
            
            # Start simulation thread
            self.simulation_thread = threading.Thread(
                target=self.simulate_audio_processing,
                daemon=True
            )
            self.simulation_thread.start()
            
            with self.state_lock:
                self.state.processing = True
                self.state.stream_active = True
                
            logger.info("Cloud AI mixer started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start mixer: {e}")
            self.is_running = False
            return False
    
    def stop(self):
        """Stop cloud audio simulation"""
        try:
            self.is_running = False
            
            # Wait for thread to finish
            if self.simulation_thread:
                self.simulation_thread.join(timeout=2)
                
            with self.state_lock:
                self.state.processing = False
                self.state.stream_active = False
                
            logger.info("Cloud AI mixer stopped")
            
        except Exception as e:
            logger.error(f"Error stopping mixer: {e}")
    
    def get_state(self) -> dict:
        """Get current state as JSON-serializable dict"""
        with self.state_lock:
            return {
                'input_rms': float(self.state.input_rms),
                'input_peak': float(self.state.input_peak),
                'output_rms': float(self.state.output_rms),
                'output_peak': float(self.state.output_peak),
                'processing': bool(self.state.processing),
                'spectrum': list(self.state.spectrum),
                'genre': str(self.state.genre),
                'timestamp': float(self.state.timestamp),
                'stream_active': bool(self.state.stream_active),
                'client_count': int(self.state.client_count),
                'latency_ms': float(self.state.latency_ms),
                'total_processed_samples': int(self.state.total_processed_samples)
            }
    
    def generate_sse_events(self) -> Generator[str, None, None]:
        """Generate SSE events for streaming"""
        with self.state_lock:
            self.state.client_count += 1
        
        try:
            while True:
                try:
                    # Get event from queue with timeout
                    event_data = self.event_queue.get(timeout=1)
                    yield f"data: {event_data}\n\n"
                    
                except queue.Empty:
                    # Send heartbeat
                    heartbeat = json.dumps({
                        'type': 'heartbeat',
                        'timestamp': time.time(),
                        'uptime': time.time() - self.metrics['uptime_start']
                    })
                    yield f"data: {heartbeat}\n\n"
                    
        finally:
            with self.state_lock:
                self.state.client_count -= 1

# Global mixer instance
mixer = CloudAIMixer()

# Flask routes
webapp_dir = os.environ.get('WEBAPP_DIR', '/app/webapp')
if not os.path.exists(webapp_dir):
    webapp_dir = '/Users/nguythe/ag06_mixer/automation-framework/webapp'

@app.route('/')
def index():
    """Serve main application"""
    if os.path.exists(os.path.join(webapp_dir, 'ai_mixer.html')):
        return send_from_directory(webapp_dir, 'ai_mixer.html')
    else:
        return jsonify({
            'name': 'Aioke',
            'version': '2.0.0',
            'status': 'ready',
            'endpoints': ['/api/stream', '/api/status', '/api/start', '/api/stop', '/health']
        })

@app.route('/health')
def health():
    """Health check endpoint for container orchestration"""
    return jsonify({
        'status': 'healthy',
        'processing': mixer.is_running,
        'uptime': time.time() - mixer.metrics['uptime_start'],
        'total_events': mixer.metrics['total_events'],
        'error_count': mixer.metrics['error_count']
    })

@app.route('/api/stream')
def stream():
    """SSE endpoint for real-time audio streaming"""
    def generate():
        for event in mixer.generate_sse_events():
            yield event
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Access-Control-Allow-Origin': '*'
        }
    )

@app.route('/api/start', methods=['POST'])
@require_auth(['write'])
def start_mixer():
    """Start the mixer (requires authentication)"""
    try:
        if mixer.start():
            logger.info(f"Mixer started by user {g.current_user.get('user_id', 'unknown')}")
            return jsonify({
                'success': True,
                'message': 'AI mixer started',
                'status': 'started',
                'user': g.current_user.get('user_id')
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to start mixer',
                'status': 'error'
            })
    except Exception as e:
        logger.error(f"Start endpoint error: {e}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}',
            'status': 'error'
        })

@app.route('/api/stop', methods=['POST'])
@require_auth(['write'])
def stop_mixer():
    """Stop the mixer (requires authentication)"""
    try:
        mixer.stop()
        logger.info(f"Mixer stopped by user {g.current_user.get('user_id', 'unknown')}")
        return jsonify({
            'success': True,
            'message': 'Mixer stopped',
            'status': 'stopped',
            'user': g.current_user.get('user_id')
        })
    except Exception as e:
        logger.error(f"Stop endpoint error: {e}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}',
            'status': 'error'
        })

@app.route('/api/status')
@optional_auth
def get_status():
    """Get mixer status - cloud optimized"""
    try:
        state = mixer.get_state()
        
        return jsonify({
            'processing': state['processing'],
            'cloud': {
                'environment': 'azure' if os.environ.get('WEBSITE_INSTANCE_ID') else 'local',
                'stream_active': state['stream_active'],
                'client_count': state['client_count']
            },
            'input_level': {
                'rms': state['input_rms'],
                'peak': state['input_peak'],
                'clipping': state['input_peak'] > -3,
                'too_quiet': state['input_rms'] < -50
            },
            'output_level': {
                'rms': state['output_rms'],
                'peak': state['output_peak']
            },
            'spectrum': state['spectrum'],
            'music': {
                'detected': state['input_rms'] > -40,
                'genre': state['genre'],
                'confidence': max(0.0, min(1.0, (state['input_rms'] + 40) / 20))
            },
            'performance': {
                'latency_ms': state['latency_ms'],
                'total_samples': state['total_processed_samples'],
                'uptime': time.time() - mixer.metrics['uptime_start']
            },
            'timestamp': state['timestamp']
        })
        
    except Exception as e:
        logger.error(f"Status endpoint error: {e}")
        return jsonify({
            'processing': False,
            'cloud': {'environment': 'error', 'stream_active': False, 'client_count': 0},
            'input_level': {'rms': -60, 'peak': -60, 'clipping': False, 'too_quiet': True},
            'spectrum': [0] * 64,
            'music': {'detected': False, 'genre': 'unknown', 'confidence': 0},
            'timestamp': time.time(),
            'error': str(e)
        })

@app.route('/api/spectrum')
def get_spectrum():
    """Get spectrum data"""
    try:
        state = mixer.get_state()
        
        return jsonify({
            'spectrum': state['spectrum'],
            'level_db': state['input_rms'],
            'peak_db': state['input_peak'],
            'classification': state['genre'],
            'peak_frequency': 440.0,
            'timestamp': state['timestamp']
        })
        
    except Exception as e:
        logger.error(f"Spectrum endpoint error: {e}")
        return jsonify({
            'spectrum': [0] * 64,
            'level_db': -60,
            'peak_db': -60,
            'classification': 'error',
            'peak_frequency': 440.0,
            'timestamp': time.time(),
            'error': str(e)
        })

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update mixer configuration"""
    try:
        data = request.json or {}
        
        if 'ai_mix' in data:
            mixer.ai_mix = float(data['ai_mix'])
        if 'bass_boost' in data:
            mixer.bass_boost = float(data['bass_boost'])
        if 'presence_boost' in data:
            mixer.presence_boost = float(data['presence_boost'])
            
        return jsonify({
            'success': True,
            'config': {
                'ai_mix': mixer.ai_mix,
                'bass_boost': mixer.bass_boost,
                'presence_boost': mixer.presence_boost
            }
        })
        
    except Exception as e:
        logger.error(f"Config endpoint error: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

# Authentication endpoints
@app.route('/auth/login', methods=['POST'])
def login():
    """Generate JWT token for API access"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        # Simple demo authentication (use proper auth in production)
        if username == 'admin' and password == 'aioke2025':
            user_data = {
                'user_id': username,
                'permissions': ['read', 'write', 'admin']
            }
            token = auth_system.generate_jwt_token(user_data)
            
            return jsonify({
                'success': True,
                'token': token,
                'expires_in': auth_system.token_expiry_hours * 3600,
                'permissions': user_data['permissions']
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid credentials'
            }), 401
    
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/auth/apikey', methods=['POST'])
@require_auth(['admin'])
def generate_api_key():
    """Generate new API key (admin only)"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        permissions = data.get('permissions', ['read'])
        
        if not user_id:
            return jsonify({'error': 'user_id required'}), 400
        
        api_key = auth_system.generate_api_key(user_id, permissions)
        
        return jsonify({
            'success': True,
            'api_key': api_key,
            'user_id': user_id,
            'permissions': permissions,
            'message': 'Store this key securely - it will not be shown again!'
        })
    
    except Exception as e:
        logger.error(f"API key generation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/auth/me')
@require_auth()
def get_user_info():
    """Get current user information"""
    try:
        return jsonify({
            'user_id': g.current_user.get('user_id'),
            'permissions': g.current_user.get('permissions', []),
            'auth_method': g.auth_method,
            'rate_limit': g.rate_limit_status,
            'last_used': g.current_user.get('last_used'),
            'usage_count': g.current_user.get('usage_count', 0)
        })
    
    except Exception as e:
        logger.error(f"User info error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🎵 CLOUD AI MIXER - SSE PRODUCTION BACKEND")
    print("=" * 60)
    print("Features:")
    print("  • Server-Sent Events (SSE) for real-time streaming")
    print("  • Cloud-optimized with health checks")
    print("  • Simulated audio processing for cloud deployment")
    print("  • Auto-scaling ready with stateless design")
    print("  • Azure Container Apps & Vercel compatible")
    print("=" * 60)
    
    # Auto-start mixer on launch
    mixer.start()
    
    # Get port from environment or default
    port = int(os.environ.get('PORT', 8080))
    
    print(f"Endpoints:")
    print(f"  • Health: http://localhost:{port}/health")
    print(f"  • SSE Stream: http://localhost:{port}/api/stream")
    print(f"  • Status: http://localhost:{port}/api/status")
    print(f"  • Spectrum: http://localhost:{port}/api/spectrum")
    print("=" * 60)
    
    # Note: In production, gunicorn will be used instead
    # This is for local development only
    if os.environ.get('PRODUCTION') != 'true':
        app.run(host='0.0.0.0', port=port, debug=False)