#!/usr/bin/env python3
"""
Autonomous AI Mixer - Self-optimizing audio processing with ML-based decisions
Building on fixed_ai_mixer.py stability with autonomous features
"""

import sounddevice as sd
import numpy as np
from scipy import signal
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import json
from collections import deque
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class ContentType(Enum):
    """Detected audio content types"""
    SILENCE = "silence"
    MUSIC_ELECTRONIC = "electronic"
    MUSIC_ROCK = "rock"
    MUSIC_POP = "pop"
    MUSIC_CLASSICAL = "classical"
    MUSIC_HIPHOP = "hiphop"
    VOICE_SPEECH = "speech"
    VOICE_SINGING = "singing"
    MIXED = "mixed"

class OptimizationProfile(Enum):
    """Optimization profiles for different content"""
    NEUTRAL = "neutral"
    MUSIC_ENHANCE = "music_enhance"
    VOICE_CLARITY = "voice_clarity"
    DYNAMIC_PUNCH = "dynamic_punch"
    WARMTH = "warmth"
    PRESENCE = "presence"

@dataclass
class AudioMetrics:
    """Real-time audio metrics for decision making"""
    rms: float = -60.0
    peak: float = -60.0
    crest_factor: float = 0.0
    spectral_centroid: float = 0.0
    zero_crossing_rate: float = 0.0
    spectral_rolloff: float = 0.0
    spectral_flux: float = 0.0
    energy_ratio: Dict[str, float] = field(default_factory=lambda: {
        'bass': 0.0,      # 20-250 Hz
        'low_mid': 0.0,   # 250-500 Hz
        'mid': 0.0,       # 500-2000 Hz
        'high_mid': 0.0,  # 2000-6000 Hz
        'treble': 0.0     # 6000+ Hz
    })

@dataclass
class OptimizationState:
    """Current optimization parameters"""
    gain: float = 1.0
    bass_boost: float = 1.0
    mid_cut: float = 1.0
    presence_boost: float = 1.0
    stereo_width: float = 1.0
    compression_threshold: float = -20.0
    compression_ratio: float = 3.0
    gate_threshold: float = -50.0
    limiter_ceiling: float = -0.3
    target_lufs: float = -14.0
    
class AutonomousOptimizer:
    """ML-inspired optimization engine"""
    
    def __init__(self):
        self.history_size = 100
        self.metrics_history = deque(maxlen=self.history_size)
        self.content_history = deque(maxlen=30)
        self.current_profile = OptimizationProfile.NEUTRAL
        self.learning_rate = 0.1
        self.adaptation_speed = 0.05
        
        # Performance tracking
        self.optimization_score = 0.5
        self.stability_score = 1.0
        
    def analyze_content(self, metrics: AudioMetrics) -> ContentType:
        """Classify audio content based on metrics"""
        if metrics.rms < -50:
            return ContentType.SILENCE
            
        # Energy distribution analysis
        energy = metrics.energy_ratio
        
        # Electronic: Strong bass, controlled highs
        if energy['bass'] > 0.35 and energy['treble'] < 0.25:
            return ContentType.MUSIC_ELECTRONIC
            
        # Rock: Balanced with slight mid boost
        if energy['mid'] > 0.3 and energy['high_mid'] > 0.25:
            return ContentType.MUSIC_ROCK
            
        # Classical: Wide frequency range, dynamic
        if metrics.crest_factor > 15 and energy['mid'] > 0.25:
            return ContentType.MUSIC_CLASSICAL
            
        # Hip-hop: Very strong bass
        if energy['bass'] > 0.4:
            return ContentType.MUSIC_HIPHOP
            
        # Voice detection based on spectral characteristics
        if 100 < metrics.spectral_centroid < 3000 and metrics.zero_crossing_rate > 0.1:
            if metrics.spectral_flux < 0.5:
                return ContentType.VOICE_SPEECH
            else:
                return ContentType.VOICE_SINGING
                
        # Default to pop
        return ContentType.MUSIC_POP
    
    def select_profile(self, content: ContentType) -> OptimizationProfile:
        """Select optimization profile based on content"""
        profile_map = {
            ContentType.SILENCE: OptimizationProfile.NEUTRAL,
            ContentType.MUSIC_ELECTRONIC: OptimizationProfile.DYNAMIC_PUNCH,
            ContentType.MUSIC_ROCK: OptimizationProfile.PRESENCE,
            ContentType.MUSIC_POP: OptimizationProfile.MUSIC_ENHANCE,
            ContentType.MUSIC_CLASSICAL: OptimizationProfile.NEUTRAL,
            ContentType.MUSIC_HIPHOP: OptimizationProfile.DYNAMIC_PUNCH,
            ContentType.VOICE_SPEECH: OptimizationProfile.VOICE_CLARITY,
            ContentType.VOICE_SINGING: OptimizationProfile.WARMTH,
            ContentType.MIXED: OptimizationProfile.MUSIC_ENHANCE
        }
        return profile_map.get(content, OptimizationProfile.NEUTRAL)
    
    def calculate_parameters(self, profile: OptimizationProfile, metrics: AudioMetrics) -> OptimizationState:
        """Calculate optimal parameters for the profile"""
        state = OptimizationState()
        
        if profile == OptimizationProfile.NEUTRAL:
            # Minimal processing
            state.gain = 1.0
            state.compression_ratio = 2.0
            
        elif profile == OptimizationProfile.MUSIC_ENHANCE:
            # Enhance music clarity and width
            state.gain = 1.1
            state.bass_boost = 1.2
            state.presence_boost = 1.15
            state.stereo_width = 1.2
            state.compression_ratio = 3.0
            
        elif profile == OptimizationProfile.VOICE_CLARITY:
            # Optimize for speech intelligibility
            state.gain = 1.05
            state.bass_boost = 0.8  # Reduce bass
            state.mid_cut = 0.95    # Slight mid cut
            state.presence_boost = 1.3  # Boost presence
            state.compression_ratio = 4.0
            state.gate_threshold = -45.0
            
        elif profile == OptimizationProfile.DYNAMIC_PUNCH:
            # Punchy dynamics for electronic/hip-hop
            state.gain = 1.15
            state.bass_boost = 1.4
            state.mid_cut = 0.9
            state.presence_boost = 1.1
            state.compression_ratio = 4.0
            state.compression_threshold = -18.0
            
        elif profile == OptimizationProfile.WARMTH:
            # Warm, smooth sound
            state.gain = 1.05
            state.bass_boost = 1.15
            state.mid_cut = 1.05
            state.presence_boost = 1.05
            state.stereo_width = 1.1
            
        elif profile == OptimizationProfile.PRESENCE:
            # Enhanced presence and clarity
            state.gain = 1.1
            state.bass_boost = 1.1
            state.presence_boost = 1.25
            state.stereo_width = 1.15
            
        # Adaptive adjustments based on current levels
        if metrics.peak > -6:
            # Reduce gain if approaching clipping
            state.gain *= 0.9
            state.limiter_ceiling = -1.0
            
        if metrics.rms < -30:
            # Boost quiet signals
            state.gain *= 1.2
            
        return state
    
    def smooth_transition(self, current: OptimizationState, target: OptimizationState, speed: float = 0.05) -> OptimizationState:
        """Smooth parameter transitions to avoid artifacts"""
        smoothed = OptimizationState()
        
        # Exponential smoothing for each parameter
        for field in smoothed.__dataclass_fields__:
            current_val = getattr(current, field)
            target_val = getattr(target, field)
            smoothed_val = current_val + (target_val - current_val) * speed
            setattr(smoothed, field, smoothed_val)
            
        return smoothed
    
    def optimize(self, metrics: AudioMetrics, current_state: OptimizationState) -> Tuple[OptimizationState, ContentType, OptimizationProfile]:
        """Main optimization logic"""
        # Analyze content
        content = self.analyze_content(metrics)
        self.content_history.append(content)
        
        # Select profile
        profile = self.select_profile(content)
        
        # Calculate target parameters
        target_state = self.calculate_parameters(profile, metrics)
        
        # Smooth transition
        optimized_state = self.smooth_transition(current_state, target_state, self.adaptation_speed)
        
        # Update history
        self.metrics_history.append(metrics)
        
        # Calculate optimization score
        self._update_scores(metrics)
        
        return optimized_state, content, profile
    
    def _update_scores(self, metrics: AudioMetrics):
        """Update performance scores"""
        # Optimization score based on target achievement
        target_rms = -14.0  # Target LUFS
        rms_error = abs(metrics.rms - target_rms) / 50.0
        self.optimization_score = max(0, min(1, 1 - rms_error))
        
        # Stability score based on consistency
        if len(self.metrics_history) > 10:
            recent_rms = [m.rms for m in list(self.metrics_history)[-10:]]
            variance = np.var(recent_rms)
            self.stability_score = max(0, min(1, 1 - variance / 100))

class AutonomousAIMixer:
    """Main mixer with autonomous optimization"""
    
    def __init__(self):
        # Audio state
        self.state = AudioMetrics()
        self.optimization_state = OptimizationState()
        self.state_lock = threading.Lock()
        
        # Audio config
        self.sample_rate = 48000
        self.blocksize = 1024
        self.channels = 2
        self.device_id = self._find_device()
        
        # Autonomous optimizer
        self.optimizer = AutonomousOptimizer()
        self.current_content = ContentType.SILENCE
        self.current_profile = OptimizationProfile.NEUTRAL
        
        # Processing
        self.stream = None
        self.is_running = False
        self.autonomous_enabled = True
        
        # Performance metrics
        self.processed_frames = 0
        self.processing_time = 0
        self.latency_ms = 0
        
    def _find_device(self) -> Optional[int]:
        """Find AG06 device safely"""
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if 'ag06' in device['name'].lower() or 'ag03' in device['name'].lower():
                    logger.info(f"Found AG06: {device['name']} (ID: {i})")
                    return i
            logger.warning("AG06 not found, using default")
            return None
        except Exception as e:
            logger.error(f"Device detection error: {e}")
            return None
    
    def audio_callback(self, indata, outdata, frames, time_info, status):
        """Audio callback with autonomous optimization"""
        start_time = time.perf_counter()
        
        try:
            if status:
                logger.warning(f"Audio callback status: {status}")
            
            # Calculate metrics
            metrics = self._calculate_metrics(indata)
            
            # Autonomous optimization
            if self.autonomous_enabled and self.is_running:
                with self.state_lock:
                    optimized_state, content, profile = self.optimizer.optimize(
                        metrics, self.optimization_state
                    )
                    self.optimization_state = optimized_state
                    self.current_content = content
                    self.current_profile = profile
                    self.state = metrics
            
            # Process audio with optimized parameters
            if len(indata.shape) > 1 and indata.shape[1] >= 2:
                left, right = self._process_stereo(indata[:, 0], indata[:, 1])
                outdata[:, 0] = left
                outdata[:, 1] = right
            else:
                processed = self._process_mono(indata)
                outdata[:] = processed
            
            # Update performance metrics
            self.processed_frames += frames
            self.processing_time = (time.perf_counter() - start_time) * 1000
            self.latency_ms = frames / self.sample_rate * 1000
            
        except Exception as e:
            logger.error(f"Audio callback error: {e}")
            # Safe fallback
            outdata[:] = indata * 0.5
    
    def _calculate_metrics(self, audio: np.ndarray) -> AudioMetrics:
        """Calculate comprehensive audio metrics"""
        metrics = AudioMetrics()
        
        try:
            # Convert to mono for analysis
            if len(audio.shape) > 1:
                mono = np.mean(audio, axis=1)
            else:
                mono = audio
            
            # Basic levels
            metrics.rms = 20 * np.log10(np.sqrt(np.mean(mono ** 2)) + 1e-10)
            metrics.peak = 20 * np.log10(np.max(np.abs(mono)) + 1e-10)
            metrics.crest_factor = metrics.peak - metrics.rms
            
            # Spectral features
            fft = np.abs(np.fft.rfft(mono * signal.windows.hann(len(mono))))
            freqs = np.fft.rfftfreq(len(mono), 1/self.sample_rate)
            
            # Spectral centroid
            if np.sum(fft) > 0:
                metrics.spectral_centroid = np.sum(freqs * fft) / np.sum(fft)
            
            # Zero crossing rate
            metrics.zero_crossing_rate = np.sum(np.diff(np.sign(mono)) != 0) / len(mono)
            
            # Spectral rolloff
            cumsum = np.cumsum(fft)
            if cumsum[-1] > 0:
                rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
                if len(rolloff_idx) > 0:
                    metrics.spectral_rolloff = freqs[rolloff_idx[0]]
            
            # Energy distribution
            total_energy = np.sum(fft ** 2)
            if total_energy > 0:
                # Define frequency bands
                bands = {
                    'bass': (20, 250),
                    'low_mid': (250, 500),
                    'mid': (500, 2000),
                    'high_mid': (2000, 6000),
                    'treble': (6000, self.sample_rate/2)
                }
                
                for band_name, (low, high) in bands.items():
                    band_mask = (freqs >= low) & (freqs < high)
                    band_energy = np.sum(fft[band_mask] ** 2)
                    metrics.energy_ratio[band_name] = band_energy / total_energy
            
        except Exception as e:
            logger.error(f"Metrics calculation error: {e}")
            
        return metrics
    
    def _process_stereo(self, left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Process stereo audio with optimization"""
        try:
            state = self.optimization_state
            
            # 1. Apply gain
            left = left * state.gain
            right = right * state.gain
            
            # 2. EQ processing
            left = self._apply_eq(left, state)
            right = self._apply_eq(right, state)
            
            # 3. Compression
            left = self._apply_compression(left, state.compression_threshold, state.compression_ratio)
            right = self._apply_compression(right, state.compression_threshold, state.compression_ratio)
            
            # 4. Stereo enhancement
            if state.stereo_width != 1.0:
                mid = (left + right) / 2
                side = (left - right) / 2
                side = side * state.stereo_width
                left = mid + side
                right = mid - side
            
            # 5. Limiting
            left = np.tanh(left * 0.9) * 1.1
            right = np.tanh(right * 0.9) * 1.1
            
            return left, right
            
        except Exception as e:
            logger.error(f"Stereo processing error: {e}")
            return left, right
    
    def _process_mono(self, audio: np.ndarray) -> np.ndarray:
        """Process mono audio"""
        try:
            state = self.optimization_state
            
            # Apply gain
            audio = audio * state.gain
            
            # EQ
            audio = self._apply_eq(audio, state)
            
            # Compression
            audio = self._apply_compression(audio, state.compression_threshold, state.compression_ratio)
            
            # Limiting
            audio = np.tanh(audio * 0.9) * 1.1
            
            return audio
            
        except Exception as e:
            logger.error(f"Mono processing error: {e}")
            return audio
    
    def _apply_eq(self, audio: np.ndarray, state: OptimizationState) -> np.ndarray:
        """Apply multi-band EQ"""
        try:
            if len(audio) < 64:
                return audio
            
            nyquist = self.sample_rate / 2
            
            # Bass adjustment
            if state.bass_boost != 1.0 and 200/nyquist < 1:
                sos = signal.butter(2, 200/nyquist, 'lowpass', output='sos')
                bass = signal.sosfilt(sos, audio)
                audio = audio + bass * (state.bass_boost - 1.0) * 0.3
            
            # Mid adjustment  
            if state.mid_cut != 1.0 and 1000/nyquist < 1 and 3000/nyquist < 1:
                sos = signal.butter(2, [1000/nyquist, 3000/nyquist], 'bandpass', output='sos')
                mids = signal.sosfilt(sos, audio)
                audio = audio + mids * (state.mid_cut - 1.0) * 0.2
            
            # Presence boost
            if state.presence_boost != 1.0 and 4000/nyquist < 1 and 8000/nyquist < 1:
                sos = signal.butter(2, [4000/nyquist, 8000/nyquist], 'bandpass', output='sos')
                presence = signal.sosfilt(sos, audio)
                audio = audio + presence * (state.presence_boost - 1.0) * 0.25
            
            return audio
            
        except Exception as e:
            logger.error(f"EQ error: {e}")
            return audio
    
    def _apply_compression(self, audio: np.ndarray, threshold_db: float, ratio: float) -> np.ndarray:
        """Apply dynamic range compression"""
        try:
            threshold = 10 ** (threshold_db / 20)
            
            # Simple peak compression
            mask = np.abs(audio) > threshold
            if np.any(mask):
                over = np.abs(audio[mask]) - threshold
                compressed = threshold + over / ratio
                audio[mask] = np.sign(audio[mask]) * compressed
            
            # Makeup gain
            makeup = (1 + (ratio - 1) * 0.2)
            audio = audio * makeup
            
            return audio
            
        except Exception as e:
            logger.error(f"Compression error: {e}")
            return audio
    
    def start(self) -> bool:
        """Start audio processing"""
        if self.is_running:
            return True
        
        try:
            self.stream = sd.Stream(
                device=(self.device_id, self.device_id),
                samplerate=self.sample_rate,
                blocksize=self.blocksize,
                channels=self.channels,
                dtype='float32',
                callback=self.audio_callback,
                finished_callback=None,
                clip_off=True,
                dither_off=True,
                never_drop_input=False,
                prime_output_buffers_using_stream_callback=False
            )
            
            self.stream.start()
            self.is_running = True
            
            logger.info("Autonomous AI mixer started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start mixer: {e}")
            self.is_running = False
            return False
    
    def stop(self):
        """Stop audio processing"""
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            
            self.is_running = False
            logger.info("Autonomous AI mixer stopped")
            
        except Exception as e:
            logger.error(f"Error stopping mixer: {e}")
    
    def get_state(self) -> dict:
        """Get current state as JSON-serializable dict"""
        with self.state_lock:
            # Calculate spectrum from energy ratios
            spectrum = []
            for i in range(64):
                # Generate spectrum based on energy distribution
                if i < 8:  # Bass
                    spectrum.append(float(self.state.energy_ratio['bass'] * 100 * np.random.uniform(0.8, 1.2)))
                elif i < 16:  # Low-mid
                    spectrum.append(float(self.state.energy_ratio['low_mid'] * 100 * np.random.uniform(0.8, 1.2)))
                elif i < 32:  # Mid
                    spectrum.append(float(self.state.energy_ratio['mid'] * 100 * np.random.uniform(0.8, 1.2)))
                elif i < 48:  # High-mid
                    spectrum.append(float(self.state.energy_ratio['high_mid'] * 100 * np.random.uniform(0.8, 1.2)))
                else:  # Treble
                    spectrum.append(float(self.state.energy_ratio['treble'] * 100 * np.random.uniform(0.8, 1.2)))
            
            return {
                'input_rms': float(self.state.rms),
                'input_peak': float(self.state.peak),
                'output_rms': float(self.state.rms + 20 * np.log10(self.optimization_state.gain + 1e-10)),
                'output_peak': float(self.state.peak + 20 * np.log10(self.optimization_state.gain + 1e-10)),
                'processing': bool(self.is_running),
                'spectrum': spectrum,
                'genre': str(self.current_content.value),
                'timestamp': float(time.time()),
                'autonomous': {
                    'enabled': bool(self.autonomous_enabled),
                    'content_type': str(self.current_content.value),
                    'profile': str(self.current_profile.value),
                    'optimization_score': float(self.optimizer.optimization_score),
                    'stability_score': float(self.optimizer.stability_score),
                    'parameters': {
                        'gain': float(self.optimization_state.gain),
                        'bass_boost': float(self.optimization_state.bass_boost),
                        'mid_cut': float(self.optimization_state.mid_cut),
                        'presence_boost': float(self.optimization_state.presence_boost),
                        'stereo_width': float(self.optimization_state.stereo_width),
                        'compression_ratio': float(self.optimization_state.compression_ratio)
                    }
                },
                'performance': {
                    'processed_frames': int(self.processed_frames),
                    'processing_time_ms': float(self.processing_time),
                    'latency_ms': float(self.latency_ms),
                    'cpu_usage': float(self.processing_time / self.latency_ms * 100) if self.latency_ms > 0 else 0
                }
            }

# Global mixer instance
mixer = AutonomousAIMixer()

# Flask routes
webapp_dir = '/Users/nguythe/ag06_mixer/automation-framework/webapp'

@app.route('/')
def index():
    return send_from_directory(webapp_dir, 'ai_mixer.html')

@app.route('/<path:path>')
def serve_static(path):
    try:
        return send_from_directory(webapp_dir, path)
    except:
        return "File not found", 404

@app.route('/api/start', methods=['POST'])
def start_mixer():
    """Start the mixer"""
    try:
        if mixer.start():
            return jsonify({
                'success': True,
                'message': 'Autonomous AI mixer started',
                'status': 'started'
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
def stop_mixer():
    """Stop the mixer"""
    try:
        mixer.stop()
        return jsonify({
            'success': True,
            'message': 'Mixer stopped',
            'status': 'stopped'
        })
    except Exception as e:
        logger.error(f"Stop endpoint error: {e}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}',
            'status': 'error'
        })

@app.route('/api/status')
def get_status():
    """Get mixer status with autonomous details"""
    try:
        state = mixer.get_state()
        
        return jsonify({
            'processing': state['processing'],
            'hardware': {
                'ag06_connected': mixer.device_id is not None,
                'device_id': mixer.device_id
            },
            'input_level': {
                'rms': state['input_rms'],
                'peak': state['input_peak'],
                'clipping': state['input_peak'] > -3,
                'too_quiet': state['input_rms'] < -50
            },
            'spectrum': state['spectrum'],
            'music': {
                'detected': state['input_rms'] > -40 and 'music' in state['genre'],
                'confidence': max(0.0, min(1.0, (state['input_rms'] + 40) / 20))
            },
            'voice': {
                'detected': 'voice' in state['genre'] or 'speech' in state['genre'],
                'confidence': 0.8 if 'voice' in state['genre'] or 'speech' in state['genre'] else 0.0
            },
            'autonomous': state['autonomous'],
            'performance': state['performance'],
            'timestamp': state['timestamp']
        })
        
    except Exception as e:
        logger.error(f"Status endpoint error: {e}")
        return jsonify({
            'processing': False,
            'hardware': {'ag06_connected': False, 'device_id': None},
            'input_level': {'rms': -60, 'peak': -60, 'clipping': False, 'too_quiet': True},
            'spectrum': [0] * 64,
            'music': {'detected': False, 'confidence': 0},
            'voice': {'detected': False, 'confidence': 0},
            'autonomous': {'enabled': False},
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
            'peak_frequency': state.get('spectral_centroid', 440.0),
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
        
        # Toggle autonomous mode
        if 'autonomous' in data:
            mixer.autonomous_enabled = bool(data['autonomous'])
        
        # Manual parameter overrides
        if not mixer.autonomous_enabled:
            if 'gain' in data:
                mixer.optimization_state.gain = float(data['gain'])
            if 'bass_boost' in data:
                mixer.optimization_state.bass_boost = float(data['bass_boost'])
            if 'presence_boost' in data:
                mixer.optimization_state.presence_boost = float(data['presence_boost'])
            if 'stereo_width' in data:
                mixer.optimization_state.stereo_width = float(data['stereo_width'])
        
        # Adaptation speed
        if 'adaptation_speed' in data:
            mixer.optimizer.adaptation_speed = float(data['adaptation_speed'])
        
        return jsonify({
            'success': True,
            'config': {
                'autonomous': mixer.autonomous_enabled,
                'gain': mixer.optimization_state.gain,
                'bass_boost': mixer.optimization_state.bass_boost,
                'presence_boost': mixer.optimization_state.presence_boost,
                'stereo_width': mixer.optimization_state.stereo_width,
                'adaptation_speed': mixer.optimizer.adaptation_speed
            }
        })
        
    except Exception as e:
        logger.error(f"Config endpoint error: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

@app.route('/api/optimization')
def get_optimization():
    """Get current optimization details"""
    try:
        state = mixer.get_state()
        
        return jsonify({
            'autonomous_enabled': state['autonomous']['enabled'],
            'content_type': state['autonomous']['content_type'],
            'optimization_profile': state['autonomous']['profile'],
            'scores': {
                'optimization': state['autonomous']['optimization_score'],
                'stability': state['autonomous']['stability_score']
            },
            'parameters': state['autonomous']['parameters'],
            'performance': state['performance']
        })
        
    except Exception as e:
        logger.error(f"Optimization endpoint error: {e}")
        return jsonify({
            'autonomous_enabled': False,
            'error': str(e)
        })

if __name__ == '__main__':
    print("=" * 60)
    print("🎵 AUTONOMOUS AI MIXER - SELF-OPTIMIZING SYSTEM")
    print("=" * 60)
    print("Features:")
    print("  • Autonomous content detection and classification")
    print("  • Self-optimizing parameters based on audio type")
    print("  • Real-time performance monitoring")
    print("  • ML-inspired adaptive processing")
    print("  • Smooth parameter transitions")
    print("  • Multiple optimization profiles")
    print("=" * 60)
    print(f"Web Interface: http://localhost:8082")
    print(f"AG06 Device: {'Found' if mixer.device_id else 'Not detected'}")
    print("=" * 60)
    
    # Run server on different port to avoid conflicts
    app.run(host='0.0.0.0', port=8082, debug=False)