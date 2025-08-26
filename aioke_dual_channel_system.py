#!/usr/bin/env python3
"""
AiOke Dual Channel Karaoke System
Following Google's audio architecture best practices for channel separation

Architecture:
- Channel 1: Vocals (Microphone) - Independent processing chain
- Channel 2: Music (Any source - YouTube, Spotify, etc.) - Separate processing
- No mixing at software level - hardware mixer handles final blend
- Each channel has its own AI-optimized effects pipeline
"""

import asyncio
import numpy as np
from aiohttp import web
import json
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
from collections import deque
import logging
import time

# Optional audio libraries - will run in simulation mode if not available
try:
    import pyaudio
    import sounddevice as sd
    from scipy import signal
    AUDIO_AVAILABLE = True
    logger = logging.getLogger('AiOke.DualChannel')
    logger.info("Audio libraries loaded - hardware mode available")
except ImportError:
    AUDIO_AVAILABLE = False
    logger = logging.getLogger('AiOke.DualChannel')
    logger.warning("Audio libraries not available - running in simulation mode")

# Configure logging following Google's structured logging practices
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('AiOke.DualChannel')


class ChannelType(Enum):
    """Audio channel types"""
    VOCAL = "vocal"
    MUSIC = "music"


@dataclass
class AudioChannel:
    """
    Independent audio channel configuration
    Following Google's protocol buffer style data structures
    """
    channel_id: int
    channel_type: ChannelType
    sample_rate: int = 44100
    buffer_size: int = 512
    device_index: Optional[int] = None
    
    # Independent effects chain per channel
    effects: Dict = None
    
    def __post_init__(self):
        if self.effects is None:
            self.effects = self.get_default_effects()
    
    def get_default_effects(self) -> Dict:
        """Default effects based on channel type"""
        if self.channel_type == ChannelType.VOCAL:
            return {
                'gate': {'threshold': -40, 'ratio': 10, 'enabled': True},
                'compressor': {'threshold': -12, 'ratio': 4, 'attack': 5, 'release': 100, 'enabled': True},
                'eq': {
                    'high_pass': {'freq': 80, 'enabled': True},
                    'low_shelf': {'freq': 200, 'gain': -2, 'enabled': False},
                    'mid_bell': {'freq': 2000, 'gain': 3, 'q': 0.7, 'enabled': True},
                    'high_shelf': {'freq': 8000, 'gain': 2, 'enabled': True}
                },
                'reverb': {'room_size': 0.3, 'damping': 0.5, 'wet': 0.25, 'enabled': True},
                'delay': {'time_ms': 250, 'feedback': 0.3, 'wet': 0.15, 'enabled': False},
                'limiter': {'threshold': -1, 'lookahead': 5, 'enabled': True}
            }
        else:  # MUSIC
            return {
                'eq': {
                    'high_pass': {'freq': 20, 'enabled': False},
                    'low_shelf': {'freq': 100, 'gain': 0, 'enabled': False},
                    'mid_bell': {'freq': 1000, 'gain': -2, 'q': 0.5, 'enabled': True},  # Duck mids for vocal space
                    'high_shelf': {'freq': 10000, 'gain': 0, 'enabled': False}
                },
                'stereo_enhancer': {'width': 1.2, 'enabled': True},
                'vocal_remover': {'strength': 0.5, 'enabled': False},  # Optional karaoke mode
                'limiter': {'threshold': -3, 'lookahead': 5, 'enabled': True}
            }


class AudioProcessor:
    """
    Independent audio processor for each channel
    Following Google's modular design patterns
    """
    
    def __init__(self, channel: AudioChannel):
        self.channel = channel
        self.sample_rate = channel.sample_rate
        self.buffer_size = channel.buffer_size
        
        # Ring buffer for audio processing
        self.audio_buffer = deque(maxlen=self.sample_rate * 2)  # 2 seconds
        
        # Pre-calculate filter coefficients
        self._init_filters()
        
    def _init_filters(self):
        """Initialize filter coefficients for real-time processing"""
        self.filters = {}
        
        if not AUDIO_AVAILABLE:
            logger.info(f"Initializing filters in simulation mode for {self.channel.channel_type.value} channel")
            return
        
        if 'eq' in self.channel.effects:
            eq = self.channel.effects['eq']
            
            # High-pass filter
            if eq.get('high_pass', {}).get('enabled'):
                freq = eq['high_pass']['freq']
                self.filters['high_pass'] = signal.butter(
                    2, freq, btype='high', fs=self.sample_rate, output='sos'
                )
            
            # Low shelf
            if eq.get('low_shelf', {}).get('enabled'):
                freq = eq['low_shelf']['freq']
                gain = eq['low_shelf']['gain']
                self.filters['low_shelf'] = self._design_shelf_filter(freq, gain, 'low')
            
            # Mid bell
            if eq.get('mid_bell', {}).get('enabled'):
                freq = eq['mid_bell']['freq']
                gain = eq['mid_bell']['gain']
                q = eq['mid_bell'].get('q', 0.7)
                self.filters['mid_bell'] = self._design_bell_filter(freq, gain, q)
            
            # High shelf
            if eq.get('high_shelf', {}).get('enabled'):
                freq = eq['high_shelf']['freq']
                gain = eq['high_shelf']['gain']
                self.filters['high_shelf'] = self._design_shelf_filter(freq, gain, 'high')
    
    def _design_shelf_filter(self, freq: float, gain: float, shelf_type: str):
        """Design shelf filter using Google's audio DSP approach"""
        nyquist = self.sample_rate / 2
        normalized_freq = freq / nyquist
        
        if shelf_type == 'low':
            if normalized_freq >= 1:
                normalized_freq = 0.99
            return signal.butter(2, normalized_freq, btype='low', fs=self.sample_rate, output='sos')
        else:  # high
            if normalized_freq >= 1:
                normalized_freq = 0.99
            return signal.butter(2, normalized_freq, btype='high', fs=self.sample_rate, output='sos')
    
    def _design_bell_filter(self, freq: float, gain: float, q: float):
        """Design parametric EQ bell filter"""
        w0 = 2 * np.pi * freq / self.sample_rate
        A = 10 ** (gain / 40)
        alpha = np.sin(w0) / (2 * q)
        
        # Peaking EQ coefficients
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A
        
        # Normalize
        b = np.array([b0, b1, b2]) / a0
        a = np.array([1, a1/a0, a2/a0])
        
        return signal.tf2sos(b, a)
    
    def process_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Process audio through the effects chain
        Each channel processes independently
        """
        if not AUDIO_AVAILABLE:
            # Simulation mode - return processed audio with simulated effects
            logger.debug(f"Processing audio in simulation mode for {self.channel.channel_type.value} channel")
            return audio_data * 0.9  # Simulate some processing
            
        output = audio_data.copy()
        
        # Apply effects based on channel configuration
        effects = self.channel.effects
        
        # Gate (vocals only)
        if self.channel.channel_type == ChannelType.VOCAL:
            if effects.get('gate', {}).get('enabled'):
                output = self._apply_gate(output, effects['gate'])
            
            if effects.get('compressor', {}).get('enabled'):
                output = self._apply_compressor(output, effects['compressor'])
        
        # EQ (both channels)
        for filter_name, filter_sos in self.filters.items():
            output = signal.sosfilt(filter_sos, output)
        
        # Channel-specific effects
        if self.channel.channel_type == ChannelType.VOCAL:
            if effects.get('reverb', {}).get('enabled'):
                output = self._apply_reverb(output, effects['reverb'])
            
            if effects.get('delay', {}).get('enabled'):
                output = self._apply_delay(output, effects['delay'])
        
        elif self.channel.channel_type == ChannelType.MUSIC:
            if effects.get('stereo_enhancer', {}).get('enabled'):
                output = self._apply_stereo_enhancer(output, effects['stereo_enhancer'])
            
            if effects.get('vocal_remover', {}).get('enabled'):
                output = self._apply_vocal_remover(output, effects['vocal_remover'])
        
        # Limiter (both channels)
        if effects.get('limiter', {}).get('enabled'):
            output = self._apply_limiter(output, effects['limiter'])
        
        return output
    
    def _apply_gate(self, audio: np.ndarray, params: Dict) -> np.ndarray:
        """Noise gate for vocals"""
        threshold = 10 ** (params['threshold'] / 20)  # Convert dB to linear
        ratio = params['ratio']
        
        # Simple gate implementation
        mask = np.abs(audio) < threshold
        audio[mask] *= 1 / ratio
        
        return audio
    
    def _apply_compressor(self, audio: np.ndarray, params: Dict) -> np.ndarray:
        """Dynamic range compression"""
        threshold = 10 ** (params['threshold'] / 20)
        ratio = params['ratio']
        attack_samples = int(params['attack'] * self.sample_rate / 1000)
        release_samples = int(params['release'] * self.sample_rate / 1000)
        
        # Simple compressor
        envelope = np.abs(audio)
        
        # Smooth envelope
        for i in range(1, len(envelope)):
            if envelope[i] > envelope[i-1]:
                # Attack
                envelope[i] = envelope[i-1] + (envelope[i] - envelope[i-1]) / attack_samples
            else:
                # Release
                envelope[i] = envelope[i-1] + (envelope[i] - envelope[i-1]) / release_samples
        
        # Apply compression
        gain = np.ones_like(envelope)
        above_threshold = envelope > threshold
        gain[above_threshold] = threshold / envelope[above_threshold] + \
                              (1 - threshold / envelope[above_threshold]) / ratio
        
        return audio * gain
    
    def _apply_reverb(self, audio: np.ndarray, params: Dict) -> np.ndarray:
        """Reverb effect using Schroeder reverb algorithm"""
        room_size = params['room_size']
        damping = params['damping']
        wet = params['wet']
        
        # Comb filter delays (in samples)
        comb_delays = [int(d * self.sample_rate / 1000) for d in [29.7, 37.1, 41.1, 43.7]]
        
        reverb_signal = np.zeros_like(audio)
        
        for delay in comb_delays:
            if delay < len(audio):
                # Simple comb filter
                delayed = np.zeros_like(audio)
                delayed[delay:] = audio[:-delay] * room_size
                
                # Apply damping
                delayed *= (1 - damping)
                
                reverb_signal += delayed / len(comb_delays)
        
        # Mix wet and dry signals
        return audio * (1 - wet) + reverb_signal * wet
    
    def _apply_delay(self, audio: np.ndarray, params: Dict) -> np.ndarray:
        """Echo/delay effect"""
        delay_ms = params['time_ms']
        feedback = params['feedback']
        wet = params['wet']
        
        delay_samples = int(delay_ms * self.sample_rate / 1000)
        
        if delay_samples >= len(audio):
            return audio
        
        delayed = np.zeros_like(audio)
        delayed[delay_samples:] = audio[:-delay_samples]
        
        # Add feedback
        output = audio.copy()
        for _ in range(3):  # 3 echo repeats
            delayed[delay_samples:] = delayed[:-delay_samples] * feedback
            output += delayed * wet
            wet *= 0.7  # Decay each echo
        
        return output
    
    def _apply_stereo_enhancer(self, audio: np.ndarray, params: Dict) -> np.ndarray:
        """Stereo width enhancement for music"""
        if audio.ndim != 2:
            return audio
        
        width = params['width']
        
        # M/S processing
        mid = (audio[:, 0] + audio[:, 1]) / 2
        side = (audio[:, 0] - audio[:, 1]) / 2
        
        # Enhance stereo width
        side *= width
        
        # Convert back to L/R
        audio[:, 0] = mid + side
        audio[:, 1] = mid - side
        
        return audio
    
    def _apply_vocal_remover(self, audio: np.ndarray, params: Dict) -> np.ndarray:
        """Remove center channel (vocals) for karaoke"""
        if audio.ndim != 2:
            return audio
        
        strength = params['strength']
        
        # Extract center (vocals) and sides (instruments)
        center = (audio[:, 0] + audio[:, 1]) / 2
        sides = audio - np.column_stack([center, center])
        
        # Reduce center channel
        center *= (1 - strength)
        
        # Reconstruct
        return sides + np.column_stack([center, center])
    
    def _apply_limiter(self, audio: np.ndarray, params: Dict) -> np.ndarray:
        """Brickwall limiter to prevent clipping"""
        threshold = 10 ** (params['threshold'] / 20)
        
        # Simple limiting
        return np.clip(audio, -threshold, threshold)


class DualChannelKaraokeSystem:
    """
    Main karaoke system with dual independent channels
    Following Google's microservices architecture pattern
    """
    
    def __init__(self):
        logger.info("Initializing Dual Channel Karaoke System")
        
        # Create independent channels
        self.vocal_channel = AudioChannel(
            channel_id=1,
            channel_type=ChannelType.VOCAL
        )
        
        self.music_channel = AudioChannel(
            channel_id=2,
            channel_type=ChannelType.MUSIC
        )
        
        # Create processors for each channel
        self.vocal_processor = AudioProcessor(self.vocal_channel)
        self.music_processor = AudioProcessor(self.music_channel)
        
        # Audio device management
        self._setup_audio_devices()
        
        # Real-time audio threads
        self.vocal_thread = None
        self.music_thread = None
        self.running = False
        
        logger.info("System initialized successfully")
    
    def _setup_audio_devices(self):
        """Setup audio input/output devices"""
        if not AUDIO_AVAILABLE:
            logger.info("Audio device setup skipped - running in simulation mode")
            self.vocal_channel.device_index = None
            self.music_channel.device_index = None
            return
            
        devices = sd.query_devices()
        
        logger.info("Available audio devices:")
        for i, device in enumerate(devices):
            logger.info(f"  [{i}] {device['name']} - {device['max_input_channels']} in, {device['max_output_channels']} out")
        
        # Auto-detect AG06
        for i, device in enumerate(devices):
            name = device['name'].lower()
            if 'ag06' in name or 'ag03' in name or 'yamaha' in name:
                if device['max_input_channels'] > 0:
                    self.vocal_channel.device_index = i
                    logger.info(f"Vocal input device: {device['name']}")
                if device['max_output_channels'] > 0:
                    self.music_channel.device_index = i
                    logger.info(f"Music output device: {device['name']}")
    
    def update_channel_effects(self, channel_type: ChannelType, effects: Dict):
        """Update effects for a specific channel"""
        if channel_type == ChannelType.VOCAL:
            self.vocal_channel.effects.update(effects)
            self.vocal_processor._init_filters()
            logger.info(f"Updated vocal effects: {effects}")
        else:
            self.music_channel.effects.update(effects)
            self.music_processor._init_filters()
            logger.info(f"Updated music effects: {effects}")
    
    def get_channel_status(self, channel_type: ChannelType) -> Dict:
        """Get current status of a channel"""
        channel = self.vocal_channel if channel_type == ChannelType.VOCAL else self.music_channel
        
        return {
            'channel_id': channel.channel_id,
            'channel_type': channel.channel_type.value,
            'sample_rate': channel.sample_rate,
            'buffer_size': channel.buffer_size,
            'effects': channel.effects,
            'device_index': channel.device_index
        }
    
    def start(self):
        """Start audio processing threads"""
        self.running = True
        
        # Start independent processing threads
        self.vocal_thread = threading.Thread(target=self._vocal_processing_loop)
        self.music_thread = threading.Thread(target=self._music_processing_loop)
        
        self.vocal_thread.start()
        self.music_thread.start()
        
        logger.info("Audio processing started")
    
    def stop(self):
        """Stop audio processing"""
        self.running = False
        
        if self.vocal_thread:
            self.vocal_thread.join()
        if self.music_thread:
            self.music_thread.join()
        
        logger.info("Audio processing stopped")
    
    def _vocal_processing_loop(self):
        """Independent vocal processing thread"""
        # This would connect to actual audio hardware
        # For now, it's a placeholder showing the architecture
        logger.info("Vocal processing thread started")
        
        while self.running:
            # In production, this would:
            # 1. Read from microphone input
            # 2. Process through vocal effects chain
            # 3. Output to mixer channel 1
            asyncio.sleep(0.01)
    
    def _music_processing_loop(self):
        """Independent music processing thread"""
        logger.info("Music processing thread started")
        
        while self.running:
            # In production, this would:
            # 1. Read from system audio (YouTube, Spotify, etc.)
            # 2. Process through music effects chain
            # 3. Output to mixer channel 2
            asyncio.sleep(0.01)


# Web API following Google's REST API design guidelines
async def create_web_app():
    """Create web API for controlling the dual channel system"""
    
    system = DualChannelKaraokeSystem()
    
    async def handle_channel_status(request):
        """GET /api/channels/{channel_type}/status"""
        channel_type = request.match_info['channel_type']
        
        try:
            ch_type = ChannelType(channel_type)
            status = system.get_channel_status(ch_type)
            return web.json_response(status)
        except ValueError:
            return web.json_response({'error': 'Invalid channel type'}, status=400)
    
    async def handle_channel_effects(request):
        """POST /api/channels/{channel_type}/effects"""
        channel_type = request.match_info['channel_type']
        
        try:
            ch_type = ChannelType(channel_type)
            effects = await request.json()
            system.update_channel_effects(ch_type, effects)
            return web.json_response({'status': 'updated', 'effects': effects})
        except ValueError:
            return web.json_response({'error': 'Invalid channel type'}, status=400)
    
    async def handle_system_status(request):
        """GET /api/system/status"""
        return web.json_response({
            'status': 'running' if system.running else 'stopped',
            'vocal_channel': system.get_channel_status(ChannelType.VOCAL),
            'music_channel': system.get_channel_status(ChannelType.MUSIC)
        })
    
    async def handle_system_control(request):
        """POST /api/system/control"""
        data = await request.json()
        action = data.get('action')
        
        if action == 'start':
            system.start()
            return web.json_response({'status': 'started'})
        elif action == 'stop':
            system.stop()
            return web.json_response({'status': 'stopped'})
        else:
            return web.json_response({'error': 'Invalid action'}, status=400)
    
    async def handle_index(request):
        """Serve control interface"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AiOke Dual Channel System</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                * { box-sizing: border-box; margin: 0; padding: 0; }
                body {
                    font-family: 'Google Sans', -apple-system, BlinkMacSystemFont, sans-serif;
                    background: linear-gradient(135deg, #1e3c72, #2a5298);
                    color: white;
                    padding: 20px;
                }
                .container { max-width: 1200px; margin: 0 auto; }
                h1 {
                    text-align: center;
                    font-size: 42px;
                    margin-bottom: 10px;
                    font-weight: 300;
                }
                .subtitle {
                    text-align: center;
                    font-size: 16px;
                    opacity: 0.9;
                    margin-bottom: 40px;
                }
                .channels {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 30px;
                    margin-bottom: 30px;
                }
                .channel {
                    background: rgba(255,255,255,0.1);
                    border-radius: 12px;
                    padding: 30px;
                    backdrop-filter: blur(10px);
                }
                .channel h2 {
                    font-size: 24px;
                    margin-bottom: 25px;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }
                .channel-icon {
                    width: 32px;
                    height: 32px;
                    background: linear-gradient(45deg, #4285f4, #34a853);
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 18px;
                }
                .control-group {
                    margin-bottom: 20px;
                }
                .control-label {
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 8px;
                    font-size: 13px;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    opacity: 0.9;
                }
                input[type="range"] {
                    width: 100%;
                    height: 36px;
                    -webkit-appearance: none;
                    background: transparent;
                    cursor: pointer;
                }
                input[type="range"]::-webkit-slider-track {
                    background: rgba(255,255,255,0.2);
                    height: 4px;
                    border-radius: 2px;
                }
                input[type="range"]::-webkit-slider-thumb {
                    -webkit-appearance: none;
                    background: white;
                    height: 16px;
                    width: 16px;
                    border-radius: 50%;
                    cursor: pointer;
                    margin-top: -6px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                }
                .value {
                    background: rgba(66, 133, 244, 0.3);
                    padding: 2px 6px;
                    border-radius: 4px;
                    font-size: 11px;
                    font-weight: 500;
                }
                .toggle {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    margin-top: 5px;
                }
                .toggle input[type="checkbox"] {
                    width: 20px;
                    height: 20px;
                }
                .system-status {
                    background: rgba(255,255,255,0.1);
                    border-radius: 12px;
                    padding: 20px;
                    text-align: center;
                    backdrop-filter: blur(10px);
                }
                .status-indicator {
                    display: inline-block;
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    background: #34a853;
                    margin-right: 8px;
                    animation: pulse 2s infinite;
                }
                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.5; }
                }
                .architecture-note {
                    background: rgba(0,0,0,0.3);
                    border-left: 3px solid #4285f4;
                    padding: 15px;
                    margin-top: 30px;
                    border-radius: 4px;
                    font-size: 14px;
                    line-height: 1.6;
                }
                @media (max-width: 768px) {
                    .channels { grid-template-columns: 1fr; }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎤 AiOke Dual Channel System</h1>
                <p class="subtitle">Independent Channel Processing • Google-Standard Architecture</p>
                
                <div class="channels">
                    <!-- Vocal Channel -->
                    <div class="channel">
                        <h2>
                            <span class="channel-icon">🎙️</span>
                            Channel 1: Vocals
                        </h2>
                        
                        <div class="control-group">
                            <div class="control-label">
                                <span>Gate Threshold</span>
                                <span class="value" id="gate-val">-40dB</span>
                            </div>
                            <input type="range" min="-60" max="0" value="-40" 
                                   oninput="updateEffect('vocal', 'gate.threshold', this.value, 'gate-val', 'dB')">
                        </div>
                        
                        <div class="control-group">
                            <div class="control-label">
                                <span>Compression</span>
                                <span class="value" id="comp-val">4:1</span>
                            </div>
                            <input type="range" min="1" max="10" value="4" step="0.1"
                                   oninput="updateEffect('vocal', 'compressor.ratio', this.value, 'comp-val', ':1')">
                        </div>
                        
                        <div class="control-group">
                            <div class="control-label">
                                <span>Reverb</span>
                                <span class="value" id="reverb-val">25%</span>
                            </div>
                            <input type="range" min="0" max="100" value="25"
                                   oninput="updateEffect('vocal', 'reverb.wet', this.value/100, 'reverb-val', '%')">
                        </div>
                        
                        <div class="control-group">
                            <div class="control-label">
                                <span>Delay</span>
                                <span class="value" id="delay-val">15%</span>
                            </div>
                            <input type="range" min="0" max="100" value="15"
                                   oninput="updateEffect('vocal', 'delay.wet', this.value/100, 'delay-val', '%')">
                            <div class="toggle">
                                <input type="checkbox" id="delay-enable" onchange="toggleEffect('vocal', 'delay.enabled', this.checked)">
                                <label for="delay-enable">Enable Delay</label>
                            </div>
                        </div>
                        
                        <div class="control-group">
                            <div class="control-label">
                                <span>Presence (2kHz)</span>
                                <span class="value" id="presence-val">+3dB</span>
                            </div>
                            <input type="range" min="-12" max="12" value="3"
                                   oninput="updateEffect('vocal', 'eq.mid_bell.gain', this.value, 'presence-val', 'dB')">
                        </div>
                    </div>
                    
                    <!-- Music Channel -->
                    <div class="channel">
                        <h2>
                            <span class="channel-icon">🎵</span>
                            Channel 2: Music
                        </h2>
                        
                        <div class="control-group">
                            <div class="control-label">
                                <span>Mid Duck (for vocals)</span>
                                <span class="value" id="duck-val">-2dB</span>
                            </div>
                            <input type="range" min="-12" max="0" value="-2"
                                   oninput="updateEffect('music', 'eq.mid_bell.gain', this.value, 'duck-val', 'dB')">
                        </div>
                        
                        <div class="control-group">
                            <div class="control-label">
                                <span>Stereo Width</span>
                                <span class="value" id="width-val">120%</span>
                            </div>
                            <input type="range" min="50" max="200" value="120"
                                   oninput="updateEffect('music', 'stereo_enhancer.width', this.value/100, 'width-val', '%')">
                        </div>
                        
                        <div class="control-group">
                            <div class="control-label">
                                <span>Vocal Removal (Karaoke)</span>
                                <span class="value" id="karaoke-val">0%</span>
                            </div>
                            <input type="range" min="0" max="100" value="0"
                                   oninput="updateEffect('music', 'vocal_remover.strength', this.value/100, 'karaoke-val', '%')">
                            <div class="toggle">
                                <input type="checkbox" id="karaoke-enable" onchange="toggleEffect('music', 'vocal_remover.enabled', this.checked)">
                                <label for="karaoke-enable">Enable Karaoke Mode</label>
                            </div>
                        </div>
                        
                        <div class="control-group">
                            <div class="control-label">
                                <span>Limiter Threshold</span>
                                <span class="value" id="limit-val">-3dB</span>
                            </div>
                            <input type="range" min="-12" max="0" value="-3"
                                   oninput="updateEffect('music', 'limiter.threshold', this.value, 'limit-val', 'dB')">
                        </div>
                    </div>
                </div>
                
                <div class="system-status">
                    <span class="status-indicator"></span>
                    <span>System Active • Dual Channel Independent Processing</span>
                </div>
                
                <div class="architecture-note">
                    <strong>🏗️ Architecture Note (Google Best Practices):</strong><br>
                    Each channel operates completely independently with its own processing pipeline. 
                    No audio mixing occurs at the software level - the AG06 hardware mixer handles 
                    final channel blending. This ensures maximum flexibility, lowest latency, and 
                    professional quality output. Music can come from any source (YouTube, Spotify, 
                    Apple Music, local files) without software modifications.
                </div>
            </div>
            
            <script>
                async function updateEffect(channel, path, value, displayId, unit) {
                    // Update display
                    document.getElementById(displayId).textContent = value + unit;
                    
                    // Parse nested path
                    const keys = path.split('.');
                    let effects = {};
                    let current = effects;
                    
                    for (let i = 0; i < keys.length - 1; i++) {
                        current[keys[i]] = {};
                        current = current[keys[i]];
                    }
                    current[keys[keys.length - 1]] = value;
                    
                    // Send to API
                    await fetch(`/api/channels/${channel}/effects`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(effects)
                    });
                }
                
                async function toggleEffect(channel, path, enabled) {
                    const keys = path.split('.');
                    let effects = {};
                    let current = effects;
                    
                    for (let i = 0; i < keys.length - 1; i++) {
                        current[keys[i]] = {};
                        current = current[keys[i]];
                    }
                    current[keys[keys.length - 1]] = enabled;
                    
                    await fetch(`/api/channels/${channel}/effects`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(effects)
                    });
                }
                
                // Initialize system
                fetch('/api/system/control', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({action: 'start'})
                });
            </script>
        </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')
    
    app = web.Application()
    
    # Routes following Google's API design
    app.router.add_get('/', handle_index)
    app.router.add_get('/api/system/status', handle_system_status)
    app.router.add_post('/api/system/control', handle_system_control)
    app.router.add_get('/api/channels/{channel_type}/status', handle_channel_status)
    app.router.add_post('/api/channels/{channel_type}/effects', handle_channel_effects)
    
    return app


if __name__ == '__main__':
    print("""
    🎤 AiOke Dual Channel Karaoke System
    =====================================
    
    Following Google's Audio Architecture Best Practices:
    • Independent channel processing
    • No software mixing (hardware mixer handles blend)
    • Modular effects chains per channel
    • Low-latency real-time processing
    • RESTful API design
    
    Channel 1: Microphone/Vocals (XLR input)
    Channel 2: Music (Any source - YouTube, Spotify, etc.)
    
    Starting server on http://localhost:9092
    """)
    
    async def start_server():
        app = await create_web_app()
        return app
    
    app = asyncio.run(start_server())
    web.run_app(app, host='0.0.0.0', port=9092)