#!/usr/bin/env python3
"""
AI-Powered Karaoke Auto-Mixer for AG06
Automatically enhances vocals in real-time for the best karaoke experience
"""

import numpy as np
import pyaudio
import threading
import queue
import time
from scipy import signal
from scipy.fft import rfft, rfftfreq, irfft
from typing import Optional, Tuple
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIKaraokeAutoMixer:
    """AI-powered auto-mixing system for karaoke vocals"""
    
    def __init__(self):
        self.sample_rate = 44100
        self.chunk_size = 1024
        self.channels = 2
        self.format = pyaudio.paFloat32
        
        # Audio interface
        self.p = pyaudio.PyAudio()
        self.stream = None
        
        # Processing parameters (AI-optimized)
        self.vocal_params = {
            'eq_bands': {
                'low_cut': 80,  # Remove rumble
                'presence_boost': (2000, 5000, 3),  # Boost vocal presence (Hz, Hz, dB)
                'air_boost': (10000, 16000, 2),  # Add "air" to vocals
                'warmth': (200, 400, 1.5),  # Add warmth
            },
            'compression': {
                'threshold': -20,  # dB
                'ratio': 3,
                'attack': 5,  # ms
                'release': 100,  # ms
                'knee': 2,  # dB
                'makeup_gain': 3  # dB
            },
            'effects': {
                'reverb_mix': 0.25,  # 25% reverb for spaciousness
                'reverb_size': 0.7,
                'delay_time': 0.08,  # 80ms delay
                'delay_feedback': 0.3,
                'delay_mix': 0.15,
                'chorus_depth': 0.2,
                'chorus_rate': 1.5,  # Hz
            },
            'auto_tune': {
                'enabled': True,
                'strength': 0.7,  # 0-1, how much to correct pitch
                'speed': 20,  # ms, how fast to correct
                'key': 'C',  # Default key
                'scale': 'major'
            },
            'enhancement': {
                'de_esser': True,
                'de_esser_threshold': -25,
                'gate_threshold': -40,  # Noise gate
                'exciter_amount': 0.15,  # Harmonic excitement
                'stereo_width': 1.2
            }
        }
        
        # AI learning parameters
        self.ai_params = {
            'vocal_detection_threshold': 0.3,
            'feedback_suppression': True,
            'room_correction': True,
            'adaptive_eq': True,
            'auto_gain': True,
            'target_loudness': -18  # LUFS
        }
        
        # Processing state
        self.is_running = False
        self.audio_queue = queue.Queue(maxsize=10)
        self.processed_queue = queue.Queue(maxsize=10)
        
        # Analysis buffers
        self.pitch_buffer = []
        self.level_buffer = []
        self.spectrum_buffer = []
        
    def find_ag06_device(self) -> Optional[Tuple[int, int]]:
        """Find AG06 device indices"""
        input_idx = None
        output_idx = None
        
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            if 'AG06' in info.get('name', '') or 'AG03' in info.get('name', ''):
                if info['maxInputChannels'] > 0:
                    input_idx = i
                    logger.info(f"Found AG06 input: {info['name']} (index {i})")
                if info['maxOutputChannels'] > 0:
                    output_idx = i
                    logger.info(f"Found AG06 output: {info['name']} (index {i})")
        
        return (input_idx, output_idx) if input_idx and output_idx else None
    
    def apply_eq(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply intelligent EQ for vocal enhancement"""
        # High-pass filter to remove rumble
        sos = signal.butter(2, self.vocal_params['eq_bands']['low_cut'], 
                           btype='highpass', fs=self.sample_rate, output='sos')
        audio_data = signal.sosfilt(sos, audio_data)
        
        # Presence boost (2-5kHz)
        low, high, gain = self.vocal_params['eq_bands']['presence_boost']
        sos = signal.butter(2, [low, high], btype='bandpass', 
                           fs=self.sample_rate, output='sos')
        presence = signal.sosfilt(sos, audio_data)
        audio_data += presence * (10**(gain/20) - 1)
        
        # Air boost (10-16kHz)
        low, high, gain = self.vocal_params['eq_bands']['air_boost']
        sos = signal.butter(2, [low, high], btype='bandpass',
                           fs=self.sample_rate, output='sos')
        air = signal.sosfilt(sos, audio_data)
        audio_data += air * (10**(gain/20) - 1)
        
        # Warmth (200-400Hz)
        low, high, gain = self.vocal_params['eq_bands']['warmth']
        sos = signal.butter(2, [low, high], btype='bandpass',
                           fs=self.sample_rate, output='sos')
        warmth = signal.sosfilt(sos, audio_data)
        audio_data += warmth * (10**(gain/20) - 1)
        
        return audio_data
    
    def apply_compression(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply dynamic compression for consistent vocal levels"""
        comp = self.vocal_params['compression']
        threshold = 10**(comp['threshold']/20)
        ratio = comp['ratio']
        makeup = 10**(comp['makeup_gain']/20)
        
        # Simple compression
        mask = np.abs(audio_data) > threshold
        compressed = audio_data.copy()
        compressed[mask] = threshold + (audio_data[mask] - threshold) / ratio
        compressed *= makeup
        
        return compressed
    
    def apply_reverb(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply reverb for spaciousness"""
        reverb_mix = self.vocal_params['effects']['reverb_mix']
        reverb_size = self.vocal_params['effects']['reverb_size']
        
        # Simple reverb using delays
        delays = [0.03, 0.05, 0.07, 0.09]  # Multiple delay taps
        reverb = np.zeros_like(audio_data)
        
        for delay_time in delays:
            delay_samples = int(delay_time * self.sample_rate)
            if delay_samples < len(audio_data):
                delayed = np.zeros_like(audio_data)
                delayed[delay_samples:] = audio_data[:-delay_samples] * reverb_size
                reverb += delayed * 0.25  # Mix delays equally
        
        return audio_data * (1 - reverb_mix) + reverb * reverb_mix
    
    def apply_delay(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply delay effect for depth"""
        delay_time = self.vocal_params['effects']['delay_time']
        delay_feedback = self.vocal_params['effects']['delay_feedback']
        delay_mix = self.vocal_params['effects']['delay_mix']
        
        delay_samples = int(delay_time * self.sample_rate)
        delayed = np.zeros_like(audio_data)
        
        if delay_samples < len(audio_data):
            delayed[delay_samples:] = audio_data[:-delay_samples] * delay_feedback
        
        return audio_data * (1 - delay_mix) + delayed * delay_mix
    
    def apply_auto_tune(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply pitch correction for better singing"""
        if not self.vocal_params['auto_tune']['enabled']:
            return audio_data
        
        strength = self.vocal_params['auto_tune']['strength']
        
        # FFT for pitch detection and correction
        fft = rfft(audio_data)
        freqs = rfftfreq(len(audio_data), 1/self.sample_rate)
        
        # Find dominant frequency (simplified pitch detection)
        magnitude = np.abs(fft)
        fundamental_idx = np.argmax(magnitude[1:500]) + 1  # Look for fundamental in vocal range
        fundamental_freq = freqs[fundamental_idx]
        
        # Find nearest note frequency (A440 tuning)
        if fundamental_freq > 80 and fundamental_freq < 2000:  # Vocal range
            # Calculate nearest note
            A4 = 440
            semitones_from_A4 = 12 * np.log2(fundamental_freq / A4)
            nearest_semitone = round(semitones_from_A4)
            target_freq = A4 * (2 ** (nearest_semitone / 12))
            
            # Apply pitch shift
            if abs(fundamental_freq - target_freq) > 1:
                shift_ratio = target_freq / fundamental_freq
                shift_amount = (shift_ratio - 1) * strength + 1
                
                # Simple pitch shift using FFT
                shifted_fft = np.zeros_like(fft)
                for i in range(len(fft)):
                    new_idx = int(i * shift_amount)
                    if 0 <= new_idx < len(shifted_fft):
                        shifted_fft[new_idx] = fft[i]
                
                audio_data = irfft(shifted_fft, n=len(audio_data))
        
        return audio_data
    
    def apply_de_esser(self, audio_data: np.ndarray) -> np.ndarray:
        """Reduce harsh sibilance in vocals"""
        if not self.vocal_params['enhancement']['de_esser']:
            return audio_data
        
        # De-essing frequency range (5-8kHz)
        sos = signal.butter(2, [5000, 8000], btype='bandpass',
                           fs=self.sample_rate, output='sos')
        sibilance = signal.sosfilt(sos, audio_data)
        
        # Compress sibilance
        threshold = 10**(self.vocal_params['enhancement']['de_esser_threshold']/20)
        mask = np.abs(sibilance) > threshold
        sibilance[mask] *= 0.5  # Reduce by 50%
        
        # Subtract processed sibilance from original
        audio_data -= sibilance * 0.5
        
        return audio_data
    
    def apply_exciter(self, audio_data: np.ndarray) -> np.ndarray:
        """Add harmonic excitement for brightness"""
        amount = self.vocal_params['enhancement']['exciter_amount']
        
        # Generate harmonics using soft clipping
        excited = np.tanh(audio_data * 3) * amount
        
        # High-pass filter the excitement
        sos = signal.butter(2, 3000, btype='highpass',
                           fs=self.sample_rate, output='sos')
        excited = signal.sosfilt(sos, excited)
        
        return audio_data + excited
    
    def process_audio(self, input_data: np.ndarray) -> np.ndarray:
        """Main AI processing pipeline"""
        # Convert to mono for processing
        if len(input_data.shape) > 1:
            mono = np.mean(input_data, axis=1)
        else:
            mono = input_data
        
        # Apply processing chain
        processed = mono
        
        # 1. Noise gate
        gate_threshold = 10**(self.vocal_params['enhancement']['gate_threshold']/20)
        mask = np.abs(processed) < gate_threshold
        processed[mask] *= 0.1
        
        # 2. EQ
        processed = self.apply_eq(processed)
        
        # 3. Compression
        processed = self.apply_compression(processed)
        
        # 4. De-esser
        processed = self.apply_de_esser(processed)
        
        # 5. Auto-tune
        processed = self.apply_auto_tune(processed)
        
        # 6. Exciter
        processed = self.apply_exciter(processed)
        
        # 7. Reverb
        processed = self.apply_reverb(processed)
        
        # 8. Delay
        processed = self.apply_delay(processed)
        
        # 9. Auto-gain to target loudness
        if self.ai_params['auto_gain']:
            current_level = np.sqrt(np.mean(processed**2))
            target_level = 10**(self.ai_params['target_loudness']/20)
            if current_level > 0:
                gain = target_level / current_level
                gain = np.clip(gain, 0.1, 10)  # Limit gain range
                processed *= gain
        
        # Convert back to stereo with width enhancement
        stereo_width = self.vocal_params['enhancement']['stereo_width']
        if len(input_data.shape) > 1:
            # Create stereo from mono with width
            left = processed * (1 + (stereo_width - 1) * 0.5)
            right = processed * (1 - (stereo_width - 1) * 0.5)
            processed = np.column_stack([left, right])
        
        # Prevent clipping
        processed = np.clip(processed, -1.0, 1.0)
        
        return processed
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Real-time audio callback"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Convert input data
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # Process audio through AI pipeline
        processed = self.process_audio(audio_data)
        
        # Convert back to bytes
        out_data = processed.astype(np.float32).tobytes()
        
        return (out_data, pyaudio.paContinue)
    
    def start(self):
        """Start the AI karaoke mixer"""
        device_info = self.find_ag06_device()
        
        if not device_info:
            logger.error("AG06 device not found! Please connect the AG06.")
            return False
        
        input_idx, output_idx = device_info
        
        try:
            # Open audio stream
            self.stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                output=True,
                input_device_index=input_idx,
                output_device_index=output_idx,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback
            )
            
            self.is_running = True
            self.stream.start_stream()
            
            logger.info("🎤 AI Karaoke Auto-Mixer started!")
            logger.info("Settings:")
            logger.info(f"  - Auto-tune: {'ON' if self.vocal_params['auto_tune']['enabled'] else 'OFF'}")
            logger.info(f"  - Reverb: {self.vocal_params['effects']['reverb_mix']*100:.0f}%")
            logger.info(f"  - Compression: {self.vocal_params['compression']['ratio']}:1")
            logger.info(f"  - Target loudness: {self.ai_params['target_loudness']} LUFS")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            return False
    
    def stop(self):
        """Stop the mixer"""
        self.is_running = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        logger.info("AI Karaoke Mixer stopped")
    
    def update_settings(self, settings: dict):
        """Update mixer settings in real-time"""
        if 'reverb' in settings:
            self.vocal_params['effects']['reverb_mix'] = settings['reverb']
        if 'auto_tune' in settings:
            self.vocal_params['auto_tune']['enabled'] = settings['auto_tune']
        if 'compression' in settings:
            self.vocal_params['compression']['ratio'] = settings['compression']
        
        logger.info(f"Settings updated: {settings}")
    
    def get_status(self) -> dict:
        """Get current mixer status"""
        return {
            'running': self.is_running,
            'auto_tune': self.vocal_params['auto_tune']['enabled'],
            'reverb': self.vocal_params['effects']['reverb_mix'],
            'compression': self.vocal_params['compression']['ratio'],
            'target_loudness': self.ai_params['target_loudness']
        }

def main():
    """Main entry point for testing"""
    mixer = AIKaraokeAutoMixer()
    
    print("🎤 AG06 AI Karaoke Auto-Mixer")
    print("=" * 40)
    print("This system automatically enhances your voice")
    print("for the best karaoke experience!")
    print("=" * 40)
    
    if mixer.start():
        print("\n✅ Mixer is running!")
        print("\nFeatures active:")
        print("  • AI Auto-tune for pitch correction")
        print("  • Smart EQ for vocal clarity")
        print("  • Dynamic compression for consistency")
        print("  • Studio reverb for professional sound")
        print("  • De-esser for smooth vocals")
        print("  • Harmonic exciter for brightness")
        print("\nSing into your microphone and hear the magic! 🎵")
        print("\nPress Ctrl+C to stop...")
        
        try:
            while mixer.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\nStopping mixer...")
            mixer.stop()
    else:
        print("❌ Failed to start mixer. Check AG06 connection.")
    
    mixer.p.terminate()

if __name__ == "__main__":
    main()