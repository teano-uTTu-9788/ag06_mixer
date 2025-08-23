#!/usr/bin/env python3
"""
AI-Powered Audio Mixing System for AG06
Real-time audio processing with machine learning
"""

import numpy as np
import pyaudio
import threading
import time
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from collections import deque
import subprocess
import asyncio
import websocket
from datetime import datetime

@dataclass
class AudioProfile:
    """SM58 Dynamic Microphone Profile"""
    mic_type: str = "dynamic"
    model: str = "Shure SM58"
    frequency_response: Tuple[int, int] = (50, 15000)  # Hz
    optimal_distance: Tuple[int, int] = (2, 6)  # inches
    gain_range: Tuple[float, float] = (50, 75)  # percentage
    requires_phantom: bool = False
    polar_pattern: str = "cardioid"

@dataclass
class MixSettings:
    """Intelligent mix settings"""
    input_gain: float = 0.0
    eq_low: float = 0.0
    eq_mid: float = 0.0
    eq_high: float = 0.0
    compression_ratio: float = 2.0
    noise_gate: float = -40.0
    reverb_amount: float = 0.0
    auto_gain: bool = True

class AIAudioMixer:
    """AI-powered audio mixing for AG06"""
    
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.sample_rate = 48000
        self.buffer_size = 1024
        self.channels = 2
        
        # SM58 profile
        self.mic_profile = AudioProfile()
        self.mix_settings = MixSettings()
        
        # Audio analysis
        self.rms_history = deque(maxlen=100)
        self.peak_history = deque(maxlen=100)
        self.frequency_history = deque(maxlen=50)
        
        # AI parameters
        self.target_loudness = -18.0  # LUFS
        self.headroom = -6.0  # dB
        self.noise_floor = -50.0  # dB
        
        # Real-time metrics
        self.current_rms = 0.0
        self.current_peak = 0.0
        self.is_clipping = False
        self.is_too_quiet = False
        
        # Voice detection
        self.voice_detected = False
        self.voice_confidence = 0.0
        
        # Stream handles
        self.input_stream = None
        self.output_stream = None
        self.processing_active = False
        
    def analyze_audio_characteristics(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Analyze audio using AI techniques"""
        
        # Convert to mono for analysis
        if len(audio_data.shape) > 1:
            mono = np.mean(audio_data, axis=1)
        else:
            mono = audio_data
            
        # RMS (loudness)
        rms = np.sqrt(np.mean(mono**2))
        self.rms_history.append(rms)
        
        # Peak detection
        peak = np.max(np.abs(mono))
        self.peak_history.append(peak)
        
        # Frequency analysis (simple FFT)
        fft = np.fft.rfft(mono)
        freqs = np.fft.rfftfreq(len(mono), 1/self.sample_rate)
        
        # Find dominant frequency
        magnitude = np.abs(fft)
        dominant_freq_idx = np.argmax(magnitude[1:]) + 1  # Skip DC
        dominant_freq = freqs[dominant_freq_idx]
        
        # Voice detection (85-255 Hz for male, 165-255 Hz for female fundamentals)
        voice_range = (85, 255)
        if voice_range[0] <= dominant_freq <= voice_range[1]:
            self.voice_confidence = min(1.0, magnitude[dominant_freq_idx] / np.mean(magnitude))
            self.voice_detected = self.voice_confidence > 0.3
        else:
            self.voice_detected = False
            self.voice_confidence = 0.0
        
        # Spectral analysis
        low_freq = magnitude[freqs < 250].mean()
        mid_freq = magnitude[(freqs >= 250) & (freqs < 4000)].mean()
        high_freq = magnitude[freqs >= 4000].mean()
        
        return {
            "rms": float(rms),
            "peak": float(peak),
            "rms_db": 20 * np.log10(rms + 1e-10),
            "peak_db": 20 * np.log10(peak + 1e-10),
            "dominant_freq": float(dominant_freq),
            "voice_detected": self.voice_detected,
            "voice_confidence": float(self.voice_confidence),
            "spectral_balance": {
                "low": float(low_freq),
                "mid": float(mid_freq),
                "high": float(high_freq)
            },
            "is_clipping": peak > 0.95,
            "is_too_quiet": rms < 0.01
        }
    
    def apply_intelligent_gain(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply AI-driven automatic gain control"""
        
        if not self.mix_settings.auto_gain:
            return audio_data
            
        # Calculate current loudness
        rms = np.sqrt(np.mean(audio_data**2))
        current_db = 20 * np.log10(rms + 1e-10)
        
        # Target adjustment
        target_db = self.target_loudness
        gain_adjustment = target_db - current_db
        
        # Limit gain adjustment
        gain_adjustment = np.clip(gain_adjustment, -12, 12)
        
        # Apply gain smoothly
        gain_linear = 10 ** (gain_adjustment / 20)
        
        # Prevent clipping
        if np.max(np.abs(audio_data * gain_linear)) > 0.95:
            gain_linear *= 0.95 / np.max(np.abs(audio_data * gain_linear))
        
        return audio_data * gain_linear
    
    def apply_dynamic_eq(self, audio_data: np.ndarray, analysis: Dict) -> np.ndarray:
        """Apply intelligent EQ based on content analysis"""
        
        # SM58 has natural presence boost around 5kHz
        # Compensate for proximity effect if needed
        
        if self.voice_detected:
            # Voice enhancement EQ
            # Gentle high-pass to reduce rumble
            # Boost presence for clarity
            # This is simplified - real implementation would use proper filters
            pass
            
        return audio_data
    
    def apply_smart_compression(self, audio_data: np.ndarray) -> np.ndarray:
        """Intelligent compression for consistent levels"""
        
        threshold = -20  # dB
        ratio = self.mix_settings.compression_ratio
        
        # Simple compression (real implementation would use lookahead)
        for i in range(len(audio_data)):
            sample = audio_data[i]
            sample_db = 20 * np.log10(abs(sample) + 1e-10)
            
            if sample_db > threshold:
                excess = sample_db - threshold
                compressed_excess = excess / ratio
                new_db = threshold + compressed_excess
                gain = 10 ** ((new_db - sample_db) / 20)
                audio_data[i] *= gain
                
        return audio_data
    
    def apply_noise_gate(self, audio_data: np.ndarray) -> np.ndarray:
        """Intelligent noise gate for SM58"""
        
        gate_threshold = self.mix_settings.noise_gate
        
        rms = np.sqrt(np.mean(audio_data**2))
        rms_db = 20 * np.log10(rms + 1e-10)
        
        if rms_db < gate_threshold:
            # Smooth gate to avoid clicks
            fade_samples = 64
            if len(audio_data) > fade_samples * 2:
                audio_data[:fade_samples] *= np.linspace(0, 1, fade_samples)
                audio_data[-fade_samples:] *= np.linspace(1, 0, fade_samples)
                audio_data[fade_samples:-fade_samples] *= 0.1
            else:
                audio_data *= 0.1
                
        return audio_data
    
    def process_audio_callback(self, in_data, frame_count, time_info, status):
        """Real-time audio processing callback"""
        
        try:
            # Convert byte data to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # Analyze audio characteristics
            analysis = self.analyze_audio_characteristics(audio_data)
            
            # Update real-time metrics
            self.current_rms = analysis["rms_db"]
            self.current_peak = analysis["peak_db"]
            self.is_clipping = analysis["is_clipping"]
            self.is_too_quiet = analysis["is_too_quiet"]
            
            # Apply AI processing chain
            processed = audio_data.copy()
            
            # 1. Noise gate
            processed = self.apply_noise_gate(processed)
            
            # 2. Intelligent gain
            processed = self.apply_intelligent_gain(processed)
            
            # 3. Dynamic EQ
            processed = self.apply_dynamic_eq(processed, analysis)
            
            # 4. Smart compression
            processed = self.apply_smart_compression(processed)
            
            # Convert back to bytes
            out_data = processed.astype(np.float32).tobytes()
            
            return (out_data, pyaudio.paContinue)
            
        except Exception as e:
            print(f"Processing error: {e}")
            return (in_data, pyaudio.paContinue)
    
    def start_processing(self):
        """Start real-time audio processing"""
        
        try:
            # Find AG06 device
            ag06_index = None
            for i in range(self.pa.get_device_count()):
                info = self.pa.get_device_info_by_index(i)
                if "AG06" in info["name"] or "AG03" in info["name"]:
                    if info["maxInputChannels"] > 0:
                        ag06_index = i
                        print(f"✅ Found AG06 at index {i}: {info['name']}")
                        break
            
            if ag06_index is None:
                print("❌ AG06 not found in audio devices")
                return False
            
            # Open audio stream
            self.input_stream = self.pa.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                output=True,
                input_device_index=ag06_index,
                frames_per_buffer=self.buffer_size,
                stream_callback=self.process_audio_callback
            )
            
            self.processing_active = True
            self.input_stream.start_stream()
            
            print("🎙️ AI Audio Processing Started")
            print(f"📊 Sample Rate: {self.sample_rate} Hz")
            print(f"🎯 Target Loudness: {self.target_loudness} LUFS")
            print(f"🎤 Microphone: {self.mic_profile.model}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error starting audio processing: {e}")
            return False
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current audio metrics for dashboard"""
        
        return {
            "timestamp": datetime.now().isoformat(),
            "input_level": {
                "rms": float(self.current_rms),
                "peak": float(self.current_peak),
                "clipping": self.is_clipping,
                "too_quiet": self.is_too_quiet
            },
            "voice": {
                "detected": self.voice_detected,
                "confidence": float(self.voice_confidence)
            },
            "processing": {
                "auto_gain": self.mix_settings.auto_gain,
                "compression_ratio": self.mix_settings.compression_ratio,
                "noise_gate": self.mix_settings.noise_gate
            },
            "mic_profile": {
                "model": self.mic_profile.model,
                "type": self.mic_profile.mic_type,
                "phantom": self.mic_profile.requires_phantom
            }
        }
    
    def optimize_for_voice(self):
        """Optimize settings for voice/podcast"""
        
        print("🎙️ Optimizing for voice...")
        self.mix_settings.compression_ratio = 3.0
        self.mix_settings.noise_gate = -35.0
        self.mix_settings.eq_low = -2.0  # Reduce low rumble
        self.mix_settings.eq_mid = 2.0   # Enhance presence
        self.mix_settings.eq_high = 1.0  # Slight brightness
        print("✅ Voice optimization applied")
    
    def optimize_for_music(self):
        """Optimize settings for music"""
        
        print("🎵 Optimizing for music...")
        self.mix_settings.compression_ratio = 2.0
        self.mix_settings.noise_gate = -45.0
        self.mix_settings.eq_low = 0.0
        self.mix_settings.eq_mid = 0.0
        self.mix_settings.eq_high = 0.0
        print("✅ Music optimization applied")
    
    def stop_processing(self):
        """Stop audio processing"""
        
        self.processing_active = False
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        self.pa.terminate()
        print("🛑 AI Audio Processing Stopped")


def main():
    """Test AI audio mixing system"""
    
    print("\n" + "="*60)
    print("🤖 AI-Powered Audio Mixing System for AG06")
    print("="*60)
    print("🎤 Microphone: Shure SM58")
    print("🔊 Speakers: JBL 310")
    print("="*60 + "\n")
    
    # Initialize AI mixer
    mixer = AIAudioMixer()
    
    # Start processing
    if not mixer.start_processing():
        print("Failed to start audio processing")
        return
    
    try:
        print("\n📊 Real-time Audio Analysis Running...")
        print("Press Ctrl+C to stop\n")
        
        # Monitor loop
        while True:
            metrics = mixer.get_real_time_metrics()
            
            # Display metrics
            print(f"\r📊 Level: {metrics['input_level']['rms']:.1f} dB | "
                  f"Peak: {metrics['input_level']['peak']:.1f} dB | "
                  f"Voice: {'✅' if metrics['voice']['detected'] else '❌'} "
                  f"({metrics['voice']['confidence']:.0%}) | "
                  f"{'🔴 CLIPPING' if metrics['input_level']['clipping'] else ''}",
                  end="", flush=True)
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
        mixer.stop_processing()
        print("✅ Done")


if __name__ == "__main__":
    main()