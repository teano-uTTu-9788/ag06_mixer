#!/usr/bin/env python3
"""
Karaoke Vocal Enhancer - Makes singing easier and sound better!
Optimized reverb, pitch correction, and vocal effects for beginners
"""

import numpy as np
import sounddevice as sd
from scipy import signal
from scipy.signal import butter, lfilter, sosfilt
import threading
import queue
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List
import json

@dataclass
class KaraokePreset:
    """Preset settings for different singing styles"""
    name: str
    reverb_amount: float      # 0-100%
    reverb_room_size: float   # Small/Medium/Large
    echo_delay: float         # ms
    echo_feedback: float      # 0-100%
    pitch_correction: float   # 0-100% strength
    vocal_warmth: float       # 0-100%
    doubler: bool            # Voice doubling effect
    confidence_boost: float   # Overall enhancement level

# Beginner-friendly presets
KARAOKE_PRESETS = {
    "Shower Singer": KaraokePreset(
        name="Shower Singer",
        reverb_amount=40,        # Generous reverb like a bathroom
        reverb_room_size=0.3,    # Small room
        echo_delay=100,
        echo_feedback=20,
        pitch_correction=60,     # Moderate pitch help
        vocal_warmth=30,
        doubler=False,
        confidence_boost=70
    ),
    "Studio Pro": KaraokePreset(
        name="Studio Pro",
        reverb_amount=25,        # Professional amount
        reverb_room_size=0.5,    # Medium room
        echo_delay=0,
        echo_feedback=0,
        pitch_correction=80,     # Strong pitch correction
        vocal_warmth=50,
        doubler=True,           # Thicken voice
        confidence_boost=85
    ),
    "Concert Hall": KaraokePreset(
        name="Concert Hall",
        reverb_amount=60,        # Big reverb
        reverb_room_size=0.9,    # Large hall
        echo_delay=250,
        echo_feedback=30,
        pitch_correction=40,
        vocal_warmth=20,
        doubler=False,
        confidence_boost=90
    ),
    "Radio Voice": KaraokePreset(
        name="Radio Voice",
        reverb_amount=15,        # Minimal reverb
        reverb_room_size=0.2,
        echo_delay=0,
        echo_feedback=0,
        pitch_correction=70,
        vocal_warmth=60,        # Warm radio sound
        doubler=False,
        confidence_boost=60
    ),
    "Karaoke King": KaraokePreset(
        name="Karaoke King",
        reverb_amount=35,        # Perfect karaoke balance
        reverb_room_size=0.6,
        echo_delay=150,
        echo_feedback=25,
        pitch_correction=75,     # Good pitch help
        vocal_warmth=40,
        doubler=True,
        confidence_boost=95      # Maximum confidence!
    )
}

class EnhancedReverb:
    """High-quality reverb optimized for vocals"""
    
    def __init__(self, sample_rate: int = 44100):
        self.fs = sample_rate
        
        # Multiple delay lines for rich reverb
        self.delays = []
        self.indices = []
        
        # Create delay network
        delay_times = [
            0.003, 0.005, 0.007, 0.011,  # Very early reflections
            0.013, 0.017, 0.019, 0.023,  # Early reflections
            0.029, 0.031, 0.037, 0.041,  # Medium reflections
            0.043, 0.047, 0.053, 0.059,  # Late reflections
            0.061, 0.067, 0.071, 0.073   # Very late reflections
        ]
        
        for delay_time in delay_times:
            delay_samples = int(delay_time * sample_rate)
            self.delays.append(np.zeros(delay_samples))
            self.indices.append(0)
        
        # Allpass filters for smooth diffusion
        self.ap_delays = []
        self.ap_indices = []
        ap_times = [0.005, 0.0067, 0.0073, 0.0089]
        
        for ap_time in ap_times:
            ap_samples = int(ap_time * sample_rate)
            self.ap_delays.append(np.zeros(ap_samples))
            self.ap_indices.append(0)
        
        # Pre-delay buffer (gives space before reverb)
        self.predelay = np.zeros(int(0.02 * sample_rate))  # 20ms
        self.predelay_idx = 0
        
        # Damping for natural sound
        self.damping_state = 0.0
        
    def process(self, audio: np.ndarray, amount: float, room_size: float) -> np.ndarray:
        """Apply lush reverb perfect for vocals"""
        
        if amount <= 0:
            return audio
        
        output = np.zeros_like(audio)
        
        # Scale parameters
        feedback = min(0.95, room_size * 0.9)  # Prevent infinite reverb
        damping = 0.3 + (1 - room_size) * 0.4
        wet_gain = amount / 100.0
        
        for i in range(len(audio)):
            # Pre-delay
            delayed_input = self.predelay[self.predelay_idx]
            self.predelay[self.predelay_idx] = audio[i]
            self.predelay_idx = (self.predelay_idx + 1) % len(self.predelay)
            
            # Sum all delays with different gains
            reverb_sum = 0.0
            for j, delay_line in enumerate(self.delays):
                # Get delayed sample
                delayed = delay_line[self.indices[j]]
                
                # Different gain for each delay (decreasing over time)
                gain = 1.0 / (j + 1)
                reverb_sum += delayed * gain
                
                # Update delay line with feedback
                if j < 8:  # Only early reflections get input
                    delay_line[self.indices[j]] = delayed_input * 0.5 + delayed * feedback * 0.7
                else:  # Late reflections just feedback
                    delay_line[self.indices[j]] = reverb_sum * feedback * 0.3
                
                self.indices[j] = (self.indices[j] + 1) % len(delay_line)
            
            # Normalize
            reverb_sum /= len(self.delays)
            
            # Apply damping (gentle low-pass filter)
            self.damping_state = reverb_sum * (1 - damping) + self.damping_state * damping
            reverb_sum = self.damping_state
            
            # Allpass diffusion for smoothness
            ap_out = reverb_sum
            for j, ap_delay in enumerate(self.ap_delays):
                delayed_ap = ap_delay[self.ap_indices[j]]
                ap_in = ap_out + delayed_ap * 0.5
                ap_delay[self.ap_indices[j]] = ap_in
                ap_out = delayed_ap - ap_in * 0.5
                self.ap_indices[j] = (self.ap_indices[j] + 1) % len(ap_delay)
            
            # Mix wet and dry (keep original voice clear)
            output[i] = audio[i] + ap_out * wet_gain
        
        return output

class PitchCorrection:
    """Auto-tune style pitch correction to help singers stay in key"""
    
    def __init__(self, sample_rate: int = 44100):
        self.fs = sample_rate
        self.buffer_size = 2048
        self.hop_size = 512
        self.window = np.hanning(self.buffer_size)
        
        # Musical notes frequencies (A4 = 440 Hz)
        self.note_frequencies = self.generate_note_frequencies()
        
    def generate_note_frequencies(self) -> List[float]:
        """Generate frequencies for musical notes"""
        # C, C#, D, D#, E, F, F#, G, G#, A, A#, B
        a4_freq = 440.0
        notes = []
        
        # Generate 8 octaves of notes
        for octave in range(-4, 4):
            for semitone in range(12):
                # A4 is the 9th semitone in octave 4
                n = octave * 12 + semitone - 9
                freq = a4_freq * (2 ** (n / 12))
                if 80 <= freq <= 2000:  # Vocal range
                    notes.append(freq)
        
        return sorted(notes)
    
    def find_nearest_note(self, frequency: float) -> float:
        """Find the nearest musical note frequency"""
        if frequency <= 0:
            return frequency
        
        min_diff = float('inf')
        nearest = frequency
        
        for note_freq in self.note_frequencies:
            diff = abs(frequency - note_freq)
            if diff < min_diff:
                min_diff = diff
                nearest = note_freq
        
        return nearest
    
    def detect_pitch(self, audio: np.ndarray) -> float:
        """Detect fundamental frequency using autocorrelation"""
        
        if len(audio) < self.buffer_size:
            return 0.0
        
        # Window the signal
        windowed = audio[-self.buffer_size:] * self.window
        
        # Autocorrelation
        correlation = np.correlate(windowed, windowed, mode='full')
        correlation = correlation[len(correlation)//2:]
        
        # Find first peak after zero lag
        min_period = int(self.fs / 1000)  # 1000 Hz max
        max_period = int(self.fs / 80)     # 80 Hz min
        
        if max_period >= len(correlation):
            return 0.0
        
        # Find peak in valid range
        peak_idx = min_period + np.argmax(correlation[min_period:max_period])
        
        if correlation[peak_idx] > 0:
            frequency = self.fs / peak_idx
            return frequency
        
        return 0.0
    
    def process(self, audio: np.ndarray, correction_amount: float) -> np.ndarray:
        """Apply pitch correction"""
        
        if correction_amount <= 0:
            return audio
        
        # Simple pitch shifting using phase vocoder
        output = np.copy(audio)
        
        # Detect pitch
        detected_freq = self.detect_pitch(audio)
        
        if detected_freq > 0:
            # Find nearest note
            target_freq = self.find_nearest_note(detected_freq)
            
            # Calculate pitch shift ratio
            shift_ratio = target_freq / detected_freq
            
            # Apply correction based on amount (0-100%)
            correction_ratio = 1.0 + (shift_ratio - 1.0) * (correction_amount / 100.0)
            
            # Simple pitch shift (simplified for real-time)
            if abs(correction_ratio - 1.0) > 0.01:
                # Resample for pitch shift
                if correction_ratio > 1.0:
                    # Pitch up - stretch then resample
                    stretched = signal.resample(audio, int(len(audio) * correction_ratio))
                    output = stretched[:len(audio)]
                else:
                    # Pitch down - resample then stretch  
                    resampled = signal.resample(audio, int(len(audio) * correction_ratio))
                    output = np.pad(resampled, (0, len(audio) - len(resampled)))
        
        return output

class VocalEnhancer:
    """Makes vocals sound warmer and fuller"""
    
    def __init__(self, sample_rate: int = 44100):
        self.fs = sample_rate
        
        # EQ bands for vocal enhancement
        # Low: Warmth (100-300 Hz)
        # Mid: Body (300-2000 Hz)  
        # High: Presence (2000-8000 Hz)
        # Air: Sparkle (8000+ Hz)
        
    def add_warmth(self, audio: np.ndarray, amount: float) -> np.ndarray:
        """Add warmth to vocals (enhance low-mids)"""
        
        if amount <= 0:
            return audio
        
        # Boost 200-400 Hz for warmth
        sos = signal.butter(2, [200, 400], btype='band', fs=self.fs, output='sos')
        warm = sosfilt(sos, audio)
        
        # Mix in warmth
        mix = amount / 100.0
        return audio + warm * mix * 0.3
    
    def add_presence(self, audio: np.ndarray) -> np.ndarray:
        """Add presence for clarity (3-5 kHz)"""
        
        # Gentle boost around 3.5 kHz
        sos = signal.butter(2, [3000, 5000], btype='band', fs=self.fs, output='sos')
        presence = sosfilt(sos, audio)
        
        return audio + presence * 0.2
    
    def add_doubler(self, audio: np.ndarray) -> np.ndarray:
        """Create voice doubling effect"""
        
        # Create slightly delayed and pitch-shifted copy
        delay_samples = int(0.02 * self.fs)  # 20ms delay
        
        if len(audio) > delay_samples:
            doubled = np.zeros_like(audio)
            doubled[delay_samples:] = audio[:-delay_samples] * 0.5
            
            # Slight pitch shift for thickness
            doubled = signal.resample(doubled, len(doubled) + 10)[:len(audio)]
            
            return audio + doubled * 0.3
        
        return audio
    
    def remove_harshness(self, audio: np.ndarray) -> np.ndarray:
        """Remove harsh frequencies (2-4 kHz)"""
        
        # Gentle cut around 2.5 kHz
        sos = signal.butter(2, [2000, 4000], btype='bandstop', fs=self.fs, output='sos')
        smooth = sosfilt(sos, audio)
        
        # Mix: mostly processed
        return smooth * 0.7 + audio * 0.3

class KaraokeVocalMixer:
    """Complete karaoke mixing system optimized for singers"""
    
    def __init__(self, sample_rate: int = 44100):
        self.fs = sample_rate
        self.block_size = 512
        
        # Initialize processors
        self.reverb = EnhancedReverb(sample_rate)
        self.pitch_corrector = PitchCorrection(sample_rate)
        self.enhancer = VocalEnhancer(sample_rate)
        
        # Current preset
        self.current_preset = KARAOKE_PRESETS["Karaoke King"]
        
        # Echo/Delay
        self.delay_buffer = np.zeros(int(1.0 * sample_rate))  # 1 second max
        self.delay_index = 0
        
        # Levels
        self.vocal_volume = 1.0
        self.music_volume = 0.7
        self.monitor_volume = 1.0
        
        # Key adjustment (semitones)
        self.key_shift = 0  # -6 to +6 semitones
        
        # Effects bypass
        self.bypass_effects = False
        
        # Meters
        self.vocal_level = 0.0
        self.music_level = 0.0
        
    def process_vocals(self, vocal: np.ndarray) -> np.ndarray:
        """Process vocal with all enhancements"""
        
        if self.bypass_effects:
            return vocal * self.vocal_volume
        
        preset = self.current_preset
        
        # 1. Input gain and noise gate
        processed = vocal * self.vocal_volume
        
        # Simple noise gate to reduce background noise
        gate_threshold = 0.01
        for i in range(len(processed)):
            if abs(processed[i]) < gate_threshold:
                processed[i] *= 0.1  # Reduce but don't eliminate
        
        # 2. Pitch correction (if enabled)
        if preset.pitch_correction > 0:
            processed = self.pitch_corrector.process(processed, preset.pitch_correction)
        
        # 3. Key adjustment (if needed)
        if self.key_shift != 0:
            # Simple pitch shift for key change
            shift_ratio = 2 ** (self.key_shift / 12.0)
            processed = signal.resample(processed, int(len(processed) * shift_ratio))
            if len(processed) > len(vocal):
                processed = processed[:len(vocal)]
            else:
                processed = np.pad(processed, (0, len(vocal) - len(processed)))
        
        # 4. Vocal enhancement
        if preset.vocal_warmth > 0:
            processed = self.enhancer.add_warmth(processed, preset.vocal_warmth)
        
        processed = self.enhancer.add_presence(processed)
        processed = self.enhancer.remove_harshness(processed)
        
        # 5. Doubler effect
        if preset.doubler:
            processed = self.enhancer.add_doubler(processed)
        
        # 6. Echo/Delay
        if preset.echo_delay > 0:
            delay_samples = int(preset.echo_delay * 0.001 * self.fs)
            feedback = preset.echo_feedback / 100.0
            
            for i in range(len(processed)):
                delay_idx = (self.delay_index - delay_samples) % len(self.delay_buffer)
                delayed = self.delay_buffer[delay_idx]
                
                self.delay_buffer[self.delay_index] = processed[i] + delayed * feedback
                self.delay_index = (self.delay_index + 1) % len(self.delay_buffer)
                
                processed[i] += delayed * 0.5
        
        # 7. Reverb (the magic ingredient!)
        processed = self.reverb.process(processed, 
                                       preset.reverb_amount,
                                       preset.reverb_room_size)
        
        # 8. Confidence boost (compression and limiting)
        if preset.confidence_boost > 0:
            boost = preset.confidence_boost / 100.0
            
            # Simple compression to even out levels
            for i in range(len(processed)):
                level = abs(processed[i])
                if level > 0.5:
                    # Compress peaks
                    processed[i] *= 0.5 / level
                elif level < 0.1 and level > 0:
                    # Boost quiet parts
                    processed[i] *= 0.1 / level * boost
            
            # Final boost
            processed *= (1.0 + boost * 0.5)
        
        # 9. Limiter to prevent clipping
        processed = np.clip(processed, -0.95, 0.95)
        
        # Update meter
        self.vocal_level = np.max(np.abs(processed))
        
        return processed
    
    def process_music(self, music: np.ndarray) -> np.ndarray:
        """Process backing track"""
        
        # Adjust music level
        processed = music * self.music_volume
        
        # Duck music slightly when vocals are present
        if self.vocal_level > 0.3:
            processed *= 0.8
        
        # Update meter
        self.music_level = np.max(np.abs(processed))
        
        return processed
    
    def mix(self, vocal: np.ndarray, music: np.ndarray) -> np.ndarray:
        """Mix vocal and music"""
        
        # Process separately
        processed_vocal = self.process_vocals(vocal)
        processed_music = self.process_music(music)
        
        # Mix together
        mix = processed_vocal + processed_music
        
        # Final master limiter
        mix = np.clip(mix, -1.0, 1.0)
        
        # Create stereo if needed
        if len(mix.shape) == 1:
            mix = np.column_stack((mix, mix))
        
        return mix
    
    def set_preset(self, preset_name: str):
        """Change preset"""
        if preset_name in KARAOKE_PRESETS:
            self.current_preset = KARAOKE_PRESETS[preset_name]
            print(f"Preset changed to: {preset_name}")
    
    def set_key(self, semitones: int):
        """Adjust key up or down"""
        self.key_shift = np.clip(semitones, -6, 6)
        if semitones > 0:
            print(f"Key raised by {semitones} semitones")
        elif semitones < 0:
            print(f"Key lowered by {-semitones} semitones")
        else:
            print("Original key")
    
    def get_presets(self) -> List[str]:
        """Get available presets"""
        return list(KARAOKE_PRESETS.keys())

def test_karaoke_system():
    """Test the karaoke system"""
    print("🎤 Karaoke Vocal Enhancer Test")
    print("=" * 50)
    
    mixer = KaraokeVocalMixer()
    
    # Create test signals
    duration = 2.0
    t = np.linspace(0, duration, int(44100 * duration))
    
    # Simulate vocal (varying pitch)
    vocal = np.sin(2 * np.pi * 440 * t) * 0.3  # A4 note
    vocal += np.sin(2 * np.pi * 554 * t) * 0.2  # C#5 note
    vocal *= (1 + 0.3 * np.sin(2 * np.pi * 5 * t))  # Vibrato
    
    # Simulate music (simple chord)
    music = np.sin(2 * np.pi * 220 * t) * 0.2   # A3
    music += np.sin(2 * np.pi * 277 * t) * 0.15  # C#4
    music += np.sin(2 * np.pi * 330 * t) * 0.15  # E4
    
    print("\nAvailable Presets:")
    for preset_name in mixer.get_presets():
        preset = KARAOKE_PRESETS[preset_name]
        print(f"  • {preset_name}: {preset.reverb_amount}% reverb, "
              f"{preset.pitch_correction}% pitch correction")
    
    print("\nTesting different presets...")
    
    for preset_name in ["Shower Singer", "Karaoke King", "Concert Hall"]:
        mixer.set_preset(preset_name)
        
        # Process a chunk
        chunk_size = 4410  # 0.1 seconds
        vocal_chunk = vocal[:chunk_size]
        music_chunk = music[:chunk_size]
        
        output = mixer.mix(vocal_chunk, music_chunk)
        
        print(f"\n{preset_name}:")
        print(f"  Vocal Level: {mixer.vocal_level:.2f}")
        print(f"  Music Level: {mixer.music_level:.2f}")
        print(f"  Reverb: {KARAOKE_PRESETS[preset_name].reverb_amount}%")
        print(f"  Output shape: {output.shape}")
    
    print("\n✅ Karaoke system ready!")
    print("\nKey features for beginners:")
    print("  • Generous reverb to smooth vocals")
    print("  • Pitch correction to stay in tune")
    print("  • Voice warmth enhancement")
    print("  • Echo effects for professional sound")
    print("  • Confidence boost processing")
    
    return mixer

if __name__ == "__main__":
    test_karaoke_system()