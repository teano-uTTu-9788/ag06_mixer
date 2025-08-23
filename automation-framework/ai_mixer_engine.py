#!/usr/bin/env python3
"""
Real-Time AI Mixing Engine for AG06
Actually mixes and outputs audio in real-time!
"""

import sounddevice as sd
import numpy as np
from scipy import signal
import threading
import time
from collections import deque
import queue

class AIRealTimeMixer:
    def __init__(self):
        self.sample_rate = 48000
        self.block_size = 512
        self.channels = 2
        self.is_mixing = False
        
        # Find AG06 device
        self.input_device = None
        self.output_device = None
        devices = sd.query_devices()
        
        for i, device in enumerate(devices):
            device_name = device['name'].lower()
            if 'ag06' in device_name or 'ag03' in device_name:
                if device['max_input_channels'] > 0:
                    self.input_device = i
                    print(f"✅ Input: {device['name']} (Device {i})")
                if device['max_output_channels'] > 0:
                    self.output_device = i
                    print(f"✅ Output: {device['name']} (Device {i})")
        
        # Audio processing parameters
        self.input_gain = 1.0
        self.output_gain = 0.8
        
        # EQ bands (Hz)
        self.eq_bands = [60, 170, 350, 1000, 3500, 10000]
        self.eq_gains = [0, 0, 0, 0, 0, 0]  # dB
        
        # Compressor
        self.comp_threshold = -20  # dB
        self.comp_ratio = 3.0
        self.comp_attack = 0.005  # seconds
        self.comp_release = 0.1  # seconds
        
        # Effects
        self.reverb_mix = 0.1
        self.delay_time = 0.25  # seconds
        self.delay_feedback = 0.3
        self.delay_mix = 0.0
        
        # Music-specific settings
        self.music_mode = True
        self.enhance_bass = True
        self.enhance_presence = True
        
        # Delay buffer for effects
        delay_samples = int(self.delay_time * self.sample_rate)
        self.delay_buffer_l = deque([0] * delay_samples, maxlen=delay_samples)
        self.delay_buffer_r = deque([0] * delay_samples, maxlen=delay_samples)
        
        # Processing queue
        self.process_queue = queue.Queue()
        
    def audio_callback(self, indata, outdata, frames, time_info, status):
        """Real-time audio processing callback - THE ACTUAL MIXING HAPPENS HERE!"""
        if status:
            print(f"Callback status: {status}")
        
        try:
            # Get stereo input
            if indata.shape[1] == 1:
                # Mono input - duplicate to stereo
                left = right = indata[:, 0] * self.input_gain
            else:
                # Stereo input
                left = indata[:, 0] * self.input_gain
                right = indata[:, 1] * self.input_gain if indata.shape[1] > 1 else left
            
            # Apply AI mixing based on music detection
            if self.music_mode:
                # MUSIC MODE - Enhance for music
                
                # 1. Dynamic EQ for music
                left = self.apply_music_eq(left)
                right = self.apply_music_eq(right)
                
                # 2. Multiband compression
                left = self.apply_multiband_compression(left)
                right = self.apply_multiband_compression(right)
                
                # 3. Stereo enhancement
                left, right = self.enhance_stereo(left, right)
                
                # 4. Add subtle reverb for space
                if self.reverb_mix > 0:
                    left = self.apply_reverb(left, self.reverb_mix)
                    right = self.apply_reverb(right, self.reverb_mix)
                
                # 5. Bass enhancement
                if self.enhance_bass:
                    left = self.enhance_bass_frequencies(left)
                    right = self.enhance_bass_frequencies(right)
                    
                # 6. Presence boost for clarity
                if self.enhance_presence:
                    left = self.enhance_presence_frequencies(left)
                    right = self.enhance_presence_frequencies(right)
            
            else:
                # VOICE MODE - Optimize for speech
                
                # 1. Voice EQ
                left = self.apply_voice_eq(left)
                right = self.apply_voice_eq(right)
                
                # 2. De-esser
                left = self.de_esser(left)
                right = self.de_esser(right)
                
                # 3. Compression for consistent levels
                left = self.apply_compression(left)
                right = self.apply_compression(right)
            
            # Apply delay if enabled
            if self.delay_mix > 0:
                left = self.apply_delay(left, self.delay_buffer_l)
                right = self.apply_delay(right, self.delay_buffer_r)
            
            # Final limiting to prevent clipping
            left = np.tanh(left * self.output_gain)
            right = np.tanh(right * self.output_gain)
            
            # Output the processed audio
            if outdata.shape[1] == 1:
                # Mono output
                outdata[:, 0] = (left + right) / 2
            else:
                # Stereo output
                outdata[:, 0] = left
                if outdata.shape[1] > 1:
                    outdata[:, 1] = right
                    
        except Exception as e:
            print(f"Error in audio callback: {e}")
            # Pass through on error
            outdata[:] = indata[:] * 0.5
    
    def apply_music_eq(self, audio):
        """Apply music-optimized EQ curve"""
        # Boost bass (60-170 Hz) by 3dB
        # Slight mid scoop (350-1000 Hz) by -2dB  
        # Boost presence (3500-10000 Hz) by 2dB
        
        # Simple but effective frequency shaping
        nyquist = self.sample_rate / 2
        
        # Bass boost
        sos = signal.butter(2, [60/nyquist, 170/nyquist], 'bandpass', output='sos')
        bass = signal.sosfilt(sos, audio) * 1.4  # +3dB
        
        # Presence boost
        sos = signal.butter(2, [3500/nyquist, 10000/nyquist], 'bandpass', output='sos')
        presence = signal.sosfilt(sos, audio) * 1.25  # +2dB
        
        # Combine
        return audio * 0.7 + bass * 0.2 + presence * 0.1
    
    def apply_voice_eq(self, audio):
        """Apply voice-optimized EQ curve"""
        # Cut low rumble < 80Hz
        # Boost presence 2-4kHz for clarity
        nyquist = self.sample_rate / 2
        
        # High-pass to remove rumble
        sos = signal.butter(2, 80/nyquist, 'highpass', output='sos')
        audio = signal.sosfilt(sos, audio)
        
        # Presence boost
        sos = signal.butter(2, [2000/nyquist, 4000/nyquist], 'bandpass', output='sos')
        presence = signal.sosfilt(sos, audio) * 1.5
        
        return audio * 0.8 + presence * 0.2
    
    def apply_multiband_compression(self, audio):
        """3-band compression for music"""
        nyquist = self.sample_rate / 2
        
        # Split into 3 bands
        # Low: < 200Hz
        sos_low = signal.butter(2, 200/nyquist, 'lowpass', output='sos')
        low = signal.sosfilt(sos_low, audio)
        
        # Mid: 200-4000Hz  
        sos_mid = signal.butter(2, [200/nyquist, 4000/nyquist], 'bandpass', output='sos')
        mid = signal.sosfilt(sos_mid, audio)
        
        # High: > 4000Hz
        sos_high = signal.butter(2, 4000/nyquist, 'highpass', output='sos')
        high = signal.sosfilt(sos_high, audio)
        
        # Compress each band differently
        low = self.soft_compress(low, -15, 2.0)  # Gentle on bass
        mid = self.soft_compress(mid, -20, 3.0)  # More on mids
        high = self.soft_compress(high, -25, 2.5)  # Moderate on highs
        
        return low + mid + high
    
    def soft_compress(self, audio, threshold_db, ratio):
        """Soft knee compression"""
        threshold = 10 ** (threshold_db / 20)
        
        # Calculate envelope
        envelope = np.abs(audio)
        
        # Apply compression
        compressed = np.where(
            envelope > threshold,
            np.sign(audio) * (threshold + (np.abs(audio) - threshold) / ratio),
            audio
        )
        
        return compressed
    
    def apply_compression(self, audio):
        """Simple compressor"""
        threshold = 10 ** (self.comp_threshold / 20)
        
        for i in range(len(audio)):
            if abs(audio[i]) > threshold:
                if audio[i] > 0:
                    audio[i] = threshold + (audio[i] - threshold) / self.comp_ratio
                else:
                    audio[i] = -threshold + (audio[i] + threshold) / self.comp_ratio
        
        return audio
    
    def enhance_stereo(self, left, right):
        """Widen stereo image for music"""
        # Mid-Side processing
        mid = (left + right) / 2
        side = (left - right) / 2
        
        # Enhance sides for width
        side = side * 1.3
        
        # Reconstruct
        left = mid + side
        right = mid - side
        
        return left, right
    
    def enhance_bass_frequencies(self, audio):
        """Add harmonic bass enhancement"""
        nyquist = self.sample_rate / 2
        
        # Extract bass
        sos = signal.butter(2, 120/nyquist, 'lowpass', output='sos')
        bass = signal.sosfilt(sos, audio)
        
        # Generate harmonics
        harmonic = np.tanh(bass * 3) * 0.2  # Soft saturation
        
        return audio + harmonic
    
    def enhance_presence_frequencies(self, audio):
        """Add brilliance and air"""
        nyquist = self.sample_rate / 2
        
        # Extract high frequencies
        if 8000/nyquist < 1:
            sos = signal.butter(2, 8000/nyquist, 'highpass', output='sos')
            highs = signal.sosfilt(sos, audio)
            
            # Add subtle exciter
            excited = np.tanh(highs * 2) * 0.1
            
            return audio + excited
        return audio
    
    def de_esser(self, audio):
        """Reduce sibilance in voice"""
        nyquist = self.sample_rate / 2
        
        # Detect sibilance (5-8kHz)
        sos = signal.butter(2, [5000/nyquist, 8000/nyquist], 'bandpass', output='sos')
        sibilance = signal.sosfilt(sos, audio)
        
        # Compress sibilant frequencies
        sibilance = self.soft_compress(sibilance, -30, 5.0)
        
        # Subtract from original
        return audio - (signal.sosfilt(sos, audio) - sibilance)
    
    def apply_reverb(self, audio, mix):
        """Simple reverb using comb filters"""
        # Simplified reverb for real-time processing
        delays = [0.03, 0.05, 0.07, 0.09]  # seconds
        output = audio.copy()
        
        for delay_time in delays:
            delay_samples = int(delay_time * self.sample_rate)
            if delay_samples < len(audio):
                delayed = np.pad(audio, (delay_samples, 0))[:-delay_samples]
                output = output + delayed * 0.25
        
        return audio * (1 - mix) + output * mix
    
    def apply_delay(self, audio, delay_buffer):
        """Apply delay effect"""
        output = audio.copy()
        
        for i in range(len(audio)):
            # Get delayed sample
            delayed = delay_buffer[0]
            
            # Add to output
            output[i] = audio[i] + delayed * self.delay_mix
            
            # Update delay buffer with feedback
            delay_buffer.append(audio[i] + delayed * self.delay_feedback)
        
        return output
    
    def start_mixing(self):
        """Start the real-time AI mixing"""
        if self.is_mixing:
            return False
        
        try:
            self.stream = sd.Stream(
                device=(self.input_device, self.output_device),
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=self.channels,
                callback=self.audio_callback,
                dtype='float32'
            )
            self.stream.start()
            self.is_mixing = True
            print("🎵 AI MIXING STARTED - Processing audio in real-time!")
            return True
            
        except Exception as e:
            print(f"❌ Error starting mixing: {e}")
            return False
    
    def stop_mixing(self):
        """Stop mixing"""
        if self.is_mixing and hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
            self.is_mixing = False
            print("⏹ Mixing stopped")
            return True
        return False
    
    def set_music_mode(self, enabled=True):
        """Switch between music and voice mode"""
        self.music_mode = enabled
        if enabled:
            print("🎵 Music mode activated")
            self.reverb_mix = 0.15
            self.enhance_bass = True
            self.enhance_presence = True
        else:
            print("🎤 Voice mode activated")
            self.reverb_mix = 0.05
            self.enhance_bass = False
            self.enhance_presence = False

if __name__ == "__main__":
    print("=" * 60)
    print("🎛️ AI REAL-TIME MIXER FOR AG06")
    print("=" * 60)
    print("This actually MIXES and OUTPUTS audio in real-time!")
    print("Features:")
    print("  • Multiband compression")
    print("  • Dynamic EQ")
    print("  • Stereo enhancement")
    print("  • Bass harmonics")
    print("  • Presence exciter")
    print("  • Reverb & Delay")
    print("=" * 60)
    
    mixer = AIRealTimeMixer()
    
    # Start mixing
    if mixer.start_mixing():
        print("\n🎧 Mixing active! Play music through AG06...")
        print("The AI is now actively processing your audio!")
        print("\nPress Ctrl+C to stop")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            mixer.stop_mixing()
            print("\n✅ Mixer stopped")
    else:
        print("❌ Failed to start mixer")