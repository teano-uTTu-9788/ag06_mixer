#!/usr/bin/env python3
"""
AiOke Real-Time Audio Mixer with Live DSP Processing
Functional mixer that processes audio in real-time with effects
"""

import numpy as np
import sounddevice as sd
import threading
import queue
import time
from dataclasses import dataclass
from typing import Optional, Callable
import json

@dataclass
class MixerChannel:
    """Single mixer channel with DSP"""
    name: str
    volume: float = 1.0  # 0-1
    pan: float = 0.5     # 0=left, 0.5=center, 1=right
    mute: bool = False
    solo: bool = False
    
    # EQ (simple 3-band)
    low_gain: float = 0.0    # -12 to +12 dB
    mid_gain: float = 0.0    # -12 to +12 dB  
    high_gain: float = 0.0   # -12 to +12 dB
    
    # Effects sends
    reverb_send: float = 0.0  # 0-1
    delay_send: float = 0.0   # 0-1
    
    # Dynamics
    compressor_on: bool = False
    comp_threshold: float = -10.0  # dB
    comp_ratio: float = 4.0

class SimpleDSP:
    """Simple but functional DSP processors"""
    
    def __init__(self, sample_rate: int = 44100):
        self.fs = sample_rate
        
        # Reverb (simple delay network)
        self.reverb_delays = [
            np.zeros(int(0.029 * sample_rate)),
            np.zeros(int(0.037 * sample_rate)),
            np.zeros(int(0.041 * sample_rate)),
            np.zeros(int(0.043 * sample_rate))
        ]
        self.reverb_indices = [0, 0, 0, 0]
        
        # Delay line
        self.delay_buffer = np.zeros(int(0.5 * sample_rate))  # 500ms max
        self.delay_index = 0
        
        # Filters for EQ
        self.setup_eq_filters()
        
    def setup_eq_filters(self):
        """Setup biquad filters for 3-band EQ"""
        # Low shelf at 200Hz
        self.low_freq = 200
        # Mid bell at 1kHz  
        self.mid_freq = 1000
        # High shelf at 5kHz
        self.high_freq = 5000
        
    def apply_eq(self, audio: np.ndarray, low_gain: float, mid_gain: float, high_gain: float) -> np.ndarray:
        """Apply simple 3-band EQ"""
        # Convert gains from dB to linear
        low_amp = 10 ** (low_gain / 20)
        mid_amp = 10 ** (mid_gain / 20)
        high_amp = 10 ** (high_gain / 20)
        
        # Simple frequency-based amplitude adjustment
        # This is simplified - real implementation would use proper filters
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/self.fs)
        
        # Apply gains to frequency bands
        for i, freq in enumerate(freqs):
            if freq < 400:  # Low band
                fft[i] *= low_amp
            elif freq < 2000:  # Mid band
                fft[i] *= mid_amp
            else:  # High band
                fft[i] *= high_amp
                
        return np.fft.irfft(fft, len(audio))
    
    def apply_compressor(self, audio: np.ndarray, threshold: float, ratio: float) -> np.ndarray:
        """Simple compressor"""
        threshold_linear = 10 ** (threshold / 20)
        
        output = np.copy(audio)
        for i in range(len(audio)):
            level = abs(audio[i])
            if level > threshold_linear:
                # Apply compression
                excess = level - threshold_linear
                compressed_excess = excess / ratio
                gain = (threshold_linear + compressed_excess) / level
                output[i] = audio[i] * gain
                
        return output
    
    def apply_reverb(self, audio: np.ndarray, amount: float) -> np.ndarray:
        """Simple reverb using delay network"""
        if len(audio) == 0:
            return audio
            
        reverb = np.zeros_like(audio)
        
        for i, sample in enumerate(audio):
            # Sum of delayed signals
            rev_sample = 0.0
            for j in range(len(self.reverb_delays)):
                # Get delayed sample
                delay_len = len(self.reverb_delays[j])
                if delay_len > 0:
                    idx = self.reverb_indices[j] % delay_len
                    rev_sample += self.reverb_delays[j][idx] * 0.25
                    
                    # Update delay line with feedback
                    self.reverb_delays[j][idx] = sample + rev_sample * 0.4
                    self.reverb_indices[j] = (self.reverb_indices[j] + 1) % delay_len
            
            reverb[i] = rev_sample
            
        return audio + reverb * amount
    
    def apply_delay(self, audio: np.ndarray, time_ms: float, feedback: float, mix: float) -> np.ndarray:
        """Simple delay effect"""
        delay_samples = int(time_ms * 0.001 * self.fs)
        output = np.copy(audio)
        
        for i in range(len(audio)):
            # Get delayed sample
            delay_idx = (self.delay_index - delay_samples) % len(self.delay_buffer)
            delayed = self.delay_buffer[delay_idx]
            
            # Write to delay buffer with feedback
            self.delay_buffer[self.delay_index] = audio[i] + delayed * feedback
            self.delay_index = (self.delay_index + 1) % len(self.delay_buffer)
            
            # Mix delayed signal
            output[i] = audio[i] * (1 - mix) + delayed * mix
            
        return output

class RealtimeMixer:
    """Main real-time audio mixer"""
    
    def __init__(self, num_channels: int = 4):
        self.sample_rate = 44100
        self.block_size = 512
        self.num_channels = num_channels
        
        # Create mixer channels
        self.channels = [
            MixerChannel(name=f"Channel {i+1}")
            for i in range(num_channels)
        ]
        
        # DSP processor
        self.dsp = SimpleDSP(self.sample_rate)
        
        # Audio buffers
        self.input_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue(maxsize=10)
        
        # Master settings
        self.master_volume = 0.8
        self.master_limiter = True
        
        # Effects returns
        self.reverb_return = 1.0
        self.delay_return = 1.0
        self.delay_time = 250  # ms
        self.delay_feedback = 0.3
        
        # Monitoring
        self.peak_levels = [0.0] * num_channels
        self.master_peak = 0.0
        
        # Processing thread
        self.running = False
        self.process_thread = None
        
    def audio_callback(self, indata, frames, time_info, status):
        """Audio input callback"""
        if status:
            print(f"Audio status: {status}")
            
        try:
            self.input_queue.put_nowait(indata.copy())
        except queue.Full:
            pass  # Drop frame if queue is full
    
    def output_callback(self, outdata, frames, time_info, status):
        """Audio output callback"""
        try:
            data = self.output_queue.get_nowait()
            outdata[:] = data
        except queue.Empty:
            outdata[:] = 0  # Silence if no data
            
    def process_audio(self):
        """Main audio processing loop"""
        while self.running:
            try:
                # Get input audio
                input_audio = self.input_queue.get(timeout=0.1)
                
                # Handle mono or stereo input
                if len(input_audio.shape) == 1 or input_audio.shape[1] == 1:
                    # Mono input - flatten if needed
                    mono_audio = input_audio.flatten() if len(input_audio.shape) > 1 else input_audio
                    
                    # Process each channel
                    channel_outputs = []
                    reverb_send_total = np.zeros(len(mono_audio))
                    delay_send_total = np.zeros(len(mono_audio))
                    
                    for i, channel in enumerate(self.channels):
                        # Distribute mono to all channels with slight variation
                        variation = 1.0 + (i * 0.02)  # Slight difference per channel
                        ch_audio = mono_audio * variation
                else:
                    # Stereo or multi-channel input
                    # Process each channel
                    channel_outputs = []
                    reverb_send_total = np.zeros(len(input_audio))
                    delay_send_total = np.zeros(len(input_audio))
                    
                    for i, channel in enumerate(self.channels):
                        if i < input_audio.shape[1]:
                            ch_audio = input_audio[:, i]
                        else:
                            ch_audio = np.zeros(len(input_audio))
                        
                        # Skip if muted (unless soloed)
                        if channel.mute and not channel.solo:
                            continue
                            
                        # Apply channel processing
                        processed = ch_audio * channel.volume
                        
                        # EQ
                        processed = self.dsp.apply_eq(
                            processed, 
                            channel.low_gain,
                            channel.mid_gain,
                            channel.high_gain
                        )
                        
                        # Compressor
                        if channel.compressor_on:
                            processed = self.dsp.apply_compressor(
                                processed,
                                channel.comp_threshold,
                                channel.comp_ratio
                            )
                        
                        # Effects sends
                        reverb_send_total += processed * channel.reverb_send
                        delay_send_total += processed * channel.delay_send
                        
                        # Pan and add to output
                        left_gain = 1.0 - channel.pan
                        right_gain = channel.pan
                        
                        channel_outputs.append((processed * left_gain, processed * right_gain))
                        
                        # Update peak meter
                        self.peak_levels[i] = np.max(np.abs(processed))
                    
                    # Sum channels
                    if channel_outputs:
                        left_sum = sum(ch[0] for ch in channel_outputs)
                        right_sum = sum(ch[1] for ch in channel_outputs)
                    else:
                        left_sum = np.zeros(len(input_audio))
                        right_sum = np.zeros(len(input_audio))
                    
                    # Apply effects returns
                    if np.any(reverb_send_total > 0):
                        reverb = self.dsp.apply_reverb(reverb_send_total, 1.0)
                        left_sum += reverb * self.reverb_return * 0.5
                        right_sum += reverb * self.reverb_return * 0.5
                    
                    if np.any(delay_send_total > 0):
                        delay = self.dsp.apply_delay(
                            delay_send_total,
                            self.delay_time,
                            self.delay_feedback,
                            1.0
                        )
                        left_sum += delay * self.delay_return * 0.5
                        right_sum += delay * self.delay_return * 0.5
                    
                    # Master processing
                    left_sum *= self.master_volume
                    right_sum *= self.master_volume
                    
                    # Limiter (simple clipping prevention)
                    if self.master_limiter:
                        left_sum = np.clip(left_sum, -1.0, 1.0)
                        right_sum = np.clip(right_sum, -1.0, 1.0)
                    
                    # Update master peak
                    self.master_peak = max(np.max(np.abs(left_sum)), np.max(np.abs(right_sum)))
                    
                    # Create stereo output
                    output = np.column_stack((left_sum, right_sum))
                    
                    # Send to output
                    try:
                        self.output_queue.put_nowait(output)
                    except queue.Full:
                        pass
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
    
    def start(self):
        """Start the mixer"""
        if not self.running:
            self.running = True
            
            # Start processing thread
            self.process_thread = threading.Thread(target=self.process_audio)
            self.process_thread.start()
            
            # Open audio streams with default device
            # Use mono input for system microphone
            self.input_stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=1,  # Mono input for Mac microphone
                callback=self.audio_callback
            )
            
            self.output_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=2,
                callback=self.output_callback
            )
            
            self.input_stream.start()
            self.output_stream.start()
            
            print("Mixer started!")
    
    def stop(self):
        """Stop the mixer"""
        if self.running:
            self.running = False
            
            # Stop streams
            if hasattr(self, 'input_stream'):
                self.input_stream.stop()
                self.input_stream.close()
                
            if hasattr(self, 'output_stream'):
                self.output_stream.stop()
                self.output_stream.close()
            
            # Wait for thread to finish
            if self.process_thread:
                self.process_thread.join()
            
            print("Mixer stopped!")
    
    def set_channel_volume(self, channel: int, volume: float):
        """Set channel volume (0-1)"""
        if 0 <= channel < self.num_channels:
            self.channels[channel].volume = np.clip(volume, 0, 1)
    
    def set_channel_pan(self, channel: int, pan: float):
        """Set channel pan (0=left, 0.5=center, 1=right)"""
        if 0 <= channel < self.num_channels:
            self.channels[channel].pan = np.clip(pan, 0, 1)
    
    def set_channel_eq(self, channel: int, low: float, mid: float, high: float):
        """Set channel EQ gains in dB"""
        if 0 <= channel < self.num_channels:
            self.channels[channel].low_gain = np.clip(low, -12, 12)
            self.channels[channel].mid_gain = np.clip(mid, -12, 12)
            self.channels[channel].high_gain = np.clip(high, -12, 12)
    
    def set_channel_reverb(self, channel: int, amount: float):
        """Set channel reverb send (0-1)"""
        if 0 <= channel < self.num_channels:
            self.channels[channel].reverb_send = np.clip(amount, 0, 1)
    
    def set_channel_delay(self, channel: int, amount: float):
        """Set channel delay send (0-1)"""
        if 0 <= channel < self.num_channels:
            self.channels[channel].delay_send = np.clip(amount, 0, 1)
    
    def toggle_mute(self, channel: int):
        """Toggle channel mute"""
        if 0 <= channel < self.num_channels:
            self.channels[channel].mute = not self.channels[channel].mute
    
    def toggle_solo(self, channel: int):
        """Toggle channel solo"""
        if 0 <= channel < self.num_channels:
            self.channels[channel].solo = not self.channels[channel].solo
    
    def get_levels(self):
        """Get current peak levels"""
        return {
            "channels": self.peak_levels,
            "master": self.master_peak
        }

def demo():
    """Demo the mixer with interactive controls"""
    mixer = RealtimeMixer(num_channels=4)
    mixer.start()
    
    print("\n=== AIOKE REALTIME AUDIO MIXER ===")
    print("\nControls:")
    print("  1-4: Select channel")
    print("  v: Set volume (0-100)")
    print("  p: Set pan (L/C/R)")
    print("  e: Set EQ (low mid high)")
    print("  r: Set reverb (0-100)")
    print("  d: Set delay (0-100)")
    print("  m: Mute channel")
    print("  s: Solo channel")
    print("  l: Show levels")
    print("  q: Quit\n")
    
    current_channel = 0
    
    try:
        while True:
            cmd = input(f"Channel {current_channel+1}> ").strip().lower()
            
            if cmd == 'q':
                break
            elif cmd in '1234':
                current_channel = int(cmd) - 1
                print(f"Selected channel {current_channel+1}")
            elif cmd == 'v':
                vol = float(input("Volume (0-100): ")) / 100
                mixer.set_channel_volume(current_channel, vol)
            elif cmd == 'p':
                pan_str = input("Pan (L/C/R or 0-100): ").upper()
                if pan_str == 'L':
                    pan = 0.0
                elif pan_str == 'C':
                    pan = 0.5
                elif pan_str == 'R':
                    pan = 1.0
                else:
                    pan = float(pan_str) / 100
                mixer.set_channel_pan(current_channel, pan)
            elif cmd == 'e':
                low = float(input("Low EQ (-12 to 12 dB): "))
                mid = float(input("Mid EQ (-12 to 12 dB): "))
                high = float(input("High EQ (-12 to 12 dB): "))
                mixer.set_channel_eq(current_channel, low, mid, high)
            elif cmd == 'r':
                reverb = float(input("Reverb (0-100): ")) / 100
                mixer.set_channel_reverb(current_channel, reverb)
            elif cmd == 'd':
                delay = float(input("Delay (0-100): ")) / 100
                mixer.set_channel_delay(current_channel, delay)
            elif cmd == 'm':
                mixer.toggle_mute(current_channel)
                print(f"Channel {current_channel+1} mute toggled")
            elif cmd == 's':
                mixer.toggle_solo(current_channel)
                print(f"Channel {current_channel+1} solo toggled")
            elif cmd == 'l':
                levels = mixer.get_levels()
                print("\nLevels:")
                for i, level in enumerate(levels["channels"]):
                    bar = '█' * int(level * 20)
                    print(f"  Ch{i+1}: {bar:<20} {level:.2f}")
                print(f"  Master: {'█' * int(levels['master'] * 20):<20} {levels['master']:.2f}\n")
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        mixer.stop()

if __name__ == "__main__":
    demo()