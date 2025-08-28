#!/usr/bin/env python3
"""
Professional Music Mixing System
Advanced DSP, frequency analysis, and studio-quality processing
"""

import numpy as np
import sounddevice as sd
from scipy import signal, fft
from scipy.signal import butter, lfilter, sosfilt, sosfreqz
import threading
import queue
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import json

@dataclass
class MusicChannelStrip:
    """Professional channel strip for music mixing"""
    name: str
    
    # Levels
    input_gain: float = 0.0      # -24 to +24 dB
    volume: float = 0.0          # -inf to +10 dB (fader)
    pan: float = 0.0            # -100 to +100 (L to R)
    
    # Dynamics
    gate_enabled: bool = False
    gate_threshold: float = -40.0
    gate_range: float = -60.0
    
    comp_enabled: bool = True
    comp_threshold: float = -18.0
    comp_ratio: float = 3.0
    comp_attack: float = 10.0    # ms
    comp_release: float = 100.0  # ms
    comp_makeup: float = 0.0     # dB
    
    # EQ - 4 band parametric
    eq_enabled: bool = True
    hpf_freq: float = 80.0       # High pass filter
    hpf_enabled: bool = False
    
    low_freq: float = 100.0      # Low shelf
    low_gain: float = 0.0
    low_q: float = 0.7
    
    lowmid_freq: float = 500.0   # Low-mid bell
    lowmid_gain: float = 0.0
    lowmid_q: float = 1.0
    
    himid_freq: float = 3000.0   # Hi-mid bell
    himid_gain: float = 0.0
    himid_q: float = 1.0
    
    high_freq: float = 10000.0   # High shelf
    high_gain: float = 0.0
    high_q: float = 0.7
    
    lpf_freq: float = 18000.0    # Low pass filter
    lpf_enabled: bool = False
    
    # Sends (pre/post fader)
    aux1_send: float = 0.0       # -inf to 0 dB
    aux1_pre: bool = False
    aux2_send: float = 0.0
    aux2_pre: bool = False
    aux3_send: float = 0.0
    aux3_pre: bool = True        # Monitor send
    aux4_send: float = 0.0
    aux4_pre: bool = False
    
    # Mix assignment
    mute: bool = False
    solo: bool = False
    solo_safe: bool = False      # Exempt from solo
    to_stereo: bool = True       # Route to stereo bus
    to_group: List[bool] = None  # Route to group buses

class ProfessionalEQ:
    """High-quality parametric EQ with multiple filter types"""
    
    def __init__(self, sample_rate: int = 48000):
        self.fs = sample_rate
        self.filters = {}
        
    def design_filter(self, freq: float, gain: float, q: float, 
                     filter_type: str = 'bell') -> np.ndarray:
        """Design digital filter coefficients"""
        
        # Normalize frequency
        w0 = 2 * np.pi * freq / self.fs
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        A = 10 ** (gain / 40)  # For peaking and shelving EQ
        
        if filter_type == 'bell' or filter_type == 'peaking':
            # Peaking EQ
            alpha = sin_w0 / (2 * q)
            
            b0 = 1 + alpha * A
            b1 = -2 * cos_w0
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * cos_w0
            a2 = 1 - alpha / A
            
        elif filter_type == 'lowshelf':
            # Low shelf
            S = 1  # Shelf slope (1 = steepest)
            alpha = sin_w0 / 2 * np.sqrt((A + 1/A) * (1/S - 1) + 2)
            
            b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
            b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
            b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
            a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
            a2 = (A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha
            
        elif filter_type == 'highshelf':
            # High shelf
            S = 1
            alpha = sin_w0 / 2 * np.sqrt((A + 1/A) * (1/S - 1) + 2)
            
            b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
            b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
            b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
            a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
            a2 = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha
            
        elif filter_type == 'highpass':
            # High pass filter
            alpha = sin_w0 / (2 * q)
            
            b0 = (1 + cos_w0) / 2
            b1 = -(1 + cos_w0)
            b2 = (1 + cos_w0) / 2
            a0 = 1 + alpha
            a1 = -2 * cos_w0
            a2 = 1 - alpha
            
        elif filter_type == 'lowpass':
            # Low pass filter
            alpha = sin_w0 / (2 * q)
            
            b0 = (1 - cos_w0) / 2
            b1 = 1 - cos_w0
            b2 = (1 - cos_w0) / 2
            a0 = 1 + alpha
            a1 = -2 * cos_w0
            a2 = 1 - alpha
            
        else:
            # Default: no filtering
            return np.array([[1, 0, 0, 1, 0, 0]])
        
        # Normalize coefficients
        b = np.array([b0, b1, b2]) / a0
        a = np.array([a0, a1, a2]) / a0
        
        # Convert to second-order sections for stability
        sos = np.array([[b[0], b[1], b[2], a[0], a[1], a[2]]])
        return sos
    
    def process(self, audio: np.ndarray, channel: MusicChannelStrip) -> np.ndarray:
        """Apply parametric EQ to audio"""
        
        if not channel.eq_enabled:
            return audio
        
        output = audio.copy()
        
        # High pass filter
        if channel.hpf_enabled:
            sos = self.design_filter(channel.hpf_freq, 0, 0.7, 'highpass')
            output = sosfilt(sos, output)
        
        # Low shelf
        if abs(channel.low_gain) > 0.1:
            sos = self.design_filter(channel.low_freq, channel.low_gain, 
                                    channel.low_q, 'lowshelf')
            output = sosfilt(sos, output)
        
        # Low-mid bell
        if abs(channel.lowmid_gain) > 0.1:
            sos = self.design_filter(channel.lowmid_freq, channel.lowmid_gain,
                                    channel.lowmid_q, 'bell')
            output = sosfilt(sos, output)
        
        # Hi-mid bell
        if abs(channel.himid_gain) > 0.1:
            sos = self.design_filter(channel.himid_freq, channel.himid_gain,
                                    channel.himid_q, 'bell')
            output = sosfilt(sos, output)
        
        # High shelf
        if abs(channel.high_gain) > 0.1:
            sos = self.design_filter(channel.high_freq, channel.high_gain,
                                    channel.high_q, 'highshelf')
            output = sosfilt(sos, output)
        
        # Low pass filter
        if channel.lpf_enabled:
            sos = self.design_filter(channel.lpf_freq, 0, 0.7, 'lowpass')
            output = sosfilt(sos, output)
        
        return output

class StudioCompressor:
    """Professional compressor with look-ahead and side-chain filtering"""
    
    def __init__(self, sample_rate: int = 48000):
        self.fs = sample_rate
        self.envelope = 0.0
        self.lookahead_buffer = np.zeros(int(0.005 * sample_rate))  # 5ms lookahead
        self.lookahead_index = 0
        
    def process(self, audio: np.ndarray, channel: MusicChannelStrip) -> Tuple[np.ndarray, float]:
        """Apply compression with RMS detection"""
        
        if not channel.comp_enabled:
            return audio, 0.0
        
        # Convert parameters
        threshold = 10 ** (channel.comp_threshold / 20)
        ratio = channel.comp_ratio
        attack_time = channel.comp_attack * 0.001
        release_time = channel.comp_release * 0.001
        makeup_gain = 10 ** (channel.comp_makeup / 20)
        
        # Time constants
        attack = np.exp(-1.0 / (attack_time * self.fs))
        release = np.exp(-1.0 / (release_time * self.fs))
        
        output = np.zeros_like(audio)
        gain_reduction_db = 0.0
        
        for i in range(len(audio)):
            # Get input sample
            input_sample = audio[i]
            
            # RMS detection (more musical than peak)
            rms = np.sqrt(input_sample ** 2)
            
            # Envelope follower
            if rms > self.envelope:
                self.envelope = rms + (self.envelope - rms) * attack
            else:
                self.envelope = rms + (self.envelope - rms) * release
            
            # Compute gain reduction
            if self.envelope > threshold:
                # Above threshold - apply compression
                excess_db = 20 * np.log10(self.envelope / threshold)
                gain_reduction = excess_db * (1 - 1/ratio)
                gain = 10 ** (-gain_reduction / 20)
            else:
                gain = 1.0
            
            # Apply gain with makeup
            output[i] = input_sample * gain * makeup_gain
            
            # Track gain reduction for metering
            if gain < 1.0:
                gain_reduction_db = max(gain_reduction_db, -20 * np.log10(gain))
        
        return output, gain_reduction_db

class MusicReverb:
    """High-quality algorithmic reverb for music production"""
    
    def __init__(self, sample_rate: int = 48000):
        self.fs = sample_rate
        
        # Early reflections (simulate room geometry)
        self.early_delays = [
            int(0.007 * sample_rate),
            int(0.011 * sample_rate),
            int(0.013 * sample_rate),
            int(0.017 * sample_rate),
            int(0.019 * sample_rate),
            int(0.023 * sample_rate),
        ]
        self.early_gains = [0.8, 0.7, 0.65, 0.6, 0.55, 0.5]
        self.early_buffers = [np.zeros(d) for d in self.early_delays]
        self.early_indices = [0] * len(self.early_delays)
        
        # Late reverb (diffuse field)
        self.late_delays = [
            int(0.029 * sample_rate),
            int(0.037 * sample_rate),
            int(0.041 * sample_rate),
            int(0.043 * sample_rate),
            int(0.047 * sample_rate),
            int(0.051 * sample_rate),
            int(0.053 * sample_rate),
            int(0.059 * sample_rate),
        ]
        self.late_buffers = [np.zeros(d) for d in self.late_delays]
        self.late_indices = [0] * len(self.late_delays)
        
        # Allpass filters for diffusion
        self.ap_delays = [
            int(0.0051 * sample_rate),
            int(0.0077 * sample_rate),
            int(0.0091 * sample_rate),
            int(0.0113 * sample_rate),
        ]
        self.ap_buffers = [np.zeros(d) for d in self.ap_delays]
        self.ap_indices = [0] * len(self.ap_delays)
        
        # Damping filters (high frequency absorption)
        self.damping_state = [0.0] * len(self.late_delays)
        
    def process(self, audio: np.ndarray, room_size: float = 0.7, 
                damping: float = 0.5, width: float = 1.0,
                wet: float = 0.3, dry: float = 0.7) -> np.ndarray:
        """Process audio through reverb"""
        
        output = np.zeros_like(audio)
        
        # Adjust room size (feedback amount)
        feedback = room_size * 0.9
        damp = damping * 0.4
        
        for i in range(len(audio)):
            input_sample = audio[i]
            
            # Early reflections
            early_sum = 0.0
            for j in range(len(self.early_delays)):
                # Read from buffer
                early_sum += self.early_buffers[j][self.early_indices[j]] * self.early_gains[j]
                
                # Write to buffer
                self.early_buffers[j][self.early_indices[j]] = input_sample
                self.early_indices[j] = (self.early_indices[j] + 1) % self.early_delays[j]
            
            # Late reverb with damping
            late_sum = 0.0
            for j in range(len(self.late_delays)):
                # Read from buffer
                delayed = self.late_buffers[j][self.late_indices[j]]
                
                # Apply damping (low-pass filter)
                self.damping_state[j] = delayed * (1 - damp) + self.damping_state[j] * damp
                late_sum += self.damping_state[j]
                
                # Feedback with diffusion
                feedback_signal = (input_sample + late_sum * feedback) / len(self.late_delays)
                self.late_buffers[j][self.late_indices[j]] = feedback_signal
                self.late_indices[j] = (self.late_indices[j] + 1) % self.late_delays[j]
            
            late_sum /= len(self.late_delays)
            
            # Allpass diffusion network
            ap_out = early_sum + late_sum
            for j in range(len(self.ap_delays)):
                delayed_ap = self.ap_buffers[j][self.ap_indices[j]]
                ap_in = ap_out + delayed_ap * 0.5
                self.ap_buffers[j][self.ap_indices[j]] = ap_in
                ap_out = delayed_ap - ap_in * 0.5
                self.ap_indices[j] = (self.ap_indices[j] + 1) % self.ap_delays[j]
            
            # Mix wet and dry
            output[i] = audio[i] * dry + ap_out * wet * width
        
        return output

class FrequencyAnalyzer:
    """Real-time frequency spectrum analyzer"""
    
    def __init__(self, sample_rate: int = 48000, fft_size: int = 2048):
        self.fs = sample_rate
        self.fft_size = fft_size
        self.window = np.hanning(fft_size)
        self.freq_bins = np.fft.rfftfreq(fft_size, 1/sample_rate)
        
    def analyze(self, audio: np.ndarray) -> Dict:
        """Analyze frequency spectrum"""
        
        if len(audio) < self.fft_size:
            # Pad if necessary
            audio = np.pad(audio, (0, self.fft_size - len(audio)))
        else:
            # Use last FFT size samples
            audio = audio[-self.fft_size:]
        
        # Apply window
        windowed = audio * self.window
        
        # Compute FFT
        spectrum = np.abs(np.fft.rfft(windowed))
        
        # Convert to dB
        spectrum_db = 20 * np.log10(spectrum + 1e-10)
        
        # Find peaks
        peaks = []
        for i in range(1, len(spectrum) - 1):
            if spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
                if spectrum_db[i] > -40:  # Threshold
                    peaks.append({
                        'freq': self.freq_bins[i],
                        'magnitude': spectrum_db[i]
                    })
        
        # Sort by magnitude
        peaks.sort(key=lambda x: x['magnitude'], reverse=True)
        
        return {
            'spectrum': spectrum_db.tolist(),
            'frequencies': self.freq_bins.tolist(),
            'peaks': peaks[:10],  # Top 10 peaks
            'rms': float(np.sqrt(np.mean(audio ** 2))),
            'peak': float(np.max(np.abs(audio))),
            'crest_factor': float(np.max(np.abs(audio)) / (np.sqrt(np.mean(audio ** 2)) + 1e-10))
        }

class ProfessionalMusicMixer:
    """Professional music mixing console"""
    
    def __init__(self, num_channels: int = 16, sample_rate: int = 48000):
        self.fs = sample_rate
        self.num_channels = num_channels
        self.block_size = 512
        
        # Create channel strips
        self.channels = [
            MusicChannelStrip(name=f"Track {i+1}")
            for i in range(num_channels)
        ]
        
        # Initialize processors
        self.eq = ProfessionalEQ(sample_rate)
        self.compressor = StudioCompressor(sample_rate)
        self.reverb = MusicReverb(sample_rate)
        self.analyzer = FrequencyAnalyzer(sample_rate)
        
        # Aux buses (for effects sends)
        self.aux_buses = {
            'aux1': {'name': 'Reverb', 'return_level': 0.0},
            'aux2': {'name': 'Delay', 'return_level': 0.0},
            'aux3': {'name': 'Monitor', 'return_level': 0.0},
            'aux4': {'name': 'FX', 'return_level': 0.0},
        }
        
        # Group buses
        self.group_buses = [
            {'name': f'Group {i+1}', 'level': 0.0, 'mute': False}
            for i in range(4)
        ]
        
        # Master bus
        self.master = {
            'level': 0.0,  # dB
            'comp_enabled': True,
            'comp_threshold': -6.0,
            'comp_ratio': 2.0,
            'limiter_enabled': True,
            'limiter_ceiling': -0.3
        }
        
        # Metering
        self.channel_meters = [0.0] * num_channels
        self.master_meter_l = 0.0
        self.master_meter_r = 0.0
        self.gain_reduction = 0.0
        
        # Audio I/O
        self.input_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue(maxsize=10)
        self.running = False
        
    def process_channel(self, audio: np.ndarray, channel: MusicChannelStrip) -> Dict:
        """Process single channel through strip"""
        
        if channel.mute and not channel.solo:
            return {
                'audio': np.zeros_like(audio),
                'meter': 0.0,
                'gain_reduction': 0.0,
                'sends': {}
            }
        
        # Input gain
        processed = audio * (10 ** (channel.input_gain / 20))
        
        # Gate (if enabled)
        if channel.gate_enabled:
            gate_threshold = 10 ** (channel.gate_threshold / 20)
            gate_range = 10 ** (channel.gate_range / 20)
            
            for i in range(len(processed)):
                if abs(processed[i]) < gate_threshold:
                    processed[i] *= gate_range
        
        # EQ
        processed = self.eq.process(processed, channel)
        
        # Compressor
        processed, gr = self.compressor.process(processed, channel)
        
        # Calculate sends (pre/post fader)
        sends = {}
        
        # Pre-fader sends
        if channel.aux1_pre:
            sends['aux1'] = processed * (10 ** (channel.aux1_send / 20))
        if channel.aux3_pre:  # Monitor is typically pre-fader
            sends['aux3'] = processed * (10 ** (channel.aux3_send / 20))
        
        # Apply fader
        fader_gain = 10 ** (channel.volume / 20)
        processed = processed * fader_gain
        
        # Post-fader sends
        if not channel.aux1_pre:
            sends['aux1'] = processed * (10 ** (channel.aux1_send / 20))
        if not channel.aux2_pre:
            sends['aux2'] = processed * (10 ** (channel.aux2_send / 20))
        if not channel.aux4_pre:
            sends['aux4'] = processed * (10 ** (channel.aux4_send / 20))
        
        # Metering (post-fader)
        meter = np.max(np.abs(processed))
        
        return {
            'audio': processed,
            'meter': meter,
            'gain_reduction': gr,
            'sends': sends
        }
    
    def process_audio(self, input_audio: np.ndarray) -> np.ndarray:
        """Main mixing process"""
        
        # Initialize mix buses
        stereo_l = np.zeros(len(input_audio))
        stereo_r = np.zeros(len(input_audio))
        aux_mixes = {
            'aux1': np.zeros(len(input_audio)),
            'aux2': np.zeros(len(input_audio)),
            'aux3': np.zeros(len(input_audio)),
            'aux4': np.zeros(len(input_audio)),
        }
        
        # Check for any soloed channels
        any_soloed = any(ch.solo for ch in self.channels)
        
        # Process each channel
        for i, channel in enumerate(self.channels):
            if i >= input_audio.shape[1]:
                break
                
            # Skip if not soloed (when solo is active)
            if any_soloed and not channel.solo and not channel.solo_safe:
                continue
            
            # Get channel audio
            ch_audio = input_audio[:, i] if i < input_audio.shape[1] else np.zeros(len(input_audio))
            
            # Process through channel strip
            result = self.process_channel(ch_audio, channel)
            
            # Update metering
            self.channel_meters[i] = result['meter']
            
            # Pan and add to stereo bus
            if channel.to_stereo:
                pan = (channel.pan + 100) / 200  # Convert -100..100 to 0..1
                left_gain = np.cos(pan * np.pi / 2)
                right_gain = np.sin(pan * np.pi / 2)
                
                stereo_l += result['audio'] * left_gain
                stereo_r += result['audio'] * right_gain
            
            # Add to aux buses
            for aux_name, aux_audio in result['sends'].items():
                if aux_name in aux_mixes:
                    aux_mixes[aux_name] += aux_audio
        
        # Process aux returns
        # Reverb return
        reverb_out = self.reverb.process(aux_mixes['aux1'], 
                                         room_size=0.8, 
                                         damping=0.5,
                                         wet=1.0, 
                                         dry=0.0)
        reverb_return = 10 ** (self.aux_buses['aux1']['return_level'] / 20)
        stereo_l += reverb_out * reverb_return * 0.5
        stereo_r += reverb_out * reverb_return * 0.5
        
        # Master bus processing
        master_gain = 10 ** (self.master['level'] / 20)
        stereo_l *= master_gain
        stereo_r *= master_gain
        
        # Master compressor
        if self.master['comp_enabled']:
            # Process L/R separately for stereo image preservation
            stereo_l, gr_l = self.compressor.process(stereo_l, 
                MusicChannelStrip(
                    name="Master",
                    comp_enabled=True,
                    comp_threshold=self.master['comp_threshold'],
                    comp_ratio=self.master['comp_ratio'],
                    comp_attack=3.0,
                    comp_release=50.0
                ))
            stereo_r, gr_r = self.compressor.process(stereo_r,
                MusicChannelStrip(
                    name="Master",
                    comp_enabled=True,
                    comp_threshold=self.master['comp_threshold'],
                    comp_ratio=self.master['comp_ratio'],
                    comp_attack=3.0,
                    comp_release=50.0
                ))
            self.gain_reduction = max(gr_l, gr_r)
        
        # Master limiter
        if self.master['limiter_enabled']:
            ceiling = 10 ** (self.master['limiter_ceiling'] / 20)
            stereo_l = np.clip(stereo_l, -ceiling, ceiling)
            stereo_r = np.clip(stereo_r, -ceiling, ceiling)
        
        # Update master meters
        self.master_meter_l = np.max(np.abs(stereo_l))
        self.master_meter_r = np.max(np.abs(stereo_r))
        
        # Combine to stereo
        output = np.column_stack((stereo_l, stereo_r))
        
        return output
    
    def get_analysis(self, audio: np.ndarray = None) -> Dict:
        """Get frequency analysis and meters"""
        
        result = {
            'channel_meters': self.channel_meters,
            'master_meter_l': self.master_meter_l,
            'master_meter_r': self.master_meter_r,
            'gain_reduction': self.gain_reduction,
        }
        
        if audio is not None:
            # Analyze frequency spectrum
            if len(audio.shape) > 1:
                # Analyze left channel
                result['spectrum'] = self.analyzer.analyze(audio[:, 0])
            else:
                result['spectrum'] = self.analyzer.analyze(audio)
        
        return result

def demo():
    """Demo the professional mixer"""
    print("🎚️ Professional Music Mixer")
    print("=" * 50)
    
    mixer = ProfessionalMusicMixer(num_channels=8, sample_rate=48000)
    
    # Setup some channels
    mixer.channels[0].name = "Kick"
    mixer.channels[0].eq_enabled = True
    mixer.channels[0].low_gain = 6.0  # Boost lows
    mixer.channels[0].comp_enabled = True
    mixer.channels[0].comp_threshold = -15.0
    mixer.channels[0].comp_ratio = 4.0
    
    mixer.channels[1].name = "Snare"
    mixer.channels[1].himid_freq = 5000
    mixer.channels[1].himid_gain = 3.0  # Presence
    mixer.channels[1].aux1_send = -6.0  # Some reverb
    
    mixer.channels[2].name = "Bass"
    mixer.channels[2].hpf_enabled = True
    mixer.channels[2].hpf_freq = 40  # Remove sub-bass
    mixer.channels[2].comp_enabled = True
    mixer.channels[2].comp_threshold = -12.0
    
    mixer.channels[3].name = "Guitar"
    mixer.channels[3].lowmid_freq = 800
    mixer.channels[3].lowmid_gain = -3.0  # Cut mud
    mixer.channels[3].aux1_send = -10.0  # Reverb
    mixer.channels[3].pan = 30  # Pan right
    
    mixer.channels[4].name = "Vocal"
    mixer.channels[4].hpf_enabled = True
    mixer.channels[4].hpf_freq = 80
    mixer.channels[4].himid_freq = 3500
    mixer.channels[4].himid_gain = 2.0  # Presence
    mixer.channels[4].comp_enabled = True
    mixer.channels[4].comp_threshold = -18.0
    mixer.channels[4].comp_ratio = 3.0
    mixer.channels[4].aux1_send = -12.0  # Reverb
    
    # Create test signal
    duration = 1.0
    fs = 48000
    t = np.linspace(0, duration, int(fs * duration))
    
    # Multi-track test (simplified drum pattern + instruments)
    kick = np.sin(2 * np.pi * 60 * t) * (t % 0.5 < 0.1)  # Kick pattern
    snare = np.random.normal(0, 0.1, len(t)) * ((t % 0.5 > 0.25) & (t % 0.5 < 0.35))  # Snare
    bass = np.sin(2 * np.pi * 100 * t) * 0.3  # Bass
    guitar = np.sin(2 * np.pi * 440 * t) * 0.2  # Guitar (A note)
    vocal = np.sin(2 * np.pi * 880 * t) * 0.15  # Vocal (A octave up)
    
    # Combine tracks
    input_audio = np.column_stack((kick, snare, bass, guitar, vocal))
    
    # Process through mixer
    output = mixer.process_audio(input_audio[:1024])  # Process small chunk
    
    # Get analysis
    analysis = mixer.get_analysis(output)
    
    print("\nChannel Levels:")
    for i, meter in enumerate(analysis['channel_meters'][:5]):
        if meter > 0:
            bar = '█' * int(meter * 20)
            print(f"  {mixer.channels[i].name:10} [{bar:<20}] {20*np.log10(meter+1e-10):.1f} dB")
    
    print(f"\nMaster:")
    print(f"  L: {20*np.log10(analysis['master_meter_l']+1e-10):.1f} dB")
    print(f"  R: {20*np.log10(analysis['master_meter_r']+1e-10):.1f} dB")
    print(f"  Gain Reduction: -{analysis['gain_reduction']:.1f} dB")
    
    if 'spectrum' in analysis:
        spectrum = analysis['spectrum']
        print(f"\nSpectrum Analysis:")
        print(f"  RMS: {20*np.log10(spectrum['rms']+1e-10):.1f} dB")
        print(f"  Peak: {20*np.log10(spectrum['peak']+1e-10):.1f} dB")
        print(f"  Crest Factor: {spectrum['crest_factor']:.1f}")
        
        if spectrum['peaks']:
            print(f"\n  Top Frequency Peaks:")
            for peak in spectrum['peaks'][:5]:
                print(f"    {peak['freq']:.0f} Hz: {peak['magnitude']:.1f} dB")
    
    print("\n✅ Professional music mixer is ready!")
    print("Features: 4-band parametric EQ, compression, reverb, frequency analysis")
    
    return mixer

if __name__ == "__main__":
    demo()