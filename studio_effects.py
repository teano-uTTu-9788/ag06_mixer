"""
Professional Audio Effects for Studio Mixing
Implements Reverb, Delay, Chorus, Stereo Imager, and Harmonic Exciter
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional
from dataclasses import dataclass
import numba

@dataclass
class ReverbParams:
    """Studio reverb parameters"""
    room_size: float = 0.7  # 0-1
    damping: float = 0.5    # 0-1
    wet_level: float = 0.3  # 0-1
    dry_level: float = 0.7  # 0-1
    pre_delay_ms: float = 20.0
    width: float = 1.0      # Stereo width
    
@dataclass  
class DelayParams:
    """Studio delay parameters"""
    delay_time_ms: float = 250.0
    feedback: float = 0.4   # 0-1
    mix: float = 0.3        # 0-1 wet/dry
    tempo_sync: bool = False
    bpm: float = 120.0
    note_division: str = "1/4"  # 1/4, 1/8, 1/16, etc
    
@dataclass
class ChorusParams:
    """Chorus/modulation parameters"""
    rate_hz: float = 2.0    # LFO rate
    depth_ms: float = 3.0   # Modulation depth
    mix: float = 0.5        # Wet/dry mix
    voices: int = 2         # Number of voices
    
@dataclass
class StereoImagerParams:
    """Stereo width control"""
    width: float = 1.0      # 0=mono, 1=normal, >1=wider
    bass_mono_freq: float = 120.0  # Mono below this frequency
    high_freq_width: float = 1.5   # Extra width for highs
    
@dataclass
class ExciterParams:
    """Harmonic exciter for brilliance"""
    drive: float = 0.3      # Saturation amount
    mix: float = 0.2        # Wet/dry
    frequency: float = 8000.0  # Enhancement frequency
    bandwidth: float = 2000.0  # Enhancement bandwidth

class PlateReverb:
    """
    Plate reverb simulation using Schroeder-Moorer architecture
    Multiple comb filters + allpass filters for dense reverb
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.fs = sample_rate
        
        # Comb filter delays (in samples) - tuned for plate sound
        self.comb_delays = [
            int(0.0297 * sample_rate),
            int(0.0371 * sample_rate),
            int(0.0411 * sample_rate),
            int(0.0437 * sample_rate),
            int(0.0497 * sample_rate),
            int(0.0531 * sample_rate),
            int(0.0571 * sample_rate),
            int(0.0631 * sample_rate)
        ]
        
        # Allpass filter delays
        self.allpass_delays = [
            int(0.005 * sample_rate),
            int(0.0067 * sample_rate),
            int(0.0073 * sample_rate),
            int(0.0089 * sample_rate)
        ]
        
        # Initialize delay buffers
        self.comb_buffers = [np.zeros(d) for d in self.comb_delays]
        self.comb_indices = [0] * len(self.comb_delays)
        
        self.allpass_buffers = [np.zeros(d) for d in self.allpass_delays]
        self.allpass_indices = [0] * len(self.allpass_delays)
        
        # Pre-delay buffer
        self.predelay_buffer = None
        self.predelay_index = 0
        
    def process(self, audio: np.ndarray, params: ReverbParams) -> np.ndarray:
        """Apply plate reverb"""
        
        # Setup pre-delay
        predelay_samples = int(params.pre_delay_ms * 0.001 * self.fs)
        if self.predelay_buffer is None or len(self.predelay_buffer) != predelay_samples:
            self.predelay_buffer = np.zeros(max(1, predelay_samples))
            self.predelay_index = 0
        
        output = np.zeros_like(audio)
        
        # Damping factor
        damp = params.damping * 0.4 + 0.4
        room = params.room_size * 0.28 + 0.7
        
        for i, sample in enumerate(audio):
            # Pre-delay
            if predelay_samples > 0:
                delayed = self.predelay_buffer[self.predelay_index]
                self.predelay_buffer[self.predelay_index] = sample
                self.predelay_index = (self.predelay_index + 1) % predelay_samples
            else:
                delayed = sample
            
            # Comb filters in parallel
            comb_sum = 0.0
            for j in range(len(self.comb_delays)):
                # Read from buffer
                out = self.comb_buffers[j][self.comb_indices[j]]
                
                # Feedback with damping
                feedback = out * room * damp
                self.comb_buffers[j][self.comb_indices[j]] = delayed + feedback
                
                # Update index
                self.comb_indices[j] = (self.comb_indices[j] + 1) % self.comb_delays[j]
                
                comb_sum += out
            
            # Normalize
            comb_sum *= 0.125  # 1/8 for 8 comb filters
            
            # Allpass filters in series
            allpass_out = comb_sum
            for j in range(len(self.allpass_delays)):
                # Read from buffer
                delayed_ap = self.allpass_buffers[j][self.allpass_indices[j]]
                
                # Allpass processing
                ap_in = allpass_out + delayed_ap * 0.5
                self.allpass_buffers[j][self.allpass_indices[j]] = ap_in
                allpass_out = delayed_ap - ap_in * 0.5
                
                # Update index
                self.allpass_indices[j] = (self.allpass_indices[j] + 1) % self.allpass_delays[j]
            
            # Mix wet and dry
            output[i] = sample * params.dry_level + allpass_out * params.wet_level
        
        return output

class StudioDelay:
    """
    Professional delay with tempo sync and feedback filtering
    """
    
    def __init__(self, sample_rate: int = 44100, max_delay_ms: float = 2000):
        self.fs = sample_rate
        self.max_samples = int(max_delay_ms * 0.001 * sample_rate)
        self.buffer = np.zeros(self.max_samples)
        self.write_index = 0
        
        # Feedback lowpass filter
        self.lpf_state = 0.0
        
    def process(self, audio: np.ndarray, params: DelayParams) -> np.ndarray:
        """Apply delay with optional tempo sync"""
        
        # Calculate delay time
        if params.tempo_sync:
            # Convert note division to delay time
            beat_ms = 60000.0 / params.bpm
            divisions = {
                "1/1": 4.0, "1/2": 2.0, "1/4": 1.0, 
                "1/8": 0.5, "1/16": 0.25, "1/32": 0.125
            }
            multiplier = divisions.get(params.note_division, 1.0)
            delay_ms = beat_ms * multiplier
        else:
            delay_ms = params.delay_time_ms
        
        delay_samples = int(delay_ms * 0.001 * self.fs)
        delay_samples = min(delay_samples, self.max_samples - 1)
        
        output = np.zeros_like(audio)
        
        # Feedback filter coefficient (gentle highcut)
        lpf_coeff = 0.7
        
        for i, sample in enumerate(audio):
            # Read from delay line
            read_index = (self.write_index - delay_samples) % self.max_samples
            delayed = self.buffer[read_index]
            
            # Apply feedback with filtering
            self.lpf_state = delayed * lpf_coeff + self.lpf_state * (1 - lpf_coeff)
            feedback_signal = self.lpf_state * params.feedback
            
            # Write to buffer
            self.buffer[self.write_index] = sample + feedback_signal
            self.write_index = (self.write_index + 1) % self.max_samples
            
            # Mix output
            output[i] = sample * (1 - params.mix) + delayed * params.mix
        
        return output

class StereoChorus:
    """
    Multi-voice chorus for thickening and stereo enhancement
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.fs = sample_rate
        self.max_delay = int(0.05 * sample_rate)  # 50ms max
        self.buffers = [np.zeros(self.max_delay) for _ in range(4)]  # Up to 4 voices
        self.indices = [0] * 4
        self.lfo_phase = 0.0
        
    def process_stereo(self, audio: np.ndarray, params: ChorusParams) -> Tuple[np.ndarray, np.ndarray]:
        """Process mono to stereo chorus"""
        
        left = np.zeros_like(audio)
        right = np.zeros_like(audio)
        
        # LFO parameters
        lfo_increment = 2 * np.pi * params.rate_hz / self.fs
        depth_samples = params.depth_ms * 0.001 * self.fs
        
        for i, sample in enumerate(audio):
            # Generate LFOs for each voice (phase offset)
            lfo_values = []
            for v in range(params.voices):
                phase = self.lfo_phase + (v * 2 * np.pi / params.voices)
                lfo = np.sin(phase) * depth_samples + depth_samples + 10  # Offset from 0
                lfo_values.append(lfo)
            
            # Process each voice
            voice_sum_l = 0.0
            voice_sum_r = 0.0
            
            for v in range(params.voices):
                # Fractional delay interpolation
                delay_samples = lfo_values[v]
                delay_int = int(delay_samples)
                delay_frac = delay_samples - delay_int
                
                # Read with linear interpolation
                idx1 = (self.indices[v] - delay_int) % self.max_delay
                idx2 = (idx1 - 1) % self.max_delay
                
                delayed = self.buffers[v][idx1] * (1 - delay_frac) + \
                         self.buffers[v][idx2] * delay_frac
                
                # Pan voices in stereo field
                pan = (v / max(1, params.voices - 1)) * 2 - 1  # -1 to 1
                left_gain = np.sqrt(0.5 * (1 - pan))
                right_gain = np.sqrt(0.5 * (1 + pan))
                
                voice_sum_l += delayed * left_gain
                voice_sum_r += delayed * right_gain
                
                # Update buffer
                self.buffers[v][self.indices[v]] = sample
                self.indices[v] = (self.indices[v] + 1) % self.max_delay
            
            # Normalize and mix
            voice_sum_l /= max(1, params.voices)
            voice_sum_r /= max(1, params.voices)
            
            left[i] = sample * (1 - params.mix) + voice_sum_l * params.mix
            right[i] = sample * (1 - params.mix) + voice_sum_r * params.mix
            
            # Update LFO
            self.lfo_phase += lfo_increment
            if self.lfo_phase > 2 * np.pi:
                self.lfo_phase -= 2 * np.pi
        
        return left, right

class StereoImager:
    """
    Control stereo width with frequency-dependent processing
    M/S (Mid/Side) processing for professional width control
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.fs = sample_rate
        
        # Crossover filters for frequency-dependent processing
        self.setup_filters()
        
    def setup_filters(self):
        """Setup crossover filters"""
        # 2nd order Butterworth at 120Hz for bass mono
        nyquist = self.fs / 2
        self.bass_b, self.bass_a = signal.butter(2, 120 / nyquist, 'low')
        self.bass_state_l = signal.lfilter_zi(self.bass_b, self.bass_a)
        self.bass_state_r = signal.lfilter_zi(self.bass_b, self.bass_a)
        
        # High frequency enhancement filter at 8kHz
        self.high_b, self.high_a = signal.butter(2, 8000 / nyquist, 'high')
        self.high_state_l = signal.lfilter_zi(self.high_b, self.high_a)
        self.high_state_r = signal.lfilter_zi(self.high_b, self.high_a)
    
    def process_stereo(self, left: np.ndarray, right: np.ndarray, 
                       params: StereoImagerParams) -> Tuple[np.ndarray, np.ndarray]:
        """Process stereo width with M/S encoding"""
        
        # Convert to Mid/Side
        mid = (left + right) * 0.5
        side = (left - right) * 0.5
        
        # Process bass frequencies (make mono)
        bass_l, self.bass_state_l = signal.lfilter(self.bass_b, self.bass_a, left, zi=self.bass_state_l)
        bass_r, self.bass_state_r = signal.lfilter(self.bass_b, self.bass_a, right, zi=self.bass_state_r)
        bass_mono = (bass_l + bass_r) * 0.5
        
        # Process high frequencies (extra width)
        high_l, self.high_state_l = signal.lfilter(self.high_b, self.high_a, left, zi=self.high_state_l)
        high_r, self.high_state_r = signal.lfilter(self.high_b, self.high_a, right, zi=self.high_state_r)
        
        # Apply width control
        # Main signal
        processed_side = side * params.width
        
        # Reconstruct from M/S
        out_left = mid + processed_side
        out_right = mid - processed_side
        
        # Add bass mono to both channels
        out_left = out_left - bass_l + bass_mono
        out_right = out_right - bass_r + bass_mono
        
        # Add enhanced highs
        if params.high_freq_width > 1.0:
            width_boost = params.high_freq_width - 1.0
            out_left += high_l * width_boost * 0.5
            out_right -= high_r * width_boost * 0.5
        
        return out_left, out_right

class HarmonicExciter:
    """
    Add harmonic brilliance and presence
    Generates subtle harmonics to enhance clarity
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.fs = sample_rate
        self.setup_filter()
        
    def setup_filter(self):
        """Setup bandpass filter for exciter"""
        nyquist = self.fs / 2
        low_freq = 6000 / nyquist
        high_freq = min(12000 / nyquist, 0.99)
        self.bp_b, self.bp_a = signal.butter(2, [low_freq, high_freq], 'band')
        self.bp_state = signal.lfilter_zi(self.bp_b, self.bp_a)
    
    @numba.jit(nopython=True)
    def soft_clip(x: np.ndarray, drive: float) -> np.ndarray:
        """Soft clipping for harmonic generation"""
        gain = 1 + drive * 10
        x_driven = x * gain
        # Soft clipping function
        return np.tanh(x_driven) / gain
    
    def process(self, audio: np.ndarray, params: ExciterParams) -> np.ndarray:
        """Add harmonic excitement"""
        
        # Bandpass filter the signal
        filtered, self.bp_state = signal.lfilter(self.bp_b, self.bp_a, audio, zi=self.bp_state)
        
        # Generate harmonics through soft saturation
        excited = np.zeros_like(filtered)
        gain = 1 + params.drive * 10
        
        for i in range(len(filtered)):
            # Soft clipping
            x_driven = filtered[i] * gain
            excited[i] = np.tanh(x_driven) / gain
        
        # Mix with dry signal
        output = audio * (1 - params.mix) + excited * params.mix
        
        return output

class StudioEffectsChain:
    """
    Complete effects chain for professional mixing
    Signal flow: Input -> Delay -> Chorus -> Reverb -> Imager -> Exciter -> Output
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.fs = sample_rate
        
        # Initialize all effects
        self.reverb = PlateReverb(sample_rate)
        self.delay = StudioDelay(sample_rate)
        self.chorus = StereoChorus(sample_rate)
        self.imager = StereoImager(sample_rate)
        self.exciter = HarmonicExciter(sample_rate)
        
    def process(self, audio: np.ndarray,
                reverb_params: Optional[ReverbParams] = None,
                delay_params: Optional[DelayParams] = None,
                chorus_params: Optional[ChorusParams] = None,
                imager_params: Optional[StereoImagerParams] = None,
                exciter_params: Optional[ExciterParams] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process through complete effects chain
        Returns stereo output
        """
        
        # Start with mono
        x = audio.copy()
        
        # 1. Delay (mono)
        if delay_params:
            x = self.delay.process(x, delay_params)
        
        # 2. Chorus (mono to stereo)
        if chorus_params:
            left, right = self.chorus.process_stereo(x, chorus_params)
        else:
            left = x
            right = x.copy()
        
        # 3. Reverb (process each channel)
        if reverb_params:
            left = self.reverb.process(left, reverb_params)
            # Slightly different params for right (wider stereo)
            reverb_r = ReverbParams(
                room_size=reverb_params.room_size,
                damping=reverb_params.damping * 0.95,
                wet_level=reverb_params.wet_level,
                dry_level=reverb_params.dry_level,
                pre_delay_ms=reverb_params.pre_delay_ms * 1.1
            )
            right = self.reverb.process(right, reverb_r)
        
        # 4. Stereo Imager
        if imager_params:
            left, right = self.imager.process_stereo(left, right, imager_params)
        
        # 5. Harmonic Exciter (both channels)
        if exciter_params:
            left = self.exciter.process(left, exciter_params)
            right = self.exciter.process(right, exciter_params)
        
        return left, right