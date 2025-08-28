#!/usr/bin/env python3
"""
AiOke AI Vocal Auto-Mixer - Top-Notch Intelligent Vocal Processing
Advanced AI-powered automatic mixing for vocals with professional results
"""

import numpy as np
import scipy.signal as signal
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import json

@dataclass
class VocalCharacteristics:
    """Detected characteristics of the vocal input"""
    gender: str  # "male", "female", "child"
    vocal_range: str  # "bass", "baritone", "tenor", "alto", "soprano"
    skill_level: str  # "beginner", "intermediate", "advanced"
    vocal_style: str  # "speaking", "singing", "rapping", "screaming"
    confidence_score: float  # 0.0 to 1.0
    pitch_accuracy: float  # 0.0 to 1.0
    breath_support: float  # 0.0 to 1.0
    tone_quality: str  # "breathy", "clear", "nasal", "warm", "bright"

@dataclass 
class AIProcessingDecisions:
    """AI-determined processing parameters"""
    # EQ Decisions
    eq_curve: Dict[str, float]
    warmth_amount: float
    presence_boost: float
    air_frequency: float
    mud_reduction: float
    
    # Dynamics
    compression_ratio: float
    compression_threshold: float
    compression_attack: float
    compression_release: float
    expansion_gate: float
    
    # Enhancement
    reverb_type: str
    reverb_amount: float
    reverb_predelay: float
    delay_time: float
    delay_feedback: float
    
    # Correction
    pitch_correction_strength: float
    pitch_correction_speed: float
    formant_shift: float
    
    # Special Processing
    saturation_amount: float
    stereo_width: float
    doubler_amount: float
    exciter_amount: float

class AIVocalAutoMixer:
    """Top-notch AI-powered automatic vocal mixer"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.learning_history = []
        
        # Professional vocal processing knowledge base
        self.vocal_profiles = {
            "beginner_male": {
                "eq": {"80Hz": -3, "200Hz": -2, "800Hz": 1, "3kHz": 3, "8kHz": 2},
                "compression": {"ratio": 4, "threshold": -15, "attack": 5, "release": 50},
                "reverb": 0.35,
                "pitch_correction": 0.8,
                "warmth": 0.3
            },
            "beginner_female": {
                "eq": {"100Hz": -4, "250Hz": -1, "1kHz": 2, "4kHz": 3, "10kHz": 2},
                "compression": {"ratio": 3.5, "threshold": -18, "attack": 3, "release": 40},
                "reverb": 0.40,
                "pitch_correction": 0.85,
                "warmth": 0.2
            },
            "intermediate_male": {
                "eq": {"80Hz": -2, "200Hz": 0, "800Hz": 0, "3kHz": 2, "8kHz": 1},
                "compression": {"ratio": 3, "threshold": -20, "attack": 10, "release": 100},
                "reverb": 0.25,
                "pitch_correction": 0.5,
                "warmth": 0.2
            },
            "intermediate_female": {
                "eq": {"100Hz": -3, "250Hz": 0, "1kHz": 1, "4kHz": 2, "10kHz": 1},
                "compression": {"ratio": 2.5, "threshold": -22, "attack": 8, "release": 80},
                "reverb": 0.30,
                "pitch_correction": 0.6,
                "warmth": 0.15
            },
            "advanced_male": {
                "eq": {"80Hz": -1, "200Hz": 0, "800Hz": 0, "3kHz": 1, "8kHz": 0},
                "compression": {"ratio": 2, "threshold": -25, "attack": 15, "release": 150},
                "reverb": 0.15,
                "pitch_correction": 0.2,
                "warmth": 0.1
            },
            "advanced_female": {
                "eq": {"100Hz": -2, "250Hz": 0, "1kHz": 0, "4kHz": 1, "10kHz": 0},
                "compression": {"ratio": 2, "threshold": -25, "attack": 12, "release": 120},
                "reverb": 0.20,
                "pitch_correction": 0.3,
                "warmth": 0.1
            }
        }
        
        # Music genre-specific mixing styles
        self.genre_styles = {
            "pop": {"brightness": 0.7, "compression": 0.8, "reverb": 0.3},
            "rock": {"brightness": 0.5, "compression": 0.6, "reverb": 0.2},
            "jazz": {"brightness": 0.3, "compression": 0.3, "reverb": 0.4},
            "classical": {"brightness": 0.4, "compression": 0.2, "reverb": 0.6},
            "hip_hop": {"brightness": 0.6, "compression": 0.9, "reverb": 0.1},
            "electronic": {"brightness": 0.8, "compression": 0.7, "reverb": 0.4}
        }
    
    def analyze_vocal(self, audio: np.ndarray) -> VocalCharacteristics:
        """Analyze vocal characteristics using AI"""
        
        # Gender detection based on fundamental frequency
        fundamental = self._detect_fundamental_frequency(audio)
        
        if fundamental < 165:  # E3
            gender = "male"
            if fundamental < 110:  # A2
                vocal_range = "bass"
            elif fundamental < 147:  # D3
                vocal_range = "baritone"
            else:
                vocal_range = "tenor"
        elif fundamental < 250:  # ~B3
            gender = "female"
            vocal_range = "alto"
        else:
            gender = "female"
            vocal_range = "soprano" if fundamental > 330 else "mezzo-soprano"
        
        # Detect skill level based on multiple factors
        pitch_accuracy = self._analyze_pitch_stability(audio)
        breath_support = self._analyze_breath_support(audio)
        vibrato_quality = self._analyze_vibrato(audio)
        
        skill_score = (pitch_accuracy * 0.4 + breath_support * 0.3 + vibrato_quality * 0.3)
        
        if skill_score < 0.4:
            skill_level = "beginner"
        elif skill_score < 0.7:
            skill_level = "intermediate"
        else:
            skill_level = "advanced"
        
        # Detect vocal style
        vocal_style = self._detect_vocal_style(audio)
        
        # Analyze tone quality
        tone_quality = self._analyze_tone_quality(audio)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(audio)
        
        return VocalCharacteristics(
            gender=gender,
            vocal_range=vocal_range,
            skill_level=skill_level,
            vocal_style=vocal_style,
            confidence_score=confidence_score,
            pitch_accuracy=pitch_accuracy,
            breath_support=breath_support,
            tone_quality=tone_quality
        )
    
    def make_ai_decisions(self, 
                         vocal_chars: VocalCharacteristics,
                         music_analysis: Dict,
                         target_style: str = "auto") -> AIProcessingDecisions:
        """Make intelligent mixing decisions based on analysis"""
        
        # Select base profile
        profile_key = f"{vocal_chars.skill_level}_{vocal_chars.gender}"
        base_profile = self.vocal_profiles.get(profile_key, self.vocal_profiles["beginner_male"])
        
        # Adaptive EQ curve
        eq_curve = base_profile["eq"].copy()
        
        # Adjust for tone quality
        if vocal_chars.tone_quality == "nasal":
            eq_curve["1kHz"] = eq_curve.get("1kHz", 0) - 3
            eq_curve["2kHz"] = eq_curve.get("2kHz", 0) - 2
        elif vocal_chars.tone_quality == "breathy":
            eq_curve["8kHz"] = eq_curve.get("8kHz", 0) + 2
            eq_curve["200Hz"] = eq_curve.get("200Hz", 0) - 1
        elif vocal_chars.tone_quality == "muddy":
            eq_curve["250Hz"] = eq_curve.get("250Hz", 0) - 4
            eq_curve["500Hz"] = eq_curve.get("500Hz", 0) - 2
        
        # Intelligent compression
        comp = base_profile["compression"]
        
        # Adjust compression for vocal style
        if vocal_chars.vocal_style == "singing":
            compression_ratio = comp["ratio"]
            compression_threshold = comp["threshold"]
        elif vocal_chars.vocal_style == "rapping":
            compression_ratio = comp["ratio"] * 1.5
            compression_threshold = comp["threshold"] + 5
        else:  # speaking
            compression_ratio = comp["ratio"] * 0.7
            compression_threshold = comp["threshold"] - 5
        
        # Smart reverb selection
        if vocal_chars.skill_level == "beginner":
            # More reverb for beginners to smooth imperfections
            reverb_amount = base_profile["reverb"] * 1.2
            reverb_type = "hall"  # Forgiving
            reverb_predelay = 20  # ms
        elif vocal_chars.skill_level == "intermediate":
            reverb_amount = base_profile["reverb"]
            reverb_type = "room"  # Natural
            reverb_predelay = 15
        else:
            reverb_amount = base_profile["reverb"] * 0.8
            reverb_type = "plate"  # Professional
            reverb_predelay = 10
        
        # Pitch correction based on accuracy
        if vocal_chars.pitch_accuracy < 0.5:
            pitch_correction_strength = 0.9  # Strong correction
            pitch_correction_speed = 20  # Fast
        elif vocal_chars.pitch_accuracy < 0.7:
            pitch_correction_strength = 0.6  # Moderate
            pitch_correction_speed = 50  # Medium
        else:
            pitch_correction_strength = 0.3  # Light touch
            pitch_correction_speed = 100  # Slow
        
        # Enhancement decisions
        warmth_amount = base_profile.get("warmth", 0.2)
        
        # Presence boost for clarity in mix
        if music_analysis.get("density", 0.5) > 0.7:
            presence_boost = 3.0  # Cut through dense mix
        else:
            presence_boost = 1.5
        
        # Air frequency for sparkle
        if vocal_chars.gender == "male":
            air_frequency = 10000  # 10kHz
        else:
            air_frequency = 12000  # 12kHz
        
        # Special processing for beginners
        if vocal_chars.skill_level == "beginner":
            saturation_amount = 0.15  # Add warmth and thickness
            doubler_amount = 0.2  # Subtle doubling for fullness
            exciter_amount = 0.1  # Gentle brightness
        else:
            saturation_amount = 0.05
            doubler_amount = 0.0
            exciter_amount = 0.05
        
        return AIProcessingDecisions(
            eq_curve=eq_curve,
            warmth_amount=warmth_amount,
            presence_boost=presence_boost,
            air_frequency=air_frequency,
            mud_reduction=2.0 if vocal_chars.tone_quality == "muddy" else 0.5,
            compression_ratio=compression_ratio,
            compression_threshold=compression_threshold,
            compression_attack=comp["attack"],
            compression_release=comp["release"],
            expansion_gate=-40 if vocal_chars.skill_level == "beginner" else -50,
            reverb_type=reverb_type,
            reverb_amount=reverb_amount,
            reverb_predelay=reverb_predelay,
            delay_time=0.0,  # No delay by default
            delay_feedback=0.0,
            pitch_correction_strength=pitch_correction_strength,
            pitch_correction_speed=pitch_correction_speed,
            formant_shift=0.0,  # Natural formants
            saturation_amount=saturation_amount,
            stereo_width=0.2,  # Subtle stereo enhancement
            doubler_amount=doubler_amount,
            exciter_amount=exciter_amount
        )
    
    def process_vocal_with_ai(self, 
                              vocal: np.ndarray,
                              music: np.ndarray,
                              genre: str = "auto") -> np.ndarray:
        """Process vocal with full AI automation"""
        
        # Analyze vocal characteristics
        vocal_chars = self.analyze_vocal(vocal)
        
        # Analyze music context
        music_analysis = self._analyze_music_context(music)
        
        # Make AI mixing decisions
        decisions = self.make_ai_decisions(vocal_chars, music_analysis, genre)
        
        # Apply processing chain
        processed = vocal.copy()
        
        # 1. Gate/Expander (remove noise)
        processed = self._apply_gate(processed, decisions.expansion_gate)
        
        # 2. EQ
        processed = self._apply_multiband_eq(processed, decisions.eq_curve)
        
        # 3. Compression
        processed = self._apply_compression(
            processed,
            decisions.compression_ratio,
            decisions.compression_threshold,
            decisions.compression_attack,
            decisions.compression_release
        )
        
        # 4. Pitch Correction
        if decisions.pitch_correction_strength > 0:
            processed = self._apply_pitch_correction(
                processed,
                decisions.pitch_correction_strength,
                decisions.pitch_correction_speed
            )
        
        # 5. Saturation (warmth)
        if decisions.saturation_amount > 0:
            processed = self._apply_saturation(processed, decisions.saturation_amount)
        
        # 6. Enhancement
        if decisions.presence_boost > 0:
            processed = self._boost_presence(processed, decisions.presence_boost)
        
        if decisions.exciter_amount > 0:
            processed = self._apply_exciter(processed, decisions.exciter_amount, decisions.air_frequency)
        
        # 7. Spatial Effects
        if decisions.doubler_amount > 0:
            processed = self._apply_doubler(processed, decisions.doubler_amount)
        
        # 8. Reverb
        if decisions.reverb_amount > 0:
            processed = self._apply_reverb(
                processed,
                decisions.reverb_type,
                decisions.reverb_amount,
                decisions.reverb_predelay
            )
        
        # 9. Final Limiting
        processed = self._apply_limiter(processed, -0.3)
        
        # Store learning data
        self._update_learning_history(vocal_chars, decisions, music_analysis)
        
        return processed
    
    def _detect_fundamental_frequency(self, audio: np.ndarray) -> float:
        """Detect fundamental frequency using autocorrelation"""
        # Autocorrelation method
        corr = np.correlate(audio, audio, mode='full')
        corr = corr[len(corr)//2:]
        
        # Find first peak after zero lag
        d = np.diff(corr)
        start = 20  # Skip very short lags
        peak = start + np.argmax(corr[start:start+500])
        
        if peak > 0:
            fundamental = self.sample_rate / peak
        else:
            fundamental = 220  # Default A3
            
        return np.clip(fundamental, 50, 2000)
    
    def _analyze_pitch_stability(self, audio: np.ndarray) -> float:
        """Analyze how stable the pitch is"""
        window_size = 2048
        hop = 1024
        pitches = []
        
        for i in range(0, len(audio) - window_size, hop):
            window = audio[i:i+window_size]
            pitch = self._detect_fundamental_frequency(window)
            pitches.append(pitch)
        
        if len(pitches) > 1:
            # Calculate pitch variation
            pitches = np.array(pitches)
            mean_pitch = np.mean(pitches)
            std_pitch = np.std(pitches)
            
            # Convert to stability score (lower variation = higher score)
            stability = 1.0 - min(std_pitch / mean_pitch, 1.0)
        else:
            stability = 0.5
            
        return stability
    
    def _analyze_breath_support(self, audio: np.ndarray) -> float:
        """Analyze breath support quality"""
        # Look for consistent amplitude envelope
        envelope = self._extract_envelope(audio)
        
        # Good breath support has steady amplitude
        variation = np.std(envelope) / (np.mean(envelope) + 1e-10)
        
        # Convert to score
        support = 1.0 - min(variation, 1.0)
        return support
    
    def _analyze_vibrato(self, audio: np.ndarray) -> float:
        """Analyze vibrato quality"""
        # Detect pitch modulation patterns
        window_size = 512
        hop = 256
        pitch_curve = []
        
        for i in range(0, len(audio) - window_size, hop):
            window = audio[i:i+window_size]
            pitch = self._detect_fundamental_frequency(window)
            pitch_curve.append(pitch)
        
        if len(pitch_curve) > 10:
            # Look for regular oscillation (vibrato)
            pitch_curve = np.array(pitch_curve)
            
            # Detrend
            detrended = pitch_curve - np.mean(pitch_curve)
            
            # Check for periodicity
            autocorr = np.correlate(detrended, detrended, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Vibrato typically 4-7 Hz
            expected_lag = int(self.sample_rate / (hop * 5.5))  # 5.5 Hz center
            
            if expected_lag < len(autocorr):
                vibrato_strength = autocorr[expected_lag] / (autocorr[0] + 1e-10)
                return np.clip(vibrato_strength, 0, 1)
        
        return 0.0
    
    def _detect_vocal_style(self, audio: np.ndarray) -> str:
        """Detect if singing, speaking, or rapping"""
        # Analyze rhythmic patterns and pitch variation
        pitch_variation = self._analyze_pitch_stability(audio)
        
        # Analyze onset density (for rap detection)
        onsets = self._detect_onsets(audio)
        onset_density = len(onsets) / (len(audio) / self.sample_rate)
        
        # Classification logic
        if onset_density > 8:  # High onset density
            return "rapping"
        elif pitch_variation < 0.3:  # Low pitch variation
            return "speaking"
        else:
            return "singing"
    
    def _analyze_tone_quality(self, audio: np.ndarray) -> str:
        """Analyze tonal characteristics"""
        # FFT analysis
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
        
        # Analyze frequency balance
        low_energy = np.sum(magnitude[freqs < 250])
        mid_energy = np.sum(magnitude[(freqs >= 250) & (freqs < 2000)])
        high_energy = np.sum(magnitude[freqs >= 2000])
        
        total_energy = low_energy + mid_energy + high_energy
        
        if total_energy > 0:
            low_ratio = low_energy / total_energy
            mid_ratio = mid_energy / total_energy
            high_ratio = high_energy / total_energy
            
            # Classify tone
            if low_ratio > 0.5:
                return "muddy"
            elif mid_ratio > 0.6 and (freqs[1000] < freqs[2000]):
                return "nasal"
            elif high_ratio > 0.4:
                return "bright"
            elif high_energy < total_energy * 0.1:
                return "warm"
            else:
                return "clear"
        
        return "clear"
    
    def _calculate_confidence(self, audio: np.ndarray) -> float:
        """Calculate vocalist confidence score"""
        # Analyze amplitude consistency
        envelope = self._extract_envelope(audio)
        
        # Confident vocals have strong, consistent amplitude
        mean_level = np.mean(envelope)
        peak_level = np.max(envelope)
        
        if peak_level > 0:
            consistency = mean_level / peak_level
        else:
            consistency = 0
            
        # Check for hesitations (sudden drops)
        drops = np.sum(np.diff(envelope) < -0.1 * peak_level)
        hesitation_penalty = min(drops / 100, 0.5)
        
        confidence = consistency - hesitation_penalty
        return np.clip(confidence, 0, 1)
    
    def _analyze_music_context(self, music: np.ndarray) -> Dict:
        """Analyze the music to understand mixing context"""
        # FFT for frequency content
        fft = np.fft.rfft(music)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(music), 1/self.sample_rate)
        
        # Analyze frequency density
        vocal_range = (80, 8000)  # Hz
        vocal_band = magnitude[(freqs >= vocal_range[0]) & (freqs <= vocal_range[1])]
        
        density = np.mean(vocal_band) / (np.max(magnitude) + 1e-10)
        
        # Analyze dynamics
        envelope = self._extract_envelope(music)
        dynamics = np.std(envelope) / (np.mean(envelope) + 1e-10)
        
        return {
            "density": density,
            "dynamics": dynamics,
            "needs_cut_through": density > 0.7,
            "frequency_mask": self._find_frequency_masks(magnitude, freqs)
        }
    
    def _find_frequency_masks(self, magnitude: np.ndarray, freqs: np.ndarray) -> List[float]:
        """Find frequencies that might mask the vocal"""
        # Look for peaks in the vocal range that could cause masking
        vocal_range = (200, 5000)
        mask_freqs = []
        
        in_range = (freqs >= vocal_range[0]) & (freqs <= vocal_range[1])
        vocal_magnitude = magnitude[in_range]
        vocal_freqs = freqs[in_range]
        
        # Find peaks
        threshold = np.mean(vocal_magnitude) + 2 * np.std(vocal_magnitude)
        peaks = vocal_magnitude > threshold
        
        if np.any(peaks):
            peak_freqs = vocal_freqs[peaks]
            mask_freqs = peak_freqs.tolist()
        
        return mask_freqs
    
    def _extract_envelope(self, audio: np.ndarray) -> np.ndarray:
        """Extract amplitude envelope"""
        return np.abs(signal.hilbert(audio))
    
    def _detect_onsets(self, audio: np.ndarray) -> List[int]:
        """Detect note onsets for rhythm analysis"""
        envelope = self._extract_envelope(audio)
        
        # Spectral flux method
        window_size = 2048
        hop = 512
        onsets = []
        
        for i in range(hop, len(audio) - window_size, hop):
            prev_window = audio[i-hop:i-hop+window_size]
            curr_window = audio[i:i+window_size]
            
            prev_fft = np.abs(np.fft.rfft(prev_window))
            curr_fft = np.abs(np.fft.rfft(curr_window))
            
            flux = np.sum(np.maximum(0, curr_fft - prev_fft))
            
            if flux > np.mean(envelope) * 2:
                onsets.append(i)
        
        return onsets
    
    def _apply_gate(self, audio: np.ndarray, threshold_db: float) -> np.ndarray:
        """Apply noise gate"""
        threshold = 10 ** (threshold_db / 20)
        envelope = self._extract_envelope(audio)
        
        gate = np.where(envelope > threshold, 1.0, 0.0)
        
        # Smooth gate to avoid clicks
        gate = signal.savgol_filter(gate, 101, 3)
        
        return audio * gate
    
    def _apply_multiband_eq(self, audio: np.ndarray, eq_curve: Dict[str, float]) -> np.ndarray:
        """Apply multiband EQ"""
        result = audio.copy()
        
        for freq_str, gain_db in eq_curve.items():
            # Parse frequency string
            if "kHz" in freq_str or "khz" in freq_str.lower():
                freq = float(freq_str.lower().replace("khz", "").replace("k", "")) * 1000
            elif "k" in freq_str.lower():
                freq = float(freq_str.lower().replace("k", "")) * 1000
            elif "Hz" in freq_str or "hz" in freq_str.lower():
                freq = float(freq_str.lower().replace("hz", ""))
            else:
                # Try to parse as number directly
                try:
                    freq = float(freq_str)
                except:
                    continue  # Skip unparseable frequencies
            
            result = self._apply_bell_filter(result, freq, gain_db)
        
        return result
    
    def _apply_bell_filter(self, audio: np.ndarray, freq: float, gain_db: float, q: float = 1.0) -> np.ndarray:
        """Apply bell-shaped EQ filter"""
        if gain_db == 0:
            return audio
            
        # Design filter
        nyquist = self.sample_rate / 2
        if freq >= nyquist:
            return audio
            
        w0 = freq / nyquist
        
        # Peaking EQ filter coefficients
        A = 10 ** (gain_db / 40)
        omega = 2 * np.pi * w0
        sin_omega = np.sin(omega)
        cos_omega = np.cos(omega)
        alpha = sin_omega / (2 * q)
        
        b0 = 1 + alpha * A
        b1 = -2 * cos_omega
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_omega
        a2 = 1 - alpha / A
        
        # Normalize
        b = [b0/a0, b1/a0, b2/a0]
        a = [1, a1/a0, a2/a0]
        
        return signal.lfilter(b, a, audio)
    
    def _apply_compression(self, audio: np.ndarray, ratio: float, threshold_db: float,
                          attack_ms: float, release_ms: float) -> np.ndarray:
        """Apply dynamic range compression"""
        threshold = 10 ** (threshold_db / 20)
        
        # Convert times to samples
        attack_samples = int(attack_ms * self.sample_rate / 1000)
        release_samples = int(release_ms * self.sample_rate / 1000)
        
        envelope = self._extract_envelope(audio)
        gain_reduction = np.ones_like(envelope)
        
        for i in range(len(envelope)):
            if envelope[i] > threshold:
                # Above threshold - apply compression
                over = envelope[i] - threshold
                compressed = threshold + over / ratio
                gain_reduction[i] = compressed / envelope[i]
            
            # Smooth gain changes
            if i > 0:
                if gain_reduction[i] < gain_reduction[i-1]:
                    # Attack
                    alpha = 1.0 - np.exp(-1.0 / attack_samples)
                else:
                    # Release
                    alpha = 1.0 - np.exp(-1.0 / release_samples)
                
                gain_reduction[i] = gain_reduction[i-1] + alpha * (gain_reduction[i] - gain_reduction[i-1])
        
        return audio * gain_reduction
    
    def _apply_pitch_correction(self, audio: np.ndarray, strength: float, speed_ms: float) -> np.ndarray:
        """Apply pitch correction (simplified auto-tune)"""
        # This is a simplified version - real pitch correction is more complex
        
        # Detect pitch
        pitch = self._detect_fundamental_frequency(audio)
        
        # Find nearest musical note
        A4 = 440
        notes = []
        for i in range(-48, 48):  # 4 octaves up and down
            note_freq = A4 * (2 ** (i/12))
            notes.append(note_freq)
        
        notes = np.array(notes)
        nearest_idx = np.argmin(np.abs(notes - pitch))
        target_pitch = notes[nearest_idx]
        
        # Calculate pitch shift ratio
        shift_ratio = target_pitch / pitch
        
        # Apply pitch shift with strength control
        shift_ratio = 1.0 + (shift_ratio - 1.0) * strength
        
        # Resample to shift pitch (simplified)
        if shift_ratio != 1.0:
            num_samples = int(len(audio) / shift_ratio)
            resampled = signal.resample(audio, num_samples)
            
            # Stretch back to original length
            processed = signal.resample(resampled, len(audio))
        else:
            processed = audio
        
        return processed
    
    def _apply_saturation(self, audio: np.ndarray, amount: float) -> np.ndarray:
        """Apply tape-style saturation for warmth"""
        # Soft clipping function
        def soft_clip(x):
            return np.tanh(x * (1 + amount * 2))
        
        # Apply saturation
        saturated = soft_clip(audio * (1 + amount))
        
        # Compensate for level change
        original_rms = np.sqrt(np.mean(audio ** 2))
        saturated_rms = np.sqrt(np.mean(saturated ** 2))
        
        if saturated_rms > 0:
            saturated *= original_rms / saturated_rms
        
        # Mix with dry signal
        return audio * (1 - amount * 0.3) + saturated * (amount * 0.3)
    
    def _boost_presence(self, audio: np.ndarray, boost_db: float) -> np.ndarray:
        """Boost presence frequencies (2-5kHz)"""
        # Apply bell filter centered at 3.5kHz
        return self._apply_bell_filter(audio, 3500, boost_db, q=0.7)
    
    def _apply_exciter(self, audio: np.ndarray, amount: float, freq: float) -> np.ndarray:
        """Apply harmonic exciter for brightness"""
        # High-pass filter
        nyquist = self.sample_rate / 2
        if freq >= nyquist:
            return audio
            
        sos = signal.butter(2, freq / nyquist, 'high', output='sos')
        high_passed = signal.sosfilt(sos, audio)
        
        # Generate harmonics through soft distortion
        excited = np.tanh(high_passed * 3) * amount
        
        # Mix with original
        return audio + excited * 0.1
    
    def _apply_doubler(self, audio: np.ndarray, amount: float) -> np.ndarray:
        """Apply vocal doubling effect"""
        # Create slightly delayed and pitched copy
        delay_samples = int(0.02 * self.sample_rate)  # 20ms
        
        # Slight pitch shift (few cents)
        pitch_shift = 1.01  # 1% higher
        doubled = signal.resample(audio, int(len(audio) / pitch_shift))
        doubled = signal.resample(doubled, len(audio))
        
        # Delay
        delayed = np.pad(doubled, (delay_samples, 0), mode='constant')[:-delay_samples]
        
        # Mix
        return audio + delayed * amount
    
    def _apply_reverb(self, audio: np.ndarray, reverb_type: str, 
                     amount: float, predelay_ms: float) -> np.ndarray:
        """Apply reverb based on type"""
        predelay_samples = int(predelay_ms * self.sample_rate / 1000)
        
        if reverb_type == "plate":
            # Plate reverb - bright and smooth
            delays = [23, 29, 31, 37, 41]
            gains = [0.7, 0.6, 0.5, 0.4, 0.3]
        elif reverb_type == "hall":
            # Hall reverb - long and spacious
            delays = [37, 43, 47, 53, 59]
            gains = [0.8, 0.7, 0.6, 0.5, 0.4]
        else:  # room
            # Room reverb - short and natural
            delays = [13, 17, 19, 23, 29]
            gains = [0.6, 0.5, 0.4, 0.3, 0.2]
        
        reverb = np.zeros_like(audio)
        
        # Predelay
        if predelay_samples > 0:
            audio = np.pad(audio, (predelay_samples, 0), mode='constant')[:-predelay_samples]
        
        # Comb filters for reverb
        for delay_ms, gain in zip(delays, gains):
            delay_samples = int(delay_ms * self.sample_rate / 1000)
            delayed = np.pad(audio, (delay_samples, 0), mode='constant')[:-delay_samples]
            reverb += delayed * gain
        
        # Mix wet and dry
        return audio * (1 - amount) + reverb * amount
    
    def _apply_limiter(self, audio: np.ndarray, threshold_db: float) -> np.ndarray:
        """Apply brickwall limiter"""
        threshold = 10 ** (threshold_db / 20)
        
        # Look-ahead buffer
        lookahead_ms = 5
        lookahead_samples = int(lookahead_ms * self.sample_rate / 1000)
        
        # Get envelope with lookahead
        envelope = self._extract_envelope(audio)
        
        # Calculate gain reduction
        gain = np.ones_like(envelope)
        
        for i in range(len(envelope) - lookahead_samples):
            # Check future samples
            future_peak = np.max(envelope[i:i+lookahead_samples])
            if future_peak > threshold:
                gain[i] = threshold / future_peak
        
        # Smooth gain changes
        gain = signal.savgol_filter(gain, 51, 3)
        
        return audio * gain
    
    def _update_learning_history(self, vocal_chars: VocalCharacteristics,
                                decisions: AIProcessingDecisions,
                                music_analysis: Dict):
        """Store learning data for continuous improvement"""
        self.learning_history.append({
            "timestamp": np.datetime64('now'),
            "vocal_characteristics": vocal_chars,
            "decisions": decisions,
            "music_context": music_analysis
        })
        
        # Keep only recent history
        if len(self.learning_history) > 1000:
            self.learning_history = self.learning_history[-1000:]
    
    def get_mixing_report(self) -> str:
        """Generate a report of the AI mixing decisions"""
        if not self.learning_history:
            return "No mixing sessions recorded yet."
        
        latest = self.learning_history[-1]
        vocal_chars = latest["vocal_characteristics"]
        decisions = latest["decisions"]
        
        report = f"""
🎤 AI Vocal Auto-Mixing Report
================================

Vocal Analysis:
- Gender: {vocal_chars.gender}
- Range: {vocal_chars.vocal_range}
- Skill Level: {vocal_chars.skill_level}
- Style: {vocal_chars.vocal_style}
- Tone Quality: {vocal_chars.tone_quality}
- Pitch Accuracy: {vocal_chars.pitch_accuracy:.1%}
- Confidence: {vocal_chars.confidence_score:.1%}

AI Decisions:
- Compression: {decisions.compression_ratio}:1 @ {decisions.compression_threshold}dB
- Reverb: {decisions.reverb_type} at {decisions.reverb_amount:.1%}
- Pitch Correction: {decisions.pitch_correction_strength:.1%} strength
- EQ Adjustments: {len(decisions.eq_curve)} bands modified
- Special Processing:
  • Saturation: {decisions.saturation_amount:.1%}
  • Doubler: {decisions.doubler_amount:.1%}
  • Exciter: {decisions.exciter_amount:.1%}

The AI has optimized the vocal for {vocal_chars.skill_level} level performance,
enhancing weak points while preserving natural character.
        """
        
        return report


def demonstrate_ai_mixing():
    """Demonstrate the AI vocal auto-mixer"""
    print("🤖 AiOke AI Vocal Auto-Mixer Demo")
    print("=" * 50)
    
    # Create mixer instance
    mixer = AIVocalAutoMixer()
    
    # Create test signals
    sample_rate = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Simulate vocal (varying pitch to simulate singing)
    vocal_freq = 220 + 50 * np.sin(2 * np.pi * 0.5 * t)  # Vibrato
    vocal = np.sin(2 * np.pi * vocal_freq * t)
    
    # Add some noise to simulate beginner
    vocal += np.random.normal(0, 0.01, len(vocal))
    
    # Simulate music (chord)
    music = (np.sin(2 * np.pi * 261.63 * t) +  # C
             np.sin(2 * np.pi * 329.63 * t) +  # E
             np.sin(2 * np.pi * 392.00 * t)) / 3  # G
    
    print("\n🎵 Processing vocal with AI auto-mixing...")
    
    # Process with AI
    processed = mixer.process_vocal_with_ai(vocal, music)
    
    # Generate report
    report = mixer.get_mixing_report()
    print(report)
    
    print("\n✅ AI mixing complete! The vocal has been intelligently processed")
    print("   based on detected characteristics and skill level.")
    
    return processed


if __name__ == "__main__":
    demonstrate_ai_mixing()