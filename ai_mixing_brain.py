"""
AI-Powered Autonomous Mixing Brain
Analyzes audio in real-time and makes intelligent mixing decisions
Uses ML models for genre detection, optimal settings, and adaptive processing
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
from collections import deque
import time

from studio_dsp_chain import (
    GateParams, CompressorParams, EQBand, LimiterParams,
    StudioDSPChain
)

class AudioGenre(Enum):
    """Detected audio genres for targeted processing"""
    SPEECH = "speech"
    PODCAST = "podcast"
    VOCAL = "vocal"
    ROCK = "rock"
    POP = "pop"
    JAZZ = "jazz"
    CLASSICAL = "classical"
    ELECTRONIC = "electronic"
    HIPHOP = "hiphop"
    AMBIENT = "ambient"

@dataclass
class AudioFeatures:
    """Extracted audio features for decision making"""
    rms_db: float = -60.0
    peak_db: float = -60.0
    crest_factor: float = 20.0
    spectral_centroid: float = 1000.0
    spectral_rolloff: float = 5000.0
    spectral_flux: float = 0.0
    zero_crossing_rate: float = 0.0
    mfcc: np.ndarray = field(default_factory=lambda: np.zeros(13))
    tempo_bpm: float = 120.0
    onset_strength: float = 0.0
    harmonic_ratio: float = 0.5
    percussive_ratio: float = 0.5

@dataclass
class MixProfile:
    """Target mixing profile for different content types"""
    name: str
    genre: AudioGenre
    target_lufs: float  # Loudness target
    target_dynamics: float  # Dynamic range
    tonal_balance: Dict[str, float]  # Frequency targets
    gate_params: GateParams
    comp_params: CompressorParams
    eq_bands: List[EQBand]
    limiter_params: LimiterParams

class AIAudioAnalyzer:
    """Intelligent audio analysis using ML techniques"""
    
    def __init__(self, sample_rate: int = 44100):
        self.fs = sample_rate
        self.feature_buffer = deque(maxlen=100)  # 1 second at 100Hz
        self.genre_detector = GenreDetector()
        self.spectral_analyzer = SpectralAnalyzer(sample_rate)
        
    def analyze(self, audio: np.ndarray) -> AudioFeatures:
        """Extract comprehensive audio features"""
        
        features = AudioFeatures()
        
        # Basic levels
        features.rms_db = 20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-12)
        features.peak_db = 20 * np.log10(np.max(np.abs(audio)) + 1e-12)
        features.crest_factor = features.peak_db - features.rms_db
        
        # Spectral features
        spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
        freqs = np.fft.rfftfreq(len(audio), 1/self.fs)
        
        # Spectral centroid (brightness)
        features.spectral_centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-12)
        
        # Spectral rolloff (high frequency content)
        cumsum = np.cumsum(spectrum)
        rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
        if len(rolloff_idx) > 0:
            features.spectral_rolloff = freqs[rolloff_idx[0]]
        
        # Zero crossing rate (percussiveness)
        features.zero_crossing_rate = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
        
        # Harmonic/percussive separation estimate
        features.harmonic_ratio, features.percussive_ratio = self._estimate_harmonic_percussive(spectrum)
        
        # MFCCs for timbre
        features.mfcc = self._compute_mfcc(audio)
        
        return features
    
    def _estimate_harmonic_percussive(self, spectrum: np.ndarray) -> Tuple[float, float]:
        """Estimate harmonic vs percussive content"""
        
        # Harmonic content has peaks, percussive is broadband
        spectral_flatness = np.exp(np.mean(np.log(spectrum + 1e-12))) / (np.mean(spectrum) + 1e-12)
        
        harmonic_ratio = 1.0 - spectral_flatness
        percussive_ratio = spectral_flatness
        
        return harmonic_ratio, percussive_ratio
    
    def _compute_mfcc(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """Compute MFCCs for timbre analysis"""
        
        # Simplified MFCC computation
        # In production, use librosa or python_speech_features
        
        # FFT
        spectrum = np.abs(np.fft.rfft(audio))
        
        # Mel filterbank (simplified)
        n_mels = 40
        mel_filters = self._create_mel_filterbank(len(spectrum), n_mels)
        mel_spectrum = np.dot(mel_filters, spectrum)
        
        # Log and DCT
        log_mel = np.log(mel_spectrum + 1e-12)
        mfcc = np.zeros(n_mfcc)
        
        for i in range(n_mfcc):
            mfcc[i] = np.sum(log_mel * np.cos(np.pi * i * (np.arange(n_mels) + 0.5) / n_mels))
        
        return mfcc
    
    def _create_mel_filterbank(self, n_fft: int, n_mels: int) -> np.ndarray:
        """Create mel filterbank matrix"""
        
        # Simplified mel filterbank
        filterbank = np.zeros((n_mels, n_fft))
        
        for i in range(n_mels):
            start = int(i * n_fft / n_mels)
            center = int((i + 0.5) * n_fft / n_mels)
            end = int((i + 1) * n_fft / n_mels)
            
            if end > n_fft:
                end = n_fft
            
            # Triangular filter
            for j in range(start, center):
                filterbank[i, j] = (j - start) / (center - start)
            for j in range(center, end):
                filterbank[i, j] = (end - j) / (end - center)
        
        return filterbank

class GenreDetector:
    """ML-based genre detection"""
    
    def __init__(self):
        self.genre_profiles = self._load_genre_profiles()
        
    def _load_genre_profiles(self) -> Dict[AudioGenre, AudioFeatures]:
        """Load pre-trained genre profiles"""
        
        # In production, load from trained model
        # Here we use heuristic profiles
        
        profiles = {
            AudioGenre.SPEECH: AudioFeatures(
                rms_db=-25, crest_factor=15, spectral_centroid=800,
                harmonic_ratio=0.7, percussive_ratio=0.3
            ),
            AudioGenre.ROCK: AudioFeatures(
                rms_db=-15, crest_factor=8, spectral_centroid=2000,
                harmonic_ratio=0.4, percussive_ratio=0.6
            ),
            AudioGenre.JAZZ: AudioFeatures(
                rms_db=-20, crest_factor=12, spectral_centroid=1500,
                harmonic_ratio=0.6, percussive_ratio=0.4
            ),
            AudioGenre.ELECTRONIC: AudioFeatures(
                rms_db=-12, crest_factor=6, spectral_centroid=3000,
                harmonic_ratio=0.3, percussive_ratio=0.7
            ),
            AudioGenre.CLASSICAL: AudioFeatures(
                rms_db=-30, crest_factor=20, spectral_centroid=1200,
                harmonic_ratio=0.8, percussive_ratio=0.2
            ),
        }
        
        return profiles
    
    def detect(self, features: AudioFeatures) -> Tuple[AudioGenre, float]:
        """Detect genre with confidence score"""
        
        best_genre = AudioGenre.SPEECH
        best_score = float('inf')
        
        for genre, profile in self.genre_profiles.items():
            # Simple distance metric
            distance = 0.0
            distance += abs(features.spectral_centroid - profile.spectral_centroid) / 1000
            distance += abs(features.crest_factor - profile.crest_factor) / 10
            distance += abs(features.harmonic_ratio - profile.harmonic_ratio)
            
            if distance < best_score:
                best_score = distance
                best_genre = genre
        
        # Convert distance to confidence (0-1)
        confidence = max(0, 1 - best_score / 3)
        
        return best_genre, confidence

class SpectralAnalyzer:
    """Advanced spectral analysis for tonal balance"""
    
    def __init__(self, sample_rate: int):
        self.fs = sample_rate
        
        # Target curves for different genres
        self.target_curves = {
            AudioGenre.SPEECH: self._create_speech_target(),
            AudioGenre.ROCK: self._create_rock_target(),
            AudioGenre.JAZZ: self._create_jazz_target(),
            AudioGenre.ELECTRONIC: self._create_electronic_target(),
            AudioGenre.CLASSICAL: self._create_classical_target(),
        }
    
    def _create_speech_target(self) -> np.ndarray:
        """Speech clarity target curve"""
        # Boost presence (2-5kHz), roll off extremes
        freqs = np.logspace(1, 4.3, 100)  # 10Hz to 20kHz
        curve = np.ones(100)
        
        # Roll off below 80Hz
        curve[freqs < 80] *= 0.3
        
        # Boost presence
        presence_mask = (freqs > 2000) & (freqs < 5000)
        curve[presence_mask] *= 1.3
        
        # Roll off above 10kHz
        curve[freqs > 10000] *= 0.7
        
        return curve
    
    def _create_rock_target(self) -> np.ndarray:
        """Rock/pop target curve - slight smile"""
        freqs = np.logspace(1, 4.3, 100)
        curve = np.ones(100)
        
        # Boost lows (80-250Hz)
        curve[(freqs > 80) & (freqs < 250)] *= 1.2
        
        # Slight mid scoop (500-2kHz)
        curve[(freqs > 500) & (freqs < 2000)] *= 0.9
        
        # Boost highs (5-10kHz)
        curve[(freqs > 5000) & (freqs < 10000)] *= 1.15
        
        return curve
    
    def _create_jazz_target(self) -> np.ndarray:
        """Jazz target - warm and smooth"""
        freqs = np.logspace(1, 4.3, 100)
        curve = np.ones(100)
        
        # Warm low-mids
        curve[(freqs > 200) & (freqs < 500)] *= 1.1
        
        # Smooth highs
        curve[freqs > 8000] *= 0.85
        
        return curve
    
    def _create_electronic_target(self) -> np.ndarray:
        """Electronic target - extended range"""
        freqs = np.logspace(1, 4.3, 100)
        curve = np.ones(100)
        
        # Extended sub-bass
        curve[freqs < 60] *= 1.3
        
        # Crisp highs
        curve[freqs > 8000] *= 1.2
        
        return curve
    
    def _create_classical_target(self) -> np.ndarray:
        """Classical target - natural balance"""
        # Mostly flat with gentle extremes rolloff
        freqs = np.logspace(1, 4.3, 100)
        curve = np.ones(100)
        
        curve[freqs < 40] *= 0.7
        curve[freqs > 15000] *= 0.8
        
        return curve
    
    def analyze_tonal_balance(self, spectrum: np.ndarray, genre: AudioGenre) -> Dict[str, float]:
        """Analyze tonal balance against target"""
        
        target = self.target_curves.get(genre, np.ones(100))
        
        # Resample spectrum to match target resolution
        resampled = np.interp(
            np.linspace(0, len(spectrum)-1, 100),
            np.arange(len(spectrum)),
            spectrum
        )
        
        # Compare to target
        deviation = resampled / (target * np.mean(resampled) + 1e-12)
        
        return {
            "sub_bass": np.mean(deviation[:10]),     # <60Hz
            "bass": np.mean(deviation[10:20]),        # 60-250Hz
            "low_mid": np.mean(deviation[20:40]),     # 250-1kHz
            "mid": np.mean(deviation[40:60]),         # 1-4kHz
            "high_mid": np.mean(deviation[60:80]),    # 4-10kHz
            "high": np.mean(deviation[80:])           # >10kHz
        }

class AutonomousMixingEngine:
    """Main AI mixing engine that makes all decisions"""
    
    def __init__(self, sample_rate: int = 44100):
        self.fs = sample_rate
        self.analyzer = AIAudioAnalyzer(sample_rate)
        self.dsp_chain = StudioDSPChain(sample_rate)
        
        # Mixing profiles for different content
        self.profiles = self._create_mixing_profiles()
        
        # State tracking
        self.current_genre = AudioGenre.SPEECH
        self.genre_confidence = 0.0
        self.adaptation_rate = 0.1  # How quickly to adapt to changes
        
        # History for smoothing decisions
        self.feature_history = deque(maxlen=50)
        self.genre_history = deque(maxlen=10)
        
    def _create_mixing_profiles(self) -> Dict[AudioGenre, MixProfile]:
        """Create optimized mixing profiles for each genre"""
        
        profiles = {}
        
        # Speech/Podcast profile
        profiles[AudioGenre.SPEECH] = MixProfile(
            name="Speech Clarity",
            genre=AudioGenre.SPEECH,
            target_lufs=-16.0,  # Podcast standard
            target_dynamics=8.0,
            tonal_balance={"low": 0.7, "mid": 1.2, "high": 1.0},
            gate_params=GateParams(
                threshold_db=-45, ratio=4, attack_ms=0.5,
                release_ms=100, range_db=-40
            ),
            comp_params=CompressorParams(
                threshold_db=-20, ratio=3, attack_ms=2,
                release_ms=50, knee_db=2, makeup_gain_db=3
            ),
            eq_bands=[
                EQBand(80, -3, 0.7, "highpass"),     # Remove rumble
                EQBand(200, -2, 0.7, "bell"),        # Reduce muddiness
                EQBand(3000, 2, 0.7, "bell"),        # Presence boost
                EQBand(8000, 1, 0.7, "highshelf"),   # Air
            ],
            limiter_params=LimiterParams(ceiling_db=-1.0, release_ms=50)
        )
        
        # Music - Rock/Pop profile
        profiles[AudioGenre.ROCK] = MixProfile(
            name="Rock Power",
            genre=AudioGenre.ROCK,
            target_lufs=-14.0,  # Streaming standard
            target_dynamics=6.0,
            tonal_balance={"low": 1.2, "mid": 0.9, "high": 1.1},
            gate_params=GateParams(
                threshold_db=-50, ratio=10, attack_ms=0.1,
                release_ms=200, range_db=-60
            ),
            comp_params=CompressorParams(
                threshold_db=-18, ratio=4, attack_ms=5,
                release_ms=100, knee_db=3, makeup_gain_db=4,
                sidechain_hpf_hz=100  # Prevent bass pumping
            ),
            eq_bands=[
                EQBand(100, 2, 0.7, "bell"),         # Punch
                EQBand(500, -1, 0.5, "bell"),        # Clean mids
                EQBand(3000, 1, 0.7, "bell"),        # Presence
                EQBand(10000, 2, 0.7, "highshelf"),  # Sparkle
            ],
            limiter_params=LimiterParams(ceiling_db=-0.3, release_ms=100)
        )
        
        # Jazz profile
        profiles[AudioGenre.JAZZ] = MixProfile(
            name="Jazz Warmth",
            genre=AudioGenre.JAZZ,
            target_lufs=-18.0,
            target_dynamics=12.0,
            tonal_balance={"low": 1.0, "mid": 1.1, "high": 0.9},
            gate_params=GateParams(
                threshold_db=-55, ratio=2, attack_ms=1,
                release_ms=300, range_db=-40
            ),
            comp_params=CompressorParams(
                threshold_db=-22, ratio=2.5, attack_ms=10,
                release_ms=150, knee_db=4, makeup_gain_db=2
            ),
            eq_bands=[
                EQBand(250, 1, 0.5, "bell"),         # Warmth
                EQBand(1000, 0.5, 0.7, "bell"),      # Body
                EQBand(8000, -1, 0.7, "highshelf"),  # Smooth highs
            ],
            limiter_params=LimiterParams(ceiling_db=-2.0, release_ms=150)
        )
        
        # Electronic profile
        profiles[AudioGenre.ELECTRONIC] = MixProfile(
            name="Electronic Energy",
            genre=AudioGenre.ELECTRONIC,
            target_lufs=-11.0,  # Club standard
            target_dynamics=4.0,
            tonal_balance={"low": 1.4, "mid": 0.8, "high": 1.2},
            gate_params=GateParams(
                threshold_db=-60, ratio=20, attack_ms=0.01,
                release_ms=50, range_db=-80
            ),
            comp_params=CompressorParams(
                threshold_db=-15, ratio=6, attack_ms=1,
                release_ms=30, knee_db=1, makeup_gain_db=5,
                sidechain_hpf_hz=30  # Keep sub-bass
            ),
            eq_bands=[
                EQBand(50, 3, 0.5, "bell"),          # Sub-bass
                EQBand(150, 1, 0.7, "bell"),         # Bass punch
                EQBand(800, -2, 0.5, "bell"),        # Clear mids
                EQBand(12000, 3, 0.7, "highshelf"),  # Crisp highs
            ],
            limiter_params=LimiterParams(ceiling_db=-0.1, release_ms=20)
        )
        
        # Classical profile
        profiles[AudioGenre.CLASSICAL] = MixProfile(
            name="Classical Natural",
            genre=AudioGenre.CLASSICAL,
            target_lufs=-23.0,  # EBU R128 standard
            target_dynamics=20.0,
            tonal_balance={"low": 1.0, "mid": 1.0, "high": 1.0},
            gate_params=GateParams(
                threshold_db=-60, ratio=2, attack_ms=5,
                release_ms=500, range_db=-30
            ),
            comp_params=CompressorParams(
                threshold_db=-30, ratio=1.5, attack_ms=20,
                release_ms=200, knee_db=6, makeup_gain_db=0
            ),
            eq_bands=[
                # Minimal EQ - preserve natural balance
                EQBand(30, -2, 0.5, "highpass"),     # Remove sub-sonics
            ],
            limiter_params=LimiterParams(ceiling_db=-3.0, release_ms=200)
        )
        
        return profiles
    
    def process(self, audio: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Main processing function - analyzes and applies intelligent mixing
        Returns processed audio and decision metrics
        """
        
        # 1. Analyze audio
        features = self.analyzer.analyze(audio)
        self.feature_history.append(features)
        
        # 2. Detect genre/content type
        genre, confidence = self.analyzer.genre_detector.detect(features)
        self.genre_history.append(genre)
        
        # 3. Smooth genre detection to avoid jumpy changes
        if confidence > 0.7 and genre != self.current_genre:
            # High confidence in new genre
            self.current_genre = genre
            self.genre_confidence = confidence
        elif len(self.genre_history) >= 5:
            # Check if consistent detection
            recent_genres = list(self.genre_history)[-5:]
            most_common = max(set(recent_genres), key=recent_genres.count)
            if recent_genres.count(most_common) >= 4:
                self.current_genre = most_common
                self.genre_confidence = 0.8
        
        # 4. Get mixing profile
        profile = self.profiles.get(self.current_genre, self.profiles[AudioGenre.SPEECH])
        
        # 5. Adaptive parameter adjustment based on current audio
        adapted_profile = self._adapt_profile(profile, features)
        
        # 6. Process through DSP chain
        processed, metrics = self.dsp_chain.process(
            audio,
            adapted_profile.gate_params,
            adapted_profile.comp_params,
            adapted_profile.eq_bands,
            adapted_profile.limiter_params
        )
        
        # 7. Add decision info to metrics
        metrics.update({
            "detected_genre": self.current_genre.value,
            "genre_confidence": self.genre_confidence,
            "profile_name": profile.name,
            "target_lufs": profile.target_lufs,
            "features": {
                "spectral_centroid": features.spectral_centroid,
                "crest_factor": features.crest_factor,
                "harmonic_ratio": features.harmonic_ratio,
            }
        })
        
        return processed, metrics
    
    def _adapt_profile(self, profile: MixProfile, features: AudioFeatures) -> MixProfile:
        """Dynamically adapt profile based on current audio characteristics"""
        
        adapted = MixProfile(
            name=profile.name,
            genre=profile.genre,
            target_lufs=profile.target_lufs,
            target_dynamics=profile.target_dynamics,
            tonal_balance=profile.tonal_balance.copy(),
            gate_params=GateParams(**vars(profile.gate_params)),
            comp_params=CompressorParams(**vars(profile.comp_params)),
            eq_bands=profile.eq_bands.copy(),
            limiter_params=LimiterParams(**vars(profile.limiter_params))
        )
        
        # Adapt gate threshold based on noise floor
        if features.rms_db < -50:
            adapted.gate_params.threshold_db = features.rms_db + 10
        
        # Adapt compressor based on dynamics
        if features.crest_factor > 15:
            # More dynamic content - gentler compression
            adapted.comp_params.ratio = max(2.0, adapted.comp_params.ratio - 1)
            adapted.comp_params.threshold_db += 3
        elif features.crest_factor < 8:
            # Already compressed - be gentle
            adapted.comp_params.ratio = min(3.0, adapted.comp_params.ratio)
        
        # Adapt EQ based on spectral balance
        if features.spectral_centroid < 500:
            # Dull - add brightness
            adapted.eq_bands.append(EQBand(8000, 2, 0.7, "highshelf"))
        elif features.spectral_centroid > 4000:
            # Harsh - smooth highs
            adapted.eq_bands.append(EQBand(6000, -2, 0.7, "bell"))
        
        return adapted

# Export main class
__all__ = ['AutonomousMixingEngine', 'AudioGenre', 'MixProfile']