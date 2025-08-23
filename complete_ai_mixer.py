"""
Complete AI Mixing System with Studio Effects
Professional mixing with DSP chain and effects for broadcast-ready output
"""

import numpy as np
from typing import Dict, Tuple, Any
from dataclasses import dataclass

from ai_mixing_brain import AutonomousMixingEngine
from studio_dsp_chain import StudioDSPChain, GateParams, CompressorParams, EQBand, LimiterParams
from studio_effects import (
    StudioEffectsChain, ReverbParams, DelayParams, 
    ChorusParams, StereoImagerParams, ExciterParams
)

@dataclass
class MixingProfile:
    """Complete mixing profile with DSP and effects"""
    name: str
    gate: GateParams
    compressor: CompressorParams
    eq_bands: list
    limiter: LimiterParams
    reverb: ReverbParams
    delay: DelayParams
    chorus: ChorusParams
    imager: StereoImagerParams
    exciter: ExciterParams

class CompleteMixingSystem:
    """
    Professional AI mixing system with complete signal chain
    DSP -> Effects -> Final limiting for broadcast-ready output
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.fs = sample_rate
        
        # Core processors
        self.ai_engine = AutonomousMixingEngine(sample_rate)
        self.dsp_chain = StudioDSPChain(sample_rate)
        self.effects_chain = StudioEffectsChain(sample_rate)
        
        # Define genre-specific effects profiles
        self.effects_profiles = self._create_effects_profiles()
        
    def _create_effects_profiles(self) -> Dict[str, Dict]:
        """Create genre-specific effects settings"""
        
        return {
            "Speech": {
                "reverb": ReverbParams(
                    room_size=0.2,
                    damping=0.8,
                    wet_level=0.1,
                    dry_level=0.9,
                    pre_delay_ms=5.0
                ),
                "delay": None,  # No delay for speech
                "chorus": None,  # No chorus for speech
                "imager": StereoImagerParams(
                    width=0.5,  # Narrower for speech
                    bass_mono_freq=200.0
                ),
                "exciter": ExciterParams(
                    drive=0.1,
                    mix=0.15,
                    frequency=3000.0  # Presence boost
                )
            },
            "Rock": {
                "reverb": ReverbParams(
                    room_size=0.6,
                    damping=0.5,
                    wet_level=0.25,
                    dry_level=0.75,
                    pre_delay_ms=15.0
                ),
                "delay": DelayParams(
                    delay_time_ms=125.0,
                    feedback=0.3,
                    mix=0.15,
                    tempo_sync=True,
                    bpm=120.0,
                    note_division="1/8"
                ),
                "chorus": ChorusParams(
                    rate_hz=1.5,
                    depth_ms=2.0,
                    mix=0.2,
                    voices=2
                ),
                "imager": StereoImagerParams(
                    width=1.2,
                    bass_mono_freq=100.0,
                    high_freq_width=1.4
                ),
                "exciter": ExciterParams(
                    drive=0.3,
                    mix=0.25,
                    frequency=8000.0
                )
            },
            "Jazz": {
                "reverb": ReverbParams(
                    room_size=0.8,
                    damping=0.4,
                    wet_level=0.35,
                    dry_level=0.65,
                    pre_delay_ms=25.0
                ),
                "delay": DelayParams(
                    delay_time_ms=333.0,
                    feedback=0.25,
                    mix=0.1,
                    tempo_sync=False
                ),
                "chorus": ChorusParams(
                    rate_hz=0.5,
                    depth_ms=1.5,
                    mix=0.15,
                    voices=3
                ),
                "imager": StereoImagerParams(
                    width=1.5,
                    bass_mono_freq=80.0,
                    high_freq_width=1.3
                ),
                "exciter": ExciterParams(
                    drive=0.15,
                    mix=0.1,
                    frequency=10000.0
                )
            },
            "Electronic": {
                "reverb": ReverbParams(
                    room_size=0.9,
                    damping=0.3,
                    wet_level=0.4,
                    dry_level=0.6,
                    pre_delay_ms=30.0
                ),
                "delay": DelayParams(
                    delay_time_ms=375.0,
                    feedback=0.5,
                    mix=0.3,
                    tempo_sync=True,
                    bpm=128.0,
                    note_division="1/4"
                ),
                "chorus": ChorusParams(
                    rate_hz=3.0,
                    depth_ms=4.0,
                    mix=0.3,
                    voices=4
                ),
                "imager": StereoImagerParams(
                    width=1.8,
                    bass_mono_freq=60.0,
                    high_freq_width=2.0
                ),
                "exciter": ExciterParams(
                    drive=0.4,
                    mix=0.3,
                    frequency=12000.0
                )
            },
            "Classical": {
                "reverb": ReverbParams(
                    room_size=0.95,
                    damping=0.2,
                    wet_level=0.45,
                    dry_level=0.55,
                    pre_delay_ms=40.0
                ),
                "delay": None,  # No delay for classical
                "chorus": ChorusParams(
                    rate_hz=0.3,
                    depth_ms=1.0,
                    mix=0.1,
                    voices=2
                ),
                "imager": StereoImagerParams(
                    width=2.0,
                    bass_mono_freq=40.0,
                    high_freq_width=1.5
                ),
                "exciter": ExciterParams(
                    drive=0.05,
                    mix=0.05,
                    frequency=15000.0
                )
            }
        }
    
    def process_complete(self, audio: np.ndarray, 
                        auto_mode: bool = True,
                        target_genre: str = None) -> Dict[str, Any]:
        """
        Complete processing with AI decision making
        Returns stereo output and comprehensive metrics
        """
        
        # Step 1: AI Analysis and genre detection
        if auto_mode:
            # Let AI detect genre and adapt
            ai_mono, ai_results = self.ai_engine.process(audio)
            genre = ai_results.get("genre", "Unknown")
            confidence = ai_results.get("confidence", 0.0)
            dsp_params = ai_results.get("profile", {})
        else:
            # Use specified genre
            genre = target_genre or "Rock"
            confidence = 1.0
            dsp_params = self.ai_engine.profiles.get(genre, {})
            
            # Process DSP manually
            ai_mono, dsp_metrics = self.dsp_chain.process(
                audio,
                dsp_params.get("gate"),
                dsp_params.get("compressor"),
                dsp_params.get("eq_bands"),
                dsp_params.get("limiter")
            )
        
        # Step 2: Apply effects based on genre
        effects_params = self.effects_profiles.get(genre, self.effects_profiles["Rock"])
        
        # Process through effects chain (mono to stereo)
        stereo_left, stereo_right = self.effects_chain.process(
            ai_mono,
            effects_params["reverb"],
            effects_params["delay"],
            effects_params["chorus"],
            effects_params["imager"],
            effects_params["exciter"]
        )
        
        # Step 3: Final limiting on stereo bus
        final_limiter = LimiterParams(
            ceiling_db=-0.1,  # Broadcast standard
            release_ms=50.0,
            lookahead_ms=5.0
        )
        
        # Process each channel through final limiter
        left_limited, left_reduction = self.dsp_chain.limiter.process(stereo_left, final_limiter)
        right_limited, right_reduction = self.dsp_chain.limiter.process(stereo_right, final_limiter)
        
        # Step 4: Calculate comprehensive metrics
        
        # LUFS measurement (simplified)
        left_rms = np.sqrt(np.mean(left_limited**2))
        right_rms = np.sqrt(np.mean(right_limited**2))
        stereo_rms = np.sqrt((left_rms**2 + right_rms**2) / 2)
        lufs = 20 * np.log10(stereo_rms + 1e-12) - 0.691  # K-weighting approximation
        
        # True peak
        true_peak_l = np.max(np.abs(left_limited))
        true_peak_r = np.max(np.abs(right_limited))
        true_peak = max(true_peak_l, true_peak_r)
        true_peak_db = 20 * np.log10(true_peak + 1e-12)
        
        # Stereo correlation
        if len(left_limited) > 0:
            correlation = np.corrcoef(left_limited, right_limited)[0, 1]
        else:
            correlation = 0.0
        
        # Dynamic range
        peak_db = 20 * np.log10(np.max(np.abs(ai_mono)) + 1e-12)
        rms_db = 20 * np.log10(np.sqrt(np.mean(ai_mono**2)) + 1e-12)
        dynamic_range = peak_db - rms_db
        
        # Compile results
        return {
            "audio": {
                "left": left_limited,
                "right": right_limited,
                "mono": ai_mono  # Pre-effects mono
            },
            "genre": genre,
            "confidence": confidence,
            "metrics": {
                "lufs": lufs,
                "true_peak_db": true_peak_db,
                "dynamic_range": dynamic_range,
                "stereo_correlation": correlation,
                "final_limiting_db": max(left_reduction, right_reduction)
            },
            "dsp": {
                "gate_reduction": dsp_metrics.get("gate_reduction_db", 0.0) if auto_mode else 0.0,
                "comp_reduction": dsp_metrics.get("comp_reduction_db", 0.0) if auto_mode else 0.0,
                "limiter_reduction": dsp_metrics.get("limiter_reduction_db", 0.0) if auto_mode else 0.0
            },
            "target_compliance": {
                "streaming": lufs >= -14 and lufs <= -13,  # Spotify/Apple Music
                "broadcast": lufs >= -24 and lufs <= -22,  # EBU R128
                "true_peak_safe": true_peak_db <= -1.0,
                "dynamic_range_good": dynamic_range >= 6 and dynamic_range <= 20
            }
        }
    
    def get_presets(self) -> Dict[str, MixingProfile]:
        """Get all available mixing presets"""
        
        presets = {}
        
        for genre in ["Speech", "Rock", "Jazz", "Electronic", "Classical"]:
            # Get DSP profile
            dsp = self.ai_engine.profiles.get(genre, {})
            effects = self.effects_profiles.get(genre, {})
            
            presets[genre] = MixingProfile(
                name=genre,
                gate=dsp.get("gate", GateParams()),
                compressor=dsp.get("compressor", CompressorParams()),
                eq_bands=dsp.get("eq_bands", []),
                limiter=dsp.get("limiter", LimiterParams()),
                reverb=effects.get("reverb", ReverbParams()),
                delay=effects.get("delay", DelayParams()),
                chorus=effects.get("chorus", ChorusParams()),
                imager=effects.get("imager", StereoImagerParams()),
                exciter=effects.get("exciter", ExciterParams())
            )
        
        return presets
    
    def process_realtime(self, audio_buffer: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simplified real-time processing for low latency
        Returns stereo output with minimal processing delay
        """
        
        # Quick genre detection (use cached if available)
        features = self.ai_engine.analyzer.analyze(audio_buffer)
        genre_scores = self.ai_engine.analyzer.genre_detector._calculate_genre_scores(features)
        genre_idx = np.argmax(list(genre_scores.values()))
        genre = list(genre_scores.keys())[genre_idx]
        
        # Get cached profile
        profile = self.ai_engine.profiles[genre]
        
        # Fast DSP processing (skip lookahead stages for RT)
        processed = audio_buffer.copy()
        
        # Simple gate
        gate_threshold = 10 ** (profile["gate"].threshold_db / 20)
        mask = np.abs(processed) > gate_threshold
        processed *= mask
        
        # Simple compression
        comp_threshold = 10 ** (profile["compressor"].threshold_db / 20)
        ratio = profile["compressor"].ratio
        over = np.abs(processed) - comp_threshold
        over[over < 0] = 0
        reduction = over * (1 - 1/ratio)
        processed *= 10 ** (-reduction / 20)
        
        # Fast stereo (simple width)
        left = processed * 0.7
        right = processed * 0.7
        
        return left, right