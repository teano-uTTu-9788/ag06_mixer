"""
Production Generative Mix AI System
Following Meta/OpenAI patterns for reliable AI generation

Key patterns implemented:
- Meta's LLaMA approach: Template-based generation with constraints
- OpenAI's safety patterns: Input validation and output filtering
- Google's Vertex AI patterns: Structured generation pipelines
- Microsoft's responsible AI: Confidence scoring and fallback mechanisms
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from abc import ABC, abstractmethod
import time
import hashlib

# Setup structured logging (Google Cloud pattern)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MixStyle(Enum):
    """Mix styles following industry standards"""
    MODERN_POP = "modern_pop"
    VINTAGE_ROCK = "vintage_rock"
    EDM = "edm"
    JAZZ = "jazz"
    CLASSICAL = "classical"
    HIP_HOP = "hip_hop"
    COUNTRY = "country"
    FOLK = "folk"
    METAL = "metal"
    REGGAE = "reggae"
    FUNK = "funk"
    BLUES = "blues"

class InstrumentType(Enum):
    """Instrument classification following audio standards"""
    VOCALS = "vocals"
    GUITAR = "guitar"
    BASS = "bass"
    DRUMS = "drums"
    KEYBOARD = "keyboard"
    SAXOPHONE = "saxophone"
    TRUMPET = "trumpet"
    VIOLIN = "violin"
    UNKNOWN = "unknown"

@dataclass
class MixTemplate:
    """Mix template following Meta's structured generation pattern"""
    style: MixStyle
    instruments: Dict[str, Dict[str, float]]
    effects: Dict[str, Dict[str, Any]]
    automation: List[Dict[str, Any]]
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GenerationRequest:
    """Generation request following OpenAI API patterns"""
    style: MixStyle
    channels: int
    constraints: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    safety_level: str = "strict"  # strict, moderate, permissive

@dataclass
class GenerationResult:
    """Generation result with confidence scoring"""
    template: MixTemplate
    alternatives: List[MixTemplate]
    confidence: float
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class IAudioAnalyzer(ABC):
    """Abstract audio analyzer interface"""
    @abstractmethod
    def analyze_content(self, audio_data: np.ndarray) -> Dict[str, Any]:
        pass

class ITemplateGenerator(ABC):
    """Abstract template generator interface"""
    @abstractmethod
    def generate_template(self, request: GenerationRequest) -> MixTemplate:
        pass

class ISafetyValidator(ABC):
    """Abstract safety validator interface"""
    @abstractmethod
    def validate_template(self, template: MixTemplate) -> Tuple[bool, str]:
        pass

class ProductionAudioAnalyzer(IAudioAnalyzer):
    """Production audio analyzer following Google's ML patterns"""
    
    def __init__(self):
        logger.info("🔧 Initializing Production Audio Analyzer")
        # Initialize with professional audio analysis parameters
        self.sample_rate = 44100
        self.frame_size = 2048
        self.hop_length = 512
        
    def analyze_content(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Analyze audio content for instrument detection and characteristics
        Following Google's MediaPipe approach for structured analysis
        """
        try:
            # Simulate professional audio analysis
            # In production: Use librosa, aubio, or similar
            
            # Spectral analysis
            rms_energy = np.sqrt(np.mean(audio_data**2))
            peak_amplitude = np.max(np.abs(audio_data))
            dynamic_range = 20 * np.log10(peak_amplitude / (rms_energy + 1e-8))
            
            # Frequency analysis simulation
            # In production: FFT analysis for frequency content
            low_energy = np.mean(np.abs(audio_data[:len(audio_data)//4]))
            mid_energy = np.mean(np.abs(audio_data[len(audio_data)//4:3*len(audio_data)//4]))
            high_energy = np.mean(np.abs(audio_data[3*len(audio_data)//4:]))
            
            # Instrument detection (simplified heuristics)
            instruments = self._detect_instruments(low_energy, mid_energy, high_energy)
            
            # Tempo detection (simplified)
            estimated_bpm = self._estimate_tempo(audio_data)
            
            analysis = {
                "rms_energy": float(rms_energy),
                "peak_amplitude": float(peak_amplitude),
                "dynamic_range": float(dynamic_range),
                "frequency_content": {
                    "low": float(low_energy),
                    "mid": float(mid_energy),
                    "high": float(high_energy)
                },
                "instruments_detected": instruments,
                "estimated_bpm": estimated_bpm,
                "analysis_confidence": 0.75 + np.random.random() * 0.2  # Simulated confidence
            }
            
            logger.info(f"📊 Audio analysis complete: {len(instruments)} instruments detected")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Audio analysis failed: {e}")
            return self._get_fallback_analysis()
    
    def _detect_instruments(self, low: float, mid: float, high: float) -> List[str]:
        """Instrument detection using frequency content heuristics"""
        instruments = []
        
        # Heuristic-based detection (production would use ML models)
        if low > 0.3:
            instruments.append("bass")
        if low > 0.4 and mid > 0.3:
            instruments.append("drums")
        if mid > 0.4:
            instruments.append("guitar")
        if high > 0.3:
            instruments.append("vocals")
        if mid > 0.2 and high > 0.2:
            instruments.append("keyboard")
            
        return instruments if instruments else ["unknown"]
    
    def _estimate_tempo(self, audio_data: np.ndarray) -> float:
        """Simplified tempo estimation"""
        # In production: Use librosa.beat.tempo or similar
        # Simulate BPM between 60-200
        return 80 + (hash(str(audio_data[:100].sum())) % 120)
    
    def _get_fallback_analysis(self) -> Dict[str, Any]:
        """Fallback analysis for error cases"""
        return {
            "rms_energy": 0.3,
            "peak_amplitude": 0.8,
            "dynamic_range": 12.0,
            "frequency_content": {"low": 0.3, "mid": 0.4, "high": 0.3},
            "instruments_detected": ["vocals", "guitar", "drums"],
            "estimated_bpm": 120.0,
            "analysis_confidence": 0.5
        }

class ProductionTemplateGenerator(ITemplateGenerator):
    """Production template generator following Meta's structured approach"""
    
    def __init__(self):
        logger.info("🎯 Initializing Production Template Generator")
        self.templates = self._load_professional_templates()
        
    def _load_professional_templates(self) -> Dict[MixStyle, Dict]:
        """Load professional mixing templates following industry standards"""
        return {
            MixStyle.MODERN_POP: {
                "eq_curve": {
                    "vocals": {"low_cut": 80, "presence": 3000, "air": 12000},
                    "guitar": {"low_cut": 100, "body": 800, "bite": 2500},
                    "bass": {"sub": 60, "punch": 100, "click": 2500},
                    "drums": {"kick": 60, "snare": 200, "cymbals": 10000}
                },
                "compression": {
                    "vocals": {"ratio": 4, "attack": 3, "release": 100},
                    "guitar": {"ratio": 3, "attack": 10, "release": 50},
                    "bass": {"ratio": 4, "attack": 1, "release": 100},
                    "drums": {"ratio": 6, "attack": 1, "release": 50}
                },
                "effects": {
                    "reverb": {"type": "hall", "size": 0.6, "decay": 1.8},
                    "delay": {"time": 0.125, "feedback": 0.25, "high_cut": 5000}
                },
                "mix_balance": {
                    "vocals": 0.9, "guitar": 0.7, "bass": 0.8, "drums": 0.85
                }
            },
            
            MixStyle.VINTAGE_ROCK: {
                "eq_curve": {
                    "vocals": {"low_cut": 100, "warmth": 500, "presence": 2500},
                    "guitar": {"low_cut": 80, "warmth": 400, "grind": 1800},
                    "bass": {"punch": 80, "midrange": 400},
                    "drums": {"kick": 50, "snare": 150, "overhead": 8000}
                },
                "compression": {
                    "vocals": {"ratio": 3, "attack": 5, "release": 150},
                    "guitar": {"ratio": 2.5, "attack": 15, "release": 80},
                    "bass": {"ratio": 3, "attack": 2, "release": 120},
                    "drums": {"ratio": 4, "attack": 2, "release": 60}
                },
                "effects": {
                    "reverb": {"type": "plate", "size": 0.5, "decay": 2.2},
                    "tape_delay": {"time": 0.25, "feedback": 0.3, "wow_flutter": 0.1}
                },
                "mix_balance": {
                    "vocals": 0.85, "guitar": 0.9, "bass": 0.75, "drums": 0.8
                }
            },
            
            MixStyle.EDM: {
                "eq_curve": {
                    "vocals": {"low_cut": 120, "body": 800, "presence": 4000, "air": 15000},
                    "synth": {"sub": 40, "body": 200, "lead": 2000, "sparkle": 12000},
                    "bass": {"sub": 50, "fundamental": 80, "harmonics": 200},
                    "drums": {"kick": 50, "snare": 180, "hi_hats": 8000}
                },
                "compression": {
                    "vocals": {"ratio": 6, "attack": 1, "release": 50},
                    "synth": {"ratio": 4, "attack": 1, "release": 30},
                    "bass": {"ratio": 8, "attack": 0.5, "release": 50},
                    "drums": {"ratio": 10, "attack": 0.1, "release": 20}
                },
                "effects": {
                    "reverb": {"type": "digital", "size": 0.8, "decay": 3.0},
                    "delay": {"time": 0.0625, "feedback": 0.4, "filter": "high_pass"},
                    "sidechain": {"trigger": "kick", "ratio": 8, "release": 100}
                },
                "mix_balance": {
                    "vocals": 0.8, "synth": 0.95, "bass": 0.9, "drums": 1.0
                }
            }
        }
    
    def generate_template(self, request: GenerationRequest) -> MixTemplate:
        """Generate mix template following Meta's structured generation approach"""
        try:
            start_time = time.time()
            
            # Get base template for style
            base_template = self.templates.get(request.style, self.templates[MixStyle.MODERN_POP])
            
            # Apply constraints and preferences
            customized_template = self._apply_customizations(base_template, request)
            
            # Create instruments configuration
            instruments = self._generate_instruments_config(customized_template, request.channels)
            
            # Generate effects chain
            effects = self._generate_effects_chain(customized_template)
            
            # Generate automation
            automation = self._generate_automation(customized_template)
            
            # Calculate confidence based on template match and constraints
            confidence = self._calculate_confidence(request, customized_template)
            
            processing_time = time.time() - start_time
            
            template = MixTemplate(
                style=request.style,
                instruments=instruments,
                effects=effects,
                automation=automation,
                confidence=confidence,
                metadata={
                    "generation_time": processing_time,
                    "template_version": "1.0",
                    "safety_level": request.safety_level
                }
            )
            
            logger.info(f"✅ Template generated: {request.style.value} (confidence: {confidence:.2f})")
            return template
            
        except Exception as e:
            logger.error(f"❌ Template generation failed: {e}")
            return self._get_fallback_template(request.style)
    
    def _apply_customizations(self, base_template: Dict, request: GenerationRequest) -> Dict:
        """Apply user constraints and preferences"""
        customized = base_template.copy()
        
        # Apply constraints (safety-first approach)
        for constraint_key, constraint_value in request.constraints.items():
            if constraint_key in customized:
                # Apply constraint with safety bounds
                customized[constraint_key] = self._apply_safe_constraint(
                    customized[constraint_key], constraint_value
                )
        
        # Apply preferences (if they don't violate constraints)
        for pref_key, pref_value in request.preferences.items():
            if pref_key in customized and self._is_safe_preference(pref_value):
                customized[pref_key] = self._blend_preference(
                    customized[pref_key], pref_value
                )
        
        return customized
    
    def _generate_instruments_config(self, template: Dict, channels: int) -> Dict[str, Dict[str, float]]:
        """Generate instruments configuration for available channels"""
        instruments = {}
        eq_curve = template.get("eq_curve", {})
        compression = template.get("compression", {})
        mix_balance = template.get("mix_balance", {})
        
        # Map channels to instruments (simplified mapping)
        instrument_mapping = ["vocals", "guitar", "bass", "drums", "keyboard", "synth", "percussion", "auxiliary"]
        
        for i in range(min(channels, len(instrument_mapping))):
            instrument = instrument_mapping[i]
            
            instruments[f"channel_{i+1}"] = {
                "instrument_type": instrument,
                "volume": mix_balance.get(instrument, 0.7),
                "pan": self._calculate_pan_position(i, channels),
                "eq": eq_curve.get(instrument, {}),
                "compression": compression.get(instrument, {}),
                "confidence": 0.8 + np.random.random() * 0.15
            }
        
        return instruments
    
    def _generate_effects_chain(self, template: Dict) -> Dict[str, Dict[str, Any]]:
        """Generate effects chain configuration"""
        effects = {}
        base_effects = template.get("effects", {})
        
        for effect_name, effect_params in base_effects.items():
            effects[effect_name] = {
                "enabled": True,
                "parameters": effect_params.copy(),
                "insert_position": self._get_optimal_insert_position(effect_name),
                "wet_dry_mix": self._calculate_wet_dry_mix(effect_name)
            }
        
        return effects
    
    def _generate_automation(self, template: Dict) -> List[Dict[str, Any]]:
        """Generate automation curves for dynamic mixing"""
        automation = []
        
        # Volume automation for intro/outro
        automation.append({
            "parameter": "master_volume",
            "points": [
                {"time": 0.0, "value": 0.0},      # Fade in
                {"time": 4.0, "value": 1.0},      # Full volume
                {"time": -8.0, "value": 1.0},     # Maintain
                {"time": -0.1, "value": 0.0}      # Fade out
            ],
            "curve": "logarithmic"
        })
        
        # Dynamic EQ automation
        automation.append({
            "parameter": "vocal_eq_presence",
            "points": [
                {"time": 0.0, "value": 0.8},      # Verse
                {"time": 60.0, "value": 1.2},     # Chorus lift
                {"time": 120.0, "value": 0.9}     # Bridge
            ],
            "curve": "smooth"
        })
        
        return automation
    
    def _calculate_confidence(self, request: GenerationRequest, template: Dict) -> float:
        """Calculate generation confidence using multiple factors"""
        base_confidence = 0.7
        
        # Style match confidence
        if request.style in self.templates:
            base_confidence += 0.15
        
        # Constraint satisfaction
        constraint_satisfaction = len(request.constraints) / max(len(request.constraints) + 1, 1)
        base_confidence += constraint_satisfaction * 0.1
        
        # Template completeness
        required_sections = ["eq_curve", "compression", "effects", "mix_balance"]
        completeness = sum(1 for section in required_sections if section in template) / len(required_sections)
        base_confidence += completeness * 0.05
        
        return min(base_confidence, 1.0)
    
    def _calculate_pan_position(self, channel_index: int, total_channels: int) -> float:
        """Calculate optimal pan position for channel"""
        if total_channels <= 1:
            return 0.0  # Center
        
        # Distribute channels across stereo field
        pan_step = 2.0 / (total_channels - 1) if total_channels > 1 else 0
        return -1.0 + (channel_index * pan_step)
    
    def _apply_safe_constraint(self, original_value: Any, constraint_value: Any) -> Any:
        """Apply constraint with safety bounds"""
        # Safety bounds to prevent extreme values
        if isinstance(constraint_value, (int, float)):
            return max(0.0, min(constraint_value, 2.0))  # Reasonable audio range
        return constraint_value
    
    def _is_safe_preference(self, preference_value: Any) -> bool:
        """Validate preference is safe to apply"""
        if isinstance(preference_value, (int, float)):
            return 0.0 <= preference_value <= 2.0
        return True
    
    def _blend_preference(self, original_value: Any, preference_value: Any) -> Any:
        """Blend preference with original value"""
        if isinstance(original_value, (int, float)) and isinstance(preference_value, (int, float)):
            # Weighted blend favoring original template
            return original_value * 0.7 + preference_value * 0.3
        return preference_value
    
    def _get_optimal_insert_position(self, effect_name: str) -> str:
        """Get optimal insert position for effect"""
        insert_positions = {
            "eq": "pre_fader",
            "compression": "pre_fader", 
            "reverb": "post_fader",
            "delay": "post_fader",
            "sidechain": "pre_fader"
        }
        return insert_positions.get(effect_name.lower(), "post_fader")
    
    def _calculate_wet_dry_mix(self, effect_name: str) -> float:
        """Calculate optimal wet/dry mix for effect"""
        wet_dry_defaults = {
            "reverb": 0.25,
            "delay": 0.15,
            "chorus": 0.3,
            "compression": 1.0,  # Fully wet
            "eq": 1.0           # Fully wet
        }
        return wet_dry_defaults.get(effect_name.lower(), 0.5)
    
    def _get_fallback_template(self, style: MixStyle) -> MixTemplate:
        """Fallback template for error cases"""
        return MixTemplate(
            style=style,
            instruments={
                "channel_1": {
                    "instrument_type": "vocals",
                    "volume": 0.8,
                    "pan": 0.0,
                    "eq": {"low_cut": 100, "presence": 3000},
                    "compression": {"ratio": 3, "attack": 5, "release": 100}
                }
            },
            effects={
                "reverb": {
                    "enabled": True,
                    "parameters": {"size": 0.5, "decay": 1.5},
                    "wet_dry_mix": 0.2
                }
            },
            automation=[],
            confidence=0.5
        )

class ProductionSafetyValidator(ISafetyValidator):
    """Production safety validator following Microsoft's responsible AI patterns"""
    
    def __init__(self):
        logger.info("🛡️ Initializing Production Safety Validator")
        self.safety_rules = self._load_safety_rules()
    
    def _load_safety_rules(self) -> Dict[str, Dict]:
        """Load audio safety validation rules"""
        return {
            "volume_limits": {
                "max_volume": 1.0,
                "min_volume": 0.0,
                "recommended_max": 0.9
            },
            "frequency_limits": {
                "min_frequency": 20,
                "max_frequency": 20000,
                "safe_boost_limit": 12  # dB
            },
            "compression_limits": {
                "max_ratio": 10,
                "min_attack": 0.1,
                "max_release": 5000
            },
            "effects_limits": {
                "max_reverb_decay": 10.0,
                "max_delay_time": 2.0,
                "max_feedback": 0.95
            }
        }
    
    def validate_template(self, template: MixTemplate) -> Tuple[bool, str]:
        """Validate template against safety rules"""
        try:
            # Validate instruments
            for channel, instrument_config in template.instruments.items():
                is_valid, error_msg = self._validate_instrument_config(instrument_config)
                if not is_valid:
                    return False, f"Instrument validation failed for {channel}: {error_msg}"
            
            # Validate effects
            for effect_name, effect_config in template.effects.items():
                is_valid, error_msg = self._validate_effect_config(effect_name, effect_config)
                if not is_valid:
                    return False, f"Effect validation failed for {effect_name}: {error_msg}"
            
            # Validate automation
            for automation_item in template.automation:
                is_valid, error_msg = self._validate_automation_config(automation_item)
                if not is_valid:
                    return False, f"Automation validation failed: {error_msg}"
            
            logger.info(f"✅ Template validation passed for {template.style.value}")
            return True, "Template passes all safety validations"
            
        except Exception as e:
            logger.error(f"❌ Template validation error: {e}")
            return False, f"Validation error: {str(e)}"
    
    def _validate_instrument_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate individual instrument configuration"""
        # Volume validation
        volume = config.get("volume", 0.7)
        if not (0.0 <= volume <= self.safety_rules["volume_limits"]["max_volume"]):
            return False, f"Volume {volume} exceeds safe limits"
        
        # Pan validation
        pan = config.get("pan", 0.0)
        if not (-1.0 <= pan <= 1.0):
            return False, f"Pan position {pan} outside valid range"
        
        # EQ validation
        eq_config = config.get("eq", {})
        for freq, gain in eq_config.items():
            if isinstance(gain, (int, float)) and abs(gain) > self.safety_rules["frequency_limits"]["safe_boost_limit"]:
                return False, f"EQ boost/cut {gain}dB exceeds safe limit"
        
        return True, "Instrument configuration valid"
    
    def _validate_effect_config(self, effect_name: str, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate individual effect configuration"""
        parameters = config.get("parameters", {})
        
        if "reverb" in effect_name.lower():
            decay = parameters.get("decay", 0)
            if decay > self.safety_rules["effects_limits"]["max_reverb_decay"]:
                return False, f"Reverb decay {decay} exceeds safe limit"
        
        if "delay" in effect_name.lower():
            delay_time = parameters.get("time", 0)
            feedback = parameters.get("feedback", 0)
            
            if delay_time > self.safety_rules["effects_limits"]["max_delay_time"]:
                return False, f"Delay time {delay_time} exceeds safe limit"
            
            if feedback > self.safety_rules["effects_limits"]["max_feedback"]:
                return False, f"Delay feedback {feedback} could cause instability"
        
        return True, "Effect configuration valid"
    
    def _validate_automation_config(self, automation: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate automation configuration"""
        points = automation.get("points", [])
        
        for point in points:
            value = point.get("value", 0)
            if isinstance(value, (int, float)):
                if not (0.0 <= value <= 2.0):  # Reasonable automation range
                    return False, f"Automation value {value} outside safe range"
        
        return True, "Automation configuration valid"

class ProductionGenerativeMixAI:
    """
    Production Generative Mix AI System
    Following Meta/OpenAI patterns for production AI systems
    """
    
    def __init__(self):
        logger.info("🚀 Initializing Production Generative Mix AI")
        
        # Initialize components using dependency injection
        self.audio_analyzer = ProductionAudioAnalyzer()
        self.template_generator = ProductionTemplateGenerator()
        self.safety_validator = ProductionSafetyValidator()
        
        # Performance metrics
        self.generation_stats = {
            "total_generations": 0,
            "successful_generations": 0,
            "average_processing_time": 0.0,
            "safety_violations": 0
        }
        
        logger.info("✅ Production Generative Mix AI initialized successfully")
    
    def generate_mix_suggestions(self, 
                                audio_data: Optional[np.ndarray] = None,
                                style: Optional[MixStyle] = None,
                                channels: int = 8,
                                constraints: Optional[Dict] = None,
                                preferences: Optional[Dict] = None) -> GenerationResult:
        """
        Generate mix suggestions following OpenAI API patterns
        
        Args:
            audio_data: Optional audio data for analysis-based generation
            style: Desired mix style (auto-detected if None)
            channels: Number of available channels
            constraints: Hard constraints that must be satisfied
            preferences: Soft preferences that should be considered
        
        Returns:
            GenerationResult with primary template and alternatives
        """
        try:
            start_time = time.time()
            self.generation_stats["total_generations"] += 1
            
            # Step 1: Analyze audio content (if provided)
            analysis = None
            if audio_data is not None:
                analysis = self.audio_analyzer.analyze_content(audio_data)
                logger.info(f"📊 Audio analysis: {analysis.get('instruments_detected', [])} detected")
            
            # Step 2: Determine style (auto-detect or use provided)
            if style is None:
                style = self._auto_detect_style(analysis) if analysis else MixStyle.MODERN_POP
            
            # Step 3: Create generation request
            request = GenerationRequest(
                style=style,
                channels=channels,
                constraints=constraints or {},
                preferences=preferences or {}
            )
            
            # Step 4: Generate primary template
            primary_template = self.template_generator.generate_template(request)
            
            # Step 5: Validate safety
            is_safe, safety_message = self.safety_validator.validate_template(primary_template)
            if not is_safe:
                logger.warning(f"⚠️ Safety validation failed: {safety_message}")
                self.generation_stats["safety_violations"] += 1
                primary_template = self._apply_safety_corrections(primary_template)
            
            # Step 6: Generate alternatives
            alternatives = self._generate_alternatives(request, primary_template)
            
            # Step 7: Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(primary_template, alternatives, analysis)
            
            processing_time = time.time() - start_time
            self.generation_stats["successful_generations"] += 1
            self.generation_stats["average_processing_time"] = (
                (self.generation_stats["average_processing_time"] * (self.generation_stats["successful_generations"] - 1) +
                 processing_time) / self.generation_stats["successful_generations"]
            )
            
            result = GenerationResult(
                template=primary_template,
                alternatives=alternatives,
                confidence=overall_confidence,
                processing_time=processing_time,
                metadata={
                    "style_detected": style.value,
                    "analysis_used": analysis is not None,
                    "safety_validated": is_safe,
                    "generation_id": self._generate_id(),
                    "alternatives_count": len(alternatives)
                }
            )
            
            logger.info(f"✅ Mix generation complete: {style.value} (confidence: {overall_confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Mix generation failed: {e}")
            return self._get_fallback_result(style or MixStyle.MODERN_POP, channels)
    
    def _auto_detect_style(self, analysis: Dict[str, Any]) -> MixStyle:
        """Auto-detect mix style based on audio analysis"""
        instruments = analysis.get("instruments_detected", [])
        bpm = analysis.get("estimated_bpm", 120)
        frequency_content = analysis.get("frequency_content", {})
        
        # Simple heuristic-based style detection
        if "synth" in instruments and bpm > 120:
            return MixStyle.EDM
        elif "guitar" in instruments and bpm < 100:
            return MixStyle.VINTAGE_ROCK
        elif "bass" in instruments and frequency_content.get("low", 0) > 0.4:
            if bpm > 140:
                return MixStyle.HIP_HOP
            else:
                return MixStyle.JAZZ
        elif "vocals" in instruments:
            return MixStyle.MODERN_POP
        else:
            return MixStyle.MODERN_POP  # Default fallback
    
    def _generate_alternatives(self, request: GenerationRequest, primary: MixTemplate) -> List[MixTemplate]:
        """Generate alternative mix suggestions"""
        alternatives = []
        
        # Generate variations with different styles
        alternative_styles = [
            MixStyle.MODERN_POP,
            MixStyle.VINTAGE_ROCK,
            MixStyle.EDM
        ]
        
        for alt_style in alternative_styles:
            if alt_style != primary.style and len(alternatives) < 2:  # Limit to 2 alternatives
                alt_request = GenerationRequest(
                    style=alt_style,
                    channels=request.channels,
                    constraints=request.constraints,
                    preferences=request.preferences
                )
                
                alt_template = self.template_generator.generate_template(alt_request)
                
                # Validate alternative
                is_safe, _ = self.safety_validator.validate_template(alt_template)
                if is_safe:
                    alternatives.append(alt_template)
        
        return alternatives
    
    def _calculate_overall_confidence(self, primary: MixTemplate, alternatives: List[MixTemplate], analysis: Optional[Dict]) -> float:
        """Calculate overall generation confidence"""
        confidence_factors = []
        
        # Primary template confidence
        confidence_factors.append(primary.confidence)
        
        # Alternative quality (having good alternatives increases confidence)
        if alternatives:
            avg_alt_confidence = sum(alt.confidence for alt in alternatives) / len(alternatives)
            confidence_factors.append(avg_alt_confidence * 0.5)  # Weight alternatives lower
        
        # Analysis quality (if audio was analyzed)
        if analysis:
            analysis_confidence = analysis.get("analysis_confidence", 0.5)
            confidence_factors.append(analysis_confidence * 0.3)
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def _apply_safety_corrections(self, template: MixTemplate) -> MixTemplate:
        """Apply safety corrections to template"""
        # Create corrected copy
        corrected = MixTemplate(
            style=template.style,
            instruments=template.instruments.copy(),
            effects=template.effects.copy(),
            automation=template.automation.copy(),
            confidence=template.confidence * 0.8,  # Reduce confidence due to corrections
            metadata=template.metadata.copy()
        )
        
        # Apply volume limits
        for channel, config in corrected.instruments.items():
            if config.get("volume", 0) > 0.9:
                config["volume"] = 0.9
        
        # Apply effect limits
        for effect_name, effect_config in corrected.effects.items():
            params = effect_config.get("parameters", {})
            if "decay" in params and params["decay"] > 5.0:
                params["decay"] = 5.0
            if "feedback" in params and params["feedback"] > 0.7:
                params["feedback"] = 0.7
        
        corrected.metadata["safety_corrected"] = True
        return corrected
    
    def _generate_id(self) -> str:
        """Generate unique generation ID"""
        timestamp = str(time.time())
        return hashlib.md5(timestamp.encode()).hexdigest()[:8]
    
    def _get_fallback_result(self, style: MixStyle, channels: int) -> GenerationResult:
        """Fallback result for error cases"""
        fallback_template = self.template_generator._get_fallback_template(style)
        
        return GenerationResult(
            template=fallback_template,
            alternatives=[],
            confidence=0.3,  # Low confidence for fallback
            processing_time=0.1,
            metadata={
                "is_fallback": True,
                "error_recovery": True
            }
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        success_rate = (
            self.generation_stats["successful_generations"] / 
            max(self.generation_stats["total_generations"], 1)
        )
        
        return {
            "total_generations": self.generation_stats["total_generations"],
            "success_rate": success_rate,
            "average_processing_time": self.generation_stats["average_processing_time"],
            "safety_violations": self.generation_stats["safety_violations"],
            "system_status": "operational" if success_rate > 0.8 else "degraded"
        }

def demo_production_generative_ai():
    """Production demo following Google's demo patterns"""
    print("🎯 Production Generative Mix AI Demo")
    print("=" * 50)
    
    try:
        # Initialize system
        print("1. Initializing Production Generative Mix AI...")
        ai_system = ProductionGenerativeMixAI()
        
        # Generate test audio data
        print("\n2. Generating test audio data...")
        test_audio = np.random.randn(44100 * 2)  # 2 seconds of test audio
        
        # Test 1: Style-based generation
        print("\n3. Testing style-based generation...")
        result1 = ai_system.generate_mix_suggestions(
            style=MixStyle.MODERN_POP,
            channels=8,
            constraints={"max_volume": 0.85},
            preferences={"vocal_prominence": 0.9}
        )
        
        print(f"   ✅ Primary template: {result1.template.style.value}")
        print(f"   ✅ Confidence: {result1.confidence:.2f}")
        print(f"   ✅ Alternatives: {len(result1.alternatives)}")
        print(f"   ✅ Processing time: {result1.processing_time:.3f}s")
        
        # Test 2: Audio-analysis based generation
        print("\n4. Testing audio-analysis based generation...")
        result2 = ai_system.generate_mix_suggestions(
            audio_data=test_audio,
            channels=6
        )
        
        print(f"   ✅ Auto-detected style: {result2.template.style.value}")
        print(f"   ✅ Confidence: {result2.confidence:.2f}")
        print(f"   ✅ Analysis used: {result2.metadata.get('analysis_used', False)}")
        
        # Test 3: Multiple styles
        print("\n5. Testing multiple mix styles...")
        styles_tested = [MixStyle.EDM, MixStyle.VINTAGE_ROCK, MixStyle.JAZZ]
        
        for style in styles_tested:
            result = ai_system.generate_mix_suggestions(style=style, channels=4)
            print(f"   ✅ {style.value}: confidence {result.confidence:.2f}")
        
        # Performance statistics
        print("\n6. Performance Statistics:")
        stats = ai_system.get_performance_stats()
        for metric, value in stats.items():
            print(f"   📊 {metric}: {value}")
        
        print("\n✅ Production Generative Mix AI Demo Complete!")
        print(f"   Generated {stats['total_generations']} templates")
        print(f"   Success rate: {stats['success_rate']:.1%}")
        print(f"   Average processing: {stats['average_processing_time']:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    demo_production_generative_ai()