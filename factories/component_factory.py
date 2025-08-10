"""
Component Factory for AG06 Mixer
Follows SOLID principles - Dependency Inversion & Factory Pattern
"""
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

# Import interfaces (abstractions we depend on)
from interfaces.audio_engine import IAudioEngine, IAudioEffects, IAudioMetrics, AudioConfig
from interfaces.midi_controller import IMidiController, IMidiDeviceDiscovery, IMidiMapping
from interfaces.preset_manager import IPresetManager, IPresetValidator, IPresetExporter
from interfaces.karaoke_integration import IVocalProcessor, IKaraokeEffects, ILyricsSync, IPerformanceScoring


class IComponentFactory(ABC):
    """Abstract factory interface - Dependency Inversion Principle"""
    
    @abstractmethod
    def create_audio_engine(self, config: Optional[AudioConfig] = None) -> IAudioEngine:
        """Create audio engine instance"""
        pass
    
    @abstractmethod
    def create_midi_controller(self, device_id: Optional[str] = None) -> IMidiController:
        """Create MIDI controller instance"""
        pass
    
    @abstractmethod
    def create_preset_manager(self, storage_path: Optional[str] = None) -> IPresetManager:
        """Create preset manager instance"""
        pass
    
    @abstractmethod
    def create_vocal_processor(self) -> IVocalProcessor:
        """Create vocal processor instance"""
        pass


class AG06ComponentFactory(IComponentFactory):
    """Concrete factory for AG06 components - Factory Pattern"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize factory with optional configuration"""
        self.config = config or {}
        self._initialize_implementations()
    
    def _initialize_implementations(self):
        """Lazy loading of concrete implementations"""
        # Import concrete implementations only when needed
        # This maintains dependency inversion - factory depends on abstractions
        pass
    
    def create_audio_engine(self, config: Optional[AudioConfig] = None) -> IAudioEngine:
        """Create audio engine with dependency injection"""
        from implementations.audio_engine import WebAudioEngine
        
        audio_config = config or AudioConfig()
        
        # Create metrics and effects as dependencies
        metrics = self._create_audio_metrics()
        effects = self._create_audio_effects()
        
        # Inject dependencies into audio engine
        return WebAudioEngine(audio_config, metrics, effects)
    
    def create_midi_controller(self, device_id: Optional[str] = None) -> IMidiController:
        """Create MIDI controller with dependency injection"""
        from implementations.midi_controller import YamahaAG06Controller
        
        # Create discovery and mapping as dependencies
        discovery = self._create_midi_discovery()
        mapping = self._create_midi_mapping()
        
        # Inject dependencies
        return YamahaAG06Controller(device_id, discovery, mapping)
    
    def create_preset_manager(self, storage_path: Optional[str] = None) -> IPresetManager:
        """Create preset manager with dependency injection"""
        from implementations.preset_manager import JsonPresetManager
        
        path = storage_path or self.config.get('preset_path', './presets')
        
        # Create validator and exporter as dependencies
        validator = self._create_preset_validator()
        exporter = self._create_preset_exporter()
        
        return JsonPresetManager(path, validator, exporter)
    
    def create_vocal_processor(self) -> IVocalProcessor:
        """Create vocal processor with dependency injection"""
        from implementations.karaoke_integration import AdvancedVocalProcessor
        
        # Create effects and scoring as dependencies
        effects = self._create_karaoke_effects()
        scoring = self._create_performance_scoring()
        
        return AdvancedVocalProcessor(effects, scoring)
    
    # Private helper methods for creating sub-components
    def _create_audio_metrics(self) -> IAudioMetrics:
        """Create audio metrics implementation"""
        from implementations.audio_engine import RealtimeAudioMetrics
        return RealtimeAudioMetrics()
    
    def _create_audio_effects(self) -> IAudioEffects:
        """Create audio effects implementation"""
        from implementations.audio_engine import ProfessionalAudioEffects
        return ProfessionalAudioEffects()
    
    def _create_midi_discovery(self) -> IMidiDeviceDiscovery:
        """Create MIDI discovery implementation"""
        from implementations.midi_controller import UsbMidiDiscovery
        return UsbMidiDiscovery()
    
    def _create_midi_mapping(self) -> IMidiMapping:
        """Create MIDI mapping implementation"""
        from implementations.midi_controller import FlexibleMidiMapping
        return FlexibleMidiMapping()
    
    def _create_preset_validator(self) -> IPresetValidator:
        """Create preset validator implementation"""
        from implementations.preset_manager import SchemaPresetValidator
        return SchemaPresetValidator()
    
    def _create_preset_exporter(self) -> IPresetExporter:
        """Create preset exporter implementation"""
        from implementations.preset_manager import MultiFormatPresetExporter
        return MultiFormatPresetExporter()
    
    def _create_karaoke_effects(self) -> IKaraokeEffects:
        """Create karaoke effects implementation"""
        from implementations.karaoke_integration import StudioKaraokeEffects
        return StudioKaraokeEffects()
    
    def _create_performance_scoring(self) -> IPerformanceScoring:
        """Create performance scoring implementation"""
        from implementations.karaoke_integration import MLPerformanceScoring
        return MLPerformanceScoring()
    
    def _create_lyrics_sync(self) -> ILyricsSync:
        """Create lyrics sync implementation"""
        from implementations.karaoke_integration import AutoLyricsSync
        return AutoLyricsSync()


class TestComponentFactory(IComponentFactory):
    """Test factory for unit testing - Dependency Inversion"""
    
    def create_audio_engine(self, config: Optional[AudioConfig] = None) -> IAudioEngine:
        """Create mock audio engine for testing"""
        # Would import from tests.mocks
        from implementations.audio_engine import WebAudioEngine
        return WebAudioEngine(config or AudioConfig(), None, None)
    
    def create_midi_controller(self, device_id: Optional[str] = None) -> IMidiController:
        """Create mock MIDI controller for testing"""
        # Would import from tests.mocks  
        from implementations.midi_controller import YamahaAG06Controller
        return YamahaAG06Controller(device_id, None, None)
    
    def create_preset_manager(self, storage_path: Optional[str] = None) -> IPresetManager:
        """Create mock preset manager for testing"""
        # Would import from tests.mocks
        from implementations.preset_manager import JsonPresetManager
        return JsonPresetManager(storage_path or "./test_presets", None, None)
    
    def create_vocal_processor(self) -> IVocalProcessor:
        """Create mock vocal processor for testing"""
        # Would import from tests.mocks
        from implementations.karaoke_integration import AdvancedVocalProcessor
        return AdvancedVocalProcessor(None, None)