"""
Implementations package - SOLID-compliant concrete implementations
"""

from .audio_engine import WebAudioEngine, ProfessionalAudioEffects, RealtimeAudioMetrics
from .midi_controller import YamahaAG06Controller, UsbMidiDiscovery, FlexibleMidiMapping
from .preset_manager import JsonPresetManager, SchemaPresetValidator, MultiFormatPresetExporter
from .karaoke_integration import AdvancedVocalProcessor, StudioKaraokeEffects, AutoLyricsSync, MLPerformanceScoring

__all__ = [
    'WebAudioEngine', 'ProfessionalAudioEffects', 'RealtimeAudioMetrics',
    'YamahaAG06Controller', 'UsbMidiDiscovery', 'FlexibleMidiMapping',
    'JsonPresetManager', 'SchemaPresetValidator', 'MultiFormatPresetExporter',
    'AdvancedVocalProcessor', 'StudioKaraokeEffects', 'AutoLyricsSync', 'MLPerformanceScoring'
]