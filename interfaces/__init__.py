"""
Interfaces package - SOLID Interface Segregation
"""

from .audio_engine import IAudioEngine, IAudioEffects, IAudioMetrics, AudioConfig
from .midi_controller import IMidiController, IMidiDeviceDiscovery, IMidiMapping
from .preset_manager import IPresetManager, IPresetValidator, IPresetExporter
from .karaoke_integration import IVocalProcessor, IKaraokeEffects, ILyricsSync, IPerformanceScoring

__all__ = [
    'IAudioEngine', 'IAudioEffects', 'IAudioMetrics', 'AudioConfig',
    'IMidiController', 'IMidiDeviceDiscovery', 'IMidiMapping',
    'IPresetManager', 'IPresetValidator', 'IPresetExporter',
    'IVocalProcessor', 'IKaraokeEffects', 'ILyricsSync', 'IPerformanceScoring'
]