"""
Karaoke Integration Interface for AG06 Mixer
Follows SOLID principles - Interface Segregation
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum


class ProcessingMode(Enum):
    """Audio processing modes for karaoke"""
    VOCAL_REMOVAL = "vocal_removal"
    VOCAL_ENHANCEMENT = "vocal_enhancement"
    PITCH_CORRECTION = "pitch_correction"
    HARMONY = "harmony"


@dataclass
class KaraokeTrack:
    """Karaoke track data - Single Responsibility"""
    title: str
    artist: str
    duration: float
    key: str
    tempo: int
    lyrics_sync: Optional[List[Tuple[float, str]]] = None


class IVocalProcessor(ABC):
    """Vocal processing interface - Single Responsibility"""
    
    @abstractmethod
    async def remove_vocals(self, audio_data: bytes) -> bytes:
        """Remove vocals from audio"""
        pass
    
    @abstractmethod
    async def enhance_vocals(self, audio_data: bytes, enhancement_level: float) -> bytes:
        """Enhance vocal clarity"""
        pass
    
    @abstractmethod
    async def apply_pitch_correction(self, audio_data: bytes, target_key: str) -> bytes:
        """Apply pitch correction to vocals"""
        pass


class IKaraokeEffects(ABC):
    """Karaoke effects interface - Single Responsibility"""
    
    @abstractmethod
    async def add_reverb(self, audio_data: bytes, room_size: float) -> bytes:
        """Add reverb effect"""
        pass
    
    @abstractmethod
    async def add_echo(self, audio_data: bytes, delay_ms: int, feedback: float) -> bytes:
        """Add echo effect"""
        pass
    
    @abstractmethod
    async def transpose_key(self, audio_data: bytes, semitones: int) -> bytes:
        """Transpose audio key"""
        pass


class ILyricsSync(ABC):
    """Lyrics synchronization interface - Single Responsibility"""
    
    @abstractmethod
    async def load_lyrics(self, track_id: str) -> List[Tuple[float, str]]:
        """Load synchronized lyrics"""
        pass
    
    @abstractmethod
    async def get_current_lyric(self, position: float) -> Optional[str]:
        """Get current lyric at position"""
        pass
    
    @abstractmethod
    async def sync_lyrics_to_audio(self, audio_file: str, lyrics: str) -> List[Tuple[float, str]]:
        """Auto-sync lyrics to audio"""
        pass


class IPerformanceScoring(ABC):
    """Performance scoring interface - Single Responsibility"""
    
    @abstractmethod
    async def analyze_pitch_accuracy(self, reference: bytes, performance: bytes) -> float:
        """Analyze pitch accuracy (0-100)"""
        pass
    
    @abstractmethod
    async def analyze_timing(self, reference: bytes, performance: bytes) -> float:
        """Analyze timing accuracy (0-100)"""
        pass
    
    @abstractmethod
    async def calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall performance score"""
        pass