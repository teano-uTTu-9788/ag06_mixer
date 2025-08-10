"""
Audio Engine Interface for AG06 Mixer
Follows SOLID principles - Interface Segregation
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AudioConfig:
    """Audio configuration following Single Responsibility"""
    sample_rate: int = 48000
    bit_depth: int = 24
    buffer_size: int = 256
    channels: int = 6
    

class IAudioEngine(ABC):
    """Audio processing interface - Single Responsibility"""
    
    @abstractmethod
    async def initialize(self, config: AudioConfig) -> bool:
        """Initialize audio engine with configuration"""
        pass
    
    @abstractmethod
    async def process_audio(self, input_buffer: bytes) -> bytes:
        """Process audio buffer"""
        pass
    
    @abstractmethod
    async def get_latency(self) -> float:
        """Get current processing latency in ms"""
        pass
    

class IAudioEffects(ABC):
    """Effects processing interface - Interface Segregation"""
    
    @abstractmethod
    async def apply_reverb(self, signal: bytes, params: Dict[str, float]) -> bytes:
        """Apply reverb effect"""
        pass
    
    @abstractmethod
    async def apply_eq(self, signal: bytes, bands: Dict[str, float]) -> bytes:
        """Apply EQ"""
        pass
    

class IAudioMetrics(ABC):
    """Audio metrics interface - Single Responsibility"""
    
    @abstractmethod
    async def get_levels(self) -> Dict[int, float]:
        """Get current channel levels"""
        pass
    
    @abstractmethod
    async def get_peak_values(self) -> Dict[int, Tuple[float, float]]:
        """Get peak and RMS values per channel"""
        pass