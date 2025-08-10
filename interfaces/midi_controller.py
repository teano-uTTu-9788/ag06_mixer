"""
MIDI Controller Interface for AG06 Mixer
Follows SOLID principles - Interface Segregation
"""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class MidiMessageType(Enum):
    """MIDI message types"""
    NOTE_ON = 0x90
    NOTE_OFF = 0x80
    CONTROL_CHANGE = 0xB0
    PROGRAM_CHANGE = 0xC0
    SYSTEM_EXCLUSIVE = 0xF0


@dataclass
class MidiMessage:
    """MIDI message data structure"""
    type: MidiMessageType
    channel: int
    data1: int
    data2: Optional[int] = None
    

class IMidiController(ABC):
    """MIDI control interface - Single Responsibility"""
    
    @abstractmethod
    async def connect(self, device_id: str) -> bool:
        """Connect to MIDI device"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from MIDI device"""
        pass
    
    @abstractmethod
    async def send_message(self, message: MidiMessage) -> None:
        """Send MIDI message"""
        pass
    
    @abstractmethod
    async def receive_message(self) -> Optional[MidiMessage]:
        """Receive MIDI message (non-blocking)"""
        pass


class IMidiDeviceDiscovery(ABC):
    """MIDI device discovery interface - Single Responsibility"""
    
    @abstractmethod
    async def scan_devices(self) -> List[Dict[str, str]]:
        """Scan for available MIDI devices"""
        pass
    
    @abstractmethod
    async def get_device_info(self, device_id: str) -> Dict[str, Any]:
        """Get detailed device information"""
        pass


class IMidiMapping(ABC):
    """MIDI parameter mapping interface - Single Responsibility"""
    
    @abstractmethod
    async def map_control(self, cc_number: int, parameter: str) -> None:
        """Map MIDI CC to parameter"""
        pass
    
    @abstractmethod
    async def get_mapping(self, cc_number: int) -> Optional[str]:
        """Get parameter mapped to CC"""
        pass
    
    @abstractmethod
    async def clear_mappings(self) -> None:
        """Clear all MIDI mappings"""
        pass