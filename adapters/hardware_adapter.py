"""
Hardware Adapter Pattern Implementation
Provides abstraction layer for external hardware systems
Follows Adapter pattern for flexibility and testability
"""
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import numpy as np


class IAudioHardwareAdapter(ABC):
    """Abstract interface for audio hardware adapters"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to hardware device"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from hardware device"""
        pass
    
    @abstractmethod
    async def read_audio(self, num_frames: int) -> np.ndarray:
        """Read audio from hardware"""
        pass
    
    @abstractmethod
    async def write_audio(self, data: np.ndarray) -> None:
        """Write audio to hardware"""
        pass
    
    @abstractmethod
    async def get_device_info(self) -> Dict[str, Any]:
        """Get hardware device information"""
        pass


class IMidiHardwareAdapter(ABC):
    """Abstract interface for MIDI hardware adapters"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to MIDI device"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from MIDI device"""
        pass
    
    @abstractmethod
    async def send_midi(self, message: bytes) -> None:
        """Send MIDI message to hardware"""
        pass
    
    @abstractmethod
    async def receive_midi(self) -> Optional[bytes]:
        """Receive MIDI message from hardware"""
        pass
    
    @abstractmethod
    async def get_device_info(self) -> Dict[str, Any]:
        """Get MIDI device information"""
        pass


class AG06AudioAdapter(IAudioHardwareAdapter):
    """
    Adapter for AG06 audio hardware
    Translates between application and hardware interfaces
    """
    
    def __init__(self, device_id: str = "AG06"):
        """
        Initialize AG06 audio adapter
        
        Args:
            device_id: Device identifier
        """
        self.device_id = device_id
        self.connected = False
        self.sample_rate = 48000
        self.channels = 2
        self.buffer_size = 512
        self._audio_buffer = None
    
    async def connect(self) -> bool:
        """Connect to AG06 audio hardware"""
        try:
            # In production, would use actual audio API (e.g., pyaudio, sounddevice)
            # For now, simulate connection
            print(f"Connecting to AG06 audio device: {self.device_id}")
            
            # Initialize audio buffer
            self._audio_buffer = np.zeros((self.buffer_size, self.channels), dtype=np.float32)
            
            self.connected = True
            print("✅ AG06 audio adapter connected")
            return True
            
        except Exception as e:
            print(f"Failed to connect AG06 audio: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from AG06 audio hardware"""
        if self.connected:
            print("Disconnecting AG06 audio adapter")
            self.connected = False
            self._audio_buffer = None
    
    async def read_audio(self, num_frames: int) -> np.ndarray:
        """
        Read audio from AG06 hardware
        
        Args:
            num_frames: Number of frames to read
            
        Returns:
            Audio data as numpy array
        """
        if not self.connected:
            raise RuntimeError("AG06 audio not connected")
        
        # In production, would read from actual hardware
        # For now, generate test signal
        t = np.linspace(0, num_frames / self.sample_rate, num_frames)
        
        # Generate stereo test signal (440Hz on left, 880Hz on right)
        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        right = 0.5 * np.sin(2 * np.pi * 880 * t)
        
        return np.column_stack((left, right)).astype(np.float32)
    
    async def write_audio(self, data: np.ndarray) -> None:
        """
        Write audio to AG06 hardware
        
        Args:
            data: Audio data to write
        """
        if not self.connected:
            raise RuntimeError("AG06 audio not connected")
        
        # In production, would write to actual hardware
        # For now, just validate data
        if data.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {data.shape[1]}")
        
        # Simulate write latency
        await asyncio.sleep(0.001)
    
    async def get_device_info(self) -> Dict[str, Any]:
        """Get AG06 device information"""
        return {
            'device_id': self.device_id,
            'name': 'Yamaha AG06',
            'connected': self.connected,
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'buffer_size': self.buffer_size,
            'latency_ms': 0.034,
            'driver': 'CoreAudio' if self.connected else 'None'
        }


class AG06MidiAdapter(IMidiHardwareAdapter):
    """
    Adapter for AG06 MIDI hardware
    Translates between application and MIDI hardware interfaces
    """
    
    def __init__(self, device_id: str = "AG06-MIDI"):
        """
        Initialize AG06 MIDI adapter
        
        Args:
            device_id: Device identifier
        """
        self.device_id = device_id
        self.connected = False
        self._midi_queue = asyncio.Queue(maxsize=100)
    
    async def connect(self) -> bool:
        """Connect to AG06 MIDI hardware"""
        try:
            # In production, would use actual MIDI API (e.g., rtmidi, mido)
            # For now, simulate connection
            print(f"Connecting to AG06 MIDI device: {self.device_id}")
            
            self.connected = True
            print("✅ AG06 MIDI adapter connected")
            
            # Start MIDI monitoring task
            asyncio.create_task(self._monitor_midi())
            
            return True
            
        except Exception as e:
            print(f"Failed to connect AG06 MIDI: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from AG06 MIDI hardware"""
        if self.connected:
            print("Disconnecting AG06 MIDI adapter")
            self.connected = False
    
    async def send_midi(self, message: bytes) -> None:
        """
        Send MIDI message to AG06 hardware
        
        Args:
            message: MIDI message bytes
        """
        if not self.connected:
            raise RuntimeError("AG06 MIDI not connected")
        
        # In production, would send to actual hardware
        # For now, just validate message
        if len(message) < 2:
            raise ValueError("Invalid MIDI message")
        
        # Log MIDI send (for debugging)
        status = message[0]
        channel = status & 0x0F
        msg_type = status & 0xF0
        
        print(f"MIDI Send: Type={msg_type:02X}, Channel={channel}, Data={message[1:]}")
    
    async def receive_midi(self) -> Optional[bytes]:
        """
        Receive MIDI message from AG06 hardware
        
        Returns:
            MIDI message bytes or None
        """
        if not self.connected:
            return None
        
        try:
            # Non-blocking receive
            message = self._midi_queue.get_nowait()
            return message
        except asyncio.QueueEmpty:
            return None
    
    async def get_device_info(self) -> Dict[str, Any]:
        """Get AG06 MIDI device information"""
        return {
            'device_id': self.device_id,
            'name': 'Yamaha AG06 MIDI',
            'connected': self.connected,
            'vendor_id': '0x0499',
            'product_id': '0x170d',
            'input_ports': 1,
            'output_ports': 1,
            'driver': 'CoreMIDI' if self.connected else 'None'
        }
    
    async def _monitor_midi(self):
        """Monitor for incoming MIDI messages (simulation)"""
        while self.connected:
            # Simulate occasional MIDI messages
            await asyncio.sleep(5)
            
            if self.connected:
                # Generate test MIDI message (CC for volume)
                test_message = bytes([0xB0, 7, 100])  # CC7 (volume) on channel 0
                await self._midi_queue.put(test_message)


class HardwareAdapterFactory:
    """
    Factory for creating hardware adapters
    Provides centralized adapter creation
    """
    
    @staticmethod
    def create_audio_adapter(device_type: str = "AG06") -> IAudioHardwareAdapter:
        """
        Create audio hardware adapter
        
        Args:
            device_type: Type of device
            
        Returns:
            Audio hardware adapter instance
        """
        if device_type == "AG06":
            return AG06AudioAdapter()
        else:
            raise ValueError(f"Unknown audio device type: {device_type}")
    
    @staticmethod
    def create_midi_adapter(device_type: str = "AG06") -> IMidiHardwareAdapter:
        """
        Create MIDI hardware adapter
        
        Args:
            device_type: Type of device
            
        Returns:
            MIDI hardware adapter instance
        """
        if device_type == "AG06":
            return AG06MidiAdapter()
        else:
            raise ValueError(f"Unknown MIDI device type: {device_type}")
    
    @staticmethod
    def create_adapters(device_type: str = "AG06") -> tuple:
        """
        Create both audio and MIDI adapters
        
        Args:
            device_type: Type of device
            
        Returns:
            Tuple of (audio_adapter, midi_adapter)
        """
        audio_adapter = HardwareAdapterFactory.create_audio_adapter(device_type)
        midi_adapter = HardwareAdapterFactory.create_midi_adapter(device_type)
        return audio_adapter, midi_adapter


# Export adapters
__all__ = [
    'IAudioHardwareAdapter',
    'IMidiHardwareAdapter',
    'AG06AudioAdapter',
    'AG06MidiAdapter',
    'HardwareAdapterFactory'
]