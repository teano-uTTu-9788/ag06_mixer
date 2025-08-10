"""
MIDI Controller Implementation for AG06 Mixer
Follows SOLID principles - Single Responsibility
"""
import asyncio
from typing import Optional, List, Dict, Any
from collections import deque

from interfaces.midi_controller import (
    IMidiController, IMidiDeviceDiscovery, IMidiMapping,
    MidiMessage, MidiMessageType
)


class YamahaAG06Controller(IMidiController):
    """Yamaha AG06 MIDI controller - Single Responsibility: MIDI Communication"""
    
    def __init__(self,
                 device_id: Optional[str],
                 discovery: IMidiDeviceDiscovery,
                 mapping: IMidiMapping):
        """Constructor with dependency injection"""
        self._device_id = device_id or "Yamaha AG06"
        self._discovery = discovery  # Injected dependency
        self._mapping = mapping     # Injected dependency
        self._connected = False
        self._message_queue = deque(maxlen=100)
    
    async def connect(self, device_id: str) -> bool:
        """Connect to MIDI device"""
        try:
            # Use discovery service to find device
            devices = await self._discovery.scan_devices()
            
            for device in devices:
                if device.get('id') == device_id or device.get('name') == device_id:
                    self._device_id = device_id
                    self._connected = True
                    return True
            
            return False
        except Exception as e:
            print(f"MIDI connection failed: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from MIDI device"""
        self._connected = False
        self._message_queue.clear()
    
    async def send_message(self, message: MidiMessage) -> None:
        """Send MIDI message"""
        if not self._connected:
            raise RuntimeError("MIDI controller not connected")
        
        # Format and send MIDI message
        midi_bytes = self._format_message(message)
        await self._send_midi_bytes(midi_bytes)
    
    async def receive_message(self) -> Optional[MidiMessage]:
        """Receive MIDI message (non-blocking)"""
        if not self._connected:
            return None
        
        if self._message_queue:
            return self._message_queue.popleft()
        
        return None
    
    def _format_message(self, message: MidiMessage) -> bytes:
        """Format MIDI message to bytes"""
        status = message.type.value | (message.channel & 0x0F)
        
        if message.data2 is not None:
            return bytes([status, message.data1, message.data2])
        else:
            return bytes([status, message.data1])
    
    async def _send_midi_bytes(self, midi_bytes: bytes) -> None:
        """Send raw MIDI bytes (hardware specific)"""
        # Actual hardware communication would go here
        pass


class UsbMidiDiscovery(IMidiDeviceDiscovery):
    """USB MIDI device discovery - Single Responsibility: Device Discovery"""
    
    def __init__(self):
        """Initialize discovery service"""
        self._device_cache = {}
    
    async def scan_devices(self) -> List[Dict[str, str]]:
        """Scan for available MIDI devices"""
        devices = []
        
        # Simulated device discovery
        # In real implementation, would scan USB devices
        devices.append({
            'id': 'Yamaha-AG06-001',
            'name': 'Yamaha AG06',
            'vendor': 'Yamaha',
            'type': 'USB Audio'
        })
        
        return devices
    
    async def get_device_info(self, device_id: str) -> Dict[str, Any]:
        """Get detailed device information"""
        if device_id in self._device_cache:
            return self._device_cache[device_id]
        
        # Fetch device info
        info = {
            'id': device_id,
            'name': 'Yamaha AG06',
            'vendor_id': '0x0499',
            'product_id': '0x170d',
            'channels': 6,
            'sample_rates': [44100, 48000, 96000, 192000],
            'bit_depths': [16, 24]
        }
        
        self._device_cache[device_id] = info
        return info


class FlexibleMidiMapping(IMidiMapping):
    """Flexible MIDI mapping - Single Responsibility: Parameter Mapping"""
    
    def __init__(self):
        """Initialize mapping service"""
        self._mappings: Dict[int, str] = {}
        self._reverse_mappings: Dict[str, int] = {}
    
    async def map_control(self, cc_number: int, parameter: str) -> None:
        """Map MIDI CC to parameter"""
        # Remove old mapping if exists
        if cc_number in self._mappings:
            old_param = self._mappings[cc_number]
            del self._reverse_mappings[old_param]
        
        # Create new mapping
        self._mappings[cc_number] = parameter
        self._reverse_mappings[parameter] = cc_number
    
    async def get_mapping(self, cc_number: int) -> Optional[str]:
        """Get parameter mapped to CC"""
        return self._mappings.get(cc_number)
    
    async def clear_mappings(self) -> None:
        """Clear all MIDI mappings"""
        self._mappings.clear()
        self._reverse_mappings.clear()
    
    def get_cc_for_parameter(self, parameter: str) -> Optional[int]:
        """Get CC number for a parameter (helper method)"""
        return self._reverse_mappings.get(parameter)