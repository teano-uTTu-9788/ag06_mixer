"""
Preset Manager Interface for AG06 Mixer
Follows SOLID principles - Interface Segregation
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Preset:
    """Preset data structure - Single Responsibility"""
    name: str
    category: str
    parameters: Dict[str, Any]
    created_at: datetime
    modified_at: datetime
    version: str = "1.0.0"


class IPresetManager(ABC):
    """Preset management interface - Single Responsibility"""
    
    @abstractmethod
    async def load_preset(self, name: str) -> Preset:
        """Load a preset by name"""
        pass
    
    @abstractmethod
    async def save_preset(self, preset: Preset) -> bool:
        """Save a preset"""
        pass
    
    @abstractmethod
    async def delete_preset(self, name: str) -> bool:
        """Delete a preset"""
        pass
    
    @abstractmethod
    async def list_presets(self, category: Optional[str] = None) -> List[str]:
        """List available presets"""
        pass


class IPresetValidator(ABC):
    """Preset validation interface - Single Responsibility"""
    
    @abstractmethod
    async def validate_preset(self, preset: Preset) -> bool:
        """Validate preset parameters"""
        pass
    
    @abstractmethod
    async def get_validation_errors(self, preset: Preset) -> List[str]:
        """Get validation error messages"""
        pass


class IPresetExporter(ABC):
    """Preset export interface - Single Responsibility"""
    
    @abstractmethod
    async def export_preset(self, preset: Preset, format: str) -> bytes:
        """Export preset to specified format"""
        pass
    
    @abstractmethod
    async def import_preset(self, data: bytes, format: str) -> Preset:
        """Import preset from data"""
        pass
    
    @abstractmethod
    async def get_supported_formats(self) -> List[str]:
        """Get list of supported export formats"""
        pass