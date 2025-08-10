"""
Preset Manager Implementation for AG06 Mixer
Follows SOLID principles - Single Responsibility
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from interfaces.preset_manager import (
    IPresetManager, IPresetValidator, IPresetExporter,
    Preset
)


class JsonPresetManager(IPresetManager):
    """JSON-based preset manager - Single Responsibility: Preset Storage"""
    
    def __init__(self,
                 storage_path: str,
                 validator: IPresetValidator,
                 exporter: IPresetExporter):
        """Constructor with dependency injection"""
        self._storage_path = Path(storage_path)
        self._validator = validator  # Injected dependency
        self._exporter = exporter   # Injected dependency
        
        # Create storage directory if it doesn't exist
        self._storage_path.mkdir(parents=True, exist_ok=True)
    
    async def load_preset(self, name: str) -> Preset:
        """Load a preset by name"""
        preset_file = self._storage_path / f"{name}.json"
        
        if not preset_file.exists():
            raise FileNotFoundError(f"Preset '{name}' not found")
        
        with open(preset_file, 'r') as f:
            data = json.load(f)
        
        preset = Preset(
            name=data['name'],
            category=data['category'],
            parameters=data['parameters'],
            created_at=datetime.fromisoformat(data['created_at']),
            modified_at=datetime.fromisoformat(data['modified_at']),
            version=data.get('version', '1.0.0')
        )
        
        # Validate before returning
        if not await self._validator.validate_preset(preset):
            errors = await self._validator.get_validation_errors(preset)
            raise ValueError(f"Invalid preset: {', '.join(errors)}")
        
        return preset
    
    async def save_preset(self, preset: Preset) -> bool:
        """Save a preset"""
        # Validate before saving
        if not await self._validator.validate_preset(preset):
            return False
        
        preset_file = self._storage_path / f"{preset.name}.json"
        
        data = {
            'name': preset.name,
            'category': preset.category,
            'parameters': preset.parameters,
            'created_at': preset.created_at.isoformat(),
            'modified_at': datetime.now().isoformat(),
            'version': preset.version
        }
        
        with open(preset_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return True
    
    async def delete_preset(self, name: str) -> bool:
        """Delete a preset"""
        preset_file = self._storage_path / f"{name}.json"
        
        if preset_file.exists():
            preset_file.unlink()
            return True
        
        return False
    
    async def list_presets(self, category: Optional[str] = None) -> List[str]:
        """List available presets"""
        presets = []
        
        for preset_file in self._storage_path.glob("*.json"):
            if category:
                # Filter by category if specified
                with open(preset_file, 'r') as f:
                    data = json.load(f)
                    if data.get('category') == category:
                        presets.append(preset_file.stem)
            else:
                presets.append(preset_file.stem)
        
        return sorted(presets)


class SchemaPresetValidator(IPresetValidator):
    """Schema-based preset validator - Single Responsibility: Validation"""
    
    def __init__(self):
        """Initialize validator"""
        self._required_params = [
            'input_gain', 'output_gain', 'eq_bands',
            'reverb_level', 'compression_ratio'
        ]
    
    async def validate_preset(self, preset: Preset) -> bool:
        """Validate preset parameters"""
        errors = await self.get_validation_errors(preset)
        return len(errors) == 0
    
    async def get_validation_errors(self, preset: Preset) -> List[str]:
        """Get validation error messages"""
        errors = []
        
        # Check required parameters
        for param in self._required_params:
            if param not in preset.parameters:
                errors.append(f"Missing required parameter: {param}")
        
        # Validate parameter ranges
        if 'input_gain' in preset.parameters:
            gain = preset.parameters['input_gain']
            if not -60 <= gain <= 12:
                errors.append(f"Input gain out of range: {gain}")
        
        if 'output_gain' in preset.parameters:
            gain = preset.parameters['output_gain']
            if not -60 <= gain <= 12:
                errors.append(f"Output gain out of range: {gain}")
        
        if 'compression_ratio' in preset.parameters:
            ratio = preset.parameters['compression_ratio']
            if not 1 <= ratio <= 20:
                errors.append(f"Compression ratio out of range: {ratio}")
        
        return errors


class MultiFormatPresetExporter(IPresetExporter):
    """Multi-format preset exporter - Single Responsibility: Import/Export"""
    
    def __init__(self):
        """Initialize exporter"""
        self._formats = ['json', 'xml', 'yaml']
    
    async def export_preset(self, preset: Preset, format: str) -> bytes:
        """Export preset to specified format"""
        if format not in self._formats:
            raise ValueError(f"Unsupported format: {format}")
        
        if format == 'json':
            data = {
                'name': preset.name,
                'category': preset.category,
                'parameters': preset.parameters,
                'created_at': preset.created_at.isoformat(),
                'modified_at': preset.modified_at.isoformat(),
                'version': preset.version
            }
            return json.dumps(data, indent=2).encode('utf-8')
        
        elif format == 'xml':
            # Simplified XML export
            xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<preset>
    <name>{preset.name}</name>
    <category>{preset.category}</category>
    <parameters>
        {self._params_to_xml(preset.parameters)}
    </parameters>
    <version>{preset.version}</version>
</preset>"""
            return xml.encode('utf-8')
        
        elif format == 'yaml':
            # Simplified YAML export
            yaml = f"""name: {preset.name}
category: {preset.category}
version: {preset.version}
parameters:
{self._params_to_yaml(preset.parameters)}"""
            return yaml.encode('utf-8')
    
    async def import_preset(self, data: bytes, format: str) -> Preset:
        """Import preset from data"""
        if format not in self._formats:
            raise ValueError(f"Unsupported format: {format}")
        
        if format == 'json':
            preset_data = json.loads(data.decode('utf-8'))
            return Preset(
                name=preset_data['name'],
                category=preset_data['category'],
                parameters=preset_data['parameters'],
                created_at=datetime.fromisoformat(preset_data['created_at']),
                modified_at=datetime.fromisoformat(preset_data['modified_at']),
                version=preset_data.get('version', '1.0.0')
            )
        
        # Other formats would be implemented similarly
        raise NotImplementedError(f"Import for {format} not yet implemented")
    
    async def get_supported_formats(self) -> List[str]:
        """Get list of supported export formats"""
        return self._formats.copy()
    
    def _params_to_xml(self, params: Dict[str, Any]) -> str:
        """Convert parameters to XML format"""
        xml_params = []
        for key, value in params.items():
            xml_params.append(f"        <{key}>{value}</{key}>")
        return "\n".join(xml_params)
    
    def _params_to_yaml(self, params: Dict[str, Any]) -> str:
        """Convert parameters to YAML format"""
        yaml_params = []
        for key, value in params.items():
            yaml_params.append(f"  {key}: {value}")
        return "\n".join(yaml_params)