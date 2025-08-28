#!/usr/bin/env python3
"""
Enhanced AiOke Detection System
Comprehensive detection for AiOke mixer with detailed diagnostics
"""

import asyncio
import subprocess
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

class EnhancedAiOkeDetector:
    """Enhanced AiOke detection with multiple methods and detailed diagnostics"""
    
    def __init__(self):
        self.detection_results = {}
        
    async def comprehensive_detection(self) -> Dict:
        """Comprehensive AiOke detection with detailed diagnostics"""
        print("🔍 Enhanced AiOke Detection Starting...")
        print("=" * 50)
        
        results = {
            "aioke_detected": False,
            "detection_methods": {},
            "diagnostics": {},
            "troubleshooting": [],
            "system_info": {}
        }
        
        # Method 1: System Audio Devices
        print("1. Checking system audio devices...")
        audio_result = await self._check_system_audio()
        results["detection_methods"]["system_audio"] = audio_result
        if audio_result["detected"]:
            results["aioke_detected"] = True
            
        # Method 2: USB Device Tree
        print("2. Checking USB device tree...")
        usb_result = await self._check_usb_devices()
        results["detection_methods"]["usb_devices"] = usb_result
        if usb_result["detected"]:
            results["aioke_detected"] = True
            
        # Method 3: IO Registry (macOS specific)
        print("3. Checking IO Registry...")
        ioreg_result = await self._check_ioreg()
        results["detection_methods"]["io_registry"] = ioreg_result
        if ioreg_result["detected"]:
            results["aioke_detected"] = True
            
        # Method 4: Core Audio (macOS specific)
        print("4. Checking Core Audio...")
        coreaudio_result = await self._check_core_audio()
        results["detection_methods"]["core_audio"] = coreaudio_result
        if coreaudio_result["detected"]:
            results["aioke_detected"] = True
            
        # Method 5: Audio MIDI Setup devices
        print("5. Checking Audio MIDI Setup...")
        midi_result = await self._check_audio_midi()
        results["detection_methods"]["audio_midi"] = midi_result
        if midi_result["detected"]:
            results["aioke_detected"] = True
            
        # Diagnostics
        results["diagnostics"] = await self._run_diagnostics()
        
        # Generate troubleshooting steps
        results["troubleshooting"] = self._generate_troubleshooting(results)
        
        # System information
        results["system_info"] = await self._get_system_info()
        
        return results
    
    async def _check_system_audio(self) -> Dict:
        """Check system audio devices for AiOke"""
        try:
            proc = await asyncio.create_subprocess_exec(
                'system_profiler', 'SPAudioDataType',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                return {"detected": False, "error": stderr.decode()}
                
            audio_info = stdout.decode()
            
            # Look for various AiOke/AG06 identifiers (backward compatibility)
            aioke_indicators = [
                'AiOke', 'aioke', 'AIOKE',
                'AG06', 'ag06', 'Yamaha AG06', 'YAMAHA AG06',
                'Yamaha Corporation AG06', 'AG-06', 'ag-06'
            ]
            
            detected = any(indicator in audio_info for indicator in aioke_indicators)
            
            if detected:
                # Extract device details
                lines = audio_info.split('\n')
                device_info = {}
                for i, line in enumerate(lines):
                    if any(indicator in line for indicator in aioke_indicators):
                        # Extract surrounding lines for context
                        device_info["name"] = line.strip()
                        for j in range(max(0, i-5), min(len(lines), i+5)):
                            if "Manufacturer" in lines[j]:
                                device_info["manufacturer"] = lines[j].split(":")[-1].strip()
                            if "Sample Rate" in lines[j]:
                                device_info["sample_rate"] = lines[j].split(":")[-1].strip()
                            if "Channels" in lines[j]:
                                device_info["channels"] = lines[j].split(":")[-1].strip()
                return {"detected": True, "device_info": device_info}
            
            return {"detected": False, "raw_output": audio_info[:500]}
            
        except Exception as e:
            return {"detected": False, "error": str(e)}
    
    async def _check_usb_devices(self) -> Dict:
        """Check USB device tree for AiOke"""
        try:
            proc = await asyncio.create_subprocess_exec(
                'system_profiler', 'SPUSBDataType', '-json',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                return {"detected": False, "error": stderr.decode()}
            
            usb_data = json.loads(stdout.decode())
            
            # Search for AiOke/AG06 in USB tree
            def search_usb_tree(items):
                for item in items:
                    if isinstance(item, dict):
                        name = item.get('_name', '').lower()
                        manufacturer = item.get('manufacturer', '').lower()
                        
                        # Check for AiOke or AG06 (Yamaha) identifiers
                        if ('aioke' in name or 'ag06' in name or 'ag-06' in name or
                            ('yamaha' in manufacturer and any(x in name for x in ['mixer', 'audio']))):
                            return {
                                "detected": True,
                                "device": {
                                    "name": item.get('_name'),
                                    "manufacturer": item.get('manufacturer'),
                                    "vendor_id": item.get('vendor_id'),
                                    "product_id": item.get('product_id'),
                                    "location": item.get('location_id')
                                }
                            }
                        
                        # Recursive search
                        if '_items' in item:
                            result = search_usb_tree(item['_items'])
                            if result["detected"]:
                                return result
                
                return {"detected": False}
            
            # Search through USB data structure
            for section in usb_data.get('SPUSBDataType', []):
                if '_items' in section:
                    result = search_usb_tree(section['_items'])
                    if result["detected"]:
                        return result
            
            # Check for Yamaha vendor ID (0x0499)
            for section in usb_data.get('SPUSBDataType', []):
                if section.get('vendor_id', '') == '0x0499':
                    return {
                        "detected": True,
                        "device": {
                            "name": "Yamaha Audio Device (possibly AiOke)",
                            "vendor_id": "0x0499",
                            "note": "Yamaha device detected, may be AiOke"
                        }
                    }
                    
            return {"detected": False, "usb_device_count": len(usb_data.get('SPUSBDataType', []))}
            
        except Exception as e:
            return {"detected": False, "error": str(e)}
    
    async def _check_ioreg(self) -> Dict:
        """Check IO Registry for AiOke"""
        try:
            proc = await asyncio.create_subprocess_exec(
                'ioreg', '-r', '-c', 'IOAudioEngine',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                return {"detected": False, "error": "ioreg command failed"}
                
            ioreg_output = stdout.decode()
            
            # Check for AiOke/AG06 in IO Registry
            if any(x in ioreg_output.lower() for x in ['aioke', 'ag06', 'ag-06']):
                return {"detected": True, "method": "IO Registry"}
            
            # Check for Yamaha audio
            if 'yamaha' in ioreg_output.lower():
                return {"detected": True, "note": "Yamaha audio device found (possibly AiOke)"}
                
            return {"detected": False}
            
        except Exception as e:
            return {"detected": False, "error": str(e)}
    
    async def _check_core_audio(self) -> Dict:
        """Check Core Audio devices"""
        try:
            # Use osascript to get audio devices
            script = """
            tell application "System Preferences"
                get name of every audio device
            end tell
            """
            
            # Alternative: Check using command line
            proc = await asyncio.create_subprocess_exec(
                'osascript', '-e',
                'set devices to (do shell script "system_profiler SPAudioDataType")',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if stdout:
                output = stdout.decode().lower()
                if any(x in output for x in ['aioke', 'ag06', 'ag-06', 'yamaha']):
                    return {"detected": True, "method": "Core Audio"}
            
            return {"detected": False}
            
        except Exception as e:
            return {"detected": False, "error": str(e)}
    
    async def _check_audio_midi(self) -> Dict:
        """Check Audio MIDI Setup configuration"""
        try:
            # Check Audio MIDI Setup preferences
            plist_path = Path("~/Library/Preferences/com.apple.audio.AudioMIDISetup.plist").expanduser()
            
            if plist_path.exists():
                proc = await asyncio.create_subprocess_exec(
                    'plutil', '-convert', 'json', '-o', '-', str(plist_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                
                if proc.returncode == 0:
                    try:
                        prefs = json.loads(stdout.decode())
                        # Check for AiOke/AG06 in device configurations
                        prefs_str = json.dumps(prefs).lower()
                        if any(x in prefs_str for x in ['aioke', 'ag06', 'ag-06']):
                            return {"detected": True, "method": "Audio MIDI Setup preferences"}
                    except:
                        pass
                        
            return {"detected": False}
            
        except Exception as e:
            return {"detected": False, "error": str(e)}
    
    async def _run_diagnostics(self) -> Dict:
        """Run system diagnostics"""
        diagnostics = {}
        
        # Check if drivers are installed
        try:
            driver_paths = [
                "/Library/Audio/Plug-Ins/HAL/YamahaSteinbergUSBAudioDriver.driver",
                "/System/Library/Extensions/YamahaUSBMIDIDriver.kext",
                "/Library/Extensions/YamahaUSBMIDIDriver.kext"
            ]
            
            drivers_found = []
            for path in driver_paths:
                if Path(path).exists():
                    drivers_found.append(path)
            
            diagnostics["drivers_installed"] = len(drivers_found) > 0
            diagnostics["driver_paths"] = drivers_found
            
        except Exception as e:
            diagnostics["driver_check_error"] = str(e)
        
        # Check USB power
        try:
            proc = await asyncio.create_subprocess_exec(
                'system_profiler', 'SPPowerDataType',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            
            if proc.returncode == 0:
                power_info = stdout.decode()
                diagnostics["usb_power_ok"] = "Current Available" in power_info
        except:
            pass
            
        # Check audio system status
        try:
            proc = await asyncio.create_subprocess_exec(
                'ps', 'aux',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            
            if proc.returncode == 0:
                processes = stdout.decode()
                diagnostics["coreaudiod_running"] = "coreaudiod" in processes
                
        except:
            pass
            
        return diagnostics
    
    def _generate_troubleshooting(self, results: Dict) -> List[str]:
        """Generate troubleshooting steps based on detection results"""
        steps = []
        
        if not results["aioke_detected"]:
            steps.append("🔌 Check that AiOke is powered on and connected via USB")
            steps.append("🔄 Try a different USB cable or port")
            
            if not results.get("diagnostics", {}).get("drivers_installed"):
                steps.append("💿 Install Yamaha Steinberg USB Driver from https://www.yamaha.com/support/")
                steps.append("🔄 Restart your Mac after driver installation")
                
            steps.append("🔒 Check System Settings > Privacy & Security for blocked software")
            steps.append("🎚️ Open Audio MIDI Setup and rescan for devices")
            
            if results.get("diagnostics", {}).get("coreaudiod_running"):
                steps.append("🔧 Try restarting Core Audio: sudo killall coreaudiod")
            
            steps.append("📖 Refer to AIOKE_TROUBLESHOOTING_GUIDE.md for detailed instructions")
            
        return steps
    
    async def _get_system_info(self) -> Dict:
        """Get system information"""
        info = {}
        
        try:
            # macOS version
            proc = await asyncio.create_subprocess_exec(
                'sw_vers', '-productVersion',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            if proc.returncode == 0:
                info["macos_version"] = stdout.decode().strip()
                
            # Hardware model
            proc = await asyncio.create_subprocess_exec(
                'sysctl', '-n', 'hw.model',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            if proc.returncode == 0:
                info["hardware_model"] = stdout.decode().strip()
                
        except:
            pass
            
        return info
    
    def print_results(self, results: Dict):
        """Pretty print detection results"""
        print("\n" + "=" * 50)
        print("🎛️ AiOke Detection Results")
        print("=" * 50)
        
        if results["aioke_detected"]:
            print("✅ AiOke DETECTED!")
            print("\nDetection Methods:")
            for method, result in results["detection_methods"].items():
                if result.get("detected"):
                    print(f"  ✓ {method}: Success")
                    if "device_info" in result:
                        for key, value in result["device_info"].items():
                            print(f"    - {key}: {value}")
        else:
            print("❌ AiOke NOT DETECTED")
            print("\nDetection Methods:")
            for method, result in results["detection_methods"].items():
                status = "✓" if result.get("detected") else "✗"
                print(f"  {status} {method}")
                
        print("\nDiagnostics:")
        for key, value in results.get("diagnostics", {}).items():
            print(f"  - {key}: {value}")
            
        if results.get("troubleshooting"):
            print("\n📝 Troubleshooting Steps:")
            for step in results["troubleshooting"]:
                print(f"  {step}")
                
        print("\nSystem Information:")
        for key, value in results.get("system_info", {}).items():
            print(f"  - {key}: {value}")
            
        print("\n" + "=" * 50)

async def main():
    """Main detection routine"""
    detector = EnhancedAiOkeDetector()
    
    print("🎛️ AiOke Enhanced Detection System")
    print("Searching for AiOke mixer on your system...")
    print()
    
    results = await detector.comprehensive_detection()
    detector.print_results(results)
    
    # Save results to file
    results_file = Path("aioke_detection_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")
    
    # Open Audio MIDI Setup if not detected
    if not results["aioke_detected"]:
        print("\n🎚️ Opening Audio MIDI Setup for manual configuration...")
        subprocess.run(['open', '/Applications/Utilities/Audio MIDI Setup.app'])

if __name__ == "__main__":
    asyncio.run(main())