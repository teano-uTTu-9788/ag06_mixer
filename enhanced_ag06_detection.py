#!/usr/bin/env python3
"""
Enhanced AG06 Detection System
Comprehensive detection for Yamaha AG06 mixer with detailed diagnostics
"""

import asyncio
import subprocess
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

class EnhancedAG06Detector:
    """Enhanced AG06 detection with multiple methods and detailed diagnostics"""
    
    def __init__(self):
        self.detection_results = {}
        
    async def comprehensive_detection(self) -> Dict:
        """Comprehensive AG06 detection with detailed diagnostics"""
        print("🔍 Enhanced AG06 Detection Starting...")
        print("=" * 50)
        
        results = {
            "ag06_detected": False,
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
            results["ag06_detected"] = True
            
        # Method 2: USB Device Tree
        print("2. Checking USB device tree...")
        usb_result = await self._check_usb_devices()
        results["detection_methods"]["usb_devices"] = usb_result
        if usb_result["detected"]:
            results["ag06_detected"] = True
            
        # Method 3: IO Registry (macOS specific)
        print("3. Checking IO Registry...")
        ioreg_result = await self._check_ioreg()
        results["detection_methods"]["io_registry"] = ioreg_result
        if ioreg_result["detected"]:
            results["ag06_detected"] = True
            
        # Method 4: Core Audio (macOS specific)
        print("4. Checking Core Audio...")
        coreaudio_result = await self._check_core_audio()
        results["detection_methods"]["core_audio"] = coreaudio_result
        if coreaudio_result["detected"]:
            results["ag06_detected"] = True
            
        # Method 5: Audio MIDI Setup devices
        print("5. Checking Audio MIDI Setup...")
        midi_result = await self._check_audio_midi()
        results["detection_methods"]["audio_midi"] = midi_result
        if midi_result["detected"]:
            results["ag06_detected"] = True
            
        # Diagnostics
        results["diagnostics"] = await self._run_diagnostics()
        
        # Generate troubleshooting steps
        results["troubleshooting"] = self._generate_troubleshooting(results)
        
        # System information
        results["system_info"] = await self._get_system_info()
        
        return results
    
    async def _check_system_audio(self) -> Dict:
        """Check system audio devices for AG06"""
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
            
            # Look for various AG06 identifiers
            ag06_indicators = [
                'AG06', 'ag06', 'Yamaha AG06', 'YAMAHA AG06',
                'Yamaha Corporation AG06', 'AG-06', 'ag-06'
            ]
            
            detected = any(indicator in audio_info for indicator in ag06_indicators)
            
            return {
                "detected": detected,
                "raw_output": audio_info,
                "indicators_found": [ind for ind in ag06_indicators if ind in audio_info],
                "method": "system_profiler SPAudioDataType"
            }
            
        except Exception as e:
            return {"detected": False, "error": str(e)}
    
    async def _check_usb_devices(self) -> Dict:
        """Check USB devices for AG06"""
        try:
            proc = await asyncio.create_subprocess_exec(
                'system_profiler', 'SPUSBDataType',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                return {"detected": False, "error": stderr.decode()}
                
            usb_info = stdout.decode()
            
            # Look for Yamaha vendor ID and AG06 product
            yamaha_indicators = [
                'Yamaha', 'yamaha', 'YAMAHA',
                'Vendor ID: 0x0499',  # Common Yamaha vendor ID
                'AG06', 'ag06', 'AG-06'
            ]
            
            detected = any(indicator in usb_info for indicator in yamaha_indicators)
            
            return {
                "detected": detected,
                "raw_output": usb_info,
                "indicators_found": [ind for ind in yamaha_indicators if ind in usb_info],
                "method": "system_profiler SPUSBDataType"
            }
            
        except Exception as e:
            return {"detected": False, "error": str(e)}
    
    async def _check_ioreg(self) -> Dict:
        """Check IO Registry for AG06"""
        try:
            proc = await asyncio.create_subprocess_exec(
                'ioreg', '-p', 'IOUSB', '-l',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                return {"detected": False, "error": stderr.decode()}
                
            ioreg_info = stdout.decode()
            
            ag06_indicators = [
                'AG06', 'ag06', 'Yamaha', 'yamaha',
                '"idVendor" = 1177',  # 0x0499 in decimal
                '"USB Product Name" = "AG06"'
            ]
            
            detected = any(indicator in ioreg_info for indicator in ag06_indicators)
            
            return {
                "detected": detected,
                "indicators_found": [ind for ind in ag06_indicators if ind in ioreg_info],
                "method": "ioreg -p IOUSB -l"
            }
            
        except Exception as e:
            return {"detected": False, "error": str(e)}
    
    async def _check_core_audio(self) -> Dict:
        """Check Core Audio devices"""
        try:
            # Use audiodevice command if available
            proc = await asyncio.create_subprocess_exec(
                'python3', '-c', '''
import sys
try:
    import sounddevice as sd
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if "AG06" in str(device) or "Yamaha" in str(device):
            print(f"FOUND: {device}")
            sys.exit(0)
    print("NOT_FOUND")
    sys.exit(1)
except ImportError:
    print("SOUNDDEVICE_NOT_AVAILABLE")
    sys.exit(2)
''',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            output = stdout.decode().strip()
            
            if proc.returncode == 0:
                return {"detected": True, "device_info": output, "method": "sounddevice"}
            elif proc.returncode == 2:
                return {"detected": False, "error": "sounddevice not available", "method": "sounddevice"}
            else:
                return {"detected": False, "method": "sounddevice"}
                
        except Exception as e:
            return {"detected": False, "error": str(e)}
    
    async def _check_audio_midi(self) -> Dict:
        """Check Audio MIDI Setup devices"""
        try:
            # Check if Audio MIDI Setup has AG06
            proc = await asyncio.create_subprocess_exec(
                'python3', '-c', '''
import subprocess
import sys

try:
    # Try to list audio devices using coreaudio
    result = subprocess.run(["system_profiler", "SPAudioDataType"], 
                          capture_output=True, text=True, timeout=10)
    
    if "AG06" in result.stdout or "Yamaha" in result.stdout:
        print("DETECTED_IN_SYSTEM")
        
    # Also check for audio units
    result2 = subprocess.run(["auval", "-a"], 
                           capture_output=True, text=True, timeout=10)
    if result2.returncode == 0 and ("AG06" in result2.stdout or "Yamaha" in result2.stdout):
        print("DETECTED_IN_AUVAL")
        
except Exception as e:
    print(f"ERROR: {e}")
''',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            
            output = stdout.decode().strip()
            detected = "DETECTED" in output
            
            return {
                "detected": detected,
                "details": output,
                "method": "audio_midi_setup"
            }
            
        except Exception as e:
            return {"detected": False, "error": str(e)}
    
    async def _run_diagnostics(self) -> Dict:
        """Run diagnostic tests"""
        diagnostics = {}
        
        try:
            # Check if AG06 driver is installed
            driver_paths = [
                "/Library/Audio/Plug-Ins/HAL/YamahaUSBAudioDriver.plugin",
                "/System/Library/Extensions/YamahaUSBAudio.kext",
                "/Applications/Yamaha Steinberg USB Driver.app"
            ]
            
            installed_drivers = []
            for path in driver_paths:
                if Path(path).exists():
                    installed_drivers.append(path)
                    
            diagnostics["drivers"] = {
                "installed_drivers": installed_drivers,
                "driver_count": len(installed_drivers)
            }
            
            # Check USB power management
            proc = await asyncio.create_subprocess_exec(
                'pmset', '-g',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL
            )
            stdout, _ = await proc.communicate()
            power_info = stdout.decode()
            
            diagnostics["power_management"] = {
                "usb_sleep": "USB sleep" in power_info,
                "power_info": power_info[:500]  # Truncate for brevity
            }
            
        except Exception as e:
            diagnostics["error"] = str(e)
            
        return diagnostics
    
    def _generate_troubleshooting(self, results: Dict) -> List[str]:
        """Generate troubleshooting steps based on detection results"""
        steps = []
        
        if not results["ag06_detected"]:
            steps.extend([
                "1. Physical Connection:",
                "   • Ensure AG06 is powered on (power LED should be lit)",
                "   • Check USB cable connection (try a different cable)",
                "   • Try a different USB port on your Mac",
                "   • Ensure USB cable supports data transfer (not just power)",
                "",
                "2. Driver Installation:",
                "   • Download latest Yamaha AG06 driver from Yamaha website",
                "   • Install Yamaha Steinberg USB Driver",
                "   • Restart Mac after driver installation",
                "",
                "3. System Settings:",
                "   • Open Audio MIDI Setup (Applications > Utilities)",
                "   • Check if AG06 appears in device list",
                "   • Open System Preferences > Sound",
                "   • Look for AG06 in Input/Output device list",
                "",
                "4. macOS Security:",
                "   • Check System Preferences > Security & Privacy",
                "   • Allow Yamaha driver if blocked",
                "   • Restart Mac after allowing driver",
                "",
                "5. Advanced Troubleshooting:",
                "   • Reset NVRAM/PRAM (restart while holding Option+Command+P+R)",
                "   • Reset SMC (System Management Controller)",
                "   • Try AG06 on another Mac to isolate hardware issues"
            ])
        else:
            steps.extend([
                "✅ AG06 Detected Successfully!",
                "",
                "Optimization Tips:",
                "• Set sample rate to 48kHz for best performance",
                "• Use 512 sample buffer size for low latency",
                "• Enable 'Advanced' UI mode in AiOke for full controls"
            ])
        
        return steps
    
    async def _get_system_info(self) -> Dict:
        """Get relevant system information"""
        try:
            proc = await asyncio.create_subprocess_exec(
                'sw_vers',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL
            )
            stdout, _ = await proc.communicate()
            system_info = stdout.decode()
            
            return {
                "macos_version": system_info,
                "timestamp": asyncio.get_event_loop().time()
            }
        except:
            return {"error": "Could not get system info"}

async def main():
    """Run enhanced AG06 detection"""
    detector = EnhancedAG06Detector()
    results = await detector.comprehensive_detection()
    
    print("\n" + "=" * 50)
    print("🎤 ENHANCED AG06 DETECTION RESULTS")
    print("=" * 50)
    
    if results["ag06_detected"]:
        print("✅ AG06 DETECTED!")
        print("\nDetection Methods That Found AG06:")
        for method, result in results["detection_methods"].items():
            if result["detected"]:
                print(f"  ✅ {method}: {result.get('indicators_found', ['Found'])}")
    else:
        print("❌ AG06 NOT DETECTED")
        print("\nDetection Methods Tried:")
        for method, result in results["detection_methods"].items():
            status = "✅" if result["detected"] else "❌"
            print(f"  {status} {method}")
            if "error" in result:
                print(f"      Error: {result['error']}")
    
    print(f"\n📊 Diagnostics:")
    if "drivers" in results["diagnostics"]:
        driver_count = results["diagnostics"]["drivers"]["driver_count"]
        print(f"  Drivers installed: {driver_count}")
        if driver_count > 0:
            for driver in results["diagnostics"]["drivers"]["installed_drivers"]:
                print(f"    • {driver}")
    
    print(f"\n🔧 Troubleshooting Steps:")
    for step in results["troubleshooting"]:
        print(f"  {step}")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())