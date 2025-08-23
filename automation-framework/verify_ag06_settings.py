#!/usr/bin/env python3
"""
AG06 Settings Verification Tool
Ensures all mixer settings are correct for your recording setup
"""

import subprocess
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple

class AG06SettingsVerifier:
    """Verifies and corrects AG06 mixer settings"""
    
    def __init__(self):
        self.settings_correct = True
        self.issues_found = []
        self.recommendations = []
        
    def check_audio_device(self) -> Tuple[bool, str]:
        """Check if AG06 is the active audio device"""
        try:
            result = subprocess.run(
                ["system_profiler", "SPAudioDataType"],
                capture_output=True,
                text=True
            )
            
            if "AG06" in result.stdout or "AG03" in result.stdout:
                # Check if it's the default device
                if "Default Output Device: Yes" in result.stdout:
                    return True, "✅ AG06 is set as default audio device"
                else:
                    self.issues_found.append("AG06 not set as default device")
                    return False, "⚠️ AG06 detected but not set as default"
            else:
                self.issues_found.append("AG06 not detected")
                return False, "❌ AG06 not detected by system"
        except Exception as e:
            return False, f"❌ Error checking audio device: {e}"
    
    def check_sample_rate(self) -> Tuple[bool, str]:
        """Verify sample rate is set correctly"""
        try:
            # Check for common professional sample rates
            result = subprocess.run(
                ["system_profiler", "SPAudioDataType"],
                capture_output=True,
                text=True
            )
            
            if "48000" in result.stdout or "44100" in result.stdout:
                return True, "✅ Sample rate set to professional standard"
            else:
                self.recommendations.append("Set sample rate to 48kHz in Audio MIDI Setup")
                return False, "⚠️ Check sample rate in Audio MIDI Setup"
        except Exception as e:
            return False, f"❌ Error checking sample rate: {e}"
    
    def verify_phantom_power_safety(self, mic_type: str = "unknown") -> Dict[str, str]:
        """Provide phantom power recommendations based on microphone type"""
        phantom_guide = {
            "condenser": {
                "setting": "ON",
                "led": "Lit",
                "note": "✅ Phantom power required for condenser microphones"
            },
            "dynamic": {
                "setting": "OFF",
                "led": "Dark",
                "note": "✅ No phantom power needed for dynamic microphones"
            },
            "ribbon_vintage": {
                "setting": "OFF",
                "led": "Dark",
                "note": "⚠️ WARNING: Never use phantom power with vintage ribbon mics!"
            },
            "ribbon_modern": {
                "setting": "Check Manual",
                "led": "Varies",
                "note": "📖 Consult microphone manual for phantom power requirements"
            },
            "line": {
                "setting": "OFF",
                "led": "Dark",
                "note": "✅ No phantom power needed for line-level inputs"
            },
            "unknown": {
                "setting": "OFF (Safe Default)",
                "led": "Dark",
                "note": "💡 Keep phantom power OFF until you verify your mic type"
            }
        }
        
        return phantom_guide.get(mic_type.lower(), phantom_guide["unknown"])
    
    def check_gain_staging(self) -> List[str]:
        """Provide gain staging recommendations"""
        recommendations = [
            "📊 Gain Staging Best Practices:",
            "• Set gain so peaks hit -12dB to -6dB",
            "• Start with gain at 12 o'clock",
            "• Dynamic mics need more gain than condensers",
            "• Use PAD if signal is too hot",
            "• Monitor input levels while adjusting"
        ]
        return recommendations
    
    def check_monitoring_setup(self) -> Dict[str, Any]:
        """Verify monitoring configuration"""
        return {
            "speakers": {
                "status": "✅ Monitor Out to JBL 310s",
                "level": "Set to comfortable listening level",
                "tip": "Start at 9 o'clock and increase gradually"
            },
            "headphones": {
                "status": "🎧 Headphone monitoring available",
                "level": "Adjust separately from speakers",
                "tip": "Use for zero-latency monitoring while recording"
            },
            "to_pc_switch": {
                "recording": "DRY CH 1-2G",
                "streaming": "LOOPBACK",
                "tip": "Use DRY for recording, LOOPBACK for streaming"
            }
        }
    
    def run_verification(self, mic_type: str = "unknown") -> Dict[str, Any]:
        """Run complete verification of AG06 settings"""
        print("\n" + "="*60)
        print("🎛️  AG06 Settings Verification Tool")
        print("="*60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "device_check": {},
            "sample_rate": {},
            "phantom_power": {},
            "gain_staging": [],
            "monitoring": {},
            "overall_status": "PASS",
            "issues": [],
            "recommendations": []
        }
        
        # Check audio device
        print("1️⃣ Checking Audio Device...")
        device_ok, device_msg = self.check_audio_device()
        print(f"   {device_msg}")
        results["device_check"] = {"status": device_ok, "message": device_msg}
        if not device_ok:
            self.settings_correct = False
        
        # Check sample rate
        print("\n2️⃣ Checking Sample Rate...")
        rate_ok, rate_msg = self.check_sample_rate()
        print(f"   {rate_msg}")
        results["sample_rate"] = {"status": rate_ok, "message": rate_msg}
        if not rate_ok:
            self.settings_correct = False
        
        # Phantom power recommendations
        print(f"\n3️⃣ Phantom Power Settings (Mic Type: {mic_type})...")
        phantom_info = self.verify_phantom_power_safety(mic_type)
        print(f"   +48V Button: {phantom_info['setting']}")
        print(f"   LED Status: {phantom_info['led']}")
        print(f"   {phantom_info['note']}")
        results["phantom_power"] = phantom_info
        
        # Gain staging
        print("\n4️⃣ Gain Staging Guidelines...")
        gain_tips = self.check_gain_staging()
        for tip in gain_tips:
            print(f"   {tip}")
        results["gain_staging"] = gain_tips
        
        # Monitoring setup
        print("\n5️⃣ Monitoring Configuration...")
        monitoring = self.check_monitoring_setup()
        print(f"   Speakers: {monitoring['speakers']['status']}")
        print(f"   TO PC Switch: {monitoring['to_pc_switch']['recording']} for recording")
        results["monitoring"] = monitoring
        
        # Final status
        print("\n" + "="*60)
        if self.settings_correct:
            print("✅ OVERALL STATUS: All critical settings verified!")
            results["overall_status"] = "PASS"
        else:
            print("⚠️ OVERALL STATUS: Some settings need attention")
            results["overall_status"] = "NEEDS_ATTENTION"
            results["issues"] = self.issues_found
            results["recommendations"] = self.recommendations
        
        print("="*60 + "\n")
        
        # Save results to file
        with open("ag06_verification_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("📄 Results saved to ag06_verification_results.json")
        
        return results
    
    def quick_check(self) -> bool:
        """Run a quick verification check"""
        device_ok, _ = self.check_audio_device()
        rate_ok, _ = self.check_sample_rate()
        return device_ok and rate_ok


def main():
    """Main entry point for AG06 settings verification"""
    import sys
    
    # Get microphone type from command line or ask
    mic_type = "unknown"
    if len(sys.argv) > 1:
        mic_type = sys.argv[1]
    else:
        print("\n🎤 What type of microphone are you using?")
        print("1. Condenser (requires phantom power)")
        print("2. Dynamic (no phantom power)")
        print("3. Ribbon (vintage - NO phantom power!)")
        print("4. Ribbon (modern - check manual)")
        print("5. Line input (keyboard, interface, etc.)")
        print("6. Not sure")
        
        choice = input("\nEnter choice (1-6): ").strip()
        mic_types = {
            "1": "condenser",
            "2": "dynamic", 
            "3": "ribbon_vintage",
            "4": "ribbon_modern",
            "5": "line",
            "6": "unknown"
        }
        mic_type = mic_types.get(choice, "unknown")
    
    # Run verification
    verifier = AG06SettingsVerifier()
    results = verifier.run_verification(mic_type)
    
    # Quick tips based on results
    print("\n💡 Quick Tips:")
    print("• Mute monitors when switching phantom power ON/OFF")
    print("• Record at 24-bit for best quality")
    print("• Leave headroom - peaks at -6dB are perfect")
    print("• Save your settings once dialed in")
    
    return 0 if results["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    exit(main())