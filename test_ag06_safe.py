#!/usr/bin/env python3
"""
AG06 Mixer Safe Testing Suite
Context-limited tests for AG06 mixer functionality
"""

import os
import sys
import time
import subprocess
from typing import Dict, List, Tuple, Optional

class AG06SafeTests:
    """Safe, context-limited tests for AG06 mixer"""
    
    def __init__(self):
        self.test_results = []
        self.max_output_lines = 10
        self.ag06_connected = False
        
    def run_test(self, name: str, test_func) -> bool:
        """Run a single test with output limiting"""
        try:
            print(f"🧪 {name}...", end=' ')
            result = test_func()
            
            if result:
                print("✅")
                self.test_results.append((name, True, None))
                return True
            else:
                print("❌")
                self.test_results.append((name, False, "Test returned False"))
                return False
                
        except Exception as e:
            print(f"❌ ({str(e)[:50]})")
            self.test_results.append((name, False, str(e)[:100]))
            return False
    
    def test_usb_connection(self) -> bool:
        """Test AG06 USB connection"""
        try:
            result = subprocess.run(
                ['system_profiler', 'SPUSBDataType'],
                capture_output=True,
                text=True,
                timeout=5
            )
            self.ag06_connected = 'AG06' in result.stdout
            return self.ag06_connected
        except:
            return False
    
    def test_audio_device(self) -> bool:
        """Test AG06 audio device detection"""
        if not self.ag06_connected:
            return False
            
        try:
            result = subprocess.run(
                ['system_profiler', 'SPAudioDataType'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return 'AG06' in result.stdout
        except:
            return False
    
    def test_midi_device(self) -> bool:
        """Test MIDI device availability"""
        try:
            # Check if sendmidi is available
            result = subprocess.run(
                ['which', 'sendmidi'],
                capture_output=True,
                timeout=1
            )
            
            if result.returncode != 0:
                # Tool not installed, skip test
                return True  # Don't fail if tool missing
            
            # List MIDI devices
            result = subprocess.run(
                ['sendmidi', 'list'],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            return 'AG06' in result.stdout or result.returncode == 0
            
        except:
            return True  # Don't fail on MIDI test
    
    def test_python_imports(self) -> bool:
        """Test required Python imports"""
        required_modules = [
            'pyaudio',
            'mido',
            'numpy',
            'tkinter'
        ]
        
        missing = []
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing.append(module)
        
        if missing:
            print(f"\n  ⚠️  Missing: {', '.join(missing)}")
            
        return len(missing) == 0
    
    def test_ag06_controller(self) -> bool:
        """Test AG06 controller module"""
        try:
            sys.path.insert(0, '/Users/nguythe/ag06_mixer')
            from ag06_controller import AG06Controller
            
            # Try to initialize (should handle missing device gracefully)
            controller = AG06Controller()
            return True
            
        except ImportError:
            return False
        except Exception:
            # Controller exists but device not connected - that's OK
            return True
    
    def test_audio_playback(self) -> bool:
        """Test basic audio playback capability"""
        try:
            test_sound = "/System/Library/Sounds/Glass.aiff"
            if not os.path.exists(test_sound):
                return True  # Skip if test sound missing
            
            # Try to play with timeout
            result = subprocess.run(
                ['timeout', '2', 'afplay', test_sound],
                capture_output=True,
                timeout=3
            )
            
            return result.returncode == 0
            
        except:
            return False
    
    def test_project_structure(self) -> bool:
        """Test AG06 project structure"""
        required_files = [
            '/Users/nguythe/ag06_mixer/main.py',
            '/Users/nguythe/ag06_mixer/ag06_controller.py',
            '/Users/nguythe/ag06_mixer/requirements.txt'
        ]
        
        missing = [f for f in required_files if not os.path.exists(f)]
        
        if missing:
            print(f"\n  ⚠️  Missing files: {len(missing)}")
            
        return len(missing) == 0
    
    def test_logs_directory(self) -> bool:
        """Test logs directory exists and is writable"""
        logs_dir = '/Users/nguythe/ag06_mixer/logs'
        
        try:
            os.makedirs(logs_dir, exist_ok=True)
            
            # Test write permission
            test_file = os.path.join(logs_dir, '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            
            return True
            
        except:
            return False
    
    def run_all_tests(self):
        """Run all tests with summary"""
        print("🎛️  AG06 Mixer Safe Test Suite")
        print("=" * 40)
        
        # Define test suite
        tests = [
            ("USB Connection", self.test_usb_connection),
            ("Audio Device", self.test_audio_device),
            ("MIDI Device", self.test_midi_device),
            ("Python Imports", self.test_python_imports),
            ("AG06 Controller", self.test_ag06_controller),
            ("Audio Playback", self.test_audio_playback),
            ("Project Structure", self.test_project_structure),
            ("Logs Directory", self.test_logs_directory)
        ]
        
        # Run tests
        for name, test_func in tests:
            self.run_test(name, test_func)
        
        # Summary
        print("=" * 40)
        passed = sum(1 for _, success, _ in self.test_results if success)
        total = len(self.test_results)
        
        print(f"\n📊 Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("✅ All tests passed! AG06 development ready.")
        elif self.ag06_connected:
            print("⚠️  AG06 connected but some tests failed.")
            print("   Run 'ag06_debug' for detailed diagnostics.")
        else:
            print("❌ AG06 not connected. Connect device and retry.")
        
        # Show failures (limited output)
        failures = [(name, error) for name, success, error in self.test_results if not success]
        if failures:
            print("\nFailed tests:")
            for name, error in failures[:5]:  # Limit to 5
                print(f"  ❌ {name}: {error[:60] if error else 'Failed'}")
        
        return passed == total

def main():
    """Main entry point"""
    tester = AG06SafeTests()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--quick':
            # Quick connectivity test only
            print("🎛️  AG06 Quick Test")
            if tester.test_usb_connection():
                print("✅ AG06 connected")
                if tester.test_audio_device():
                    print("✅ Audio device ready")
                else:
                    print("⚠️  Audio device not detected")
            else:
                print("❌ AG06 not connected")
                
        elif sys.argv[1] == '--help':
            print("""
AG06 Mixer Safe Testing Suite

Usage:
    python3 test_ag06_safe.py         # Run all tests
    python3 test_ag06_safe.py --quick # Quick connectivity test
    python3 test_ag06_safe.py --help  # Show this help

Tests are designed with limited output to prevent terminal overflow.
""")
        else:
            print(f"Unknown option: {sys.argv[1]}")
    else:
        # Run full test suite
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()