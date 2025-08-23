#!/usr/bin/env python3
"""
Test AG06 audio on macOS using system commands
"""

import subprocess
import time
import os

def test_audio_devices():
    """Test and list audio devices using macOS system commands"""
    print("🔊 AG06 Audio Device Test (macOS)")
    print("=" * 50)
    
    # List audio devices using system_profiler
    print("\n📋 Checking Audio Devices:")
    result = subprocess.run(
        ["system_profiler", "SPAudioDataType"],
        capture_output=True,
        text=True
    )
    
    ag06_found = False
    if "AG06" in result.stdout or "AG03" in result.stdout:
        ag06_found = True
        print("✅ AG06 DETECTED in system!")
    
    # Parse output to show key devices
    lines = result.stdout.split('\n')
    for i, line in enumerate(lines):
        if 'AG06' in line or 'AG03' in line:
            print(f"  ✅ {line.strip()}")
            # Show next few lines for context
            for j in range(1, 4):
                if i+j < len(lines):
                    print(f"     {lines[i+j].strip()}")
    
    if not ag06_found:
        print("❌ AG06 not found in system audio devices")
        print("\n📝 Available audio devices:")
        for line in lines:
            if "Device:" in line or "Name:" in line:
                print(f"  {line.strip()}")
    
    return ag06_found

def test_audio_output():
    """Test audio output using macOS 'say' command"""
    print("\n🔊 Testing Audio Output...")
    print("  Playing test sound through default output device...")
    
    # Use macOS 'say' command for simple audio test
    subprocess.run(["say", "-v", "Samantha", "Testing AG06 audio output. This is a karaoke mixer test."])
    print("  ✅ Audio output test complete")
    
    # Also play a system sound
    print("  Playing system beep...")
    subprocess.run(["afplay", "/System/Library/Sounds/Ping.aiff"])
    print("  ✅ System sound played")

def test_input_devices():
    """Check available input devices"""
    print("\n🎤 Checking Input Devices:")
    
    # Use SwitchAudioSource if available
    try:
        result = subprocess.run(
            ["SwitchAudioSource", "-a", "-t", "input"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'AG06' in line or 'AG03' in line:
                    print(f"  ✅ {line} - AVAILABLE FOR INPUT")
                else:
                    print(f"  • {line}")
        else:
            print("  ℹ️  SwitchAudioSource not available, using system_profiler")
            test_with_system_profiler()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("  ℹ️  SwitchAudioSource not available, using system_profiler")
        test_with_system_profiler()

def test_with_system_profiler():
    """Fallback method using system_profiler"""
    result = subprocess.run(
        ["system_profiler", "SPAudioDataType", "-detailLevel", "full"],
        capture_output=True,
        text=True
    )
    
    if "Input Devices" in result.stdout:
        print("  Input devices found in system")
        if "AG06" in result.stdout or "AG03" in result.stdout:
            print("  ✅ AG06 available as input device")

def generate_test_tone():
    """Generate a test tone using sox if available"""
    print("\n🎵 Generating Test Tone:")
    
    # Check if sox is installed
    try:
        subprocess.run(["which", "sox"], check=True, capture_output=True)
        print("  Generating 440Hz tone for 1 second...")
        subprocess.run([
            "sox", "-n", "-d", "synth", "1", "sine", "440",
            "vol", "0.3"
        ])
        print("  ✅ Test tone played")
    except:
        print("  ℹ️  Sox not installed, using system sounds instead")
        subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"])
        print("  ✅ Alternative sound played")

def main():
    print("🎵 AG06 Audio System Test for macOS")
    print("=" * 50)
    
    # Test 1: Device detection
    ag06_found = test_audio_devices()
    
    if not ag06_found:
        print("\n⚠️  AG06 not detected as audio device")
        print("\n📝 Troubleshooting:")
        print("  1. Check USB connection to AG06")
        print("  2. Check System Preferences > Sound")
        print("  3. Make sure AG06 is powered on")
        print("  4. Try unplugging and reconnecting USB")
    
    # Test 2: Output test
    print("\n" + "=" * 50)
    test_audio_output()
    
    # Test 3: Input device check
    print("\n" + "=" * 50)
    test_input_devices()
    
    # Test 4: Generate test tone
    print("\n" + "=" * 50)
    generate_test_tone()
    
    print("\n" + "=" * 50)
    if ag06_found:
        print("✅ AG06 detected in system!")
        print("🎤 Ready for karaoke with AI auto-mixing!")
    else:
        print("⚠️  AG06 not detected - please check connections")
    
    print("\n📝 Next Steps:")
    print("  1. If AG06 is connected, select it in System Preferences > Sound")
    print("  2. Run the AI karaoke mixer: python3 ai_karaoke_mixer.py")
    print("  3. Start singing and enjoy auto-tuned, enhanced vocals!")

if __name__ == "__main__":
    main()