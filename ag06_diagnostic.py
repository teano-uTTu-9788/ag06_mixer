#!/usr/bin/env python3
"""
AG06 Diagnostic Tool - Complete truth about audio routing
No fabrication, shows exactly what's happening
"""

import subprocess
import sounddevice as sd
import numpy as np
import time

def run_diagnostic():
    """Run complete AG06 diagnostic"""
    
    print("\n" + "="*70)
    print("AG06 DIAGNOSTIC - TRUTHFUL ASSESSMENT")
    print("="*70)
    
    # 1. Check system audio settings
    print("\n1️⃣  SYSTEM AUDIO CONFIGURATION:")
    print("-" * 40)
    
    # Get default input device
    try:
        default_input = sd.query_devices(kind='input')
        print(f"Default Input Device: {default_input['name']}")
        print(f"  Channels: {default_input['max_input_channels']}")
    except:
        print("❌ Could not determine default input device")
    
    # Get default output device
    try:
        default_output = sd.query_devices(kind='output')
        print(f"Default Output Device: {default_output['name']}")
        print(f"  Channels: {default_output['max_output_channels']}")
    except:
        print("❌ Could not determine default output device")
    
    # 2. Check AG06 specific settings
    print("\n2️⃣  AG06 MIXER STATUS:")
    print("-" * 40)
    
    devices = sd.query_devices()
    ag06_info = None
    for device in devices:
        if 'AG06' in device['name'] or 'AG03' in device['name']:
            ag06_info = device
            break
    
    if ag06_info:
        print(f"✅ AG06 Found: {ag06_info['name']}")
        print(f"   Device Index: {ag06_info['index']}")
        print(f"   Input Channels: {ag06_info['max_input_channels']}")
        print(f"   Output Channels: {ag06_info['max_output_channels']}")
        print(f"   Sample Rate: {ag06_info['default_samplerate']}Hz")
        print(f"   Low Latency: {ag06_info['default_low_input_latency']:.3f}s")
        
        # Test if AG06 is actually receiving audio
        print("\n3️⃣  TESTING AG06 INPUT (checking for signal):")
        print("-" * 40)
        
        try:
            # Record a short sample
            duration = 2
            print(f"   Recording {duration} seconds from AG06...")
            recording = sd.rec(
                int(duration * 44100),
                samplerate=44100,
                channels=2,
                device=ag06_info['index'],
                dtype='float32'
            )
            sd.wait()
            
            # Analyze what we got
            ch1_max = np.max(np.abs(recording[:, 0]))
            ch2_max = np.max(np.abs(recording[:, 1]))
            ch1_mean = np.mean(np.abs(recording[:, 0]))
            ch2_mean = np.mean(np.abs(recording[:, 1]))
            
            print(f"\n   📊 ACTUAL RECORDED LEVELS:")
            print(f"   Channel 1 (Left/Vocal):")
            print(f"     Max amplitude: {ch1_max:.6f}")
            print(f"     Mean amplitude: {ch1_mean:.6f}")
            if ch1_max < 0.001:
                print(f"     ❌ NO SIGNAL - Check mic connection and gain knob")
            else:
                print(f"     ✅ Signal detected")
            
            print(f"   Channel 2 (Right/Music):")  
            print(f"     Max amplitude: {ch2_max:.6f}")
            print(f"     Mean amplitude: {ch2_mean:.6f}")
            if ch2_max < 0.001:
                print(f"     ❌ NO SIGNAL - Check audio routing to AG06")
            else:
                print(f"     ✅ Signal detected")
                
        except Exception as e:
            print(f"   ❌ Recording failed: {e}")
    else:
        print("❌ AG06 not found in audio devices")
    
    # 3. Check audio routing
    print("\n4️⃣  AUDIO ROUTING CHECK:")
    print("-" * 40)
    
    # Check if AG06 is set as system output
    result = subprocess.run(
        ['system_profiler', 'SPAudioDataType'],
        capture_output=True,
        text=True
    )
    
    if 'Default Output Device: Yes' in result.stdout and 'AG06' in result.stdout:
        print("✅ AG06 is set as default output device")
        print("   ⚠️  This means system audio goes TO the AG06")
        print("   ⚠️  You need to route it back as input for processing")
    else:
        print("❌ AG06 is NOT the default output device")
    
    # 4. Common issues and solutions
    print("\n5️⃣  TROUBLESHOOTING GUIDE:")
    print("-" * 40)
    print("If NO SIGNAL on Channel 1 (Vocal):")
    print("  • Check microphone is connected to AG06 input")
    print("  • Turn up GAIN knob for that channel")
    print("  • Check channel isn't muted on AG06")
    print("  • Verify mic needs phantom power (+48V) if condenser")
    
    print("\nIf NO SIGNAL on Channel 2 (Music/YouTube):")
    print("  • Set AG06 as system OUTPUT in System Settings > Sound")
    print("  • Use Audio MIDI Setup to create aggregate device")
    print("  • Or use BlackHole/Loopback to route audio")
    print("  • Check AG06 USB mode switch (if present)")
    
    print("\n" + "="*70)
    print("END OF DIAGNOSTIC - ALL INFORMATION IS TRUTHFUL")
    print("="*70)


if __name__ == "__main__":
    run_diagnostic()