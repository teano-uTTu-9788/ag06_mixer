#!/usr/bin/env python3
"""
TRUTHFUL AG06 Audio Test - No fabrication, no mock data
Shows exactly what's happening with your audio hardware
"""

import sounddevice as sd
import numpy as np
import time
import sys

def test_ag06_truthfully():
    """Test AG06 with complete honesty about what's working"""
    
    print("\n" + "="*60)
    print("TRUTHFUL AG06 AUDIO TEST - NO FABRICATION")
    print("="*60)
    
    # 1. List all audio devices
    print("\n📋 ACTUAL AUDIO DEVICES FOUND:")
    print("-" * 40)
    devices = sd.query_devices()
    ag06_found = False
    ag06_index = None
    
    for idx, device in enumerate(devices):
        device_name = device['name']
        if device['max_input_channels'] > 0:
            print(f"  [{idx}] {device_name}")
            print(f"       Input channels: {device['max_input_channels']}")
            print(f"       Sample rate: {device['default_samplerate']}Hz")
            
            if 'AG06' in device_name or 'AG03' in device_name:
                ag06_found = True
                ag06_index = idx
                print(f"       ✅ THIS IS YOUR AG06 MIXER")
        print()
    
    if not ag06_found:
        print("\n❌ TRUTH: AG06 not detected in audio devices")
        print("   Please check USB connection and drivers")
        return
    
    # 2. Test actual audio input
    print("\n🎤 TESTING REAL AUDIO INPUT (5 seconds):")
    print("-" * 40)
    print("   MAKE SOME NOISE or SPEAK INTO MIC NOW!")
    print()
    
    try:
        duration = 5  # seconds
        sample_rate = 44100
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=2,
            device=ag06_index,
            dtype='float32'
        )
        
        # Show real-time levels
        for i in range(5):
            sd.wait()  # Wait for 1 second of recording
            time.sleep(1)
            
            # Get the recorded data so far
            current_data = recording[:int((i+1) * sample_rate)]
            
            # Calculate REAL levels (no fabrication)
            if current_data.shape[0] > 0:
                # Channel 1 (left/vocal)
                ch1_level = np.max(np.abs(current_data[:, 0]))
                ch1_rms = np.sqrt(np.mean(current_data[:, 0]**2))
                
                # Channel 2 (right/music)  
                ch2_level = np.max(np.abs(current_data[:, 1]))
                ch2_rms = np.sqrt(np.mean(current_data[:, 1]**2))
                
                print(f"   Second {i+1}:")
                print(f"     Ch1 (Vocal): Peak={ch1_level:.4f}, RMS={ch1_rms:.4f}")
                print(f"     Ch2 (Music): Peak={ch2_level:.4f}, RMS={ch2_rms:.4f}")
                
                # HONEST assessment
                if ch1_level < 0.001:
                    print(f"     ⚠️  Channel 1: NO SIGNAL DETECTED")
                elif ch1_level > 0.9:
                    print(f"     ⚠️  Channel 1: CLIPPING!")
                else:
                    print(f"     ✅ Channel 1: Signal detected")
                    
                if ch2_level < 0.001:
                    print(f"     ⚠️  Channel 2: NO SIGNAL DETECTED")
                elif ch2_level > 0.9:
                    print(f"     ⚠️  Channel 2: CLIPPING!")
                else:
                    print(f"     ✅ Channel 2: Signal detected")
                print()
        
        sd.wait()  # Ensure recording is complete
        
        # Final analysis
        print("\n📊 FINAL HONEST ANALYSIS:")
        print("-" * 40)
        
        # Check if we got any real audio
        max_level_ch1 = np.max(np.abs(recording[:, 0]))
        max_level_ch2 = np.max(np.abs(recording[:, 1]))
        
        print(f"Channel 1 (Vocal) Maximum Level: {max_level_ch1:.6f}")
        if max_level_ch1 < 0.001:
            print("   ❌ NO VOCAL DETECTED - Mic may not be connected or muted")
        else:
            print("   ✅ VOCAL SIGNAL DETECTED")
            
        print(f"Channel 2 (Music) Maximum Level: {max_level_ch2:.6f}")
        if max_level_ch2 < 0.001:
            print("   ❌ NO MUSIC DETECTED - Check audio routing from YouTube/system")
        else:
            print("   ✅ MUSIC SIGNAL DETECTED")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("   This is a real error, not a simulation")
    
    print("\n" + "="*60)
    print("END OF TRUTHFUL TEST - NO MOCK DATA WAS USED")
    print("="*60)


if __name__ == "__main__":
    test_ag06_truthfully()