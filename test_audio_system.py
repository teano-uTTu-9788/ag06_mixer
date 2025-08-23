#!/usr/bin/env python3
"""
Test audio input/output functionality with AG06
"""

import pyaudio
import numpy as np
import time
import sys

def test_audio_devices():
    """Test and list audio devices"""
    p = pyaudio.PyAudio()
    
    print("🔊 Audio Device Test")
    print("=" * 50)
    
    # Find AG06
    ag06_input = None
    ag06_output = None
    
    print("\n📋 Available Audio Devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        device_type = []
        if info['maxInputChannels'] > 0:
            device_type.append("INPUT")
        if info['maxOutputChannels'] > 0:
            device_type.append("OUTPUT")
        
        print(f"  [{i}] {info['name']} - {', '.join(device_type)}")
        
        if 'AG06' in info['name'] or 'AG03' in info['name']:
            if info['maxInputChannels'] > 0:
                ag06_input = i
            if info['maxOutputChannels'] > 0:
                ag06_output = i
            print(f"       ✅ AG06 DETECTED!")
    
    print(f"\n🎛️ AG06 Status:")
    if ag06_input is not None:
        print(f"  ✅ Input device found (index: {ag06_input})")
    else:
        print(f"  ❌ Input device NOT found")
    
    if ag06_output is not None:
        print(f"  ✅ Output device found (index: {ag06_output})")
    else:
        print(f"  ❌ Output device NOT found")
    
    p.terminate()
    return ag06_input, ag06_output

def test_audio_passthrough(duration=5):
    """Test audio passthrough (input -> output)"""
    p = pyaudio.PyAudio()
    
    # Find AG06
    ag06_input = None
    ag06_output = None
    
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if 'AG06' in info['name'] or 'AG03' in info['name']:
            if info['maxInputChannels'] > 0:
                ag06_input = i
            if info['maxOutputChannels'] > 0:
                ag06_output = i
    
    if ag06_input is None or ag06_output is None:
        print("❌ AG06 not found for passthrough test")
        p.terminate()
        return False
    
    print(f"\n🎤 Testing Audio Passthrough for {duration} seconds...")
    print("  Speak into your microphone - you should hear yourself")
    
    try:
        # Open stream
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=2,
            rate=44100,
            input=True,
            output=True,
            input_device_index=ag06_input,
            output_device_index=ag06_output,
            frames_per_buffer=512
        )
        
        start_time = time.time()
        max_level = 0
        
        while time.time() - start_time < duration:
            # Read from input
            data = stream.read(512, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.float32)
            
            # Calculate level
            level = np.max(np.abs(audio_data))
            max_level = max(max_level, level)
            
            # Visual level meter
            meter_length = int(level * 50)
            meter = "█" * meter_length + "░" * (50 - meter_length)
            sys.stdout.write(f"\r  Level: [{meter}] {level:.3f}")
            sys.stdout.flush()
            
            # Write to output
            stream.write(data)
        
        stream.stop_stream()
        stream.close()
        
        print(f"\n  ✅ Passthrough test complete")
        print(f"  📊 Max level detected: {max_level:.3f}")
        
        if max_level > 0.01:
            print("  ✅ Audio input is working!")
        else:
            print("  ⚠️  No audio detected - check your microphone")
        
        return True
        
    except Exception as e:
        print(f"\n  ❌ Error during passthrough: {e}")
        return False
    finally:
        p.terminate()

def test_tone_generation():
    """Generate test tones to verify output"""
    p = pyaudio.PyAudio()
    
    # Find AG06 output
    ag06_output = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if ('AG06' in info['name'] or 'AG03' in info['name']) and info['maxOutputChannels'] > 0:
            ag06_output = i
            break
    
    if ag06_output is None:
        print("❌ AG06 output not found")
        p.terminate()
        return False
    
    print("\n🔊 Generating Test Tones...")
    
    try:
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=2,
            rate=44100,
            output=True,
            output_device_index=ag06_output,
            frames_per_buffer=512
        )
        
        # Generate tones
        frequencies = [440, 880, 440]  # A4, A5, A4
        names = ["A4 (440Hz)", "A5 (880Hz)", "A4 (440Hz)"]
        
        for freq, name in zip(frequencies, names):
            print(f"  Playing {name}...")
            
            # Generate sine wave
            duration = 0.5
            sample_rate = 44100
            t = np.linspace(0, duration, int(sample_rate * duration))
            tone = np.sin(2 * np.pi * freq * t) * 0.3  # 30% volume
            
            # Convert to stereo
            stereo = np.column_stack([tone, tone])
            data = stereo.flatten().astype(np.float32).tobytes()
            
            # Play tone
            stream.write(data)
            time.sleep(0.2)
        
        stream.stop_stream()
        stream.close()
        
        print("  ✅ Test tones complete")
        return True
        
    except Exception as e:
        print(f"  ❌ Error generating tones: {e}")
        return False
    finally:
        p.terminate()

def main():
    print("🎵 AG06 Audio System Test")
    print("=" * 50)
    
    # Test 1: Device detection
    input_idx, output_idx = test_audio_devices()
    
    if input_idx is None or output_idx is None:
        print("\n❌ AG06 not properly connected")
        print("\n📝 Troubleshooting:")
        print("  1. Check USB connection")
        print("  2. Check System Preferences > Sound")
        print("  3. Restart the AG06 device")
        return
    
    # Test 2: Tone generation
    print("\n" + "=" * 50)
    test_tone_generation()
    
    # Test 3: Audio passthrough
    print("\n" + "=" * 50)
    test_audio_passthrough(duration=5)
    
    print("\n" + "=" * 50)
    print("✅ Audio system test complete!")
    print("\n🎤 Ready for karaoke!")

if __name__ == "__main__":
    main()