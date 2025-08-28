#!/usr/bin/env python3
"""
Verify Real Audio System - Ensures no mock data
Tests actual audio flow through AG06
"""

import asyncio
import aiohttp
import websockets
import json
import sounddevice as sd
import numpy as np
import time

async def verify_system():
    """Verify the system is processing real audio"""
    
    print("\n" + "="*60)
    print("REAL AUDIO SYSTEM VERIFICATION")
    print("="*60)
    
    # 1. Check AG06 hardware
    print("\n1️⃣  Hardware Check:")
    print("-" * 40)
    
    devices = sd.query_devices()
    ag06_found = False
    for device in devices:
        if 'AG06' in device['name'] or 'AG03' in device['name']:
            ag06_found = True
            print(f"✅ AG06 Found: {device['name']}")
            print(f"   Input Channels: {device['max_input_channels']}")
            break
    
    if not ag06_found:
        print("❌ AG06 not found - cannot process real audio")
        return
    
    # 2. Test server status
    print("\n2️⃣  Server Status:")
    print("-" * 40)
    
    try:
        async with aiohttp.ClientSession() as session:
            # Check status endpoint
            async with session.get('http://localhost:9099/status') as resp:
                status = await resp.json()
                
                print(f"Server Status: {status.get('status', 'unknown')}")
                print(f"Real Audio Detected: {status.get('real_audio_detected', False)}")
                print(f"Vocal Level: {status.get('vocal_level', 0):.6f}")
                print(f"Music Level: {status.get('music_level', 0):.6f}")
                print(f"No Mock Data: {status.get('no_mock_data', False)}")
                
                if not status.get('real_audio_detected'):
                    print("\n⚠️  NO REAL AUDIO DETECTED")
                    print("   • Check microphone connection to AG06")
                    print("   • Check audio routing for music input")
                    print("   • Verify gain levels on AG06")
                else:
                    print("\n✅ REAL AUDIO IS BEING PROCESSED")
                    
    except aiohttp.ClientError as e:
        print(f"❌ Server not responding: {e}")
        print("   Start server with: python3 aioke_enterprise_real.py")
        return
    
    # 3. Test WebSocket connection
    print("\n3️⃣  WebSocket Real-Time Stream:")
    print("-" * 40)
    
    try:
        async with websockets.connect('ws://localhost:8765') as websocket:
            print("Connected to WebSocket")
            print("Monitoring real-time audio metrics (5 seconds)...")
            
            end_time = time.time() + 5
            max_vocal = 0
            max_music = 0
            
            while time.time() < end_time:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    
                    if data.get('type') == 'audio_metrics':
                        metrics = data['data']
                        vocal = metrics.get('vocal_level', 0)
                        music = metrics.get('music_level', 0)
                        
                        max_vocal = max(max_vocal, vocal)
                        max_music = max(max_music, music)
                        
                        # Show real-time status
                        if vocal > 0.001 or music > 0.001:
                            print(f"  📊 Real Audio: Vocal={vocal:.4f}, Music={music:.4f}")
                        else:
                            print(f"  ⚠️  No signal: Vocal={vocal:.6f}, Music={music:.6f}")
                            
                except asyncio.TimeoutError:
                    print("  ⏳ Waiting for audio data...")
            
            print(f"\nMax Levels Detected:")
            print(f"  Vocal: {max_vocal:.6f}")
            print(f"  Music: {max_music:.6f}")
            
            if max_vocal < 0.001 and max_music < 0.001:
                print("\n❌ NO REAL AUDIO DETECTED DURING TEST")
            else:
                print("\n✅ REAL AUDIO CONFIRMED")
                
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    
    # 4. Direct audio test
    print("\n4️⃣  Direct AG06 Test (3 seconds):")
    print("-" * 40)
    print("Make noise or play music NOW...")
    
    try:
        recording = sd.rec(int(3 * 44100), samplerate=44100, channels=2, device='AG06/AG03')
        
        for i in range(3):
            await asyncio.sleep(1)
            print(f"  Recording... {i+1}/3")
        
        sd.wait()
        
        # Analyze recording
        ch1_max = np.max(np.abs(recording[:, 0]))
        ch2_max = np.max(np.abs(recording[:, 1]))
        
        print(f"\nDirect Recording Results:")
        print(f"  Channel 1 (Vocal): {ch1_max:.6f}")
        print(f"  Channel 2 (Music): {ch2_max:.6f}")
        
        if ch1_max < 0.001 and ch2_max < 0.001:
            print("  ❌ No audio captured from AG06")
        else:
            print("  ✅ Audio captured successfully")
            
    except Exception as e:
        print(f"❌ Recording error: {e}")
    
    # 5. Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    print("\nSystem Components:")
    print(f"  AG06 Hardware: {'✅ Connected' if ag06_found else '❌ Not Found'}")
    print(f"  HTTP Server: {'✅ Running' if 'status' in locals() else '❌ Not Running'}")
    print(f"  WebSocket: {'✅ Streaming' if 'max_vocal' in locals() else '❌ Not Streaming'}")
    
    print("\nAudio Signal Status:")
    if 'status' in locals():
        if status.get('real_audio_detected'):
            print("  ✅ PROCESSING REAL AUDIO")
        else:
            print("  ❌ NO AUDIO SIGNAL DETECTED")
            print("\n  Troubleshooting:")
            print("  1. Check mic is connected to AG06 and gain is up")
            print("  2. Route system audio through AG06 or BlackHole")
            print("  3. Verify AG06 USB connection is secure")
    
    print("\n" + "="*60)
    print("END OF VERIFICATION")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(verify_system())