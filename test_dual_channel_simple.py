#!/usr/bin/env python3
"""
Simple test for the dual-channel karaoke system
Tests the basic functionality and API endpoints
"""

import asyncio
import aiohttp
import json
import time

async def test_dual_channel_system():
    """Test the dual channel karaoke system endpoints"""
    
    base_url = "http://localhost:9092"
    
    print("🎤 Testing Dual Channel Karaoke System")
    print("=" * 50)
    
    tests_passed = 0
    tests_total = 0
    
    async with aiohttp.ClientSession() as session:
        
        # Test 1: Health Check
        tests_total += 1
        print(f"Test {tests_total}: Health Check", end="... ")
        try:
            async with session.get(f"{base_url}/api/health") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"✅ PASS - {data.get('status', 'unknown')}")
                    tests_passed += 1
                else:
                    print(f"❌ FAIL - Status {resp.status}")
        except Exception as e:
            print(f"❌ FAIL - {e}")
        
        # Test 2: System Status
        tests_total += 1
        print(f"Test {tests_total}: System Status", end="... ")
        try:
            async with session.get(f"{base_url}/api/system/status") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"✅ PASS - Channels: {len(data.get('channels', []))}")
                    tests_passed += 1
                else:
                    print(f"❌ FAIL - Status {resp.status}")
        except Exception as e:
            print(f"❌ FAIL - {e}")
        
        # Test 3: Vocal Channel Status
        tests_total += 1
        print(f"Test {tests_total}: Vocal Channel Status", end="... ")
        try:
            async with session.get(f"{base_url}/api/channels/vocal/status") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"✅ PASS - Channel: {data.get('channel_type', 'unknown')}")
                    tests_passed += 1
                else:
                    print(f"❌ FAIL - Status {resp.status}")
        except Exception as e:
            print(f"❌ FAIL - {e}")
        
        # Test 4: Music Channel Status  
        tests_total += 1
        print(f"Test {tests_total}: Music Channel Status", end="... ")
        try:
            async with session.get(f"{base_url}/api/channels/music/status") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"✅ PASS - Channel: {data.get('channel_type', 'unknown')}")
                    tests_passed += 1
                else:
                    print(f"❌ FAIL - Status {resp.status}")
        except Exception as e:
            print(f"❌ FAIL - {e}")
        
        # Test 5: Update Vocal Effects
        tests_total += 1
        print(f"Test {tests_total}: Update Vocal Effects", end="... ")
        try:
            effects = {
                "reverb": {"room_size": 0.7, "dampening": 0.3, "enabled": True},
                "eq": {"presence": 2.0, "enabled": True}
            }
            async with session.post(f"{base_url}/api/channels/vocal/effects", 
                                  json=effects) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"✅ PASS - Status: {data.get('status', 'unknown')}")
                    tests_passed += 1
                else:
                    print(f"❌ FAIL - Status {resp.status}")
        except Exception as e:
            print(f"❌ FAIL - {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {tests_passed}/{tests_total} ({(tests_passed/tests_total)*100:.1f}%)")
    
    if tests_passed == tests_total:
        print("🎉 All tests passed! Dual-channel system is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check system status.")
        return False

def main():
    """Main test function"""
    print("Waiting for server to start...")
    time.sleep(3)  # Give server time to start
    
    success = asyncio.run(test_dual_channel_system())
    return success

if __name__ == '__main__':
    main()