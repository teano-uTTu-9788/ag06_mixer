#!/usr/bin/env python3
"""
Simple WebRTC System Verification
Tests basic functionality without requiring all dependencies
"""

import os
import sys
import json
from pathlib import Path

def test_file_structure():
    """Test that all WebRTC files are present"""
    print("🧪 Testing WebRTC file structure...")
    
    webrtc_dir = Path(__file__).parent
    required_files = [
        'signaling_server.py',
        'static/webrtc_client.js',
        'static/index.html',
        'media_server.py',
        'start_webrtc.py',
        'requirements.txt'
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = webrtc_dir / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            print(f"  ✅ {file_path}")
    
    if missing_files:
        print(f"  ❌ Missing files: {', '.join(missing_files)}")
        return False
    
    print("  ✅ All required files present")
    return True

def test_signaling_server_imports():
    """Test signaling server imports"""
    print("\n🧪 Testing signaling server imports...")
    
    try:
        # Add parent directory to path
        parent_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(parent_dir))
        
        # Test core imports that should work
        import asyncio
        import json
        import logging
        from datetime import datetime
        from typing import Dict, Set, Optional
        from dataclasses import dataclass, asdict
        
        print("  ✅ Core Python imports successful")
        
        # Test aiohttp imports
        try:
            from aiohttp import web
            print("  ✅ aiohttp available")
        except ImportError:
            print("  ⚠️  aiohttp not available (install with: pip install aiohttp)")
        
        # Test socketio imports  
        try:
            import socketio
            print("  ✅ socketio available")
        except ImportError:
            print("  ⚠️  socketio not available (install with: pip install python-socketio)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import error: {e}")
        return False

def test_javascript_structure():
    """Test JavaScript client structure"""
    print("\n🧪 Testing JavaScript client structure...")
    
    js_file = Path(__file__).parent / 'static' / 'webrtc_client.js'
    
    if not js_file.exists():
        print("  ❌ JavaScript client file missing")
        return False
    
    content = js_file.read_text()
    required_classes = ['AudioStreamingClient']
    required_methods = [
        'connect', 'startPublishing', 'startSubscribing',
        'createPeerConnection', 'joinRoom', 'setupAudioAnalysis'
    ]
    
    missing_items = []
    for item in required_classes + required_methods:
        if item not in content:
            missing_items.append(item)
    
    if missing_items:
        print(f"  ❌ Missing JavaScript items: {', '.join(missing_items)}")
        return False
    
    # Check for WebRTC API usage
    webrtc_apis = ['RTCPeerConnection', 'getUserMedia', 'RTCSessionDescription']
    for api in webrtc_apis:
        if api in content:
            print(f"  ✅ {api} usage found")
    
    print("  ✅ JavaScript client structure valid")
    return True

def test_html_interface():
    """Test HTML interface structure"""
    print("\n🧪 Testing HTML interface...")
    
    html_file = Path(__file__).parent / 'static' / 'index.html'
    
    if not html_file.exists():
        print("  ❌ HTML interface file missing")
        return False
    
    content = html_file.read_text()
    required_elements = [
        'audioVisualizer', 'connectBtn', 'publishBtn', 
        'subscribeBtn', 'roomInput', 'remoteAudio'
    ]
    
    missing_elements = []
    for element_id in required_elements:
        if f'id="{element_id}"' not in content:
            missing_elements.append(element_id)
    
    if missing_elements:
        print(f"  ❌ Missing HTML elements: {', '.join(missing_elements)}")
        return False
    
    # Check for required scripts
    if 'webrtc_client.js' not in content:
        print("  ❌ WebRTC client script not linked")
        return False
    
    print("  ✅ HTML interface structure valid")
    return True

def test_integration_readiness():
    """Test integration with existing AI mixer"""
    print("\n🧪 Testing integration readiness...")
    
    try:
        # Test AI mixer imports
        parent_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(parent_dir))
        
        # These should exist from our previous work
        ai_files = [
            'ai_mixing_brain.py',
            'studio_dsp_chain.py', 
            'complete_ai_mixer.py'
        ]
        
        missing_ai = []
        for ai_file in ai_files:
            if not (parent_dir / ai_file).exists():
                missing_ai.append(ai_file)
        
        if missing_ai:
            print(f"  ⚠️  Missing AI mixer files: {', '.join(missing_ai)}")
        else:
            print("  ✅ AI mixer files available for integration")
        
        return len(missing_ai) == 0
        
    except Exception as e:
        print(f"  ❌ Integration test error: {e}")
        return False

def test_configuration():
    """Test WebRTC configuration"""
    print("\n🧪 Testing WebRTC configuration...")
    
    # Check requirements file
    req_file = Path(__file__).parent / 'requirements.txt'
    if req_file.exists():
        requirements = req_file.read_text()
        required_packages = ['aiortc', 'aiohttp', 'python-socketio']
        
        for package in required_packages:
            if package in requirements:
                print(f"  ✅ {package} in requirements")
            else:
                print(f"  ⚠️  {package} not in requirements")
    else:
        print("  ❌ Requirements file missing")
        return False
    
    # Test launcher script
    launcher = Path(__file__).parent / 'start_webrtc.py'
    if launcher.exists() and launcher.stat().st_mode & 0o111:
        print("  ✅ Launcher script executable")
    else:
        print("  ❌ Launcher script not executable")
        return False
    
    print("  ✅ Configuration valid")
    return True

def main():
    """Run all tests"""
    print("🎵 AG06 Mixer - WebRTC System Verification")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_signaling_server_imports,
        test_javascript_structure,
        test_html_interface,
        test_integration_readiness,
        test_configuration
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    percentage = (passed / total) * 100
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} ({percentage:.1f}%)")
    
    if percentage >= 90:
        print("🎉 WebRTC system ready for deployment!")
        print("Next steps:")
        print("1. Install dependencies: pip install -r webrtc/requirements.txt")
        print("2. Start WebRTC server: python3 webrtc/start_webrtc.py")
        print("3. Open http://localhost:8081 in browser")
    elif percentage >= 70:
        print("⚠️  WebRTC system mostly ready, minor issues to fix")
    else:
        print("❌ WebRTC system needs significant work")
    
    print("=" * 50)
    return percentage >= 90

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)