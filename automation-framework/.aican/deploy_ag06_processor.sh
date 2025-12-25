#!/bin/bash
# AG06 Audio Processor Deployment Script
# Optimized workflow implementation

echo "🎛️ AG06 AUDIO PROCESSOR DEPLOYMENT"
echo "================================="

# Check Python version
python_version=$(python3 --version 2>&1)
echo "Python version: $python_version"

# Install required packages
echo "📦 Installing required packages..."
pip3 install flask flask-socketio sounddevice numpy scipy librosa aubio pyaudio

# Install system dependencies (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 Installing macOS audio dependencies..."
    brew install portaudio sox
    pip3 install pyobjc-core pyobjc-framework-CoreAudio
fi

# Create necessary directories
mkdir -p templates static logs

# Set permissions
chmod +x optimized_ag06_flask_app.py

# Check AG06 device
echo "🔍 Checking for AG06 device..."
python3 -c "
import sounddevice as sd
devices = sd.query_devices()
ag06_found = False
for i, device in enumerate(devices):
    if 'ag06' in device['name'].lower() or 'ag03' in device['name'].lower():
        print(f'✅ Found AG06: {device["name"]} (Device {i})')
        ag06_found = True
        break
if not ag06_found:
    print('❌ AG06 device not found')
    print('Available input devices:')
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f'  {i}: {device["name"]}')
"

echo "🚀 Starting AG06 Audio Processor..."
python3 optimized_ag06_flask_app.py
