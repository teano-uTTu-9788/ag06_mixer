#!/bin/bash

echo "🎚️ AiOke Real-Time Audio Mixer Launcher"
echo "======================================="
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python $PYTHON_VERSION detected"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "📦 Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet numpy scipy sounddevice flask flask-cors websockets

echo ""
echo "🚀 Starting AiOke Real-Time Mixer..."
echo ""
echo "📌 Open your browser to: http://localhost:8080"
echo "📌 WebSocket control: ws://localhost:8765"
echo ""
echo "Press Ctrl+C to stop the mixer"
echo ""

# Run the mixer server
python3 mixer_server.py