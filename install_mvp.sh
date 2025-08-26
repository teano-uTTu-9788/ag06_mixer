#!/bin/bash
# Aioke - MVP Installer for Friends & Family
# One-click installation and launch

set -e  # Exit on any error

echo "🎛️  Aioke - MVP Installation"
echo "======================================"

# Check Python version
echo "🔍 Checking system requirements..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+ and try again."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "  ✅ Python $PYTHON_VERSION found"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🚀 Activating environment..."
source venv/bin/activate

# Install dependencies
echo "📦 Installing AI dependencies..."
pip install --quiet opencv-python mediapipe numpy flask

echo "✅ Installation complete!"
echo ""
echo "🎉 Aioke ready to launch! Run:"
echo "   source venv/bin/activate"
echo "   python3 launch_mvp.py"
echo ""
echo "📱 Or use the quick launcher:"
echo "   ./quick_launch.sh"