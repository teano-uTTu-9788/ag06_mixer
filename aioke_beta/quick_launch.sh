#!/bin/bash
# Aioke - Quick Launch Script
# For friends and family testing

echo "🚀 Launching Aioke MVP..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Environment activated"
else
    echo "⚠️  Virtual environment not found. Run ./install_mvp.sh first"
    exit 1
fi

# Launch MVP
echo "🎛️  Starting Aioke..."
python3 launch_mvp.py