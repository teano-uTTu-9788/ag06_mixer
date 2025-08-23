#!/bin/bash
# AG06 Mixer Modern Development Launcher

echo "🚀 Starting AG06 Mixer Modern Development Environment"
echo "=================================================="

# Kill any existing processes
pkill -f "ag06.*web" 2>/dev/null
lsof -ti:8080 | xargs kill -9 2>/dev/null
lsof -ti:3000 | xargs kill -9 2>/dev/null

# Check if npm dependencies are installed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Start the simple web interface for now
echo "🎛️  Starting AG06 Web Interface..."
python3 web_interface.py &
WEB_PID=$!

echo "✅ Web interface started (PID: $WEB_PID)"
echo ""
echo "📱 Access Points:"
echo "  - Web UI: http://localhost:8080"
echo "  - Metrics: http://localhost:9090"
echo "  - Grafana: http://localhost:3001 (admin/admin)"
echo ""
echo "Press Ctrl+C to stop all services"

# Open browser
sleep 2
open "http://localhost:8080"

# Wait for interrupt
wait $WEB_PID