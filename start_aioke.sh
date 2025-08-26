#!/bin/bash
# AiOke Production Startup Script

# Set environment variables
export PORT=9090
export YOUTUBE_API_KEY="${YOUTUBE_API_KEY:-demo_mode}"
export PYTHONUNBUFFERED=1

# Create logs directory
mkdir -p logs

# Kill any existing servers
pkill -f aioke_integrated_server 2>/dev/null
pkill -f "port.*9090" 2>/dev/null
sleep 2

# Start server with logging
echo "<¤ Starting AiOke Production Server..."
echo "=Í Port: $PORT"
echo "= YouTube API: ${YOUTUBE_API_KEY:0:10}..."
echo "=Ý Logs: logs/aioke.log"

python3 aioke_integrated_server.py > logs/aioke.log 2>&1 &
PID=$!
echo " Started with PID: $PID"

# Save PID for management
echo $PID > aioke.pid

# Wait and check health
sleep 3
echo ""
echo "= Checking server health..."
curl -s http://localhost:$PORT/api/health | python3 -m json.tool || echo "  Server not ready yet"
echo ""
echo "=ñ Access URLs:"
echo "   Local: http://localhost:$PORT/aioke_enhanced_interface.html"
echo "   iPad:  http://$(ipconfig getifaddr en0 2>/dev/null || echo YOUR_IP):$PORT/aioke_enhanced_interface.html"