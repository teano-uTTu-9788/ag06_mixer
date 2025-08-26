#!/bin/bash
# Check AiOke Server Status

echo "🎤 AiOke System Status"
echo "=" * 40

# Check process
if [ -f aioke.pid ]; then
    PID=$(cat aioke.pid)
    if ps -p $PID > /dev/null; then
        echo "✅ Server running (PID: $PID)"
    else
        echo "❌ Server not running (stale PID file)"
    fi
else
    echo "❌ No PID file found"
fi

# Check port
if lsof -ti:9090 > /dev/null; then
    echo "✅ Port 9090 is active"
else
    echo "❌ Port 9090 is not listening"
fi

# Check health
echo ""
echo "🔍 Health Check:"
curl -s http://localhost:9090/api/health 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f'  Status: {data[\"status\"]}')
    print(f'  Uptime: {data[\"uptime\"]}')
    print(f'  Songs Played: {data[\"songs_played\"]}')
    print(f'  YouTube API: {\"Enabled\" if data[\"youtube_api\"] else \"Demo Mode\"}')
except:
    print('  ❌ Server not responding')
"

# Check logs
if [ -f logs/aioke.log ]; then
    echo ""
    echo "📝 Recent Logs:"
    tail -5 logs/aioke.log
fi

echo ""
echo "📱 Access URLs:"
echo "   Local: http://localhost:9090/aioke_enhanced_interface.html"
echo "   iPad:  http://$(ipconfig getifaddr en0 2>/dev/null || echo YOUR_IP):9090/aioke_enhanced_interface.html"