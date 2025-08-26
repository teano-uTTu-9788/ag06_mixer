#!/bin/bash
# Stop AiOke Server

if [ -f aioke.pid ]; then
    PID=$(cat aioke.pid)
    echo "🛑 Stopping AiOke server (PID: $PID)..."
    kill $PID 2>/dev/null
    rm aioke.pid
    echo "✅ Server stopped"
else
    echo "⚠️ No PID file found. Searching for process..."
    pkill -f aioke_integrated_server
    echo "✅ All AiOke processes terminated"
fi