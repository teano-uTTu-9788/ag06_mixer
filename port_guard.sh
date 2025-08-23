#!/bin/bash
# Port guard - ensure clean port before starting
# Kill any process using our port to prevent conflicts

PORT=${API_PORT:-5001}

echo "🔒 Port Guard: Checking port $PORT..."

# Find and kill processes on port
PIDS=$(lsof -ti:$PORT 2>/dev/null)

if [ -n "$PIDS" ]; then
    echo "⚠️  Found processes on port $PORT: $PIDS"
    echo "🔪 Killing processes..."
    kill -9 $PIDS 2>/dev/null
    sleep 1
    echo "✅ Port $PORT cleared"
else
    echo "✅ Port $PORT is already free"
fi

# Verify port is free
if lsof -ti:$PORT >/dev/null 2>&1; then
    echo "❌ Failed to clear port $PORT"
    exit 1
fi

# Also check for rogue auto-continue processes
echo "🔍 Checking for auto-continue interference..."
if pgrep -f "auto.*continue" >/dev/null; then
    echo "⚠️  Found auto-continue processes, killing..."
    pkill -f "auto.*continue"
fi

# Clear any curl aliases/functions
echo "🔧 Clearing shell hijacks..."
unalias curl 2>/dev/null || true
unset -f curl 2>/dev/null || true
hash -r

echo "✅ Environment clean and ready"