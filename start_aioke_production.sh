#!/bin/bash

# Start AiOke Production Server with Google/Meta Best Practices
echo "🚀 Starting AiOke Production Server (Google/Meta Patterns)"

# Kill any existing instances
pkill -f aioke_production_google_meta.py 2>/dev/null

# Create logs directory
mkdir -p logs

# Start the server with proper error handling
python3 aioke_production_google_meta.py 2>&1 | tee -a logs/aioke_production.log &
PID=$!

echo "✅ Started with PID: $PID"

# Wait for startup
sleep 3

# Test health endpoint
if curl -s http://localhost:9090/health/live | grep -q "healthy"; then
    echo "✅ Server is healthy"
    
    # Show access URLs
    echo ""
    echo "📊 Access URLs:"
    echo "   Health: http://localhost:9090/health/live"
    echo "   Metrics: http://localhost:9090/metrics"
    echo "   Status: http://localhost:9090/api/status"
    echo "   Features: http://localhost:9090/api/features"
else
    echo "❌ Server failed to start properly"
    exit 1
fi