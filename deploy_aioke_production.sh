#!/bin/bash
# AiOke Complete Production Deployment Script

echo "🎤 AiOke Production Deployment"
echo "=============================="
echo ""

# 1. Check prerequisites
echo "1️⃣ Checking prerequisites..."
command -v python3 >/dev/null 2>&1 || { echo "❌ Python 3 required"; exit 1; }
command -v pip3 >/dev/null 2>&1 || { echo "❌ pip3 required"; exit 1; }
echo "✅ Prerequisites satisfied"
echo ""

# 2. Install dependencies
echo "2️⃣ Installing dependencies..."
pip3 install aiohttp aiohttp-cors requests numpy --quiet
echo "✅ Dependencies installed"
echo ""

# 3. Create directory structure
echo "3️⃣ Creating directory structure..."
mkdir -p logs
mkdir -p backups
echo "✅ Directories created"
echo ""

# 4. Set permissions
echo "4️⃣ Setting permissions..."
chmod +x start_aioke.sh stop_aioke.sh status_aioke.sh test_all_features.sh
echo "✅ Permissions set"
echo ""

# 5. Configure environment
echo "5️⃣ Configuring environment..."
cat > .env << EOF
# AiOke Environment Configuration
PORT=9090
YOUTUBE_API_KEY=${YOUTUBE_API_KEY:-}
ENVIRONMENT=production
LOG_LEVEL=INFO
EOF
echo "✅ Environment configured"
echo ""

# 6. Stop any existing servers
echo "6️⃣ Stopping existing servers..."
./stop_aioke.sh 2>/dev/null || true
echo "✅ Existing servers stopped"
echo ""

# 7. Start production server
echo "7️⃣ Starting production server..."
./start_aioke.sh
echo ""

# 8. Wait for server to initialize
echo "8️⃣ Waiting for server initialization..."
sleep 5
echo ""

# 9. Run health check
echo "9️⃣ Running health check..."
if curl -s http://localhost:9090/api/health | python3 -m json.tool > /dev/null 2>&1; then
    echo "✅ Server is healthy"
else
    echo "❌ Server health check failed"
    exit 1
fi
echo ""

# 10. Run feature tests
echo "🔟 Running feature tests..."
./test_all_features.sh | tail -10
echo ""

# 11. Display access information
echo "=============================="
echo "✅ DEPLOYMENT COMPLETE"
echo "=============================="
echo ""
echo "📱 Access Information:"
echo ""
echo "Local Access:"
echo "  http://localhost:9090"
echo ""
echo "Network Access:"
IP=$(ipconfig getifaddr en0 2>/dev/null || hostname -I | awk '{print $1}' 2>/dev/null || echo "YOUR_IP")
echo "  http://$IP:9090"
echo ""
echo "iPad Installation:"
echo "  1. Open Safari on iPad"
echo "  2. Go to: http://$IP:9090"
echo "  3. Tap Share → Add to Home Screen"
echo "  4. Name it 'AiOke' and tap Add"
echo ""
echo "Management Commands:"
echo "  ./start_aioke.sh   - Start server"
echo "  ./stop_aioke.sh    - Stop server"
echo "  ./status_aioke.sh  - Check status"
echo "  ./test_all_features.sh - Test features"
echo ""
echo "Logs:"
echo "  tail -f logs/aioke.log"
echo ""
echo "🎤 Enjoy AiOke!"