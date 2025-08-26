#!/bin/bash
# Quick tunnel setup for ChatGPT Custom GPT integration

set -euo pipefail

echo "🚀 Setting up secure tunnel for ChatGPT Custom GPT"
echo "================================================"

# Check if server is running
if ! lsof -i :8090 >/dev/null 2>&1; then
    echo "❌ Server not running on port 8090"
    echo "Start your server first:"
    echo "  ./deploy_production_secure.sh"
    exit 1
fi

echo "✅ Server running on port 8090"

# Get API token
API_TOKEN=$(grep CHATGPT_API_TOKEN .env.enterprise | cut -d= -f2 2>/dev/null || echo "")
if [[ -z "$API_TOKEN" ]]; then
    echo "❌ API token not found in .env.enterprise"
    echo "Generate environment first:"
    echo "  ./setup_hardened_enterprise.sh"
    exit 1
fi

echo "✅ API token found: ${API_TOKEN:0:20}..."

# Check for tunnel tools
TUNNEL_CMD=""
if command -v ngrok >/dev/null 2>&1; then
    TUNNEL_CMD="ngrok"
elif command -v localtunnel >/dev/null 2>&1; then
    TUNNEL_CMD="lt"
else
    echo "Installing ngrok..."
    if command -v brew >/dev/null 2>&1; then
        brew install ngrok
        TUNNEL_CMD="ngrok"
    else
        echo "Installing localtunnel..."
        npm install -g localtunnel
        TUNNEL_CMD="lt"
    fi
fi

echo "✅ Tunnel tool ready: $TUNNEL_CMD"

# Start tunnel
echo ""
echo "🌐 Starting secure tunnel..."
echo "Keep this terminal open while using Custom GPT"
echo ""

if [[ "$TUNNEL_CMD" == "ngrok" ]]; then
    echo "Starting ngrok tunnel..."
    echo "Copy the HTTPS URL for your Custom GPT configuration"
    echo ""
    ngrok http 8090
elif [[ "$TUNNEL_CMD" == "lt" ]]; then
    echo "Starting localtunnel..."
    echo "Copy the HTTPS URL for your Custom GPT configuration"
    echo ""
    npx localtunnel --port 8090
fi