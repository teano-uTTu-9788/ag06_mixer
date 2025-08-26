#!/bin/bash

# Vercel Deployment Script for Web UI
# Leverages your paid Vercel plan for advanced features

set -e

echo "================================================"
echo "🚀 AG06 MIXER - VERCEL UI DEPLOYMENT"
echo "================================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Check Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    print_warning "Vercel CLI not found. Installing..."
    npm i -g vercel
fi

# Get backend URL from deployment-info.json if it exists
BACKEND_URL=""
if [ -f "deployment-info.json" ]; then
    BACKEND_URL=$(cat deployment-info.json | grep backend_url | cut -d'"' -f4)
    print_status "Found backend URL: $BACKEND_URL"
else
    print_warning "No deployment-info.json found. Using default backend URL."
    BACKEND_URL="https://ag06-mixer-backend.azurecontainerapps.io"
fi

# Set environment variables for Vercel
print_status "Setting environment variables..."

vercel env add NEXT_PUBLIC_API_URL production <<< "$BACKEND_URL"
vercel env add NEXT_PUBLIC_SSE_ENDPOINT production <<< "$BACKEND_URL/api/stream"

# Deploy to Vercel
print_status "Deploying to Vercel..."

# Production deployment with your paid plan features
vercel --prod \
  --name ag06-mixer-ui \
  --yes \
  --env NEXT_PUBLIC_API_URL="$BACKEND_URL" \
  --env NEXT_PUBLIC_SSE_ENDPOINT="$BACKEND_URL/api/stream" \
  --build-env NEXT_PUBLIC_API_URL="$BACKEND_URL" \
  --build-env NEXT_PUBLIC_SSE_ENDPOINT="$BACKEND_URL/api/stream"

# Get deployment URL
DEPLOYMENT_URL=$(vercel ls --json | jq -r '.[] | select(.name=="ag06-mixer-ui") | .url' | head -1)

if [ -z "$DEPLOYMENT_URL" ]; then
    DEPLOYMENT_URL=$(vercel inspect --json | jq -r '.url')
fi

# Configure custom domain (if you have one)
# vercel domains add mixer.yourdomain.com

# Set up analytics (included in paid plan)
print_status "Configuring Vercel Analytics..."
vercel env add NEXT_PUBLIC_VERCEL_ANALYTICS_ID production <<< "auto"

echo ""
echo "================================================"
echo "✅ VERCEL DEPLOYMENT COMPLETE!"
echo "================================================"
echo ""
echo "🌐 Web UI URL: https://$DEPLOYMENT_URL"
echo ""
echo "🎯 Features enabled with your paid plan:"
echo "  • Web Analytics"
echo "  • Speed Insights"
echo "  • Enhanced performance"
echo "  • Priority support"
echo "  • Team collaboration"
echo ""
echo "📱 Test the app:"
echo "  1. Open https://$DEPLOYMENT_URL"
echo "  2. Click 'Connect to Stream'"
echo "  3. Watch real-time audio metrics"
echo ""
echo "🔧 Configuration:"
echo "  • Backend API: $BACKEND_URL"
echo "  • SSE Endpoint: $BACKEND_URL/api/stream"
echo ""
echo "📊 View analytics:"
echo "  vercel analytics"
echo ""
echo "🚀 Future deployments:"
echo "  git push → Automatic deployment"
echo ""
echo "================================================"

# Update deployment info
if [ -f "deployment-info.json" ]; then
    # Add frontend URL to existing file
    cat deployment-info.json | jq ". + {\"frontend_url\": \"https://$DEPLOYMENT_URL\"}" > deployment-info.tmp.json
    mv deployment-info.tmp.json deployment-info.json
else
    # Create new deployment info
    cat > deployment-info.json <<EOF
{
  "frontend_url": "https://$DEPLOYMENT_URL",
  "backend_url": "$BACKEND_URL",
  "deployment_date": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF
fi

print_status "Deployment info updated in deployment-info.json"