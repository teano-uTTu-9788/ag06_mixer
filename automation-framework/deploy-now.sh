#!/bin/bash

# Direct Azure deployment (bypassing auto-continue)
# Using Azure Cloud Build instead of local Docker

set -e

echo "🚀 Starting Azure deployment..."

# Direct path to Azure CLI
AZ_CLI="/opt/homebrew/bin/az"
if [ ! -f "$AZ_CLI" ]; then
    AZ_CLI="/usr/local/bin/az"
fi

# Configuration
RESOURCE_GROUP="ag06-mixer-rg"
LOCATION="eastus"
ACR_NAME="ag06mixeracr"
CONTAINER_APP_NAME="ag06-mixer-backend"
CONTAINER_APP_ENV="ag06-mixer-env"

echo "✅ Using Azure CLI at: $AZ_CLI"

# Check if already logged in
if ! $AZ_CLI account show &>/dev/null; then
    echo "🔐 Please login to Azure..."
    $AZ_CLI login --use-device-code
fi

echo "✅ Azure login successful"

# Create resource group
echo "📦 Creating resource group..."
$AZ_CLI group create --name $RESOURCE_GROUP --location $LOCATION --output table

# Create Azure Container Registry
echo "🐳 Creating Azure Container Registry..."
$AZ_CLI acr create \
    --resource-group $RESOURCE_GROUP \
    --name $ACR_NAME \
    --sku Basic \
    --admin-enabled true \
    --output table

# Get ACR credentials
echo "🔑 Getting ACR credentials..."
ACR_USERNAME=$($AZ_CLI acr credential show --name $ACR_NAME --query username -o tsv)
ACR_PASSWORD=$($AZ_CLI acr credential show --name $ACR_NAME --query passwords[0].value -o tsv)
ACR_LOGIN_SERVER=$($AZ_CLI acr show --name $ACR_NAME --query loginServer -o tsv)

echo "✅ ACR created: $ACR_LOGIN_SERVER"

# Build and push using Azure Container Registry Tasks (no local Docker needed!)
echo "🏗️  Building and pushing with ACR Tasks..."
$AZ_CLI acr build \
    --registry $ACR_NAME \
    --image ag06-mixer:latest \
    --resource-group $RESOURCE_GROUP \
    .

echo "✅ Image built and pushed successfully"

# Create Container Apps environment
echo "🌐 Creating Container Apps environment..."
$AZ_CLI containerapp env create \
    --name $CONTAINER_APP_ENV \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --output table

# Deploy Container App
echo "🚀 Deploying Container App..."
$AZ_CLI containerapp create \
    --name $CONTAINER_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --environment $CONTAINER_APP_ENV \
    --image $ACR_LOGIN_SERVER/ag06-mixer:latest \
    --target-port 8080 \
    --ingress 'external' \
    --registry-server $ACR_LOGIN_SERVER \
    --registry-username $ACR_USERNAME \
    --registry-password $ACR_PASSWORD \
    --cpu 0.25 \
    --memory 0.5Gi \
    --min-replicas 0 \
    --max-replicas 3 \
    --env-vars \
        PRODUCTION=true \
        WEBAPP_DIR=/app/webapp \
        OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES \
    --output table

# Get the app URL
APP_URL=$($AZ_CLI containerapp show \
    --name $CONTAINER_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --query properties.configuration.ingress.fqdn -o tsv)

echo ""
echo "🎉 DEPLOYMENT COMPLETE!"
echo "=================================="
echo "Backend URL: https://$APP_URL"
echo ""
echo "Test endpoints:"
echo "  Health: https://$APP_URL/health"
echo "  Status: https://$APP_URL/api/status"  
echo "  SSE Stream: https://$APP_URL/api/stream"
echo ""

# Save deployment info
cat > deployment-info.json <<EOF
{
  "backend_url": "https://$APP_URL",
  "resource_group": "$RESOURCE_GROUP",
  "container_app": "$CONTAINER_APP_NAME",
  "acr_server": "$ACR_LOGIN_SERVER",
  "deployment_date": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF

echo "✅ Deployment info saved to deployment-info.json"
echo ""
echo "🧪 Testing deployment..."

# Test health endpoint
if curl -f -s "https://$APP_URL/health" | grep -q "healthy"; then
    echo "✅ Backend health check passed"
else
    echo "⚠️  Backend health check pending (may need a few minutes to start)"
fi

echo ""
echo "🎯 Next step: Deploy frontend to Vercel"
echo "   Run: ./deploy-vercel.sh"