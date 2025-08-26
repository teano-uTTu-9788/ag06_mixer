#!/bin/bash

# Azure Container Apps Deployment Script
# Using free student credits ($100) 

set -e

echo "================================================"
echo "🚀 AG06 MIXER - AZURE CONTAINER APPS DEPLOYMENT"
echo "================================================"

# Configuration
RESOURCE_GROUP="ag06-mixer-rg"
LOCATION="eastus"
ACR_NAME="ag06mixeracr"
CONTAINER_APP_NAME="ag06-mixer-backend"
CONTAINER_APP_ENV="ag06-mixer-env"
IMAGE_NAME="ag06-mixer"
IMAGE_TAG="latest"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check Azure CLI is installed
if ! command -v az &> /dev/null; then
    print_error "Azure CLI not found. Please install: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

# Login to Azure (using student account)
print_status "Logging into Azure..."
az login --use-device-code

# Set subscription (if you have multiple)
# az account set --subscription "YOUR_SUBSCRIPTION_ID"

# Create resource group
print_status "Creating resource group..."
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create Azure Container Registry (Basic tier for student budget)
print_status "Creating Azure Container Registry (Basic tier)..."
az acr create \
    --resource-group $RESOURCE_GROUP \
    --name $ACR_NAME \
    --sku Basic \
    --admin-enabled true

# Get ACR credentials
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query passwords[0].value -o tsv)
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --query loginServer -o tsv)

print_status "ACR created: $ACR_LOGIN_SERVER"

# Build and push Docker image
print_status "Building Docker image..."
docker build -t $IMAGE_NAME:$IMAGE_TAG .

# Tag image for ACR
docker tag $IMAGE_NAME:$IMAGE_TAG $ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG

# Login to ACR
print_status "Logging into ACR..."
echo $ACR_PASSWORD | docker login $ACR_LOGIN_SERVER -u $ACR_USERNAME --password-stdin

# Push image to ACR
print_status "Pushing image to ACR..."
docker push $ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG

# Create Container Apps environment
print_status "Creating Container Apps environment..."
az containerapp env create \
    --name $CONTAINER_APP_ENV \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION

# Deploy Container App
print_status "Deploying Container App..."
az containerapp create \
    --name $CONTAINER_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --environment $CONTAINER_APP_ENV \
    --image $ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG \
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
        OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Get the app URL
APP_URL=$(az containerapp show \
    --name $CONTAINER_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --query properties.configuration.ingress.fqdn -o tsv)

# Enable CORS for the app (if needed)
print_status "Configuring CORS..."
az containerapp ingress cors enable \
    --name $CONTAINER_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --allowed-origins "*" \
    --allowed-methods "GET" "POST" "OPTIONS" \
    --allowed-headers "*"

# Set up auto-scaling rules
print_status "Configuring auto-scaling..."
az containerapp update \
    --name $CONTAINER_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --scale-rule-name "http-rule" \
    --scale-rule-type "http" \
    --scale-rule-http-concurrency 10

echo ""
echo "================================================"
echo "✅ DEPLOYMENT COMPLETE!"
echo "================================================"
echo ""
echo "📱 Backend API URL: https://$APP_URL"
echo ""
echo "🔍 Test endpoints:"
echo "  • Health: https://$APP_URL/health"
echo "  • Status: https://$APP_URL/api/status"
echo "  • SSE Stream: https://$APP_URL/api/stream"
echo "  • Spectrum: https://$APP_URL/api/spectrum"
echo ""
echo "💰 Cost Optimization Tips:"
echo "  • Min replicas set to 0 (scales to zero when idle)"
echo "  • Using 0.25 vCPU and 0.5GB RAM (minimal resources)"
echo "  • Basic ACR tier (cheapest option)"
echo "  • Auto-scaling based on HTTP concurrency"
echo ""
echo "📊 Monitor usage:"
echo "  az containerapp logs show -n $CONTAINER_APP_NAME -g $RESOURCE_GROUP --follow"
echo ""
echo "🗑️  To delete all resources (stop charges):"
echo "  az group delete --name $RESOURCE_GROUP --yes"
echo ""
echo "================================================"

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

print_status "Deployment info saved to deployment-info.json"