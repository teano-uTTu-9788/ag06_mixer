#!/bin/bash
# AiOke GCP Deployment Script
# Following Google Cloud Platform best practices

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 AiOke GCP Deployment Script${NC}"
echo "================================="

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="aioke-production"
IMAGE_NAME="gcr.io/${PROJECT_ID}/aioke"

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        echo -e "${RED}❌ gcloud CLI not found. Please install it first.${NC}"
        echo "Visit: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
    
    # Check if logged in
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        echo -e "${YELLOW}Not logged in to GCP. Running 'gcloud auth login'...${NC}"
        gcloud auth login
    fi
    
    # Set project
    echo -e "${YELLOW}Setting project to: ${PROJECT_ID}${NC}"
    gcloud config set project ${PROJECT_ID}
    
    echo -e "${GREEN}✅ Prerequisites checked${NC}"
}

# Enable required APIs
enable_apis() {
    echo -e "${YELLOW}Enabling required GCP APIs...${NC}"
    
    gcloud services enable \
        run.googleapis.com \
        containerregistry.googleapis.com \
        cloudbuild.googleapis.com \
        secretmanager.googleapis.com \
        youtube.googleapis.com \
        --project=${PROJECT_ID}
    
    echo -e "${GREEN}✅ APIs enabled${NC}"
}

# Create secrets
setup_secrets() {
    echo -e "${YELLOW}Setting up secrets...${NC}"
    
    # Check if YouTube API key exists
    if [ -z "${YOUTUBE_API_KEY:-}" ]; then
        echo -e "${YELLOW}Enter your YouTube API Key:${NC}"
        read -s YOUTUBE_API_KEY
    fi
    
    # Create secret if it doesn't exist
    if ! gcloud secrets describe youtube-api-key --project=${PROJECT_ID} &>/dev/null; then
        echo "${YOUTUBE_API_KEY}" | gcloud secrets create youtube-api-key \
            --data-file=- \
            --replication-policy="automatic" \
            --project=${PROJECT_ID}
        echo -e "${GREEN}✅ Secret created${NC}"
    else
        echo -e "${YELLOW}Secret already exists${NC}"
    fi
    
    # Grant Cloud Run access to the secret
    gcloud secrets add-iam-policy-binding youtube-api-key \
        --member="serviceAccount:${PROJECT_ID}-compute@developer.gserviceaccount.com" \
        --role="roles/secretmanager.secretAccessor" \
        --project=${PROJECT_ID}
}

# Build and push Docker image
build_and_push() {
    echo -e "${YELLOW}Building Docker image...${NC}"
    
    # Use Cloud Build for consistency
    gcloud builds submit \
        --tag ${IMAGE_NAME}:latest \
        --file Dockerfile.gcp \
        --project=${PROJECT_ID} \
        .
    
    echo -e "${GREEN}✅ Image built and pushed to GCR${NC}"
}

# Deploy to Cloud Run
deploy_service() {
    echo -e "${YELLOW}Deploying to Cloud Run...${NC}"
    
    gcloud run deploy ${SERVICE_NAME} \
        --image ${IMAGE_NAME}:latest \
        --platform managed \
        --region ${REGION} \
        --allow-unauthenticated \
        --port 8080 \
        --memory 2Gi \
        --cpu 2 \
        --max-instances 100 \
        --min-instances 1 \
        --set-env-vars="ENVIRONMENT=production" \
        --set-secrets="YOUTUBE_API_KEY=youtube-api-key:latest" \
        --project=${PROJECT_ID}
    
    echo -e "${GREEN}✅ Service deployed${NC}"
}

# Get service URL
get_service_url() {
    SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
        --platform managed \
        --region ${REGION} \
        --format 'value(status.url)' \
        --project=${PROJECT_ID})
    
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}🎉 AiOke deployed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "🌐 Service URL: ${YELLOW}${SERVICE_URL}${NC}"
    echo ""
    echo "📱 iPad Access Instructions:"
    echo "1. Open Safari on your iPad"
    echo "2. Navigate to: ${SERVICE_URL}"
    echo "3. Tap Share → Add to Home Screen"
    echo "4. Name it 'AiOke' and tap Add"
    echo ""
    echo "🎤 Features:"
    echo "• YouTube karaoke search"
    echo "• Real-time vocal processing"
    echo "• AI-powered mixing"
    echo "• Voice commands"
    echo "• Progressive Web App"
    echo ""
    echo -e "${GREEN}========================================${NC}"
}

# Main deployment flow
main() {
    echo ""
    echo "Deploying AiOke to Google Cloud Platform"
    echo "Project: ${PROJECT_ID}"
    echo "Region: ${REGION}"
    echo ""
    
    check_prerequisites
    enable_apis
    setup_secrets
    build_and_push
    deploy_service
    get_service_url
    
    echo -e "${GREEN}Deployment complete!${NC}"
}

# Run main function
main "$@"