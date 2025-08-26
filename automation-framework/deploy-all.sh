#!/bin/bash

# Complete AG06 Mixer Deployment Orchestrator
# Deploys backend to Azure and frontend to Vercel

set -e

echo "=============================================="
echo "🚀 AG06 MIXER - COMPLETE CLOUD DEPLOYMENT"
echo "=============================================="
echo ""
echo "This script will deploy:"
echo "  1. Python backend → Azure Container Apps"
echo "  2. Web UI → Vercel (with your paid plan)"
echo "  3. Configure GitHub Actions CI/CD"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }

# Check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."
    
    local missing=()
    
    command -v docker &> /dev/null || missing+=("docker")
    command -v az &> /dev/null || missing+=("azure-cli")
    command -v vercel &> /dev/null || missing+=("vercel")
    command -v git &> /dev/null || missing+=("git")
    command -v node &> /dev/null || missing+=("node")
    
    if [ ${#missing[@]} -gt 0 ]; then
        print_error "Missing required tools: ${missing[*]}"
        echo "Please install missing tools and try again."
        exit 1
    fi
    
    print_status "All prerequisites installed"
}

# Deploy backend to Azure
deploy_backend() {
    echo ""
    echo "===== BACKEND DEPLOYMENT (Azure) ====="
    
    if [ -f "./deploy-azure.sh" ]; then
        print_info "Starting Azure deployment..."
        chmod +x ./deploy-azure.sh
        ./deploy-azure.sh
        
        if [ $? -eq 0 ]; then
            print_status "Backend deployed successfully"
        else
            print_error "Backend deployment failed"
            exit 1
        fi
    else
        print_error "deploy-azure.sh not found"
        exit 1
    fi
}

# Deploy frontend to Vercel
deploy_frontend() {
    echo ""
    echo "===== FRONTEND DEPLOYMENT (Vercel) ====="
    
    if [ -f "./deploy-vercel.sh" ]; then
        print_info "Starting Vercel deployment..."
        chmod +x ./deploy-vercel.sh
        ./deploy-vercel.sh
        
        if [ $? -eq 0 ]; then
            print_status "Frontend deployed successfully"
        else
            print_error "Frontend deployment failed"
            exit 1
        fi
    else
        print_error "deploy-vercel.sh not found"
        exit 1
    fi
}

# Setup GitHub Actions
setup_github_actions() {
    echo ""
    echo "===== GITHUB ACTIONS SETUP ====="
    
    print_info "Setting up GitHub Actions..."
    
    # Check if we're in a git repo
    if [ ! -d ".git" ]; then
        print_warning "Not in a git repository. Initializing..."
        git init
        git add .
        git commit -m "Initial commit: AG06 Mixer Cloud Deployment"
    fi
    
    # Create GitHub repo (if needed)
    if ! git remote | grep -q origin; then
        print_info "No remote origin found."
        echo "Please create a GitHub repository and run:"
        echo "  git remote add origin https://github.com/YOUR_USERNAME/ag06-mixer.git"
        echo "  git push -u origin main"
        print_warning "Skipping GitHub Actions setup for now"
        return
    fi
    
    # Push workflow file
    if [ -f ".github/workflows/deploy-aca.yml" ]; then
        git add .github/workflows/deploy-aca.yml
        git commit -m "Add Azure deployment workflow" || true
        git push origin main || print_warning "Could not push to GitHub"
        
        print_status "GitHub Actions workflow added"
        print_info "Follow AZURE_OIDC_SETUP.md to configure OIDC authentication"
    fi
}

# Test deployment
test_deployment() {
    echo ""
    echo "===== DEPLOYMENT TESTING ====="
    
    if [ -f "deployment-info.json" ]; then
        BACKEND_URL=$(cat deployment-info.json | grep backend_url | cut -d'"' -f4)
        FRONTEND_URL=$(cat deployment-info.json | grep frontend_url | cut -d'"' -f4)
        
        print_info "Testing backend health..."
        if curl -f "$BACKEND_URL/health" &> /dev/null; then
            print_status "Backend is healthy"
        else
            print_warning "Backend health check failed"
        fi
        
        print_info "Testing frontend..."
        if curl -f "$FRONTEND_URL" &> /dev/null; then
            print_status "Frontend is accessible"
        else
            print_warning "Frontend not accessible"
        fi
    else
        print_warning "No deployment info found, skipping tests"
    fi
}

# Generate summary
generate_summary() {
    echo ""
    echo "=============================================="
    echo "📋 DEPLOYMENT SUMMARY"
    echo "=============================================="
    
    if [ -f "deployment-info.json" ]; then
        BACKEND_URL=$(cat deployment-info.json | grep backend_url | cut -d'"' -f4 || echo "Not deployed")
        FRONTEND_URL=$(cat deployment-info.json | grep frontend_url | cut -d'"' -f4 || echo "Not deployed")
        
        echo ""
        echo "🎯 ENDPOINTS:"
        echo "  Backend API: $BACKEND_URL"
        echo "  Frontend UI: $FRONTEND_URL"
        echo ""
        echo "📱 TEST THE APP:"
        echo "  1. Open $FRONTEND_URL"
        echo "  2. Click 'Connect to Stream'"
        echo "  3. Watch real-time audio processing"
        echo ""
        echo "🔍 API ENDPOINTS:"
        echo "  • Health: $BACKEND_URL/health"
        echo "  • Status: $BACKEND_URL/api/status"
        echo "  • Stream: $BACKEND_URL/api/stream"
        echo "  • Spectrum: $BACKEND_URL/api/spectrum"
        echo ""
        echo "💰 COST OPTIMIZATION:"
        echo "  • Azure: Scales to 0 when idle"
        echo "  • Vercel: Using paid plan features"
        echo "  • Total estimated cost: ~$0-5/month"
        echo ""
        echo "🚀 NEXT STEPS:"
        echo "  1. Configure OIDC (see AZURE_OIDC_SETUP.md)"
        echo "  2. Set up custom domains (optional)"
        echo "  3. Configure monitoring alerts"
        echo "  4. Deploy iOS app with CloudKit"
        echo ""
        echo "🗑️  CLEANUP COMMANDS:"
        echo "  Azure: az group delete --name ag06-mixer-rg --yes"
        echo "  Vercel: vercel remove ag06-mixer-ui --yes"
    else
        print_error "No deployment information available"
    fi
    
    echo ""
    echo "=============================================="
    echo "✨ DEPLOYMENT COMPLETE!"
    echo "=============================================="
}

# Main execution
main() {
    echo "Starting deployment process..."
    echo ""
    
    # Step 1: Check prerequisites
    check_prerequisites
    
    # Step 2: Deploy backend
    read -p "Deploy backend to Azure? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        deploy_backend
    else
        print_info "Skipping backend deployment"
    fi
    
    # Step 3: Deploy frontend
    read -p "Deploy frontend to Vercel? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        deploy_frontend
    else
        print_info "Skipping frontend deployment"
    fi
    
    # Step 4: Setup GitHub Actions
    read -p "Setup GitHub Actions CI/CD? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        setup_github_actions
    else
        print_info "Skipping GitHub Actions setup"
    fi
    
    # Step 5: Test deployment
    test_deployment
    
    # Step 6: Generate summary
    generate_summary
}

# Run main function
main "$@"