#!/bin/bash
set -e

echo "🚀 AG06 Mixer Production Deployment"
echo "=================================="

# Configuration
DOMAIN="ag06mixer.com"
REGION="us-east-1"
CLUSTER_NAME="ag06mixer-cluster"
NAMESPACE="ag06mixer"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    commands=("terraform" "kubectl" "aws" "helm")
    for cmd in "${commands[@]}"; do
        if ! command -v $cmd &> /dev/null; then
            log_error "$cmd is not installed"
            exit 1
        fi
    done
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

# Deploy infrastructure
deploy_infrastructure() {
    log_info "Deploying infrastructure with Terraform..."
    
    cd infrastructure
    
    # Initialize Terraform
    terraform init
    
    # Plan deployment
    terraform plan -out=tfplan
    
    # Apply infrastructure
    terraform apply tfplan
    
    # Get outputs
    export VPC_ID=$(terraform output -raw vpc_id)
    export CLUSTER_NAME=$(terraform output -raw cluster_name)
    export DB_ENDPOINT=$(terraform output -raw database_endpoint)
    export ALB_DNS=$(terraform output -raw load_balancer_dns)
    
    cd ..
    
    log_info "Infrastructure deployed successfully"
}

# Configure kubectl
configure_kubectl() {
    log_info "Configuring kubectl for EKS cluster..."
    
    aws eks update-kubeconfig --region $REGION --name $CLUSTER_NAME
    
    # Verify connection
    kubectl get nodes
    
    log_info "kubectl configured successfully"
}

# Deploy application
deploy_application() {
    log_info "Deploying AG06 Mixer application..."
    
    # Apply Kubernetes manifests
    kubectl apply -f k8s-manifests/
    
    # Wait for deployment to be ready
    kubectl wait --for=condition=available --timeout=300s deployment/ag06mixer-app -n $NAMESPACE
    
    log_info "Application deployed successfully"
}

# Deploy monitoring
deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    
    # Add Prometheus Helm repo
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    
    # Install kube-prometheus-stack
    helm install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --wait
    
    log_info "Monitoring deployed successfully"
}

# Validate deployment
validate_deployment() {
    log_info "Validating deployment..."
    
    # Check application health
    kubectl get pods -n $NAMESPACE
    
    # Get service endpoints
    kubectl get services -n $NAMESPACE
    
    # Test health endpoint
    APP_URL="https://$DOMAIN"
    if curl -f -s "$APP_URL/health" > /dev/null; then
        log_info "Application health check passed"
    else
        log_warn "Application health check failed - may need DNS propagation time"
    fi
    
    log_info "Deployment validation complete"
}

# Main deployment flow
main() {
    log_info "Starting AG06 Mixer production deployment..."
    
    check_prerequisites
    deploy_infrastructure
    configure_kubectl
    deploy_application
    deploy_monitoring
    validate_deployment
    
    log_info "🎉 Deployment completed successfully!"
    log_info "Your application will be available at: https://$DOMAIN"
    log_info "API endpoint: https://api.$DOMAIN"
    log_info "Monitoring: kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80"
}

# Run main function
main "$@"
