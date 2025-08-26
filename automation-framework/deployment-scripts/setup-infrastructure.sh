#!/bin/bash
set -e

echo "🏗️  Setting up AG06 Mixer Infrastructure"
echo "======================================="

# Create S3 bucket for Terraform state
create_terraform_state_bucket() {
    BUCKET_NAME="ag06mixer-terraform-state-$(date +%s)"
    REGION="us-east-1"
    
    echo "Creating S3 bucket for Terraform state: $BUCKET_NAME"
    
    aws s3api create-bucket \
        --bucket $BUCKET_NAME \
        --region $REGION \
        --create-bucket-configuration LocationConstraint=$REGION
    
    # Enable versioning
    aws s3api put-bucket-versioning \
        --bucket $BUCKET_NAME \
        --versioning-configuration Status=Enabled
    
    # Enable encryption
    aws s3api put-bucket-encryption \
        --bucket $BUCKET_NAME \
        --server-side-encryption-configuration '{
            "Rules": [{
                "ApplyServerSideEncryptionByDefault": {
                    "SSEAlgorithm": "AES256"
                }
            }]
        }'
    
    echo "Terraform state bucket created: $BUCKET_NAME"
    echo "Update the backend configuration in main.tf with this bucket name"
}

# Set up AWS load balancer controller
setup_alb_controller() {
    echo "Setting up AWS Load Balancer Controller..."
    
    # Download IAM policy
    curl -o iam_policy.json https://raw.githubusercontent.com/kubernetes-sigs/aws-load-balancer-controller/v2.6.0/docs/install/iam_policy.json
    
    # Create IAM policy
    aws iam create-policy \
        --policy-name AWSLoadBalancerControllerIAMPolicy \
        --policy-document file://iam_policy.json
    
    # Install AWS Load Balancer Controller
    helm repo add eks https://aws.github.io/eks-charts
    helm repo update
    
    kubectl create namespace kube-system || true
    
    helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
        -n kube-system \
        --set clusterName=ag06mixer-cluster \
        --set serviceAccount.create=false \
        --set serviceAccount.name=aws-load-balancer-controller
}

# Main setup
main() {
    create_terraform_state_bucket
    echo ""
    echo "✅ Infrastructure setup complete"
    echo "Next steps:"
    echo "1. Update the S3 bucket name in infrastructure/main.tf"
    echo "2. Run ./deploy.sh to deploy the full stack"
}

main "$@"
