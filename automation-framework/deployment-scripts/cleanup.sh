#!/bin/bash
set -e

echo "🧹 Cleaning up AG06 Mixer deployment"
echo "===================================="

REGION="us-east-1"
CLUSTER_NAME="ag06mixer-cluster"

# Remove Kubernetes resources
cleanup_kubernetes() {
    echo "Removing Kubernetes resources..."
    
    # Delete application
    kubectl delete -f k8s-manifests/ --ignore-not-found=true
    
    # Remove monitoring
    helm uninstall prometheus -n monitoring --ignore-not-found
    kubectl delete namespace monitoring --ignore-not-found=true
    
    echo "Kubernetes resources cleaned up"
}

# Destroy infrastructure
cleanup_infrastructure() {
    echo "Destroying infrastructure..."
    
    cd infrastructure
    terraform destroy -auto-approve
    cd ..
    
    echo "Infrastructure destroyed"
}

# Main cleanup
main() {
    echo "⚠️  WARNING: This will destroy ALL resources!"
    read -p "Are you sure you want to continue? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cleanup_kubernetes
        cleanup_infrastructure
        echo "✅ Cleanup complete"
    else
        echo "Cleanup cancelled"
    fi
}

main "$@"
