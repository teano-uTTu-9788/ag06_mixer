#!/bin/bash
set -e

echo "📊 Setting up monitoring for AG06 Mixer"
echo "======================================="

# Install monitoring stack
install_monitoring() {
    echo "Installing Prometheus and Grafana..."
    
    # Add Helm repositories
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update
    
    # Install kube-prometheus-stack
    helm install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
        --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
        --set grafana.adminPassword=admin123 \
        --wait
    
    echo "Monitoring stack installed"
}

# Configure Grafana dashboards
configure_dashboards() {
    echo "Configuring Grafana dashboards..."
    
    # Port forward to access Grafana
    kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80 &
    
    echo "Grafana will be available at http://localhost:3000"
    echo "Username: admin"
    echo "Password: admin123"
}

# Main monitoring setup
main() {
    install_monitoring
    configure_dashboards
    
    echo "✅ Monitoring setup complete"
    echo ""
    echo "Access monitoring:"
    echo "kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80"
}

main "$@"
