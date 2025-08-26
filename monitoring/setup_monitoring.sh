#!/bin/bash

###############################################################################
# AI Mixer Monitoring Setup Script
# 
# Sets up complete monitoring stack:
# - Prometheus for metrics collection
# - Grafana for visualization
# - Alertmanager for alerting
# - Node Exporter for system metrics
###############################################################################

set -e
set -o pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
MONITORING_NAMESPACE="ai-mixer-monitoring"

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed"
    fi
    
    if ! command -v helm &> /dev/null; then
        warning "Helm is not installed - using kubectl instead"
    fi
    
    log "Prerequisites check completed"
}

# Create monitoring namespace
create_namespace() {
    log "Creating monitoring namespace..."
    
    kubectl create namespace ${MONITORING_NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
    
    log "Monitoring namespace created"
}

# Deploy Prometheus
deploy_prometheus() {
    log "Deploying Prometheus..."
    
    # Create ConfigMap for Prometheus configuration
    kubectl create configmap prometheus-config \
        --from-file="${SCRIPT_DIR}/prometheus_config.yml" \
        --from-file="${SCRIPT_DIR}/alert_rules.yml" \
        -n ${MONITORING_NAMESPACE} \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy Prometheus
    cat << EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: ${MONITORING_NAMESPACE}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: config
          mountPath: /etc/prometheus
        - name: storage
          mountPath: /prometheus
        args:
          - '--config.file=/etc/prometheus/prometheus_config.yml'
          - '--storage.tsdb.path=/prometheus'
          - '--web.console.libraries=/etc/prometheus/console_libraries'
          - '--web.console.templates=/etc/prometheus/consoles'
          - '--storage.tsdb.retention.time=15d'
          - '--web.enable-lifecycle'
      volumes:
      - name: config
        configMap:
          name: prometheus-config
      - name: storage
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: prometheus
  namespace: ${MONITORING_NAMESPACE}
spec:
  selector:
    app: prometheus
  ports:
  - port: 9090
    targetPort: 9090
  type: ClusterIP
EOF
    
    log "Prometheus deployed"
}

# Deploy Grafana
deploy_grafana() {
    log "Deploying Grafana..."
    
    # Create ConfigMap for Grafana dashboard
    kubectl create configmap grafana-dashboard \
        --from-file="${SCRIPT_DIR}/grafana_dashboard.json" \
        -n ${MONITORING_NAMESPACE} \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy Grafana
    cat << EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: ${MONITORING_NAMESPACE}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:latest
        ports:
        - containerPort: 3000
        env:
        - name: GF_SECURITY_ADMIN_PASSWORD
          value: "admin123"
        - name: GF_DASHBOARDS_JSON_ENABLED
          value: "true"
        - name: GF_DASHBOARDS_JSON_PATH
          value: "/var/lib/grafana/dashboards"
        volumeMounts:
        - name: dashboard
          mountPath: /var/lib/grafana/dashboards
        - name: storage
          mountPath: /var/lib/grafana
      volumes:
      - name: dashboard
        configMap:
          name: grafana-dashboard
      - name: storage
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: grafana
  namespace: ${MONITORING_NAMESPACE}
spec:
  selector:
    app: grafana
  ports:
  - port: 3000
    targetPort: 3000
  type: LoadBalancer
EOF
    
    log "Grafana deployed"
}

# Deploy Alertmanager
deploy_alertmanager() {
    log "Deploying Alertmanager..."
    
    # Create ConfigMap for Alertmanager configuration
    kubectl create configmap alertmanager-config \
        --from-file="${SCRIPT_DIR}/alertmanager_config.yml" \
        -n ${MONITORING_NAMESPACE} \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy Alertmanager
    cat << EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: alertmanager
  namespace: ${MONITORING_NAMESPACE}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: alertmanager
  template:
    metadata:
      labels:
        app: alertmanager
    spec:
      containers:
      - name: alertmanager
        image: prom/alertmanager:latest
        ports:
        - containerPort: 9093
        volumeMounts:
        - name: config
          mountPath: /etc/alertmanager
        - name: storage
          mountPath: /alertmanager
        args:
          - '--config.file=/etc/alertmanager/alertmanager_config.yml'
          - '--storage.path=/alertmanager'
      volumes:
      - name: config
        configMap:
          name: alertmanager-config
      - name: storage
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: alertmanager
  namespace: ${MONITORING_NAMESPACE}
spec:
  selector:
    app: alertmanager
  ports:
  - port: 9093
    targetPort: 9093
  type: ClusterIP
EOF
    
    log "Alertmanager deployed"
}

# Deploy Node Exporter
deploy_node_exporter() {
    log "Deploying Node Exporter..."
    
    cat << EOF | kubectl apply -f -
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: node-exporter
  namespace: ${MONITORING_NAMESPACE}
spec:
  selector:
    matchLabels:
      app: node-exporter
  template:
    metadata:
      labels:
        app: node-exporter
    spec:
      hostNetwork: true
      hostPID: true
      containers:
      - name: node-exporter
        image: prom/node-exporter:latest
        ports:
        - containerPort: 9100
        args:
          - '--path.procfs=/host/proc'
          - '--path.sysfs=/host/sys'
          - '--collector.filesystem.ignored-mount-points'
          - '^/(sys|proc|dev|host|etc|rootfs/var/lib/docker/containers|rootfs/var/lib/docker/overlay2|rootfs/run/docker/netns|rootfs/var/lib/docker/aufs)($$|/)'
        volumeMounts:
        - name: proc
          mountPath: /host/proc
          readOnly: true
        - name: sys
          mountPath: /host/sys
          readOnly: true
      volumes:
      - name: proc
        hostPath:
          path: /proc
      - name: sys
        hostPath:
          path: /sys
      tolerations:
      - operator: Exists
---
apiVersion: v1
kind: Service
metadata:
  name: node-exporter
  namespace: ${MONITORING_NAMESPACE}
spec:
  selector:
    app: node-exporter
  ports:
  - port: 9100
    targetPort: 9100
  type: ClusterIP
EOF
    
    log "Node Exporter deployed"
}

# Setup service monitors for AI Mixer components
setup_service_monitors() {
    log "Setting up service monitors..."
    
    cat << EOF | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  name: ai-mixer-metrics
  namespace: ai-mixer-global
  labels:
    app: ai-mixer
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/path: "/metrics"
    prometheus.io/port: "8080"
spec:
  selector:
    app: ai-mixer
  ports:
  - name: metrics
    port: 8080
    targetPort: 8080
  type: ClusterIP
EOF
    
    log "Service monitors configured"
}

# Wait for deployments
wait_for_deployments() {
    log "Waiting for deployments to be ready..."
    
    kubectl -n ${MONITORING_NAMESPACE} wait --for=condition=available --timeout=300s \
        deployment/prometheus \
        deployment/grafana \
        deployment/alertmanager
    
    log "All deployments are ready"
}

# Display access information
display_access_info() {
    log "========================================="
    log "Monitoring Stack Deployment Complete!"
    log "========================================="
    
    # Get Grafana external IP
    GRAFANA_IP=$(kubectl -n ${MONITORING_NAMESPACE} get service grafana -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
    
    if [ "$GRAFANA_IP" = "pending" ] || [ -z "$GRAFANA_IP" ]; then
        log "Grafana dashboard: http://localhost:3000 (use port-forward)"
        log "To access: kubectl port-forward -n ${MONITORING_NAMESPACE} svc/grafana 3000:3000"
    else
        log "Grafana dashboard: http://${GRAFANA_IP}:3000"
    fi
    
    log "Default credentials: admin / admin123"
    log ""
    log "Prometheus: http://localhost:9090 (use port-forward)"
    log "To access: kubectl port-forward -n ${MONITORING_NAMESPACE} svc/prometheus 9090:9090"
    log ""
    log "Alertmanager: http://localhost:9093 (use port-forward)"
    log "To access: kubectl port-forward -n ${MONITORING_NAMESPACE} svc/alertmanager 9093:9093"
    log "========================================="
}

# Main function
main() {
    log "========================================="
    log "AI Mixer Monitoring Stack Setup"
    log "========================================="
    
    check_prerequisites
    create_namespace
    deploy_prometheus
    deploy_grafana
    deploy_alertmanager
    deploy_node_exporter
    setup_service_monitors
    wait_for_deployments
    display_access_info
    
    log "Monitoring setup completed successfully!"
}

# Run main function
main "$@"