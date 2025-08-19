#!/bin/bash

# Blue-Green Deployment Script for AG06 Mixer
# Implements zero-downtime deployment with automatic rollback

set -e

# Configuration
NAMESPACE="ag06-mixer"
APP_NAME="ag06-mixer"
DEPLOYMENT_TIMEOUT=300
HEALTH_CHECK_RETRIES=30
PERFORMANCE_THRESHOLD_MS=5
MEMORY_THRESHOLD_MB=15
ERROR_RATE_THRESHOLD=0.01

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get current active version
get_active_version() {
    kubectl get service ${APP_NAME}-service -n ${NAMESPACE} -o jsonpath='{.spec.selector.version}'
}

# Get inactive version
get_inactive_version() {
    local active=$1
    if [ "$active" = "blue" ]; then
        echo "green"
    else
        echo "blue"
    fi
}

# Check deployment health
check_deployment_health() {
    local version=$1
    local deployment="${APP_NAME}-${version}"
    
    log_info "Checking health of ${deployment}..."
    
    # Wait for rollout to complete
    if ! kubectl rollout status deployment/${deployment} -n ${NAMESPACE} --timeout=${DEPLOYMENT_TIMEOUT}s; then
        log_error "Deployment rollout failed"
        return 1
    fi
    
    # Check pod readiness
    local ready_pods=$(kubectl get deployment ${deployment} -n ${NAMESPACE} -o jsonpath='{.status.readyReplicas}')
    local desired_pods=$(kubectl get deployment ${deployment} -n ${NAMESPACE} -o jsonpath='{.spec.replicas}')
    
    if [ "$ready_pods" != "$desired_pods" ]; then
        log_error "Not all pods are ready: ${ready_pods}/${desired_pods}"
        return 1
    fi
    
    log_success "All pods are ready: ${ready_pods}/${desired_pods}"
    return 0
}

# Run smoke tests
run_smoke_tests() {
    local version=$1
    local service="${APP_NAME}-${version}-service"
    
    log_info "Running smoke tests for ${version} deployment..."
    
    # Port forward to test service
    kubectl port-forward service/${service} 8080:8080 -n ${NAMESPACE} &
    local port_forward_pid=$!
    sleep 5
    
    # Health check
    if ! curl -f http://localhost:8080/health; then
        log_error "Health check failed"
        kill $port_forward_pid
        return 1
    fi
    
    # API test
    if ! curl -f http://localhost:8080/api/mixer/status; then
        log_error "API test failed"
        kill $port_forward_pid
        return 1
    fi
    
    kill $port_forward_pid
    log_success "Smoke tests passed"
    return 0
}

# Check performance metrics
check_performance_metrics() {
    local version=$1
    local deployment="${APP_NAME}-${version}"
    
    log_info "Checking performance metrics for ${deployment}..."
    
    # Get metrics from Prometheus
    local prometheus_url="http://prometheus:9090"
    
    # Check latency
    local latency=$(kubectl exec -n ${NAMESPACE} deployment/prometheus -- \
        curl -s "${prometheus_url}/api/v1/query?query=audio_latency_ms{deployment_version=\"${version}\"}" | \
        jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0")
    
    if (( $(echo "$latency > $PERFORMANCE_THRESHOLD_MS" | bc -l) )); then
        log_error "Latency ${latency}ms exceeds threshold ${PERFORMANCE_THRESHOLD_MS}ms"
        return 1
    fi
    
    # Check memory usage per channel
    local memory=$(kubectl exec -n ${NAMESPACE} deployment/prometheus -- \
        curl -s "${prometheus_url}/api/v1/query?query=(memory_usage_mb/channel_count){deployment_version=\"${version}\"}" | \
        jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0")
    
    if (( $(echo "$memory > $MEMORY_THRESHOLD_MB" | bc -l) )); then
        log_error "Memory usage ${memory}MB/channel exceeds threshold ${MEMORY_THRESHOLD_MB}MB"
        return 1
    fi
    
    # Check error rate
    local error_rate=$(kubectl exec -n ${NAMESPACE} deployment/prometheus -- \
        curl -s "${prometheus_url}/api/v1/query?query=rate(http_requests_errors_total[5m]){deployment_version=\"${version}\"}" | \
        jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0")
    
    if (( $(echo "$error_rate > $ERROR_RATE_THRESHOLD" | bc -l) )); then
        log_error "Error rate ${error_rate} exceeds threshold ${ERROR_RATE_THRESHOLD}"
        return 1
    fi
    
    log_success "Performance metrics within thresholds"
    log_info "  Latency: ${latency}ms (threshold: ${PERFORMANCE_THRESHOLD_MS}ms)"
    log_info "  Memory: ${memory}MB/channel (threshold: ${MEMORY_THRESHOLD_MB}MB)"
    log_info "  Error rate: ${error_rate} (threshold: ${ERROR_RATE_THRESHOLD})"
    return 0
}

# Switch traffic
switch_traffic() {
    local target_version=$1
    
    log_info "Switching traffic to ${target_version} deployment..."
    
    kubectl patch service ${APP_NAME}-service -n ${NAMESPACE} \
        -p "{\"spec\":{\"selector\":{\"version\":\"${target_version}\"}}}"
    
    log_success "Traffic switched to ${target_version}"
}

# Rollback deployment
rollback() {
    local original_version=$1
    
    log_warning "Rolling back to ${original_version}..."
    switch_traffic "$original_version"
    log_success "Rollback completed"
}

# Monitor deployment
monitor_deployment() {
    local version=$1
    local duration=$2
    
    log_info "Monitoring ${version} deployment for ${duration} seconds..."
    
    local end_time=$(($(date +%s) + duration))
    
    while [ $(date +%s) -lt $end_time ]; do
        if ! check_performance_metrics "$version"; then
            return 1
        fi
        sleep 10
    done
    
    log_success "Monitoring completed successfully"
    return 0
}

# Main deployment flow
main() {
    local new_image=${1:-}
    
    if [ -z "$new_image" ]; then
        log_error "Usage: $0 <new-image-tag>"
        exit 1
    fi
    
    log_info "Starting blue-green deployment for AG06 Mixer"
    log_info "New image: ${new_image}"
    
    # Get current state
    local active_version=$(get_active_version)
    local inactive_version=$(get_inactive_version "$active_version")
    
    log_info "Current active version: ${active_version}"
    log_info "Will deploy to: ${inactive_version}"
    
    # Update inactive deployment with new image
    log_info "Updating ${inactive_version} deployment with new image..."
    kubectl set image deployment/${APP_NAME}-${inactive_version} \
        ${APP_NAME}=${new_image} \
        -n ${NAMESPACE}
    
    # Wait for deployment to be ready
    if ! check_deployment_health "$inactive_version"; then
        log_error "Deployment health check failed"
        exit 1
    fi
    
    # Run smoke tests
    if ! run_smoke_tests "$inactive_version"; then
        log_error "Smoke tests failed"
        exit 1
    fi
    
    # Check performance before switching
    if ! check_performance_metrics "$inactive_version"; then
        log_error "Performance metrics check failed"
        exit 1
    fi
    
    # Switch traffic to new version
    switch_traffic "$inactive_version"
    
    # Monitor for 5 minutes
    if ! monitor_deployment "$inactive_version" 300; then
        log_error "Monitoring detected issues, initiating rollback"
        rollback "$active_version"
        exit 1
    fi
    
    # Scale down old version (optional)
    read -p "Scale down ${active_version} deployment? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Scaling down ${active_version} deployment..."
        kubectl scale deployment/${APP_NAME}-${active_version} --replicas=0 -n ${NAMESPACE}
        log_success "Old deployment scaled down"
    fi
    
    log_success "Blue-green deployment completed successfully!"
    log_info "New active version: ${inactive_version}"
}

# Run main function
main "$@"