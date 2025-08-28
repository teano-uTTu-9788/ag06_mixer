#!/usr/bin/env bash
#
# SRE Observability - Google SRE-inspired observability with OpenTelemetry, Prometheus, and Grafana
# Following Google SRE 2025 best practices with intent-based configuration and golden signals
#
set -euo pipefail

# Source framework dependencies
source "$(dirname "${BASH_SOURCE[0]}")/../core/bootstrap.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../core/logger.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../core/config.sh"

# SRE Observability Configuration
if [[ -z "${SRE_OBSERVABILITY_VERSION:-}" ]]; then
    readonly SRE_OBSERVABILITY_VERSION="1.0.0"
fi
if [[ -z "${OTEL_SERVICE_NAME:-}" ]]; then
    readonly OTEL_SERVICE_NAME="terminal-automation-framework"
fi
if [[ -z "${OTEL_SERVICE_VERSION:-}" ]]; then
    readonly OTEL_SERVICE_VERSION="${FRAMEWORK_VERSION:-1.0.0}"
fi

# Google SRE Golden Signals
readonly SRE_GOLDEN_SIGNALS=("latency" "traffic" "errors" "saturation")

# Observability State
SRE_OBSERVABILITY_INITIALIZED=false
SRE_METRICS_ENABLED=false
SRE_TRACING_ENABLED=false
SRE_LOGGING_ENABLED=false

# Initialize SRE observability stack
sre::observability::init() {
    if [[ "$SRE_OBSERVABILITY_INITIALIZED" == "true" ]]; then
        return 0
    fi
    
    log::info "Initializing SRE Observability Stack v${SRE_OBSERVABILITY_VERSION}"
    
    # Initialize framework only if not already initialized
    if [[ "${FRAMEWORK_INITIALIZED:-false}" != "true" ]]; then
        framework::init
    fi
    config::load
    
    # Create observability directories
    local obs_dir="${FRAMEWORK_DATA_DIR}/observability"
    mkdir -p "${obs_dir}"/{metrics,traces,logs,config}
    
    # Setup OpenTelemetry environment
    export OTEL_SERVICE_NAME="$OTEL_SERVICE_NAME"
    export OTEL_SERVICE_VERSION="$OTEL_SERVICE_VERSION"
    export OTEL_RESOURCE_ATTRIBUTES="service.name=${OTEL_SERVICE_NAME},service.version=${OTEL_SERVICE_VERSION}"
    
    SRE_OBSERVABILITY_INITIALIZED=true
    log::success "SRE Observability Stack initialized"
}

# Google SRE Four Golden Signals implementation
sre::golden_signals::collect() {
    local signal="$1"
    local value="${2:-0}"
    local labels="${3:-}"
    
    case "$signal" in
        latency)
            sre::metrics::histogram "request_duration_seconds" "$value" "$labels"
            ;;
        traffic)
            sre::metrics::counter "requests_total" "$value" "$labels"
            ;;
        errors)
            sre::metrics::counter "errors_total" "$value" "$labels"
            ;;
        saturation)
            sre::metrics::gauge "resource_utilization" "$value" "$labels"
            ;;
        *)
            log::error "Unknown golden signal: $signal"
            return 1
            ;;
    esac
}

# Prometheus-compatible metrics collection
sre::metrics::counter() {
    local metric_name="$1"
    local value="${2:-1}"
    local labels="${3:-}"
    
    local timestamp=$(date +%s)
    local metric_line="${metric_name}{${labels}} ${value} ${timestamp}000"
    
    echo "$metric_line" >> "${FRAMEWORK_DATA_DIR}/observability/metrics/prometheus.txt"
    log::trace "Counter metric: $metric_line"
}

sre::metrics::gauge() {
    local metric_name="$1"
    local value="$2"
    local labels="${3:-}"
    
    local timestamp=$(date +%s)
    local metric_line="${metric_name}{${labels}} ${value} ${timestamp}000"
    
    echo "$metric_line" >> "${FRAMEWORK_DATA_DIR}/observability/metrics/prometheus.txt"
    log::trace "Gauge metric: $metric_line"
}

sre::metrics::histogram() {
    local metric_name="$1"
    local value="$2"
    local labels="${3:-}"
    
    local timestamp=$(date +%s)
    
    # Histogram buckets (Google SRE recommended latency buckets)
    local buckets=(0.005 0.01 0.025 0.05 0.1 0.25 0.5 1 2.5 5 10)
    
    for bucket in "${buckets[@]}"; do
        if (( $(echo "$value <= $bucket" | bc -l) )); then
            local bucket_metric="${metric_name}_bucket{${labels},le=\"${bucket}\"} 1 ${timestamp}000"
            echo "$bucket_metric" >> "${FRAMEWORK_DATA_DIR}/observability/metrics/prometheus.txt"
        fi
    done
    
    # Count and sum
    echo "${metric_name}_count{${labels}} 1 ${timestamp}000" >> "${FRAMEWORK_DATA_DIR}/observability/metrics/prometheus.txt"
    echo "${metric_name}_sum{${labels}} ${value} ${timestamp}000" >> "${FRAMEWORK_DATA_DIR}/observability/metrics/prometheus.txt"
    
    log::trace "Histogram metric: ${metric_name} = ${value}"
}

# OpenTelemetry distributed tracing
sre::tracing::span_start() {
    local span_name="$1"
    local parent_span_id="${2:-}"
    
    local span_id=$(openssl rand -hex 8)
    local trace_id="${OTEL_TRACE_ID:-$(openssl rand -hex 16)}"
    
    export OTEL_TRACE_ID="$trace_id"
    export OTEL_PARENT_SPAN_ID="$parent_span_id"
    export OTEL_SPAN_ID="$span_id"
    
    local span_start=$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ")
    
    cat > "${FRAMEWORK_DATA_DIR}/observability/traces/${span_id}.json" << EOF
{
    "traceId": "$trace_id",
    "spanId": "$span_id",
    "parentSpanId": "$parent_span_id",
    "operationName": "$span_name",
    "startTime": "$span_start",
    "tags": {
        "service.name": "$OTEL_SERVICE_NAME",
        "service.version": "$OTEL_SERVICE_VERSION"
    },
    "process": {
        "serviceName": "$OTEL_SERVICE_NAME",
        "tags": {}
    }
}
EOF
    
    echo "$span_id"
}

sre::tracing::span_finish() {
    local span_id="$1"
    local status="${2:-ok}"
    
    local span_end=$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ")
    local span_file="${FRAMEWORK_DATA_DIR}/observability/traces/${span_id}.json"
    
    if [[ -f "$span_file" ]]; then
        # Update span with end time and status
        jq --arg endTime "$span_end" --arg status "$status" \
           '.endTime = $endTime | .status = $status' \
           "$span_file" > "${span_file}.tmp" && mv "${span_file}.tmp" "$span_file"
    fi
    
    log::trace "Span finished: $span_id status=$status"
}

# Structured logging with correlation IDs
sre::logging::structured() {
    local level="$1"
    local message="$2"
    local component="${3:-unknown}"
    
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ")
    local correlation_id="${OTEL_TRACE_ID:-$(openssl rand -hex 8)}"
    
    local log_entry=$(jq -n \
        --arg timestamp "$timestamp" \
        --arg level "$level" \
        --arg message "$message" \
        --arg component "$component" \
        --arg correlationId "$correlation_id" \
        --arg service "$OTEL_SERVICE_NAME" \
        --arg version "$OTEL_SERVICE_VERSION" \
        '{
            timestamp: $timestamp,
            level: $level,
            message: $message,
            component: $component,
            correlationId: $correlationId,
            service: $service,
            version: $version
        }')
    
    echo "$log_entry" >> "${FRAMEWORK_DATA_DIR}/observability/logs/structured.json"
    
    # Also log to traditional format for compatibility
    echo "[$timestamp] [$level] [$component] [$correlation_id] $message" >> "${FRAMEWORK_DATA_DIR}/observability/logs/application.log"
}

# SRE Service Level Indicators (SLIs)
sre::sli::availability() {
    local service="$1"
    local status="${2:-success}" # success or failure
    
    local labels="service=\"${service}\",status=\"${status}\""
    sre::metrics::counter "sli_availability_total" 1 "$labels"
    
    log::trace "SLI availability recorded: $service $status"
}

sre::sli::latency() {
    local service="$1"
    local latency_seconds="$2"
    local operation="${3:-request}"
    
    local labels="service=\"${service}\",operation=\"${operation}\""
    sre::metrics::histogram "sli_latency_seconds" "$latency_seconds" "$labels"
    
    log::trace "SLI latency recorded: $service $operation ${latency_seconds}s"
}

sre::sli::throughput() {
    local service="$1"
    local requests_per_second="$2"
    local endpoint="${3:-default}"
    
    local labels="service=\"${service}\",endpoint=\"${endpoint}\""
    sre::metrics::gauge "sli_throughput_rps" "$requests_per_second" "$labels"
    
    log::trace "SLI throughput recorded: $service $endpoint ${requests_per_second}rps"
}

# Error budget tracking
sre::error_budget::calculate() {
    local service="$1"
    local slo_target="${2:-99.9}" # 99.9% availability SLO
    
    local total_requests=$(grep "sli_availability_total.*service=\"${service}\"" \
                          "${FRAMEWORK_DATA_DIR}/observability/metrics/prometheus.txt" | \
                          awk '{sum += $2} END {print sum+0}')
    
    local failed_requests=$(grep "sli_availability_total.*service=\"${service}\".*status=\"failure\"" \
                           "${FRAMEWORK_DATA_DIR}/observability/metrics/prometheus.txt" | \
                           awk '{sum += $2} END {print sum+0}')
    
    if [[ $total_requests -gt 0 ]]; then
        local success_rate=$(echo "scale=3; (($total_requests - $failed_requests) / $total_requests) * 100" | bc)
        local error_budget_remaining=$(echo "scale=3; $success_rate - $slo_target" | bc)
        
        echo "Service: $service"
        echo "Success Rate: ${success_rate}%"
        echo "SLO Target: ${slo_target}%"
        echo "Error Budget Remaining: ${error_budget_remaining}%"
        
        # Store error budget metrics
        sre::metrics::gauge "sre_error_budget_remaining" "$error_budget_remaining" "service=\"${service}\""
    else
        log::warn "No availability data found for service: $service"
    fi
}

# Alerting rules (Google SRE style)
sre::alerting::define_rule() {
    local rule_name="$1"
    local expression="$2"
    local duration="${3:-5m}"
    local severity="${4:-warning}"
    local description="$5"
    
    local alert_rule=$(cat << EOF
- alert: $rule_name
  expr: $expression
  for: $duration
  labels:
    severity: $severity
  annotations:
    description: "$description"
    summary: "SRE Alert: $rule_name"
EOF
)
    
    echo "$alert_rule" >> "${FRAMEWORK_DATA_DIR}/observability/config/alerting_rules.yml"
    log::info "Alert rule defined: $rule_name"
}

# Generate Prometheus configuration
sre::config::generate_prometheus() {
    local prometheus_config="${FRAMEWORK_DATA_DIR}/observability/config/prometheus.yml"
    
    cat > "$prometheus_config" << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alerting_rules.yml"

scrape_configs:
  - job_name: 'terminal-automation-framework'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
    scrape_interval: 5s
    
  - job_name: 'pushgateway'
    static_configs:
      - targets: ['localhost:9091']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - localhost:9093
EOF
    
    log::info "Prometheus configuration generated: $prometheus_config"
}

# Generate Grafana dashboard
sre::config::generate_grafana_dashboard() {
    local dashboard_config="${FRAMEWORK_DATA_DIR}/observability/config/grafana_dashboard.json"
    
    local dashboard=$(jq -n '{
        "dashboard": {
            "id": null,
            "title": "Terminal Automation Framework - SRE Dashboard",
            "tags": ["sre", "terminal-automation"],
            "timezone": "browser",
            "panels": [
                {
                    "id": 1,
                    "title": "Request Rate (Traffic)",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "rate(requests_total[5m])",
                            "legendFormat": "Requests/sec"
                        }
                    ]
                },
                {
                    "id": 2, 
                    "title": "Request Duration (Latency)",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.95, rate(request_duration_seconds_bucket[5m]))",
                            "legendFormat": "95th percentile"
                        }
                    ]
                },
                {
                    "id": 3,
                    "title": "Error Rate",
                    "type": "graph", 
                    "targets": [
                        {
                            "expr": "rate(errors_total[5m])",
                            "legendFormat": "Errors/sec"
                        }
                    ]
                },
                {
                    "id": 4,
                    "title": "Resource Utilization (Saturation)",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "resource_utilization",
                            "legendFormat": "Utilization %"
                        }
                    ]
                }
            ],
            "time": {
                "from": "now-1h",
                "to": "now"
            },
            "refresh": "5s"
        }
    }')
    
    echo "$dashboard" > "$dashboard_config"
    log::info "Grafana dashboard generated: $dashboard_config"
}

# Command-line interface for SRE observability
sre::observability::main() {
    local command="${1:-help}"
    shift || true
    
    case "$command" in
        init)
            sre::observability::init
            ;;
        metric)
            if [[ $# -lt 3 ]]; then
                echo "Usage: sre observability metric <type> <name> <value> [labels]"
                exit 1
            fi
            local metric_type="$1"
            local metric_name="$2" 
            local metric_value="$3"
            local metric_labels="${4:-}"
            
            case "$metric_type" in
                counter)
                    sre::metrics::counter "$metric_name" "$metric_value" "$metric_labels"
                    ;;
                gauge)
                    sre::metrics::gauge "$metric_name" "$metric_value" "$metric_labels"
                    ;;
                histogram)
                    sre::metrics::histogram "$metric_name" "$metric_value" "$metric_labels"
                    ;;
                *)
                    log::error "Unknown metric type: $metric_type"
                    exit 1
                    ;;
            esac
            ;;
        golden-signal)
            if [[ $# -lt 2 ]]; then
                echo "Usage: sre observability golden-signal <signal> <value> [labels]"
                exit 1
            fi
            sre::golden_signals::collect "$1" "$2" "${3:-}"
            ;;
        trace)
            local trace_command="$1"
            case "$trace_command" in
                start)
                    span_id=$(sre::tracing::span_start "$2" "${3:-}")
                    echo "Started span: $span_id"
                    ;;
                finish)
                    sre::tracing::span_finish "$2" "${3:-ok}"
                    ;;
                *)
                    echo "Usage: sre observability trace {start|finish} <span_name> [parent_span_id|status]"
                    exit 1
                    ;;
            esac
            ;;
        sli)
            local sli_type="$1"
            case "$sli_type" in
                availability)
                    sre::sli::availability "$2" "${3:-success}"
                    ;;
                latency)
                    sre::sli::latency "$2" "$3" "${4:-request}"
                    ;;
                throughput)
                    sre::sli::throughput "$2" "$3" "${4:-default}"
                    ;;
                *)
                    echo "Usage: sre observability sli {availability|latency|throughput} <service> <value> [operation]"
                    exit 1
                    ;;
            esac
            ;;
        error-budget)
            sre::error_budget::calculate "$1" "${2:-99.9}"
            ;;
        config)
            sre::observability::init
            sre::config::generate_prometheus
            sre::config::generate_grafana_dashboard
            ;;
        help|*)
            cat << EOF
SRE Observability - Google SRE-inspired observability stack

Usage: sre observability <command> [options]

Commands:
  init                                    Initialize observability stack
  metric <type> <name> <value> [labels]   Record metric (counter|gauge|histogram)
  golden-signal <signal> <value> [labels] Record golden signal (latency|traffic|errors|saturation)
  trace start <span_name> [parent_id]     Start distributed trace span
  trace finish <span_id> [status]         Finish distributed trace span
  sli <type> <service> <value> [operation] Record SLI (availability|latency|throughput)
  error-budget <service> [slo_target]     Calculate error budget
  config                                  Generate Prometheus and Grafana configs

Examples:
  sre observability init
  sre observability metric counter requests_total 1 'method="GET",status="200"'
  sre observability golden-signal latency 0.042 'service="api"'
  sre observability sli availability api-service success
  sre observability error-budget api-service 99.95

Based on Google SRE practices with OpenTelemetry, Prometheus, and Grafana integration.
EOF
            ;;
    esac
}

# Export functions for use by other modules
export -f sre::observability::init sre::golden_signals::collect
export -f sre::metrics::counter sre::metrics::gauge sre::metrics::histogram
export -f sre::tracing::span_start sre::tracing::span_finish
export -f sre::logging::structured sre::sli::availability sre::sli::latency sre::sli::throughput

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    sre::observability::main "$@"
fi