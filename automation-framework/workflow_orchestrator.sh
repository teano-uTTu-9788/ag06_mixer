#!/usr/bin/env bash

# Workflow Orchestrator - Main entry point for the automation framework
# Integrates all components for comprehensive workflow management

set -euo pipefail

# Framework paths
FRAMEWORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAEMON_DIR="$FRAMEWORK_DIR/daemon"
MONITORING_DIR="$FRAMEWORK_DIR/monitoring"
CHAOS_DIR="$FRAMEWORK_DIR/chaos"
CONFIG_DIR="$FRAMEWORK_DIR/config"

# Configuration
ORCHESTRATOR_CONFIG="$CONFIG_DIR/orchestrator.conf"
ORCHESTRATOR_LOG="/tmp/orchestrator.log"
ORCHESTRATOR_STATE="/tmp/orchestrator.state"

# Components (using functions for compatibility)
get_component() {
    case "$1" in
        daemon) echo "$DAEMON_DIR/workflow_daemon.sh" ;;
        hermetic) echo "$DAEMON_DIR/hermetic_env.sh" ;;
        parallel) echo "$DAEMON_DIR/parallel_executor.sh" ;;
        metrics) echo "$MONITORING_DIR/metrics_collector.py" ;;
        chaos) echo "$CHAOS_DIR/chaos_test.sh" ;;
    esac
}

# Initialize orchestrator
init_orchestrator() {
    echo "Initializing Workflow Orchestrator..."
    
    # Create directories
    mkdir -p "$CONFIG_DIR" /tmp/orchestrator
    
    # Create default configuration if not exists
    if [[ ! -f "$ORCHESTRATOR_CONFIG" ]]; then
        create_default_config
    fi
    
    # Initialize state
    echo '{
        "status": "initialized",
        "components": {},
        "workflows": [],
        "start_time": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'"
    }' > "$ORCHESTRATOR_STATE"
    
    # Make all scripts executable
    chmod +x "$DAEMON_DIR"/*.sh "$CHAOS_DIR"/*.sh 2>/dev/null || true
    
    echo "Orchestrator initialized successfully"
}

# Create default configuration
create_default_config() {
    cat > "$ORCHESTRATOR_CONFIG" <<EOF
# Workflow Orchestrator Configuration

# Component settings
DAEMON_ENABLED=true
HERMETIC_ENABLED=true
PARALLEL_ENABLED=true
METRICS_ENABLED=true
CHAOS_ENABLED=false

# Performance settings
MAX_PARALLEL_WORKFLOWS=10
MAX_PARALLEL_WORKERS=8
WORKER_TIMEOUT=300

# Monitoring settings
METRICS_INTERVAL=10
METRICS_PORT=8080
METRICS_RETENTION_DAYS=7

# Hermetic environment settings
DEFAULT_HERMETIC_ENV=production
HERMETIC_CACHE_SIZE=1024

# Chaos testing settings (disabled by default)
CHAOS_PROBABILITY=0.01
CHAOS_MAX_DURATION=60

# Circuit breaker settings
CIRCUIT_BREAKER_THRESHOLD=5
CIRCUIT_BREAKER_TIMEOUT=60

# Logging settings
LOG_LEVEL=INFO
LOG_MAX_SIZE=100M
LOG_RETENTION_DAYS=30
EOF
    
    echo "Default configuration created: $ORCHESTRATOR_CONFIG"
}

# Start all components
start_components() {
    echo "Starting framework components..."
    
    source "$ORCHESTRATOR_CONFIG"
    
    local started=0
    local failed=0
    
    # Start workflow daemon
    if [[ "$DAEMON_ENABLED" == "true" ]]; then
        echo "Starting workflow daemon..."
        if "$(get_component daemon)" start; then
            update_component_state "daemon" "running"
            ((started++))
        else
            update_component_state "daemon" "failed"
            ((failed++))
        fi
    fi
    
    # Start parallel executor
    if [[ "$PARALLEL_ENABLED" == "true" ]]; then
        echo "Starting parallel executor..."
        if "$(get_component parallel)" start; then
            update_component_state "parallel" "running"
            ((started++))
        else
            update_component_state "parallel" "failed"
            ((failed++))
        fi
    fi
    
    # Start metrics collector
    if [[ "$METRICS_ENABLED" == "true" ]]; then
        echo "Starting metrics collector..."
        python3 "$(get_component metrics)" > /tmp/metrics_collector.log 2>&1 &
        if [[ $? -eq 0 ]]; then
            update_component_state "metrics" "running"
            ((started++))
            echo "Metrics dashboard available at http://localhost:$METRICS_PORT"
        else
            update_component_state "metrics" "failed"
            ((failed++))
        fi
    fi
    
    echo ""
    echo "Components started: $started"
    echo "Components failed: $failed"
    
    if [[ $failed -eq 0 ]]; then
        echo "All components started successfully!"
        return 0
    else
        echo "Some components failed to start"
        return 1
    fi
}

# Stop all components
stop_components() {
    echo "Stopping framework components..."
    
    local stopped=0
    
    # Stop workflow daemon
    if "$(get_component daemon)" stop 2>/dev/null; then
        update_component_state "daemon" "stopped"
        ((stopped++))
    fi
    
    # Stop parallel executor
    if "$(get_component parallel)" stop 2>/dev/null; then
        update_component_state "parallel" "stopped"
        ((stopped++))
    fi
    
    # Stop metrics collector
    if pkill -f "metrics_collector.py" 2>/dev/null; then
        update_component_state "metrics" "stopped"
        ((stopped++))
    fi
    
    echo "Components stopped: $stopped"
}

# Update component state
update_component_state() {
    local component="$1"
    local status="$2"
    
    local state=$(cat "$ORCHESTRATOR_STATE")
    state=$(echo "$state" | jq --arg comp "$component" --arg stat "$status" \
        '.components[$comp] = {"status": $stat, "updated": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'"}')
    echo "$state" > "$ORCHESTRATOR_STATE"
}

# Execute workflow
execute_workflow() {
    local workflow_file="$1"
    local environment="${2:-default}"
    
    if [[ ! -f "$workflow_file" ]]; then
        echo "Workflow file not found: $workflow_file"
        return 1
    fi
    
    echo "Executing workflow: $workflow_file"
    echo "Environment: $environment"
    
    # Create hermetic environment if enabled
    source "$ORCHESTRATOR_CONFIG"
    if [[ "$HERMETIC_ENABLED" == "true" ]]; then
        echo "Creating hermetic environment..."
        "$(get_component hermetic)" create "$environment"
        
        # Activate environment
        source "/tmp/hermetic_envs/$environment/activate"
    fi
    
    # Parse workflow file
    local workflow_name=$(basename "$workflow_file" .yaml)
    local workflow_id="wf_$(date +%s%N)"
    
    # Record workflow start
    record_workflow_start "$workflow_id" "$workflow_name"
    
    # Execute workflow steps
    if [[ "$workflow_file" == *.yaml ]] || [[ "$workflow_file" == *.yml ]]; then
        execute_yaml_workflow "$workflow_file" "$workflow_id"
    elif [[ "$workflow_file" == *.sh ]]; then
        execute_shell_workflow "$workflow_file" "$workflow_id"
    else
        echo "Unsupported workflow format"
        return 1
    fi
    
    # Record workflow completion
    record_workflow_end "$workflow_id" "$?"
}

# Execute YAML workflow
execute_yaml_workflow() {
    local workflow_file="$1"
    local workflow_id="$2"
    
    echo "Parsing YAML workflow..."
    
    # Use Python to parse YAML
    python3 -c "
import yaml
import json
import sys

with open('$workflow_file', 'r') as f:
    workflow = yaml.safe_load(f)
    
# Convert to JSON for shell processing
print(json.dumps(workflow))
" > "/tmp/${workflow_id}.json"
    
    local workflow=$(cat "/tmp/${workflow_id}.json")
    
    # Extract workflow properties
    local name=$(echo "$workflow" | jq -r '.name')
    local mode=$(echo "$workflow" | jq -r '.execution_mode // "sequential"')
    
    echo "Workflow: $name"
    echo "Execution mode: $mode"
    
    # Execute steps based on mode
    case "$mode" in
        sequential)
            execute_sequential_steps "$workflow" "$workflow_id"
            ;;
        parallel)
            execute_parallel_steps "$workflow" "$workflow_id"
            ;;
        conditional)
            execute_conditional_steps "$workflow" "$workflow_id"
            ;;
        *)
            echo "Unknown execution mode: $mode"
            return 1
            ;;
    esac
}

# Execute shell workflow
execute_shell_workflow() {
    local workflow_file="$1"
    local workflow_id="$2"
    
    echo "Executing shell workflow..."
    
    # Execute with monitoring
    (
        export WORKFLOW_ID="$workflow_id"
        bash "$workflow_file"
    ) 2>&1 | tee "/tmp/${workflow_id}.log"
    
    return ${PIPESTATUS[0]}
}

# Execute sequential steps
execute_sequential_steps() {
    local workflow="$1"
    local workflow_id="$2"
    
    echo "Executing steps sequentially..."
    
    local steps=$(echo "$workflow" | jq -r '.steps[]')
    local step_count=$(echo "$workflow" | jq '.steps | length')
    local current=0
    
    echo "$workflow" | jq -c '.steps[]' | while read -r step; do
        ((current++))
        
        local step_name=$(echo "$step" | jq -r '.name')
        local step_command=$(echo "$step" | jq -r '.command')
        
        echo ""
        echo "[$current/$step_count] Executing step: $step_name"
        
        # Execute step
        if eval "$step_command"; then
            echo "✓ Step completed: $step_name"
        else
            echo "✗ Step failed: $step_name"
            return 1
        fi
    done
}

# Execute parallel steps
execute_parallel_steps() {
    local workflow="$1"
    local workflow_id="$2"
    
    echo "Executing steps in parallel..."
    
    # Add all steps to parallel queue
    echo "$workflow" | jq -c '.steps[]' | while read -r step; do
        local step_name=$(echo "$step" | jq -r '.name')
        local step_command=$(echo "$step" | jq -r '.command')
        local priority=$(echo "$step" | jq -r '.priority // 5')
        
        "$(get_component parallel)" add "$step_command" "$priority"
    done
    
    # Wait for completion
    while [[ $(find /tmp/parallel_work_queue -type f 2>/dev/null | wc -l) -gt 0 ]]; do
        sleep 2
    done
    
    # Get results
    "$(get_component parallel)" stats
}

# Execute conditional steps
execute_conditional_steps() {
    local workflow="$1"
    local workflow_id="$2"
    
    echo "Executing conditional steps..."
    
    echo "$workflow" | jq -c '.steps[]' | while read -r step; do
        local step_name=$(echo "$step" | jq -r '.name')
        local step_command=$(echo "$step" | jq -r '.command')
        local condition=$(echo "$step" | jq -r '.condition // ""')
        
        # Check condition
        if [[ -n "$condition" ]]; then
            if ! eval "$condition"; then
                echo "Skipping step (condition not met): $step_name"
                continue
            fi
        fi
        
        echo "Executing step: $step_name"
        eval "$step_command"
    done
}

# Record workflow start
record_workflow_start() {
    local workflow_id="$1"
    local workflow_name="$2"
    
    local state=$(cat "$ORCHESTRATOR_STATE")
    state=$(echo "$state" | jq --arg id "$workflow_id" --arg name "$workflow_name" \
        '.workflows += [{"id": $id, "name": $name, "status": "running", "start_time": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'"}]')
    echo "$state" > "$ORCHESTRATOR_STATE"
}

# Record workflow end
record_workflow_end() {
    local workflow_id="$1"
    local exit_code="$2"
    
    local status="success"
    if [[ $exit_code -ne 0 ]]; then
        status="failed"
    fi
    
    local state=$(cat "$ORCHESTRATOR_STATE")
    state=$(echo "$state" | jq --arg id "$workflow_id" --arg stat "$status" \
        '(.workflows[] | select(.id == $id)) |= (. + {"status": $stat, "end_time": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'"})')
    echo "$state" > "$ORCHESTRATOR_STATE"
}

# Health check
health_check() {
    echo "Performing health check..."
    
    local healthy=0
    local unhealthy=0
    
    # Check daemon
    if "$(get_component daemon)" status >/dev/null 2>&1; then
        echo "✓ Workflow daemon: healthy"
        ((healthy++))
    else
        echo "✗ Workflow daemon: unhealthy"
        ((unhealthy++))
    fi
    
    # Check parallel executor
    if [[ -d /tmp/parallel_workers ]]; then
        echo "✓ Parallel executor: healthy"
        ((healthy++))
    else
        echo "✗ Parallel executor: unhealthy"
        ((unhealthy++))
    fi
    
    # Check metrics collector
    if curl -s http://localhost:8080/metrics >/dev/null 2>&1; then
        echo "✓ Metrics collector: healthy"
        ((healthy++))
    else
        echo "✗ Metrics collector: unhealthy"
        ((unhealthy++))
    fi
    
    echo ""
    echo "Healthy components: $healthy"
    echo "Unhealthy components: $unhealthy"
    
    if [[ $unhealthy -eq 0 ]]; then
        echo "System is healthy ✓"
        return 0
    else
        echo "System has issues ✗"
        return 1
    fi
}

# Show dashboard
show_dashboard() {
    clear
    
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║           WORKFLOW AUTOMATION FRAMEWORK DASHBOARD            ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Component status
    echo "COMPONENTS:"
    echo "──────────────────────────────────────────────────────────────"
    
    local state=$(cat "$ORCHESTRATOR_STATE" 2>/dev/null || echo '{}')
    
    for component in daemon parallel metrics; do
        local status=$(echo "$state" | jq -r ".components.$component.status // \"unknown\"")
        local icon="⚪"
        
        case "$status" in
            running) icon="🟢" ;;
            stopped) icon="🔴" ;;
            failed) icon="🟡" ;;
        esac
        
        printf "  %s %-15s %s\n" "$icon" "$component" "$status"
    done
    
    echo ""
    echo "WORKFLOWS:"
    echo "──────────────────────────────────────────────────────────────"
    
    # Recent workflows
    echo "$state" | jq -r '.workflows[-5:] | reverse | .[] | "  \(.status | ascii_upcase | .[0:1]) \(.name) (\(.id))"' 2>/dev/null || echo "  No workflows"
    
    echo ""
    echo "METRICS:"
    echo "──────────────────────────────────────────────────────────────"
    
    # System metrics
    local cpu=$(ps aux | awk '{sum+=$3} END {print sum}')
    local mem=$(ps aux | awk '{sum+=$4} END {print sum}')
    
    printf "  CPU Usage: %.1f%%\n" "$cpu"
    printf "  Memory Usage: %.1f%%\n" "$mem"
    
    echo ""
    echo "Dashboard URL: http://localhost:8080"
    echo "──────────────────────────────────────────────────────────────"
}

# Main menu
show_menu() {
    cat <<EOF
Workflow Automation Framework

Usage: $0 <command> [options]

Commands:
  init                Initialize orchestrator
  start               Start all components
  stop                Stop all components
  restart             Restart all components
  
  execute <file>      Execute workflow file
  status              Show component status
  health              Perform health check
  dashboard           Show live dashboard
  
  config              Edit configuration
  logs                Show component logs
  clean               Clean all temporary files
  
  chaos <test>        Run chaos engineering test
  benchmark           Run performance benchmark
  
  help                Show this help

Examples:
  $0 start
  $0 execute workflows/deploy.yaml
  $0 chaos network
  $0 dashboard

Configuration: $ORCHESTRATOR_CONFIG
EOF
}

# Main command handler
main() {
    local action="${1:-help}"
    shift || true
    
    case "$action" in
        init)
            init_orchestrator
            ;;
        start)
            init_orchestrator
            start_components
            ;;
        stop)
            stop_components
            ;;
        restart)
            stop_components
            sleep 2
            start_components
            ;;
        execute)
            execute_workflow "$@"
            ;;
        status)
            cat "$ORCHESTRATOR_STATE" | jq .
            ;;
        health)
            health_check
            ;;
        dashboard)
            while true; do
                show_dashboard
                sleep 5
            done
            ;;
        config)
            ${EDITOR:-vi} "$ORCHESTRATOR_CONFIG"
            ;;
        logs)
            tail -f "$ORCHESTRATOR_LOG" /tmp/workflow_daemon/daemon.log /tmp/metrics_collector.log 2>/dev/null
            ;;
        clean)
            stop_components
            rm -rf /tmp/orchestrator /tmp/workflow_daemon /tmp/parallel_* /tmp/hermetic_* /tmp/chaos_*
            echo "Cleaned all temporary files"
            ;;
        chaos)
            "$(get_component chaos)" "$@"
            ;;
        benchmark)
            echo "Running performance benchmark..."
            time {
                for i in {1..100}; do
                    "$(get_component parallel)" add "echo Test $i" 5
                done
                "$(get_component parallel)" stats
            }
            ;;
        help)
            show_menu
            ;;
        *)
            echo "Unknown command: $action"
            show_menu
            exit 1
            ;;
    esac
}

# Run main function
main "$@"