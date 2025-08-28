#!/usr/bin/env bash
#
# Netflix Chaos Engineering - Resilience patterns inspired by Netflix's chaos engineering practices
# Following Netflix 2025 practices with circuit breakers, bulkheads, and self-healing systems
#
set -euo pipefail

# Source framework dependencies
source "$(dirname "${BASH_SOURCE[0]}")/../core/bootstrap.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../core/logger.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../core/config.sh"

# Netflix Chaos Engineering Configuration
if [[ -z "${NETFLIX_CHAOS_VERSION:-}" ]]; then
    readonly NETFLIX_CHAOS_VERSION="1.0.0"
fi
if [[ -z "${CHAOS_MONKEY_ENABLED:-}" ]]; then
    readonly CHAOS_MONKEY_ENABLED="${CHAOS_ENABLED:-false}"
fi

# Circuit breaker states (Netflix Hystrix pattern)
readonly CIRCUIT_BREAKER_CLOSED="closed"
readonly CIRCUIT_BREAKER_OPEN="open" 
readonly CIRCUIT_BREAKER_HALF_OPEN="half_open"

# Netflix Chaos State (using variables for macOS bash compatibility)
NETFLIX_CHAOS_INITIALIZED=false
# Note: Using variable names with prefixes instead of associative arrays for macOS compatibility

# Sanitize service name for use as bash variable name
netflix::chaos::sanitize_name() {
    local name="$1"
    # Replace hyphens and other special characters with underscores
    echo "$name" | sed 's/[^a-zA-Z0-9]/_/g'
}

# Load circuit breaker state from persistent storage
netflix::circuit_breaker::load_state() {
    local service_name="$1"
    local safe_name=$(netflix::chaos::sanitize_name "$service_name")
    local state_file="${FRAMEWORK_DATA_DIR}/chaos/circuit-breakers/${service_name}.json"
    
    if [[ -f "$state_file" ]]; then
        local state=$(jq -r '.state' "$state_file" 2>/dev/null || echo "$CIRCUIT_BREAKER_CLOSED")
        local failures=$(jq -r '.failures // 0' "$state_file" 2>/dev/null || echo "0")
        local last_failure=$(jq -r '.last_failure // 0' "$state_file" 2>/dev/null || echo "0")
        local half_open_successes=$(jq -r '.half_open_successes // 0' "$state_file" 2>/dev/null || echo "0")
        local failure_threshold=$(jq -r '.failure_threshold // 5' "$state_file" 2>/dev/null || echo "5")
        local timeout=$(jq -r '.timeout // 60' "$state_file" 2>/dev/null || echo "60")
        local success_threshold=$(jq -r '.success_threshold // 3' "$state_file" 2>/dev/null || echo "3")
        
        eval "CB_${safe_name}_state='$state'"
        eval "CB_${safe_name}_failures=$failures"
        eval "CB_${safe_name}_last_failure=$last_failure"
        eval "CB_${safe_name}_half_open_successes=$half_open_successes"
        eval "CB_${safe_name}_failure_threshold=$failure_threshold"
        eval "CB_${safe_name}_timeout=$timeout"
        eval "CB_${safe_name}_success_threshold=$success_threshold"
        eval "CB_${safe_name}_original_name='$service_name'"
    fi
}

# Save circuit breaker state to persistent storage
netflix::circuit_breaker::save_state() {
    local service_name="$1"
    local safe_name=$(netflix::chaos::sanitize_name "$service_name")
    local state_file="${FRAMEWORK_DATA_DIR}/chaos/circuit-breakers/${service_name}.json"
    
    local state_var="CB_${safe_name}_state"
    local failures_var="CB_${safe_name}_failures"
    local last_failure_var="CB_${safe_name}_last_failure"
    local half_open_successes_var="CB_${safe_name}_half_open_successes"
    local failure_threshold_var="CB_${safe_name}_failure_threshold"
    local timeout_var="CB_${safe_name}_timeout"
    local success_threshold_var="CB_${safe_name}_success_threshold"
    
    local state="$(eval echo "\$${state_var}")"
    local failures="$(eval echo "\$${failures_var}")"
    local last_failure="$(eval echo "\$${last_failure_var}")"
    local half_open_successes="$(eval echo "\$${half_open_successes_var}")"
    local failure_threshold="$(eval echo "\$${failure_threshold_var}")"
    local timeout="$(eval echo "\$${timeout_var}")"
    local success_threshold="$(eval echo "\$${success_threshold_var}")"
    
    cat > "$state_file" << EOF
{
    "service": "$service_name",
    "state": "$state",
    "failures": $failures,
    "last_failure": $last_failure,
    "half_open_successes": $half_open_successes,
    "failure_threshold": $failure_threshold,
    "timeout": $timeout,
    "success_threshold": $success_threshold,
    "updated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
}

# Initialize Netflix-inspired chaos engineering
netflix::chaos::init() {
    if [[ "$NETFLIX_CHAOS_INITIALIZED" == "true" ]]; then
        return 0
    fi
    
    log::info "Initializing Netflix Chaos Engineering v${NETFLIX_CHAOS_VERSION}"
    
    # Initialize framework only if not already initialized
    if [[ "${FRAMEWORK_INITIALIZED:-false}" != "true" ]]; then
        framework::init
    fi
    config::load
    
    # Create chaos engineering directories
    local chaos_dir="${FRAMEWORK_DATA_DIR}/chaos"
    mkdir -p "${chaos_dir}"/{experiments,circuit-breakers,metrics,failures}
    
    # Initialize chaos monkey if enabled
    if [[ "$CHAOS_MONKEY_ENABLED" == "true" ]]; then
        netflix::chaos::monkey_init
    fi
    
    NETFLIX_CHAOS_INITIALIZED=true
    log::success "Netflix Chaos Engineering initialized"
}

# Circuit Breaker Pattern (Netflix Hystrix inspired)
netflix::circuit_breaker::create() {
    local service_name="$1"
    local failure_threshold="${2:-5}"        # failures before opening
    local timeout_seconds="${3:-60}"         # timeout before half-open
    local success_threshold="${4:-3}"        # successes to close from half-open
    
    log::info "Creating circuit breaker for service: $service_name"
    
    # Sanitize service name for variable use
    local safe_name=$(netflix::chaos::sanitize_name "$service_name")
    
    # Initialize circuit breaker state (using dynamic variable names)
    eval "CB_${safe_name}_state=\"$CIRCUIT_BREAKER_CLOSED\""
    eval "CB_${safe_name}_failures=0"
    eval "CB_${safe_name}_failure_threshold=\"$failure_threshold\""
    eval "CB_${safe_name}_timeout=\"$timeout_seconds\""
    eval "CB_${safe_name}_success_threshold=\"$success_threshold\""
    eval "CB_${safe_name}_last_failure=$(date +%s)"
    eval "CB_${safe_name}_half_open_successes=0"
    
    # Store original name mapping
    eval "CB_${safe_name}_original_name=\"$service_name\""
    
    # Save initial state to persistent storage
    netflix::circuit_breaker::save_state "$service_name"
    
    # Create circuit breaker config file
    cat > "${FRAMEWORK_DATA_DIR}/chaos/circuit-breakers/${service_name}.json" << EOF
{
    "service": "$service_name",
    "state": "$CIRCUIT_BREAKER_CLOSED", 
    "failure_threshold": $failure_threshold,
    "timeout_seconds": $timeout_seconds,
    "success_threshold": $success_threshold,
    "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "metrics": {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "circuit_open_count": 0
    }
}
EOF
    
    log::success "Circuit breaker created for: $service_name"
}

netflix::circuit_breaker::call() {
    local service_name="$1"
    local command="$2"
    local fallback_command="${3:-echo 'Service unavailable - fallback response'}"
    
    # Load current state from persistent storage
    netflix::circuit_breaker::load_state "$service_name"
    
    # Sanitize service name for variable use
    local safe_name=$(netflix::chaos::sanitize_name "$service_name")
    
    local state_var="CB_${safe_name}_state"
    local state="$(eval echo "\$${state_var}")"
    state="${state:-$CIRCUIT_BREAKER_CLOSED}"
    
    case "$state" in
        "$CIRCUIT_BREAKER_CLOSED")
            # Circuit is closed, attempt the call
            if eval "$command"; then
                netflix::circuit_breaker::on_success "$service_name"
                return 0
            else
                netflix::circuit_breaker::on_failure "$service_name"
                return 1
            fi
            ;;
        "$CIRCUIT_BREAKER_OPEN")
            # Circuit is open, check if timeout has passed
            local last_failure_var="CB_${safe_name}_last_failure"
            local timeout_var="CB_${safe_name}_timeout"
            local last_failure="$(eval echo "\$${last_failure_var}")"
            local timeout="$(eval echo "\$${timeout_var}")"
            local current_time=$(date +%s)
            
            if (( current_time - last_failure > timeout )); then
                # Move to half-open state
                eval "CB_${safe_name}_state='$CIRCUIT_BREAKER_HALF_OPEN'"
                log::info "Circuit breaker moving to half-open: $service_name"
                netflix::circuit_breaker::call "$service_name" "$command" "$fallback_command"
            else
                # Still open, use fallback
                log::warn "Circuit breaker open for $service_name, using fallback"
                eval "$fallback_command"
                return 1
            fi
            ;;
        "$CIRCUIT_BREAKER_HALF_OPEN")
            # Circuit is half-open, test with limited requests
            if eval "$command"; then
                netflix::circuit_breaker::on_half_open_success "$service_name"
                return 0
            else
                netflix::circuit_breaker::on_half_open_failure "$service_name"
                eval "$fallback_command"
                return 1
            fi
            ;;
    esac
}

netflix::circuit_breaker::on_success() {
    local service_name="$1"
    local safe_name=$(netflix::chaos::sanitize_name "$service_name")
    
    # Reset failure count
    eval "CB_${safe_name}_failures=0"
    
    # Save updated state
    netflix::circuit_breaker::save_state "$service_name"
    
    log::trace "Circuit breaker success for: $service_name"
}

netflix::circuit_breaker::on_failure() {
    local service_name="$1"
    local safe_name=$(netflix::chaos::sanitize_name "$service_name")
    local failures_var="CB_${safe_name}_failures"
    local threshold_var="CB_${safe_name}_failure_threshold"
    local current_failures="$(eval echo "\$${failures_var}")"
    local failure_threshold="$(eval echo "\$${threshold_var}")"
    
    # Increment failure count
    eval "CB_${safe_name}_failures=$((current_failures + 1))"
    eval "CB_${safe_name}_last_failure=$(date +%s)"
    
    # Check if threshold exceeded
    if (( current_failures + 1 >= failure_threshold )); then
        eval "CB_${safe_name}_state='$CIRCUIT_BREAKER_OPEN'"
        log::warn "Circuit breaker opened for: $service_name (failures: $((current_failures + 1)))"
    fi
    
    # Save updated state
    netflix::circuit_breaker::save_state "$service_name"
    
    log::trace "Circuit breaker failure for: $service_name (count: $((current_failures + 1)))"
}

netflix::circuit_breaker::on_half_open_success() {
    local service_name="$1"
    local safe_name=$(netflix::chaos::sanitize_name "$service_name")
    local successes_var="CB_${safe_name}_half_open_successes"
    local success_threshold_var="CB_${safe_name}_success_threshold"
    local successes="$(eval echo "\$${successes_var}")"
    local success_threshold="$(eval echo "\$${success_threshold_var}")"
    
    # Increment success count
    eval "CB_${safe_name}_half_open_successes=$((successes + 1))"
    
    # Check if we can close the circuit
    if (( successes + 1 >= success_threshold )); then
        eval "CB_${safe_name}_state='$CIRCUIT_BREAKER_CLOSED'"
        eval "CB_${safe_name}_failures=0"
        eval "CB_${safe_name}_half_open_successes=0"
        log::info "Circuit breaker closed for: $service_name (successes: $((successes + 1)))"
    fi
    
    # Save updated state
    netflix::circuit_breaker::save_state "$service_name"
}

netflix::circuit_breaker::on_half_open_failure() {
    local service_name="$1"
    local safe_name=$(netflix::chaos::sanitize_name "$service_name")
    
    # Return to open state
    eval "CB_${safe_name}_state='$CIRCUIT_BREAKER_OPEN'"
    eval "CB_${safe_name}_half_open_successes=0"
    eval "CB_${safe_name}_last_failure=$(date +%s)"
    
    # Save updated state
    netflix::circuit_breaker::save_state "$service_name"
    
    log::warn "Circuit breaker reopened for: $service_name"
}

# Bulkhead Pattern (Netflix isolation)
netflix::bulkhead::create() {
    local resource_name="$1"
    local max_concurrent="${2:-10}"
    local queue_size="${3:-50}"
    
    log::info "Creating bulkhead for resource: $resource_name"
    
    # Create resource pool directory
    local pool_dir="${FRAMEWORK_DATA_DIR}/chaos/bulkheads/${resource_name}"
    mkdir -p "$pool_dir"/{active,queue,completed}
    
    # Initialize bulkhead configuration
    cat > "${pool_dir}/config.json" << EOF
{
    "resource": "$resource_name",
    "max_concurrent": $max_concurrent,
    "queue_size": $queue_size,
    "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "metrics": {
        "total_requests": 0,
        "active_requests": 0,
        "queued_requests": 0,
        "rejected_requests": 0
    }
}
EOF
    
    log::success "Bulkhead created for: $resource_name"
}

netflix::bulkhead::execute() {
    local resource_name="$1"
    local command="$2"
    local request_id="${3:-$(openssl rand -hex 8)}"
    
    local pool_dir="${FRAMEWORK_DATA_DIR}/chaos/bulkheads/${resource_name}"
    local max_concurrent=$(jq -r '.max_concurrent' "${pool_dir}/config.json")
    local queue_size=$(jq -r '.queue_size' "${pool_dir}/config.json")
    
    # Check current active requests
    local active_count=$(find "${pool_dir}/active" -type f 2>/dev/null | wc -l)
    
    if (( active_count < max_concurrent )); then
        # Execute immediately
        netflix::bulkhead::_execute_request "$resource_name" "$command" "$request_id"
    else
        # Check queue capacity
        local queue_count=$(find "${pool_dir}/queue" -type f 2>/dev/null | wc -l)
        
        if (( queue_count < queue_size )); then
            # Queue the request
            echo "$command" > "${pool_dir}/queue/${request_id}"
            log::info "Request queued for $resource_name: $request_id"
        else
            # Reject the request
            log::warn "Request rejected for $resource_name (bulkhead full): $request_id"
            return 1
        fi
    fi
}

netflix::bulkhead::_execute_request() {
    local resource_name="$1"
    local command="$2"
    local request_id="$3"
    
    local pool_dir="${FRAMEWORK_DATA_DIR}/chaos/bulkheads/${resource_name}"
    
    # Mark as active
    echo "$command" > "${pool_dir}/active/${request_id}"
    
    # Execute in background
    {
        local start_time=$(date +%s)
        local result=0
        
        eval "$command" || result=$?
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        # Record completion
        cat > "${pool_dir}/completed/${request_id}" << EOF
{
    "request_id": "$request_id",
    "command": "$command",
    "result": $result,
    "duration": $duration,
    "completed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
        
        # Remove from active
        rm -f "${pool_dir}/active/${request_id}"
        
        # Process next queued request if any
        netflix::bulkhead::_process_queue "$resource_name"
        
    } &
    
    log::trace "Bulkhead executing request: $resource_name:$request_id"
}

netflix::bulkhead::_process_queue() {
    local resource_name="$1"
    local pool_dir="${FRAMEWORK_DATA_DIR}/chaos/bulkheads/${resource_name}"
    
    # Get next queued request
    local next_request=$(find "${pool_dir}/queue" -type f 2>/dev/null | head -n 1)
    
    if [[ -n "$next_request" ]]; then
        local request_id=$(basename "$next_request")
        local command=$(cat "$next_request")
        
        # Remove from queue
        rm -f "$next_request"
        
        # Execute
        netflix::bulkhead::_execute_request "$resource_name" "$command" "$request_id"
    fi
}

# Chaos Monkey (Netflix's famous failure injection)
netflix::chaos::monkey_init() {
    log::info "Initializing Chaos Monkey"
    
    # Create chaos monkey configuration
    cat > "${FRAMEWORK_DATA_DIR}/chaos/chaos_monkey.json" << EOF
{
    "enabled": true,
    "schedule": {
        "weekdays_only": true,
        "business_hours_only": true,
        "start_hour": 9,
        "end_hour": 17
    },
    "experiments": {
        "kill_process": {
            "enabled": true,
            "probability": 0.1,
            "targets": ["dev", "background_jobs"]
        },
        "network_latency": {
            "enabled": true,
            "probability": 0.05,
            "delay_ms": [100, 500, 1000]
        },
        "disk_space": {
            "enabled": true,
            "probability": 0.02
        }
    }
}
EOF
    
    log::success "Chaos Monkey initialized"
}

netflix::chaos::monkey_unleash() {
    local experiment_type="${1:-random}"
    
    if [[ "$CHAOS_MONKEY_ENABLED" != "true" ]]; then
        log::warn "Chaos Monkey is disabled"
        return 0
    fi
    
    # Check if we're in business hours (safety)
    local current_hour=$(date +%H)
    if (( current_hour < 9 || current_hour > 17 )); then
        log::info "Chaos Monkey sleeping (outside business hours)"
        return 0
    fi
    
    log::warn "🐒 Chaos Monkey is unleashing chaos: $experiment_type"
    
    case "$experiment_type" in
        "kill_process"|"random")
            netflix::chaos::experiment_kill_process
            ;;
        "network_latency")
            netflix::chaos::experiment_network_latency
            ;;
        "disk_space")
            netflix::chaos::experiment_disk_space
            ;;
        "cpu_stress")
            netflix::chaos::experiment_cpu_stress
            ;;
        *)
            log::error "Unknown chaos experiment: $experiment_type"
            return 1
            ;;
    esac
}

netflix::chaos::experiment_kill_process() {
    # Find non-critical processes to terminate
    local target_processes=("sleep" "cat" "tail")
    
    for process in "${target_processes[@]}"; do
        local pids=($(pgrep "$process" 2>/dev/null || true))
        
        if [[ ${#pids[@]} -gt 0 ]]; then
            local random_pid=${pids[$RANDOM % ${#pids[@]}]}
            log::warn "🐒 Chaos Monkey killing process: $process (PID: $random_pid)"
            kill -TERM "$random_pid" 2>/dev/null || true
            
            # Record chaos experiment
            netflix::chaos::record_experiment "kill_process" "Terminated $process PID $random_pid"
            return 0
        fi
    done
    
    log::info "🐒 Chaos Monkey: No suitable processes to kill"
}

netflix::chaos::experiment_network_latency() {
    # Simulate network latency (requires root for real implementation)
    local delay_ms=${1:-$((100 + RANDOM % 400))}
    
    log::warn "🐒 Chaos Monkey simulating ${delay_ms}ms network latency"
    
    # In a real implementation, this would use tc (traffic control) or similar
    # For simulation, we'll just add delays to network-related commands
    export CHAOS_NETWORK_DELAY_MS="$delay_ms"
    
    netflix::chaos::record_experiment "network_latency" "Added ${delay_ms}ms delay"
    
    # Remove delay after 60 seconds
    (sleep 60; unset CHAOS_NETWORK_DELAY_MS) &
}

netflix::chaos::experiment_disk_space() {
    local temp_file="/tmp/chaos_monkey_disk_fill_$$"
    local size_mb=$((10 + RANDOM % 90)) # 10-100MB
    
    log::warn "🐒 Chaos Monkey filling ${size_mb}MB of disk space"
    
    # Create temporary file to consume disk space
    dd if=/dev/zero of="$temp_file" bs=1M count="$size_mb" 2>/dev/null
    
    netflix::chaos::record_experiment "disk_space" "Filled ${size_mb}MB disk space"
    
    # Clean up after 5 minutes
    (sleep 300; rm -f "$temp_file") &
}

netflix::chaos::experiment_cpu_stress() {
    local duration="${1:-30}"
    local cores="${2:-1}"
    
    log::warn "🐒 Chaos Monkey stressing $cores CPU cores for ${duration}s"
    
    # CPU stress simulation
    for ((i=0; i<cores; i++)); do
        (
            end_time=$((SECONDS + duration))
            while (( SECONDS < end_time )); do
                : # Busy loop
            done
        ) &
    done
    
    netflix::chaos::record_experiment "cpu_stress" "Stressed $cores cores for ${duration}s"
}

netflix::chaos::record_experiment() {
    local experiment_type="$1"
    local description="$2"
    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    
    local experiment_id="chaos_$(date +%s)_$$"
    
    cat > "${FRAMEWORK_DATA_DIR}/chaos/experiments/${experiment_id}.json" << EOF
{
    "id": "$experiment_id",
    "type": "$experiment_type", 
    "description": "$description",
    "timestamp": "$timestamp",
    "status": "executed"
}
EOF
    
    eval "CHAOS_EXP_${experiment_id}='$experiment_type:$description'"
}

# Self-healing system (Netflix resilience)
netflix::self_healing::monitor() {
    local service_name="$1"
    local health_check_command="$2"
    local recovery_command="$3"
    local check_interval="${4:-60}"
    
    log::info "Starting self-healing monitor for: $service_name"
    
    # Sanitize service name for variable use
    local safe_name=$(netflix::chaos::sanitize_name "$service_name")
    
    # Background monitoring loop
    {
        while true; do
            if eval "$health_check_command"; then
                eval "SVC_HEALTH_${safe_name}='healthy'"
                log::trace "Health check passed: $service_name"
            else
                log::warn "Health check failed: $service_name"
                eval "SVC_HEALTH_${safe_name}='unhealthy'"
                
                # Attempt recovery
                log::info "Attempting self-healing for: $service_name"
                if eval "$recovery_command"; then
                    log::success "Self-healing successful: $service_name"
                    eval "SVC_HEALTH_${safe_name}='recovered'"
                else
                    log::error "Self-healing failed: $service_name"
                    eval "SVC_HEALTH_${safe_name}='failed'"
                fi
            fi
            
            sleep "$check_interval"
        done
    } &
    
    local monitor_pid=$!
    echo "$monitor_pid" > "${FRAMEWORK_DATA_DIR}/chaos/monitors/${service_name}.pid"
    
    log::success "Self-healing monitor started for: $service_name (PID: $monitor_pid)"
}

# Command-line interface
netflix::chaos::main() {
    local command="${1:-help}"
    shift || true
    
    case "$command" in
        init)
            netflix::chaos::init
            ;;
        circuit-breaker)
            local cb_command="${1:-help}"
            shift || true
            case "$cb_command" in
                create)
                    netflix::circuit_breaker::create "$@"
                    ;;
                call)
                    if [[ $# -lt 2 ]]; then
                        echo "Usage: netflix chaos circuit-breaker call <service> <command> [fallback]"
                        exit 1
                    fi
                    netflix::circuit_breaker::call "$@"
                    ;;
                status)
                    local service="${1:-all}"
                    if [[ "$service" == "all" ]]; then
                        # List all circuit breaker states from persistent storage
                        find "${FRAMEWORK_DATA_DIR}/chaos/circuit-breakers" -name "*.json" 2>/dev/null | while read -r state_file; do
                            if [[ -f "$state_file" ]]; then
                                local service_name=$(jq -r '.service' "$state_file")
                                local state=$(jq -r '.state' "$state_file")
                                local failures=$(jq -r '.failures // 0' "$state_file")
                                echo "$service_name: $state (failures: $failures)"
                            fi
                        done
                    else
                        local state_file="${FRAMEWORK_DATA_DIR}/chaos/circuit-breakers/${service}.json"
                        if [[ -f "$state_file" ]]; then
                            local state=$(jq -r '.state' "$state_file")
                            local failures=$(jq -r '.failures // 0' "$state_file")
                            echo "$service: $state (failures: $failures)"
                        else
                            echo "$service: not found"
                        fi
                    fi
                    ;;
                *)
                    echo "Usage: netflix chaos circuit-breaker {create|call|status}"
                    exit 1
                    ;;
            esac
            ;;
        bulkhead)
            local bh_command="${1:-help}"
            shift || true
            case "$bh_command" in
                create)
                    netflix::bulkhead::create "$@"
                    ;;
                execute)
                    if [[ $# -lt 2 ]]; then
                        echo "Usage: netflix chaos bulkhead execute <resource> <command> [request_id]"
                        exit 1
                    fi
                    netflix::bulkhead::execute "$@"
                    ;;
                *)
                    echo "Usage: netflix chaos bulkhead {create|execute}"
                    exit 1
                    ;;
            esac
            ;;
        monkey)
            local monkey_command="${1:-unleash}"
            case "$monkey_command" in
                unleash)
                    netflix::chaos::monkey_unleash "${1:-random}"
                    ;;
                enable)
                    export CHAOS_MONKEY_ENABLED=true
                    log::info "🐒 Chaos Monkey enabled"
                    ;;
                disable)
                    export CHAOS_MONKEY_ENABLED=false
                    log::info "🐒 Chaos Monkey disabled"
                    ;;
                *)
                    echo "Usage: netflix chaos monkey {unleash|enable|disable}"
                    exit 1
                    ;;
            esac
            ;;
        self-healing)
            if [[ $# -lt 3 ]]; then
                echo "Usage: netflix chaos self-healing <service> <health_check> <recovery_command> [interval]"
                exit 1
            fi
            netflix::self_healing::monitor "$@"
            ;;
        experiments)
            log::info "Recent Chaos Experiments:"
            find "${FRAMEWORK_DATA_DIR}/chaos/experiments" -name "*.json" -mtime -1 2>/dev/null | \
            sort -r | head -10 | while read -r exp_file; do
                local exp_type=$(jq -r '.type' "$exp_file")
                local exp_desc=$(jq -r '.description' "$exp_file")
                local exp_time=$(jq -r '.timestamp' "$exp_file")
                echo "  [$exp_time] $exp_type: $exp_desc"
            done
            ;;
        help|*)
            cat << EOF
Netflix Chaos Engineering - Resilience patterns inspired by Netflix

Usage: netflix chaos <command> [options]

Commands:
  init                                    Initialize chaos engineering
  circuit-breaker create <service> [threshold] [timeout] [success_threshold]
  circuit-breaker call <service> <command> [fallback]
  circuit-breaker status [service]       Show circuit breaker states
  bulkhead create <resource> [max_concurrent] [queue_size]
  bulkhead execute <resource> <command> [request_id]
  monkey unleash [experiment_type]       Unleash chaos monkey
  monkey enable/disable                  Enable/disable chaos monkey
  self-healing <service> <health_check> <recovery> [interval]
  experiments                           Show recent chaos experiments

Experiment Types:
  kill_process, network_latency, disk_space, cpu_stress

Examples:
  netflix chaos init
  netflix chaos circuit-breaker create api-service 3 30 2
  netflix chaos circuit-breaker call api-service 'curl -f http://api/' 'echo "Service down"'
  netflix chaos bulkhead create database 5 20
  netflix chaos monkey unleash kill_process
  netflix chaos self-healing web-server 'curl -f http://localhost/' './restart-server.sh'

Following Netflix patterns:
- Circuit breakers for cascading failure prevention  
- Bulkheads for resource isolation
- Chaos Monkey for failure injection
- Self-healing for automatic recovery
EOF
            ;;
    esac
}

# Export functions for use by other modules
export -f netflix::chaos::init netflix::circuit_breaker::create netflix::circuit_breaker::call
export -f netflix::bulkhead::create netflix::bulkhead::execute netflix::self_healing::monitor

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    netflix::chaos::main "$@"
fi