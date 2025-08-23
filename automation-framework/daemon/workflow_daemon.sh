#!/usr/bin/env bash

# Workflow Daemon - Persistent background service for workflow management
# Inspired by Google Bazel's daemon pattern for improved performance

set -euo pipefail

DAEMON_DIR="/tmp/workflow_daemon"
DAEMON_PID_FILE="$DAEMON_DIR/daemon.pid"
DAEMON_SOCKET="$DAEMON_DIR/daemon.sock"
DAEMON_LOG="$DAEMON_DIR/daemon.log"
DAEMON_STATE="$DAEMON_DIR/state.json"

# Circuit breaker configuration
CIRCUIT_BREAKER_THRESHOLD=5
CIRCUIT_BREAKER_TIMEOUT=60
CIRCUIT_BREAKER_STATE="$DAEMON_DIR/circuit_breaker.state"

# Initialize daemon directory
init_daemon_dir() {
    mkdir -p "$DAEMON_DIR"
    chmod 700 "$DAEMON_DIR"
}

# Check if daemon is running
is_daemon_running() {
    if [[ -f "$DAEMON_PID_FILE" ]]; then
        local pid=$(cat "$DAEMON_PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

# Start the daemon
start_daemon() {
    if is_daemon_running; then
        echo "Daemon already running (PID: $(cat $DAEMON_PID_FILE))"
        return 0
    fi
    
    init_daemon_dir
    
    # Start daemon in background
    (
        trap cleanup EXIT
        trap 'handle_signal TERM' TERM
        trap 'handle_signal INT' INT
        
        echo $$ > "$DAEMON_PID_FILE"
        echo "Daemon started (PID: $$)" >> "$DAEMON_LOG"
        
        # Initialize state
        echo '{"status":"running","start_time":"'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'","workflows":[]}' > "$DAEMON_STATE"
        
        # Main daemon loop
        while true; do
            # Process incoming commands via socket
            if [[ -p "$DAEMON_SOCKET" ]]; then
                while IFS= read -r cmd < "$DAEMON_SOCKET"; do
                    process_command "$cmd" >> "$DAEMON_LOG" 2>&1
                done
            else
                # Create named pipe if it doesn't exist
                mkfifo "$DAEMON_SOCKET" 2>/dev/null || true
            fi
            
            # Health check
            perform_health_check
            
            # Clean up old workflows
            cleanup_old_workflows
            
            sleep 1
        done
    ) &
    
    local daemon_pid=$!
    echo "$daemon_pid" > "$DAEMON_PID_FILE"
    echo "Workflow daemon started (PID: $daemon_pid)"
    
    # Wait for daemon to be ready
    sleep 2
    
    if is_daemon_running; then
        echo "Daemon successfully started"
        return 0
    else
        echo "Failed to start daemon"
        return 1
    fi
}

# Stop the daemon
stop_daemon() {
    if ! is_daemon_running; then
        echo "Daemon not running"
        return 0
    fi
    
    local pid=$(cat "$DAEMON_PID_FILE")
    echo "Stopping daemon (PID: $pid)..."
    
    kill -TERM "$pid" 2>/dev/null || true
    
    # Wait for daemon to stop
    local count=0
    while kill -0 "$pid" 2>/dev/null && [[ $count -lt 10 ]]; do
        sleep 1
        ((count++))
    done
    
    if kill -0 "$pid" 2>/dev/null; then
        echo "Force killing daemon..."
        kill -9 "$pid" 2>/dev/null || true
    fi
    
    rm -f "$DAEMON_PID_FILE"
    echo "Daemon stopped"
}

# Process incoming commands
process_command() {
    local cmd="$1"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    case "$cmd" in
        status)
            cat "$DAEMON_STATE"
            ;;
        execute:*)
            local workflow="${cmd#execute:}"
            execute_workflow "$workflow"
            ;;
        circuit_breaker:*)
            handle_circuit_breaker "${cmd#circuit_breaker:}"
            ;;
        health)
            perform_health_check
            ;;
        *)
            echo "Unknown command: $cmd"
            ;;
    esac
}

# Execute workflow with circuit breaker
execute_workflow() {
    local workflow="$1"
    
    # Check circuit breaker
    if is_circuit_open; then
        echo "Circuit breaker OPEN - workflow execution blocked"
        return 1
    fi
    
    echo "Executing workflow: $workflow"
    
    # Update state
    local state=$(cat "$DAEMON_STATE")
    state=$(echo "$state" | jq --arg wf "$workflow" --arg ts "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
        '.workflows += [{"name": $wf, "start_time": $ts, "status": "running"}]')
    echo "$state" > "$DAEMON_STATE"
    
    # Execute with timeout and monitoring
    (
        timeout 300 bash -c "$workflow" 2>&1
        local exit_code=$?
        
        if [[ $exit_code -eq 0 ]]; then
            record_success
        else
            record_failure
        fi
        
        return $exit_code
    ) &
    
    local workflow_pid=$!
    echo "Workflow started (PID: $workflow_pid)"
}

# Circuit breaker implementation
is_circuit_open() {
    if [[ ! -f "$CIRCUIT_BREAKER_STATE" ]]; then
        echo '{"state":"closed","failures":0,"last_failure":null}' > "$CIRCUIT_BREAKER_STATE"
        return 1
    fi
    
    local state=$(cat "$CIRCUIT_BREAKER_STATE" | jq -r '.state')
    [[ "$state" == "open" ]]
}

record_failure() {
    local state=$(cat "$CIRCUIT_BREAKER_STATE")
    local failures=$(echo "$state" | jq -r '.failures')
    ((failures++))
    
    if [[ $failures -ge $CIRCUIT_BREAKER_THRESHOLD ]]; then
        state=$(echo "$state" | jq --arg ts "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
            '.state = "open" | .failures = '$failures' | .last_failure = $ts | .open_until = "'$(date -u -d "+$CIRCUIT_BREAKER_TIMEOUT seconds" +"%Y-%m-%dT%H:%M:%SZ")'"')
        echo "Circuit breaker OPENED due to $failures failures"
    else
        state=$(echo "$state" | jq --arg ts "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
            '.failures = '$failures' | .last_failure = $ts')
    fi
    
    echo "$state" > "$CIRCUIT_BREAKER_STATE"
}

record_success() {
    local state=$(cat "$CIRCUIT_BREAKER_STATE")
    state=$(echo "$state" | jq '.state = "closed" | .failures = 0')
    echo "$state" > "$CIRCUIT_BREAKER_STATE"
}

# Health check
perform_health_check() {
    local mem_usage=$(ps aux | awk -v pid=$$ '$2 == pid {print $4}')
    local cpu_usage=$(ps aux | awk -v pid=$$ '$2 == pid {print $3}')
    
    echo "{\"timestamp\":\"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\",\"memory\":\"$mem_usage\",\"cpu\":\"$cpu_usage\",\"status\":\"healthy\"}"
}

# Clean up old workflows
cleanup_old_workflows() {
    # Remove workflows older than 1 hour
    local state=$(cat "$DAEMON_STATE")
    state=$(echo "$state" | jq --arg cutoff "$(date -u -d '-1 hour' +"%Y-%m-%dT%H:%M:%SZ")" \
        '.workflows = [.workflows[] | select(.start_time > $cutoff)]')
    echo "$state" > "$DAEMON_STATE"
}

# Signal handlers
handle_signal() {
    local signal="$1"
    echo "Received signal: $signal" >> "$DAEMON_LOG"
    cleanup
    exit 0
}

# Cleanup on exit
cleanup() {
    echo "Cleaning up daemon resources..." >> "$DAEMON_LOG"
    rm -f "$DAEMON_SOCKET" "$DAEMON_PID_FILE"
}

# Send command to daemon
send_command() {
    local cmd="$1"
    if ! is_daemon_running; then
        echo "Daemon not running. Starting..."
        start_daemon
    fi
    
    echo "$cmd" > "$DAEMON_SOCKET"
}

# Main command handler
main() {
    local action="${1:-status}"
    
    case "$action" in
        start)
            start_daemon
            ;;
        stop)
            stop_daemon
            ;;
        restart)
            stop_daemon
            sleep 2
            start_daemon
            ;;
        status)
            if is_daemon_running; then
                echo "Daemon running (PID: $(cat $DAEMON_PID_FILE))"
                send_command "status"
            else
                echo "Daemon not running"
            fi
            ;;
        execute)
            shift
            send_command "execute:$*"
            ;;
        health)
            send_command "health"
            ;;
        *)
            echo "Usage: $0 {start|stop|restart|status|execute|health}"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"