#!/usr/bin/env bash

# Parallel Execution Framework - Advanced concurrent task execution
# Implements work-stealing queue pattern for optimal performance

set -euo pipefail

# Configuration
MAX_WORKERS=${MAX_WORKERS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}
WORK_QUEUE="/tmp/parallel_work_queue"
RESULT_DIR="/tmp/parallel_results"
WORKER_DIR="/tmp/parallel_workers"
STATS_FILE="/tmp/parallel_stats.json"

# Initialize directories
init_parallel_env() {
    mkdir -p "$WORK_QUEUE" "$RESULT_DIR" "$WORKER_DIR"
    
    # Initialize statistics
    echo '{
        "start_time": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'",
        "total_tasks": 0,
        "completed_tasks": 0,
        "failed_tasks": 0,
        "workers": '$MAX_WORKERS',
        "tasks": []
    }' > "$STATS_FILE"
}

# Add task to work queue
add_task() {
    local task_id="task_$(date +%s%N)"
    local task_cmd="$1"
    local priority="${2:-5}"  # Default priority 5 (1=highest, 10=lowest)
    local dependencies="${3:-}"
    
    local task_file="$WORK_QUEUE/${priority}_${task_id}"
    
    cat > "$task_file" <<EOF
{
    "id": "$task_id",
    "command": "$task_cmd",
    "priority": $priority,
    "dependencies": "$dependencies",
    "status": "pending",
    "created": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF
    
    # Update statistics
    local stats=$(cat "$STATS_FILE")
    stats=$(echo "$stats" | jq '.total_tasks += 1')
    echo "$stats" > "$STATS_FILE"
    
    echo "Task added: $task_id (priority: $priority)"
    return 0
}

# Worker process
worker_process() {
    local worker_id="$1"
    local worker_pid=$$
    local worker_file="$WORKER_DIR/worker_${worker_id}.pid"
    
    echo "$worker_pid" > "$worker_file"
    
    echo "[Worker $worker_id] Started (PID: $worker_pid)"
    
    while true; do
        # Try to get next task (work-stealing)
        local task_file=$(find "$WORK_QUEUE" -type f -name "*.task*" 2>/dev/null | sort | head -n1)
        
        if [[ -z "$task_file" ]]; then
            # No tasks available, try to steal from other workers
            task_file=$(steal_task "$worker_id")
            
            if [[ -z "$task_file" ]]; then
                # No work available, wait
                sleep 0.5
                continue
            fi
        fi
        
        # Atomic task acquisition
        local lock_file="${task_file}.lock"
        if ! (set -C; echo "$worker_id" > "$lock_file") 2>/dev/null; then
            # Another worker got this task
            continue
        fi
        
        # Process task
        process_task "$worker_id" "$task_file"
        
        # Clean up
        rm -f "$task_file" "$lock_file"
        
        # Check if should continue
        if [[ -f "$WORKER_DIR/stop" ]]; then
            break
        fi
    done
    
    echo "[Worker $worker_id] Stopped"
    rm -f "$worker_file"
}

# Process individual task
process_task() {
    local worker_id="$1"
    local task_file="$2"
    
    local task=$(cat "$task_file")
    local task_id=$(echo "$task" | jq -r '.id')
    local task_cmd=$(echo "$task" | jq -r '.command')
    local dependencies=$(echo "$task" | jq -r '.dependencies')
    
    echo "[Worker $worker_id] Processing task: $task_id"
    
    # Check dependencies
    if [[ -n "$dependencies" && "$dependencies" != "null" ]]; then
        if ! check_dependencies "$dependencies"; then
            echo "[Worker $worker_id] Dependencies not met for $task_id, requeuing"
            # Requeue with lower priority
            local new_priority=10
            mv "$task_file" "$WORK_QUEUE/${new_priority}_${task_id}.requeued"
            return 1
        fi
    fi
    
    # Execute task
    local start_time=$(date +%s%N)
    local result_file="$RESULT_DIR/${task_id}.result"
    local exit_code=0
    
    # Run task with timeout
    if timeout 300 bash -c "$task_cmd" > "$result_file" 2>&1; then
        echo "[Worker $worker_id] Task $task_id completed successfully"
        local status="success"
    else
        exit_code=$?
        echo "[Worker $worker_id] Task $task_id failed with exit code: $exit_code"
        local status="failed"
    fi
    
    local end_time=$(date +%s%N)
    local duration=$((($end_time - $start_time) / 1000000))  # Convert to milliseconds
    
    # Update task result
    cat > "${result_file}.json" <<EOF
{
    "task_id": "$task_id",
    "worker_id": "$worker_id",
    "status": "$status",
    "exit_code": $exit_code,
    "duration_ms": $duration,
    "completed": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF
    
    # Update statistics
    update_statistics "$task_id" "$status" "$duration"
    
    return 0
}

# Work-stealing implementation
steal_task() {
    local worker_id="$1"
    
    # Look for tasks in other workers' queues
    for other_worker in "$WORKER_DIR"/*.queue; do
        if [[ -f "$other_worker" ]]; then
            local stolen_task=$(head -n1 "$other_worker" 2>/dev/null)
            if [[ -n "$stolen_task" ]]; then
                # Remove from other worker's queue
                tail -n +2 "$other_worker" > "${other_worker}.tmp"
                mv "${other_worker}.tmp" "$other_worker"
                
                echo "[Worker $worker_id] Stole task from $(basename $other_worker)"
                echo "$stolen_task"
                return 0
            fi
        fi
    done
    
    return 1
}

# Check task dependencies
check_dependencies() {
    local dependencies="$1"
    
    IFS=',' read -ra deps <<< "$dependencies"
    for dep in "${deps[@]}"; do
        dep=$(echo "$dep" | tr -d ' ')
        if [[ ! -f "$RESULT_DIR/${dep}.result.json" ]]; then
            return 1
        fi
        
        local status=$(jq -r '.status' "$RESULT_DIR/${dep}.result.json")
        if [[ "$status" != "success" ]]; then
            return 1
        fi
    done
    
    return 0
}

# Update statistics
update_statistics() {
    local task_id="$1"
    local status="$2"
    local duration="$3"
    
    local stats=$(cat "$STATS_FILE")
    
    if [[ "$status" == "success" ]]; then
        stats=$(echo "$stats" | jq '.completed_tasks += 1')
    else
        stats=$(echo "$stats" | jq '.failed_tasks += 1')
    fi
    
    stats=$(echo "$stats" | jq --arg id "$task_id" --arg status "$status" --arg dur "$duration" \
        '.tasks += [{"id": $id, "status": $status, "duration_ms": ($dur | tonumber)}]')
    
    echo "$stats" > "$STATS_FILE"
}

# Start parallel execution
start_parallel_execution() {
    init_parallel_env
    
    echo "Starting parallel execution with $MAX_WORKERS workers"
    
    # Start worker processes
    for ((i=1; i<=MAX_WORKERS; i++)); do
        worker_process "$i" &
    done
    
    echo "All workers started"
}

# Stop parallel execution
stop_parallel_execution() {
    echo "Stopping parallel execution..."
    
    # Signal workers to stop
    touch "$WORKER_DIR/stop"
    
    # Wait for workers to finish current tasks
    local count=0
    while [[ $(find "$WORKER_DIR" -name "*.pid" 2>/dev/null | wc -l) -gt 0 ]] && [[ $count -lt 30 ]]; do
        sleep 1
        ((count++))
    done
    
    # Force kill remaining workers
    for worker_pid_file in "$WORKER_DIR"/*.pid; do
        if [[ -f "$worker_pid_file" ]]; then
            local pid=$(cat "$worker_pid_file")
            kill -9 "$pid" 2>/dev/null || true
            rm -f "$worker_pid_file"
        fi
    done
    
    rm -f "$WORKER_DIR/stop"
    echo "Parallel execution stopped"
}

# Get execution statistics
get_statistics() {
    if [[ ! -f "$STATS_FILE" ]]; then
        echo "No statistics available"
        return 1
    fi
    
    local stats=$(cat "$STATS_FILE")
    
    echo "Parallel Execution Statistics:"
    echo "==============================="
    echo "Workers: $(echo "$stats" | jq -r '.workers')"
    echo "Total Tasks: $(echo "$stats" | jq -r '.total_tasks')"
    echo "Completed: $(echo "$stats" | jq -r '.completed_tasks')"
    echo "Failed: $(echo "$stats" | jq -r '.failed_tasks')"
    
    # Calculate average duration
    local avg_duration=$(echo "$stats" | jq '[.tasks[].duration_ms] | add / length')
    echo "Average Duration: ${avg_duration}ms"
    
    # Show task distribution
    echo ""
    echo "Task Distribution:"
    echo "$stats" | jq -r '.tasks[] | "  - \(.id): \(.status) (\(.duration_ms)ms)"' | head -10
    
    # Performance metrics
    local total_time=$(echo "$stats" | jq '[.tasks[].duration_ms] | add')
    local parallel_time=$(echo "$stats" | jq '[.tasks[].duration_ms] | max')
    local speedup=$(echo "scale=2; $total_time / $parallel_time" | bc 2>/dev/null || echo "N/A")
    
    echo ""
    echo "Performance Metrics:"
    echo "  Total Sequential Time: ${total_time}ms"
    echo "  Parallel Execution Time: ${parallel_time}ms"
    echo "  Speedup: ${speedup}x"
}

# Execute task batch
execute_batch() {
    local batch_file="$1"
    
    if [[ ! -f "$batch_file" ]]; then
        echo "Batch file not found: $batch_file"
        return 1
    fi
    
    echo "Executing batch: $batch_file"
    
    # Parse and add tasks
    while IFS= read -r line; do
        if [[ -n "$line" && ! "$line" =~ ^# ]]; then
            # Parse priority and dependencies if specified
            if [[ "$line" =~ ^([0-9]+):(.*)$ ]]; then
                local priority="${BASH_REMATCH[1]}"
                local cmd="${BASH_REMATCH[2]}"
            else
                local priority=5
                local cmd="$line"
            fi
            
            add_task "$cmd" "$priority"
        fi
    done < "$batch_file"
    
    # Start execution
    start_parallel_execution
    
    # Wait for completion
    while [[ $(find "$WORK_QUEUE" -type f 2>/dev/null | wc -l) -gt 0 ]]; do
        sleep 1
    done
    
    # Stop workers
    stop_parallel_execution
    
    # Show results
    get_statistics
}

# Main command handler
main() {
    local action="${1:-help}"
    shift || true
    
    case "$action" in
        add)
            add_task "$@"
            ;;
        start)
            start_parallel_execution
            ;;
        stop)
            stop_parallel_execution
            ;;
        stats)
            get_statistics
            ;;
        batch)
            execute_batch "$@"
            ;;
        clean)
            rm -rf "$WORK_QUEUE" "$RESULT_DIR" "$WORKER_DIR"
            echo "Cleaned parallel execution environment"
            ;;
        help)
            cat <<EOF
Parallel Execution Framework

Usage: $0 <command> [options]

Commands:
  add <cmd> [priority] [deps]  Add task to queue
  start                         Start worker pool
  stop                          Stop all workers
  stats                         Show execution statistics
  batch <file>                  Execute batch file
  clean                         Clean all temporary files
  help                          Show this help

Examples:
  $0 add "sleep 5 && echo done" 1
  $0 add "compile project" 2 "task_123,task_456"
  $0 batch tasks.txt
  $0 stats

Batch File Format:
  # Comments start with #
  5:normal priority task
  1:high priority task
  10:low priority task
EOF
            ;;
        *)
            echo "Unknown command: $action"
            echo "Run '$0 help' for usage"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"