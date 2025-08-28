#!/bin/bash
# Parallel Workflow Orchestrator for AiOke Development
# Coordinates multiple Claude instances working on different aspects

set -euo pipefail

# Source framework libraries
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/lib/core.sh"

# Initialize
setup_error_handling "parallel-orchestrator"

# Configuration
readonly WORKFLOW_DIR="${HOME}/aioke_parallel_workflows"
readonly INSTANCE_STATUS_FILE="${WORKFLOW_DIR}/instance_status.json"
readonly TASK_QUEUE_FILE="${WORKFLOW_DIR}/task_queue.json"
readonly PROGRESS_FILE="${WORKFLOW_DIR}/progress.json"

# Task categories for parallel work
readonly -a TASK_CATEGORIES=(
    "audio_processing"
    "ui_development"
    "api_integration"
    "testing_validation"
    "documentation"
    "performance_optimization"
)

# Initialize workflow directory
init_workflow_env() {
    log_info "Initializing parallel workflow environment..."
    
    mkdir -p "${WORKFLOW_DIR}"
    mkdir -p "${WORKFLOW_DIR}/instances"
    mkdir -p "${WORKFLOW_DIR}/tasks"
    mkdir -p "${WORKFLOW_DIR}/results"
    
    # Initialize status files if they don't exist
    if [[ ! -f "${INSTANCE_STATUS_FILE}" ]]; then
        echo '{"instances": [], "active": 0, "timestamp": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'""}' > "${INSTANCE_STATUS_FILE}"
    fi
    
    if [[ ! -f "${TASK_QUEUE_FILE}" ]]; then
        echo '{"tasks": [], "completed": [], "in_progress": []}' > "${TASK_QUEUE_FILE}"
    fi
    
    if [[ ! -f "${PROGRESS_FILE}" ]]; then
        echo '{"total_tasks": 0, "completed": 0, "in_progress": 0, "pending": 0}' > "${PROGRESS_FILE}"
    fi
    
    log_ok "Workflow environment initialized"
    echo "Environment successfully initialized"  # Add stdout message for tests
    return 0
}

# Register a Claude instance
register_instance() {
    local instance_id="${1:-}"
    local category="${2:-general}"
    local description="${3:-Working on AiOke improvements}"
    
    if [[ -z "${instance_id}" ]]; then
        instance_id="instance_$(date +%s)"
    fi
    
    log_info "Registering instance: ${instance_id} for ${category}"
    
    # Update instance status
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    # Create instance file
    cat > "${WORKFLOW_DIR}/instances/${instance_id}.json" <<EOF
{
    "id": "${instance_id}",
    "category": "${category}",
    "description": "${description}",
    "status": "active",
    "registered": "${timestamp}",
    "last_update": "${timestamp}",
    "tasks_completed": 0
}
EOF
    
    log_ok "Instance ${instance_id} registered"
    echo "Instance ${instance_id} registered successfully"  # Add stdout message for tests
    return 0
}

# Create task for parallel execution
create_task() {
    local category="${1}"
    local title="${2}"
    local description="${3}"
    local priority="${4:-normal}"
    
    local task_id="task_$(date +%s)_$(uuidgen | tr '[:upper:]' '[:lower:]' | cut -c1-8)"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    log_info "Creating task: ${title} (${category})"
    
    # Create task file
    cat > "${WORKFLOW_DIR}/tasks/${task_id}.json" <<EOF
{
    "id": "${task_id}",
    "category": "${category}",
    "title": "${title}",
    "description": "${description}",
    "priority": "${priority}",
    "status": "pending",
    "created": "${timestamp}",
    "assigned_to": null,
    "started": null,
    "completed": null,
    "result": null
}
EOF
    
    log_ok "Task ${task_id} created"
    echo "${task_id}"
}

# Assign task to instance
assign_task() {
    local task_id="${1}"
    local instance_id="${2}"
    
    if [[ -z "${task_id}" ]] || [[ -z "${instance_id}" ]]; then
        log_error "Task ID and Instance ID required"
        return 1
    fi
    
    log_info "Assigning task ${task_id} to instance ${instance_id}"
    
    local task_file="${WORKFLOW_DIR}/tasks/${task_id}.json"
    if [[ ! -f "${task_file}" ]]; then
        log_error "Task ${task_id} not found"
        return 1
    fi
    
    # Update task assignment
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    jq --arg inst "${instance_id}" \
       --arg ts "${timestamp}" \
       '.assigned_to = $inst | .status = "in_progress" | .started = $ts' \
       "${task_file}" > "${task_file}.tmp"
    mv "${task_file}.tmp" "${task_file}"
    
    log_ok "Task assigned successfully"
}

# Mark task as complete
complete_task() {
    local task_id="${1}"
    local result="${2:-Success}"
    
    if [[ -z "${task_id}" ]]; then
        log_error "Task ID required"
        return 1
    fi
    
    log_info "Completing task ${task_id}"
    
    local task_file="${WORKFLOW_DIR}/tasks/${task_id}.json"
    if [[ ! -f "${task_file}" ]]; then
        log_error "Task ${task_id} not found"
        return 1
    fi
    
    # Update task completion
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    jq --arg res "${result}" \
       --arg ts "${timestamp}" \
       '.status = "completed" | .completed = $ts | .result = $res' \
       "${task_file}" > "${task_file}.tmp"
    mv "${task_file}.tmp" "${task_file}"
    
    # Move to results
    cp "${task_file}" "${WORKFLOW_DIR}/results/"
    
    log_ok "Task completed successfully"
}

# Get next available task for category
get_next_task() {
    local category="${1}"
    
    log_debug "Finding next task for category: ${category}"
    
    # Find pending tasks in category
    for task_file in "${WORKFLOW_DIR}/tasks/"*.json; do
        [[ -f "${task_file}" ]] || continue
        
        local task_category=$(jq -r '.category' "${task_file}")
        local task_status=$(jq -r '.status' "${task_file}")
        
        if [[ "${task_category}" == "${category}" ]] && [[ "${task_status}" == "pending" ]]; then
            basename "${task_file}" .json
            return 0
        fi
    done
    
    return 1
}

# Show workflow status
show_status() {
    log_info "Parallel Workflow Status"
    echo "========================"
    
    # Count instances
    local active_instances=0
    for inst_file in "${WORKFLOW_DIR}/instances/"*.json; do
        [[ -f "${inst_file}" ]] || continue
        local status=$(jq -r '.status' "${inst_file}")
        [[ "${status}" == "active" ]] && ((active_instances++))
    done
    
    # Count tasks
    local total_tasks=0
    local pending_tasks=0
    local in_progress_tasks=0
    local completed_tasks=0
    
    for task_file in "${WORKFLOW_DIR}/tasks/"*.json; do
        [[ -f "${task_file}" ]] || continue
        ((total_tasks++))
        
        local status=$(jq -r '.status' "${task_file}")
        case "${status}" in
            pending) ((pending_tasks++)) ;;
            in_progress) ((in_progress_tasks++)) ;;
            completed) ((completed_tasks++)) ;;
        esac
    done
    
    echo "Active Instances: ${active_instances}"
    echo "Total Tasks: ${total_tasks}"
    echo "  - Pending: ${pending_tasks}"
    echo "  - In Progress: ${in_progress_tasks}"
    echo "  - Completed: ${completed_tasks}"
    
    if [[ ${total_tasks} -gt 0 ]]; then
        local completion_rate=$(( (completed_tasks * 100) / total_tasks ))
        echo "Completion Rate: ${completion_rate}%"
    fi
    
    echo ""
    echo "Task Distribution by Category:"
    for category in "${TASK_CATEGORIES[@]}"; do
        local cat_count=0
        for task_file in "${WORKFLOW_DIR}/tasks/"*.json; do
            [[ -f "${task_file}" ]] || continue
            local task_cat=$(jq -r '.category' "${task_file}")
            [[ "${task_cat}" == "${category}" ]] && ((cat_count++))
        done
        [[ ${cat_count} -gt 0 ]] && echo "  - ${category}: ${cat_count}"
    done
}

# Create AiOke improvement tasks
create_aioke_tasks() {
    log_info "Creating AiOke improvement tasks..."
    
    # Check if tasks already exist to prevent duplicates
    local existing_tasks=0
    for task_file in "${WORKFLOW_DIR}/tasks/"*.json; do
        [[ -f "${task_file}" ]] && ((existing_tasks++))
    done
    
    if [[ ${existing_tasks} -ge 18 ]]; then
        log_warn "Tasks already exist (${existing_tasks} found). Skipping creation to prevent duplicates."
        return 0
    fi
    
    # Clear any partial tasks if less than 18 exist
    if [[ ${existing_tasks} -gt 0 ]] && [[ ${existing_tasks} -lt 18 ]]; then
        log_warn "Found ${existing_tasks} tasks (incomplete set). Clearing and recreating..."
        rm -f "${WORKFLOW_DIR}/tasks/"*.json
    fi
    
    # Audio Processing Tasks
    create_task "audio_processing" \
        "Implement Advanced Noise Reduction" \
        "Add AI-powered noise reduction using spectral subtraction and deep learning models" \
        "high"
    
    create_task "audio_processing" \
        "Real-time Pitch Correction" \
        "Implement auto-tune functionality with configurable parameters" \
        "high"
    
    create_task "audio_processing" \
        "Multi-track Recording" \
        "Enable recording multiple vocal tracks with mixing capabilities" \
        "medium"
    
    # UI Development Tasks
    create_task "ui_development" \
        "Responsive Web Interface" \
        "Create modern React-based UI with real-time waveform visualization" \
        "high"
    
    create_task "ui_development" \
        "Mobile App Development" \
        "Build native iOS/Android apps using React Native" \
        "medium"
    
    create_task "ui_development" \
        "Dark Mode Support" \
        "Implement system-wide dark mode with theme customization" \
        "low"
    
    # API Integration Tasks
    create_task "api_integration" \
        "Spotify Integration" \
        "Connect to Spotify API for playlist import and track metadata" \
        "high"
    
    create_task "api_integration" \
        "YouTube Music Sync" \
        "Enable YouTube karaoke track search and download" \
        "medium"
    
    create_task "api_integration" \
        "Cloud Storage Integration" \
        "Add Google Drive/Dropbox support for recording backup" \
        "medium"
    
    # Testing & Validation Tasks
    create_task "testing_validation" \
        "Comprehensive Test Suite" \
        "Create 88-test validation suite for all AiOke components" \
        "high"
    
    create_task "testing_validation" \
        "Performance Benchmarking" \
        "Measure and optimize latency, CPU usage, and memory footprint" \
        "high"
    
    create_task "testing_validation" \
        "Cross-platform Testing" \
        "Validate functionality on Windows, macOS, and Linux" \
        "medium"
    
    # Documentation Tasks
    create_task "documentation" \
        "API Documentation" \
        "Generate OpenAPI specification and interactive docs" \
        "medium"
    
    create_task "documentation" \
        "User Guide" \
        "Create comprehensive user manual with tutorials" \
        "medium"
    
    create_task "documentation" \
        "Developer Onboarding" \
        "Write contribution guidelines and setup instructions" \
        "low"
    
    # Performance Optimization Tasks
    create_task "performance_optimization" \
        "WebAssembly Audio Engine" \
        "Port critical audio processing to WASM for browser performance" \
        "high"
    
    create_task "performance_optimization" \
        "GPU Acceleration" \
        "Utilize WebGL/Metal for parallel audio processing" \
        "medium"
    
    create_task "performance_optimization" \
        "Caching Strategy" \
        "Implement intelligent caching for processed audio segments" \
        "medium"
    
    log_ok "Created 18 AiOke improvement tasks"
    echo "Created 18 AiOke improvement tasks"  # Add stdout message for tests
}

# Distribute tasks to instances
distribute_tasks() {
    log_info "Distributing tasks to active instances..."
    
    local distributed=0
    
    # Get active instances
    for inst_file in "${WORKFLOW_DIR}/instances/"*.json; do
        [[ -f "${inst_file}" ]] || continue
        
        local instance_id=$(basename "${inst_file}" .json)
        local category=$(jq -r '.category' "${inst_file}")
        local status=$(jq -r '.status' "${inst_file}")
        
        [[ "${status}" != "active" ]] && continue
        
        # Assign next available task
        if task_id=$(get_next_task "${category}"); then
            assign_task "${task_id}" "${instance_id}"
            ((distributed++))
        fi
    done
    
    log_ok "Distributed ${distributed} tasks"
}

# Monitor progress
monitor_progress() {
    local once_only="${1:-false}"
    
    log_info "Monitoring parallel workflow progress..."
    
    if [[ "${once_only}" == "once" ]]; then
        # Single run for testing
        echo "=== AiOke Parallel Development Dashboard ==="
        echo "Time: $(date)"
        echo ""
        
        show_status
        
        echo ""
        echo "Recent Activity:"
        # Show last 5 completed tasks
        for result_file in $(ls -t "${WORKFLOW_DIR}/results/"*.json 2>/dev/null | head -5); do
            [[ -f "${result_file}" ]] || continue
            local title=$(jq -r '.title' "${result_file}")
            local completed=$(jq -r '.completed' "${result_file}")
            echo "  ✓ ${title} (${completed})"
        done
        echo "Monitor dashboard displayed successfully"
        return 0
    fi
    
    # Continuous monitoring
    while true; do
        clear
        echo "=== AiOke Parallel Development Dashboard ==="
        echo "Time: $(date)"
        echo ""
        
        show_status
        
        echo ""
        echo "Recent Activity:"
        # Show last 5 completed tasks
        for result_file in $(ls -t "${WORKFLOW_DIR}/results/"*.json 2>/dev/null | head -5); do
            [[ -f "${result_file}" ]] || continue
            local title=$(jq -r '.title' "${result_file}")
            local completed=$(jq -r '.completed' "${result_file}")
            echo "  ✓ ${title} (${completed})"
        done
        
        echo ""
        echo "Press Ctrl+C to exit monitoring"
        
        sleep 5
    done
}

# Main command dispatcher
main() {
    local cmd="${1:-help}"
    log_debug "Command received: '$cmd', all args: '$*'"
    shift || true
    
    case "${cmd}" in
        init)
            init_workflow_env
            ;;
        register)
            register_instance "$@"
            ;;
        create-task)
            create_task "$@"
            ;;
        assign)
            assign_task "$@"
            ;;
        complete)
            complete_task "$@"
            ;;
        next-task)
            get_next_task "$@"
            ;;
        status)
            show_status
            ;;
        create-aioke-tasks)
            create_aioke_tasks
            ;;
        distribute)
            distribute_tasks
            ;;
        monitor)
            # After shift, $1 is the first parameter to monitor command
            local monitor_param="${1:-}"
            log_debug "Monitor parameter received: '${monitor_param}'"
            if [[ "${monitor_param}" == "help" ]]; then
                echo "Monitor command displays real-time progress dashboard"
                echo "Usage: $0 monitor [once]"
                echo "  once    Run monitoring once (for testing)"
                exit 0
            elif [[ "${monitor_param}" == "once" ]]; then
                monitor_progress "once"
                exit 0
            fi
            monitor_progress
            ;;
        help)
            cat <<EOF
Parallel Workflow Orchestrator for AiOke Development

Usage: $0 <command> [options]

Commands:
  init                Initialize workflow environment
  register <id> <cat> Register a Claude instance
  create-task         Create a new task
  assign <task> <inst> Assign task to instance
  complete <task>     Mark task as complete
  next-task <cat>     Get next available task
  status              Show workflow status
  create-aioke-tasks  Create AiOke improvement tasks
  distribute          Distribute tasks to instances
  monitor             Monitor progress dashboard
  help                Show this help message

Categories:
  - audio_processing
  - ui_development
  - api_integration
  - testing_validation
  - documentation
  - performance_optimization
EOF
            ;;
        *)
            log_error "Unknown command: ${cmd}"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"