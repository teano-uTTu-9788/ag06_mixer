#!/bin/bash
# Universal Parallel Workflow Orchestrator
# Coordinates multiple Claude instances working on any shared repository/project
# Repository-agnostic and project-agnostic collaboration framework

set -euo pipefail

# Source framework libraries
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/lib/core.sh"

# Initialize
setup_error_handling "universal-orchestrator"

# Version
readonly VERSION="1.0.0"

# Universal task categories for parallel work (repository-agnostic)
readonly -a TASK_CATEGORIES=(
    "backend_development"
    "frontend_development"
    "database_design"
    "api_integration"
    "testing_validation"
    "documentation"
    "devops_infrastructure" 
    "security_compliance"
    "performance_optimization"
    "monitoring_observability"
)

# Auto-detect project context
detect_project_context() {
    local project_name
    local repo_root
    
    # Try to detect git repository root
    if git rev-parse --show-toplevel >/dev/null 2>&1; then
        repo_root="$(git rev-parse --show-toplevel)"
        project_name="$(basename "${repo_root}")"
    else
        # Fall back to current directory
        repo_root="$(pwd)"
        project_name="$(basename "${repo_root}")"
    fi
    
    echo "${project_name}:${repo_root}"
}

# Configuration (auto-detected or user-specified)
setup_configuration() {
    local project_info
    project_info="$(detect_project_context)"
    
    # Avoid readonly conflicts if already set
    if [[ -z "${PROJECT_NAME:-}" ]]; then
        readonly PROJECT_NAME="${1:-$(echo "${project_info}" | cut -d: -f1)}"
    fi
    if [[ -z "${PROJECT_ROOT:-}" ]]; then
        readonly PROJECT_ROOT="${2:-$(echo "${project_info}" | cut -d: -f2)}"
    fi
    if [[ -z "${WORKFLOW_DIR:-}" ]]; then
        readonly WORKFLOW_DIR="${HOME}/.universal_workflows/${PROJECT_NAME}"
        readonly INSTANCE_STATUS_FILE="${WORKFLOW_DIR}/instance_status.json"
        readonly TASK_QUEUE_FILE="${WORKFLOW_DIR}/task_queue.json"
        readonly PROGRESS_FILE="${WORKFLOW_DIR}/progress.json"
        readonly PROJECT_CONFIG_FILE="${WORKFLOW_DIR}/project_config.json"
    fi
    
    log_info "Project: ${PROJECT_NAME}"
    log_info "Root: ${PROJECT_ROOT}"
    log_info "Workflow Dir: ${WORKFLOW_DIR}"
}

# Initialize workflow directory
init_workflow_env() {
    setup_configuration "$@"
    
    log_info "Initializing universal parallel workflow environment..."
    log_info "Project: ${PROJECT_NAME} at ${PROJECT_ROOT}"
    
    mkdir -p "${WORKFLOW_DIR}"
    mkdir -p "${WORKFLOW_DIR}/instances"
    mkdir -p "${WORKFLOW_DIR}/tasks"
    mkdir -p "${WORKFLOW_DIR}/results"
    mkdir -p "${WORKFLOW_DIR}/logs"
    
    # Initialize status files if they don't exist
    if [[ ! -f "${INSTANCE_STATUS_FILE}" ]]; then
        cat > "${INSTANCE_STATUS_FILE}" <<EOF
{
  "instances": [],
  "active": 0,
  "project": "${PROJECT_NAME}",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF
    fi
    
    if [[ ! -f "${TASK_QUEUE_FILE}" ]]; then
        cat > "${TASK_QUEUE_FILE}" <<EOF
{
  "tasks": [],
  "completed": [],
  "in_progress": [],
  "project": "${PROJECT_NAME}"
}
EOF
    fi
    
    if [[ ! -f "${PROGRESS_FILE}" ]]; then
        cat > "${PROGRESS_FILE}" <<EOF
{
  "total_tasks": 0,
  "completed": 0,
  "in_progress": 0,
  "pending": 0,
  "project": "${PROJECT_NAME}",
  "last_updated": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF
    fi
    
    # Create project configuration
    if [[ ! -f "${PROJECT_CONFIG_FILE}" ]]; then
        cat > "${PROJECT_CONFIG_FILE}" <<EOF
{
  "name": "${PROJECT_NAME}",
  "root": "${PROJECT_ROOT}",
  "categories": [
    "backend_development",
    "frontend_development", 
    "database_design",
    "api_development",
    "testing_validation",
    "documentation",
    "devops_deployment",
    "performance_optimization",
    "security_implementation",
    "code_refactoring"
  ],
  "file_patterns": {
    "backend": ["*.py", "*.js", "*.ts", "*.go", "*.rs", "*.java"],
    "frontend": ["*.html", "*.css", "*.jsx", "*.tsx", "*.vue", "*.svelte"],
    "config": ["*.json", "*.yaml", "*.yml", "*.toml", "*.ini"],
    "docs": ["*.md", "*.rst", "*.txt"]
  },
  "created": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "version": "${VERSION}"
}
EOF
    fi
    
    log_ok "Universal workflow environment initialized for ${PROJECT_NAME}"
    echo "Environment successfully initialized for ${PROJECT_NAME}"
}

# Register a Claude instance
register_instance() {
    setup_configuration
    
    local instance_id="${1:-}"
    local specialization="${2:-general}"
    local description="${3:-Claude instance}"
    
    if [[ -z "${instance_id}" ]]; then
        log_error "Instance ID required"
        echo "Usage: $0 register <instance-id> [specialization] [description]"
        return 1
    fi
    
    log_info "Registering instance: ${instance_id}"
    
    # Create instance file
    local instance_file="${WORKFLOW_DIR}/instances/${instance_id}.json"
    cat > "${instance_file}" <<EOF
{
  "id": "${instance_id}",
  "specialization": "${specialization}",
  "description": "${description}",
  "status": "active",
  "registered": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "tasks_completed": 0,
  "current_task": null,
  "project": "${PROJECT_NAME}"
}
EOF
    
    log_ok "Instance ${instance_id} registered"
    echo "Instance registered: ${instance_id}"
}

# Create a generic task
create_task() {
    setup_configuration
    
    local title="${1:-}"
    local description="${2:-}"
    local category="${3:-general}"
    local priority="${4:-medium}"
    local files="${5:-}"
    
    if [[ -z "${title}" ]]; then
        log_error "Task title required"
        echo "Usage: $0 create-task <title> [description] [category] [priority] [files]"
        return 1
    fi
    
    local task_id="task_$(date +%s)_$(openssl rand -hex 4 2>/dev/null || echo "$(date +%N | cut -c1-8)")"
    local task_file="${WORKFLOW_DIR}/tasks/${task_id}.json"
    
    cat > "${task_file}" <<EOF
{
  "id": "${task_id}",
  "title": "${title}",
  "description": "${description}",
  "category": "${category}",
  "priority": "${priority}",
  "status": "pending",
  "files": $(if [[ -n "${files}" ]]; then echo "\"${files}\""; else echo "null"; fi),
  "created": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "assigned_to": null,
  "started": null,
  "completed": null,
  "project": "${PROJECT_NAME}",
  "result": null
}
EOF
    
    log_ok "Task created: ${task_id}"
    echo "${task_id}"
}

# Create tasks based on repository analysis
analyze_and_create_tasks() {
    setup_configuration
    
    log_info "Analyzing repository structure to create tasks..."
    
    # AiOke-specific development tasks following Google/AWS/Microsoft best practices
    local -a common_tasks=(
        "AiOke Audio Engine Optimization:Optimize real-time audio processing following Google's gRPC patterns:aioke_audio_processing:high"
        "AiOke Karaoke Algorithm Enhancement:Enhance karaoke vocal separation using AWS machine learning patterns:aioke_karaoke_engine:high" 
        "AiOke UI Component Library:Build reusable React components following Microsoft Fluent design system:aioke_ui_components:medium"
        "AiOke Performance Monitoring:Implement observability following OpenTelemetry standards:aioke_monitoring:high"
        "AiOke Deployment Pipeline:Create blue-green deployment using AWS/Google Cloud patterns:aioke_deployment:high"
        "AiOke Documentation System:Create comprehensive docs following Google's documentation standards:aioke_documentation:medium"
        "AiOke API Gateway Design:Design GraphQL API following Netflix federation patterns:aioke_api_design:medium"
        "AiOke Vocal Effects Processing:Implement professional vocal effects using WebAssembly:aioke_vocal_effects:medium"
        "AiOke Auto-Scaling Configuration:Configure Kubernetes HPA following Google GKE best practices:aioke_scaling:medium"
        "AiOke CI/CD Security:Implement security scanning following Microsoft DevSecOps patterns:aioke_ci_cd:high"
        "AiOke Mobile Integration:Prepare mobile SDK following React Native/Flutter patterns:aioke_mobile:medium"
        "AiOke Analytics Dashboard:Build analytics using Google Analytics/AWS CloudWatch patterns:aioke_analytics:medium"
        
        # Legacy tasks for backward compatibility
        "Code Review and Quality Analysis:Review codebase for SOLID principles, patterns, and improvements:code_refactoring:high"
        "Documentation Update:Update and improve project documentation:documentation:medium"
    )
    
    # Detect project type and add specific tasks
    if [[ -f "${PROJECT_ROOT}/package.json" ]]; then
        common_tasks+=("Frontend Bundle Optimization:Optimize webpack/build configuration:frontend_development:medium")
        common_tasks+=("Node.js Performance:Optimize Node.js backend performance:backend_development:medium")
    fi
    
    if [[ -f "${PROJECT_ROOT}/requirements.txt" ]] || [[ -f "${PROJECT_ROOT}/pyproject.toml" ]]; then
        common_tasks+=("Python Code Optimization:Optimize Python code performance:backend_development:medium")
        common_tasks+=("Python Dependencies:Update Python dependencies safely:devops_deployment:medium")
    fi
    
    if [[ -f "${PROJECT_ROOT}/Dockerfile" ]] || [[ -f "${PROJECT_ROOT}/docker-compose.yml" ]]; then
        common_tasks+=("Docker Optimization:Optimize Docker images and containers:devops_deployment:medium")
    fi
    
    if [[ -d "${PROJECT_ROOT}/.github" ]]; then
        common_tasks+=("CI/CD Enhancement:Improve GitHub Actions workflows:devops_deployment:medium")
    fi
    
    # Create tasks
    local created_count=0
    for task_info in "${common_tasks[@]}"; do
        IFS=':' read -r title description category priority <<< "${task_info}"
        create_task "${title}" "${description}" "${category}" "${priority}" >/dev/null
        ((created_count++))
    done
    
    log_ok "Created ${created_count} tasks based on repository analysis"
    echo "Created ${created_count} tasks for ${PROJECT_NAME}"
}

# Get next available task
get_next_task() {
    setup_configuration
    
    local category="${1:-}"
    local instance_id="${2:-}"
    
    # Find pending tasks
    local task_file
    local task_id
    
    if [[ -n "${category}" ]]; then
        # Look for tasks in specific category
        for task_file in "${WORKFLOW_DIR}/tasks/"*.json; do
            [[ -f "${task_file}" ]] || continue
            
            if jq -e --arg cat "${category}" '.status == "pending" and .category == $cat' "${task_file}" >/dev/null 2>&1; then
                task_id="$(jq -r '.id' "${task_file}")"
                
                # Assign task if instance provided
                if [[ -n "${instance_id}" ]]; then
                    assign_task_to_instance "${task_id}" "${instance_id}"
                fi
                
                echo "${task_id}"
                return 0
            fi
        done
    else
        # Find any pending task
        for task_file in "${WORKFLOW_DIR}/tasks/"*.json; do
            [[ -f "${task_file}" ]] || continue
            
            if jq -e '.status == "pending"' "${task_file}" >/dev/null 2>&1; then
                task_id="$(jq -r '.id' "${task_file}")"
                
                # Assign task if instance provided
                if [[ -n "${instance_id}" ]]; then
                    assign_task_to_instance "${task_id}" "${instance_id}"
                fi
                
                echo "${task_id}"
                return 0
            fi
        done
    fi
    
    return 1
}

# Assign task to instance
assign_task_to_instance() {
    setup_configuration
    
    local task_id="${1:-}"
    local instance_id="${2:-}"
    
    if [[ -z "${task_id}" ]] || [[ -z "${instance_id}" ]]; then
        log_error "Both task ID and instance ID required"
        return 1
    fi
    
    local task_file="${WORKFLOW_DIR}/tasks/${task_id}.json"
    local instance_file="${WORKFLOW_DIR}/instances/${instance_id}.json"
    
    if [[ ! -f "${task_file}" ]]; then
        log_error "Task not found: ${task_id}"
        return 1
    fi
    
    if [[ ! -f "${instance_file}" ]]; then
        log_error "Instance not found: ${instance_id}"
        return 1
    fi
    
    # Update task status
    jq --arg instance "${instance_id}" --arg timestamp "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
        '.status = "in_progress" | .assigned_to = $instance | .started = $timestamp' \
        "${task_file}" > "${task_file}.tmp" && mv "${task_file}.tmp" "${task_file}"
    
    # Update instance current task
    jq --arg task "${task_id}" \
        '.current_task = $task' \
        "${instance_file}" > "${instance_file}.tmp" && mv "${instance_file}.tmp" "${instance_file}"
    
    log_ok "Task ${task_id} assigned to ${instance_id}"
    echo "Task assigned: ${task_id} -> ${instance_id}"
}

# Complete a task
complete_task() {
    setup_configuration
    
    local task_id="${1:-}"
    local result="${2:-Task completed}"
    
    if [[ -z "${task_id}" ]]; then
        log_error "Task ID required"
        return 1
    fi
    
    local task_file="${WORKFLOW_DIR}/tasks/${task_id}.json"
    
    if [[ ! -f "${task_file}" ]]; then
        log_error "Task not found: ${task_id}"
        return 1
    fi
    
    # Update task status
    jq --arg result "${result}" --arg timestamp "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
        '.status = "completed" | .result = $result | .completed = $timestamp' \
        "${task_file}" > "${task_file}.tmp" && mv "${task_file}.tmp" "${task_file}"
    
    # Move to results
    local result_file="${WORKFLOW_DIR}/results/${task_id}.json"
    cp "${task_file}" "${result_file}"
    
    # Update instance
    local instance_id
    instance_id="$(jq -r '.assigned_to // empty' "${task_file}")"
    if [[ -n "${instance_id}" ]]; then
        local instance_file="${WORKFLOW_DIR}/instances/${instance_id}.json"
        if [[ -f "${instance_file}" ]]; then
            jq '.tasks_completed += 1 | .current_task = null' \
                "${instance_file}" > "${instance_file}.tmp" && mv "${instance_file}.tmp" "${instance_file}"
        fi
    fi
    
    log_ok "Task ${task_id} completed"
    echo "Task completed: ${task_id}"
}

# Show workflow status
show_status() {
    setup_configuration
    
    log_info "Universal Parallel Workflow Status"
    echo "========================================"
    echo "Project: ${PROJECT_NAME}"
    echo "Root: ${PROJECT_ROOT}"
    echo ""
    
    # Count instances
    local active_instances=0
    if ls "${WORKFLOW_DIR}/instances/"*.json >/dev/null 2>&1; then
        active_instances=$(ls "${WORKFLOW_DIR}/instances/"*.json | wc -l | tr -d ' ')
    fi
    
    # Count tasks by status
    local total_tasks=0
    local pending_tasks=0
    local in_progress_tasks=0
    local completed_tasks=0
    
    if ls "${WORKFLOW_DIR}/tasks/"*.json >/dev/null 2>&1; then
        for task_file in "${WORKFLOW_DIR}/tasks/"*.json; do
            [[ -f "${task_file}" ]] || continue
            
            ((total_tasks++))
            
            local status
            status="$(jq -r '.status' "${task_file}" 2>/dev/null || echo "unknown")"
            
            case "${status}" in
                "pending") ((pending_tasks++)) ;;
                "in_progress") ((in_progress_tasks++)) ;;
                "completed") ((completed_tasks++)) ;;
            esac
        done
    fi
    
    # Calculate completion rate
    local completion_rate=0
    if [[ ${total_tasks} -gt 0 ]]; then
        completion_rate=$((completed_tasks * 100 / total_tasks))
    fi
    
    echo "Active Instances: ${active_instances}"
    echo "Total Tasks: ${total_tasks}"
    echo "  - Pending: ${pending_tasks}"
    echo "  - In Progress: ${in_progress_tasks}"
    echo "  - Completed: ${completed_tasks}"
    echo "Completion Rate: ${completion_rate}%"
    echo ""
    
    # Show task distribution by category
    if [[ ${total_tasks} -gt 0 ]]; then
        echo "Task Distribution by Category:"
        
        # Use a simpler approach to count categories
        for category in "${TASK_CATEGORIES[@]}"; do
            local cat_count=0
            for task_file in "${WORKFLOW_DIR}/tasks/"*.json; do
                [[ -f "${task_file}" ]] || continue
                local task_cat
                task_cat="$(jq -r '.category // "uncategorized"' "${task_file}" 2>/dev/null)"
                [[ "${task_cat}" == "${category}" ]] && ((cat_count++))
            done
            [[ ${cat_count} -gt 0 ]] && echo "  - ${category}: ${cat_count}"
        done
    fi
    
    echo ""
    log_info "Universal Parallel Workflow Status"
}

# Monitor progress dashboard
monitor_progress() {
    setup_configuration
    
    local once_only="${1:-false}"
    
    log_info "Monitoring universal parallel workflow progress..."
    
    if [[ "${once_only}" == "once" ]]; then
        # Single run for testing
        echo "=== Universal Parallel Development Dashboard ==="
        echo "Project: ${PROJECT_NAME}"
        echo "Time: $(date)"
        echo ""
        
        show_status
        echo "Monitor dashboard displayed successfully"
        return 0
    fi
    
    # Continuous monitoring
    while true; do
        clear
        echo "=== Universal Parallel Development Dashboard ==="
        echo "Project: ${PROJECT_NAME}"
        echo "Time: $(date)"
        echo ""
        
        show_status
        
        echo "Recent Activity:"
        # Show recent task completions
        if ls "${WORKFLOW_DIR}/results/"*.json >/dev/null 2>&1; then
            local recent_file
            recent_file="$(ls -t "${WORKFLOW_DIR}/results/"*.json 2>/dev/null | head -1)"
            if [[ -n "${recent_file}" ]]; then
                local task_title
                task_title="$(jq -r '.title' "${recent_file}" 2>/dev/null || echo "Unknown task")"
                local completed_time
                completed_time="$(jq -r '.completed // "Unknown"' "${recent_file}" 2>/dev/null)"
                echo "- Last completed: ${task_title} (${completed_time})"
            fi
        fi
        
        echo ""
        echo "Press Ctrl+C to exit monitoring"
        
        sleep 5
    done
}

# Distribute tasks to active instances
distribute_tasks() {
    setup_configuration
    
    log_info "Distributing tasks to active instances..."
    
    # Get active instances
    local -a instances
    if ls "${WORKFLOW_DIR}/instances/"*.json >/dev/null 2>&1; then
        for instance_file in "${WORKFLOW_DIR}/instances/"*.json; do
            [[ -f "${instance_file}" ]] || continue
            
            local instance_id
            instance_id="$(jq -r '.id' "${instance_file}")"
            instances+=("${instance_id}")
        done
    fi
    
    if [[ ${#instances[@]} -eq 0 ]]; then
        log_warn "No active instances found for task distribution"
        return 1
    fi
    
    log_info "Found ${#instances[@]} active instances"
    
    # Assign pending tasks round-robin
    local assigned=0
    local instance_index=0
    
    for task_file in "${WORKFLOW_DIR}/tasks/"*.json; do
        [[ -f "${task_file}" ]] || continue
        
        if jq -e '.status == "pending"' "${task_file}" >/dev/null 2>&1; then
            local task_id
            task_id="$(jq -r '.id' "${task_file}")"
            
            assign_task_to_instance "${task_id}" "${instances[${instance_index}]}"
            ((assigned++))
            
            # Round-robin to next instance
            instance_index=$(((instance_index + 1) % ${#instances[@]}))
        fi
    done
    
    log_ok "Distributed ${assigned} tasks to ${#instances[@]} instances"
    echo "Tasks distributed: ${assigned}"
}

# Main command dispatcher
main() {
    local cmd="${1:-help}"
    log_debug "Command received: '$cmd', all args: '$*'"
    shift || true
    
    case "${cmd}" in
        init)
            init_workflow_env "$@"
            ;;
        register)
            register_instance "$@"
            ;;
        create-task)
            create_task "$@"
            ;;
        analyze)
            analyze_and_create_tasks "$@"
            ;;
        assign)
            assign_task_to_instance "$@"
            ;;
        complete)
            complete_task "$@"
            ;;
        next-task)
            get_next_task "$@"
            ;;
        status)
            show_status "$@"
            ;;
        distribute)
            distribute_tasks "$@"
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
Universal Parallel Workflow Orchestrator v${VERSION}
Coordinates multiple Claude instances on any shared repository/project

Usage: $0 <command> [options]

Commands:
  init [project] [root]   Initialize workflow for project (auto-detects if not specified)
  register <id> [spec]    Register a Claude instance with specialization
  create-task <title>     Create a new task manually
  analyze                 Analyze repository and create appropriate tasks
  assign <task> <inst>    Assign task to instance
  complete <task>         Mark task as complete
  next-task [category]    Get next available task (optionally by category)
  status                  Show workflow status
  distribute              Distribute pending tasks to active instances
  monitor [once]          Monitor progress dashboard
  help                    Show this help message

Examples:
  $0 init                           # Initialize for current project
  $0 init my-app /path/to/project   # Initialize for specific project
  $0 register claude-backend backend_development
  $0 analyze                        # Create tasks based on repo analysis
  $0 next-task backend_development  # Get next backend task
  $0 distribute                     # Auto-assign tasks to instances

Categories (auto-detected based on project):
  - backend_development
  - frontend_development  
  - database_design
  - api_development
  - testing_validation
  - documentation
  - devops_deployment
  - performance_optimization
  - security_implementation
  - code_refactoring

Project Types Supported:
  - Node.js/JavaScript projects (package.json)
  - Python projects (requirements.txt, pyproject.toml)
  - Docker projects (Dockerfile, docker-compose.yml)
  - GitHub projects (.github workflows)
  - Any git repository or directory

Workflow Directory: ~/.universal_workflows/[project-name]/
EOF
            ;;
        version)
            echo "Universal Parallel Workflow Orchestrator v${VERSION}"
            ;;
        *)
            log_error "Unknown command: $cmd"
            echo ""
            log_info "Available commands:"
            log_info "  init, register, create-task, analyze, assign, complete"
            log_info "  next-task, status, distribute, monitor, help, version"
            echo ""
            log_info "Run '$0 help' for detailed usage information"
            return 2
            ;;
    esac
}

# Run main function with all arguments
main "$@"