#!/bin/bash
# Reset workflow script for clean restart

set -euo pipefail

# Source the core library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/lib/core.sh"

# Workflow directory
WORKFLOW_DIR="${HOME}/aioke_parallel_workflows"

reset_workflow() {
    log_warn "This will reset the entire parallel workflow environment!"
    log_warn "All tasks, instances, and results will be cleared."
    
    # Skip confirmation in CI or if FORCE_RESET is set
    if [[ "${FORCE_RESET:-}" == "true" ]] || [[ "${CI:-}" == "true" ]]; then
        log_info "Force reset enabled"
    elif ! confirm "Are you sure you want to reset the workflow?" "n"; then
        log_info "Reset cancelled"
        return 0
    fi
    
    # Proceed with reset
    log_info "Resetting workflow environment..."
    
    # Clear all workflow data
    rm -rf "${WORKFLOW_DIR}/tasks/"*.json 2>/dev/null || true
    rm -rf "${WORKFLOW_DIR}/instances/"*.json 2>/dev/null || true
    rm -rf "${WORKFLOW_DIR}/results/"*.json 2>/dev/null || true
    
    # Reset status files
    echo '{"instances": [], "active": 0, "timestamp": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'""}' > "${WORKFLOW_DIR}/instance_status.json"
    echo '{"tasks": [], "completed": [], "in_progress": []}' > "${WORKFLOW_DIR}/task_queue.json"
    echo '{"total_tasks": 0, "completed": 0, "in_progress": 0, "pending": 0}' > "${WORKFLOW_DIR}/progress.json"
    
    log_ok "Workflow environment reset successfully"
    log_info "You can now run './dev parallel:init' to reinitialize"
}

# Run reset
reset_workflow