#!/bin/bash
# Git automation library following Google/Meta best practices
# Provides: Git operations, branch management, hooks, CI/CD integration
# Used by: Deployment scripts, development workflows

# shellcheck source=./core.sh
source "$(dirname "${BASH_SOURCE[0]}")/core.sh"

# Git configuration
readonly DEFAULT_BRANCH="${DEFAULT_BRANCH:-main}"
readonly COMMIT_MSG_TEMPLATE="${PROJECT_ROOT}/.gitmessage"

# Git repository initialization and validation
git_init_if_needed() {
    if [[ -d ".git" ]]; then
        log_debug "Git repository already initialized"
        return 0
    fi
    
    log_info "Initializing Git repository..."
    
    git init
    
    # Set default branch name (Google/GitHub standard)
    git checkout -b "${DEFAULT_BRANCH}"
    
    # Configure commit message template if it exists
    if [[ -f "${COMMIT_MSG_TEMPLATE}" ]]; then
        git config commit.template "${COMMIT_MSG_TEMPLATE}"
    fi
    
    log_ok "Git repository initialized with branch: ${DEFAULT_BRANCH}"
}

git_validate_repo() {
    if ! git rev-parse --git-dir >/dev/null 2>&1; then
        log_fatal "Not in a Git repository. Run 'git init' first."
    fi
    
    log_debug "Git repository validation passed"
}

# Configuration management (Google-style standardization)
git_setup_user() {
    local name="$1"
    local email="$2"
    local global="${3:-false}"
    
    if [[ -z "${name}" ]] || [[ -z "${email}" ]]; then
        log_error "Both name and email are required"
        return 1
    fi
    
    local scope_flag=""
    if [[ "${global}" == "true" ]]; then
        scope_flag="--global"
        log_info "Setting global Git user configuration"
    else
        log_info "Setting local Git user configuration"
    fi
    
    git config ${scope_flag} user.name "${name}"
    git config ${scope_flag} user.email "${email}"
    
    log_ok "Git user configured: ${name} <${email}>"
}

git_setup_defaults() {
    local global="${1:-false}"
    
    local scope_flag=""
    if [[ "${global}" == "true" ]]; then
        scope_flag="--global"
        log_info "Setting global Git defaults"
    else
        log_info "Setting local Git defaults"
    fi
    
    # Core settings (Google recommendations)
    git config ${scope_flag} core.autocrlf false
    git config ${scope_flag} core.filemode false
    git config ${scope_flag} core.ignorecase false
    git config ${scope_flag} core.precomposeunicode true
    git config ${scope_flag} core.quotepath false
    
    # Push settings (safer defaults)
    git config ${scope_flag} push.default simple
    git config ${scope_flag} push.followTags true
    
    # Pull settings (rebase by default - Meta practice)
    git config ${scope_flag} pull.rebase true
    
    # Branch settings
    git config ${scope_flag} branch.autosetupmerge always
    git config ${scope_flag} branch.autosetuprebase always
    
    # Merge settings (Google-style)
    git config ${scope_flag} merge.tool vimdiff
    git config ${scope_flag} merge.conflictstyle diff3
    
    # Color settings
    git config ${scope_flag} color.ui auto
    
    # Credential caching (macOS)
    if is_macos; then
        git config ${scope_flag} credential.helper osxkeychain
    fi
    
    log_ok "Git defaults configured"
}

# Branch management (Meta-style feature branch workflow)
git_create_branch() {
    local branch_name="$1"
    local base_branch="${2:-${DEFAULT_BRANCH}}"
    
    if [[ -z "${branch_name}" ]]; then
        log_error "Branch name is required"
        return 1
    fi
    
    git_validate_repo
    
    # Validate branch name (Google naming conventions)
    if ! [[ "${branch_name}" =~ ^[a-z0-9/_-]+$ ]]; then
        log_error "Branch name must contain only lowercase letters, numbers, hyphens, underscores, and forward slashes"
        return 1
    fi
    
    # Check if branch already exists
    if git rev-parse --verify "${branch_name}" >/dev/null 2>&1; then
        log_warn "Branch '${branch_name}' already exists"
        return 1
    fi
    
    log_info "Creating branch '${branch_name}' from '${base_branch}'"
    
    # Ensure we're on the base branch and it's up to date
    git checkout "${base_branch}"
    git pull origin "${base_branch}" || log_warn "Could not pull latest changes from origin"
    
    # Create and checkout new branch
    git checkout -b "${branch_name}"
    
    log_ok "Branch '${branch_name}' created and checked out"
}

git_delete_branch() {
    local branch_name="$1"
    local force="${2:-false}"
    
    if [[ -z "${branch_name}" ]]; then
        log_error "Branch name is required"
        return 1
    fi
    
    git_validate_repo
    
    # Safety check - don't delete main/master branches
    if [[ "${branch_name}" == "main" ]] || [[ "${branch_name}" == "master" ]]; then
        log_fatal "Cannot delete main/master branch"
    fi
    
    # Check if branch exists
    if ! git rev-parse --verify "${branch_name}" >/dev/null 2>&1; then
        log_warn "Branch '${branch_name}' does not exist"
        return 1
    fi
    
    # Switch to default branch if currently on the branch to be deleted
    local current_branch
    current_branch=$(git branch --show-current)
    if [[ "${current_branch}" == "${branch_name}" ]]; then
        log_info "Switching to ${DEFAULT_BRANCH} before deleting current branch"
        git checkout "${DEFAULT_BRANCH}"
    fi
    
    # Delete branch
    local delete_flag="-d"
    if [[ "${force}" == "true" ]]; then
        delete_flag="-D"
        log_warn "Force deleting branch '${branch_name}'"
    else
        log_info "Deleting branch '${branch_name}'"
    fi
    
    if git branch "${delete_flag}" "${branch_name}"; then
        log_ok "Branch '${branch_name}' deleted locally"
        
        # Try to delete remote branch if it exists
        if git ls-remote --exit-code origin "${branch_name}" >/dev/null 2>&1; then
            if confirm "Delete remote branch 'origin/${branch_name}'?" "n"; then
                git push origin --delete "${branch_name}"
                log_ok "Remote branch deleted"
            fi
        fi
    else
        log_error "Failed to delete branch '${branch_name}'"
        return 1
    fi
}

# Commit operations (Google-style commit hygiene)
git_commit_staged() {
    local message="$1"
    local sign="${2:-false}"
    
    if [[ -z "${message}" ]]; then
        log_error "Commit message is required"
        return 1
    fi
    
    git_validate_repo
    
    # Check if there are staged changes
    if ! git diff --cached --exit-code >/dev/null; then
        log_info "Creating commit with message: ${message}"
        
        local sign_flag=""
        if [[ "${sign}" == "true" ]]; then
            sign_flag="--signoff"
        fi
        
        git commit ${sign_flag} -m "${message}"
        log_ok "Commit created successfully"
    else
        log_warn "No staged changes to commit"
        return 1
    fi
}

git_commit_all() {
    local message="$1"
    local sign="${2:-false}"
    
    if [[ -z "${message}" ]]; then
        log_error "Commit message is required"
        return 1
    fi
    
    git_validate_repo
    
    # Check if there are any changes
    if git diff --exit-code >/dev/null && git diff --cached --exit-code >/dev/null; then
        log_warn "No changes to commit"
        return 1
    fi
    
    log_info "Committing all changes with message: ${message}"
    
    local sign_flag=""
    if [[ "${sign}" == "true" ]]; then
        sign_flag="--signoff"
    fi
    
    git add --all
    git commit ${sign_flag} -m "${message}"
    
    log_ok "All changes committed successfully"
}

# Export all functions
export -f git_init_if_needed git_validate_repo
export -f git_setup_user git_setup_defaults
export -f git_create_branch git_delete_branch
export -f git_commit_staged git_commit_all