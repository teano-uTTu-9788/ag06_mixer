#!/usr/bin/env bash
# Git Operations Library - Meta/Google Pattern
# Provides Git automation and workflow management

# Require core library
deps::require "core"

# ============================================================================
# Repository Management (Google Monorepo Pattern)
# ============================================================================

git::is_repo() {
    git rev-parse --git-dir &>/dev/null
}

git::root() {
    git rev-parse --show-toplevel 2>/dev/null
}

git::current_branch() {
    git rev-parse --abbrev-ref HEAD 2>/dev/null
}

git::default_branch() {
    # Try to detect default branch (main, master, develop)
    local default_branches=("main" "master" "develop")
    
    for branch in "${default_branches[@]}"; do
        if git show-ref --verify --quiet "refs/heads/$branch"; then
            echo "$branch"
            return 0
        fi
    done
    
    # Fallback to remote HEAD
    git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@'
}

git::has_changes() {
    ! git diff-index --quiet HEAD --
}

git::has_staged() {
    ! git diff-index --quiet --cached HEAD --
}

git::has_untracked() {
    [[ -n "$(git ls-files --others --exclude-standard)" ]]
}

# ============================================================================
# Commit Operations (Meta's Rapid Release Pattern)
# ============================================================================

git::commit() {
    local message="$1"
    local options="${2:-}"
    
    if ! git::has_staged; then
        log::warn "No staged changes to commit"
        return 1
    fi
    
    log::info "Creating commit: $message"
    
    # Add co-author if in pair programming mode
    if [[ -n "${GIT_CO_AUTHOR:-}" ]]; then
        message="$message

Co-authored-by: $GIT_CO_AUTHOR"
    fi
    
    git commit -m "$message" $options
}

git::commit_all() {
    local message="$1"
    
    if ! git::has_changes && ! git::has_untracked; then
        log::info "No changes to commit"
        return 0
    fi
    
    log::info "Staging all changes..."
    git add -A
    
    git::commit "$message"
}

git::amend() {
    local message="${1:-}"
    
    if [[ -n "$message" ]]; then
        git commit --amend -m "$message"
    else
        git commit --amend --no-edit
    fi
}

# ============================================================================
# Branch Management (Netflix Trunk-Based Development Pattern)
# ============================================================================

git::create_branch() {
    local branch="$1"
    local from="${2:-$(git::default_branch)}"
    
    log::info "Creating branch '$branch' from '$from'"
    
    # Ensure we have latest from remote
    git fetch origin "$from"
    
    # Create and checkout new branch
    git checkout -b "$branch" "origin/$from"
}

git::switch_branch() {
    local branch="$1"
    
    if git::has_changes; then
        log::warn "You have uncommitted changes. Stashing them..."
        git stash push -m "Auto-stash before switching to $branch"
    fi
    
    git checkout "$branch"
    
    # Apply stash if it exists
    if git stash list | grep -q "Auto-stash before switching to $branch"; then
        log::info "Applying stashed changes..."
        git stash pop
    fi
}

git::delete_branch() {
    local branch="$1"
    local force="${2:-false}"
    
    if [[ "$branch" == "$(git::current_branch)" ]]; then
        log::error "Cannot delete current branch"
        return 1
    fi
    
    if [[ "$force" == "true" ]]; then
        git branch -D "$branch"
        git push origin --delete "$branch" 2>/dev/null || true
    else
        git branch -d "$branch"
        git push origin --delete "$branch" 2>/dev/null || true
    fi
}

git::sync_branch() {
    local branch="${1:-$(git::current_branch)}"
    local upstream="${2:-$(git::default_branch)}"
    
    log::info "Syncing '$branch' with '$upstream'"
    
    # Save current work
    local stashed=false
    if git::has_changes; then
        git stash push -m "Auto-stash for sync"
        stashed=true
    fi
    
    # Update upstream
    git fetch origin "$upstream"
    
    # Rebase or merge based on preference
    if [[ "${GIT_SYNC_STRATEGY:-rebase}" == "rebase" ]]; then
        git rebase "origin/$upstream"
    else
        git merge "origin/$upstream"
    fi
    
    # Restore work
    if [[ "$stashed" == "true" ]]; then
        git stash pop
    fi
}

# ============================================================================
# Remote Operations (Dropbox CI/CD Pattern)
# ============================================================================

git::push() {
    local branch="${1:-$(git::current_branch)}"
    local force="${2:-false}"
    
    log::info "Pushing branch '$branch' to origin"
    
    if [[ "$force" == "true" ]]; then
        git push --force-with-lease origin "$branch"
    else
        git push origin "$branch"
    fi
}

git::pull() {
    local branch="${1:-$(git::current_branch)}"
    
    log::info "Pulling latest changes for '$branch'"
    
    # Use rebase by default to maintain linear history
    if [[ "${GIT_PULL_STRATEGY:-rebase}" == "rebase" ]]; then
        git pull --rebase origin "$branch"
    else
        git pull origin "$branch"
    fi
}

git::fetch_all() {
    log::info "Fetching all remotes..."
    git fetch --all --prune
}

# ============================================================================
# Tag Management (Semantic Versioning Pattern)
# ============================================================================

git::tag() {
    local tag="$1"
    local message="${2:-Release $tag}"
    
    log::info "Creating tag: $tag"
    
    git tag -a "$tag" -m "$message"
    git push origin "$tag"
}

git::latest_tag() {
    git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0"
}

git::next_version() {
    local current=$(git::latest_tag)
    local bump_type="${1:-patch}"  # major, minor, patch
    
    # Remove 'v' prefix if present
    current="${current#v}"
    
    # Parse version
    IFS='.' read -r major minor patch <<< "$current"
    
    case "$bump_type" in
        major)
            ((major++))
            minor=0
            patch=0
            ;;
        minor)
            ((minor++))
            patch=0
            ;;
        patch)
            ((patch++))
            ;;
    esac
    
    echo "v${major}.${minor}.${patch}"
}

# ============================================================================
# GitHub Integration (Meta's Code Review Pattern)
# ============================================================================

git::create_pr() {
    local title="$1"
    local body="${2:-}"
    local base="${3:-$(git::default_branch)}"
    local draft="${4:-false}"
    
    if ! command -v gh &>/dev/null; then
        log::error "GitHub CLI (gh) is not installed"
        return 1
    fi
    
    log::info "Creating pull request: $title"
    
    local options=""
    [[ "$draft" == "true" ]] && options="--draft"
    
    gh pr create \
        --title "$title" \
        --body "$body" \
        --base "$base" \
        $options
}

git::pr_status() {
    if ! command -v gh &>/dev/null; then
        log::error "GitHub CLI (gh) is not installed"
        return 1
    fi
    
    gh pr status
}

# ============================================================================
# Hooks Management (Google's Code Quality Pattern)
# ============================================================================

git::install_hooks() {
    local hooks_dir="${1:-.githooks}"
    
    if [[ ! -d "$hooks_dir" ]]; then
        log::error "Hooks directory not found: $hooks_dir"
        return 1
    fi
    
    log::info "Installing Git hooks from $hooks_dir"
    
    # Set hooks path
    git config core.hooksPath "$hooks_dir"
    
    # Make hooks executable
    find "$hooks_dir" -type f -exec chmod +x {} \;
    
    log::info "Git hooks installed successfully"
}

git::run_pre_commit() {
    local hook=".git/hooks/pre-commit"
    
    if [[ -x "$hook" ]]; then
        log::info "Running pre-commit hooks..."
        "$hook"
    else
        log::debug "No pre-commit hook found"
    fi
}

# ============================================================================
# Cleanup Operations (Netflix's Trunk-Based Pattern)
# ============================================================================

git::clean_branches() {
    local dry_run="${1:-true}"
    
    log::info "Cleaning up merged branches..."
    
    # Get default branch
    local default_branch=$(git::default_branch)
    
    # Switch to default branch
    git checkout "$default_branch"
    git pull origin "$default_branch"
    
    # Find merged branches
    local branches=$(git branch --merged | grep -v "^\*\|$default_branch\|main\|master\|develop")
    
    if [[ -z "$branches" ]]; then
        log::info "No branches to clean"
        return 0
    fi
    
    echo "Branches to delete:"
    echo "$branches"
    
    if [[ "$dry_run" == "false" ]]; then
        echo "$branches" | xargs -n 1 git branch -d
        log::info "Branches deleted"
    else
        log::info "Run with dry_run=false to actually delete branches"
    fi
}

git::gc() {
    log::info "Running Git garbage collection..."
    git gc --aggressive --prune=now
}

# ============================================================================
# Status and Information (Developer Experience Pattern)
# ============================================================================

git::status_summary() {
    local format="${1:-plain}"
    
    case "$format" in
        json)
            echo "{"
            echo "  \"branch\": \"$(git::current_branch)\","
            echo "  \"has_changes\": $(git::has_changes && echo "true" || echo "false"),"
            echo "  \"has_staged\": $(git::has_staged && echo "true" || echo "false"),"
            echo "  \"has_untracked\": $(git::has_untracked && echo "true" || echo "false"),"
            echo "  \"ahead\": $(git rev-list --count HEAD..@{u} 2>/dev/null || echo 0),"
            echo "  \"behind\": $(git rev-list --count @{u}..HEAD 2>/dev/null || echo 0)"
            echo "}"
            ;;
        *)
            echo "Branch: $(git::current_branch)"
            echo "Changes: $(git::has_changes && echo "Yes" || echo "No")"
            echo "Staged: $(git::has_staged && echo "Yes" || echo "No")"
            echo "Untracked: $(git::has_untracked && echo "Yes" || echo "No")"
            ;;
    esac
}

git::log_pretty() {
    local limit="${1:-10}"
    
    git log \
        --pretty=format:"%C(yellow)%h%Creset %C(blue)%an%Creset %C(cyan)%ar%Creset %s" \
        --graph \
        -n "$limit"
}