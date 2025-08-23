#!/usr/bin/env bash
# shellcheck shell=bash
# Git workflow utilities

# Source dependencies
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./log.sh
source "${SCRIPT_DIR}/log.sh"
# shellcheck source=./utils.sh
source "${SCRIPT_DIR}/utils.sh"

require_clean_work_tree() {
  git update-index -q --refresh
  
  if ! git diff-files --quiet --ignore-submodules --; then
    die "Unstaged changes present"
  fi
  
  if ! git diff-index --cached --quiet HEAD --ignore-submodules --; then
    die "Uncommitted changes present"
  fi
}

get_default_branch() {
  # Try to get from remote
  local default_branch
  default_branch=$(git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@')
  
  if [[ -z "$default_branch" ]]; then
    # Fallback to common defaults
    if git show-ref --verify --quiet refs/heads/main; then
      echo "main"
    elif git show-ref --verify --quiet refs/heads/master; then
      echo "master"
    else
      echo "main"
    fi
  else
    echo "$default_branch"
  fi
}

create_feature_branch() {
  local branch_name="$1"
  local base_branch="${2:-$(get_default_branch)}"
  
  log_info "Creating feature branch: $branch_name from $base_branch"
  
  # Ensure we're up to date
  git fetch origin "$base_branch"
  
  # Create and checkout new branch
  git checkout -b "$branch_name" "origin/$base_branch"
  
  log_ok "Feature branch created: $branch_name"
}

push_current_branch() {
  local current_branch
  current_branch=$(current_branch)
  
  log_info "Pushing branch: $current_branch"
  git push -u origin "$current_branch"
  log_ok "Branch pushed"
}

create_pull_request() {
  local title="${1:-chore: automation update}"
  local body="${2:-Automated via dev CLI}"
  
  if ! command -v gh >/dev/null 2>&1; then
    die "GitHub CLI (gh) not installed"
  fi
  
  log_info "Creating pull request: $title"
  gh pr create -t "$title" -b "$body" --web
}

sync_with_upstream() {
  local upstream="${1:-upstream}"
  local branch="${2:-$(get_default_branch)}"
  
  # Check if upstream exists
  if ! git remote | grep -q "^${upstream}$"; then
    log_error "Upstream remote '${upstream}' not configured"
    log_info "Add with: git remote add ${upstream} <upstream-url>"
    return 1
  fi
  
  log_info "Syncing with upstream/${branch}"
  
  git fetch "$upstream"
  git checkout "$branch"
  git merge "${upstream}/${branch}"
  git push origin "$branch"
  
  log_ok "Synced with upstream"
}

# Conventional commit helper
conventional_commit() {
  local type="$1"
  local message="$2"
  local scope="${3:-}"
  
  local valid_types=("feat" "fix" "docs" "style" "refactor" "perf" "test" "build" "ci" "chore" "revert")
  
  # Validate type
  local valid=false
  for t in "${valid_types[@]}"; do
    if [[ "$t" == "$type" ]]; then
      valid=true
      break
    fi
  done
  
  if [[ "$valid" != "true" ]]; then
    log_error "Invalid commit type: $type"
    log_info "Valid types: ${valid_types[*]}"
    return 1
  fi
  
  # Build commit message
  local commit_msg="$type"
  if [[ -n "$scope" ]]; then
    commit_msg="${commit_msg}(${scope})"
  fi
  commit_msg="${commit_msg}: ${message}"
  
  log_info "Committing: $commit_msg"
  git commit -m "$commit_msg"
  log_ok "Committed"
}