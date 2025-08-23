#!/usr/bin/env bash
# Git workflow automation following GitHub Flow and best practices
# Implements feature branches, pull requests, and automated workflows

set -euo pipefail

# Script metadata
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly FRAMEWORK_ROOT="$(dirname "$(dirname "${SCRIPT_DIR}")")"

# Source core libraries
source "${FRAMEWORK_ROOT}/lib/core/colors.sh"
source "${FRAMEWORK_ROOT}/lib/core/logger.sh"
source "${FRAMEWORK_ROOT}/lib/core/utils.sh"
source "${FRAMEWORK_ROOT}/lib/core/validation.sh"

# Git configuration
readonly DEFAULT_BRANCH="${GIT_DEFAULT_BRANCH:-main}"
readonly REMOTE_NAME="${GIT_REMOTE:-origin}"

# Check if in a git repository
check_git_repo() {
  if ! git rev-parse --git-dir >/dev/null 2>&1; then
    log_error "Not in a git repository"
    return 1
  fi
  return 0
}

# Get current branch name
get_current_branch() {
  git rev-parse --abbrev-ref HEAD
}

# Check if branch exists
branch_exists() {
  local branch="$1"
  git show-ref --verify --quiet "refs/heads/${branch}"
}

# Check if remote branch exists
remote_branch_exists() {
  local branch="$1"
  local remote="${2:-${REMOTE_NAME}}"
  git ls-remote --exit-code --heads "${remote}" "${branch}" >/dev/null 2>&1
}

# Feature branch workflow
feature_start() {
  local feature_name="$1"
  
  if ! check_git_repo; then
    return 1
  fi
  
  # Validate feature name
  if ! validate_alphanumeric "${feature_name}" true true; then
    log_error "Invalid feature name. Use only alphanumeric characters, dashes, and underscores"
    return 1
  fi
  
  local branch_name="feature/${feature_name}"
  
  # Check if branch already exists
  if branch_exists "${branch_name}"; then
    log_error "Branch '${branch_name}' already exists"
    return 1
  fi
  
  # Update main branch
  log_info "Updating ${DEFAULT_BRANCH} branch..."
  git checkout "${DEFAULT_BRANCH}"
  git pull "${REMOTE_NAME}" "${DEFAULT_BRANCH}"
  
  # Create and checkout feature branch
  log_info "Creating feature branch: ${branch_name}"
  git checkout -b "${branch_name}"
  
  log_success "Feature branch '${branch_name}' created and checked out"
  log_info "Start making your changes, then run: dev git feature-finish ${feature_name}"
}

feature_finish() {
  local feature_name="$1"
  local no_pr="${2:-false}"
  
  if ! check_git_repo; then
    return 1
  fi
  
  local branch_name="feature/${feature_name}"
  local current_branch
  current_branch="$(get_current_branch)"
  
  # Check if on the feature branch
  if [[ "${current_branch}" != "${branch_name}" ]]; then
    log_error "Not on branch '${branch_name}'. Current branch: ${current_branch}"
    return 1
  fi
  
  # Check for uncommitted changes
  if ! git diff-index --quiet HEAD --; then
    log_warning "You have uncommitted changes. Commit them first."
    return 1
  fi
  
  # Push to remote
  log_info "Pushing feature branch to remote..."
  git push -u "${REMOTE_NAME}" "${branch_name}"
  
  # Create pull request if not disabled
  if [[ "${no_pr}" != "true" ]]; then
    if command_exists gh; then
      log_info "Creating pull request..."
      gh pr create --base "${DEFAULT_BRANCH}" --head "${branch_name}" --web
    else
      log_info "GitHub CLI not installed. Create pull request manually at:"
      echo "  https://github.com/$(git remote get-url origin | sed 's/.*github.com[:/]\(.*\)\.git/\1/')/pull/new/${branch_name}"
    fi
  fi
  
  log_success "Feature branch pushed successfully"
}

# Hotfix workflow
hotfix_start() {
  local hotfix_name="$1"
  
  if ! check_git_repo; then
    return 1
  fi
  
  # Validate hotfix name
  if ! validate_alphanumeric "${hotfix_name}" true true; then
    log_error "Invalid hotfix name"
    return 1
  fi
  
  local branch_name="hotfix/${hotfix_name}"
  
  # Check if branch already exists
  if branch_exists "${branch_name}"; then
    log_error "Branch '${branch_name}' already exists"
    return 1
  fi
  
  # Create from main/master
  log_info "Creating hotfix branch from ${DEFAULT_BRANCH}..."
  git checkout "${DEFAULT_BRANCH}"
  git pull "${REMOTE_NAME}" "${DEFAULT_BRANCH}"
  git checkout -b "${branch_name}"
  
  log_success "Hotfix branch '${branch_name}' created"
}

hotfix_finish() {
  local hotfix_name="$1"
  
  if ! check_git_repo; then
    return 1
  fi
  
  local branch_name="hotfix/${hotfix_name}"
  local current_branch
  current_branch="$(get_current_branch)"
  
  if [[ "${current_branch}" != "${branch_name}" ]]; then
    log_error "Not on hotfix branch '${branch_name}'"
    return 1
  fi
  
  # Push and create PR
  log_info "Pushing hotfix branch..."
  git push -u "${REMOTE_NAME}" "${branch_name}"
  
  if command_exists gh; then
    log_info "Creating pull request for hotfix..."
    gh pr create --base "${DEFAULT_BRANCH}" --head "${branch_name}" --label "hotfix" --web
  fi
  
  log_success "Hotfix branch pushed successfully"
}

# Release workflow
release_start() {
  local version="$1"
  
  if ! check_git_repo; then
    return 1
  fi
  
  # Validate version
  if ! validate_semver "${version}"; then
    log_error "Invalid version format. Use semantic versioning (e.g., 1.2.3)"
    return 1
  fi
  
  local branch_name="release/${version}"
  
  if branch_exists "${branch_name}"; then
    log_error "Release branch '${branch_name}' already exists"
    return 1
  fi
  
  # Create release branch
  log_info "Creating release branch: ${branch_name}"
  git checkout "${DEFAULT_BRANCH}"
  git pull "${REMOTE_NAME}" "${DEFAULT_BRANCH}"
  git checkout -b "${branch_name}"
  
  log_success "Release branch '${branch_name}' created"
}

release_finish() {
  local version="$1"
  
  if ! check_git_repo; then
    return 1
  fi
  
  local branch_name="release/${version}"
  local current_branch
  current_branch="$(get_current_branch)"
  
  if [[ "${current_branch}" != "${branch_name}" ]]; then
    log_error "Not on release branch '${branch_name}'"
    return 1
  fi
  
  # Tag the release
  log_info "Creating release tag v${version}..."
  git tag -a "v${version}" -m "Release version ${version}"
  
  # Push branch and tag
  log_info "Pushing release branch and tag..."
  git push -u "${REMOTE_NAME}" "${branch_name}"
  git push "${REMOTE_NAME}" "v${version}"
  
  # Create GitHub release if gh is available
  if command_exists gh; then
    log_info "Creating GitHub release..."
    gh release create "v${version}" --generate-notes
  fi
  
  log_success "Release ${version} created successfully"
}

# Sync with upstream
sync_fork() {
  local upstream="${1:-upstream}"
  
  if ! check_git_repo; then
    return 1
  fi
  
  # Check if upstream remote exists
  if ! git remote | grep -q "^${upstream}$"; then
    log_error "Upstream remote '${upstream}' not configured"
    log_info "Add it with: git remote add ${upstream} <upstream-url>"
    return 1
  fi
  
  log_info "Syncing with upstream..."
  
  # Fetch upstream
  git fetch "${upstream}"
  
  # Checkout main branch
  git checkout "${DEFAULT_BRANCH}"
  
  # Merge upstream
  git merge "${upstream}/${DEFAULT_BRANCH}"
  
  # Push to origin
  git push "${REMOTE_NAME}" "${DEFAULT_BRANCH}"
  
  log_success "Fork synced with upstream"
}

# Clean up old branches
cleanup_branches() {
  local dry_run="${1:-false}"
  
  if ! check_git_repo; then
    return 1
  fi
  
  log_info "Finding merged branches..."
  
  # Get merged branches (excluding main/master and current)
  local current_branch
  current_branch="$(get_current_branch)"
  
  local merged_branches
  merged_branches=$(git branch --merged "${DEFAULT_BRANCH}" | 
    grep -v "^[* ] ${DEFAULT_BRANCH}$" | 
    grep -v "^[* ] ${current_branch}$" |
    sed 's/^[ *]*//')
  
  if [[ -z "${merged_branches}" ]]; then
    log_info "No merged branches to clean up"
    return 0
  fi
  
  log_info "Merged branches to delete:"
  echo "${merged_branches}" | log_indent
  
  if [[ "${dry_run}" == "true" ]]; then
    log_info "Dry run - no branches deleted"
    return 0
  fi
  
  # Confirm deletion
  echo
  read -rp "Delete these branches? [y/N] " confirm
  if [[ "${confirm}" != "y" ]]; then
    log_info "Cleanup cancelled"
    return 0
  fi
  
  # Delete branches
  echo "${merged_branches}" | while read -r branch; do
    log_info "Deleting branch: ${branch}"
    git branch -d "${branch}"
  done
  
  log_success "Branch cleanup complete"
}

# Stash operations
stash_save() {
  local message="${1:-WIP}"
  
  if ! check_git_repo; then
    return 1
  fi
  
  log_info "Stashing changes: ${message}"
  git stash push -m "${message}"
  log_success "Changes stashed"
}

stash_pop() {
  if ! check_git_repo; then
    return 1
  fi
  
  log_info "Applying stashed changes..."
  git stash pop
  log_success "Stash applied"
}

stash_list() {
  if ! check_git_repo; then
    return 1
  fi
  
  log_info "Stashed changes:"
  git stash list
}

# Commit with conventional commits
commit() {
  local type="$1"
  local message="$2"
  local scope="${3:-}"
  
  if ! check_git_repo; then
    return 1
  fi
  
  # Validate commit type
  local valid_types=("feat" "fix" "docs" "style" "refactor" "perf" "test" "build" "ci" "chore" "revert")
  if ! array_contains "${type}" "${valid_types[@]}"; then
    log_error "Invalid commit type: ${type}"
    echo "Valid types: ${valid_types[*]}"
    return 1
  fi
  
  # Build commit message
  local commit_message="${type}"
  [[ -n "${scope}" ]] && commit_message="${commit_message}(${scope})"
  commit_message="${commit_message}: ${message}"
  
  log_info "Committing: ${commit_message}"
  git commit -m "${commit_message}"
  log_success "Changes committed"
}

# Interactive rebase
rebase_interactive() {
  local commits="${1:-3}"
  
  if ! check_git_repo; then
    return 1
  fi
  
  if ! validate_integer "${commits}" 1; then
    log_error "Invalid number of commits: ${commits}"
    return 1
  fi
  
  log_info "Starting interactive rebase for last ${commits} commits..."
  git rebase -i "HEAD~${commits}"
}

# Show git status with formatting
status() {
  if ! check_git_repo; then
    return 1
  fi
  
  log_section "Git Status"
  
  # Current branch
  local current_branch
  current_branch="$(get_current_branch)"
  log_info "Current branch: ${BOLD}${current_branch}${RESET}"
  
  # Remote tracking
  local tracking
  tracking=$(git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null || echo "none")
  log_info "Tracking: ${tracking}"
  
  echo
  git status --short
}

# Main command handler
main() {
  local command="${1:-help}"
  shift || true
  
  case "${command}" in
    feature|feature-start)
      feature_start "$@"
      ;;
    feature-finish)
      feature_finish "$@"
      ;;
    hotfix|hotfix-start)
      hotfix_start "$@"
      ;;
    hotfix-finish)
      hotfix_finish "$@"
      ;;
    release|release-start)
      release_start "$@"
      ;;
    release-finish)
      release_finish "$@"
      ;;
    sync|sync-fork)
      sync_fork "$@"
      ;;
    cleanup)
      cleanup_branches "$@"
      ;;
    stash|stash-save)
      stash_save "$@"
      ;;
    stash-pop)
      stash_pop "$@"
      ;;
    stash-list)
      stash_list "$@"
      ;;
    commit)
      commit "$@"
      ;;
    rebase)
      rebase_interactive "$@"
      ;;
    status|st)
      status "$@"
      ;;
    help)
      cat <<EOF
${BOLD}Git Workflow Automation${RESET}

${BOLD}USAGE:${RESET}
    dev git <command> [options]

${BOLD}COMMANDS:${RESET}
    ${GREEN}feature${RESET} <name>        Start a feature branch
    ${GREEN}feature-finish${RESET} <name>  Finish feature (push & PR)
    ${GREEN}hotfix${RESET} <name>          Start a hotfix branch
    ${GREEN}hotfix-finish${RESET} <name>   Finish hotfix
    ${GREEN}release${RESET} <version>     Start a release branch
    ${GREEN}release-finish${RESET} <ver>  Finish release (tag & push)
    ${GREEN}sync${RESET} [upstream]       Sync fork with upstream
    ${GREEN}cleanup${RESET} [--dry-run]   Clean merged branches
    ${GREEN}stash${RESET} [message]       Stash changes
    ${GREEN}stash-pop${RESET}             Apply stashed changes
    ${GREEN}stash-list${RESET}            List stashes
    ${GREEN}commit${RESET} <type> <msg>   Conventional commit
    ${GREEN}rebase${RESET} [n]            Interactive rebase
    ${GREEN}status${RESET}                Show formatted status

${BOLD}COMMIT TYPES:${RESET}
    feat, fix, docs, style, refactor, perf, test,
    build, ci, chore, revert

${BOLD}EXAMPLES:${RESET}
    dev git feature my-new-feature
    dev git commit feat "Add new feature"
    dev git release 1.2.3
    dev git cleanup --dry-run
EOF
      ;;
    *)
      log_error "Unknown command: ${command}"
      echo "Run 'dev git help' for usage information."
      exit 1
      ;;
  esac
}

# Only run main if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi