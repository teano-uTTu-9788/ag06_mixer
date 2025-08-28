#!/usr/bin/env bash
# Core framework utilities
# Based on Google Shell Style Guide and Netflix microservices patterns

set -euo pipefail

# Framework configuration
export FRAMEWORK_VERSION="1.0.0"
export FRAMEWORK_DEBUG="${FRAMEWORK_DEBUG:-false}"

# Color definitions
export RED='\033[0;31m'
export GREEN='\033[0;32m'
export YELLOW='\033[1;33m'
export BLUE='\033[0;34m'
export BOLD='\033[1m'
export RESET='\033[0m'

# ============================================================================
# Basic Logging (Simplified)
# ============================================================================

log::info() {
    echo -e "${BLUE}[INFO]${RESET} $*" >&2
}

log::success() {
    echo -e "${GREEN}[SUCCESS]${RESET} $*" >&2
}

log::warn() {
    echo -e "${YELLOW}[WARN]${RESET} $*" >&2
}

log::error() {
    echo -e "${RED}[ERROR]${RESET} $*" >&2
}

log::debug() {
    if [[ "$FRAMEWORK_DEBUG" == "true" ]]; then
        echo -e "[DEBUG] $*" >&2
    fi
}

# ============================================================================
# Library Loading System (Simplified)
# ============================================================================

# Use simple approach for library loading to avoid Bash 3.2 issues
LOADED_LIBS=""
AUTOMATION_LIB_DIR="${AUTOMATION_LIB_DIR:-$(dirname "${BASH_SOURCE[0]}")/lib}"

deps::require() {
    local lib_name="$1"
    local lib_path="${AUTOMATION_LIB_DIR}/${lib_name}.sh"
    
    # Check if already loaded (simple string check)
    if [[ "$LOADED_LIBS" == *"$lib_name"* ]]; then
        return 0  # Already loaded
    fi
    
    if [[ ! -f "$lib_path" ]]; then
        log::debug "Library not found (optional): $lib_path"
        return 0  # Make it optional
    fi
    
    # shellcheck source=/dev/null
    source "$lib_path"
    LOADED_LIBS="$LOADED_LIBS $lib_name"
    
    log::debug "Loaded library: $lib_name"
}

# ============================================================================
# Framework Bootstrap Functions
# ============================================================================

framework::bootstrap() {
    log::info "Bootstrapping development environment..."
    
    # Load required dependencies
    deps::require "homebrew"
    deps::require "git"
    
    # Install Homebrew if needed
    if ! brew::is_installed; then
        homebrew::install
    fi
    
    # Install from Brewfile
    if [[ -f "Brewfile" ]]; then
        homebrew::bundle_install
    fi
    
    # Setup development environment
    _setup_python_env
    _setup_node_env
    _setup_git_hooks
    
    log::success "Bootstrap complete!"
}

framework::doctor() {
    log::info "Running system health check..."
    
    local issues=0
    
    # Check required commands
    local required_commands=("git" "brew")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &>/dev/null; then
            log::error "Required command not found: $cmd"
            ((issues++))
        else
            log::success "✓ $cmd installed"
        fi
    done
    
    # Check Homebrew health
    deps::require "homebrew"
    if homebrew::is_installed; then
        if homebrew::doctor_check; then
            log::success "✓ Homebrew is healthy"
        else
            log::warn "⚠ Homebrew has issues"
            ((issues++))
        fi
    fi
    
    # Report results
    if [[ $issues -eq 0 ]]; then
        log::success "All checks passed! System is healthy."
        return 0
    else
        log::error "Found $issues issues. Run './dev bootstrap' to fix."
        return 1
    fi
}

framework::format() {
    log::info "Formatting code..."
    
    # Python formatting
    if command -v black &>/dev/null; then
        log::info "Running black..."
        black .
    fi
    
    # Shell formatting
    if command -v shfmt &>/dev/null; then
        log::info "Running shfmt..."
        find . -name "*.sh" -exec shfmt -w -i 4 {} \;
    fi
    
    log::success "Formatting complete!"
}

framework::lint() {
    log::info "Running linters..."
    
    local failed=0
    
    # Shell linting
    if command -v shellcheck &>/dev/null; then
        log::info "Running shellcheck..."
        find . -name "*.sh" -exec shellcheck {} \; || ((failed++))
    fi
    
    # Python linting
    if command -v ruff &>/dev/null; then
        log::info "Running ruff..."
        ruff check . || ((failed++))
    fi
    
    if [[ $failed -eq 0 ]]; then
        log::success "All linters passed!"
        return 0
    else
        log::error "$failed linters reported issues"
        return 1
    fi
}

framework::test() {
    log::info "Running tests..."
    
    deps::require "testing"
    
    # Run shell tests if available
    if testing::has_bats; then
        testing::run_bats
    fi
    
    # Run Python tests if available
    if testing::has_python; then
        testing::run_python
    fi
    
    log::success "Tests complete!"
}

framework::ci() {
    log::info "Running CI checks locally..."
    
    local failed=0
    
    # Run quality checks
    framework::format || ((failed++))
    framework::lint || ((failed++))
    framework::test || ((failed++))
    
    if [[ $failed -eq 0 ]]; then
        log::success "All CI checks passed! ✅"
        return 0
    else
        log::error "$failed CI checks failed ❌"
        return 1
    fi
}

framework::build() {
    log::info "Building project..."
    
    # Python build
    if [[ -f "pyproject.toml" ]]; then
        log::info "Building Python project..."
        python -m build
    fi
    
    # Node build
    if [[ -f "package.json" ]]; then
        log::info "Building Node project..."
        npm run build || true
    fi
    
    log::success "Build complete!"
}

framework::deploy() {
    local environment="${1:-staging}"
    log::info "Deploying to $environment..."
    
    # Pre-deployment checks
    if ! framework::ci; then
        log::error "CI checks failed. Aborting deployment."
        return 1
    fi
    
    log::info "Deployment would continue here..."
    log::success "Deployment simulation complete!"
}

framework::clean() {
    log::info "Cleaning build artifacts..."
    
    # Python cleanup
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    rm -rf build/ dist/ .pytest_cache/ 2>/dev/null || true
    
    # Node cleanup
    rm -rf node_modules/ dist/ 2>/dev/null || true
    
    log::success "Cleanup complete!"
}

framework::help() {
    echo "${BOLD}${BLUE}Developer CLI - Terminal Automation Framework${RESET}"
    echo ""
    echo "Usage: ./dev <command> [options]"
    echo ""
    echo "Commands:"
    echo "  ${GREEN}bootstrap${RESET}    Setup development environment"
    echo "  ${GREEN}doctor${RESET}       Check system health"
    echo "  ${GREEN}build${RESET}        Build the project"
    echo "  ${GREEN}test${RESET}         Run test suite"
    echo "  ${GREEN}lint${RESET}         Run code linters"
    echo "  ${GREEN}fmt${RESET}          Format code"
    echo "  ${GREEN}ci${RESET}           Run CI checks locally"
    echo "  ${GREEN}deploy${RESET}       Deploy to environment"
    echo "  ${GREEN}clean${RESET}        Clean build artifacts"
    echo "  ${GREEN}help${RESET}         Show this help"
    echo ""
    echo "Examples:"
    echo "  ./dev bootstrap    # Setup development environment"
    echo "  ./dev ci           # Run all quality checks"
    echo "  ./dev deploy prod  # Deploy to production"
}

framework::version() {
    echo "Terminal Automation Framework v${FRAMEWORK_VERSION}"
}

# ============================================================================
# Private Helper Functions
# ============================================================================

_setup_python_env() {
    if command -v python3 &>/dev/null; then
        log::info "Setting up Python environment..."
        if [[ ! -d ".venv" ]]; then
            python3 -m venv .venv
        fi
        if [[ -f "requirements.txt" ]]; then
            .venv/bin/pip install -r requirements.txt
        fi
    fi
}

_setup_node_env() {
    if command -v npm &>/dev/null && [[ -f "package.json" ]]; then
        log::info "Installing Node dependencies..."
        npm install
    fi
}

_setup_git_hooks() {
    if [[ -d ".githooks" ]]; then
        deps::require "git"
        git::install_hooks
    fi
}