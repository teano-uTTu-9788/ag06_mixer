#!/bin/bash
# Homebrew automation library following macOS best practices
# Provides: Homebrew installation, package management, environment setup
# Used by: Bootstrap scripts, package installers

# shellcheck source=./core.sh
source "$(dirname "${BASH_SOURCE[0]}")/core.sh"

# Homebrew configuration
readonly HOMEBREW_PREFIX="${HOMEBREW_PREFIX:-$(command -v brew >/dev/null && brew --prefix || echo '/opt/homebrew')}"
readonly HOMEBREW_REPOSITORY="${HOMEBREW_PREFIX}/Homebrew"
readonly HOMEBREW_BREWFILE="${PROJECT_ROOT}/Brewfile"

# Homebrew installation (official method with safety checks)
install_homebrew() {
    if command -v brew >/dev/null 2>&1; then
        log_info "Homebrew already installed at: $(command -v brew)"
        return 0
    fi
    
    log_info "Installing Homebrew..."
    
    # Check system requirements
    if ! is_macos; then
        log_fatal "This script is designed for macOS only"
    fi
    
    # Check for Xcode command line tools
    if ! xcode-select -p >/dev/null 2>&1; then
        log_info "Installing Xcode Command Line Tools..."
        xcode-select --install
        log_warn "Please complete the Xcode Command Line Tools installation and run this script again"
        exit 1
    fi
    
    # Install Homebrew using official script
    timer_start
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    timer_end
    
    # Add to shell profile
    setup_homebrew_environment
    
    log_ok "Homebrew installation completed"
}

# Environment setup (Shell profile configuration)
setup_homebrew_environment() {
    local shell_profile
    
    # Determine shell profile file
    case "${SHELL##*/}" in
        bash) shell_profile="${HOME}/.bash_profile" ;;
        zsh) shell_profile="${HOME}/.zshrc" ;;
        *) 
            log_warn "Unsupported shell: ${SHELL}. Manual PATH setup may be required."
            return 1
            ;;
    esac
    
    # Check if Homebrew is already in PATH
    if [[ ":$PATH:" == *":${HOMEBREW_PREFIX}/bin:"* ]]; then
        log_debug "Homebrew already in PATH"
        return 0
    fi
    
    log_info "Setting up Homebrew environment in ${shell_profile}"
    
    # Create backup of shell profile
    if [[ -f "${shell_profile}" ]]; then
        cp "${shell_profile}" "${shell_profile}.backup.$(date +%s)"
    fi
    
    # Add Homebrew initialization
    {
        echo ""
        echo "# Homebrew initialization (added by automation framework)"
        echo "if [[ -f \"${HOMEBREW_PREFIX}/bin/brew\" ]]; then"
        echo "    eval \"\$(${HOMEBREW_PREFIX}/bin/brew shellenv)\""
        echo "fi"
    } >> "${shell_profile}"
    
    # Source the profile for current session
    # shellcheck disable=SC1090
    source "${shell_profile}"
    
    log_ok "Homebrew environment configured"
}

# Package management functions
brew_update() {
    log_info "Updating Homebrew..."
    
    timer_start
    brew update
    timer_end
    
    log_ok "Homebrew updated successfully"
}

brew_upgrade() {
    local packages=("$@")
    
    if [[ ${#packages[@]} -eq 0 ]]; then
        log_info "Upgrading all Homebrew packages..."
        timer_start
        brew upgrade
        timer_end
    else
        log_info "Upgrading specific packages: ${packages[*]}"
        timer_start
        brew upgrade "${packages[@]}"
        timer_end
    fi
    
    log_ok "Package upgrade completed"
}

brew_install() {
    local packages=("$@")
    local failed_packages=()
    
    if [[ ${#packages[@]} -eq 0 ]]; then
        log_warn "No packages specified for installation"
        return 1
    fi
    
    log_info "Installing packages: ${packages[*]}"
    
    for package in "${packages[@]}"; do
        if brew list "${package}" >/dev/null 2>&1; then
            log_debug "Package '${package}' already installed"
            continue
        fi
        
        log_info "Installing ${package}..."
        if retry 3 2 brew install "${package}"; then
            log_ok "Successfully installed ${package}"
        else
            log_error "Failed to install ${package}"
            failed_packages+=("${package}")
        fi
    done
    
    if [[ ${#failed_packages[@]} -gt 0 ]]; then
        log_error "Failed to install packages: ${failed_packages[*]}"
        return 1
    fi
    
    log_ok "All packages installed successfully"
}

brew_cask_install() {
    local casks=("$@")
    local failed_casks=()
    
    if [[ ${#casks[@]} -eq 0 ]]; then
        log_warn "No casks specified for installation"
        return 1
    fi
    
    log_info "Installing casks: ${casks[*]}"
    
    for cask in "${casks[@]}"; do
        if brew list --cask "${cask}" >/dev/null 2>&1; then
            log_debug "Cask '${cask}' already installed"
            continue
        fi
        
        log_info "Installing ${cask}..."
        if retry 3 2 brew install --cask "${cask}"; then
            log_ok "Successfully installed ${cask}"
        else
            log_error "Failed to install ${cask}"
            failed_casks+=("${cask}")
        fi
    done
    
    if [[ ${#failed_casks[@]} -gt 0 ]]; then
        log_error "Failed to install casks: ${failed_casks[*]}"
        return 1
    fi
    
    log_ok "All casks installed successfully"
}

# Brewfile management (Bundler-style dependency management)
brew_bundle_install() {
    local brewfile="${1:-${HOMEBREW_BREWFILE}}"
    
    validate_file "${brewfile}"
    
    log_info "Installing packages from Brewfile: ${brewfile}"
    
    timer_start
    if brew bundle install --file="${brewfile}"; then
        timer_end
        log_ok "Brewfile installation completed"
    else
        timer_end
        log_error "Brewfile installation failed"
        return 1
    fi
}

brew_bundle_check() {
    local brewfile="${1:-${HOMEBREW_BREWFILE}}"
    
    validate_file "${brewfile}"
    
    log_info "Checking Brewfile dependencies: ${brewfile}"
    
    if brew bundle check --file="${brewfile}"; then
        log_ok "All Brewfile dependencies satisfied"
        return 0
    else
        log_warn "Some Brewfile dependencies missing"
        return 1
    fi
}

brew_bundle_cleanup() {
    local brewfile="${1:-${HOMEBREW_BREWFILE}}"
    
    validate_file "${brewfile}"
    
    log_info "Cleaning up packages not in Brewfile"
    
    if confirm "This will uninstall packages not listed in the Brewfile. Continue?" "n"; then
        brew bundle cleanup --file="${brewfile}"
        log_ok "Cleanup completed"
    else
        log_info "Cleanup cancelled"
    fi
}

# Service management (launchd integration)
brew_service_start() {
    local service="$1"
    
    if [[ -z "${service}" ]]; then
        log_error "Service name required"
        return 1
    fi
    
    log_info "Starting service: ${service}"
    
    if brew services start "${service}"; then
        log_ok "Service ${service} started"
    else
        log_error "Failed to start service ${service}"
        return 1
    fi
}

brew_service_stop() {
    local service="$1"
    
    if [[ -z "${service}" ]]; then
        log_error "Service name required"
        return 1
    fi
    
    log_info "Stopping service: ${service}"
    
    if brew services stop "${service}"; then
        log_ok "Service ${service} stopped"
    else
        log_error "Failed to stop service ${service}"
        return 1
    fi
}

brew_service_restart() {
    local service="$1"
    
    if [[ -z "${service}" ]]; then
        log_error "Service name required"
        return 1
    fi
    
    log_info "Restarting service: ${service}"
    
    if brew services restart "${service}"; then
        log_ok "Service ${service} restarted"
    else
        log_error "Failed to restart service ${service}"
        return 1
    fi
}

brew_services_list() {
    log_info "Listing Homebrew services:"
    brew services list
}

# Maintenance functions
brew_cleanup() {
    log_info "Cleaning up Homebrew cache and outdated packages..."
    
    timer_start
    brew cleanup
    timer_end
    
    log_ok "Homebrew cleanup completed"
}

brew_doctor() {
    log_info "Running Homebrew diagnostics..."
    
    if brew doctor; then
        log_ok "Homebrew is healthy"
        return 0
    else
        log_warn "Homebrew has some issues that may need attention"
        return 1
    fi
}

brew_info_system() {
    log_info "Homebrew system information:"
    echo "----------------------------------------"
    echo "Homebrew Version: $(brew --version | head -n1)"
    echo "Homebrew Prefix: ${HOMEBREW_PREFIX}"
    echo "Homebrew Repository: ${HOMEBREW_REPOSITORY}"
    echo "Architecture: $(get_arch)"
    echo "Operating System: $(get_os)"
    echo "Installed Packages: $(brew list | wc -l)"
    echo "Installed Casks: $(brew list --cask 2>/dev/null | wc -l || echo "0")"
    echo "----------------------------------------"
}

# Tap management
brew_tap_add() {
    local tap="$1"
    
    if [[ -z "${tap}" ]]; then
        log_error "Tap name required"
        return 1
    fi
    
    if brew tap | grep -q "^${tap}$"; then
        log_debug "Tap '${tap}' already added"
        return 0
    fi
    
    log_info "Adding tap: ${tap}"
    
    if brew tap "${tap}"; then
        log_ok "Tap ${tap} added successfully"
    else
        log_error "Failed to add tap ${tap}"
        return 1
    fi
}

brew_tap_remove() {
    local tap="$1"
    
    if [[ -z "${tap}" ]]; then
        log_error "Tap name required"
        return 1
    fi
    
    if ! brew tap | grep -q "^${tap}$"; then
        log_debug "Tap '${tap}' not found"
        return 0
    fi
    
    log_info "Removing tap: ${tap}"
    
    if brew untap "${tap}"; then
        log_ok "Tap ${tap} removed successfully"
    else
        log_error "Failed to remove tap ${tap}"
        return 1
    fi
}

# Export all functions
export -f install_homebrew setup_homebrew_environment
export -f brew_update brew_upgrade brew_install brew_cask_install
export -f brew_bundle_install brew_bundle_check brew_bundle_cleanup
export -f brew_service_start brew_service_stop brew_service_restart brew_services_list
export -f brew_cleanup brew_doctor brew_info_system
export -f brew_tap_add brew_tap_remove