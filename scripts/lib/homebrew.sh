#!/usr/bin/env bash
# Homebrew Management Library - Enterprise Pattern
# Provides Homebrew automation and package management

# Require core library
deps::require "core"

# ============================================================================
# Homebrew Installation (Enterprise Control Pattern)
# ============================================================================

brew::install() {
    if brew::is_installed; then
        log::info "Homebrew is already installed"
        return 0
    fi
    
    log::info "Installing Homebrew..."
    
    # Non-interactive installation (Enterprise pattern)
    NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || {
        log::error "Failed to install Homebrew"
        return 1
    }
    
    # Setup PATH for current session
    brew::setup_path
    
    log::info "Homebrew installed successfully"
}

brew::is_installed() {
    command -v brew &> /dev/null
}

brew::setup_path() {
    if platform::is_macos; then
        if [[ "$(platform::arch)" == "arm64" ]]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        else
            eval "$(/usr/local/bin/brew shellenv)"
        fi
    fi
}

brew::version() {
    brew --version 2>/dev/null | head -n1 | awk '{print $2}'
}

# ============================================================================
# Package Management (Netflix Productivity Pattern)
# ============================================================================

brew::tap() {
    local tap="$1"
    
    if brew tap | grep -q "^$tap\$"; then
        log::debug "Tap '$tap' already added"
        return 0
    fi
    
    log::info "Adding tap: $tap"
    retry::exponential_backoff 3 2 10 brew tap "$tap"
}

brew::install_package() {
    local package="$1"
    local options="${2:-}"
    
    if brew::is_package_installed "$package"; then
        log::debug "Package '$package' is already installed"
        return 0
    fi
    
    log::info "Installing package: $package"
    
    if [[ -n "$options" ]]; then
        retry::exponential_backoff 3 2 10 brew install "$package" $options
    else
        retry::exponential_backoff 3 2 10 brew install "$package"
    fi
}

brew::install_cask() {
    local cask="$1"
    local options="${2:-}"
    
    if brew::is_cask_installed "$cask"; then
        log::debug "Cask '$cask' is already installed"
        return 0
    fi
    
    log::info "Installing cask: $cask"
    
    if [[ -n "$options" ]]; then
        retry::exponential_backoff 3 2 10 brew install --cask "$cask" $options
    else
        retry::exponential_backoff 3 2 10 brew install --cask "$cask"
    fi
}

brew::is_package_installed() {
    local package="$1"
    brew list --formula | grep -q "^${package}\$"
}

brew::is_cask_installed() {
    local cask="$1"
    brew list --cask | grep -q "^${cask}\$"
}

# ============================================================================
# Brewfile Management (Dropbox's Dependency Pattern)
# ============================================================================

brew::bundle_install() {
    local brewfile="${1:-Brewfile}"
    
    if [[ ! -f "$brewfile" ]]; then
        log::error "Brewfile not found: $brewfile"
        return 1
    fi
    
    log::info "Installing dependencies from Brewfile: $brewfile"
    
    # Use --no-lock to avoid Brewfile.lock.json in CI
    local options="--file=$brewfile"
    [[ "${CI:-false}" == "true" ]] && options="$options --no-lock"
    
    retry::exponential_backoff 3 5 30 brew bundle $options
}

brew::bundle_cleanup() {
    local brewfile="${1:-Brewfile}"
    
    log::info "Cleaning up packages not in Brewfile"
    brew bundle cleanup --file="$brewfile" --force
}

brew::bundle_check() {
    local brewfile="${1:-Brewfile}"
    
    if brew bundle check --file="$brewfile" &>/dev/null; then
        log::info "All Brewfile dependencies are satisfied"
        return 0
    else
        log::warn "Some Brewfile dependencies are missing"
        return 1
    fi
}

brew::generate_brewfile() {
    local output="${1:-Brewfile}"
    
    log::info "Generating Brewfile: $output"
    
    {
        echo "# Generated Brewfile - $(date)"
        echo ""
        echo "# Taps"
        brew tap | while read -r tap; do
            echo "tap \"$tap\""
        done
        echo ""
        echo "# Formulae"
        brew list --formula | while read -r formula; do
            echo "brew \"$formula\""
        done
        echo ""
        echo "# Casks"
        brew list --cask | while read -r cask; do
            echo "cask \"$cask\""
        done
    } > "$output"
    
    log::info "Brewfile generated at: $output"
}

# ============================================================================
# Maintenance Operations (Google's SRE Pattern)
# ============================================================================

brew::update() {
    log::info "Updating Homebrew..."
    retry::exponential_backoff 3 5 30 brew update
}

brew::upgrade() {
    local package="${1:-}"
    
    if [[ -n "$package" ]]; then
        log::info "Upgrading package: $package"
        brew upgrade "$package"
    else
        log::info "Upgrading all packages..."
        brew upgrade
    fi
}

brew::cleanup() {
    log::info "Cleaning up Homebrew cache..."
    brew cleanup -s
    
    # Remove old versions
    brew cleanup --prune=all
    
    # Clean cache
    rm -rf "$(brew --cache)"
    
    log::info "Homebrew cleanup complete"
}

brew::doctor() {
    log::info "Running Homebrew diagnostics..."
    
    if brew doctor; then
        log::info "Homebrew is healthy"
        return 0
    else
        log::warn "Homebrew has some issues. Run 'brew doctor' for details"
        return 1
    fi
}

# ============================================================================
# Analytics and Reporting (Meta's Monitoring Pattern)
# ============================================================================

brew::list_installed() {
    local output_format="${1:-plain}"  # plain, json, csv
    
    case "$output_format" in
        json)
            echo '{"formulae":['
            brew list --formula | while read -r formula; do
                echo "\"$formula\","
            done | sed '$ s/,$//'
            echo '],"casks":['
            brew list --cask | while read -r cask; do
                echo "\"$cask\","
            done | sed '$ s/,$//'
            echo ']}'
            ;;
        csv)
            echo "Type,Name,Version"
            brew list --formula --versions | while read -r line; do
                echo "formula,$line"
            done
            brew list --cask --versions | while read -r line; do
                echo "cask,$line"
            done
            ;;
        *)
            echo "=== Formulae ==="
            brew list --formula --versions
            echo ""
            echo "=== Casks ==="
            brew list --cask --versions
            ;;
    esac
}

brew::outdated() {
    local formulae_count=$(brew outdated --formula | wc -l | tr -d ' ')
    local casks_count=$(brew outdated --cask | wc -l | tr -d ' ')
    
    if [[ $formulae_count -eq 0 && $casks_count -eq 0 ]]; then
        log::info "All packages are up to date"
        return 0
    fi
    
    log::warn "Found $formulae_count outdated formulae and $casks_count outdated casks"
    
    if [[ $formulae_count -gt 0 ]]; then
        echo "Outdated formulae:"
        brew outdated --formula --verbose
    fi
    
    if [[ $casks_count -gt 0 ]]; then
        echo "Outdated casks:"
        brew outdated --cask --verbose
    fi
    
    return 1
}

# ============================================================================
# Services Management (Netflix's Service Pattern)
# ============================================================================

brew::service_start() {
    local service="$1"
    
    log::info "Starting service: $service"
    brew services start "$service"
}

brew::service_stop() {
    local service="$1"
    
    log::info "Stopping service: $service"
    brew services stop "$service"
}

brew::service_restart() {
    local service="$1"
    
    log::info "Restarting service: $service"
    brew services restart "$service"
}

brew::service_list() {
    brew services list
}

# ============================================================================
# Security Operations (Google Security Pattern)
# ============================================================================

brew::audit() {
    local package="${1:-}"
    
    if [[ -n "$package" ]]; then
        log::info "Auditing package: $package"
        brew audit "$package"
    else
        log::info "Auditing all packages..."
        brew audit --installed
    fi
}

brew::verify_checksums() {
    log::info "Verifying package checksums..."
    
    local failed=0
    brew list --formula | while read -r formula; do
        if ! brew fetch --force --retry "$formula" &>/dev/null; then
            log::warn "Checksum verification failed for: $formula"
            ((failed++))
        fi
    done
    
    if [[ $failed -eq 0 ]]; then
        log::info "All checksums verified successfully"
        return 0
    else
        log::error "$failed packages failed checksum verification"
        return 1
    fi
}