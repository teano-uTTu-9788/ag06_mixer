#!/usr/bin/env bash

# Hermetic Environment Manager - Isolated build environments
# Ensures reproducible builds by controlling all dependencies

set -euo pipefail

HERMETIC_BASE="/tmp/hermetic_envs"
HERMETIC_CACHE="/tmp/hermetic_cache"

# Create hermetic environment
create_hermetic_env() {
    local env_name="${1:-default}"
    local env_dir="$HERMETIC_BASE/$env_name"
    
    echo "Creating hermetic environment: $env_name"
    
    # Create isolated directory structure
    mkdir -p "$env_dir"/{bin,lib,include,share,tmp,cache}
    mkdir -p "$HERMETIC_CACHE"
    
    # Create environment configuration
    cat > "$env_dir/env.conf" <<EOF
HERMETIC_ENV_NAME=$env_name
HERMETIC_ENV_DIR=$env_dir
HERMETIC_ENV_PATH=$env_dir/bin
HERMETIC_ENV_LIB=$env_dir/lib
HERMETIC_ENV_INCLUDE=$env_dir/include
HERMETIC_ENV_CACHE=$HERMETIC_CACHE/$env_name
HERMETIC_ENV_TMP=$env_dir/tmp
HERMETIC_ENV_CREATED=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF
    
    # Create activation script
    cat > "$env_dir/activate" <<'EOF'
#!/usr/bin/env bash

# Save original environment
export HERMETIC_OLD_PATH="$PATH"
export HERMETIC_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export HERMETIC_OLD_PKG_CONFIG_PATH="${PKG_CONFIG_PATH:-}"
export HERMETIC_OLD_PS1="${PS1:-}"

# Source environment configuration
source "$(dirname "${BASH_SOURCE[0]}")/env.conf"

# Set hermetic paths
export PATH="$HERMETIC_ENV_PATH:$PATH"
export LD_LIBRARY_PATH="$HERMETIC_ENV_LIB:$LD_LIBRARY_PATH"
export PKG_CONFIG_PATH="$HERMETIC_ENV_LIB/pkgconfig:$PKG_CONFIG_PATH"
export CPATH="$HERMETIC_ENV_INCLUDE:${CPATH:-}"
export LIBRARY_PATH="$HERMETIC_ENV_LIB:${LIBRARY_PATH:-}"

# Set build isolation flags
export HERMETIC_BUILD=1
export CC="${CC:-gcc} -I$HERMETIC_ENV_INCLUDE -L$HERMETIC_ENV_LIB"
export CXX="${CXX:-g++} -I$HERMETIC_ENV_INCLUDE -L$HERMETIC_ENV_LIB"
export LDFLAGS="-L$HERMETIC_ENV_LIB ${LDFLAGS:-}"
export CFLAGS="-I$HERMETIC_ENV_INCLUDE ${CFLAGS:-}"
export CXXFLAGS="-I$HERMETIC_ENV_INCLUDE ${CXXFLAGS:-}"

# Set cache and temp directories
export TMPDIR="$HERMETIC_ENV_TMP"
export TEMP="$HERMETIC_ENV_TMP"
export TMP="$HERMETIC_ENV_TMP"
export CACHE_DIR="$HERMETIC_ENV_CACHE"

# Update prompt
export PS1="[hermetic:$HERMETIC_ENV_NAME] $PS1"

# Deactivation function
deactivate_hermetic() {
    export PATH="$HERMETIC_OLD_PATH"
    export LD_LIBRARY_PATH="$HERMETIC_OLD_LD_LIBRARY_PATH"
    export PKG_CONFIG_PATH="$HERMETIC_OLD_PKG_CONFIG_PATH"
    export PS1="$HERMETIC_OLD_PS1"
    
    unset HERMETIC_BUILD HERMETIC_ENV_NAME HERMETIC_ENV_DIR
    unset HERMETIC_OLD_PATH HERMETIC_OLD_LD_LIBRARY_PATH
    unset HERMETIC_OLD_PKG_CONFIG_PATH HERMETIC_OLD_PS1
    unset -f deactivate_hermetic
}

echo "Hermetic environment '$HERMETIC_ENV_NAME' activated"
echo "Type 'deactivate_hermetic' to exit"
EOF
    
    chmod +x "$env_dir/activate"
    
    # Create wrapper scripts for common tools
    create_tool_wrappers "$env_dir"
    
    echo "Hermetic environment created: $env_dir"
    echo "To activate: source $env_dir/activate"
}

# Create tool wrappers for isolation
create_tool_wrappers() {
    local env_dir="$1"
    local bin_dir="$env_dir/bin"
    
    # Git wrapper with isolated config
    cat > "$bin_dir/git" <<'EOF'
#!/usr/bin/env bash
export GIT_CONFIG_NOSYSTEM=1
export HOME="$HERMETIC_ENV_DIR"
exec /usr/bin/git "$@"
EOF
    
    # NPM wrapper with isolated cache
    cat > "$bin_dir/npm" <<'EOF'
#!/usr/bin/env bash
export npm_config_cache="$HERMETIC_ENV_CACHE/npm"
export npm_config_prefix="$HERMETIC_ENV_DIR"
exec /usr/bin/npm "$@"
EOF
    
    # Python wrapper with isolated environment
    cat > "$bin_dir/python" <<'EOF'
#!/usr/bin/env bash
export PYTHONHOME="$HERMETIC_ENV_DIR"
export PYTHONPATH="$HERMETIC_ENV_LIB/python"
export PYTHONUSERBASE="$HERMETIC_ENV_DIR"
exec /usr/bin/python3 "$@"
EOF
    
    # Make all wrappers executable
    chmod +x "$bin_dir"/*
}

# Install dependency in hermetic environment
install_dependency() {
    local env_name="$1"
    local dep_type="$2"
    local dep_spec="$3"
    local env_dir="$HERMETIC_BASE/$env_name"
    
    if [[ ! -d "$env_dir" ]]; then
        echo "Environment not found: $env_name"
        return 1
    fi
    
    echo "Installing $dep_type dependency: $dep_spec"
    
    case "$dep_type" in
        npm)
            (
                source "$env_dir/activate"
                npm install --prefix "$env_dir" "$dep_spec"
            )
            ;;
        pip)
            (
                source "$env_dir/activate"
                pip install --target "$env_dir/lib/python" "$dep_spec"
            )
            ;;
        binary)
            # Download and install binary
            local binary_name=$(basename "$dep_spec")
            curl -L "$dep_spec" -o "$env_dir/bin/$binary_name"
            chmod +x "$env_dir/bin/$binary_name"
            ;;
        *)
            echo "Unknown dependency type: $dep_type"
            return 1
            ;;
    esac
}

# Lock environment dependencies
lock_dependencies() {
    local env_name="$1"
    local env_dir="$HERMETIC_BASE/$env_name"
    local lock_file="$env_dir/dependencies.lock"
    
    echo "Locking dependencies for environment: $env_name"
    
    # Generate dependency lock file
    cat > "$lock_file" <<EOF
{
  "environment": "$env_name",
  "created": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "dependencies": {
EOF
    
    # List npm dependencies
    if [[ -f "$env_dir/package.json" ]]; then
        echo '    "npm": {' >> "$lock_file"
        (cd "$env_dir" && npm list --json --depth=0 | jq -r '.dependencies | to_entries[] | "      \"" + .key + "\": \"" + .value.version + "\","') >> "$lock_file"
        echo '    },' >> "$lock_file"
    fi
    
    # List Python dependencies
    if [[ -d "$env_dir/lib/python" ]]; then
        echo '    "python": {' >> "$lock_file"
        (cd "$env_dir/lib/python" && pip list --format=json | jq -r '.[] | "      \"" + .name + "\": \"" + .version + "\","') >> "$lock_file"
        echo '    }' >> "$lock_file"
    fi
    
    echo '  }' >> "$lock_file"
    echo '}' >> "$lock_file"
    
    echo "Dependencies locked to: $lock_file"
}

# Verify environment integrity
verify_environment() {
    local env_name="$1"
    local env_dir="$HERMETIC_BASE/$env_name"
    
    echo "Verifying hermetic environment: $env_name"
    
    local issues=0
    
    # Check directory structure
    for dir in bin lib include share tmp cache; do
        if [[ ! -d "$env_dir/$dir" ]]; then
            echo "  ✗ Missing directory: $dir"
            ((issues++))
        else
            echo "  ✓ Directory exists: $dir"
        fi
    done
    
    # Check configuration
    if [[ ! -f "$env_dir/env.conf" ]]; then
        echo "  ✗ Missing env.conf"
        ((issues++))
    else
        echo "  ✓ Configuration exists"
    fi
    
    # Check activation script
    if [[ ! -x "$env_dir/activate" ]]; then
        echo "  ✗ Missing or non-executable activate script"
        ((issues++))
    else
        echo "  ✓ Activation script ready"
    fi
    
    # Check lock file if exists
    if [[ -f "$env_dir/dependencies.lock" ]]; then
        echo "  ✓ Dependencies locked"
    else
        echo "  ⚠ No dependency lock file"
    fi
    
    if [[ $issues -eq 0 ]]; then
        echo "Environment verification passed ✓"
        return 0
    else
        echo "Environment verification failed with $issues issues ✗"
        return 1
    fi
}

# Clean hermetic environment
clean_environment() {
    local env_name="$1"
    local env_dir="$HERMETIC_BASE/$env_name"
    
    if [[ ! -d "$env_dir" ]]; then
        echo "Environment not found: $env_name"
        return 1
    fi
    
    echo "Cleaning hermetic environment: $env_name"
    
    # Clean temporary files
    rm -rf "$env_dir/tmp"/*
    
    # Clean cache if requested
    if [[ "${2:-}" == "--cache" ]]; then
        rm -rf "$HERMETIC_CACHE/$env_name"/*
        echo "Cache cleared"
    fi
    
    echo "Environment cleaned"
}

# List all hermetic environments
list_environments() {
    echo "Hermetic environments:"
    
    if [[ ! -d "$HERMETIC_BASE" ]]; then
        echo "  No environments found"
        return 0
    fi
    
    for env_dir in "$HERMETIC_BASE"/*; do
        if [[ -d "$env_dir" ]]; then
            local env_name=$(basename "$env_dir")
            local created="unknown"
            
            if [[ -f "$env_dir/env.conf" ]]; then
                created=$(grep HERMETIC_ENV_CREATED "$env_dir/env.conf" | cut -d= -f2)
            fi
            
            echo "  - $env_name (created: $created)"
        fi
    done
}

# Main command handler
main() {
    local action="${1:-help}"
    shift || true
    
    case "$action" in
        create)
            create_hermetic_env "$@"
            ;;
        install)
            install_dependency "$@"
            ;;
        lock)
            lock_dependencies "$@"
            ;;
        verify)
            verify_environment "$@"
            ;;
        clean)
            clean_environment "$@"
            ;;
        list)
            list_environments
            ;;
        help)
            cat <<EOF
Hermetic Environment Manager

Usage: $0 <command> [options]

Commands:
  create <name>                 Create new hermetic environment
  install <env> <type> <spec>   Install dependency (npm/pip/binary)
  lock <env>                    Lock current dependencies
  verify <env>                  Verify environment integrity
  clean <env> [--cache]         Clean environment
  list                          List all environments
  help                          Show this help

Examples:
  $0 create myproject
  $0 install myproject npm express
  $0 install myproject pip requests
  $0 lock myproject
  $0 verify myproject
EOF
            ;;
        *)
            echo "Unknown command: $action"
            echo "Run '$0 help' for usage"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"