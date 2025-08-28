#!/usr/bin/env bash
#
# Test helper functions for BATS tests
# Provides common utilities and setup for test suite

# Test constants
readonly TEST_TEMP_DIR="${BATS_TMPDIR:-/tmp}/terminal-automation-test-$$"
readonly TEST_CONFIG_DIR="$TEST_TEMP_DIR/config"
readonly TEST_DATA_DIR="$TEST_TEMP_DIR/data"

# Setup test environment
test_setup_common() {
    # Create test directories
    mkdir -p "$TEST_TEMP_DIR" "$TEST_CONFIG_DIR" "$TEST_DATA_DIR"
    
    # Set test environment variables
    export FRAMEWORK_CONFIG_DIR="$TEST_CONFIG_DIR"
    export FRAMEWORK_DATA_DIR="$TEST_DATA_DIR"
    export FRAMEWORK_CACHE_DIR="$TEST_TEMP_DIR/cache"
    export LOG_LEVEL="ERROR"  # Reduce noise in tests
    export LOG_FORMAT="simple"
    
    # Ensure clean environment
    unset FRAMEWORK_INITIALIZED
    unset CONFIG_LOADED
}

# Cleanup test environment  
test_teardown_common() {
    # Remove test directories
    if [[ -n "$TEST_TEMP_DIR" ]] && [[ -d "$TEST_TEMP_DIR" ]]; then
        rm -rf "$TEST_TEMP_DIR"
    fi
}

# Create a test configuration file
create_test_config() {
    local config_content="$1"
    local config_file="$TEST_CONFIG_DIR/config"
    
    echo "$config_content" > "$config_file"
    echo "$config_file"
}

# Create a test script
create_test_script() {
    local script_name="$1"
    local script_content="$2"
    local script_path="$TEST_TEMP_DIR/$script_name"
    
    echo "$script_content" > "$script_path"
    chmod +x "$script_path"
    echo "$script_path"
}

# Assert output contains string
assert_output_contains() {
    local expected="$1"
    [[ "$output" =~ $expected ]]
}

# Assert output matches exactly
assert_output_equals() {
    local expected="$1"
    [[ "$output" == "$expected" ]]
}

# Assert file exists
assert_file_exists() {
    local file="$1"
    [[ -f "$file" ]]
}

# Assert directory exists
assert_dir_exists() {
    local dir="$1"
    [[ -d "$dir" ]]
}

# Assert command succeeds
assert_success() {
    [[ "$status" -eq 0 ]]
}

# Assert command fails
assert_failure() {
    [[ "$status" -ne 0 ]]
}

# Mock external command
mock_command() {
    local command="$1"
    local mock_output="$2"
    local mock_exit_code="${3:-0}"
    
    local mock_script="$TEST_TEMP_DIR/mock_$command"
    
    cat > "$mock_script" << EOF
#!/bin/bash
echo "$mock_output"
exit $mock_exit_code
EOF
    
    chmod +x "$mock_script"
    export PATH="$TEST_TEMP_DIR:$PATH"
}

# Restore original command after mocking
restore_command() {
    local command="$1"
    local mock_script="$TEST_TEMP_DIR/mock_$command"
    
    if [[ -f "$mock_script" ]]; then
        rm "$mock_script"
    fi
}

# Skip test if command not available
skip_if_missing() {
    local command="$1"
    if ! command -v "$command" >/dev/null 2>&1; then
        skip "$command not available"
    fi
}

# Skip test on specific OS
skip_if_not_macos() {
    if [[ "$(uname -s)" != "Darwin" ]]; then
        skip "Test requires macOS"
    fi
}

skip_if_not_linux() {
    if [[ "$(uname -s)" != "Linux" ]]; then
        skip "Test requires Linux"
    fi
}

# Helper to run command with timeout
run_with_timeout() {
    local timeout="$1"
    shift
    
    if command -v timeout >/dev/null 2>&1; then
        run timeout "$timeout" "$@"
    else
        # Fallback for systems without timeout
        run "$@"
    fi
}

# Generate test data
generate_test_email() {
    echo "test-$(date +%s)@example.com"
}

generate_test_uuid() {
    if command -v uuidgen >/dev/null 2>&1; then
        uuidgen | tr '[:upper:]' '[:lower:]'
    else
        # Fallback UUID generation
        echo "$(openssl rand -hex 4)-$(openssl rand -hex 2)-$(openssl rand -hex 2)-$(openssl rand -hex 2)-$(openssl rand -hex 6)"
    fi
}

generate_test_semver() {
    echo "1.0.0"
}

# Log test information
test_log() {
    echo "# $*" >&3
}

# Export functions for use in tests
export -f test_setup_common test_teardown_common
export -f create_test_config create_test_script
export -f assert_output_contains assert_output_equals
export -f assert_file_exists assert_dir_exists
export -f assert_success assert_failure
export -f mock_command restore_command
export -f skip_if_missing skip_if_not_macos skip_if_not_linux
export -f run_with_timeout
export -f generate_test_email generate_test_uuid generate_test_semver
export -f test_log