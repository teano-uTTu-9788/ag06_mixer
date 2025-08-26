#!/usr/bin/env bash
# Testing Library - Test Automation Pattern
# Provides testing utilities for shell scripts and applications

# Require core library
deps::require "core"

# ============================================================================
# Test Framework Setup (Google Test Pattern)
# ============================================================================

# Test counters
TEST_TOTAL=0
TEST_PASSED=0
TEST_FAILED=0
TEST_SKIPPED=0

# Test output settings
TEST_VERBOSE="${TEST_VERBOSE:-false}"
TEST_STOP_ON_FAIL="${TEST_STOP_ON_FAIL:-false}"

# ============================================================================
# Assertion Functions (JUnit Pattern)
# ============================================================================

assert::equals() {
    local expected="$1"
    local actual="$2"
    local message="${3:-Assertion failed}"
    
    if [[ "$expected" == "$actual" ]]; then
        test::pass "$message"
        return 0
    else
        test::fail "$message: expected '$expected', got '$actual'"
        return 1
    fi
}

assert::not_equals() {
    local unexpected="$1"
    local actual="$2"
    local message="${3:-Assertion failed}"
    
    if [[ "$unexpected" != "$actual" ]]; then
        test::pass "$message"
        return 0
    else
        test::fail "$message: value should not be '$unexpected'"
        return 1
    fi
}

assert::true() {
    local condition="$1"
    local message="${2:-Assertion failed}"
    
    if eval "$condition"; then
        test::pass "$message"
        return 0
    else
        test::fail "$message: condition '$condition' is false"
        return 1
    fi
}

assert::false() {
    local condition="$1"
    local message="${2:-Assertion failed}"
    
    if ! eval "$condition"; then
        test::pass "$message"
        return 0
    else
        test::fail "$message: condition '$condition' is true"
        return 1
    fi
}

assert::contains() {
    local haystack="$1"
    local needle="$2"
    local message="${3:-Assertion failed}"
    
    if [[ "$haystack" == *"$needle"* ]]; then
        test::pass "$message"
        return 0
    else
        test::fail "$message: '$haystack' does not contain '$needle'"
        return 1
    fi
}

assert::matches() {
    local string="$1"
    local pattern="$2"
    local message="${3:-Assertion failed}"
    
    if [[ "$string" =~ $pattern ]]; then
        test::pass "$message"
        return 0
    else
        test::fail "$message: '$string' does not match pattern '$pattern'"
        return 1
    fi
}

assert::file_exists() {
    local file="$1"
    local message="${2:-File should exist}"
    
    if [[ -f "$file" ]]; then
        test::pass "$message: $file"
        return 0
    else
        test::fail "$message: $file not found"
        return 1
    fi
}

assert::directory_exists() {
    local dir="$1"
    local message="${2:-Directory should exist}"
    
    if [[ -d "$dir" ]]; then
        test::pass "$message: $dir"
        return 0
    else
        test::fail "$message: $dir not found"
        return 1
    fi
}

assert::command_succeeds() {
    local command="$1"
    local message="${2:-Command should succeed}"
    
    if eval "$command" &>/dev/null; then
        test::pass "$message"
        return 0
    else
        test::fail "$message: command '$command' failed"
        return 1
    fi
}

assert::command_fails() {
    local command="$1"
    local message="${2:-Command should fail}"
    
    if ! eval "$command" &>/dev/null; then
        test::pass "$message"
        return 0
    else
        test::fail "$message: command '$command' succeeded unexpectedly"
        return 1
    fi
}

# ============================================================================
# Test Execution (Meta Test Runner Pattern)
# ============================================================================

test::run() {
    local test_name="$1"
    local test_function="$2"
    
    ((TEST_TOTAL++))
    
    if [[ "$TEST_VERBOSE" == "true" ]]; then
        echo -n "Running $test_name... "
    fi
    
    # Create test environment
    local test_dir="/tmp/test_$$_$RANDOM"
    mkdir -p "$test_dir"
    local original_dir="$(pwd)"
    cd "$test_dir"
    
    # Run test in subshell to isolate
    (
        set +e  # Don't exit on error in tests
        $test_function
    )
    local result=$?
    
    # Cleanup
    cd "$original_dir"
    rm -rf "$test_dir"
    
    # Report result
    if [[ $result -eq 0 ]]; then
        ((TEST_PASSED++))
        if [[ "$TEST_VERBOSE" == "true" ]]; then
            echo -e "${GREEN}PASSED${NC}"
        else
            echo -n "."
        fi
    else
        ((TEST_FAILED++))
        if [[ "$TEST_VERBOSE" == "true" ]]; then
            echo -e "${RED}FAILED${NC}"
        else
            echo -n "F"
        fi
        
        if [[ "$TEST_STOP_ON_FAIL" == "true" ]]; then
            test::report
            exit 1
        fi
    fi
    
    return $result
}

test::suite() {
    local suite_name="$1"
    shift
    local test_functions=("$@")
    
    echo "Running test suite: $suite_name"
    echo "=================================="
    
    for test_func in "${test_functions[@]}"; do
        local test_name="${test_func#test_}"
        test_name="${test_name//_/ }"
        test::run "$test_name" "$test_func"
    done
    
    echo ""
    test::report
}

test::pass() {
    local message="${1:-Test passed}"
    [[ "$TEST_VERBOSE" == "true" ]] && log::debug "✓ $message"
    return 0
}

test::fail() {
    local message="${1:-Test failed}"
    log::error "✗ $message"
    return 1
}

test::skip() {
    local message="${1:-Test skipped}"
    ((TEST_SKIPPED++))
    [[ "$TEST_VERBOSE" == "true" ]] && log::warn "⊘ $message"
    return 0
}

test::report() {
    echo ""
    echo "Test Results:"
    echo "============="
    echo "Total:   $TEST_TOTAL"
    echo -e "Passed:  ${GREEN}$TEST_PASSED${NC}"
    echo -e "Failed:  ${RED}$TEST_FAILED${NC}"
    echo -e "Skipped: ${YELLOW}$TEST_SKIPPED${NC}"
    
    local success_rate=0
    if [[ $TEST_TOTAL -gt 0 ]]; then
        success_rate=$((TEST_PASSED * 100 / TEST_TOTAL))
    fi
    
    echo ""
    echo "Success rate: ${success_rate}%"
    
    if [[ $TEST_FAILED -eq 0 ]]; then
        echo -e "${GREEN}All tests passed!${NC}"
        return 0
    else
        echo -e "${RED}Some tests failed!${NC}"
        return 1
    fi
}

# ============================================================================
# BATS Integration (Bash Automated Testing System)
# ============================================================================

testing::install_bats() {
    log::info "Installing BATS testing framework..."
    
    if command -v bats &>/dev/null; then
        log::info "BATS is already installed"
        return 0
    fi
    
    if platform::is_macos && brew::is_installed; then
        brew install bats-core
    else
        # Install from source
        local install_dir="${HOME}/.local"
        git clone https://github.com/bats-core/bats-core.git /tmp/bats-core
        cd /tmp/bats-core
        ./install.sh "$install_dir"
        cd -
        rm -rf /tmp/bats-core
        
        # Add to PATH
        export PATH="${install_dir}/bin:$PATH"
    fi
    
    log::info "BATS installed successfully"
}

testing::run_bats() {
    local test_dir="${1:-tests}"
    local options="${2:-}"
    
    if ! command -v bats &>/dev/null; then
        log::error "BATS is not installed. Run 'testing::install_bats' first."
        return 1
    fi
    
    log::info "Running BATS tests in $test_dir..."
    bats $options "$test_dir"/*.bats
}

# ============================================================================
# Mock Functions (Dependency Injection Pattern)
# ============================================================================

mock::create() {
    local function_name="$1"
    local mock_behavior="$2"
    
    # Save original function if it exists
    if declare -f "$function_name" &>/dev/null; then
        eval "original_${function_name}() { $(declare -f "$function_name" | tail -n +2) }"
    fi
    
    # Create mock function
    eval "$function_name() { $mock_behavior; }"
}

mock::restore() {
    local function_name="$1"
    
    # Restore original function if it was saved
    if declare -f "original_${function_name}" &>/dev/null; then
        eval "$function_name() { $(declare -f "original_${function_name}" | tail -n +2) }"
        unset -f "original_${function_name}"
    else
        unset -f "$function_name"
    fi
}

mock::call_count() {
    local function_name="$1"
    echo "${MOCK_CALL_COUNT[$function_name]:-0}"
}

# ============================================================================
# Coverage Analysis (Code Coverage Pattern)
# ============================================================================

coverage::start() {
    # Enable tracing
    set -x
    export PS4='+ $(date "+%s.%N") ${BASH_SOURCE}:${LINENO}: ${FUNCNAME[0]:+${FUNCNAME[0]}(): }'
    exec 2> "${COVERAGE_FILE:-/tmp/coverage_$$.log}"
}

coverage::stop() {
    set +x
    exec 2>&1
}

coverage::analyze() {
    local coverage_file="${1:-/tmp/coverage_$$.log}"
    local source_dir="${2:-.}"
    
    if [[ ! -f "$coverage_file" ]]; then
        log::error "Coverage file not found: $coverage_file"
        return 1
    fi
    
    log::info "Analyzing coverage..."
    
    # Extract covered lines
    local covered_files=$(grep -o "${source_dir}/[^:]*" "$coverage_file" | sort -u)
    
    echo "Coverage Report:"
    echo "================"
    
    for file in $covered_files; do
        if [[ -f "$file" ]]; then
            local total_lines=$(wc -l < "$file")
            local covered_lines=$(grep -c "$file" "$coverage_file")
            local coverage=$((covered_lines * 100 / total_lines))
            
            printf "%-50s %3d%% (%d/%d lines)\n" "$file" "$coverage" "$covered_lines" "$total_lines"
        fi
    done
}

# ============================================================================
# Benchmark Functions (Performance Testing Pattern)
# ============================================================================

benchmark::time() {
    local name="$1"
    local command="$2"
    local iterations="${3:-10}"
    
    log::info "Benchmarking '$name' ($iterations iterations)..."
    
    local total_time=0
    local min_time=999999
    local max_time=0
    
    for ((i=1; i<=iterations; i++)); do
        local start=$(date +%s%N)
        eval "$command" &>/dev/null
        local end=$(date +%s%N)
        
        local elapsed=$((end - start))
        total_time=$((total_time + elapsed))
        
        [[ $elapsed -lt $min_time ]] && min_time=$elapsed
        [[ $elapsed -gt $max_time ]] && max_time=$elapsed
    done
    
    local avg_time=$((total_time / iterations))
    
    # Convert nanoseconds to milliseconds
    local avg_ms=$((avg_time / 1000000))
    local min_ms=$((min_time / 1000000))
    local max_ms=$((max_time / 1000000))
    
    echo "Benchmark Results for '$name':"
    echo "  Average: ${avg_ms}ms"
    echo "  Min:     ${min_ms}ms"
    echo "  Max:     ${max_ms}ms"
}

benchmark::compare() {
    local name1="$1"
    local command1="$2"
    local name2="$3"
    local command2="$4"
    local iterations="${5:-10}"
    
    log::info "Comparing performance..."
    
    # Run benchmarks
    local output1=$(benchmark::time "$name1" "$command1" "$iterations")
    local output2=$(benchmark::time "$name2" "$command2" "$iterations")
    
    echo "$output1"
    echo ""
    echo "$output2"
    
    # Extract average times for comparison
    local avg1=$(echo "$output1" | grep "Average:" | awk '{print $2}' | tr -d 'ms')
    local avg2=$(echo "$output2" | grep "Average:" | awk '{print $2}' | tr -d 'ms')
    
    echo ""
    if [[ $avg1 -lt $avg2 ]]; then
        local speedup=$((avg2 * 100 / avg1 - 100))
        echo -e "${GREEN}$name1 is ${speedup}% faster${NC}"
    elif [[ $avg2 -lt $avg1 ]]; then
        local speedup=$((avg1 * 100 / avg2 - 100))
        echo -e "${GREEN}$name2 is ${speedup}% faster${NC}"
    else
        echo "Both perform equally"
    fi
}

# ============================================================================
# Integration Test Helpers (E2E Testing Pattern)
# ============================================================================

integration::setup() {
    local test_name="${1:-integration_test}"
    
    # Create isolated test environment
    export TEST_ENV_DIR="/tmp/${test_name}_$$"
    mkdir -p "$TEST_ENV_DIR"
    
    # Save original environment
    export ORIGINAL_PATH="$PATH"
    export ORIGINAL_HOME="$HOME"
    
    # Set test environment
    export HOME="$TEST_ENV_DIR"
    export PATH="$TEST_ENV_DIR/bin:$PATH"
    
    log::debug "Integration test environment created: $TEST_ENV_DIR"
}

integration::teardown() {
    # Restore original environment
    export PATH="$ORIGINAL_PATH"
    export HOME="$ORIGINAL_HOME"
    
    # Cleanup test environment
    if [[ -n "$TEST_ENV_DIR" ]] && [[ -d "$TEST_ENV_DIR" ]]; then
        rm -rf "$TEST_ENV_DIR"
    fi
    
    log::debug "Integration test environment cleaned up"
}

integration::wait_for_port() {
    local port="$1"
    local timeout="${2:-30}"
    local host="${3:-localhost}"
    
    log::info "Waiting for port $port to be available..."
    
    local elapsed=0
    while [[ $elapsed -lt $timeout ]]; do
        if nc -z "$host" "$port" 2>/dev/null; then
            log::info "Port $port is available"
            return 0
        fi
        sleep 1
        ((elapsed++))
    done
    
    log::error "Port $port did not become available within ${timeout}s"
    return 1
}

integration::wait_for_url() {
    local url="$1"
    local timeout="${2:-30}"
    local expected_status="${3:-200}"
    
    log::info "Waiting for URL $url to respond..."
    
    local elapsed=0
    while [[ $elapsed -lt $timeout ]]; do
        local status=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null)
        if [[ "$status" == "$expected_status" ]]; then
            log::info "URL is responding with status $status"
            return 0
        fi
        sleep 1
        ((elapsed++))
    done
    
    log::error "URL did not respond with status $expected_status within ${timeout}s"
    return 1
}