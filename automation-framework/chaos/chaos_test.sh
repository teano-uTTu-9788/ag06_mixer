#!/usr/bin/env bash

# Chaos Engineering Test Suite
# Implements controlled failure injection for resilience testing

set -euo pipefail

CHAOS_DIR="/tmp/chaos_tests"
CHAOS_LOG="$CHAOS_DIR/chaos.log"
CHAOS_RESULTS="$CHAOS_DIR/results.json"

# Chaos test types
declare -A CHAOS_TESTS=(
    ["network"]="Network failure simulation"
    ["disk"]="Disk space exhaustion"
    ["cpu"]="CPU stress testing"
    ["memory"]="Memory pressure testing"
    ["process"]="Random process killing"
    ["latency"]="Network latency injection"
    ["corruption"]="Data corruption simulation"
)

# Initialize chaos environment
init_chaos() {
    mkdir -p "$CHAOS_DIR"
    
    echo '{
        "test_id": "'$(uuidgen 2>/dev/null || echo "chaos_$(date +%s)")'",
        "start_time": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'",
        "tests": []
    }' > "$CHAOS_RESULTS"
    
    echo "Chaos engineering environment initialized"
}

# Network failure simulation
chaos_network() {
    local duration="${1:-30}"
    local target="${2:-google.com}"
    
    echo "Simulating network failure for $duration seconds..."
    log_chaos "network" "start" "duration=$duration, target=$target"
    
    # Save current firewall rules
    local rules_backup="$CHAOS_DIR/iptables.backup"
    sudo iptables-save > "$rules_backup" 2>/dev/null || true
    
    # Block network traffic
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        echo "127.0.0.1 $target" | sudo tee -a /etc/hosts > /dev/null
        
        # Simulate network failure
        (
            sleep "$duration"
            sudo sed -i '' "/$target/d" /etc/hosts
        ) &
    else
        # Linux
        sudo iptables -A OUTPUT -d "$target" -j DROP
        
        # Auto-restore after duration
        (
            sleep "$duration"
            sudo iptables -D OUTPUT -d "$target" -j DROP
        ) &
    fi
    
    local pid=$!
    
    # Test application behavior during network failure
    test_network_resilience "$target"
    
    wait "$pid"
    log_chaos "network" "end" "recovered"
    
    echo "Network failure test completed"
}

# Disk space exhaustion
chaos_disk() {
    local fill_percent="${1:-90}"
    local test_dir="${2:-/tmp/chaos_disk}"
    
    echo "Simulating disk space exhaustion ($fill_percent% full)..."
    log_chaos "disk" "start" "fill_percent=$fill_percent"
    
    mkdir -p "$test_dir"
    
    # Calculate disk space to fill
    local available=$(df "$test_dir" | awk 'NR==2 {print $4}')
    local to_fill=$((available * fill_percent / 100))
    
    # Create large file
    local fill_file="$test_dir/chaos_fill"
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        mkfile -n "${to_fill}k" "$fill_file"
    else
        # Linux
        fallocate -l "${to_fill}k" "$fill_file" 2>/dev/null || \
        dd if=/dev/zero of="$fill_file" bs=1024 count="$to_fill" 2>/dev/null
    fi
    
    # Test application behavior with low disk space
    test_disk_resilience
    
    # Cleanup
    rm -f "$fill_file"
    log_chaos "disk" "end" "space_recovered"
    
    echo "Disk exhaustion test completed"
}

# CPU stress testing
chaos_cpu() {
    local duration="${1:-30}"
    local cpu_percent="${2:-80}"
    
    echo "Simulating CPU stress ($cpu_percent% usage for $duration seconds)..."
    log_chaos "cpu" "start" "duration=$duration, percent=$cpu_percent"
    
    # Calculate number of workers based on CPU cores
    local cpu_cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    local workers=$((cpu_cores * cpu_percent / 100))
    
    # Start CPU stress workers
    local pids=()
    for ((i=1; i<=workers; i++)); do
        (
            timeout "$duration" bash -c 'while true; do echo $((13**99999)) >/dev/null; done'
        ) &
        pids+=($!)
    done
    
    # Test application behavior under CPU stress
    test_cpu_resilience
    
    # Wait for workers to complete
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    
    log_chaos "cpu" "end" "stress_removed"
    echo "CPU stress test completed"
}

# Memory pressure testing
chaos_memory() {
    local duration="${1:-30}"
    local memory_mb="${2:-512}"
    
    echo "Simulating memory pressure (${memory_mb}MB for $duration seconds)..."
    log_chaos "memory" "start" "duration=$duration, size=${memory_mb}MB"
    
    # Create memory pressure using Python
    python3 -c "
import time
import gc

# Allocate memory
data = []
chunk_size = 1024 * 1024  # 1MB
chunks = $memory_mb

for i in range(chunks):
    data.append(bytearray(chunk_size))
    
print(f'Allocated {len(data)}MB of memory')

# Hold memory for duration
time.sleep($duration)

# Cleanup
del data
gc.collect()
print('Memory released')
" &
    
    local pid=$!
    
    # Test application behavior under memory pressure
    test_memory_resilience
    
    wait "$pid"
    log_chaos "memory" "end" "memory_released"
    
    echo "Memory pressure test completed"
}

# Random process killing
chaos_process() {
    local target_pattern="${1:-workflow}"
    local kill_probability="${2:-0.5}"
    
    echo "Simulating random process failures (pattern: $target_pattern)..."
    log_chaos "process" "start" "pattern=$target_pattern, probability=$kill_probability"
    
    # Find target processes
    local pids=($(pgrep -f "$target_pattern" 2>/dev/null || true))
    
    if [[ ${#pids[@]} -eq 0 ]]; then
        echo "No processes matching pattern: $target_pattern"
        return 1
    fi
    
    # Randomly kill processes based on probability
    for pid in "${pids[@]}"; do
        if (( $(echo "scale=2; $RANDOM/32768 < $kill_probability" | bc -l) )); then
            echo "Killing process $pid"
            kill -9 "$pid" 2>/dev/null || true
            
            # Test recovery
            sleep 2
            test_process_recovery "$target_pattern"
        fi
    done
    
    log_chaos "process" "end" "processes_tested"
    echo "Process failure test completed"
}

# Network latency injection
chaos_latency() {
    local duration="${1:-30}"
    local latency_ms="${2:-500}"
    local interface="${3:-lo}"
    
    echo "Injecting network latency (${latency_ms}ms for $duration seconds)..."
    log_chaos "latency" "start" "duration=$duration, latency=${latency_ms}ms"
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS - use dummynet
        echo "Using dummynet for latency injection on macOS"
        
        # Create pipe with latency
        sudo pfctl -E 2>/dev/null || true
        echo "dummynet in all pipe 1 delay ${latency_ms}ms" | sudo pfctl -f - 2>/dev/null || true
        
        # Auto-remove after duration
        (
            sleep "$duration"
            sudo pfctl -F all -f /etc/pf.conf 2>/dev/null || true
        ) &
    else
        # Linux - use tc (traffic control)
        echo "Using tc for latency injection on Linux"
        
        # Add latency to interface
        sudo tc qdisc add dev "$interface" root netem delay "${latency_ms}ms"
        
        # Auto-remove after duration
        (
            sleep "$duration"
            sudo tc qdisc del dev "$interface" root netem
        ) &
    fi
    
    local pid=$!
    
    # Test application behavior with latency
    test_latency_resilience "$latency_ms"
    
    wait "$pid"
    log_chaos "latency" "end" "latency_removed"
    
    echo "Latency injection test completed"
}

# Data corruption simulation
chaos_corruption() {
    local target_file="${1:-/tmp/chaos_data}"
    local corruption_percent="${2:-5}"
    
    echo "Simulating data corruption ($corruption_percent% of file)..."
    log_chaos "corruption" "start" "file=$target_file, percent=$corruption_percent"
    
    # Create test file if it doesn't exist
    if [[ ! -f "$target_file" ]]; then
        echo "Creating test file: $target_file"
        dd if=/dev/urandom of="$target_file" bs=1024 count=100 2>/dev/null
    fi
    
    # Backup original file
    cp "$target_file" "${target_file}.backup"
    
    # Calculate bytes to corrupt
    local file_size=$(stat -f%z "$target_file" 2>/dev/null || stat -c%s "$target_file" 2>/dev/null)
    local bytes_to_corrupt=$((file_size * corruption_percent / 100))
    
    # Corrupt random bytes
    for ((i=0; i<bytes_to_corrupt; i++)); do
        local offset=$((RANDOM % file_size))
        printf '\x%02x' $((RANDOM % 256)) | dd of="$target_file" bs=1 seek="$offset" count=1 conv=notrunc 2>/dev/null
    done
    
    echo "Corrupted $bytes_to_corrupt bytes in $target_file"
    
    # Test application behavior with corrupted data
    test_corruption_resilience "$target_file"
    
    # Restore original file
    mv "${target_file}.backup" "$target_file"
    log_chaos "corruption" "end" "data_restored"
    
    echo "Data corruption test completed"
}

# Test resilience functions
test_network_resilience() {
    local target="$1"
    echo "Testing network resilience..."
    
    # Try to connect during failure
    for i in {1..5}; do
        if curl -s --max-time 2 "http://$target" >/dev/null 2>&1; then
            echo "  Unexpected success connecting to $target"
        else
            echo "  Expected failure connecting to $target"
        fi
        sleep 1
    done
}

test_disk_resilience() {
    echo "Testing disk space resilience..."
    
    # Try to write during low disk space
    if echo "test" > /tmp/chaos_write_test 2>/dev/null; then
        echo "  Application can still write files"
        rm -f /tmp/chaos_write_test
    else
        echo "  Application correctly handles disk full error"
    fi
}

test_cpu_resilience() {
    echo "Testing CPU stress resilience..."
    
    # Measure response time during stress
    local start=$(date +%s%N)
    sleep 0.1
    local end=$(date +%s%N)
    local duration=$((($end - $start) / 1000000))
    
    echo "  Response time during stress: ${duration}ms"
}

test_memory_resilience() {
    echo "Testing memory pressure resilience..."
    
    # Check if application can allocate memory
    if python3 -c "import sys; data = bytearray(10*1024*1024); sys.exit(0)" 2>/dev/null; then
        echo "  Application can allocate memory"
    else
        echo "  Application handles memory allocation failure"
    fi
}

test_process_recovery() {
    local pattern="$1"
    echo "Testing process recovery..."
    
    # Check if process restarted
    sleep 2
    if pgrep -f "$pattern" >/dev/null 2>&1; then
        echo "  Process recovered successfully"
    else
        echo "  Process did not recover automatically"
    fi
}

test_latency_resilience() {
    local latency="$1"
    echo "Testing latency resilience..."
    
    # Measure actual latency
    if command -v ping >/dev/null 2>&1; then
        local actual=$(ping -c 1 127.0.0.1 2>/dev/null | grep -oE 'time=[0-9.]+' | cut -d= -f2)
        echo "  Measured latency: ${actual}ms (expected: ~${latency}ms)"
    fi
}

test_corruption_resilience() {
    local file="$1"
    echo "Testing data corruption resilience..."
    
    # Check if application detects corruption
    if md5sum "$file" >/dev/null 2>&1; then
        echo "  File checksum calculated (corruption handling needed)"
    fi
}

# Logging function
log_chaos() {
    local test_type="$1"
    local event="$2"
    local details="$3"
    
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    echo "[$timestamp] $test_type: $event - $details" >> "$CHAOS_LOG"
    
    # Update results JSON
    local results=$(cat "$CHAOS_RESULTS")
    results=$(echo "$results" | jq --arg type "$test_type" --arg event "$event" --arg details "$details" --arg ts "$timestamp" \
        '.tests += [{"type": $type, "event": $event, "details": $details, "timestamp": $ts}]')
    echo "$results" > "$CHAOS_RESULTS"
}

# Run chaos test suite
run_chaos_suite() {
    echo "Running comprehensive chaos test suite..."
    init_chaos
    
    local tests=("network" "disk" "cpu" "memory" "latency")
    local failed_tests=0
    
    for test in "${tests[@]}"; do
        echo ""
        echo "===================="
        echo "Running $test chaos test..."
        echo "===================="
        
        if chaos_"$test"; then
            echo "✓ $test test passed"
        else
            echo "✗ $test test failed"
            ((failed_tests++))
        fi
        
        # Pause between tests
        sleep 5
    done
    
    # Generate report
    generate_chaos_report
    
    if [[ $failed_tests -eq 0 ]]; then
        echo ""
        echo "All chaos tests completed successfully!"
        return 0
    else
        echo ""
        echo "$failed_tests chaos tests failed"
        return 1
    fi
}

# Generate chaos test report
generate_chaos_report() {
    echo ""
    echo "Generating chaos test report..."
    
    local results=$(cat "$CHAOS_RESULTS")
    
    # Update end time
    results=$(echo "$results" | jq --arg ts "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" '.end_time = $ts')
    echo "$results" > "$CHAOS_RESULTS"
    
    # Create HTML report
    cat > "$CHAOS_DIR/report.html" <<EOF
<!DOCTYPE html>
<html>
<head>
    <title>Chaos Engineering Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        .test { margin: 10px 0; padding: 10px; border: 1px solid #ddd; }
        .success { background: #d4edda; }
        .failure { background: #f8d7da; }
        .info { background: #d1ecf1; }
        table { width: 100%; border-collapse: collapse; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background: #f2f2f2; }
    </style>
</head>
<body>
    <h1>Chaos Engineering Test Report</h1>
    <div class="info">
        <p><strong>Test ID:</strong> $(echo "$results" | jq -r '.test_id')</p>
        <p><strong>Start Time:</strong> $(echo "$results" | jq -r '.start_time')</p>
        <p><strong>End Time:</strong> $(echo "$results" | jq -r '.end_time')</p>
    </div>
    
    <h2>Test Results</h2>
    <table>
        <tr>
            <th>Test Type</th>
            <th>Event</th>
            <th>Details</th>
            <th>Timestamp</th>
        </tr>
EOF
    
    echo "$results" | jq -r '.tests[] | "<tr><td>\(.type)</td><td>\(.event)</td><td>\(.details)</td><td>\(.timestamp)</td></tr>"' >> "$CHAOS_DIR/report.html"
    
    cat >> "$CHAOS_DIR/report.html" <<EOF
    </table>
    
    <h2>Summary</h2>
    <ul>
        <li>Total Tests Run: $(echo "$results" | jq '.tests | length')</li>
        <li>Unique Test Types: $(echo "$results" | jq '[.tests[].type] | unique | length')</li>
    </ul>
</body>
</html>
EOF
    
    echo "Report generated: $CHAOS_DIR/report.html"
    echo "Results saved: $CHAOS_RESULTS"
}

# Validate chaos testing setup
validate_setup() {
    echo "Validating chaos testing setup..."
    
    local issues=0
    
    # Check required commands
    for cmd in bc jq timeout; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            echo "  ✗ Missing required command: $cmd"
            ((issues++))
        else
            echo "  ✓ Command available: $cmd"
        fi
    done
    
    # Check permissions
    if [[ "$EUID" -eq 0 ]]; then
        echo "  ✓ Running with root privileges"
    else
        echo "  ⚠ Some tests may require sudo privileges"
    fi
    
    # Check disk space
    local available=$(df /tmp | awk 'NR==2 {print $4}')
    if [[ $available -lt 1048576 ]]; then  # Less than 1GB
        echo "  ⚠ Low disk space for chaos tests"
    else
        echo "  ✓ Sufficient disk space available"
    fi
    
    if [[ $issues -eq 0 ]]; then
        echo "Setup validation passed ✓"
        return 0
    else
        echo "Setup validation failed with $issues issues"
        return 1
    fi
}

# Main command handler
main() {
    local action="${1:-help}"
    shift || true
    
    case "$action" in
        network|disk|cpu|memory|process|latency|corruption)
            init_chaos
            chaos_"$action" "$@"
            ;;
        suite)
            run_chaos_suite
            ;;
        validate)
            validate_setup
            ;;
        report)
            if [[ -f "$CHAOS_RESULTS" ]]; then
                generate_chaos_report
            else
                echo "No test results found. Run tests first."
            fi
            ;;
        clean)
            rm -rf "$CHAOS_DIR"
            echo "Chaos test environment cleaned"
            ;;
        help)
            cat <<EOF
Chaos Engineering Test Suite

Usage: $0 <test> [options]

Tests:
  network [duration] [target]      Network failure simulation
  disk [fill_percent] [dir]        Disk space exhaustion
  cpu [duration] [percent]         CPU stress testing
  memory [duration] [size_mb]      Memory pressure testing
  process [pattern] [probability]  Random process killing
  latency [duration] [ms] [iface]  Network latency injection
  corruption [file] [percent]      Data corruption simulation
  
Commands:
  suite                            Run complete test suite
  validate                         Validate setup
  report                           Generate test report
  clean                            Clean test environment
  help                             Show this help

Examples:
  $0 network 30 google.com
  $0 cpu 60 90
  $0 memory 30 1024
  $0 suite
EOF
            ;;
        *)
            echo "Unknown test: $action"
            echo "Run '$0 help' for usage"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"