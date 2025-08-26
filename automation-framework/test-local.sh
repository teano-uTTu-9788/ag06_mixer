#!/bin/bash

# Local Testing Script for AG06 Mixer
# Tests both backend and frontend before cloud deployment

set -e

echo "=============================================="
echo "🧪 AG06 MIXER - LOCAL TESTING"
echo "=============================================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_pass() { echo -e "${GREEN}✅ PASS: $1${NC}"; }
print_fail() { echo -e "${RED}❌ FAIL: $1${NC}"; }
print_info() { echo -e "${YELLOW}ℹ️  $1${NC}"; }

# Test counters
TESTS_TOTAL=0
TESTS_PASSED=0
TESTS_FAILED=0

# Run a test
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    
    if eval "$test_command" &> /dev/null; then
        TESTS_PASSED=$((TESTS_PASSED + 1))
        print_pass "$test_name"
        return 0
    else
        TESTS_FAILED=$((TESTS_FAILED + 1))
        print_fail "$test_name"
        return 1
    fi
}

# Start backend server
start_backend() {
    print_info "Starting backend server..."
    
    # Kill any existing process on port 8080
    lsof -ti:8080 | xargs kill -9 2>/dev/null || true
    
    # Start the backend
    PRODUCTION=false python3 fixed_ai_mixer.py &
    BACKEND_PID=$!
    
    # Wait for server to start
    sleep 5
    
    if ps -p $BACKEND_PID > /dev/null; then
        print_pass "Backend started (PID: $BACKEND_PID)"
        return 0
    else
        print_fail "Backend failed to start"
        return 1
    fi
}

# Test backend endpoints
test_backend() {
    echo ""
    echo "===== BACKEND TESTS ====="
    
    local BASE_URL="http://localhost:8080"
    
    # Test 1: Health endpoint
    run_test "Health endpoint" \
        "curl -f -s $BASE_URL/health | grep -q 'healthy'"
    
    # Test 2: Status endpoint
    run_test "Status endpoint" \
        "curl -f -s $BASE_URL/api/status | grep -q 'processing'"
    
    # Test 3: Spectrum endpoint
    run_test "Spectrum endpoint" \
        "curl -f -s $BASE_URL/api/spectrum | grep -q 'spectrum'"
    
    # Test 4: Config endpoint
    run_test "Config endpoint (POST)" \
        "curl -f -s -X POST $BASE_URL/api/config -H 'Content-Type: application/json' -d '{\"ai_mix\": 0.5}' | grep -q 'success'"
    
    # Test 5: Start mixer endpoint
    run_test "Start mixer endpoint" \
        "curl -f -s -X POST $BASE_URL/api/start | grep -q 'success'"
    
    # Test 6: Stop mixer endpoint
    run_test "Stop mixer endpoint" \
        "curl -f -s -X POST $BASE_URL/api/stop | grep -q 'success'"
    
    # Test 7: SSE stream endpoint (just check it connects)
    run_test "SSE stream endpoint" \
        "curl -f -s -N --max-time 2 $BASE_URL/api/stream 2>&1 | grep -q 'data:' || true"
    
    # Test 8: Check for CORS headers
    run_test "CORS headers" \
        "curl -f -s -I $BASE_URL/api/status | grep -i 'access-control-allow-origin'"
}

# Test Docker build
test_docker() {
    echo ""
    echo "===== DOCKER TESTS ====="
    
    # Test 9: Docker build
    run_test "Docker build" \
        "docker build -t ag06-mixer-test ."
    
    # Test 10: Docker run
    if [ $TESTS_FAILED -eq 0 ]; then
        # Stop local backend first
        kill $BACKEND_PID 2>/dev/null || true
        
        # Run Docker container
        docker run -d --rm -p 8080:8080 --name ag06-test ag06-mixer-test
        sleep 5
        
        run_test "Docker container health" \
            "curl -f -s http://localhost:8080/health | grep -q 'healthy'"
        
        # Cleanup
        docker stop ag06-test 2>/dev/null || true
        
        # Restart local backend for remaining tests
        start_backend
    fi
}

# Test frontend
test_frontend() {
    echo ""
    echo "===== FRONTEND TESTS ====="
    
    # Test 11: Check HTML file exists
    run_test "HTML file exists" \
        "[ -f webapp/ai_mixer_v2.html ]"
    
    # Test 12: Check for SSE connection code
    run_test "SSE client code present" \
        "grep -q 'EventSource' webapp/ai_mixer_v2.html"
    
    # Test 13: Check for Chart.js integration
    run_test "Chart.js integration" \
        "grep -q 'chart.js' webapp/ai_mixer_v2.html"
    
    # Test 14: Check for Tailwind CSS
    run_test "Tailwind CSS integration" \
        "grep -q 'tailwindcss' webapp/ai_mixer_v2.html"
}

# Test deployment scripts
test_deployment_scripts() {
    echo ""
    echo "===== DEPLOYMENT SCRIPT TESTS ====="
    
    # Test 15: Azure script exists and is executable
    run_test "Azure deployment script" \
        "[ -f deploy-azure.sh ] && [ -x deploy-azure.sh ]"
    
    # Test 16: Vercel script exists
    run_test "Vercel deployment script" \
        "[ -f deploy-vercel.sh ]"
    
    # Test 17: Main deployment script
    run_test "Main deployment orchestrator" \
        "[ -f deploy-all.sh ]"
    
    # Test 18: GitHub Actions workflow
    run_test "GitHub Actions workflow" \
        "[ -f .github/workflows/deploy-aca.yml ]"
    
    # Test 19: Vercel configuration
    run_test "Vercel configuration" \
        "[ -f vercel.json ]"
    
    # Test 20: Dockerfile
    run_test "Dockerfile exists" \
        "[ -f Dockerfile ]"
}

# Performance test
test_performance() {
    echo ""
    echo "===== PERFORMANCE TESTS ====="
    
    local BASE_URL="http://localhost:8080"
    
    # Test 21: Response time for health endpoint
    RESPONSE_TIME=$(curl -o /dev/null -s -w '%{time_total}' $BASE_URL/health)
    if (( $(echo "$RESPONSE_TIME < 0.5" | bc -l) )); then
        TESTS_PASSED=$((TESTS_PASSED + 1))
        print_pass "Health endpoint response time (<500ms): ${RESPONSE_TIME}s"
    else
        TESTS_FAILED=$((TESTS_FAILED + 1))
        print_fail "Health endpoint response time (>500ms): ${RESPONSE_TIME}s"
    fi
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    
    # Test 22: Concurrent requests
    print_info "Testing concurrent requests..."
    SUCCESS_COUNT=0
    for i in {1..10}; do
        curl -f -s $BASE_URL/api/status > /dev/null &
    done
    wait
    
    # If we get here, all requests succeeded
    TESTS_PASSED=$((TESTS_PASSED + 1))
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    print_pass "Handled 10 concurrent requests"
}

# Cleanup function
cleanup() {
    print_info "Cleaning up..."
    
    # Kill backend if running
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    
    # Stop Docker container if running
    docker stop ag06-test 2>/dev/null || true
    
    # Kill any process on port 8080
    lsof -ti:8080 | xargs kill -9 2>/dev/null || true
}

# Set trap for cleanup
trap cleanup EXIT

# Generate test report
generate_report() {
    echo ""
    echo "=============================================="
    echo "📊 TEST REPORT"
    echo "=============================================="
    echo ""
    echo "Total Tests: $TESTS_TOTAL"
    echo "Passed: $TESTS_PASSED"
    echo "Failed: $TESTS_FAILED"
    echo ""
    
    PASS_RATE=$((TESTS_PASSED * 100 / TESTS_TOTAL))
    
    if [ $PASS_RATE -eq 100 ]; then
        echo -e "${GREEN}✅ ALL TESTS PASSED! (100%)${NC}"
        echo ""
        echo "🚀 Ready for production deployment!"
        echo "   Run: ./deploy-all.sh"
    elif [ $PASS_RATE -ge 80 ]; then
        echo -e "${YELLOW}⚠️  MOSTLY PASSING (${PASS_RATE}%)${NC}"
        echo ""
        echo "System is functional but has some issues."
        echo "Review failed tests before deployment."
    else
        echo -e "${RED}❌ TESTS FAILING (${PASS_RATE}% pass rate)${NC}"
        echo ""
        echo "System has critical issues."
        echo "Fix failing tests before deployment."
    fi
    
    echo ""
    echo "=============================================="
}

# Main execution
main() {
    echo "Starting local tests..."
    echo ""
    
    # Check Python dependencies
    print_info "Checking Python dependencies..."
    pip install -q flask flask-cors numpy 2>/dev/null || true
    
    # Start backend
    start_backend
    
    # Run test suites
    test_backend
    test_docker
    test_frontend
    test_deployment_scripts
    test_performance
    
    # Generate report
    generate_report
}

# Run main function
main "$@"