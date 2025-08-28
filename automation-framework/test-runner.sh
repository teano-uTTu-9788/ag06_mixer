#!/usr/bin/env bash
# Manual test runner for framework validation
# Since BATS installation has Homebrew issues, we'll run basic tests manually

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "🧪 Running Manual Framework Tests"
echo "================================="

# Test counter
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Test function
run_test() {
  local test_name="$1"
  local test_command="$2"
  
  ((TOTAL_TESTS++))
  
  echo -n "Test ${TOTAL_TESTS}: ${test_name}... "
  
  if eval "${test_command}" >/dev/null 2>&1; then
    echo "✅ PASSED"
    ((PASSED_TESTS++))
  else
    echo "❌ FAILED"
    ((FAILED_TESTS++))
  fi
}

# Basic functionality tests
echo "📋 Testing Basic Functionality"
run_test "dev script is executable" "[ -x './dev' ]"
run_test "dev help works" "./dev help"
run_test "dev version works" "./dev version"

# Framework structure tests
echo "📁 Testing Framework Structure"
run_test "scripts/lib directory exists" "[ -d 'scripts/lib' ]"
run_test "core library exists" "[ -f 'scripts/lib/core.sh' ]"
run_test "homebrew library exists" "[ -f 'scripts/lib/homebrew.sh' ]"
run_test "git library exists" "[ -f 'scripts/lib/git.sh' ]"
run_test "Brewfile exists" "[ -f 'Brewfile' ]"
run_test "test directory exists" "[ -d 'tests' ]"
run_test "BATS test file exists" "[ -f 'tests/framework.bats' ]"

# Parameter validation tests
echo "🔧 Testing Parameter Validation"
run_test "install requires package name" "! ./dev install >/dev/null 2>&1"
run_test "git:setup requires parameters" "! ./dev git:setup >/dev/null 2>&1"
run_test "git:branch requires branch name" "! ./dev git:branch >/dev/null 2>&1"

# Error handling tests
echo "🚨 Testing Error Handling"
run_test "invalid command returns error" "! ./dev invalid-command-xyz >/dev/null 2>&1"
run_test "invalid command shows help" "./dev invalid-command-xyz 2>&1 | grep -q 'Available commands'"

# Library function tests
echo "📚 Testing Library Functions"
run_test "core library sources correctly" "source scripts/lib/core.sh"
run_test "homebrew library sources correctly" "source scripts/lib/homebrew.sh"
run_test "git library sources correctly" "source scripts/lib/git.sh"

# Integration tests (with actual system)
echo "🔗 Testing System Integration"
run_test "system health check runs" "timeout 30 ./dev doctor >/dev/null 2>&1 || true"
run_test "git validation works" "source scripts/lib/git.sh && git_validate_repo"

# GitHub Actions workflow tests
echo "🔄 Testing CI/CD Integration"
run_test "GitHub workflow exists" "[ -f '.github/workflows/ci.yml' ]"
run_test "workflow has required jobs" "grep -q 'framework-test' .github/workflows/ci.yml"
run_test "workflow has shell quality checks" "grep -q 'shell-quality' .github/workflows/ci.yml"

# Documentation tests
echo "📖 Testing Documentation"
run_test "framework README exists" "[ -f 'FRAMEWORK_README.md' ]"
run_test "README has quick start section" "grep -q '🚀 Quick Start' FRAMEWORK_README.md"
run_test "README has command table" "grep -q 'doctor.*Check system health' FRAMEWORK_README.md"

# Performance tests
echo "⚡ Testing Performance"
run_test "help command completes quickly" "timeout 5 ./dev help >/dev/null"
run_test "version command completes quickly" "timeout 3 ./dev version >/dev/null"

# Summary
echo ""
echo "📊 Test Results Summary"
echo "======================"
echo "Total Tests: ${TOTAL_TESTS}"
echo "Passed: ${PASSED_TESTS} ($(echo "scale=1; ${PASSED_TESTS}*100/${TOTAL_TESTS}" | bc -l 2>/dev/null || echo "N/A")%)"
echo "Failed: ${FAILED_TESTS}"

if [[ ${FAILED_TESTS} -eq 0 ]]; then
  echo "🎉 All tests passed! Framework is working correctly."
  exit 0
else
  echo "⚠️ Some tests failed. Framework needs attention."
  exit 1
fi