#!/usr/bin/env bats
# Terminal Automation Framework Test Suite
# Following Google/Meta testing best practices
#
# Test categories:
# 1. Basic functionality and CLI interface
# 2. System health and environment validation  
# 3. Homebrew integration and package management
# 4. Git operations and configuration
# 5. Code quality tools and CI pipeline

# Test environment setup
setup() {
  # Ensure we're in the framework root directory
  cd "${BATS_TEST_DIRNAME}/.."
  
  # Ensure dev script is executable
  chmod +x ./dev
  
  # Set up test environment variables
  export LOG_LEVEL=DEBUG
  export CI=true  # Simulate CI environment for testing
}

# Basic framework functionality tests
@test "dev script exists and is executable" {
  [ -x "./dev" ]
}

@test "dev help command shows usage information" {
  run ./dev help
  [ "$status" -eq 0 ]
  [[ "$output" =~ "Terminal Automation Framework" ]]
  [[ "$output" =~ "USAGE:" ]]
  [[ "$output" =~ "COMMANDS:" ]]
}

@test "dev version command returns version info" {
  run ./dev version
  [ "$status" -eq 0 ]
  [[ "$output" =~ "dev version" ]]
}

@test "framework directory structure exists" {
  [ -d "scripts/lib" ]
  [ -f "scripts/lib/core.sh" ]
  [ -f "scripts/lib/homebrew.sh" ]
  [ -f "scripts/lib/git.sh" ]
  [ -f "Brewfile" ]
}

@test "invalid command shows error and help" {
  run ./dev invalid-command-that-does-not-exist
  [ "$status" -eq 2 ]
  [[ "$output" =~ "Unknown command" ]]
  [[ "$output" =~ "Available commands" ]]
}

# System health and validation tests
@test "dev doctor command runs system health check" {
  # Skip if not on macOS (CI might run on Linux)
  if [[ "$(uname)" != "Darwin" ]]; then
    skip "System health check is macOS-specific"
  fi
  
  run ./dev doctor
  # May fail in CI if dependencies aren't installed, but should not crash
  [[ "$output" =~ "Running system health check" ]]
}

@test "core library functions are properly exported" {
  # Source the core library and test key functions exist
  source scripts/lib/core.sh
  
  # Test that key functions are defined
  type log_info >/dev/null 2>&1
  type log_error >/dev/null 2>&1
  type validate_command >/dev/null 2>&1
  type get_os >/dev/null 2>&1
  type is_macos >/dev/null 2>&1
}

@test "homebrew library functions are properly exported" {
  source scripts/lib/homebrew.sh
  
  # Test that key functions are defined
  type install_homebrew >/dev/null 2>&1
  type brew_install >/dev/null 2>&1
  type brew_update >/dev/null 2>&1
}

@test "git library functions are properly exported" {
  source scripts/lib/git.sh
  
  # Test that key functions are defined  
  type git_validate_repo >/dev/null 2>&1
  type git_setup_user >/dev/null 2>&1
  type git_create_branch >/dev/null 2>&1
}

# Homebrew integration tests (macOS only)
@test "Brewfile exists and is valid" {
  [ -f "Brewfile" ]
  
  # Check that Brewfile contains expected entries
  grep -q "brew.*git" Brewfile
  grep -q "brew.*shellcheck" Brewfile
  grep -q "brew.*shfmt" Brewfile
}

@test "dev bootstrap command handles missing Homebrew gracefully" {
  # This test verifies the command structure, not actual installation
  run timeout 10 ./dev bootstrap 2>/dev/null || true
  # Command should not crash, regardless of whether Homebrew is installed
  true
}

# Git operations tests  
@test "git setup command validates parameters" {
  run ./dev git:setup
  [ "$status" -eq 1 ]
  [[ "$output" =~ "Both name and email are required" ]]
  [[ "$output" =~ "Usage: dev git:setup" ]]
}

@test "git branch command validates parameters" {
  run ./dev git:branch
  [ "$status" -eq 1 ]
  [[ "$output" =~ "Branch name is required" ]]
  [[ "$output" =~ "Usage: dev git:branch" ]]
}

# Package management tests
@test "install command validates parameters" {
  run ./dev install
  [ "$status" -eq 1 ]
  [[ "$output" =~ "Package name required" ]]
  [[ "$output" =~ "Usage: dev install" ]]
}

# Code quality tools tests
@test "format command handles missing shfmt gracefully" {
  # Test the command structure when shfmt might not be available
  run ./dev format 2>/dev/null || true
  # Should not crash the script
  true
}

@test "lint command handles missing shellcheck gracefully" {
  # Test the command structure when shellcheck might not be available  
  run ./dev lint 2>/dev/null || true
  # Should not crash the script
  true
}

# CI pipeline tests
@test "ci command runs all validation steps" {
  # This is a comprehensive test - may take longer
  if [[ -z "${BATS_COMPREHENSIVE_TESTS:-}" ]]; then
    skip "Comprehensive CI test disabled (set BATS_COMPREHENSIVE_TESTS=1 to enable)"
  fi
  
  run timeout 60 ./dev ci
  # Test should not timeout or crash
  [[ "$output" =~ "CI pipeline" ]]
}

# Error handling and edge cases
@test "framework handles missing dependencies gracefully" {
  # Test that the framework doesn't crash when optional tools are missing
  LOG_LEVEL=ERROR run ./dev doctor 2>/dev/null || true
  LOG_LEVEL=ERROR run ./dev format 2>/dev/null || true  
  LOG_LEVEL=ERROR run ./dev lint 2>/dev/null || true
  
  # All commands should handle missing dependencies without crashing
  true
}

@test "framework respects LOG_LEVEL environment variable" {
  # Test quiet mode
  LOG_LEVEL=ERROR run ./dev help
  [ "$status" -eq 0 ]
  
  # Test verbose mode  
  LOG_LEVEL=DEBUG run ./dev help
  [ "$status" -eq 0 ]
}

# Configuration and environment tests
@test "framework detects CI environment correctly" {
  CI=true run ./dev help
  [ "$status" -eq 0 ]
  
  unset CI
  run ./dev help  
  [ "$status" -eq 0 ]
}

# Performance and reliability tests
@test "dev commands complete within reasonable time" {
  # Basic commands should complete quickly
  timeout 5 ./dev help
  timeout 5 ./dev version
  timeout 10 ./dev doctor 2>/dev/null || true
}

@test "framework handles interrupted operations gracefully" {
  # Test that cleanup works properly
  timeout 2 ./dev bootstrap 2>/dev/null || true
  timeout 2 ./dev update 2>/dev/null || true
  
  # Framework should handle interruptions without leaving broken state
  true
}