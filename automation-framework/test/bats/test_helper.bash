#!/usr/bin/env bash
# BATS test helper functions

# Skip test if command is not available
skip_if_missing() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    skip "$cmd not available"
  fi
}

# Skip if not on macOS
skip_if_not_macos() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    skip "Test requires macOS"
  fi
}

# Skip if not in CI
skip_if_not_ci() {
  if [[ -z "${CI:-}" && -z "${GITHUB_ACTIONS:-}" ]]; then
    skip "Test requires CI environment"
  fi
}

# Create temporary file
temp_file() {
  mktemp "${BATS_TMPDIR}/bats-test-XXXXXX"
}

# Create temporary directory
temp_dir() {
  mktemp -d "${BATS_TMPDIR}/bats-test-XXXXXX"
}

# Clean up function
cleanup() {
  # Remove any temporary files created during tests
  rm -rf "${BATS_TMPDIR}/bats-test-"*
}

# Setup function called before each test
setup() {
  # Set test environment variables
  export LOG_LEVEL="error"  # Reduce noise in tests
  export NO_COLOR="1"       # Disable colors in tests
}

# Teardown function called after each test
teardown() {
  cleanup
}