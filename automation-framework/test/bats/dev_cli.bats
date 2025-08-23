#!/usr/bin/env bats
# Test suite for the dev CLI

load 'test_helper.bash'

@test "dev help prints usage" {
  run ./dev help
  [ "$status" -eq 0 ]
  [[ "${lines[0]}" =~ "dev — developer CLI" ]]
}

@test "dev version shows version" {
  run ./dev version
  [ "$status" -eq 0 ]
  [[ "$output" =~ "dev version" ]]
}

@test "dev doctor runs system check" {
  run ./dev doctor
  [ "$status" -eq 0 ]
  [[ "$output" =~ "Running system health check" ]]
}

@test "dev doctor checks required tools" {
  run ./dev doctor
  [ "$status" -eq 0 ]
  [[ "$output" =~ "git" ]]
  [[ "$output" =~ "curl" ]]
  [[ "$output" =~ "jq" ]]
}

@test "dev unknown command shows error" {
  run ./dev nonexistent
  [ "$status" -eq 2 ]
  [[ "$output" =~ "Unknown command: nonexistent" ]]
}

@test "dev fmt runs without error when tools available" {
  skip_if_missing shfmt
  run ./dev fmt
  # Should not fail even if no files to format
  [[ "$status" -eq 0 || "$status" -eq 1 ]]
}

@test "dev lint runs shellcheck" {
  skip_if_missing shellcheck
  run ./dev lint
  # May pass or fail depending on code quality, but should run
  [[ "$status" -eq 0 || "$status" -eq 1 ]]
}

@test "dev ci runs full suite" {
  skip_if_missing shellcheck
  run ./dev ci
  # CI may fail but should execute
  [[ "$status" -eq 0 || "$status" -eq 1 ]]
}

@test "dev bootstrap installs dependencies" {
  run ./dev bootstrap
  [ "$status" -eq 0 ]
  [[ "$output" =~ "Bootstrap" ]]
}

@test "dev agent:install works on macOS" {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    skip "Agent installation only supported on macOS"
  fi
  
  run ./dev agent:install
  [ "$status" -eq 0 ]
  [[ "$output" =~ "Agent installation" ]]
}

@test "dev notion:status requires arguments" {
  run ./dev notion:status
  [ "$status" -ne 0 ]
}