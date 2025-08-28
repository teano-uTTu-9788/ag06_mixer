#!/usr/bin/env bats
#
# Basic functionality tests for Terminal Automation Framework
# Using BATS (Bash Automated Testing System)

load '../test_helper'

setup() {
    # Ensure dev CLI is executable
    chmod +x "$BATS_TEST_DIRNAME/../../dev"
    
    # Set up test environment
    export PATH="$BATS_TEST_DIRNAME/../..:$PATH"
}

@test "dev CLI is executable" {
    [ -x "$BATS_TEST_DIRNAME/../../dev" ]
}

@test "dev CLI shows version" {
    run ./dev version
    [ "$status" -eq 0 ]
    [[ "$output" =~ "Terminal Automation Framework" ]]
}

@test "dev CLI shows help" {
    run ./dev help
    [ "$status" -eq 0 ]
    [[ "$output" =~ "Usage:" ]]
    [[ "$output" =~ "Commands:" ]]
}

@test "dev CLI handles unknown command gracefully" {
    run ./dev nonexistent-command
    [ "$status" -eq 1 ]
    [[ "$output" =~ "Unknown command" ]]
}

@test "dev doctor runs system checks" {
    run ./dev doctor --quiet
    # Should pass or provide useful feedback
    [[ "$status" -eq 0 || "$status" -eq 1 ]]
}

@test "dev clean removes build artifacts" {
    # Create some fake build artifacts
    mkdir -p build
    touch build/test-artifact.tar.gz
    
    run ./dev clean
    [ "$status" -eq 0 ]
    [ ! -d build ]
}

@test "framework loads without errors" {
    run bash -c "source lib/core/bootstrap.sh && echo 'Framework loaded'"
    [ "$status" -eq 0 ]
    [[ "$output" =~ "Framework loaded" ]]
}

@test "logger functions work correctly" {
    run bash -c "
        source lib/core/bootstrap.sh
        source lib/core/logger.sh
        log::info 'Test message'
    "
    [ "$status" -eq 0 ]
    [[ "$output" =~ "Test message" ]]
}

@test "config loading works" {
    run bash -c "
        source lib/core/bootstrap.sh
        source lib/core/logger.sh
        source lib/core/config.sh
        config::get 'log_level' 'INFO'
    "
    [ "$status" -eq 0 ]
}

@test "validation functions work" {
    run bash -c "
        source lib/core/bootstrap.sh
        source lib/core/logger.sh
        source lib/utils/validation.sh
        validate::not_empty 'test_value'
    "
    [ "$status" -eq 0 ]
}

@test "validation correctly fails on empty input" {
    run bash -c "
        source lib/core/bootstrap.sh
        source lib/core/logger.sh
        source lib/utils/validation.sh
        validate::not_empty ''
    "
    [ "$status" -eq 1 ]
}

@test "email validation works correctly" {
    run bash -c "
        source lib/core/bootstrap.sh
        source lib/core/logger.sh
        source lib/utils/validation.sh
        validate::email 'test@example.com'
    "
    [ "$status" -eq 0 ]
}

@test "email validation fails on invalid email" {
    run bash -c "
        source lib/core/bootstrap.sh
        source lib/core/logger.sh
        source lib/utils/validation.sh
        validate::email 'invalid-email'
    "
    [ "$status" -eq 1 ]
}