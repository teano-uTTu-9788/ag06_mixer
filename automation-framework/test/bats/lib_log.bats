#!/usr/bin/env bats
# Test suite for log.sh library

load 'test_helper.bash'

setup() {
  export NO_COLOR=1
  export LOG_LEVEL=debug
  source scripts/lib/log.sh
}

@test "log_info outputs formatted message" {
  run log_info "test message"
  [ "$status" -eq 0 ]
  [[ "$output" =~ "[INFO] test message" ]]
}

@test "log_error outputs to stderr" {
  run bash -c "source scripts/lib/log.sh; log_error 'error message' 2>&1"
  [ "$status" -eq 0 ]
  [[ "$output" =~ "[ERROR] error message" ]]
}

@test "log_debug respects LOG_LEVEL" {
  export LOG_LEVEL=info
  source scripts/lib/log.sh
  
  run log_debug "debug message"
  [ "$status" -eq 0 ]
  [[ "$output" == "" ]]
}

@test "log_debug shows when LOG_LEVEL is debug" {
  export LOG_LEVEL=debug
  source scripts/lib/log.sh
  
  run log_debug "debug message"
  [ "$status" -eq 0 ]
  [[ "$output" =~ "[DEBUG] debug message" ]]
}

@test "log_ok outputs success message" {
  run log_ok "success message"
  [ "$status" -eq 0 ]
  [[ "$output" =~ "[OK] success message" ]]
}

@test "die outputs error and exits" {
  run bash -c "source scripts/lib/log.sh; die 'fatal error'"
  [ "$status" -eq 1 ]
  [[ "$output" =~ "[ERROR] fatal error" ]]
}

@test "_ts outputs timestamp" {
  run _ts
  [ "$status" -eq 0 ]
  [[ "$output" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T ]]
}

@test "NO_COLOR disables colors" {
  export NO_COLOR=1
  
  run bash -c "source scripts/lib/log.sh; _color green; echo 'test'; _reset"
  [ "$status" -eq 0 ]
  [[ "$output" == "test" ]]
}