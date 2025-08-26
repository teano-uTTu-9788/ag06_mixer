#!/usr/bin/env bash

# Environment setup for macOS compatibility
# Creates necessary directories and symlinks

set -euo pipefail

echo "Setting up environment for workflow automation framework..."

# Use macOS temp directory
if [[ "$OSTYPE" == "darwin"* ]]; then
    TEMP_BASE="${TMPDIR:-/var/folders/tmp}"
else
    TEMP_BASE="/tmp"
fi

# Create working directories
mkdir -p "$TEMP_BASE/workflow_daemon"
mkdir -p "$TEMP_BASE/parallel_work_queue"
mkdir -p "$TEMP_BASE/parallel_results"
mkdir -p "$TEMP_BASE/parallel_workers"
mkdir -p "$TEMP_BASE/hermetic_envs"
mkdir -p "$TEMP_BASE/hermetic_cache"
mkdir -p "$TEMP_BASE/chaos_tests"
mkdir -p "$TEMP_BASE/orchestrator"

# Create symlinks for compatibility
if [[ ! -e /tmp && "$OSTYPE" == "darwin"* ]]; then
    echo "Note: /tmp symlink not available, using $TEMP_BASE"
    # Export for use by other scripts
    export WORKFLOW_TEMP="$TEMP_BASE"
else
    export WORKFLOW_TEMP="/tmp"
fi

# Update configuration with correct paths
cat > /tmp/framework_env.sh <<EOF
#!/bin/bash
# Framework environment configuration
export WORKFLOW_TEMP="$WORKFLOW_TEMP"
export DAEMON_DIR="$WORKFLOW_TEMP/workflow_daemon"
export WORK_QUEUE="$WORKFLOW_TEMP/parallel_work_queue"
export RESULT_DIR="$WORKFLOW_TEMP/parallel_results"
export WORKER_DIR="$WORKFLOW_TEMP/parallel_workers"
export HERMETIC_BASE="$WORKFLOW_TEMP/hermetic_envs"
export HERMETIC_CACHE="$WORKFLOW_TEMP/hermetic_cache"
export CHAOS_DIR="$WORKFLOW_TEMP/chaos_tests"
export ORCHESTRATOR_STATE="$WORKFLOW_TEMP/orchestrator.state"
export ORCHESTRATOR_LOG="$WORKFLOW_TEMP/orchestrator.log"
EOF

echo "Environment setup complete!"
echo "Temp directory: $WORKFLOW_TEMP"
echo ""
echo "To use the framework, source the environment:"
echo "  source /tmp/framework_env.sh"