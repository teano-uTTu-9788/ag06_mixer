#!/usr/bin/env bash

# Quick Start Script for Workflow Automation Framework
# Sets up and demonstrates the complete system

set -euo pipefail

FRAMEWORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     WORKFLOW AUTOMATION FRAMEWORK - QUICK START GUIDE        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Function to print colored output
print_step() {
    echo -e "\n\033[1;34m▶ $1\033[0m"
}

print_success() {
    echo -e "\033[1;32m✓ $1\033[0m"
}

print_error() {
    echo -e "\033[1;31m✗ $1\033[0m"
}

print_info() {
    echo -e "\033[1;33mℹ $1\033[0m"
}

# Check dependencies
check_dependencies() {
    print_step "Checking system dependencies..."
    
    local missing=()
    
    # Required commands
    for cmd in bash python3 jq curl; do
        if command -v "$cmd" >/dev/null 2>&1; then
            print_success "$cmd is installed"
        else
            print_error "$cmd is not installed"
            missing+=("$cmd")
        fi
    done
    
    # Python packages
    if python3 -c "import psutil" 2>/dev/null; then
        print_success "Python psutil module is installed"
    else
        print_info "Installing psutil module..."
        pip3 install psutil --user
    fi
    
    if [[ ${#missing[@]} -gt 0 ]]; then
        print_error "Missing dependencies: ${missing[*]}"
        print_info "Please install missing dependencies and run again"
        exit 1
    fi
    
    print_success "All dependencies satisfied!"
}

# Make scripts executable
setup_permissions() {
    print_step "Setting up permissions..."
    
    chmod +x "$FRAMEWORK_DIR"/*.sh
    chmod +x "$FRAMEWORK_DIR"/daemon/*.sh
    chmod +x "$FRAMEWORK_DIR"/chaos/*.sh
    chmod +x "$FRAMEWORK_DIR"/monitoring/*.py
    
    print_success "Permissions configured"
}

# Initialize the framework
initialize_framework() {
    print_step "Initializing framework..."
    
    "$FRAMEWORK_DIR/workflow_orchestrator.sh" init
    
    print_success "Framework initialized"
}

# Start core components
start_framework() {
    print_step "Starting framework components..."
    
    "$FRAMEWORK_DIR/workflow_orchestrator.sh" start
    
    print_success "Framework started"
}

# Run demo workflows
run_demos() {
    print_step "Running demonstration workflows..."
    
    # Demo 1: Simple sequential workflow
    print_info "Demo 1: Sequential task execution"
    cat > /tmp/demo_sequential.sh <<'EOF'
#!/bin/bash
echo "Task 1: Initializing..."
sleep 1
echo "Task 2: Processing data..."
sleep 1
echo "Task 3: Generating report..."
sleep 1
echo "Task 4: Cleanup..."
echo "Sequential workflow completed!"
EOF
    chmod +x /tmp/demo_sequential.sh
    
    "$FRAMEWORK_DIR/daemon/workflow_daemon.sh" execute /tmp/demo_sequential.sh
    
    sleep 2
    
    # Demo 2: Parallel task execution
    print_info "Demo 2: Parallel task execution"
    
    for i in {1..5}; do
        "$FRAMEWORK_DIR/daemon/parallel_executor.sh" add "echo 'Parallel task $i' && sleep 2" 5
    done
    
    sleep 5
    "$FRAMEWORK_DIR/daemon/parallel_executor.sh" stats
    
    # Demo 3: Hermetic environment
    print_info "Demo 3: Hermetic environment isolation"
    
    "$FRAMEWORK_DIR/daemon/hermetic_env.sh" create demo_env
    "$FRAMEWORK_DIR/daemon/hermetic_env.sh" verify demo_env
    
    # Demo 4: Metrics collection
    print_info "Demo 4: Metrics collection and monitoring"
    
    # Custom metric
    echo '[{"name":"demo.metric","value":42,"type":"gauge"}]' > /tmp/custom_metrics.json
    
    sleep 5
    
    # Check metrics
    if curl -s http://localhost:8080/metrics >/dev/null 2>&1; then
        print_success "Metrics dashboard is accessible at http://localhost:8080"
    fi
}

# Run basic chaos test
run_chaos_test() {
    print_step "Running chaos engineering test (optional)..."
    
    read -p "Run chaos test? (y/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Running CPU stress test for 10 seconds..."
        "$FRAMEWORK_DIR/chaos/chaos_test.sh" cpu 10 50
        print_success "Chaos test completed"
    else
        print_info "Skipping chaos test"
    fi
}

# Show status and next steps
show_summary() {
    print_step "Framework Status"
    
    "$FRAMEWORK_DIR/workflow_orchestrator.sh" health
    
    echo ""
    print_step "Quick Start Complete!"
    
    cat <<EOF

📚 NEXT STEPS:
──────────────────────────────────────────────────────────────

1. View the metrics dashboard:
   Open http://localhost:8080 in your browser

2. Execute the sample workflow:
   $FRAMEWORK_DIR/workflow_orchestrator.sh execute config/sample_workflow.yaml

3. Create a hermetic environment:
   $FRAMEWORK_DIR/daemon/hermetic_env.sh create myproject

4. Run parallel tasks:
   $FRAMEWORK_DIR/daemon/parallel_executor.sh batch my_tasks.txt

5. Monitor system status:
   $FRAMEWORK_DIR/workflow_orchestrator.sh dashboard

6. Run chaos tests (carefully!):
   $FRAMEWORK_DIR/chaos/chaos_test.sh validate
   $FRAMEWORK_DIR/chaos/chaos_test.sh suite

7. View logs:
   $FRAMEWORK_DIR/workflow_orchestrator.sh logs

8. Stop the framework:
   $FRAMEWORK_DIR/workflow_orchestrator.sh stop

📖 DOCUMENTATION:
──────────────────────────────────────────────────────────────

- Configuration: $FRAMEWORK_DIR/config/orchestrator.conf
- Sample workflow: $FRAMEWORK_DIR/config/sample_workflow.yaml
- Component help: <component>.sh help

🛠 AVAILABLE COMPONENTS:
──────────────────────────────────────────────────────────────

✓ Workflow Daemon - Persistent background service
✓ Hermetic Environments - Isolated build environments  
✓ Parallel Executor - Concurrent task execution
✓ Metrics Collector - Real-time monitoring
✓ Chaos Testing - Resilience validation
✓ CI/CD Pipeline - Automated deployment

💡 TIPS:
──────────────────────────────────────────────────────────────

• Use hermetic environments for reproducible builds
• Enable parallel execution for independent tasks
• Monitor metrics to identify bottlenecks
• Run chaos tests in staging before production
• Check health regularly with 'orchestrator health'

🔧 TROUBLESHOOTING:
──────────────────────────────────────────────────────────────

If components fail to start:
1. Check permissions: ls -la $FRAMEWORK_DIR/**/*.sh
2. Verify dependencies: $0 (this script)
3. Check logs: tail -f /tmp/*.log
4. Clean and restart: orchestrator clean && orchestrator start

Need help? Run: $FRAMEWORK_DIR/workflow_orchestrator.sh help

EOF
}

# Cleanup function
cleanup() {
    print_info "Cleaning up..."
    rm -f /tmp/demo_* 2>/dev/null || true
}

# Main execution
main() {
    trap cleanup EXIT
    
    check_dependencies
    setup_permissions
    initialize_framework
    start_framework
    
    echo ""
    read -p "Run demonstration workflows? (y/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_demos
        run_chaos_test
    fi
    
    show_summary
}

# Run main function
main "$@"