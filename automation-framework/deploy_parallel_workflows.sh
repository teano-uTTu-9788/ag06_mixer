#!/bin/bash
# Parallel Workflow Deployment Script for AiOke Development
# Deploys and coordinates multiple Claude instances working on AiOke improvements

set -euo pipefail

# Source framework libraries
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/lib/core.sh"

# Initialize error handling
setup_error_handling "parallel-deploy"

# Configuration
readonly WORKFLOW_DIR="${HOME}/aioke_parallel_workflows"
readonly DEPLOYMENT_LOG="${WORKFLOW_DIR}/deployment.log"

# Instance configurations for parallel work
readonly INSTANCE_CONFIGS=(
    "audio_specialist:audio_processing:Specialized Claude for advanced audio processing and DSP"
    "ui_specialist:ui_development:Specialized Claude for React/React Native UI development"
    "api_specialist:api_integration:Specialized Claude for API integrations and external services"
    "test_specialist:testing_validation:Specialized Claude for comprehensive testing and QA"
    "docs_specialist:documentation:Specialized Claude for technical documentation"
    "perf_specialist:performance_optimization:Specialized Claude for performance optimization and WASM"
)

# Main deployment function
main() {
    local action="${1:-deploy}"
    
    case "${action}" in
        deploy)
            deploy_parallel_framework
            ;;
        status)
            show_deployment_status
            ;;
        cleanup)
            cleanup_deployment
            ;;
        help)
            show_help
            ;;
        *)
            log_error "Unknown action: ${action}"
            show_help
            exit 1
            ;;
    esac
}

# Deploy the parallel workflow framework
deploy_parallel_framework() {
    log_info "🚀 Deploying Parallel Workflow Framework for AiOke Development"
    
    # Step 1: Initialize workflow environment
    log_info "📋 Step 1: Initializing workflow environment..."
    "${SCRIPT_DIR}/parallel_workflow_orchestrator.sh" init
    
    # Step 2: Create AiOke improvement tasks if not already created
    local task_count
    task_count=$(find "${WORKFLOW_DIR}/tasks" -name "*.json" 2>/dev/null | wc -l || echo "0")
    
    if [[ ${task_count} -eq 0 ]]; then
        log_info "📝 Step 2: Creating AiOke improvement tasks..."
        "${SCRIPT_DIR}/parallel_workflow_orchestrator.sh" create-aioke-tasks > /dev/null
        log_ok "Created 18 AiOke improvement tasks"
    else
        log_info "📝 Step 2: Found ${task_count} existing tasks"
    fi
    
    # Step 3: Register specialized instances
    log_info "👥 Step 3: Registering specialized Claude instances..."
    for config in "${INSTANCE_CONFIGS[@]}"; do
        local instance_id="${config%%:*}"
        local remainder="${config#*:}"
        local category="${remainder%%:*}"
        local description="${remainder#*:}"
        
        log_info "  Registering ${instance_id} for ${category}"
        "${SCRIPT_DIR}/parallel_workflow_orchestrator.sh" register "${instance_id}" "${category}" "${description}"
    done
    
    # Step 4: Show deployment status
    log_info "📊 Step 4: Deployment status..."
    "${SCRIPT_DIR}/parallel_workflow_orchestrator.sh" status
    
    # Step 5: Create instance coordination guide
    create_instance_coordination_guide
    
    # Step 6: Create monitoring dashboard
    create_monitoring_dashboard
    
    log_ok "✅ Parallel Workflow Framework deployed successfully!"
    log_info "🎯 Next steps:"
    log_info "  1. Each Claude instance should run: ./dev parallel:distribute"
    log_info "  2. Monitor progress with: ./dev parallel:monitor"
    log_info "  3. Check status with: ./dev parallel:status"
}

# Create instance coordination guide
create_instance_coordination_guide() {
    log_info "📖 Creating instance coordination guide..."
    
    cat > "${WORKFLOW_DIR}/INSTANCE_COORDINATION_GUIDE.md" <<'EOF'
# AiOke Parallel Development - Instance Coordination Guide

## Overview
This guide coordinates multiple Claude instances working in parallel on AiOke improvements.

## Instance Assignments

### Audio Specialist (`audio_specialist`)
**Specialization**: Advanced audio processing and DSP
**Tasks**:
- Implement Advanced Noise Reduction (AI-powered, spectral subtraction, deep learning)
- Real-time Pitch Correction (auto-tune with configurable parameters)  
- Multi-track Recording (multiple vocal tracks with mixing capabilities)

**Technical Focus**:
- Audio processing algorithms
- Real-time DSP implementation
- Audio format handling and conversion
- Performance optimization for audio workloads

### UI Specialist (`ui_specialist`)
**Specialization**: React/React Native UI development
**Tasks**:
- Responsive Web Interface (React-based with real-time waveform visualization)
- Mobile App Development (React Native for iOS/Android)
- Dark Mode Support (system-wide with theme customization)

**Technical Focus**:
- Modern React patterns and hooks
- Responsive design and CSS-in-JS
- Mobile app development with React Native
- UI/UX best practices and accessibility

### API Specialist (`api_specialist`)
**Specialization**: API integrations and external services
**Tasks**:
- Spotify Integration (playlist import, track metadata)
- YouTube Music Sync (karaoke track search and download)
- Cloud Storage Integration (Google Drive/Dropbox backup)

**Technical Focus**:
- REST API integration patterns
- OAuth and authentication flows
- Data synchronization and caching
- Rate limiting and error handling

### Test Specialist (`test_specialist`)
**Specialization**: Comprehensive testing and QA
**Tasks**:
- Comprehensive Test Suite (88-test validation for all components)
- Performance Benchmarking (latency, CPU, memory optimization)
- Cross-platform Testing (Windows, macOS, Linux validation)

**Technical Focus**:
- Test automation frameworks
- Performance testing and profiling
- Cross-platform compatibility
- CI/CD pipeline integration

### Documentation Specialist (`docs_specialist`)
**Specialization**: Technical documentation
**Tasks**:
- API Documentation (OpenAPI spec, interactive docs)
- User Guide (comprehensive manual with tutorials)
- Developer Onboarding (contribution guidelines, setup instructions)

**Technical Focus**:
- Technical writing and documentation
- API documentation standards
- User experience documentation
- Developer experience optimization

### Performance Specialist (`perf_specialist`)
**Specialization**: Performance optimization and WebAssembly
**Tasks**:
- WebAssembly Audio Engine (WASM for browser performance)
- GPU Acceleration (WebGL/Metal for parallel audio processing)
- Caching Strategy (intelligent caching for processed audio segments)

**Technical Focus**:
- WebAssembly and low-level optimization
- GPU programming and parallel processing
- Caching strategies and data structures
- Performance profiling and benchmarking

## Coordination Protocol

### Getting Next Task
Each specialist should:
1. Get next available task: `./dev parallel:next-task <category>`
2. Assign task to self: `./dev parallel:assign <task_id> <instance_id>`
3. Work on the task
4. Mark complete: `./dev parallel:complete <task_id> <result>`

### Communication
- Status updates: Use `./dev parallel:status` to see overall progress
- Task coordination: Check task distribution before claiming work
- Progress sharing: Update task results with detailed outcomes

### Quality Standards
- All implementations must follow SOLID principles
- Code must be production-ready with proper error handling
- Include comprehensive tests for new functionality
- Document all APIs and complex algorithms
- Follow existing AiOke architectural patterns

## Example Workflow

```bash
# 1. Get next task for your specialty
TASK_ID=$(./dev parallel:next-task audio_processing)

# 2. Assign to yourself
./dev parallel:assign $TASK_ID audio_specialist

# 3. Work on the task (implement the feature)
# ... implement advanced noise reduction ...

# 4. Mark as complete with results
./dev parallel:complete $TASK_ID "Implemented AI-powered noise reduction using spectral subtraction with 15dB improvement"

# 5. Check overall status
./dev parallel:status
```

## Success Metrics
- **Audio Processing**: Measurable audio quality improvements (SNR, THD, latency)
- **UI Development**: User experience metrics (load time, responsiveness, accessibility)
- **API Integration**: Reliability metrics (success rate, response time, error handling)
- **Testing**: Coverage metrics (test coverage, performance benchmarks, platform compatibility)
- **Documentation**: Completeness metrics (API coverage, user guide sections, examples)
- **Performance**: Quantifiable improvements (speed, memory usage, throughput)

## Integration Points
All specialists should ensure their work integrates properly:
- **Audio + UI**: Real-time visualization of audio processing
- **Audio + Performance**: Optimized audio pipelines
- **API + UI**: Seamless integration of external services
- **Testing + All**: Comprehensive validation of all components
- **Documentation + All**: Complete documentation for all features

## Final Integration
Once individual components are complete:
1. Integration testing across all components
2. End-to-end workflow validation
3. Performance optimization of the complete system
4. Final documentation and deployment preparation

Happy parallel development! 🚀
EOF

    log_ok "Instance coordination guide created"
}

# Create monitoring dashboard
create_monitoring_dashboard() {
    log_info "📊 Creating monitoring dashboard..."
    
    cat > "${WORKFLOW_DIR}/monitoring_dashboard.html" <<'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AiOke Parallel Development Dashboard</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .dashboard {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #2d3748;
            margin-bottom: 30px;
            font-size: 2.5em;
            text-align: center;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: #f7fafc;
            border-radius: 12px;
            padding: 24px;
            border-left: 4px solid #667eea;
        }
        .stat-number {
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 8px;
        }
        .stat-label {
            color: #718096;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .progress-section {
            margin-bottom: 30px;
        }
        .category-progress {
            margin-bottom: 16px;
        }
        .category-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
        }
        .progress-bar {
            background: #e2e8f0;
            border-radius: 6px;
            height: 8px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s ease;
        }
        .instances-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .instance-card {
            background: #f7fafc;
            border-radius: 12px;
            padding: 20px;
            border-top: 3px solid #48bb78;
        }
        .instance-header {
            display: flex;
            justify-content: between;
            align-items: center;
            margin-bottom: 12px;
        }
        .instance-name {
            font-weight: bold;
            color: #2d3748;
        }
        .status-badge {
            background: #48bb78;
            color: white;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.75em;
            text-transform: uppercase;
        }
        .refresh-button {
            position: fixed;
            top: 20px;
            right: 20px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            font-size: 24px;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            transition: transform 0.2s ease;
        }
        .refresh-button:hover {
            transform: scale(1.1);
        }
    </style>
</head>
<body>
    <button class="refresh-button" onclick="location.reload()" title="Refresh Dashboard">🔄</button>
    
    <div class="dashboard">
        <h1>🎵 AiOke Parallel Development Dashboard</h1>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number" id="total-tasks">18</div>
                <div class="stat-label">Total Tasks</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="completed-tasks">0</div>
                <div class="stat-label">Completed</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="active-instances">6</div>
                <div class="stat-label">Active Instances</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="completion-rate">0%</div>
                <div class="stat-label">Completion Rate</div>
            </div>
        </div>
        
        <div class="progress-section">
            <h2>📊 Category Progress</h2>
            <div class="category-progress">
                <div class="category-header">
                    <span>🎵 Audio Processing</span>
                    <span>0/3</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: 0%"></div>
                </div>
            </div>
            <div class="category-progress">
                <div class="category-header">
                    <span>🎨 UI Development</span>
                    <span>0/3</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: 0%"></div>
                </div>
            </div>
            <div class="category-progress">
                <div class="category-header">
                    <span>🔗 API Integration</span>
                    <span>0/3</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: 0%"></div>
                </div>
            </div>
            <div class="category-progress">
                <div class="category-header">
                    <span>🧪 Testing & Validation</span>
                    <span>0/3</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: 0%"></div>
                </div>
            </div>
            <div class="category-progress">
                <div class="category-header">
                    <span>📚 Documentation</span>
                    <span>0/3</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: 0%"></div>
                </div>
            </div>
            <div class="category-progress">
                <div class="category-header">
                    <span>⚡ Performance Optimization</span>
                    <span>0/3</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: 0%"></div>
                </div>
            </div>
        </div>
        
        <div class="instances-section">
            <h2>🤖 Active Instances</h2>
            <div class="instances-grid">
                <div class="instance-card">
                    <div class="instance-header">
                        <span class="instance-name">🎵 Audio Specialist</span>
                        <span class="status-badge">Active</span>
                    </div>
                    <div>Specializing in advanced audio processing and DSP</div>
                    <div style="margin-top: 8px; font-size: 0.9em; color: #718096;">Tasks: 0 completed</div>
                </div>
                <div class="instance-card">
                    <div class="instance-header">
                        <span class="instance-name">🎨 UI Specialist</span>
                        <span class="status-badge">Active</span>
                    </div>
                    <div>Specializing in React/React Native UI development</div>
                    <div style="margin-top: 8px; font-size: 0.9em; color: #718096;">Tasks: 0 completed</div>
                </div>
                <div class="instance-card">
                    <div class="instance-header">
                        <span class="instance-name">🔗 API Specialist</span>
                        <span class="status-badge">Active</span>
                    </div>
                    <div>Specializing in API integrations and external services</div>
                    <div style="margin-top: 8px; font-size: 0.9em; color: #718096;">Tasks: 0 completed</div>
                </div>
                <div class="instance-card">
                    <div class="instance-header">
                        <span class="instance-name">🧪 Test Specialist</span>
                        <span class="status-badge">Active</span>
                    </div>
                    <div>Specializing in comprehensive testing and QA</div>
                    <div style="margin-top: 8px; font-size: 0.9em; color: #718096;">Tasks: 0 completed</div>
                </div>
                <div class="instance-card">
                    <div class="instance-header">
                        <span class="instance-name">📚 Documentation Specialist</span>
                        <span class="status-badge">Active</span>
                    </div>
                    <div>Specializing in technical documentation</div>
                    <div style="margin-top: 8px; font-size: 0.9em; color: #718096;">Tasks: 0 completed</div>
                </div>
                <div class="instance-card">
                    <div class="instance-header">
                        <span class="instance-name">⚡ Performance Specialist</span>
                        <span class="status-badge">Active</span>
                    </div>
                    <div>Specializing in performance optimization and WebAssembly</div>
                    <div style="margin-top: 8px; font-size: 0.9em; color: #718096;">Tasks: 0 completed</div>
                </div>
            </div>
        </div>
        
        <div style="margin-top: 40px; padding: 20px; background: #edf2f7; border-radius: 12px; text-align: center;">
            <h3>🎯 Current Status: Development Phase</h3>
            <p>All instances registered and ready for parallel development. Tasks distributed across 6 specialized categories.</p>
            <p><strong>Next:</strong> Each Claude instance should pick up tasks from their specialty area and begin implementation.</p>
        </div>
    </div>
    
    <script>
        // Auto-refresh every 30 seconds
        setTimeout(() => location.reload(), 30000);
    </script>
</body>
</html>
EOF

    log_ok "Monitoring dashboard created at: ${WORKFLOW_DIR}/monitoring_dashboard.html"
}

# Show deployment status
show_deployment_status() {
    log_info "📊 Parallel Workflow Deployment Status"
    echo "========================================"
    
    if [[ -d "${WORKFLOW_DIR}" ]]; then
        log_ok "✅ Workflow environment: Initialized"
        
        # Count tasks
        local task_count
        task_count=$(find "${WORKFLOW_DIR}/tasks" -name "*.json" 2>/dev/null | wc -l || echo "0")
        log_info "📝 Tasks created: ${task_count}"
        
        # Count instances
        local instance_count
        instance_count=$(find "${WORKFLOW_DIR}/instances" -name "*.json" 2>/dev/null | wc -l || echo "0")
        log_info "👥 Instances registered: ${instance_count}"
        
        # Show detailed status
        echo ""
        "${SCRIPT_DIR}/parallel_workflow_orchestrator.sh" status
        
        echo ""
        log_info "📊 Dashboard: file://${WORKFLOW_DIR}/monitoring_dashboard.html"
        log_info "📖 Guide: ${WORKFLOW_DIR}/INSTANCE_COORDINATION_GUIDE.md"
    else
        log_warn "❌ Workflow environment not initialized"
        log_info "Run: ./deploy_parallel_workflows.sh deploy"
    fi
}

# Cleanup deployment
cleanup_deployment() {
    log_warn "🧹 Cleaning up parallel workflow deployment..."
    
    if [[ -d "${WORKFLOW_DIR}" ]]; then
        rm -rf "${WORKFLOW_DIR}"
        log_ok "Workflow directory removed"
    fi
    
    log_ok "Cleanup completed"
}

# Show help
show_help() {
    cat <<HELP
Parallel Workflow Deployment Script for AiOke Development

Usage: $0 [action]

Actions:
  deploy    Deploy the complete parallel workflow framework
  status    Show current deployment status  
  cleanup   Remove all workflow files and reset
  help      Show this help message

Examples:
  ./deploy_parallel_workflows.sh deploy   # Full deployment
  ./deploy_parallel_workflows.sh status   # Check status
  ./deploy_parallel_workflows.sh cleanup  # Clean reset

The deployment creates:
- 18 AiOke improvement tasks across 6 categories
- 6 specialized Claude instance registrations
- Instance coordination guide
- Real-time monitoring dashboard
- Complete workflow orchestration system
HELP
}

# Run main function
main "$@"