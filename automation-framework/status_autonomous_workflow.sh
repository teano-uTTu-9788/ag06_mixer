#!/usr/bin/env bash
# Check Status of Autonomous Universal Workflow System

PID_FILE="autonomous_workflow.pid"
LOG_FILE="autonomous_workflow.log"

echo "🔍 AUTONOMOUS WORKFLOW STATUS"
echo "============================="

# Check if running
if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "✅ Status: RUNNING (PID: $PID)"
        
        # Show uptime
        START_TIME=$(ps -o lstart= -p "$PID")
        echo "🕐 Started: $START_TIME"
        
        # Show recent activity
        if [[ -f "$LOG_FILE" ]]; then
            echo -e "\n📋 Recent Activity (last 10 lines):"
            tail -10 "$LOG_FILE" | sed 's/^/   /'
            
            echo -e "\n📊 Activity Summary:"
            echo "   Cycles completed: $(grep -c "AUTONOMOUS CYCLE" "$LOG_FILE" 2>/dev/null || echo "0")"
            echo "   Tasks completed: $(grep -c "Completed task:" "$LOG_FILE" 2>/dev/null || echo "0")"
            echo "   Instances registered: $(grep -c "Registered:" "$LOG_FILE" 2>/dev/null || echo "0")"
        fi
        
    else
        echo "❌ Status: NOT RUNNING (stale PID file)"
        rm -f "$PID_FILE"
    fi
else
    echo "❌ Status: NOT RUNNING"
fi

# Check workflow state
echo -e "\n🗂️  Workflow State:"
if [[ -d ~/.universal_workflows ]]; then
    PROJECTS=$(find ~/.universal_workflows -maxdepth 1 -type d | tail -n +2 | wc -l)
    echo "   Active projects: $PROJECTS"
    
    if [[ $PROJECTS -gt 0 ]]; then
        for project_dir in ~/.universal_workflows/*/; do
            if [[ -d "$project_dir" ]]; then
                project_name=$(basename "$project_dir")
                tasks=$(find "$project_dir/tasks" -name "*.json" 2>/dev/null | wc -l)
                instances=$(find "$project_dir/instances" -name "*.json" 2>/dev/null | wc -l)
                echo "   - $project_name: $tasks tasks, $instances instances"
            fi
        done
    fi
else
    echo "   No workflow directory found"
fi

echo -e "\n🎛️  Control Commands:"
echo "   View live logs:  tail -f $LOG_FILE"
echo "   Stop agent:     ./stop_autonomous_workflow.sh"
echo "   Restart agent:  ./start_autonomous_workflow.sh"