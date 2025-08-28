#!/usr/bin/env bash
# Stop Autonomous Universal Workflow System

PID_FILE="autonomous_workflow.pid"

if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "🛑 Stopping autonomous workflow agent (PID: $PID)..."
        kill "$PID"
        sleep 2
        
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "⚡ Force killing agent..."
            kill -9 "$PID"
        fi
        
        rm -f "$PID_FILE"
        echo "✅ Autonomous workflow stopped"
    else
        echo "⚠️ Agent not running (removing stale PID file)"
        rm -f "$PID_FILE"
    fi
else
    echo "❌ No PID file found - agent may not be running"
fi