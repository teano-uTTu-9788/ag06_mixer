#!/usr/bin/env bash
# Start Autonomous Universal Workflow System
# Creates truly autonomous operation with self-managing Claude instances

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/autonomous_workflow.log"
PID_FILE="$SCRIPT_DIR/autonomous_workflow.pid"

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🤖 AUTONOMOUS UNIVERSAL WORKFLOW STARTUP${NC}"
echo "========================================"

# Check if already running
if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Autonomous workflow already running (PID: $PID)${NC}"
        echo "Use ./stop_autonomous_workflow.sh to stop"
        exit 1
    else
        echo -e "${YELLOW}Removing stale PID file${NC}"
        rm -f "$PID_FILE"
    fi
fi

# Start autonomous agent
echo -e "${BLUE}Starting autonomous workflow agent...${NC}"
cd "$SCRIPT_DIR"

# Run in background with logging
nohup python3 autonomous_workflow_agent.py > "$LOG_FILE" 2>&1 &
AGENT_PID=$!

# Save PID
echo "$AGENT_PID" > "$PID_FILE"

echo -e "${GREEN}✅ Autonomous workflow started${NC}"
echo -e "   PID: $AGENT_PID"
echo -e "   Log: $LOG_FILE"
echo -e "   Config: autonomous_config.json"

# Show initial status
sleep 3
if ps -p "$AGENT_PID" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Agent is running successfully${NC}"
    
    # Show recent log output
    echo -e "\n${BLUE}📋 Recent Activity:${NC}"
    tail -10 "$LOG_FILE" | sed 's/^/   /'
    
    echo -e "\n${BLUE}📚 Management Commands:${NC}"
    echo "   Monitor logs:     tail -f $LOG_FILE"
    echo "   Check status:     ./status_autonomous_workflow.sh"
    echo "   Stop agent:       ./stop_autonomous_workflow.sh"
    echo "   Restart agent:    ./restart_autonomous_workflow.sh"
    
    echo -e "\n${BLUE}🎯 Autonomous Features Active:${NC}"
    echo "   • Auto-registers 4 Claude instances"
    echo "   • Analyzes project every hour"
    echo "   • Distributes tasks every 5 minutes"
    echo "   • Monitors progress every minute"
    echo "   • Simulates task completion (demo mode)"
    
else
    echo -e "${RED}❌ Failed to start autonomous agent${NC}"
    if [[ -f "$LOG_FILE" ]]; then
        echo "Error logs:"
        tail -20 "$LOG_FILE"
    fi
    exit 1
fi