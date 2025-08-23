#!/bin/bash
# AG06 Mixer Development Workflow
# Safe, context-limited workflow for AG06 mixer app development

# ============================================
# CONFIGURATION
# ============================================

export AG06_PROJECT="/Users/nguythe/ag06_mixer"
export AG06_SAFE_MODE=true
export AUTO_CONTINUE_LEVEL=manual
export MAX_TERMINAL_OUTPUT=150

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================
# WORKFLOW FUNCTIONS
# ============================================

# Initialize development session
init_session() {
    echo -e "${BLUE}🎛️  Initializing AG06 Development Session${NC}"
    echo "=========================================="
    
    # Set safety configurations
    source "$AG06_PROJECT/terminal_safety.sh" 2>/dev/null || echo "⚠️  Safety config not found"
    
    # Check AG06 connection
    if system_profiler SPUSBDataType 2>/dev/null | grep -q "AG06"; then
        echo -e "${GREEN}✅ AG06 Connected${NC}"
    else
        echo -e "${YELLOW}⚠️  AG06 Not Connected - Connect device for full functionality${NC}"
    fi
    
    # Create session log
    mkdir -p "$AG06_PROJECT/logs"
    export SESSION_LOG="$AG06_PROJECT/logs/session_$(date +%Y%m%d_%H%M%S).log"
    echo "Session started at $(date)" > "$SESSION_LOG"
    echo -e "${GREEN}✅ Session log: $SESSION_LOG${NC}"
    
    # Check Python environment
    if [ -f "$AG06_PROJECT/requirements.txt" ]; then
        echo -e "${BLUE}📦 Checking dependencies...${NC}"
        python3 -m pip list 2>/dev/null | grep -E "(pyaudio|mido|numpy|tkinter)" | head -5
    fi
    
    echo ""
}

# Start development with monitoring
start_development() {
    echo -e "${BLUE}🚀 Starting AG06 Development Environment${NC}"
    
    # Start monitor in background
    if [ -f "$AG06_PROJECT/ag06_dev_monitor.py" ]; then
        python3 "$AG06_PROJECT/ag06_dev_monitor.py" --status
    fi
    
    echo ""
    echo -e "${GREEN}Ready for development!${NC}"
    echo "Commands:"
    echo "  test    - Run safe test suite"
    echo "  run     - Start mixer app"
    echo "  monitor - Open device monitor"
    echo "  debug   - Save debug snapshot"
    echo ""
}

# Run safe tests
run_tests() {
    echo -e "${BLUE}🧪 Running Safe Test Suite${NC}"
    echo "------------------------"
    
    if [ -f "$AG06_PROJECT/test_ag06_safe.py" ]; then
        python3 "$AG06_PROJECT/test_ag06_safe.py" 2>&1 | tee -a "$SESSION_LOG" | tail -n 30
    else
        echo -e "${RED}❌ Test suite not found${NC}"
    fi
}

# Start mixer application
run_app() {
    echo -e "${BLUE}🎛️  Starting AG06 Mixer App${NC}"
    
    cd "$AG06_PROJECT" || return 1
    
    # Check for main app file
    if [ -f "main.py" ]; then
        echo "Running Python app (output limited)..."
        python3 main.py 2>&1 | tee -a "$SESSION_LOG" | tail -n "$MAX_TERMINAL_OUTPUT"
    elif [ -f "package.json" ]; then
        echo "Running Node.js app (output limited)..."
        npm start 2>&1 | tee -a "$SESSION_LOG" | tail -n "$MAX_TERMINAL_OUTPUT"
    else
        echo -e "${RED}❌ No main.py or package.json found${NC}"
        echo "Create main.py to start development"
    fi
}

# Open device monitor
open_monitor() {
    echo -e "${BLUE}📊 Opening AG06 Device Monitor${NC}"
    
    if [ -f "$AG06_PROJECT/ag06_dev_monitor.py" ]; then
        python3 "$AG06_PROJECT/ag06_dev_monitor.py"
    else
        echo -e "${RED}❌ Monitor not found${NC}"
    fi
}

# Save debug information
save_debug() {
    echo -e "${BLUE}🐛 Saving Debug Information${NC}"
    
    local debug_file="$AG06_PROJECT/logs/debug_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "AG06 Debug Report - $(date)"
        echo "======================================"
        echo ""
        
        echo "USB Status:"
        system_profiler SPUSBDataType 2>/dev/null | grep -A10 "AG06" | head -15
        echo ""
        
        echo "Audio Status:"
        system_profiler SPAudioDataType 2>/dev/null | grep -A10 "AG06" | head -15
        echo ""
        
        echo "Python Environment:"
        python3 --version
        python3 -m pip list 2>/dev/null | grep -E "(pyaudio|mido|numpy)" | head -10
        echo ""
        
        echo "Process Status:"
        ps aux | grep -E "(ag06|mixer)" | grep -v grep | head -5
        echo ""
        
        echo "Recent Errors (from session log):"
        [ -f "$SESSION_LOG" ] && grep -i "error" "$SESSION_LOG" | tail -10
        
    } > "$debug_file" 2>&1
    
    echo -e "${GREEN}✅ Debug saved to: $debug_file${NC}"
    echo "   View with: cat $debug_file | less"
}

# Git operations with safety
git_safe() {
    echo -e "${BLUE}📝 Git Operations${NC}"
    
    cd "$AG06_PROJECT" || return 1
    
    # Show status (limited output)
    echo "Git Status:"
    git status -s 2>/dev/null | head -20
    
    echo ""
    echo -n "Commit changes? [y/N]: "
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo -n "Commit message: "
        read -r message
        
        git add -A
        git commit -m "AG06: $message" 2>&1 | tail -10
        echo -e "${GREEN}✅ Changes committed${NC}"
    fi
}

# Clean up session
cleanup_session() {
    echo -e "${BLUE}🧹 Cleaning up session${NC}"
    
    # Kill any runaway processes
    pkill -f "ag06_dev_monitor" 2>/dev/null
    
    # Clean old logs (keep last 10)
    if [ -d "$AG06_PROJECT/logs" ]; then
        ls -t "$AG06_PROJECT/logs"/*.log 2>/dev/null | tail -n +11 | xargs rm -f 2>/dev/null
        echo -e "${GREEN}✅ Old logs cleaned${NC}"
    fi
    
    # Save session summary
    if [ -f "$SESSION_LOG" ]; then
        echo "" >> "$SESSION_LOG"
        echo "Session ended at $(date)" >> "$SESSION_LOG"
        echo -e "${GREEN}✅ Session log saved${NC}"
    fi
}

# ============================================
# MAIN WORKFLOW MENU
# ============================================

show_menu() {
    echo -e """
${BLUE}╔══════════════════════════════════════════════╗
║        AG06 MIXER DEVELOPMENT WORKFLOW       ║
╚══════════════════════════════════════════════╝${NC}

1) Initialize Session
2) Run Tests
3) Start Mixer App
4) Open Monitor
5) Save Debug Info
6) Git Operations
7) Audio/MIDI Test Menu
8) Clean Session
9) Quick Status

h) Help
q) Quit

Select option: """
}

# Main workflow loop
main_workflow() {
    # Auto-initialize on start
    init_session
    start_development
    
    while true; do
        show_menu
        read -r option
        
        case $option in
            1) init_session ;;
            2) run_tests ;;
            3) run_app ;;
            4) open_monitor ;;
            5) save_debug ;;
            6) git_safe ;;
            7) source "$AG06_PROJECT/midi_audio_safe_env.sh" && ag06_safe_menu ;;
            8) cleanup_session ;;
            9) python3 "$AG06_PROJECT/ag06_dev_monitor.py" --status ;;
            h) echo -e """
${BLUE}Help:${NC}
  This workflow provides safe, context-limited development
  for the AG06 Mixer app. All commands limit output to
  prevent terminal overflow.
  
  Start with option 1 to initialize, then 2 to test.
  Use option 3 to run your app with automatic output limiting.
  
  Session logs are saved in logs/ directory.
""" ;;
            q) cleanup_session
               echo -e "${GREEN}👋 Goodbye!${NC}"
               break ;;
            *) echo -e "${RED}Invalid option${NC}" ;;
        esac
        
        echo ""
    done
}

# ============================================
# ENTRY POINT
# ============================================

# Handle command line arguments
case "${1:-}" in
    init)
        init_session
        ;;
    test)
        run_tests
        ;;
    run)
        run_app
        ;;
    monitor)
        open_monitor
        ;;
    debug)
        save_debug
        ;;
    help)
        echo """
AG06 Mixer Development Workflow

Usage:
    ./ag06_dev_workflow.sh          # Interactive menu
    ./ag06_dev_workflow.sh init     # Initialize session
    ./ag06_dev_workflow.sh test     # Run tests
    ./ag06_dev_workflow.sh run      # Start mixer app
    ./ag06_dev_workflow.sh monitor  # Open device monitor
    ./ag06_dev_workflow.sh debug    # Save debug info
    ./ag06_dev_workflow.sh help     # Show this help

This workflow ensures safe, context-limited development
to prevent terminal overflow and system issues.
"""
        ;;
    *)
        main_workflow
        ;;
esac