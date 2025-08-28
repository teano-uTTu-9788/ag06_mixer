#!/bin/bash
# Terminal color definitions - Google style

# Check if terminal supports colors
if [[ -t 1 ]] && [[ "$(tput colors 2>/dev/null)" -ge 8 ]]; then
    # Regular colors
    export BLACK='\033[0;30m'
    export RED='\033[0;31m'
    export GREEN='\033[0;32m'
    export YELLOW='\033[0;33m'
    export BLUE='\033[0;34m'
    export PURPLE='\033[0;35m'
    export CYAN='\033[0;36m'
    export WHITE='\033[0;37m'
    
    # Bold colors
    export BOLD_BLACK='\033[1;30m'
    export BOLD_RED='\033[1;31m'
    export BOLD_GREEN='\033[1;32m'
    export BOLD_YELLOW='\033[1;33m'
    export BOLD_BLUE='\033[1;34m'
    export BOLD_PURPLE='\033[1;35m'
    export BOLD_CYAN='\033[1;36m'
    export BOLD_WHITE='\033[1;37m'
    
    # Background colors
    export BG_BLACK='\033[40m'
    export BG_RED='\033[41m'
    export BG_GREEN='\033[42m'
    export BG_YELLOW='\033[43m'
    export BG_BLUE='\033[44m'
    export BG_PURPLE='\033[45m'
    export BG_CYAN='\033[46m'
    export BG_WHITE='\033[47m'
    
    # Text formatting
    export BOLD='\033[1m'
    export DIM='\033[2m'
    export ITALIC='\033[3m'
    export UNDERLINE='\033[4m'
    export BLINK='\033[5m'
    export REVERSE='\033[7m'
    export STRIKETHROUGH='\033[9m'
    
    # Reset
    export RESET='\033[0m'
    
    # Semantic colors for logging
    export COLOR_DEBUG="$DIM"
    export COLOR_INFO="$BLUE"
    export COLOR_WARN="$YELLOW"
    export COLOR_ERROR="$RED"
    export COLOR_FATAL="$BOLD_RED"
    export COLOR_SUCCESS="$GREEN"
    
else
    # No color support - set empty values
    export BLACK=''
    export RED=''
    export GREEN=''
    export YELLOW=''
    export BLUE=''
    export PURPLE=''
    export CYAN=''
    export WHITE=''
    export BOLD_BLACK=''
    export BOLD_RED=''
    export BOLD_GREEN=''
    export BOLD_YELLOW=''
    export BOLD_BLUE=''
    export BOLD_PURPLE=''
    export BOLD_CYAN=''
    export BOLD_WHITE=''
    export BG_BLACK=''
    export BG_RED=''
    export BG_GREEN=''
    export BG_YELLOW=''
    export BG_BLUE=''
    export BG_PURPLE=''
    export BG_CYAN=''
    export BG_WHITE=''
    export BOLD=''
    export DIM=''
    export ITALIC=''
    export UNDERLINE=''
    export BLINK=''
    export REVERSE=''
    export STRIKETHROUGH=''
    export RESET=''
    export COLOR_DEBUG=''
    export COLOR_INFO=''
    export COLOR_WARN=''
    export COLOR_ERROR=''
    export COLOR_FATAL=''
    export COLOR_SUCCESS=''
fi

# ============================================================================
# Color Utility Functions
# ============================================================================

colors::print() {
    local color="$1"
    shift
    echo -e "${color}$*${RESET}"
}

colors::red() {
    colors::print "$RED" "$@"
}

colors::green() {
    colors::print "$GREEN" "$@"
}

colors::yellow() {
    colors::print "$YELLOW" "$@"
}

colors::blue() {
    colors::print "$BLUE" "$@"
}

colors::purple() {
    colors::print "$PURPLE" "$@"
}

colors::cyan() {
    colors::print "$CYAN" "$@"
}

colors::bold() {
    colors::print "$BOLD" "$@"
}

colors::dim() {
    colors::print "$DIM" "$@"
}

# ============================================================================
# Color Test Function
# ============================================================================

colors::test() {
    echo "Color Test - Terminal Automation Framework"
    echo ""
    
    echo -e "${RED}■${RESET} Red"
    echo -e "${GREEN}■${RESET} Green"  
    echo -e "${YELLOW}■${RESET} Yellow"
    echo -e "${BLUE}■${RESET} Blue"
    echo -e "${PURPLE}■${RESET} Purple"
    echo -e "${CYAN}■${RESET} Cyan"
    echo -e "${WHITE}■${RESET} White"
    
    echo ""
    echo -e "${BOLD}Bold text${RESET}"
    echo -e "${DIM}Dim text${RESET}"
    echo -e "${ITALIC}Italic text${RESET}"
    echo -e "${UNDERLINE}Underlined text${RESET}"
    
    echo ""
    echo "Log level colors:"
    echo -e "${COLOR_DEBUG}[DEBUG] Debug message${RESET}"
    echo -e "${COLOR_INFO}[INFO] Info message${RESET}"
    echo -e "${COLOR_WARN}[WARN] Warning message${RESET}"
    echo -e "${COLOR_ERROR}[ERROR] Error message${RESET}"
    echo -e "${COLOR_FATAL}[FATAL] Fatal message${RESET}"
    echo -e "${COLOR_SUCCESS}[SUCCESS] Success message${RESET}"
}