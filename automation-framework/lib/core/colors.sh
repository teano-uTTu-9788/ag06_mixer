#!/usr/bin/env bash
# Color definitions for terminal output
# Following industry best practices for accessible color schemes

# Prevent multiple sourcing
[[ -n "${_COLORS_SH_LOADED:-}" ]] && return 0
readonly _COLORS_SH_LOADED=1

# Check if output supports colors
if [[ -t 1 ]] && command -v tput >/dev/null 2>&1; then
  # Use tput for better compatibility
  readonly COLORS_SUPPORTED=1
  
  # Text formatting
  readonly RESET="$(tput sgr0)"
  readonly BOLD="$(tput bold)"
  readonly DIM="$(tput dim)"
  readonly ITALIC="$(tput sitm 2>/dev/null || echo '')"
  readonly UNDERLINE="$(tput smul)"
  readonly BLINK="$(tput blink)"
  readonly REVERSE="$(tput rev)"
  readonly HIDDEN="$(tput invis)"
  
  # Regular colors
  readonly BLACK="$(tput setaf 0)"
  readonly RED="$(tput setaf 1)"
  readonly GREEN="$(tput setaf 2)"
  readonly YELLOW="$(tput setaf 3)"
  readonly BLUE="$(tput setaf 4)"
  readonly MAGENTA="$(tput setaf 5)"
  readonly CYAN="$(tput setaf 6)"
  readonly WHITE="$(tput setaf 7)"
  readonly GRAY="$(tput setaf 8)"
  
  # Bright colors
  readonly BRIGHT_RED="$(tput setaf 9)"
  readonly BRIGHT_GREEN="$(tput setaf 10)"
  readonly BRIGHT_YELLOW="$(tput setaf 11)"
  readonly BRIGHT_BLUE="$(tput setaf 12)"
  readonly BRIGHT_MAGENTA="$(tput setaf 13)"
  readonly BRIGHT_CYAN="$(tput setaf 14)"
  readonly BRIGHT_WHITE="$(tput setaf 15)"
  
  # Background colors
  readonly BG_BLACK="$(tput setab 0)"
  readonly BG_RED="$(tput setab 1)"
  readonly BG_GREEN="$(tput setab 2)"
  readonly BG_YELLOW="$(tput setab 3)"
  readonly BG_BLUE="$(tput setab 4)"
  readonly BG_MAGENTA="$(tput setab 5)"
  readonly BG_CYAN="$(tput setab 6)"
  readonly BG_WHITE="$(tput setab 7)"
  
  # Semantic colors
  readonly COLOR_SUCCESS="${GREEN}"
  readonly COLOR_ERROR="${RED}"
  readonly COLOR_WARNING="${YELLOW}"
  readonly COLOR_INFO="${BLUE}"
  readonly COLOR_DEBUG="${CYAN}"
  readonly COLOR_MUTED="${GRAY}"
  
else
  # No color support - define empty strings
  readonly COLORS_SUPPORTED=0
  
  # Text formatting
  readonly RESET=""
  readonly BOLD=""
  readonly DIM=""
  readonly ITALIC=""
  readonly UNDERLINE=""
  readonly BLINK=""
  readonly REVERSE=""
  readonly HIDDEN=""
  
  # Regular colors
  readonly BLACK=""
  readonly RED=""
  readonly GREEN=""
  readonly YELLOW=""
  readonly BLUE=""
  readonly MAGENTA=""
  readonly CYAN=""
  readonly WHITE=""
  readonly GRAY=""
  
  # Bright colors
  readonly BRIGHT_RED=""
  readonly BRIGHT_GREEN=""
  readonly BRIGHT_YELLOW=""
  readonly BRIGHT_BLUE=""
  readonly BRIGHT_MAGENTA=""
  readonly BRIGHT_CYAN=""
  readonly BRIGHT_WHITE=""
  
  # Background colors
  readonly BG_BLACK=""
  readonly BG_RED=""
  readonly BG_GREEN=""
  readonly BG_YELLOW=""
  readonly BG_BLUE=""
  readonly BG_MAGENTA=""
  readonly BG_CYAN=""
  readonly BG_WHITE=""
  
  # Semantic colors
  readonly COLOR_SUCCESS=""
  readonly COLOR_ERROR=""
  readonly COLOR_WARNING=""
  readonly COLOR_INFO=""
  readonly COLOR_DEBUG=""
  readonly COLOR_MUTED=""
fi

# Helper functions

# Print colored text
print_color() {
  local color="$1"
  shift
  echo -e "${color}$*${RESET}"
}

# Print with semantic colors
print_success() {
  print_color "${COLOR_SUCCESS}" "$@"
}

print_error() {
  print_color "${COLOR_ERROR}" "$@" >&2
}

print_warning() {
  print_color "${COLOR_WARNING}" "$@"
}

print_info() {
  print_color "${COLOR_INFO}" "$@"
}

print_debug() {
  print_color "${COLOR_DEBUG}" "$@"
}

print_muted() {
  print_color "${COLOR_MUTED}" "$@"
}

# Colorize output from commands
colorize_output() {
  local pattern="$1"
  local color="$2"
  sed -E "s/${pattern}/${color}&${RESET}/g"
}

# Strip color codes from text
strip_colors() {
  sed -E "s/\x1B\[[0-9;]*[mK]//g"
}

# Export all color variables and functions
export RESET BOLD DIM ITALIC UNDERLINE BLINK REVERSE HIDDEN
export BLACK RED GREEN YELLOW BLUE MAGENTA CYAN WHITE GRAY
export BRIGHT_RED BRIGHT_GREEN BRIGHT_YELLOW BRIGHT_BLUE
export BRIGHT_MAGENTA BRIGHT_CYAN BRIGHT_WHITE
export BG_BLACK BG_RED BG_GREEN BG_YELLOW BG_BLUE BG_MAGENTA BG_CYAN BG_WHITE
export COLOR_SUCCESS COLOR_ERROR COLOR_WARNING COLOR_INFO COLOR_DEBUG COLOR_MUTED
export COLORS_SUPPORTED

export -f print_color
export -f print_success
export -f print_error
export -f print_warning
export -f print_info
export -f print_debug
export -f print_muted
export -f colorize_output
export -f strip_colors