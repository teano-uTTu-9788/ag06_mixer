#!/usr/bin/env bash
#
# MCP Tools - Tool registry and management for Model Context Protocol
# Following Google Cloud Functions and Microsoft Azure Functions patterns
#
set -euo pipefail

# Source framework dependencies
source "$(dirname "${BASH_SOURCE[0]}")/../core/bootstrap.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../core/logger.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../core/config.sh"

# Tool registry storage
readonly MCP_TOOLS_REGISTRY="/tmp/mcp_tools_registry.json"

# Initialize tools registry
mcp::tools::init() {
    log::debug "Initializing MCP tools registry"
    
    if [[ ! -f "$MCP_TOOLS_REGISTRY" ]]; then
        cat > "$MCP_TOOLS_REGISTRY" << 'EOF'
{
  "tools": [],
  "version": "1.0.0",
  "last_updated": ""
}
EOF
    fi
}

# Register a new tool
mcp::tools::register() {
    local tool_name="$1"
    local tool_description="$2"
    local input_schema="$3"
    local handler_script="$4"
    
    mcp::tools::init
    
    log::info "Registering MCP tool: $tool_name"
    
    # Validate required parameters
    if [[ -z "$tool_name" ]] || [[ -z "$tool_description" ]] || [[ -z "$input_schema" ]]; then
        log::error "Missing required parameters for tool registration"
        return 1
    fi
    
    # Create tool definition
    local tool_def
    tool_def=$(jq -n \
        --arg name "$tool_name" \
        --arg description "$tool_description" \
        --argjson inputSchema "$input_schema" \
        --arg handler "$handler_script" \
        --arg created "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        '{
            name: $name,
            description: $description,
            inputSchema: $inputSchema,
            handler: $handler,
            created: $created,
            enabled: true
        }')
    
    # Add to registry
    local updated_registry
    updated_registry=$(jq --argjson tool "$tool_def" \
        '.tools += [$tool] | .last_updated = now | .last_updated |= strftime("%Y-%m-%dT%H:%M:%SZ")' \
        "$MCP_TOOLS_REGISTRY")
    
    echo "$updated_registry" > "$MCP_TOOLS_REGISTRY"
    
    log::success "Tool '$tool_name' registered successfully"
}

# Unregister a tool
mcp::tools::unregister() {
    local tool_name="$1"
    
    mcp::tools::init
    
    log::info "Unregistering MCP tool: $tool_name"
    
    local updated_registry
    updated_registry=$(jq --arg name "$tool_name" \
        '.tools = (.tools | map(select(.name != $name))) | .last_updated = now | .last_updated |= strftime("%Y-%m-%dT%H:%M:%SZ")' \
        "$MCP_TOOLS_REGISTRY")
    
    echo "$updated_registry" > "$MCP_TOOLS_REGISTRY"
    
    log::success "Tool '$tool_name' unregistered"
}

# List all registered tools
mcp::tools::list() {
    mcp::tools::init
    
    log::info "Listing registered MCP tools"
    
    if ! jq -e '.tools | length > 0' "$MCP_TOOLS_REGISTRY" >/dev/null; then
        echo "No tools registered"
        return 0
    fi
    
    echo "Registered MCP Tools:"
    echo "===================="
    
    jq -r '.tools[] | 
        "• \(.name)" + 
        (if .enabled then " ✅" else " ❌" end) + 
        "\n  Description: \(.description)" + 
        "\n  Handler: \(.handler // "built-in")" + 
        "\n  Created: \(.created)\n"' \
        "$MCP_TOOLS_REGISTRY"
}

# Get tool information
mcp::tools::get() {
    local tool_name="$1"
    
    mcp::tools::init
    
    local tool_info
    tool_info=$(jq -r --arg name "$tool_name" '.tools[] | select(.name == $name)' "$MCP_TOOLS_REGISTRY")
    
    if [[ -z "$tool_info" ]] || [[ "$tool_info" == "null" ]]; then
        log::error "Tool not found: $tool_name"
        return 1
    fi
    
    echo "$tool_info"
}

# Enable/disable tool
mcp::tools::toggle() {
    local tool_name="$1"
    local enabled="${2:-toggle}"
    
    mcp::tools::init
    
    local current_state
    current_state=$(jq -r --arg name "$tool_name" '.tools[] | select(.name == $name) | .enabled' "$MCP_TOOLS_REGISTRY")
    
    if [[ -z "$current_state" ]] || [[ "$current_state" == "null" ]]; then
        log::error "Tool not found: $tool_name"
        return 1
    fi
    
    local new_state
    case "$enabled" in
        true|enable|on)
            new_state="true"
            ;;
        false|disable|off)
            new_state="false"
            ;;
        toggle|*)
            if [[ "$current_state" == "true" ]]; then
                new_state="false"
            else
                new_state="true"
            fi
            ;;
    esac
    
    local updated_registry
    updated_registry=$(jq --arg name "$tool_name" --argjson enabled "$new_state" \
        '(.tools[] | select(.name == $name) | .enabled) = $enabled | .last_updated = now | .last_updated |= strftime("%Y-%m-%dT%H:%M:%SZ")' \
        "$MCP_TOOLS_REGISTRY")
    
    echo "$updated_registry" > "$MCP_TOOLS_REGISTRY"
    
    local action
    if [[ "$new_state" == "true" ]]; then
        action="enabled"
    else
        action="disabled"
    fi
    
    log::success "Tool '$tool_name' $action"
}

# Execute tool handler
mcp::tools::execute() {
    local tool_name="$1"
    local arguments="${2:-{}}"
    
    log::debug "Executing MCP tool: $tool_name"
    
    # Get tool information
    local tool_info
    tool_info=$(mcp::tools::get "$tool_name")
    if [[ $? -ne 0 ]]; then
        return 1
    fi
    
    # Check if tool is enabled
    local enabled
    enabled=$(echo "$tool_info" | jq -r '.enabled')
    if [[ "$enabled" != "true" ]]; then
        log::error "Tool '$tool_name' is disabled"
        return 1
    fi
    
    # Get handler script
    local handler_script
    handler_script=$(echo "$tool_info" | jq -r '.handler // empty')
    
    if [[ -n "$handler_script" ]] && [[ -f "$handler_script" ]]; then
        # External handler script
        log::debug "Executing external handler: $handler_script"
        bash "$handler_script" "$tool_name" "$arguments"
    else
        # Built-in handler
        mcp::tools::execute_builtin "$tool_name" "$arguments"
    fi
}

# Execute built-in tool handlers
mcp::tools::execute_builtin() {
    local tool_name="$1"
    local arguments="$2"
    
    case "$tool_name" in
        "run_command")
            mcp::tools::builtin::run_command "$arguments"
            ;;
        "validate_input")
            mcp::tools::builtin::validate_input "$arguments"
            ;;
        "framework_doctor")
            mcp::tools::builtin::framework_doctor "$arguments"
            ;;
        "create_project")
            mcp::tools::builtin::create_project "$arguments"
            ;;
        "analyze_logs")
            mcp::tools::builtin::analyze_logs "$arguments"
            ;;
        *)
            log::error "Unknown built-in tool: $tool_name"
            return 1
            ;;
    esac
}

# Built-in tool: Run command
mcp::tools::builtin::run_command() {
    local args="$1"
    
    local command
    command=$(echo "$args" | jq -r '.command // empty')
    local working_dir
    working_dir=$(echo "$args" | jq -r '.working_dir // empty')
    
    if [[ -z "$command" ]]; then
        echo "Error: Command parameter is required"
        return 1
    fi
    
    # Security: Basic command validation
    local dangerous_commands=("rm -rf /" "dd if=" "mkfs" "fdisk" "parted" "sudo rm" "sudo dd")
    for dangerous in "${dangerous_commands[@]}"; do
        if [[ "$command" == *"$dangerous"* ]]; then
            echo "Error: Dangerous command blocked: $command"
            return 1
        fi
    done
    
    # Execute command
    local result
    if [[ -n "$working_dir" ]] && [[ -d "$working_dir" ]]; then
        result=$(cd "$working_dir" && bash -c "$command" 2>&1)
    else
        result=$(bash -c "$command" 2>&1)
    fi
    
    local exit_code=$?
    
    cat << EOF
Command: $command
Exit Code: $exit_code
Output:
$result
EOF
}

# Built-in tool: Validate input
mcp::tools::builtin::validate_input() {
    local args="$1"
    
    local input_value
    input_value=$(echo "$args" | jq -r '.input // empty')
    local validation_type
    validation_type=$(echo "$args" | jq -r '.type // empty')
    
    if [[ -z "$input_value" ]] || [[ -z "$validation_type" ]]; then
        echo "Error: Both 'input' and 'type' parameters are required"
        return 1
    fi
    
    # Load validation functions
    source "$(dirname "${BASH_SOURCE[0]}")/../utils/validation.sh"
    
    local validation_result="❌ Invalid"
    case "$validation_type" in
        email)
            if validate::email "$input_value" 2>/dev/null; then
                validation_result="✅ Valid"
            fi
            ;;
        url)
            if validate::url "$input_value" 2>/dev/null; then
                validation_result="✅ Valid"
            fi
            ;;
        hostname)
            if validate::hostname "$input_value" 2>/dev/null; then
                validation_result="✅ Valid"
            fi
            ;;
        path)
            if validate::safe_path "$input_value" 2>/dev/null; then
                validation_result="✅ Valid"
            fi
            ;;
        *)
            echo "Error: Unsupported validation type: $validation_type"
            return 1
            ;;
    esac
    
    cat << EOF
Input: $input_value
Type: $validation_type
Result: $validation_result
EOF
}

# Built-in tool: Framework doctor
mcp::tools::builtin::framework_doctor() {
    local args="$1"
    
    # Run framework health check
    local result
    result=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && ./dev doctor 2>&1)
    
    cat << EOF
Framework Health Check Results:
==============================
$result
EOF
}

# Built-in tool: Create project
mcp::tools::builtin::create_project() {
    local args="$1"
    
    local project_name
    project_name=$(echo "$args" | jq -r '.name // empty')
    local project_type
    project_type=$(echo "$args" | jq -r '.type // "basic"')
    
    if [[ -z "$project_name" ]]; then
        echo "Error: Project name is required"
        return 1
    fi
    
    # Validate project name
    if [[ ! "$project_name" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        echo "Error: Invalid project name. Use only letters, numbers, underscores, and hyphens."
        return 1
    fi
    
    # Create project directory
    if [[ -d "$project_name" ]]; then
        echo "Error: Directory '$project_name' already exists"
        return 1
    fi
    
    mkdir -p "$project_name"
    cd "$project_name"
    
    # Create basic project structure
    case "$project_type" in
        cli)
            mkdir -p bin lib test
            echo "#!/usr/bin/env bash" > bin/"$project_name"
            chmod +x bin/"$project_name"
            ;;
        webapp)
            mkdir -p src public test
            echo "<!DOCTYPE html><html><head><title>$project_name</title></head><body><h1>Welcome to $project_name</h1></body></html>" > public/index.html
            ;;
        *)
            mkdir -p src test
            ;;
    esac
    
    # Create README
    cat > README.md << EOF
# $project_name

Project created with Terminal Automation Framework MCP Tools

## Type
$project_type

## Structure
$(tree . 2>/dev/null || find . -type d | sort)

## Getting Started
Add your project details here...
EOF
    
    cat << EOF
Project '$project_name' created successfully!
Type: $project_type
Location: $(pwd)

Files created:
$(find . -type f | sort)
EOF
}

# Built-in tool: Analyze logs
mcp::tools::builtin::analyze_logs() {
    local args="$1"
    
    local log_file
    log_file=$(echo "$args" | jq -r '.file // "/tmp/framework.log"')
    local lines
    lines=$(echo "$args" | jq -r '.lines // "50"')
    
    if [[ ! -f "$log_file" ]]; then
        echo "Error: Log file not found: $log_file"
        return 1
    fi
    
    echo "Log Analysis for: $log_file"
    echo "================================"
    echo ""
    
    # Basic log analysis
    echo "Recent entries (last $lines lines):"
    tail -n "$lines" "$log_file"
    
    echo ""
    echo "Error summary:"
    grep -i error "$log_file" | tail -5 || echo "No errors found"
    
    echo ""
    echo "Warning summary:"
    grep -i warn "$log_file" | tail -5 || echo "No warnings found"
    
    echo ""
    echo "File info:"
    ls -la "$log_file"
}

# Register built-in tools
mcp::tools::register_builtins() {
    log::info "Registering built-in MCP tools"
    
    # Tool: run_command
    mcp::tools::register \
        "run_command" \
        "Execute shell commands safely in the terminal automation framework" \
        '{"type": "object", "properties": {"command": {"type": "string", "description": "Shell command to execute"}, "working_dir": {"type": "string", "description": "Working directory (optional)"}}, "required": ["command"]}' \
        ""
    
    # Tool: validate_input
    mcp::tools::register \
        "validate_input" \
        "Validate various types of input using framework validators" \
        '{"type": "object", "properties": {"input": {"type": "string", "description": "Input to validate"}, "type": {"type": "string", "enum": ["email", "url", "path", "hostname"], "description": "Validation type"}}, "required": ["input", "type"]}' \
        ""
    
    # Tool: framework_doctor
    mcp::tools::register \
        "framework_doctor" \
        "Run system health checks on the terminal automation framework" \
        '{"type": "object", "properties": {}, "additionalProperties": false}' \
        ""
    
    # Tool: create_project
    mcp::tools::register \
        "create_project" \
        "Create a new project with the terminal automation framework" \
        '{"type": "object", "properties": {"name": {"type": "string", "description": "Project name"}, "type": {"type": "string", "enum": ["basic", "cli", "webapp"], "description": "Project type"}}, "required": ["name"]}' \
        ""
    
    # Tool: analyze_logs
    mcp::tools::register \
        "analyze_logs" \
        "Analyze log files for errors and patterns" \
        '{"type": "object", "properties": {"file": {"type": "string", "description": "Path to log file"}, "lines": {"type": "string", "description": "Number of recent lines to analyze"}}}' \
        ""
    
    log::success "Built-in tools registered successfully"
}

# Main CLI interface
mcp::tools::main() {
    local command="${1:-help}"
    
    case "$command" in
        init)
            mcp::tools::init
            mcp::tools::register_builtins
            ;;
        list)
            mcp::tools::list
            ;;
        register)
            if [[ $# -lt 4 ]]; then
                echo "Usage: $0 register <name> <description> <schema_json> [handler_script]"
                exit 1
            fi
            mcp::tools::register "$2" "$3" "$4" "${5:-}"
            ;;
        unregister)
            if [[ $# -lt 2 ]]; then
                echo "Usage: $0 unregister <name>"
                exit 1
            fi
            mcp::tools::unregister "$2"
            ;;
        get)
            if [[ $# -lt 2 ]]; then
                echo "Usage: $0 get <name>"
                exit 1
            fi
            mcp::tools::get "$2"
            ;;
        enable|disable|toggle)
            if [[ $# -lt 2 ]]; then
                echo "Usage: $0 $command <name>"
                exit 1
            fi
            mcp::tools::toggle "$2" "$command"
            ;;
        execute)
            if [[ $# -lt 2 ]]; then
                echo "Usage: $0 execute <name> [arguments_json]"
                exit 1
            fi
            mcp::tools::execute "$2" "${3:-{}}"
            ;;
        help|*)
            echo "MCP Tools - Model Context Protocol Tool Management"
            echo ""
            echo "Usage: $0 <command> [args...]"
            echo ""
            echo "Commands:"
            echo "  init                           Initialize tools registry and register built-ins"
            echo "  list                          List all registered tools"
            echo "  register <name> <desc> <schema> [handler]  Register a new tool"
            echo "  unregister <name>             Unregister a tool"
            echo "  get <name>                    Get tool information"
            echo "  enable <name>                 Enable a tool"
            echo "  disable <name>                Disable a tool"
            echo "  toggle <name>                 Toggle tool enabled/disabled"
            echo "  execute <name> [args]         Execute a tool"
            echo "  help                          Show this help"
            ;;
    esac
}

# Export functions
export -f mcp::tools::register mcp::tools::execute mcp::tools::list

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    mcp::tools::main "$@"
fi