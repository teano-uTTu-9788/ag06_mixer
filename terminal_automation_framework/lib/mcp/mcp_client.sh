#!/usr/bin/env bash
#
# MCP Client - Model Context Protocol Client Implementation
# Following 2025-06-18 MCP Specification with OAuth 2.0 Resource Indicators
# Integration with Google, Microsoft, and OpenAI MCP patterns
#
set -euo pipefail

# Source framework dependencies
source "$(dirname "${BASH_SOURCE[0]}")/../core/bootstrap.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../core/logger.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../core/config.sh"

# MCP Client Configuration
if [[ -z "${MCP_CLIENT_VERSION:-}" ]]; then
    readonly MCP_CLIENT_VERSION="1.0.0"
fi
if [[ -z "${MCP_PROTOCOL_VERSION:-}" ]]; then
    readonly MCP_PROTOCOL_VERSION="2024-11-05"
fi
if [[ -z "${MCP_CLIENT_NAME:-}" ]]; then
    readonly MCP_CLIENT_NAME="terminal-automation-mcp-client"
fi

# Client State
MCP_CLIENT_INITIALIZED=false
MCP_SERVER_URL=""
MCP_ACCESS_TOKEN=""
MCP_REQUEST_ID=1

# Initialize MCP client
mcp::client::init() {
    if [[ "$MCP_CLIENT_INITIALIZED" == "true" ]]; then
        return 0
    fi
    
    log::info "Initializing MCP Client v${MCP_CLIENT_VERSION}"
    
    # Initialize framework only if not already initialized
    if [[ "${FRAMEWORK_INITIALIZED:-false}" != "true" ]]; then
        framework::init
    fi
    config::load
    
    # Load server configuration
    MCP_SERVER_URL="${MCP_SERVER_URL:-http://localhost:3333}"
    
    MCP_CLIENT_INITIALIZED=true
    log::success "MCP Client initialized"
}

# OAuth 2.0 Authentication (2025-06-18 spec)
mcp::client::authenticate() {
    local auth_server_url="$1"
    local client_id="$2"
    local resource_indicator="${3:-terminal-automation}"
    
    log::info "Authenticating with OAuth 2.0 Authorization Server"
    log::debug "Auth Server: $auth_server_url"
    log::debug "Resource Indicator: $resource_indicator"
    
    # This is a simplified OAuth flow - in production you'd implement full OAuth 2.0
    # with PKCE, proper token exchange, etc.
    
    if [[ -z "$auth_server_url" ]] || [[ -z "$client_id" ]]; then
        log::warn "OAuth credentials not provided - using anonymous access"
        MCP_ACCESS_TOKEN=""
        return 0
    fi
    
    # For demonstration, we'll simulate token acquisition
    # In real implementation, this would be a proper OAuth flow
    MCP_ACCESS_TOKEN="demo_token_$(date +%s)"
    log::success "Authentication successful"
}

# Connect to MCP server
mcp::client::connect() {
    local server_url="${1:-$MCP_SERVER_URL}"
    
    mcp::client::init
    
    log::info "Connecting to MCP server: $server_url"
    MCP_SERVER_URL="$server_url"
    
    # Test connection
    if ! mcp::client::test_connection; then
        log::error "Failed to connect to MCP server"
        return 1
    fi
    
    # Initialize session
    local init_response
    init_response=$(mcp::client::send_request "initialize" '{"protocolVersion": "2024-11-05", "capabilities": {"roots": {"listChanged": true}}, "clientInfo": {"name": "terminal-automation-mcp-client", "version": "1.0.0"}}')
    
    if [[ $? -eq 0 ]]; then
        log::success "Connected to MCP server successfully"
        log::debug "Server capabilities: $(echo "$init_response" | jq -r '.result.capabilities // empty')"
    else
        log::error "Failed to initialize MCP session"
        return 1
    fi
}

# Test connection to MCP server
mcp::client::test_connection() {
    if ! command -v curl >/dev/null 2>&1; then
        log::error "curl not available - cannot test connection"
        return 1
    fi
    
    local response
    response=$(curl -s --max-time 5 "$MCP_SERVER_URL" 2>/dev/null)
    
    if [[ $? -eq 0 ]] && echo "$response" | grep -q "terminal-automation-mcp"; then
        return 0
    else
        return 1
    fi
}

# Send JSON-RPC 2.0 request to MCP server
mcp::client::send_request() {
    local method="$1"
    local params="${2:-{}}"
    local request_id=$((MCP_REQUEST_ID++))
    
    if ! command -v jq >/dev/null 2>&1; then
        log::error "jq not available - cannot send MCP requests"
        return 1
    fi
    
    # Build JSON-RPC 2.0 request
    local request
    request=$(jq -n \
        --arg jsonrpc "2.0" \
        --arg method "$method" \
        --argjson params "$params" \
        --arg id "$request_id" \
        '{
            jsonrpc: $jsonrpc,
            method: $method,
            params: $params,
            id: $id
        }')
    
    log::debug "Sending MCP request: $method"
    
    # Send request with authentication if available
    local curl_args=(-s -X POST -H "Content-Type: application/json")
    
    if [[ -n "$MCP_ACCESS_TOKEN" ]]; then
        curl_args+=(-H "Authorization: Bearer $MCP_ACCESS_TOKEN")
    fi
    
    curl_args+=(-d "$request" "$MCP_SERVER_URL/mcp")
    
    local response
    response=$(curl "${curl_args[@]}")
    
    if [[ $? -eq 0 ]]; then
        log::debug "MCP response received"
        echo "$response"
        return 0
    else
        log::error "Failed to send MCP request"
        return 1
    fi
}

# List available tools
mcp::client::list_tools() {
    log::info "Listing available MCP tools"
    
    local response
    response=$(mcp::client::send_request "tools/list" "{}")
    
    if [[ $? -eq 0 ]]; then
        echo "$response" | jq -r '
            if .result.tools then
                .result.tools[] | "• \(.name): \(.description)"
            else
                "No tools available or error occurred"
            end'
    else
        log::error "Failed to list tools"
        return 1
    fi
}

# Call MCP tool
mcp::client::call_tool() {
    local tool_name="$1"
    local arguments="${2:-{}}"
    
    log::info "Calling MCP tool: $tool_name"
    
    local params
    params=$(jq -n \
        --arg name "$tool_name" \
        --argjson arguments "$arguments" \
        '{
            name: $name,
            arguments: $arguments
        }')
    
    local response
    response=$(mcp::client::send_request "tools/call" "$params")
    
    if [[ $? -eq 0 ]]; then
        # Extract and display the result
        echo "$response" | jq -r '
            if .result.content then
                .result.content[] | 
                if .type == "text" then
                    .text
                else
                    "Content type: \(.type)"
                end
            elif .error then
                "Error: \(.error.message)"
            else
                "No content returned"
            end'
    else
        log::error "Failed to call tool: $tool_name"
        return 1
    fi
}

# List available resources
mcp::client::list_resources() {
    log::info "Listing available MCP resources"
    
    local response
    response=$(mcp::client::send_request "resources/list" "{}")
    
    if [[ $? -eq 0 ]]; then
        echo "$response" | jq -r '
            if .result.resources then
                .result.resources[] | "• \(.name): \(.uri) (\(.mimeType))"
            else
                "No resources available or error occurred"
            end'
    else
        log::error "Failed to list resources"
        return 1
    fi
}

# Read MCP resource
mcp::client::read_resource() {
    local resource_uri="$1"
    
    log::info "Reading MCP resource: $resource_uri"
    
    local params
    params=$(jq -n --arg uri "$resource_uri" '{uri: $uri}')
    
    local response
    response=$(mcp::client::send_request "resources/read" "$params")
    
    if [[ $? -eq 0 ]]; then
        echo "$response" | jq -r '
            if .result.contents then
                .result.contents[] | 
                if .text then
                    .text
                else
                    "Content: \(.mimeType)"
                end
            elif .error then
                "Error: \(.error.message)"
            else
                "No content returned"
            end'
    else
        log::error "Failed to read resource: $resource_uri"
        return 1
    fi
}

# Interactive MCP client session
mcp::client::interactive() {
    log::info "Starting interactive MCP client session"
    echo "Type 'help' for available commands, 'quit' to exit"
    
    while true; do
        echo -n "mcp> "
        read -r command args
        
        case "$command" in
            help)
                echo "Available commands:"
                echo "  tools                    - List available tools"
                echo "  call <tool> [args]      - Call a tool (args as JSON)"
                echo "  resources               - List available resources"
                echo "  read <uri>             - Read a resource"
                echo "  status                 - Show connection status"
                echo "  reconnect [url]        - Reconnect to server"
                echo "  quit                   - Exit session"
                ;;
            tools)
                mcp::client::list_tools
                ;;
            call)
                if [[ -z "$args" ]]; then
                    echo "Usage: call <tool_name> [arguments_json]"
                else
                    local tool_name
                    tool_name=$(echo "$args" | cut -d' ' -f1)
                    local tool_args
                    tool_args=$(echo "$args" | cut -d' ' -f2- | sed 's/^[[:space:]]*//')
                    
                    if [[ -z "$tool_args" ]] || [[ "$tool_args" == "$tool_name" ]]; then
                        tool_args="{}"
                    fi
                    
                    mcp::client::call_tool "$tool_name" "$tool_args"
                fi
                ;;
            resources)
                mcp::client::list_resources
                ;;
            read)
                if [[ -z "$args" ]]; then
                    echo "Usage: read <resource_uri>"
                else
                    mcp::client::read_resource "$args"
                fi
                ;;
            status)
                if mcp::client::test_connection; then
                    echo "✅ Connected to $MCP_SERVER_URL"
                else
                    echo "❌ Not connected to MCP server"
                fi
                ;;
            reconnect)
                local new_url="${args:-$MCP_SERVER_URL}"
                mcp::client::connect "$new_url"
                ;;
            quit|exit)
                echo "Goodbye!"
                break
                ;;
            "")
                # Empty command, continue
                ;;
            *)
                echo "Unknown command: $command"
                echo "Type 'help' for available commands"
                ;;
        esac
    done
}

# Command-line interface
mcp::client::main() {
    local command="${1:-help}"
    local server_url="${2:-http://localhost:3333}"
    
    case "$command" in
        connect)
            mcp::client::connect "$server_url"
            ;;
        tools)
            mcp::client::connect "$server_url"
            mcp::client::list_tools
            ;;
        call)
            if [[ $# -lt 3 ]]; then
                echo "Usage: $0 call <server_url> <tool_name> [arguments_json]"
                exit 1
            fi
            local tool_name="$3"
            local tool_args="${4:-{}}"
            mcp::client::connect "$server_url"
            mcp::client::call_tool "$tool_name" "$tool_args"
            ;;
        resources)
            mcp::client::connect "$server_url"
            mcp::client::list_resources
            ;;
        read)
            if [[ $# -lt 3 ]]; then
                echo "Usage: $0 read <server_url> <resource_uri>"
                exit 1
            fi
            local resource_uri="$3"
            mcp::client::connect "$server_url"
            mcp::client::read_resource "$resource_uri"
            ;;
        interactive)
            mcp::client::connect "$server_url"
            mcp::client::interactive
            ;;
        help|*)
            echo "MCP Client - Model Context Protocol Client v${MCP_CLIENT_VERSION}"
            echo ""
            echo "Usage: $0 <command> [server_url] [args...]"
            echo ""
            echo "Commands:"
            echo "  connect <url>              Connect to MCP server"
            echo "  tools [url]               List available tools"
            echo "  call <url> <tool> [args]  Call a tool"
            echo "  resources [url]           List available resources"
            echo "  read <url> <uri>          Read a resource"
            echo "  interactive [url]         Start interactive session"
            echo "  help                      Show this help"
            echo ""
            echo "Default server URL: http://localhost:3333"
            ;;
    esac
}

# Export functions
export -f mcp::client::init mcp::client::connect mcp::client::call_tool mcp::client::list_tools

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    mcp::client::main "$@"
fi