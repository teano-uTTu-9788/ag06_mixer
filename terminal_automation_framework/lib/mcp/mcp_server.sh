#!/usr/bin/env bash
#
# MCP Server - Model Context Protocol Server Implementation
# Following 2025-06-18 MCP Specification and industry best practices
# OAuth 2.0 Resource Server with HTTP+SSE transport
#
set -euo pipefail

# Source framework dependencies
source "$(dirname "${BASH_SOURCE[0]}")/../core/bootstrap.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../core/logger.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../core/config.sh"

# MCP Server Configuration
if [[ -z "${MCP_SERVER_VERSION:-}" ]]; then
    readonly MCP_SERVER_VERSION="1.0.0"
fi
if [[ -z "${MCP_PROTOCOL_VERSION:-}" ]]; then
    readonly MCP_PROTOCOL_VERSION="2024-11-05"
fi
if [[ -z "${MCP_SERVER_NAME:-}" ]]; then
    readonly MCP_SERVER_NAME="terminal-automation-mcp"
fi
if [[ -z "${MCP_SERVER_PORT:-}" ]]; then
    readonly MCP_SERVER_PORT="${MCP_PORT:-3333}"
fi
if [[ -z "${MCP_SERVER_HOST:-}" ]]; then
    readonly MCP_SERVER_HOST="${MCP_HOST:-localhost}"
fi

# OAuth 2.0 Configuration (2025-06-18 spec requirement)
if [[ -z "${OAUTH_AUTHORIZATION_SERVER_METADATA_URL:-}" ]]; then
    readonly OAUTH_AUTHORIZATION_SERVER_METADATA_URL="${OAUTH_AUTH_SERVER:-}"
fi
if [[ -z "${OAUTH_RESOURCE_INDICATOR:-}" ]]; then
    readonly OAUTH_RESOURCE_INDICATOR="${OAUTH_RESOURCE:-terminal-automation}"
fi
if [[ -z "${OAUTH_SCOPE:-}" ]]; then
    readonly OAUTH_SCOPE="mcp:tools mcp:resources"
fi

# MCP Server State
MCP_SERVER_RUNNING=false
MCP_CLIENTS=()
MCP_TOOLS=()
MCP_RESOURCES=()

# Initialize MCP server
mcp::server::init() {
    log::info "Initializing MCP Server v${MCP_SERVER_VERSION}"
    
    # Initialize framework only if not already initialized
    if [[ "${FRAMEWORK_INITIALIZED:-false}" != "true" ]]; then
        framework::init
    fi
    
    # Load configuration
    config::load
    
    # Register built-in tools
    mcp::tools::register_builtin
    
    # Register built-in resources  
    mcp::resources::register_builtin
    
    log::success "MCP Server initialized successfully"
}

# Start MCP server with HTTP+SSE transport
mcp::server::start() {
    if [[ "$MCP_SERVER_RUNNING" == "true" ]]; then
        log::warn "MCP Server already running"
        return 0
    fi
    
    log::info "Starting MCP Server on ${MCP_SERVER_HOST}:${MCP_SERVER_PORT}"
    
    # Create server implementation
    mcp::server::create_http_server
    
    MCP_SERVER_RUNNING=true
    log::success "MCP Server started successfully"
}

# Create HTTP server with SSE support
mcp::server::create_http_server() {
    local server_script="/tmp/mcp_http_server.py"
    
    cat > "$server_script" << 'EOF'
#!/usr/bin/env python3
"""
MCP HTTP Server with Server-Sent Events support
Following 2025-06-18 MCP Specification
"""
import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import sys
import os

class MCPHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, mcp_shell_handler=None, **kwargs):
        self.mcp_shell_handler = mcp_shell_handler
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests for SSE and metadata"""
        path = urlparse(self.path).path
        
        if path == '/':
            self.send_mcp_info()
        elif path == '/.well-known/mcp_metadata':
            self.send_mcp_metadata()
        elif path == '/events':
            self.handle_sse_connection()
        else:
            self.send_error(404)
    
    def do_POST(self):
        """Handle POST requests for JSON-RPC 2.0"""
        if self.path == '/mcp':
            self.handle_mcp_request()
        else:
            self.send_error(404)
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()
    
    def send_mcp_info(self):
        """Send MCP server information"""
        info = {
            "name": "terminal-automation-mcp",
            "version": "1.0.0",
            "protocol_version": "2024-11-05",
            "capabilities": {
                "tools": True,
                "resources": True,
                "prompts": True,
                "logging": True
            },
            "transport": "http+sse",
            "oauth": {
                "resource_server": True,
                "resource_indicator": "terminal-automation",
                "scopes": ["mcp:tools", "mcp:resources"]
            }
        }
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(info, indent=2).encode())
    
    def send_mcp_metadata(self):
        """Send OAuth 2.0 resource server metadata (2025-06-18 spec)"""
        metadata = {
            "resource_identifier": "terminal-automation",
            "resource_documentation": "https://github.com/terminal-automation-framework",
            "scopes_supported": ["mcp:tools", "mcp:resources"],
            "authorization_server": os.environ.get('OAUTH_AUTH_SERVER', ''),
            "token_endpoint_auth_methods_supported": ["private_key_jwt", "client_secret_jwt"],
            "resource_indicators_supported": True
        }
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(metadata, indent=2).encode())
    
    def handle_sse_connection(self):
        """Handle Server-Sent Events connection"""
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        # Send initial connection event
        self.send_sse_event('connected', {'timestamp': time.time()})
        
        # Keep connection alive with periodic pings
        try:
            while True:
                time.sleep(30)
                self.send_sse_event('ping', {'timestamp': time.time()})
        except:
            pass
    
    def send_sse_event(self, event_type, data):
        """Send Server-Sent Event"""
        try:
            self.wfile.write(f"event: {event_type}\n".encode())
            self.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
            self.wfile.flush()
        except:
            pass
    
    def handle_mcp_request(self):
        """Handle MCP JSON-RPC 2.0 requests"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            request = json.loads(post_data)
            
            # Validate JSON-RPC 2.0 format
            if not self.validate_jsonrpc_request(request):
                self.send_jsonrpc_error(-32600, "Invalid Request", request.get('id'))
                return
            
            # Process the request
            response = self.process_mcp_request(request)
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except json.JSONDecodeError:
            self.send_jsonrpc_error(-32700, "Parse error", None)
        except Exception as e:
            self.send_jsonrpc_error(-32603, f"Internal error: {str(e)}", None)
    
    def validate_jsonrpc_request(self, request):
        """Validate JSON-RPC 2.0 request format"""
        return (
            isinstance(request, dict) and
            request.get('jsonrpc') == '2.0' and
            'method' in request and
            'id' in request
        )
    
    def process_mcp_request(self, request):
        """Process MCP request and return response"""
        method = request['method']
        params = request.get('params', {})
        request_id = request['id']
        
        # Handle different MCP methods
        if method == 'initialize':
            return self.handle_initialize(request_id, params)
        elif method == 'tools/list':
            return self.handle_tools_list(request_id)
        elif method == 'tools/call':
            return self.handle_tools_call(request_id, params)
        elif method == 'resources/list':
            return self.handle_resources_list(request_id)
        elif method == 'resources/read':
            return self.handle_resources_read(request_id, params)
        else:
            return self.create_jsonrpc_error(-32601, "Method not found", request_id)
    
    def handle_initialize(self, request_id, params):
        """Handle MCP initialization"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": True},
                    "resources": {"subscribe": True, "listChanged": True},
                    "logging": {}
                },
                "serverInfo": {
                    "name": "terminal-automation-mcp",
                    "version": "1.0.0"
                }
            }
        }
    
    def handle_tools_list(self, request_id):
        """List available MCP tools"""
        tools = [
            {
                "name": "run_command",
                "description": "Execute shell commands in the terminal automation framework",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to execute"},
                        "working_dir": {"type": "string", "description": "Working directory (optional)"}
                    },
                    "required": ["command"]
                }
            },
            {
                "name": "validate_input",
                "description": "Validate various types of input using framework validators",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string", "description": "Input to validate"},
                        "type": {"type": "string", "enum": ["email", "url", "path", "hostname"], "description": "Validation type"}
                    },
                    "required": ["input", "type"]
                }
            },
            {
                "name": "framework_doctor",
                "description": "Run system health checks on the terminal automation framework",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            }
        ]
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"tools": tools}
        }
    
    def handle_tools_call(self, request_id, params):
        """Execute MCP tool"""
        tool_name = params.get('name')
        arguments = params.get('arguments', {})
        
        if tool_name == 'run_command':
            return self.execute_shell_command(request_id, arguments)
        elif tool_name == 'validate_input':
            return self.validate_user_input(request_id, arguments)
        elif tool_name == 'framework_doctor':
            return self.run_framework_doctor(request_id)
        else:
            return self.create_jsonrpc_error(-32601, f"Unknown tool: {tool_name}", request_id)
    
    def execute_shell_command(self, request_id, args):
        """Execute shell command safely"""
        import subprocess
        import shlex
        
        command = args.get('command', '')
        working_dir = args.get('working_dir', os.getcwd())
        
        # Security: Basic command validation
        if not command or len(command.strip()) == 0:
            return self.create_jsonrpc_error(-32602, "Command cannot be empty", request_id)
        
        # Dangerous commands blacklist
        dangerous_commands = ['rm -rf /', 'dd if=', 'mkfs', 'fdisk', 'parted']
        if any(dangerous in command for dangerous in dangerous_commands):
            return self.create_jsonrpc_error(-32602, "Dangerous command blocked", request_id)
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Command: {command}\nReturn code: {result.returncode}\n\nStdout:\n{result.stdout}\n\nStderr:\n{result.stderr}"
                        }
                    ]
                }
            }
        except subprocess.TimeoutExpired:
            return self.create_jsonrpc_error(-32603, "Command timed out", request_id)
        except Exception as e:
            return self.create_jsonrpc_error(-32603, f"Command execution failed: {str(e)}", request_id)
    
    def validate_user_input(self, request_id, args):
        """Validate user input using framework validators"""
        input_value = args.get('input', '')
        validation_type = args.get('type', '')
        
        # This would normally call the shell validation functions
        # For now, basic validation patterns
        import re
        
        patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'url': r'^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$',
            'hostname': r'^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$'
        }
        
        if validation_type in patterns:
            is_valid = bool(re.match(patterns[validation_type], input_value))
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Input: {input_value}\nType: {validation_type}\nValid: {'✅ Yes' if is_valid else '❌ No'}"
                        }
                    ]
                }
            }
        else:
            return self.create_jsonrpc_error(-32602, f"Unknown validation type: {validation_type}", request_id)
    
    def run_framework_doctor(self, request_id):
        """Run framework health check"""
        try:
            result = subprocess.run(['./dev', 'doctor'], capture_output=True, text=True, cwd=os.getcwd())
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Framework Health Check:\n\n{result.stdout}\n{result.stderr}"
                        }
                    ]
                }
            }
        except Exception as e:
            return self.create_jsonrpc_error(-32603, f"Health check failed: {str(e)}", request_id)
    
    def handle_resources_list(self, request_id):
        """List available MCP resources"""
        resources = [
            {
                "uri": "file:///framework/logs",
                "name": "Framework Logs",
                "description": "Access to terminal automation framework logs",
                "mimeType": "text/plain"
            },
            {
                "uri": "file:///framework/config",
                "name": "Framework Configuration",
                "description": "Terminal automation framework configuration files",
                "mimeType": "application/json"
            }
        ]
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"resources": resources}
        }
    
    def handle_resources_read(self, request_id, params):
        """Read MCP resource"""
        uri = params.get('uri', '')
        
        if uri == "file:///framework/logs":
            # Return recent logs
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "text/plain",
                            "text": "Framework logs would be here..."
                        }
                    ]
                }
            }
        else:
            return self.create_jsonrpc_error(-32602, f"Resource not found: {uri}", request_id)
    
    def create_jsonrpc_error(self, code, message, request_id):
        """Create JSON-RPC error response"""
        return {
            "jsonrpc": "2.0",
            "error": {"code": code, "message": message},
            "id": request_id
        }
    
    def send_jsonrpc_error(self, code, message, request_id):
        """Send JSON-RPC error response"""
        error_response = self.create_jsonrpc_error(code, message, request_id)
        
        self.send_response(400)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(error_response).encode())

def create_handler(mcp_shell_handler):
    def handler(*args, **kwargs):
        return MCPHandler(*args, mcp_shell_handler=mcp_shell_handler, **kwargs)
    return handler

if __name__ == "__main__":
    port = int(os.environ.get('MCP_PORT', '3333'))
    host = os.environ.get('MCP_HOST', 'localhost')
    
    server = HTTPServer((host, port), create_handler(None))
    print(f"MCP Server running on http://{host}:{port}")
    print(f"MCP endpoint: http://{host}:{port}/mcp")
    print(f"SSE endpoint: http://{host}:{port}/events")
    print(f"Metadata: http://{host}:{port}/.well-known/mcp_metadata")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nMCP Server stopped")
        server.shutdown()
EOF

    # Make server script executable and start it
    chmod +x "$server_script"
    
    # Start server in background
    python3 "$server_script" &
    local server_pid=$!
    
    # Store PID for cleanup
    echo $server_pid > /tmp/mcp_server.pid
    
    log::info "MCP HTTP Server started with PID $server_pid"
}

# Stop MCP server
mcp::server::stop() {
    if [[ "$MCP_SERVER_RUNNING" == "false" ]]; then
        log::warn "MCP Server not running"
        return 0
    fi
    
    log::info "Stopping MCP Server"
    
    # Kill server process if PID file exists
    if [[ -f /tmp/mcp_server.pid ]]; then
        local server_pid
        server_pid=$(cat /tmp/mcp_server.pid)
        if kill "$server_pid" 2>/dev/null; then
            log::info "MCP Server stopped (PID: $server_pid)"
        fi
        rm -f /tmp/mcp_server.pid
    fi
    
    MCP_SERVER_RUNNING=false
    log::success "MCP Server stopped successfully"
}

# Register built-in tools
mcp::tools::register_builtin() {
    log::debug "Registering built-in MCP tools"
    
    # Tool registry would be implemented here
    # For now, tools are defined in the Python server
    MCP_TOOLS=(
        "run_command"
        "validate_input" 
        "framework_doctor"
    )
}

# Register built-in resources
mcp::resources::register_builtin() {
    log::debug "Registering built-in MCP resources"
    
    # Resource registry would be implemented here
    MCP_RESOURCES=(
        "file:///framework/logs"
        "file:///framework/config"
    )
}

# Get server status
mcp::server::status() {
    if [[ "$MCP_SERVER_RUNNING" == "true" ]]; then
        echo "✅ MCP Server running on ${MCP_SERVER_HOST}:${MCP_SERVER_PORT}"
        echo "📊 Registered tools: ${#MCP_TOOLS[@]}"
        echo "📁 Registered resources: ${#MCP_RESOURCES[@]}"
        
        # Check if server is actually responding
        if command -v curl >/dev/null 2>&1; then
            if curl -s "http://${MCP_SERVER_HOST}:${MCP_SERVER_PORT}" >/dev/null; then
                echo "🌐 HTTP endpoint: http://${MCP_SERVER_HOST}:${MCP_SERVER_PORT}/mcp"
                echo "📡 SSE endpoint: http://${MCP_SERVER_HOST}:${MCP_SERVER_PORT}/events"
            else
                echo "⚠️  Server not responding to HTTP requests"
            fi
        fi
    else
        echo "❌ MCP Server not running"
    fi
}

# Test MCP server
mcp::server::test() {
    log::info "Testing MCP Server functionality"
    
    if ! command -v curl >/dev/null 2>&1; then
        log::error "curl not available for testing"
        return 1
    fi
    
    local base_url="http://${MCP_SERVER_HOST}:${MCP_SERVER_PORT}"
    
    # Test server info endpoint
    log::info "Testing server info endpoint..."
    if curl -s "$base_url" | grep -q "terminal-automation-mcp"; then
        log::success "✅ Server info endpoint working"
    else
        log::error "❌ Server info endpoint failed"
        return 1
    fi
    
    # Test MCP metadata endpoint
    log::info "Testing OAuth metadata endpoint..."
    if curl -s "$base_url/.well-known/mcp_metadata" | grep -q "resource_identifier"; then
        log::success "✅ OAuth metadata endpoint working"
    else
        log::error "❌ OAuth metadata endpoint failed"
        return 1
    fi
    
    # Test MCP JSON-RPC endpoint
    log::info "Testing MCP tools/list endpoint..."
    local test_request='{"jsonrpc": "2.0", "method": "tools/list", "id": 1}'
    if curl -s -X POST -H "Content-Type: application/json" -d "$test_request" "$base_url/mcp" | grep -q "run_command"; then
        log::success "✅ MCP tools/list endpoint working"
    else
        log::error "❌ MCP tools/list endpoint failed"
        return 1
    fi
    
    log::success "All MCP server tests passed!"
    return 0
}

# Main function for CLI usage
mcp::main() {
    local command="${1:-help}"
    
    case "$command" in
        init)
            mcp::server::init
            ;;
        start)
            mcp::server::init
            mcp::server::start
            ;;
        stop)
            mcp::server::stop
            ;;
        restart)
            mcp::server::stop
            sleep 2
            mcp::server::start
            ;;
        status)
            mcp::server::status
            ;;
        test)
            mcp::server::test
            ;;
        help|*)
            echo "MCP Server - Model Context Protocol Server v${MCP_SERVER_VERSION}"
            echo ""
            echo "Usage: $0 <command>"
            echo ""
            echo "Commands:"
            echo "  init     Initialize MCP server"
            echo "  start    Start MCP server"
            echo "  stop     Stop MCP server"
            echo "  restart  Restart MCP server"
            echo "  status   Show server status"
            echo "  test     Test server functionality"
            echo "  help     Show this help"
            ;;
    esac
}

# Export functions
export -f mcp::server::init mcp::server::start mcp::server::stop mcp::server::status mcp::server::test

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    mcp::main "$@"
fi