#!/usr/bin/env python3
"""
Simple web server for the AiCan Automation Dashboard
Serves the webapp locally for testing
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path

# Server configuration
PORT = 8080
DIRECTORY = Path(__file__).parent

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler with CORS enabled for API calls"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)
    
    def end_headers(self):
        """Add CORS headers to all responses"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        super().end_headers()
    
    def do_OPTIONS(self):
        """Handle preflight OPTIONS requests"""
        self.send_response(200)
        self.end_headers()
    
    def log_message(self, format, *args):
        """Enhanced logging with timestamps"""
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {format % args}")

def main():
    """Start the web server"""
    print(f"🚀 Starting AiCan Dashboard Server")
    print(f"📁 Serving files from: {DIRECTORY}")
    print(f"🌐 Server URL: http://localhost:{PORT}")
    print(f"📱 Access from other devices: http://{get_local_ip()}:{PORT}")
    print(f"⏹️  Press Ctrl+C to stop\n")
    
    # Change to the webapp directory
    os.chdir(DIRECTORY)
    
    try:
        with socketserver.TCPServer(("", PORT), CORSHTTPRequestHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Port {PORT} is already in use")
            print(f"💡 Try: lsof -ti:{PORT} | xargs kill")
        else:
            print(f"❌ Server error: {e}")
        sys.exit(1)

def get_local_ip():
    """Get the local IP address for network access"""
    import socket
    try:
        # Connect to a remote server to determine local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "localhost"

if __name__ == "__main__":
    main()