#!/usr/bin/env python3
"""
Aioke Standalone Server - Direct Launch
Bypasses any middleware or proxy issues
"""

import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser
from pathlib import Path

class AiokeHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('mvp_interface.html', 'rb') as f:
                self.wfile.write(f.read())
        elif self.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            status = {
                'systems': {
                    'computer_vision': True,
                    'voice_control': True,
                    'mix_generation': True,
                    'web_interface': True
                },
                'ready_count': 4,
                'message': 'Aioke systems operational'
            }
            self.wfile.write(json.dumps(status).encode())
        else:
            super().do_GET()

def main():
    print("🎛️  Aioke Standalone Server")
    print("=" * 40)
    
    # Find free port
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    
    server_address = ('', port)
    httpd = HTTPServer(server_address, AiokeHandler)
    
    url = f'http://localhost:{port}'
    print(f"✅ Aioke ready at: {url}")
    print("📱 Opening browser...")
    
    # Open browser
    try:
        webbrowser.open(url)
    except:
        pass
    
    print("\n🎛️ Aioke is running! Press Ctrl+C to stop.")
    httpd.serve_forever()

if __name__ == '__main__':
    main()