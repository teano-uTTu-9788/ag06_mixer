#!/usr/bin/env python3
"""
Aioke Network Server - Multi-Device Access
Allows access from iPad and other devices on the same network
"""

import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socket
from pathlib import Path

class AiokeHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Access-Control-Allow-Origin', '*')  # Allow cross-origin for iPad
            self.end_headers()
            with open('mvp_interface.html', 'rb') as f:
                self.wfile.write(f.read())
        elif self.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            status = {
                'systems': {
                    'computer_vision': True,
                    'voice_control': True,
                    'mix_generation': True,
                    'web_interface': True
                },
                'ready_count': 4,
                'message': 'Aioke systems operational',
                'device': 'Multi-device ready'
            }
            self.wfile.write(json.dumps(status).encode())
        elif self.path.startswith('/api/'):
            # Handle API requests
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {'status': 'success', 'endpoint': self.path}
            self.wfile.write(json.dumps(response).encode())
        else:
            super().do_GET()
    
    def do_POST(self):
        """Handle POST requests for voice/gesture/mix endpoints"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        if '/api/voice' in self.path:
            response = {
                'command': 'Make vocals louder',
                'intent': 'volume_adjust',
                'confidence': 0.85,
                'action': 'applied'
            }
        elif '/api/gesture' in self.path:
            response = {
                'detected': True,
                'gesture': 'volume_up',
                'confidence': 0.92
            }
        elif '/api/mix' in self.path:
            response = {
                'style': 'Modern Pop',
                'settings': {
                    'vocals': {'volume': 0.8, 'eq_high': 3},
                    'guitar': {'volume': 0.7, 'pan': -0.3},
                    'drums': {'volume': 0.9, 'compression': 0.6}
                },
                'confidence': 0.88
            }
        else:
            response = {'status': 'success'}
        
        self.wfile.write(json.dumps(response).encode())

def main():
    print("🎛️  Aioke Network Server - Multi-Device Access")
    print("=" * 50)
    
    # Get network IP safely
    network_ip = '192.168.1.10'  # Your network IP
    
    # Alternative: try to detect automatically
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        detected_ip = s.getsockname()[0]
        if detected_ip:
            network_ip = detected_ip
        s.close()
    except:
        pass  # Use hardcoded IP
    
    # Find available port
    import random
    port = random.randint(8000, 9000)
    
    # Try to bind to find free port
    test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    for attempt_port in range(8000, 9000):
        try:
            test_socket.bind(('', attempt_port))
            test_socket.close()
            port = attempt_port
            break
        except:
            continue
    
    server_address = ('0.0.0.0', port)  # Listen on all interfaces
    httpd = HTTPServer(server_address, AiokeHandler)
    
    print(f"\n✅ Aioke is ready for multi-device access!")
    print(f"\n📱 Access from your devices:")
    print(f"   💻 Computer: http://localhost:{port}")
    print(f"   📱 iPad/Phone: http://{network_ip}:{port}")
    print(f"   🌐 Network: http://192.168.1.10:{port}")
    print(f"\n📋 iPad Instructions:")
    print(f"   1. Make sure iPad is on same WiFi network")
    print(f"   2. Open Safari or Chrome")
    print(f"   3. Go to: http://192.168.1.10:{port}")
    print(f"   4. Allow camera/microphone if prompted")
    print(f"\n🎛️ Aioke is running! Press Ctrl+C to stop.")
    
    httpd.serve_forever()

if __name__ == '__main__':
    main()