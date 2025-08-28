#!/usr/bin/env python3
"""
AiOke WebSocket Server for Real-Time Mixer Control
Bridges the web interface with the audio mixer
"""

import asyncio
import websockets
import json
import threading
from flask import Flask, send_file
from flask_cors import CORS
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from realtime_mixer import RealtimeMixer

class MixerWebServer:
    """WebSocket server for mixer control"""
    
    def __init__(self):
        self.mixer = RealtimeMixer(num_channels=4)
        self.clients = set()
        self.running = False
        
        # Flask app for serving the HTML interface
        self.app = Flask(__name__)
        CORS(self.app)
        
        @self.app.route('/')
        def index():
            return send_file('mixer_web_interface.html')
        
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connection"""
        self.clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.clients)}")
        
        try:
            async for message in websocket:
                data = json.loads(message)
                await self.process_command(data, websocket)
        except websockets.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            print(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def process_command(self, data, websocket):
        """Process mixer control commands"""
        cmd_type = data.get('type')
        channel = data.get('channel')
        value = data.get('value')
        
        # Channel commands
        if channel is not None:
            if cmd_type == 'volume':
                self.mixer.set_channel_volume(channel, value)
                
            elif cmd_type == 'pan':
                self.mixer.set_channel_pan(channel, value)
                
            elif cmd_type == 'eq_low':
                ch = self.mixer.channels[channel]
                self.mixer.set_channel_eq(channel, value, ch.mid_gain, ch.high_gain)
                
            elif cmd_type == 'eq_mid':
                ch = self.mixer.channels[channel]
                self.mixer.set_channel_eq(channel, ch.low_gain, value, ch.high_gain)
                
            elif cmd_type == 'eq_high':
                ch = self.mixer.channels[channel]
                self.mixer.set_channel_eq(channel, ch.low_gain, ch.mid_gain, value)
                
            elif cmd_type == 'reverb_send':
                self.mixer.set_channel_reverb(channel, value)
                
            elif cmd_type == 'delay_send':
                self.mixer.set_channel_delay(channel, value)
                
            elif cmd_type == 'mute':
                if value:
                    self.mixer.channels[channel].mute = True
                else:
                    self.mixer.channels[channel].mute = False
                    
            elif cmd_type == 'solo':
                if value:
                    self.mixer.channels[channel].solo = True
                else:
                    self.mixer.channels[channel].solo = False
        
        # Master commands
        else:
            if cmd_type == 'master_volume':
                self.mixer.master_volume = value
                
            elif cmd_type == 'master_limiter':
                self.mixer.master_limiter = value
                
            elif cmd_type == 'reverb_return':
                self.mixer.reverb_return = value
                
            elif cmd_type == 'delay_return':
                self.mixer.delay_return = value
                
            elif cmd_type == 'delay_time':
                self.mixer.delay_time = value
                
            elif cmd_type == 'delay_feedback':
                self.mixer.delay_feedback = value
                
            elif cmd_type == 'get_levels':
                # Send current levels back
                levels = self.mixer.get_levels()
                await websocket.send(json.dumps({
                    'type': 'levels',
                    'levels': levels
                }))
    
    async def broadcast_levels(self):
        """Broadcast level meters to all connected clients"""
        while self.running:
            if self.clients:
                levels = self.mixer.get_levels()
                message = json.dumps({
                    'type': 'levels',
                    'levels': levels
                })
                
                # Send to all connected clients
                disconnected = set()
                for client in self.clients:
                    try:
                        await client.send(message)
                    except websockets.ConnectionClosed:
                        disconnected.add(client)
                
                # Remove disconnected clients
                self.clients -= disconnected
            
            await asyncio.sleep(0.05)  # 20 Hz update rate
    
    def run_flask(self):
        """Run Flask server in background thread"""
        self.app.run(host='0.0.0.0', port=8080, debug=False)
    
    async def start_websocket_server(self):
        """Start WebSocket server"""
        self.running = True
        
        # Start level broadcasting
        broadcast_task = asyncio.create_task(self.broadcast_levels())
        
        # Start WebSocket server
        async with websockets.serve(self.handle_client, 'localhost', 8765):
            print("WebSocket server started on ws://localhost:8765")
            await asyncio.Future()  # Run forever
    
    def start(self):
        """Start mixer and servers"""
        print("\n=== AIOKE REAL-TIME MIXER SERVER ===\n")
        
        # Start the audio mixer
        self.mixer.start()
        print("✅ Audio mixer started")
        
        # Start Flask server in background thread
        flask_thread = threading.Thread(target=self.run_flask, daemon=True)
        flask_thread.start()
        print("✅ Web interface available at: http://localhost:8080")
        
        # Start WebSocket server
        print("✅ WebSocket server starting...")
        
        try:
            asyncio.run(self.start_websocket_server())
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            self.stop()
    
    def stop(self):
        """Stop mixer and servers"""
        self.running = False
        self.mixer.stop()
        print("Mixer server stopped")

def main():
    """Main entry point"""
    server = MixerWebServer()
    
    print("Starting AiOke Real-Time Mixer...")
    print("\n📌 Open your browser to: http://localhost:8080\n")
    
    try:
        server.start()
    except Exception as e:
        print(f"Error: {e}")
        server.stop()

if __name__ == "__main__":
    main()