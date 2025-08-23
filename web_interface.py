#!/usr/bin/env python3
"""
AG06 Mixer Web Interface
Real-time web control panel for AG06 mixer development
"""

from aiohttp import web
import aiohttp
from aiohttp import web
import json
import asyncio
import weakref
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AG06WebInterface:
    """Web interface for AG06 Mixer control"""
    
    def __init__(self):
        self.app = web.Application()
        self.websockets = weakref.WeakSet()
        self.mixer_state = {
            'connected': False,
            'channels': {
                '1': {'gain': 0, 'mute': False, 'solo': False, 'pan': 0},
                '2': {'gain': 0, 'mute': False, 'solo': False, 'pan': 0},
            },
            'master': {'volume': 75, 'mute': False},
            'monitor': {'level': 50, 'mix': 50},
            'effects': {'reverb': 0, 'delay': 0, 'chorus': 0},
            'loopback': False,
            'phantom_power': False
        }
        self.setup_routes()
        
    def setup_routes(self):
        """Configure web routes"""
        self.app.router.add_get('/', self.index)
        self.app.router.add_get('/api/status', self.get_status)
        self.app.router.add_post('/api/control', self.handle_control)
        self.app.router.add_get('/ws', self.websocket_handler)
        # Only add static route if directory exists
        static_path = Path(__file__).parent / 'static'
        if static_path.exists():
            self.app.router.add_static('/static', path=static_path, show_index=False)
        
    async def index(self, request):
        """Serve main HTML interface"""
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AG06 Mixer Control Panel</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .header h1 {
            color: #333;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .status {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 500;
        }
        
        .status.connected {
            background: #10b981;
            color: white;
        }
        
        .status.disconnected {
            background: #ef4444;
            color: white;
        }
        
        .mixer-panel {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .channels {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .channel {
            background: #f3f4f6;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }
        
        .channel h3 {
            color: #4b5563;
            margin-bottom: 15px;
        }
        
        .fader {
            width: 100%;
            height: 150px;
            -webkit-appearance: none;
            appearance: none;
            background: linear-gradient(to right, #ddd 0%, #ddd 100%);
            border-radius: 5px;
            outline: none;
            writing-mode: bt-lr; /* IE */
            -webkit-appearance: slider-vertical; /* WebKit */
            cursor: pointer;
        }
        
        .fader::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 40px;
            height: 20px;
            background: #667eea;
            cursor: pointer;
            border-radius: 5px;
        }
        
        .controls {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-top: 15px;
        }
        
        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.3s;
        }
        
        .btn-mute {
            background: #ef4444;
            color: white;
        }
        
        .btn-mute.active {
            background: #dc2626;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
        }
        
        .btn-solo {
            background: #f59e0b;
            color: white;
        }
        
        .btn-solo.active {
            background: #d97706;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
        }
        
        .master-section {
            background: #e5e7eb;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }
        
        .effects-section {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .effect {
            background: #f9fafb;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        
        .knob {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 10px auto;
            position: relative;
            cursor: pointer;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        }
        
        .knob::after {
            content: '';
            position: absolute;
            width: 4px;
            height: 25px;
            background: white;
            top: 5px;
            left: 50%;
            transform: translateX(-50%);
            border-radius: 2px;
        }
        
        .value-display {
            margin-top: 10px;
            font-size: 18px;
            font-weight: bold;
            color: #4b5563;
        }
        
        .monitor-section {
            background: #f3f4f6;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            display: flex;
            justify-content: space-around;
            align-items: center;
        }
        
        .toggle-switch {
            position: relative;
            width: 60px;
            height: 30px;
            background: #cbd5e1;
            border-radius: 15px;
            cursor: pointer;
            transition: background 0.3s;
        }
        
        .toggle-switch.active {
            background: #10b981;
        }
        
        .toggle-switch::after {
            content: '';
            position: absolute;
            width: 26px;
            height: 26px;
            border-radius: 50%;
            background: white;
            top: 2px;
            left: 2px;
            transition: transform 0.3s;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        .toggle-switch.active::after {
            transform: translateX(30px);
        }
        
        .log-panel {
            background: #1f2937;
            color: #10b981;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            max-height: 200px;
            overflow-y: auto;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .connecting {
            animation: pulse 1.5s infinite;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>
                🎛️ AG06 Mixer Control Panel
                <span id="status" class="status disconnected">
                    <span class="indicator">●</span> Disconnected
                </span>
            </h1>
        </div>
        
        <div class="mixer-panel">
            <h2 style="margin-bottom: 20px; color: #374151;">Channel Controls</h2>
            
            <div class="channels">
                <div class="channel">
                    <h3>Channel 1</h3>
                    <input type="range" class="fader" id="ch1-gain" min="-60" max="10" value="0" orient="vertical">
                    <div class="value-display"><span id="ch1-gain-val">0</span> dB</div>
                    <div class="controls">
                        <button class="btn btn-mute" id="ch1-mute">Mute</button>
                        <button class="btn btn-solo" id="ch1-solo">Solo</button>
                    </div>
                </div>
                
                <div class="channel">
                    <h3>Channel 2</h3>
                    <input type="range" class="fader" id="ch2-gain" min="-60" max="10" value="0" orient="vertical">
                    <div class="value-display"><span id="ch2-gain-val">0</span> dB</div>
                    <div class="controls">
                        <button class="btn btn-mute" id="ch2-mute">Mute</button>
                        <button class="btn btn-solo" id="ch2-solo">Solo</button>
                    </div>
                </div>
                
                <div class="channel">
                    <h3>Master</h3>
                    <input type="range" class="fader" id="master-volume" min="0" max="100" value="75" orient="vertical">
                    <div class="value-display"><span id="master-volume-val">75</span>%</div>
                    <div class="controls">
                        <button class="btn btn-mute" id="master-mute">Mute</button>
                    </div>
                </div>
            </div>
            
            <div class="master-section">
                <h3 style="margin-bottom: 15px; color: #374151;">Effects</h3>
                <div class="effects-section">
                    <div class="effect">
                        <label>Reverb</label>
                        <div class="knob" id="reverb-knob"></div>
                        <div class="value-display"><span id="reverb-val">0</span>%</div>
                    </div>
                    <div class="effect">
                        <label>Delay</label>
                        <div class="knob" id="delay-knob"></div>
                        <div class="value-display"><span id="delay-val">0</span>%</div>
                    </div>
                    <div class="effect">
                        <label>Chorus</label>
                        <div class="knob" id="chorus-knob"></div>
                        <div class="value-display"><span id="chorus-val">0</span>%</div>
                    </div>
                </div>
            </div>
            
            <div class="monitor-section">
                <div>
                    <label>Monitor Level</label>
                    <input type="range" id="monitor-level" min="0" max="100" value="50">
                    <span id="monitor-level-val">50%</span>
                </div>
                <div>
                    <label>Monitor Mix</label>
                    <input type="range" id="monitor-mix" min="0" max="100" value="50">
                    <span id="monitor-mix-val">50%</span>
                </div>
                <div>
                    <label>Loopback</label>
                    <div class="toggle-switch" id="loopback-toggle"></div>
                </div>
                <div>
                    <label>Phantom Power</label>
                    <div class="toggle-switch" id="phantom-toggle"></div>
                </div>
            </div>
            
            <div class="log-panel" id="log">
                <div>🎛️ AG06 Mixer Web Interface Started</div>
                <div>⏳ Connecting to mixer...</div>
            </div>
        </div>
    </div>
    
    <script>
        let ws;
        let mixerState = {};
        
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            ws.onopen = () => {
                updateStatus(true);
                log('✅ Connected to AG06 Mixer');
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.type === 'state') {
                    updateMixerState(data.state);
                } else if (data.type === 'log') {
                    log(data.message);
                }
            };
            
            ws.onclose = () => {
                updateStatus(false);
                log('❌ Disconnected from mixer');
                setTimeout(connectWebSocket, 3000);
            };
            
            ws.onerror = (error) => {
                log('⚠️ Connection error');
            };
        }
        
        function updateStatus(connected) {
            const statusEl = document.getElementById('status');
            if (connected) {
                statusEl.className = 'status connected';
                statusEl.innerHTML = '<span class="indicator">●</span> Connected';
            } else {
                statusEl.className = 'status disconnected';
                statusEl.innerHTML = '<span class="indicator">●</span> Disconnected';
            }
        }
        
        function updateMixerState(state) {
            mixerState = state;
            
            // Update UI elements
            if (state.channels) {
                Object.keys(state.channels).forEach(ch => {
                    const channel = state.channels[ch];
                    const gainEl = document.getElementById(`ch${ch}-gain`);
                    const muteBtn = document.getElementById(`ch${ch}-mute`);
                    const soloBtn = document.getElementById(`ch${ch}-solo`);
                    
                    if (gainEl) gainEl.value = channel.gain;
                    if (muteBtn) muteBtn.classList.toggle('active', channel.mute);
                    if (soloBtn) soloBtn.classList.toggle('active', channel.solo);
                });
            }
            
            if (state.master) {
                document.getElementById('master-volume').value = state.master.volume;
                document.getElementById('master-mute').classList.toggle('active', state.master.mute);
            }
            
            if (state.monitor) {
                document.getElementById('monitor-level').value = state.monitor.level;
                document.getElementById('monitor-mix').value = state.monitor.mix;
            }
            
            document.getElementById('loopback-toggle').classList.toggle('active', state.loopback);
            document.getElementById('phantom-toggle').classList.toggle('active', state.phantom_power);
        }
        
        function sendControl(control, value) {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'control', control, value }));
                log(`📤 ${control}: ${value}`);
            }
        }
        
        function log(message) {
            const logEl = document.getElementById('log');
            const entry = document.createElement('div');
            const timestamp = new Date().toLocaleTimeString();
            entry.textContent = `[${timestamp}] ${message}`;
            logEl.appendChild(entry);
            logEl.scrollTop = logEl.scrollHeight;
            
            // Keep only last 50 messages
            while (logEl.children.length > 50) {
                logEl.removeChild(logEl.firstChild);
            }
        }
        
        // Set up event listeners
        document.addEventListener('DOMContentLoaded', () => {
            // Channel controls
            ['1', '2'].forEach(ch => {
                const gain = document.getElementById(`ch${ch}-gain`);
                const gainVal = document.getElementById(`ch${ch}-gain-val`);
                const mute = document.getElementById(`ch${ch}-mute`);
                const solo = document.getElementById(`ch${ch}-solo`);
                
                gain.addEventListener('input', (e) => {
                    gainVal.textContent = e.target.value;
                    sendControl(`ch${ch}_gain`, e.target.value);
                });
                
                mute.addEventListener('click', () => {
                    mute.classList.toggle('active');
                    sendControl(`ch${ch}_mute`, mute.classList.contains('active'));
                });
                
                solo.addEventListener('click', () => {
                    solo.classList.toggle('active');
                    sendControl(`ch${ch}_solo`, solo.classList.contains('active'));
                });
            });
            
            // Master controls
            const masterVol = document.getElementById('master-volume');
            const masterVolVal = document.getElementById('master-volume-val');
            const masterMute = document.getElementById('master-mute');
            
            masterVol.addEventListener('input', (e) => {
                masterVolVal.textContent = e.target.value;
                sendControl('master_volume', e.target.value);
            });
            
            masterMute.addEventListener('click', () => {
                masterMute.classList.toggle('active');
                sendControl('master_mute', masterMute.classList.contains('active'));
            });
            
            // Monitor controls
            const monitorLevel = document.getElementById('monitor-level');
            const monitorLevelVal = document.getElementById('monitor-level-val');
            const monitorMix = document.getElementById('monitor-mix');
            const monitorMixVal = document.getElementById('monitor-mix-val');
            
            monitorLevel.addEventListener('input', (e) => {
                monitorLevelVal.textContent = e.target.value + '%';
                sendControl('monitor_level', e.target.value);
            });
            
            monitorMix.addEventListener('input', (e) => {
                monitorMixVal.textContent = e.target.value + '%';
                sendControl('monitor_mix', e.target.value);
            });
            
            // Toggle switches
            document.getElementById('loopback-toggle').addEventListener('click', (e) => {
                e.target.classList.toggle('active');
                sendControl('loopback', e.target.classList.contains('active'));
            });
            
            document.getElementById('phantom-toggle').addEventListener('click', (e) => {
                e.target.classList.toggle('active');
                sendControl('phantom_power', e.target.classList.contains('active'));
            });
            
            // Connect WebSocket
            connectWebSocket();
        });
    </script>
</body>
</html>"""
        return web.Response(text=html_content, content_type='text/html')
    
    async def get_status(self, request):
        """Get current mixer status"""
        return web.json_response(self.mixer_state)
    
    async def handle_control(self, request):
        """Handle control changes from web interface"""
        data = await request.json()
        control = data.get('control')
        value = data.get('value')
        
        logger.info(f"Control change: {control} = {value}")
        
        # Update mixer state
        if control.startswith('ch'):
            channel = control[2]
            param = control[4:]
            if channel in self.mixer_state['channels']:
                self.mixer_state['channels'][channel][param] = value
        elif control.startswith('master_'):
            param = control[7:]
            self.mixer_state['master'][param] = value
        elif control.startswith('monitor_'):
            param = control[8:]
            self.mixer_state['monitor'][param] = value
        elif control in ['loopback', 'phantom_power']:
            self.mixer_state[control] = value
        elif control in ['reverb', 'delay', 'chorus']:
            self.mixer_state['effects'][control] = value
        
        # Broadcast update to all connected clients
        await self.broadcast_state()
        
        return web.json_response({'status': 'ok'})
    
    async def websocket_handler(self, request):
        """Handle WebSocket connections"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.websockets.add(ws)
        
        # Send initial state
        await ws.send_json({
            'type': 'state',
            'state': self.mixer_state
        })
        
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await self.handle_ws_message(ws, data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f'WebSocket error: {ws.exception()}')
        finally:
            self.websockets.discard(ws)
            
        return ws
    
    async def handle_ws_message(self, ws, data):
        """Handle WebSocket messages"""
        msg_type = data.get('type')
        
        if msg_type == 'control':
            control = data.get('control')
            value = data.get('value')
            
            # Process control change
            await self.handle_control_change(control, value)
            
            # Broadcast to all clients
            await self.broadcast_state()
    
    async def handle_control_change(self, control, value):
        """Process control changes"""
        logger.info(f"Processing: {control} = {value}")
        
        # Update internal state
        if control.startswith('ch') and '_' in control:
            ch, param = control.split('_', 1)
            channel_num = ch[2:]
            if channel_num in self.mixer_state['channels']:
                self.mixer_state['channels'][channel_num][param] = value
        elif control.startswith('master_'):
            param = control.replace('master_', '')
            self.mixer_state['master'][param] = value
        elif control.startswith('monitor_'):
            param = control.replace('monitor_', '')
            self.mixer_state['monitor'][param] = value
        elif control in self.mixer_state:
            self.mixer_state[control] = value
        elif control in self.mixer_state.get('effects', {}):
            self.mixer_state['effects'][control] = value
    
    async def broadcast_state(self):
        """Broadcast mixer state to all connected clients"""
        message = {
            'type': 'state',
            'state': self.mixer_state
        }
        
        # Send to all connected WebSocket clients
        for ws in self.websockets:
            try:
                await ws.send_json(message)
            except ConnectionResetError:
                pass
    
    async def simulate_ag06_connection(self):
        """Simulate AG06 device connection for development"""
        await asyncio.sleep(2)
        self.mixer_state['connected'] = True
        await self.broadcast_state()
        
        # Log connection
        for ws in self.websockets:
            try:
                await ws.send_json({
                    'type': 'log',
                    'message': '✅ AG06 device connected (simulated)'
                })
            except:
                pass
    
    def run(self, host='0.0.0.0', port=8080):
        """Start the web server"""
        logger.info(f"🎛️  Starting AG06 Mixer Web Interface on http://{host}:{port}")
        
        # Start background tasks
        async def start_background_tasks(app):
            app['ag06_connection'] = asyncio.create_task(self.simulate_ag06_connection())
        
        self.app.on_startup.append(start_background_tasks)
        
        # Run the app
        web.run_app(self.app, host=host, port=port)

if __name__ == '__main__':
    interface = AG06WebInterface()
    interface.run()