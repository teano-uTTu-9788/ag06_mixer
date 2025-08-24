#!/usr/bin/env python3
"""
WebRTC Audio Streaming Server Launcher
Starts both signaling server and media server with static file serving
"""

import asyncio
import logging
import os
from pathlib import Path
import signal
import sys
from aiohttp import web, web_request
import aioredis
import socketio

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent))

try:
    from signaling_server import app as signaling_app, signaling_server, sio
    from media_server import WebRTCMediaServer, MediaServerIntegration
    logger.info("Successfully imported WebRTC components")
except ImportError as e:
    logger.error(f"Failed to import WebRTC components: {e}")
    sys.exit(1)

class WebRTCServerManager:
    """Manages the complete WebRTC server stack"""
    
    def __init__(self, host='0.0.0.0', port=8080, static_port=8081):
        self.host = host
        self.port = port
        self.static_port = static_port
        self.media_server = None
        self.integration = None
        self.static_runner = None
        self.signaling_runner = None
        
    async def setup_static_server(self):
        """Setup static file server for the HTML interface"""
        static_app = web.Application()
        
        # Add static file serving
        static_dir = current_dir / 'static'
        static_app.router.add_static('/', str(static_dir), name='static')
        
        # Add CORS headers
        async def add_cors_headers(request, handler):
            response = await handler(request)
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return response
        
        static_app.middlewares.append(add_cors_headers)
        
        # Health check for static server
        async def static_health(request):
            return web.json_response({
                'status': 'healthy',
                'server': 'webrtc-static',
                'static_dir': str(static_dir)
            })
        
        static_app.router.add_get('/health', static_health)
        
        return static_app
    
    async def setup_media_server(self):
        """Setup the WebRTC media server"""
        self.media_server = WebRTCMediaServer()
        self.integration = MediaServerIntegration(self.media_server)
        
        # Integrate with signaling server
        await self.integrate_with_signaling()
    
    async def integrate_with_signaling(self):
        """Integrate media server with signaling server events"""
        
        # Override signaling server events to include media processing
        original_offer = None
        original_answer = None
        original_ice_candidate = None
        original_audio_stream_metadata = None
        original_disconnect = None
        
        # Store original handlers
        if hasattr(sio, '_handlers'):
            for namespace, handlers in sio._handlers.items():
                if 'offer' in handlers:
                    original_offer = handlers['offer']
                if 'answer' in handlers:
                    original_answer = handlers['answer']
                if 'ice_candidate' in handlers:
                    original_ice_candidate = handlers['ice_candidate']
                if 'audio_stream_metadata' in handlers:
                    original_audio_stream_metadata = handlers['audio_stream_metadata']
                if 'disconnect' in handlers:
                    original_disconnect = handlers['disconnect']
        
        # Enhanced offer handler
        @sio.event
        async def offer(sid, data):
            logger.info(f"Processing offer with media server for {sid}")
            
            try:
                # Process through media server
                if 'offer' in data:
                    answer = await self.integration.on_offer(sid, data['offer'])
                    
                    # Send answer back
                    await sio.emit('answer', {
                        'answer': answer,
                        'from': 'server'
                    }, to=sid)
                
                # Also call original handler
                if original_offer:
                    return await original_offer(sid, data)
                return True
                
            except Exception as e:
                logger.error(f"Error in offer handler: {e}")
                await sio.emit('error', {'message': str(e)}, to=sid)
                return False
        
        # Enhanced answer handler  
        @sio.event
        async def answer(sid, data):
            logger.info(f"Processing answer with media server for {sid}")
            
            try:
                if 'answer' in data:
                    await self.integration.on_answer(sid, data['answer'])
                
                # Also call original handler
                if original_answer:
                    return await original_answer(sid, data)
                return True
                
            except Exception as e:
                logger.error(f"Error in answer handler: {e}")
                return False
        
        # Enhanced ICE candidate handler
        @sio.event
        async def ice_candidate(sid, data):
            logger.info(f"Processing ICE candidate with media server for {sid}")
            
            try:
                if 'candidate' in data:
                    await self.integration.on_ice_candidate(sid, data['candidate'])
                
                # Also call original handler
                if original_ice_candidate:
                    return await original_ice_candidate(sid, data)
                return True
                
            except Exception as e:
                logger.error(f"Error in ICE candidate handler: {e}")
                return False
        
        # Enhanced metadata handler
        @sio.event
        async def audio_stream_metadata(sid, data):
            logger.info(f"Processing stream metadata for {sid}")
            
            try:
                await self.integration.on_stream_config(sid, data)
                
                # Also call original handler
                if original_audio_stream_metadata:
                    return await original_audio_stream_metadata(sid, data)
                return True
                
            except Exception as e:
                logger.error(f"Error in metadata handler: {e}")
                return False
        
        # Enhanced disconnect handler
        @sio.event
        async def disconnect(sid):
            logger.info(f"Cleaning up media server resources for {sid}")
            
            try:
                await self.integration.on_disconnect(sid)
                
                # Also call original handler
                if original_disconnect:
                    return await original_disconnect(sid)
                return True
                
            except Exception as e:
                logger.error(f"Error in disconnect handler: {e}")
                return False
        
        # Add stats endpoint
        @sio.event
        async def get_stream_stats(sid):
            try:
                stats = await self.integration.get_stats(sid)
                await sio.emit('stream_stats', stats, to=sid)
                return stats
            except Exception as e:
                logger.error(f"Error getting stats: {e}")
                return None
    
    async def start_servers(self):
        """Start both signaling and static servers"""
        logger.info("Starting WebRTC servers...")
        
        # Setup media server
        await self.setup_media_server()
        
        # Start signaling server
        signaling_runner = web.AppRunner(signaling_app)
        await signaling_runner.setup()
        signaling_site = web.TCPSite(signaling_runner, self.host, self.port)
        await signaling_site.start()
        self.signaling_runner = signaling_runner
        
        logger.info(f"🎵 WebRTC Signaling Server started on http://{self.host}:{self.port}")
        
        # Start static file server
        static_app = await self.setup_static_server()
        static_runner = web.AppRunner(static_app)
        await static_runner.setup()
        static_site = web.TCPSite(static_runner, self.host, self.static_port)
        await static_site.start()
        self.static_runner = static_runner
        
        logger.info(f"📁 Static File Server started on http://{self.host}:{self.static_port}")
        logger.info(f"🌐 Open http://{self.host}:{self.static_port} in your browser")
        
        # Wait for signaling server setup
        if hasattr(signaling_server, 'setup'):
            await signaling_server.setup()
    
    async def stop_servers(self):
        """Stop all servers"""
        logger.info("Stopping WebRTC servers...")
        
        if self.static_runner:
            await self.static_runner.cleanup()
        
        if self.signaling_runner:
            await self.signaling_runner.cleanup()
        
        if hasattr(signaling_server, 'cleanup'):
            await signaling_server.cleanup()
        
        if self.media_server:
            await self.media_server.cleanup_all()
        
        logger.info("All servers stopped")


async def main():
    """Main server startup"""
    # Configuration
    HOST = os.getenv('WEBRTC_HOST', '0.0.0.0')
    SIGNALING_PORT = int(os.getenv('WEBRTC_SIGNALING_PORT', '8080'))
    STATIC_PORT = int(os.getenv('WEBRTC_STATIC_PORT', '8081'))
    
    # Create server manager
    server_manager = WebRTCServerManager(
        host=HOST,
        port=SIGNALING_PORT,
        static_port=STATIC_PORT
    )
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(server_manager.stop_servers())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start servers
        await server_manager.start_servers()
        
        # Print startup summary
        print("\n" + "="*60)
        print("🎵 AG06 AI Mixer - WebRTC Audio Streaming")
        print("="*60)
        print(f"📡 Signaling Server: http://{HOST}:{SIGNALING_PORT}")
        print(f"🌐 Web Interface:    http://{HOST}:{STATIC_PORT}")
        print(f"📊 Health Check:     http://{HOST}:{SIGNALING_PORT}/health")
        print(f"📈 Stats:            http://{HOST}:{SIGNALING_PORT}/api/stats")
        print("="*60)
        print("🚀 WebRTC server ready! Open the web interface to start streaming.")
        print("💡 Features: Real-time AI mixing, genre detection, studio effects")
        print("🔧 Controls: Noise gate, compression, EQ, reverb")
        print("="*60 + "\n")
        
        # Keep servers running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        await server_manager.stop_servers()


if __name__ == "__main__":
    # Check dependencies
    try:
        import aiohttp
        import socketio
        import aioredis
        # import aiortc  # May not be installed yet
        logger.info("Core dependencies available")
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        print("\n🚨 Missing Dependencies!")
        print("Please install required packages:")
        print("pip install aiohttp python-socketio aioredis aiortc av")
        print("brew install portaudio && pip install pyaudio")
        sys.exit(1)
    
    # Check if Redis is running (optional)
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        logger.info("Redis connection successful")
    except Exception:
        logger.warning("Redis not available - using in-memory storage")
    
    # Run server
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown complete")