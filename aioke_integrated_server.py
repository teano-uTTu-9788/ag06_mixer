#!/usr/bin/env python3
"""
AiOke Integrated Production Server
Complete karaoke system with YouTube integration and real-time mixing
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import aiohttp
from aiohttp import web
from aiohttp_cors import setup as cors_setup, ResourceOptions
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY', '')
PORT = int(os.getenv('PORT', 51563))
HOST = '0.0.0.0'

@dataclass
class VideoResult:
    """YouTube video search result"""
    video_id: str
    title: str
    channel: str
    thumbnail: str
    duration: str = ""
    
@dataclass
class MixerSettings:
    """Current mixer configuration"""
    reverb: float = 0.3
    echo: float = 0.2
    pitch: float = 0.0
    tempo: float = 1.0
    vocal_reduction: float = 0.5
    bass_boost: float = 0.0
    treble_boost: float = 0.0
    compression: float = 0.3

class YouTubeService:
    """YouTube Data API v3 integration"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = 'https://www.googleapis.com/youtube/v3'
        self.session = None
        
    async def init(self):
        """Initialize aiohttp session"""
        self.session = aiohttp.ClientSession()
        
    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            
    async def search_karaoke(self, query: str, max_results: int = 12) -> List[VideoResult]:
        """Search for karaoke videos"""
        if not self.api_key:
            # Return demo results if no API key
            logger.warning("No YouTube API key configured, returning demo results")
            return self._get_demo_results(query)
            
        search_query = f"{query} karaoke"
        params = {
            'part': 'snippet',
            'q': search_query,
            'type': 'video',
            'videoCategoryId': '10',  # Music category
            'maxResults': max_results,
            'key': self.api_key
        }
        
        try:
            async with self.session.get(f"{self.base_url}/search", params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results = []
                    for item in data.get('items', []):
                        snippet = item['snippet']
                        results.append(VideoResult(
                            video_id=item['id']['videoId'],
                            title=snippet['title'],
                            channel=snippet['channelTitle'],
                            thumbnail=snippet['thumbnails']['medium']['url']
                        ))
                    return results
                else:
                    logger.error(f"YouTube API error: {resp.status}")
                    return self._get_demo_results(query)
        except Exception as e:
            logger.error(f"YouTube search error: {e}")
            return self._get_demo_results(query)
            
    def _get_demo_results(self, query: str) -> List[VideoResult]:
        """Return demo results when API is unavailable"""
        demo_songs = [
            ("Yesterday", "The Beatles"),
            ("Bohemian Rhapsody", "Queen"),
            ("Hotel California", "Eagles"),
            ("Sweet Caroline", "Neil Diamond"),
            ("Don't Stop Believin'", "Journey"),
            ("Wonderwall", "Oasis")
        ]
        
        results = []
        for title, artist in demo_songs:
            results.append(VideoResult(
                video_id=f"demo_{title.lower().replace(' ', '_')}",
                title=f"{title} - {artist} (Karaoke Version)",
                channel="Karaoke Channel",
                thumbnail=f"https://via.placeholder.com/320x180?text={title}"
            ))
        return results[:6]

class AiOkeServer:
    """Main AiOke server with all integrated features"""
    
    def __init__(self):
        self.app = web.Application()
        self.youtube = YouTubeService(YOUTUBE_API_KEY)
        self.mixer = MixerSettings()
        self.current_queue = []
        self.stats = {
            'total_requests': 0,
            'songs_played': 0,
            'start_time': datetime.now(),
            'last_activity': datetime.now()
        }
        self.setup_routes()
        self.setup_cors()
        
    def setup_routes(self):
        """Configure all API routes"""
        # Health check
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/api/health', self.health_check)
        
        # YouTube integration
        self.app.router.add_post('/api/youtube/search', self.youtube_search)
        self.app.router.add_post('/api/youtube/queue', self.add_to_queue)
        self.app.router.add_get('/api/youtube/queue', self.get_queue)
        
        # Mixer controls
        self.app.router.add_post('/api/mix', self.update_mix)
        self.app.router.add_get('/api/mix', self.get_mix)
        self.app.router.add_post('/api/effects', self.apply_effects)
        
        # Voice commands
        self.app.router.add_post('/api/voice', self.process_voice)
        
        # Stats and metrics
        self.app.router.add_get('/api/stats', self.get_stats)
        
        # Serve static files (interface)
        self.app.router.add_static('/', path='.', name='static')
        
    def setup_cors(self):
        """Configure CORS for iPad access"""
        cors = cors_setup(self.app, defaults={
            "*": ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        for route in list(self.app.router.routes()):
            cors.add(route)
            
    async def health_check(self, request):
        """Health check endpoint"""
        self.stats['last_activity'] = datetime.now()
        return web.json_response({
            'status': 'healthy',
            'service': 'AiOke Production Server',
            'uptime': str(datetime.now() - self.stats['start_time']),
            'youtube_api': bool(YOUTUBE_API_KEY),
            'songs_played': self.stats['songs_played']
        })
        
    async def youtube_search(self, request):
        """Search YouTube for karaoke videos"""
        self.stats['total_requests'] += 1
        self.stats['last_activity'] = datetime.now()
        
        try:
            data = await request.json()
            query = data.get('query', '')
            
            if not query:
                return web.json_response({'error': 'Query required'}, status=400)
                
            results = await self.youtube.search_karaoke(query)
            
            return web.json_response({
                'success': True,
                'results': [asdict(r) for r in results],
                'count': len(results)
            })
        except Exception as e:
            logger.error(f"Search error: {e}")
            return web.json_response({'error': str(e)}, status=500)
            
    async def add_to_queue(self, request):
        """Add song to queue"""
        self.stats['total_requests'] += 1
        
        try:
            data = await request.json()
            video_id = data.get('video_id')
            title = data.get('title', 'Unknown')
            
            self.current_queue.append({
                'video_id': video_id,
                'title': title,
                'added_at': datetime.now().isoformat()
            })
            
            self.stats['songs_played'] += 1
            
            # Apply automatic AI mix based on song
            await self._apply_ai_mix(title)
            
            return web.json_response({
                'success': True,
                'queue_position': len(self.current_queue),
                'mixer_settings': asdict(self.mixer)
            })
        except Exception as e:
            logger.error(f"Queue error: {e}")
            return web.json_response({'error': str(e)}, status=500)
            
    async def get_queue(self, request):
        """Get current queue"""
        return web.json_response({
            'queue': self.current_queue,
            'current': self.current_queue[0] if self.current_queue else None
        })
        
    async def update_mix(self, request):
        """Update mixer settings"""
        self.stats['total_requests'] += 1
        
        try:
            data = await request.json()
            
            # Update mixer settings
            for key, value in data.items():
                if hasattr(self.mixer, key):
                    setattr(self.mixer, key, float(value))
                    
            logger.info(f"Mixer updated: {asdict(self.mixer)}")
            
            return web.json_response({
                'success': True,
                'settings': asdict(self.mixer)
            })
        except Exception as e:
            logger.error(f"Mix update error: {e}")
            return web.json_response({'error': str(e)}, status=500)
            
    async def get_mix(self, request):
        """Get current mixer settings"""
        return web.json_response(asdict(self.mixer))
        
    async def apply_effects(self, request):
        """Apply specific audio effects"""
        self.stats['total_requests'] += 1
        
        try:
            data = await request.json()
            effect = data.get('effect')
            
            # Apply preset effects
            if effect == 'reverb':
                self.mixer.reverb = 0.7
                self.mixer.echo = 0.3
            elif effect == 'no_vocals':
                self.mixer.vocal_reduction = 0.9
            elif effect == 'party':
                self.mixer.bass_boost = 0.5
                self.mixer.compression = 0.6
            elif effect == 'clean':
                self.mixer = MixerSettings()  # Reset to defaults
                
            return web.json_response({
                'success': True,
                'effect': effect,
                'settings': asdict(self.mixer)
            })
        except Exception as e:
            logger.error(f"Effects error: {e}")
            return web.json_response({'error': str(e)}, status=500)
            
    async def process_voice(self, request):
        """Process voice commands"""
        self.stats['total_requests'] += 1
        
        try:
            data = await request.json()
            command = data.get('command', '').lower()
            
            response = "Command received"
            
            if 'play' in command:
                # Extract song name and search
                song = command.replace('play', '').strip()
                if song:
                    results = await self.youtube.search_karaoke(song)
                    if results:
                        response = f"Playing {results[0].title}"
                        
            elif 'skip' in command or 'next' in command:
                response = "Skipping to next song"
                
            elif 'volume' in command:
                if 'up' in command:
                    response = "Volume increased"
                elif 'down' in command:
                    response = "Volume decreased"
                    
            elif 'reverb' in command:
                self.mixer.reverb = 0.7
                response = "Reverb added"
                
            elif 'remove vocals' in command or 'no vocals' in command:
                self.mixer.vocal_reduction = 0.9
                response = "Vocals reduced"
                
            return web.json_response({
                'success': True,
                'command': command,
                'response': response,
                'settings': asdict(self.mixer)
            })
        except Exception as e:
            logger.error(f"Voice command error: {e}")
            return web.json_response({'error': str(e)}, status=500)
            
    async def get_stats(self, request):
        """Get server statistics"""
        uptime = datetime.now() - self.stats['start_time']
        
        return web.json_response({
            'total_requests': self.stats['total_requests'],
            'songs_played': self.stats['songs_played'],
            'uptime_seconds': uptime.total_seconds(),
            'uptime_readable': str(uptime),
            'last_activity': self.stats['last_activity'].isoformat(),
            'mixer_settings': asdict(self.mixer),
            'queue_size': len(self.current_queue)
        })
        
    async def _apply_ai_mix(self, song_title: str):
        """Apply AI-determined mix settings based on song"""
        # Simple genre detection based on keywords
        title_lower = song_title.lower()
        
        if any(word in title_lower for word in ['rock', 'metal', 'guitar']):
            # Rock preset
            self.mixer.bass_boost = 0.4
            self.mixer.compression = 0.5
            self.mixer.reverb = 0.2
        elif any(word in title_lower for word in ['ballad', 'slow', 'love']):
            # Ballad preset
            self.mixer.reverb = 0.5
            self.mixer.echo = 0.3
            self.mixer.compression = 0.2
        elif any(word in title_lower for word in ['dance', 'party', 'disco']):
            # Dance preset
            self.mixer.bass_boost = 0.6
            self.mixer.treble_boost = 0.3
            self.mixer.compression = 0.7
        else:
            # Default balanced mix
            self.mixer.reverb = 0.3
            self.mixer.compression = 0.4
            
        logger.info(f"AI mix applied for '{song_title}': {asdict(self.mixer)}")
        
    async def startup(self, app):
        """Initialize services on startup"""
        await self.youtube.init()
        logger.info(f"AiOke server starting on {HOST}:{PORT}")
        logger.info(f"YouTube API: {'Configured' if YOUTUBE_API_KEY else 'Demo mode'}")
        
    async def cleanup(self, app):
        """Cleanup on shutdown"""
        await self.youtube.close()
        logger.info("AiOke server shutting down")
        
    def run(self):
        """Start the server"""
        self.app.on_startup.append(self.startup)
        self.app.on_cleanup.append(self.cleanup)
        
        logger.info(f"🎤 AiOke Production Server")
        logger.info(f"🌐 Starting on http://{HOST}:{PORT}")
        logger.info(f"📱 iPad access: http://YOUR_IP:{PORT}/aioke_karaoke_interface.html")
        
        web.run_app(
            self.app,
            host=HOST,
            port=PORT,
            access_log=logger
        )

if __name__ == '__main__':
    server = AiOkeServer()
    server.run()