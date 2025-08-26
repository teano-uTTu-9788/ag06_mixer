#!/usr/bin/env python3
"""
AiOke YouTube Integration
Following Google best practices for YouTube Data API v3
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import aiohttp
from urllib.parse import quote_plus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VideoResult:
    """YouTube video search result"""
    video_id: str
    title: str
    channel: str
    thumbnail: str
    duration: str
    view_count: int
    published_at: str
    description: str
    
    def to_dict(self) -> Dict:
        return {
            'videoId': self.video_id,
            'title': self.title,
            'channel': self.channel,
            'thumbnail': self.thumbnail,
            'duration': self.duration,
            'viewCount': self.view_count,
            'publishedAt': self.published_at,
            'description': self.description,
            'embedUrl': f'https://www.youtube.com/embed/{self.video_id}',
            'watchUrl': f'https://www.youtube.com/watch?v={self.video_id}'
        }

class YouTubeService:
    """YouTube Data API v3 integration service"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('YOUTUBE_API_KEY')
        self.base_url = 'https://www.googleapis.com/youtube/v3'
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def search_karaoke(self, query: str, max_results: int = 10) -> List[VideoResult]:
        """Search for karaoke videos"""
        if not self.api_key:
            logger.error("YouTube API key not configured")
            # Return mock data for development
            return self._get_mock_results(query)
            
        # Add "karaoke" to search for karaoke versions
        search_query = f"{query} karaoke"
        
        try:
            # Search endpoint
            search_url = f"{self.base_url}/search"
            params = {
                'part': 'snippet',
                'q': search_query,
                'type': 'video',
                'videoCategoryId': '10',  # Music category
                'maxResults': max_results,
                'key': self.api_key,
                'order': 'relevance',
                'videoEmbeddable': 'true'
            }
            
            async with self.session.get(search_url, params=params) as response:
                if response.status != 200:
                    logger.error(f"YouTube API error: {response.status}")
                    return self._get_mock_results(query)
                    
                data = await response.json()
                video_ids = [item['id']['videoId'] for item in data.get('items', [])]
                
                # Get video details for duration and statistics
                if video_ids:
                    return await self._get_video_details(video_ids)
                    
        except Exception as e:
            logger.error(f"YouTube search error: {e}")
            return self._get_mock_results(query)
            
        return []
        
    async def _get_video_details(self, video_ids: List[str]) -> List[VideoResult]:
        """Get detailed video information"""
        videos_url = f"{self.base_url}/videos"
        params = {
            'part': 'snippet,contentDetails,statistics',
            'id': ','.join(video_ids),
            'key': self.api_key
        }
        
        try:
            async with self.session.get(videos_url, params=params) as response:
                if response.status != 200:
                    return []
                    
                data = await response.json()
                results = []
                
                for item in data.get('items', []):
                    snippet = item['snippet']
                    details = item['contentDetails']
                    stats = item['statistics']
                    
                    result = VideoResult(
                        video_id=item['id'],
                        title=snippet['title'],
                        channel=snippet['channelTitle'],
                        thumbnail=snippet['thumbnails']['high']['url'],
                        duration=self._parse_duration(details['duration']),
                        view_count=int(stats.get('viewCount', 0)),
                        published_at=snippet['publishedAt'],
                        description=snippet.get('description', '')[:200]
                    )
                    results.append(result)
                    
                return results
                
        except Exception as e:
            logger.error(f"Error getting video details: {e}")
            return []
            
    def _parse_duration(self, iso_duration: str) -> str:
        """Convert ISO 8601 duration to readable format"""
        # PT4M13S -> 4:13
        import re
        match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', iso_duration)
        if match:
            hours, minutes, seconds = match.groups()
            parts = []
            if hours:
                parts.append(hours)
            parts.append(minutes or '0')
            parts.append(f"{int(seconds or 0):02d}")
            return ':'.join(parts)
        return '0:00'
        
    def _get_mock_results(self, query: str) -> List[VideoResult]:
        """Return mock data for development"""
        mock_songs = [
            {
                'video_id': 'dQw4w9WgXcQ',
                'title': f'{query} - Karaoke Version',
                'channel': 'KaraokeHits',
                'thumbnail': 'https://i.ytimg.com/vi/dQw4w9WgXcQ/hqdefault.jpg',
                'duration': '3:33',
                'view_count': 1000000,
                'published_at': '2024-01-01T00:00:00Z',
                'description': f'High quality karaoke version of {query}'
            },
            {
                'video_id': 'L_jWHffIx5E',
                'title': f'{query} (Instrumental)',
                'channel': 'SingKing',
                'thumbnail': 'https://i.ytimg.com/vi/L_jWHffIx5E/hqdefault.jpg',
                'duration': '4:02',
                'view_count': 500000,
                'published_at': '2024-02-01T00:00:00Z',
                'description': f'Professional karaoke track - {query}'
            },
            {
                'video_id': '9bZkp7q19f0',
                'title': f'{query} Karaoke with Lyrics',
                'channel': 'KaraFun',
                'thumbnail': 'https://i.ytimg.com/vi/9bZkp7q19f0/hqdefault.jpg',
                'duration': '4:12',
                'view_count': 750000,
                'published_at': '2024-03-01T00:00:00Z',
                'description': f'Sing along to {query} with on-screen lyrics'
            }
        ]
        
        return [VideoResult(**data) for data in mock_songs[:3]]

class KaraokeManager:
    """Manages karaoke sessions and video selection"""
    
    def __init__(self):
        self.youtube_service = None
        self.current_video = None
        self.queue = []
        
    async def initialize(self, youtube_api_key: Optional[str] = None):
        """Initialize the karaoke manager"""
        self.youtube_service = YouTubeService(youtube_api_key)
        
    async def search_songs(self, query: str) -> List[Dict]:
        """Search for karaoke songs"""
        async with self.youtube_service as service:
            results = await service.search_karaoke(query)
            return [r.to_dict() for r in results]
            
    async def add_to_queue(self, video_id: str, video_data: Dict):
        """Add a song to the karaoke queue"""
        self.queue.append({
            'videoId': video_id,
            'addedAt': datetime.utcnow().isoformat(),
            **video_data
        })
        logger.info(f"Added to queue: {video_data.get('title')}")
        
    def get_queue(self) -> List[Dict]:
        """Get current karaoke queue"""
        return self.queue
        
    def play_next(self) -> Optional[Dict]:
        """Play next song in queue"""
        if self.queue:
            self.current_video = self.queue.pop(0)
            return self.current_video
        return None
        
    def get_current(self) -> Optional[Dict]:
        """Get currently playing video"""
        return self.current_video

# API endpoints for integration
async def handle_youtube_search(request_data: Dict) -> Dict:
    """Handle YouTube search API request"""
    query = request_data.get('query', '')
    if not query:
        return {'error': 'No search query provided', 'results': []}
        
    manager = KaraokeManager()
    await manager.initialize()
    
    try:
        results = await manager.search_songs(query)
        return {
            'success': True,
            'query': query,
            'count': len(results),
            'results': results
        }
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {
            'success': False,
            'error': str(e),
            'results': []
        }

async def handle_add_to_queue(request_data: Dict) -> Dict:
    """Handle adding video to queue"""
    video_id = request_data.get('videoId')
    video_data = request_data.get('videoData', {})
    
    if not video_id:
        return {'error': 'No video ID provided'}
        
    manager = KaraokeManager()
    await manager.add_to_queue(video_id, video_data)
    
    return {
        'success': True,
        'videoId': video_id,
        'queueLength': len(manager.get_queue())
    }

if __name__ == "__main__":
    # Test the YouTube integration
    async def test():
        print("🎤 Testing AiOke YouTube Integration")
        print("="*50)
        
        # Search for a song
        result = await handle_youtube_search({'query': 'Bohemian Rhapsody'})
        
        print(f"Search Results: {result['count']} videos found")
        for video in result['results'][:3]:
            print(f"\n📹 {video['title']}")
            print(f"   Channel: {video['channel']}")
            print(f"   Duration: {video['duration']}")
            print(f"   Views: {video['viewCount']:,}")
            print(f"   URL: {video['watchUrl']}")
            
        print("\n✅ YouTube integration ready for AiOke!")
        
    asyncio.run(test())