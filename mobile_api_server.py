#!/usr/bin/env python3
"""
AI Mixer Mobile API Server

Optimized API endpoints specifically designed for mobile app integration:
- Low-latency audio processing
- WebSocket streaming support
- Mobile-friendly response formats
- Efficient bandwidth usage
- Battery optimization considerations
"""

import asyncio
import base64
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import struct

from flask import Flask, request, jsonify, websocket
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import numpy as np
import jwt
import redis
from werkzeug.exceptions import BadRequest, Unauthorized, PaymentRequired

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app with mobile optimization
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 1MB max upload
CORS(app, origins=['*'])  # Configure properly for production

# Rate limiting with Redis backend
limiter = Limiter(
    app,
    key_func=get_remote_address,
    storage_uri="redis://localhost:6379"
)

# Redis for session management
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Audio processing configuration optimized for mobile
MOBILE_CONFIG = {
    'sample_rate': 48000,
    'frame_size': 960,  # 20ms at 48kHz
    'max_processing_time': 0.05,  # 50ms max
    'buffer_optimization': True,
    'battery_aware': True
}

@dataclass
class MobileSession:
    """Mobile app session management"""
    session_id: str
    user_id: str
    device_id: str
    platform: str  # ios or android
    app_version: str
    subscription_tier: str  # free, pro, studio
    created_at: datetime
    last_activity: datetime
    processing_minutes_used: int
    daily_limit: int

@dataclass
class ProcessingResult:
    """Mobile-optimized processing result"""
    processed_audio: bytes
    metadata: Dict[str, Any]
    processing_time_ms: float
    battery_impact: str  # low, medium, high
    data_usage_bytes: int

class MobileAudioProcessor:
    """Audio processor optimized for mobile constraints"""
    
    def __init__(self):
        self.processing_stats = {
            'total_requests': 0,
            'average_latency': 0,
            'battery_optimizations_applied': 0
        }
    
    def process_audio_mobile(self, audio_data: bytes, session: MobileSession, 
                           settings: Dict[str, Any]) -> ProcessingResult:
        """Process audio with mobile optimizations"""
        start_time = time.time()
        
        try:
            # Convert bytes to numpy array
            if len(audio_data) != MOBILE_CONFIG['frame_size'] * 4:  # 4 bytes per float32
                raise ValueError(f"Expected {MOBILE_CONFIG['frame_size']} samples")
            
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            
            # Apply battery-aware processing
            processing_level = self._get_processing_level(session)
            
            # Core DSP processing (simplified for mobile)
            processed_array = self._apply_mobile_dsp(audio_array, processing_level, settings)
            
            # AI genre classification (optimized)
            genre_info = self._classify_genre_mobile(audio_array, session)
            
            # Calculate metrics
            processing_time = (time.time() - start_time) * 1000
            peak_db = 20 * np.log10(np.max(np.abs(processed_array)) + 1e-10)
            rms_db = 20 * np.log10(np.sqrt(np.mean(processed_array**2)) + 1e-10)
            
            # Determine battery impact
            battery_impact = self._calculate_battery_impact(processing_time, processing_level)
            
            # Create result
            result = ProcessingResult(
                processed_audio=processed_array.tobytes(),
                metadata={
                    'peak_db': float(peak_db),
                    'rms_db': float(rms_db),
                    'genre': genre_info['genre'],
                    'confidence': genre_info['confidence'],
                    'processing_level': processing_level,
                    'optimizations_applied': genre_info.get('optimizations', [])
                },
                processing_time_ms=processing_time,
                battery_impact=battery_impact,
                data_usage_bytes=len(processed_array.tobytes())
            )
            
            # Update stats
            self.processing_stats['total_requests'] += 1
            self.processing_stats['average_latency'] = (
                (self.processing_stats['average_latency'] * (self.processing_stats['total_requests'] - 1) + processing_time) /
                self.processing_stats['total_requests']
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Mobile audio processing failed: {e}")
            raise
    
    def _get_processing_level(self, session: MobileSession) -> str:
        """Determine processing level based on subscription and battery"""
        if session.subscription_tier == 'studio':
            return 'high_quality'
        elif session.subscription_tier == 'pro':
            return 'balanced'
        else:
            return 'battery_saver'
    
    def _apply_mobile_dsp(self, audio: np.ndarray, level: str, settings: Dict[str, Any]) -> np.ndarray:
        """Apply DSP processing optimized for mobile"""
        processed = audio.copy()
        
        if level == 'battery_saver':
            # Minimal processing for battery savings
            # Simple noise gate
            gate_threshold = settings.get('noise_gate_threshold', -40)
            processed = np.where(np.abs(processed) > 10**(gate_threshold/20), processed, 0)
            
            # Basic limiting
            processed = np.clip(processed, -0.95, 0.95)
            
        elif level == 'balanced':
            # Moderate processing
            # Noise gate with hysteresis
            gate_threshold = settings.get('noise_gate_threshold', -35)
            processed = self._apply_noise_gate(processed, gate_threshold)
            
            # Simple compressor
            processed = self._apply_simple_compressor(processed, ratio=3.0, threshold=-20)
            
            # Basic EQ
            processed = self._apply_simple_eq(processed, settings.get('eq_settings', {}))
            
        else:  # high_quality
            # Full processing chain
            processed = self._apply_noise_gate(processed, -30)
            processed = self._apply_compressor(processed, ratio=4.0, threshold=-18)
            processed = self._apply_eq(processed, settings.get('eq_settings', {}))
            processed = self._apply_limiter(processed, ceiling=-0.3)
        
        return processed
    
    def _apply_noise_gate(self, audio: np.ndarray, threshold_db: float) -> np.ndarray:
        """Mobile-optimized noise gate"""
        threshold_linear = 10**(threshold_db/20)
        return np.where(np.abs(audio) > threshold_linear, audio, audio * 0.1)
    
    def _apply_simple_compressor(self, audio: np.ndarray, ratio: float, threshold: float) -> np.ndarray:
        """Simplified compressor for mobile"""
        threshold_linear = 10**(threshold/20)
        compressed = audio.copy()
        
        over_threshold = np.abs(audio) > threshold_linear
        compressed[over_threshold] = (
            np.sign(audio[over_threshold]) * threshold_linear *
            (np.abs(audio[over_threshold]) / threshold_linear) ** (1/ratio)
        )
        
        return compressed
    
    def _apply_compressor(self, audio: np.ndarray, ratio: float, threshold: float) -> np.ndarray:
        """Full compressor with attack/release"""
        # Simplified implementation for mobile
        return self._apply_simple_compressor(audio, ratio, threshold)
    
    def _apply_simple_eq(self, audio: np.ndarray, eq_settings: Dict[str, Any]) -> np.ndarray:
        """Basic 3-band EQ for mobile"""
        # Simplified frequency shaping
        low_gain = eq_settings.get('low', 0)
        mid_gain = eq_settings.get('mid', 0)
        high_gain = eq_settings.get('high', 0)
        
        # Very basic frequency emphasis (not true filtering)
        if low_gain != 0:
            audio *= (1 + low_gain * 0.1)
        if mid_gain != 0:
            audio *= (1 + mid_gain * 0.1)
        if high_gain != 0:
            audio *= (1 + high_gain * 0.1)
            
        return audio
    
    def _apply_eq(self, audio: np.ndarray, eq_settings: Dict[str, Any]) -> np.ndarray:
        """Full EQ processing"""
        # For now, use simplified EQ
        return self._apply_simple_eq(audio, eq_settings)
    
    def _apply_limiter(self, audio: np.ndarray, ceiling: float) -> np.ndarray:
        """Mobile-optimized limiter"""
        ceiling_linear = 10**(ceiling/20)
        return np.clip(audio, -ceiling_linear, ceiling_linear)
    
    def _classify_genre_mobile(self, audio: np.ndarray, session: MobileSession) -> Dict[str, Any]:
        """Mobile-optimized genre classification"""
        if session.subscription_tier == 'free':
            # Basic classification for free tier
            energy = np.mean(audio**2)
            
            if energy < 0.001:
                return {'genre': 'speech', 'confidence': 0.6, 'optimizations': ['energy_based']}
            elif energy > 0.1:
                return {'genre': 'rock', 'confidence': 0.7, 'optimizations': ['energy_based']}
            else:
                return {'genre': 'jazz', 'confidence': 0.5, 'optimizations': ['energy_based']}
        else:
            # Enhanced classification for paid tiers
            # Extract basic features
            energy = np.mean(audio**2)
            spectral_centroid = self._calculate_spectral_centroid(audio)
            zero_crossing_rate = self._calculate_zcr(audio)
            
            # Simple rule-based classification
            if spectral_centroid < 1000:
                genre = 'speech'
                confidence = 0.8
            elif energy > 0.1 and spectral_centroid > 2000:
                genre = 'rock'
                confidence = 0.85
            elif zero_crossing_rate < 0.1:
                genre = 'classical'
                confidence = 0.75
            elif spectral_centroid > 3000:
                genre = 'electronic'
                confidence = 0.8
            else:
                genre = 'jazz'
                confidence = 0.7
            
            return {
                'genre': genre, 
                'confidence': confidence,
                'optimizations': ['spectral_analysis', 'feature_extraction']
            }
    
    def _calculate_spectral_centroid(self, audio: np.ndarray) -> float:
        """Simple spectral centroid calculation"""
        fft = np.abs(np.fft.fft(audio))
        freqs = np.fft.fftfreq(len(audio), 1/MOBILE_CONFIG['sample_rate'])
        return np.sum(freqs[:len(freqs)//2] * fft[:len(fft)//2]) / np.sum(fft[:len(fft)//2])
    
    def _calculate_zcr(self, audio: np.ndarray) -> float:
        """Zero crossing rate calculation"""
        return np.mean(np.abs(np.diff(np.sign(audio))))
    
    def _calculate_battery_impact(self, processing_time: float, level: str) -> str:
        """Estimate battery impact"""
        if level == 'battery_saver' or processing_time < 20:
            return 'low'
        elif level == 'balanced' or processing_time < 40:
            return 'medium'
        else:
            return 'high'

# Initialize processor
audio_processor = MobileAudioProcessor()

# Session management
def create_session(device_id: str, platform: str, app_version: str) -> MobileSession:
    """Create new mobile session"""
    session_id = f"mobile_{int(time.time())}_{device_id[:8]}"
    
    session = MobileSession(
        session_id=session_id,
        user_id=f"user_{device_id}",  # Anonymous user based on device
        device_id=device_id,
        platform=platform,
        app_version=app_version,
        subscription_tier='free',  # Default to free
        created_at=datetime.utcnow(),
        last_activity=datetime.utcnow(),
        processing_minutes_used=0,
        daily_limit=30  # 30 minutes for free tier
    )
    
    # Store in Redis with 24-hour expiry
    redis_client.setex(
        f"session:{session_id}",
        86400,  # 24 hours
        json.dumps(asdict(session), default=str)
    )
    
    return session

def get_session(session_id: str) -> Optional[MobileSession]:
    """Retrieve mobile session"""
    try:
        session_data = redis_client.get(f"session:{session_id}")
        if session_data:
            data = json.loads(session_data)
            # Convert datetime strings back to datetime objects
            data['created_at'] = datetime.fromisoformat(data['created_at'])
            data['last_activity'] = datetime.fromisoformat(data['last_activity'])
            return MobileSession(**data)
        return None
    except Exception as e:
        logger.error(f"Session retrieval failed: {e}")
        return None

def update_session_activity(session: MobileSession):
    """Update session last activity"""
    session.last_activity = datetime.utcnow()
    redis_client.setex(
        f"session:{session.session_id}",
        86400,
        json.dumps(asdict(session), default=str)
    )

# API Endpoints
@app.route('/mobile/v1/session', methods=['POST'])
@limiter.limit("10 per minute")
def create_mobile_session():
    """Create new mobile session"""
    try:
        data = request.get_json()
        
        required_fields = ['device_id', 'platform', 'app_version']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        session = create_session(
            device_id=data['device_id'],
            platform=data['platform'],
            app_version=data['app_version']
        )
        
        # Generate JWT token
        token = jwt.encode({
            'session_id': session.session_id,
            'device_id': session.device_id,
            'exp': datetime.utcnow() + timedelta(days=1)
        }, 'mobile_secret_key', algorithm='HS256')
        
        return jsonify({
            'session_id': session.session_id,
            'token': token,
            'expires_at': (datetime.utcnow() + timedelta(days=1)).isoformat(),
            'subscription_tier': session.subscription_tier,
            'daily_limit_minutes': session.daily_limit,
            'processing_minutes_used': session.processing_minutes_used
        })
        
    except Exception as e:
        logger.error(f"Session creation failed: {e}")
        return jsonify({'error': 'Session creation failed'}), 500

@app.route('/mobile/v1/process', methods=['POST'])
@limiter.limit("60 per minute")  # Free tier limit
def process_audio():
    """Process audio for mobile app"""
    try:
        # Verify authentication
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing or invalid authorization'}), 401
        
        token = auth_header.split(' ')[1]
        try:
            payload = jwt.decode(token, 'mobile_secret_key', algorithms=['HS256'])
            session_id = payload['session_id']
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        
        # Get session
        session = get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 401
        
        # Check usage limits
        if session.subscription_tier == 'free':
            if session.processing_minutes_used >= session.daily_limit:
                return jsonify({
                    'error': 'Daily processing limit exceeded',
                    'subscription_required': True,
                    'upgrade_url': '/subscription/upgrade'
                }), 402
        
        # Get audio data
        if request.content_type != 'application/octet-stream':
            return jsonify({'error': 'Content-Type must be application/octet-stream'}), 400
        
        audio_data = request.get_data()
        if len(audio_data) == 0:
            return jsonify({'error': 'No audio data provided'}), 400
        
        # Get processing settings
        settings = {}
        if 'X-Processing-Settings' in request.headers:
            try:
                settings = json.loads(request.headers['X-Processing-Settings'])
            except json.JSONDecodeError:
                pass
        
        # Process audio
        result = audio_processor.process_audio_mobile(audio_data, session, settings)
        
        # Update session usage
        session.processing_minutes_used += 1  # Increment by 1 minute per request
        update_session_activity(session)
        
        # Return result
        return jsonify({
            'processed_buffer': base64.b64encode(result.processed_audio).decode('utf-8'),
            'metadata': result.metadata,
            'performance': {
                'processing_time_ms': result.processing_time_ms,
                'battery_impact': result.battery_impact,
                'data_usage_bytes': result.data_usage_bytes
            },
            'usage': {
                'processing_minutes_used': session.processing_minutes_used,
                'daily_limit_minutes': session.daily_limit,
                'subscription_tier': session.subscription_tier
            }
        })
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        return jsonify({'error': 'Processing failed'}), 500

@app.route('/mobile/v1/preferences', methods=['GET'])
@limiter.limit("30 per minute")
def get_preferences():
    """Get user preferences"""
    # Authentication check (simplified)
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return jsonify({'error': 'Authentication required'}), 401
    
    return jsonify({
        'genre_preference': 'auto',
        'dsp_settings': {
            'noise_gate': {'enabled': True, 'threshold': -35},
            'compressor': {'enabled': True, 'ratio': 4.0, 'threshold': -18},
            'eq': {'enabled': True, 'low': 0, 'mid': 0, 'high': 0},
            'limiter': {'enabled': True, 'ceiling': -0.3}
        },
        'battery_optimization': True,
        'processing_quality': 'balanced'
    })

@app.route('/mobile/v1/subscription', methods=['GET'])
@limiter.limit("30 per minute")
def get_subscription():
    """Get subscription status"""
    # Authentication and session retrieval (simplified)
    return jsonify({
        'tier': 'free',
        'expires_at': None,
        'features': [
            'basic_processing',
            'genre_classification',
            '30_minutes_daily'
        ],
        'upgrade_options': [
            {
                'tier': 'pro',
                'price': 4.99,
                'currency': 'USD',
                'billing_period': 'monthly',
                'features': ['8_hours_daily', 'advanced_dsp', 'export_quality']
            },
            {
                'tier': 'studio',
                'price': 9.99,
                'currency': 'USD',
                'billing_period': 'monthly',
                'features': ['unlimited', 'pro_effects', 'batch_processing']
            }
        ]
    })

@app.route('/mobile/v1/health', methods=['GET'])
def mobile_health():
    """Mobile API health check"""
    return jsonify({
        'status': 'healthy',
        'api_version': '1.0.0',
        'server_time': datetime.utcnow().isoformat(),
        'performance': {
            'average_latency_ms': audio_processor.processing_stats['average_latency'],
            'total_requests': audio_processor.processing_stats['total_requests'],
            'battery_optimizations': audio_processor.processing_stats['battery_optimizations_applied']
        },
        'supported_platforms': ['ios', 'android'],
        'supported_formats': ['float32_pcm_48khz']
    })

@app.route('/mobile/v1/analytics/usage', methods=['POST'])
@limiter.limit("100 per minute")
def track_usage():
    """Track anonymous usage analytics"""
    try:
        data = request.get_json()
        
        # Log analytics (in production, send to analytics service)
        logger.info(f"Usage analytics: {json.dumps(data)}")
        
        return jsonify({'status': 'recorded'})
        
    except Exception as e:
        logger.error(f"Analytics tracking failed: {e}")
        return jsonify({'error': 'Tracking failed'}), 500

# WebSocket support for real-time streaming will be added next
@app.route('/mobile/v1/stream')
def stream_audio():
    """WebSocket endpoint for real-time audio streaming"""
    # This will be implemented with WebSocket support
    return jsonify({
        'message': 'WebSocket streaming endpoint - upgrade connection to WebSocket',
        'upgrade_url': 'ws://localhost:5000/mobile/v1/stream',
        'protocol': 'ws'
    })

if __name__ == '__main__':
    logger.info("Starting AI Mixer Mobile API Server...")
    logger.info(f"Mobile optimization config: {MOBILE_CONFIG}")
    
    # Start server optimized for mobile
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # Disable debug for performance
        threaded=True
    )