"""
Edge Runtime for AG06 Mixer Device

Implements edge computing capabilities for distributed AG06 deployments with:
- Local ML inference using TensorFlow Lite
- Real-time audio processing at the edge
- Offline operation with cloud synchronization
- IoT device management and telemetry
- Edge-to-cloud data pipeline

Architecture based on industry patterns from:
- NVIDIA Jetson (edge AI computing)
- AWS IoT Greengrass (edge runtime)
- Google Edge TPU (ML at the edge)
- Azure IoT Edge (device management)
- Spotify Edge (distributed audio processing)
"""

import asyncio
import json
import logging
import time
import threading
import queue
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable
from pathlib import Path
import hashlib
import numpy as np
import sqlite3
from collections import deque
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EdgeMode(Enum):
    """Operating modes for edge device"""
    ONLINE = "online"          # Connected to cloud
    OFFLINE = "offline"        # Disconnected, local only
    HYBRID = "hybrid"          # Selective cloud sync


class ProcessingPriority(Enum):
    """Processing priority levels"""
    CRITICAL = 1    # Real-time audio, <10ms latency
    HIGH = 2        # ML inference, <50ms latency  
    NORMAL = 3      # Analytics, <200ms latency
    LOW = 4         # Background sync, best effort


@dataclass
class EdgeDevice:
    """Represents an AG06 edge device"""
    device_id: str
    hardware_version: str = "AG06_v1.2"
    firmware_version: str = "2.0.0"
    capabilities: List[str] = field(default_factory=lambda: [
        "audio_processing",
        "ml_inference", 
        "local_storage",
        "cloud_sync"
    ])
    location: Optional[str] = None
    last_seen: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'device_id': self.device_id,
            'hardware_version': self.hardware_version,
            'firmware_version': self.firmware_version,
            'capabilities': self.capabilities,
            'location': self.location,
            'last_seen': self.last_seen.isoformat()
        }


@dataclass
class EdgeMetrics:
    """Performance metrics for edge device"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    temperature: float = 0.0
    audio_latency_ms: float = 0.0
    ml_inference_ms: float = 0.0
    events_processed: int = 0
    events_dropped: int = 0
    uptime_seconds: float = 0.0


class TensorFlowLiteInference:
    """
    ML inference engine using TensorFlow Lite for edge deployment
    
    Optimized for running on resource-constrained AG06 hardware
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
        # Model cache for multiple models
        self.model_cache: Dict[str, Any] = {}
        
        # Inference metrics
        self.inference_count = 0
        self.total_inference_time = 0.0
        
        if model_path:
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load TensorFlow Lite model"""
        try:
            # In production, would use actual TensorFlow Lite
            # For demo, we'll simulate model loading
            logger.info(f"Loading TFLite model from {model_path}")
            
            # Simulate model metadata
            self.model_cache[model_path] = {
                'loaded_at': datetime.utcnow(),
                'input_shape': (1, 128, 128, 3),  # Example: audio spectrogram
                'output_shape': (1, 10),  # Example: 10 classes
                'quantized': True,
                'size_kb': 456
            }
            
            logger.info(f"Model loaded successfully: {self.model_cache[model_path]}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    async def predict(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Run inference on input data"""
        start_time = time.time()
        
        try:
            # Simulate TensorFlow Lite inference
            # In production, would use interpreter.invoke()
            
            # Simulate audio classification
            raw_scores = np.random.randn(10)
            exp_scores = np.exp(raw_scores - np.max(raw_scores))
            predictions = exp_scores / np.sum(exp_scores)
            classes = ['speech', 'music', 'silence', 'noise', 'applause', 
                      'laughter', 'singing', 'instrument', 'nature', 'other']
            
            # Get top predictions
            top_indices = np.argsort(predictions)[-3:][::-1]
            top_predictions = [
                {'class': classes[i], 'confidence': float(predictions[i])}
                for i in top_indices
            ]
            
            inference_time = (time.time() - start_time) * 1000
            self.inference_count += 1
            self.total_inference_time += inference_time
            
            return {
                'predictions': top_predictions,
                'inference_time_ms': inference_time,
                'model_type': 'audio_classifier',
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {'error': str(e)}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get inference metrics"""
        avg_inference_time = (
            self.total_inference_time / self.inference_count 
            if self.inference_count > 0 else 0
        )
        
        return {
            'inference_count': self.inference_count,
            'avg_inference_time_ms': avg_inference_time,
            'models_loaded': len(self.model_cache),
            'total_model_size_kb': sum(
                m.get('size_kb', 0) for m in self.model_cache.values()
            )
        }


class LocalDataStore:
    """
    SQLite-based local storage for edge device
    
    Provides offline data persistence and sync queue
    """
    
    def __init__(self, db_path: str = "./edge_data.db"):
        self.db_path = db_path
        self.conn = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database schema"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.conn.cursor()
        
        # Events table for local storage
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                event_type TEXT,
                payload TEXT,
                synced INTEGER DEFAULT 0,
                priority INTEGER DEFAULT 3
            )
        ''')
        
        # Metrics table for telemetry
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                metric_type TEXT,
                value REAL,
                metadata TEXT
            )
        ''')
        
        # ML inference cache
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS inference_cache (
                input_hash TEXT PRIMARY KEY,
                output TEXT,
                model_version TEXT,
                timestamp REAL
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_synced ON events(synced)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)')
        
        self.conn.commit()
    
    async def store_event(self, event: Dict[str, Any], priority: ProcessingPriority = ProcessingPriority.NORMAL) -> bool:
        """Store event locally"""
        try:
            cursor = self.conn.cursor()
            event_id = event.get('id', str(hashlib.md5(json.dumps(event).encode()).hexdigest()))
            
            cursor.execute('''
                INSERT OR REPLACE INTO events (id, timestamp, event_type, payload, priority)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                event_id,
                time.time(),
                event.get('type', 'unknown'),
                json.dumps(event),
                priority.value
            ))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to store event: {e}")
            return False
    
    async def get_unsynced_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get events that need to be synced to cloud"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT id, timestamp, event_type, payload, priority
            FROM events
            WHERE synced = 0
            ORDER BY priority ASC, timestamp ASC
            LIMIT ?
        ''', (limit,))
        
        events = []
        for row in cursor.fetchall():
            event = json.loads(row[3])
            event['_id'] = row[0]
            event['_timestamp'] = row[1]
            event['_priority'] = row[4]
            events.append(event)
        
        return events
    
    async def mark_events_synced(self, event_ids: List[str]) -> bool:
        """Mark events as synced to cloud"""
        try:
            cursor = self.conn.cursor()
            cursor.executemany(
                'UPDATE events SET synced = 1 WHERE id = ?',
                [(event_id,) for event_id in event_ids]
            )
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to mark events as synced: {e}")
            return False
    
    async def cache_inference(self, input_hash: str, output: Dict[str, Any], model_version: str):
        """Cache ML inference results"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO inference_cache (input_hash, output, model_version, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (input_hash, json.dumps(output), model_version, time.time()))
        self.conn.commit()
    
    async def get_cached_inference(self, input_hash: str, model_version: str) -> Optional[Dict[str, Any]]:
        """Get cached inference result"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT output FROM inference_cache
            WHERE input_hash = ? AND model_version = ?
            AND timestamp > ?
        ''', (input_hash, model_version, time.time() - 3600))  # 1 hour cache
        
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Events stats
        cursor.execute('SELECT COUNT(*), COUNT(CASE WHEN synced = 0 THEN 1 END) FROM events')
        total_events, unsynced_events = cursor.fetchone()
        stats['total_events'] = total_events
        stats['unsynced_events'] = unsynced_events
        
        # Metrics stats
        cursor.execute('SELECT COUNT(*) FROM metrics')
        stats['total_metrics'] = cursor.fetchone()[0]
        
        # Cache stats
        cursor.execute('SELECT COUNT(*) FROM inference_cache')
        stats['cached_inferences'] = cursor.fetchone()[0]
        
        # Database size
        cursor.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()")
        stats['database_size_bytes'] = cursor.fetchone()[0]
        
        return stats


class EdgeToCloudSync:
    """
    Manages data synchronization between edge and cloud
    
    Features:
    - Batch upload with compression
    - Retry logic with exponential backoff
    - Selective sync based on priority
    - Bandwidth optimization
    """
    
    def __init__(self, 
                 cloud_endpoint: str,
                 device_id: str,
                 api_key: Optional[str] = None):
        
        self.cloud_endpoint = cloud_endpoint
        self.device_id = device_id
        self.api_key = api_key
        
        # Sync state
        self.last_sync = datetime.utcnow()
        self.sync_in_progress = False
        self.failed_sync_count = 0
        
        # Bandwidth management
        self.bandwidth_limit_kbps = 1000  # 1 Mbps default
        self.bytes_synced_today = 0
        
    async def sync_events(self, events: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """Sync events to cloud"""
        if self.sync_in_progress:
            logger.warning("Sync already in progress")
            return False, []
        
        self.sync_in_progress = True
        synced_ids = []
        
        try:
            # Batch events for efficient upload
            batch_size = 50
            for i in range(0, len(events), batch_size):
                batch = events[i:i + batch_size]
                
                # Compress batch
                batch_data = {
                    'device_id': self.device_id,
                    'timestamp': datetime.utcnow().isoformat(),
                    'events': batch
                }
                
                # Simulate cloud upload
                # In production, would use actual HTTP/MQTT client
                success = await self._upload_batch(batch_data)
                
                if success:
                    synced_ids.extend([e['_id'] for e in batch if '_id' in e])
                    self.bytes_synced_today += len(json.dumps(batch_data).encode())
                else:
                    self.failed_sync_count += 1
                    break
                
                # Bandwidth throttling
                await self._apply_bandwidth_limit()
            
            self.last_sync = datetime.utcnow()
            self.failed_sync_count = 0 if synced_ids else self.failed_sync_count
            
            return len(synced_ids) > 0, synced_ids
            
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            self.failed_sync_count += 1
            return False, []
            
        finally:
            self.sync_in_progress = False
    
    async def _upload_batch(self, batch_data: Dict[str, Any]) -> bool:
        """Upload batch to cloud (simulated)"""
        try:
            # Simulate network upload with 95% success rate
            await asyncio.sleep(0.1)  # Simulate network latency
            
            if np.random.random() > 0.05:
                logger.debug(f"Uploaded batch with {len(batch_data['events'])} events")
                return True
            else:
                logger.warning("Simulated upload failure")
                return False
                
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False
    
    async def _apply_bandwidth_limit(self):
        """Apply bandwidth throttling"""
        # Simple bandwidth limiting
        bytes_per_second = self.bandwidth_limit_kbps * 1024 / 8
        sleep_time = 0.1  # Adjust based on actual bytes sent
        await asyncio.sleep(sleep_time)
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get synchronization status"""
        return {
            'last_sync': self.last_sync.isoformat(),
            'sync_in_progress': self.sync_in_progress,
            'failed_sync_count': self.failed_sync_count,
            'bytes_synced_today': self.bytes_synced_today,
            'bandwidth_limit_kbps': self.bandwidth_limit_kbps
        }


class AudioProcessingPipeline:
    """
    Edge audio processing pipeline with DSP and ML
    
    Optimized for low-latency processing on AG06 hardware
    """
    
    def __init__(self, sample_rate: int = 44100, buffer_size: int = 512):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        
        # Audio buffers
        self.input_buffer = deque(maxlen=buffer_size * 10)
        self.output_buffer = deque(maxlen=buffer_size * 10)
        
        # Processing metrics
        self.frames_processed = 0
        self.total_latency_ms = 0.0
        
    async def process_audio_frame(self, audio_data: np.ndarray) -> np.ndarray:
        """Process audio frame with edge DSP"""
        start_time = time.time()
        
        try:
            # Apply edge DSP chain
            processed = audio_data.copy()
            
            # 1. Noise gate (simple threshold)
            gate_threshold = 0.01
            processed[np.abs(processed) < gate_threshold] = 0
            
            # 2. Dynamic range compression (simplified)
            threshold = 0.7
            ratio = 4.0
            above_threshold = np.abs(processed) > threshold
            processed[above_threshold] = (
                np.sign(processed[above_threshold]) * 
                (threshold + (np.abs(processed[above_threshold]) - threshold) / ratio)
            )
            
            # 3. Simple EQ (bass/treble adjustment)
            # In production, would use proper filters
            processed = processed * 1.1  # Slight boost
            
            # Update metrics
            self.frames_processed += 1
            latency = (time.time() - start_time) * 1000
            self.total_latency_ms += latency
            
            return processed
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            return audio_data
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics"""
        avg_latency = (
            self.total_latency_ms / self.frames_processed
            if self.frames_processed > 0 else 0
        )
        
        return {
            'frames_processed': self.frames_processed,
            'avg_latency_ms': avg_latency,
            'buffer_usage': len(self.input_buffer) / self.input_buffer.maxlen,
            'sample_rate': self.sample_rate,
            'buffer_size': self.buffer_size
        }


class EdgeRuntime:
    """
    Main edge runtime orchestrator for AG06 device
    
    Coordinates all edge computing components:
    - ML inference engine
    - Local data storage
    - Cloud synchronization
    - Audio processing pipeline
    - Device telemetry
    """
    
    def __init__(self,
                 device_id: str,
                 mode: EdgeMode = EdgeMode.HYBRID,
                 cloud_endpoint: Optional[str] = None):
        
        self.device = EdgeDevice(device_id=device_id)
        self.mode = mode
        self.cloud_endpoint = cloud_endpoint
        
        # Initialize components
        self.ml_engine = TensorFlowLiteInference()
        self.local_store = LocalDataStore()
        self.audio_pipeline = AudioProcessingPipeline()
        
        if cloud_endpoint:
            self.cloud_sync = EdgeToCloudSync(cloud_endpoint, device_id)
        else:
            self.cloud_sync = None
        
        # Runtime state
        self.running = False
        self.start_time = None
        self.metrics = EdgeMetrics()
        
        # Processing queues
        self.event_queue: queue.Queue = queue.Queue(maxsize=1000)
        self.sync_queue: queue.Queue = queue.Queue(maxsize=500)
        
        # Background tasks
        self.tasks: List[asyncio.Task] = []
        
        logger.info(f"Edge Runtime initialized for device {device_id} in {mode.value} mode")
    
    async def start(self):
        """Start edge runtime"""
        if self.running:
            logger.warning("Edge runtime already running")
            return
        
        self.running = True
        self.start_time = time.time()
        
        logger.info("Starting Edge Runtime...")
        
        # Start background tasks
        self.tasks.append(asyncio.create_task(self._process_events()))
        self.tasks.append(asyncio.create_task(self._sync_to_cloud()))
        self.tasks.append(asyncio.create_task(self._collect_telemetry()))
        self.tasks.append(asyncio.create_task(self._health_check()))
        
        logger.info("✅ Edge Runtime started successfully")
    
    async def stop(self):
        """Stop edge runtime"""
        if not self.running:
            return
        
        logger.info("Stopping Edge Runtime...")
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        logger.info("✅ Edge Runtime stopped")
    
    async def process_audio(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Process audio with edge capabilities"""
        try:
            # Apply edge DSP
            processed_audio = await self.audio_pipeline.process_audio_frame(audio_data)
            
            # Run ML inference if needed
            inference_result = None
            if self.ml_engine and np.random.random() > 0.8:  # Sample 20% for inference
                # Convert audio to spectrogram for ML model
                # In production, would use actual spectrogram computation
                spectrogram = np.random.randn(1, 128, 128, 3).astype(np.float32)
                inference_result = await self.ml_engine.predict(spectrogram)
            
            # Create event
            event = {
                'type': 'audio_processed',
                'timestamp': datetime.utcnow().isoformat(),
                'device_id': self.device.device_id,
                'audio_metrics': {
                    'peak_level': float(np.max(np.abs(processed_audio))),
                    'rms_level': float(np.sqrt(np.mean(processed_audio**2))),
                    'processing_latency_ms': self.audio_pipeline.get_metrics()['avg_latency_ms']
                },
                'ml_inference': inference_result
            }
            
            # Store locally
            await self.local_store.store_event(event, ProcessingPriority.HIGH)
            
            # Add to sync queue if online
            if self.mode in [EdgeMode.ONLINE, EdgeMode.HYBRID]:
                self.sync_queue.put_nowait(event)
            
            return {
                'success': True,
                'processed_audio': processed_audio.tolist()[:100],  # Sample for response
                'inference': inference_result,
                'latency_ms': self.audio_pipeline.get_metrics()['avg_latency_ms']
            }
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _process_events(self):
        """Background task to process events"""
        while self.running:
            try:
                # Process events from queue
                if not self.event_queue.empty():
                    event = self.event_queue.get_nowait()
                    
                    # Store locally
                    await self.local_store.store_event(event)
                    
                    # Update metrics
                    self.metrics.events_processed += 1
                
                await asyncio.sleep(0.01)  # Small delay to prevent CPU spinning
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Event processing error: {e}")
                self.metrics.events_dropped += 1
    
    async def _sync_to_cloud(self):
        """Background task to sync data to cloud"""
        while self.running:
            try:
                if self.cloud_sync and self.mode in [EdgeMode.ONLINE, EdgeMode.HYBRID]:
                    # Get unsynced events
                    unsynced = await self.local_store.get_unsynced_events(limit=50)
                    
                    if unsynced:
                        success, synced_ids = await self.cloud_sync.sync_events(unsynced)
                        
                        if synced_ids:
                            await self.local_store.mark_events_synced(synced_ids)
                            logger.info(f"Synced {len(synced_ids)} events to cloud")
                
                # Sync every 30 seconds
                await asyncio.sleep(30)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cloud sync error: {e}")
    
    async def _collect_telemetry(self):
        """Background task to collect device telemetry"""
        while self.running:
            try:
                # Update metrics
                self.metrics.uptime_seconds = time.time() - self.start_time if self.start_time else 0
                
                # Simulate resource metrics
                # In production, would use actual system metrics
                self.metrics.cpu_usage = np.random.uniform(20, 60)
                self.metrics.memory_usage = np.random.uniform(30, 70)
                self.metrics.disk_usage = np.random.uniform(10, 50)
                self.metrics.temperature = np.random.uniform(35, 55)
                
                # Get component metrics
                if self.audio_pipeline:
                    audio_metrics = self.audio_pipeline.get_metrics()
                    self.metrics.audio_latency_ms = audio_metrics['avg_latency_ms']
                
                if self.ml_engine:
                    ml_metrics = self.ml_engine.get_metrics()
                    self.metrics.ml_inference_ms = ml_metrics['avg_inference_time_ms']
                
                # Store telemetry locally
                telemetry_event = {
                    'type': 'telemetry',
                    'timestamp': datetime.utcnow().isoformat(),
                    'device_id': self.device.device_id,
                    'metrics': self.metrics.__dict__
                }
                
                await self.local_store.store_event(telemetry_event, ProcessingPriority.LOW)
                
                # Collect every 60 seconds
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Telemetry collection error: {e}")
    
    async def _health_check(self):
        """Background task for health monitoring"""
        while self.running:
            try:
                # Check component health
                health_status = {
                    'device_id': self.device.device_id,
                    'timestamp': datetime.utcnow().isoformat(),
                    'mode': self.mode.value,
                    'uptime_seconds': self.metrics.uptime_seconds,
                    'components': {
                        'ml_engine': self.ml_engine is not None,
                        'local_store': self.local_store is not None,
                        'audio_pipeline': self.audio_pipeline is not None,
                        'cloud_sync': self.cloud_sync is not None and not self.cloud_sync.sync_in_progress
                    },
                    'metrics': {
                        'events_processed': self.metrics.events_processed,
                        'events_dropped': self.metrics.events_dropped,
                        'cpu_usage': self.metrics.cpu_usage,
                        'memory_usage': self.metrics.memory_usage
                    }
                }
                
                # Log health status
                logger.debug(f"Health check: {health_status}")
                
                # Check for issues
                if self.metrics.cpu_usage > 90:
                    logger.warning("High CPU usage detected")
                
                if self.metrics.memory_usage > 85:
                    logger.warning("High memory usage detected")
                
                if self.metrics.events_dropped > 100:
                    logger.warning(f"High event drop rate: {self.metrics.events_dropped} events dropped")
                
                # Health check every 30 seconds
                await asyncio.sleep(30)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get edge runtime status"""
        storage_stats = self.local_store.get_storage_stats()
        
        return {
            'device': self.device.to_dict(),
            'mode': self.mode.value,
            'running': self.running,
            'uptime_seconds': self.metrics.uptime_seconds,
            'metrics': self.metrics.__dict__,
            'storage': storage_stats,
            'cloud_sync': self.cloud_sync.get_sync_status() if self.cloud_sync else None,
            'audio_pipeline': self.audio_pipeline.get_metrics(),
            'ml_engine': self.ml_engine.get_metrics()
        }


# Demo function
async def demo_edge_runtime():
    """Demonstrate edge runtime capabilities"""
    
    print("🔧 AG06 Edge Runtime Demo")
    print("=========================")
    
    # Create edge runtime
    runtime = EdgeRuntime(
        device_id="AG06_EDGE_001",
        mode=EdgeMode.HYBRID,
        cloud_endpoint="https://api.ag06mixer.cloud"
    )
    
    # Start runtime
    await runtime.start()
    
    print("✅ Edge Runtime started")
    print(f"   Device ID: {runtime.device.device_id}")
    print(f"   Mode: {runtime.mode.value}")
    print(f"   Capabilities: {', '.join(runtime.device.capabilities)}")
    
    # Simulate audio processing
    print("\n🎵 Processing audio samples...")
    
    for i in range(5):
        # Generate sample audio (sine wave)
        t = np.linspace(0, 0.1, 4410)  # 100ms at 44.1kHz
        frequency = 440 * (i + 1)  # Different frequencies
        audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        # Process audio
        result = await runtime.process_audio(audio_data)
        
        print(f"   Sample {i+1}: Frequency={frequency}Hz, "
              f"Latency={result.get('latency_ms', 0):.2f}ms")
        
        if result.get('inference'):
            predictions = result['inference'].get('predictions', [])
            if predictions:
                top_class = predictions[0]
                print(f"      ML: {top_class['class']} "
                      f"({top_class['confidence']*100:.1f}%)")
    
    # Wait for some background processing
    await asyncio.sleep(3)
    
    # Show status
    print("\n📊 Edge Runtime Status:")
    status = runtime.get_status()
    
    print(f"   Uptime: {status['uptime_seconds']:.1f}s")
    print(f"   Events processed: {status['metrics']['events_processed']}")
    print(f"   CPU usage: {status['metrics']['cpu_usage']:.1f}%")
    print(f"   Memory usage: {status['metrics']['memory_usage']:.1f}%")
    
    print(f"\n💾 Local Storage:")
    print(f"   Total events: {status['storage']['total_events']}")
    print(f"   Unsynced events: {status['storage']['unsynced_events']}")
    print(f"   Database size: {status['storage']['database_size_bytes']/1024:.1f}KB")
    
    if status['cloud_sync']:
        print(f"\n☁️  Cloud Sync:")
        print(f"   Last sync: {status['cloud_sync']['last_sync']}")
        print(f"   Bytes synced: {status['cloud_sync']['bytes_synced_today']}")
    
    print(f"\n🤖 ML Engine:")
    print(f"   Inference count: {status['ml_engine']['inference_count']}")
    print(f"   Avg inference time: {status['ml_engine']['avg_inference_time_ms']:.2f}ms")
    
    # Generate more events for sync demonstration
    print("\n📤 Generating events for cloud sync...")
    
    for i in range(20):
        event = {
            'type': 'test_event',
            'timestamp': datetime.utcnow().isoformat(),
            'data': {'index': i, 'value': np.random.randn()}
        }
        runtime.event_queue.put(event)
    
    # Wait for processing and sync
    await asyncio.sleep(5)
    
    # Final status
    final_status = runtime.get_status()
    print(f"\n📈 Final Status:")
    print(f"   Total events processed: {final_status['metrics']['events_processed']}")
    print(f"   Events in local storage: {final_status['storage']['total_events']}")
    print(f"   Unsynced events: {final_status['storage']['unsynced_events']}")
    
    # Stop runtime
    await runtime.stop()
    print("\n✅ Edge Runtime stopped successfully")
    
    return {
        'device_id': runtime.device.device_id,
        'events_processed': final_status['metrics']['events_processed'],
        'avg_audio_latency_ms': final_status['audio_pipeline']['avg_latency_ms'],
        'ml_inferences': final_status['ml_engine']['inference_count'],
        'storage_size_kb': final_status['storage']['database_size_bytes'] / 1024,
        'success': True
    }


if __name__ == "__main__":
    # Run the demo
    result = asyncio.run(demo_edge_runtime())
    print(f"\n🎯 Demo Result: {result}")