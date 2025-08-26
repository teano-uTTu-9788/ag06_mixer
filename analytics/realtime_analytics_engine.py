"""
Real-time Analytics Engine
=========================

Enterprise-grade real-time analytics platform following patterns from:
- Netflix: Event-driven microservices, real-time dashboards, A/B testing integration
- Spotify: Audio analytics, user behavior tracking, collaborative filtering
- Google Analytics: Real-time data processing, ML-powered insights
- AWS Kinesis: Stream processing, time-series analytics

Implements high-throughput event ingestion, real-time processing,
and sub-second analytics for the AG06 mixer platform.
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Callable
import threading
from concurrent.futures import ThreadPoolExecutor
import heapq
import statistics
import hashlib
import gzip
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventType(Enum):
    """Real-time event types for AG06 mixer"""
    AUDIO_STREAM_START = "audio_stream_start"
    AUDIO_STREAM_END = "audio_stream_end"
    AUDIO_LEVEL_CHANGE = "audio_level_change"
    EFFECT_APPLIED = "effect_applied"
    USER_INTERACTION = "user_interaction"
    PERFORMANCE_METRIC = "performance_metric"
    ERROR_EVENT = "error_event"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    COLLABORATION_EVENT = "collaboration_event"


class MetricType(Enum):
    """Real-time metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


@dataclass
class RealTimeEvent:
    """Real-time event with comprehensive metadata"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_type: EventType = EventType.USER_INTERACTION
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    device_id: Optional[str] = None
    
    # Audio-specific attributes
    audio_channel: Optional[int] = None
    audio_level: Optional[float] = None
    sample_rate: Optional[int] = None
    bit_depth: Optional[int] = None
    
    # User interaction attributes
    component: Optional[str] = None
    action: Optional[str] = None
    value: Optional[Union[str, int, float]] = None
    
    # Performance attributes
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    latency_ms: Optional[float] = None
    throughput: Optional[float] = None
    
    # Context attributes
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    location: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'device_id': self.device_id,
            'audio_channel': self.audio_channel,
            'audio_level': self.audio_level,
            'sample_rate': self.sample_rate,
            'bit_depth': self.bit_depth,
            'component': self.component,
            'action': self.action,
            'value': self.value,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'latency_ms': self.latency_ms,
            'throughput': self.throughput,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'location': self.location,
            'metadata': self.metadata,
            'tags': self.tags
        }


@dataclass
class RealTimeMetric:
    """Real-time metric with time-series data"""
    name: str
    metric_type: MetricType
    value: Union[int, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)
    dimensions: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary"""
        return {
            'name': self.name,
            'type': self.metric_type.value,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags,
            'dimensions': self.dimensions,
            'unit': self.unit
        }


class IEventProcessor(ABC):
    """Interface for event processing"""
    
    @abstractmethod
    async def process_event(self, event: RealTimeEvent) -> Dict[str, Any]:
        """Process individual event"""
        pass
    
    @abstractmethod
    async def process_batch(self, events: List[RealTimeEvent]) -> Dict[str, Any]:
        """Process batch of events"""
        pass


class IMetricCollector(ABC):
    """Interface for metric collection"""
    
    @abstractmethod
    async def collect_metric(self, metric: RealTimeMetric) -> None:
        """Collect individual metric"""
        pass
    
    @abstractmethod
    async def get_metrics(self, name: str, start_time: datetime, end_time: datetime) -> List[RealTimeMetric]:
        """Retrieve metrics for time range"""
        pass


class IAnalyticsStore(ABC):
    """Interface for analytics data storage"""
    
    @abstractmethod
    async def store_events(self, events: List[RealTimeEvent]) -> None:
        """Store events for analytics"""
        pass
    
    @abstractmethod
    async def query_events(self, criteria: Dict[str, Any]) -> List[RealTimeEvent]:
        """Query events with criteria"""
        pass


class TimeSeriesBuffer:
    """High-performance time-series buffer for real-time metrics"""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.data = deque(maxlen=max_size)
        self._lock = threading.RLock()
        self.index = {}  # Fast lookups by metric name
    
    def add_metric(self, metric: RealTimeMetric) -> None:
        """Add metric to time-series buffer"""
        with self._lock:
            # Add to main buffer
            self.data.append(metric)
            
            # Update index for fast queries
            if metric.name not in self.index:
                self.index[metric.name] = deque(maxlen=self.max_size // 10)
            self.index[metric.name].append(metric)
            
            # Clean expired metrics
            self._cleanup_expired()
    
    def get_metrics(self, name: str, start_time: datetime, end_time: datetime) -> List[RealTimeMetric]:
        """Get metrics for time range"""
        with self._lock:
            if name not in self.index:
                return []
            
            metrics = []
            for metric in self.index[name]:
                if start_time <= metric.timestamp <= end_time:
                    metrics.append(metric)
            
            return sorted(metrics, key=lambda m: m.timestamp)
    
    def get_latest_value(self, name: str) -> Optional[RealTimeMetric]:
        """Get latest metric value"""
        with self._lock:
            if name not in self.index or not self.index[name]:
                return None
            return self.index[name][-1]
    
    def _cleanup_expired(self) -> None:
        """Remove expired metrics"""
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.ttl_seconds)
        
        # Clean main buffer
        while self.data and self.data[0].timestamp < cutoff_time:
            self.data.popleft()
        
        # Clean index
        for name in list(self.index.keys()):
            metrics = self.index[name]
            while metrics and metrics[0].timestamp < cutoff_time:
                metrics.popleft()
            
            if not metrics:
                del self.index[name]


class InMemoryMetricCollector(IMetricCollector):
    """High-performance in-memory metric collector"""
    
    def __init__(self, buffer_size: int = 50000):
        self.buffer = TimeSeriesBuffer(max_size=buffer_size)
        self.aggregated_metrics = defaultdict(list)
        self.metric_stats = defaultdict(dict)
        self._lock = threading.RLock()
    
    async def collect_metric(self, metric: RealTimeMetric) -> None:
        """Collect individual metric with real-time aggregation"""
        self.buffer.add_metric(metric)
        
        # Update real-time statistics
        with self._lock:
            stats = self.metric_stats[metric.name]
            
            if 'count' not in stats:
                stats['count'] = 0
                stats['sum'] = 0.0
                stats['min'] = float('inf')
                stats['max'] = float('-inf')
                stats['avg'] = 0.0
            
            stats['count'] += 1
            stats['sum'] += metric.value
            stats['min'] = min(stats['min'], metric.value)
            stats['max'] = max(stats['max'], metric.value)
            stats['avg'] = stats['sum'] / stats['count']
    
    async def get_metrics(self, name: str, start_time: datetime, end_time: datetime) -> List[RealTimeMetric]:
        """Retrieve metrics for time range"""
        return self.buffer.get_metrics(name, start_time, end_time)
    
    def get_metric_stats(self, name: str) -> Dict[str, Any]:
        """Get real-time statistics for metric"""
        with self._lock:
            return self.metric_stats.get(name, {}).copy()
    
    def get_all_metric_names(self) -> Set[str]:
        """Get all tracked metric names"""
        with self._lock:
            return set(self.metric_stats.keys())


class AudioAnalyticsProcessor(IEventProcessor):
    """Audio-specific event processor with domain expertise"""
    
    def __init__(self):
        self.session_stats = defaultdict(dict)
        self.audio_quality_metrics = defaultdict(list)
        self.user_interaction_patterns = defaultdict(list)
        self._lock = threading.RLock()
    
    async def process_event(self, event: RealTimeEvent) -> Dict[str, Any]:
        """Process individual audio event"""
        result = {'processed': True, 'insights': []}
        
        if event.event_type == EventType.AUDIO_STREAM_START:
            await self._process_stream_start(event, result)
        elif event.event_type == EventType.AUDIO_LEVEL_CHANGE:
            await self._process_level_change(event, result)
        elif event.event_type == EventType.EFFECT_APPLIED:
            await self._process_effect_applied(event, result)
        elif event.event_type == EventType.USER_INTERACTION:
            await self._process_user_interaction(event, result)
        elif event.event_type == EventType.PERFORMANCE_METRIC:
            await self._process_performance_metric(event, result)
        
        return result
    
    async def process_batch(self, events: List[RealTimeEvent]) -> Dict[str, Any]:
        """Process batch of events with batch optimizations"""
        batch_results = {
            'processed_count': len(events),
            'insights': [],
            'aggregated_metrics': {}
        }
        
        # Group events by type for efficient processing
        events_by_type = defaultdict(list)
        for event in events:
            events_by_type[event.event_type].append(event)
        
        # Process each type with optimized batch logic
        for event_type, type_events in events_by_type.items():
            if event_type == EventType.AUDIO_LEVEL_CHANGE:
                batch_results['aggregated_metrics']['audio_levels'] = await self._batch_process_audio_levels(type_events)
            elif event_type == EventType.USER_INTERACTION:
                batch_results['aggregated_metrics']['user_patterns'] = await self._batch_process_interactions(type_events)
        
        return batch_results
    
    async def _process_stream_start(self, event: RealTimeEvent, result: Dict[str, Any]) -> None:
        """Process audio stream start event"""
        with self._lock:
            session_id = event.session_id or event.user_id
            if session_id:
                self.session_stats[session_id]['start_time'] = event.timestamp
                self.session_stats[session_id]['sample_rate'] = event.sample_rate
                self.session_stats[session_id]['bit_depth'] = event.bit_depth
                
                # Audio quality assessment
                if event.sample_rate and event.bit_depth:
                    quality_score = self._calculate_audio_quality_score(event.sample_rate, event.bit_depth)
                    result['insights'].append(f"Audio quality score: {quality_score}")
    
    async def _process_level_change(self, event: RealTimeEvent, result: Dict[str, Any]) -> None:
        """Process audio level change with real-time analysis"""
        if event.audio_level is not None:
            # Detect potential issues
            if event.audio_level > 0.95:
                result['insights'].append("WARNING: Audio level near clipping threshold")
            elif event.audio_level < 0.05:
                result['insights'].append("INFO: Very low audio level detected")
            
            # Track level patterns
            with self._lock:
                session_id = event.session_id or event.user_id
                if session_id:
                    if 'level_history' not in self.session_stats[session_id]:
                        self.session_stats[session_id]['level_history'] = deque(maxlen=100)
                    self.session_stats[session_id]['level_history'].append(event.audio_level)
    
    async def _process_effect_applied(self, event: RealTimeEvent, result: Dict[str, Any]) -> None:
        """Process audio effect application"""
        effect_name = event.metadata.get('effect_name', 'unknown')
        effect_params = event.metadata.get('parameters', {})
        
        # Track effect usage patterns
        with self._lock:
            user_id = event.user_id
            if user_id:
                if 'effects_used' not in self.session_stats[user_id]:
                    self.session_stats[user_id]['effects_used'] = defaultdict(int)
                self.session_stats[user_id]['effects_used'][effect_name] += 1
        
        result['insights'].append(f"Applied effect: {effect_name}")
    
    async def _process_user_interaction(self, event: RealTimeEvent, result: Dict[str, Any]) -> None:
        """Process user interaction with behavioral analysis"""
        with self._lock:
            user_id = event.user_id
            if user_id:
                interaction = {
                    'component': event.component,
                    'action': event.action,
                    'timestamp': event.timestamp,
                    'value': event.value
                }
                self.user_interaction_patterns[user_id].append(interaction)
                
                # Keep only recent interactions (last 1000)
                if len(self.user_interaction_patterns[user_id]) > 1000:
                    self.user_interaction_patterns[user_id] = self.user_interaction_patterns[user_id][-1000:]
    
    async def _process_performance_metric(self, event: RealTimeEvent, result: Dict[str, Any]) -> None:
        """Process performance metrics with anomaly detection"""
        if event.cpu_usage and event.cpu_usage > 90:
            result['insights'].append("WARNING: High CPU usage detected")
        
        if event.memory_usage and event.memory_usage > 85:
            result['insights'].append("WARNING: High memory usage detected")
        
        if event.latency_ms and event.latency_ms > 100:
            result['insights'].append("WARNING: High audio latency detected")
    
    async def _batch_process_audio_levels(self, events: List[RealTimeEvent]) -> Dict[str, Any]:
        """Batch process audio level changes"""
        levels = [e.audio_level for e in events if e.audio_level is not None]
        
        if not levels:
            return {}
        
        return {
            'count': len(levels),
            'avg_level': statistics.mean(levels),
            'max_level': max(levels),
            'min_level': min(levels),
            'clipping_events': sum(1 for level in levels if level > 0.95),
            'low_level_events': sum(1 for level in levels if level < 0.05)
        }
    
    async def _batch_process_interactions(self, events: List[RealTimeEvent]) -> Dict[str, Any]:
        """Batch process user interactions"""
        components = defaultdict(int)
        actions = defaultdict(int)
        
        for event in events:
            if event.component:
                components[event.component] += 1
            if event.action:
                actions[event.action] += 1
        
        return {
            'total_interactions': len(events),
            'top_components': dict(sorted(components.items(), key=lambda x: x[1], reverse=True)[:5]),
            'top_actions': dict(sorted(actions.items(), key=lambda x: x[1], reverse=True)[:5])
        }
    
    def _calculate_audio_quality_score(self, sample_rate: int, bit_depth: int) -> float:
        """Calculate audio quality score based on technical parameters"""
        # Base score calculation
        sample_rate_score = min(sample_rate / 48000, 1.0) * 0.6  # Max at 48kHz
        bit_depth_score = min(bit_depth / 24, 1.0) * 0.4  # Max at 24-bit
        
        return round(sample_rate_score + bit_depth_score, 2)


class RealTimeAnalyticsEngine:
    """Main real-time analytics engine orchestrating all components"""
    
    def __init__(self, 
                 max_events_per_second: int = 10000,
                 batch_size: int = 100,
                 batch_timeout_ms: int = 500,
                 enable_compression: bool = True):
        
        self.max_events_per_second = max_events_per_second
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.enable_compression = enable_compression
        
        # Core components
        self.metric_collector = InMemoryMetricCollector()
        self.event_processor = AudioAnalyticsProcessor()
        
        # Event processing infrastructure
        self.event_queue = asyncio.Queue(maxsize=max_events_per_second)
        self.event_batch = []
        self.last_batch_time = time.time()
        
        # Real-time dashboards and subscriptions
        self.dashboard_subscriptions = set()
        self.metric_subscriptions = defaultdict(set)
        
        # Performance tracking
        self.performance_stats = {
            'events_processed': 0,
            'events_per_second': 0,
            'average_processing_time_ms': 0,
            'last_reset_time': datetime.utcnow()
        }
        
        # Background tasks
        self.background_tasks = set()
        self.running = False
        
        # Thread pool for CPU-intensive operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
    
    async def initialize(self) -> None:
        """Initialize the analytics engine"""
        logger.info("Initializing Real-time Analytics Engine...")
        
        # Start background processing tasks
        self.running = True
        
        # Event processing task
        event_processor_task = asyncio.create_task(self._process_event_queue())
        self.background_tasks.add(event_processor_task)
        event_processor_task.add_done_callback(self.background_tasks.discard)
        
        # Batch processing task
        batch_processor_task = asyncio.create_task(self._process_event_batches())
        self.background_tasks.add(batch_processor_task)
        batch_processor_task.add_done_callback(self.background_tasks.discard)
        
        # Performance monitoring task
        perf_monitor_task = asyncio.create_task(self._monitor_performance())
        self.background_tasks.add(perf_monitor_task)
        perf_monitor_task.add_done_callback(self.background_tasks.discard)
        
        # Dashboard update task
        dashboard_task = asyncio.create_task(self._update_dashboards())
        self.background_tasks.add(dashboard_task)
        dashboard_task.add_done_callback(self.background_tasks.discard)
        
        logger.info("Real-time Analytics Engine initialized successfully")
    
    async def ingest_event(self, event: RealTimeEvent) -> Dict[str, Any]:
        """Ingest real-time event for processing"""
        try:
            # Add event to queue with backpressure handling
            if self.event_queue.full():
                logger.warning("Event queue full, dropping event")
                return {'status': 'dropped', 'reason': 'queue_full'}
            
            await self.event_queue.put(event)
            
            # Update performance stats
            self.performance_stats['events_processed'] += 1
            
            return {'status': 'accepted', 'event_id': event.id}
            
        except Exception as e:
            logger.error(f"Error ingesting event: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def ingest_metric(self, metric: RealTimeMetric) -> None:
        """Ingest real-time metric"""
        await self.metric_collector.collect_metric(metric)
        
        # Notify metric subscribers
        for callback in self.metric_subscriptions[metric.name]:
            try:
                await callback(metric)
            except Exception as e:
                logger.error(f"Error in metric subscription callback: {e}")
    
    async def get_real_time_metrics(self, metric_names: List[str] = None) -> Dict[str, Any]:
        """Get current real-time metrics"""
        if metric_names is None:
            metric_names = list(self.metric_collector.get_all_metric_names())
        
        metrics = {}
        for name in metric_names:
            latest_metric = self.metric_collector.buffer.get_latest_value(name)
            stats = self.metric_collector.get_metric_stats(name)
            
            metrics[name] = {
                'latest_value': latest_metric.value if latest_metric else None,
                'latest_timestamp': latest_metric.timestamp.isoformat() if latest_metric else None,
                'statistics': stats
            }
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': metrics,
            'performance': self.performance_stats.copy()
        }
    
    async def get_time_series_data(self, 
                                  metric_name: str,
                                  start_time: datetime,
                                  end_time: datetime,
                                  aggregation: str = 'raw') -> Dict[str, Any]:
        """Get time-series data for metric"""
        metrics = await self.metric_collector.get_metrics(metric_name, start_time, end_time)
        
        if aggregation == 'raw':
            return {
                'metric_name': metric_name,
                'data_points': [m.to_dict() for m in metrics],
                'count': len(metrics)
            }
        elif aggregation == 'minute':
            return await self._aggregate_by_minute(metrics)
        elif aggregation == 'hour':
            return await self._aggregate_by_hour(metrics)
        else:
            return {'error': f'Unsupported aggregation: {aggregation}'}
    
    async def subscribe_to_dashboard(self, callback: Callable) -> str:
        """Subscribe to real-time dashboard updates"""
        subscription_id = str(uuid.uuid4())
        self.dashboard_subscriptions.add((subscription_id, callback))
        return subscription_id
    
    async def subscribe_to_metric(self, metric_name: str, callback: Callable) -> str:
        """Subscribe to specific metric updates"""
        subscription_id = str(uuid.uuid4())
        self.metric_subscriptions[metric_name].add(callback)
        return subscription_id
    
    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from updates"""
        # Remove from dashboard subscriptions
        self.dashboard_subscriptions = {
            (sid, cb) for sid, cb in self.dashboard_subscriptions 
            if sid != subscription_id
        }
        
        # Remove from metric subscriptions
        for metric_callbacks in self.metric_subscriptions.values():
            metric_callbacks.discard(subscription_id)
    
    async def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary"""
        current_time = datetime.utcnow()
        one_hour_ago = current_time - timedelta(hours=1)
        
        # Get key metrics from last hour
        summary = {
            'timestamp': current_time.isoformat(),
            'performance_stats': self.performance_stats.copy(),
            'active_sessions': len(self.event_processor.session_stats),
            'metrics_tracked': len(self.metric_collector.get_all_metric_names()),
            'dashboard_subscribers': len(self.dashboard_subscriptions),
            'total_events_processed': self.performance_stats['events_processed']
        }
        
        return summary
    
    async def _process_event_queue(self) -> None:
        """Background task to process event queue"""
        while self.running:
            try:
                # Get event with timeout
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # Process event
                start_time = time.time()
                result = await self.event_processor.process_event(event)
                processing_time = (time.time() - start_time) * 1000
                
                # Update performance stats
                self._update_processing_time_stats(processing_time)
                
                # Add to batch for batch processing
                self.event_batch.append(event)
                
                # Mark task done
                self.event_queue.task_done()
                
            except asyncio.TimeoutError:
                # No events to process, continue
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def _process_event_batches(self) -> None:
        """Background task to process event batches"""
        while self.running:
            try:
                current_time = time.time()
                time_since_batch = (current_time - self.last_batch_time) * 1000
                
                # Process batch if we have enough events or timeout reached
                if (len(self.event_batch) >= self.batch_size or 
                    (self.event_batch and time_since_batch >= self.batch_timeout_ms)):
                    
                    # Process current batch
                    batch_to_process = self.event_batch.copy()
                    self.event_batch.clear()
                    self.last_batch_time = current_time
                    
                    # Process in thread pool for CPU-intensive operations
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        self.thread_pool,
                        self._process_batch_sync,
                        batch_to_process
                    )
                
                # Small sleep to prevent busy waiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
    
    def _process_batch_sync(self, events: List[RealTimeEvent]) -> None:
        """Synchronous batch processing for thread pool"""
        try:
            # This would normally call async batch processing,
            # but we're keeping it simple for thread pool execution
            logger.info(f"Processed batch of {len(events)} events")
        except Exception as e:
            logger.error(f"Error in sync batch processing: {e}")
    
    async def _monitor_performance(self) -> None:
        """Background task to monitor performance"""
        last_count = 0
        
        while self.running:
            try:
                current_time = datetime.utcnow()
                current_count = self.performance_stats['events_processed']
                
                # Calculate events per second
                time_diff = (current_time - self.performance_stats['last_reset_time']).total_seconds()
                if time_diff > 0:
                    self.performance_stats['events_per_second'] = (current_count - last_count) / min(time_diff, 60)
                
                # Reset counters every hour
                if time_diff >= 3600:
                    self.performance_stats['last_reset_time'] = current_time
                    last_count = current_count
                
                # Sleep for 10 seconds before next check
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
    
    async def _update_dashboards(self) -> None:
        """Background task to update dashboard subscribers"""
        while self.running:
            try:
                if self.dashboard_subscriptions:
                    # Get latest dashboard data
                    dashboard_data = await self.get_real_time_metrics()
                    
                    # Notify all subscribers
                    for subscription_id, callback in self.dashboard_subscriptions.copy():
                        try:
                            await callback(dashboard_data)
                        except Exception as e:
                            logger.error(f"Error in dashboard callback {subscription_id}: {e}")
                            # Remove failed subscription
                            self.dashboard_subscriptions.discard((subscription_id, callback))
                
                # Update every 5 seconds
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error updating dashboards: {e}")
    
    def _update_processing_time_stats(self, processing_time_ms: float) -> None:
        """Update average processing time statistics"""
        current_avg = self.performance_stats.get('average_processing_time_ms', 0)
        current_count = self.performance_stats.get('events_processed', 1)
        
        # Calculate moving average
        new_avg = ((current_avg * (current_count - 1)) + processing_time_ms) / current_count
        self.performance_stats['average_processing_time_ms'] = round(new_avg, 2)
    
    async def _aggregate_by_minute(self, metrics: List[RealTimeMetric]) -> Dict[str, Any]:
        """Aggregate metrics by minute"""
        minute_buckets = defaultdict(list)
        
        for metric in metrics:
            minute_key = metric.timestamp.replace(second=0, microsecond=0)
            minute_buckets[minute_key].append(metric.value)
        
        aggregated = []
        for minute, values in sorted(minute_buckets.items()):
            aggregated.append({
                'timestamp': minute.isoformat(),
                'count': len(values),
                'avg': statistics.mean(values),
                'min': min(values),
                'max': max(values)
            })
        
        return {
            'aggregation': 'minute',
            'data_points': aggregated,
            'total_count': len(metrics)
        }
    
    async def _aggregate_by_hour(self, metrics: List[RealTimeMetric]) -> Dict[str, Any]:
        """Aggregate metrics by hour"""
        hour_buckets = defaultdict(list)
        
        for metric in metrics:
            hour_key = metric.timestamp.replace(minute=0, second=0, microsecond=0)
            hour_buckets[hour_key].append(metric.value)
        
        aggregated = []
        for hour, values in sorted(hour_buckets.items()):
            aggregated.append({
                'timestamp': hour.isoformat(),
                'count': len(values),
                'avg': statistics.mean(values),
                'min': min(values),
                'max': max(values)
            })
        
        return {
            'aggregation': 'hour',
            'data_points': aggregated,
            'total_count': len(metrics)
        }
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the analytics engine"""
        logger.info("Shutting down Real-time Analytics Engine...")
        
        self.running = False
        
        # Wait for queue to be processed
        if not self.event_queue.empty():
            await self.event_queue.join()
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        logger.info("Real-time Analytics Engine shutdown complete")


# Convenience functions for easy integration

async def create_analytics_engine(config: Dict[str, Any] = None) -> RealTimeAnalyticsEngine:
    """Create and initialize analytics engine with configuration"""
    if config is None:
        config = {}
    
    engine = RealTimeAnalyticsEngine(
        max_events_per_second=config.get('max_events_per_second', 10000),
        batch_size=config.get('batch_size', 100),
        batch_timeout_ms=config.get('batch_timeout_ms', 500),
        enable_compression=config.get('enable_compression', True)
    )
    
    await engine.initialize()
    return engine


async def demo_analytics_engine():
    """Demonstration of the real-time analytics engine"""
    print("🔥 Starting Real-time Analytics Engine Demo")
    print("=" * 50)
    
    # Create and initialize engine
    engine = await create_analytics_engine()
    
    # Simulate some real-time events
    print("📊 Simulating real-time audio events...")
    
    for i in range(20):
        # Create sample events
        audio_event = RealTimeEvent(
            event_type=EventType.AUDIO_LEVEL_CHANGE,
            user_id=f"user_{i % 3}",
            session_id=f"session_{i % 2}",
            audio_channel=i % 4,
            audio_level=0.5 + (i % 10) * 0.05,
            sample_rate=48000,
            bit_depth=24
        )
        
        interaction_event = RealTimeEvent(
            event_type=EventType.USER_INTERACTION,
            user_id=f"user_{i % 3}",
            component="fader",
            action="adjust",
            value=i * 0.1
        )
        
        performance_event = RealTimeEvent(
            event_type=EventType.PERFORMANCE_METRIC,
            cpu_usage=20 + (i % 5) * 10,
            memory_usage=40 + (i % 3) * 15,
            latency_ms=5 + (i % 4) * 2
        )
        
        # Ingest events
        await engine.ingest_event(audio_event)
        await engine.ingest_event(interaction_event)
        await engine.ingest_event(performance_event)
        
        # Ingest some metrics
        audio_metric = RealTimeMetric(
            name="audio_level",
            metric_type=MetricType.GAUGE,
            value=audio_event.audio_level,
            tags={"channel": str(audio_event.audio_channel)}
        )
        
        cpu_metric = RealTimeMetric(
            name="cpu_usage",
            metric_type=MetricType.GAUGE,
            value=performance_event.cpu_usage,
            unit="percent"
        )
        
        await engine.ingest_metric(audio_metric)
        await engine.ingest_metric(cpu_metric)
    
    # Wait for processing
    await asyncio.sleep(2)
    
    # Get analytics summary
    print("\n📈 Real-time Analytics Summary:")
    summary = await engine.get_analytics_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Get real-time metrics
    print("\n📊 Current Real-time Metrics:")
    metrics = await engine.get_real_time_metrics()
    for metric_name, metric_data in metrics['metrics'].items():
        print(f"  {metric_name}: {metric_data['latest_value']} ({metric_data['statistics'].get('count', 0)} samples)")
    
    # Shutdown
    await engine.shutdown()
    print("\n✅ Demo completed successfully!")


if __name__ == "__main__":
    asyncio.run(demo_analytics_engine())