"""
Data Platform Orchestrator for AG06 Mixer

Integrates all Phase 8 components into a unified data platform:
- Real-time Analytics Engine (analytics/realtime_analytics_engine.py)
- Data Lake System (data_platform/data_lake_system.py)  
- Stream Processing Engine (streaming/stream_processing_engine.py)

Provides enterprise-grade data platform capabilities following patterns from:
- Netflix (Lambda architecture, real-time + batch processing)
- Spotify (audio analytics and streaming data)
- Google (data mesh, domain-oriented ownership)
- AWS (unified data platform, Lake Formation)
- Databricks (unified analytics platform)
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from uuid import uuid4

# Import our Phase 8 components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.realtime_analytics_engine import RealTimeAnalyticsEngine, RealTimeEvent, EventType
from data_platform.data_lake_system import DataLakeSystem, DataFormat, StorageTier
from streaming.stream_processing_engine import (
    StreamProcessingEngine, StreamEvent, AudioLevelProcessor, 
    SessionAggregationProcessor, StreamPartitionStrategy
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPlatformMode(Enum):
    """Operating modes for the data platform"""
    DEVELOPMENT = "development"      # Single-node, in-memory
    STAGING = "staging"             # Multi-node, persistent storage
    PRODUCTION = "production"       # Full distributed, high availability


@dataclass
class PlatformMetrics:
    """Comprehensive metrics for the data platform"""
    # Stream processing metrics
    events_processed_per_second: float = 0.0
    events_in_flight: int = 0
    processing_latency_ms: float = 0.0
    
    # Analytics metrics
    analytics_queries_per_second: float = 0.0
    active_dashboards: int = 0
    real_time_subscribers: int = 0
    
    # Data lake metrics
    data_ingested_mb_per_second: float = 0.0
    total_storage_gb: float = 0.0
    query_response_time_ms: float = 0.0
    
    # System health
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0
    error_rate_percent: float = 0.0
    
    # Business metrics
    active_users: int = 0
    audio_sessions_active: int = 0
    total_interactions_today: int = 0


class DataPlatformOrchestrator:
    """
    Unified orchestrator for AG06 Mixer Data Platform
    
    Features:
    - Unified event ingestion and processing pipeline
    - Real-time analytics with streaming insights
    - Long-term storage with data lake architecture
    - Cross-component data flow and orchestration
    - Comprehensive monitoring and observability
    - Auto-scaling and resource optimization
    """
    
    def __init__(self,
                 mode: DataPlatformMode = DataPlatformMode.DEVELOPMENT,
                 data_lake_base_path: str = "./data_lake",
                 max_events_per_second: int = 100000,
                 enable_monitoring: bool = True):
        
        self.mode = mode
        self.enable_monitoring = enable_monitoring
        self.running = False
        
        # Initialize core components
        self.analytics_engine = RealTimeAnalyticsEngine(
            max_events_per_second=max_events_per_second,
            batch_size=100,
            enable_compression=True
        )
        
        self.data_lake = DataLakeSystem(
            base_path=data_lake_base_path
        )
        
        self.stream_processor = StreamProcessingEngine(
            max_throughput_per_second=max_events_per_second,
            default_partition_count=16 if mode == DataPlatformMode.PRODUCTION else 4
        )
        
        # Cross-component integration
        self.event_flows: Dict[str, List[Callable]] = {}
        self.data_pipelines: List[asyncio.Task] = []
        
        # Metrics and monitoring
        self.metrics = PlatformMetrics()
        self.metrics_history: List[Dict] = []
        self._last_metrics_update = time.time()
        
        # Component status
        self.component_status = {
            'analytics_engine': False,
            'data_lake': False,
            'stream_processor': False
        }
        
        logger.info(f"Data Platform Orchestrator initialized in {mode.value} mode")
    
    async def initialize(self) -> bool:
        """Initialize all platform components"""
        try:
            logger.info("Initializing Data Platform components...")
            
            # Initialize Analytics Engine
            logger.info("Starting Real-time Analytics Engine...")
            await self.analytics_engine.initialize()
            self.component_status['analytics_engine'] = True
            
            # Initialize Data Lake
            logger.info("Starting Data Lake System...")
            await self.data_lake.initialize()
            self.component_status['data_lake'] = True
            
            # Initialize Stream Processor
            logger.info("Starting Stream Processing Engine...")
            
            # Register audio-specific processors
            audio_processor = AudioLevelProcessor(
                smoothing_factor=0.8,
                anomaly_threshold=2.0
            )
            
            session_processor = SessionAggregationProcessor(
                session_timeout_minutes=30
            )
            
            self.stream_processor.register_processor(
                "audio_analytics",
                audio_processor,
                ["audio_level_change", "audio_effect_change", "audio_input_change"],
                StreamPartitionStrategy.AUDIO_CHANNEL
            )
            
            self.stream_processor.register_processor(
                "session_analytics",
                session_processor,
                ["user_interaction", "session_start", "session_end", "session_activity"],
                StreamPartitionStrategy.USER_ID
            )
            
            await self.stream_processor.start_processing()
            self.component_status['stream_processor'] = True
            
            # Set up cross-component data flows
            await self._setup_data_flows()
            
            logger.info("✅ All Data Platform components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Data Platform: {e}")
            return False
    
    async def start(self) -> bool:
        """Start the unified data platform"""
        if self.running:
            logger.warning("Data Platform is already running")
            return True
        
        # Initialize components if not already done
        if not all(self.component_status.values()):
            success = await self.initialize()
            if not success:
                return False
        
        self.running = True
        logger.info("🚀 Data Platform Orchestrator starting...")
        
        # Start cross-component data pipelines
        await self._start_data_pipelines()
        
        # Start monitoring if enabled
        if self.enable_monitoring:
            await self._start_monitoring()
        
        logger.info("✅ Data Platform Orchestrator running successfully")
        return True
    
    async def stop(self):
        """Stop the unified data platform"""
        if not self.running:
            return
        
        logger.info("Stopping Data Platform Orchestrator...")
        self.running = False
        
        # Stop data pipelines
        for task in self.data_pipelines:
            task.cancel()
        
        if self.data_pipelines:
            await asyncio.gather(*self.data_pipelines, return_exceptions=True)
        
        # Stop components
        await self.stream_processor.stop_processing()
        await self.data_lake.shutdown()
        await self.analytics_engine.shutdown()
        
        logger.info("✅ Data Platform Orchestrator stopped")
    
    async def ingest_event(self, event_data: Dict[str, Any]) -> bool:
        """
        Unified event ingestion - routes to appropriate processing pipelines
        
        This is the main entry point for all AG06 mixer events
        """
        try:
            # Create unified event object
            stream_event = StreamEvent(
                event_type=event_data.get('event_type', 'unknown'),
                payload=event_data.get('payload', {}),
                partition_key=event_data.get('partition_key'),
                headers=event_data.get('headers', {})
            )
            
            # Route to stream processor for real-time processing
            await self.stream_processor.publish_event("audio_platform", stream_event)
            
            # Route to analytics engine for real-time metrics
            # Map event type to EventType enum
            event_type_mapping = {
                'audio_level_change': EventType.AUDIO_LEVEL_CHANGE,
                'audio_stream_start': EventType.AUDIO_STREAM_START,
                'audio_stream_end': EventType.AUDIO_STREAM_END,
                'user_interaction': EventType.USER_INTERACTION,
                'session_start': EventType.SESSION_START,
                'session_end': EventType.SESSION_END
            }
            
            analytics_event = RealTimeEvent(
                event_type=event_type_mapping.get(stream_event.event_type, EventType.USER_INTERACTION),
                timestamp=stream_event.timestamp,
                user_id=stream_event.payload.get('user_id'),
                session_id=stream_event.payload.get('session_id'),
                audio_channel=stream_event.payload.get('channel'),
                audio_level=stream_event.payload.get('level'),
                component=stream_event.payload.get('target_element'),
                action=stream_event.payload.get('action_type'),
                value=stream_event.payload.get('level')
            )
            
            await self.analytics_engine.ingest_event(analytics_event)
            
            # Store in data lake for long-term analytics (async)
            asyncio.create_task(self._store_to_data_lake(event_data))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to ingest event: {e}")
            return False
    
    async def _store_to_data_lake(self, event_data: Dict[str, Any]):
        """Store event data to data lake with appropriate schema"""
        try:
            event_type = event_data.get('event_type', 'unknown')
            payload = event_data.get('payload', {})
            
            # Route to appropriate data lake schema
            if event_type.startswith('audio_'):
                await self.data_lake.ingest_data(
                    'audio_streams',
                    {
                        'timestamp': datetime.utcnow().isoformat(),
                        'event_type': event_type,
                        'channel': payload.get('channel'),
                        'level': payload.get('level'),
                        'effect_type': payload.get('effect_type'),
                        'settings': payload.get('settings', {})
                    },
                    partition_values={
                        'year': datetime.utcnow().year,
                        'month': datetime.utcnow().month,
                        'day': datetime.utcnow().day
                    },
                    storage_tier=StorageTier.HOT
                )
            
            elif event_type in ['user_interaction', 'session_start', 'session_end']:
                await self.data_lake.ingest_data(
                    'user_interactions',
                    {
                        'timestamp': datetime.utcnow().isoformat(),
                        'event_type': event_type,
                        'user_id': payload.get('user_id'),
                        'session_id': payload.get('session_id'),
                        'action_type': payload.get('action_type'),
                        'target_element': payload.get('target_element'),
                        'interaction_data': payload.get('interaction_data', {})
                    },
                    partition_values={
                        'year': datetime.utcnow().year,
                        'month': datetime.utcnow().month,
                        'day': datetime.utcnow().day
                    },
                    storage_tier=StorageTier.HOT
                )
                
        except Exception as e:
            logger.error(f"Failed to store event to data lake: {e}")
    
    async def _setup_data_flows(self):
        """Set up cross-component data flows and integrations"""
        logger.info("Setting up cross-component data flows...")
        
        # Analytics → Data Lake flow (for processed metrics)
        self.event_flows['analytics_to_lake'] = [
            self._flow_analytics_metrics_to_lake
        ]
        
        # Stream Processing → Analytics flow (for processed events)
        self.event_flows['stream_to_analytics'] = [
            self._flow_processed_events_to_analytics
        ]
        
        # Data Lake → Analytics flow (for historical analysis)
        self.event_flows['lake_to_analytics'] = [
            self._flow_historical_data_to_analytics
        ]
    
    async def _start_data_pipelines(self):
        """Start cross-component data pipeline tasks"""
        logger.info("Starting data pipeline tasks...")
        
        # Start analytics metrics collection pipeline
        analytics_pipeline = asyncio.create_task(
            self._analytics_metrics_pipeline()
        )
        self.data_pipelines.append(analytics_pipeline)
        
        # Start data lake optimization pipeline
        lake_optimization_pipeline = asyncio.create_task(
            self._data_lake_optimization_pipeline()
        )
        self.data_pipelines.append(lake_optimization_pipeline)
        
        # Start cross-component health monitoring
        health_pipeline = asyncio.create_task(
            self._health_monitoring_pipeline()
        )
        self.data_pipelines.append(health_pipeline)
    
    async def _analytics_metrics_pipeline(self):
        """Pipeline for collecting and storing analytics metrics"""
        while self.running:
            try:
                # Get real-time metrics from analytics engine
                analytics_metrics = await self.analytics_engine.get_real_time_metrics()
                
                # Store aggregated metrics to data lake
                await self.data_lake.ingest_data(
                    'performance_metrics',
                    {
                        'timestamp': datetime.utcnow().isoformat(),
                        'metric_type': 'analytics_performance',
                        'events_processed': analytics_metrics.get('events_processed', 0),
                        'active_subscribers': analytics_metrics.get('active_subscribers', 0),
                        'processing_latency_ms': analytics_metrics.get('avg_processing_time_ms', 0),
                        'memory_usage_mb': analytics_metrics.get('memory_usage_mb', 0)
                    },
                    partition_values={
                        'year': datetime.utcnow().year,
                        'month': datetime.utcnow().month,
                        'day': datetime.utcnow().day
                    },
                    storage_tier=StorageTier.WARM
                )
                
                await asyncio.sleep(60)  # Collect metrics every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in analytics metrics pipeline: {e}")
                await asyncio.sleep(30)
    
    async def _data_lake_optimization_pipeline(self):
        """Pipeline for data lake maintenance and optimization"""
        while self.running:
            try:
                # Trigger data lake optimization
                await self.data_lake.optimize_storage()
                
                # Clean up old partitions based on retention policies
                cutoff_date = datetime.utcnow() - timedelta(days=90)  # 90 day retention
                
                # This would typically query and clean old partitions
                # Implementation depends on specific retention requirements
                
                await asyncio.sleep(3600)  # Run optimization hourly
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in data lake optimization pipeline: {e}")
                await asyncio.sleep(1800)  # Retry in 30 minutes
    
    async def _health_monitoring_pipeline(self):
        """Pipeline for monitoring component health and metrics"""
        while self.running:
            try:
                # Collect metrics from all components
                await self._update_platform_metrics()
                
                # Store metrics history
                metrics_snapshot = {
                    'timestamp': datetime.utcnow().isoformat(),
                    **self.metrics.__dict__
                }
                self.metrics_history.append(metrics_snapshot)
                
                # Keep only last 1000 metrics snapshots
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                await asyncio.sleep(30)  # Update metrics every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring pipeline: {e}")
                await asyncio.sleep(60)
    
    async def _start_monitoring(self):
        """Start platform monitoring and alerting"""
        logger.info("Starting platform monitoring...")
        
        # This would typically integrate with monitoring systems like:
        # - Prometheus for metrics collection
        # - Grafana for dashboards
        # - AlertManager for alerting
        # - Custom health check endpoints
        
        # For now, we'll log key metrics periodically
        monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.data_pipelines.append(monitoring_task)
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                await self._update_platform_metrics()
                
                # Log key metrics every 5 minutes
                logger.info(f"Platform Health: "
                           f"Events/sec: {self.metrics.events_processed_per_second:.1f}, "
                           f"Memory: {self.metrics.memory_usage_percent:.1f}%, "
                           f"Active users: {self.metrics.active_users}, "
                           f"Error rate: {self.metrics.error_rate_percent:.2f}%")
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _update_platform_metrics(self):
        """Update comprehensive platform metrics"""
        try:
            # Get metrics from each component
            analytics_metrics = await self.analytics_engine.get_real_time_metrics()
            stream_metrics = self.stream_processor.get_metrics()
            # data_lake_metrics = await self.data_lake.get_metrics()  # Would implement this
            
            # Update platform metrics
            self.metrics.events_processed_per_second = stream_metrics.get('events_per_second', 0.0)
            self.metrics.processing_latency_ms = analytics_metrics.get('avg_processing_time_ms', 0.0)
            self.metrics.active_dashboards = analytics_metrics.get('active_subscribers', 0)
            self.metrics.real_time_subscribers = analytics_metrics.get('active_subscribers', 0)
            
            # System metrics (would typically use psutil or system monitoring)
            import psutil
            self.metrics.cpu_usage_percent = psutil.cpu_percent()
            self.metrics.memory_usage_percent = psutil.virtual_memory().percent
            self.metrics.disk_usage_percent = psutil.disk_usage('/').percent
            
            # Calculate error rate
            total_events = stream_metrics.get('events_processed', 0) + stream_metrics.get('events_failed', 0)
            if total_events > 0:
                self.metrics.error_rate_percent = (stream_metrics.get('events_failed', 0) / total_events) * 100
            
        except Exception as e:
            logger.error(f"Error updating platform metrics: {e}")
    
    # Flow methods for cross-component integration
    async def _flow_analytics_metrics_to_lake(self):
        """Flow analytics metrics to data lake"""
        pass  # Implementation would depend on specific requirements
    
    async def _flow_processed_events_to_analytics(self):
        """Flow processed stream events to analytics"""
        pass  # Implementation would depend on specific requirements
        
    async def _flow_historical_data_to_analytics(self):
        """Flow historical data from lake to analytics"""
        pass  # Implementation would depend on specific requirements
    
    # Public API methods
    async def get_platform_status(self) -> Dict[str, Any]:
        """Get comprehensive platform status"""
        await self._update_platform_metrics()
        
        return {
            'running': self.running,
            'mode': self.mode.value,
            'components': self.component_status,
            'metrics': self.metrics.__dict__,
            'uptime_seconds': time.time() - self._last_metrics_update,
            'active_pipelines': len(self.data_pipelines)
        }
    
    async def get_platform_metrics(self) -> PlatformMetrics:
        """Get current platform metrics"""
        await self._update_platform_metrics()
        return self.metrics
    
    async def get_metrics_history(self, hours: int = 24) -> List[Dict]:
        """Get metrics history for specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        filtered_history = [
            metrics for metrics in self.metrics_history
            if datetime.fromisoformat(metrics['timestamp']) > cutoff_time
        ]
        
        return filtered_history
    
    async def query_data_lake(self, dataset_name: str, filters: Dict = None) -> List[Dict]:
        """Query data lake through unified interface"""
        return await self.data_lake.query_data(dataset_name, filters or {})
    
    async def get_real_time_analytics(self, metric_names: List[str] = None) -> Dict:
        """Get real-time analytics data"""
        return await self.analytics_engine.get_real_time_metrics(metric_names)


# Demo function
async def demo_data_platform():
    """Demonstrate the unified data platform"""
    
    print("🏗️ AG06 Mixer Data Platform Demo")
    print("=================================")
    
    # Create orchestrator
    orchestrator = DataPlatformOrchestrator(
        mode=DataPlatformMode.DEVELOPMENT,
        max_events_per_second=10000
    )
    
    # Start platform
    success = await orchestrator.start()
    if not success:
        print("❌ Failed to start data platform")
        return
    
    print("✅ Data Platform started successfully")
    
    # Generate sample events
    sample_events = [
        {
            'event_type': 'audio_level_change',
            'payload': {
                'channel': 'input_1',
                'level': 0.8,
                'user_id': 'user_123',
                'session_id': 'session_456'
            },
            'partition_key': 'input_1'
        },
        {
            'event_type': 'user_interaction',
            'payload': {
                'user_id': 'user_123',
                'session_id': 'session_456',
                'action_type': 'button_click',
                'target_element': 'mute_button_1'
            },
            'partition_key': 'user_123'
        },
        {
            'event_type': 'session_start',
            'payload': {
                'user_id': 'user_123',
                'session_id': 'session_456',
                'device_model': 'AG06',
                'firmware_version': '1.2.0'
            },
            'partition_key': 'user_123'
        }
    ]
    
    # Ingest events
    print("\n📥 Ingesting sample events...")
    for event in sample_events:
        success = await orchestrator.ingest_event(event)
        print(f"   {event['event_type']}: {success}")
    
    # Wait for processing
    await asyncio.sleep(3)
    
    # Show platform status
    print("\n📊 Platform Status:")
    status = await orchestrator.get_platform_status()
    print(f"   Running: {status['running']}")
    print(f"   Mode: {status['mode']}")
    print(f"   Components active: {sum(status['components'].values())}/3")
    print(f"   Events/sec: {status['metrics']['events_processed_per_second']:.1f}")
    print(f"   Memory usage: {status['metrics']['memory_usage_percent']:.1f}%")
    print(f"   Active pipelines: {status['active_pipelines']}")
    
    # Query data lake
    print("\n🗄️ Data Lake Query Results:")
    try:
        audio_data = await orchestrator.query_data_lake('audio_streams')
        user_data = await orchestrator.query_data_lake('user_interactions')
        print(f"   Audio events stored: {len(audio_data)}")
        print(f"   User interaction events stored: {len(user_data)}")
    except Exception as e:
        print(f"   Query error: {e}")
    
    # Generate more events for throughput test
    print("\n⚡ Testing platform throughput...")
    start_time = time.time()
    
    for i in range(500):
        event = {
            'event_type': 'audio_level_change',
            'payload': {
                'channel': f'input_{i % 4}',
                'level': 0.5 + (i % 10) * 0.05,
                'user_id': f'user_{i % 50}',
                'session_id': f'session_{i % 25}'
            },
            'partition_key': f'input_{i % 4}'
        }
        await orchestrator.ingest_event(event)
    
    # Wait for processing
    await asyncio.sleep(5)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Final metrics
    final_status = await orchestrator.get_platform_status()
    print(f"\n📈 Performance Results:")
    print(f"   Events ingested: 503")
    print(f"   Processing time: {duration:.2f}s")
    print(f"   Throughput: {503 / duration:.0f} events/sec")
    print(f"   Platform events/sec: {final_status['metrics']['events_processed_per_second']:.1f}")
    print(f"   Error rate: {final_status['metrics']['error_rate_percent']:.2f}%")
    
    # Stop platform
    await orchestrator.stop()
    print("\n✅ Data Platform stopped successfully")
    
    return {
        'events_processed': 503,
        'throughput_events_per_sec': 503 / duration,
        'platform_throughput': final_status['metrics']['events_processed_per_second'],
        'components_active': sum(final_status['components'].values()),
        'success': True
    }


if __name__ == "__main__":
    # Run the demo
    result = asyncio.run(demo_data_platform())
    print(f"\n🎯 Demo Result: {result}")