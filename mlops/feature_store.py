"""
Production ML Feature Store
Following Uber Michelangelo, Netflix, Google Vertex AI best practices
"""

import numpy as np
import asyncio
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
import json
import hashlib
import threading
from abc import ABC, abstractmethod
from enum import Enum
import time
import pickle
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """Feature data types"""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    EMBEDDING = "embedding"
    TEXT = "text"
    TIMESTAMP = "timestamp"
    BINARY = "binary"


class FeatureSource(Enum):
    """Feature data sources"""
    BATCH = "batch"  # Offline batch processing
    STREAMING = "streaming"  # Real-time streams
    ON_DEMAND = "on_demand"  # Computed on request
    CACHED = "cached"  # Pre-computed and cached


@dataclass
class FeatureDefinition:
    """Feature schema definition"""
    name: str
    feature_type: FeatureType
    source: FeatureSource
    description: str
    entity: str  # Entity this feature belongs to (user, item, session)
    ttl_seconds: int = 3600  # Time to live
    default_value: Any = None
    transformation: Optional[str] = None  # SQL or function name
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"


@dataclass
class FeatureValue:
    """Feature value with metadata"""
    name: str
    value: Any
    timestamp: datetime
    entity_id: str
    source: FeatureSource
    version: str = "1.0"
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureVector:
    """Collection of features for an entity"""
    entity_id: str
    entity_type: str
    features: Dict[str, FeatureValue]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'entity_id': self.entity_id,
            'entity_type': self.entity_type,
            'features': {name: fv.value for name, fv in self.features.items()},
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


class FeatureStore(ABC):
    """Abstract feature store interface"""
    
    @abstractmethod
    async def register_feature(self, feature_def: FeatureDefinition):
        """Register a new feature definition"""
        pass
        
    @abstractmethod
    async def write_features(self, features: List[FeatureValue]):
        """Write feature values"""
        pass
        
    @abstractmethod
    async def read_features(self, entity_id: str, feature_names: List[str], 
                           timestamp: Optional[datetime] = None) -> FeatureVector:
        """Read feature values for an entity"""
        pass
        
    @abstractmethod
    async def read_feature_vector(self, entity_id: str, entity_type: str,
                                 timestamp: Optional[datetime] = None) -> FeatureVector:
        """Read all features for an entity"""
        pass


class InMemoryFeatureStore(FeatureStore):
    """
    In-memory feature store for development and testing
    Production systems would use Redis, Cassandra, or cloud solutions
    """
    
    def __init__(self):
        self.feature_definitions: Dict[str, FeatureDefinition] = {}
        self.feature_data: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(deque))  # entity -> feature -> values
        self.entity_features: Dict[str, List[str]] = defaultdict(list)  # entity_type -> feature_names
        self.lock = threading.RLock()
        self.metrics = FeatureStoreMetrics()
        
    async def register_feature(self, feature_def: FeatureDefinition):
        """Register feature definition"""
        with self.lock:
            self.feature_definitions[feature_def.name] = feature_def
            if feature_def.name not in self.entity_features[feature_def.entity]:
                self.entity_features[feature_def.entity].append(feature_def.name)
            logger.info(f"Registered feature: {feature_def.name} for entity: {feature_def.entity}")
            
    async def write_features(self, features: List[FeatureValue]):
        """Write feature values to store"""
        with self.lock:
            for feature in features:
                entity_id = feature.entity_id
                feature_name = feature.name
                
                # Store with TTL (simulate with max length)
                if feature_name in self.feature_definitions:
                    ttl = self.feature_definitions[feature_name].ttl_seconds
                    max_entries = max(1, ttl // 60)  # Rough estimate
                    if len(self.feature_data[entity_id][feature_name]) >= max_entries:
                        self.feature_data[entity_id][feature_name].popleft()
                
                self.feature_data[entity_id][feature_name].append(feature)
                
        self.metrics.record_write(len(features))
        logger.debug(f"Wrote {len(features)} feature values")
        
    async def read_features(self, entity_id: str, feature_names: List[str],
                           timestamp: Optional[datetime] = None) -> FeatureVector:
        """Read specific features for entity"""
        current_time = timestamp or datetime.now()
        features = {}
        
        with self.lock:
            for feature_name in feature_names:
                if feature_name in self.feature_definitions:
                    entity_type = self.feature_definitions[feature_name].entity
                    
                    # Find most recent valid feature value
                    feature_values = self.feature_data[entity_id][feature_name]
                    
                    valid_value = None
                    for fv in reversed(feature_values):  # Most recent first
                        if fv.timestamp <= current_time:
                            # Check TTL
                            ttl = self.feature_definitions[feature_name].ttl_seconds
                            if (current_time - fv.timestamp).total_seconds() <= ttl:
                                valid_value = fv
                                break
                                
                    if valid_value:
                        features[feature_name] = valid_value
                    else:
                        # Use default value
                        default = self.feature_definitions[feature_name].default_value
                        if default is not None:
                            features[feature_name] = FeatureValue(
                                name=feature_name,
                                value=default,
                                timestamp=current_time,
                                entity_id=entity_id,
                                source=FeatureSource.CACHED,
                                metadata={'is_default': True}
                            )
                            
        # Determine entity type from first feature
        entity_type = "unknown"
        if features:
            first_feature = next(iter(feature_names))
            if first_feature in self.feature_definitions:
                entity_type = self.feature_definitions[first_feature].entity
                
        self.metrics.record_read(len(features))
        
        return FeatureVector(
            entity_id=entity_id,
            entity_type=entity_type,
            features=features,
            timestamp=current_time
        )
        
    async def read_feature_vector(self, entity_id: str, entity_type: str,
                                 timestamp: Optional[datetime] = None) -> FeatureVector:
        """Read all features for entity type"""
        feature_names = self.entity_features.get(entity_type, [])
        return await self.read_features(entity_id, feature_names, timestamp)
        
    def get_feature_definitions(self, entity_type: Optional[str] = None) -> List[FeatureDefinition]:
        """Get feature definitions"""
        if entity_type:
            return [fd for fd in self.feature_definitions.values() if fd.entity == entity_type]
        return list(self.feature_definitions.values())


class FeatureComputer:
    """Computes derived features on-demand"""
    
    def __init__(self, feature_store: FeatureStore):
        self.feature_store = feature_store
        self.transformations: Dict[str, Callable] = {}
        
    def register_transformation(self, name: str, func: Callable):
        """Register a feature transformation function"""
        self.transformations[name] = func
        logger.info(f"Registered transformation: {name}")
        
    async def compute_features(self, entity_id: str, feature_names: List[str],
                             context: Dict[str, Any] = None) -> List[FeatureValue]:
        """Compute derived features on demand"""
        computed_features = []
        context = context or {}
        
        for feature_name in feature_names:
            if feature_name in self.transformations:
                try:
                    # Get base features needed for computation
                    base_vector = await self.feature_store.read_feature_vector(
                        entity_id, context.get('entity_type', 'user')
                    )
                    
                    # Apply transformation
                    transform_func = self.transformations[feature_name]
                    computed_value = await self._apply_transformation(
                        transform_func, base_vector, context
                    )
                    
                    feature_value = FeatureValue(
                        name=feature_name,
                        value=computed_value,
                        timestamp=datetime.now(),
                        entity_id=entity_id,
                        source=FeatureSource.ON_DEMAND,
                        metadata={'computed': True, 'transformation': feature_name}
                    )
                    
                    computed_features.append(feature_value)
                    
                except Exception as e:
                    logger.error(f"Failed to compute feature {feature_name}: {e}")
                    
        return computed_features
        
    async def _apply_transformation(self, func: Callable, feature_vector: FeatureVector,
                                   context: Dict[str, Any]) -> Any:
        """Apply transformation function safely"""
        if asyncio.iscoroutinefunction(func):
            return await func(feature_vector, context)
        else:
            return func(feature_vector, context)


class FeaturePipeline:
    """
    Feature processing pipeline for real-time and batch features
    Following Uber's Michelangelo architecture
    """
    
    def __init__(self, feature_store: FeatureStore):
        self.feature_store = feature_store
        self.computer = FeatureComputer(feature_store)
        self.processors: Dict[str, Callable] = {}
        self.batch_processors: List[Callable] = []
        self.streaming_processors: List[Callable] = []
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.metrics = FeatureStoreMetrics()
        
    def register_processor(self, name: str, processor: Callable, 
                          source_type: FeatureSource):
        """Register a feature processor"""
        self.processors[name] = processor
        
        if source_type == FeatureSource.BATCH:
            self.batch_processors.append(processor)
        elif source_type == FeatureSource.STREAMING:
            self.streaming_processors.append(processor)
            
        logger.info(f"Registered {source_type.value} processor: {name}")
        
    async def process_event(self, event: Dict[str, Any]) -> List[FeatureValue]:
        """Process streaming event to extract features"""
        features = []
        
        for processor in self.streaming_processors:
            try:
                processor_features = await self._run_processor(processor, event)
                if processor_features:
                    features.extend(processor_features)
            except Exception as e:
                logger.error(f"Streaming processor failed: {e}")
                
        # Write features to store
        if features:
            await self.feature_store.write_features(features)
            self.metrics.record_processed_event()
            
        return features
        
    async def process_batch(self, batch_data: List[Dict[str, Any]]) -> List[FeatureValue]:
        """Process batch data to extract features"""
        all_features = []
        
        for processor in self.batch_processors:
            try:
                processor_features = await self._run_batch_processor(processor, batch_data)
                if processor_features:
                    all_features.extend(processor_features)
            except Exception as e:
                logger.error(f"Batch processor failed: {e}")
                
        # Write features to store
        if all_features:
            await self.feature_store.write_features(all_features)
            self.metrics.record_batch_processed(len(batch_data))
            
        return all_features
        
    async def _run_processor(self, processor: Callable, data: Any) -> List[FeatureValue]:
        """Run processor safely"""
        if asyncio.iscoroutinefunction(processor):
            return await processor(data)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, processor, data)
            
    async def _run_batch_processor(self, processor: Callable, 
                                  batch_data: List[Dict[str, Any]]) -> List[FeatureValue]:
        """Run batch processor"""
        if asyncio.iscoroutinefunction(processor):
            return await processor(batch_data)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, processor, batch_data)


class FeatureStoreMetrics:
    """Metrics for feature store operations"""
    
    def __init__(self):
        self.read_count = 0
        self.write_count = 0
        self.processed_events = 0
        self.batch_processed = 0
        self.start_time = datetime.now()
        self.lock = threading.Lock()
        
    def record_read(self, feature_count: int):
        with self.lock:
            self.read_count += feature_count
            
    def record_write(self, feature_count: int):
        with self.lock:
            self.write_count += feature_count
            
    def record_processed_event(self):
        with self.lock:
            self.processed_events += 1
            
    def record_batch_processed(self, batch_size: int):
        with self.lock:
            self.batch_processed += batch_size
            
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            runtime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
            return {
                'reads_total': self.read_count,
                'writes_total': self.write_count,
                'events_processed': self.processed_events,
                'batch_items_processed': self.batch_processed,
                'runtime_hours': runtime_hours,
                'reads_per_hour': self.read_count / max(runtime_hours, 0.01),
                'writes_per_hour': self.write_count / max(runtime_hours, 0.01)
            }


class FeatureServingAPI:
    """
    Feature serving API for real-time ML inference
    Following Google Vertex AI and AWS SageMaker patterns
    """
    
    def __init__(self, feature_store: FeatureStore, pipeline: FeaturePipeline):
        self.feature_store = feature_store
        self.pipeline = pipeline
        self.cache: Dict[str, Tuple[FeatureVector, datetime]] = {}
        self.cache_ttl = 300  # 5 minutes
        self.lock = threading.Lock()
        
    async def get_features_for_inference(self, entity_id: str, entity_type: str,
                                       feature_names: Optional[List[str]] = None,
                                       include_computed: bool = False) -> Dict[str, Any]:
        """Get features optimized for ML inference"""
        cache_key = f"{entity_type}:{entity_id}:{hash(tuple(feature_names or []))}"
        
        # Check cache first
        cached_vector = self._get_cached_vector(cache_key)
        if cached_vector:
            return self._format_for_inference(cached_vector, feature_names)
            
        # Fetch from feature store
        if feature_names:
            feature_vector = await self.feature_store.read_features(
                entity_id, feature_names
            )
        else:
            feature_vector = await self.feature_store.read_feature_vector(
                entity_id, entity_type
            )
            
        # Add computed features if requested
        if include_computed:
            computed_features = await self.pipeline.computer.compute_features(
                entity_id, [f"computed_{entity_type}_score"], 
                {'entity_type': entity_type}
            )
            for cf in computed_features:
                feature_vector.features[cf.name] = cf
                
        # Cache result
        self._cache_vector(cache_key, feature_vector)
        
        return self._format_for_inference(feature_vector, feature_names)
        
    def _get_cached_vector(self, cache_key: str) -> Optional[FeatureVector]:
        """Get cached feature vector if still valid"""
        with self.lock:
            if cache_key in self.cache:
                vector, cached_at = self.cache[cache_key]
                if (datetime.now() - cached_at).total_seconds() < self.cache_ttl:
                    return vector
                else:
                    del self.cache[cache_key]
        return None
        
    def _cache_vector(self, cache_key: str, vector: FeatureVector):
        """Cache feature vector"""
        with self.lock:
            self.cache[cache_key] = (vector, datetime.now())
            
    def _format_for_inference(self, vector: FeatureVector, 
                            feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Format features for ML model inference"""
        features = {}
        
        for name, feature_value in vector.features.items():
            if feature_names is None or name in feature_names:
                value = feature_value.value
                
                # Handle different feature types
                if isinstance(value, (list, np.ndarray)):
                    features[name] = value
                elif isinstance(value, (int, float)):
                    features[name] = float(value)
                elif isinstance(value, str):
                    # For categorical features, might need encoding
                    features[name] = value
                else:
                    features[name] = str(value)
                    
        return {
            'entity_id': vector.entity_id,
            'entity_type': vector.entity_type,
            'features': features,
            'timestamp': vector.timestamp.isoformat(),
            'feature_count': len(features)
        }


# Example feature processors and transformations
def extract_user_features(user_event: Dict[str, Any]) -> List[FeatureValue]:
    """Extract user features from event"""
    user_id = user_event.get('user_id')
    timestamp = datetime.now()
    
    features = []
    
    # Basic user features
    if 'age' in user_event:
        features.append(FeatureValue(
            name='user_age',
            value=user_event['age'],
            timestamp=timestamp,
            entity_id=user_id,
            source=FeatureSource.STREAMING
        ))
        
    if 'country' in user_event:
        features.append(FeatureValue(
            name='user_country',
            value=user_event['country'],
            timestamp=timestamp,
            entity_id=user_id,
            source=FeatureSource.STREAMING
        ))
        
    # Computed features
    session_duration = user_event.get('session_duration', 0)
    features.append(FeatureValue(
        name='user_session_minutes',
        value=session_duration / 60.0,
        timestamp=timestamp,
        entity_id=user_id,
        source=FeatureSource.STREAMING
    ))
    
    return features


def compute_user_engagement_score(feature_vector: FeatureVector, 
                                context: Dict[str, Any]) -> float:
    """Compute user engagement score from features"""
    features = feature_vector.features
    
    # Get feature values with defaults
    session_minutes = features.get('user_session_minutes', FeatureValue(
        name='user_session_minutes', value=0.0, timestamp=datetime.now(),
        entity_id=feature_vector.entity_id, source=FeatureSource.CACHED
    )).value
    
    page_views = features.get('user_page_views', FeatureValue(
        name='user_page_views', value=0, timestamp=datetime.now(),
        entity_id=feature_vector.entity_id, source=FeatureSource.CACHED
    )).value
    
    # Simple engagement score formula
    engagement_score = min(100.0, (session_minutes * 2) + (page_views * 0.5))
    
    return float(engagement_score)


# Example usage and demonstration
async def demo_feature_store():
    """Demonstrate the ML feature store system"""
    print("🏪 Production ML Feature Store Demo")
    print("Following Uber Michelangelo, Netflix best practices\n")
    
    # Initialize feature store
    feature_store = InMemoryFeatureStore()
    pipeline = FeaturePipeline(feature_store)
    serving_api = FeatureServingAPI(feature_store, pipeline)
    
    # Register feature definitions
    print("📋 Registering feature definitions...")
    
    user_features = [
        FeatureDefinition(
            name='user_age',
            feature_type=FeatureType.NUMERICAL,
            source=FeatureSource.STREAMING,
            description='User age in years',
            entity='user',
            ttl_seconds=86400  # 24 hours
        ),
        FeatureDefinition(
            name='user_country',
            feature_type=FeatureType.CATEGORICAL,
            source=FeatureSource.STREAMING,
            description='User country code',
            entity='user',
            ttl_seconds=86400
        ),
        FeatureDefinition(
            name='user_session_minutes',
            feature_type=FeatureType.NUMERICAL,
            source=FeatureSource.STREAMING,
            description='Session duration in minutes',
            entity='user',
            ttl_seconds=3600  # 1 hour
        ),
        FeatureDefinition(
            name='user_page_views',
            feature_type=FeatureType.NUMERICAL,
            source=FeatureSource.BATCH,
            description='Total page views in last 30 days',
            entity='user',
            default_value=0,
            ttl_seconds=86400
        )
    ]
    
    for feature_def in user_features:
        await feature_store.register_feature(feature_def)
        
    # Register processors and transformations
    pipeline.register_processor(
        'user_events', 
        extract_user_features, 
        FeatureSource.STREAMING
    )
    
    pipeline.computer.register_transformation(
        'computed_user_score',
        compute_user_engagement_score
    )
    
    # Simulate streaming events
    print("\n📡 Processing streaming events...")
    
    streaming_events = [
        {
            'user_id': 'user_001',
            'age': 25,
            'country': 'US',
            'session_duration': 1800,  # 30 minutes
            'event_type': 'session_end'
        },
        {
            'user_id': 'user_002',
            'age': 32,
            'country': 'UK',
            'session_duration': 600,   # 10 minutes
            'event_type': 'session_end'
        },
        {
            'user_id': 'user_001',  # Same user, different session
            'session_duration': 2400,  # 40 minutes
            'event_type': 'session_end'
        }
    ]
    
    for event in streaming_events:
        features_extracted = await pipeline.process_event(event)
        print(f"   Extracted {len(features_extracted)} features from {event['event_type']} for {event['user_id']}")
    
    # Add batch features
    print("\n📊 Adding batch-computed features...")
    
    batch_features = [
        FeatureValue(
            name='user_page_views',
            value=150,
            timestamp=datetime.now(),
            entity_id='user_001',
            source=FeatureSource.BATCH
        ),
        FeatureValue(
            name='user_page_views',
            value=85,
            timestamp=datetime.now(),
            entity_id='user_002',
            source=FeatureSource.BATCH
        )
    ]
    
    await feature_store.write_features(batch_features)
    
    # Demonstrate feature serving for inference
    print("\n🤖 Serving features for ML inference...")
    
    test_users = ['user_001', 'user_002']
    
    for user_id in test_users:
        print(f"\n--- Features for {user_id} ---")
        
        # Get all features
        inference_data = await serving_api.get_features_for_inference(
            entity_id=user_id,
            entity_type='user',
            include_computed=True
        )
        
        print(f"🎯 Feature Vector for Inference:")
        print(f"   Entity: {inference_data['entity_id']} ({inference_data['entity_type']})")
        print(f"   Features ({inference_data['feature_count']}):")
        
        for feature_name, feature_value in inference_data['features'].items():
            print(f"     {feature_name}: {feature_value}")
            
        print(f"   Timestamp: {inference_data['timestamp']}")
        
        # Get specific features
        specific_features = await serving_api.get_features_for_inference(
            entity_id=user_id,
            entity_type='user',
            feature_names=['user_age', 'user_session_minutes']
        )
        
        print(f"🎯 Specific Features:")
        for name, value in specific_features['features'].items():
            print(f"     {name}: {value}")
    
    # Feature store statistics
    print(f"\n📈 Feature Store Statistics:")
    definitions = feature_store.get_feature_definitions()
    print(f"   Registered Features: {len(definitions)}")
    
    for entity_type in ['user']:
        entity_features = feature_store.get_feature_definitions(entity_type)
        print(f"   {entity_type.title()} Features: {len(entity_features)}")
        for fd in entity_features:
            print(f"     - {fd.name} ({fd.feature_type.value}, {fd.source.value})")
    
    # Pipeline metrics
    print(f"\n📊 Pipeline Metrics:")
    metrics = pipeline.metrics.get_stats()
    for key, value in metrics.items():
        formatted_value = f"{value:.2f}" if isinstance(value, float) else value
        print(f"   {key.replace('_', ' ').title()}: {formatted_value}")
    
    # Demonstrate feature freshness
    print(f"\n⏰ Feature Freshness Check:")
    for user_id in test_users:
        vector = await feature_store.read_feature_vector(user_id, 'user')
        print(f"   {user_id}:")
        for name, fv in vector.features.items():
            age_seconds = (datetime.now() - fv.timestamp).total_seconds()
            print(f"     {name}: {age_seconds:.1f}s old (source: {fv.source.value})")


if __name__ == "__main__":
    asyncio.run(demo_feature_store())