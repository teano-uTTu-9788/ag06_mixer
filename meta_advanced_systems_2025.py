#!/usr/bin/env python3
"""
Meta (Facebook) Advanced Systems Implementation 2025
Implements TAO, Prophet, PyTorch Serving, and other Meta-scale systems
"""

import asyncio
import json
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import hashlib
import threading
from datetime import datetime, timedelta
import uuid
import random

# ============================================================================
# META TAO - The Associations and Objects System
# ============================================================================

@dataclass
class TAOObject:
    """Represents an object in TAO (like a user, post, etc)"""
    id: int
    type: str
    data: Dict[str, Any]
    created_at: float
    updated_at: float
    shard_id: Optional[int] = None

@dataclass
class TAOAssociation:
    """Represents an association (edge) between objects"""
    id1: int  # Source object
    atype: str  # Association type (friend, likes, follows)
    id2: int  # Destination object
    time: float
    data: Optional[Dict] = None
    
class TAOShard:
    """TAO shard that stores objects and associations"""
    
    def __init__(self, shard_id: int):
        self.shard_id = shard_id
        self.objects: Dict[int, TAOObject] = {}
        self.associations: Dict[str, List[TAOAssociation]] = defaultdict(list)
        self.assoc_counts: Dict[str, int] = defaultdict(int)
        self.lock = threading.Lock()
        
    def add_object(self, obj: TAOObject) -> bool:
        """Add an object to this shard"""
        with self.lock:
            obj.shard_id = self.shard_id
            self.objects[obj.id] = obj
            return True
    
    def add_association(self, assoc: TAOAssociation) -> bool:
        """Add an association"""
        with self.lock:
            key = f"{assoc.id1}:{assoc.atype}"
            self.associations[key].append(assoc)
            self.assoc_counts[key] += 1
            
            # Also index reverse for bidirectional queries
            if assoc.atype in ["friend", "follows"]:
                reverse_key = f"{assoc.id2}:{assoc.atype}:reverse"
                self.associations[reverse_key].append(assoc)
            return True
    
    def get_associations(self, id1: int, atype: str, limit: int = 50) -> List[TAOAssociation]:
        """Get associations for an object"""
        with self.lock:
            key = f"{id1}:{atype}"
            assocs = self.associations.get(key, [])
            # Sort by time descending
            assocs.sort(key=lambda a: a.time, reverse=True)
            return assocs[:limit]
    
    def get_association_count(self, id1: int, atype: str) -> int:
        """Get count of associations"""
        with self.lock:
            key = f"{id1}:{atype}"
            return self.assoc_counts.get(key, 0)

class TAOCache:
    """TAO cache layer (like Facebook's Memcache layer)"""
    
    def __init__(self, size_limit: int = 10000):
        self.cache: Dict[str, Any] = {}
        self.size_limit = size_limit
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get from cache"""
        with self.lock:
            if key in self.cache:
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any):
        """Set in cache with LRU eviction"""
        with self.lock:
            if len(self.cache) >= self.size_limit:
                # Simple eviction - remove first item
                first_key = next(iter(self.cache))
                del self.cache[first_key]
            self.cache[key] = value
    
    def invalidate(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        with self.lock:
            keys_to_remove = [k for k in self.cache if pattern in k]
            for key in keys_to_remove:
                del self.cache[key]
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate"""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0

class TAOSystem:
    """The TAO distributed graph system"""
    
    def __init__(self, num_shards: int = 4):
        self.shards: List[TAOShard] = []
        self.cache = TAOCache()
        self.num_shards = num_shards
        self.object_counter = 0
        self.query_stats = defaultdict(int)
        
        # Initialize shards
        for i in range(num_shards):
            self.shards.append(TAOShard(i))
    
    def _get_shard(self, obj_id: int) -> TAOShard:
        """Get shard for an object ID"""
        return self.shards[obj_id % self.num_shards]
    
    def create_object(self, obj_type: str, data: Dict) -> TAOObject:
        """Create a new object"""
        self.object_counter += 1
        obj = TAOObject(
            id=self.object_counter,
            type=obj_type,
            data=data,
            created_at=time.time(),
            updated_at=time.time()
        )
        
        shard = self._get_shard(obj.id)
        shard.add_object(obj)
        
        # Invalidate relevant cache
        self.cache.invalidate(f"obj:{obj.id}")
        
        return obj
    
    def create_association(self, id1: int, atype: str, id2: int, data: Dict = None) -> TAOAssociation:
        """Create an association between objects"""
        assoc = TAOAssociation(
            id1=id1,
            atype=atype,
            id2=id2,
            time=time.time(),
            data=data
        )
        
        shard = self._get_shard(id1)
        shard.add_association(assoc)
        
        # Invalidate cache
        self.cache.invalidate(f"assoc:{id1}:{atype}")
        
        return assoc
    
    def assoc_range(self, id1: int, atype: str, limit: int = 50) -> List[TAOAssociation]:
        """Range query for associations (e.g., get user's friends)"""
        self.query_stats['assoc_range'] += 1
        
        # Check cache first
        cache_key = f"assoc:{id1}:{atype}:{limit}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        # Get from shard
        shard = self._get_shard(id1)
        assocs = shard.get_associations(id1, atype, limit)
        
        # Cache result
        self.cache.set(cache_key, assocs)
        
        return assocs
    
    def assoc_count(self, id1: int, atype: str) -> int:
        """Count associations (e.g., count friends)"""
        self.query_stats['assoc_count'] += 1
        
        # Check cache
        cache_key = f"count:{id1}:{atype}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Get from shard
        shard = self._get_shard(id1)
        count = shard.get_association_count(id1, atype)
        
        # Cache result
        self.cache.set(cache_key, count)
        
        return count

# ============================================================================
# META PROPHET - Time Series Forecasting
# ============================================================================

class ProphetForecaster:
    """Simplified Prophet time series forecasting"""
    
    def __init__(self):
        self.trend_params = None
        self.seasonality_params = None
        self.holiday_params = None
        self.fitted = False
        
    def fit(self, timestamps: List[float], values: List[float]):
        """Fit the Prophet model"""
        # Simplified trend fitting
        x = np.array(timestamps)
        y = np.array(values)
        
        # Linear trend
        coeffs = np.polyfit(x, y, 1)
        self.trend_params = {"slope": coeffs[0], "intercept": coeffs[1]}
        
        # Simplified seasonality (daily, weekly patterns)
        # Extract hour of day and day of week
        hours = [(t % 86400) / 3600 for t in timestamps]
        days = [(t % 604800) / 86400 for t in timestamps]
        
        # Simple averaging for seasonality
        hourly_avg = {}
        for h, v in zip(hours, values):
            hour_bucket = int(h)
            if hour_bucket not in hourly_avg:
                hourly_avg[hour_bucket] = []
            hourly_avg[hour_bucket].append(v)
        
        self.seasonality_params = {
            "hourly": {h: np.mean(vals) for h, vals in hourly_avg.items()},
            "weekly_amplitude": np.std(values) * 0.1  # Simplified
        }
        
        self.fitted = True
    
    def predict(self, future_timestamps: List[float]) -> List[float]:
        """Make predictions"""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        
        for t in future_timestamps:
            # Trend component
            trend = self.trend_params["slope"] * t + self.trend_params["intercept"]
            
            # Seasonality component
            hour = int((t % 86400) / 3600)
            hourly_effect = self.seasonality_params["hourly"].get(hour, 0)
            
            # Weekly pattern (sine wave approximation)
            day_of_week = (t % 604800) / 86400
            weekly_effect = self.seasonality_params["weekly_amplitude"] * np.sin(2 * np.pi * day_of_week / 7)
            
            # Combine components
            prediction = trend + hourly_effect + weekly_effect
            
            # Add some noise for realism
            prediction += random.gauss(0, abs(prediction) * 0.05)
            
            predictions.append(max(0, prediction))  # Ensure non-negative
        
        return predictions
    
    def get_components(self) -> Dict:
        """Get model components for interpretability"""
        return {
            "trend": self.trend_params,
            "seasonality": {
                "hourly_patterns": len(self.seasonality_params.get("hourly", {})),
                "weekly_amplitude": self.seasonality_params.get("weekly_amplitude", 0)
            }
        }

# ============================================================================
# META PYTORCH SERVING - Model Serving Infrastructure
# ============================================================================

@dataclass
class ModelVersion:
    """Represents a model version"""
    model_id: str
    version: int
    framework: str
    size_mb: float
    created_at: float
    metrics: Dict[str, float] = field(default_factory=dict)
    status: str = "staging"  # staging, production, retired

class TorchServeHandler:
    """PyTorch model serving handler"""
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = None  # Would load actual model
        self.batch_size = 32
        self.timeout_ms = 100
        self.requests_processed = 0
        
    def preprocess(self, data: Any) -> Any:
        """Preprocess input data"""
        # Simulate preprocessing
        return {"preprocessed": data, "timestamp": time.time()}
    
    def inference(self, data: Any) -> Any:
        """Run model inference"""
        # Simulate inference
        time.sleep(random.uniform(0.01, 0.05))  # Simulate processing time
        self.requests_processed += 1
        
        return {
            "predictions": [random.random() for _ in range(10)],
            "model_id": self.model_id,
            "processing_time_ms": random.uniform(10, 50)
        }
    
    def postprocess(self, data: Any) -> Any:
        """Postprocess model output"""
        # Simulate postprocessing
        if "predictions" in data:
            data["predictions"] = [round(p, 4) for p in data["predictions"]]
        return data

class PyTorchServing:
    """PyTorch model serving system (like TorchServe)"""
    
    def __init__(self):
        self.models: Dict[str, ModelVersion] = {}
        self.handlers: Dict[str, TorchServeHandler] = {}
        self.request_queue = deque()
        self.metrics = defaultdict(list)
        self.ab_tests = {}
        
    def register_model(self, model_id: str, framework: str = "pytorch", size_mb: float = 100) -> ModelVersion:
        """Register a new model"""
        version = len([m for m in self.models.values() if m.model_id == model_id]) + 1
        
        model = ModelVersion(
            model_id=model_id,
            version=version,
            framework=framework,
            size_mb=size_mb,
            created_at=time.time()
        )
        
        key = f"{model_id}:v{version}"
        self.models[key] = model
        self.handlers[key] = TorchServeHandler(model_id)
        
        return model
    
    def predict(self, model_id: str, data: Any, version: Optional[int] = None) -> Dict:
        """Make prediction with a model"""
        # Get model version
        if version:
            key = f"{model_id}:v{version}"
        else:
            # Get latest production version
            prod_models = [m for m in self.models.values() 
                          if m.model_id == model_id and m.status == "production"]
            if not prod_models:
                # Fall back to latest staging
                staging_models = [m for m in self.models.values() 
                                if m.model_id == model_id and m.status == "staging"]
                if staging_models:
                    model = max(staging_models, key=lambda m: m.version)
                else:
                    raise ValueError(f"No model found: {model_id}")
            else:
                model = max(prod_models, key=lambda m: m.version)
            key = f"{model.model_id}:v{model.version}"
        
        if key not in self.handlers:
            raise ValueError(f"Model not found: {key}")
        
        handler = self.handlers[key]
        
        # Process request
        start_time = time.time()
        
        preprocessed = handler.preprocess(data)
        inference_result = handler.inference(preprocessed)
        final_result = handler.postprocess(inference_result)
        
        # Record metrics
        latency_ms = (time.time() - start_time) * 1000
        self.metrics[key].append(latency_ms)
        
        final_result["model_version"] = key
        final_result["latency_ms"] = latency_ms
        
        return final_result
    
    def setup_ab_test(self, test_id: str, model_a: str, model_b: str, traffic_split: float = 0.5):
        """Set up A/B test between models"""
        self.ab_tests[test_id] = {
            "model_a": model_a,
            "model_b": model_b,
            "traffic_split": traffic_split,
            "results_a": [],
            "results_b": []
        }
    
    def predict_with_ab_test(self, test_id: str, data: Any) -> Dict:
        """Predict with A/B testing"""
        if test_id not in self.ab_tests:
            raise ValueError(f"A/B test not found: {test_id}")
        
        test = self.ab_tests[test_id]
        
        # Route traffic
        use_model_a = random.random() < test["traffic_split"]
        model = test["model_a"] if use_model_a else test["model_b"]
        
        result = self.predict(model, data)
        
        # Record for analysis
        if use_model_a:
            test["results_a"].append(result.get("latency_ms", 0))
        else:
            test["results_b"].append(result.get("latency_ms", 0))
        
        result["ab_test"] = {
            "test_id": test_id,
            "variant": "A" if use_model_a else "B"
        }
        
        return result
    
    def promote_to_production(self, model_id: str, version: int):
        """Promote model version to production"""
        key = f"{model_id}:v{version}"
        if key in self.models:
            # Retire old production versions
            for model in self.models.values():
                if model.model_id == model_id and model.status == "production":
                    model.status = "retired"
            
            # Promote new version
            self.models[key].status = "production"
            return True
        return False

# ============================================================================
# META HYDRA - Configuration Management
# ============================================================================

class HydraConfig:
    """Meta's Hydra-like configuration system"""
    
    def __init__(self):
        self.base_config = {}
        self.overrides = []
        self.composed = {}
        self.config_groups = defaultdict(dict)
        
    def add_config_group(self, group: str, name: str, config: Dict):
        """Add a configuration to a group"""
        self.config_groups[group][name] = config
    
    def compose(self, groups: Dict[str, str], overrides: List[str] = None) -> Dict:
        """Compose configuration from groups and overrides"""
        composed = {}
        
        # Apply group configs
        for group, selection in groups.items():
            if group in self.config_groups and selection in self.config_groups[group]:
                composed.update(self.config_groups[group][selection])
        
        # Apply overrides (format: "key=value" or "key.nested=value")
        if overrides:
            for override in overrides:
                if "=" in override:
                    key, value = override.split("=", 1)
                    # Handle nested keys
                    keys = key.split(".")
                    current = composed
                    for k in keys[:-1]:
                        if k not in current:
                            current[k] = {}
                        current = current[k]
                    
                    # Try to parse value
                    try:
                        value = json.loads(value)
                    except:
                        pass  # Keep as string
                    
                    current[keys[-1]] = value
        
        self.composed = composed
        return composed
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        keys = key.split(".")
        current = self.composed
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current

# ============================================================================
# META SYSTEMS ORCHESTRATOR
# ============================================================================

class MetaSystemsOrchestrator:
    """Orchestrates all Meta-style systems"""
    
    def __init__(self):
        self.tao = TAOSystem()
        self.prophet = ProphetForecaster()
        self.pytorch_serving = PyTorchServing()
        self.hydra = HydraConfig()
        self.metrics = defaultdict(int)
        
        # Initialize Hydra configs
        self._init_hydra_configs()
    
    def _init_hydra_configs(self):
        """Initialize Hydra configuration groups"""
        # Model configs
        self.hydra.add_config_group("model", "small", {
            "batch_size": 32,
            "hidden_size": 256,
            "num_layers": 4
        })
        
        self.hydra.add_config_group("model", "large", {
            "batch_size": 16,
            "hidden_size": 1024,
            "num_layers": 12
        })
        
        # Training configs
        self.hydra.add_config_group("training", "fast", {
            "learning_rate": 0.001,
            "epochs": 10,
            "optimizer": "adam"
        })
        
        self.hydra.add_config_group("training", "accurate", {
            "learning_rate": 0.0001,
            "epochs": 100,
            "optimizer": "sgd"
        })
    
    async def demonstrate_tao(self):
        """Demonstrate TAO graph database"""
        print("\n🔗 Meta TAO Demonstration")
        print("=" * 50)
        
        # Create users
        user1 = self.tao.create_object("user", {"name": "Alice", "age": 25})
        user2 = self.tao.create_object("user", {"name": "Bob", "age": 30})
        user3 = self.tao.create_object("user", {"name": "Charlie", "age": 28})
        
        print(f"Created {3} users")
        
        # Create associations
        self.tao.create_association(user1.id, "friend", user2.id)
        self.tao.create_association(user1.id, "friend", user3.id)
        self.tao.create_association(user2.id, "friend", user3.id)
        
        # Create posts
        post1 = self.tao.create_object("post", {"content": "Hello TAO!", "author": user1.id})
        post2 = self.tao.create_object("post", {"content": "Graph databases rock!", "author": user2.id})
        
        # Create likes
        self.tao.create_association(user2.id, "likes", post1.id)
        self.tao.create_association(user3.id, "likes", post1.id)
        self.tao.create_association(user1.id, "likes", post2.id)
        
        print(f"Created associations: friends, posts, likes")
        
        # Query associations
        print("\nQuerying TAO:")
        friends = self.tao.assoc_range(user1.id, "friend", limit=10)
        print(f"  User 1 friends: {len(friends)} connections")
        
        likes_count = self.tao.assoc_count(post1.id, "likes")
        print(f"  Post 1 likes: {likes_count}")
        
        # Show cache performance
        print(f"\nCache Performance:")
        print(f"  Hit rate: {self.tao.cache.get_hit_rate():.1f}%")
        print(f"  Hits: {self.tao.cache.hits}, Misses: {self.tao.cache.misses}")
        
        # Show sharding
        print(f"\nSharding:")
        for shard in self.tao.shards:
            print(f"  Shard {shard.shard_id}: {len(shard.objects)} objects, "
                  f"{sum(len(v) for v in shard.associations.values())} associations")
        
        self.metrics['tao_objects'] = self.tao.object_counter
    
    async def demonstrate_prophet(self):
        """Demonstrate Prophet forecasting"""
        print("\n📈 Meta Prophet Demonstration")
        print("=" * 50)
        
        # Generate sample time series data
        now = time.time()
        timestamps = [now - 3600 * i for i in range(168, 0, -1)]  # Last week hourly
        
        # Generate values with trend and seasonality
        values = []
        for t in timestamps:
            hour = (t % 86400) / 3600
            day = (t % 604800) / 86400
            
            # Base trend
            trend = 100 + (t - timestamps[0]) / 3600 * 0.5
            
            # Daily pattern (peak at noon)
            daily = 20 * np.sin(2 * np.pi * (hour - 6) / 24)
            
            # Weekly pattern (higher on weekends)
            weekly = 10 if day > 5 else 0
            
            # Add noise
            noise = random.gauss(0, 5)
            
            value = max(0, trend + daily + weekly + noise)
            values.append(value)
        
        # Fit Prophet model
        self.prophet.fit(timestamps, values)
        print("Fitted Prophet model on 168 hourly data points")
        
        # Make predictions
        future_timestamps = [now + 3600 * i for i in range(1, 25)]  # Next 24 hours
        predictions = self.prophet.predict(future_timestamps)
        
        print(f"\nPredictions for next 24 hours:")
        print(f"  Min: {min(predictions):.1f}")
        print(f"  Max: {max(predictions):.1f}")
        print(f"  Mean: {np.mean(predictions):.1f}")
        
        # Show components
        components = self.prophet.get_components()
        print(f"\nModel Components:")
        print(f"  Trend: slope={components['trend']['slope']:.4f}")
        print(f"  Hourly patterns: {components['seasonality']['hourly_patterns']}")
        print(f"  Weekly amplitude: {components['seasonality']['weekly_amplitude']:.2f}")
        
        self.metrics['prophet_predictions'] = len(predictions)
    
    async def demonstrate_pytorch_serving(self):
        """Demonstrate PyTorch model serving"""
        print("\n🤖 Meta PyTorch Serving Demonstration")
        print("=" * 50)
        
        # Register models
        model_v1 = self.pytorch_serving.register_model("recommendation_model", size_mb=150)
        model_v2 = self.pytorch_serving.register_model("recommendation_model", size_mb=180)
        
        print(f"Registered model versions:")
        print(f"  {model_v1.model_id} v{model_v1.version} ({model_v1.size_mb}MB)")
        print(f"  {model_v2.model_id} v{model_v2.version} ({model_v2.size_mb}MB)")
        
        # Promote v2 to production
        self.pytorch_serving.promote_to_production("recommendation_model", 2)
        print(f"\nPromoted v2 to production")
        
        # Set up A/B test
        self.pytorch_serving.setup_ab_test(
            "model_comparison",
            "recommendation_model:v1",
            "recommendation_model:v2",
            traffic_split=0.5
        )
        
        # Run predictions with A/B test
        print("\nRunning A/B test predictions:")
        for i in range(10):
            result = self.pytorch_serving.predict_with_ab_test(
                "model_comparison",
                {"user_id": i, "context": "homepage"}
            )
            variant = result["ab_test"]["variant"]
            latency = result["latency_ms"]
            print(f"  Request {i}: Variant {variant}, Latency {latency:.1f}ms")
        
        # Show A/B test results
        test = self.pytorch_serving.ab_tests["model_comparison"]
        if test["results_a"] and test["results_b"]:
            print(f"\nA/B Test Results:")
            print(f"  Model A avg latency: {np.mean(test['results_a']):.1f}ms")
            print(f"  Model B avg latency: {np.mean(test['results_b']):.1f}ms")
        
        self.metrics['models_served'] = len(self.pytorch_serving.models)
    
    async def demonstrate_hydra(self):
        """Demonstrate Hydra configuration"""
        print("\n⚙️ Meta Hydra Configuration")
        print("=" * 50)
        
        # Compose configuration
        config = self.hydra.compose(
            groups={
                "model": "large",
                "training": "accurate"
            },
            overrides=[
                "training.learning_rate=0.0005",
                "model.dropout=0.2",
                "experiment.name=production_v2"
            ]
        )
        
        print("Composed configuration:")
        print(json.dumps(config, indent=2))
        
        # Access nested values
        print(f"\nAccessing values:")
        print(f"  Batch size: {self.hydra.get('model.batch_size', 'not found')}")
        print(f"  Learning rate: {self.hydra.get('training.learning_rate', 'not found')}")
        print(f"  Experiment name: {self.hydra.get('experiment.name', 'not found')}")
        
        self.metrics['hydra_configs'] = len(self.hydra.config_groups)
    
    def get_metrics(self) -> Dict:
        """Get all system metrics"""
        return {
            "meta_systems": {
                "tao": {
                    "objects": self.metrics.get('tao_objects', 0),
                    "shards": len(self.tao.shards),
                    "cache_hit_rate": self.tao.cache.get_hit_rate()
                },
                "prophet": {
                    "predictions_made": self.metrics.get('prophet_predictions', 0),
                    "model_fitted": self.prophet.fitted
                },
                "pytorch_serving": {
                    "models": self.metrics.get('models_served', 0),
                    "ab_tests": len(self.pytorch_serving.ab_tests),
                    "requests_processed": sum(h.requests_processed for h in self.pytorch_serving.handlers.values())
                },
                "hydra": {
                    "config_groups": self.metrics.get('hydra_configs', 0),
                    "composed_configs": len(self.hydra.composed)
                }
            }
        }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main demonstration of Meta systems"""
    orchestrator = MetaSystemsOrchestrator()
    
    print("🌐 META ADVANCED SYSTEMS DEMONSTRATION")
    print("Implementing TAO, Prophet, PyTorch Serving, and Hydra")
    
    # Run all demonstrations
    await orchestrator.demonstrate_tao()
    await asyncio.sleep(0.5)
    
    await orchestrator.demonstrate_prophet()
    await asyncio.sleep(0.5)
    
    await orchestrator.demonstrate_pytorch_serving()
    await asyncio.sleep(0.5)
    
    await orchestrator.demonstrate_hydra()
    
    # Show final metrics
    print("\n📊 FINAL METRICS")
    print("=" * 50)
    metrics = orchestrator.get_metrics()
    print(json.dumps(metrics, indent=2))
    
    return orchestrator

if __name__ == "__main__":
    asyncio.run(main())