"""
Production ML Model Serving Platform
Following TensorFlow Serving, TorchServe, and Seldon Core patterns
"""

import asyncio
import json
import time
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
from collections import defaultdict, deque
import hashlib
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor

class ModelFramework(Enum):
    """Supported ML frameworks"""
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    ONNX = "onnx"
    CUSTOM = "custom"

class ServingMode(Enum):
    """Model serving modes"""
    ONLINE = "online"  # Real-time inference
    BATCH = "batch"  # Batch prediction
    STREAMING = "streaming"  # Stream processing
    EDGE = "edge"  # Edge deployment

class ModelVersion(Enum):
    """Model versioning strategy"""
    LATEST = "latest"
    STABLE = "stable"
    CANARY = "canary"
    SHADOW = "shadow"  # Shadow mode for testing

@dataclass
class ModelConfig:
    """Model configuration"""
    model_id: str
    model_name: str
    framework: ModelFramework
    version: str
    serving_mode: ServingMode
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    batch_size: int = 32
    max_latency_ms: int = 100
    min_replicas: int = 1
    max_replicas: int = 10
    memory_mb: int = 512
    cpu_cores: float = 1.0
    gpu_enabled: bool = False
    warmup_samples: int = 100
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PredictionRequest:
    """Prediction request"""
    request_id: str
    model_id: str
    inputs: Union[np.ndarray, Dict[str, Any]]
    version: Optional[str] = None
    timeout_ms: int = 1000
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PredictionResponse:
    """Prediction response"""
    request_id: str
    model_id: str
    outputs: Union[np.ndarray, Dict[str, Any]]
    version: str
    latency_ms: float
    cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class ModelRegistry:
    """Central model registry (like MLflow Model Registry)"""
    
    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}
        self.versions: Dict[str, List[str]] = defaultdict(list)
        self.aliases: Dict[str, str] = {}  # alias -> model_id:version
        
    def register_model(self, config: ModelConfig, model_artifact: Any):
        """Register new model"""
        model_key = f"{config.model_id}:{config.version}"
        
        self.models[model_key] = {
            "config": config,
            "artifact": model_artifact,
            "registered_at": time.time(),
            "metrics": {},
            "status": "ready"
        }
        
        self.versions[config.model_id].append(config.version)
        
        # Update aliases
        if config.version == "latest":
            self.aliases["latest"] = model_key
    
    def get_model(self, model_id: str, version: Optional[str] = None) -> Optional[Dict]:
        """Get model by ID and version"""
        if version is None:
            version = "latest"
        
        if version in self.aliases:
            model_key = self.aliases[version]
        else:
            model_key = f"{model_id}:{version}"
        
        return self.models.get(model_key)
    
    def list_models(self) -> List[str]:
        """List all registered models"""
        return list(set(k.split(":")[0] for k in self.models.keys()))
    
    def get_versions(self, model_id: str) -> List[str]:
        """Get all versions of a model"""
        return self.versions.get(model_id, [])

class ModelServer:
    """High-performance model serving engine"""
    
    def __init__(self, max_workers: int = 10):
        self.registry = ModelRegistry()
        self.loaded_models: Dict[str, Any] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.cache = PredictionCache()
        self.batcher = RequestBatcher()
        self.metrics = ServingMetrics()
        self.autoscaler = AutoScaler()
        
    async def load_model(self, config: ModelConfig) -> bool:
        """Load model into memory"""
        model_key = f"{config.model_id}:{config.version}"
        
        if model_key in self.loaded_models:
            return True
        
        # Simulate model loading
        await asyncio.sleep(0.5)  # Loading time
        
        # Create mock model based on framework
        if config.framework == ModelFramework.TENSORFLOW:
            model = TensorFlowModel(config)
        elif config.framework == ModelFramework.PYTORCH:
            model = PyTorchModel(config)
        elif config.framework == ModelFramework.SKLEARN:
            model = SklearnModel(config)
        else:
            model = GenericModel(config)
        
        # Warmup model
        await self._warmup_model(model, config)
        
        self.loaded_models[model_key] = model
        
        # Register in registry
        self.registry.register_model(config, model)
        
        return True
    
    async def _warmup_model(self, model: Any, config: ModelConfig):
        """Warmup model with sample predictions"""
        for _ in range(min(config.warmup_samples, 10)):
            # Generate dummy input
            if isinstance(config.input_schema, dict):
                dummy_input = np.random.randn(1, 10)
            else:
                dummy_input = np.random.randn(1, 10)
            
            await model.predict(dummy_input)
    
    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Run prediction"""
        start_time = time.time()
        
        # Check cache
        cached_result = await self.cache.get(request)
        if cached_result:
            return cached_result
        
        # Get model
        version = request.version or "latest"
        model_key = f"{request.model_id}:{version}"
        
        if model_key not in self.loaded_models:
            # Try to load model
            model_data = self.registry.get_model(request.model_id, version)
            if model_data:
                await self.load_model(model_data["config"])
            else:
                raise ValueError(f"Model {model_key} not found")
        
        model = self.loaded_models[model_key]
        
        # Run prediction
        if isinstance(model.config.serving_mode, ServingMode) and \
           model.config.serving_mode == ServingMode.BATCH:
            # Add to batch
            future = await self.batcher.add_request(request, model)
            outputs = await future
        else:
            # Direct prediction
            outputs = await model.predict(request.inputs)
        
        # Create response
        response = PredictionResponse(
            request_id=request.request_id,
            model_id=request.model_id,
            outputs=outputs,
            version=version,
            latency_ms=(time.time() - start_time) * 1000
        )
        
        # Cache result
        await self.cache.set(request, response)
        
        # Track metrics
        await self.metrics.record_prediction(response)
        
        # Check if autoscaling needed
        await self.autoscaler.check_scaling(self.metrics)
        
        return response
    
    async def batch_predict(self, requests: List[PredictionRequest]) -> List[PredictionResponse]:
        """Batch prediction"""
        tasks = [self.predict(req) for req in requests]
        return await asyncio.gather(*tasks)

class TensorFlowModel:
    """TensorFlow model wrapper"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = self._create_mock_model()
    
    def _create_mock_model(self):
        """Create mock TensorFlow model"""
        # In production, load actual TF model
        return lambda x: np.random.randn(*x.shape)
    
    async def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Run TensorFlow prediction"""
        await asyncio.sleep(np.random.uniform(0.01, 0.05))  # Simulate inference
        return self.model(inputs)

class PyTorchModel:
    """PyTorch model wrapper"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = self._create_mock_model()
    
    def _create_mock_model(self):
        """Create mock PyTorch model"""
        return lambda x: np.random.randn(*x.shape)
    
    async def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Run PyTorch prediction"""
        await asyncio.sleep(np.random.uniform(0.01, 0.05))
        return self.model(inputs)

class SklearnModel:
    """Scikit-learn model wrapper"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = self._create_mock_model()
    
    def _create_mock_model(self):
        """Create mock sklearn model"""
        return lambda x: np.random.randn(x.shape[0], 1)
    
    async def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Run sklearn prediction"""
        await asyncio.sleep(np.random.uniform(0.005, 0.02))
        return self.model(inputs)

class GenericModel:
    """Generic model wrapper"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    async def predict(self, inputs: Any) -> Any:
        """Run generic prediction"""
        await asyncio.sleep(np.random.uniform(0.01, 0.03))
        
        if isinstance(inputs, np.ndarray):
            return np.random.randn(*inputs.shape)
        else:
            return {"prediction": np.random.random()}

class PredictionCache:
    """LRU cache for predictions"""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 300):
        self.cache: Dict[str, Tuple[PredictionResponse, float]] = {}
        self.access_order = deque(maxlen=max_size)
        self.max_size = max_size
        self.ttl = ttl_seconds
    
    def _get_cache_key(self, request: PredictionRequest) -> str:
        """Generate cache key"""
        if isinstance(request.inputs, np.ndarray):
            input_hash = hashlib.md5(request.inputs.tobytes()).hexdigest()
        else:
            input_hash = hashlib.md5(str(request.inputs).encode()).hexdigest()
        
        return f"{request.model_id}:{request.version}:{input_hash}"
    
    async def get(self, request: PredictionRequest) -> Optional[PredictionResponse]:
        """Get cached prediction"""
        key = self._get_cache_key(request)
        
        if key in self.cache:
            response, timestamp = self.cache[key]
            
            # Check TTL
            if time.time() - timestamp > self.ttl:
                del self.cache[key]
                return None
            
            # Update access order
            self.access_order.append(key)
            
            # Mark as cached
            cached_response = PredictionResponse(
                request_id=request.request_id,
                model_id=response.model_id,
                outputs=response.outputs,
                version=response.version,
                latency_ms=0,
                cached=True
            )
            
            return cached_response
        
        return None
    
    async def set(self, request: PredictionRequest, response: PredictionResponse):
        """Cache prediction"""
        key = self._get_cache_key(request)
        
        # Evict if at capacity
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            if self.access_order:
                lru_key = self.access_order.popleft()
                if lru_key in self.cache:
                    del self.cache[lru_key]
        
        self.cache[key] = (response, time.time())
        self.access_order.append(key)

class RequestBatcher:
    """Batch requests for efficient processing"""
    
    def __init__(self, max_batch_size: int = 32, max_wait_ms: int = 50):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.batches: Dict[str, List[Tuple[PredictionRequest, asyncio.Future]]] = defaultdict(list)
        self.processing = False
    
    async def add_request(self, request: PredictionRequest, model: Any) -> asyncio.Future:
        """Add request to batch"""
        future = asyncio.Future()
        model_key = f"{request.model_id}:{request.version}"
        
        self.batches[model_key].append((request, future))
        
        # Start processing if batch is full
        if len(self.batches[model_key]) >= self.max_batch_size:
            asyncio.create_task(self._process_batch(model_key, model))
        elif not self.processing:
            # Start timer for batch processing
            asyncio.create_task(self._batch_timer(model_key, model))
        
        return future
    
    async def _batch_timer(self, model_key: str, model: Any):
        """Process batch after timeout"""
        await asyncio.sleep(self.max_wait_ms / 1000)
        await self._process_batch(model_key, model)
    
    async def _process_batch(self, model_key: str, model: Any):
        """Process batched requests"""
        if model_key not in self.batches or not self.batches[model_key]:
            return
        
        batch = self.batches[model_key]
        self.batches[model_key] = []
        
        # Combine inputs
        inputs = []
        for req, _ in batch:
            if isinstance(req.inputs, np.ndarray):
                inputs.append(req.inputs)
            else:
                inputs.append(np.array([req.inputs]))
        
        if inputs:
            batch_input = np.vstack(inputs)
        else:
            batch_input = np.array([])
        
        # Run batch prediction
        batch_output = await model.predict(batch_input)
        
        # Split outputs and resolve futures
        for i, (req, future) in enumerate(batch):
            if i < len(batch_output):
                output = batch_output[i:i+1]
            else:
                output = batch_output[0:1]  # Fallback
            
            future.set_result(output)

class ServingMetrics:
    """Model serving metrics"""
    
    def __init__(self):
        self.predictions = 0
        self.total_latency = 0.0
        self.cache_hits = 0
        self.model_metrics: Dict[str, Dict] = defaultdict(lambda: {
            "count": 0,
            "total_latency": 0,
            "errors": 0
        })
    
    async def record_prediction(self, response: PredictionResponse):
        """Record prediction metrics"""
        self.predictions += 1
        self.total_latency += response.latency_ms
        
        if response.cached:
            self.cache_hits += 1
        
        model_key = f"{response.model_id}:{response.version}"
        self.model_metrics[model_key]["count"] += 1
        self.model_metrics[model_key]["total_latency"] += response.latency_ms
    
    def get_stats(self) -> Dict[str, Any]:
        """Get serving statistics"""
        avg_latency = self.total_latency / self.predictions if self.predictions > 0 else 0
        cache_hit_rate = self.cache_hits / self.predictions if self.predictions > 0 else 0
        
        model_stats = {}
        for model_key, metrics in self.model_metrics.items():
            if metrics["count"] > 0:
                model_stats[model_key] = {
                    "predictions": metrics["count"],
                    "avg_latency_ms": metrics["total_latency"] / metrics["count"],
                    "error_rate": metrics["errors"] / metrics["count"]
                }
        
        return {
            "total_predictions": self.predictions,
            "average_latency_ms": avg_latency,
            "cache_hit_rate": cache_hit_rate,
            "models": model_stats
        }

class AutoScaler:
    """Auto-scaling for model replicas"""
    
    def __init__(self):
        self.replica_counts: Dict[str, int] = defaultdict(lambda: 1)
        self.scaling_history: List[Dict] = []
    
    async def check_scaling(self, metrics: ServingMetrics):
        """Check if scaling is needed"""
        stats = metrics.get_stats()
        
        for model_key, model_stats in stats.get("models", {}).items():
            current_replicas = self.replica_counts[model_key]
            avg_latency = model_stats.get("avg_latency_ms", 0)
            
            # Scale up if latency too high
            if avg_latency > 100 and current_replicas < 10:
                self.replica_counts[model_key] = min(10, current_replicas + 1)
                self.scaling_history.append({
                    "model": model_key,
                    "action": "scale_up",
                    "replicas": self.replica_counts[model_key],
                    "reason": f"high_latency: {avg_latency:.1f}ms"
                })
            
            # Scale down if latency very low
            elif avg_latency < 20 and current_replicas > 1:
                self.replica_counts[model_key] = max(1, current_replicas - 1)
                self.scaling_history.append({
                    "model": model_key,
                    "action": "scale_down",
                    "replicas": self.replica_counts[model_key],
                    "reason": f"low_latency: {avg_latency:.1f}ms"
                })

class ModelMonitor:
    """Model performance monitoring and drift detection"""
    
    def __init__(self):
        self.baseline_metrics: Dict[str, Dict] = {}
        self.drift_scores: Dict[str, float] = defaultdict(float)
        self.alerts: List[Dict] = []
    
    async def set_baseline(self, model_id: str, metrics: Dict):
        """Set baseline metrics for model"""
        self.baseline_metrics[model_id] = metrics
    
    async def check_drift(self, model_id: str, current_metrics: Dict) -> float:
        """Check for model drift"""
        if model_id not in self.baseline_metrics:
            return 0.0
        
        baseline = self.baseline_metrics[model_id]
        
        # Simple drift detection (KL divergence, PSI, etc.)
        drift_score = 0.0
        
        for metric, baseline_value in baseline.items():
            if metric in current_metrics:
                current_value = current_metrics[metric]
                # Simplified drift calculation
                drift_score += abs(current_value - baseline_value) / max(baseline_value, 1)
        
        self.drift_scores[model_id] = drift_score
        
        # Alert if drift too high
        if drift_score > 0.3:  # 30% drift threshold
            self.alerts.append({
                "model_id": model_id,
                "drift_score": drift_score,
                "timestamp": time.time(),
                "severity": "high" if drift_score > 0.5 else "medium"
            })
        
        return drift_score

# Example usage
async def main():
    """Demonstrate ML model serving platform"""
    
    print("🤖 Production ML Model Serving Platform")
    print("=" * 60)
    
    # Initialize model server
    server = ModelServer(max_workers=5)
    
    # Register models
    models = [
        ModelConfig(
            model_id="audio_classifier",
            model_name="Audio Genre Classifier",
            framework=ModelFramework.TENSORFLOW,
            version="v1.0",
            serving_mode=ServingMode.ONLINE,
            input_schema={"audio": "float32[1, 16000]"},
            output_schema={"genres": "float32[10]"},
            max_latency_ms=50
        ),
        ModelConfig(
            model_id="speech_recognizer",
            model_name="Speech Recognition",
            framework=ModelFramework.PYTORCH,
            version="v2.1",
            serving_mode=ServingMode.BATCH,
            input_schema={"audio": "float32[batch, 16000]"},
            output_schema={"text": "string"},
            batch_size=16
        ),
        ModelConfig(
            model_id="anomaly_detector",
            model_name="Anomaly Detection",
            framework=ModelFramework.SKLEARN,
            version="v1.0",
            serving_mode=ServingMode.STREAMING,
            input_schema={"features": "float32[100]"},
            output_schema={"anomaly_score": "float32"}
        )
    ]
    
    print("\n📦 Loading Models:")
    for config in models:
        loaded = await server.load_model(config)
        print(f"  ✓ {config.model_name} ({config.framework.value}) - {config.version}")
    
    # Run predictions
    print("\n🔮 Running Predictions:")
    print("-" * 40)
    
    # Single prediction
    request = PredictionRequest(
        request_id="req_001",
        model_id="audio_classifier",
        inputs=np.random.randn(1, 16000),
        version="v1.0"
    )
    
    response = await server.predict(request)
    print(f"\nAudio Classifier:")
    print(f"  Latency: {response.latency_ms:.2f}ms")
    print(f"  Cached: {response.cached}")
    
    # Batch prediction
    batch_requests = [
        PredictionRequest(
            request_id=f"batch_{i}",
            model_id="speech_recognizer",
            inputs=np.random.randn(1, 16000)
        )
        for i in range(5)
    ]
    
    batch_responses = await server.batch_predict(batch_requests)
    print(f"\nSpeech Recognizer (Batch):")
    print(f"  Batch Size: {len(batch_responses)}")
    print(f"  Avg Latency: {np.mean([r.latency_ms for r in batch_responses]):.2f}ms")
    
    # Test caching
    cached_response = await server.predict(request)  # Same request
    print(f"\nCache Test:")
    print(f"  Cache Hit: {cached_response.cached}")
    print(f"  Latency: {cached_response.latency_ms:.2f}ms")
    
    # Get metrics
    stats = server.metrics.get_stats()
    
    print("\n📊 Serving Metrics:")
    print("-" * 40)
    print(f"Total Predictions: {stats['total_predictions']}")
    print(f"Average Latency: {stats['average_latency_ms']:.2f}ms")
    print(f"Cache Hit Rate: {stats['cache_hit_rate']*100:.1f}%")
    
    print("\n✅ ML model serving platform operational!")
    
    return server

if __name__ == "__main__":
    asyncio.run(main())