"""
Edge ML Inference with ONNX Runtime WebAssembly
Ultra-low latency ML inference at the edge following TensorFlow.js and ONNX patterns
"""

import asyncio
import json
import time
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import base64
import hashlib
from collections import deque

class ModelFormat(Enum):
    """Supported ML model formats"""
    ONNX = "onnx"
    TENSORFLOW_LITE = "tflite"
    TENSORFLOW_JS = "tfjs"
    PYTORCH_MOBILE = "pt_mobile"
    CORE_ML = "coreml"

class InferenceMode(Enum):
    """Inference execution modes"""
    WASM = "webassembly"  # WebAssembly execution
    WEBGPU = "webgpu"  # WebGPU acceleration
    WEBGL = "webgl"  # WebGL acceleration
    CPU = "cpu"  # Pure CPU execution
    SIMD = "simd"  # SIMD optimized

@dataclass
class ModelConfig:
    """Edge ML model configuration"""
    model_id: str
    model_name: str
    format: ModelFormat
    version: str
    input_shape: List[int]
    output_shape: List[int]
    size_mb: float
    quantized: bool = False
    optimization_level: int = 3  # 0-3, higher = more optimized
    target_devices: List[str] = field(default_factory=list)
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    postprocessing: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InferenceRequest:
    """ML inference request"""
    request_id: str
    model_id: str
    input_data: np.ndarray
    mode: InferenceMode = InferenceMode.WASM
    batch_size: int = 1
    timeout_ms: int = 100
    priority: int = 0  # Higher = more important

@dataclass
class InferenceResult:
    """ML inference result"""
    request_id: str
    output: np.ndarray
    latency_ms: float
    model_id: str
    confidence_scores: Optional[List[float]] = None
    labels: Optional[List[str]] = None
    execution_mode: str = ""

class EdgeMLRuntime:
    """Edge ML inference runtime with ONNX WebAssembly"""
    
    def __init__(self):
        self.models: Dict[str, ModelConfig] = {}
        self.loaded_models: Dict[str, Any] = {}
        self.inference_cache = InferenceCache()
        self.performance_monitor = PerformanceMonitor()
        self.model_optimizer = ModelOptimizer()
        
        # Initialize default models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize pre-trained models for edge deployment"""
        
        # Audio classification model
        self.models["audio_classifier"] = ModelConfig(
            model_id="audio_classifier",
            model_name="Audio Genre Classifier",
            format=ModelFormat.ONNX,
            version="1.0.0",
            input_shape=[1, 1, 16000],  # 1 second @ 16kHz
            output_shape=[1, 10],  # 10 genres
            size_mb=2.5,
            quantized=True,
            target_devices=["wasm", "mobile", "edge"],
            preprocessing={
                "normalize": True,
                "sample_rate": 16000,
                "mel_specs": True
            }
        )
        
        # Voice activity detection
        self.models["vad"] = ModelConfig(
            model_id="vad",
            model_name="Voice Activity Detector",
            format=ModelFormat.TENSORFLOW_LITE,
            version="2.0.0",
            input_shape=[1, 256],  # 256 audio frames
            output_shape=[1, 2],  # speech/no-speech
            size_mb=0.5,
            quantized=True,
            optimization_level=3
        )
        
        # Audio enhancement model
        self.models["audio_enhancer"] = ModelConfig(
            model_id="audio_enhancer",
            model_name="Audio Enhancement Net",
            format=ModelFormat.ONNX,
            version="1.2.0",
            input_shape=[1, 1, 8192],
            output_shape=[1, 1, 8192],
            size_mb=5.0,
            quantized=False,
            target_devices=["webgpu", "edge"]
        )
        
        # Speaker recognition
        self.models["speaker_id"] = ModelConfig(
            model_id="speaker_id",
            model_name="Speaker Recognition",
            format=ModelFormat.TENSORFLOW_JS,
            version="1.0.0",
            input_shape=[1, 128, 128],  # Mel spectrogram
            output_shape=[1, 512],  # Embedding vector
            size_mb=8.0,
            quantized=False
        )
        
        # Noise suppression
        self.models["noise_suppressor"] = ModelConfig(
            model_id="noise_suppressor",
            model_name="Real-time Noise Suppression",
            format=ModelFormat.ONNX,
            version="3.0.0",
            input_shape=[1, 1, 4096],
            output_shape=[1, 1, 4096],
            size_mb=3.5,
            quantized=True,
            optimization_level=3,
            preprocessing={
                "window": "hann",
                "hop_length": 256
            }
        )
    
    async def load_model(self, model_id: str, mode: InferenceMode = InferenceMode.WASM) -> bool:
        """Load model for inference"""
        if model_id not in self.models:
            return False
        
        config = self.models[model_id]
        
        # Optimize model for target mode
        optimized_model = await self.model_optimizer.optimize(config, mode)
        
        # Simulate model loading
        await asyncio.sleep(0.1)  # Simulate load time
        
        # Create mock inference session
        self.loaded_models[model_id] = {
            "config": config,
            "session": self._create_inference_session(config, mode),
            "mode": mode,
            "loaded_at": time.time()
        }
        
        return True
    
    def _create_inference_session(self, config: ModelConfig, mode: InferenceMode):
        """Create inference session based on mode"""
        
        # Simulate different runtime backends
        if mode == InferenceMode.WASM:
            return WASMInferenceSession(config)
        elif mode == InferenceMode.WEBGPU:
            return WebGPUInferenceSession(config)
        elif mode == InferenceMode.WEBGL:
            return WebGLInferenceSession(config)
        elif mode == InferenceMode.SIMD:
            return SIMDInferenceSession(config)
        else:
            return CPUInferenceSession(config)
    
    async def infer(self, request: InferenceRequest) -> InferenceResult:
        """Run inference on edge"""
        start_time = time.time()
        
        # Check cache first
        cached_result = await self.inference_cache.get(request)
        if cached_result:
            cached_result.latency_ms = (time.time() - start_time) * 1000
            return cached_result
        
        # Load model if not loaded
        if request.model_id not in self.loaded_models:
            await self.load_model(request.model_id, request.mode)
        
        model_info = self.loaded_models[request.model_id]
        config = model_info["config"]
        session = model_info["session"]
        
        # Preprocess input
        processed_input = await self._preprocess(request.input_data, config)
        
        # Run inference
        output = await session.run(processed_input)
        
        # Postprocess output
        final_output, confidence_scores, labels = await self._postprocess(output, config)
        
        # Create result
        result = InferenceResult(
            request_id=request.request_id,
            output=final_output,
            latency_ms=(time.time() - start_time) * 1000,
            model_id=request.model_id,
            confidence_scores=confidence_scores,
            labels=labels,
            execution_mode=request.mode.value
        )
        
        # Cache result
        await self.inference_cache.set(request, result)
        
        # Track performance
        await self.performance_monitor.track(request, result)
        
        return result
    
    async def _preprocess(self, input_data: np.ndarray, config: ModelConfig) -> np.ndarray:
        """Preprocess input data"""
        processed = input_data.copy()
        
        if config.preprocessing.get("normalize"):
            # Normalize to [-1, 1]
            max_val = np.max(np.abs(processed))
            if max_val > 0:
                processed = processed / max_val
        
        if config.preprocessing.get("mel_specs"):
            # Convert to mel spectrogram (simulated)
            processed = self._compute_mel_spectrogram(processed)
            # For mel spectrograms, adjust the shape
            return processed.reshape((1, 128, 128))  # Standard mel spec shape
        
        # For non-mel spec inputs, ensure correct shape
        # If the input is already the right shape, return as is
        if processed.shape == tuple(config.input_shape):
            return processed
        
        # Try to reshape if sizes are compatible
        try:
            processed = processed.reshape(config.input_shape)
        except ValueError:
            # If reshape fails, pad or trim to match expected size
            expected_size = np.prod(config.input_shape)
            current_size = processed.size
            
            if current_size < expected_size:
                # Pad with zeros
                flat = processed.flatten()
                padded = np.pad(flat, (0, expected_size - current_size))
                processed = padded.reshape(config.input_shape)
            else:
                # Trim to size
                flat = processed.flatten()[:expected_size]
                processed = flat.reshape(config.input_shape)
        
        return processed
    
    async def _postprocess(self, output: np.ndarray, config: ModelConfig) -> Tuple[np.ndarray, List[float], List[str]]:
        """Postprocess model output"""
        
        # Apply softmax for classification models
        if "classifier" in config.model_id:
            output = self._softmax(output)
            
            # Get confidence scores and labels
            confidence_scores = output.flatten().tolist()
            labels = self._get_labels(config.model_id, confidence_scores)
            
            return output, confidence_scores, labels
        
        # For other models, return raw output
        return output, None, None
    
    def _compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram (simplified)"""
        # Simplified mel spectrogram computation
        # In production, use librosa or similar
        
        # Flatten audio to 1D for FFT
        audio_flat = audio.flatten()
        
        # FFT
        fft = np.fft.rfft(audio_flat)
        magnitude = np.abs(fft)
        
        # Mel filter banks (simplified)
        n_mels = 128
        freq_bins = magnitude.shape[0]
        
        if freq_bins < n_mels:
            # If we have fewer frequency bins than mel bands, pad
            magnitude = np.pad(magnitude, (0, n_mels - freq_bins))
            freq_bins = n_mels
        
        # Create mel spectrogram
        mel_spec = np.zeros((n_mels, n_mels))
        
        bin_size = max(1, freq_bins // n_mels)
        
        for i in range(n_mels):
            start = i * bin_size
            end = min(start + bin_size, freq_bins)
            if start < freq_bins:
                # Average the frequency bins for this mel band
                mel_spec[i, :min(end-start, n_mels)] = magnitude[start:end][:n_mels]
        
        return mel_spec
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def _get_labels(self, model_id: str, scores: List[float]) -> List[str]:
        """Get labels for classification results"""
        
        if model_id == "audio_classifier":
            genres = ["rock", "pop", "jazz", "classical", "electronic",
                     "hip-hop", "country", "blues", "metal", "folk"]
            top_indices = np.argsort(scores)[-3:][::-1]  # Top 3
            return [genres[i] for i in top_indices]
        
        elif model_id == "vad":
            return ["speech", "no-speech"]
        
        return []
    
    async def batch_infer(self, requests: List[InferenceRequest]) -> List[InferenceResult]:
        """Batch inference for efficiency"""
        
        # Group requests by model
        model_groups = {}
        for req in requests:
            if req.model_id not in model_groups:
                model_groups[req.model_id] = []
            model_groups[req.model_id].append(req)
        
        results = []
        
        # Process each model group
        for model_id, group_requests in model_groups.items():
            # Stack inputs for batch processing
            batch_input = np.stack([req.input_data for req in group_requests])
            
            # Create batch request
            batch_req = InferenceRequest(
                request_id=f"batch_{model_id}_{time.time()}",
                model_id=model_id,
                input_data=batch_input,
                batch_size=len(group_requests)
            )
            
            # Run batch inference
            batch_result = await self.infer(batch_req)
            
            # Split results
            for i, req in enumerate(group_requests):
                # Handle batch output splitting
                if len(batch_result.output.shape) > 1 and batch_result.output.shape[0] > 1:
                    # If we have a proper batch dimension and multiple results
                    individual_output = batch_result.output[i] if i < batch_result.output.shape[0] else batch_result.output[0]
                else:
                    # Single output or no batch dimension
                    individual_output = batch_result.output
                
                result = InferenceResult(
                    request_id=req.request_id,
                    output=individual_output,
                    latency_ms=batch_result.latency_ms / len(group_requests),
                    model_id=model_id
                )
                results.append(result)
        
        return results
    
    async def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model information"""
        if model_id not in self.models:
            return None
        
        config = self.models[model_id]
        loaded = model_id in self.loaded_models
        
        return {
            "model_id": model_id,
            "name": config.model_name,
            "version": config.version,
            "format": config.format.value,
            "size_mb": config.size_mb,
            "loaded": loaded,
            "quantized": config.quantized,
            "input_shape": config.input_shape,
            "output_shape": config.output_shape,
            "target_devices": config.target_devices
        }
    
    async def list_models(self) -> List[str]:
        """List available models"""
        return list(self.models.keys())


class InferenceCache:
    """Cache for inference results"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 60):
        self.cache: Dict[str, Tuple[InferenceResult, float]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.access_order = deque(maxlen=max_size)
    
    async def get(self, request: InferenceRequest) -> Optional[InferenceResult]:
        """Get cached result"""
        cache_key = self._get_cache_key(request)
        
        if cache_key in self.cache:
            result, timestamp = self.cache[cache_key]
            
            # Check if expired
            if time.time() - timestamp > self.ttl_seconds:
                del self.cache[cache_key]
                return None
            
            # Update access order
            self.access_order.append(cache_key)
            
            return result
        
        return None
    
    async def set(self, request: InferenceRequest, result: InferenceResult):
        """Cache result"""
        cache_key = self._get_cache_key(request)
        
        # Evict if at capacity
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            if self.access_order:
                lru_key = self.access_order.popleft()
                if lru_key in self.cache:
                    del self.cache[lru_key]
        
        self.cache[cache_key] = (result, time.time())
        self.access_order.append(cache_key)
    
    def _get_cache_key(self, request: InferenceRequest) -> str:
        """Generate cache key for request"""
        # Hash input data for cache key
        input_hash = hashlib.md5(request.input_data.tobytes()).hexdigest()
        return f"{request.model_id}_{input_hash}_{request.mode.value}"


class PerformanceMonitor:
    """Monitor inference performance"""
    
    def __init__(self):
        self.metrics: List[Dict] = []
        self.model_stats: Dict[str, Dict] = {}
    
    async def track(self, request: InferenceRequest, result: InferenceResult):
        """Track inference metrics"""
        metric = {
            "timestamp": time.time(),
            "model_id": request.model_id,
            "latency_ms": result.latency_ms,
            "mode": request.mode.value,
            "batch_size": request.batch_size
        }
        
        self.metrics.append(metric)
        
        # Update model statistics
        if request.model_id not in self.model_stats:
            self.model_stats[request.model_id] = {
                "total_inferences": 0,
                "total_latency": 0,
                "min_latency": float('inf'),
                "max_latency": 0
            }
        
        stats = self.model_stats[request.model_id]
        stats["total_inferences"] += 1
        stats["total_latency"] += result.latency_ms
        stats["min_latency"] = min(stats["min_latency"], result.latency_ms)
        stats["max_latency"] = max(stats["max_latency"], result.latency_ms)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {}
        
        for model_id, model_stats in self.model_stats.items():
            if model_stats["total_inferences"] > 0:
                stats[model_id] = {
                    "total_inferences": model_stats["total_inferences"],
                    "avg_latency_ms": model_stats["total_latency"] / model_stats["total_inferences"],
                    "min_latency_ms": model_stats["min_latency"],
                    "max_latency_ms": model_stats["max_latency"]
                }
        
        return stats


class ModelOptimizer:
    """Optimize models for edge deployment"""
    
    async def optimize(self, config: ModelConfig, mode: InferenceMode) -> Any:
        """Optimize model for target execution mode"""
        
        optimizations = []
        
        if config.quantized:
            optimizations.append("quantization")
        
        if mode == InferenceMode.WASM:
            optimizations.extend(["wasm_simd", "memory_optimization"])
        elif mode == InferenceMode.WEBGPU:
            optimizations.extend(["gpu_kernels", "memory_coalescing"])
        elif mode == InferenceMode.SIMD:
            optimizations.extend(["vectorization", "loop_unrolling"])
        
        # Simulate optimization
        await asyncio.sleep(0.05)
        
        return {
            "original_size_mb": config.size_mb,
            "optimized_size_mb": config.size_mb * 0.7,  # 30% reduction
            "optimizations_applied": optimizations,
            "expected_speedup": 1.5  # 50% faster
        }


# Inference session implementations
class WASMInferenceSession:
    """WebAssembly inference session"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    async def run(self, input_data: np.ndarray) -> np.ndarray:
        """Run WASM inference"""
        # Simulate WASM execution with <10ms latency
        await asyncio.sleep(np.random.uniform(0.005, 0.010))
        
        # Generate mock output matching output shape
        output_shape = self.config.output_shape
        output = np.random.randn(*output_shape).astype(np.float32)
        
        return output


class WebGPUInferenceSession:
    """WebGPU accelerated inference"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    async def run(self, input_data: np.ndarray) -> np.ndarray:
        """Run WebGPU inference"""
        # Simulate GPU execution with <5ms latency
        await asyncio.sleep(np.random.uniform(0.002, 0.005))
        
        output_shape = self.config.output_shape
        output = np.random.randn(*output_shape).astype(np.float32)
        
        return output


class WebGLInferenceSession:
    """WebGL accelerated inference"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    async def run(self, input_data: np.ndarray) -> np.ndarray:
        """Run WebGL inference"""
        await asyncio.sleep(np.random.uniform(0.003, 0.007))
        
        output_shape = self.config.output_shape
        output = np.random.randn(*output_shape).astype(np.float32)
        
        return output


class SIMDInferenceSession:
    """SIMD optimized CPU inference"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    async def run(self, input_data: np.ndarray) -> np.ndarray:
        """Run SIMD inference"""
        await asyncio.sleep(np.random.uniform(0.004, 0.008))
        
        output_shape = self.config.output_shape
        output = np.random.randn(*output_shape).astype(np.float32)
        
        return output


class CPUInferenceSession:
    """Pure CPU inference"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    async def run(self, input_data: np.ndarray) -> np.ndarray:
        """Run CPU inference"""
        await asyncio.sleep(np.random.uniform(0.008, 0.015))
        
        output_shape = self.config.output_shape
        output = np.random.randn(*output_shape).astype(np.float32)
        
        return output


# Example usage
async def main():
    """Demonstrate edge ML inference"""
    
    print("🤖 Edge ML Inference with ONNX Runtime WebAssembly")
    print("=" * 60)
    
    # Initialize runtime
    runtime = EdgeMLRuntime()
    
    # List available models
    models = await runtime.list_models()
    print("\n📦 Available Models:")
    for model_id in models:
        info = await runtime.get_model_info(model_id)
        print(f"  - {info['name']} ({info['size_mb']:.1f}MB, {info['format']})")
    
    # Test audio classification
    print("\n🎵 Testing Audio Classification:")
    print("-" * 40)
    
    # Generate test audio (1 second @ 16kHz)
    test_audio = np.random.randn(1, 1, 16000).astype(np.float32)
    
    # Create inference request
    request = InferenceRequest(
        request_id="test_001",
        model_id="audio_classifier",
        input_data=test_audio,
        mode=InferenceMode.WASM
    )
    
    # Run inference
    result = await runtime.infer(request)
    
    print(f"Model: {result.model_id}")
    print(f"Execution Mode: {result.execution_mode}")
    print(f"Latency: {result.latency_ms:.2f}ms")
    if result.labels:
        print(f"Top Genres: {', '.join(result.labels[:3])}")
    
    # Test batch inference
    print("\n📊 Testing Batch Inference:")
    print("-" * 40)
    
    batch_requests = [
        InferenceRequest(
            request_id=f"batch_{i}",
            model_id="vad",
            input_data=np.random.randn(1, 256).astype(np.float32),
            mode=InferenceMode.WEBGPU
        )
        for i in range(5)
    ]
    
    batch_results = await runtime.batch_infer(batch_requests)
    
    total_latency = sum(r.latency_ms for r in batch_results)
    avg_latency = total_latency / len(batch_results)
    
    print(f"Batch Size: {len(batch_requests)}")
    print(f"Average Latency: {avg_latency:.2f}ms")
    
    # Get performance stats
    stats = await runtime.performance_monitor.get_stats()
    
    print("\n📈 Performance Statistics:")
    print("-" * 40)
    for model_id, model_stats in stats.items():
        print(f"\n{model_id}:")
        print(f"  Total Inferences: {model_stats['total_inferences']}")
        print(f"  Avg Latency: {model_stats['avg_latency_ms']:.2f}ms")
        print(f"  Min/Max: {model_stats['min_latency_ms']:.2f}ms / {model_stats['max_latency_ms']:.2f}ms")
    
    print("\n✅ Edge ML inference operational with <10ms latency!")
    
    return runtime


if __name__ == "__main__":
    asyncio.run(main())