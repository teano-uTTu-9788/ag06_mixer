"""
Optimized AI Mixer Integration
Integrates optimized ML models with the real-time audio processing pipeline
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import threading
import time
from dataclasses import dataclass
from enum import Enum

# Setup logging
logger = logging.getLogger(__name__)

class ModelBackend(Enum):
    """Supported ML model backends"""
    SKLEARN = "sklearn"
    TFLITE = "tflite"
    ONNX = "onnx"
    NUMPY = "numpy"

@dataclass
class InferenceMetrics:
    """Model inference performance metrics"""
    inference_time_ms: float
    confidence_score: float
    model_backend: ModelBackend
    cache_hit: bool = False

class ModelCache:
    """Cache for model predictions to reduce inference overhead"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
    
    def _generate_key(self, features: np.ndarray) -> str:
        """Generate cache key from features"""
        # Simple hash of feature values
        return str(hash(features.tobytes()))
    
    def get(self, features: np.ndarray) -> Optional[Tuple[Any, float]]:
        """Get cached prediction if available and valid"""
        key = self._generate_key(features)
        current_time = time.time()
        
        with self._lock:
            if key in self.cache:
                timestamp = self.timestamps[key]
                if current_time - timestamp < self.ttl_seconds:
                    return self.cache[key], timestamp
                else:
                    # Expired entry
                    del self.cache[key]
                    del self.timestamps[key]
        
        return None
    
    def set(self, features: np.ndarray, prediction: Any):
        """Cache a prediction"""
        key = self._generate_key(features)
        current_time = time.time()
        
        with self._lock:
            # Remove oldest entries if cache is full
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
            
            self.cache[key] = prediction
            self.timestamps[key] = current_time
    
    def clear(self):
        """Clear all cached entries"""
        with self._lock:
            self.cache.clear()
            self.timestamps.clear()

class OptimizedGenreDetector:
    """Optimized genre detection with multiple backend support"""
    
    def __init__(self, model_dir: str = "models/optimized"):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.active_backend = ModelBackend.SKLEARN
        self.cache = ModelCache()
        self.genres = ['speech', 'rock', 'jazz', 'electronic', 'classical']
        self._load_available_models()
    
    def _load_available_models(self):
        """Load all available optimized models"""
        model_files = {
            ModelBackend.SKLEARN: "genre_classifier.pkl",
            ModelBackend.TFLITE: "tflite_mobile_optimized.tflite",
            ModelBackend.ONNX: "onnx_edge_optimized.onnx",
            ModelBackend.NUMPY: "genre_classifier_numpy.pkl"
        }
        
        for backend, filename in model_files.items():
            model_path = self.model_dir / filename
            if model_path.exists():
                try:
                    self._load_model(backend, str(model_path))
                    logger.info(f"Loaded {backend.value} model")
                except Exception as e:
                    logger.warning(f"Failed to load {backend.value} model: {e}")
    
    def _load_model(self, backend: ModelBackend, model_path: str):
        """Load a specific model backend"""
        if backend == ModelBackend.SKLEARN or backend == ModelBackend.NUMPY:
            import pickle
            with open(model_path, 'rb') as f:
                self.models[backend] = pickle.load(f)
        
        elif backend == ModelBackend.TFLITE:
            try:
                import tensorflow as tf
                interpreter = tf.lite.Interpreter(model_path=model_path)
                interpreter.allocate_tensors()
                self.models[backend] = interpreter
            except ImportError:
                logger.warning("TensorFlow Lite not available")
        
        elif backend == ModelBackend.ONNX:
            try:
                import onnxruntime as ort
                session = ort.InferenceSession(model_path)
                self.models[backend] = session
            except ImportError:
                logger.warning("ONNX Runtime not available")
    
    def set_backend(self, backend: ModelBackend):
        """Set the active inference backend"""
        if backend in self.models:
            self.active_backend = backend
            logger.info(f"Switched to {backend.value} backend")
        else:
            logger.warning(f"Backend {backend.value} not available")
    
    def detect(self, features: np.ndarray) -> Tuple[str, float, InferenceMetrics]:
        """Detect genre with performance metrics"""
        start_time = time.perf_counter()
        
        # Check cache first
        cached_result = self.cache.get(features)
        if cached_result is not None:
            prediction, _ = cached_result
            inference_time = (time.perf_counter() - start_time) * 1000
            
            metrics = InferenceMetrics(
                inference_time_ms=inference_time,
                confidence_score=prediction[1] if isinstance(prediction, tuple) else 0.9,
                model_backend=self.active_backend,
                cache_hit=True
            )
            
            genre = prediction[0] if isinstance(prediction, tuple) else prediction
            confidence = prediction[1] if isinstance(prediction, tuple) else 0.9
            return genre, confidence, metrics
        
        # Run inference
        genre, confidence = self._run_inference(features)
        
        # Cache result
        self.cache.set(features, (genre, confidence))
        
        # Calculate metrics
        inference_time = (time.perf_counter() - start_time) * 1000
        metrics = InferenceMetrics(
            inference_time_ms=inference_time,
            confidence_score=confidence,
            model_backend=self.active_backend,
            cache_hit=False
        )
        
        return genre, confidence, metrics
    
    def _run_inference(self, features: np.ndarray) -> Tuple[str, float]:
        """Run inference with the active backend"""
        if self.active_backend not in self.models:
            logger.warning(f"Backend {self.active_backend.value} not available, using fallback")
            return self._fallback_detection(features)
        
        try:
            if self.active_backend == ModelBackend.SKLEARN:
                return self._sklearn_inference(features)
            elif self.active_backend == ModelBackend.TFLITE:
                return self._tflite_inference(features)
            elif self.active_backend == ModelBackend.ONNX:
                return self._onnx_inference(features)
            elif self.active_backend == ModelBackend.NUMPY:
                return self._numpy_inference(features)
        except Exception as e:
            logger.error(f"Inference failed with {self.active_backend.value}: {e}")
            return self._fallback_detection(features)
    
    def _sklearn_inference(self, features: np.ndarray) -> Tuple[str, float]:
        """Run inference with sklearn model"""
        model_data = self.models[ModelBackend.SKLEARN]
        
        if isinstance(model_data, dict):
            # Loaded model data structure
            model = model_data.get('model')
            scaler = model_data.get('feature_scaler')
            
            if scaler:
                features = scaler.transform(features.reshape(1, -1))
            else:
                features = features.reshape(1, -1)
            
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features)[0]
                prediction = np.argmax(probabilities)
                confidence = probabilities[prediction]
            else:
                prediction = model.predict(features)[0]
                confidence = 0.8  # Default confidence
            
        else:
            # Direct model object
            features = features.reshape(1, -1)
            if hasattr(model_data, 'predict_proba'):
                probabilities = model_data.predict_proba(features)[0]
                prediction = np.argmax(probabilities)
                confidence = probabilities[prediction]
            else:
                prediction = model_data.predict(features)[0]
                confidence = 0.8
        
        genre = self.genres[int(prediction)] if prediction < len(self.genres) else 'unknown'
        return genre, float(confidence)
    
    def _tflite_inference(self, features: np.ndarray) -> Tuple[str, float]:
        """Run inference with TensorFlow Lite"""
        interpreter = self.models[ModelBackend.TFLITE]
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Prepare input
        input_data = features.reshape(1, -1).astype(input_details[0]['dtype'])
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Process output
        if len(output_data) == len(self.genres):
            prediction = np.argmax(output_data)
            confidence = output_data[prediction]
        else:
            prediction = int(output_data[0]) if len(output_data) == 1 else 0
            confidence = 0.8
        
        genre = self.genres[prediction] if prediction < len(self.genres) else 'unknown'
        return genre, float(confidence)
    
    def _onnx_inference(self, features: np.ndarray) -> Tuple[str, float]:
        """Run inference with ONNX"""
        session = self.models[ModelBackend.ONNX]
        
        # Get input name
        input_name = session.get_inputs()[0].name
        
        # Prepare input
        input_data = features.reshape(1, -1).astype(np.float32)
        
        # Run inference
        outputs = session.run(None, {input_name: input_data})
        output_data = outputs[0][0]
        
        # Process output
        if len(output_data) == len(self.genres):
            prediction = np.argmax(output_data)
            confidence = output_data[prediction]
        else:
            prediction = int(output_data[0]) if len(output_data) == 1 else 0
            confidence = 0.8
        
        genre = self.genres[prediction] if prediction < len(self.genres) else 'unknown'
        return genre, float(confidence)
    
    def _numpy_inference(self, features: np.ndarray) -> Tuple[str, float]:
        """Run inference with numpy model"""
        model_data = self.models[ModelBackend.NUMPY]
        
        # This would use the SimpleGenreClassifier from audio_model_factory
        if hasattr(model_data, 'predict_proba'):
            probabilities = model_data.predict_proba(features.reshape(1, -1))[0]
            prediction = np.argmax(probabilities)
            confidence = probabilities[prediction]
        else:
            prediction = model_data.predict(features.reshape(1, -1))[0]
            confidence = 0.8
        
        genre = self.genres[prediction] if prediction < len(self.genres) else 'unknown'
        return genre, float(confidence)
    
    def _fallback_detection(self, features: np.ndarray) -> Tuple[str, float]:
        """Fallback genre detection using simple heuristics"""
        # Simple heuristic based on spectral characteristics
        energy = np.mean(features)
        high_freq_energy = np.mean(features[-5:]) if len(features) > 5 else 0
        
        if energy < 0.1:
            return 'speech', 0.6
        elif high_freq_energy > 0.3:
            return 'electronic', 0.5
        elif energy > 0.8:
            return 'rock', 0.5
        else:
            return 'jazz', 0.4
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'active_backend': self.active_backend.value,
            'available_backends': [b.value for b in self.models.keys()],
            'cache_size': len(self.cache.cache),
            'cache_max_size': self.cache.max_size
        }

class OptimizedAIMixer:
    """AI Mixer with optimized ML models for real-time performance"""
    
    def __init__(self, sample_rate: int = 48000, model_dir: str = "models/optimized"):
        self.sample_rate = sample_rate
        self.genre_detector = OptimizedGenreDetector(model_dir)
        self.current_genre = "speech"
        self.confidence = 0.0
        self.processing_stats = {
            'total_frames': 0,
            'total_inference_time': 0.0,
            'avg_inference_time': 0.0,
            'cache_hit_rate': 0.0
        }
        self._setup_audio_processor()
    
    def _setup_audio_processor(self):
        """Setup the audio processing components"""
        try:
            # Import our existing audio processing components
            import sys
            from pathlib import Path
            parent_dir = Path(__file__).parent.parent
            sys.path.insert(0, str(parent_dir))
            
            from complete_ai_mixer import CompleteMixingSystem
            self.mixer = CompleteMixingSystem(self.sample_rate)
            logger.info("Integrated with existing AI mixer")
            
        except ImportError as e:
            logger.warning(f"Could not import existing mixer: {e}")
            self.mixer = None
    
    def extract_features(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract features for ML inference"""
        # Use the same feature extraction as our trained models
        try:
            from .audio_model_factory import SimpleGenreClassifier, AudioModelConfig
            
            config = AudioModelConfig()
            classifier = SimpleGenreClassifier(config)
            features = classifier.extract_features(audio_data)
            return features
            
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            # Fallback to simple spectral features
            return self._simple_features(audio_data)
    
    def _simple_features(self, audio_data: np.ndarray) -> np.ndarray:
        """Simple fallback feature extraction"""
        # Ensure mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Basic spectral features
        fft = np.fft.fft(audio_data)
        magnitude = np.abs(fft[:len(fft)//2])
        
        # 13 features to match MFCC
        n_features = 13
        bands = np.array_split(magnitude, n_features)
        features = np.array([np.mean(band) for band in bands])
        
        return features
    
    def process_frame(self, audio_frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process a single audio frame with optimized AI analysis"""
        # Extract features for ML analysis
        features = self.extract_features(audio_frame)
        
        # Run optimized genre detection
        genre, confidence, metrics = self.genre_detector.detect(features)
        
        # Update processing stats
        self._update_stats(metrics)
        
        # Update genre if confidence is high enough
        if confidence > 0.7:
            self.current_genre = genre
            self.confidence = confidence
        
        # Process audio with existing mixer if available
        if self.mixer:
            try:
                processed_audio = self.mixer.process(audio_frame, self.sample_rate)
            except Exception as e:
                logger.warning(f"Mixer processing failed: {e}")
                processed_audio = audio_frame
        else:
            # Basic processing without full mixer
            processed_audio = self._basic_processing(audio_frame)
        
        # Return processed audio and metadata
        metadata = {
            'genre': self.current_genre,
            'confidence': self.confidence,
            'inference_time_ms': metrics.inference_time_ms,
            'model_backend': metrics.model_backend.value,
            'cache_hit': metrics.cache_hit,
            'processing_stats': self.processing_stats.copy()
        }
        
        return processed_audio, metadata
    
    def _basic_processing(self, audio_frame: np.ndarray) -> np.ndarray:
        """Basic audio processing when full mixer is not available"""
        # Simple genre-based processing
        if self.current_genre == 'speech':
            # Speech: light compression and noise gate
            processed = self._apply_compression(audio_frame, ratio=2.0)
        elif self.current_genre == 'rock':
            # Rock: more aggressive processing
            processed = self._apply_compression(audio_frame, ratio=4.0)
        elif self.current_genre == 'classical':
            # Classical: preserve dynamics
            processed = audio_frame * 0.95  # Slight gain reduction
        else:
            # Default processing
            processed = self._apply_compression(audio_frame, ratio=3.0)
        
        return processed
    
    def _apply_compression(self, audio: np.ndarray, ratio: float = 3.0, threshold: float = 0.5) -> np.ndarray:
        """Simple audio compression"""
        # Basic compressor
        output = audio.copy()
        
        # Find samples above threshold
        above_threshold = np.abs(output) > threshold
        
        # Apply compression to samples above threshold
        if np.any(above_threshold):
            excess = np.abs(output[above_threshold]) - threshold
            compressed_excess = excess / ratio
            
            # Apply with original sign
            output[above_threshold] = np.sign(output[above_threshold]) * (threshold + compressed_excess)
        
        return output
    
    def _update_stats(self, metrics: InferenceMetrics):
        """Update processing statistics"""
        self.processing_stats['total_frames'] += 1
        self.processing_stats['total_inference_time'] += metrics.inference_time_ms
        
        # Calculate running averages
        total_frames = self.processing_stats['total_frames']
        self.processing_stats['avg_inference_time'] = (
            self.processing_stats['total_inference_time'] / total_frames
        )
        
        # Update cache hit rate
        cache_hits = getattr(self, '_cache_hits', 0)
        if metrics.cache_hit:
            cache_hits += 1
            self._cache_hits = cache_hits
        
        self.processing_stats['cache_hit_rate'] = (
            cache_hits / total_frames if total_frames > 0 else 0.0
        )
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report"""
        return {
            'current_settings': {
                'genre': self.current_genre,
                'confidence': self.confidence,
                'active_backend': self.genre_detector.active_backend.value
            },
            'performance_metrics': self.processing_stats,
            'model_stats': self.genre_detector.get_performance_stats(),
            'recommendations': self._get_optimization_recommendations()
        }
    
    def _get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on current performance"""
        recommendations = []
        
        avg_time = self.processing_stats.get('avg_inference_time', 0)
        cache_hit_rate = self.processing_stats.get('cache_hit_rate', 0)
        
        if avg_time > 10:  # ms
            recommendations.append("Consider switching to TensorFlow Lite or ONNX for faster inference")
        
        if cache_hit_rate < 0.3:
            recommendations.append("Low cache hit rate - consider increasing cache size or TTL")
        
        if ModelBackend.TFLITE in self.genre_detector.models and self.genre_detector.active_backend != ModelBackend.TFLITE:
            recommendations.append("TensorFlow Lite model available - try switching for mobile optimization")
        
        if len(recommendations) == 0:
            recommendations.append("Performance is optimal for current configuration")
        
        return recommendations

def demo_optimized_ai_mixer():
    """Demonstrate the optimized AI mixer"""
    print("🚀 Optimized AI Mixer Demo")
    print("=" * 40)
    
    # Create optimized mixer
    mixer = OptimizedAIMixer()
    
    # Generate test audio
    duration = 1.0
    sample_rate = 48000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Test different audio types
    test_signals = {
        'speech': np.sin(2 * np.pi * 800 * t) * 0.5,  # Speech-like frequency
        'rock': np.sin(2 * np.pi * 220 * t) * 0.8,    # Rock-like bass
        'jazz': np.sin(2 * np.pi * 440 * t) * 0.6     # Jazz-like mid
    }
    
    print("\n🧪 Testing genre detection and processing...")
    
    for signal_type, audio in test_signals.items():
        print(f"\nTesting {signal_type} audio:")
        
        # Process audio frame
        processed, metadata = mixer.process_frame(audio)
        
        print(f"  Detected genre: {metadata['genre']} (confidence: {metadata['confidence']:.2f})")
        print(f"  Inference time: {metadata['inference_time_ms']:.2f}ms")
        print(f"  Model backend: {metadata['model_backend']}")
        print(f"  Cache hit: {metadata['cache_hit']}")
    
    # Get optimization report
    report = mixer.get_optimization_report()
    print("\n📊 Performance Report:")
    print(f"  Average inference time: {report['performance_metrics']['avg_inference_time']:.2f}ms")
    print(f"  Cache hit rate: {report['performance_metrics']['cache_hit_rate']:.2%}")
    print(f"  Total frames processed: {report['performance_metrics']['total_frames']}")
    
    print("\n💡 Recommendations:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
    
    print("\n✅ Optimized AI Mixer demo completed!")
    return mixer

if __name__ == "__main__":
    demo_optimized_ai_mixer()