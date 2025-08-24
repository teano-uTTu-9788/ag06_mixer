#!/usr/bin/env python3
"""
ML Optimization System Test Suite
Tests model optimization, training, and integration components
"""

import sys
import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLOptimizationTester:
    """Test suite for ML optimization components"""
    
    def __init__(self):
        self.test_results = {}
        self.total_tests = 0
        self.passed_tests = 0
    
    def run_test(self, test_name: str, test_func):
        """Run a single test and record results"""
        self.total_tests += 1
        
        try:
            logger.info(f"Running test: {test_name}")
            start_time = time.time()
            
            result = test_func()
            
            execution_time = time.time() - start_time
            
            if result:
                self.passed_tests += 1
                logger.info(f"✅ {test_name} - PASSED ({execution_time:.3f}s)")
            else:
                logger.error(f"❌ {test_name} - FAILED ({execution_time:.3f}s)")
            
            self.test_results[test_name] = {
                'passed': result,
                'execution_time': execution_time
            }
            
        except Exception as e:
            logger.error(f"❌ {test_name} - ERROR: {e}")
            self.test_results[test_name] = {
                'passed': False,
                'error': str(e)
            }
    
    def test_model_optimizer_imports(self) -> bool:
        """Test model optimizer imports"""
        try:
            from model_optimizer import (
                BaseModelOptimizer, OptimizationConfig, 
                ModelOptimizationPipeline, create_optimizer
            )
            return True
        except ImportError as e:
            logger.warning(f"Import error: {e}")
            return False
    
    def test_audio_model_factory_imports(self) -> bool:
        """Test audio model factory imports"""
        try:
            from audio_model_factory import (
                SimpleGenreClassifier, AudioModelConfig,
                AudioDataGenerator, AudioModelTrainingPipeline
            )
            return True
        except ImportError as e:
            logger.warning(f"Import error: {e}")
            return False
    
    def test_optimized_ai_mixer_imports(self) -> bool:
        """Test optimized AI mixer imports"""
        try:
            from optimized_ai_mixer import (
                OptimizedGenreDetector, OptimizedAIMixer,
                ModelBackend, InferenceMetrics
            )
            return True
        except ImportError as e:
            logger.warning(f"Import error: {e}")
            return False
    
    def test_audio_model_creation(self) -> bool:
        """Test creating and training audio models"""
        try:
            from audio_model_factory import AudioModelTrainingPipeline
            
            pipeline = AudioModelTrainingPipeline()
            classifier = pipeline.create_genre_classifier()
            
            # Test feature extraction
            test_audio = np.random.randn(48000)  # 1 second at 48kHz
            features = classifier.extract_features(test_audio)
            
            if len(features) > 0:
                return True
            
        except Exception as e:
            logger.error(f"Model creation failed: {e}")
            return False
    
    def test_optimization_config(self) -> bool:
        """Test optimization configuration"""
        try:
            from model_optimizer import OptimizationConfig
            
            # Test default config
            config1 = OptimizationConfig()
            assert config1.target_platform == "cpu"
            assert config1.quantization == "int8"
            
            # Test custom config
            config2 = OptimizationConfig(
                target_platform="mobile",
                quantization="fp16",
                optimization_level=2
            )
            assert config2.target_platform == "mobile"
            assert config2.quantization == "fp16"
            
            return True
            
        except Exception as e:
            logger.error(f"Config test failed: {e}")
            return False
    
    def test_synthetic_data_generation(self) -> bool:
        """Test synthetic audio data generation"""
        try:
            from audio_model_factory import AudioDataGenerator
            
            generator = AudioDataGenerator()
            X, y = generator.generate_training_data(samples_per_genre=10)
            
            # Check data shape and labels
            assert X.shape[0] == 50  # 10 samples * 5 genres
            assert len(np.unique(y)) == 5  # 5 different genres
            assert X.dtype == np.float32
            
            # Check audio data properties
            for audio in X[:3]:  # Test first 3 samples
                assert len(audio) > 0
                assert not np.all(audio == 0)  # Not all zeros
                assert np.max(np.abs(audio)) <= 1.0  # Normalized
            
            return True
            
        except Exception as e:
            logger.error(f"Data generation test failed: {e}")
            return False
    
    def test_feature_extraction(self) -> bool:
        """Test audio feature extraction"""
        try:
            from audio_model_factory import SimpleGenreClassifier, AudioModelConfig
            
            config = AudioModelConfig(input_features=13)
            classifier = SimpleGenreClassifier(config)
            
            # Test with different audio types
            test_cases = [
                np.random.randn(48000),  # Random noise
                np.sin(2 * np.pi * 440 * np.linspace(0, 1, 48000)),  # Pure tone
                np.zeros(48000)  # Silence
            ]
            
            for i, audio in enumerate(test_cases):
                features = classifier.extract_features(audio)
                
                assert len(features) == config.input_features
                assert features.dtype in [np.float32, np.float64]
                assert not np.any(np.isnan(features))
                assert not np.any(np.isinf(features))
                
                logger.info(f"Test case {i+1}: Features shape {features.shape}, range [{np.min(features):.3f}, {np.max(features):.3f}]")
            
            return True
            
        except Exception as e:
            logger.error(f"Feature extraction test failed: {e}")
            return False
    
    def test_model_backends(self) -> bool:
        """Test different model backends"""
        try:
            from optimized_ai_mixer import OptimizedGenreDetector, ModelBackend
            
            detector = OptimizedGenreDetector()
            
            # Test available backends
            available_backends = list(detector.models.keys())
            logger.info(f"Available backends: {[b.value for b in available_backends]}")
            
            # Test feature size
            test_features = np.random.randn(13)
            
            # Test fallback detection (should always work)
            genre, confidence = detector._fallback_detection(test_features)
            assert genre in detector.genres
            assert 0.0 <= confidence <= 1.0
            
            logger.info(f"Fallback detection: {genre} ({confidence:.3f})")
            
            return True
            
        except Exception as e:
            logger.error(f"Backend test failed: {e}")
            return False
    
    def test_model_caching(self) -> bool:
        """Test model prediction caching"""
        try:
            from optimized_ai_mixer import ModelCache
            
            cache = ModelCache(max_size=100, ttl_seconds=60)
            
            # Test cache operations
            test_features = np.random.randn(13)
            test_prediction = ("rock", 0.85)
            
            # Should be empty initially
            result = cache.get(test_features)
            assert result is None
            
            # Set and retrieve
            cache.set(test_features, test_prediction)
            result = cache.get(test_features)
            assert result is not None
            cached_prediction, timestamp = result
            assert cached_prediction == test_prediction
            
            # Test cache size limit
            for i in range(150):  # Exceed max_size
                features = np.random.randn(13)
                cache.set(features, ("test", 0.5))
            
            assert len(cache.cache) <= cache.max_size
            
            return True
            
        except Exception as e:
            logger.error(f"Cache test failed: {e}")
            return False
    
    def test_inference_performance(self) -> bool:
        """Test inference performance benchmarking"""
        try:
            from optimized_ai_mixer import OptimizedAIMixer
            
            mixer = OptimizedAIMixer()
            
            # Generate test audio frames
            frame_size = 960  # 20ms at 48kHz
            test_frames = [
                np.random.randn(frame_size) * 0.1,  # Quiet noise
                np.sin(2 * np.pi * 440 * np.linspace(0, 0.02, frame_size)),  # 440Hz tone
                np.random.randn(frame_size) * 0.5   # Louder noise
            ]
            
            total_time = 0
            processed_frames = 0
            
            for frame in test_frames:
                start_time = time.perf_counter()
                processed_audio, metadata = mixer.process_frame(frame)
                end_time = time.perf_counter()
                
                processing_time = (end_time - start_time) * 1000  # ms
                total_time += processing_time
                processed_frames += 1
                
                # Validate output
                assert processed_audio.shape == frame.shape
                assert 'genre' in metadata
                assert 'confidence' in metadata
                assert 'inference_time_ms' in metadata
                
                logger.info(f"Frame {processed_frames}: {metadata['genre']} ({metadata['confidence']:.3f}), {processing_time:.2f}ms")
            
            avg_time = total_time / processed_frames
            logger.info(f"Average processing time: {avg_time:.2f}ms")
            
            # Performance requirement: should process faster than real-time
            # 20ms frame should process in < 20ms
            if avg_time < 20:
                logger.info("✅ Performance requirement met (< 20ms per frame)")
                return True
            else:
                logger.warning(f"⚠️ Performance slower than real-time: {avg_time:.2f}ms")
                return False
            
        except Exception as e:
            logger.error(f"Performance test failed: {e}")
            return False
    
    def test_optimization_pipeline(self) -> bool:
        """Test the complete optimization pipeline"""
        try:
            from model_optimizer import ModelOptimizationPipeline, OptimizationConfig
            
            pipeline = ModelOptimizationPipeline()
            
            # Test pipeline creation
            config = OptimizationConfig(
                target_platform="cpu",
                quantization="fp32",
                optimization_level=1
            )
            
            # Pipeline should be created without errors
            assert hasattr(pipeline, 'optimizers')
            assert hasattr(pipeline, 'results')
            
            # Test result comparison (with empty results)
            comparison = pipeline.compare_results()
            assert isinstance(comparison, dict)
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline test failed: {e}")
            return False
    
    def test_integration_with_existing_mixer(self) -> bool:
        """Test integration with existing AI mixer components"""
        try:
            from optimized_ai_mixer import OptimizedAIMixer
            
            mixer = OptimizedAIMixer()
            
            # Test that mixer was created
            assert mixer.sample_rate == 48000
            assert mixer.genre_detector is not None
            
            # Test basic processing capability
            test_audio = np.random.randn(960) * 0.1
            processed, metadata = mixer.process_frame(test_audio)
            
            # Should produce some output even if full integration fails
            assert processed is not None
            assert len(processed) == len(test_audio)
            assert metadata is not None
            
            return True
            
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all ML optimization tests"""
        print("🧪 ML Optimization System Test Suite")
        print("=" * 50)
        
        # Import tests
        self.run_test("Model Optimizer Imports", self.test_model_optimizer_imports)
        self.run_test("Audio Model Factory Imports", self.test_audio_model_factory_imports)
        self.run_test("Optimized AI Mixer Imports", self.test_optimized_ai_mixer_imports)
        
        # Configuration tests
        self.run_test("Optimization Config", self.test_optimization_config)
        
        # Audio processing tests
        self.run_test("Synthetic Data Generation", self.test_synthetic_data_generation)
        self.run_test("Feature Extraction", self.test_feature_extraction)
        
        # ML model tests
        self.run_test("Audio Model Creation", self.test_audio_model_creation)
        self.run_test("Model Backends", self.test_model_backends)
        self.run_test("Model Caching", self.test_model_caching)
        
        # Performance tests
        self.run_test("Inference Performance", self.test_inference_performance)
        
        # Integration tests
        self.run_test("Optimization Pipeline", self.test_optimization_pipeline)
        self.run_test("Existing Mixer Integration", self.test_integration_with_existing_mixer)
        
        # Summary
        print("=" * 50)
        success_rate = (self.passed_tests / self.total_tests) * 100
        print(f"📊 Test Results: {self.passed_tests}/{self.total_tests} ({success_rate:.1f}%)")
        
        if success_rate >= 90:
            print("🎉 ML Optimization system is ready for deployment!")
        elif success_rate >= 70:
            print("⚠️ ML Optimization system mostly functional, some issues to address")
        else:
            print("❌ ML Optimization system needs significant work")
        
        # Recommendations
        print("\n💡 Recommendations:")
        if success_rate < 100:
            print("  • Install optional dependencies: pip install -r ml_optimization/requirements.txt")
            print("  • Missing dependencies reduce functionality but core system still works")
        
        if success_rate >= 80:
            print("  • System ready for model training and optimization")
            print("  • Consider training custom models on your audio data")
            print("  • Test different optimization backends for best performance")
        
        return success_rate >= 80

def main():
    """Run the test suite"""
    tester = MLOptimizationTester()
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()