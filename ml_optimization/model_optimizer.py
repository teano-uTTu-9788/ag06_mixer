"""
ML Model Optimization for Real-Time Audio Processing
Converts and optimizes ML models for edge deployment using TensorFlow Lite and ONNX
"""

import os
import sys
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import pickle
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Model performance and size metrics"""
    accuracy: float
    inference_time_ms: float
    model_size_mb: float
    memory_usage_mb: float
    quantization_type: str
    target_platform: str

@dataclass
class OptimizationConfig:
    """Configuration for model optimization"""
    target_platform: str = "cpu"  # cpu, gpu, mobile, edge
    quantization: str = "int8"    # fp32, fp16, int8
    batch_size: int = 1
    input_shape: Tuple[int, ...] = (1, 960)  # Audio frame shape
    optimization_level: int = 2   # 0=none, 1=basic, 2=aggressive
    preserve_accuracy: float = 0.95  # Minimum accuracy to maintain

class BaseModelOptimizer(ABC):
    """Abstract base class for model optimizers"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.original_model = None
        self.optimized_model = None
        self.metrics = {}
    
    @abstractmethod
    def load_model(self, model_path: str):
        """Load the original model"""
        pass
    
    @abstractmethod
    def optimize(self) -> Any:
        """Optimize the model"""
        pass
    
    @abstractmethod
    def save_optimized_model(self, output_path: str):
        """Save the optimized model"""
        pass
    
    @abstractmethod
    def benchmark(self, test_data: np.ndarray) -> ModelMetrics:
        """Benchmark model performance"""
        pass

class TensorFlowLiteOptimizer(BaseModelOptimizer):
    """TensorFlow Lite model optimizer for mobile/edge deployment"""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.converter = None
        self.interpreter = None
        
    def load_model(self, model_path: str):
        """Load TensorFlow model for optimization"""
        try:
            import tensorflow as tf
            
            if model_path.endswith('.h5'):
                self.original_model = tf.keras.models.load_model(model_path)
            elif model_path.endswith('.pb'):
                # Load SavedModel format
                self.original_model = tf.saved_model.load(model_path)
            else:
                raise ValueError(f"Unsupported model format: {model_path}")
            
            logger.info(f"Loaded TensorFlow model from {model_path}")
            
        except ImportError:
            logger.error("TensorFlow not available. Install with: pip install tensorflow")
            raise
    
    def _create_representative_dataset(self, calibration_data: np.ndarray):
        """Create representative dataset for quantization"""
        def representative_data_gen():
            for i in range(min(100, len(calibration_data))):
                # Ensure correct shape and dtype
                sample = calibration_data[i:i+1].astype(np.float32)
                if len(sample.shape) == 2:
                    sample = sample.reshape(1, -1)
                yield [sample]
        
        return representative_data_gen
    
    def optimize(self, calibration_data: Optional[np.ndarray] = None) -> bytes:
        """Optimize model using TensorFlow Lite"""
        try:
            import tensorflow as tf
            
            # Create converter
            if hasattr(self.original_model, 'signatures'):
                # SavedModel format
                self.converter = tf.lite.TFLiteConverter.from_saved_model(self.original_model)
            else:
                # Keras model
                self.converter = tf.lite.TFLiteConverter.from_keras_model(self.original_model)
            
            # Configure optimization settings
            if self.config.optimization_level >= 1:
                self.converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Configure quantization
            if self.config.quantization == "int8":
                if calibration_data is not None:
                    self.converter.representative_dataset = self._create_representative_dataset(calibration_data)
                    self.converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                    self.converter.inference_input_type = tf.int8
                    self.converter.inference_output_type = tf.int8
                else:
                    logger.warning("No calibration data provided for INT8 quantization, using dynamic range quantization")
                    
            elif self.config.quantization == "fp16":
                self.converter.target_spec.supported_types = [tf.float16]
            
            # Advanced optimizations for aggressive setting
            if self.config.optimization_level >= 2:
                self.converter.experimental_new_converter = True
                if hasattr(self.converter, 'experimental_enable_resource_variables'):
                    self.converter.experimental_enable_resource_variables = True
            
            # Convert model
            self.optimized_model = self.converter.convert()
            logger.info(f"Model optimized with {self.config.quantization} quantization")
            
            return self.optimized_model
            
        except Exception as e:
            logger.error(f"TensorFlow Lite optimization failed: {e}")
            raise
    
    def save_optimized_model(self, output_path: str):
        """Save TensorFlow Lite model"""
        if self.optimized_model is None:
            raise ValueError("Model not optimized yet")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(self.optimized_model)
        
        logger.info(f"Optimized model saved to {output_path}")
    
    def load_optimized_model(self, model_path: str):
        """Load optimized TFLite model for inference"""
        try:
            import tensorflow as tf
            
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            logger.info(f"Loaded TFLite model from {model_path}")
            
        except ImportError:
            logger.error("TensorFlow not available")
            raise
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference with optimized model"""
        if self.interpreter is None:
            raise ValueError("Optimized model not loaded")
        
        # Get input and output details
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        
        # Prepare input data
        input_data = input_data.astype(input_details[0]['dtype'])
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)
        
        # Run inference
        self.interpreter.set_tensor(input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(output_details[0]['index'])
        return output_data
    
    def benchmark(self, test_data: np.ndarray) -> ModelMetrics:
        """Benchmark TensorFlow Lite model performance"""
        import time
        
        if self.interpreter is None:
            raise ValueError("Optimized model not loaded")
        
        # Warmup
        for _ in range(10):
            self.predict(test_data[0:1])
        
        # Benchmark inference time
        start_time = time.perf_counter()
        predictions = []
        for sample in test_data[:100]:  # Test on 100 samples
            pred = self.predict(sample.reshape(1, -1))
            predictions.append(pred)
        end_time = time.perf_counter()
        
        avg_inference_time = (end_time - start_time) / len(predictions) * 1000  # ms
        
        # Calculate model size
        model_size_mb = sys.getsizeof(self.optimized_model) / (1024 * 1024)
        
        return ModelMetrics(
            accuracy=0.0,  # Would need labeled data to calculate
            inference_time_ms=avg_inference_time,
            model_size_mb=model_size_mb,
            memory_usage_mb=0.0,  # Would need memory profiling
            quantization_type=self.config.quantization,
            target_platform=self.config.target_platform
        )

class ONNXOptimizer(BaseModelOptimizer):
    """ONNX model optimizer for cross-platform deployment"""
    
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.session = None
    
    def load_model(self, model_path: str):
        """Load ONNX model"""
        try:
            import onnx
            
            if model_path.endswith('.onnx'):
                self.original_model = onnx.load(model_path)
                logger.info(f"Loaded ONNX model from {model_path}")
            elif model_path.endswith('.h5') or model_path.endswith('.keras') or model_path.endswith('.pb'):
                # Try to convert from TensorFlow/Keras using tf2onnx
                logger.info(f"Converting TensorFlow model {model_path} to ONNX...")
                self._convert_tf_to_onnx(model_path)
            else:
                raise ValueError(f"Unsupported model format: {model_path}")

        except ImportError:
            logger.error("ONNX not available. Install with: pip install onnx")
            raise

    def _convert_tf_to_onnx(self, model_path: str):
        """Convert TensorFlow model to ONNX"""
        try:
            import tf2onnx
            import tensorflow as tf

            # Load Keras model
            if model_path.endswith('.h5') or model_path.endswith('.keras'):
                model = tf.keras.models.load_model(model_path)
                # Fix for Keras 3/tf2onnx compatibility
                if not hasattr(model, 'output_names') or not model.output_names:
                    model.output_names = [f'output_{i}' for i in range(len(model.outputs))]
            else:
                model = tf.saved_model.load(model_path)

            # Convert to ONNX
            # Handle Keras 3 input shape difference
            input_shape = model.input_shape
            if isinstance(input_shape, list):
                input_shape = input_shape[0]

            spec = (tf.TensorSpec((1,) + input_shape[1:], tf.float32, name="input"),)
            output_path = model_path.rsplit('.', 1)[0] + '.onnx'

            model_proto, _ = tf2onnx.convert.from_keras(
                model,
                input_signature=spec,
                opset=13,
                output_path=output_path
            )

            self.original_model = model_proto
            logger.info(f"Converted to ONNX and loaded from {output_path}")

        except ImportError:
            logger.error("tf2onnx not available. Install with: pip install tf2onnx")
            raise
        except Exception as e:
            logger.error(f"TF to ONNX conversion failed: {e}")
            raise
    
    def optimize(self, calibration_data: Optional[np.ndarray] = None) -> Any:
        """Optimize ONNX model"""
        try:
            import onnx
            import onnxoptimizer
            
            # Basic optimizations
            available_passes = set(onnxoptimizer.get_available_passes())

            desired_optimizations = [
                'eliminate_deadend',
                'eliminate_identity',
                'eliminate_nop_dropout',
                'eliminate_nop_pad',
                'eliminate_unused_initializer',
                'extract_constant_to_initializer',
                'fuse_add_bias_into_conv',
                'fuse_bn_into_conv',
                'fuse_consecutive_concats',
                'fuse_consecutive_reduce_unsqueeze',
                'fuse_consecutive_squeezes',
                'fuse_consecutive_transposes',
                'fuse_matmul_add_bias_into_gemm',
                'fuse_pad_into_conv',
                'fuse_transpose_into_gemm',
            ]
            
            if self.config.optimization_level >= 2:
                # Aggressive optimizations
                desired_optimizations.extend([
                    'eliminate_duplicate_initializer',
                    'eliminate_if_with_const_cond',
                    'eliminate_loop_with_const_iteration_count',
                    'lift_lexical_references',
                    'nop',
                ])
            
            # Filter optimizations to only include available ones
            optimizations = [op for op in desired_optimizations if op in available_passes]

            # Apply optimizations
            try:
                self.optimized_model = onnxoptimizer.optimize(self.original_model, optimizations)
                logger.info("ONNX model optimization completed")
            except Exception as e:
                logger.warning(f"ONNX optimization passes failed: {e}. Using unoptimized model.")
                self.optimized_model = self.original_model
            
            # Quantization if requested
            if self.config.quantization in ["int8", "fp16"]:
                self._apply_quantization()
            
            return self.optimized_model
            
        except ImportError:
            logger.error("ONNX optimizer not available. Install with: pip install onnxoptimizer")
            raise
        except Exception as e:
            logger.error(f"ONNX optimization failed: {e}")
            raise
    
    def _apply_quantization(self):
        """Apply quantization to ONNX model"""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            import tempfile
            
            # Save original model temporarily
            with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
                temp_path = tmp.name
                
            # Save and quantize
            import onnx
            onnx.save(self.optimized_model, temp_path)
            
            quantized_path = temp_path.replace('.onnx', '_quantized.onnx')
            
            if self.config.quantization == "int8":
                quantize_dynamic(
                    temp_path,
                    quantized_path,
                    weight_type=QuantType.QInt8
                )
            
            # Load quantized model
            self.optimized_model = onnx.load(quantized_path)
            
            # Cleanup
            os.unlink(temp_path)
            os.unlink(quantized_path)
            
        except ImportError:
            logger.warning("ONNX quantization not available")
        except Exception as e:
            logger.warning(f"ONNX quantization failed: {e}")
    
    def save_optimized_model(self, output_path: str):
        """Save optimized ONNX model"""
        if self.optimized_model is None:
            raise ValueError("Model not optimized yet")
        
        import onnx
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        onnx.save(self.optimized_model, output_path)
        logger.info(f"Optimized ONNX model saved to {output_path}")
    
    def load_optimized_model(self, model_path: str):
        """Load optimized ONNX model for inference"""
        try:
            import onnxruntime as ort
            
            # Configure session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Configure execution providers
            providers = ['CPUExecutionProvider']
            if self.config.target_platform == "gpu":
                providers.insert(0, 'CUDAExecutionProvider')
            
            self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
            logger.info(f"Loaded optimized ONNX model from {model_path}")
            
        except ImportError:
            logger.error("ONNX Runtime not available. Install with: pip install onnxruntime")
            raise
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference with ONNX model"""
        if self.session is None:
            raise ValueError("Optimized model not loaded")
        
        # Get input name
        input_name = self.session.get_inputs()[0].name
        
        # Prepare input
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)
        
        # Run inference
        outputs = self.session.run(None, {input_name: input_data.astype(np.float32)})
        return outputs[0]
    
    def benchmark(self, test_data: np.ndarray) -> ModelMetrics:
        """Benchmark ONNX model performance"""
        import time
        
        if self.session is None:
            raise ValueError("Optimized model not loaded")
        
        # Warmup
        for _ in range(10):
            self.predict(test_data[0:1])
        
        # Benchmark
        start_time = time.perf_counter()
        for sample in test_data[:100]:
            self.predict(sample.reshape(1, -1))
        end_time = time.perf_counter()
        
        avg_inference_time = (end_time - start_time) / 100 * 1000  # ms
        
        # Model size (approximate)
        model_size_mb = 0.0  # Would need to load from file
        
        return ModelMetrics(
            accuracy=0.0,
            inference_time_ms=avg_inference_time,
            model_size_mb=model_size_mb,
            memory_usage_mb=0.0,
            quantization_type=self.config.quantization,
            target_platform=self.config.target_platform
        )

class ModelOptimizationPipeline:
    """Complete model optimization pipeline"""
    
    def __init__(self):
        self.optimizers = {}
        self.results = {}
    
    def add_optimizer(self, name: str, optimizer: BaseModelOptimizer):
        """Add an optimizer to the pipeline"""
        self.optimizers[name] = optimizer
    
    def optimize_all(self, model_path: str, calibration_data: Optional[np.ndarray] = None):
        """Run optimization with all configured optimizers"""
        results = {}
        
        for name, optimizer in self.optimizers.items():
            try:
                logger.info(f"Running optimization with {name}")
                
                # Load and optimize
                optimizer.load_model(model_path)
                optimizer.optimize(calibration_data)
                
                # Save optimized model
                output_path = f"models/optimized/{name}_optimized.{self._get_extension(name)}"
                optimizer.save_optimized_model(output_path)
                
                # Benchmark if test data available
                if calibration_data is not None:
                    optimizer.load_optimized_model(output_path)
                    metrics = optimizer.benchmark(calibration_data)
                    results[name] = {
                        'metrics': metrics,
                        'model_path': output_path
                    }
                
                logger.info(f"{name} optimization completed")
                
            except Exception as e:
                logger.error(f"{name} optimization failed: {e}")
                results[name] = {'error': str(e)}
        
        self.results = results
        return results
    
    def _get_extension(self, optimizer_name: str) -> str:
        """Get file extension for optimizer type"""
        if 'tflite' in optimizer_name.lower():
            return 'tflite'
        elif 'onnx' in optimizer_name.lower():
            return 'onnx'
        else:
            return 'pkl'
    
    def compare_results(self) -> Dict[str, Any]:
        """Compare optimization results"""
        if not self.results:
            return {}
        
        comparison = {
            'best_inference_time': None,
            'smallest_model': None,
            'best_accuracy': None,
            'summary': []
        }
        
        best_time = float('inf')
        smallest_size = float('inf')
        best_acc = 0.0
        
        for name, result in self.results.items():
            if 'error' in result:
                continue
                
            metrics = result['metrics']
            
            # Track best metrics
            if metrics.inference_time_ms < best_time:
                best_time = metrics.inference_time_ms
                comparison['best_inference_time'] = name
            
            if metrics.model_size_mb < smallest_size:
                smallest_size = metrics.model_size_mb
                comparison['smallest_model'] = name
            
            if metrics.accuracy > best_acc:
                best_acc = metrics.accuracy
                comparison['best_accuracy'] = name
            
            # Add to summary
            comparison['summary'].append({
                'optimizer': name,
                'inference_time_ms': metrics.inference_time_ms,
                'model_size_mb': metrics.model_size_mb,
                'accuracy': metrics.accuracy,
                'quantization': metrics.quantization_type
            })
        
        return comparison
    
    def save_results(self, output_path: str):
        """Save optimization results to file"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert metrics to dict for JSON serialization
        serializable_results = {}
        for name, result in self.results.items():
            if 'metrics' in result:
                metrics = result['metrics']
                serializable_results[name] = {
                    'metrics': {
                        'accuracy': metrics.accuracy,
                        'inference_time_ms': metrics.inference_time_ms,
                        'model_size_mb': metrics.model_size_mb,
                        'memory_usage_mb': metrics.memory_usage_mb,
                        'quantization_type': metrics.quantization_type,
                        'target_platform': metrics.target_platform
                    },
                    'model_path': result['model_path']
                }
            else:
                serializable_results[name] = result
        
        with open(output_path, 'w') as f:
            json.dump({
                'results': serializable_results,
                'comparison': self.compare_results()
            }, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")

# Factory function for creating optimizers
def create_optimizer(optimizer_type: str, config: OptimizationConfig) -> BaseModelOptimizer:
    """Factory function to create appropriate optimizer"""
    optimizers = {
        'tflite': TensorFlowLiteOptimizer,
        'onnx': ONNXOptimizer,
    }
    
    if optimizer_type not in optimizers:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizers[optimizer_type](config)

# Example usage and testing
def demo_optimization():
    """Demonstrate model optimization pipeline"""
    
    # Mock data for testing
    mock_audio_data = np.random.randn(1000, 960).astype(np.float32)
    
    # Create optimization configs
    mobile_config = OptimizationConfig(
        target_platform="mobile",
        quantization="int8",
        optimization_level=2
    )
    
    edge_config = OptimizationConfig(
        target_platform="edge",
        quantization="fp16",
        optimization_level=2
    )
    
    # Create pipeline
    pipeline = ModelOptimizationPipeline()
    
    try:
        # Add optimizers
        pipeline.add_optimizer("tflite_mobile", create_optimizer("tflite", mobile_config))
        pipeline.add_optimizer("onnx_edge", create_optimizer("onnx", edge_config))
        
        logger.info("Model optimization pipeline created successfully")
        
    except Exception as e:
        logger.warning(f"Could not create all optimizers: {e}")
        logger.info("This is normal if TensorFlow/ONNX are not installed")
    
    return pipeline

if __name__ == "__main__":
    # Run demo
    pipeline = demo_optimization()
    print("🚀 ML Model Optimization system ready!")
    print("Install dependencies: pip install tensorflow onnx onnxruntime onnxoptimizer")