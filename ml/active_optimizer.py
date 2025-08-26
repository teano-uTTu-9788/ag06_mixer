#!/usr/bin/env python3
"""
Active ML Workflow Optimizer
Real-time machine learning optimization with gradient descent
"""

import json
import time
import asyncio
import random
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque
import math

@dataclass
class PerformanceMetric:
    timestamp: str
    metric_name: str
    value: float
    workflow_id: str
    configuration: Dict[str, Any]
    context: Dict[str, Any]

class ActiveOptimizer:
    """Real-time ML-based workflow optimizer"""
    
    def __init__(self, learning_rate: float = 0.01, history_size: int = 1000):
        self.learning_rate = learning_rate
        self.history_size = history_size
        
        # Performance history
        self.metrics_history = deque(maxlen=history_size)
        self.configurations = {}  # config_id -> config_params
        self.performance_scores = {}  # config_id -> performance_score
        
        # ML parameters
        self.feature_weights = np.random.normal(0, 0.1, 10)  # 10 features
        self.baseline_performance = 1.0
        self.optimization_iterations = 0
        
        # A/B testing
        self.ab_experiments = {}
        self.experiment_results = {}
        
        print("🧠 Active ML Optimizer initialized")
        print(f"   Learning rate: {learning_rate}")
        print(f"   History size: {history_size}")
    
    def extract_features(self, config: Dict[str, Any], context: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from configuration and context"""
        features = np.zeros(10)
        
        # Feature 1-3: Configuration parameters (normalized)
        if 'batch_size' in config:
            features[0] = min(config['batch_size'] / 100.0, 1.0)
        if 'timeout_ms' in config:
            features[1] = min(config['timeout_ms'] / 5000.0, 1.0)
        if 'retry_count' in config:
            features[2] = min(config['retry_count'] / 5.0, 1.0)
        
        # Feature 4-6: Context/environment
        if 'cpu_percent' in context:
            features[3] = context['cpu_percent'] / 100.0
        if 'memory_percent' in context:
            features[4] = context['memory_percent'] / 100.0
        if 'active_workflows' in context:
            features[5] = min(context['active_workflows'] / 10.0, 1.0)
        
        # Feature 7-10: Time-based features
        current_hour = datetime.now().hour
        features[6] = current_hour / 24.0
        features[7] = 1.0 if 9 <= current_hour <= 17 else 0.0  # Business hours
        features[8] = len(self.metrics_history) / self.history_size
        features[9] = self.optimization_iterations / 1000.0
        
        return features
    
    def predict_performance(self, config: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Predict performance score using learned model"""
        features = self.extract_features(config, context)
        raw_score = np.dot(self.feature_weights, features)
        
        # Apply sigmoid to normalize to 0-2 range (1.0 = baseline)
        normalized_score = 2.0 / (1.0 + np.exp(-raw_score))
        return normalized_score
    
    async def record_performance(self, workflow_id: str, 
                               duration_ms: float, 
                               success: bool,
                               config: Dict[str, Any],
                               context: Dict[str, Any] = None) -> str:
        """Record actual performance for learning"""
        
        if context is None:
            context = {}
        
        # Calculate performance score (lower latency + success = higher score)
        if success:
            # Baseline: 500ms = 1.0 score, better performance = higher score
            perf_score = min(2.0, 500.0 / max(duration_ms, 50.0))
        else:
            perf_score = 0.1  # Failure penalty
        
        metric = PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            metric_name="workflow_performance",
            value=perf_score,
            workflow_id=workflow_id,
            configuration=config.copy(),
            context=context.copy()
        )
        
        self.metrics_history.append(metric)
        
        # Update ML model with this data point
        await self._update_model(metric)
        
        print(f"📊 Performance recorded: {perf_score:.3f} (Duration: {duration_ms}ms, Success: {success})")
        return metric.timestamp
    
    async def _update_model(self, metric: PerformanceMetric):
        """Update ML model using gradient descent"""
        features = self.extract_features(metric.configuration, metric.context)
        predicted = np.dot(self.feature_weights, features)
        
        # Sigmoid transformation
        predicted_score = 2.0 / (1.0 + np.exp(-predicted))
        actual_score = metric.value
        
        # Calculate error and gradient
        error = actual_score - predicted_score
        
        # Gradient of sigmoid: sigmoid(x) * (1 - sigmoid(x))
        sigmoid_grad = predicted_score * (2.0 - predicted_score) / 2.0
        gradient = error * sigmoid_grad * features
        
        # Update weights
        self.feature_weights += self.learning_rate * gradient
        
        self.optimization_iterations += 1
        
        if self.optimization_iterations % 50 == 0:
            print(f"🔬 Model updated (iteration {self.optimization_iterations})")
            print(f"   Error: {error:.4f}, Predicted: {predicted_score:.3f}, Actual: {actual_score:.3f}")
    
    def suggest_configuration(self, workflow_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest optimal configuration based on learned model"""
        
        # Define configuration space
        config_options = {
            'batch_size': [1, 5, 10, 20, 50],
            'timeout_ms': [1000, 2000, 3000, 5000],
            'retry_count': [0, 1, 2, 3],
            'parallel_workers': [1, 2, 4],
            'circuit_breaker_threshold': [3, 5, 10]
        }
        
        best_config = {}
        best_score = -1
        
        # Sample and evaluate configurations
        for _ in range(20):  # Sample 20 random configurations
            config = {}
            for param, options in config_options.items():
                config[param] = random.choice(options)
            
            predicted_score = self.predict_performance(config, context)
            
            if predicted_score > best_score:
                best_score = predicted_score
                best_config = config.copy()
        
        best_config['predicted_score'] = best_score
        print(f"💡 Suggested config (score: {best_score:.3f}): {best_config}")
        
        return best_config
    
    def start_ab_experiment(self, experiment_name: str, 
                           config_a: Dict[str, Any], 
                           config_b: Dict[str, Any],
                           traffic_split: float = 0.5) -> str:
        """Start A/B test experiment"""
        experiment_id = f"exp_{int(time.time())}_{experiment_name}"
        
        self.ab_experiments[experiment_id] = {
            "name": experiment_name,
            "config_a": config_a,
            "config_b": config_b,
            "traffic_split": traffic_split,
            "start_time": datetime.now().isoformat(),
            "samples_a": [],
            "samples_b": []
        }
        
        print(f"🧪 A/B experiment started: {experiment_name} ({experiment_id})")
        return experiment_id
    
    def get_ab_config(self, experiment_id: str) -> Tuple[str, Dict[str, Any]]:
        """Get configuration for A/B experiment"""
        if experiment_id not in self.ab_experiments:
            raise ValueError(f"Unknown experiment: {experiment_id}")
        
        exp = self.ab_experiments[experiment_id]
        
        # Random assignment based on traffic split
        if random.random() < exp["traffic_split"]:
            return "A", exp["config_a"]
        else:
            return "B", exp["config_b"]
    
    def record_ab_result(self, experiment_id: str, variant: str, performance_score: float):
        """Record A/B experiment result"""
        if experiment_id not in self.ab_experiments:
            return
        
        exp = self.ab_experiments[experiment_id]
        
        if variant == "A":
            exp["samples_a"].append(performance_score)
        elif variant == "B":
            exp["samples_b"].append(performance_score)
        
        # Auto-evaluate after collecting enough samples
        if len(exp["samples_a"]) >= 30 and len(exp["samples_b"]) >= 30:
            self._evaluate_ab_experiment(experiment_id)
    
    def _evaluate_ab_experiment(self, experiment_id: str):
        """Evaluate A/B experiment results"""
        exp = self.ab_experiments[experiment_id]
        samples_a = exp["samples_a"]
        samples_b = exp["samples_b"]
        
        if not samples_a or not samples_b:
            return
        
        mean_a = np.mean(samples_a)
        mean_b = np.mean(samples_b)
        std_a = np.std(samples_a)
        std_b = np.std(samples_b)
        
        # Simple t-test approximation
        n_a, n_b = len(samples_a), len(samples_b)
        pooled_std = math.sqrt(((n_a-1)*std_a**2 + (n_b-1)*std_b**2) / (n_a+n_b-2))
        t_stat = (mean_a - mean_b) / (pooled_std * math.sqrt(1/n_a + 1/n_b))
        
        # Critical value for 95% confidence (approximation)
        is_significant = abs(t_stat) > 2.0
        
        result = {
            "experiment_id": experiment_id,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "improvement": (mean_b - mean_a) / mean_a * 100 if mean_a > 0 else 0,
            "significant": is_significant,
            "winner": "B" if mean_b > mean_a and is_significant else "A" if is_significant else "No clear winner",
            "confidence": "95%" if is_significant else "<95%",
            "samples": {"A": n_a, "B": n_b}
        }
        
        self.experiment_results[experiment_id] = result
        
        print(f"🏆 A/B experiment results for {exp['name']}:")
        print(f"   Config A: {mean_a:.3f} ± {std_a:.3f}")
        print(f"   Config B: {mean_b:.3f} ± {std_b:.3f}")
        print(f"   Winner: {result['winner']} ({result['improvement']:+.1f}% improvement)")
        print(f"   Significant: {result['significant']} ({result['confidence']})")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary"""
        if not self.metrics_history:
            return {"status": "no_data", "message": "No performance data collected yet"}
        
        recent_metrics = list(self.metrics_history)[-100:]  # Last 100 metrics
        recent_scores = [m.value for m in recent_metrics]
        
        # Calculate improvement over time
        if len(recent_scores) >= 10:
            early_avg = np.mean(recent_scores[:10])
            late_avg = np.mean(recent_scores[-10:])
            improvement_pct = ((late_avg - early_avg) / early_avg * 100) if early_avg > 0 else 0
        else:
            improvement_pct = 0
        
        return {
            "status": "active",
            "total_samples": len(self.metrics_history),
            "optimization_iterations": self.optimization_iterations,
            "recent_performance": {
                "mean": float(np.mean(recent_scores)),
                "std": float(np.std(recent_scores)),
                "min": float(np.min(recent_scores)),
                "max": float(np.max(recent_scores))
            },
            "improvement_percent": improvement_pct,
            "feature_weights": self.feature_weights.tolist(),
            "active_experiments": len([exp for exp in self.ab_experiments.values() 
                                     if len(exp["samples_a"]) < 30 or len(exp["samples_b"]) < 30]),
            "completed_experiments": len(self.experiment_results)
        }
    
    def export_model(self, filename: str = None) -> str:
        """Export learned model and data"""
        if filename is None:
            filename = f"optimizer_model_{int(time.time())}.json"
        
        data = {
            "export_timestamp": datetime.now().isoformat(),
            "model": {
                "feature_weights": self.feature_weights.tolist(),
                "learning_rate": self.learning_rate,
                "optimization_iterations": self.optimization_iterations
            },
            "metrics_history": [asdict(m) for m in self.metrics_history],
            "ab_experiments": self.ab_experiments,
            "experiment_results": self.experiment_results,
            "summary": self.get_optimization_summary()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filename

# Global optimizer instance
_optimizer = None

def get_optimizer() -> ActiveOptimizer:
    """Get global optimizer instance"""
    global _optimizer
    if _optimizer is None:
        _optimizer = ActiveOptimizer()
    return _optimizer

async def demo_ml_optimization():
    """Demo ML optimization functionality"""
    optimizer = get_optimizer()
    
    print("🚀 ML Optimization Demo")
    
    # Simulate workflow performance data
    base_config = {
        'batch_size': 10,
        'timeout_ms': 2000,
        'retry_count': 2,
        'parallel_workers': 2,
        'circuit_breaker_threshold': 5
    }
    
    context = {
        'cpu_percent': 45.0,
        'memory_percent': 60.0,
        'active_workflows': 3
    }
    
    print("📊 Recording baseline performance...")
    for i in range(50):
        # Simulate realistic performance variations
        duration = random.gauss(800, 200)  # Mean 800ms, std 200ms
        success = random.random() > 0.05   # 95% success rate
        
        # Add some noise to config
        config_variant = base_config.copy()
        config_variant['batch_size'] += random.randint(-2, 2)
        
        await optimizer.record_performance(
            f"demo_workflow_{i:03d}",
            max(100, duration),  # Minimum 100ms
            success,
            config_variant,
            context
        )
        
        # Small delay to show real-time learning
        await asyncio.sleep(0.01)
    
    print("\n💡 Getting optimized configuration...")
    optimized_config = optimizer.suggest_configuration("demo", context)
    
    print("\n🧪 Starting A/B experiment...")
    config_b = optimized_config.copy()
    del config_b['predicted_score']  # Remove prediction from actual config
    
    exp_id = optimizer.start_ab_experiment(
        "batch_size_optimization",
        base_config,
        config_b
    )
    
    # Simulate A/B test data
    for i in range(60):
        variant, config = optimizer.get_ab_config(exp_id)
        
        # Config B should perform slightly better on average
        if variant == "B":
            perf_score = random.gauss(1.3, 0.2)  # Better performance
        else:
            perf_score = random.gauss(1.1, 0.2)  # Baseline performance
        
        optimizer.record_ab_result(exp_id, variant, max(0.1, perf_score))
    
    print("\n📈 Optimization Summary:")
    summary = optimizer.get_optimization_summary()
    print(f"   Improvement: {summary['improvement_percent']:+.1f}%")
    print(f"   Iterations: {summary['optimization_iterations']}")
    print(f"   Recent performance: {summary['recent_performance']['mean']:.3f} ± {summary['recent_performance']['std']:.3f}")
    
    # Export model
    filename = optimizer.export_model()
    print(f"💾 Model exported to: {filename}")
    
    print("✅ ML optimization demo complete")

if __name__ == "__main__":
    asyncio.run(demo_ml_optimization())