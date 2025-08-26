"""
Production Anomaly Detection System
Following Google, Netflix, Microsoft best practices for real-time anomaly detection
"""

import numpy as np
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
import json
import pickle
from enum import Enum
import hashlib
import threading
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies detected"""
    POINT = "point"  # Single data point anomaly
    CONTEXTUAL = "contextual"  # Anomalous in specific context
    COLLECTIVE = "collective"  # Pattern-based anomaly
    SEASONAL = "seasonal"  # Time-based pattern anomaly


class AnomalySeverity(Enum):
    """Severity levels following industry standards"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyAlert:
    """Anomaly detection result"""
    id: str
    timestamp: datetime
    metric_name: str
    value: float
    anomaly_score: float
    severity: AnomalySeverity
    anomaly_type: AnomalyType
    context: Dict[str, Any]
    confidence: float
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""


@dataclass
class DetectionResult:
    """Detection result with metadata"""
    is_anomaly: bool
    score: float
    threshold: float
    percentile: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class IsolationForest:
    """
    Isolation Forest implementation following scikit-learn patterns
    Optimized for real-time anomaly detection at scale
    """
    
    def __init__(self, n_estimators: int = 100, max_samples: Union[int, float] = 256, 
                 contamination: float = 0.1, max_features: Union[int, float] = 1.0,
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.max_depth = 0
        self.is_fitted = False
        
    def fit(self, X: np.ndarray) -> 'IsolationForest':
        """Fit isolation forest on training data"""
        n_samples, n_features = X.shape
        
        if isinstance(self.max_samples, float):
            self.max_samples = int(self.max_samples * n_samples)
        self.max_samples = min(self.max_samples, n_samples)
        
        if isinstance(self.max_features, float):
            self.max_features = int(self.max_features * n_features)
        self.max_features = min(self.max_features, n_features)
        
        self.max_depth = int(np.ceil(np.log2(max(self.max_samples, 2))))
        
        np.random.seed(self.random_state)
        
        for i in range(self.n_estimators):
            # Sample data for tree
            indices = np.random.choice(n_samples, self.max_samples, replace=False)
            sample_data = X[indices]
            
            # Build tree
            tree = self._build_tree(sample_data, 0)
            self.trees.append(tree)
            
        self.is_fitted = True
        return self
        
    def _build_tree(self, X: np.ndarray, depth: int) -> Dict:
        """Build isolation tree recursively"""
        n_samples, n_features = X.shape
        
        if depth >= self.max_depth or n_samples <= 1:
            return {'type': 'leaf', 'size': n_samples}
            
        # Random feature and split point
        feature = np.random.choice(n_features if self.max_features == n_features 
                                  else np.random.choice(n_features, self.max_features, replace=False))
        feature_values = X[:, feature]
        
        if len(np.unique(feature_values)) == 1:
            return {'type': 'leaf', 'size': n_samples}
            
        min_val, max_val = feature_values.min(), feature_values.max()
        split_point = np.random.uniform(min_val, max_val)
        
        left_mask = feature_values < split_point
        right_mask = ~left_mask
        
        left_data = X[left_mask]
        right_data = X[right_mask]
        
        return {
            'type': 'node',
            'feature': feature,
            'split_point': split_point,
            'left': self._build_tree(left_data, depth + 1) if len(left_data) > 0 else {'type': 'leaf', 'size': 0},
            'right': self._build_tree(right_data, depth + 1) if len(right_data) > 0 else {'type': 'leaf', 'size': 0}
        }
        
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Calculate anomaly scores"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
            
        scores = []
        for sample in X:
            path_lengths = []
            
            for tree in self.trees:
                path_length = self._path_length(sample, tree, 0)
                path_lengths.append(path_length)
                
            avg_path_length = np.mean(path_lengths)
            # Anomaly score: shorter paths = higher anomaly score
            c_n = 2 * (np.log(self.max_samples - 1) + 0.5772156649) - (2 * (self.max_samples - 1) / self.max_samples)
            score = 2 ** (-avg_path_length / c_n)
            scores.append(score)
            
        return np.array(scores)
        
    def _path_length(self, sample: np.ndarray, tree: Dict, depth: int) -> float:
        """Calculate path length for a sample in a tree"""
        if tree['type'] == 'leaf':
            # Average path length for unsuccessful search in BST
            size = tree['size']
            if size <= 1:
                return depth
            return depth + 2 * (np.log(size - 1) + 0.5772156649) - (2 * (size - 1) / size)
            
        feature = tree['feature']
        split_point = tree['split_point']
        
        if sample[feature] < split_point:
            return self._path_length(sample, tree['left'], depth + 1)
        else:
            return self._path_length(sample, tree['right'], depth + 1)
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies (1 for normal, -1 for anomaly)"""
        scores = self.decision_function(X)
        threshold = np.percentile(scores, (1 - self.contamination) * 100)
        return np.where(scores > threshold, -1, 1)


class StreamingStats:
    """Streaming statistics calculator for online anomaly detection"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self.sum = 0.0
        self.sum_squares = 0.0
        self.count = 0
        
    def update(self, value: float):
        """Update with new value"""
        if len(self.values) == self.window_size:
            # Remove oldest value
            old_value = self.values[0]
            self.sum -= old_value
            self.sum_squares -= old_value ** 2
            self.count -= 1
            
        self.values.append(value)
        self.sum += value
        self.sum_squares += value ** 2
        self.count += 1
        
    @property
    def mean(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0
        
    @property
    def variance(self) -> float:
        if self.count <= 1:
            return 0.0
        return (self.sum_squares - (self.sum ** 2) / self.count) / (self.count - 1)
        
    @property
    def std(self) -> float:
        return np.sqrt(self.variance)


class AnomalyDetector(ABC):
    """Abstract base class for anomaly detectors"""
    
    @abstractmethod
    async def detect(self, data: Dict[str, Any]) -> DetectionResult:
        pass
        
    @abstractmethod
    def fit(self, training_data: List[Dict[str, Any]]):
        pass


class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest based anomaly detector"""
    
    def __init__(self, feature_columns: List[str], **kwargs):
        self.feature_columns = feature_columns
        self.model = IsolationForest(**kwargs)
        self.scaler_stats = {}
        self.is_fitted = False
        
    def fit(self, training_data: List[Dict[str, Any]]):
        """Fit the model on training data"""
        # Extract features
        X = self._extract_features(training_data)
        
        # Calculate scaling statistics
        for i, col in enumerate(self.feature_columns):
            values = X[:, i]
            self.scaler_stats[col] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
            
        # Normalize features
        X_normalized = self._normalize_features(X)
        
        # Fit model
        self.model.fit(X_normalized)
        self.is_fitted = True
        
    def _extract_features(self, data: List[Dict[str, Any]]) -> np.ndarray:
        """Extract feature matrix from data"""
        features = []
        for sample in data:
            feature_vector = []
            for col in self.feature_columns:
                value = sample.get(col, 0.0)
                if isinstance(value, (int, float)):
                    feature_vector.append(float(value))
                else:
                    feature_vector.append(0.0)
            features.append(feature_vector)
        return np.array(features)
        
    def _normalize_features(self, X: np.ndarray) -> np.ndarray:
        """Normalize features using stored statistics"""
        X_normalized = np.copy(X)
        for i, col in enumerate(self.feature_columns):
            if col in self.scaler_stats:
                mean = self.scaler_stats[col]['mean']
                std = self.scaler_stats[col]['std']
                if std > 0:
                    X_normalized[:, i] = (X[:, i] - mean) / std
        return X_normalized
        
    async def detect(self, data: Dict[str, Any]) -> DetectionResult:
        """Detect anomaly in single data point"""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detection")
            
        # Extract and normalize features
        X = self._extract_features([data])
        X_normalized = self._normalize_features(X)
        
        # Get anomaly score
        score = self.model.decision_function(X_normalized)[0]
        
        # Calculate percentile and threshold
        # For isolation forest, higher scores indicate anomalies
        threshold = 0.5  # Typical threshold for isolation forest
        percentile = score * 100  # Convert to percentile
        
        is_anomaly = score > threshold
        
        return DetectionResult(
            is_anomaly=is_anomaly,
            score=score,
            threshold=threshold,
            percentile=percentile,
            metadata={
                'detector_type': 'isolation_forest',
                'features_used': self.feature_columns
            }
        )


class StatisticalDetector(AnomalyDetector):
    """Statistical anomaly detector using Z-score and modified Z-score"""
    
    def __init__(self, metric_name: str, window_size: int = 1000, threshold: float = 3.0):
        self.metric_name = metric_name
        self.stats = StreamingStats(window_size)
        self.threshold = threshold
        self.is_fitted = False
        
    def fit(self, training_data: List[Dict[str, Any]]):
        """Initialize with training data"""
        for sample in training_data:
            value = sample.get(self.metric_name, 0.0)
            if isinstance(value, (int, float)):
                self.stats.update(float(value))
        self.is_fitted = True
        
    async def detect(self, data: Dict[str, Any]) -> DetectionResult:
        """Detect anomaly using statistical methods"""
        value = data.get(self.metric_name, 0.0)
        if not isinstance(value, (int, float)):
            value = 0.0
            
        value = float(value)
        
        if not self.is_fitted or self.stats.count < 10:
            # Not enough data for reliable detection
            return DetectionResult(
                is_anomaly=False,
                score=0.0,
                threshold=self.threshold,
                percentile=50.0,
                metadata={'detector_type': 'statistical', 'reason': 'insufficient_data'}
            )
            
        # Calculate Z-score
        z_score = abs((value - self.stats.mean) / self.stats.std) if self.stats.std > 0 else 0.0
        
        # Update statistics with new value
        self.stats.update(value)
        
        is_anomaly = z_score > self.threshold
        percentile = min(99.9, z_score / self.threshold * 50 + 50)  # Rough percentile conversion
        
        return DetectionResult(
            is_anomaly=is_anomaly,
            score=z_score,
            threshold=self.threshold,
            percentile=percentile,
            metadata={
                'detector_type': 'statistical',
                'z_score': z_score,
                'mean': self.stats.mean,
                'std': self.stats.std
            }
        )


class AnomalyDetectionPipeline:
    """
    Production anomaly detection pipeline
    Following Google, Netflix best practices for real-time monitoring
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.detectors: Dict[str, AnomalyDetector] = {}
        self.alert_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.metrics = AnomalyMetrics()
        self.is_running = False
        self.lock = threading.Lock()
        
    def register_detector(self, name: str, detector: AnomalyDetector):
        """Register an anomaly detector"""
        with self.lock:
            self.detectors[name] = detector
            logger.info(f"Registered detector: {name}")
            
    def train_detectors(self, training_data: Dict[str, List[Dict[str, Any]]]):
        """Train all registered detectors"""
        for detector_name, detector in self.detectors.items():
            if detector_name in training_data:
                logger.info(f"Training detector: {detector_name}")
                detector.fit(training_data[detector_name])
                
    async def detect_anomalies(self, data: Dict[str, Any]) -> List[AnomalyAlert]:
        """Run anomaly detection on incoming data"""
        alerts = []
        
        # Run all detectors concurrently
        tasks = []
        for detector_name, detector in self.detectors.items():
            task = asyncio.create_task(self._run_detector(detector_name, detector, data))
            tasks.append(task)
            
        # Wait for all detectors
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for detector_name, result in zip(self.detectors.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Detector {detector_name} failed: {result}")
                continue
                
            if result and result.is_anomaly:
                alert = self._create_alert(detector_name, data, result)
                alerts.append(alert)
                self._record_alert(detector_name, alert)
                
        self.metrics.update_detection_stats(len(alerts), len(self.detectors))
        return alerts
        
    async def _run_detector(self, detector_name: str, detector: AnomalyDetector, 
                           data: Dict[str, Any]) -> Optional[DetectionResult]:
        """Run single detector with error handling"""
        try:
            return await detector.detect(data)
        except Exception as e:
            logger.error(f"Detector {detector_name} error: {e}")
            return None
            
    def _create_alert(self, detector_name: str, data: Dict[str, Any], 
                     result: DetectionResult) -> AnomalyAlert:
        """Create anomaly alert from detection result"""
        # Determine severity based on score and percentile
        if result.percentile >= 99.9:
            severity = AnomalySeverity.CRITICAL
        elif result.percentile >= 99:
            severity = AnomalySeverity.HIGH
        elif result.percentile >= 95:
            severity = AnomalySeverity.MEDIUM
        else:
            severity = AnomalySeverity.LOW
            
        # Determine anomaly type based on detector
        anomaly_type = AnomalyType.POINT
        if "statistical" in result.metadata.get('detector_type', ''):
            anomaly_type = AnomalyType.CONTEXTUAL
        elif "isolation" in result.metadata.get('detector_type', ''):
            anomaly_type = AnomalyType.COLLECTIVE
            
        # Generate alert ID
        alert_id = hashlib.md5(
            f"{detector_name}_{datetime.now().isoformat()}_{result.score}".encode()
        ).hexdigest()[:8]
        
        return AnomalyAlert(
            id=alert_id,
            timestamp=datetime.now(),
            metric_name=detector_name,
            value=data.get('value', 0.0),
            anomaly_score=result.score,
            severity=severity,
            anomaly_type=anomaly_type,
            context=data,
            confidence=min(result.percentile / 100.0, 1.0),
            explanation=f"Detected by {detector_name} with score {result.score:.3f}"
        )
        
    def _record_alert(self, detector_name: str, alert: AnomalyAlert):
        """Record alert in history"""
        self.alert_history[detector_name].append(alert)
        
    def get_alert_history(self, detector_name: Optional[str] = None, 
                         hours: int = 24) -> List[AnomalyAlert]:
        """Get recent alert history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        if detector_name:
            alerts = list(self.alert_history.get(detector_name, []))
        else:
            alerts = []
            for detector_alerts in self.alert_history.values():
                alerts.extend(detector_alerts)
                
        # Filter by time
        recent_alerts = [alert for alert in alerts if alert.timestamp >= cutoff_time]
        return sorted(recent_alerts, key=lambda x: x.timestamp, reverse=True)
        
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection pipeline statistics"""
        return {
            'detectors_registered': len(self.detectors),
            'total_alerts_24h': len(self.get_alert_history(hours=24)),
            'detection_metrics': self.metrics.get_stats(),
            'detector_status': {name: 'active' for name in self.detectors.keys()}
        }


class AnomalyMetrics:
    """Metrics collection for anomaly detection system"""
    
    def __init__(self):
        self.detection_count = 0
        self.anomaly_count = 0
        self.detector_performance = defaultdict(lambda: {'detections': 0, 'anomalies': 0})
        self.last_reset = datetime.now()
        
    def update_detection_stats(self, anomalies_found: int, detectors_run: int):
        """Update detection statistics"""
        self.detection_count += detectors_run
        self.anomaly_count += anomalies_found
        
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        runtime_hours = (datetime.now() - self.last_reset).total_seconds() / 3600
        
        return {
            'total_detections': self.detection_count,
            'total_anomalies': self.anomaly_count,
            'anomaly_rate': self.anomaly_count / max(self.detection_count, 1),
            'runtime_hours': runtime_hours,
            'detections_per_hour': self.detection_count / max(runtime_hours, 0.01)
        }


# Example usage and demonstration
async def demo_anomaly_detection():
    """Demonstrate the anomaly detection system"""
    print("🔍 Production Anomaly Detection System Demo")
    print("Following Google/Netflix best practices\n")
    
    # Initialize pipeline
    pipeline = AnomalyDetectionPipeline()
    
    # Register detectors
    # 1. Isolation Forest for multivariate anomalies
    iso_detector = IsolationForestDetector(
        feature_columns=['cpu_usage', 'memory_usage', 'request_rate'],
        contamination=0.1
    )
    pipeline.register_detector('system_metrics', iso_detector)
    
    # 2. Statistical detector for single metric
    stat_detector = StatisticalDetector('response_time', threshold=3.0)
    pipeline.register_detector('response_time', stat_detector)
    
    # Generate training data
    print("📊 Generating training data...")
    np.random.seed(42)
    
    # Normal system metrics
    normal_data = []
    for _ in range(1000):
        normal_data.append({
            'cpu_usage': np.random.normal(50, 10),
            'memory_usage': np.random.normal(60, 15),
            'request_rate': np.random.normal(100, 20),
            'response_time': np.random.normal(200, 50)
        })
        
    training_data = {
        'system_metrics': normal_data,
        'response_time': normal_data
    }
    
    # Train detectors
    print("🎯 Training anomaly detectors...")
    pipeline.train_detectors(training_data)
    
    # Test with normal and anomalous data
    print("\n🧪 Testing anomaly detection...")
    
    test_cases = [
        # Normal case
        {
            'name': 'Normal Operation',
            'data': {
                'cpu_usage': 55.0,
                'memory_usage': 65.0,
                'request_rate': 95.0,
                'response_time': 210.0,
                'timestamp': datetime.now()
            }
        },
        # CPU spike anomaly
        {
            'name': 'CPU Spike',
            'data': {
                'cpu_usage': 95.0,  # Anomalous
                'memory_usage': 60.0,
                'request_rate': 100.0,
                'response_time': 220.0,
                'timestamp': datetime.now()
            }
        },
        # Memory leak
        {
            'name': 'Memory Leak',
            'data': {
                'cpu_usage': 50.0,
                'memory_usage': 150.0,  # Anomalous
                'request_rate': 90.0,
                'response_time': 350.0,  # Also high
                'timestamp': datetime.now()
            }
        },
        # Traffic surge
        {
            'name': 'Traffic Surge',
            'data': {
                'cpu_usage': 80.0,
                'memory_usage': 90.0,
                'request_rate': 500.0,  # Anomalous
                'response_time': 800.0,  # Very high
                'timestamp': datetime.now()
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        alerts = await pipeline.detect_anomalies(test_case['data'])
        
        if alerts:
            for alert in alerts:
                print(f"🚨 ANOMALY DETECTED:")
                print(f"   Detector: {alert.metric_name}")
                print(f"   Score: {alert.anomaly_score:.3f}")
                print(f"   Severity: {alert.severity.value.upper()}")
                print(f"   Confidence: {alert.confidence:.1%}")
                print(f"   Type: {alert.anomaly_type.value}")
                print(f"   Explanation: {alert.explanation}")
        else:
            print("✅ No anomalies detected - system operating normally")
    
    # Show pipeline statistics
    print(f"\n📈 Detection Pipeline Statistics:")
    stats = pipeline.get_detection_stats()
    for key, value in stats.items():
        if key != 'detection_metrics':
            print(f"   {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n📊 Detection Metrics:")
    metrics = stats['detection_metrics']
    for key, value in metrics.items():
        print(f"   {key.replace('_', ' ').title()}: {value:.3f}" if isinstance(value, float) else f"   {key.replace('_', ' ').title()}: {value}")
    
    # Show recent alerts
    recent_alerts = pipeline.get_alert_history(hours=1)
    print(f"\n🔔 Recent Alerts ({len(recent_alerts)} in last hour):")
    for alert in recent_alerts[:5]:  # Show last 5
        print(f"   {alert.timestamp.strftime('%H:%M:%S')} - {alert.metric_name} - {alert.severity.value.upper()}")


if __name__ == "__main__":
    asyncio.run(demo_anomaly_detection())