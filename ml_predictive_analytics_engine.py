#!/usr/bin/env python3
"""
ML-Powered Predictive Analytics Engine - Phase 2
Advanced intelligence system for autonomous optimization and predictive insights
"""

import asyncio
import sys
import os
import json
import logging
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import traceback

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Advanced ML libraries with fallbacks
try:
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn not available - using lightweight ML fallbacks")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️  pandas not available - using native data structures")

# Import our existing systems
from integrated_workflow_system import IntegratedWorkflowSystem
from performance_optimization_monitoring import PerformanceOptimizationMonitor

class PredictionType(Enum):
    FAILURE_PREDICTION = "failure_prediction"
    PERFORMANCE_FORECASTING = "performance_forecasting"
    RESOURCE_OPTIMIZATION = "resource_optimization"
    ANOMALY_DETECTION = "anomaly_detection"
    WORKFLOW_OPTIMIZATION = "workflow_optimization"

class AnalyticsModel(Enum):
    ISOLATION_FOREST = "isolation_forest"
    RANDOM_FOREST = "random_forest" 
    LIGHTWEIGHT_STATS = "lightweight_stats"
    DBSCAN_CLUSTERING = "dbscan_clustering"
    LINEAR_REGRESSION = "linear_regression"

@dataclass
class PredictiveInsight:
    insight_id: str
    prediction_type: PredictionType
    model_used: AnalyticsModel
    confidence_score: float
    prediction: Dict[str, Any]
    recommended_actions: List[str]
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    predicted_impact: Dict[str, float]
    time_horizon: str  # immediate, short_term, medium_term, long_term
    supporting_data: Dict[str, Any]
    timestamp: datetime
    expires_at: datetime

@dataclass
class ModelPerformance:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    last_trained: datetime
    training_samples: int
    validation_score: float

class LightweightMLEngine:
    """Lightweight ML engine for systems without sklearn"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    def detect_anomalies(self, data: List[float], contamination: float = 0.1) -> List[bool]:
        """Simple statistical anomaly detection"""
        if not data:
            return []
        
        mean_val = sum(data) / len(data)
        variance = sum((x - mean_val) ** 2 for x in data) / len(data)
        std_dev = variance ** 0.5
        
        # Use 3-sigma rule for anomaly detection
        threshold = 3 * std_dev
        anomalies = [abs(x - mean_val) > threshold for x in data]
        
        # Ensure we don't exceed contamination rate
        num_anomalies = sum(anomalies)
        max_anomalies = int(len(data) * contamination)
        
        if num_anomalies > max_anomalies:
            # Keep only the most extreme anomalies
            deviations = [(abs(x - mean_val), i) for i, x in enumerate(data)]
            deviations.sort(reverse=True)
            
            anomalies = [False] * len(data)
            for _, idx in deviations[:max_anomalies]:
                anomalies[idx] = True
        
        return anomalies
    
    def predict_trend(self, data: List[float], steps: int = 5) -> List[float]:
        """Simple linear trend prediction"""
        if len(data) < 2:
            return [data[-1] if data else 0.0] * steps
        
        # Calculate linear trend
        x = list(range(len(data)))
        n = len(data)
        
        sum_x = sum(x)
        sum_y = sum(data)
        sum_xy = sum(x[i] * data[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        # Linear regression coefficients
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        intercept = (sum_y - slope * sum_x) / n
        
        # Predict future values
        predictions = []
        for i in range(steps):
            future_x = len(data) + i
            predicted_y = slope * future_x + intercept
            predictions.append(predicted_y)
        
        return predictions

class MLPredictiveAnalyticsEngine:
    """Advanced ML-powered predictive analytics engine"""
    
    def __init__(self, engine_id: str = "ml_analytics_001"):
        self.engine_id = engine_id
        self.workflow_system = None
        self.performance_monitor = None
        self.lightweight_ml = LightweightMLEngine()
        
        # Model storage
        self.trained_models = {}
        self.model_performance = {}
        self.prediction_history = []
        self.feature_scalers = {}
        
        # Analytics configuration
        self.config = {
            "anomaly_detection_contamination": 0.05,  # 5% expected anomalies
            "prediction_confidence_threshold": 0.7,
            "model_retrain_interval_hours": 24,
            "max_prediction_history": 10000,
            "feature_window_minutes": 60,
            "prediction_horizons": {
                "immediate": 5,      # 5 minutes
                "short_term": 60,    # 1 hour  
                "medium_term": 1440, # 24 hours
                "long_term": 10080   # 1 week
            }
        }
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Metrics tracking
        self.metrics = {
            "total_predictions": 0,
            "successful_predictions": 0,
            "model_accuracy": {},
            "prediction_confidence_avg": 0.0,
            "anomalies_detected": 0,
            "preventive_actions_triggered": 0,
            "engine_uptime_start": datetime.now()
        }
        
        print(f"🧠 ML Predictive Analytics Engine {self.engine_id} initialized")
        print(f"   ✅ Models: {'scikit-learn' if SKLEARN_AVAILABLE else 'Lightweight ML'}")
        print(f"   ✅ Data processing: {'pandas' if PANDAS_AVAILABLE else 'Native Python'}")
        print(f"   ✅ Prediction types: {len(PredictionType)} supported")
        print(f"   ✅ Analytics models: {len(AnalyticsModel)} available")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging for the analytics engine"""
        logger = logging.getLogger(f"ml_analytics_{self.engine_id}")
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s | ML-ANALYTICS-{self.engine_id} | %(levelname)s | %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    async def initialize(self) -> bool:
        """Initialize the ML analytics engine and dependencies"""
        try:
            self.logger.info("🧠 Initializing ML Predictive Analytics Engine...")
            
            # Initialize workflow system
            self.workflow_system = IntegratedWorkflowSystem()
            
            # Initialize performance monitor
            self.performance_monitor = PerformanceOptimizationMonitor()
            await self.performance_monitor.initialize()
            
            # Initialize base models
            await self._initialize_base_models()
            
            self.logger.info("✅ ML Predictive Analytics Engine fully initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ML analytics engine: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    async def _initialize_base_models(self):
        """Initialize base ML models for different prediction types"""
        
        if SKLEARN_AVAILABLE:
            # Anomaly detection model
            self.trained_models[AnalyticsModel.ISOLATION_FOREST] = IsolationForest(
                contamination=self.config["anomaly_detection_contamination"],
                random_state=42
            )
            
            # Performance forecasting model
            self.trained_models[AnalyticsModel.RANDOM_FOREST] = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            
            # Clustering for pattern analysis
            self.trained_models[AnalyticsModel.DBSCAN_CLUSTERING] = DBSCAN(
                eps=0.5,
                min_samples=5
            )
            
            self.logger.info("✅ Advanced ML models initialized (scikit-learn)")
        else:
            self.logger.info("✅ Lightweight ML models initialized (statistical)")
    
    async def collect_system_features(self) -> Dict[str, Any]:
        """Collect comprehensive system features for ML analysis"""
        
        features = {
            "timestamp": datetime.now(),
            "system_metrics": {},
            "workflow_metrics": {},
            "performance_metrics": {}
        }
        
        try:
            # Get system health data
            if self.workflow_system:
                health = await self.workflow_system.get_system_health()
                features["system_metrics"] = {
                    "overall_status_score": 1.0 if health.get("overall_status") == "healthy" else 0.0,
                    "workflow_count": health.get("active_workflows", 0),
                    "event_count": health.get("total_events", 0),
                    "ml_optimizer_learning_rate": health.get("ml_optimizer", {}).get("learning_rate", 0.01)
                }
            
            # Get performance data
            if self.performance_monitor:
                perf_data = await self.performance_monitor.get_current_metrics()
                if perf_data:
                    features["performance_metrics"] = {
                        "cpu_percent": perf_data.get("system", {}).get("cpu_percent", 0.0),
                        "memory_percent": perf_data.get("system", {}).get("memory_percent", 0.0),
                        "disk_io_read": perf_data.get("system", {}).get("disk_io", {}).get("read_bytes", 0),
                        "disk_io_write": perf_data.get("system", {}).get("disk_io", {}).get("write_bytes", 0),
                        "network_sent": perf_data.get("system", {}).get("network_io", {}).get("bytes_sent", 0),
                        "network_recv": perf_data.get("system", {}).get("network_io", {}).get("bytes_recv", 0)
                    }
            
            # Calculate derived features
            features["derived_metrics"] = self._calculate_derived_features(features)
            
        except Exception as e:
            self.logger.warning(f"Error collecting features: {e}")
        
        return features
    
    def _calculate_derived_features(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Calculate derived features from raw metrics"""
        
        derived = {}
        
        try:
            perf = features.get("performance_metrics", {})
            system = features.get("system_metrics", {})
            
            # Resource utilization ratios
            derived["memory_cpu_ratio"] = perf.get("memory_percent", 0) / max(perf.get("cpu_percent", 1), 1)
            derived["disk_network_ratio"] = (perf.get("disk_io_read", 0) + perf.get("disk_io_write", 0)) / max(
                perf.get("network_sent", 1) + perf.get("network_recv", 1), 1
            )
            
            # System health indicators  
            derived["workflow_load_factor"] = system.get("workflow_count", 0) * system.get("event_count", 0)
            derived["system_stress_indicator"] = (
                perf.get("cpu_percent", 0) * 0.4 +
                perf.get("memory_percent", 0) * 0.4 +
                min(derived["memory_cpu_ratio"], 5.0) * 0.2
            )
            
            # Trend indicators (would be calculated from historical data in production)
            derived["performance_trend"] = 0.0  # Placeholder for trend calculation
            derived["anomaly_likelihood"] = min(derived["system_stress_indicator"] / 100.0, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Error calculating derived features: {e}")
            
        return derived
    
    async def detect_anomalies(self, data_window_minutes: int = 60) -> PredictiveInsight:
        """Detect system anomalies using ML analysis"""
        
        insight_id = f"anomaly_{int(time.time())}"
        
        try:
            # Collect recent feature data
            features = await self.collect_system_features()
            
            # For demo, simulate feature history
            feature_history = []
            for i in range(100):  # Simulate 100 data points
                sim_features = [
                    features["performance_metrics"].get("cpu_percent", 0) + np.random.normal(0, 5),
                    features["performance_metrics"].get("memory_percent", 0) + np.random.normal(0, 3),
                    features["derived_metrics"].get("system_stress_indicator", 0) + np.random.normal(0, 2)
                ]
                feature_history.append(sim_features)
            
            # Detect anomalies
            if SKLEARN_AVAILABLE and AnalyticsModel.ISOLATION_FOREST in self.trained_models:
                model = self.trained_models[AnalyticsModel.ISOLATION_FOREST]
                model.fit(feature_history)
                
                current_features = [[
                    features["performance_metrics"].get("cpu_percent", 0),
                    features["performance_metrics"].get("memory_percent", 0), 
                    features["derived_metrics"].get("system_stress_indicator", 0)
                ]]
                
                anomaly_score = model.decision_function(current_features)[0]
                is_anomaly = model.predict(current_features)[0] == -1
                confidence = abs(anomaly_score)
                model_used = AnalyticsModel.ISOLATION_FOREST
                
            else:
                # Lightweight anomaly detection
                stress_values = [f[2] for f in feature_history]  # System stress indicators
                anomalies = self.lightweight_ml.detect_anomalies(stress_values)
                current_stress = features["derived_metrics"].get("system_stress_indicator", 0)
                
                is_anomaly = current_stress > np.mean(stress_values) + 2 * np.std(stress_values)
                confidence = min(abs(current_stress - np.mean(stress_values)) / max(np.std(stress_values), 1), 1.0)
                anomaly_score = confidence if is_anomaly else -confidence
                model_used = AnalyticsModel.LIGHTWEIGHT_STATS
            
            # Determine severity and recommendations
            if is_anomaly and confidence > 0.8:
                severity = "CRITICAL"
                recommended_actions = [
                    "Investigate resource usage patterns immediately",
                    "Check for runaway processes or memory leaks",
                    "Consider scaling system resources",
                    "Review recent system changes"
                ]
            elif is_anomaly and confidence > 0.6:
                severity = "HIGH"
                recommended_actions = [
                    "Monitor system closely for continued anomalies",
                    "Review system performance metrics",
                    "Prepare for potential resource scaling"
                ]
            elif is_anomaly:
                severity = "MEDIUM"
                recommended_actions = [
                    "Continue monitoring system behavior",
                    "Log anomaly for trend analysis"
                ]
            else:
                severity = "LOW"
                recommended_actions = ["System operating within normal parameters"]
            
            # Update metrics
            self.metrics["total_predictions"] += 1
            if is_anomaly:
                self.metrics["anomalies_detected"] += 1
            
            insight = PredictiveInsight(
                insight_id=insight_id,
                prediction_type=PredictionType.ANOMALY_DETECTION,
                model_used=model_used,
                confidence_score=confidence,
                prediction={
                    "is_anomaly": is_anomaly,
                    "anomaly_score": float(anomaly_score),
                    "confidence": confidence,
                    "affected_metrics": ["cpu_percent", "memory_percent", "system_stress_indicator"]
                },
                recommended_actions=recommended_actions,
                severity=severity,
                predicted_impact={
                    "performance_degradation": confidence * 0.3 if is_anomaly else 0.0,
                    "system_instability_risk": confidence * 0.2 if is_anomaly else 0.0
                },
                time_horizon="immediate",
                supporting_data=features,
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(minutes=30)
            )
            
            self.prediction_history.append(insight)
            
            # Keep history manageable
            if len(self.prediction_history) > self.config["max_prediction_history"]:
                self.prediction_history = self.prediction_history[-self.config["max_prediction_history"]:]
            
            self.logger.info(f"🔍 Anomaly detection complete - Anomaly: {is_anomaly}, Confidence: {confidence:.3f}")
            
            return insight
            
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {e}")
            self.logger.error(traceback.format_exc())
            
            # Return fallback insight
            return PredictiveInsight(
                insight_id=insight_id,
                prediction_type=PredictionType.ANOMALY_DETECTION,
                model_used=AnalyticsModel.LIGHTWEIGHT_STATS,
                confidence_score=0.1,
                prediction={"error": str(e), "is_anomaly": False},
                recommended_actions=["Check ML analytics engine logs"],
                severity="LOW",
                predicted_impact={},
                time_horizon="immediate",
                supporting_data={},
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(minutes=5)
            )
    
    async def predict_performance_trends(self, forecast_minutes: int = 60) -> PredictiveInsight:
        """Predict system performance trends using ML forecasting"""
        
        insight_id = f"performance_forecast_{int(time.time())}"
        
        try:
            # Collect current features
            features = await self.collect_system_features()
            
            # Simulate historical performance data
            historical_performance = []
            base_cpu = features["performance_metrics"].get("cpu_percent", 30.0)
            base_memory = features["performance_metrics"].get("memory_percent", 50.0)
            
            for i in range(100):
                # Simulate trending data with noise
                trend_factor = i * 0.1
                cpu_val = base_cpu + trend_factor + np.random.normal(0, 3)
                memory_val = base_memory + trend_factor * 0.5 + np.random.normal(0, 2)
                historical_performance.append([cpu_val, memory_val])
            
            # Predict future performance
            if SKLEARN_AVAILABLE and AnalyticsModel.RANDOM_FOREST in self.trained_models:
                model = self.trained_models[AnalyticsModel.RANDOM_FOREST]
                
                # Prepare training data (features: time index, targets: performance metrics)
                X = [[i] for i in range(len(historical_performance))]
                y_cpu = [p[0] for p in historical_performance]
                y_memory = [p[1] for p in historical_performance]
                
                # Train models
                cpu_model = RandomForestRegressor(n_estimators=50, random_state=42)
                memory_model = RandomForestRegressor(n_estimators=50, random_state=42)
                
                cpu_model.fit(X, y_cpu)
                memory_model.fit(X, y_memory)
                
                # Predict future values
                future_steps = min(forecast_minutes // 5, 20)  # Predict up to 20 steps
                future_X = [[len(historical_performance) + i] for i in range(future_steps)]
                
                cpu_predictions = cpu_model.predict(future_X)
                memory_predictions = memory_model.predict(future_X)
                
                # Calculate prediction confidence based on model performance
                cpu_score = cpu_model.score(X, y_cpu)
                memory_score = memory_model.score(X, y_memory)
                confidence = (cpu_score + memory_score) / 2
                
                model_used = AnalyticsModel.RANDOM_FOREST
                
            else:
                # Lightweight trend prediction
                cpu_values = [p[0] for p in historical_performance]
                memory_values = [p[1] for p in historical_performance]
                
                future_steps = min(forecast_minutes // 5, 20)
                cpu_predictions = self.lightweight_ml.predict_trend(cpu_values, future_steps)
                memory_predictions = self.lightweight_ml.predict_trend(memory_values, future_steps)
                
                # Simple confidence based on trend consistency
                confidence = 0.7  # Moderate confidence for simple linear trends
                model_used = AnalyticsModel.LIGHTWEIGHT_STATS
            
            # Analyze predictions for issues
            max_cpu_predicted = max(cpu_predictions) if cpu_predictions else 0
            max_memory_predicted = max(memory_predictions) if memory_predictions else 0
            
            # Determine severity and recommendations
            if max_cpu_predicted > 90 or max_memory_predicted > 90:
                severity = "CRITICAL"
                recommended_actions = [
                    "Immediate resource scaling required",
                    "Investigate high resource consumers",
                    "Consider load balancing or system optimization",
                    "Prepare emergency scaling procedures"
                ]
            elif max_cpu_predicted > 80 or max_memory_predicted > 80:
                severity = "HIGH"
                recommended_actions = [
                    "Plan resource scaling within next hour",
                    "Monitor system load closely",
                    "Review resource allocation policies"
                ]
            elif max_cpu_predicted > 70 or max_memory_predicted > 70:
                severity = "MEDIUM"
                recommended_actions = [
                    "Consider proactive resource optimization",
                    "Monitor trends for continued growth"
                ]
            else:
                severity = "LOW"
                recommended_actions = ["Performance trending within acceptable ranges"]
            
            # Update metrics
            self.metrics["total_predictions"] += 1
            self.metrics["successful_predictions"] += 1
            
            insight = PredictiveInsight(
                insight_id=insight_id,
                prediction_type=PredictionType.PERFORMANCE_FORECASTING,
                model_used=model_used,
                confidence_score=confidence,
                prediction={
                    "forecast_horizon_minutes": forecast_minutes,
                    "cpu_trend": cpu_predictions[:10],  # First 10 predictions
                    "memory_trend": memory_predictions[:10],
                    "max_cpu_predicted": max_cpu_predicted,
                    "max_memory_predicted": max_memory_predicted,
                    "trend_direction": "increasing" if max_cpu_predicted > base_cpu else "stable"
                },
                recommended_actions=recommended_actions,
                severity=severity,
                predicted_impact={
                    "performance_degradation": max(0, (max_cpu_predicted - 70) / 30) if max_cpu_predicted > 70 else 0,
                    "resource_exhaustion_risk": max(0, (max_memory_predicted - 80) / 20) if max_memory_predicted > 80 else 0
                },
                time_horizon="medium_term",
                supporting_data=features,
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(minutes=forecast_minutes)
            )
            
            self.prediction_history.append(insight)
            
            self.logger.info(f"📈 Performance forecasting complete - Max CPU: {max_cpu_predicted:.1f}%, Max Memory: {max_memory_predicted:.1f}%")
            
            return insight
            
        except Exception as e:
            self.logger.error(f"Error in performance forecasting: {e}")
            self.logger.error(traceback.format_exc())
            
            # Return fallback insight
            return PredictiveInsight(
                insight_id=insight_id,
                prediction_type=PredictionType.PERFORMANCE_FORECASTING,
                model_used=AnalyticsModel.LIGHTWEIGHT_STATS,
                confidence_score=0.1,
                prediction={"error": str(e)},
                recommended_actions=["Check ML analytics engine logs"],
                severity="LOW",
                predicted_impact={},
                time_horizon="medium_term",
                supporting_data={},
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(minutes=30)
            )
    
    async def optimize_resource_allocation(self) -> PredictiveInsight:
        """Generate ML-powered resource optimization recommendations"""
        
        insight_id = f"resource_opt_{int(time.time())}"
        
        try:
            # Collect system features
            features = await self.collect_system_features()
            
            # Analyze current resource utilization patterns
            current_cpu = features["performance_metrics"].get("cpu_percent", 0)
            current_memory = features["performance_metrics"].get("memory_percent", 0)
            workflow_count = features["system_metrics"].get("workflow_count", 0)
            
            # Calculate optimization opportunities
            optimizations = []
            predicted_savings = {}
            
            # CPU optimization analysis
            if current_cpu < 30:
                optimizations.append({
                    "resource": "cpu",
                    "current_utilization": current_cpu,
                    "recommendation": "downscale",
                    "suggested_change": f"Reduce CPU allocation by {max(10, 50-current_cpu)}%",
                    "estimated_savings": f"{(50-current_cpu) * 0.02:.1f}% cost reduction"
                })
                predicted_savings["cpu_cost_reduction"] = (50-current_cpu) * 0.02
            elif current_cpu > 80:
                optimizations.append({
                    "resource": "cpu", 
                    "current_utilization": current_cpu,
                    "recommendation": "upscale",
                    "suggested_change": f"Increase CPU allocation by {current_cpu-70}%",
                    "estimated_impact": f"Prevent {(current_cpu-70) * 0.1:.1f}% performance degradation"
                })
                predicted_savings["performance_improvement"] = (current_cpu-70) * 0.1
            
            # Memory optimization analysis
            if current_memory < 40:
                optimizations.append({
                    "resource": "memory",
                    "current_utilization": current_memory,
                    "recommendation": "optimize_allocation",
                    "suggested_change": f"Redistribute memory from over-allocated processes",
                    "estimated_savings": f"{(60-current_memory) * 0.015:.1f}% memory cost reduction"
                })
                predicted_savings["memory_optimization"] = (60-current_memory) * 0.015
            elif current_memory > 85:
                optimizations.append({
                    "resource": "memory",
                    "current_utilization": current_memory,
                    "recommendation": "increase_allocation",
                    "suggested_change": f"Add {current_memory-80}% additional memory",
                    "estimated_impact": "Prevent memory pressure and swapping"
                })
            
            # Workflow-based optimization
            if workflow_count > 0:
                workflow_efficiency = min(100, (current_cpu + current_memory) / workflow_count)
                if workflow_efficiency < 50:
                    optimizations.append({
                        "resource": "workflow_scheduling",
                        "current_efficiency": workflow_efficiency,
                        "recommendation": "optimize_scheduling",
                        "suggested_change": "Implement intelligent workflow batching",
                        "estimated_improvement": f"{100-workflow_efficiency:.1f}% efficiency gain"
                    })
                    predicted_savings["workflow_efficiency"] = 100-workflow_efficiency
            
            # Determine overall recommendations
            if not optimizations:
                severity = "LOW"
                recommended_actions = ["System resource allocation is optimal"]
                confidence = 0.9
            elif len(optimizations) >= 3:
                severity = "HIGH"
                recommended_actions = [
                    "Multiple optimization opportunities identified",
                    "Implement resource optimization plan",
                    "Monitor resource utilization after changes"
                ]
                confidence = 0.8
            else:
                severity = "MEDIUM"
                recommended_actions = [
                    "Moderate optimization opportunities available",
                    "Consider implementing suggested changes during low-usage periods"
                ]
                confidence = 0.75
            
            # Calculate total predicted impact
            total_cost_savings = sum(v for k, v in predicted_savings.items() if "cost" in k or "optimization" in k)
            total_performance_gain = sum(v for k, v in predicted_savings.items() if "performance" in k or "efficiency" in k)
            
            insight = PredictiveInsight(
                insight_id=insight_id,
                prediction_type=PredictionType.RESOURCE_OPTIMIZATION,
                model_used=AnalyticsModel.LIGHTWEIGHT_STATS,
                confidence_score=confidence,
                prediction={
                    "optimizations_found": len(optimizations),
                    "optimization_details": optimizations,
                    "current_resource_state": {
                        "cpu_percent": current_cpu,
                        "memory_percent": current_memory,
                        "workflow_count": workflow_count
                    }
                },
                recommended_actions=recommended_actions + [opt["suggested_change"] for opt in optimizations],
                severity=severity,
                predicted_impact={
                    "cost_savings_percent": total_cost_savings,
                    "performance_improvement_percent": total_performance_gain,
                    "efficiency_gain": predicted_savings.get("workflow_efficiency", 0)
                },
                time_horizon="short_term",
                supporting_data=features,
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=2)
            )
            
            self.prediction_history.append(insight)
            self.metrics["total_predictions"] += 1
            self.metrics["successful_predictions"] += 1
            
            self.logger.info(f"🔧 Resource optimization analysis complete - {len(optimizations)} opportunities found")
            
            return insight
            
        except Exception as e:
            self.logger.error(f"Error in resource optimization: {e}")
            self.logger.error(traceback.format_exc())
            return PredictiveInsight(
                insight_id=insight_id,
                prediction_type=PredictionType.RESOURCE_OPTIMIZATION,
                model_used=AnalyticsModel.LIGHTWEIGHT_STATS,
                confidence_score=0.1,
                prediction={"error": str(e)},
                recommended_actions=["Check ML analytics engine logs"],
                severity="LOW",
                predicted_impact={},
                time_horizon="short_term",
                supporting_data={},
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(minutes=30)
            )
    
    async def predict_failure_risks(self) -> PredictiveInsight:
        """Predict potential system failure risks using ML analysis"""
        
        insight_id = f"failure_risk_{int(time.time())}"
        
        try:
            # Collect comprehensive system data
            features = await self.collect_system_features()
            
            # Calculate failure risk indicators
            risk_factors = []
            risk_scores = {}
            
            # Resource exhaustion risk
            cpu_percent = features["performance_metrics"].get("cpu_percent", 0)
            memory_percent = features["performance_metrics"].get("memory_percent", 0)
            
            if cpu_percent > 90:
                risk_factors.append("Critical CPU usage")
                risk_scores["cpu_exhaustion"] = min(1.0, (cpu_percent - 70) / 30)
            
            if memory_percent > 90:
                risk_factors.append("Critical memory usage")
                risk_scores["memory_exhaustion"] = min(1.0, (memory_percent - 70) / 30)
            
            # System stress indicators
            stress_indicator = features["derived_metrics"].get("system_stress_indicator", 0)
            if stress_indicator > 80:
                risk_factors.append("High system stress")
                risk_scores["system_stress"] = min(1.0, stress_indicator / 100)
            
            # I/O bottleneck risk  
            disk_read = features["performance_metrics"].get("disk_io_read", 0)
            disk_write = features["performance_metrics"].get("disk_io_write", 0)
            total_io = disk_read + disk_write
            
            if total_io > 1e9:  # 1GB I/O
                risk_factors.append("High disk I/O activity")
                risk_scores["io_bottleneck"] = min(1.0, total_io / 5e9)  # Scale to 5GB max
            
            # Network saturation risk
            network_sent = features["performance_metrics"].get("network_sent", 0)
            network_recv = features["performance_metrics"].get("network_recv", 0)
            total_network = network_sent + network_recv
            
            if total_network > 5e8:  # 500MB network
                risk_factors.append("High network activity")
                risk_scores["network_saturation"] = min(1.0, total_network / 1e9)  # Scale to 1GB max
            
            # Calculate overall failure risk
            if risk_scores:
                overall_risk = sum(risk_scores.values()) / len(risk_scores)
                max_individual_risk = max(risk_scores.values())
                confidence = min(0.9, max_individual_risk + 0.1)
            else:
                overall_risk = 0.0
                max_individual_risk = 0.0
                confidence = 0.8  # High confidence in low-risk assessment
            
            # Determine severity and time to potential failure
            if overall_risk > 0.8:
                severity = "CRITICAL"
                time_to_failure = "immediate"
                recommended_actions = [
                    "IMMEDIATE ACTION REQUIRED",
                    "Scale system resources immediately",
                    "Investigate resource consumption patterns",
                    "Prepare for emergency system restart if needed",
                    "Alert system administrators"
                ]
            elif overall_risk > 0.6:
                severity = "HIGH"
                time_to_failure = "short_term"
                recommended_actions = [
                    "Take preventive action within 1 hour",
                    "Reduce system load",
                    "Monitor critical resources closely",
                    "Prepare scaling procedures"
                ]
            elif overall_risk > 0.3:
                severity = "MEDIUM"
                time_to_failure = "medium_term"
                recommended_actions = [
                    "Monitor system trends",
                    "Consider proactive resource optimization",
                    "Review system capacity planning"
                ]
            else:
                severity = "LOW"
                time_to_failure = "long_term"
                recommended_actions = ["System operating within safe parameters"]
            
            # Update metrics
            self.metrics["total_predictions"] += 1
            if overall_risk > 0.5:
                self.metrics["preventive_actions_triggered"] += 1
            
            insight = PredictiveInsight(
                insight_id=insight_id,
                prediction_type=PredictionType.FAILURE_PREDICTION,
                model_used=AnalyticsModel.LIGHTWEIGHT_STATS,
                confidence_score=confidence,
                prediction={
                    "overall_failure_risk": overall_risk,
                    "risk_factors": risk_factors,
                    "risk_breakdown": risk_scores,
                    "time_to_potential_failure": time_to_failure,
                    "primary_risk_factor": max(risk_scores.keys(), key=lambda k: risk_scores[k]) if risk_scores else None
                },
                recommended_actions=recommended_actions,
                severity=severity,
                predicted_impact={
                    "system_downtime_probability": overall_risk,
                    "data_loss_risk": max_individual_risk * 0.3,
                    "service_degradation_probability": min(1.0, overall_risk * 1.2)
                },
                time_horizon=time_to_failure,
                supporting_data=features,
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(minutes=15)
            )
            
            self.prediction_history.append(insight)
            
            self.logger.info(f"⚠️  Failure risk analysis complete - Overall risk: {overall_risk:.3f}, Factors: {len(risk_factors)}")
            
            return insight
            
        except Exception as e:
            self.logger.error(f"Error in failure prediction: {e}")
            self.logger.error(traceback.format_exc())
            return PredictiveInsight(
                insight_id=insight_id,
                prediction_type=PredictionType.FAILURE_PREDICTION,
                model_used=AnalyticsModel.LIGHTWEIGHT_STATS,
                confidence_score=0.1,
                prediction={"error": str(e)},
                recommended_actions=["Check ML analytics engine logs"],
                severity="LOW",
                predicted_impact={},
                time_horizon="long_term",
                supporting_data={},
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(minutes=15)
            )
    
    async def generate_comprehensive_insights(self) -> Dict[str, PredictiveInsight]:
        """Generate comprehensive ML-powered insights across all prediction types"""
        
        self.logger.info("🔮 Generating comprehensive predictive insights...")
        
        insights = {}
        
        try:
            # Run all prediction types
            insights["anomaly_detection"] = await self.detect_anomalies()
            insights["performance_forecasting"] = await self.predict_performance_trends()
            insights["resource_optimization"] = await self.optimize_resource_allocation()
            insights["failure_prediction"] = await self.predict_failure_risks()
            
            # Update average confidence
            total_confidence = sum(insight.confidence_score for insight in insights.values())
            self.metrics["prediction_confidence_avg"] = total_confidence / len(insights)
            
            self.logger.info(f"✅ Generated {len(insights)} predictive insights")
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive insights: {e}")
        
        return insights
    
    async def get_analytics_status(self) -> Dict[str, Any]:
        """Get comprehensive analytics engine status"""
        
        uptime = (datetime.now() - self.metrics["engine_uptime_start"]).total_seconds()
        
        return {
            "engine_id": self.engine_id,
            "status": "operational",
            "uptime_seconds": uptime,
            "uptime_human": f"{uptime/3600:.1f} hours",
            "capabilities": {
                "sklearn_available": SKLEARN_AVAILABLE,
                "pandas_available": PANDAS_AVAILABLE,
                "prediction_types": len(PredictionType),
                "model_types": len(AnalyticsModel)
            },
            "metrics": self.metrics,
            "active_models": list(self.trained_models.keys()),
            "prediction_history_count": len(self.prediction_history),
            "configuration": self.config
        }
    
    async def run_comprehensive_demo(self) -> Dict[str, Any]:
        """Run comprehensive demonstration of ML analytics capabilities"""
        
        self.logger.info("🎯 Starting comprehensive ML analytics demonstration...")
        
        start_time = datetime.now()
        
        # Generate all types of insights
        insights = await self.generate_comprehensive_insights()
        
        # Calculate demo metrics
        processing_time = (datetime.now() - start_time).total_seconds()
        
        demo_results = {
            "demo_summary": {
                "insights_generated": len(insights),
                "processing_time_seconds": processing_time,
                "average_confidence": self.metrics["prediction_confidence_avg"],
                "models_used": list(set(insight.model_used for insight in insights.values()))
            },
            "insights": {k: asdict(v) for k, v in insights.items()},
            "analytics_status": await self.get_analytics_status()
        }
        
        self.logger.info(f"🎉 ML Analytics demo complete - {len(insights)} insights generated in {processing_time:.2f}s")
        
        return demo_results

async def main():
    """Main entry point for ML Predictive Analytics Engine"""
    print("🧠 Starting ML Predictive Analytics Engine - Phase 2")
    print("=" * 80)
    
    # Initialize analytics engine
    engine = MLPredictiveAnalyticsEngine()
    
    if not await engine.initialize():
        print("❌ Failed to initialize ML analytics engine")
        return
    
    # Run comprehensive demonstration
    demo_results = await engine.run_comprehensive_demo()
    
    print("\n" + "=" * 80)
    print("📋 ML Analytics Demo Results:")
    print(f"   Insights Generated: {demo_results['demo_summary']['insights_generated']}")
    print(f"   Processing Time: {demo_results['demo_summary']['processing_time_seconds']:.2f}s")
    print(f"   Average Confidence: {demo_results['demo_summary']['average_confidence']:.3f}")
    print(f"   Models Used: {', '.join(str(m) for m in demo_results['demo_summary']['models_used'])}")
    
    # Show insight summary
    print(f"\n📊 Generated Insights:")
    for insight_type, insight_data in demo_results["insights"].items():
        confidence = insight_data["confidence_score"]
        severity = insight_data["severity"]
        print(f"   • {insight_type.replace('_', ' ').title()}: {severity} ({confidence:.3f} confidence)")
    
    # Export results
    results_file = "ml_predictive_analytics_results.json"
    with open(results_file, "w") as f:
        json.dump(demo_results, f, indent=2, default=str)
    
    print(f"\n📄 Full results exported: {results_file}")
    print("\n✅ ML Predictive Analytics Engine Phase 2 demonstration complete!")

if __name__ == "__main__":
    # Install dependencies if needed
    try:
        import numpy as np
        import sklearn
    except ImportError as e:
        print(f"⚠️  Missing ML dependency: {e}")
        print("Installing required packages...")
        import subprocess
        packages = ["numpy", "scikit-learn", "pandas"]
        for package in packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ Installed {package}")
            except:
                print(f"❌ Failed to install {package} - continuing with fallbacks")
    
    asyncio.run(main())