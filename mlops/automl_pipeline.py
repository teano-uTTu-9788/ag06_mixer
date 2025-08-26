"""
Production AutoML Pipeline
Following Google AutoML, Microsoft Azure ML, AWS SageMaker Autopilot best practices
"""

import numpy as np
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import json
import hashlib
import threading
import time
import pickle
from abc import ABC, abstractmethod
from enum import Enum
import itertools
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskType(Enum):
    """ML task types"""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    ANOMALY_DETECTION = "anomaly_detection"


class ModelType(Enum):
    """Supported model types"""
    RANDOM_FOREST = "random_forest"
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM = "svm"
    NEURAL_NETWORK = "neural_network"
    LINEAR_REGRESSION = "linear_regression"


class OptimizationMetric(Enum):
    """Optimization metrics"""
    ACCURACY = "accuracy"
    F1_SCORE = "f1"
    ROC_AUC = "roc_auc"
    R2_SCORE = "r2"
    RMSE = "rmse"
    MAE = "mae"


@dataclass
class ModelConfig:
    """Model configuration for hyperparameter optimization"""
    model_type: ModelType
    hyperparameters: Dict[str, Any]
    preprocessing_steps: List[str] = field(default_factory=list)
    feature_selection: Optional[str] = None
    cross_validation_folds: int = 5


@dataclass
class ExperimentResult:
    """Result of a single AutoML experiment"""
    experiment_id: str
    model_config: ModelConfig
    task_type: TaskType
    metric_scores: Dict[str, float]
    training_time: float
    model_size_mb: float
    feature_importance: Dict[str, float] = field(default_factory=dict)
    cross_val_scores: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AutoMLJob:
    """AutoML job configuration"""
    job_id: str
    dataset_name: str
    task_type: TaskType
    target_column: str
    optimization_metric: OptimizationMetric
    max_runtime_hours: float = 2.0
    max_models: int = 50
    early_stopping_patience: int = 10
    cross_validation_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    include_ensemble: bool = True
    feature_selection: bool = True
    preprocessing: bool = True
    created_at: datetime = field(default_factory=datetime.now)


class HyperparameterOptimizer:
    """
    Hyperparameter optimization following Google AutoML patterns
    Implements Random Search and Grid Search strategies
    """
    
    def __init__(self, optimization_strategy: str = "random_search"):
        self.optimization_strategy = optimization_strategy
        self.search_spaces = self._define_search_spaces()
        
    def _define_search_spaces(self) -> Dict[ModelType, Dict[str, List]]:
        """Define hyperparameter search spaces for each model type"""
        return {
            ModelType.RANDOM_FOREST: {
                'n_estimators': [50, 100, 200, 300, 500],
                'max_depth': [None, 10, 20, 30, 50],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'max_features': ['auto', 'sqrt', 'log2', None]
            },
            ModelType.LOGISTIC_REGRESSION: {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga', 'lbfgs'],
                'max_iter': [100, 200, 500, 1000]
            },
            ModelType.SVM: {
                'C': [0.1, 1.0, 10.0, 100.0],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
                'degree': [2, 3, 4, 5]  # For poly kernel
            },
            ModelType.NEURAL_NETWORK: {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
                'activation': ['relu', 'tanh', 'logistic'],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'learning_rate': ['constant', 'adaptive'],
                'max_iter': [200, 500, 1000]
            },
            ModelType.LINEAR_REGRESSION: {
                'fit_intercept': [True, False],
                'normalize': [True, False]
            }
        }
        
    def generate_configurations(self, model_type: ModelType, 
                              max_configs: int = 20) -> List[Dict[str, Any]]:
        """Generate hyperparameter configurations"""
        search_space = self.search_spaces.get(model_type, {})
        
        if self.optimization_strategy == "grid_search":
            return self._grid_search_configs(search_space, max_configs)
        else:
            return self._random_search_configs(search_space, max_configs)
            
    def _grid_search_configs(self, search_space: Dict[str, List], 
                           max_configs: int) -> List[Dict[str, Any]]:
        """Generate grid search configurations"""
        if not search_space:
            return [{}]
            
        keys = list(search_space.keys())
        values = list(search_space.values())
        
        configs = []
        for combination in itertools.product(*values):
            if len(configs) >= max_configs:
                break
            config = dict(zip(keys, combination))
            configs.append(config)
            
        return configs[:max_configs]
        
    def _random_search_configs(self, search_space: Dict[str, List], 
                             max_configs: int) -> List[Dict[str, Any]]:
        """Generate random search configurations"""
        if not search_space:
            return [{}]
            
        configs = []
        np.random.seed(42)
        
        for _ in range(max_configs):
            config = {}
            for param, values in search_space.items():
                if isinstance(values, list):
                    config[param] = np.random.choice(values)
                else:
                    config[param] = values
            configs.append(config)
            
        return configs


class DataPreprocessor:
    """
    Automated data preprocessing following industry best practices
    """
    
    def __init__(self):
        self.encoders = {}
        self.scalers = {}
        self.feature_names = []
        self.preprocessing_steps = []
        
    def fit_transform(self, X: np.ndarray, y: np.ndarray = None, 
                     column_names: List[str] = None) -> np.ndarray:
        """Fit preprocessing pipeline and transform data"""
        self.feature_names = column_names or [f"feature_{i}" for i in range(X.shape[1])]
        X_processed = X.copy()
        
        # Detect data types and apply appropriate preprocessing
        for i, feature_name in enumerate(self.feature_names):
            feature_data = X_processed[:, i]
            
            # Handle categorical features (if they are strings)
            if self._is_categorical(feature_data):
                encoder = LabelEncoder()
                X_processed[:, i] = encoder.fit_transform(feature_data.astype(str))
                self.encoders[feature_name] = encoder
                self.preprocessing_steps.append(f"Label encoded {feature_name}")
                
        # Scale numerical features
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X_processed.astype(float))
        self.scalers['standard'] = scaler
        self.preprocessing_steps.append("Applied standard scaling")
        
        return X_processed
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using fitted preprocessors"""
        X_processed = X.copy()
        
        # Apply encoders
        for i, feature_name in enumerate(self.feature_names):
            if feature_name in self.encoders:
                encoder = self.encoders[feature_name]
                feature_data = X_processed[:, i]
                
                # Handle unseen labels
                try:
                    X_processed[:, i] = encoder.transform(feature_data.astype(str))
                except ValueError:
                    # Use most frequent class for unseen labels
                    most_frequent = encoder.classes_[0]
                    feature_data_fixed = np.where(
                        np.isin(feature_data.astype(str), encoder.classes_),
                        feature_data.astype(str),
                        most_frequent
                    )
                    X_processed[:, i] = encoder.transform(feature_data_fixed)
                    
        # Apply scaling
        if 'standard' in self.scalers:
            X_processed = self.scalers['standard'].transform(X_processed.astype(float))
            
        return X_processed
        
    def _is_categorical(self, feature_data: np.ndarray) -> bool:
        """Detect if feature is categorical"""
        try:
            # Try to convert to float
            float_data = feature_data.astype(float)
            # If successful and has many unique values, treat as numerical
            unique_ratio = len(np.unique(float_data)) / len(float_data)
            return unique_ratio < 0.05  # Less than 5% unique values
        except (ValueError, TypeError):
            return True  # String data is categorical


class ModelTrainer:
    """
    Model training with automated selection and optimization
    """
    
    def __init__(self, task_type: TaskType):
        self.task_type = task_type
        self.model_classes = self._get_model_classes()
        
    def _get_model_classes(self) -> Dict[ModelType, type]:
        """Get appropriate model classes for task type"""
        if self.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
            return {
                ModelType.RANDOM_FOREST: RandomForestClassifier,
                ModelType.LOGISTIC_REGRESSION: LogisticRegression,
                ModelType.SVM: SVC,
                ModelType.NEURAL_NETWORK: MLPClassifier
            }
        elif self.task_type == TaskType.REGRESSION:
            return {
                ModelType.RANDOM_FOREST: RandomForestRegressor,
                ModelType.LINEAR_REGRESSION: LinearRegression,
                ModelType.SVM: SVR,
                ModelType.NEURAL_NETWORK: MLPRegressor
            }
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
            
    def train_model(self, model_config: ModelConfig, X_train: np.ndarray, 
                   y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> ExperimentResult:
        """Train a single model with given configuration"""
        start_time = time.time()
        
        # Create model instance
        model_class = self.model_classes[model_config.model_type]
        
        # Handle hyperparameter compatibility
        filtered_params = self._filter_hyperparameters(
            model_config.model_type, model_config.hyperparameters
        )
        
        try:
            model = model_class(**filtered_params)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            metric_scores = self._calculate_metrics(y_val, y_pred)
            
            # Cross-validation scores
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=model_config.cross_validation_folds,
                scoring=self._get_scoring_metric()
            )
            
            # Feature importance (if available)
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                feature_importance = {
                    f"feature_{i}": importance 
                    for i, importance in enumerate(model.feature_importances_)
                }
            elif hasattr(model, 'coef_'):
                coef = model.coef_
                if coef.ndim > 1:
                    coef = np.mean(np.abs(coef), axis=0)
                feature_importance = {
                    f"feature_{i}": abs(coef[i]) 
                    for i in range(len(coef))
                }
                
            training_time = time.time() - start_time
            
            # Estimate model size
            model_size_mb = len(pickle.dumps(model)) / (1024 * 1024)
            
            # Generate experiment ID
            experiment_id = hashlib.md5(
                f"{model_config.model_type.value}_{time.time()}".encode()
            ).hexdigest()[:8]
            
            return ExperimentResult(
                experiment_id=experiment_id,
                model_config=model_config,
                task_type=self.task_type,
                metric_scores=metric_scores,
                training_time=training_time,
                model_size_mb=model_size_mb,
                feature_importance=feature_importance,
                cross_val_scores=cv_scores.tolist(),
                metadata={
                    'model_instance': model,
                    'validation_samples': len(X_val),
                    'training_samples': len(X_train)
                }
            )
            
        except Exception as e:
            logger.error(f"Model training failed for {model_config.model_type.value}: {e}")
            return ExperimentResult(
                experiment_id=f"failed_{time.time()}",
                model_config=model_config,
                task_type=self.task_type,
                metric_scores={'error': -1.0},
                training_time=time.time() - start_time,
                model_size_mb=0.0,
                metadata={'error': str(e)}
            )
            
    def _filter_hyperparameters(self, model_type: ModelType, 
                               hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Filter hyperparameters based on model compatibility"""
        model_class = self.model_classes[model_type]
        
        # Get valid parameters for the model
        try:
            # Create a dummy instance to get valid parameters
            dummy_model = model_class()
            valid_params = set(dummy_model.get_params().keys())
        except Exception:
            # If we can't create dummy instance, return all parameters
            return hyperparameters
            
        # Filter hyperparameters
        filtered = {k: v for k, v in hyperparameters.items() if k in valid_params}
        
        # Handle special cases for SVM
        if model_type == ModelType.SVM and 'kernel' in filtered:
            if filtered['kernel'] != 'poly' and 'degree' in filtered:
                del filtered['degree']  # degree only applies to poly kernel
                
        return filtered
        
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate appropriate metrics for task type"""
        metrics = {}
        
        try:
            if self.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
                metrics['accuracy'] = accuracy_score(y_true, y_pred)
                metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
                
            elif self.task_type == TaskType.REGRESSION:
                metrics['r2_score'] = r2_score(y_true, y_pred)
                metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
                metrics['mae'] = np.mean(np.abs(y_true - y_pred))
                
        except Exception as e:
            logger.warning(f"Metric calculation failed: {e}")
            metrics['error'] = -1.0
            
        return metrics
        
    def _get_scoring_metric(self) -> str:
        """Get appropriate scoring metric for cross-validation"""
        if self.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
            return 'accuracy'
        else:
            return 'r2'


class AutoMLPipeline:
    """
    Complete AutoML pipeline following Google AutoML and Azure ML best practices
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.hyperparameter_optimizer = HyperparameterOptimizer()
        self.preprocessor = DataPreprocessor()
        self.results: List[ExperimentResult] = []
        self.best_model = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.is_running = False
        self.current_job: Optional[AutoMLJob] = None
        
    async def run_automl_job(self, job: AutoMLJob, X: np.ndarray, y: np.ndarray,
                           feature_names: Optional[List[str]] = None) -> List[ExperimentResult]:
        """Run complete AutoML job"""
        self.current_job = job
        self.is_running = True
        
        logger.info(f"Starting AutoML job: {job.job_id}")
        logger.info(f"Task: {job.task_type.value}, Metric: {job.optimization_metric.value}")
        logger.info(f"Dataset shape: {X.shape}, Max runtime: {job.max_runtime_hours}h")
        
        start_time = time.time()
        
        try:
            # Preprocess data
            logger.info("Preprocessing data...")
            X_processed = self.preprocessor.fit_transform(X, y, feature_names)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_processed, y, 
                test_size=job.test_size,
                random_state=job.random_state,
                stratify=y if job.task_type in [TaskType.BINARY_CLASSIFICATION, 
                                               TaskType.MULTICLASS_CLASSIFICATION] else None
            )
            
            logger.info(f"Train set: {X_train.shape}, Validation set: {X_val.shape}")
            
            # Generate model configurations
            model_configs = self._generate_model_configurations(job)
            logger.info(f"Generated {len(model_configs)} model configurations")
            
            # Train models
            results = await self._train_models_parallel(
                model_configs, X_train, y_train, X_val, y_val, job
            )
            
            # Select best model
            self.best_model = self._select_best_model(results, job.optimization_metric)
            
            runtime_hours = (time.time() - start_time) / 3600
            logger.info(f"AutoML job completed in {runtime_hours:.2f} hours")
            logger.info(f"Best model: {self.best_model.model_config.model_type.value}")
            logger.info(f"Best score: {self._get_optimization_score(self.best_model, job.optimization_metric):.4f}")
            
            self.results = results
            return results
            
        except Exception as e:
            logger.error(f"AutoML job failed: {e}")
            raise
        finally:
            self.is_running = False
            
    def _generate_model_configurations(self, job: AutoMLJob) -> List[ModelConfig]:
        """Generate model configurations for the job"""
        configs = []
        
        # Determine which models to try based on task type
        if job.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
            model_types = [ModelType.RANDOM_FOREST, ModelType.LOGISTIC_REGRESSION, 
                          ModelType.SVM, ModelType.NEURAL_NETWORK]
        else:
            model_types = [ModelType.RANDOM_FOREST, ModelType.LINEAR_REGRESSION,
                          ModelType.SVM, ModelType.NEURAL_NETWORK]
            
        # Generate configurations for each model type
        configs_per_model = max(1, job.max_models // len(model_types))
        
        for model_type in model_types:
            hyperparams_list = self.hyperparameter_optimizer.generate_configurations(
                model_type, configs_per_model
            )
            
            for hyperparams in hyperparams_list:
                config = ModelConfig(
                    model_type=model_type,
                    hyperparameters=hyperparams,
                    cross_validation_folds=job.cross_validation_folds
                )
                configs.append(config)
                
        return configs[:job.max_models]
        
    async def _train_models_parallel(self, model_configs: List[ModelConfig],
                                   X_train: np.ndarray, y_train: np.ndarray,
                                   X_val: np.ndarray, y_val: np.ndarray,
                                   job: AutoMLJob) -> List[ExperimentResult]:
        """Train models in parallel"""
        trainer = ModelTrainer(job.task_type)
        results = []
        
        # Submit training jobs
        loop = asyncio.get_event_loop()
        futures = []
        
        for config in model_configs:
            future = loop.run_in_executor(
                self.executor,
                trainer.train_model,
                config, X_train, y_train, X_val, y_val
            )
            futures.append(future)
            
        # Collect results with timeout
        completed = 0
        timeout_seconds = job.max_runtime_hours * 3600
        start_time = time.time()
        
        for future in as_completed(futures, timeout=timeout_seconds):
            try:
                result = await future
                results.append(result)
                completed += 1
                
                # Log progress
                if completed % 5 == 0:
                    elapsed_hours = (time.time() - start_time) / 3600
                    logger.info(f"Completed {completed}/{len(model_configs)} models "
                              f"({elapsed_hours:.1f}h elapsed)")
                    
                # Early stopping if we have enough good results
                if (completed >= job.early_stopping_patience and 
                    self._has_convergence(results, job.optimization_metric)):
                    logger.info(f"Early stopping triggered after {completed} models")
                    break
                    
            except Exception as e:
                logger.warning(f"Model training failed: {e}")
                continue
                
            # Check runtime limit
            if (time.time() - start_time) > timeout_seconds:
                logger.warning("Runtime limit reached, stopping training")
                break
                
        # Cancel remaining futures
        for future in futures:
            if not future.done():
                future.cancel()
                
        return results
        
    def _select_best_model(self, results: List[ExperimentResult], 
                          metric: OptimizationMetric) -> Optional[ExperimentResult]:
        """Select best model based on optimization metric"""
        if not results:
            return None
            
        # Filter out failed results
        valid_results = [r for r in results if 'error' not in r.metric_scores]
        
        if not valid_results:
            return None
            
        # Sort by optimization metric
        metric_key = metric.value
        if metric_key == 'rmse':  # Lower is better for RMSE
            best_result = min(valid_results, 
                            key=lambda x: x.metric_scores.get(metric_key, float('inf')))
        else:  # Higher is better for accuracy, f1, r2
            best_result = max(valid_results,
                            key=lambda x: x.metric_scores.get(metric_key, -float('inf')))
            
        return best_result
        
    def _get_optimization_score(self, result: ExperimentResult, 
                              metric: OptimizationMetric) -> float:
        """Get optimization score from result"""
        return result.metric_scores.get(metric.value, 0.0)
        
    def _has_convergence(self, results: List[ExperimentResult], 
                        metric: OptimizationMetric, min_results: int = 10) -> bool:
        """Check if optimization has converged"""
        if len(results) < min_results:
            return False
            
        # Get recent scores
        recent_results = results[-min_results:]
        scores = [self._get_optimization_score(r, metric) for r in recent_results]
        
        # Check if improvement has plateaued
        if len(scores) >= 5:
            recent_scores = scores[-5:]
            score_range = max(recent_scores) - min(recent_scores)
            return score_range < 0.001  # Very small improvement
            
        return False
        
    def get_leaderboard(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get top performing models"""
        if not self.results:
            return []
            
        # Filter valid results
        valid_results = [r for r in self.results if 'error' not in r.metric_scores]
        
        # Sort by primary metric
        if self.current_job:
            metric = self.current_job.optimization_metric
            if metric.value == 'rmse':
                valid_results.sort(key=lambda x: x.metric_scores.get(metric.value, float('inf')))
            else:
                valid_results.sort(key=lambda x: x.metric_scores.get(metric.value, -float('inf')), reverse=True)
                
        # Format leaderboard
        leaderboard = []
        for i, result in enumerate(valid_results[:top_k]):
            leaderboard.append({
                'rank': i + 1,
                'experiment_id': result.experiment_id,
                'model_type': result.model_config.model_type.value,
                'scores': result.metric_scores,
                'training_time': f"{result.training_time:.2f}s",
                'model_size_mb': f"{result.model_size_mb:.2f}MB",
                'cv_score_mean': f"{np.mean(result.cross_val_scores):.4f}",
                'cv_score_std': f"{np.std(result.cross_val_scores):.4f}"
            })
            
        return leaderboard


# Example usage and demonstration
async def demo_automl_pipeline():
    """Demonstrate the AutoML pipeline"""
    print("🤖 Production AutoML Pipeline Demo")
    print("Following Google AutoML, Azure ML, AWS SageMaker best practices\n")
    
    # Generate synthetic dataset for classification
    print("📊 Generating synthetic dataset...")
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Create features with some noise and correlation
    X = np.random.randn(n_samples, n_features)
    
    # Create meaningful target based on some features
    target_weights = np.random.randn(n_features)
    target_weights[:5] *= 3  # Make first 5 features more important
    
    # Binary classification target
    linear_combination = X @ target_weights
    y = (linear_combination > np.median(linear_combination)).astype(int)
    
    # Add some categorical features (simulated)
    X[:, -2] = np.random.choice(['A', 'B', 'C'], n_samples)
    X[:, -1] = np.random.choice(['Type1', 'Type2'], n_samples)
    
    feature_names = [f"numerical_feature_{i}" for i in range(n_features-2)] + \
                   ['categorical_A', 'categorical_B']
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target distribution: {np.bincount(y)}")
    
    # Create AutoML job
    print("\n🎯 Creating AutoML job...")
    
    job = AutoMLJob(
        job_id="demo_classification_job",
        dataset_name="synthetic_dataset",
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="target",
        optimization_metric=OptimizationMetric.ACCURACY,
        max_runtime_hours=0.1,  # 6 minutes for demo
        max_models=20,
        early_stopping_patience=8,
        cross_validation_folds=3  # Reduced for demo
    )
    
    print(f"Job Configuration:")
    print(f"  Task Type: {job.task_type.value}")
    print(f"  Optimization Metric: {job.optimization_metric.value}")
    print(f"  Max Models: {job.max_models}")
    print(f"  Max Runtime: {job.max_runtime_hours} hours")
    
    # Initialize and run AutoML pipeline
    print("\n🚀 Starting AutoML pipeline...")
    
    pipeline = AutoMLPipeline()
    
    # Run the job
    results = await pipeline.run_automl_job(job, X, y, feature_names)
    
    print(f"\n✅ AutoML job completed!")
    print(f"Trained {len(results)} models successfully")
    
    # Show results
    if pipeline.best_model:
        print(f"\n🏆 Best Model Results:")
        best = pipeline.best_model
        print(f"  Model Type: {best.model_config.model_type.value}")
        print(f"  Hyperparameters: {best.model_config.hyperparameters}")
        print(f"  Performance:")
        for metric, score in best.metric_scores.items():
            print(f"    {metric.upper()}: {score:.4f}")
        print(f"  Cross-validation: {np.mean(best.cross_val_scores):.4f} ± {np.std(best.cross_val_scores):.4f}")
        print(f"  Training Time: {best.training_time:.2f} seconds")
        print(f"  Model Size: {best.model_size_mb:.2f} MB")
        
        # Feature importance
        if best.feature_importance:
            print(f"\n📊 Top Feature Importance:")
            sorted_features = sorted(best.feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True)
            for i, (feature, importance) in enumerate(sorted_features[:5]):
                feature_name = feature_names[int(feature.split('_')[-1])] if feature.startswith('feature_') else feature
                print(f"    {i+1}. {feature_name}: {importance:.4f}")
    
    # Show leaderboard
    print(f"\n🏅 Model Leaderboard (Top 5):")
    leaderboard = pipeline.get_leaderboard(top_k=5)
    
    for entry in leaderboard:
        print(f"  {entry['rank']}. {entry['model_type'].title()}")
        print(f"     Accuracy: {entry['scores'].get('accuracy', 0):.4f}")
        print(f"     F1 Score: {entry['scores'].get('f1_score', 0):.4f}")
        print(f"     Training Time: {entry['training_time']}")
        print(f"     CV Score: {entry['cv_score_mean']} ± {entry['cv_score_std']}")
        print()
    
    # Show preprocessing steps
    print(f"🔧 Data Preprocessing Steps Applied:")
    for step in pipeline.preprocessor.preprocessing_steps:
        print(f"  - {step}")
    
    # Regression example
    print(f"\n" + "="*60)
    print("🔢 Regression Task Example")
    print("="*60)
    
    # Create regression dataset
    X_reg = np.random.randn(800, 15)
    y_reg = 3 * X_reg[:, 0] + 2 * X_reg[:, 1] - X_reg[:, 2] + 0.5 * np.random.randn(800)
    
    regression_job = AutoMLJob(
        job_id="demo_regression_job",
        dataset_name="synthetic_regression",
        task_type=TaskType.REGRESSION,
        target_column="target",
        optimization_metric=OptimizationMetric.R2_SCORE,
        max_runtime_hours=0.05,  # 3 minutes
        max_models=15
    )
    
    # Run regression AutoML
    reg_results = await pipeline.run_automl_job(regression_job, X_reg, y_reg)
    
    if pipeline.best_model:
        best_reg = pipeline.best_model
        print(f"\n🏆 Best Regression Model:")
        print(f"  Model Type: {best_reg.model_config.model_type.value}")
        print(f"  R² Score: {best_reg.metric_scores.get('r2_score', 0):.4f}")
        print(f"  RMSE: {best_reg.metric_scores.get('rmse', 0):.4f}")
        print(f"  MAE: {best_reg.metric_scores.get('mae', 0):.4f}")
        print(f"  Training Time: {best_reg.training_time:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(demo_automl_pipeline())