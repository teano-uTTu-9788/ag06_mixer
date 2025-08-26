"""
Machine Learning package for AG06 Workflow System
Provides ML-driven optimization and A/B testing
"""

from .active_optimizer import (
    ActiveOptimizer,
    get_optimizer,
    PerformanceMetric
)

__all__ = [
    'ActiveOptimizer',
    'get_optimizer',
    'PerformanceMetric'
]