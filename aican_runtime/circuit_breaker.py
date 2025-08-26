#!/usr/bin/env python3
"""
Circuit Breaker Implementation for Fault Tolerance
Prevents cascade failures in distributed systems
"""

import asyncio
import time
import threading
from typing import Any, Callable, Optional
from enum import Enum


class CircuitBreakerState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker rejects a call"""
    pass


class CircuitBreaker:
    """Thread-safe circuit breaker for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60.0):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            reset_timeout: Time in seconds before attempting to reset
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()
    
    @property
    def state(self) -> str:
        """Get current circuit breaker state as string"""
        return self._state.value
    
    @property
    def failure_count(self) -> int:
        """Get current failure count"""
        return self._failure_count
    
    @property
    def last_failure_time(self) -> Optional[float]:
        """Get timestamp of last failure"""
        return self._last_failure_time
    
    def record_success(self):
        """Record a successful operation"""
        with self._lock:
            self._failure_count = 0
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.CLOSED
    
    def record_failure(self):
        """Record a failed operation"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitBreakerState.OPEN
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt to reset"""
        if self._state != CircuitBreakerState.OPEN:
            return False
        
        if self._last_failure_time is None:
            return False
        
        return (time.time() - self._last_failure_time) >= self.reset_timeout
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        
        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
        
        Returns:
            Function result
        
        Raises:
            CircuitBreakerError: If circuit is open
        """
        with self._lock:
            if self._state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitBreakerState.HALF_OPEN
                else:
                    raise CircuitBreakerError("Circuit breaker is OPEN")
        
        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Record success
            self.record_success()
            return result
            
        except Exception as e:
            # Record failure
            self.record_failure()
            raise e
    
    def reset(self):
        """Manually reset the circuit breaker"""
        with self._lock:
            self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None


# Factory function for easy usage
def create_circuit_breaker(failure_threshold: int = 5, 
                          reset_timeout: float = 60.0) -> CircuitBreaker:
    """Create a new circuit breaker instance"""
    return CircuitBreaker(failure_threshold, reset_timeout)