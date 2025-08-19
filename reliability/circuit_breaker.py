"""
Circuit Breaker Pattern for Reliability
MANU Compliance: Reliability Requirements
"""
import asyncio
import time
from enum import Enum
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass
import threading


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Failing, rejecting requests
    HALF_OPEN = "half_open" # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5          # Failures before opening
    recovery_timeout: float = 60.0      # Seconds before trying half-open
    success_threshold: int = 3          # Successes needed to close from half-open
    timeout: float = 10.0               # Request timeout in seconds
    expected_exception: type = Exception # Exception type to count as failure


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance
    Prevents cascading failures by failing fast when service is down
    """
    
    def __init__(self, 
                 name: str,
                 config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker
        
        Args:
            name: Circuit breaker name
            config: Configuration parameters
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.lock = threading.RLock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit breaker is open
        """
        with self.lock:
            # Check if we should transition states
            self._update_state()
            
            # If open, reject immediately
            if self.state == CircuitBreakerState.OPEN:
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN"
                )
            
            # If half-open, only allow limited requests
            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.success_count >= 1:  # Only one test request at a time
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is HALF_OPEN, test in progress"
                    )
        
        # Execute the function with timeout
        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )
            
            # Record success
            self._record_success()
            return result
            
        except asyncio.TimeoutError:
            self._record_failure(asyncio.TimeoutError("Request timeout"))
            raise
        except self.config.expected_exception as e:
            self._record_failure(e)
            raise
        except Exception as e:
            # Unexpected exceptions don't count as failures
            raise e
    
    def _update_state(self):
        """Update circuit breaker state based on current conditions"""
        if self.state == CircuitBreakerState.OPEN:
            # Check if we should move to half-open
            if (self.last_failure_time and 
                time.time() - self.last_failure_time >= self.config.recovery_timeout):
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                print(f"Circuit breaker '{self.name}' transitioned to HALF_OPEN")
        
        elif self.state == CircuitBreakerState.HALF_OPEN:
            # Check if we should close (enough successes)
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                print(f"Circuit breaker '{self.name}' transitioned to CLOSED")
    
    def _record_success(self):
        """Record a successful call"""
        with self.lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = 0  # Reset failure count on success
    
    def _record_failure(self, exception: Exception):
        """Record a failed call"""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitBreakerState.CLOSED:
                # Check if we should open
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
                    print(f"Circuit breaker '{self.name}' transitioned to OPEN")
            
            elif self.state == CircuitBreakerState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self.state = CircuitBreakerState.OPEN
                self.success_count = 0
                print(f"Circuit breaker '{self.name}' transitioned back to OPEN")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time,
            'config': {
                'failure_threshold': self.config.failure_threshold,
                'recovery_timeout': self.config.recovery_timeout,
                'success_threshold': self.config.success_threshold,
                'timeout': self.config.timeout
            }
        }
    
    def reset(self):
        """Reset circuit breaker to closed state"""
        with self.lock:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            print(f"Circuit breaker '{self.name}' manually reset to CLOSED")


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers
    """
    
    def __init__(self):
        """Initialize circuit breaker registry"""
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
    
    def get_breaker(self, 
                   name: str, 
                   config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """
        Get or create a circuit breaker
        
        Args:
            name: Circuit breaker name
            config: Configuration (only used for new breakers)
            
        Returns:
            Circuit breaker instance
        """
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get states of all circuit breakers"""
        with self._lock:
            return {name: breaker.get_state() 
                   for name, breaker in self._breakers.items()}
    
    def reset_breaker(self, name: str) -> bool:
        """Reset a specific circuit breaker"""
        with self._lock:
            if name in self._breakers:
                self._breakers[name].reset()
                return True
            return False
    
    def reset_all(self):
        """Reset all circuit breakers"""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()


class TimeoutManager:
    """
    Timeout management for operations
    Provides configurable timeouts with fallbacks
    """
    
    def __init__(self):
        """Initialize timeout manager"""
        self.default_timeouts = {
            'http_request': 30.0,
            'database_query': 10.0,
            'audio_processing': 5.0,
            'midi_operation': 2.0,
            'file_operation': 15.0
        }
        self.operation_timeouts = {}
    
    def set_timeout(self, operation_type: str, timeout: float):
        """
        Set timeout for operation type
        
        Args:
            operation_type: Type of operation
            timeout: Timeout in seconds
        """
        self.operation_timeouts[operation_type] = timeout
    
    def get_timeout(self, operation_type: str) -> float:
        """
        Get timeout for operation type
        
        Args:
            operation_type: Type of operation
            
        Returns:
            Timeout in seconds
        """
        return (self.operation_timeouts.get(operation_type) or
                self.default_timeouts.get(operation_type, 30.0))
    
    async def with_timeout(self, 
                          operation_type: str,
                          coro,
                          fallback_value=None):
        """
        Execute coroutine with timeout
        
        Args:
            operation_type: Type of operation
            coro: Coroutine to execute
            fallback_value: Value to return on timeout
            
        Returns:
            Operation result or fallback value
        """
        timeout = self.get_timeout(operation_type)
        
        try:
            result = await asyncio.wait_for(coro, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            if fallback_value is not None:
                return fallback_value
            raise


# Global instances
circuit_breaker_registry = CircuitBreakerRegistry()
timeout_manager = TimeoutManager()

# Decorator for easy circuit breaker usage
def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """
    Decorator for circuit breaker protection
    
    Args:
        name: Circuit breaker name
        config: Optional configuration
    """
    def decorator(func):
        breaker = circuit_breaker_registry.get_breaker(name, config)
        
        async def wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)
        
        return wrapper
    return decorator


# Export reliability components
__all__ = [
    'CircuitBreaker',
    'CircuitBreakerConfig',
    'CircuitBreakerError',
    'CircuitBreakerRegistry',
    'TimeoutManager',
    'circuit_breaker_registry',
    'timeout_manager',
    'circuit_breaker'
]