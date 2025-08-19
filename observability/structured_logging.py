"""
Structured Logging System for Production
MANU Compliance: Observability Requirements
"""
import json
import logging
import logging.config
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import contextmanager
import threading
import uuid


class StructuredLogger:
    """
    Structured logging with JSON format
    Provides consistent logging across the AG-06 mixer system
    """
    
    def __init__(self, name: str = "ag06_mixer"):
        """
        Initialize structured logger
        
        Args:
            name: Logger name
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.correlation_id = threading.local()
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup structured logging configuration"""
        # Configure structured logging
        logging_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'structured': {
                    'format': '%(message)s'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'structured',
                    'stream': sys.stdout
                },
                'file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': 'ag06_mixer.log',
                    'formatter': 'structured',
                    'maxBytes': 10485760,  # 10MB
                    'backupCount': 5
                }
            },
            'loggers': {
                self.name: {
                    'level': 'INFO',
                    'handlers': ['console', 'file'],
                    'propagate': False
                }
            }
        }
        
        logging.config.dictConfig(logging_config)
    
    def _create_log_record(self, 
                          level: str,
                          message: str,
                          **kwargs) -> Dict[str, Any]:
        """
        Create structured log record
        
        Args:
            level: Log level
            message: Log message
            **kwargs: Additional fields
            
        Returns:
            Structured log record
        """
        record = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': level.upper(),
            'logger': self.name,
            'message': message,
            'correlation_id': getattr(self.correlation_id, 'value', None),
            'thread_id': threading.current_thread().ident,
            'thread_name': threading.current_thread().name
        }
        
        # Add extra fields
        record.update(kwargs)
        
        return record
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        record = self._create_log_record('debug', message, **kwargs)
        self.logger.debug(json.dumps(record))
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        record = self._create_log_record('info', message, **kwargs)
        self.logger.info(json.dumps(record))
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        record = self._create_log_record('warning', message, **kwargs)
        self.logger.warning(json.dumps(record))
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log error message"""
        if error:
            kwargs.update({
                'error_type': type(error).__name__,
                'error_message': str(error),
                'stack_trace': str(error.__traceback__) if error.__traceback__ else None
            })
        
        record = self._create_log_record('error', message, **kwargs)
        self.logger.error(json.dumps(record))
    
    def critical(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log critical message"""
        if error:
            kwargs.update({
                'error_type': type(error).__name__,
                'error_message': str(error),
                'stack_trace': str(error.__traceback__) if error.__traceback__ else None
            })
        
        record = self._create_log_record('critical', message, **kwargs)
        self.logger.critical(json.dumps(record))
    
    @contextmanager
    def correlation_context(self, correlation_id: Optional[str] = None):
        """
        Context manager for correlation ID
        
        Args:
            correlation_id: Optional correlation ID (generates if not provided)
        """
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())
        
        old_id = getattr(self.correlation_id, 'value', None)
        self.correlation_id.value = correlation_id
        
        try:
            yield correlation_id
        finally:
            if old_id is not None:
                self.correlation_id.value = old_id
            else:
                delattr(self.correlation_id, 'value')


class AuditLogger:
    """
    Audit logging for security events
    Tracks security-relevant actions and events
    """
    
    def __init__(self):
        """Initialize audit logger"""
        self.logger = StructuredLogger("ag06_audit")
    
    def log_authentication(self, 
                          user_id: str, 
                          success: bool, 
                          ip_address: Optional[str] = None,
                          user_agent: Optional[str] = None):
        """
        Log authentication attempt
        
        Args:
            user_id: User identifier
            success: Whether authentication succeeded
            ip_address: Client IP address
            user_agent: Client user agent
        """
        self.logger.info(
            f"Authentication {'successful' if success else 'failed'} for user {user_id}",
            event_type="authentication",
            user_id=user_id,
            success=success,
            ip_address=ip_address,
            user_agent=user_agent,
            severity="HIGH" if not success else "INFO"
        )
    
    def log_authorization(self, 
                         user_id: str, 
                         resource: str, 
                         action: str, 
                         granted: bool):
        """
        Log authorization decision
        
        Args:
            user_id: User identifier
            resource: Resource being accessed
            action: Action being performed
            granted: Whether access was granted
        """
        self.logger.info(
            f"Access {'granted' if granted else 'denied'} for user {user_id} to {resource}:{action}",
            event_type="authorization",
            user_id=user_id,
            resource=resource,
            action=action,
            granted=granted,
            severity="HIGH" if not granted else "INFO"
        )
    
    def log_data_access(self, 
                       user_id: str, 
                       data_type: str, 
                       operation: str,
                       record_count: Optional[int] = None):
        """
        Log data access
        
        Args:
            user_id: User identifier
            data_type: Type of data accessed
            operation: Operation performed (read, write, delete)
            record_count: Number of records affected
        """
        self.logger.info(
            f"Data {operation} by user {user_id} on {data_type}",
            event_type="data_access",
            user_id=user_id,
            data_type=data_type,
            operation=operation,
            record_count=record_count,
            severity="INFO"
        )
    
    def log_security_event(self, 
                          event_type: str, 
                          description: str, 
                          severity: str = "HIGH",
                          **kwargs):
        """
        Log security event
        
        Args:
            event_type: Type of security event
            description: Event description
            severity: Event severity
            **kwargs: Additional fields
        """
        self.logger.critical(
            description,
            event_type=event_type,
            security_event=True,
            severity=severity,
            **kwargs
        )


class PerformanceLogger:
    """
    Performance logging for monitoring and optimization
    Tracks performance metrics and bottlenecks
    """
    
    def __init__(self):
        """Initialize performance logger"""
        self.logger = StructuredLogger("ag06_performance")
    
    def log_request_timing(self, 
                          endpoint: str, 
                          method: str, 
                          duration_ms: float,
                          status_code: Optional[int] = None,
                          user_id: Optional[str] = None):
        """
        Log request timing
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            duration_ms: Request duration in milliseconds
            status_code: HTTP status code
            user_id: User identifier
        """
        severity = "WARNING" if duration_ms > 1000 else "INFO"
        
        self.logger.info(
            f"{method} {endpoint} completed in {duration_ms:.2f}ms",
            metric_type="request_timing",
            endpoint=endpoint,
            method=method,
            duration_ms=duration_ms,
            status_code=status_code,
            user_id=user_id,
            severity=severity
        )
    
    def log_audio_processing(self, 
                           operation: str, 
                           samples: int, 
                           duration_ms: float,
                           latency_ms: Optional[float] = None):
        """
        Log audio processing metrics
        
        Args:
            operation: Audio operation type
            samples: Number of samples processed
            duration_ms: Processing duration
            latency_ms: Processing latency
        """
        throughput = samples / (duration_ms / 1000) if duration_ms > 0 else 0
        
        self.logger.info(
            f"Audio {operation}: {samples} samples in {duration_ms:.2f}ms",
            metric_type="audio_processing",
            operation=operation,
            samples=samples,
            duration_ms=duration_ms,
            latency_ms=latency_ms,
            throughput_samples_per_sec=throughput
        )
    
    def log_system_metrics(self, 
                          cpu_percent: float, 
                          memory_percent: float,
                          disk_percent: Optional[float] = None,
                          active_connections: Optional[int] = None):
        """
        Log system resource metrics
        
        Args:
            cpu_percent: CPU usage percentage
            memory_percent: Memory usage percentage
            disk_percent: Disk usage percentage
            active_connections: Number of active connections
        """
        severity = "WARNING" if cpu_percent > 80 or memory_percent > 80 else "INFO"
        
        self.logger.info(
            f"System metrics: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%",
            metric_type="system_metrics",
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_percent=disk_percent,
            active_connections=active_connections,
            severity=severity
        )


# Global logger instances
structured_logger = StructuredLogger()
audit_logger = AuditLogger()
performance_logger = PerformanceLogger()

# Export logging components
__all__ = [
    'StructuredLogger',
    'AuditLogger', 
    'PerformanceLogger',
    'structured_logger',
    'audit_logger',
    'performance_logger'
]