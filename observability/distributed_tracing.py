"""
Distributed Tracing System for Production
MANU Compliance: Observability Requirements
"""
import time
import uuid
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
from contextlib import contextmanager
import json


@dataclass
class Span:
    """
    Represents a single span in distributed tracing
    """
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    operation_name: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"  # ok, error, timeout
    
    def finish(self):
        """Mark span as finished"""
        if self.end_time is None:
            self.end_time = time.time()
            self.duration_ms = (self.end_time - self.start_time) * 1000
    
    def set_tag(self, key: str, value: Any):
        """Set a tag on the span"""
        self.tags[key] = value
    
    def log_event(self, event: str, **fields):
        """Log an event within the span"""
        log_entry = {
            'timestamp': time.time(),
            'event': event,
            **fields
        }
        self.logs.append(log_entry)
    
    def set_error(self, error: Exception):
        """Mark span as having an error"""
        self.status = "error"
        self.set_tag('error', True)
        self.set_tag('error.type', type(error).__name__)
        self.set_tag('error.message', str(error))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary"""
        return {
            'trace_id': self.trace_id,
            'span_id': self.span_id,
            'parent_span_id': self.parent_span_id,
            'operation_name': self.operation_name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_ms': self.duration_ms,
            'tags': self.tags,
            'logs': self.logs,
            'status': self.status
        }


class Tracer:
    """
    Distributed tracing implementation
    Tracks request flows across system components
    """
    
    def __init__(self, service_name: str = "ag06_mixer"):
        """
        Initialize tracer
        
        Args:
            service_name: Name of the service
        """
        self.service_name = service_name
        self.active_spans = threading.local()
        self.finished_spans = []
        self._lock = threading.Lock()
    
    def start_span(self, 
                   operation_name: str,
                   parent_span: Optional[Span] = None,
                   tags: Optional[Dict[str, Any]] = None) -> Span:
        """
        Start a new span
        
        Args:
            operation_name: Name of the operation
            parent_span: Parent span (if any)
            tags: Initial tags for the span
            
        Returns:
            New span
        """
        # Get trace ID and parent span
        if parent_span:
            trace_id = parent_span.trace_id
            parent_span_id = parent_span.span_id
        elif hasattr(self.active_spans, 'current_span') and self.active_spans.current_span:
            current = self.active_spans.current_span
            trace_id = current.trace_id
            parent_span_id = current.span_id
        else:
            trace_id = str(uuid.uuid4())
            parent_span_id = None
        
        # Create new span
        span = Span(
            trace_id=trace_id,
            span_id=str(uuid.uuid4()),
            parent_span_id=parent_span_id,
            operation_name=operation_name
        )
        
        # Set service tag
        span.set_tag('service.name', self.service_name)
        span.set_tag('span.kind', 'internal')
        
        # Add provided tags
        if tags:
            for key, value in tags.items():
                span.set_tag(key, value)
        
        return span
    
    def finish_span(self, span: Span):
        """
        Finish a span and store it
        
        Args:
            span: Span to finish
        """
        span.finish()
        
        with self._lock:
            self.finished_spans.append(span)
            
            # Keep only recent spans in memory (last 1000)
            if len(self.finished_spans) > 1000:
                self.finished_spans = self.finished_spans[-1000:]
    
    @contextmanager
    def span(self, operation_name: str, **tags):
        """
        Context manager for automatic span lifecycle
        
        Args:
            operation_name: Operation name
            **tags: Tags to set on span
        """
        span = self.start_span(operation_name, tags=tags)
        
        # Set as active span
        old_span = getattr(self.active_spans, 'current_span', None)
        self.active_spans.current_span = span
        
        try:
            yield span
        except Exception as e:
            span.set_error(e)
            raise
        finally:
            self.finish_span(span)
            self.active_spans.current_span = old_span
    
    def get_current_span(self) -> Optional[Span]:
        """Get currently active span"""
        return getattr(self.active_spans, 'current_span', None)
    
    def inject_trace_context(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Inject trace context into headers
        
        Args:
            headers: Headers dictionary to inject into
            
        Returns:
            Updated headers with trace context
        """
        current_span = self.get_current_span()
        if current_span:
            headers['X-Trace-Id'] = current_span.trace_id
            headers['X-Span-Id'] = current_span.span_id
        
        return headers
    
    def extract_trace_context(self, headers: Dict[str, str]) -> Optional[Span]:
        """
        Extract trace context from headers
        
        Args:
            headers: Headers to extract from
            
        Returns:
            Parent span if trace context found
        """
        trace_id = headers.get('X-Trace-Id')
        span_id = headers.get('X-Span-Id')
        
        if trace_id and span_id:
            # Create a parent span from extracted context
            parent_span = Span(
                trace_id=trace_id,
                span_id=span_id,
                operation_name="extracted_parent"
            )
            return parent_span
        
        return None
    
    def get_traces(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent traces
        
        Args:
            limit: Maximum number of traces to return
            
        Returns:
            List of trace dictionaries
        """
        with self._lock:
            # Group spans by trace ID
            traces = {}
            for span in self.finished_spans[-limit:]:
                trace_id = span.trace_id
                if trace_id not in traces:
                    traces[trace_id] = []
                traces[trace_id].append(span.to_dict())
            
            return list(traces.values())


class TracingMiddleware:
    """
    Middleware for automatic request tracing
    """
    
    def __init__(self, tracer: Tracer):
        """
        Initialize tracing middleware
        
        Args:
            tracer: Tracer instance
        """
        self.tracer = tracer
    
    async def trace_request(self, 
                           operation_name: str,
                           func,
                           *args,
                           **kwargs):
        """
        Trace a function call
        
        Args:
            operation_name: Name of the operation
            func: Function to trace
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        with self.tracer.span(operation_name) as span:
            # Add request details to span
            span.set_tag('operation.name', operation_name)
            span.set_tag('component', 'ag06_mixer')
            
            # Extract correlation ID if available
            correlation_id = kwargs.get('correlation_id')
            if correlation_id:
                span.set_tag('correlation.id', correlation_id)
            
            try:
                result = await func(*args, **kwargs)
                span.set_tag('success', True)
                return result
            except Exception as e:
                span.set_error(e)
                span.set_tag('success', False)
                raise


class DistributedTracingCollector:
    """
    Collects and exports tracing data
    """
    
    def __init__(self, tracer: Tracer):
        """
        Initialize tracing collector
        
        Args:
            tracer: Tracer instance
        """
        self.tracer = tracer
        self.export_queue = []
    
    def collect_spans(self) -> List[Span]:
        """
        Collect finished spans for export
        
        Returns:
            List of spans ready for export
        """
        with self.tracer._lock:
            spans = self.tracer.finished_spans.copy()
            self.tracer.finished_spans.clear()
            return spans
    
    def export_spans(self, spans: List[Span]):
        """
        Export spans to external system
        
        Args:
            spans: Spans to export
        """
        # In production, export to:
        # - Jaeger
        # - Zipkin
        # - AWS X-Ray
        # - Google Cloud Trace
        
        # For now, write to file
        export_data = [span.to_dict() for span in spans]
        
        with open('traces.jsonl', 'a') as f:
            for span_data in export_data:
                f.write(json.dumps(span_data) + '\n')
    
    def flush(self):
        """Flush all pending spans"""
        spans = self.collect_spans()
        if spans:
            self.export_spans(spans)


# Global tracer instance
tracer = Tracer()
tracing_middleware = TracingMiddleware(tracer)
tracing_collector = DistributedTracingCollector(tracer)

# Export tracing components
__all__ = [
    'Span',
    'Tracer', 
    'TracingMiddleware',
    'DistributedTracingCollector',
    'tracer',
    'tracing_middleware',
    'tracing_collector'
]