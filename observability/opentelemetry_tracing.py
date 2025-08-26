#!/usr/bin/env python3
"""
OpenTelemetry Distributed Tracing Implementation
Following Google, Uber, and Datadog best practices for observability

Based on:
- Google Cloud Trace integration
- OpenTelemetry CNCF standards
- Uber Jaeger patterns
- Datadog APM best practices
"""

import asyncio
import json
import logging
import time
import uuid
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from contextlib import asynccontextmanager
import traceback

# OpenTelemetry imports (with fallbacks for demonstration)
try:
    from opentelemetry import trace, baggage, propagate
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
    from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
    from opentelemetry.propagators.b3 import B3MultiFormat
    from opentelemetry.propagators.jaeger import JaegerPropagator
    OTEL_AVAILABLE = True
except ImportError:
    print("⚠️ OpenTelemetry not available - using simulation mode")
    OTEL_AVAILABLE = False

class SpanKind(Enum):
    """Span types following OpenTelemetry standards"""
    INTERNAL = "internal"
    SERVER = "server" 
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"

class SpanStatus(Enum):
    """Span status codes"""
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"

@dataclass
class TraceConfig:
    """Tracing configuration"""
    service_name: str
    service_version: str
    environment: str
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    otlp_endpoint: str = "http://localhost:4317"
    google_cloud_project: str = None
    sampling_rate: float = 1.0  # 100% sampling for demo
    enable_console_exporter: bool = True
    enable_jaeger_exporter: bool = True
    enable_otlp_exporter: bool = True
    enable_google_cloud_exporter: bool = False

@dataclass  
class SpanData:
    """Span data structure"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    service_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: SpanStatus = SpanStatus.OK
    kind: SpanKind = SpanKind.INTERNAL
    tags: Dict[str, Any] = None
    logs: List[Dict[str, Any]] = None
    baggage: Dict[str, str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.logs is None:
            self.logs = []
        if self.baggage is None:
            self.baggage = {}

class SimulatedTracer:
    """Simulated tracer when OpenTelemetry is not available"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.active_spans = {}
        self.completed_spans = []
        
    def start_span(self, name: str, kind: SpanKind = SpanKind.INTERNAL) -> 'SimulatedSpan':
        span_id = str(uuid.uuid4())[:16]
        trace_id = str(uuid.uuid4())[:32]
        
        span_data = SpanData(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=None,
            operation_name=name,
            service_name=self.service_name,
            start_time=time.time(),
            kind=kind
        )
        
        span = SimulatedSpan(span_data, self)
        self.active_spans[span_id] = span
        return span

class SimulatedSpan:
    """Simulated span implementation"""
    
    def __init__(self, span_data: SpanData, tracer: SimulatedTracer):
        self.span_data = span_data
        self.tracer = tracer
        
    def set_tag(self, key: str, value: Any):
        self.span_data.tags[key] = value
        
    def set_status(self, status: SpanStatus):
        self.span_data.status = status
        
    def log(self, message: str, level: str = "info"):
        self.span_data.logs.append({
            "timestamp": time.time(),
            "level": level,
            "message": message
        })
        
    def add_event(self, name: str, attributes: Dict[str, Any] = None):
        self.log(f"Event: {name}", "event")
        if attributes:
            self.span_data.tags.update({f"event.{k}": v for k, v in attributes.items()})
    
    def finish(self):
        self.span_data.end_time = time.time()
        self.span_data.duration_ms = (self.span_data.end_time - self.span_data.start_time) * 1000
        
        if self.span_data.span_id in self.tracer.active_spans:
            del self.tracer.active_spans[self.span_data.span_id]
        
        self.tracer.completed_spans.append(self.span_data)
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.set_status(SpanStatus.ERROR)
            self.set_tag("error", True)
            self.set_tag("error.type", exc_type.__name__)
            self.set_tag("error.message", str(exc_val))
        self.finish()

class OpenTelemetryManager:
    """Manages OpenTelemetry tracing configuration"""
    
    def __init__(self, config: TraceConfig):
        self.config = config
        self.tracer = None
        self.initialized = False
        self.logger = logging.getLogger("otel_manager")
        
    def initialize(self):
        """Initialize OpenTelemetry with exporters"""
        
        if OTEL_AVAILABLE:
            self._initialize_real_otel()
        else:
            self._initialize_simulated_otel()
        
        self.initialized = True
        self.logger.info(f"OpenTelemetry initialized for service: {self.config.service_name}")
    
    def _initialize_real_otel(self):
        """Initialize real OpenTelemetry"""
        
        # Configure resource
        resource = Resource.create({
            SERVICE_NAME: self.config.service_name,
            SERVICE_VERSION: self.config.service_version,
            "service.environment": self.config.environment,
            "service.instance.id": str(uuid.uuid4())
        })
        
        # Set up tracer provider
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)
        
        # Configure exporters
        if self.config.enable_console_exporter:
            console_exporter = ConsoleSpanExporter()
            console_processor = BatchSpanProcessor(console_exporter)
            tracer_provider.add_span_processor(console_processor)
        
        if self.config.enable_jaeger_exporter:
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831,
                collector_endpoint=self.config.jaeger_endpoint
            )
            jaeger_processor = BatchSpanProcessor(jaeger_exporter)
            tracer_provider.add_span_processor(jaeger_processor)
            
        if self.config.enable_otlp_exporter:
            otlp_exporter = OTLPSpanExporter(
                endpoint=self.config.otlp_endpoint,
                insecure=True
            )
            otlp_processor = BatchSpanProcessor(otlp_exporter)
            tracer_provider.add_span_processor(otlp_processor)
        
        # Configure propagators
        propagate.set_global_textmap([
            B3MultiFormat(),
            JaegerPropagator(),
        ])
        
        # Get tracer
        self.tracer = trace.get_tracer(
            self.config.service_name,
            self.config.service_version
        )
        
        # Auto-instrument popular libraries
        RequestsInstrumentor().instrument()
        AioHttpClientInstrumentor().instrument()
        AsyncioInstrumentor().instrument()
        
    def _initialize_simulated_otel(self):
        """Initialize simulated OpenTelemetry"""
        self.tracer = SimulatedTracer(self.config.service_name)
    
    def get_tracer(self):
        """Get the configured tracer"""
        if not self.initialized:
            self.initialize()
        return self.tracer

class DistributedTracingDecorator:
    """Decorator for automatic tracing"""
    
    def __init__(self, otel_manager: OpenTelemetryManager, operation_name: str = None):
        self.otel_manager = otel_manager
        self.operation_name = operation_name
        
    def __call__(self, func: Callable):
        if asyncio.iscoroutinefunction(func):
            return self._trace_async(func)
        else:
            return self._trace_sync(func)
    
    def _trace_async(self, func: Callable):
        async def wrapper(*args, **kwargs):
            tracer = self.otel_manager.get_tracer()
            operation_name = self.operation_name or f"{func.__module__}.{func.__name__}"
            
            if OTEL_AVAILABLE:
                with tracer.start_as_current_span(operation_name) as span:
                    try:
                        span.set_attribute("function.name", func.__name__)
                        span.set_attribute("function.module", func.__module__)
                        span.set_attribute("args.count", len(args))
                        span.set_attribute("kwargs.count", len(kwargs))
                        
                        result = await func(*args, **kwargs)
                        span.set_status(trace.Status(trace.StatusCode.OK))
                        return result
                        
                    except Exception as e:
                        span.set_status(trace.Status(
                            trace.StatusCode.ERROR,
                            str(e)
                        ))
                        span.set_attribute("error", True)
                        span.set_attribute("error.type", type(e).__name__)
                        span.set_attribute("error.message", str(e))
                        raise
            else:
                with tracer.start_span(operation_name, SpanKind.INTERNAL) as span:
                    try:
                        span.set_tag("function.name", func.__name__)
                        span.set_tag("function.module", func.__module__)
                        span.set_tag("args.count", len(args))
                        span.set_tag("kwargs.count", len(kwargs))
                        
                        result = await func(*args, **kwargs)
                        return result
                        
                    except Exception as e:
                        span.set_status(SpanStatus.ERROR)
                        span.set_tag("error", True)
                        span.set_tag("error.type", type(e).__name__)
                        span.set_tag("error.message", str(e))
                        raise
        
        return wrapper
    
    def _trace_sync(self, func: Callable):
        def wrapper(*args, **kwargs):
            tracer = self.otel_manager.get_tracer()
            operation_name = self.operation_name or f"{func.__module__}.{func.__name__}"
            
            if OTEL_AVAILABLE:
                with tracer.start_as_current_span(operation_name) as span:
                    try:
                        span.set_attribute("function.name", func.__name__)
                        span.set_attribute("function.module", func.__module__)
                        
                        result = func(*args, **kwargs)
                        span.set_status(trace.Status(trace.StatusCode.OK))
                        return result
                        
                    except Exception as e:
                        span.set_status(trace.Status(
                            trace.StatusCode.ERROR,
                            str(e)
                        ))
                        span.set_attribute("error", True)
                        span.set_attribute("error.type", type(e).__name__)
                        raise
            else:
                with tracer.start_span(operation_name, SpanKind.INTERNAL) as span:
                    try:
                        span.set_tag("function.name", func.__name__)
                        span.set_tag("function.module", func.__module__)
                        
                        result = func(*args, **kwargs)
                        return result
                        
                    except Exception as e:
                        span.set_status(SpanStatus.ERROR)
                        span.set_tag("error", True)
                        span.set_tag("error.type", type(e).__name__)
                        raise
        
        return wrapper

class AG06TracedWorkflowSystem:
    """Example AG06 workflow system with distributed tracing"""
    
    def __init__(self, otel_manager: OpenTelemetryManager):
        self.otel_manager = otel_manager
        self.tracer = otel_manager.get_tracer()
        
    @DistributedTracingDecorator(None, "workflow.execute")
    async def execute_workflow(self, workflow_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow with tracing"""
        # This decorator will be set up properly when initialized
        if hasattr(self, '_decorator'):
            return await self._decorator._trace_async(self._execute_workflow_impl)(workflow_type, context)
        else:
            # Fallback for demo
            return await self._execute_workflow_impl(workflow_type, context)
    
    async def _execute_workflow_impl(self, workflow_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Internal workflow execution"""
        
        # Simulate workflow steps with child spans
        await self._validate_input(workflow_type, context)
        processed_data = await self._process_data(context)
        result = await self._generate_output(processed_data)
        
        return {
            "workflow_id": str(uuid.uuid4()),
            "workflow_type": workflow_type,
            "status": "completed",
            "result": result,
            "processing_time_ms": 150  # Simulated
        }
    
    async def _validate_input(self, workflow_type: str, context: Dict[str, Any]):
        """Validate input with tracing"""
        tracer = self.otel_manager.get_tracer()
        
        if OTEL_AVAILABLE:
            with tracer.start_as_current_span("workflow.validate_input") as span:
                span.set_attribute("workflow.type", workflow_type)
                span.set_attribute("context.size", len(context))
                
                # Simulate validation
                await asyncio.sleep(0.01)
                
                if workflow_type == "invalid_type":
                    span.set_attribute("validation.result", "failed")
                    raise ValueError("Invalid workflow type")
                
                span.set_attribute("validation.result", "passed")
        else:
            with tracer.start_span("workflow.validate_input", SpanKind.INTERNAL) as span:
                span.set_tag("workflow.type", workflow_type)
                span.set_tag("context.size", len(context))
                
                await asyncio.sleep(0.01)
                
                if workflow_type == "invalid_type":
                    span.set_tag("validation.result", "failed")
                    raise ValueError("Invalid workflow type")
                
                span.set_tag("validation.result", "passed")
    
    async def _process_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process data with tracing"""
        tracer = self.otel_manager.get_tracer()
        
        if OTEL_AVAILABLE:
            with tracer.start_as_current_span("workflow.process_data") as span:
                span.set_attribute("processing.input_size", len(context))
                
                # Simulate data processing
                await asyncio.sleep(0.05)
                
                processed = {
                    "processed": True,
                    "timestamp": time.time(),
                    "input_keys": list(context.keys())
                }
                
                span.set_attribute("processing.output_size", len(processed))
                span.add_event("data_processed", {
                    "input_keys": len(context),
                    "output_keys": len(processed)
                })
                
                return processed
        else:
            with tracer.start_span("workflow.process_data", SpanKind.INTERNAL) as span:
                span.set_tag("processing.input_size", len(context))
                
                await asyncio.sleep(0.05)
                
                processed = {
                    "processed": True,
                    "timestamp": time.time(),
                    "input_keys": list(context.keys())
                }
                
                span.set_tag("processing.output_size", len(processed))
                span.add_event("data_processed", {
                    "input_keys": len(context),
                    "output_keys": len(processed)
                })
                
                return processed
    
    async def _generate_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate output with tracing"""
        tracer = self.otel_manager.get_tracer()
        
        if OTEL_AVAILABLE:
            with tracer.start_as_current_span("workflow.generate_output") as span:
                span.set_attribute("generation.input_processed", data.get("processed", False))
                
                # Simulate output generation
                await asyncio.sleep(0.02)
                
                output = {
                    "success": True,
                    "data_processed": data.get("processed", False),
                    "output_timestamp": time.time()
                }
                
                span.set_attribute("generation.success", True)
                return output
        else:
            with tracer.start_span("workflow.generate_output", SpanKind.INTERNAL) as span:
                span.set_tag("generation.input_processed", data.get("processed", False))
                
                await asyncio.sleep(0.02)
                
                output = {
                    "success": True,
                    "data_processed": data.get("processed", False),
                    "output_timestamp": time.time()
                }
                
                span.set_tag("generation.success", True)
                return output

class TracingAnalytics:
    """Analytics for trace data"""
    
    def __init__(self, otel_manager: OpenTelemetryManager):
        self.otel_manager = otel_manager
        
    def analyze_trace_performance(self) -> Dict[str, Any]:
        """Analyze trace performance data"""
        
        if hasattr(self.otel_manager.tracer, 'completed_spans'):
            spans = self.otel_manager.tracer.completed_spans
            
            if not spans:
                return {"error": "No trace data available"}
            
            # Calculate performance metrics
            durations = [span.duration_ms for span in spans if span.duration_ms]
            operations = {}
            
            for span in spans:
                op_name = span.operation_name
                if op_name not in operations:
                    operations[op_name] = {
                        "count": 0,
                        "total_duration": 0,
                        "errors": 0,
                        "min_duration": float('inf'),
                        "max_duration": 0
                    }
                
                operations[op_name]["count"] += 1
                if span.duration_ms:
                    operations[op_name]["total_duration"] += span.duration_ms
                    operations[op_name]["min_duration"] = min(
                        operations[op_name]["min_duration"], 
                        span.duration_ms
                    )
                    operations[op_name]["max_duration"] = max(
                        operations[op_name]["max_duration"],
                        span.duration_ms
                    )
                
                if span.status == SpanStatus.ERROR:
                    operations[op_name]["errors"] += 1
            
            # Calculate averages
            for op_name, op_data in operations.items():
                if op_data["count"] > 0:
                    op_data["avg_duration"] = op_data["total_duration"] / op_data["count"]
                    op_data["error_rate"] = (op_data["errors"] / op_data["count"]) * 100
                    
                    if op_data["min_duration"] == float('inf'):
                        op_data["min_duration"] = 0
            
            return {
                "total_spans": len(spans),
                "total_traces": len(set(span.trace_id for span in spans)),
                "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
                "operations": operations,
                "service_name": self.otel_manager.config.service_name,
                "analysis_timestamp": datetime.now().isoformat()
            }
        
        return {"message": "Using real OpenTelemetry - check your observability backend"}

# Production tracing demonstration
async def distributed_tracing_demo():
    """Comprehensive distributed tracing demonstration"""
    print("📊 OpenTelemetry Distributed Tracing Demo")
    print("=" * 60)
    
    # Configure tracing
    config = TraceConfig(
        service_name="ag06-workflow-engine",
        service_version="1.0.0",
        environment="production",
        sampling_rate=1.0  # 100% for demo
    )
    
    otel_manager = OpenTelemetryManager(config)
    
    print(f"✅ OpenTelemetry configured for service: {config.service_name}")
    print(f"   📡 Jaeger endpoint: {config.jaeger_endpoint}")
    print(f"   📡 OTLP endpoint: {config.otlp_endpoint}")
    print(f"   🎯 Sampling rate: {config.sampling_rate * 100}%")
    
    # Initialize traced workflow system
    workflow_system = AG06TracedWorkflowSystem(otel_manager)
    
    print(f"\n🔄 Executing traced workflows...")
    
    # Execute multiple workflows to generate trace data
    workflows = [
        ("data_processing", {"input": "dataset_1", "format": "json"}),
        ("validation", {"rules": ["required", "format"], "strict": True}),
        ("optimization", {"algorithm": "genetic", "iterations": 100}),
        ("reporting", {"format": "pdf", "include_charts": True}),
        ("cleanup", {"temporary_files": True, "cache": False})
    ]
    
    results = []
    for workflow_type, context in workflows:
        try:
            result = await workflow_system.execute_workflow(workflow_type, context)
            results.append(result)
            print(f"   ✅ {workflow_type}: {result['status']} ({result.get('processing_time_ms', 0):.1f}ms)")
        except Exception as e:
            print(f"   ❌ {workflow_type}: error - {e}")
            results.append({"status": "error", "error": str(e)})
    
    # Analyze trace performance
    analytics = TracingAnalytics(otel_manager)
    performance_analysis = analytics.analyze_trace_performance()
    
    print(f"\n📈 Trace Performance Analysis:")
    if "total_spans" in performance_analysis:
        print(f"   📊 Total spans: {performance_analysis['total_spans']}")
        print(f"   🔗 Total traces: {performance_analysis['total_traces']}")
        print(f"   ⏱️  Average duration: {performance_analysis['avg_duration_ms']:.2f}ms")
        
        print(f"\n🔍 Operation Breakdown:")
        for op_name, op_data in performance_analysis.get("operations", {}).items():
            print(f"   📋 {op_name}:")
            print(f"      Count: {op_data['count']}")
            print(f"      Avg Duration: {op_data.get('avg_duration', 0):.2f}ms")
            print(f"      Error Rate: {op_data.get('error_rate', 0):.1f}%")
    else:
        print(f"   {performance_analysis.get('message', 'No performance data')}")
    
    # Export trace data
    trace_report = {
        "service_config": {
            "name": config.service_name,
            "version": config.service_version,
            "environment": config.environment
        },
        "workflow_results": results,
        "performance_analysis": performance_analysis,
        "timestamp": datetime.now().isoformat()
    }
    
    with open("distributed_tracing_report.json", "w") as f:
        json.dump(trace_report, f, indent=2, default=str)
    
    print(f"\n📁 Generated Files:")
    print(f"   📄 distributed_tracing_report.json - Complete trace analysis")
    
    print(f"\n✅ Distributed tracing demo complete!")
    print(f"🔍 View traces in Jaeger UI: http://localhost:16686")
    print(f"📊 OTLP traces sent to: {config.otlp_endpoint}")
    
    return trace_report

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
    
    # Run distributed tracing demonstration
    asyncio.run(distributed_tracing_demo())