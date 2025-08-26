#!/usr/bin/env python3
"""
Enterprise ChatGPT API Server 2025 Edition
Following Google/Meta/Netflix/Amazon best practices
"""

import os
import time
import uuid
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from dataclasses import dataclass
import subprocess
import tempfile
import threading
from collections import defaultdict, deque
import hashlib
import hmac

try:
    from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from fastapi.responses import StreamingResponse, JSONResponse
    from pydantic import BaseModel, Field, validator
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    import uvicorn
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    import structlog
except ImportError as e:
    print(f"Installing required packages: {e}")
    subprocess.check_call([
        "pip", "install", 
        "fastapi", "uvicorn", "python-multipart", "pydantic",
        "prometheus-client", "opentelemetry-api", "opentelemetry-sdk",
        "opentelemetry-exporter-jaeger", "opentelemetry-instrumentation-fastapi",
        "structlog"
    ])
    from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from fastapi.responses import StreamingResponse, JSONResponse
    from pydantic import BaseModel, Field, validator
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    import uvicorn
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    import structlog

# Google SRE: Structured Logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Netflix: Circuit Breaker Pattern
@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_max_calls: int = 3

class CircuitBreakerState:
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

class CircuitBreaker:
    """Netflix-style Circuit Breaker with failure detection"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
        self._lock = threading.Lock()

    def can_proceed(self) -> bool:
        with self._lock:
            if self.state == CircuitBreakerState.CLOSED:
                return True
            elif self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time > self.config.recovery_timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_calls = 0
                    return True
                return False
            elif self.state == CircuitBreakerState.HALF_OPEN:
                return self.half_open_calls < self.config.half_open_max_calls

    def record_success(self):
        with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= 2:  # Netflix: Require multiple successes
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
            else:
                self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self):
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
            elif self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.half_open_calls += 1

# Google: Rate Limiting with Token Bucket + Sliding Window
class TokenBucketRateLimiter:
    """Google-style rate limiting with burst capacity"""
    
    def __init__(self, max_tokens: int = 100, refill_rate: float = 10.0):
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate  # tokens per second
        self.buckets = defaultdict(lambda: {
            "tokens": max_tokens,
            "last_refill": time.time(),
            "request_times": deque()
        })
        self._lock = threading.Lock()

    def check_rate_limit(self, key: str, tokens_required: int = 1) -> tuple[bool, Dict[str, Any]]:
        with self._lock:
            now = time.time()
            bucket = self.buckets[key]
            
            # Refill tokens
            time_passed = now - bucket["last_refill"]
            tokens_to_add = time_passed * self.refill_rate
            bucket["tokens"] = min(self.max_tokens, bucket["tokens"] + tokens_to_add)
            bucket["last_refill"] = now
            
            # Clean old request times (sliding window)
            while bucket["request_times"] and now - bucket["request_times"][0] > 3600:  # 1 hour window
                bucket["request_times"].popleft()
            
            # Check rate limit
            if bucket["tokens"] >= tokens_required:
                bucket["tokens"] -= tokens_required
                bucket["request_times"].append(now)
                return True, {
                    "remaining_tokens": bucket["tokens"],
                    "requests_last_hour": len(bucket["request_times"]),
                    "reset_time": now + (self.max_tokens - bucket["tokens"]) / self.refill_rate
                }
            
            return False, {
                "remaining_tokens": bucket["tokens"],
                "requests_last_hour": len(bucket["request_times"]),
                "retry_after": (tokens_required - bucket["tokens"]) / self.refill_rate
            }

# Meta: Feature Flags System
class FeatureFlags:
    """Meta-style feature flag system with gradual rollout"""
    
    def __init__(self):
        self.flags = {
            "enhanced_security": {"enabled": True, "rollout_percent": 100},
            "streaming_response": {"enabled": True, "rollout_percent": 50},
            "advanced_telemetry": {"enabled": True, "rollout_percent": 100},
            "code_analysis": {"enabled": False, "rollout_percent": 0},
            "ai_assistance": {"enabled": False, "rollout_percent": 10}
        }
    
    def is_enabled(self, flag_name: str, user_id: str = None) -> bool:
        flag = self.flags.get(flag_name, {"enabled": False, "rollout_percent": 0})
        if not flag["enabled"]:
            return False
            
        if flag["rollout_percent"] >= 100:
            return True
            
        # Consistent hash-based rollout
        if user_id:
            hash_value = int(hashlib.md5(f"{flag_name}:{user_id}".encode()).hexdigest(), 16)
            return (hash_value % 100) < flag["rollout_percent"]
        
        return False

# Amazon: Request/Response Models with Validation
class CodeExecutionRequest(BaseModel):
    code: str = Field(..., max_length=100000, min_length=1)
    language: str = Field(default="python", regex="^(python|javascript|typescript)$")
    timeout: int = Field(default=30, ge=1, le=300)
    streaming: bool = Field(default=False)
    analysis: bool = Field(default=False)
    
    @validator('code')
    def validate_code_security(cls, v):
        dangerous_patterns = [
            'import os', 'import subprocess', 'import sys',
            'exec(', 'eval(', '__import__', 'compile(',
            'open(', 'file(', 'input(', 'raw_input(',
            'exit(', 'quit(', 'delete', 'remove', 'unlink',
            'socket', 'urllib', 'requests', 'http',
            '__builtins__', 'globals(', 'locals(',
            'setattr(', 'getattr(', 'hasattr(', 'delattr('
        ]
        
        code_lower = v.lower()
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                raise ValueError(f"Security validation failed: {pattern} not allowed")
        return v

class CodeExecutionResponse(BaseModel):
    status: str
    output: str
    error: str
    execution_time: float
    language: str
    trace_id: str
    memory_usage: Optional[int] = None
    analysis: Optional[Dict[str, Any]] = None

# Prometheus Metrics (Google SRE)
REQUEST_COUNT = Counter(
    'chatgpt_api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'chatgpt_api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint']
)

CODE_EXECUTION_DURATION = Histogram(
    'chatgpt_code_execution_duration_seconds',
    'Code execution duration',
    ['language', 'status']
)

ACTIVE_REQUESTS = Gauge(
    'chatgpt_api_active_requests',
    'Active API requests'
)

CIRCUIT_BREAKER_STATE = Gauge(
    'chatgpt_circuit_breaker_state',
    'Circuit breaker state (0=CLOSED, 1=HALF_OPEN, 2=OPEN)'
)

# OpenTelemetry Setup (Google/Amazon observability)
def setup_tracing():
    trace.set_tracer_provider(TracerProvider())
    jaeger_exporter = JaegerExporter(
        agent_host_name="localhost",
        agent_port=14268,
    )
    
    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)

# Global instances
circuit_breaker = CircuitBreaker(CircuitBreakerConfig())
rate_limiter = TokenBucketRateLimiter(max_tokens=100, refill_rate=1.67)  # 100 requests/minute
feature_flags = FeatureFlags()
start_time = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Google: Graceful startup and shutdown"""
    logger.info("Starting ChatGPT Enterprise API Server 2025")
    setup_tracing()
    yield
    logger.info("Shutting down ChatGPT Enterprise API Server 2025")

def create_app() -> FastAPI:
    app = FastAPI(
        title="ChatGPT Enterprise API 2025",
        description="Production-grade code execution API following Google/Meta/Netflix/Amazon best practices",
        version="4.0.0",
        lifespan=lifespan
    )

    # Security middlewares
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure for production
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://chat.openai.com", "https://chatgpt.com"],  # Restrict in production
        allow_credentials=True,
        allow_methods=["POST", "GET"],
        allow_headers=["*"],
    )

    # OpenTelemetry instrumentation
    FastAPIInstrumentor.instrument_app(app)

    security = HTTPBearer()

    async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
        """Enhanced token verification with rate limiting"""
        token = credentials.credentials
        expected_token = os.getenv("CHATGPT_API_TOKEN", "cgt_default_token")
        
        # HMAC verification for production
        if feature_flags.is_enabled("enhanced_security", token):
            secret_key = os.getenv("API_SECRET_KEY", "default_secret").encode()
            expected_hmac = hmac.new(secret_key, token.encode(), hashlib.sha256).hexdigest()
            provided_hmac = hmac.new(secret_key, expected_token.encode(), hashlib.sha256).hexdigest()
            
            if not hmac.compare_digest(expected_hmac, provided_hmac):
                logger.warning("HMAC token verification failed", token_prefix=token[:10])
                raise HTTPException(status_code=401, detail="Invalid authentication token")
        else:
            if token != expected_token:
                logger.warning("Basic token verification failed", token_prefix=token[:10])
                raise HTTPException(status_code=401, detail="Invalid authentication token")
        
        return token

    async def check_rate_limit(request: Request):
        """Enhanced rate limiting with client identification"""
        # Use multiple identifiers for better rate limiting
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        auth_header = request.headers.get("authorization", "")
        
        # Create composite key
        rate_limit_key = f"{client_ip}:{hashlib.md5(f'{user_agent}:{auth_header}'.encode()).hexdigest()[:8]}"
        
        allowed, info = rate_limiter.check_rate_limit(rate_limit_key)
        if not allowed:
            logger.warning("Rate limit exceeded", 
                         client_ip=client_ip, 
                         retry_after=info.get("retry_after", 60))
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Try again in {info.get('retry_after', 60):.1f} seconds.",
                headers={"Retry-After": str(int(info.get("retry_after", 60)))}
            )

    async def execute_code_safely(
        request: CodeExecutionRequest,
        trace_id: str
    ) -> CodeExecutionResponse:
        """Enhanced code execution with telemetry"""
        start_time = time.time()
        
        with trace.get_tracer(__name__).start_as_current_span("code_execution") as span:
            span.set_attribute("code.language", request.language)
            span.set_attribute("code.length", len(request.code))
            span.set_attribute("trace_id", trace_id)
            
            try:
                # Create temporary file
                suffix_map = {"python": ".py", "javascript": ".js", "typescript": ".ts"}
                suffix = suffix_map.get(request.language, ".txt")
                
                with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
                    f.write(request.code)
                    temp_file = f.name

                try:
                    # Execute based on language
                    if request.language == "python":
                        cmd = ["python3", temp_file]
                    elif request.language in ["javascript", "typescript"]:
                        cmd = ["node", temp_file]
                    else:
                        raise ValueError(f"Unsupported language: {request.language}")

                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=request.timeout,
                        env={"PYTHONPATH": "", "NODE_PATH": ""}  # Security: Clean environment
                    )

                    execution_time = time.time() - start_time
                    
                    # Record metrics
                    CODE_EXECUTION_DURATION.labels(
                        language=request.language,
                        status="success" if result.returncode == 0 else "error"
                    ).observe(execution_time)
                    
                    response_data = {
                        "status": "success" if result.returncode == 0 else "error",
                        "output": result.stdout,
                        "error": result.stderr,
                        "execution_time": execution_time,
                        "language": request.language,
                        "trace_id": trace_id
                    }
                    
                    # Optional code analysis
                    if request.analysis and feature_flags.is_enabled("code_analysis"):
                        response_data["analysis"] = {
                            "lines_of_code": len(request.code.split('\n')),
                            "complexity_estimate": len(request.code) // 50,
                            "performance_tier": "fast" if execution_time < 0.1 else "moderate" if execution_time < 1.0 else "slow"
                        }
                    
                    return CodeExecutionResponse(**response_data)

                finally:
                    # Clean up
                    try:
                        os.unlink(temp_file)
                    except:
                        pass

            except subprocess.TimeoutExpired:
                span.set_attribute("error", "timeout")
                return CodeExecutionResponse(
                    status="timeout",
                    output="",
                    error=f"Execution timed out after {request.timeout} seconds",
                    execution_time=time.time() - start_time,
                    language=request.language,
                    trace_id=trace_id
                )
            except Exception as e:
                span.set_attribute("error", str(e))
                logger.error("Code execution failed", error=str(e), trace_id=trace_id)
                return CodeExecutionResponse(
                    status="error",
                    output="",
                    error=str(e),
                    execution_time=time.time() - start_time,
                    language=request.language,
                    trace_id=trace_id
                )

    # API Endpoints
    @app.post("/execute", response_model=CodeExecutionResponse)
    async def execute_code(
        request: CodeExecutionRequest,
        background_tasks: BackgroundTasks,
        token: str = Depends(verify_token),
        rate_limit: None = Depends(check_rate_limit)
    ):
        """Execute code with enterprise features"""
        trace_id = str(uuid.uuid4())[:8]
        
        ACTIVE_REQUESTS.inc()
        REQUEST_COUNT.labels(method="POST", endpoint="/execute", status="in_progress").inc()
        
        try:
            # Circuit breaker check
            if not circuit_breaker.can_proceed():
                CIRCUIT_BREAKER_STATE.set(2)  # OPEN
                raise HTTPException(
                    status_code=503, 
                    detail="Circuit breaker is open - service temporarily unavailable"
                )

            result = await execute_code_safely(request, trace_id)
            
            # Update circuit breaker
            if result.status in ["success", "error"]:  # Successful execution, even with code errors
                circuit_breaker.record_success()
                CIRCUIT_BREAKER_STATE.set(0)  # CLOSED
            else:
                circuit_breaker.record_failure()
                CIRCUIT_BREAKER_STATE.set(1 if circuit_breaker.state == "HALF_OPEN" else 2)
            
            REQUEST_COUNT.labels(method="POST", endpoint="/execute", status=result.status).inc()
            return result

        except Exception as e:
            circuit_breaker.record_failure()
            logger.error("Request processing failed", error=str(e), trace_id=trace_id)
            REQUEST_COUNT.labels(method="POST", endpoint="/execute", status="server_error").inc()
            raise HTTPException(status_code=500, detail="Internal server error")
        finally:
            ACTIVE_REQUESTS.dec()

    @app.get("/health")
    async def health_check():
        """Enhanced health check with dependencies"""
        uptime = time.time() - start_time
        
        # Check circuit breaker health
        cb_health = "healthy" if circuit_breaker.state == "CLOSED" else "degraded" if circuit_breaker.state == "HALF_OPEN" else "unhealthy"
        
        overall_status = "healthy"
        if cb_health != "healthy":
            overall_status = "degraded"
            
        return {
            "status": overall_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "4.0.0-enterprise",
            "uptime": uptime,
            "components": {
                "circuit_breaker": {
                    "status": cb_health,
                    "state": circuit_breaker.state,
                    "failure_count": circuit_breaker.failure_count
                },
                "rate_limiter": {
                    "status": "healthy",
                    "active_buckets": len(rate_limiter.buckets)
                },
                "feature_flags": {
                    "status": "healthy",
                    "enabled_flags": [k for k, v in feature_flags.flags.items() if v["enabled"]]
                }
            }
        }

    @app.get("/metrics")
    async def get_metrics():
        """Prometheus metrics endpoint"""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/status")
    async def get_status(token: str = Depends(verify_token)):
        """Detailed system status for operators"""
        return {
            "uptime": time.time() - start_time,
            "circuit_breaker": {
                "state": circuit_breaker.state,
                "failure_count": circuit_breaker.failure_count,
                "success_count": circuit_breaker.success_count,
                "config": {
                    "failure_threshold": circuit_breaker.config.failure_threshold,
                    "recovery_timeout": circuit_breaker.config.recovery_timeout
                }
            },
            "rate_limiter": {
                "active_buckets": len(rate_limiter.buckets),
                "max_tokens": rate_limiter.max_tokens,
                "refill_rate": rate_limiter.refill_rate
            },
            "feature_flags": feature_flags.flags,
            "system": {
                "python_version": subprocess.check_output(["python3", "--version"], text=True).strip(),
                "node_version": subprocess.check_output(["node", "--version"], text=True).strip() if os.system("which node") == 0 else "not_available"
            }
        }

    return app

def main():
    """Main entry point with production configuration"""
    # Load environment variables
    env_file = ".env.enterprise"
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

    # Generate tokens if needed
    if not os.getenv("CHATGPT_API_TOKEN"):
        token = f"cgt_{uuid.uuid4().hex[:32]}"
        logger.info("Generated new API token", token_prefix=token[:10])
        with open(env_file, "a") as f:
            f.write(f"\nCHATGPT_API_TOKEN={token}\n")
        os.environ["CHATGPT_API_TOKEN"] = token

    if not os.getenv("API_SECRET_KEY"):
        secret = uuid.uuid4().hex
        with open(env_file, "a") as f:
            f.write(f"API_SECRET_KEY={secret}\n")
        os.environ["API_SECRET_KEY"] = secret

    app = create_app()
    
    logger.info(
        "Starting ChatGPT Enterprise API Server 2025",
        token=os.getenv('CHATGPT_API_TOKEN', 'not_set')[:10] + "...",
        features={
            "enhanced_security": feature_flags.is_enabled("enhanced_security"),
            "streaming_response": feature_flags.is_enabled("streaming_response"),
            "advanced_telemetry": feature_flags.is_enabled("advanced_telemetry")
        }
    )
    
    # Production configuration
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8090,
        log_level="info",
        access_log=True,
        workers=1,  # Single worker for development; scale for production
        loop="auto",
        http="auto"
    )

if __name__ == "__main__":
    main()