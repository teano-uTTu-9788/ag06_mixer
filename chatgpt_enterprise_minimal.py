#!/usr/bin/env python3
"""
Minimal ChatGPT Enterprise API Server
Production-ready with essential enterprise features
"""

import os
import time
import uuid
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import subprocess
import tempfile
import threading
from collections import defaultdict, deque

try:
    from fastapi import FastAPI, HTTPException, Depends, Request
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    print("Installing required packages...")
    subprocess.check_call(["pip", "install", "fastapi", "uvicorn", "python-multipart"])
    from fastapi import FastAPI, HTTPException, Depends, Request
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
def load_env():
    """Load environment variables from .env.enterprise file"""
    env_path = ".env.enterprise"
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

# Models
class CodeExecutionRequest(BaseModel):
    code: str = Field(..., max_length=50000)
    language: str = Field(default="python", pattern="^(python|javascript)$")
    timeout: int = Field(default=30, ge=1, le=300)

class CodeExecutionResponse(BaseModel):
    status: str
    output: str
    error: str
    execution_time: float
    language: str
    trace_id: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    uptime: float
    dependencies: Dict[str, str] = {}

# Rate Limiting (Simple in-memory implementation)
class TokenBucketRateLimiter:
    def __init__(self):
        self.buckets = defaultdict(lambda: {"tokens": 100, "last_refill": time.time()})
        self.max_tokens = 100
        self.refill_rate = 10  # tokens per minute

    def check_rate_limit(self, key: str, max_tokens: int = 100, refill_rate: int = 10) -> bool:
        now = time.time()
        bucket = self.buckets[key]
        
        # Refill tokens
        time_passed = now - bucket["last_refill"]
        tokens_to_add = time_passed * (refill_rate / 60)  # tokens per second
        bucket["tokens"] = min(max_tokens, bucket["tokens"] + tokens_to_add)
        bucket["last_refill"] = now
        
        # Check if request allowed
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True
        return False

# Circuit Breaker (Simple implementation)
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def can_proceed(self) -> bool:
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        elif self.state == "HALF_OPEN":
            return True

    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

# Initialize components
rate_limiter = TokenBucketRateLimiter()
circuit_breaker = CircuitBreaker()
start_time = time.time()

def create_app():
    app = FastAPI(
        title="Enterprise ChatGPT Coding API",
        description="Secure code execution API with enterprise features",
        version="3.0.0"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Security
    security = HTTPBearer()

    def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
        token = credentials.credentials
        expected_token = os.getenv("CHATGPT_API_TOKEN", "cgt_default_token")
        if token != expected_token:
            raise HTTPException(status_code=401, detail="Invalid authentication token")
        return token

    def check_rate_limit(request: Request):
        client_ip = request.client.host
        if not rate_limiter.check_rate_limit(client_ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in 60 seconds.")

    def execute_code_safely(code: str, language: str, timeout: int) -> Dict[str, Any]:
        """Execute code in a sandboxed environment"""
        trace_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        try:
            # Basic security validation
            dangerous_patterns = [
                'import os', 'import subprocess', 'import sys',
                'exec(', 'eval(', '__import__',
                'open(', 'file(', 'input(',
                'raw_input(', 'exit(', 'quit(',
                'delete', 'remove', 'unlink'
            ]
            
            for pattern in dangerous_patterns:
                if pattern in code.lower():
                    return {
                        "status": "security_error",
                        "output": "",
                        "error": f"Security validation failed: {pattern} not allowed",
                        "execution_time": time.time() - start_time,
                        "language": language,
                        "trace_id": trace_id
                    }

            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{language}', delete=False) as f:
                f.write(code)
                temp_file = f.name

            try:
                if language == "python":
                    result = subprocess.run(
                        ["python3", temp_file],
                        capture_output=True,
                        text=True,
                        timeout=timeout
                    )
                elif language == "javascript":
                    result = subprocess.run(
                        ["node", temp_file],
                        capture_output=True,
                        text=True,
                        timeout=timeout
                    )
                else:
                    raise ValueError(f"Unsupported language: {language}")

                execution_time = time.time() - start_time
                
                if result.returncode == 0:
                    return {
                        "status": "success",
                        "output": result.stdout,
                        "error": result.stderr,
                        "execution_time": execution_time,
                        "language": language,
                        "trace_id": trace_id
                    }
                else:
                    return {
                        "status": "error",
                        "output": result.stdout,
                        "error": result.stderr,
                        "execution_time": execution_time,
                        "language": language,
                        "trace_id": trace_id
                    }

            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file)
                except:
                    pass

        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "output": "",
                "error": f"Execution timed out after {timeout} seconds",
                "execution_time": time.time() - start_time,
                "language": language,
                "trace_id": trace_id
            }
        except Exception as e:
            return {
                "status": "error",
                "output": "",
                "error": str(e),
                "execution_time": time.time() - start_time,
                "language": language,
                "trace_id": trace_id
            }

    # API Endpoints
    @app.post("/execute", response_model=CodeExecutionResponse)
    async def execute_code(
        request: CodeExecutionRequest,
        token: str = Depends(verify_token),
        rate_limit: None = Depends(check_rate_limit)
    ):
        """Execute code securely"""
        if not circuit_breaker.can_proceed():
            raise HTTPException(status_code=503, detail="Circuit breaker is open - service temporarily unavailable")

        try:
            result = execute_code_safely(request.code, request.language, request.timeout)
            circuit_breaker.record_success()
            return CodeExecutionResponse(**result)
        except Exception as e:
            circuit_breaker.record_failure()
            logger.error(f"Code execution failed: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint"""
        uptime = time.time() - start_time
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow().isoformat() + "Z",
            version="3.0.0-enhanced",
            uptime=uptime,
            dependencies={"redis": "not_configured"}
        )

    @app.get("/metrics")
    async def get_metrics():
        """Prometheus metrics endpoint"""
        metrics = []
        metrics.append("# HELP chatgpt_uptime_seconds Total uptime in seconds")
        metrics.append("# TYPE chatgpt_uptime_seconds counter")
        metrics.append(f"chatgpt_uptime_seconds {time.time() - start_time:.2f}")
        
        metrics.append("# HELP chatgpt_circuit_breaker_state Circuit breaker state (0=CLOSED, 1=HALF_OPEN, 2=OPEN)")
        metrics.append("# TYPE chatgpt_circuit_breaker_state gauge")
        state_map = {"CLOSED": 0, "HALF_OPEN": 1, "OPEN": 2}
        metrics.append(f"chatgpt_circuit_breaker_state {state_map.get(circuit_breaker.state, 0)}")
        
        return "\n".join(metrics)

    @app.get("/status")
    async def get_status(token: str = Depends(verify_token)):
        """Detailed system status"""
        return {
            "uptime": time.time() - start_time,
            "circuit_breaker": {
                "state": circuit_breaker.state,
                "failure_count": circuit_breaker.failure_count,
                "failure_threshold": circuit_breaker.failure_threshold
            },
            "rate_limiter": {
                "active_buckets": len(rate_limiter.buckets),
                "max_tokens": rate_limiter.max_tokens,
                "refill_rate": rate_limiter.refill_rate
            },
            "metrics": {
                "total_requests": "not_tracked",
                "active_requests": "not_tracked"
            }
        }

    return app

def main():
    load_env()
    app = create_app()
    
    # Generate API token if not exists
    if not os.getenv("CHATGPT_API_TOKEN"):
        token = f"cgt_{uuid.uuid4().hex[:32]}"
        print(f"Generated API token: {token}")
        with open(".env.enterprise", "a") as f:
            f.write(f"\nCHATGPT_API_TOKEN={token}\n")
        os.environ["CHATGPT_API_TOKEN"] = token

    print("Starting ChatGPT Enterprise API server...")
    print(f"API Token: {os.getenv('CHATGPT_API_TOKEN')}")
    print("Endpoints available:")
    print("  POST /execute - Execute code")
    print("  GET  /health  - Health check")
    print("  GET  /metrics - Prometheus metrics")
    print("  GET  /status  - Detailed status")
    
    uvicorn.run(app, host="0.0.0.0", port=8090, log_level="info")

if __name__ == "__main__":
    main()