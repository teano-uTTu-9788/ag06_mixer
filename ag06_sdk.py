#!/usr/bin/env python3
"""
AG06 Production SDK - Python Client Library
Enterprise-grade SDK for AG06 workflow system integration
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import time
import warnings

# Version info
__version__ = "1.0.0"
__author__ = "AG06 Team"

class SDKException(Exception):
    """Base exception for AG06 SDK errors"""
    pass

class AuthenticationError(SDKException):
    """Authentication-related errors"""
    pass

class APIError(SDKException):
    """API-related errors"""
    pass

class ConnectionError(SDKException):
    """Connection-related errors"""
    pass

@dataclass
class APIResponse:
    """Standardized API response wrapper"""
    success: bool
    data: Any = None
    error: str = None
    status_code: int = 200
    timestamp: str = None
    request_id: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

class AG06Client:
    """
    Main AG06 SDK Client for enterprise integration
    
    Provides comprehensive access to AG06 workflow system:
    - System status and health monitoring
    - Workflow execution and management
    - ML analytics and predictive insights
    - Real-time dashboard integration
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: str = None,
        timeout: int = 30,
        retry_attempts: int = 3,
        log_level: LogLevel = LogLevel.INFO
    ):
        """
        Initialize AG06 SDK Client
        
        Args:
            base_url: AG06 API base URL
            api_key: Authentication API key (optional for local development)
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts for failed requests
            log_level: Logging level for SDK operations
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Setup logging
        self.logger = self._setup_logging(log_level)
        
        # Configuration tracking
        self.config = {
            "base_url": self.base_url,
            "timeout": self.timeout,
            "retry_attempts": self.retry_attempts,
            "sdk_version": __version__,
            "initialized_at": datetime.now().isoformat()
        }
        
        self.logger.info(f"AG06 SDK v{__version__} initialized for {self.base_url}")
    
    def _setup_logging(self, log_level: LogLevel) -> logging.Logger:
        """Setup SDK logging configuration"""
        logger = logging.getLogger(f"ag06_sdk")
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | AG06-SDK | %(levelname)s | %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        logger.setLevel(getattr(logging, log_level.value))
        return logger
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def _ensure_session(self):
        """Ensure aiohttp session is created"""
        if self.session is None or self.session.closed:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            connector = aiohttp.TCPConnector(limit=100, ttl_dns_cache=300)
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            
            self.session = aiohttp.ClientSession(
                headers=headers,
                connector=connector,
                timeout=timeout
            )
            
            self.logger.debug("HTTP session created")
    
    async def close(self):
        """Close HTTP session and cleanup resources"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.logger.debug("HTTP session closed")
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Dict = None,
        params: Dict = None
    ) -> APIResponse:
        """
        Make HTTP request with retry logic and error handling
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            data: Request body data
            params: Query parameters
            
        Returns:
            APIResponse object with standardized response data
        """
        await self._ensure_session()
        
        url = f"{self.base_url}{endpoint}"
        self.logger.debug(f"{method} {url}")
        
        for attempt in range(self.retry_attempts):
            try:
                async with self.session.request(
                    method=method,
                    url=url,
                    json=data if data else None,
                    params=params if params else None
                ) as response:
                    
                    response_text = await response.text()
                    
                    # Try to parse JSON response
                    try:
                        response_data = json.loads(response_text) if response_text else {}
                    except json.JSONDecodeError:
                        response_data = {"raw_response": response_text}
                    
                    # Create standardized response
                    api_response = APIResponse(
                        success=response.status < 400,
                        data=response_data,
                        status_code=response.status,
                        request_id=response.headers.get("X-Request-ID")
                    )
                    
                    if not api_response.success:
                        error_msg = response_data.get("error", f"HTTP {response.status}")
                        api_response.error = error_msg
                        self.logger.warning(f"API Error: {error_msg}")
                    
                    return api_response
            
            except asyncio.TimeoutError:
                error_msg = f"Request timeout after {self.timeout}s"
                self.logger.warning(f"Attempt {attempt + 1}: {error_msg}")
                if attempt == self.retry_attempts - 1:
                    return APIResponse(success=False, error=error_msg)
            
            except Exception as e:
                error_msg = f"Request failed: {str(e)}"
                self.logger.warning(f"Attempt {attempt + 1}: {error_msg}")
                if attempt == self.retry_attempts - 1:
                    return APIResponse(success=False, error=error_msg)
            
            # Wait before retry (exponential backoff)
            if attempt < self.retry_attempts - 1:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
        
        return APIResponse(success=False, error="Max retry attempts exceeded")
    
    # System Health and Status
    async def get_health(self) -> APIResponse:
        """Get system health status"""
        return await self._make_request("GET", "/health")
    
    async def get_system_status(self) -> APIResponse:
        """Get comprehensive system status"""
        return await self._make_request("GET", "/api/v1/system/status")
    
    # Workflow Management
    async def list_workflows(self, status: str = None, limit: int = 50) -> APIResponse:
        """
        List workflows with optional filtering
        
        Args:
            status: Filter by workflow status (pending, running, completed, failed)
            limit: Maximum number of workflows to return
        """
        params = {"limit": limit}
        if status:
            params["status"] = status
        
        return await self._make_request("GET", "/api/v1/workflows", params=params)
    
    async def execute_workflow(
        self,
        workflow_type: str,
        context: Dict[str, Any] = None,
        priority: int = 1
    ) -> APIResponse:
        """
        Execute a workflow
        
        Args:
            workflow_type: Type of workflow to execute
            context: Workflow execution context
            priority: Workflow priority (1=highest, 5=lowest)
        """
        data = {
            "workflow_type": workflow_type,
            "context": context or {},
            "priority": priority
        }
        
        return await self._make_request("POST", "/api/v1/workflows", data=data)
    
    async def get_workflow_status(self, workflow_id: str) -> APIResponse:
        """Get status of specific workflow"""
        return await self._make_request("GET", f"/api/v1/workflows/{workflow_id}")
    
    async def cancel_workflow(self, workflow_id: str) -> APIResponse:
        """Cancel running workflow"""
        return await self._make_request("DELETE", f"/api/v1/workflows/{workflow_id}")
    
    # ML Analytics and Insights
    async def get_ml_insights(
        self,
        insight_types: List[str] = None,
        time_horizon: str = "short_term"
    ) -> APIResponse:
        """
        Get ML predictive insights
        
        Args:
            insight_types: Types of insights to generate
            time_horizon: Prediction time horizon (immediate, short_term, medium_term, long_term)
        """
        params = {"time_horizon": time_horizon}
        if insight_types:
            params["types"] = ",".join(insight_types)
        
        return await self._make_request("GET", "/api/v1/analytics/insights", params=params)
    
    async def get_anomaly_detection(self, window_minutes: int = 60) -> APIResponse:
        """Get anomaly detection results"""
        params = {"window_minutes": window_minutes}
        return await self._make_request("GET", "/api/v1/analytics/anomalies", params=params)
    
    async def get_performance_forecast(self, forecast_hours: int = 24) -> APIResponse:
        """Get performance forecasting results"""
        params = {"forecast_hours": forecast_hours}
        return await self._make_request("GET", "/api/v1/analytics/forecast", params=params)
    
    async def get_resource_optimization(self) -> APIResponse:
        """Get resource optimization recommendations"""
        return await self._make_request("GET", "/api/v1/analytics/optimization")
    
    # Dashboard Integration
    async def get_dashboard_data(self, components: List[str] = None) -> APIResponse:
        """
        Get dashboard data for specific components
        
        Args:
            components: List of dashboard components to retrieve
        """
        params = {}
        if components:
            params["components"] = ",".join(components)
        
        return await self._make_request("GET", "/api/v1/dashboard/data", params=params)
    
    async def get_metrics(
        self,
        metric_names: List[str] = None,
        start_time: str = None,
        end_time: str = None
    ) -> APIResponse:
        """
        Get system metrics
        
        Args:
            metric_names: Specific metrics to retrieve
            start_time: Start time for metric range (ISO format)
            end_time: End time for metric range (ISO format)
        """
        params = {}
        if metric_names:
            params["metrics"] = ",".join(metric_names)
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        
        return await self._make_request("GET", "/api/v1/metrics", params=params)


class WorkflowClient:
    """Specialized client for workflow operations"""
    
    def __init__(self, ag06_client: AG06Client):
        self.client = ag06_client
    
    async def bulk_execute(
        self,
        workflows: List[Dict[str, Any]],
        max_concurrent: int = 5
    ) -> List[APIResponse]:
        """
        Execute multiple workflows concurrently
        
        Args:
            workflows: List of workflow configurations
            max_concurrent: Maximum concurrent executions
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_single(workflow_config):
            async with semaphore:
                return await self.client.execute_workflow(**workflow_config)
        
        tasks = [execute_single(config) for config in workflows]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def wait_for_completion(
        self,
        workflow_id: str,
        timeout: int = 300,
        poll_interval: int = 5
    ) -> APIResponse:
        """
        Wait for workflow completion with polling
        
        Args:
            workflow_id: Workflow ID to monitor
            timeout: Maximum wait time in seconds
            poll_interval: Polling interval in seconds
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status_response = await self.client.get_workflow_status(workflow_id)
            
            if not status_response.success:
                return status_response
            
            status = status_response.data.get("status", "unknown")
            if status in ["completed", "failed", "cancelled"]:
                return status_response
            
            await asyncio.sleep(poll_interval)
        
        return APIResponse(
            success=False,
            error=f"Workflow {workflow_id} did not complete within {timeout}s"
        )


class AnalyticsClient:
    """Specialized client for ML analytics and insights"""
    
    def __init__(self, ag06_client: AG06Client):
        self.client = ag06_client
    
    async def comprehensive_analysis(self) -> Dict[str, APIResponse]:
        """Get comprehensive system analysis"""
        
        # Execute all analytics endpoints concurrently
        tasks = {
            "insights": self.client.get_ml_insights(),
            "anomalies": self.client.get_anomaly_detection(),
            "forecast": self.client.get_performance_forecast(),
            "optimization": self.client.get_resource_optimization()
        }
        
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        # Combine results
        analysis = {}
        for key, result in zip(tasks.keys(), results):
            if isinstance(result, APIResponse):
                analysis[key] = result
            else:
                analysis[key] = APIResponse(
                    success=False,
                    error=f"Analysis failed: {str(result)}"
                )
        
        return analysis
    
    async def anomaly_monitor(
        self,
        callback: Callable[[Dict], None],
        interval: int = 60,
        threshold: float = 0.7
    ):
        """
        Continuous anomaly monitoring with callback
        
        Args:
            callback: Function to call when anomalies detected
            interval: Monitoring interval in seconds
            threshold: Anomaly confidence threshold
        """
        while True:
            try:
                response = await self.client.get_anomaly_detection()
                
                if response.success and response.data:
                    confidence = response.data.get("confidence_score", 0)
                    if confidence >= threshold:
                        callback(response.data)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.client.logger.error(f"Anomaly monitoring error: {e}")
                await asyncio.sleep(interval)


# Convenience functions for quick usage
async def quick_health_check(base_url: str = "http://localhost:8080") -> Dict[str, Any]:
    """Quick health check without full client initialization"""
    async with AG06Client(base_url=base_url) as client:
        response = await client.get_health()
        return {
            "healthy": response.success,
            "data": response.data,
            "error": response.error
        }

async def quick_workflow_execute(
    workflow_type: str,
    context: Dict[str, Any] = None,
    base_url: str = "http://localhost:8080"
) -> Dict[str, Any]:
    """Quick workflow execution without full client setup"""
    async with AG06Client(base_url=base_url) as client:
        response = await client.execute_workflow(workflow_type, context)
        return {
            "success": response.success,
            "data": response.data,
            "error": response.error
        }

async def quick_insights(base_url: str = "http://localhost:8080") -> Dict[str, Any]:
    """Quick ML insights retrieval"""
    async with AG06Client(base_url=base_url) as client:
        analytics = AnalyticsClient(client)
        results = await analytics.comprehensive_analysis()
        return {key: result.data for key, result in results.items() if result.success}


# Example usage and demonstration
async def sdk_demo():
    """Comprehensive SDK demonstration"""
    print("🚀 AG06 SDK Demo Starting")
    print("=" * 50)
    
    # Initialize client
    async with AG06Client(base_url="http://localhost:8080") as client:
        print(f"✅ Connected to AG06 at {client.base_url}")
        
        # 1. Health check
        print("\n📋 System Health Check:")
        health = await client.get_health()
        print(f"   Status: {'✅ Healthy' if health.success else '❌ Unhealthy'}")
        if health.data:
            print(f"   Data: {health.data}")
        
        # 2. System status
        print("\n📊 System Status:")
        status = await client.get_system_status()
        if status.success and status.data:
            print(f"   Workflows: {status.data.get('workflows', 'N/A')}")
            print(f"   Agents: {status.data.get('agents', 'N/A')}")
        
        # 3. ML Analytics
        print("\n🤖 ML Analytics:")
        insights = await client.get_ml_insights()
        if insights.success and insights.data:
            print(f"   Insights: {len(insights.data.get('insights', []))} generated")
            print(f"   Confidence: {insights.data.get('average_confidence', 'N/A')}")
        
        # 4. Workflow execution
        print("\n⚙️ Workflow Execution:")
        workflow_result = await client.execute_workflow(
            "demo_workflow",
            {"demo": True, "timestamp": datetime.now().isoformat()}
        )
        print(f"   Execution: {'✅ Started' if workflow_result.success else '❌ Failed'}")
        
        # 5. Specialized clients
        print("\n🔧 Specialized Clients:")
        analytics_client = AnalyticsClient(client)
        comprehensive = await analytics_client.comprehensive_analysis()
        print(f"   Analytics Results: {len([r for r in comprehensive.values() if r.success])}/4 successful")
        
        workflow_client = WorkflowClient(client)
        print(f"   Workflow Client: ✅ Ready for bulk operations")
    
    print("\n✅ AG06 SDK Demo Complete!")
    return True


if __name__ == "__main__":
    # Run demo
    try:
        asyncio.run(sdk_demo())
    except KeyboardInterrupt:
        print("\n⏹️ Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")