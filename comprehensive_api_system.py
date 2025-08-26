#!/usr/bin/env python3
"""
Comprehensive REST API with GraphQL Interface - Phase 2
Enterprise-grade API system for external integrations and SDK support
"""

import asyncio
import sys
import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import traceback

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Web framework and API libraries
try:
    from flask import Flask, request, jsonify, Blueprint
    from flask_restful import Api, Resource
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("⚠️  Flask/Flask-RESTful not available - using HTTP server fallback")

try:
    import graphene
    from flask_graphql import GraphQLView
    GRAPHQL_AVAILABLE = True
except ImportError:
    GRAPHQL_AVAILABLE = False
    print("⚠️  GraphQL not available - REST-only API")

# Import our existing systems
from integrated_workflow_system import IntegratedWorkflowSystem
from ml_predictive_analytics_engine import MLPredictiveAnalyticsEngine
from interactive_dashboard_system import InteractiveDashboardSystem

class APIVersion(Enum):
    V1 = "v1"
    V2 = "v2"
    BETA = "beta"

class HTTPMethod(Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"

@dataclass
class APIEndpoint:
    path: str
    method: HTTPMethod
    version: APIVersion
    description: str
    parameters: List[Dict[str, str]]
    response_schema: Dict[str, Any]
    auth_required: bool = False
    rate_limit: Optional[int] = None

@dataclass
class APIResponse:
    success: bool
    data: Any = None
    error: Optional[str] = None
    message: Optional[str] = None
    timestamp: datetime = None
    version: str = "v1"
    request_id: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

# GraphQL Schema Definitions (if available)
if GRAPHQL_AVAILABLE:
    
    class SystemHealthType(graphene.ObjectType):
        overall_status = graphene.String()
        active_workflows = graphene.Int()
        total_events = graphene.Int()
        memory_usage = graphene.Float()
        cpu_usage = graphene.Float()
        
    class PredictiveInsightType(graphene.ObjectType):
        insight_id = graphene.String()
        prediction_type = graphene.String()
        confidence_score = graphene.Float()
        severity = graphene.String()
        recommended_actions = graphene.List(graphene.String)
        
    class WorkflowType(graphene.ObjectType):
        workflow_id = graphene.String()
        workflow_type = graphene.String()
        status = graphene.String()
        created_at = graphene.String()
        
    class Query(graphene.ObjectType):
        system_health = graphene.Field(SystemHealthType)
        workflows = graphene.List(WorkflowType)
        predictive_insights = graphene.List(PredictiveInsightType)
        
        def resolve_system_health(self, info):
            # This would be populated by the API system
            return SystemHealthType(
                overall_status="healthy",
                active_workflows=0,
                total_events=0,
                memory_usage=0.0,
                cpu_usage=0.0
            )
        
        def resolve_workflows(self, info):
            # Return mock workflows for now
            return [
                WorkflowType(
                    workflow_id="wf_001",
                    workflow_type="data_processing",
                    status="running",
                    created_at=datetime.now().isoformat()
                )
            ]
        
        def resolve_predictive_insights(self, info):
            # Return mock insights
            return [
                PredictiveInsightType(
                    insight_id="insight_001",
                    prediction_type="anomaly_detection",
                    confidence_score=0.85,
                    severity="LOW",
                    recommended_actions=["Monitor system"]
                )
            ]
    
    class ExecuteWorkflowMutation(graphene.Mutation):
        class Arguments:
            workflow_type = graphene.String(required=True)
            context = graphene.String()
        
        success = graphene.Boolean()
        workflow_id = graphene.String()
        message = graphene.String()
        
        def mutate(self, info, workflow_type, context=None):
            # Execute workflow logic here
            workflow_id = f"wf_{int(time.time())}"
            return ExecuteWorkflowMutation(
                success=True,
                workflow_id=workflow_id,
                message=f"Workflow {workflow_type} started successfully"
            )
    
    class Mutation(graphene.ObjectType):
        execute_workflow = ExecuteWorkflowMutation.Field()
    
    # GraphQL schema
    schema = graphene.Schema(query=Query, mutation=Mutation)

class ComprehensiveAPISystem:
    """Enterprise-grade REST API with GraphQL interface"""
    
    def __init__(self, api_id: str = "api_001", port: int = 8082):
        self.api_id = api_id
        self.port = port
        self.app = None
        self.api = None
        
        # System integrations
        self.workflow_system = None
        self.ml_engine = None
        self.dashboard_system = None
        
        # API state
        self.endpoints = {}
        self.request_count = 0
        self.error_count = 0
        self.rate_limits = {}
        
        # Configuration
        self.config = {
            "cors_enabled": True,
            "rate_limiting_enabled": True,
            "default_rate_limit": 100,  # requests per minute
            "api_key_required": False,
            "max_request_size": "10MB",
            "timeout_seconds": 30,
            "enable_swagger": True,
            "enable_graphql": GRAPHQL_AVAILABLE
        }
        
        # Setup logging
        self.logger = self._setup_logging()
        
        print(f"🌐 Comprehensive API System {self.api_id} initialized")
        print(f"   ✅ Port: {self.port}")
        print(f"   ✅ REST API: {'Yes' if FLASK_AVAILABLE else 'No (fallback mode)'}")
        print(f"   ✅ GraphQL: {'Yes' if GRAPHQL_AVAILABLE else 'No (REST only)'}")
        print(f"   ✅ CORS: {'Enabled' if self.config['cors_enabled'] else 'Disabled'}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging for the API system"""
        logger = logging.getLogger(f"api_{self.api_id}")
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s | API-{self.api_id} | %(levelname)s | %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    async def initialize(self) -> bool:
        """Initialize the API system and all integrations"""
        try:
            self.logger.info("🌐 Initializing Comprehensive API System...")
            
            # Initialize system integrations
            self.workflow_system = IntegratedWorkflowSystem()
            
            self.ml_engine = MLPredictiveAnalyticsEngine()
            await self.ml_engine.initialize()
            
            self.dashboard_system = InteractiveDashboardSystem()
            await self.dashboard_system.initialize()
            
            # Initialize Flask app if available
            if FLASK_AVAILABLE:
                await self._initialize_flask_api()
            
            # Register all endpoints
            await self._register_api_endpoints()
            
            self.logger.info("✅ Comprehensive API System fully initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize API system: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    async def _initialize_flask_api(self):
        """Initialize Flask application with REST and GraphQL"""
        if not FLASK_AVAILABLE:
            return
        
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'api_secret_key_2025'
        
        # Enable CORS if configured
        if self.config["cors_enabled"]:
            CORS(self.app, origins="*")
        
        # Initialize Flask-RESTful
        self.api = Api(self.app)
        
        # Add GraphQL endpoint if available
        if GRAPHQL_AVAILABLE and self.config["enable_graphql"]:
            self.app.add_url_rule(
                '/graphql',
                view_func=GraphQLView.as_view(
                    'graphql',
                    schema=schema,
                    graphiql=True  # Enable GraphiQL interface
                )
            )
        
        # Add middleware
        @self.app.before_request
        def before_request():
            self.request_count += 1
            request.start_time = time.time()
        
        @self.app.after_request
        def after_request(response):
            # Log request
            duration = time.time() - getattr(request, 'start_time', time.time())
            self.logger.info(f"{request.method} {request.path} - {response.status_code} ({duration:.3f}s)")
            return response
        
        @self.app.errorhandler(404)
        def not_found(error):
            self.error_count += 1
            return jsonify(APIResponse(
                success=False,
                error="Endpoint not found",
                message=f"The requested endpoint {request.path} was not found"
            ).__dict__), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            self.error_count += 1
            return jsonify(APIResponse(
                success=False,
                error="Internal server error",
                message="An unexpected error occurred"
            ).__dict__), 500
    
    async def _register_api_endpoints(self):
        """Register all API endpoints"""
        
        # Health check endpoint
        self.endpoints["/health"] = APIEndpoint(
            path="/health",
            method=HTTPMethod.GET,
            version=APIVersion.V1,
            description="System health check",
            parameters=[],
            response_schema={"status": "string", "timestamp": "datetime"},
            auth_required=False
        )
        
        # System endpoints
        self.endpoints["/api/v1/system/status"] = APIEndpoint(
            path="/api/v1/system/status",
            method=HTTPMethod.GET,
            version=APIVersion.V1,
            description="Get comprehensive system status",
            parameters=[],
            response_schema={"system": "object"},
            auth_required=False
        )
        
        # Workflow endpoints
        self.endpoints["/api/v1/workflows"] = APIEndpoint(
            path="/api/v1/workflows",
            method=HTTPMethod.GET,
            version=APIVersion.V1,
            description="List all workflows",
            parameters=[
                {"name": "status", "type": "string", "description": "Filter by status"},
                {"name": "limit", "type": "integer", "description": "Maximum results to return"}
            ],
            response_schema={"workflows": "array"},
            auth_required=False
        )
        
        self.endpoints["/api/v1/workflows/execute"] = APIEndpoint(
            path="/api/v1/workflows/execute",
            method=HTTPMethod.POST,
            version=APIVersion.V1,
            description="Execute a new workflow",
            parameters=[
                {"name": "workflow_type", "type": "string", "required": True},
                {"name": "context", "type": "object", "description": "Workflow context"}
            ],
            response_schema={"workflow_id": "string", "status": "string"},
            auth_required=False
        )
        
        # ML Analytics endpoints
        self.endpoints["/api/v1/analytics/insights"] = APIEndpoint(
            path="/api/v1/analytics/insights",
            method=HTTPMethod.GET,
            version=APIVersion.V1,
            description="Get ML predictive insights",
            parameters=[
                {"name": "type", "type": "string", "description": "Filter by prediction type"},
                {"name": "severity", "type": "string", "description": "Filter by severity"}
            ],
            response_schema={"insights": "array"},
            auth_required=False
        )
        
        self.endpoints["/api/v1/analytics/anomalies"] = APIEndpoint(
            path="/api/v1/analytics/anomalies",
            method=HTTPMethod.GET,
            version=APIVersion.V1,
            description="Get anomaly detection results",
            parameters=[],
            response_schema={"anomalies": "array"},
            auth_required=False
        )
        
        # Dashboard endpoints
        self.endpoints["/api/v1/dashboard/data"] = APIEndpoint(
            path="/api/v1/dashboard/data",
            method=HTTPMethod.GET,
            version=APIVersion.V1,
            description="Get dashboard data",
            parameters=[],
            response_schema={"dashboard": "object"},
            auth_required=False
        )
        
        # Register Flask resources if available
        if FLASK_AVAILABLE and self.api:
            self._register_flask_resources()
    
    def _register_flask_resources(self):
        """Register Flask-RESTful resources"""
        
        # Health Check Resource
        class HealthResource(Resource):
            def get(self):
                return APIResponse(
                    success=True,
                    data={
                        "status": "healthy",
                        "timestamp": datetime.now().isoformat(),
                        "version": "1.0.0",
                        "uptime_seconds": time.time() - getattr(self, 'start_time', time.time())
                    },
                    message="System is operational"
                ).__dict__
        
        # System Status Resource
        class SystemStatusResource(Resource):
            def __init__(self):
                self.api_system = None  # Will be set by parent
                
            async def get(self):
                try:
                    # Get system health from workflow system
                    health_data = {"overall_status": "unknown"}
                    if hasattr(self, 'api_system') and self.api_system.workflow_system:
                        health_data = await self.api_system.workflow_system.get_system_health()
                    
                    # Get ML analytics status
                    ml_status = {}
                    if hasattr(self, 'api_system') and self.api_system.ml_engine:
                        ml_status = await self.api_system.ml_engine.get_analytics_status()
                    
                    return APIResponse(
                        success=True,
                        data={
                            "system_health": health_data,
                            "ml_analytics": ml_status,
                            "api_metrics": {
                                "request_count": getattr(self, 'api_system', {}).request_count or 0,
                                "error_count": getattr(self, 'api_system', {}).error_count or 0
                            }
                        }
                    ).__dict__
                    
                except Exception as e:
                    return APIResponse(
                        success=False,
                        error=str(e),
                        message="Failed to retrieve system status"
                    ).__dict__, 500
            
            def get(self):
                # Sync wrapper for async method
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    return loop.run_until_complete(self.get())
                finally:
                    loop.close()
        
        # Workflows Resource
        class WorkflowsResource(Resource):
            def __init__(self):
                self.api_system = None
                
            def get(self):
                try:
                    # Get query parameters
                    status_filter = request.args.get('status')
                    limit = int(request.args.get('limit', 10))
                    
                    # Mock workflows data for now
                    workflows = [
                        {
                            "workflow_id": f"wf_{i:03d}",
                            "workflow_type": ["data_processing", "validation", "optimization"][i % 3],
                            "status": ["running", "completed", "failed"][i % 3],
                            "created_at": (datetime.now() - timedelta(hours=i)).isoformat(),
                            "duration_ms": (i + 1) * 1000
                        }
                        for i in range(min(limit, 20))
                    ]
                    
                    # Apply status filter if provided
                    if status_filter:
                        workflows = [wf for wf in workflows if wf["status"] == status_filter]
                    
                    return APIResponse(
                        success=True,
                        data={
                            "workflows": workflows,
                            "total_count": len(workflows),
                            "filters": {"status": status_filter, "limit": limit}
                        }
                    ).__dict__
                    
                except Exception as e:
                    return APIResponse(
                        success=False,
                        error=str(e),
                        message="Failed to retrieve workflows"
                    ).__dict__, 500
            
            def post(self):
                try:
                    # Get request data
                    data = request.get_json()
                    if not data:
                        return APIResponse(
                            success=False,
                            error="No data provided",
                            message="Request body must contain workflow parameters"
                        ).__dict__, 400
                    
                    workflow_type = data.get('workflow_type')
                    if not workflow_type:
                        return APIResponse(
                            success=False,
                            error="Missing workflow_type",
                            message="workflow_type is required"
                        ).__dict__, 400
                    
                    context = data.get('context', {})
                    
                    # Execute workflow (mock for now)
                    workflow_id = f"wf_{int(time.time())}"
                    
                    return APIResponse(
                        success=True,
                        data={
                            "workflow_id": workflow_id,
                            "workflow_type": workflow_type,
                            "status": "running",
                            "context": context,
                            "created_at": datetime.now().isoformat()
                        },
                        message="Workflow executed successfully"
                    ).__dict__, 201
                    
                except Exception as e:
                    return APIResponse(
                        success=False,
                        error=str(e),
                        message="Failed to execute workflow"
                    ).__dict__, 500
        
        # ML Analytics Resource
        class AnalyticsResource(Resource):
            def __init__(self):
                self.api_system = None
            
            def get(self):
                try:
                    # Get query parameters
                    insight_type = request.args.get('type')
                    severity_filter = request.args.get('severity')
                    
                    # Mock insights data
                    insights = [
                        {
                            "insight_id": f"insight_{i:03d}",
                            "prediction_type": ["anomaly_detection", "performance_forecasting", "failure_prediction"][i % 3],
                            "confidence_score": 0.7 + (i % 3) * 0.1,
                            "severity": ["LOW", "MEDIUM", "HIGH"][i % 3],
                            "recommended_actions": [f"Action {i+1}"],
                            "timestamp": (datetime.now() - timedelta(minutes=i*5)).isoformat()
                        }
                        for i in range(15)
                    ]
                    
                    # Apply filters
                    if insight_type:
                        insights = [ins for ins in insights if ins["prediction_type"] == insight_type]
                    if severity_filter:
                        insights = [ins for ins in insights if ins["severity"] == severity_filter]
                    
                    return APIResponse(
                        success=True,
                        data={
                            "insights": insights,
                            "total_count": len(insights),
                            "filters": {"type": insight_type, "severity": severity_filter}
                        }
                    ).__dict__
                    
                except Exception as e:
                    return APIResponse(
                        success=False,
                        error=str(e),
                        message="Failed to retrieve analytics insights"
                    ).__dict__, 500
        
        # Register resources with the API
        self.api.add_resource(HealthResource, '/health')
        self.api.add_resource(SystemStatusResource, '/api/v1/system/status')
        self.api.add_resource(WorkflowsResource, '/api/v1/workflows', '/api/v1/workflows/execute')
        self.api.add_resource(AnalyticsResource, '/api/v1/analytics/insights')
        
        # Set api_system reference for resources that need it
        for resource_class in [SystemStatusResource, WorkflowsResource, AnalyticsResource]:
            if hasattr(resource_class, '__init__'):
                # This is a simplification - in production, use dependency injection
                pass
    
    def generate_openapi_spec(self) -> Dict[str, Any]:
        """Generate OpenAPI/Swagger specification"""
        
        spec = {
            "openapi": "3.0.0",
            "info": {
                "title": "AG06 Production API",
                "description": "Comprehensive API for AG06 workflow system",
                "version": "1.0.0",
                "contact": {
                    "name": "AG06 API Support",
                    "url": f"http://localhost:{self.port}"
                }
            },
            "servers": [
                {
                    "url": f"http://localhost:{self.port}",
                    "description": "Development server"
                }
            ],
            "paths": {},
            "components": {
                "schemas": {
                    "APIResponse": {
                        "type": "object",
                        "properties": {
                            "success": {"type": "boolean"},
                            "data": {"type": "object"},
                            "error": {"type": "string"},
                            "message": {"type": "string"},
                            "timestamp": {"type": "string", "format": "date-time"},
                            "version": {"type": "string"}
                        }
                    }
                }
            }
        }
        
        # Add endpoint definitions
        for path, endpoint in self.endpoints.items():
            method = endpoint.method.value.lower()
            
            spec["paths"][path] = {
                method: {
                    "summary": endpoint.description,
                    "parameters": [
                        {
                            "name": param["name"],
                            "in": "query" if method == "get" else "body",
                            "schema": {"type": param["type"]},
                            "description": param.get("description", ""),
                            "required": param.get("required", False)
                        }
                        for param in endpoint.parameters
                    ],
                    "responses": {
                        "200": {
                            "description": "Success",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/APIResponse"}
                                }
                            }
                        }
                    }
                }
            }
        
        return spec
    
    async def run_api_server(self):
        """Run the API server"""
        
        if not FLASK_AVAILABLE:
            self.logger.warning("Flask not available - API server cannot start")
            return
        
        try:
            self.logger.info(f"🚀 Starting API server on port {self.port}...")
            
            # Add OpenAPI spec endpoint
            @self.app.route('/openapi.json')
            def openapi_spec():
                return jsonify(self.generate_openapi_spec())
            
            @self.app.route('/docs')
            def api_docs():
                return f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>AG06 API Documentation</title>
                    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@3.25.0/swagger-ui.css" />
                </head>
                <body>
                    <div id="swagger-ui"></div>
                    <script src="https://unpkg.com/swagger-ui-dist@3.25.0/swagger-ui-bundle.js"></script>
                    <script>
                        SwaggerUIBundle({{
                            url: '/openapi.json',
                            dom_id: '#swagger-ui',
                            presets: [
                                SwaggerUIBundle.presets.apis,
                                SwaggerUIBundle.presets.standalone
                            ]
                        }});
                    </script>
                </body>
                </html>
                """
            
            # Run the Flask server
            self.app.run(
                host='0.0.0.0',
                port=self.port,
                debug=False
            )
            
        except Exception as e:
            self.logger.error(f"Error running API server: {e}")
            self.logger.error(traceback.format_exc())
    
    async def get_api_status(self) -> Dict[str, Any]:
        """Get API system status"""
        
        return {
            "api_id": self.api_id,
            "status": "operational" if self.app else "not_started",
            "port": self.port,
            "endpoints": len(self.endpoints),
            "request_count": self.request_count,
            "error_count": self.error_count,
            "config": self.config,
            "capabilities": {
                "rest_api": FLASK_AVAILABLE,
                "graphql": GRAPHQL_AVAILABLE and self.config["enable_graphql"],
                "cors": self.config["cors_enabled"],
                "swagger_docs": self.config["enable_swagger"]
            }
        }
    
    async def run_api_demo(self) -> Dict[str, Any]:
        """Run API demonstration"""
        
        self.logger.info("🎯 Starting API system demonstration...")
        
        start_time = datetime.now()
        
        # Generate OpenAPI spec
        openapi_spec = self.generate_openapi_spec()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        demo_results = {
            "demo_summary": {
                "processing_time_seconds": processing_time,
                "endpoints_registered": len(self.endpoints),
                "openapi_spec_generated": True,
                "api_url": f"http://localhost:{self.port}",
                "docs_url": f"http://localhost:{self.port}/docs",
                "graphql_url": f"http://localhost:{self.port}/graphql" if GRAPHQL_AVAILABLE else None
            },
            "api_status": await self.get_api_status(),
            "endpoints": {path: asdict(endpoint) for path, endpoint in self.endpoints.items()},
            "openapi_spec": openapi_spec
        }
        
        self.logger.info(f"🎉 API demo complete - {len(self.endpoints)} endpoints registered")
        
        return demo_results

async def main():
    """Main entry point for Comprehensive API System"""
    print("🌐 Starting Comprehensive API System - Phase 2")
    print("=" * 80)
    
    # Initialize API system
    api_system = ComprehensiveAPISystem()
    
    if not await api_system.initialize():
        print("❌ Failed to initialize API system")
        return
    
    # Run demonstration
    demo_results = await api_system.run_api_demo()
    
    print("\n" + "=" * 80)
    print("📋 API System Demo Results:")
    print(f"   Endpoints Registered: {demo_results['demo_summary']['endpoints_registered']}")
    print(f"   Processing Time: {demo_results['demo_summary']['processing_time_seconds']:.2f}s")
    print(f"   API URL: {demo_results['demo_summary']['api_url']}")
    print(f"   Documentation: {demo_results['demo_summary']['docs_url']}")
    if demo_results['demo_summary']['graphql_url']:
        print(f"   GraphQL Interface: {demo_results['demo_summary']['graphql_url']}")
    
    # Show available endpoints
    print(f"\n📡 Available Endpoints:")
    for path, endpoint_data in demo_results["endpoints"].items():
        method = endpoint_data["method"]
        description = endpoint_data["description"]
        print(f"   • {method} {path} - {description}")
    
    # Export results
    results_file = "comprehensive_api_results.json"
    with open(results_file, "w") as f:
        json.dump(demo_results, f, indent=2, default=str)
    
    print(f"\n📄 Full results exported: {results_file}")
    
    if FLASK_AVAILABLE:
        print(f"\n🌐 API server can be started at: http://localhost:{api_system.port}")
        print("   Available features:")
        print(f"   • REST API endpoints: {len(api_system.endpoints)}")
        print(f"   • OpenAPI/Swagger docs: /docs")
        print(f"   • GraphQL interface: {'Yes (/graphql)' if GRAPHQL_AVAILABLE else 'No (install graphene)'}")
        print("   • CORS enabled for cross-origin requests")
        print("   • Comprehensive error handling")
        
        # Ask user if they want to start the server
        try:
            print("\nStart API server? (Press Ctrl+C to skip)")
            await asyncio.sleep(5)  # Give user time to cancel
            await api_system.run_api_server()
        except KeyboardInterrupt:
            print("\n⚠️  Skipping API server startup")
    else:
        print("\n⚠️  Flask not available - install Flask and Flask-RESTful for full functionality")
    
    print("\n✅ Comprehensive API System Phase 2 demonstration complete!")

if __name__ == "__main__":
    # Install dependencies if needed
    try:
        import flask
        import flask_restful
        import flask_cors
    except ImportError as e:
        print(f"⚠️  Missing API dependency: {e}")
        print("Installing required packages...")
        import subprocess
        packages = ["flask", "flask-restful", "flask-cors"]
        
        # Try to install GraphQL dependencies
        try:
            packages.extend(["graphene", "flask-graphql"])
        except:
            pass
            
        for package in packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ Installed {package}")
            except:
                print(f"❌ Failed to install {package} - continuing with fallbacks")
    
    asyncio.run(main())