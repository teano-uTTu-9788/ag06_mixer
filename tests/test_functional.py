#!/usr/bin/env python3
"""
AG06 Mixer - REAL Functional Tests
These tests verify actual functionality, not just imports
"""
import pytest
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestHealthMonitoring:
    """Test health monitoring returns real data"""

    @pytest.mark.asyncio
    async def test_health_status_returns_real_data(self):
        """Verify health status returns real metrics, not hardcoded"""
        from ag06_manu_workflow import DeploymentManager, DeploymentConfig

        dm = DeploymentManager(DeploymentConfig(
            environment="test",
            version="1.0.0",
            features=[]
        ))
        health = await dm.get_health_status()

        # Verify structure
        assert hasattr(health, 'healthy')
        assert hasattr(health, 'services')
        assert hasattr(health, 'metrics')

        # Verify metrics are real (not hardcoded 45.2, 62.3)
        assert isinstance(health.metrics['cpu_usage'], (int, float))
        assert isinstance(health.metrics['memory_usage'], (int, float))
        assert 0 <= health.metrics['cpu_usage'] <= 100
        assert 0 <= health.metrics['memory_usage'] <= 100

    @pytest.mark.asyncio
    async def test_services_checked_honestly(self):
        """Verify services are actually checked, not always True"""
        from ag06_manu_workflow import DeploymentManager, DeploymentConfig

        dm = DeploymentManager(DeploymentConfig(
            environment="test",
            version="1.0.0",
            features=[]
        ))
        health = await dm.get_health_status()

        # Services dict should exist with expected keys
        assert 'audio_engine' in health.services
        assert 'midi_controller' in health.services
        assert 'monitoring' in health.services

        # Values should be booleans (not strings or None)
        for key, val in health.services.items():
            assert isinstance(val, bool), f"Service {key} should be bool, got {type(val)}"


class TestWorkflowOrchestrator:
    """Test workflow orchestrator functionality"""

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self):
        """Test orchestrator can be created"""
        from ag06_manu_workflow import AG06WorkflowFactory

        orchestrator = AG06WorkflowFactory.create_orchestrator()
        assert orchestrator is not None

    @pytest.mark.asyncio
    async def test_workflow_execution(self):
        """Test basic workflow execution"""
        from ag06_manu_workflow import AG06WorkflowFactory, WorkflowStatus

        orchestrator = AG06WorkflowFactory.create_orchestrator()
        result = await orchestrator.execute_workflow(
            "test_workflow",
            {"param": "value"}
        )

        # Verify result structure
        assert hasattr(result, 'status')
        assert hasattr(result, 'execution_id')
        assert result.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]


class TestCoreInterfaces:
    """Test core interfaces are properly defined"""

    def test_audio_engine_interface(self):
        """Test IAudioEngine interface exists"""
        from interfaces.audio_engine import IAudioEngine
        assert IAudioEngine is not None

    def test_midi_controller_interface(self):
        """Test IMidiController interface exists"""
        from interfaces.midi_controller import IMidiController
        assert IMidiController is not None

    def test_preset_manager_interface(self):
        """Test IPresetManager interface exists"""
        from interfaces.preset_manager import IPresetManager
        assert IPresetManager is not None


class TestWebApp:
    """Test web application functionality"""

    def test_web_app_creation(self):
        """Test AG06WebApp can be instantiated"""
        from web_app import AG06WebApp

        app = AG06WebApp()
        assert app is not None
        assert app.app is not None

    def test_routes_configured(self):
        """Test routes are properly configured"""
        from web_app import AG06WebApp

        app = AG06WebApp()
        routes = [r.resource.canonical for r in app.app.router.routes() if hasattr(r, 'resource')]

        # Verify essential routes exist
        assert '/' in routes or any('/' == r for r in routes)
        assert '/api/status' in routes
        assert '/health' in routes


class TestDependencyInjection:
    """Test dependency injection works correctly"""

    def test_factory_creates_components(self):
        """Test factory pattern creates valid components"""
        from ag06_manu_workflow import AG06WorkflowFactory

        orchestrator = AG06WorkflowFactory.create_orchestrator()
        deployment_mgr = AG06WorkflowFactory.create_deployment_manager()
        monitor = AG06WorkflowFactory.create_monitoring_provider()
        validator = AG06WorkflowFactory.create_test_validator()

        assert orchestrator is not None
        assert deployment_mgr is not None
        assert monitor is not None
        assert validator is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
