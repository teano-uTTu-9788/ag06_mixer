"""AGMixer - Real smoke tests (no fake success)"""
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_project_structure():
    """Test that essential project files exist"""
    assert (project_root / "main.py").exists()
    assert (project_root / "interfaces").exists()
    assert (project_root / "implementations").exists()
    assert (project_root / "core").exists()

def test_basic_imports():
    """Test basic module imports work"""
    # Test interface imports
    from interfaces.audio_engine import IAudioEngine
    assert IAudioEngine is not None
    
    # Test implementation imports
    from implementations.audio_engine import AudioEngineImpl
    assert AudioEngineImpl is not None

def test_main_module_structure():
    """Test main module has required components"""
    import main
    assert hasattr(main, 'AG06MixerApplication')
    assert hasattr(main, 'main')

def test_manu_workflow_basic():
    """Test MANU workflow basic import"""
    import ag06_manu_workflow
    assert hasattr(ag06_manu_workflow, 'AG06WorkflowFactory')

def test_web_app_basic():
    """Test web app basic import"""
    import web_app
    assert hasattr(web_app, 'AG06WebApp')
    assert hasattr(web_app, 'main')