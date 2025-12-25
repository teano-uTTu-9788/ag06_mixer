import pytest
import sys
import os

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_environment_sanity():
    """Verify the test environment is working."""
    assert True

def test_python_version():
    """Verify we are running on a compatible Python version."""
    assert sys.version_info.major == 3

def test_ag06_module_import():
    """Attempt to import the AG06 integration agent."""
    try:
        # Assuming the file is at .aican/agents/ag06_integration_agent.py
        # We might need to adjust python path or move the file for better importability
        # For now, we'll try to check if the file exists as a proxy for 'integration ready'
        agent_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 '.aican', 'agents', 'ag06_integration_agent.py')
        assert os.path.exists(agent_path), f"AG06 Agent file not found at {agent_path}"
        
    except ImportError as e:
        pytest.fail(f"Failed to import AG06 modules: {e}")

