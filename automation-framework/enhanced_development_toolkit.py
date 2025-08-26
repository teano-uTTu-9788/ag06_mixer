#!/usr/bin/env python3
"""
Enhanced Development Toolkit - 2025 Edition
Automated tool downloading, installation, and integration system

Features:
- Automated tool detection and installation
- Industry-standard development tool stack
- Integration with existing workflow systems
- Performance monitoring and optimization
- Version management and updates
"""

import asyncio
import json
import logging
import subprocess
import sys
import shutil
import platform
import os
import tempfile
import urllib.request
import zipfile
import tarfile
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DevelopmentTool:
    """Represents a development tool with installation details"""
    name: str
    category: str
    description: str
    installation_method: str  # pip, brew, npm, binary, git
    package_name: str
    version: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    post_install_commands: List[str] = field(default_factory=list)
    verification_command: str = ""
    importance: str = "medium"  # low, medium, high, critical
    platform_specific: Dict[str, Any] = field(default_factory=dict)

class ToolInstaller:
    """Handles installation of development tools"""
    
    def __init__(self):
        self.platform = platform.system().lower()
        self.installed_tools = set()
        self.failed_installations = []
        
        # Load tool definitions
        self.tools_catalog = self._load_tools_catalog()
    
    def _load_tools_catalog(self) -> Dict[str, DevelopmentTool]:
        """Load catalog of development tools"""
        tools = {}
        
        # AI/ML Development Tools
        tools['langchain'] = DevelopmentTool(
            name='LangChain',
            category='ai_frameworks',
            description='Framework for developing applications with LLMs',
            installation_method='pip',
            package_name='langchain',
            importance='high',
            verification_command='python -c "import langchain; print(langchain.__version__)"'
        )
        
        tools['langgraph'] = DevelopmentTool(
            name='LangGraph',
            category='ai_frameworks',
            description='Stateful multi-agent orchestration framework',
            installation_method='pip',
            package_name='langgraph',
            importance='critical',
            verification_command='python -c "import langgraph; print(langgraph.__version__)"'
        )
        
        tools['openai'] = DevelopmentTool(
            name='OpenAI Python SDK',
            category='ai_apis',
            description='Official OpenAI Python client library',
            installation_method='pip',
            package_name='openai',
            importance='high',
            verification_command='python -c "import openai; print(openai.__version__)"'
        )
        
        # Audio Processing Tools
        tools['sounddevice'] = DevelopmentTool(
            name='SoundDevice',
            category='audio',
            description='Audio I/O library for real-time audio processing',
            installation_method='pip',
            package_name='sounddevice',
            dependencies=['portaudio'],
            importance='critical',
            verification_command='python -c "import sounddevice; print(sounddevice.__version__)"'
        )
        
        tools['librosa'] = DevelopmentTool(
            name='Librosa',
            category='audio',
            description='Audio analysis library for music and audio analysis',
            installation_method='pip',
            package_name='librosa',
            importance='high',
            verification_command='python -c "import librosa; print(librosa.__version__)"'
        )
        
        tools['aubio'] = DevelopmentTool(
            name='Aubio',
            category='audio',
            description='Tools for audio and music analysis',
            installation_method='pip',
            package_name='aubio',
            importance='medium',
            verification_command='python -c "import aubio; print(aubio.version)"'
        )
        
        # Web Development Tools
        tools['flask'] = DevelopmentTool(
            name='Flask',
            category='web_frameworks',
            description='Lightweight WSGI web application framework',
            installation_method='pip',
            package_name='flask',
            importance='high',
            verification_command='python -c "import flask; print(flask.__version__)"'
        )
        
        tools['flask_socketio'] = DevelopmentTool(
            name='Flask-SocketIO',
            category='web_frameworks',
            description='WebSocket support for Flask applications',
            installation_method='pip',
            package_name='flask-socketio',
            dependencies=['flask'],
            importance='high',
            verification_command='python -c "import flask_socketio; print(flask_socketio.__version__)"'
        )
        
        # Scientific Computing
        tools['numpy'] = DevelopmentTool(
            name='NumPy',
            category='scientific',
            description='Fundamental package for scientific computing',
            installation_method='pip',
            package_name='numpy',
            importance='critical',
            verification_command='python -c "import numpy; print(numpy.__version__)"'
        )
        
        tools['scipy'] = DevelopmentTool(
            name='SciPy',
            category='scientific',
            description='Library for mathematics, science, and engineering',
            installation_method='pip',
            package_name='scipy',
            dependencies=['numpy'],
            importance='high',
            verification_command='python -c "import scipy; print(scipy.__version__)"'
        )
        
        tools['pandas'] = DevelopmentTool(
            name='Pandas',
            category='data_analysis',
            description='Data structures and analysis tools',
            installation_method='pip',
            package_name='pandas',
            dependencies=['numpy'],
            importance='high',
            verification_command='python -c "import pandas; print(pandas.__version__)"'
        )
        
        # Testing and Quality Tools
        tools['pytest'] = DevelopmentTool(
            name='PyTest',
            category='testing',
            description='Testing framework for Python',
            installation_method='pip',
            package_name='pytest',
            importance='high',
            verification_command='pytest --version'
        )
        
        tools['black'] = DevelopmentTool(
            name='Black',
            category='code_quality',
            description='Code formatter for Python',
            installation_method='pip',
            package_name='black',
            importance='medium',
            verification_command='black --version'
        )
        
        tools['mypy'] = DevelopmentTool(
            name='MyPy',
            category='code_quality',
            description='Static type checker for Python',
            installation_method='pip',
            package_name='mypy',
            importance='medium',
            verification_command='mypy --version'
        )
        
        # Development Utilities
        tools['rich'] = DevelopmentTool(
            name='Rich',
            category='utilities',
            description='Rich text and beautiful formatting for terminal',
            installation_method='pip',
            package_name='rich',
            importance='medium',
            verification_command='python -c "import rich; print(rich.__version__)"'
        )
        
        tools['click'] = DevelopmentTool(
            name='Click',
            category='utilities',
            description='Python package for creating command line interfaces',
            installation_method='pip',
            package_name='click',
            importance='medium',
            verification_command='python -c "import click; print(click.__version__)"'
        )
        
        # System-specific tools
        if self.platform == 'darwin':  # macOS
            tools['homebrew'] = DevelopmentTool(
                name='Homebrew',
                category='package_managers',
                description='Package manager for macOS',
                installation_method='binary',
                package_name='brew',
                importance='critical',
                verification_command='brew --version',
                platform_specific={'darwin': {'install_script': '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'}}
            )
            
            tools['portaudio'] = DevelopmentTool(
                name='PortAudio',
                category='system_libraries',
                description='Cross-platform audio I/O library',
                installation_method='brew',
                package_name='portaudio',
                importance='high',
                verification_command='brew list portaudio'
            )
        
        return tools
    
    async def check_tool_availability(self, tool: DevelopmentTool) -> bool:
        """Check if a tool is already installed"""
        if not tool.verification_command:
            return False
        
        try:
            process = await asyncio.create_subprocess_shell(
                tool.verification_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info(f"✅ {tool.name} is already installed")
                self.installed_tools.add(tool.name)
                return True
            else:
                return False
                
        except Exception as e:
            logger.debug(f"Tool check failed for {tool.name}: {e}")
            return False
    
    async def install_tool(self, tool: DevelopmentTool) -> bool:
        """Install a development tool"""
        logger.info(f"📦 Installing {tool.name}...")
        
        try:
            # Handle different installation methods
            if tool.installation_method == 'pip':
                success = await self._install_via_pip(tool)
            elif tool.installation_method == 'brew':
                success = await self._install_via_brew(tool)
            elif tool.installation_method == 'npm':
                success = await self._install_via_npm(tool)
            elif tool.installation_method == 'binary':
                success = await self._install_binary(tool)
            elif tool.installation_method == 'git':
                success = await self._install_via_git(tool)
            else:
                logger.error(f"Unsupported installation method: {tool.installation_method}")
                return False
            
            if success:
                # Run post-install commands
                if tool.post_install_commands:
                    for command in tool.post_install_commands:
                        await self._run_command(command)
                
                # Verify installation
                if await self.check_tool_availability(tool):
                    logger.info(f"✅ {tool.name} installed successfully")
                    return True
                else:
                    logger.error(f"❌ {tool.name} installation verification failed")
                    return False
            else:
                logger.error(f"❌ {tool.name} installation failed")
                self.failed_installations.append(tool.name)
                return False
                
        except Exception as e:
            logger.error(f"❌ Error installing {tool.name}: {e}")
            self.failed_installations.append(tool.name)
            return False
    
    async def _install_via_pip(self, tool: DevelopmentTool) -> bool:
        """Install tool via pip"""
        command = f"{sys.executable} -m pip install {tool.package_name}"
        if tool.version:
            command += f"=={tool.version}"
        
        return await self._run_command(command)
    
    async def _install_via_brew(self, tool: DevelopmentTool) -> bool:
        """Install tool via Homebrew"""
        if self.platform != 'darwin':
            logger.warning(f"Homebrew not available on {self.platform}")
            return False
        
        command = f"brew install {tool.package_name}"
        return await self._run_command(command)
    
    async def _install_via_npm(self, tool: DevelopmentTool) -> bool:
        """Install tool via npm"""
        command = f"npm install -g {tool.package_name}"
        if tool.version:
            command += f"@{tool.version}"
        
        return await self._run_command(command)
    
    async def _install_binary(self, tool: DevelopmentTool) -> bool:
        """Install binary tool"""
        platform_config = tool.platform_specific.get(self.platform, {})
        install_script = platform_config.get('install_script')
        
        if install_script:
            return await self._run_command(install_script)
        else:
            logger.error(f"No binary installation method for {tool.name} on {self.platform}")
            return False
    
    async def _install_via_git(self, tool: DevelopmentTool) -> bool:
        """Install tool via git"""
        temp_dir = tempfile.mkdtemp()
        try:
            clone_command = f"git clone {tool.package_name} {temp_dir}/repo"
            if not await self._run_command(clone_command):
                return False
            
            # Run installation from cloned repo
            install_command = f"cd {temp_dir}/repo && python setup.py install"
            return await self._run_command(install_command)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    async def _run_command(self, command: str) -> bool:
        """Run a shell command"""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return True
            else:
                logger.error(f"Command failed: {command}")
                if stderr:
                    logger.error(f"Error output: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return False
    
    async def install_tool_category(self, category: str, force_reinstall: bool = False) -> Dict[str, bool]:
        """Install all tools in a specific category"""
        results = {}
        category_tools = [tool for tool in self.tools_catalog.values() if tool.category == category]
        
        logger.info(f"Installing {len(category_tools)} tools in category '{category}'")
        
        for tool in category_tools:
            if not force_reinstall and await self.check_tool_availability(tool):
                results[tool.name] = True
                continue
            
            results[tool.name] = await self.install_tool(tool)
        
        return results
    
    async def install_essential_tools(self) -> Dict[str, bool]:
        """Install essential development tools"""
        essential_tools = [
            tool for tool in self.tools_catalog.values() 
            if tool.importance in ['critical', 'high']
        ]
        
        logger.info(f"Installing {len(essential_tools)} essential tools")
        results = {}
        
        # Sort by importance and dependencies
        essential_tools.sort(key=lambda t: (
            0 if t.importance == 'critical' else 1,
            len(t.dependencies)
        ))
        
        for tool in essential_tools:
            # Check and install dependencies first
            if tool.dependencies:
                for dep_name in tool.dependencies:
                    if dep_name in self.tools_catalog:
                        dep_tool = self.tools_catalog[dep_name]
                        if not await self.check_tool_availability(dep_tool):
                            await self.install_tool(dep_tool)
            
            # Install main tool
            if not await self.check_tool_availability(tool):
                results[tool.name] = await self.install_tool(tool)
            else:
                results[tool.name] = True
        
        return results
    
    def get_installation_report(self) -> Dict[str, Any]:
        """Get detailed installation report"""
        total_tools = len(self.tools_catalog)
        installed_count = len(self.installed_tools)
        failed_count = len(self.failed_installations)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'platform': self.platform,
            'total_tools': total_tools,
            'installed_tools': installed_count,
            'failed_installations': failed_count,
            'success_rate': (installed_count / total_tools) * 100 if total_tools > 0 else 0,
            'installed_tool_list': list(self.installed_tools),
            'failed_tool_list': self.failed_installations,
            'categories': self._get_category_summary()
        }
    
    def _get_category_summary(self) -> Dict[str, Dict[str, int]]:
        """Get installation summary by category"""
        categories = {}
        
        for tool in self.tools_catalog.values():
            if tool.category not in categories:
                categories[tool.category] = {'total': 0, 'installed': 0, 'failed': 0}
            
            categories[tool.category]['total'] += 1
            
            if tool.name in self.installed_tools:
                categories[tool.category]['installed'] += 1
            elif tool.name in self.failed_installations:
                categories[tool.category]['failed'] += 1
        
        return categories

class DevelopmentEnvironmentSetup:
    """Sets up complete development environment"""
    
    def __init__(self):
        self.installer = ToolInstaller()
        self.config_dir = Path.home() / '.ag06_dev_env'
        self.config_dir.mkdir(exist_ok=True)
    
    async def setup_complete_environment(self) -> Dict[str, Any]:
        """Set up complete development environment"""
        logger.info("🚀 Setting up complete development environment...")
        
        setup_results = {
            'started_at': datetime.now().isoformat(),
            'platform': platform.system(),
            'python_version': sys.version,
            'phases': {}
        }
        
        # Phase 1: Install system dependencies
        logger.info("Phase 1: Installing system dependencies...")
        system_results = await self._setup_system_dependencies()
        setup_results['phases']['system_dependencies'] = system_results
        
        # Phase 2: Install essential development tools
        logger.info("Phase 2: Installing essential development tools...")
        essential_results = await self.installer.install_essential_tools()
        setup_results['phases']['essential_tools'] = essential_results
        
        # Phase 3: Install AI/ML frameworks
        logger.info("Phase 3: Installing AI/ML frameworks...")
        ai_results = await self.installer.install_tool_category('ai_frameworks')
        setup_results['phases']['ai_frameworks'] = ai_results
        
        # Phase 4: Install audio processing tools
        logger.info("Phase 4: Installing audio processing tools...")
        audio_results = await self.installer.install_tool_category('audio')
        setup_results['phases']['audio_tools'] = audio_results
        
        # Phase 5: Install web development tools
        logger.info("Phase 5: Installing web development tools...")
        web_results = await self.installer.install_tool_category('web_frameworks')
        setup_results['phases']['web_frameworks'] = web_results
        
        # Phase 6: Configure development environment
        logger.info("Phase 6: Configuring development environment...")
        config_results = await self._configure_environment()
        setup_results['phases']['configuration'] = config_results
        
        # Phase 7: Verify installation
        logger.info("Phase 7: Verifying installation...")
        verification_results = await self._verify_environment()
        setup_results['phases']['verification'] = verification_results
        
        setup_results['completed_at'] = datetime.now().isoformat()
        setup_results['overall_success'] = verification_results.get('all_verified', False)
        
        # Save setup report
        report_file = self.config_dir / f'setup_report_{int(time.time())}.json'
        with open(report_file, 'w') as f:
            json.dump(setup_results, f, indent=2)
        
        return setup_results
    
    async def _setup_system_dependencies(self) -> Dict[str, Any]:
        """Set up system-level dependencies"""
        results = {'status': 'completed', 'actions': []}
        
        if platform.system() == 'Darwin':  # macOS
            # Install Homebrew if not present
            homebrew_tool = self.installer.tools_catalog.get('homebrew')
            if homebrew_tool and not await self.installer.check_tool_availability(homebrew_tool):
                success = await self.installer.install_tool(homebrew_tool)
                results['actions'].append({
                    'action': 'install_homebrew',
                    'success': success
                })
            
            # Install system audio libraries
            portaudio_tool = self.installer.tools_catalog.get('portaudio')
            if portaudio_tool:
                success = await self.installer.install_tool(portaudio_tool)
                results['actions'].append({
                    'action': 'install_portaudio',
                    'success': success
                })
        
        return results
    
    async def _configure_environment(self) -> Dict[str, Any]:
        """Configure development environment"""
        config_results = {'configurations': []}
        
        # Create virtual environment configuration
        venv_config = {
            'python_version': sys.version.split()[0],
            'pip_version': await self._get_pip_version(),
            'installed_packages': await self._get_installed_packages()
        }
        
        config_file = self.config_dir / 'environment.json'
        with open(config_file, 'w') as f:
            json.dump(venv_config, f, indent=2)
        
        config_results['configurations'].append({
            'type': 'virtual_environment',
            'file': str(config_file),
            'success': True
        })
        
        # Create development scripts
        scripts_dir = self.config_dir / 'scripts'
        scripts_dir.mkdir(exist_ok=True)
        
        # Create quick setup script
        setup_script = scripts_dir / 'quick_setup.sh'
        with open(setup_script, 'w') as f:
            f.write('''#!/bin/bash
# AG06 Development Environment Quick Setup
echo "🚀 AG06 Development Environment"
echo "Python: $(python3 --version)"
echo "Pip: $(pip3 --version | cut -d' ' -f1-2)"
echo ""
echo "Installed tools:"
pip3 list | grep -E "(langchain|langgraph|flask|numpy|sounddevice)"
echo ""
echo "✅ Environment ready!"
''')
        
        setup_script.chmod(0o755)
        
        config_results['configurations'].append({
            'type': 'setup_script',
            'file': str(setup_script),
            'success': True
        })
        
        return config_results
    
    async def _verify_environment(self) -> Dict[str, Any]:
        """Verify complete environment setup"""
        verification_results = {
            'tools_verified': {},
            'all_verified': True
        }
        
        # Verify essential tools
        essential_tools = [
            'numpy', 'scipy', 'flask', 'sounddevice', 'langchain', 'langgraph'
        ]
        
        for tool_name in essential_tools:
            if tool_name in self.installer.tools_catalog:
                tool = self.installer.tools_catalog[tool_name]
                is_available = await self.installer.check_tool_availability(tool)
                verification_results['tools_verified'][tool_name] = is_available
                
                if not is_available:
                    verification_results['all_verified'] = False
        
        # Test basic functionality
        try:
            import numpy
            import flask
            verification_results['numpy_test'] = True
            verification_results['flask_test'] = True
        except ImportError as e:
            verification_results['import_test_error'] = str(e)
            verification_results['all_verified'] = False
        
        return verification_results
    
    async def _get_pip_version(self) -> str:
        """Get pip version"""
        try:
            process = await asyncio.create_subprocess_shell(
                f"{sys.executable} -m pip --version",
                stdout=asyncio.subprocess.PIPE
            )
            stdout, _ = await process.communicate()
            return stdout.decode().strip().split()[1]
        except Exception:
            return "unknown"
    
    async def _get_installed_packages(self) -> List[str]:
        """Get list of installed packages"""
        try:
            process = await asyncio.create_subprocess_shell(
                f"{sys.executable} -m pip list --format=freeze",
                stdout=asyncio.subprocess.PIPE
            )
            stdout, _ = await process.communicate()
            packages = stdout.decode().strip().split('\n')
            return [pkg.split('==')[0] for pkg in packages if pkg]
        except Exception:
            return []

async def demonstrate_toolkit_setup():
    """Demonstrate the enhanced development toolkit"""
    print("🛠️  Enhanced Development Toolkit Demonstration")
    print("=" * 60)
    
    # Initialize environment setup
    env_setup = DevelopmentEnvironmentSetup()
    
    print("Starting complete environment setup...")
    print("This may take several minutes depending on your system...\n")
    
    # Run complete setup
    setup_results = await env_setup.setup_complete_environment()
    
    # Display results
    print("✅ Environment setup completed!")
    print(f"Platform: {setup_results['platform']}")
    print(f"Python: {setup_results['python_version'].split()[0]}")
    print(f"Overall success: {setup_results['overall_success']}")
    
    # Display phase results
    print("\n📊 Setup Phase Results:")
    for phase_name, phase_results in setup_results['phases'].items():
        if isinstance(phase_results, dict):
            if 'actions' in phase_results:
                successful_actions = sum(1 for action in phase_results['actions'] if action.get('success', False))
                total_actions = len(phase_results['actions'])
                print(f"  {phase_name}: {successful_actions}/{total_actions} successful")
            elif phase_results:  # Tool installation results
                successful = sum(1 for success in phase_results.values() if success)
                total = len(phase_results)
                print(f"  {phase_name}: {successful}/{total} tools installed")
    
    # Display installation report
    installation_report = env_setup.installer.get_installation_report()
    print(f"\n📈 Installation Summary:")
    print(f"  Total tools: {installation_report['total_tools']}")
    print(f"  Successfully installed: {installation_report['installed_tools']}")
    print(f"  Failed installations: {installation_report['failed_installations']}")
    print(f"  Success rate: {installation_report['success_rate']:.1f}%")
    
    # Display category breakdown
    if installation_report['categories']:
        print(f"\n📂 Category Breakdown:")
        for category, stats in installation_report['categories'].items():
            print(f"  {category}: {stats['installed']}/{stats['total']} installed")
    
    # Display failed installations if any
    if installation_report['failed_tool_list']:
        print(f"\n⚠️  Failed installations:")
        for failed_tool in installation_report['failed_tool_list']:
            print(f"  - {failed_tool}")
        print("These can be installed manually if needed.")
    
    return env_setup, setup_results

if __name__ == "__main__":
    asyncio.run(demonstrate_toolkit_setup())