#!/usr/bin/env python3
"""
Terminal Automation Framework for macOS
A CI/CD-compatible, modular terminal automation framework following best practices 
from Google, Meta, Microsoft, AWS, and Netflix.

Architecture inspired by multi-agent orchestration patterns with:
- Concurrent execution with semaphore control
- Cost/performance tracking
- Structured logging and error handling
- Pluggable automation modules
- Homebrew integration
- GitHub Actions compatibility
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml

# Configure structured logging following industry standards
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp":"%(asctime)s","level":"%(levelname)s","component":"%(name)s","message":"%(message)s"}',
    datefmt='%Y-%m-%dT%H:%M:%S'
)
logger = logging.getLogger('terminal-automation')


class AutomationError(Exception):
    """Custom exception for automation framework errors."""
    pass


@dataclass
class AutomationResponse:
    """Response container with metadata following enterprise patterns."""
    result: Any
    duration: float = 0.0
    cost: float = 0.0
    succeeded: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'result': self.result,
            'duration': self.duration,
            'cost': self.cost,
            'succeeded': self.succeeded,
            'error': self.error,
            'metadata': self.metadata
        }


class BaseAutomationModule:
    """Base class for automation modules following Google Bazel patterns."""
    
    def __init__(self, name: str, cost_per_operation: float = 0.01):
        self.name = name
        self.cost_per_operation = cost_per_operation
        
    async def execute(self, task: str, context: Dict[str, Any] = None) -> AutomationResponse:
        """Execute automation task with timing and error handling."""
        start_time = time.time()
        self._log_start(task)
        
        try:
            result = await self._execute_implementation(task, context or {})
            duration = time.time() - start_time
            response = AutomationResponse(
                result=result,
                duration=duration,
                cost=self.cost_per_operation,
                succeeded=True,
                metadata={'module': self.name, 'task': task}
            )
        except Exception as e:
            duration = time.time() - start_time
            response = AutomationResponse(
                result=None,
                duration=duration,
                cost=self.cost_per_operation,
                succeeded=False,
                error=str(e),
                metadata={'module': self.name, 'task': task}
            )
            
        self._log_end(response)
        return response
    
    async def _execute_implementation(self, task: str, context: Dict[str, Any]) -> Any:
        """Override in subclasses with specific implementation."""
        raise NotImplementedError(f"Module {self.name} must implement _execute_implementation")
    
    def _log_start(self, task: str) -> None:
        logger.info(f"{self.name}: Starting task → {task}")
        
    def _log_end(self, response: AutomationResponse) -> None:
        status = "succeeded" if response.succeeded else f"failed ({response.error})"
        logger.info(f"{self.name}: Completed {status} in {response.duration:.3f}s, cost: {response.cost:.3f}")


class HomebrewModule(BaseAutomationModule):
    """Homebrew package management automation following macOS best practices."""
    
    async def _execute_implementation(self, task: str, context: Dict[str, Any]) -> Any:
        if task.startswith('install'):
            package = task.replace('install ', '')
            return await self._install_package(package)
        elif task.startswith('update'):
            return await self._update_packages()
        elif task.startswith('cleanup'):
            return await self._cleanup()
        elif task.startswith('list'):
            return await self._list_installed()
        else:
            raise AutomationError(f"Unknown homebrew task: {task}")
    
    async def _install_package(self, package: str) -> Dict[str, Any]:
        """Install package with verification."""
        try:
            # Check if already installed
            check_result = await self._run_command(['brew', 'list', package])
            if check_result['returncode'] == 0:
                return {'status': 'already_installed', 'package': package}
            
            # Install package
            install_result = await self._run_command(['brew', 'install', package])
            if install_result['returncode'] == 0:
                return {'status': 'installed', 'package': package, 'output': install_result['stdout']}
            else:
                raise AutomationError(f"Installation failed: {install_result['stderr']}")
                
        except Exception as e:
            raise AutomationError(f"Homebrew install error: {e}")
    
    async def _update_packages(self) -> Dict[str, Any]:
        """Update all packages."""
        update_result = await self._run_command(['brew', 'update'])
        upgrade_result = await self._run_command(['brew', 'upgrade'])
        
        return {
            'update_status': update_result['returncode'],
            'upgrade_status': upgrade_result['returncode'],
            'updated_packages': self._parse_upgrade_output(upgrade_result['stdout'])
        }
    
    async def _cleanup(self) -> Dict[str, Any]:
        """Clean up old versions and cache."""
        cleanup_result = await self._run_command(['brew', 'cleanup', '--prune=all'])
        return {
            'status': cleanup_result['returncode'],
            'output': cleanup_result['stdout']
        }
    
    async def _list_installed(self) -> List[str]:
        """List installed packages."""
        list_result = await self._run_command(['brew', 'list', '--formula'])
        if list_result['returncode'] == 0:
            return list_result['stdout'].strip().split('\n') if list_result['stdout'].strip() else []
        return []
    
    def _parse_upgrade_output(self, output: str) -> List[str]:
        """Parse brew upgrade output to extract updated packages."""
        packages = []
        for line in output.split('\n'):
            if line.startswith('==> Upgrading'):
                package = line.split()[2] if len(line.split()) > 2 else 'unknown'
                packages.append(package)
        return packages
    
    async def _run_command(self, cmd: List[str]) -> Dict[str, Any]:
        """Run command with proper error handling."""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            return {
                'returncode': process.returncode,
                'stdout': stdout.decode('utf-8'),
                'stderr': stderr.decode('utf-8')
            }
        except Exception as e:
            raise AutomationError(f"Command execution failed: {e}")


class GitModule(BaseAutomationModule):
    """Git operations automation following Netflix deployment patterns."""
    
    async def _execute_implementation(self, task: str, context: Dict[str, Any]) -> Any:
        if task.startswith('clone'):
            repo_url = task.replace('clone ', '')
            return await self._clone_repository(repo_url, context.get('target_dir'))
        elif task.startswith('commit'):
            message = task.replace('commit ', '')
            return await self._commit_changes(message)
        elif task.startswith('push'):
            branch = task.replace('push ', '') or 'main'
            return await self._push_changes(branch)
        elif task.startswith('status'):
            return await self._get_status()
        else:
            raise AutomationError(f"Unknown git task: {task}")
    
    async def _clone_repository(self, repo_url: str, target_dir: Optional[str] = None) -> Dict[str, Any]:
        """Clone repository with validation."""
        cmd = ['git', 'clone', repo_url]
        if target_dir:
            cmd.append(target_dir)
            
        result = await self._run_command(cmd)
        if result['returncode'] == 0:
            return {'status': 'cloned', 'repository': repo_url, 'target': target_dir}
        else:
            raise AutomationError(f"Clone failed: {result['stderr']}")
    
    async def _commit_changes(self, message: str) -> Dict[str, Any]:
        """Commit changes with structured commit message."""
        # Add all changes
        add_result = await self._run_command(['git', 'add', '.'])
        if add_result['returncode'] != 0:
            raise AutomationError(f"Git add failed: {add_result['stderr']}")
        
        # Commit changes
        commit_result = await self._run_command(['git', 'commit', '-m', message])
        if commit_result['returncode'] == 0:
            return {'status': 'committed', 'message': message}
        else:
            # Check if nothing to commit
            if 'nothing to commit' in commit_result['stdout']:
                return {'status': 'no_changes', 'message': 'Working tree clean'}
            raise AutomationError(f"Commit failed: {commit_result['stderr']}")
    
    async def _push_changes(self, branch: str) -> Dict[str, Any]:
        """Push changes to remote."""
        push_result = await self._run_command(['git', 'push', 'origin', branch])
        if push_result['returncode'] == 0:
            return {'status': 'pushed', 'branch': branch}
        else:
            raise AutomationError(f"Push failed: {push_result['stderr']}")
    
    async def _get_status(self) -> Dict[str, Any]:
        """Get repository status."""
        status_result = await self._run_command(['git', 'status', '--porcelain'])
        log_result = await self._run_command(['git', 'log', '--oneline', '-5'])
        
        return {
            'modified_files': status_result['stdout'].strip().split('\n') if status_result['stdout'].strip() else [],
            'recent_commits': log_result['stdout'].strip().split('\n') if log_result['stdout'].strip() else [],
            'clean': not bool(status_result['stdout'].strip())
        }
    
    async def _run_command(self, cmd: List[str]) -> Dict[str, Any]:
        """Run git command with proper error handling."""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            return {
                'returncode': process.returncode,
                'stdout': stdout.decode('utf-8'),
                'stderr': stderr.decode('utf-8')
            }
        except Exception as e:
            raise AutomationError(f"Git command execution failed: {e}")


class DockerModule(BaseAutomationModule):
    """Docker containerization automation following cloud-native patterns."""
    
    async def _execute_implementation(self, task: str, context: Dict[str, Any]) -> Any:
        if task.startswith('build'):
            tag = task.replace('build ', '') or 'latest'
            return await self._build_image(tag, context.get('dockerfile_path', '.'))
        elif task.startswith('run'):
            image = task.replace('run ', '')
            return await self._run_container(image, context)
        elif task.startswith('ps'):
            return await self._list_containers()
        elif task.startswith('stop'):
            container = task.replace('stop ', '')
            return await self._stop_container(container)
        else:
            raise AutomationError(f"Unknown docker task: {task}")
    
    async def _build_image(self, tag: str, dockerfile_path: str) -> Dict[str, Any]:
        """Build Docker image."""
        cmd = ['docker', 'build', '-t', tag, dockerfile_path]
        result = await self._run_command(cmd)
        
        if result['returncode'] == 0:
            return {'status': 'built', 'tag': tag, 'path': dockerfile_path}
        else:
            raise AutomationError(f"Docker build failed: {result['stderr']}")
    
    async def _run_container(self, image: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run Docker container with configuration."""
        cmd = ['docker', 'run']
        
        # Add context-based options
        if context.get('detached', False):
            cmd.append('-d')
        if context.get('interactive', False):
            cmd.extend(['-i', '-t'])
        if context.get('ports'):
            for port_mapping in context['ports']:
                cmd.extend(['-p', port_mapping])
        if context.get('environment'):
            for env_var in context['environment']:
                cmd.extend(['-e', env_var])
        
        cmd.append(image)
        
        if context.get('command'):
            cmd.extend(context['command'].split())
        
        result = await self._run_command(cmd)
        
        if result['returncode'] == 0:
            container_id = result['stdout'].strip()[:12] if result['stdout'].strip() else 'unknown'
            return {'status': 'running', 'container_id': container_id, 'image': image}
        else:
            raise AutomationError(f"Docker run failed: {result['stderr']}")
    
    async def _list_containers(self) -> List[Dict[str, str]]:
        """List running containers."""
        result = await self._run_command(['docker', 'ps', '--format', 'json'])
        
        if result['returncode'] == 0 and result['stdout'].strip():
            containers = []
            for line in result['stdout'].strip().split('\n'):
                try:
                    container = json.loads(line)
                    containers.append({
                        'id': container.get('ID', ''),
                        'image': container.get('Image', ''),
                        'status': container.get('Status', ''),
                        'names': container.get('Names', '')
                    })
                except json.JSONDecodeError:
                    continue
            return containers
        return []
    
    async def _stop_container(self, container: str) -> Dict[str, Any]:
        """Stop Docker container."""
        result = await self._run_command(['docker', 'stop', container])
        
        if result['returncode'] == 0:
            return {'status': 'stopped', 'container': container}
        else:
            raise AutomationError(f"Docker stop failed: {result['stderr']}")
    
    async def _run_command(self, cmd: List[str]) -> Dict[str, Any]:
        """Run docker command with proper error handling."""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            return {
                'returncode': process.returncode,
                'stdout': stdout.decode('utf-8'),
                'stderr': stderr.decode('utf-8')
            }
        except Exception as e:
            raise AutomationError(f"Docker command execution failed: {e}")


class TestingModule(BaseAutomationModule):
    """Testing automation module following Google/Meta testing practices."""
    
    async def _execute_implementation(self, task: str, context: Dict[str, Any]) -> Any:
        if task.startswith('pytest'):
            test_path = task.replace('pytest ', '') or 'tests/'
            return await self._run_pytest(test_path, context)
        elif task.startswith('bats'):
            test_file = task.replace('bats ', '')
            return await self._run_bats(test_file)
        elif task.startswith('coverage'):
            return await self._run_coverage(context)
        else:
            raise AutomationError(f"Unknown testing task: {task}")
    
    async def _run_pytest(self, test_path: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run pytest with coverage and reporting."""
        cmd = ['python', '-m', 'pytest', test_path]
        
        # Add context-based options
        if context.get('verbose', False):
            cmd.append('-v')
        if context.get('coverage', False):
            cmd.extend(['--cov=.', '--cov-report=json'])
        if context.get('parallel', False):
            cmd.extend(['-n', str(context.get('parallel_workers', 4))])
        
        result = await self._run_command(cmd)
        
        return {
            'status': 'passed' if result['returncode'] == 0 else 'failed',
            'returncode': result['returncode'],
            'output': result['stdout'],
            'errors': result['stderr']
        }
    
    async def _run_bats(self, test_file: str) -> Dict[str, Any]:
        """Run BATS tests for shell scripts."""
        result = await self._run_command(['bats', test_file])
        
        return {
            'status': 'passed' if result['returncode'] == 0 else 'failed',
            'returncode': result['returncode'],
            'output': result['stdout'],
            'errors': result['stderr']
        }
    
    async def _run_coverage(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate coverage report."""
        format_type = context.get('format', 'term')
        cmd = ['python', '-m', 'coverage', 'report', f'--format={format_type}']
        
        result = await self._run_command(cmd)
        
        return {
            'status': 'generated' if result['returncode'] == 0 else 'failed',
            'output': result['stdout'],
            'format': format_type
        }
    
    async def _run_command(self, cmd: List[str]) -> Dict[str, Any]:
        """Run testing command with proper error handling."""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            return {
                'returncode': process.returncode,
                'stdout': stdout.decode('utf-8'),
                'stderr': stderr.decode('utf-8')
            }
        except Exception as e:
            raise AutomationError(f"Testing command execution failed: {e}")


class TerminalAutomationOrchestrator:
    """
    Main orchestrator following multi-agent patterns with enterprise features:
    - Concurrent execution with semaphore control
    - Cost tracking and performance monitoring  
    - Error handling with graceful degradation
    - Configuration-driven workflows
    - GitHub Actions compatibility
    """
    
    def __init__(self, config_path: Optional[str] = None, max_concurrent: int = 5):
        """Initialize orchestrator with configuration."""
        self.modules: Dict[str, BaseAutomationModule] = {}
        self.config = self._load_configuration(config_path)
        self.total_cost = 0.0
        self.execution_history: List[Dict[str, Any]] = []
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Initialize default modules
        self._initialize_default_modules()
        
    def _load_configuration(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not config_path:
            config_path = os.environ.get('AUTOMATION_CONFIG', 'automation.yaml')
            
        if not os.path.exists(config_path):
            logger.info(f"Configuration file {config_path} not found, using defaults")
            return self._get_default_config()
            
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'framework': {
                'version': '1.0.0',
                'max_concurrent': 5,
                'timeout': 300
            },
            'logging': {
                'level': 'INFO',
                'format': 'structured'
            },
            'modules': {
                'homebrew': {'enabled': True, 'cost_per_operation': 0.01},
                'git': {'enabled': True, 'cost_per_operation': 0.005},
                'docker': {'enabled': True, 'cost_per_operation': 0.02},
                'testing': {'enabled': True, 'cost_per_operation': 0.01}
            }
        }
    
    def _initialize_default_modules(self):
        """Initialize default automation modules."""
        module_config = self.config.get('modules', {})
        
        if module_config.get('homebrew', {}).get('enabled', True):
            cost = module_config.get('homebrew', {}).get('cost_per_operation', 0.01)
            self.modules['homebrew'] = HomebrewModule('HomebrewModule', cost)
            
        if module_config.get('git', {}).get('enabled', True):
            cost = module_config.get('git', {}).get('cost_per_operation', 0.005)
            self.modules['git'] = GitModule('GitModule', cost)
            
        if module_config.get('docker', {}).get('enabled', True):
            cost = module_config.get('docker', {}).get('cost_per_operation', 0.02)
            self.modules['docker'] = DockerModule('DockerModule', cost)
            
        if module_config.get('testing', {}).get('enabled', True):
            cost = module_config.get('testing', {}).get('cost_per_operation', 0.01)
            self.modules['testing'] = TestingModule('TestingModule', cost)
        
        logger.info(f"Initialized {len(self.modules)} automation modules: {list(self.modules.keys())}")
    
    def register_module(self, name: str, module: BaseAutomationModule):
        """Register custom automation module."""
        self.modules[name] = module
        logger.info(f"Registered custom module: {name}")
    
    async def execute_workflow(self, workflow_name: str, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute workflow with tasks following enterprise patterns.
        
        Args:
            workflow_name: Name of the workflow for tracking
            tasks: List of task dictionaries with format:
                   {'module': 'git', 'task': 'status', 'context': {...}}
        
        Returns:
            Dictionary with aggregated results, cost breakdown, and metadata
        """
        workflow_start = time.time()
        logger.info(f"Orchestrator: Starting workflow '{workflow_name}' with {len(tasks)} tasks")
        
        # Launch all tasks concurrently with semaphore control
        task_futures = []
        for i, task_config in enumerate(tasks):
            module_name = task_config.get('module')
            task_description = task_config.get('task')
            context = task_config.get('context', {})
            
            if module_name not in self.modules:
                raise AutomationError(f"Module '{module_name}' not registered")
            
            task_futures.append(
                self._execute_task(f"task_{i}", module_name, task_description, context)
            )
        
        # Wait for all tasks to complete
        responses = await asyncio.gather(*task_futures, return_exceptions=True)
        
        # Process results
        results = {}
        cost_breakdown = {}
        failed_tasks = []
        
        for i, response in enumerate(responses):
            task_key = f"task_{i}"
            
            if isinstance(response, Exception):
                logger.error(f"Task {task_key} raised exception: {response}")
                results[task_key] = {
                    'succeeded': False,
                    'error': str(response),
                    'result': None
                }
                failed_tasks.append(task_key)
            else:
                task_module, task_response = response
                results[task_key] = task_response.to_dict()
                cost_breakdown[task_key] = task_response.cost
                self.total_cost += task_response.cost
                
                if not task_response.succeeded:
                    failed_tasks.append(task_key)
        
        workflow_duration = time.time() - workflow_start
        
        # Create workflow summary
        workflow_result = {
            'workflow_name': workflow_name,
            'results': results,
            'cost_breakdown': cost_breakdown,
            'total_cost': self.total_cost,
            'duration': workflow_duration,
            'tasks_total': len(tasks),
            'tasks_succeeded': len(tasks) - len(failed_tasks),
            'tasks_failed': len(failed_tasks),
            'failed_tasks': failed_tasks,
            'timestamp': datetime.utcnow().isoformat(),
            'succeeded': len(failed_tasks) == 0
        }
        
        # Store in execution history
        self.execution_history.append(workflow_result)
        
        logger.info(
            f"Orchestrator: Workflow '{workflow_name}' completed in {workflow_duration:.3f}s, "
            f"cost: {self.total_cost:.3f}, success rate: {workflow_result['tasks_succeeded']}/{workflow_result['tasks_total']}"
        )
        
        return workflow_result
    
    async def _execute_task(
        self, 
        task_id: str, 
        module_name: str, 
        task_description: str, 
        context: Dict[str, Any]
    ) -> Tuple[str, AutomationResponse]:
        """Execute single task with concurrency control and error handling."""
        async with self.semaphore:
            try:
                module = self.modules[module_name]
                response = await module.execute(task_description, context)
                return module_name, response
            except Exception as e:
                logger.error(f"Failed to execute task {task_id} ({module_name}): {e}")
                error_response = AutomationResponse(
                    result=None,
                    succeeded=False,
                    error=str(e),
                    metadata={'task_id': task_id, 'module': module_name}
                )
                return module_name, error_response
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary for reporting."""
        if not self.execution_history:
            return {'total_workflows': 0, 'total_cost': 0.0}
            
        total_workflows = len(self.execution_history)
        successful_workflows = sum(1 for w in self.execution_history if w['succeeded'])
        
        return {
            'total_workflows': total_workflows,
            'successful_workflows': successful_workflows,
            'success_rate': (successful_workflows / total_workflows) * 100 if total_workflows > 0 else 0,
            'total_cost': self.total_cost,
            'average_cost_per_workflow': self.total_cost / total_workflows if total_workflows > 0 else 0,
            'total_tasks': sum(w['tasks_total'] for w in self.execution_history),
            'successful_tasks': sum(w['tasks_succeeded'] for w in self.execution_history),
            'recent_workflows': self.execution_history[-5:] if self.execution_history else []
        }
    
    def export_results(self, output_path: str = 'automation_results.json'):
        """Export execution results for CI/CD integration."""
        summary = self.get_execution_summary()
        summary['execution_history'] = self.execution_history
        summary['export_timestamp'] = datetime.utcnow().isoformat()
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Exported automation results to {output_path}")
        return output_path


# CLI Interface for standalone usage
async def main():
    """Main entry point for CLI usage."""
    if len(sys.argv) < 2:
        print("Usage: python terminal_automation_framework.py <workflow_file>")
        print("Example workflow file format (YAML):")
        print("""
workflows:
  - name: "setup-development"
    tasks:
      - module: "homebrew"
        task: "install git"
        context: {}
      - module: "git" 
        task: "clone https://github.com/example/repo.git"
        context:
          target_dir: "example-repo"
      - module: "testing"
        task: "pytest tests/"
        context:
          verbose: true
          coverage: true
        """)
        sys.exit(1)
    
    workflow_file = sys.argv[1]
    
    try:
        with open(workflow_file, 'r') as f:
            workflow_config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading workflow file: {e}")
        sys.exit(1)
    
    orchestrator = TerminalAutomationOrchestrator()
    
    for workflow in workflow_config.get('workflows', []):
        workflow_name = workflow['name']
        tasks = workflow['tasks']
        
        result = await orchestrator.execute_workflow(workflow_name, tasks)
        
        if result['succeeded']:
            print(f"✅ Workflow '{workflow_name}' completed successfully")
        else:
            print(f"❌ Workflow '{workflow_name}' failed")
            print(f"Failed tasks: {result['failed_tasks']}")
    
    # Export results
    orchestrator.export_results()
    
    # Print summary
    summary = orchestrator.get_execution_summary()
    print(f"\n📊 Execution Summary:")
    print(f"Total workflows: {summary['total_workflows']}")
    print(f"Success rate: {summary['success_rate']:.1f}%")
    print(f"Total cost: {summary['total_cost']:.3f}")


if __name__ == "__main__":
    asyncio.run(main())