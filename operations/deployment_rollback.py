"""
Deployment Rollback System for Production
MANU Compliance: Operations Requirements
"""
import os
import json
import shutil
import subprocess
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum


class DeploymentStatus(Enum):
    """Deployment status values"""
    PENDING = "pending"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"


@dataclass
class DeploymentRecord:
    """Record of a deployment"""
    deployment_id: str
    version: str
    timestamp: datetime
    status: DeploymentStatus
    commit_hash: Optional[str] = None
    config_snapshot: Optional[Dict[str, Any]] = None
    health_check_passed: bool = False
    rollback_available: bool = True
    backup_path: Optional[str] = None
    deployment_notes: Optional[str] = None


class HealthChecker:
    """
    Health check system for deployment validation
    """
    
    def __init__(self):
        """Initialize health checker"""
        self.checks = {}
        self.timeout = 30.0
        self._setup_default_checks()
    
    def _setup_default_checks(self):
        """Setup default health checks"""
        
        async def system_health():
            """Check system health"""
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            return {
                'status': 'healthy' if cpu_percent < 90 and memory_percent < 90 else 'unhealthy',
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent
            }
        
        async def audio_engine_health():
            """Check audio engine health"""
            try:
                # Check if audio engine can be imported and initialized
                from implementations.audio_engine import AG06AudioEngine
                
                # Basic initialization test
                engine = AG06AudioEngine()
                await engine.initialize()
                
                return {
                    'status': 'healthy',
                    'component': 'audio_engine',
                    'message': 'Audio engine initialized successfully'
                }
                
            except Exception as e:
                return {
                    'status': 'unhealthy',
                    'component': 'audio_engine',
                    'error': str(e)
                }
        
        async def optimization_agent_health():
            """Check optimization agent health"""
            try:
                status_file = Path('/Users/nguythe/ag06_mixer/ag06_optimization_status.json')
                if status_file.exists():
                    with open(status_file) as f:
                        data = json.load(f)
                    
                    is_running = data.get('running', False)
                    last_metric = data.get('last_metric', {})
                    
                    # Check if last metric is recent (within 5 minutes)
                    if last_metric.get('timestamp'):
                        last_time = datetime.fromisoformat(last_metric['timestamp'].replace('Z', '+00:00'))
                        time_diff = datetime.now().astimezone() - last_time.astimezone()
                        is_recent = time_diff < timedelta(minutes=5)
                    else:
                        is_recent = False
                    
                    return {
                        'status': 'healthy' if is_running and is_recent else 'unhealthy',
                        'component': 'optimization_agent',
                        'running': is_running,
                        'recent_activity': is_recent,
                        'optimizations': data.get('optimizations', 0)
                    }
                else:
                    return {
                        'status': 'unhealthy',
                        'component': 'optimization_agent',
                        'error': 'Status file not found'
                    }
                    
            except Exception as e:
                return {
                    'status': 'unhealthy',
                    'component': 'optimization_agent',
                    'error': str(e)
                }
        
        self.add_check('system', system_health)
        self.add_check('audio_engine', audio_engine_health)
        self.add_check('optimization_agent', optimization_agent_health)
    
    def add_check(self, name: str, check_func):
        """
        Add health check
        
        Args:
            name: Check name
            check_func: Async function that returns health status
        """
        self.checks[name] = check_func
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """
        Run all health checks
        
        Returns:
            Health check results
        """
        results = {}
        overall_healthy = True
        
        for check_name, check_func in self.checks.items():
            try:
                result = await asyncio.wait_for(check_func(), timeout=self.timeout)
                results[check_name] = result
                
                if result.get('status') != 'healthy':
                    overall_healthy = False
                    
            except asyncio.TimeoutError:
                results[check_name] = {
                    'status': 'unhealthy',
                    'error': 'Health check timed out'
                }
                overall_healthy = False
                
            except Exception as e:
                results[check_name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                overall_healthy = False
        
        return {
            'overall_status': 'healthy' if overall_healthy else 'unhealthy',
            'timestamp': datetime.utcnow().isoformat(),
            'checks': results
        }


class ConfigurationManager:
    """
    Configuration management for deployments
    """
    
    def __init__(self, config_dir: str = "/Users/nguythe/ag06_mixer/config"):
        """
        Initialize configuration manager
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.snapshots_dir = self.config_dir / "snapshots"
        self.snapshots_dir.mkdir(exist_ok=True)
    
    def create_config_snapshot(self, deployment_id: str) -> str:
        """
        Create configuration snapshot
        
        Args:
            deployment_id: Deployment identifier
            
        Returns:
            Path to configuration snapshot
        """
        snapshot_path = self.snapshots_dir / f"config_{deployment_id}.json"
        
        # Collect current configuration
        config_data = {
            'deployment_id': deployment_id,
            'timestamp': datetime.utcnow().isoformat(),
            'environment_variables': dict(os.environ),
            'system_info': self._get_system_info(),
            'file_checksums': self._get_file_checksums()
        }
        
        with open(snapshot_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        return str(snapshot_path)
    
    def restore_config_snapshot(self, snapshot_path: str) -> bool:
        """
        Restore configuration from snapshot
        
        Args:
            snapshot_path: Path to configuration snapshot
            
        Returns:
            True if restore successful
        """
        try:
            with open(snapshot_path) as f:
                config_data = json.load(f)
            
            # Restore critical environment variables
            critical_vars = [
                'DATABASE_URL', 'API_KEY', 'JWT_SECRET', 
                'ENCRYPTION_KEY', 'LOG_LEVEL'
            ]
            
            for var in critical_vars:
                if var in config_data['environment_variables']:
                    os.environ[var] = config_data['environment_variables'][var]
            
            return True
            
        except Exception as e:
            print(f"Failed to restore configuration: {e}")
            return False
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        import psutil
        
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'disk_total': psutil.disk_usage('/').total,
            'boot_time': psutil.boot_time()
        }
    
    def _get_file_checksums(self) -> Dict[str, str]:
        """Get checksums of important files"""
        import hashlib
        
        checksums = {}
        important_files = [
            'implementations/audio_engine.py',
            'implementations/midi_controller.py',
            'core/event_store_optimized.py'
        ]
        
        for file_path in important_files:
            full_path = Path('/Users/nguythe/ag06_mixer') / file_path
            if full_path.exists():
                with open(full_path, 'rb') as f:
                    content = f.read()
                    checksums[file_path] = hashlib.sha256(content).hexdigest()
        
        return checksums


class DeploymentRollbackManager:
    """
    Manages deployment rollbacks for the AG-06 mixer system
    """
    
    def __init__(self, 
                 deployment_dir: str = "/Users/nguythe/ag06_mixer",
                 backup_dir: str = "/Users/nguythe/ag06_mixer_backups"):
        """
        Initialize rollback manager
        
        Args:
            deployment_dir: Directory containing current deployment
            backup_dir: Directory for deployment backups
        """
        self.deployment_dir = Path(deployment_dir)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
        self.deployments_log = self.backup_dir / "deployments.json"
        self.deployments = self._load_deployments_log()
        
        self.health_checker = HealthChecker()
        self.config_manager = ConfigurationManager()
        
        # Maximum number of backups to keep
        self.max_backups = 10
    
    def _load_deployments_log(self) -> List[DeploymentRecord]:
        """Load deployments log"""
        if self.deployments_log.exists():
            try:
                with open(self.deployments_log) as f:
                    data = json.load(f)
                
                deployments = []
                for item in data.get('deployments', []):
                    # Convert timestamp string back to datetime
                    item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                    item['status'] = DeploymentStatus(item['status'])
                    deployments.append(DeploymentRecord(**item))
                
                return deployments
                
            except Exception as e:
                print(f"Failed to load deployments log: {e}")
        
        return []
    
    def _save_deployments_log(self):
        """Save deployments log"""
        data = {
            'deployments': [
                {**asdict(deployment), 
                 'timestamp': deployment.timestamp.isoformat(),
                 'status': deployment.status.value}
                for deployment in self.deployments
            ]
        }
        
        with open(self.deployments_log, 'w') as f:
            json.dump(data, f, indent=2)
    
    async def create_deployment_backup(self, 
                                     deployment_id: str, 
                                     version: str,
                                     notes: str = "") -> DeploymentRecord:
        """
        Create backup before deployment
        
        Args:
            deployment_id: Unique deployment identifier
            version: Version being deployed
            notes: Optional deployment notes
            
        Returns:
            Deployment record
        """
        backup_path = self.backup_dir / f"backup_{deployment_id}"
        
        # Create backup
        if backup_path.exists():
            shutil.rmtree(backup_path)
        
        shutil.copytree(self.deployment_dir, backup_path, 
                       ignore=shutil.ignore_patterns('*.pyc', '__pycache__', '.git'))
        
        # Create configuration snapshot
        config_snapshot_path = self.config_manager.create_config_snapshot(deployment_id)
        
        # Get current commit hash if available
        commit_hash = self._get_current_commit_hash()
        
        # Create deployment record
        deployment_record = DeploymentRecord(
            deployment_id=deployment_id,
            version=version,
            timestamp=datetime.utcnow(),
            status=DeploymentStatus.PENDING,
            commit_hash=commit_hash,
            config_snapshot={'path': config_snapshot_path},
            backup_path=str(backup_path),
            deployment_notes=notes
        )
        
        # Add to deployments list
        self.deployments.append(deployment_record)
        
        # Clean up old backups
        self._cleanup_old_backups()
        
        # Save deployments log
        self._save_deployments_log()
        
        return deployment_record
    
    async def validate_deployment(self, deployment_id: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate deployment health
        
        Args:
            deployment_id: Deployment identifier
            
        Returns:
            Tuple of (is_healthy, health_results)
        """
        # Find deployment record
        deployment = self._find_deployment(deployment_id)
        if not deployment:
            return False, {'error': 'Deployment not found'}
        
        # Run health checks
        health_results = await self.health_checker.run_all_checks()
        is_healthy = health_results.get('overall_status') == 'healthy'
        
        # Update deployment record
        deployment.health_check_passed = is_healthy
        deployment.status = DeploymentStatus.DEPLOYED if is_healthy else DeploymentStatus.FAILED
        
        self._save_deployments_log()
        
        return is_healthy, health_results
    
    async def rollback_deployment(self, deployment_id: str) -> Tuple[bool, str]:
        """
        Rollback to previous deployment
        
        Args:
            deployment_id: Deployment to rollback from
            
        Returns:
            Tuple of (success, message)
        """
        # Find deployment record
        deployment = self._find_deployment(deployment_id)
        if not deployment:
            return False, "Deployment not found"
        
        if not deployment.rollback_available:
            return False, "Rollback not available for this deployment"
        
        if not deployment.backup_path or not Path(deployment.backup_path).exists():
            return False, "Backup not found"
        
        try:
            # Update status
            deployment.status = DeploymentStatus.ROLLING_BACK
            self._save_deployments_log()
            
            # Create backup of current state before rollback
            current_backup = self.backup_dir / f"pre_rollback_{deployment_id}"
            if current_backup.exists():
                shutil.rmtree(current_backup)
            
            shutil.copytree(self.deployment_dir, current_backup,
                           ignore=shutil.ignore_patterns('*.pyc', '__pycache__', '.git'))
            
            # Restore from backup
            # Remove current deployment (keep backups)
            temp_dir = self.deployment_dir.parent / "temp_deployment"
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            
            shutil.move(self.deployment_dir, temp_dir)
            
            # Restore backup
            shutil.copytree(deployment.backup_path, self.deployment_dir)
            
            # Restore configuration
            if deployment.config_snapshot:
                config_path = deployment.config_snapshot.get('path')
                if config_path:
                    self.config_manager.restore_config_snapshot(config_path)
            
            # Validate rollback
            await asyncio.sleep(2)  # Give system time to stabilize
            health_results = await self.health_checker.run_all_checks()
            
            if health_results.get('overall_status') == 'healthy':
                # Rollback successful
                deployment.status = DeploymentStatus.ROLLED_BACK
                shutil.rmtree(temp_dir)  # Clean up
                
                message = f"Rollback successful for deployment {deployment_id}"
                
            else:
                # Rollback failed, restore current deployment
                shutil.rmtree(self.deployment_dir)
                shutil.move(temp_dir, self.deployment_dir)
                deployment.status = DeploymentStatus.FAILED
                
                message = f"Rollback failed for deployment {deployment_id}: {health_results}"
            
            self._save_deployments_log()
            
            return health_results.get('overall_status') == 'healthy', message
            
        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            self._save_deployments_log()
            return False, f"Rollback error: {str(e)}"
    
    def get_deployment_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get deployment history
        
        Args:
            limit: Maximum number of deployments to return
            
        Returns:
            List of deployment records
        """
        recent_deployments = sorted(self.deployments, 
                                  key=lambda d: d.timestamp, 
                                  reverse=True)[:limit]
        
        return [
            {
                'deployment_id': d.deployment_id,
                'version': d.version,
                'timestamp': d.timestamp.isoformat(),
                'status': d.status.value,
                'health_check_passed': d.health_check_passed,
                'rollback_available': d.rollback_available,
                'commit_hash': d.commit_hash,
                'deployment_notes': d.deployment_notes
            }
            for d in recent_deployments
        ]
    
    def _find_deployment(self, deployment_id: str) -> Optional[DeploymentRecord]:
        """Find deployment by ID"""
        for deployment in self.deployments:
            if deployment.deployment_id == deployment_id:
                return deployment
        return None
    
    def _get_current_commit_hash(self) -> Optional[str]:
        """Get current git commit hash"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=self.deployment_dir,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    def _cleanup_old_backups(self):
        """Remove old backups beyond retention limit"""
        if len(self.deployments) > self.max_backups:
            old_deployments = sorted(self.deployments, key=lambda d: d.timestamp)[:-self.max_backups]
            
            for deployment in old_deployments:
                if deployment.backup_path and Path(deployment.backup_path).exists():
                    try:
                        shutil.rmtree(deployment.backup_path)
                        deployment.rollback_available = False
                    except Exception as e:
                        print(f"Failed to cleanup backup {deployment.backup_path}: {e}")


# Create global rollback manager
rollback_manager = DeploymentRollbackManager()

# Export deployment components
__all__ = [
    'DeploymentStatus',
    'DeploymentRecord',
    'HealthChecker',
    'ConfigurationManager', 
    'DeploymentRollbackManager',
    'rollback_manager'
]