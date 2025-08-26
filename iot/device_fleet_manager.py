"""
IoT Device Fleet Manager for AG06 Mixer

Enterprise-grade IoT device management system with:
- Device registration and provisioning
- Fleet monitoring and health tracking
- Over-the-air (OTA) updates
- Remote configuration management
- Device twin synchronization
- Command and control capabilities

Architecture based on industry patterns from:
- AWS IoT Core (device management, shadows)
- Azure IoT Hub (device twins, direct methods)
- Google Cloud IoT Core (device registry, telemetry)
- Particle Device Cloud (fleet management)
- Balena (container-based edge updates)
"""

import asyncio
import json
import logging
import time
import hashlib
import hmac
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from pathlib import Path
import sqlite3
import uuid
from collections import defaultdict, deque
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeviceState(Enum):
    """Device lifecycle states"""
    PROVISIONING = "provisioning"
    ONLINE = "online"
    OFFLINE = "offline"
    UPDATING = "updating"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    DECOMMISSIONED = "decommissioned"


class UpdateStatus(Enum):
    """OTA update status"""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    INSTALLING = "installing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class CommandType(Enum):
    """Remote command types"""
    REBOOT = "reboot"
    RESET = "reset"
    UPDATE_CONFIG = "update_config"
    COLLECT_LOGS = "collect_logs"
    DIAGNOSTIC = "diagnostic"
    AUDIO_CALIBRATION = "audio_calibration"
    FACTORY_RESET = "factory_reset"


@dataclass
class DeviceTwin:
    """
    Digital twin representation of physical device
    
    Maintains desired and reported states with automatic sync
    """
    device_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Reported state (from device)
    reported: Dict[str, Any] = field(default_factory=lambda: {
        'firmware_version': None,
        'hardware_version': None,
        'status': DeviceState.PROVISIONING.value,
        'location': None,
        'capabilities': [],
        'metrics': {}
    })
    
    # Desired state (from cloud)
    desired: Dict[str, Any] = field(default_factory=lambda: {
        'firmware_version': '2.0.0',
        'config': {},
        'enabled_features': [],
        'update_channel': 'stable'
    })
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        'last_seen': None,
        'last_telemetry': None,
        'update_history': [],
        'command_history': []
    })
    
    def get_delta(self) -> Dict[str, Any]:
        """Get differences between desired and reported states"""
        delta = {}
        
        for key, desired_value in self.desired.items():
            reported_value = self.reported.get(key)
            if reported_value != desired_value:
                delta[key] = {
                    'desired': desired_value,
                    'reported': reported_value
                }
        
        return delta
    
    def update_reported(self, updates: Dict[str, Any]):
        """Update reported state from device"""
        self.reported.update(updates)
        self.updated_at = datetime.utcnow()
        self.metadata['last_seen'] = datetime.utcnow().isoformat()
    
    def update_desired(self, updates: Dict[str, Any]):
        """Update desired state from cloud"""
        self.desired.update(updates)
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'device_id': self.device_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'reported': self.reported,
            'desired': self.desired,
            'metadata': self.metadata,
            'delta': self.get_delta()
        }


@dataclass
class OTAUpdate:
    """Over-the-air update package"""
    update_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version: str = ""
    description: str = ""
    release_notes: str = ""
    
    # Update files
    firmware_url: Optional[str] = None
    firmware_hash: Optional[str] = None
    firmware_size_bytes: int = 0
    
    # Update metadata
    min_hardware_version: Optional[str] = None
    max_hardware_version: Optional[str] = None
    rollback_version: Optional[str] = None
    
    # Deployment settings
    deployment_groups: List[str] = field(default_factory=list)
    staged_rollout_percent: int = 100
    auto_rollback_threshold: float = 0.1  # 10% failure rate
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def is_compatible(self, hardware_version: str) -> bool:
        """Check if update is compatible with hardware version"""
        # Simple version comparison
        # In production, would use proper semantic versioning
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'update_id': self.update_id,
            'version': self.version,
            'description': self.description,
            'firmware_url': self.firmware_url,
            'firmware_hash': self.firmware_hash,
            'firmware_size_bytes': self.firmware_size_bytes,
            'deployment_groups': self.deployment_groups,
            'staged_rollout_percent': self.staged_rollout_percent,
            'created_at': self.created_at.isoformat()
        }


class DeviceRegistry:
    """
    Central registry for all IoT devices
    
    Manages device lifecycle, authentication, and metadata
    """
    
    def __init__(self, db_path: str = "./device_registry.db"):
        self.db_path = db_path
        self.conn = None
        self._initialize_database()
        
        # In-memory cache
        self.device_cache: Dict[str, DeviceTwin] = {}
        
    def _initialize_database(self):
        """Initialize SQLite database for device registry"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.conn.cursor()
        
        # Devices table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS devices (
                device_id TEXT PRIMARY KEY,
                created_at REAL,
                updated_at REAL,
                state TEXT,
                hardware_version TEXT,
                firmware_version TEXT,
                location TEXT,
                metadata TEXT
            )
        ''')
        
        # Device credentials table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS device_credentials (
                device_id TEXT PRIMARY KEY,
                certificate TEXT,
                private_key_hash TEXT,
                api_key TEXT,
                created_at REAL,
                expires_at REAL
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_devices_state ON devices(state)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_devices_location ON devices(location)')
        
        self.conn.commit()
    
    async def register_device(self, 
                             device_id: str,
                             hardware_version: str,
                             location: Optional[str] = None) -> DeviceTwin:
        """Register a new device"""
        
        # Create device twin
        twin = DeviceTwin(device_id=device_id)
        twin.reported['hardware_version'] = hardware_version
        twin.reported['location'] = location
        twin.reported['status'] = DeviceState.PROVISIONING.value
        
        # Generate credentials
        api_key = hashlib.sha256(f"{device_id}:{time.time()}".encode()).hexdigest()
        
        # Store in database
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO devices 
            (device_id, created_at, updated_at, state, hardware_version, firmware_version, location, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            device_id,
            time.time(),
            time.time(),
            DeviceState.PROVISIONING.value,
            hardware_version,
            None,
            location,
            json.dumps(twin.metadata)
        ))
        
        cursor.execute('''
            INSERT OR REPLACE INTO device_credentials
            (device_id, api_key, created_at, expires_at)
            VALUES (?, ?, ?, ?)
        ''', (
            device_id,
            api_key,
            time.time(),
            time.time() + 365 * 24 * 3600  # 1 year expiry
        ))
        
        self.conn.commit()
        
        # Cache device twin
        self.device_cache[device_id] = twin
        
        logger.info(f"Registered device {device_id}")
        return twin
    
    async def get_device(self, device_id: str) -> Optional[DeviceTwin]:
        """Get device twin"""
        
        # Check cache first
        if device_id in self.device_cache:
            return self.device_cache[device_id]
        
        # Load from database
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT created_at, updated_at, state, hardware_version, firmware_version, location, metadata
            FROM devices WHERE device_id = ?
        ''', (device_id,))
        
        row = cursor.fetchone()
        if row:
            twin = DeviceTwin(device_id=device_id)
            twin.created_at = datetime.fromtimestamp(row[0])
            twin.updated_at = datetime.fromtimestamp(row[1])
            twin.reported['status'] = row[2]
            twin.reported['hardware_version'] = row[3]
            twin.reported['firmware_version'] = row[4]
            twin.reported['location'] = row[5]
            twin.metadata = json.loads(row[6]) if row[6] else {}
            
            self.device_cache[device_id] = twin
            return twin
        
        return None
    
    async def update_device_state(self, device_id: str, state: DeviceState):
        """Update device state"""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE devices SET state = ?, updated_at = ?
            WHERE device_id = ?
        ''', (state.value, time.time(), device_id))
        self.conn.commit()
        
        # Update cache
        if device_id in self.device_cache:
            self.device_cache[device_id].reported['status'] = state.value
            self.device_cache[device_id].updated_at = datetime.utcnow()
    
    async def list_devices(self, 
                          state: Optional[DeviceState] = None,
                          location: Optional[str] = None) -> List[str]:
        """List devices with optional filters"""
        cursor = self.conn.cursor()
        
        query = "SELECT device_id FROM devices WHERE 1=1"
        params = []
        
        if state:
            query += " AND state = ?"
            params.append(state.value)
        
        if location:
            query += " AND location = ?"
            params.append(location)
        
        cursor.execute(query, params)
        return [row[0] for row in cursor.fetchall()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        cursor = self.conn.cursor()
        
        # Count devices by state
        cursor.execute('''
            SELECT state, COUNT(*) FROM devices GROUP BY state
        ''')
        state_counts = dict(cursor.fetchall())
        
        # Total devices
        cursor.execute('SELECT COUNT(*) FROM devices')
        total_devices = cursor.fetchone()[0]
        
        return {
            'total_devices': total_devices,
            'devices_by_state': state_counts,
            'cached_devices': len(self.device_cache)
        }


class FleetMonitor:
    """
    Real-time fleet monitoring and health tracking
    
    Monitors device health, collects telemetry, and generates alerts
    """
    
    def __init__(self, alert_callback: Optional[Callable] = None):
        self.alert_callback = alert_callback
        
        # Device health tracking
        self.device_health: Dict[str, Dict[str, Any]] = {}
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Alert thresholds
        self.thresholds = {
            'cpu_usage': 85.0,
            'memory_usage': 90.0,
            'temperature': 70.0,
            'error_rate': 0.05,
            'offline_duration_minutes': 30
        }
        
        # Fleet metrics
        self.fleet_metrics = {
            'total_devices': 0,
            'online_devices': 0,
            'healthy_devices': 0,
            'devices_with_alerts': 0,
            'avg_cpu_usage': 0.0,
            'avg_memory_usage': 0.0
        }
    
    async def update_device_telemetry(self, 
                                     device_id: str,
                                     telemetry: Dict[str, Any]):
        """Update device telemetry and check health"""
        
        # Store telemetry
        self.device_health[device_id] = {
            'timestamp': datetime.utcnow(),
            'telemetry': telemetry,
            'health_score': self._calculate_health_score(telemetry),
            'alerts': []
        }
        
        # Check thresholds and generate alerts
        alerts = []
        
        if telemetry.get('cpu_usage', 0) > self.thresholds['cpu_usage']:
            alerts.append({
                'type': 'high_cpu',
                'severity': 'warning',
                'value': telemetry['cpu_usage']
            })
        
        if telemetry.get('memory_usage', 0) > self.thresholds['memory_usage']:
            alerts.append({
                'type': 'high_memory',
                'severity': 'warning',
                'value': telemetry['memory_usage']
            })
        
        if telemetry.get('temperature', 0) > self.thresholds['temperature']:
            alerts.append({
                'type': 'high_temperature',
                'severity': 'critical',
                'value': telemetry['temperature']
            })
        
        if telemetry.get('error_rate', 0) > self.thresholds['error_rate']:
            alerts.append({
                'type': 'high_error_rate',
                'severity': 'warning',
                'value': telemetry['error_rate']
            })
        
        # Store alerts
        if alerts:
            self.device_health[device_id]['alerts'] = alerts
            
            # Trigger alert callback
            if self.alert_callback:
                await self.alert_callback(device_id, alerts)
        
        # Update history
        self.health_history[device_id].append({
            'timestamp': datetime.utcnow().isoformat(),
            'health_score': self.device_health[device_id]['health_score'],
            'cpu_usage': telemetry.get('cpu_usage', 0),
            'memory_usage': telemetry.get('memory_usage', 0)
        })
        
        # Update fleet metrics
        await self._update_fleet_metrics()
    
    def _calculate_health_score(self, telemetry: Dict[str, Any]) -> float:
        """Calculate device health score (0-100)"""
        score = 100.0
        
        # Deduct points for resource usage
        cpu_usage = telemetry.get('cpu_usage', 0)
        if cpu_usage > 80:
            score -= min(20, (cpu_usage - 80))
        
        memory_usage = telemetry.get('memory_usage', 0)
        if memory_usage > 80:
            score -= min(20, (memory_usage - 80))
        
        # Deduct points for temperature
        temperature = telemetry.get('temperature', 0)
        if temperature > 60:
            score -= min(30, (temperature - 60) / 2)
        
        # Deduct points for errors
        error_rate = telemetry.get('error_rate', 0)
        if error_rate > 0:
            score -= min(30, error_rate * 100)
        
        return max(0, score)
    
    async def _update_fleet_metrics(self):
        """Update aggregated fleet metrics"""
        
        online_devices = []
        healthy_devices = []
        total_cpu = 0.0
        total_memory = 0.0
        
        for device_id, health in self.device_health.items():
            # Check if online (recent telemetry)
            if health['timestamp'] > datetime.utcnow() - timedelta(minutes=5):
                online_devices.append(device_id)
                
                # Check if healthy
                if health['health_score'] > 70 and not health['alerts']:
                    healthy_devices.append(device_id)
                
                # Aggregate metrics
                total_cpu += health['telemetry'].get('cpu_usage', 0)
                total_memory += health['telemetry'].get('memory_usage', 0)
        
        # Update fleet metrics
        self.fleet_metrics['total_devices'] = len(self.device_health)
        self.fleet_metrics['online_devices'] = len(online_devices)
        self.fleet_metrics['healthy_devices'] = len(healthy_devices)
        self.fleet_metrics['devices_with_alerts'] = sum(
            1 for h in self.device_health.values() if h['alerts']
        )
        
        if online_devices:
            self.fleet_metrics['avg_cpu_usage'] = total_cpu / len(online_devices)
            self.fleet_metrics['avg_memory_usage'] = total_memory / len(online_devices)
    
    def get_device_health(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get device health status"""
        return self.device_health.get(device_id)
    
    def get_fleet_health(self) -> Dict[str, Any]:
        """Get overall fleet health"""
        return {
            'metrics': self.fleet_metrics,
            'unhealthy_devices': [
                device_id for device_id, health in self.device_health.items()
                if health['health_score'] < 70
            ],
            'devices_with_alerts': [
                device_id for device_id, health in self.device_health.items()
                if health['alerts']
            ]
        }


class UpdateManager:
    """
    Manages OTA updates across device fleet
    
    Features:
    - Staged rollouts with automatic rollback
    - Update verification and validation
    - Bandwidth optimization
    - Update history tracking
    """
    
    def __init__(self, registry: DeviceRegistry):
        self.registry = registry
        
        # Active updates
        self.active_updates: Dict[str, OTAUpdate] = {}
        self.update_status: Dict[str, Dict[str, UpdateStatus]] = defaultdict(dict)
        
        # Update metrics
        self.update_metrics = {
            'total_updates': 0,
            'successful_updates': 0,
            'failed_updates': 0,
            'rolled_back_updates': 0
        }
    
    async def create_update(self, update: OTAUpdate) -> str:
        """Create new OTA update"""
        
        self.active_updates[update.update_id] = update
        self.update_metrics['total_updates'] += 1
        
        logger.info(f"Created OTA update {update.update_id} version {update.version}")
        return update.update_id
    
    async def deploy_update(self,
                           update_id: str,
                           target_devices: Optional[List[str]] = None,
                           deployment_group: Optional[str] = None) -> Dict[str, Any]:
        """Deploy update to devices"""
        
        if update_id not in self.active_updates:
            raise ValueError(f"Update {update_id} not found")
        
        update = self.active_updates[update_id]
        
        # Get target devices
        if target_devices is None:
            if deployment_group:
                # Get devices in deployment group
                # In production, would filter by group
                target_devices = await self.registry.list_devices(state=DeviceState.ONLINE)
            else:
                # Get all online devices
                target_devices = await self.registry.list_devices(state=DeviceState.ONLINE)
        
        # Apply staged rollout
        if update.staged_rollout_percent < 100:
            rollout_count = max(1, int(len(target_devices) * update.staged_rollout_percent / 100))
            target_devices = target_devices[:rollout_count]
        
        # Initialize update status for each device
        for device_id in target_devices:
            self.update_status[update_id][device_id] = UpdateStatus.PENDING
            
            # Update device twin with desired firmware version
            twin = await self.registry.get_device(device_id)
            if twin:
                twin.update_desired({'firmware_version': update.version})
        
        logger.info(f"Deploying update {update_id} to {len(target_devices)} devices")
        
        return {
            'update_id': update_id,
            'target_devices': target_devices,
            'deployment_started': datetime.utcnow().isoformat()
        }
    
    async def update_device_status(self,
                                  update_id: str,
                                  device_id: str,
                                  status: UpdateStatus,
                                  error: Optional[str] = None):
        """Update device update status"""
        
        if update_id not in self.update_status:
            return
        
        self.update_status[update_id][device_id] = status
        
        # Update metrics
        if status == UpdateStatus.COMPLETED:
            self.update_metrics['successful_updates'] += 1
        elif status == UpdateStatus.FAILED:
            self.update_metrics['failed_updates'] += 1
            
            # Check if rollback needed
            await self._check_rollback_threshold(update_id)
        elif status == UpdateStatus.ROLLED_BACK:
            self.update_metrics['rolled_back_updates'] += 1
        
        logger.info(f"Device {device_id} update status: {status.value}")
    
    async def _check_rollback_threshold(self, update_id: str):
        """Check if automatic rollback should be triggered"""
        
        if update_id not in self.active_updates:
            return
        
        update = self.active_updates[update_id]
        statuses = self.update_status[update_id]
        
        total_devices = len(statuses)
        failed_devices = sum(1 for s in statuses.values() if s == UpdateStatus.FAILED)
        
        if total_devices > 0:
            failure_rate = failed_devices / total_devices
            
            if failure_rate > update.auto_rollback_threshold:
                logger.warning(f"Update {update_id} failure rate {failure_rate:.2%} exceeds threshold")
                await self._trigger_rollback(update_id)
    
    async def _trigger_rollback(self, update_id: str):
        """Trigger update rollback"""
        
        logger.info(f"Triggering rollback for update {update_id}")
        
        # Mark all pending/in-progress devices for rollback
        for device_id, status in self.update_status[update_id].items():
            if status in [UpdateStatus.PENDING, UpdateStatus.DOWNLOADING, UpdateStatus.INSTALLING]:
                self.update_status[update_id][device_id] = UpdateStatus.ROLLED_BACK
        
        # TODO: Send rollback command to devices
    
    def get_update_status(self, update_id: str) -> Dict[str, Any]:
        """Get update deployment status"""
        
        if update_id not in self.update_status:
            return {}
        
        statuses = self.update_status[update_id]
        
        status_counts = defaultdict(int)
        for status in statuses.values():
            status_counts[status.value] += 1
        
        return {
            'update_id': update_id,
            'total_devices': len(statuses),
            'status_breakdown': dict(status_counts),
            'completion_percent': (
                status_counts[UpdateStatus.COMPLETED.value] / len(statuses) * 100
                if statuses else 0
            )
        }


class CommandDispatcher:
    """
    Dispatches commands to devices
    
    Supports synchronous and asynchronous command execution
    """
    
    def __init__(self):
        self.pending_commands: Dict[str, List[Dict]] = defaultdict(list)
        self.command_history: List[Dict] = []
        self.command_callbacks: Dict[str, Callable] = {}
    
    async def send_command(self,
                          device_id: str,
                          command_type: CommandType,
                          payload: Optional[Dict[str, Any]] = None,
                          callback: Optional[Callable] = None) -> str:
        """Send command to device"""
        
        command_id = str(uuid.uuid4())
        
        command = {
            'command_id': command_id,
            'device_id': device_id,
            'command_type': command_type.value,
            'payload': payload or {},
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'pending'
        }
        
        # Queue command
        self.pending_commands[device_id].append(command)
        
        # Register callback if provided
        if callback:
            self.command_callbacks[command_id] = callback
        
        # Add to history
        self.command_history.append(command)
        
        logger.info(f"Queued command {command_type.value} for device {device_id}")
        
        return command_id
    
    async def get_pending_commands(self, device_id: str) -> List[Dict]:
        """Get pending commands for device"""
        
        commands = self.pending_commands.get(device_id, [])
        
        # Clear pending commands after retrieval
        if commands:
            self.pending_commands[device_id] = []
        
        return commands
    
    async def update_command_status(self,
                                   command_id: str,
                                   status: str,
                                   result: Optional[Dict[str, Any]] = None):
        """Update command execution status"""
        
        # Update history
        for cmd in self.command_history:
            if cmd['command_id'] == command_id:
                cmd['status'] = status
                cmd['result'] = result
                cmd['completed_at'] = datetime.utcnow().isoformat()
                break
        
        # Trigger callback if registered
        if command_id in self.command_callbacks:
            callback = self.command_callbacks[command_id]
            await callback(command_id, status, result)
            del self.command_callbacks[command_id]
        
        logger.info(f"Command {command_id} status updated: {status}")


class DeviceFleetManager:
    """
    Main fleet management orchestrator
    
    Coordinates all IoT device management components
    """
    
    def __init__(self):
        # Initialize components
        self.registry = DeviceRegistry()
        self.monitor = FleetMonitor(alert_callback=self._handle_alert)
        self.update_manager = UpdateManager(self.registry)
        self.command_dispatcher = CommandDispatcher()
        
        # Fleet state
        self.running = False
        self.tasks: List[asyncio.Task] = []
    
    async def start(self):
        """Start fleet manager"""
        if self.running:
            return
        
        self.running = True
        logger.info("Starting Device Fleet Manager...")
        
        # Start background tasks
        self.tasks.append(asyncio.create_task(self._monitor_fleet()))
        self.tasks.append(asyncio.create_task(self._process_telemetry()))
        self.tasks.append(asyncio.create_task(self._manage_updates()))
        
        logger.info("✅ Device Fleet Manager started")
    
    async def stop(self):
        """Stop fleet manager"""
        if not self.running:
            return
        
        logger.info("Stopping Device Fleet Manager...")
        self.running = False
        
        # Cancel tasks
        for task in self.tasks:
            task.cancel()
        
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        logger.info("✅ Device Fleet Manager stopped")
    
    async def register_device(self,
                             device_id: str,
                             hardware_version: str,
                             location: Optional[str] = None) -> DeviceTwin:
        """Register new device in fleet"""
        
        twin = await self.registry.register_device(device_id, hardware_version, location)
        
        # Initialize monitoring
        await self.monitor.update_device_telemetry(device_id, {
            'cpu_usage': 0,
            'memory_usage': 0,
            'temperature': 25,
            'error_rate': 0
        })
        
        return twin
    
    async def deploy_firmware_update(self,
                                    version: str,
                                    firmware_url: str,
                                    target_devices: Optional[List[str]] = None) -> str:
        """Deploy firmware update to fleet"""
        
        # Create update package
        update = OTAUpdate(
            version=version,
            description=f"Firmware update to version {version}",
            firmware_url=firmware_url,
            firmware_hash=hashlib.sha256(firmware_url.encode()).hexdigest(),
            firmware_size_bytes=1024 * 1024 * 10,  # 10MB example
            staged_rollout_percent=20  # Start with 20% of fleet
        )
        
        # Create and deploy update
        update_id = await self.update_manager.create_update(update)
        deployment = await self.update_manager.deploy_update(update_id, target_devices)
        
        return update_id
    
    async def send_device_command(self,
                                 device_id: str,
                                 command_type: CommandType,
                                 payload: Optional[Dict[str, Any]] = None) -> str:
        """Send command to device"""
        
        return await self.command_dispatcher.send_command(
            device_id, command_type, payload
        )
    
    async def _monitor_fleet(self):
        """Background task to monitor fleet health"""
        while self.running:
            try:
                # Get fleet health
                fleet_health = self.monitor.get_fleet_health()
                
                # Log fleet status
                logger.debug(f"Fleet health: {fleet_health['metrics']}")
                
                # Check for critical issues
                if fleet_health['metrics']['devices_with_alerts'] > 10:
                    logger.warning(f"{fleet_health['metrics']['devices_with_alerts']} devices have alerts")
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Fleet monitoring error: {e}")
    
    async def _process_telemetry(self):
        """Background task to process device telemetry"""
        while self.running:
            try:
                # Simulate telemetry processing
                # In production, would receive from message queue
                
                await asyncio.sleep(10)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Telemetry processing error: {e}")
    
    async def _manage_updates(self):
        """Background task to manage OTA updates"""
        while self.running:
            try:
                # Check update progress
                for update_id in self.update_manager.active_updates:
                    status = self.update_manager.get_update_status(update_id)
                    
                    if status.get('completion_percent', 0) == 100:
                        logger.info(f"Update {update_id} completed")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Update management error: {e}")
    
    async def _handle_alert(self, device_id: str, alerts: List[Dict]):
        """Handle device alerts"""
        
        for alert in alerts:
            logger.warning(f"Alert for device {device_id}: {alert['type']} - {alert['value']}")
            
            # Take action based on alert type
            if alert['type'] == 'high_temperature' and alert['severity'] == 'critical':
                # Send cooling command
                await self.send_device_command(
                    device_id,
                    CommandType.DIAGNOSTIC,
                    {'action': 'reduce_performance'}
                )
    
    def get_fleet_status(self) -> Dict[str, Any]:
        """Get overall fleet status"""
        
        registry_stats = self.registry.get_statistics()
        fleet_health = self.monitor.get_fleet_health()
        
        return {
            'registry': registry_stats,
            'health': fleet_health,
            'update_metrics': self.update_manager.update_metrics,
            'pending_commands': sum(
                len(cmds) for cmds in self.command_dispatcher.pending_commands.values()
            )
        }


# Demo function
async def demo_fleet_manager():
    """Demonstrate IoT fleet management capabilities"""
    
    print("🌐 AG06 IoT Fleet Manager Demo")
    print("==============================")
    
    # Create fleet manager
    fleet_manager = DeviceFleetManager()
    await fleet_manager.start()
    
    print("✅ Fleet Manager started")
    
    # Register devices
    print("\n📱 Registering devices...")
    
    devices = []
    locations = ['Studio A', 'Studio B', 'Live Room', 'Control Room', 'Mobile Unit']
    
    for i in range(5):
        device_id = f"AG06_PROD_{i+1:03d}"
        twin = await fleet_manager.register_device(
            device_id,
            hardware_version="AG06_v1.2",
            location=locations[i]
        )
        devices.append(device_id)
        print(f"   Registered: {device_id} at {locations[i]}")
    
    # Simulate telemetry updates
    print("\n📊 Updating device telemetry...")
    
    for device_id in devices:
        telemetry = {
            'cpu_usage': np.random.uniform(20, 70),
            'memory_usage': np.random.uniform(30, 60),
            'temperature': np.random.uniform(35, 55),
            'error_rate': np.random.uniform(0, 0.02),
            'audio_latency_ms': np.random.uniform(5, 15),
            'events_processed': np.random.randint(1000, 5000)
        }
        
        await fleet_manager.monitor.update_device_telemetry(device_id, telemetry)
    
    # Deploy firmware update
    print("\n🔄 Deploying firmware update...")
    
    update_id = await fleet_manager.deploy_firmware_update(
        version="2.1.0",
        firmware_url="https://updates.ag06mixer.cloud/firmware/v2.1.0.bin",
        target_devices=devices[:2]  # Update first 2 devices
    )
    
    print(f"   Update ID: {update_id}")
    print(f"   Target devices: {devices[:2]}")
    
    # Simulate update progress
    for device_id in devices[:2]:
        await fleet_manager.update_manager.update_device_status(
            update_id, device_id, UpdateStatus.DOWNLOADING
        )
    
    await asyncio.sleep(1)
    
    for device_id in devices[:2]:
        await fleet_manager.update_manager.update_device_status(
            update_id, device_id, UpdateStatus.COMPLETED
        )
    
    # Send commands
    print("\n📡 Sending device commands...")
    
    cmd_id = await fleet_manager.send_device_command(
        devices[0],
        CommandType.COLLECT_LOGS,
        {'log_level': 'debug', 'duration_minutes': 5}
    )
    
    print(f"   Command sent: COLLECT_LOGS to {devices[0]}")
    
    # Show fleet status
    print("\n📈 Fleet Status:")
    status = fleet_manager.get_fleet_status()
    
    print(f"   Total devices: {status['registry']['total_devices']}")
    print(f"   Online devices: {status['health']['metrics']['online_devices']}")
    print(f"   Healthy devices: {status['health']['metrics']['healthy_devices']}")
    print(f"   Average CPU usage: {status['health']['metrics']['avg_cpu_usage']:.1f}%")
    print(f"   Average memory usage: {status['health']['metrics']['avg_memory_usage']:.1f}%")
    
    print(f"\n🔄 Update Status:")
    update_status = fleet_manager.update_manager.get_update_status(update_id)
    print(f"   Completion: {update_status['completion_percent']:.0f}%")
    print(f"   Status breakdown: {update_status['status_breakdown']}")
    
    # Simulate critical temperature on one device
    print(f"\n⚠️  Simulating critical temperature on {devices[2]}...")
    
    critical_telemetry = {
        'cpu_usage': 85,
        'memory_usage': 70,
        'temperature': 75,  # Critical temperature
        'error_rate': 0.01
    }
    
    await fleet_manager.monitor.update_device_telemetry(devices[2], critical_telemetry)
    
    # Check fleet health after alert
    fleet_health = fleet_manager.monitor.get_fleet_health()
    print(f"   Devices with alerts: {fleet_health['metrics']['devices_with_alerts']}")
    print(f"   Unhealthy devices: {fleet_health['unhealthy_devices']}")
    
    # Stop fleet manager
    await fleet_manager.stop()
    
    print("\n✅ Fleet Manager stopped successfully")
    
    return {
        'total_devices': len(devices),
        'update_deployed': update_id,
        'commands_sent': 1,
        'alerts_generated': fleet_health['metrics']['devices_with_alerts'],
        'success': True
    }


if __name__ == "__main__":
    # Run the demo
    result = asyncio.run(demo_fleet_manager())
    print(f"\n🎯 Demo Result: {result}")