"""
Application Lifecycle Manager
Separates lifecycle management from business logic
Fixes SRP violation in AG06MixerApplication
"""
import asyncio
from typing import Optional, Any, Dict
from enum import Enum


class ApplicationState(Enum):
    """Application lifecycle states"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class ApplicationLifecycleManager:
    """
    Manages application lifecycle - Single Responsibility: Lifecycle Management
    Separated from business logic to follow SRP
    """
    
    def __init__(self):
        """Initialize lifecycle manager"""
        self._state = ApplicationState.UNINITIALIZED
        self._initialization_callbacks = []
        self._startup_callbacks = []
        self._shutdown_callbacks = []
        self._error_handlers = []
        self._metadata: Dict[str, Any] = {}
    
    @property
    def state(self) -> ApplicationState:
        """Get current application state"""
        return self._state
    
    @property
    def is_running(self) -> bool:
        """Check if application is running"""
        return self._state == ApplicationState.RUNNING
    
    def register_initialization(self, callback) -> None:
        """Register initialization callback"""
        self._initialization_callbacks.append(callback)
    
    def register_startup(self, callback) -> None:
        """Register startup callback"""
        self._startup_callbacks.append(callback)
    
    def register_shutdown(self, callback) -> None:
        """Register shutdown callback"""
        self._shutdown_callbacks.append(callback)
    
    def register_error_handler(self, handler) -> None:
        """Register error handler"""
        self._error_handlers.append(handler)
    
    async def initialize(self) -> bool:
        """
        Initialize application
        
        Returns:
            True if initialization successful
        """
        if self._state != ApplicationState.UNINITIALIZED:
            print(f"Cannot initialize from state: {self._state}")
            return False
        
        try:
            self._state = ApplicationState.INITIALIZING
            
            # Run initialization callbacks
            for callback in self._initialization_callbacks:
                await callback()
            
            self._state = ApplicationState.INITIALIZED
            print("✅ Application lifecycle: Initialized")
            return True
            
        except Exception as e:
            await self._handle_error(e, "initialization")
            return False
    
    async def start(self) -> bool:
        """
        Start application
        
        Returns:
            True if startup successful
        """
        if self._state != ApplicationState.INITIALIZED:
            print(f"Cannot start from state: {self._state}")
            return False
        
        try:
            self._state = ApplicationState.STARTING
            
            # Run startup callbacks
            for callback in self._startup_callbacks:
                await callback()
            
            self._state = ApplicationState.RUNNING
            print("✅ Application lifecycle: Running")
            return True
            
        except Exception as e:
            await self._handle_error(e, "startup")
            return False
    
    async def stop(self) -> bool:
        """
        Stop application
        
        Returns:
            True if shutdown successful
        """
        if self._state not in [ApplicationState.RUNNING, ApplicationState.ERROR]:
            print(f"Cannot stop from state: {self._state}")
            return False
        
        try:
            self._state = ApplicationState.STOPPING
            
            # Run shutdown callbacks
            for callback in self._shutdown_callbacks:
                try:
                    await callback()
                except Exception as e:
                    print(f"Shutdown callback error: {e}")
            
            self._state = ApplicationState.STOPPED
            print("✅ Application lifecycle: Stopped")
            return True
            
        except Exception as e:
            await self._handle_error(e, "shutdown")
            return False
    
    async def restart(self) -> bool:
        """
        Restart application
        
        Returns:
            True if restart successful
        """
        if await self.stop():
            # Reset to initialized state for restart
            self._state = ApplicationState.INITIALIZED
            return await self.start()
        return False
    
    async def _handle_error(self, error: Exception, phase: str) -> None:
        """
        Handle lifecycle error
        
        Args:
            error: Exception that occurred
            phase: Phase where error occurred
        """
        self._state = ApplicationState.ERROR
        print(f"❌ Lifecycle error during {phase}: {error}")
        
        # Run error handlers
        for handler in self._error_handlers:
            try:
                await handler(error, phase)
            except Exception as e:
                print(f"Error handler failed: {e}")
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get lifecycle metadata
        
        Args:
            key: Metadata key
            default: Default value if not found
            
        Returns:
            Metadata value
        """
        return self._metadata.get(key, default)
    
    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set lifecycle metadata
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self._metadata[key] = value
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get lifecycle status
        
        Returns:
            Status dictionary
        """
        return {
            'state': self._state.value,
            'is_running': self.is_running,
            'initialization_callbacks': len(self._initialization_callbacks),
            'startup_callbacks': len(self._startup_callbacks),
            'shutdown_callbacks': len(self._shutdown_callbacks),
            'error_handlers': len(self._error_handlers),
            'metadata': self._metadata
        }


class TaskDelegator:
    """
    Handles task delegation - Single Responsibility: Task Routing
    Separated from lifecycle management to follow SRP
    """
    
    def __init__(self, orchestrator):
        """
        Initialize task delegator
        
        Args:
            orchestrator: Workflow orchestrator for task execution
        """
        self._orchestrator = orchestrator
        self._task_handlers = {}
        self._task_metrics = {
            'total': 0,
            'successful': 0,
            'failed': 0
        }
    
    def register_handler(self, task_type: str, handler) -> None:
        """
        Register task handler
        
        Args:
            task_type: Type of task
            handler: Handler function
        """
        self._task_handlers[task_type] = handler
    
    async def delegate_task(self, task_type: str, parameters: Dict[str, Any]) -> Any:
        """
        Delegate task to appropriate handler
        
        Args:
            task_type: Type of task
            parameters: Task parameters
            
        Returns:
            Task result
        """
        self._task_metrics['total'] += 1
        
        try:
            if task_type in self._task_handlers:
                # Use registered handler
                result = await self._task_handlers[task_type](parameters)
            else:
                # Delegate to orchestrator
                from core.workflow_orchestrator import WorkflowTask, TaskType
                
                # Map string to TaskType enum
                task_enum = TaskType[task_type.upper()] if hasattr(TaskType, task_type.upper()) else TaskType.AUDIO_PROCESSING
                
                task = WorkflowTask(
                    type=task_enum,
                    parameters=parameters,
                    priority=1
                )
                result = await self._orchestrator.execute_task(task)
            
            self._task_metrics['successful'] += 1
            return result
            
        except Exception as e:
            self._task_metrics['failed'] += 1
            raise e
    
    def get_metrics(self) -> Dict[str, int]:
        """
        Get task delegation metrics
        
        Returns:
            Metrics dictionary
        """
        return self._task_metrics.copy()


# Factory function for dependency injection
def create_lifecycle_manager() -> ApplicationLifecycleManager:
    """
    Factory to create lifecycle manager
    
    Returns:
        Lifecycle manager instance
    """
    return ApplicationLifecycleManager()


def create_task_delegator(orchestrator) -> TaskDelegator:
    """
    Factory to create task delegator
    
    Args:
        orchestrator: Workflow orchestrator
        
    Returns:
        Task delegator instance
    """
    return TaskDelegator(orchestrator)