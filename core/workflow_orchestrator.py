"""
Workflow Orchestrator for AG06 Mixer
Follows SOLID principles - Single Responsibility & Open/Closed
"""
import asyncio
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass

from interfaces.audio_engine import IAudioEngine, AudioConfig
from interfaces.midi_controller import IMidiController
from interfaces.preset_manager import IPresetManager
from interfaces.karaoke_integration import IVocalProcessor
from core.dependency_container import inject, ServiceLocator


class TaskType(Enum):
    """Task types for workflow processing"""
    AUDIO_PROCESSING = "audio_processing"
    MIDI_CONTROL = "midi_control"
    PRESET_MANAGEMENT = "preset_management"
    KARAOKE_PROCESSING = "karaoke_processing"
    SYSTEM_CONFIG = "system_config"


@dataclass
class WorkflowTask:
    """Workflow task data structure"""
    type: TaskType
    parameters: Dict[str, Any]
    priority: int = 0
    async_execution: bool = True


@dataclass
class WorkflowResult:
    """Workflow execution result"""
    success: bool
    task_type: TaskType
    data: Optional[Any] = None
    error: Optional[str] = None


class ITaskHandler:
    """Interface for task handlers - Open/Closed Principle"""
    
    async def can_handle(self, task: WorkflowTask) -> bool:
        """Check if handler can process this task"""
        raise NotImplementedError
    
    async def handle(self, task: WorkflowTask) -> WorkflowResult:
        """Handle the task"""
        raise NotImplementedError


class AudioTaskHandler(ITaskHandler):
    """Audio processing task handler - Single Responsibility"""
    
    def __init__(self, audio_engine: IAudioEngine):
        self._audio_engine = audio_engine
    
    async def can_handle(self, task: WorkflowTask) -> bool:
        return task.type == TaskType.AUDIO_PROCESSING
    
    async def handle(self, task: WorkflowTask) -> WorkflowResult:
        try:
            audio_data = task.parameters.get('audio_data')
            processed = await self._audio_engine.process_audio(audio_data)
            
            return WorkflowResult(
                success=True,
                task_type=task.type,
                data=processed
            )
        except Exception as e:
            return WorkflowResult(
                success=False,
                task_type=task.type,
                error=str(e)
            )


class MidiTaskHandler(ITaskHandler):
    """MIDI control task handler - Single Responsibility"""
    
    def __init__(self, midi_controller: IMidiController):
        self._midi_controller = midi_controller
    
    async def can_handle(self, task: WorkflowTask) -> bool:
        return task.type == TaskType.MIDI_CONTROL
    
    async def handle(self, task: WorkflowTask) -> WorkflowResult:
        try:
            message = task.parameters.get('midi_message')
            await self._midi_controller.send_message(message)
            
            return WorkflowResult(
                success=True,
                task_type=task.type
            )
        except Exception as e:
            return WorkflowResult(
                success=False,
                task_type=task.type,
                error=str(e)
            )


class PresetTaskHandler(ITaskHandler):
    """Preset management task handler - Single Responsibility"""
    
    def __init__(self, preset_manager: IPresetManager):
        self._preset_manager = preset_manager
    
    async def can_handle(self, task: WorkflowTask) -> bool:
        return task.type == TaskType.PRESET_MANAGEMENT
    
    async def handle(self, task: WorkflowTask) -> WorkflowResult:
        try:
            action = task.parameters.get('action')
            
            if action == 'load':
                preset_name = task.parameters.get('preset_name')
                preset = await self._preset_manager.load_preset(preset_name)
                return WorkflowResult(success=True, task_type=task.type, data=preset)
            
            elif action == 'save':
                preset = task.parameters.get('preset')
                success = await self._preset_manager.save_preset(preset)
                return WorkflowResult(success=success, task_type=task.type)
            
            else:
                return WorkflowResult(
                    success=False,
                    task_type=task.type,
                    error=f"Unknown preset action: {action}"
                )
        except Exception as e:
            return WorkflowResult(
                success=False,
                task_type=task.type,
                error=str(e)
            )


class KaraokeTaskHandler(ITaskHandler):
    """Karaoke processing task handler - Single Responsibility"""
    
    def __init__(self, vocal_processor: IVocalProcessor):
        self._vocal_processor = vocal_processor
    
    async def can_handle(self, task: WorkflowTask) -> bool:
        return task.type == TaskType.KARAOKE_PROCESSING
    
    async def handle(self, task: WorkflowTask) -> WorkflowResult:
        try:
            action = task.parameters.get('action')
            audio_data = task.parameters.get('audio_data')
            
            if action == 'remove_vocals':
                processed = await self._vocal_processor.remove_vocals(audio_data)
            elif action == 'enhance_vocals':
                level = task.parameters.get('enhancement_level', 1.0)
                processed = await self._vocal_processor.enhance_vocals(audio_data, level)
            else:
                return WorkflowResult(
                    success=False,
                    task_type=task.type,
                    error=f"Unknown karaoke action: {action}"
                )
            
            return WorkflowResult(
                success=True,
                task_type=task.type,
                data=processed
            )
        except Exception as e:
            return WorkflowResult(
                success=False,
                task_type=task.type,
                error=str(e)
            )


class AG06WorkflowOrchestrator:
    """Main workflow orchestrator - Single Responsibility: Coordination"""
    
    def __init__(self):
        """Initialize orchestrator with dependency injection"""
        self._handlers: List[ITaskHandler] = []
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
    
    def register_handler(self, handler: ITaskHandler) -> None:
        """Register a task handler - Open/Closed Principle"""
        self._handlers.append(handler)
    
    async def initialize(self) -> None:
        """Initialize orchestrator and register handlers"""
        # Get services from DI container
        locator = ServiceLocator.get_instance()
        
        # Register handlers for different task types
        self.register_handler(
            AudioTaskHandler(locator.resolve(IAudioEngine))
        )
        self.register_handler(
            MidiTaskHandler(locator.resolve(IMidiController))
        )
        self.register_handler(
            PresetTaskHandler(locator.resolve(IPresetManager))
        )
        self.register_handler(
            KaraokeTaskHandler(locator.resolve(IVocalProcessor))
        )
    
    async def execute_task(self, task: WorkflowTask) -> WorkflowResult:
        """Execute a single task"""
        # Find appropriate handler
        for handler in self._handlers:
            if await handler.can_handle(task):
                return await handler.handle(task)
        
        # No handler found
        return WorkflowResult(
            success=False,
            task_type=task.type,
            error="No handler registered for task type"
        )
    
    async def execute_workflow(self, tasks: List[WorkflowTask]) -> List[WorkflowResult]:
        """Execute a workflow of multiple tasks"""
        results = []
        
        # Sort tasks by priority
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)
        
        for task in sorted_tasks:
            if task.async_execution:
                # Queue for async execution
                await self._task_queue.put(task)
            else:
                # Execute synchronously
                result = await self.execute_task(task)
                results.append(result)
        
        # Process async tasks
        if not self._task_queue.empty():
            async_results = await self._process_async_tasks()
            results.extend(async_results)
        
        return results
    
    async def _process_async_tasks(self) -> List[WorkflowResult]:
        """Process async tasks from queue"""
        results = []
        tasks = []
        
        while not self._task_queue.empty():
            task = await self._task_queue.get()
            tasks.append(self.execute_task(task))
        
        if tasks:
            results = await asyncio.gather(*tasks)
        
        return results
    
    async def start(self) -> None:
        """Start the orchestrator"""
        self._running = True
        await self.initialize()
    
    async def stop(self) -> None:
        """Stop the orchestrator"""
        self._running = False
        # Clear task queue
        while not self._task_queue.empty():
            self._task_queue.get_nowait()