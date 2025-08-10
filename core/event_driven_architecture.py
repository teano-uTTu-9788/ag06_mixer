"""
Event-Driven Architecture for AG06 Mixer
Research-driven implementation of reactive patterns
Based on 2025 event sourcing and CQRS research
"""
import asyncio
from typing import Dict, Any, List, Optional, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict
import json
import uuid
from abc import ABC, abstractmethod


class EventType(Enum):
    """Event types for the system"""
    # Audio Events
    AUDIO_BUFFER_RECEIVED = "audio.buffer.received"
    AUDIO_PROCESSED = "audio.processed"
    AUDIO_EFFECT_APPLIED = "audio.effect.applied"
    
    # MIDI Events
    MIDI_MESSAGE_RECEIVED = "midi.message.received"
    MIDI_CONTROL_CHANGED = "midi.control.changed"
    MIDI_DEVICE_CONNECTED = "midi.device.connected"
    
    # Preset Events
    PRESET_LOADED = "preset.loaded"
    PRESET_SAVED = "preset.saved"
    PRESET_DELETED = "preset.deleted"
    
    # System Events
    SYSTEM_STARTED = "system.started"
    SYSTEM_STOPPED = "system.stopped"
    PERFORMANCE_DEGRADED = "performance.degraded"
    ERROR_OCCURRED = "error.occurred"


@dataclass
class Event:
    """Base event class following Event Sourcing pattern"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: EventType = EventType.SYSTEM_STARTED
    timestamp: datetime = field(default_factory=datetime.now)
    aggregate_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize event to dictionary"""
        return {
            'id': self.id,
            'type': self.type.value,
            'timestamp': self.timestamp.isoformat(),
            'aggregate_id': self.aggregate_id,
            'payload': self.payload,
            'metadata': self.metadata,
            'version': self.version
        }


class IEventHandler(ABC):
    """Event handler interface - Command Query Responsibility Segregation"""
    
    @abstractmethod
    async def handle(self, event: Event) -> None:
        """Handle an event"""
        pass
    
    @abstractmethod
    def can_handle(self, event_type: EventType) -> bool:
        """Check if handler can process this event type"""
        pass


class EventBus:
    """Central event bus - Mediator pattern for decoupling"""
    
    def __init__(self):
        """Initialize event bus"""
        self._handlers: Dict[EventType, List[IEventHandler]] = defaultdict(list)
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._event_store: List[Event] = []
        self._processing_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
    
    def register_handler(self, event_type: EventType, handler: IEventHandler) -> None:
        """Register an event handler"""
        self._handlers[event_type].append(handler)
    
    def subscribe(self, pattern: str, callback: Callable) -> None:
        """Subscribe to events matching pattern"""
        self._subscribers[pattern].append(callback)
    
    async def publish(self, event: Event) -> None:
        """Publish an event to the bus"""
        # Store event (Event Sourcing)
        self._event_store.append(event)
        
        # Queue for processing
        await self._processing_queue.put(event)
        
        # Notify subscribers matching pattern
        for pattern, callbacks in self._subscribers.items():
            if self._matches_pattern(event.type.value, pattern):
                for callback in callbacks:
                    asyncio.create_task(self._safe_callback(callback, event))
    
    async def start(self) -> None:
        """Start event processing"""
        self._running = True
        asyncio.create_task(self._process_events())
    
    async def stop(self) -> None:
        """Stop event processing"""
        self._running = False
    
    async def _process_events(self) -> None:
        """Process events from queue"""
        while self._running:
            try:
                # Get event with timeout to allow checking _running
                event = await asyncio.wait_for(
                    self._processing_queue.get(), 
                    timeout=1.0
                )
                
                # Process with registered handlers
                handlers = self._handlers.get(event.type, [])
                for handler in handlers:
                    if handler.can_handle(event.type):
                        asyncio.create_task(self._safe_handle(handler, event))
                        
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error processing event: {e}")
    
    async def _safe_handle(self, handler: IEventHandler, event: Event) -> None:
        """Safely handle event with error catching"""
        try:
            await handler.handle(event)
        except Exception as e:
            error_event = Event(
                type=EventType.ERROR_OCCURRED,
                payload={'error': str(e), 'original_event': event.id}
            )
            await self.publish(error_event)
    
    async def _safe_callback(self, callback: Callable, event: Event) -> None:
        """Safely execute callback"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                callback(event)
        except Exception as e:
            print(f"Callback error: {e}")
    
    def _matches_pattern(self, event_type: str, pattern: str) -> bool:
        """Check if event type matches pattern"""
        if pattern == "*":
            return True
        if pattern.endswith("*"):
            return event_type.startswith(pattern[:-1])
        return event_type == pattern
    
    def get_events(self, 
                   aggregate_id: Optional[str] = None,
                   event_type: Optional[EventType] = None,
                   since: Optional[datetime] = None) -> List[Event]:
        """Query events from event store"""
        events = self._event_store
        
        if aggregate_id:
            events = [e for e in events if e.aggregate_id == aggregate_id]
        
        if event_type:
            events = [e for e in events if e.type == event_type]
        
        if since:
            events = [e for e in events if e.timestamp >= since]
        
        return events


class AudioProcessingHandler(IEventHandler):
    """Handler for audio processing events"""
    
    def __init__(self, audio_engine):
        """Initialize with audio engine dependency"""
        self._audio_engine = audio_engine
        self._processing_count = 0
    
    async def handle(self, event: Event) -> None:
        """Handle audio events"""
        if event.type == EventType.AUDIO_BUFFER_RECEIVED:
            # Process audio buffer
            audio_data = event.payload.get('audio_data')
            if audio_data:
                processed = await self._audio_engine.process_audio(audio_data)
                
                # Publish processed event
                processed_event = Event(
                    type=EventType.AUDIO_PROCESSED,
                    aggregate_id=event.aggregate_id,
                    payload={'processed_data': processed}
                )
                # Would publish back to bus
                self._processing_count += 1
    
    def can_handle(self, event_type: EventType) -> bool:
        """Check if can handle event type"""
        return event_type in [
            EventType.AUDIO_BUFFER_RECEIVED,
            EventType.AUDIO_EFFECT_APPLIED
        ]


class EventSourcingRepository:
    """Repository pattern with event sourcing"""
    
    def __init__(self, event_bus: EventBus):
        """Initialize repository"""
        self._event_bus = event_bus
        self._snapshots: Dict[str, Any] = {}
        self._snapshot_interval = 10
    
    async def save(self, aggregate_id: str, events: List[Event]) -> None:
        """Save events for an aggregate"""
        for event in events:
            event.aggregate_id = aggregate_id
            await self._event_bus.publish(event)
        
        # Create snapshot if needed
        if len(events) % self._snapshot_interval == 0:
            self._create_snapshot(aggregate_id)
    
    def load(self, aggregate_id: str) -> Any:
        """Load aggregate from events"""
        # Start from snapshot if available
        aggregate = self._snapshots.get(aggregate_id, {})
        
        # Apply events since snapshot
        events = self._event_bus.get_events(aggregate_id=aggregate_id)
        for event in events:
            aggregate = self._apply_event(aggregate, event)
        
        return aggregate
    
    def _apply_event(self, state: Any, event: Event) -> Any:
        """Apply event to aggregate state"""
        # Event application logic
        if event.type == EventType.PRESET_LOADED:
            state['current_preset'] = event.payload.get('preset_name')
        elif event.type == EventType.MIDI_CONTROL_CHANGED:
            if 'controls' not in state:
                state['controls'] = {}
            cc = event.payload.get('cc_number')
            value = event.payload.get('value')
            state['controls'][cc] = value
        
        return state
    
    def _create_snapshot(self, aggregate_id: str) -> None:
        """Create snapshot of aggregate state"""
        self._snapshots[aggregate_id] = self.load(aggregate_id)


class CommandBus:
    """Command bus for CQRS pattern"""
    
    def __init__(self, event_bus: EventBus):
        """Initialize command bus"""
        self._event_bus = event_bus
        self._handlers: Dict[Type, Callable] = {}
    
    def register_handler(self, command_type: Type, handler: Callable) -> None:
        """Register command handler"""
        self._handlers[command_type] = handler
    
    async def send(self, command: Any) -> Any:
        """Send command for processing"""
        command_type = type(command)
        
        if command_type not in self._handlers:
            raise ValueError(f"No handler for command {command_type.__name__}")
        
        handler = self._handlers[command_type]
        
        # Execute command
        result = await handler(command)
        
        # Publish domain events
        if hasattr(result, 'events'):
            for event in result.events:
                await self._event_bus.publish(event)
        
        return result


@dataclass
class LoadPresetCommand:
    """Command to load a preset"""
    preset_name: str
    aggregate_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ProcessAudioCommand:
    """Command to process audio"""
    audio_data: bytes
    effects: List[str] = field(default_factory=list)
    aggregate_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class QueryBus:
    """Query bus for CQRS read model"""
    
    def __init__(self):
        """Initialize query bus"""
        self._handlers: Dict[Type, Callable] = {}
        self._cache: Dict[str, Any] = {}
    
    def register_handler(self, query_type: Type, handler: Callable) -> None:
        """Register query handler"""
        self._handlers[query_type] = handler
    
    async def ask(self, query: Any) -> Any:
        """Execute query"""
        query_type = type(query)
        
        # Check cache
        cache_key = f"{query_type.__name__}:{json.dumps(vars(query), sort_keys=True)}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if query_type not in self._handlers:
            raise ValueError(f"No handler for query {query_type.__name__}")
        
        handler = self._handlers[query_type]
        result = await handler(query)
        
        # Cache result
        self._cache[cache_key] = result
        
        return result
    
    def invalidate_cache(self, pattern: Optional[str] = None) -> None:
        """Invalidate cache entries"""
        if pattern is None:
            self._cache.clear()
        else:
            keys_to_remove = [k for k in self._cache if pattern in k]
            for key in keys_to_remove:
                del self._cache[key]


@dataclass
class GetSystemStatusQuery:
    """Query to get system status"""
    include_metrics: bool = True


@dataclass
class GetPresetListQuery:
    """Query to get preset list"""
    category: Optional[str] = None


class EventDrivenAG06System:
    """Complete event-driven system for AG06"""
    
    def __init__(self):
        """Initialize event-driven system"""
        self.event_bus = EventBus()
        self.command_bus = CommandBus(self.event_bus)
        self.query_bus = QueryBus()
        self.repository = EventSourcingRepository(self.event_bus)
    
    async def initialize(self) -> None:
        """Initialize the system"""
        # Start event bus
        await self.event_bus.start()
        
        # Register handlers
        self._register_handlers()
        
        # Publish system started event
        await self.event_bus.publish(Event(
            type=EventType.SYSTEM_STARTED,
            payload={'timestamp': datetime.now().isoformat()}
        ))
    
    def _register_handlers(self) -> None:
        """Register all event, command and query handlers"""
        # Would register actual handlers here
        pass
    
    async def shutdown(self) -> None:
        """Shutdown the system"""
        # Publish system stopped event
        await self.event_bus.publish(Event(
            type=EventType.SYSTEM_STOPPED,
            payload={'timestamp': datetime.now().isoformat()}
        ))
        
        # Stop event bus
        await self.event_bus.stop()