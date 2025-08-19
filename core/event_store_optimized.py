"""
Optimized Event Store with Snapshots and Archival
Implements Event Sourcing pattern with optimization
Based on Fowler (2005) and CQRS best practices
"""
import asyncio
import json
import gzip
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Event:
    """Domain event for event sourcing"""
    event_id: str
    event_type: str
    aggregate_id: str
    timestamp: datetime
    data: Dict[str, Any]
    version: int = 1


@dataclass
class Snapshot:
    """Snapshot of aggregate state"""
    aggregate_id: str
    version: int
    timestamp: datetime
    state: Dict[str, Any]


class OptimizedEventStore:
    """
    Event store with optimization features:
    - Automatic snapshots every N events
    - Compressed archival for old events
    - Memory-bounded in-memory cache
    - Async I/O for persistence
    """
    
    def __init__(self, 
                 snapshot_interval: int = 100,
                 archive_after_days: int = 30,
                 max_memory_events: int = 10000,
                 storage_path: Optional[Path] = None):
        """
        Initialize optimized event store
        
        Args:
            snapshot_interval: Create snapshot every N events
            archive_after_days: Archive events older than N days
            max_memory_events: Maximum events to keep in memory
            storage_path: Path for persistent storage
        """
        # Configuration
        self.snapshot_interval = snapshot_interval
        self.archive_after_days = archive_after_days
        self.max_memory_events = max_memory_events
        self.storage_path = storage_path or Path("/tmp/ag06_events")
        
        # Storage
        self._events: List[Event] = []
        self._snapshots: Dict[str, List[Snapshot]] = {}
        self._archived_events: List[str] = []
        
        # Optimization tracking
        self._event_count = 0
        self._last_snapshot_version = {}
        self._memory_usage = 0
        
        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    async def append_event(self, event: Event) -> None:
        """
        Append event to store with optimization
        
        Args:
            event: Event to append
        """
        # Add to in-memory store
        self._events.append(event)
        self._event_count += 1
        
        # Check if snapshot needed
        aggregate_events = sum(1 for e in self._events if e.aggregate_id == event.aggregate_id)
        last_snapshot = self._last_snapshot_version.get(event.aggregate_id, 0)
        
        if aggregate_events - last_snapshot >= self.snapshot_interval:
            await self._create_snapshot(event.aggregate_id)
        
        # Check memory bounds
        if len(self._events) > self.max_memory_events:
            await self._archive_old_events()
        
        # Persist to disk asynchronously
        asyncio.create_task(self._persist_event(event))
    
    async def get_events(self, 
                        aggregate_id: str, 
                        after_version: int = 0) -> List[Event]:
        """
        Get events for aggregate, using snapshots for optimization
        
        Args:
            aggregate_id: Aggregate ID to get events for
            after_version: Only get events after this version
            
        Returns:
            List of events for the aggregate
        """
        # Check if we have a recent snapshot
        if aggregate_id in self._snapshots:
            snapshots = self._snapshots[aggregate_id]
            # Find closest snapshot before requested version
            for snapshot in reversed(snapshots):
                if snapshot.version <= after_version:
                    after_version = snapshot.version
                    break
        
        # Get events from memory
        events = [
            e for e in self._events 
            if e.aggregate_id == aggregate_id and e.version > after_version
        ]
        
        # If not enough in memory, load from disk
        if not events and after_version == 0:
            events = await self._load_events_from_disk(aggregate_id)
        
        return sorted(events, key=lambda e: e.version)
    
    async def get_snapshot(self, aggregate_id: str) -> Optional[Snapshot]:
        """
        Get latest snapshot for aggregate
        
        Args:
            aggregate_id: Aggregate ID
            
        Returns:
            Latest snapshot or None
        """
        if aggregate_id in self._snapshots:
            snapshots = self._snapshots[aggregate_id]
            return snapshots[-1] if snapshots else None
        return None
    
    async def _create_snapshot(self, aggregate_id: str) -> None:
        """
        Create snapshot for aggregate
        
        Args:
            aggregate_id: Aggregate ID to snapshot
        """
        # Get all events for aggregate
        events = await self.get_events(aggregate_id)
        
        if not events:
            return
        
        # Build aggregate state from events
        state = {}
        for event in events:
            # Apply event to state (simplified)
            state.update(event.data)
        
        # Create snapshot
        snapshot = Snapshot(
            aggregate_id=aggregate_id,
            version=events[-1].version,
            timestamp=datetime.now(),
            state=state
        )
        
        # Store snapshot
        if aggregate_id not in self._snapshots:
            self._snapshots[aggregate_id] = []
        self._snapshots[aggregate_id].append(snapshot)
        
        # Update tracking
        self._last_snapshot_version[aggregate_id] = snapshot.version
        
        # Persist snapshot
        await self._persist_snapshot(snapshot)
    
    async def _archive_old_events(self) -> None:
        """Archive events older than threshold"""
        cutoff_date = datetime.now() - timedelta(days=self.archive_after_days)
        
        # Separate old and current events
        old_events = []
        current_events = []
        
        for event in self._events:
            if event.timestamp < cutoff_date:
                old_events.append(event)
            else:
                current_events.append(event)
        
        if old_events:
            # Archive old events to compressed file
            archive_file = self.storage_path / f"archive_{datetime.now().isoformat()}.json.gz"
            
            async with asyncio.Lock():
                with gzip.open(archive_file, 'wt') as f:
                    events_data = [asdict(e) for e in old_events]
                    json.dump(events_data, f, default=str)
            
            # Update in-memory store
            self._events = current_events
            self._archived_events.append(str(archive_file))
            
            print(f"Archived {len(old_events)} events to {archive_file}")
    
    async def _persist_event(self, event: Event) -> None:
        """
        Persist event to disk
        
        Args:
            event: Event to persist
        """
        event_file = self.storage_path / f"event_{event.event_id}.json"
        
        try:
            async with asyncio.Lock():
                with open(event_file, 'w') as f:
                    json.dump(asdict(event), f, default=str)
        except Exception as e:
            print(f"Failed to persist event: {e}")
    
    async def _persist_snapshot(self, snapshot: Snapshot) -> None:
        """
        Persist snapshot to disk
        
        Args:
            snapshot: Snapshot to persist
        """
        snapshot_file = self.storage_path / f"snapshot_{snapshot.aggregate_id}_{snapshot.version}.json"
        
        try:
            async with asyncio.Lock():
                with open(snapshot_file, 'w') as f:
                    json.dump(asdict(snapshot), f, default=str)
        except Exception as e:
            print(f"Failed to persist snapshot: {e}")
    
    async def _load_events_from_disk(self, aggregate_id: str) -> List[Event]:
        """
        Load events from disk for aggregate
        
        Args:
            aggregate_id: Aggregate ID
            
        Returns:
            List of events
        """
        events = []
        
        # Load from event files
        for event_file in self.storage_path.glob("event_*.json"):
            try:
                with open(event_file, 'r') as f:
                    event_data = json.load(f)
                    if event_data.get('aggregate_id') == aggregate_id:
                        # Convert back to Event object
                        event_data['timestamp'] = datetime.fromisoformat(event_data['timestamp'])
                        events.append(Event(**event_data))
            except Exception as e:
                print(f"Failed to load event from {event_file}: {e}")
        
        return events
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get event store statistics
        
        Returns:
            Statistics dictionary
        """
        return {
            'total_events': self._event_count,
            'memory_events': len(self._events),
            'snapshots': sum(len(s) for s in self._snapshots.values()),
            'archived_files': len(self._archived_events),
            'aggregates': len(self._last_snapshot_version),
            'memory_bounded': len(self._events) <= self.max_memory_events
        }


# Factory function for dependency injection
def create_optimized_event_store(config: Optional[Dict[str, Any]] = None) -> OptimizedEventStore:
    """
    Factory to create optimized event store
    
    Args:
        config: Optional configuration
        
    Returns:
        Configured event store instance
    """
    if config is None:
        config = {}
    
    return OptimizedEventStore(
        snapshot_interval=config.get('snapshot_interval', 100),
        archive_after_days=config.get('archive_after_days', 30),
        max_memory_events=config.get('max_memory_events', 10000),
        storage_path=config.get('storage_path')
    )