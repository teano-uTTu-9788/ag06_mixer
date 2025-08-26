"""
Persistence package for AG06 Workflow System
Provides event storage, replay, and deduplication
"""

from .event_store import (
    EventStore,
    get_event_store,
    EventRecord
)

__all__ = [
    'EventStore',
    'get_event_store',
    'EventRecord'
]