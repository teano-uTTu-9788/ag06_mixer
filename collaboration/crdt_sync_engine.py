"""
CRDT-based Real-time Collaboration Engine
Following Google Docs, Figma, and Linear practices for conflict-free collaboration
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Optional, Any, Tuple, Union
from collections import defaultdict
import hashlib
import bisect

class CRDTType(Enum):
    """CRDT data types"""
    LWWREG = "last-write-wins-register"  # Simple values
    GSET = "grow-only-set"  # Add-only sets
    TWOPSET = "two-phase-set"  # Add and remove
    PNCOUNTER = "pn-counter"  # Increment/decrement
    ORMAP = "observed-remove-map"  # Collaborative maps
    RGA = "replicated-growable-array"  # Ordered lists
    WOOT = "without-operational-transform"  # Text editing

class ConflictResolution(Enum):
    """Conflict resolution strategies"""
    LAST_WRITE_WINS = "lww"
    MULTI_VALUE = "mv"
    CUSTOM_MERGE = "custom"
    VECTOR_CLOCK = "vector"

@dataclass
class VectorClock:
    """Vector clock for causality tracking"""
    clock: Dict[str, int] = field(default_factory=dict)
    
    def increment(self, node_id: str):
        """Increment clock for node"""
        self.clock[node_id] = self.clock.get(node_id, 0) + 1
    
    def update(self, other: 'VectorClock'):
        """Update with another vector clock"""
        for node_id, timestamp in other.clock.items():
            self.clock[node_id] = max(self.clock.get(node_id, 0), timestamp)
    
    def happens_before(self, other: 'VectorClock') -> bool:
        """Check if this clock happens before another"""
        for node_id, timestamp in self.clock.items():
            if timestamp > other.clock.get(node_id, 0):
                return False
        return any(t < other.clock.get(n, 0) for n, t in other.clock.items())
    
    def concurrent_with(self, other: 'VectorClock') -> bool:
        """Check if clocks are concurrent"""
        return not self.happens_before(other) and not other.happens_before(self)

@dataclass
class CRDTOperation:
    """CRDT operation for replication"""
    op_id: str
    op_type: str  # add, remove, update
    crdt_id: str
    value: Any
    timestamp: float
    node_id: str
    vector_clock: VectorClock
    metadata: Dict[str, Any] = field(default_factory=dict)

class LWWRegister:
    """Last-Write-Wins Register CRDT"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.value = None
        self.timestamp = 0
        self.vector_clock = VectorClock()
    
    def set(self, value: Any) -> CRDTOperation:
        """Set value with timestamp"""
        self.timestamp = time.time()
        self.value = value
        self.vector_clock.increment(self.node_id)
        
        return CRDTOperation(
            op_id=str(uuid.uuid4()),
            op_type="set",
            crdt_id="lww_register",
            value=value,
            timestamp=self.timestamp,
            node_id=self.node_id,
            vector_clock=self.vector_clock
        )
    
    def merge(self, operation: CRDTOperation):
        """Merge with remote operation"""
        if operation.timestamp > self.timestamp:
            self.value = operation.value
            self.timestamp = operation.timestamp
            self.vector_clock.update(operation.vector_clock)
    
    def get(self) -> Any:
        """Get current value"""
        return self.value

class GrowOnlySet:
    """Grow-Only Set CRDT"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.elements: Set[Any] = set()
        self.vector_clock = VectorClock()
    
    def add(self, element: Any) -> CRDTOperation:
        """Add element to set"""
        self.elements.add(element)
        self.vector_clock.increment(self.node_id)
        
        return CRDTOperation(
            op_id=str(uuid.uuid4()),
            op_type="add",
            crdt_id="g_set",
            value=element,
            timestamp=time.time(),
            node_id=self.node_id,
            vector_clock=self.vector_clock
        )
    
    def merge(self, operation: CRDTOperation):
        """Merge with remote operation"""
        self.elements.add(operation.value)
        self.vector_clock.update(operation.vector_clock)
    
    def contains(self, element: Any) -> bool:
        """Check if element exists"""
        return element in self.elements
    
    def size(self) -> int:
        """Get set size"""
        return len(self.elements)

class PNCounter:
    """Positive-Negative Counter CRDT"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.positive: Dict[str, int] = defaultdict(int)
        self.negative: Dict[str, int] = defaultdict(int)
        self.vector_clock = VectorClock()
    
    def increment(self, delta: int = 1) -> CRDTOperation:
        """Increment counter"""
        self.positive[self.node_id] += delta
        self.vector_clock.increment(self.node_id)
        
        return CRDTOperation(
            op_id=str(uuid.uuid4()),
            op_type="increment",
            crdt_id="pn_counter",
            value=delta,
            timestamp=time.time(),
            node_id=self.node_id,
            vector_clock=self.vector_clock
        )
    
    def decrement(self, delta: int = 1) -> CRDTOperation:
        """Decrement counter"""
        self.negative[self.node_id] += delta
        self.vector_clock.increment(self.node_id)
        
        return CRDTOperation(
            op_id=str(uuid.uuid4()),
            op_type="decrement",
            crdt_id="pn_counter",
            value=delta,
            timestamp=time.time(),
            node_id=self.node_id,
            vector_clock=self.vector_clock
        )
    
    def merge(self, operation: CRDTOperation):
        """Merge with remote operation"""
        if operation.op_type == "increment":
            self.positive[operation.node_id] = max(
                self.positive[operation.node_id],
                operation.value
            )
        else:
            self.negative[operation.node_id] = max(
                self.negative[operation.node_id],
                operation.value
            )
        self.vector_clock.update(operation.vector_clock)
    
    def value(self) -> int:
        """Get counter value"""
        return sum(self.positive.values()) - sum(self.negative.values())

class RGADocument:
    """Replicated Growable Array for collaborative text editing"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.tombstones: Set[str] = set()
        self.vector_clock = VectorClock()
        
        # Character structure: (char_id, char, visible)
        self.characters: List[Tuple[str, str, bool]] = []
        self.char_index: Dict[str, int] = {}
    
    def insert(self, position: int, text: str) -> List[CRDTOperation]:
        """Insert text at position"""
        operations = []
        
        for i, char in enumerate(text):
            char_id = f"{self.node_id}:{uuid.uuid4()}"
            
            # Find the predecessor character
            if position + i > 0 and position + i <= len(self.characters):
                prev_id = self.characters[position + i - 1][0]
            else:
                prev_id = None
            
            # Insert character
            self.characters.insert(position + i, (char_id, char, True))
            self.char_index[char_id] = position + i
            
            self.vector_clock.increment(self.node_id)
            
            operations.append(CRDTOperation(
                op_id=str(uuid.uuid4()),
                op_type="insert",
                crdt_id="rga_document",
                value={"char": char, "char_id": char_id, "prev_id": prev_id},
                timestamp=time.time(),
                node_id=self.node_id,
                vector_clock=self.vector_clock
            ))
        
        return operations
    
    def delete(self, position: int, length: int) -> List[CRDTOperation]:
        """Delete text at position"""
        operations = []
        
        for i in range(length):
            if position < len(self.characters):
                char_id, char, visible = self.characters[position]
                
                if visible:
                    # Mark as tombstone
                    self.tombstones.add(char_id)
                    self.characters[position] = (char_id, char, False)
                    
                    self.vector_clock.increment(self.node_id)
                    
                    operations.append(CRDTOperation(
                        op_id=str(uuid.uuid4()),
                        op_type="delete",
                        crdt_id="rga_document",
                        value={"char_id": char_id},
                        timestamp=time.time(),
                        node_id=self.node_id,
                        vector_clock=self.vector_clock
                    ))
        
        return operations
    
    def merge(self, operation: CRDTOperation):
        """Merge remote operation"""
        if operation.op_type == "insert":
            char_data = operation.value
            char_id = char_data["char_id"]
            
            if char_id not in self.char_index:
                # Find insertion position
                prev_id = char_data.get("prev_id")
                
                if prev_id and prev_id in self.char_index:
                    position = self.char_index[prev_id] + 1
                else:
                    position = 0
                
                # Insert character
                self.characters.insert(position, 
                    (char_id, char_data["char"], True))
                
                # Update indices
                self._rebuild_index()
        
        elif operation.op_type == "delete":
            char_id = operation.value["char_id"]
            
            if char_id in self.char_index:
                idx = self.char_index[char_id]
                if idx < len(self.characters):
                    self.tombstones.add(char_id)
                    old_char = self.characters[idx]
                    self.characters[idx] = (old_char[0], old_char[1], False)
        
        self.vector_clock.update(operation.vector_clock)
    
    def _rebuild_index(self):
        """Rebuild character index"""
        self.char_index.clear()
        for i, (char_id, _, _) in enumerate(self.characters):
            self.char_index[char_id] = i
    
    def get_text(self) -> str:
        """Get visible text"""
        return "".join(char for _, char, visible in self.characters if visible)

class CollaborationSession:
    """Real-time collaboration session"""
    
    def __init__(self, session_id: str, node_id: str):
        self.session_id = session_id
        self.node_id = node_id
        self.participants: Set[str] = {node_id}
        
        # CRDT instances
        self.registers: Dict[str, LWWRegister] = {}
        self.sets: Dict[str, GrowOnlySet] = {}
        self.counters: Dict[str, PNCounter] = {}
        self.documents: Dict[str, RGADocument] = {}
        
        # Operation log for catch-up
        self.operation_log: List[CRDTOperation] = []
        self.operation_buffer: List[CRDTOperation] = []
        
        # Conflict detection
        self.conflicts: List[Dict[str, Any]] = []
        
        # Performance metrics
        self.metrics = CollaborationMetrics()
    
    def create_register(self, name: str) -> LWWRegister:
        """Create LWW register"""
        if name not in self.registers:
            self.registers[name] = LWWRegister(self.node_id)
        return self.registers[name]
    
    def create_set(self, name: str) -> GrowOnlySet:
        """Create grow-only set"""
        if name not in self.sets:
            self.sets[name] = GrowOnlySet(self.node_id)
        return self.sets[name]
    
    def create_counter(self, name: str) -> PNCounter:
        """Create PN counter"""
        if name not in self.counters:
            self.counters[name] = PNCounter(self.node_id)
        return self.counters[name]
    
    def create_document(self, name: str) -> RGADocument:
        """Create collaborative document"""
        if name not in self.documents:
            self.documents[name] = RGADocument(self.node_id)
        return self.documents[name]
    
    async def apply_operation(self, operation: CRDTOperation):
        """Apply CRDT operation"""
        start_time = time.time()
        
        # Add to log
        self.operation_log.append(operation)
        
        # Route to appropriate CRDT
        if "register" in operation.crdt_id:
            for register in self.registers.values():
                register.merge(operation)
        
        elif "set" in operation.crdt_id:
            for gset in self.sets.values():
                gset.merge(operation)
        
        elif "counter" in operation.crdt_id:
            for counter in self.counters.values():
                counter.merge(operation)
        
        elif "document" in operation.crdt_id:
            for document in self.documents.values():
                document.merge(operation)
        
        # Track metrics
        latency = (time.time() - start_time) * 1000
        await self.metrics.track_operation(operation, latency)
    
    async def synchronize(self, remote_ops: List[CRDTOperation]):
        """Synchronize with remote operations"""
        # Sort by vector clock for causal consistency
        sorted_ops = self._sort_by_causality(remote_ops)
        
        for op in sorted_ops:
            if op.node_id != self.node_id:  # Don't apply own operations
                await self.apply_operation(op)
    
    def _sort_by_causality(self, operations: List[CRDTOperation]) -> List[CRDTOperation]:
        """Sort operations by causal order"""
        # Simplified: sort by timestamp
        # In production, use full vector clock comparison
        return sorted(operations, key=lambda op: op.timestamp)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current collaboration state"""
        return {
            "session_id": self.session_id,
            "node_id": self.node_id,
            "participants": list(self.participants),
            "registers": {name: reg.get() for name, reg in self.registers.items()},
            "sets": {name: list(s.elements) for name, s in self.sets.items()},
            "counters": {name: c.value() for name, c in self.counters.items()},
            "documents": {name: doc.get_text() for name, doc in self.documents.items()},
            "operation_count": len(self.operation_log)
        }

class CollaborationMetrics:
    """Metrics for collaboration performance"""
    
    def __init__(self):
        self.operation_count = 0
        self.total_latency = 0
        self.conflicts_resolved = 0
        self.bytes_synced = 0
    
    async def track_operation(self, operation: CRDTOperation, latency: float):
        """Track operation metrics"""
        self.operation_count += 1
        self.total_latency += latency
        
        # Estimate operation size
        op_size = len(json.dumps({
            "value": str(operation.value),
            "clock": operation.vector_clock.clock
        }))
        self.bytes_synced += op_size
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collaboration statistics"""
        avg_latency = self.total_latency / self.operation_count if self.operation_count > 0 else 0
        
        return {
            "operations": self.operation_count,
            "avg_latency_ms": avg_latency,
            "conflicts_resolved": self.conflicts_resolved,
            "bytes_synced": self.bytes_synced,
            "throughput_ops_sec": self.operation_count / (self.total_latency / 1000) if self.total_latency > 0 else 0
        }

class CollaborationEngine:
    """Main collaboration engine orchestrator"""
    
    def __init__(self):
        self.sessions: Dict[str, CollaborationSession] = {}
        self.node_id = str(uuid.uuid4())
        self.network = P2PNetwork(self.node_id)
    
    async def create_session(self, session_id: str) -> CollaborationSession:
        """Create new collaboration session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = CollaborationSession(session_id, self.node_id)
        return self.sessions[session_id]
    
    async def join_session(self, session_id: str, peer_id: str) -> CollaborationSession:
        """Join existing session"""
        session = await self.create_session(session_id)
        session.participants.add(peer_id)
        
        # Request state sync from peers
        await self.network.request_sync(session_id, peer_id)
        
        return session
    
    async def broadcast_operation(self, session_id: str, operation: CRDTOperation):
        """Broadcast operation to all peers"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            
            # Apply locally
            await session.apply_operation(operation)
            
            # Broadcast to peers
            for peer_id in session.participants:
                if peer_id != self.node_id:
                    await self.network.send_operation(peer_id, operation)
    
    async def handle_remote_operation(self, operation: CRDTOperation):
        """Handle operation from remote peer"""
        # Find the session for this operation
        for session in self.sessions.values():
            if operation.node_id in session.participants:
                await session.apply_operation(operation)
                break

class P2PNetwork:
    """Peer-to-peer network for CRDT synchronization"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.peers: Dict[str, str] = {}  # peer_id -> address
        self.connections: Dict[str, Any] = {}  # Active connections
    
    async def send_operation(self, peer_id: str, operation: CRDTOperation):
        """Send operation to peer"""
        # Simulate network send
        await asyncio.sleep(0.001)  # 1ms network latency
        
        # In production, use WebSocket or WebRTC
        print(f"Sent operation {operation.op_id} to {peer_id}")
    
    async def request_sync(self, session_id: str, peer_id: str):
        """Request state sync from peer"""
        # Simulate sync request
        await asyncio.sleep(0.005)  # 5ms for state transfer
        
        print(f"Requested sync for session {session_id} from {peer_id}")
    
    async def connect_peer(self, peer_id: str, address: str):
        """Connect to peer"""
        self.peers[peer_id] = address
        # Simulate connection establishment
        await asyncio.sleep(0.01)
        
        self.connections[peer_id] = {"connected": True, "address": address}

# Example usage
async def main():
    """Demonstrate CRDT-based collaboration"""
    
    print("🤝 CRDT-based Real-time Collaboration Engine")
    print("=" * 60)
    
    # Create collaboration engine
    engine = CollaborationEngine()
    
    # Create a session
    session = await engine.create_session("audio-mixing-session")
    
    print(f"\n📝 Session Created: {session.session_id}")
    print(f"Node ID: {session.node_id}")
    
    # Test collaborative text editing
    print("\n✏️ Collaborative Text Editing:")
    print("-" * 40)
    
    doc = session.create_document("lyrics")
    
    # User 1 inserts text
    ops1 = doc.insert(0, "Hello ")
    for op in ops1:
        await engine.broadcast_operation(session.session_id, op)
    
    # User 2 inserts text (simulated)
    doc2 = RGADocument("user2")
    ops2 = doc2.insert(0, "World! ")
    for op in ops2:
        await session.apply_operation(op)
    
    print(f"Document content: '{doc.get_text()}'")
    
    # Test counter
    print("\n🔢 Collaborative Counter:")
    print("-" * 40)
    
    counter = session.create_counter("play_count")
    
    # Multiple users increment
    op1 = counter.increment(5)
    await engine.broadcast_operation(session.session_id, op1)
    
    op2 = counter.increment(3)
    await engine.broadcast_operation(session.session_id, op2)
    
    print(f"Counter value: {counter.value()}")
    
    # Test set
    print("\n📦 Collaborative Set:")
    print("-" * 40)
    
    tags = session.create_set("tags")
    
    # Add tags from different users
    op3 = tags.add("rock")
    await engine.broadcast_operation(session.session_id, op3)
    
    op4 = tags.add("electronic")
    await engine.broadcast_operation(session.session_id, op4)
    
    print(f"Tags: {tags.elements}")
    
    # Get session state
    state = session.get_state()
    
    print("\n📊 Session State:")
    print("-" * 40)
    print(f"Participants: {len(state['participants'])}")
    print(f"Operations: {state['operation_count']}")
    print(f"Documents: {list(state['documents'].keys())}")
    
    # Get metrics
    metrics = session.metrics.get_stats()
    
    print("\n📈 Performance Metrics:")
    print("-" * 40)
    print(f"Operations: {metrics['operations']}")
    print(f"Avg Latency: {metrics['avg_latency_ms']:.2f}ms")
    print(f"Throughput: {metrics['throughput_ops_sec']:.0f} ops/sec")
    
    print("\n✅ CRDT collaboration engine operational!")
    
    return engine

if __name__ == "__main__":
    asyncio.run(main())