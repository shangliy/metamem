"""Core data models for MetaMem typed memory system."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    """The 5 typed memory stores."""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    FAILURE = "failure"
    INSTRUCTION = "instruction"


@dataclass
class MemoryUnit:
    """Base for all global memory entries."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    content: str = ""
    type: MemoryType = MemoryType.SEMANTIC
    summary: str = ""  # One-line for Layer 1 (progressive disclosure)

    # Retrieval indexes
    embedding: list[float] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    # Evolution metadata
    confidence: float = 0.7
    importance: float = 0.5
    access_count: int = 0
    success_count: int = 0
    failure_count: int = 0

    # Lineage
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    source_sessions: list[str] = field(default_factory=list)
    supersedes: str | None = None
    superseded_by: str | None = None
    status: str = "active"  # "active" | "superseded" | "decayed"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for storage."""
        d = {
            "id": self.id,
            "content": self.content,
            "type": self.type.value if isinstance(self.type, MemoryType) else self.type,
            "summary": self.summary,
            "entities": self.entities,
            "tags": self.tags,
            "confidence": self.confidence,
            "importance": self.importance,
            "access_count": self.access_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "source_sessions": self.source_sessions,
            "supersedes": self.supersedes,
            "superseded_by": self.superseded_by,
            "status": self.status,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoryUnit":
        """Deserialize from dict."""
        mem_type = d.get("type", "semantic")
        if isinstance(mem_type, str):
            mem_type = MemoryType(mem_type)
        return cls(
            id=d.get("id", str(uuid.uuid4())[:12]),
            content=d.get("content", ""),
            type=mem_type,
            summary=d.get("summary", ""),
            entities=d.get("entities", []),
            tags=d.get("tags", []),
            confidence=d.get("confidence", 0.7),
            importance=d.get("importance", 0.5),
            access_count=d.get("access_count", 0),
            success_count=d.get("success_count", 0),
            failure_count=d.get("failure_count", 0),
            created_at=d.get("created_at", time.time()),
            last_accessed=d.get("last_accessed", time.time()),
            source_sessions=d.get("source_sessions", []),
            supersedes=d.get("supersedes"),
            superseded_by=d.get("superseded_by"),
            status=d.get("status", "active"),
        )


@dataclass
class EpisodicMemory(MemoryUnit):
    """What happened — events and outcomes."""
    type: MemoryType = MemoryType.EPISODIC
    event_description: str = ""
    outcome: str = ""
    participants: list[str] = field(default_factory=list)
    causal_links: list[str] = field(default_factory=list)


@dataclass
class SemanticMemory(MemoryUnit):
    """Facts and knowledge."""
    type: MemoryType = MemoryType.SEMANTIC
    category: str = ""  # "architecture" | "api" | "domain"
    valid_as_of: float = field(default_factory=time.time)


@dataclass
class ProceduralMemory(MemoryUnit):
    """Skills — how to do things."""
    type: MemoryType = MemoryType.PROCEDURAL
    skill_name: str = ""
    steps: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    success_rate: float = 0.0
    last_succeeded: float | None = None
    last_failed: float | None = None


@dataclass
class FailureMemory(MemoryUnit):
    """Anti-patterns — what NOT to do."""
    type: MemoryType = MemoryType.FAILURE
    what_failed: str = ""
    why: str = ""
    fix: str = ""
    severity: str = "medium"  # "low" | "medium" | "high" | "critical"
    failure_condition: str = ""
    related_procedures: list[str] = field(default_factory=list)


@dataclass
class InstructionMemory(MemoryUnit):
    """User preferences and rules."""
    type: MemoryType = MemoryType.INSTRUCTION
    rule: str = ""
    scope: str = "global"  # "global" | "project:X"
    source: str = "explicit"  # "explicit" | "inferred" | "learned"
    revoked: bool = False


# ── Evolution Data Structures ──


@dataclass
class TaskResult:
    """Outcome of a task execution."""
    status: str = "success"  # "success" | "failure" | "partial"
    output: str = ""
    error_analysis: str | None = None
    correction: str | None = None
    failure_condition: str | None = None
    contradicts_memory: str | None = None  # Memory ID proven wrong


@dataclass
class TaskExecution:
    """Links a task to its result and the memories that influenced it."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    description: str = ""
    memories_retrieved: list[str] = field(default_factory=list)
    memories_used: list[str] = field(default_factory=list)
    result: TaskResult = field(default_factory=TaskResult)
    timestamp: float = field(default_factory=time.time)


@dataclass
class EvolutionAction:
    """A mutation applied to global memory."""
    action: str = ""  # "reinforce" | "decay" | "refine" | "supersede" | "create"
    target_memory_id: str | None = None
    new_memory_id: str | None = None
    reason: str = ""
    triggered_by: str = ""  # TaskExecution ID
    timestamp: float = field(default_factory=time.time)


# ── Session Data Structures ──


@dataclass
class Event:
    """Raw event in a session."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: str = "message"  # "message" | "tool_call" | "tool_result" | "error"
    content: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Topic:
    """Auto-segmented topic within a session."""
    topic_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    summary: str = ""
    events: list[Event] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float = field(default_factory=time.time)


@dataclass
class SessionMemory:
    """Per-conversation session memory."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    created_at: float = field(default_factory=time.time)
    project: str | None = None
    topics: list[Topic] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    summary: str = ""
    status: str = "active"  # "active" | "finalized"
