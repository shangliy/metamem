"""MetaMem — Unified lifelong memory for LLM agents.

Typed stores (episodic, semantic, procedural, failure, instruction) with
evolution from task results and MCP integration for Claude Code.
"""

__version__ = "0.1.0"

from metamem.models import (
    MemoryType,
    MemoryUnit,
    EpisodicMemory,
    SemanticMemory,
    ProceduralMemory,
    FailureMemory,
    InstructionMemory,
    TaskExecution,
    TaskResult,
    EvolutionAction,
)
from metamem.store import MemoryStore
from metamem.retriever import RetrievalEngine, RetrievalConfig
from metamem.evolution import EvolutionEngine

__all__ = [
    "MemoryType",
    "MemoryUnit",
    "EpisodicMemory",
    "SemanticMemory",
    "ProceduralMemory",
    "FailureMemory",
    "InstructionMemory",
    "TaskExecution",
    "TaskResult",
    "EvolutionAction",
    "MemoryStore",
    "RetrievalEngine",
    "RetrievalConfig",
    "EvolutionEngine",
]
