"""MetaMem — Unified lifelong memory for LLM agents.

Typed stores (episodic, semantic, procedural, failure, instruction) with
evolution from task results and MCP integration for Claude Code.
"""

__version__ = "0.1.0"

from mem_engram.models import (
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
from mem_engram.store import MemoryStore
from mem_engram.retriever import RetrievalEngine, RetrievalConfig
from mem_engram.evolution import EvolutionEngine
from mem_engram.session import SessionManager, detect_project

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
