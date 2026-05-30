"""MetaMem MCP Server — Exposes memory tools to Claude Code and other MCP clients.

Tools:
- mem_search: Search memory index (Layer 1 — compact)
- mem_timeline: Chronological context (Layer 2)
- mem_get: Full memory details (Layer 3)
- mem_store: Explicitly store a typed memory
- mem_instruct: Save a user preference/rule
- mem_feedback: Report task result for evolution
- mem_stats: Get memory system statistics
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from .evolution import EvolutionConfig, EvolutionEngine
from .models import (
    MemoryType,
    MemoryUnit,
    TaskExecution,
    TaskResult,
)
from .retriever import RetrievalConfig, RetrievalEngine
from .session import SessionManager, detect_project
from .store import MemoryStore

logger = logging.getLogger(__name__)

# Global instances (initialized on serve())
_store: MemoryStore | None = None
_retriever: RetrievalEngine | None = None
_evolution: EvolutionEngine | None = None
_session: SessionManager | None = None


def _init():
    """Initialize global memory system with project detection."""
    global _store, _retriever, _evolution, _session
    if _store is not None:
        return

    data_dir = os.environ.get("METAMEM_DATA_DIR", os.path.expanduser("~/.metamem"))

    # Detect project and use project-scoped store
    project = os.environ.get("METAMEM_PROJECT", "")
    cwd = os.environ.get("METAMEM_CWD", os.getcwd())
    if not project:
        project = detect_project(cwd)

    _session = SessionManager.start(project=project, cwd=cwd, data_dir=data_dir)

    # Try to load embedder
    embedder = None
    try:
        from sentence_transformers import SentenceTransformer
        model_name = os.environ.get("METAMEM_EMBED_MODEL", "all-MiniLM-L6-v2")
        embedder = SentenceTransformer(model_name)
    except ImportError:
        logger.warning("sentence-transformers not available — semantic search disabled")

    # Use project-scoped store from session manager
    _store = _session.store
    _store.embedder = embedder
    _retriever = RetrievalEngine(_store, RetrievalConfig())
    _evolution = EvolutionEngine(_store, EvolutionConfig())


def _get_embedding(text: str):
    """Get embedding for a query if embedder is available."""
    if _store and _store.embedder:
        import numpy as np
        return _store.embedder.encode(text)
    return None


# ── MCP Tool Handlers ──


def mem_search(query: str, type: str | None = None, limit: int = 10) -> dict[str, Any]:
    """Search memory index — Layer 1 (compact, ~50 tokens/result).

    Args:
        query: Natural language search query
        type: Filter by memory type (episodic/semantic/procedural/failure/instruction)
        limit: Max results to return (default: 10)

    Returns:
        Compact index of matching memories with IDs, types, summaries, and scores.
    """
    _init()
    embedding = _get_embedding(query)
    results = _retriever.search(query, query_embedding=embedding)

    # Filter by type if specified
    if type:
        mem_type = MemoryType(type)
        results = [r for r in results if r.memory.type == mem_type]

    index = [
        {
            "id": rm.memory.id,
            "type": rm.memory.type.value,
            "summary": rm.memory.summary or rm.memory.content[:100],
            "confidence": round(rm.memory.confidence, 2),
            "score": round(rm.score, 3),
        }
        for rm in results[:limit]
    ]
    return {"results": index, "total": len(results), "query": query}


def mem_timeline(memory_id: str, window: int = 5) -> dict[str, Any]:
    """Get chronological context around a memory — Layer 2.

    Args:
        memory_id: ID of the memory to center on
        window: Number of surrounding memories to include

    Returns:
        Timeline of memories around the target, ordered by time.
    """
    _init()
    timeline = _retriever.search_timeline(memory_id, window=window)
    return {"timeline": timeline, "center_id": memory_id}


def mem_get(ids: list[str]) -> dict[str, Any]:
    """Get full memory details — Layer 3 (~500 tokens/result).

    Args:
        ids: List of memory IDs to fetch full details for

    Returns:
        Complete memory objects with all metadata.
    """
    _init()
    memories = _retriever.get_full(ids)
    return {"memories": memories, "count": len(memories)}


def mem_store(
    content: str,
    type: str = "semantic",
    summary: str = "",
    entities: list[str] | None = None,
    tags: list[str] | None = None,
    session_id: str = "",
) -> dict[str, Any]:
    """Store a new typed memory.

    Args:
        content: The memory content text
        type: Memory type (episodic/semantic/procedural/failure/instruction)
        summary: One-line summary for search index
        entities: Related entities (people, services, concepts)
        tags: Tags for structured filtering
        session_id: Source session ID

    Returns:
        Created memory ID and confirmation.
    """
    _init()
    mem = MemoryUnit(
        content=content,
        type=MemoryType(type),
        summary=summary or content[:80],
        entities=entities or [],
        tags=tags or [],
        source_sessions=[session_id] if session_id else [],
    )
    mem_id = _store.add(mem)
    return {"id": mem_id, "type": type, "stored": True}


def mem_instruct(
    rule: str,
    scope: str = "global",
) -> dict[str, Any]:
    """Save a user preference or instruction.

    Args:
        rule: The rule/preference to remember (e.g., "Always use pnpm")
        scope: Scope — "global" or "project:<name>"

    Returns:
        Created instruction memory ID.
    """
    _init()
    mem_id = _evolution.learn_instruction(rule=rule, scope=scope, source="explicit")
    return {"id": mem_id, "rule": rule, "scope": scope, "stored": True}


def mem_feedback(
    description: str,
    memories_used: list[str],
    status: str = "success",
    error_analysis: str = "",
    correction: str = "",
    contradicts_memory: str = "",
) -> dict[str, Any]:
    """Report task result for memory evolution.

    Args:
        description: What the task was
        memories_used: List of memory IDs that were used
        status: "success" | "failure" | "partial"
        error_analysis: Why it failed (if failure)
        correction: What the correct info/approach is
        contradicts_memory: ID of memory proven wrong

    Returns:
        Evolution actions taken.
    """
    _init()
    task = TaskExecution(
        description=description,
        memories_used=memories_used,
        memories_retrieved=memories_used,
        result=TaskResult(
            status=status,
            error_analysis=error_analysis or None,
            correction=correction or None,
            contradicts_memory=contradicts_memory or None,
        ),
    )
    actions = _evolution.on_task_result(task)
    return {
        "actions": [
            {"action": a.action, "target": a.target_memory_id, "reason": a.reason}
            for a in actions
        ],
        "count": len(actions),
    }


def mem_stats() -> dict[str, Any]:
    """Get memory system statistics.

    Returns:
        Store stats (counts by type, avg confidence, etc.) and evolution stats.
    """
    _init()
    return {
        "project": _session.config.project,
        "project_dir": str(_session.project_dir),
        "session_id": _session.session_id,
        "store": _store.stats(),
        "evolution": _evolution.get_stats(),
    }


def mem_context() -> dict[str, Any]:
    """Get project context injection — previous work summary for session continuity.

    Returns:
        Context from previous sessions: instructions, last session summary,
        project skills, warnings, and key facts.
    """
    _init()
    context = _session.get_context_injection()
    return {
        "project": _session.config.project,
        "context": context,
        "sessions": _session.list_sessions(limit=5),
    }


def mem_event(event_type: str, content: str) -> dict[str, Any]:
    """Record a session event for memory tracking.

    Args:
        event_type: "message" | "tool_call" | "tool_result" | "observation" | "error"
        content: What happened

    Returns:
        Confirmation with event ID.
    """
    _init()
    _session.add_event(event_type, content)
    return {"recorded": True, "session_id": _session.session_id, "events_total": len(_session.session.events)}


# ── MCP Server Protocol ──

TOOLS = [
    {
        "name": "mem_search",
        "description": "Search memory index. Returns compact results (~50 tokens each). "
                       "Use this first, then mem_get for full details on relevant IDs.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query"},
                "type": {"type": "string", "enum": ["episodic", "semantic", "procedural", "failure", "instruction"],
                         "description": "Filter by memory type (optional)"},
                "limit": {"type": "integer", "default": 10, "description": "Max results"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "mem_timeline",
        "description": "Get chronological context around a memory. "
                       "Shows what happened before/after a specific observation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Memory ID to center on"},
                "window": {"type": "integer", "default": 5},
            },
            "required": ["memory_id"],
        },
    },
    {
        "name": "mem_get",
        "description": "Get full memory details by IDs (~500 tokens each). "
                       "Use after mem_search to fetch details for relevant results.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ids": {"type": "array", "items": {"type": "string"},
                        "description": "Memory IDs to fetch"},
            },
            "required": ["ids"],
        },
    },
    {
        "name": "mem_store",
        "description": "Store a new typed memory (fact, skill, failure case, etc).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Memory content"},
                "type": {"type": "string", "enum": ["episodic", "semantic", "procedural", "failure", "instruction"],
                         "default": "semantic"},
                "summary": {"type": "string", "description": "One-line summary"},
                "entities": {"type": "array", "items": {"type": "string"}},
                "tags": {"type": "array", "items": {"type": "string"}},
                "session_id": {"type": "string", "default": ""},
            },
            "required": ["content"],
        },
    },
    {
        "name": "mem_instruct",
        "description": "Save a user preference or rule (e.g., 'Always use pnpm').",
        "inputSchema": {
            "type": "object",
            "properties": {
                "rule": {"type": "string", "description": "The rule to remember"},
                "scope": {"type": "string", "default": "global",
                          "description": "'global' or 'project:<name>'"},
            },
            "required": ["rule"],
        },
    },
    {
        "name": "mem_feedback",
        "description": "Report task result for memory evolution. "
                       "Call after using memories to complete a task.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "What the task was"},
                "memories_used": {"type": "array", "items": {"type": "string"},
                                  "description": "Memory IDs used"},
                "status": {"type": "string", "enum": ["success", "failure", "partial"]},
                "error_analysis": {"type": "string", "default": ""},
                "correction": {"type": "string", "default": ""},
                "contradicts_memory": {"type": "string", "default": ""},
            },
            "required": ["description", "memories_used", "status"],
        },
    },
    {
        "name": "mem_stats",
        "description": "Get memory system statistics (counts, confidence, evolution actions).",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "mem_context",
        "description": "Get project context from previous sessions. "
                       "Call at session start to continue previous work seamlessly.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "mem_event",
        "description": "Record a session event for memory tracking. "
                       "Call to log important actions/observations during the session.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "event_type": {"type": "string",
                               "enum": ["message", "tool_call", "tool_result", "observation", "error"],
                               "description": "Type of event"},
                "content": {"type": "string", "description": "What happened"},
            },
            "required": ["event_type", "content"],
        },
    },
]

# Handler dispatch
_HANDLERS = {
    "mem_search": mem_search,
    "mem_timeline": mem_timeline,
    "mem_get": mem_get,
    "mem_store": mem_store,
    "mem_instruct": mem_instruct,
    "mem_feedback": mem_feedback,
    "mem_stats": mem_stats,
    "mem_context": mem_context,
    "mem_event": mem_event,
}


async def handle_tool_call(name: str, arguments: dict) -> str:
    """Handle an MCP tool call."""
    handler = _HANDLERS.get(name)
    if not handler:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = handler(**arguments)
        return json.dumps(result, default=str)
    except Exception as e:
        logger.exception("Tool call failed: %s", name)
        return json.dumps({"error": str(e)})


def serve():
    """Start the MCP server (stdio transport)."""
    import asyncio
    import sys

    async def _run():
        """Simple stdio MCP server implementation."""
        _init()
        logger.info("MetaMem MCP server started")

        while True:
            try:
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                msg = json.loads(line.strip())
                method = msg.get("method", "")
                msg_id = msg.get("id")

                if method == "initialize":
                    response = {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "metamem", "version": "0.1.0"},
                        },
                    }
                elif method == "tools/list":
                    response = {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {"tools": TOOLS},
                    }
                elif method == "tools/call":
                    params = msg.get("params", {})
                    tool_name = params.get("name", "")
                    arguments = params.get("arguments", {})
                    result_text = await handle_tool_call(tool_name, arguments)
                    response = {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "content": [{"type": "text", "text": result_text}],
                        },
                    }
                elif method == "notifications/initialized":
                    continue  # No response needed
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {"code": -32601, "message": f"Method not found: {method}"},
                    }

                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()

            except json.JSONDecodeError:
                continue
            except Exception as e:
                logger.exception("Server error")
                if msg_id:
                    error_resp = {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {"code": -32603, "message": str(e)},
                    }
                    sys.stdout.write(json.dumps(error_resp) + "\n")
                    sys.stdout.flush()

    asyncio.run(_run())


if __name__ == "__main__":
    serve()
