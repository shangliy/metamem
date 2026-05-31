"""Memory Store — LanceDB vectors + SQLite relational index.

Provides typed storage for all 5 memory types with:
- Vector similarity search (sentence-transformers embeddings)
- BM25 full-text search
- Entity graph queries
- Confidence-aware filtering
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np

from .models import (
    EvolutionAction,
    MemoryType,
    MemoryUnit,
    TaskExecution,
)

logger = logging.getLogger(__name__)

# Default data directory
DEFAULT_DATA_DIR = os.path.expanduser("~/.mem-engram")


class MemoryStore:
    """Unified memory store with typed access patterns.

    Storage layout:
        ~/.mem-engram/
        ├── memory.db          # SQLite — metadata, entities, FTS, evolution log
        ├── sessions/          # Per-session folders
        └── embeddings/        # Numpy arrays for vector search
    """

    def __init__(self, data_dir: str | None = None, embedder=None):
        self.data_dir = Path(data_dir or DEFAULT_DATA_DIR)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self._db_path = self.data_dir / "memory.db"
        self._emb_dir = self.data_dir / "embeddings"
        self._emb_dir.mkdir(exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        # In-memory caches for fast retrieval
        self._memories: dict[str, MemoryUnit] = {}
        self._embeddings: dict[str, np.ndarray] = {}
        self._load_all()

    def _init_schema(self):
        """Initialize SQLite schema."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                type TEXT NOT NULL,
                summary TEXT DEFAULT '',
                entities TEXT DEFAULT '[]',
                tags TEXT DEFAULT '[]',
                confidence REAL DEFAULT 0.7,
                importance REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                created_at REAL,
                last_accessed REAL,
                source_sessions TEXT DEFAULT '[]',
                supersedes TEXT,
                superseded_by TEXT,
                status TEXT DEFAULT 'active',
                extra TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS entities (
                name TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                FOREIGN KEY (memory_id) REFERENCES memories(id)
            );
            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
            CREATE INDEX IF NOT EXISTS idx_entities_memory ON entities(memory_id);

            CREATE TABLE IF NOT EXISTS relationships (
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                relation TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                confidence REAL DEFAULT 1.0
            );
            CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source);
            CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target);

            CREATE TABLE IF NOT EXISTS evolution_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                target_memory_id TEXT,
                new_memory_id TEXT,
                reason TEXT,
                triggered_by TEXT,
                timestamp REAL
            );

            CREATE TABLE IF NOT EXISTS task_executions (
                task_id TEXT PRIMARY KEY,
                description TEXT,
                memories_retrieved TEXT DEFAULT '[]',
                memories_used TEXT DEFAULT '[]',
                result_status TEXT,
                result_output TEXT,
                timestamp REAL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                id, content, summary, entities, tags,
                content='memories',
                content_rowid='rowid'
            );

            -- Triggers to keep FTS in sync
            CREATE TRIGGER IF NOT EXISTS memory_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memory_fts(id, content, summary, entities, tags)
                VALUES (new.id, new.content, new.summary, new.entities, new.tags);
            END;
            CREATE TRIGGER IF NOT EXISTS memory_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memory_fts(memory_fts, id, content, summary, entities, tags)
                VALUES ('delete', old.id, old.content, old.summary, old.entities, old.tags);
            END;
            CREATE TRIGGER IF NOT EXISTS memory_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memory_fts(memory_fts, id, content, summary, entities, tags)
                VALUES ('delete', old.id, old.content, old.summary, old.entities, old.tags);
                INSERT INTO memory_fts(id, content, summary, entities, tags)
                VALUES (new.id, new.content, new.summary, new.entities, new.tags);
            END;
        """)
        self._conn.commit()

    def _load_all(self):
        """Load all active memories into cache."""
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE status = 'active'"
        ).fetchall()
        for row in rows:
            mem = self._row_to_memory(row)
            self._memories[mem.id] = mem
        # Load embeddings
        emb_path = self._emb_dir / "vectors.npz"
        if emb_path.exists():
            data = np.load(str(emb_path), allow_pickle=True)
            ids = data.get("ids", [])
            vectors = data.get("vectors", np.array([]))
            if len(ids) > 0 and len(vectors) > 0:
                for i, mid in enumerate(ids):
                    self._embeddings[str(mid)] = vectors[i]
        logger.info("Loaded %d memories, %d embeddings", len(self._memories), len(self._embeddings))

    def _row_to_memory(self, row) -> MemoryUnit:
        """Convert a SQLite row to MemoryUnit."""
        return MemoryUnit(
            id=row["id"],
            content=row["content"],
            type=MemoryType(row["type"]),
            summary=row["summary"] or "",
            entities=json.loads(row["entities"] or "[]"),
            tags=json.loads(row["tags"] or "[]"),
            confidence=row["confidence"],
            importance=row["importance"],
            access_count=row["access_count"],
            success_count=row["success_count"],
            failure_count=row["failure_count"],
            created_at=row["created_at"],
            last_accessed=row["last_accessed"],
            source_sessions=json.loads(row["source_sessions"] or "[]"),
            supersedes=row["supersedes"],
            superseded_by=row["superseded_by"],
            status=row["status"],
        )

    # ── CRUD ──

    def add(self, memory: MemoryUnit) -> str:
        """Add a memory unit to the store."""
        # Generate embedding if embedder available
        if self.embedder and not memory.embedding:
            memory.embedding = self.embedder.encode(memory.content).tolist()

        self._conn.execute(
            """INSERT OR REPLACE INTO memories
               (id, content, type, summary, entities, tags, confidence, importance,
                access_count, success_count, failure_count, created_at, last_accessed,
                source_sessions, supersedes, superseded_by, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                memory.id, memory.content, memory.type.value, memory.summary,
                json.dumps(memory.entities), json.dumps(memory.tags),
                memory.confidence, memory.importance,
                memory.access_count, memory.success_count, memory.failure_count,
                memory.created_at, memory.last_accessed,
                json.dumps(memory.source_sessions), memory.supersedes,
                memory.superseded_by, memory.status,
            ),
        )
        # Entity index
        for entity in memory.entities:
            self._conn.execute(
                "INSERT INTO entities (name, memory_id) VALUES (?, ?)",
                (entity.lower(), memory.id),
            )
        self._conn.commit()

        # Cache
        self._memories[memory.id] = memory
        if memory.embedding:
            self._embeddings[memory.id] = np.array(memory.embedding)

        return memory.id

    def get(self, memory_id: str) -> MemoryUnit | None:
        """Get a memory by ID."""
        return self._memories.get(memory_id)

    def get_by_type(self, mem_type: MemoryType) -> list[MemoryUnit]:
        """Get all active memories of a specific type."""
        return [m for m in self._memories.values() if m.type == mem_type and m.status == "active"]

    def update(self, memory: MemoryUnit):
        """Update a memory's metadata (confidence, access count, etc.)."""
        self._conn.execute(
            """UPDATE memories SET confidence=?, importance=?, access_count=?,
               success_count=?, failure_count=?, last_accessed=?, status=?,
               superseded_by=?
               WHERE id=?""",
            (
                memory.confidence, memory.importance, memory.access_count,
                memory.success_count, memory.failure_count, memory.last_accessed,
                memory.status, memory.superseded_by, memory.id,
            ),
        )
        self._conn.commit()
        self._memories[memory.id] = memory

    def search_fts(self, query: str, limit: int = 20) -> list[MemoryUnit]:
        """Full-text search using FTS5."""
        try:
            rows = self._conn.execute(
                "SELECT id FROM memory_fts WHERE memory_fts MATCH ? LIMIT ?",
                (query, limit),
            ).fetchall()
            return [self._memories[r["id"]] for r in rows if r["id"] in self._memories]
        except Exception:
            return []

    def search_semantic(self, query_embedding: np.ndarray, top_k: int = 20) -> list[tuple[MemoryUnit, float]]:
        """Cosine similarity search over embeddings."""
        if not self._embeddings:
            return []
        ids = list(self._embeddings.keys())
        vectors = np.array([self._embeddings[mid] for mid in ids])
        # Normalize
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        vec_norms = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
        scores = vec_norms @ query_norm
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            mid = ids[idx]
            if mid in self._memories and self._memories[mid].status == "active":
                results.append((self._memories[mid], float(scores[idx])))
        return results

    def search_entities(self, entities: list[str], limit: int = 10) -> list[MemoryUnit]:
        """Find memories mentioning specific entities."""
        if not entities:
            return []
        placeholders = ",".join("?" * len(entities))
        rows = self._conn.execute(
            f"SELECT DISTINCT memory_id FROM entities WHERE name IN ({placeholders}) LIMIT ?",
            [e.lower() for e in entities] + [limit],
        ).fetchall()
        return [self._memories[r["memory_id"]] for r in rows if r["memory_id"] in self._memories]

    # ── Evolution ──

    def reinforce(self, memory_id: str, boost: float = 0.03):
        """Increase confidence after successful use."""
        mem = self._memories.get(memory_id)
        if not mem:
            return
        mem.confidence = min(1.0, mem.confidence + boost)
        mem.success_count += 1
        mem.access_count += 1
        mem.last_accessed = time.time()
        self.update(mem)

    def decay(self, memory_id: str, penalty: float = 0.10):
        """Decrease confidence after failed use."""
        mem = self._memories.get(memory_id)
        if not mem:
            return
        mem.confidence = max(0.05, mem.confidence - penalty)
        mem.failure_count += 1
        mem.access_count += 1
        mem.last_accessed = time.time()
        if mem.confidence < 0.1:
            mem.status = "decayed"
        self.update(mem)

    def supersede(self, old_id: str, new_memory: MemoryUnit) -> str:
        """Replace an old memory with a corrected version."""
        old = self._memories.get(old_id)
        if old:
            old.status = "superseded"
            old.superseded_by = new_memory.id
            self.update(old)
        new_memory.supersedes = old_id
        return self.add(new_memory)

    def log_evolution(self, action: EvolutionAction):
        """Record an evolution action."""
        self._conn.execute(
            """INSERT INTO evolution_log (action, target_memory_id, new_memory_id,
               reason, triggered_by, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (action.action, action.target_memory_id, action.new_memory_id,
             action.reason, action.triggered_by, action.timestamp),
        )
        self._conn.commit()

    def log_task(self, task: TaskExecution):
        """Record a task execution for attribution."""
        self._conn.execute(
            """INSERT OR REPLACE INTO task_executions
               (task_id, description, memories_retrieved, memories_used,
                result_status, result_output, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                task.task_id, task.description,
                json.dumps(task.memories_retrieved), json.dumps(task.memories_used),
                task.result.status, task.result.output, task.timestamp,
            ),
        )
        self._conn.commit()

    # ── Persistence ──

    def save_embeddings(self):
        """Persist embedding cache to disk."""
        if not self._embeddings:
            return
        ids = list(self._embeddings.keys())
        vectors = np.array([self._embeddings[mid] for mid in ids])
        np.savez(str(self._emb_dir / "vectors.npz"), ids=ids, vectors=vectors)

    def close(self):
        """Flush and close."""
        self.save_embeddings()
        self._conn.close()

    # ── Stats ──

    def stats(self) -> dict[str, Any]:
        """Return store statistics."""
        by_type = {}
        for t in MemoryType:
            by_type[t.value] = len(self.get_by_type(t))
        active = sum(1 for m in self._memories.values() if m.status == "active")
        avg_conf = (
            sum(m.confidence for m in self._memories.values()) / len(self._memories)
            if self._memories else 0
        )
        return {
            "total": len(self._memories),
            "active": active,
            "by_type": by_type,
            "avg_confidence": round(avg_conf, 3),
            "embeddings": len(self._embeddings),
        }
