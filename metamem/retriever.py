"""Intent-Aware Multi-View Retrieval Engine.

Combines:
- Semantic search (embedding similarity)
- Lexical search (BM25/FTS5)
- Entity graph traversal
- Intent classification → per-type weighting
- Confidence filtering
- Progressive disclosure (3-layer: index → timeline → full)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .models import MemoryType, MemoryUnit
from .store import MemoryStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Evolvable retrieval configuration — every field is a tunable dimension."""

    # View top-k
    semantic_top_k: int = 20
    keyword_top_k: int = 8
    structured_top_k: int = 5
    max_context: int = 25

    # Fusion mode: "rrf" | "weighted_sum" | "first_found" | "semantic_only" | "keyword_only"
    fusion_mode: str = "rrf"
    weight_semantic: float = 1.5
    weight_keyword: float = 1.2
    weight_structured: float = 0.7

    # Typed store weights (NEW — our differentiator)
    weight_episodic: float = 1.0
    weight_semantic_type: float = 1.0
    weight_procedural: float = 1.0
    weight_failure: float = 0.8
    weight_instruction: float = 1.2

    # Intent-aware routing
    enable_intent_routing: bool = True
    intent_model: str = "rule_based"  # "rule_based" | "llm"

    # Confidence filtering
    min_confidence_threshold: float = 0.1
    confidence_boost_weight: float = 0.3  # How much confidence affects final ranking

    # Evolution feedback
    enable_result_feedback: bool = True
    feedback_reinforcement: float = 0.03
    feedback_decay: float = 0.10

    # Entity graph
    enable_entity_graph: bool = True
    entity_hop_depth: int = 1

    # Time decay
    time_decay_half_life_days: float | None = None

    # Reflection (re-retrieve with refined query)
    reflection_rounds: int = 0

    # Per-category overrides (used by benchmark evolution)
    per_category_overrides: dict[str, Any] = field(default_factory=dict)


class QueryIntent:
    """Classified intent of a retrieval query."""
    HOW_TO = "how_to"          # → weight procedural
    WHAT_IS = "what_is"        # → weight semantic
    WHAT_HAPPENED = "what_happened"  # → weight episodic
    AVOID_MISTAKE = "avoid_mistake"  # → weight failure
    PREFERENCE = "preference"  # → weight instruction
    GENERAL = "general"


def classify_intent(query: str) -> str:
    """Rule-based intent classification for retrieval routing."""
    q = query.lower().strip()

    # Procedural signals
    if any(p in q for p in ["how to", "how do", "steps to", "procedure", "workflow", "deploy", "install", "setup"]):
        return QueryIntent.HOW_TO

    # Failure signals
    if any(p in q for p in ["error", "fail", "bug", "crash", "don't", "avoid", "wrong", "mistake", "issue"]):
        return QueryIntent.AVOID_MISTAKE

    # Episodic signals
    if any(p in q for p in ["when did", "what happened", "last time", "yesterday", "history", "previously"]):
        return QueryIntent.WHAT_HAPPENED

    # Instruction signals
    if any(p in q for p in ["prefer", "always", "never", "rule", "convention", "standard"]):
        return QueryIntent.PREFERENCE

    # Semantic (fact) signals
    if any(p in q for p in ["what is", "what are", "who is", "where", "which", "define"]):
        return QueryIntent.WHAT_IS

    return QueryIntent.GENERAL


def _intent_type_weights(intent: str) -> dict[MemoryType, float]:
    """Map intent to per-type weight multipliers."""
    base = {t: 1.0 for t in MemoryType}
    if intent == QueryIntent.HOW_TO:
        base[MemoryType.PROCEDURAL] = 3.0
        base[MemoryType.FAILURE] = 1.5
    elif intent == QueryIntent.AVOID_MISTAKE:
        base[MemoryType.FAILURE] = 3.0
        base[MemoryType.PROCEDURAL] = 1.5
    elif intent == QueryIntent.WHAT_HAPPENED:
        base[MemoryType.EPISODIC] = 3.0
    elif intent == QueryIntent.PREFERENCE:
        base[MemoryType.INSTRUCTION] = 3.0
    elif intent == QueryIntent.WHAT_IS:
        base[MemoryType.SEMANTIC] = 3.0
    return base


@dataclass
class RetrievedMemory:
    """A memory with its retrieval score and metadata."""
    memory: MemoryUnit
    score: float
    source: str = ""  # "semantic" | "keyword" | "entity" | "type_boost"


class RetrievalEngine:
    """Multi-view, intent-aware retrieval with progressive disclosure."""

    def __init__(self, store: MemoryStore, config: RetrievalConfig | None = None):
        self.store = store
        self.config = config or RetrievalConfig()

    def search(
        self,
        query: str,
        config: RetrievalConfig | None = None,
        query_embedding: np.ndarray | None = None,
    ) -> list[RetrievedMemory]:
        """Full multi-view retrieval pipeline.

        Returns memories ranked by fused score, filtered by confidence.
        """
        cfg = config or self.config
        results: dict[str, RetrievedMemory] = {}

        # Step 1: Intent classification
        intent = classify_intent(query) if cfg.enable_intent_routing else QueryIntent.GENERAL
        type_weights = _intent_type_weights(intent)

        # Step 2: Semantic search
        if cfg.fusion_mode != "keyword_only" and cfg.semantic_top_k > 0:
            if query_embedding is not None:
                sem_results = self.store.search_semantic(query_embedding, top_k=cfg.semantic_top_k)
                for mem, score in sem_results:
                    results[mem.id] = RetrievedMemory(
                        memory=mem, score=score * cfg.weight_semantic, source="semantic"
                    )

        # Step 3: Keyword/FTS search
        if cfg.fusion_mode != "semantic_only" and cfg.keyword_top_k > 0:
            kw_results = self.store.search_fts(query, limit=cfg.keyword_top_k)
            for i, mem in enumerate(kw_results):
                kw_score = 1.0 / (i + 1) * cfg.weight_keyword  # Rank-based score
                if mem.id in results:
                    results[mem.id].score += kw_score
                else:
                    results[mem.id] = RetrievedMemory(memory=mem, score=kw_score, source="keyword")

        # Step 4: Entity graph search
        if cfg.enable_entity_graph and cfg.structured_top_k > 0:
            entities = _extract_entities(query)
            if entities:
                ent_results = self.store.search_entities(entities, limit=cfg.structured_top_k)
                for i, mem in enumerate(ent_results):
                    ent_score = 1.0 / (i + 1) * cfg.weight_structured
                    if mem.id in results:
                        results[mem.id].score += ent_score
                    else:
                        results[mem.id] = RetrievedMemory(
                            memory=mem, score=ent_score, source="entity"
                        )

        # Step 5: Apply type weights (intent-aware boosting)
        for rm in results.values():
            type_w = type_weights.get(rm.memory.type, 1.0)
            # Also apply config-level type weight
            cfg_type_w = _get_config_type_weight(rm.memory.type, cfg)
            rm.score *= type_w * cfg_type_w

        # Step 6: Confidence boost
        if cfg.confidence_boost_weight > 0:
            for rm in results.values():
                rm.score *= (1.0 + cfg.confidence_boost_weight * rm.memory.confidence)

        # Step 7: Filter by confidence threshold
        filtered = [
            rm for rm in results.values()
            if rm.memory.confidence >= cfg.min_confidence_threshold
            and rm.memory.status == "active"
        ]

        # Step 8: Sort and limit
        filtered.sort(key=lambda x: x.score, reverse=True)
        return filtered[: cfg.max_context]

    def search_index(self, query: str, limit: int = 10, **kwargs) -> list[dict]:
        """Layer 1: Compact index results (~50 tokens each)."""
        results = self.search(query, **kwargs)
        return [
            {
                "id": rm.memory.id,
                "type": rm.memory.type.value,
                "summary": rm.memory.summary or rm.memory.content[:80],
                "confidence": round(rm.memory.confidence, 2),
                "score": round(rm.score, 3),
            }
            for rm in results[:limit]
        ]

    def search_timeline(self, memory_id: str, window: int = 5) -> list[dict]:
        """Layer 2: Chronological context around a memory."""
        target = self.store.get(memory_id)
        if not target:
            return []
        # Find memories from same session(s), sorted by time
        session_mems = []
        for mem in self.store._memories.values():
            if mem.status != "active":
                continue
            if set(mem.source_sessions) & set(target.source_sessions):
                session_mems.append(mem)
        session_mems.sort(key=lambda m: m.created_at)
        # Find target index and return window
        idx = next((i for i, m in enumerate(session_mems) if m.id == memory_id), -1)
        if idx < 0:
            return [{"id": target.id, "content": target.content, "type": target.type.value}]
        start = max(0, idx - window // 2)
        end = min(len(session_mems), idx + window // 2 + 1)
        return [
            {
                "id": m.id,
                "type": m.type.value,
                "content": m.content[:200],
                "timestamp": m.created_at,
            }
            for m in session_mems[start:end]
        ]

    def get_full(self, memory_ids: list[str]) -> list[dict]:
        """Layer 3: Full memory details (~500 tokens each)."""
        results = []
        for mid in memory_ids:
            mem = self.store.get(mid)
            if mem:
                mem.access_count += 1
                mem.last_accessed = time.time()
                self.store.update(mem)
                results.append(mem.to_dict())
        return results


def _extract_entities(query: str) -> list[str]:
    """Simple entity extraction from query (capitalized words, quoted terms)."""
    entities = []
    # Quoted strings
    for m in re.finditer(r'"([^"]+)"', query):
        entities.append(m.group(1))
    # Capitalized words (likely proper nouns)
    for word in query.split():
        if word[0].isupper() and len(word) > 2 and word.lower() not in {
            "what", "when", "where", "who", "how", "which", "the", "did", "does",
            "can", "could", "would", "should", "may", "might", "is", "are", "was",
        }:
            entities.append(word)
    return entities


def _get_config_type_weight(mem_type: MemoryType, cfg: RetrievalConfig) -> float:
    """Get the config-level weight for a memory type."""
    mapping = {
        MemoryType.EPISODIC: cfg.weight_episodic,
        MemoryType.SEMANTIC: cfg.weight_semantic_type,
        MemoryType.PROCEDURAL: cfg.weight_procedural,
        MemoryType.FAILURE: cfg.weight_failure,
        MemoryType.INSTRUCTION: cfg.weight_instruction,
    }
    return mapping.get(mem_type, 1.0)


def format_context(results: list[RetrievedMemory], max_tokens: int = 4000) -> str:
    """Format retrieved memories into an injection-ready context string."""
    lines = []
    token_est = 0
    for rm in results:
        entry = f"[{rm.memory.type.value}|conf:{rm.memory.confidence:.2f}] {rm.memory.content}"
        entry_tokens = len(entry.split()) * 1.3  # rough token estimate
        if token_est + entry_tokens > max_tokens:
            break
        lines.append(entry)
        token_est += entry_tokens
    return "\n".join(lines)
