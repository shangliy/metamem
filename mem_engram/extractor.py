"""Memory Extractor — Converts raw session conversations into typed MemoryUnits.

Uses LLM-driven extraction to:
1. Identify facts, skills, failures, preferences from conversation turns
2. Classify each into the appropriate memory type
3. Extract entities and relationships
4. Generate compressed summaries
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

from .models import (
    EpisodicMemory,
    FailureMemory,
    InstructionMemory,
    MemoryType,
    MemoryUnit,
    ProceduralMemory,
    SemanticMemory,
)

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction system. Given a conversation segment,
extract structured memories into typed categories.

For each memory, output a JSON object with:
- "type": one of "episodic", "semantic", "procedural", "failure", "instruction"
- "content": the memory content (concise, atomic fact)
- "summary": one-line summary (max 80 chars)
- "entities": list of entities mentioned (people, services, tools, concepts)
- "tags": relevant tags

Memory type guidelines:
- episodic: Events that happened ("deployed v2", "fixed the auth bug")
- semantic: Facts and knowledge ("API uses OAuth2", "rate limit is 60/min")
- procedural: How-to skills ("Deploy: build → push → restart")
- failure: What went wrong and why ("OOM caused by unbounded cache")
- instruction: User preferences ("Always use pnpm, not npm")

Rules:
1. Be CONCISE — each memory should be 1-2 sentences max
2. Be SPECIFIC — include exact values, names, versions
3. Be ATOMIC — one fact per memory (split compound facts)
4. SKIP small-talk, greetings, and low-information content

Output a JSON array of memory objects. Return [] if no extractable memories."""

EXTRACTION_USER_TEMPLATE = """Extract memories from this conversation segment:

{conversation}

Return a JSON array of typed memories:"""


class MemoryExtractor:
    """Extracts typed memories from conversation sessions."""

    def __init__(self, llm_call: Callable | None = None):
        self.llm_call = llm_call

    def extract_from_turns(
        self,
        turns: list[dict],
        session_id: str = "",
        chunk_size: int = 20,
    ) -> list[MemoryUnit]:
        """Extract memories from conversation turns.

        Args:
            turns: List of {speaker, text, ...} dicts
            session_id: Source session ID for lineage
            chunk_size: Number of turns per extraction batch

        Returns:
            List of typed MemoryUnit instances
        """
        all_memories: list[MemoryUnit] = []

        # Process in chunks
        for i in range(0, len(turns), chunk_size):
            chunk = turns[i:i + chunk_size]
            conversation_text = self._format_turns(chunk)

            if self.llm_call:
                memories = self._extract_with_llm(conversation_text, session_id)
            else:
                memories = self._extract_heuristic(chunk, session_id)

            all_memories.extend(memories)

        return all_memories

    def extract_from_sessions(
        self,
        sessions: list[tuple[str, str, list[dict]]],
    ) -> list[MemoryUnit]:
        """Extract memories from multiple sessions.

        Args:
            sessions: List of (session_id, date_str, turns)

        Returns:
            All extracted memories with session lineage
        """
        all_memories: list[MemoryUnit] = []
        for session_id, date_str, turns in sessions:
            memories = self.extract_from_turns(turns, session_id=session_id)
            all_memories.extend(memories)
        return all_memories

    def _format_turns(self, turns: list[dict]) -> str:
        """Format turns into readable conversation text."""
        lines = []
        for turn in turns:
            speaker = turn.get("speaker", turn.get("role", "unknown"))
            text = turn.get("text", turn.get("content", ""))
            lines.append(f"{speaker}: {text}")
        return "\n".join(lines)

    def _extract_with_llm(self, conversation: str, session_id: str) -> list[MemoryUnit]:
        """Use LLM to extract typed memories."""
        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": EXTRACTION_USER_TEMPLATE.format(conversation=conversation)},
        ]

        response = self.llm_call(messages, max_tokens=2048, temperature=0.1)
        if not response:
            return []

        # Parse JSON from response
        memories = self._parse_extraction_response(response, session_id)
        return memories

    def _parse_extraction_response(self, response: str, session_id: str) -> list[MemoryUnit]:
        """Parse LLM extraction response into MemoryUnit list."""
        # Try to find JSON array in response
        try:
            # Handle markdown code blocks
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                items = json.loads(json_match.group())
            else:
                items = json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse extraction response")
            return []

        memories: list[MemoryUnit] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            mem = self._dict_to_typed_memory(item, session_id)
            if mem:
                memories.append(mem)

        return memories

    def _dict_to_typed_memory(self, item: dict, session_id: str) -> MemoryUnit | None:
        """Convert an extracted dict to the appropriate typed MemoryUnit."""
        mem_type = item.get("type", "semantic")
        content = item.get("content", "")
        if not content:
            return None

        base_kwargs = {
            "content": content,
            "summary": item.get("summary", content[:80]),
            "entities": item.get("entities", []),
            "tags": item.get("tags", []),
            "source_sessions": [session_id] if session_id else [],
            "confidence": 0.7,
        }

        if mem_type == "episodic":
            return EpisodicMemory(
                **base_kwargs,
                event_description=content,
                outcome=item.get("outcome", ""),
            )
        elif mem_type == "procedural":
            return ProceduralMemory(
                **base_kwargs,
                skill_name=item.get("skill_name", content[:50]),
                steps=item.get("steps", []),
                preconditions=item.get("preconditions", []),
            )
        elif mem_type == "failure":
            return FailureMemory(
                **base_kwargs,
                what_failed=item.get("what_failed", content),
                why=item.get("why", ""),
                fix=item.get("fix", ""),
                severity=item.get("severity", "medium"),
            )
        elif mem_type == "instruction":
            return InstructionMemory(
                **base_kwargs,
                rule=item.get("rule", content),
                scope=item.get("scope", "global"),
                source="inferred",
            )
        else:
            return SemanticMemory(
                **base_kwargs,
                category=item.get("category", "general"),
            )

    def _extract_heuristic(self, turns: list[dict], session_id: str) -> list[MemoryUnit]:
        """Fallback heuristic extraction without LLM.

        Simple rule-based extraction for when no LLM is available.
        """
        memories: list[MemoryUnit] = []

        for turn in turns:
            text = turn.get("text", turn.get("content", ""))
            speaker = turn.get("speaker", turn.get("role", ""))

            if not text or len(text) < 20:
                continue

            # Detect failure patterns
            if any(w in text.lower() for w in ["error", "failed", "bug", "crash", "traceback"]):
                memories.append(FailureMemory(
                    content=text[:200],
                    summary=f"Error: {text[:60]}",
                    what_failed=text[:100],
                    why="",
                    fix="",
                    source_sessions=[session_id] if session_id else [],
                ))
            # Detect instructions
            elif any(w in text.lower() for w in ["always", "never", "prefer", "use instead"]):
                if speaker in ("user", "human"):
                    memories.append(InstructionMemory(
                        content=text[:200],
                        summary=f"Rule: {text[:60]}",
                        rule=text[:200],
                        source="inferred",
                        source_sessions=[session_id] if session_id else [],
                    ))
            # Detect procedural (step-by-step)
            elif re.search(r'(step \d|first.*then|1\.|->|→)', text.lower()):
                memories.append(ProceduralMemory(
                    content=text[:300],
                    summary=f"Procedure: {text[:60]}",
                    skill_name=text[:50],
                    source_sessions=[session_id] if session_id else [],
                ))
            # Default: semantic memory for substantial content
            elif len(text) > 50:
                memories.append(SemanticMemory(
                    content=text[:200],
                    summary=text[:80],
                    source_sessions=[session_id] if session_id else [],
                ))

        return memories
