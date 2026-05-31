"""Evolution Engine — Task-result-driven memory improvement.

Observes task outcomes and mutates memories accordingly:
- Success → reinforce contributing memories
- Failure → decay misleading memories, create failure cases
- Partial → add caveats/conditions
- Contradictions → supersede old memories
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .models import (
    EvolutionAction,
    FailureMemory,
    InstructionMemory,
    MemoryType,
    MemoryUnit,
    ProceduralMemory,
    TaskExecution,
    TaskResult,
)
from .store import MemoryStore

logger = logging.getLogger(__name__)


@dataclass
class EvolutionConfig:
    """Configuration for the evolution engine."""
    # Confidence adjustment
    reinforce_boost: float = 0.03
    decay_penalty: float = 0.10
    partial_boost: float = 0.01

    # Skill extraction
    auto_extract_skills: bool = True
    min_repetitions_for_skill: int = 3

    # Consolidation
    consolidation_enabled: bool = True
    similarity_threshold: float = 0.85  # Merge memories above this similarity

    # Forgetting curve
    decay_enabled: bool = True
    decay_half_life_days: float = 30.0
    min_importance_before_decay: float = 0.1


class EvolutionEngine:
    """Evolves memory quality based on task results.

    Core loop:
        Memory retrieved → Agent acts → Result observed → Memory mutated
    """

    def __init__(
        self,
        store: MemoryStore,
        config: EvolutionConfig | None = None,
        llm_call: Callable | None = None,
    ):
        self.store = store
        self.config = config or EvolutionConfig()
        self.llm_call = llm_call
        self._actions: list[EvolutionAction] = []

    def on_task_result(
        self,
        task: TaskExecution,
    ) -> list[EvolutionAction]:
        """Process a task result and evolve memories accordingly.

        Returns list of evolution actions taken.
        """
        actions: list[EvolutionAction] = []
        result = task.result

        if result.status == "success":
            actions.extend(self._handle_success(task))
        elif result.status == "failure":
            actions.extend(self._handle_failure(task))
        elif result.status == "partial":
            actions.extend(self._handle_partial(task))

        # Log task for attribution tracking
        self.store.log_task(task)

        # Log evolution actions
        for action in actions:
            self.store.log_evolution(action)

        self._actions.extend(actions)
        return actions

    def _handle_success(self, task: TaskExecution) -> list[EvolutionAction]:
        """Reinforce memories that contributed to success."""
        actions = []

        for mem_id in task.memories_used:
            mem = self.store.get(mem_id)
            if not mem:
                continue

            self.store.reinforce(mem_id, self.config.reinforce_boost)
            actions.append(EvolutionAction(
                action="reinforce",
                target_memory_id=mem_id,
                reason=f"Contributed to successful task: {task.description[:50]}",
                triggered_by=task.task_id,
            ))

        return actions

    def _handle_failure(self, task: TaskExecution) -> list[EvolutionAction]:
        """Decay misleading memories, create failure cases."""
        actions = []
        result = task.result

        # Decay memories that were used but led to failure
        for mem_id in task.memories_used:
            mem = self.store.get(mem_id)
            if not mem:
                continue

            self.store.decay(mem_id, self.config.decay_penalty)
            actions.append(EvolutionAction(
                action="decay",
                target_memory_id=mem_id,
                reason=f"Led to failure: {result.error_analysis or task.description[:50]}",
                triggered_by=task.task_id,
            ))

        # Create a failure case memory
        if result.error_analysis:
            failure_mem = FailureMemory(
                content=f"FAILURE: {result.error_analysis}",
                what_failed=task.description,
                why=result.error_analysis or "Unknown",
                fix=result.correction or "Needs investigation",
                failure_condition=result.failure_condition or "",
                severity="medium",
                confidence=0.8,
                source_sessions=list(set(
                    s for mid in task.memories_used
                    if (m := self.store.get(mid)) for s in m.source_sessions
                )),
                entities=list(set(
                    e for mid in task.memories_used
                    if (m := self.store.get(mid)) for e in m.entities
                )),
            )
            failure_mem.summary = f"Don't: {task.description[:60]} — {result.error_analysis[:60]}"
            new_id = self.store.add(failure_mem)
            actions.append(EvolutionAction(
                action="create",
                new_memory_id=new_id,
                reason=f"Failure case from task: {task.description[:50]}",
                triggered_by=task.task_id,
            ))

        # Handle contradictions
        if result.contradicts_memory:
            old_mem = self.store.get(result.contradicts_memory)
            if old_mem and result.correction:
                new_mem = MemoryUnit(
                    content=result.correction,
                    type=old_mem.type,
                    summary=f"Corrected: {result.correction[:80]}",
                    entities=old_mem.entities,
                    tags=old_mem.tags,
                    confidence=0.8,
                    source_sessions=old_mem.source_sessions,
                )
                new_id = self.store.supersede(result.contradicts_memory, new_mem)
                actions.append(EvolutionAction(
                    action="supersede",
                    target_memory_id=result.contradicts_memory,
                    new_memory_id=new_id,
                    reason=f"Contradicted by task result: {result.correction[:50]}",
                    triggered_by=task.task_id,
                ))

        return actions

    def _handle_partial(self, task: TaskExecution) -> list[EvolutionAction]:
        """Add caveats/conditions to partially successful memories."""
        actions = []
        result = task.result

        for mem_id in task.memories_used:
            mem = self.store.get(mem_id)
            if not mem:
                continue

            # Small boost (it partially worked)
            self.store.reinforce(mem_id, self.config.partial_boost)

            # If there's a correction, add it as context
            if result.correction and isinstance(mem, ProceduralMemory):
                mem.caveats.append(result.correction)
                self.store.update(mem)

            actions.append(EvolutionAction(
                action="refine",
                target_memory_id=mem_id,
                reason=f"Partial success — added caveat: {result.correction or 'needs review'}",
                triggered_by=task.task_id,
            ))

        return actions

    def extract_skill(
        self,
        description: str,
        steps: list[str],
        context: str = "",
    ) -> str:
        """Create a procedural memory from a successful workflow."""
        proc_mem = ProceduralMemory(
            content=f"Skill: {description}\nSteps: {'; '.join(steps)}",
            skill_name=description,
            steps=steps,
            summary=f"How to: {description}",
            confidence=0.6,
            success_rate=1.0,
            last_succeeded=time.time(),
        )
        return self.store.add(proc_mem)

    def learn_instruction(
        self,
        rule: str,
        scope: str = "global",
        source: str = "explicit",
    ) -> str:
        """Create an instruction memory from user preference."""
        inst_mem = InstructionMemory(
            content=f"Rule: {rule}",
            rule=rule,
            scope=scope,
            source=source,
            summary=f"Instruction: {rule[:80]}",
            confidence=1.0 if source == "explicit" else 0.7,
            importance=0.9,
        )
        return self.store.add(inst_mem)

    def consolidate(self):
        """Merge similar memories and compress the store.

        Uses embedding similarity to find near-duplicates and merges them,
        keeping the higher-confidence version and combining metadata.
        """
        if not self.config.consolidation_enabled:
            return

        # Group by type
        for mem_type in MemoryType:
            mems = self.store.get_by_type(mem_type)
            if len(mems) < 2:
                continue

            # Simple O(n²) pairwise similarity check (fine for <10k memories)
            merged_ids: set[str] = set()
            for i, m1 in enumerate(mems):
                if m1.id in merged_ids:
                    continue
                for m2 in mems[i + 1:]:
                    if m2.id in merged_ids:
                        continue
                    # Check content overlap
                    if _content_similarity(m1.content, m2.content) > self.config.similarity_threshold:
                        # Keep higher confidence, merge metadata
                        keeper = m1 if m1.confidence >= m2.confidence else m2
                        loser = m2 if keeper is m1 else m1
                        keeper.access_count += loser.access_count
                        keeper.success_count += loser.success_count
                        keeper.entities = list(set(keeper.entities + loser.entities))
                        keeper.source_sessions = list(set(keeper.source_sessions + loser.source_sessions))
                        self.store.update(keeper)
                        loser.status = "superseded"
                        loser.superseded_by = keeper.id
                        self.store.update(loser)
                        merged_ids.add(loser.id)

    def apply_forgetting_curve(self):
        """Decay importance of old, unaccessed memories."""
        if not self.config.decay_enabled:
            return

        now = time.time()
        half_life_sec = self.config.decay_half_life_days * 86400

        for mem in list(self.store._memories.values()):
            if mem.status != "active":
                continue
            age = now - mem.last_accessed
            decay_factor = 0.5 ** (age / half_life_sec)
            new_importance = mem.importance * decay_factor
            if new_importance < self.config.min_importance_before_decay:
                mem.importance = self.config.min_importance_before_decay
            else:
                mem.importance = new_importance
            self.store.update(mem)

    def get_stats(self) -> dict[str, Any]:
        """Return evolution statistics."""
        return {
            "total_actions": len(self._actions),
            "actions_by_type": _count_by(self._actions, lambda a: a.action),
            "store_stats": self.store.stats(),
        }


def _content_similarity(a: str, b: str) -> float:
    """Simple Jaccard similarity between two texts."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _count_by(items: list, key_fn) -> dict[str, int]:
    """Count items by a key function."""
    counts: dict[str, int] = {}
    for item in items:
        k = key_fn(item)
        counts[k] = counts.get(k, 0) + 1
    return counts
