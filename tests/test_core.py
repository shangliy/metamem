"""Tests for MetaMem core functionality."""

import tempfile
import time

import pytest

from mem_engram.models import (
    EpisodicMemory,
    EvolutionAction,
    FailureMemory,
    InstructionMemory,
    MemoryType,
    MemoryUnit,
    ProceduralMemory,
    SemanticMemory,
    TaskExecution,
    TaskResult,
)
from mem_engram.store import MemoryStore
from mem_engram.retriever import (
    RetrievalConfig,
    RetrievalEngine,
    classify_intent,
    QueryIntent,
)
from mem_engram.evolution import EvolutionConfig, EvolutionEngine
from mem_engram.extractor import MemoryExtractor


# ── Models ──

class TestModels:
    def test_memory_unit_create(self):
        mem = MemoryUnit(content="Test fact", type=MemoryType.SEMANTIC)
        assert mem.content == "Test fact"
        assert mem.type == MemoryType.SEMANTIC
        assert mem.confidence == 0.7
        assert mem.status == "active"

    def test_memory_unit_serialization(self):
        mem = MemoryUnit(content="Test", entities=["Alice", "Bob"])
        d = mem.to_dict()
        assert d["content"] == "Test"
        assert d["entities"] == ["Alice", "Bob"]
        assert d["type"] == "semantic"

        restored = MemoryUnit.from_dict(d)
        assert restored.content == "Test"
        assert restored.entities == ["Alice", "Bob"]

    def test_typed_memories(self):
        proc = ProceduralMemory(
            content="Deploy steps",
            skill_name="deploy",
            steps=["build", "push", "restart"],
        )
        assert proc.type == MemoryType.PROCEDURAL
        assert proc.steps == ["build", "push", "restart"]

        fail = FailureMemory(
            content="OOM crash",
            what_failed="Service crashed",
            why="Unbounded cache",
            fix="Add eviction policy",
        )
        assert fail.type == MemoryType.FAILURE
        assert fail.severity == "medium"

        inst = InstructionMemory(
            content="Use pnpm",
            rule="Always use pnpm, not npm",
            scope="project:web",
        )
        assert inst.type == MemoryType.INSTRUCTION
        assert inst.scope == "project:web"


# ── Store ──

class TestStore:
    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="metamem_test_")
        self.store = MemoryStore(data_dir=self.tmp_dir)

    def teardown_method(self):
        self.store.close()

    def test_add_and_get(self):
        mem = MemoryUnit(content="Test memory", type=MemoryType.SEMANTIC)
        mem_id = self.store.add(mem)
        retrieved = self.store.get(mem_id)
        assert retrieved is not None
        assert retrieved.content == "Test memory"

    def test_get_by_type(self):
        self.store.add(SemanticMemory(content="Fact 1"))
        self.store.add(SemanticMemory(content="Fact 2"))
        self.store.add(ProceduralMemory(content="Skill 1"))

        semantics = self.store.get_by_type(MemoryType.SEMANTIC)
        assert len(semantics) == 2
        procs = self.store.get_by_type(MemoryType.PROCEDURAL)
        assert len(procs) == 1

    def test_fts_search(self):
        self.store.add(MemoryUnit(content="Docker image building best practices"))
        self.store.add(MemoryUnit(content="Python virtual environments"))
        results = self.store.search_fts("Docker")
        assert len(results) >= 1
        assert "Docker" in results[0].content

    def test_reinforce(self):
        mem = MemoryUnit(content="Good memory", confidence=0.7)
        mem_id = self.store.add(mem)
        self.store.reinforce(mem_id, boost=0.1)
        updated = self.store.get(mem_id)
        assert updated.confidence == pytest.approx(0.8, abs=0.01)
        assert updated.success_count == 1

    def test_decay(self):
        mem = MemoryUnit(content="Bad memory", confidence=0.7)
        mem_id = self.store.add(mem)
        self.store.decay(mem_id, penalty=0.2)
        updated = self.store.get(mem_id)
        assert updated.confidence == pytest.approx(0.5, abs=0.01)
        assert updated.failure_count == 1

    def test_supersede(self):
        old = MemoryUnit(content="Rate limit is 100/min")
        old_id = self.store.add(old)
        new = MemoryUnit(content="Rate limit is 60/min")
        new_id = self.store.supersede(old_id, new)
        old_updated = self.store.get(old_id)
        assert old_updated.status == "superseded"
        assert old_updated.superseded_by == new_id
        new_mem = self.store.get(new_id)
        assert new_mem.supersedes == old_id

    def test_entity_search(self):
        self.store.add(MemoryUnit(content="UserService has OOM", entities=["UserService", "OOM"]))
        self.store.add(MemoryUnit(content="AuthService uses JWT", entities=["AuthService", "JWT"]))
        results = self.store.search_entities(["UserService"])
        assert len(results) == 1
        assert "UserService" in results[0].entities

    def test_stats(self):
        self.store.add(SemanticMemory(content="Fact"))
        self.store.add(FailureMemory(content="Bug"))
        stats = self.store.stats()
        assert stats["total"] == 2
        assert stats["active"] == 2
        assert stats["by_type"]["semantic"] == 1
        assert stats["by_type"]["failure"] == 1


# ── Retriever ──

class TestRetriever:
    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="metamem_test_")
        self.store = MemoryStore(data_dir=self.tmp_dir)
        self.retriever = RetrievalEngine(self.store)

    def teardown_method(self):
        self.store.close()

    def test_intent_classification(self):
        assert classify_intent("how to deploy") == QueryIntent.HOW_TO
        assert classify_intent("what is OAuth") == QueryIntent.WHAT_IS
        assert classify_intent("what happened yesterday") == QueryIntent.WHAT_HAPPENED
        assert classify_intent("error in production") == QueryIntent.AVOID_MISTAKE
        assert classify_intent("always use TypeScript") == QueryIntent.PREFERENCE
        assert classify_intent("random query") == QueryIntent.GENERAL

    def test_search_with_fts(self):
        self.store.add(MemoryUnit(content="Deploy using Docker compose"))
        self.store.add(MemoryUnit(content="Python testing with pytest"))
        results = self.retriever.search("Docker deploy")
        assert len(results) >= 1

    def test_confidence_filtering(self):
        self.store.add(MemoryUnit(content="High confidence", confidence=0.9))
        self.store.add(MemoryUnit(content="Low confidence", confidence=0.05))
        config = RetrievalConfig(min_confidence_threshold=0.1)
        results = self.retriever.search("confidence", config=config)
        # Low confidence should be filtered
        contents = [r.memory.content for r in results]
        assert "Low confidence" not in contents

    def test_progressive_disclosure(self):
        mem = MemoryUnit(content="Full detail here", summary="Short summary")
        self.store.add(mem)
        index = self.retriever.search_index("detail")
        assert len(index) >= 1
        assert "summary" in index[0]
        full = self.retriever.get_full([index[0]["id"]])
        assert len(full) == 1
        assert full[0]["content"] == "Full detail here"


# ── Evolution ──

class TestEvolution:
    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="metamem_test_")
        self.store = MemoryStore(data_dir=self.tmp_dir)
        self.engine = EvolutionEngine(self.store)

    def teardown_method(self):
        self.store.close()

    def test_success_reinforces(self):
        mem = MemoryUnit(content="Good advice", confidence=0.7)
        mem_id = self.store.add(mem)
        task = TaskExecution(
            description="Deploy app",
            memories_used=[mem_id],
            result=TaskResult(status="success"),
        )
        actions = self.engine.on_task_result(task)
        assert len(actions) == 1
        assert actions[0].action == "reinforce"
        updated = self.store.get(mem_id)
        assert updated.confidence > 0.7

    def test_failure_decays_and_creates(self):
        mem = MemoryUnit(content="Wrong advice", confidence=0.7)
        mem_id = self.store.add(mem)
        task = TaskExecution(
            description="Deploy failed",
            memories_used=[mem_id],
            result=TaskResult(
                status="failure",
                error_analysis="Registry auth expired",
                correction="Check auth first",
            ),
        )
        actions = self.engine.on_task_result(task)
        # Should have decay + create failure
        action_types = [a.action for a in actions]
        assert "decay" in action_types
        assert "create" in action_types
        # Original decayed
        updated = self.store.get(mem_id)
        assert updated.confidence < 0.7
        # Failure memory created
        failures = self.store.get_by_type(MemoryType.FAILURE)
        assert len(failures) == 1

    def test_contradiction_supersedes(self):
        old = MemoryUnit(content="API limit is 100/min", confidence=0.8)
        old_id = self.store.add(old)
        task = TaskExecution(
            description="Hit rate limit",
            memories_used=[old_id],
            result=TaskResult(
                status="failure",
                error_analysis="Rate limited at 60",
                correction="API limit is 60/min",
                contradicts_memory=old_id,
            ),
        )
        actions = self.engine.on_task_result(task)
        action_types = [a.action for a in actions]
        assert "supersede" in action_types
        old_updated = self.store.get(old_id)
        assert old_updated.status == "superseded"

    def test_extract_skill(self):
        mem_id = self.engine.extract_skill(
            description="Deploy to production",
            steps=["build", "test", "push", "restart"],
        )
        mem = self.store.get(mem_id)
        assert mem.type == MemoryType.PROCEDURAL

    def test_learn_instruction(self):
        mem_id = self.engine.learn_instruction(
            rule="Always use pnpm",
            scope="project:web",
        )
        mem = self.store.get(mem_id)
        assert mem.type == MemoryType.INSTRUCTION
        assert mem.confidence == 1.0


# ── Extractor ──

class TestExtractor:
    def test_heuristic_extraction(self):
        extractor = MemoryExtractor(llm_call=None)
        turns = [
            {"speaker": "user", "text": "I got an error: Connection refused when trying to reach the database"},
            {"speaker": "assistant", "text": "First, check if the DB is running. Then verify the connection string. Finally restart the service."},
            {"speaker": "user", "text": "Always use port 5433 for the test database, never 5432"},
        ]
        memories = extractor.extract_from_turns(turns, session_id="test_session")
        assert len(memories) >= 2
        types = [m.type for m in memories]
        assert MemoryType.FAILURE in types
        assert MemoryType.INSTRUCTION in types or MemoryType.PROCEDURAL in types
