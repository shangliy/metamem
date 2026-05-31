"""Benchmark runner — MetaMem retrieval evaluated with EvolveMem's benchmark infrastructure.

Borrows from EvolveMem:
- LoCoMoAdapter: data loading, per-category answer prompts, token_f1 scoring
- BenchmarkSample / BenchmarkAdapter protocol
- Evolution diagnosis loop structure

MetaMem's own stack:
- MemoryExtractor: LLM-driven typed memory extraction from conversation turns
- MemoryStore: SQLite + embeddings typed store
- RetrievalEngine: intent-aware multi-view retrieval (semantic + FTS + entity graph)

LLM: Anthropic (claude-haiku for extraction/eval, override via LLM_MODEL env var)
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np

# ── EvolveMem adapters (borrowed) — lazy import so path can be set at runtime ─
def _import_evolvemem():
    _EVOLVEMEM = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../EvolveMem")
    )
    if _EVOLVEMEM not in sys.path:
        sys.path.insert(0, _EVOLVEMEM)
    from evolvemem.benchmarks import LoCoMoAdapter, BenchmarkSample  # noqa
    from evolvemem.benchmarks.base import token_f1                   # noqa
    return LoCoMoAdapter, BenchmarkSample, token_f1

# ── MetaMem stack ─────────────────────────────────────────────────────────────
from ..extractor import MemoryExtractor
from ..models import MemoryType, MemoryUnit
from ..retriever import RetrievalConfig, RetrievalEngine, format_context
from ..store import MemoryStore

logger = logging.getLogger(__name__)


# ── LLM (Anthropic) ──────────────────────────────────────────────────────────

def _make_llm_call(model: str | None = None) -> Callable:
    import anthropic
    _model = model or os.environ.get("LLM_MODEL", "claude-haiku-4-5-20251001")
    client = anthropic.Anthropic()

    def llm_call(messages, max_tokens: int = 1024, temperature: float = 0.1):
        system, user_messages = "", []
        for m in messages:
            if m.get("role") == "system":
                system = m.get("content", "")
            else:
                user_messages.append({"role": m["role"], "content": m["content"]})
        if not user_messages:
            return ""
        kwargs: dict = dict(model=_model, max_tokens=min(max_tokens, 4096),
                            messages=user_messages)
        if system:
            kwargs["system"] = system
        for attempt in range(3):
            try:
                r = client.messages.create(**kwargs)
                return (r.content[0].text or "").strip()
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                else:
                    logger.warning("LLM call failed: %s", e)
                    return ""
        return ""

    return llm_call


# ── Scoring ───────────────────────────────────────────────────────────────────

def _parse_answer(raw: str) -> str:
    """Extract answer field from JSON response, fallback to raw."""
    try:
        m = re.search(r'"answer"\s*:\s*"([^"]*)"', raw)
        if m:
            return m.group(1).strip()
        obj = json.loads(raw)
        return str(obj.get("answer", raw)).strip()
    except Exception:
        return raw.strip()


# ── Evolution ─────────────────────────────────────────────────────────────────

@dataclass
class RoundResult:
    round_id: int
    f1: float
    by_category: dict[str, float] = field(default_factory=dict)
    total: int = 0
    correct: int = 0
    config: dict = field(default_factory=dict)
    improvements: list[str] = field(default_factory=list)
    memory_count: int = 0


@dataclass
class EvolutionResult:
    rounds: list[RoundResult] = field(default_factory=list)
    best_round: int = 0
    best_f1: float = 0.0
    final_config: dict = field(default_factory=dict)

    def trajectory(self) -> str:
        lines = ["Round |   F1   | Memories | Config changes"]
        lines.append("-" * 60)
        for r in self.rounds:
            changes = ", ".join(r.improvements[:3]) if r.improvements else "—"
            lines.append(f"  {r.round_id:2d}  | {r.f1:.4f} |   {r.memory_count:4d}   | {changes}")
        lines.append(f"\nBest: Round {self.best_round} → F1={self.best_f1:.4f}")
        return "\n".join(lines)


def _diagnose_and_suggest(
    results: list[dict],
    config: RetrievalConfig,
    llm_call: Callable,
) -> list[str]:
    """Use LLM to diagnose failures and propose RetrievalConfig changes."""
    failures = [r for r in results if r.get("score", 0) < 0.3]
    if not failures:
        return []

    summary = "\n".join(
        f"Cat{r.get('category',0)} Q: {r['question'][:80]} | Gold: {r['reference'][:30]} | Got: {r['prediction'][:30]}"
        for r in failures[:10]
    )
    prompt = (
        f"Analyze these memory-QA failures and suggest config changes.\n\n"
        f"Current config: fusion={config.fusion_mode}, sem_k={config.semantic_top_k}, "
        f"kw_k={config.keyword_top_k}, intent={config.enable_intent_routing}, "
        f"conf_boost={config.confidence_boost_weight}\n\n"
        f"Failures:\n{summary}\n\n"
        "Suggest 1-3 changes as JSON array: [{\"field\": \"...\", \"value\": ...}]\n"
        "Valid fields: semantic_top_k, keyword_top_k, fusion_mode, enable_intent_routing, "
        "confidence_boost_weight, weight_episodic, weight_semantic_type, weight_procedural"
    )
    raw = llm_call([{"role": "user", "content": prompt}], max_tokens=256)
    suggestions = []
    try:
        m = re.search(r'\[.*?\]', raw, re.DOTALL)
        if m:
            for item in json.loads(m.group()):
                if "field" in item and "value" in item:
                    suggestions.append(f"{item['field']}={item['value']}")
    except Exception:
        pass
    return suggestions


def _apply_suggestions(config: RetrievalConfig, suggestions: list[str]) -> RetrievalConfig:
    import copy
    cfg = copy.deepcopy(config)
    for s in suggestions:
        try:
            k, v = s.split("=", 1)
            k = k.strip()
            if not hasattr(cfg, k):
                continue
            cur = getattr(cfg, k)
            if isinstance(cur, bool):
                setattr(cfg, k, v.strip().lower() == "true")
            elif isinstance(cur, int):
                setattr(cfg, k, int(float(v)))
            elif isinstance(cur, float):
                setattr(cfg, k, float(v))
            else:
                setattr(cfg, k, v.strip().strip("'\""))
        except Exception:
            continue
    return cfg


# ── Main runner ───────────────────────────────────────────────────────────────

def run_benchmark(
    benchmark_name: str = "locomo",
    data_path: str | None = None,
    max_rounds: int = 3,
    initial: str = "weak",
    sample_indices: list[int] | None = None,
    max_qa: int | None = None,
    no_embeddings: bool = False,
):
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    print("=" * 64)
    print(f"MetaMem Benchmark  ·  {benchmark_name}  ·  initial={initial}")
    print("=" * 64)

    llm_call = _make_llm_call()
    print(f"LLM: {os.environ.get('LLM_MODEL', 'claude-haiku-4-5-20251001')}")

    # Embedder
    embedder = None
    if not no_embeddings:
        try:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            print("Embedder: all-MiniLM-L6-v2")
        except Exception as e:
            print(f"Embedder unavailable ({e}) — semantic search disabled")

    # Load data via EvolveMem adapter
    LoCoMoAdapter, BenchmarkSample, token_f1 = _import_evolvemem()
    if benchmark_name == "locomo":
        adapter = LoCoMoAdapter()
        path = data_path or "data/locomo10.json"
        samples = adapter.load(path, sample_indices=sample_indices, max_qa=max_qa)
    else:
        raise ValueError(f"Benchmark '{benchmark_name}' not yet wired — only 'locomo' for now")

    all_sessions: list[tuple[str, str, list[dict]]] = []
    all_qa: list[dict] = []
    for s in samples:
        for sid, date, turns in s.sessions:
            all_sessions.append((f"{s.sample_id}::{sid}", date, turns))
        for qa in s.qa_pairs:
            qa2 = dict(qa); qa2["_sample_id"] = s.sample_id
            all_qa.append(qa2)

    print(f"Samples: {len(samples)} | Sessions: {len(all_sessions)} | QA: {len(all_qa)}")
    print(f"Primary metric: {adapter.primary_metric}")

    # Extract memories from all sessions (MetaMem extractor)
    tmp_dir = tempfile.mkdtemp(prefix="mem_engram_bench_")
    store = MemoryStore(data_dir=tmp_dir, embedder=embedder)
    extractor = MemoryExtractor(llm_call=llm_call)

    print("\nExtracting memories from sessions...")
    memories = extractor.extract_from_sessions(all_sessions)
    for mem in memories:
        store.add(mem)
    print(f"Extracted {len(memories)} typed memories → {store.stats()}")

    # Initial RetrievalConfig
    if initial == "weak":
        ret_cfg = RetrievalConfig(
            semantic_top_k=0, keyword_top_k=5, structured_top_k=0,
            max_context=8, fusion_mode="keyword_only",
            enable_intent_routing=False, enable_entity_graph=False,
            confidence_boost_weight=0.0, enable_result_feedback=False,
        )
    else:
        ret_cfg = RetrievalConfig()

    # Evolution loop
    evolution = EvolutionResult()
    best_f1 = 0.0
    run_id = time.strftime(f"{benchmark_name}_{initial}_%Y%m%d_%H%M%S")
    results_dir = f"benchmark_results/{benchmark_name}/{run_id}"
    os.makedirs(results_dir, exist_ok=True)

    for round_id in range(max_rounds):
        print(f"\n--- Round {round_id} "
              f"[fusion={ret_cfg.fusion_mode} sem_k={ret_cfg.semantic_top_k} "
              f"kw_k={ret_cfg.keyword_top_k} intent={ret_cfg.enable_intent_routing}] ---")

        retriever = RetrievalEngine(store, ret_cfg)
        round_results: list[dict] = []
        total_f1 = 0.0
        by_cat: dict[str, list[float]] = {}

        for qa in all_qa:
            question = qa.get("question", "")
            reference = qa.get("answer", "")
            category = int(qa.get("category", 0))
            if not question or not reference:
                continue

            # Retrieve context (MetaMem retriever)
            query_emb = embedder.encode(question) if embedder else None
            retrieved = retriever.search(question, config=ret_cfg, query_embedding=query_emb)
            context = format_context(retrieved, max_tokens=2000)

            # Build answer prompt (EvolveMem per-category prompt)
            system, user = adapter.build_answer_prompt(question, context, qa)
            raw = llm_call([{"role": "system", "content": system},
                            {"role": "user", "content": user}], max_tokens=256)
            prediction = _parse_answer(raw)

            # Score (EvolveMem token_f1)
            score = token_f1(prediction, reference)
            total_f1 += score
            by_cat.setdefault(str(category), []).append(score)

            round_results.append({
                "question": question, "reference": reference,
                "prediction": prediction, "score": score,
                "category": category,
                "memories_used": [rm.memory.id for rm in retrieved[:5]],
            })

            # Evolution feedback — reinforce/decay retrieved memories
            if ret_cfg.enable_result_feedback:
                for rm in retrieved[:5]:
                    if score > 0.5:
                        store.reinforce(rm.memory.id)
                    elif score < 0.1:
                        store.decay(rm.memory.id, penalty=0.05)

        n = len(round_results)
        avg_f1 = total_f1 / max(n, 1)
        cat_f1 = {cat: sum(scores) / len(scores) for cat, scores in by_cat.items()}

        print(f"  F1={avg_f1:.4f}  correct(>0.5)={sum(1 for r in round_results if r['score']>0.5)}/{n}")
        for cat, cf1 in sorted(cat_f1.items()):
            print(f"    cat{cat}: {cf1:.4f}")

        rr = RoundResult(
            round_id=round_id, f1=avg_f1, by_category=cat_f1,
            total=n, correct=sum(1 for r in round_results if r["score"] > 0.5),
            config=asdict(ret_cfg), memory_count=len(store._memories),
        )

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            evolution.best_round = round_id
            evolution.best_f1 = avg_f1
            evolution.final_config = asdict(ret_cfg)

        # Save round detail
        round_file = os.path.join(results_dir, f"round_{round_id}.jsonl")
        with open(round_file, "w") as f:
            for r in round_results:
                f.write(json.dumps(r) + "\n")

        # Diagnose + evolve (skip last round)
        if round_id < max_rounds - 1:
            suggestions = _diagnose_and_suggest(round_results, ret_cfg, llm_call)
            if suggestions:
                ret_cfg = _apply_suggestions(ret_cfg, suggestions)
                rr.improvements = suggestions
                print(f"  → Improvements: {suggestions}")
            else:
                # Manual stepping when diagnosis is silent
                if round_id == 0 and ret_cfg.semantic_top_k == 0:
                    ret_cfg.semantic_top_k = 15
                    ret_cfg.fusion_mode = "rrf"
                    rr.improvements = ["semantic_top_k=15", "fusion_mode=rrf"]
                elif round_id == 1 and not ret_cfg.enable_intent_routing:
                    ret_cfg.enable_intent_routing = True
                    ret_cfg.confidence_boost_weight = 0.3
                    ret_cfg.enable_result_feedback = True
                    rr.improvements = ["intent_routing=True", "conf_boost=0.3", "feedback=True"]
                if rr.improvements:
                    print(f"  → Manual step: {rr.improvements}")

        evolution.rounds.append(rr)

    # Final report
    print(f"\n{'=' * 64}")
    print("EVOLUTION COMPLETE")
    print("=" * 64)
    print(evolution.trajectory())

    summary = {
        "run_id": run_id, "benchmark": benchmark_name, "initial": initial,
        "best_round": evolution.best_round, "best_f1": evolution.best_f1,
        "final_config": evolution.final_config,
        "store_stats": store.stats(),
        "rounds": [asdict(r) for r in evolution.rounds],
    }
    with open(os.path.join(results_dir, "evolution_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nArtifacts: {results_dir}/")

    store.close()
    return evolution
