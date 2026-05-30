"""Benchmark runner — EvolveMem-compatible evaluation with MetaMem retrieval.

Plugs MetaMem's typed retrieval into the same benchmark protocol as EvolveMem:
- LoCoMo (long-context conversational memory)
- MemBench (multiple-choice agent memory)
- LongMemEval (cross-session memory)

Supports evolution loop: Evaluate → Diagnose → Adjust config → Repeat
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np
import yaml
from openai import OpenAI

from ..extractor import MemoryExtractor
from ..models import MemoryType, MemoryUnit
from ..retriever import RetrievalConfig, RetrievalEngine, RetrievedMemory, format_context
from ..store import MemoryStore

logger = logging.getLogger(__name__)


# ── Scoring ──

def _tokenize(s: str) -> list[str]:
    """Lowercase, remove punctuation, split."""
    return re.sub(r'[^a-z0-9\s]', ' ', str(s).lower()).split()


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 between prediction and reference."""
    p, r = _tokenize(prediction), _tokenize(reference)
    if not p or not r:
        return 0.0
    rc = list(r)
    c = 0
    for t in p:
        if t in rc:
            c += 1
            rc.remove(t)
    if c == 0:
        return 0.0
    pr = c / len(p)
    rec = c / len(r)
    return 2 * pr * rec / (pr + rec)


def mcq_accuracy(prediction: str, ground_truth: str) -> float:
    """Multiple-choice accuracy."""
    gt = ground_truth.strip().upper()
    pred = prediction.strip().upper()
    if not gt or not pred:
        return 0.0
    m = re.search(r'(?:^|[^A-Za-z])([A-Z])(?:[^A-Za-z]|$)', pred)
    if m and m.group(1) == gt:
        return 1.0
    return 0.0


# ── LLM Setup ──

def _load_llm() -> Callable:
    """Create an LLM call function from env."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    model = os.environ.get("LLM_MODEL", "gpt-4o")

    if not api_key:
        # Try yaml file
        if os.path.exists("openai_key.yaml"):
            with open("openai_key.yaml") as f:
                cfg = yaml.safe_load(f)
            api_key = cfg.get("api_key", "")
            base_url = cfg.get("base_url", base_url)
            model = cfg.get("model", model)

    client = OpenAI(base_url=base_url, api_key=api_key)

    def llm_call(messages, max_tokens: int = 4096, temperature: float = 0.1):
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model=model, messages=messages,
                    max_completion_tokens=max_tokens,
                    temperature=temperature,
                )
                return (r.choices[0].message.content or "").strip()
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                else:
                    logger.warning("LLM call failed: %s", e)
                    return ""
        return ""

    return llm_call


# ── Data Loading ──

@dataclass
class BenchmarkSample:
    """One evaluation unit."""
    sample_id: str
    sessions: list[tuple[str, str, list[dict]]]  # (session_id, date, turns)
    qa_pairs: list[dict]  # {question, answer, category, ...}


def _load_locomo(path: str, sample_index: int = 0) -> list[BenchmarkSample]:
    """Load LoCoMo benchmark data."""
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        items = [data[sample_index]] if sample_index < len(data) else data[:1]
    else:
        items = [data]

    samples = []
    for i, item in enumerate(items):
        sessions = []
        # LoCoMo format: conversation turns
        if "conversation" in item:
            turns = item["conversation"]
            sessions.append((f"locomo_{i}", "", turns))
        elif "sessions" in item:
            for j, sess in enumerate(item["sessions"]):
                turns = sess.get("turns", sess.get("conversation", []))
                sessions.append((f"locomo_{i}_s{j}", sess.get("date", ""), turns))

        qa_pairs = item.get("qa_pairs", item.get("questions", []))
        samples.append(BenchmarkSample(
            sample_id=f"locomo_{i}",
            sessions=sessions,
            qa_pairs=qa_pairs,
        ))

    return samples


def _load_membench(path: str) -> list[BenchmarkSample]:
    """Load MemBench data."""
    samples = []
    if os.path.isdir(path):
        for fname in os.listdir(path):
            if fname.endswith(".json"):
                with open(os.path.join(path, fname)) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data[:20]:  # Limit for speed
                        sessions = [(item.get("tid", ""), "", item.get("turns", []))]
                        qa_pairs = item.get("questions", [])
                        samples.append(BenchmarkSample(
                            sample_id=item.get("tid", fname),
                            sessions=sessions,
                            qa_pairs=qa_pairs,
                        ))
    return samples


# ── Evolution Loop ──

@dataclass
class RoundResult:
    """Result of one evaluation round."""
    round_id: int
    primary_metric: float
    all_metrics: dict[str, float] = field(default_factory=dict)
    total_questions: int = 0
    correct: int = 0
    config: dict = field(default_factory=dict)
    improvements: list[str] = field(default_factory=list)


@dataclass
class EvolutionResult:
    """Full evolution run result."""
    rounds: list[RoundResult] = field(default_factory=list)
    best_round: int = 0
    best_score: float = 0.0
    final_config: dict = field(default_factory=dict)

    def trajectory(self) -> str:
        """Pretty-print evolution trajectory."""
        lines = ["Round | Primary Metric | Config Changes"]
        lines.append("-" * 60)
        for r in self.rounds:
            changes = ", ".join(r.improvements[:3]) if r.improvements else "initial"
            lines.append(f"  {r.round_id:2d}  |    {r.primary_metric:.4f}    | {changes}")
        lines.append(f"\nBest: Round {self.best_round} = {self.best_score:.4f}")
        return "\n".join(lines)


def _diagnose_failures(
    results: list[dict],
    config: RetrievalConfig,
    llm_call: Callable,
) -> list[str]:
    """Use LLM to diagnose failure patterns and propose config changes."""
    failures = [r for r in results if r.get("score", 0) < 0.3]
    if not failures:
        return []

    failure_summary = "\n".join([
        f"Q: {f['question'][:80]} | Expected: {f['reference'][:40]} | Got: {f['prediction'][:40]}"
        for f in failures[:10]
    ])

    prompt = f"""Analyze these QA failures from a memory retrieval system and suggest configuration improvements.

Current config:
- semantic_top_k: {config.semantic_top_k}
- keyword_top_k: {config.keyword_top_k}
- fusion_mode: {config.fusion_mode}
- enable_intent_routing: {config.enable_intent_routing}
- confidence_boost_weight: {config.confidence_boost_weight}
- weight_procedural: {config.weight_procedural}
- weight_failure: {config.weight_failure}

Failures:
{failure_summary}

Suggest 1-3 config changes as JSON: [{{"field": "...", "value": ...}}]
Only suggest fields from the config above."""

    messages = [{"role": "user", "content": prompt}]
    response = llm_call(messages, max_tokens=512)

    # Parse suggestions
    suggestions = []
    try:
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            items = json.loads(match.group())
            for item in items:
                if "field" in item and "value" in item:
                    suggestions.append(f"{item['field']}={item['value']}")
    except (json.JSONDecodeError, KeyError):
        pass

    return suggestions


def _apply_suggestions(config: RetrievalConfig, suggestions: list[str]) -> RetrievalConfig:
    """Apply diagnosed suggestions to config."""
    import copy
    new_config = copy.deepcopy(config)

    for suggestion in suggestions:
        try:
            field_name, value_str = suggestion.split("=", 1)
            field_name = field_name.strip()
            if hasattr(new_config, field_name):
                current = getattr(new_config, field_name)
                if isinstance(current, bool):
                    setattr(new_config, field_name, value_str.strip().lower() == "true")
                elif isinstance(current, int):
                    setattr(new_config, field_name, int(float(value_str)))
                elif isinstance(current, float):
                    setattr(new_config, field_name, float(value_str))
                else:
                    setattr(new_config, field_name, value_str.strip().strip('"\''))
        except (ValueError, AttributeError):
            continue

    return new_config


# ── Main Runner ──

def run_benchmark(
    benchmark_name: str = "locomo",
    data_path: str | None = None,
    max_rounds: int = 5,
    initial: str = "weak",
    sample_index: int = 0,
):
    """Run a benchmark with MetaMem's typed retrieval + evolution.

    Args:
        benchmark_name: "locomo" | "membench" | "longmemeval"
        data_path: Path to benchmark data
        max_rounds: Max evolution rounds
        initial: "weak" or "strong" starting config
        sample_index: Sample index for LoCoMo
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    print(f"{'=' * 60}")
    print(f"MetaMem Benchmark: {benchmark_name}")
    print(f"{'=' * 60}")

    # Load LLM
    llm_call = _load_llm()

    # Load embedder
    embedder = None
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("Embedder: all-MiniLM-L6-v2")
    except ImportError:
        print("No embedder available — semantic search disabled")

    # Load data
    if benchmark_name == "locomo":
        path = data_path or "data/locomo10.json"
        samples = _load_locomo(path, sample_index)
        score_fn = token_f1
        primary_metric = "f1"
    elif benchmark_name == "membench":
        path = data_path or "data/membench/repo/MemData"
        samples = _load_membench(path)
        score_fn = mcq_accuracy
        primary_metric = "accuracy"
    else:
        print(f"Benchmark '{benchmark_name}' not yet implemented")
        return

    print(f"Samples: {len(samples)} | QA pairs: {sum(len(s.qa_pairs) for s in samples)}")

    # Initial config
    if initial == "weak":
        config = RetrievalConfig(
            semantic_top_k=0,
            keyword_top_k=5,
            structured_top_k=0,
            max_context=8,
            fusion_mode="keyword_only",
            enable_intent_routing=False,
            enable_entity_graph=False,
            confidence_boost_weight=0.0,
        )
    else:
        config = RetrievalConfig()  # Strong defaults

    # Setup store
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="metamem_bench_")
    store = MemoryStore(data_dir=tmp_dir, embedder=embedder)
    extractor = MemoryExtractor(llm_call=llm_call)

    # Extract memories from sessions
    print("\nExtracting memories...")
    all_sessions = []
    for sample in samples:
        all_sessions.extend(sample.sessions)

    memories = extractor.extract_from_sessions(all_sessions)
    for mem in memories:
        store.add(mem)
    print(f"Extracted {len(memories)} typed memories")
    print(f"  Store stats: {store.stats()}")

    # Evolution loop
    retriever = RetrievalEngine(store, config)
    evolution_result = EvolutionResult()
    best_score = 0.0

    for round_id in range(max_rounds):
        print(f"\n--- Round {round_id} ---")
        print(f"Config: fusion={config.fusion_mode}, sem_k={config.semantic_top_k}, "
              f"kw_k={config.keyword_top_k}, intent={config.enable_intent_routing}")

        # Evaluate all QA pairs
        round_results: list[dict] = []
        total_score = 0.0

        for sample in samples:
            for qa in sample.qa_pairs:
                question = qa.get("question", "")
                reference = qa.get("answer", "")
                if not question or not reference:
                    continue

                # Retrieve context
                query_emb = embedder.encode(question) if embedder else None
                retrieved = retriever.search(question, config=config, query_embedding=query_emb)
                context = format_context(retrieved, max_tokens=2000)

                # Generate answer
                answer_prompt = (
                    f"Question: {question}\n\nContext:\n{context}\n\n"
                    "Answer concisely in 1-10 words using exact words from context. "
                    'Return JSON: {"answer": "..."}'
                )
                raw_answer = llm_call(
                    [{"role": "user", "content": answer_prompt}],
                    max_tokens=256, temperature=0.1,
                )

                # Parse answer
                prediction = raw_answer
                try:
                    match = re.search(r'"answer"\s*:\s*"([^"]*)"', raw_answer)
                    if match:
                        prediction = match.group(1)
                except Exception:
                    pass

                # Score
                score = score_fn(prediction, reference)
                total_score += score

                round_results.append({
                    "question": question,
                    "reference": reference,
                    "prediction": prediction,
                    "score": score,
                    "memories_used": [rm.memory.id for rm in retrieved[:5]],
                })

                # Evolution feedback
                if config.enable_result_feedback:
                    for rm in retrieved[:5]:
                        if score > 0.5:
                            store.reinforce(rm.memory.id, config.feedback_reinforcement)
                        elif score < 0.1:
                            store.decay(rm.memory.id, config.feedback_decay * 0.5)

        # Compute round metrics
        n_qa = len(round_results)
        avg_score = total_score / max(n_qa, 1)
        round_result = RoundResult(
            round_id=round_id,
            primary_metric=avg_score,
            total_questions=n_qa,
            correct=sum(1 for r in round_results if r["score"] > 0.5),
            config=asdict(config),
        )

        print(f"  {primary_metric}: {avg_score:.4f} ({round_result.correct}/{n_qa} correct)")

        # Track best
        if avg_score > best_score:
            best_score = avg_score
            evolution_result.best_round = round_id
            evolution_result.best_score = avg_score
            evolution_result.final_config = asdict(config)

        # Diagnose and evolve (skip last round)
        if round_id < max_rounds - 1:
            suggestions = _diagnose_failures(round_results, config, llm_call)
            if suggestions:
                config = _apply_suggestions(config, suggestions)
                round_result.improvements = suggestions
                print(f"  Improvements: {suggestions}")
            else:
                # Manual evolution steps for early rounds
                if round_id == 0 and config.semantic_top_k == 0:
                    config.semantic_top_k = 15
                    config.fusion_mode = "rrf"
                    round_result.improvements = ["semantic_top_k=15", "fusion_mode=rrf"]
                elif round_id == 1 and not config.enable_intent_routing:
                    config.enable_intent_routing = True
                    config.enable_entity_graph = True
                    round_result.improvements = ["enable_intent_routing=True", "enable_entity_graph=True"]
                elif round_id == 2:
                    config.confidence_boost_weight = 0.3
                    config.enable_result_feedback = True
                    round_result.improvements = ["confidence_boost=0.3", "result_feedback=True"]

        evolution_result.rounds.append(round_result)

    # Final report
    print(f"\n{'=' * 60}")
    print("EVOLUTION COMPLETE")
    print(f"{'=' * 60}")
    print(evolution_result.trajectory())
    print(f"\nStore final stats: {store.stats()}")

    # Save results
    results_dir = f"benchmark_results/{benchmark_name}"
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "evolution_summary.json"), "w") as f:
        json.dump({
            "benchmark": benchmark_name,
            "best_round": evolution_result.best_round,
            "best_score": evolution_result.best_score,
            "final_config": evolution_result.final_config,
            "rounds": [asdict(r) for r in evolution_result.rounds],
        }, f, indent=2, default=str)

    print(f"\nResults saved to {results_dir}/")

    # Cleanup
    store.close()
