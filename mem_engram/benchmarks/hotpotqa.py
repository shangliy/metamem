"""HotpotQA benchmark adapter for MetaMem core library.

Dataset: hotpot_qa (distractor split) — publicly available on HuggingFace.
  10 paragraphs per question (2 supporting + 8 distractors).
  Task: retrieve the 2 supporting paragraphs and answer the question.

Metrics:
  - Answer F1  : token-level F1 vs ground truth answer
  - Retrieval recall : fraction of supporting paragraphs retrieved in top-k
  - By type   : bridge vs comparison
  - By level  : easy / medium / hard

Why this fits MetaMem:
  - Pure retrieval challenge (signal vs. noise)
  - Multi-hop reasoning (bridge questions need 2 connected facts)
  - No LLM extraction needed — store raw paragraphs, test retrieval only
  - 7405 validation examples — statistically meaningful
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np

from ..models import MemoryType, MemoryUnit
from ..retriever import RetrievalConfig, RetrievalEngine, IterativeRetrievalEngine, format_context
from ..store import MemoryStore

logger = logging.getLogger(__name__)


# ── Scoring ───────────────────────────────────────────────────────────────────

_PUNCT = re.compile(r"[^a-z0-9\s]")
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "yes": "yes", "no": "no",
}


def _tokenize(s: str) -> list[str]:
    tokens = _PUNCT.sub(" ", str(s).lower()).split()
    return [_NUMBER_WORDS.get(t, t) for t in tokens]


def token_f1(pred: str, ref: str) -> float:
    p, r = _tokenize(pred), _tokenize(ref)
    if not p or not r:
        return 0.0
    rc = list(r)
    c = sum(1 for t in p if t in rc and not rc.remove(t))  # type: ignore[func-returns-value]
    if c == 0:
        return 0.0
    pr, rec = c / len(p), c / len(r)
    return 2 * pr * rec / (pr + rec)


def retrieval_recall(retrieved_titles: list[str], supporting_titles: list[str]) -> float:
    """Fraction of supporting paragraph titles found in retrieved set."""
    if not supporting_titles:
        return 1.0
    hit = sum(1 for t in supporting_titles if t in retrieved_titles)
    return hit / len(supporting_titles)


# ── Data loading ──────────────────────────────────────────────────────────────

@dataclass
class HotpotSample:
    question: str
    answer: str
    qtype: str          # "bridge" | "comparison"
    level: str          # "easy" | "medium" | "hard"
    paragraphs: list[dict]         # [{"title": str, "text": str}]
    supporting_titles: list[str]   # which paragraph titles contain the answer


def load_hotpotqa(
    n_samples: int = 200,
    seed: int = 42,
    qtype_filter: str | None = None,   # "bridge" | "comparison" | None
) -> list[HotpotSample]:
    """Load n_samples from hotpot_qa validation (distractor) split."""
    from datasets import load_dataset
    ds = load_dataset("hotpot_qa", "distractor", split="validation", streaming=True)

    samples: list[HotpotSample] = []
    for row in ds:
        if qtype_filter and row["type"] != qtype_filter:
            continue
        paragraphs = [
            {"title": title, "text": " ".join(sents)}
            for title, sents in zip(
                row["context"]["title"],
                row["context"]["sentences"],
            )
        ]
        samples.append(HotpotSample(
            question=row["question"],
            answer=row["answer"],
            qtype=row["type"],
            level=row["level"],
            paragraphs=paragraphs,
            supporting_titles=list(set(row["supporting_facts"]["title"])),
        ))
        if len(samples) >= n_samples:
            break

    return samples


# ── Per-sample store ──────────────────────────────────────────────────────────

def _build_sample_store(
    sample: HotpotSample,
    embedder,
    tmp_root: str,
) -> tuple[MemoryStore, dict[str, str]]:
    """Store the 10 paragraphs as raw semantic MemoryUnits. Returns (store, id→title map)."""
    store_dir = tempfile.mkdtemp(dir=tmp_root, prefix="hpqa_")
    store = MemoryStore(data_dir=store_dir, embedder=embedder)
    id_to_title: dict[str, str] = {}

    for para in sample.paragraphs:
        mem = MemoryUnit(
            content=para["text"],
            type=MemoryType.SEMANTIC,
            summary=para["title"],
            entities=[para["title"]],
            tags=["hotpotqa", "paragraph"],
            importance=0.7,
            confidence=0.9,
        )
        store.add(mem)
        id_to_title[mem.id] = para["title"]

    return store, id_to_title


# ── Answer generation ─────────────────────────────────────────────────────────

def _answer(question: str, context: str, llm_call: Callable) -> str:
    system = "You are a precise QA assistant. Answer based only on the provided context."
    user = (
        f"Question: {question}\n\nContext:\n{context}\n\n"
        "Rules:\n"
        "1. Answer in 1-5 words using exact words from context.\n"
        "2. Yes/No questions → 'yes' or 'no'.\n"
        "3. Never refuse; pick the most supported answer.\n"
        'Return JSON: {"answer": "..."}'
    )
    raw = llm_call(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=64, temperature=0.0,
    )
    try:
        m = re.search(r'"answer"\s*:\s*"([^"]*)"', raw)
        return m.group(1).strip() if m else raw.strip()
    except Exception:
        return raw.strip()


# ── Round result ──────────────────────────────────────────────────────────────

@dataclass
class HotpotRound:
    round_id: int
    f1: float
    recall: float               # retrieval recall for supporting paragraphs
    by_type: dict[str, dict]    # {"bridge": {f1, recall, n}, "comparison": ...}
    by_level: dict[str, dict]   # {"easy": ..., "medium": ..., "hard": ...}
    total: int = 0
    config: dict = field(default_factory=dict)
    improvements: list[str] = field(default_factory=list)

    def summary_line(self) -> str:
        return (
            f"Round {self.round_id:2d}  F1={self.f1:.4f}  "
            f"Recall={self.recall:.4f}  n={self.total}"
        )


@dataclass
class HotpotResult:
    rounds: list[HotpotRound] = field(default_factory=list)
    best_round: int = 0
    best_f1: float = 0.0
    final_config: dict = field(default_factory=dict)

    def trajectory(self) -> str:
        lines = [
            f"{'Round':>5}  {'F1':>7}  {'Recall':>7}  {'Config change'}",
            "-" * 65,
        ]
        for r in self.rounds:
            ch = ", ".join(r.improvements[:3]) if r.improvements else "—"
            lines.append(
                f"{r.round_id:>5}  {r.f1:>7.4f}  {r.recall:>7.4f}  {ch}"
            )
        lines.append(f"\nBest: Round {self.best_round} → F1={self.best_f1:.4f}")
        return "\n".join(lines)


# ── Diagnosis ────────────────────────────────────────────────────────────────

def _diagnose(failures: list[dict], config: RetrievalConfig, llm_call: Callable) -> list[str]:
    if not failures:
        return []
    summary = "\n".join(
        f"type={r['qtype']} Q: {r['question'][:70]} | gold: {r['answer'][:20]} | pred: {r['prediction'][:20]}"
        for r in failures[:8]
    )
    prompt = (
        f"These HotpotQA failures come from a memory retrieval system.\n\n"
        f"Config: fusion={config.fusion_mode} sem_k={config.semantic_top_k} "
        f"kw_k={config.keyword_top_k} intent={config.enable_intent_routing}\n\n"
        f"Failures:\n{summary}\n\n"
        "Suggest 1-3 config changes as JSON array: [{\"field\": \"...\", \"value\": ...}]\n"
        "Valid fields: semantic_top_k (int), keyword_top_k (int), fusion_mode (str: rrf|weighted_sum|semantic_only), "
        "enable_intent_routing (bool), confidence_boost_weight (float), max_context (int)"
    )
    raw = llm_call([{"role": "user", "content": prompt}], max_tokens=256)
    suggestions: list[str] = []
    try:
        m = re.search(r"\[.*?\]", raw, re.DOTALL)
        if m:
            for item in json.loads(m.group()):
                if "field" in item and "value" in item:
                    suggestions.append(f"{item['field']}={item['value']}")
    except Exception:
        pass
    return suggestions


def _apply(config: RetrievalConfig, suggestions: list[str]) -> RetrievalConfig:
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


# ── Main benchmark ────────────────────────────────────────────────────────────

def run_hotpotqa(
    n_samples: int = 100,
    max_rounds: int = 3,
    initial: str = "weak",
    qtype_filter: str | None = None,
    no_embeddings: bool = False,
    llm_call: Callable | None = None,
    iterative: bool = False,
):
    """Run HotpotQA memory retrieval benchmark with evolution.

    Args:
        n_samples: number of examples to evaluate per round
        max_rounds: number of evolution rounds
        initial: "weak" (keyword-only) or "strong" (full config)
        qtype_filter: "bridge" | "comparison" | None for both
        no_embeddings: disable sentence-transformer embeddings
        llm_call: optional pre-built LLM call fn; creates Anthropic default if None
    """
    import anthropic as _anthropic

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")

    print("=" * 65)
    print(f"MetaMem × HotpotQA  ·  n={n_samples}  ·  initial={initial}")
    print("=" * 65)

    # LLM
    if llm_call is None:
        _model = os.environ.get("LLM_MODEL", "claude-haiku-4-5-20251001")
        _client = _anthropic.Anthropic()

        def llm_call(messages, max_tokens=512, temperature=0.0):  # type: ignore[misc]
            system, user_msgs = "", []
            for m in messages:
                if m.get("role") == "system":
                    system = m["content"]
                else:
                    user_msgs.append({"role": m["role"], "content": m["content"]})
            if not user_msgs:
                return ""
            kw: dict = dict(model=_model, max_tokens=min(max_tokens, 4096), messages=user_msgs)
            if system:
                kw["system"] = system
            for attempt in range(3):
                try:
                    r = _client.messages.create(**kw)
                    return (r.content[0].text or "").strip()
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2 * (attempt + 1))
                    else:
                        logger.warning("LLM failed: %s", e)
                        return ""
            return ""

    print(f"LLM: {os.environ.get('LLM_MODEL', 'claude-haiku-4-5-20251001')}")

    # Embedder
    embedder = None
    if not no_embeddings:
        try:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            print("Embedder: all-MiniLM-L6-v2")
        except Exception as e:
            print(f"Embedder unavailable ({e})")

    # Load data (once — same samples reused across rounds)
    print(f"\nLoading {n_samples} HotpotQA examples...")
    samples = load_hotpotqa(n_samples=n_samples, qtype_filter=qtype_filter)
    type_dist = {}
    for s in samples:
        type_dist[s.qtype] = type_dist.get(s.qtype, 0) + 1
    print(f"Loaded {len(samples)} examples: {type_dist}")

    # Initial config
    if initial == "weak":
        ret_cfg = RetrievalConfig(
            semantic_top_k=0, keyword_top_k=3, structured_top_k=0,
            max_context=5, fusion_mode="keyword_only",
            enable_intent_routing=False, enable_entity_graph=False,
            confidence_boost_weight=0.0, enable_result_feedback=False,
        )
    else:
        ret_cfg = RetrievalConfig(
            semantic_top_k=5, keyword_top_k=3, structured_top_k=0,
            max_context=5, fusion_mode="rrf",
            enable_intent_routing=True, enable_entity_graph=False,
            confidence_boost_weight=0.2, enable_result_feedback=False,
        )

    tmp_root = tempfile.mkdtemp(prefix="mem_engram_hpqa_")
    run_id = time.strftime(f"hotpotqa_{initial}_%Y%m%d_%H%M%S")
    results_dir = f"benchmark_results/hotpotqa/{run_id}"
    os.makedirs(results_dir, exist_ok=True)

    result = HotpotResult()
    best_f1 = 0.0

    for round_id in range(max_rounds):
        print(f"\n--- Round {round_id}  "
              f"[fusion={ret_cfg.fusion_mode} sem_k={ret_cfg.semantic_top_k} "
              f"kw_k={ret_cfg.keyword_top_k} intent={ret_cfg.enable_intent_routing}] ---")

        round_rows: list[dict] = []
        total_f1 = total_recall = 0.0
        by_type: dict[str, dict] = {}
        by_level: dict[str, dict] = {}

        for idx, sample in enumerate(samples):
            # Build per-sample store (raw paragraph storage — no LLM extraction)
            store, id_to_title = _build_sample_store(sample, embedder, tmp_root)

            # Retrieve — one-shot or iterative (LLM-driven gap detection)
            if iterative and llm_call is not None:
                retriever_iter = IterativeRetrievalEngine(
                    store, llm_call, ret_cfg, embedder, max_rounds=2
                )
                retrieved = retriever_iter.search(sample.question, ret_cfg)
            else:
                retriever_base = RetrievalEngine(store, ret_cfg)
                q_emb = embedder.encode(sample.question) if embedder else None
                retrieved = retriever_base.search(
                    sample.question, config=ret_cfg, query_embedding=q_emb
                )

            retrieved_titles = [id_to_title.get(rm.memory.id, "") for rm in retrieved]
            context = format_context(retrieved, max_tokens=1500)

            # Answer
            prediction = _answer(sample.question, context, llm_call)

            # Score
            f1 = token_f1(prediction, sample.answer)
            rec = retrieval_recall(retrieved_titles, sample.supporting_titles)
            total_f1 += f1
            total_recall += rec

            # Bucket by type / level
            for bucket, key in [(by_type, sample.qtype), (by_level, sample.level)]:
                b = bucket.setdefault(key, {"f1": 0.0, "recall": 0.0, "n": 0})
                b["f1"] += f1; b["recall"] += rec; b["n"] += 1

            row = dict(
                question=sample.question, answer=sample.answer,
                prediction=prediction, f1=f1, recall=rec,
                qtype=sample.qtype, level=sample.level,
                supporting=sample.supporting_titles,
                retrieved_titles=retrieved_titles[:5],
            )
            round_rows.append(row)
            store.close()

            if (idx + 1) % 20 == 0:
                print(f"  [{idx+1}/{len(samples)}] "
                      f"avg_f1={total_f1/(idx+1):.3f} avg_recall={total_recall/(idx+1):.3f}")

        n = len(round_rows)
        avg_f1 = total_f1 / max(n, 1)
        avg_recall = total_recall / max(n, 1)

        # Normalize buckets
        for d in [by_type, by_level]:
            for k, b in d.items():
                b["f1"] /= max(b["n"], 1)
                b["recall"] /= max(b["n"], 1)

        print(f"  F1={avg_f1:.4f}  Recall={avg_recall:.4f}  n={n}")
        for t, b in sorted(by_type.items()):
            print(f"    {t:12s}: F1={b['f1']:.4f}  Recall={b['recall']:.4f}  n={b['n']}")
        for lvl in ["easy", "medium", "hard"]:
            if lvl in by_level:
                b = by_level[lvl]
                print(f"    {lvl:12s}: F1={b['f1']:.4f}  Recall={b['recall']:.4f}  n={b['n']}")

        rr = HotpotRound(
            round_id=round_id, f1=avg_f1, recall=avg_recall,
            by_type=by_type, by_level=by_level, total=n, config=asdict(ret_cfg),
        )

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            result.best_round = round_id
            result.best_f1 = avg_f1
            result.final_config = asdict(ret_cfg)

        # Save round detail
        with open(os.path.join(results_dir, f"round_{round_id}.jsonl"), "w") as f:
            for row in round_rows:
                f.write(json.dumps(row) + "\n")

        # Diagnose + evolve
        if round_id < max_rounds - 1:
            failures = [r for r in round_rows if r["f1"] < 0.3]
            suggestions = _diagnose(failures, ret_cfg, llm_call)
            if suggestions:
                ret_cfg = _apply(ret_cfg, suggestions)
                rr.improvements = suggestions
                print(f"  → Suggested: {suggestions}")
            else:
                # Manual stepping
                if round_id == 0 and ret_cfg.semantic_top_k == 0:
                    ret_cfg.semantic_top_k = 5
                    ret_cfg.fusion_mode = "rrf"
                    rr.improvements = ["semantic_top_k=5", "fusion_mode=rrf"]
                elif round_id == 1:
                    ret_cfg.enable_intent_routing = True
                    ret_cfg.semantic_top_k = 8
                    rr.improvements = ["intent_routing=True", "semantic_top_k=8"]
                if rr.improvements:
                    print(f"  → Manual step: {rr.improvements}")

        result.rounds.append(rr)

    print(f"\n{'=' * 65}")
    print("EVOLUTION COMPLETE")
    print("=" * 65)
    print(result.trajectory())

    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump({
            "run_id": run_id, "n_samples": n_samples, "initial": initial,
            "best_round": result.best_round, "best_f1": result.best_f1,
            "final_config": result.final_config,
            "rounds": [asdict(r) for r in result.rounds],
        }, f, indent=2, default=str)

    print(f"\nArtifacts: {results_dir}/")
    return result
