"""Head-to-head benchmark: MetaMem vs SimpleMem on HotpotQA.

Same 50 examples, same LLM (claude-haiku), same embedder (each uses its own default).
Measures: Answer F1 + Retrieval Recall (supporting paragraph hit rate).

SimpleMem:  sliding-window LLM extraction + HybridRetriever (planning + reflection)
MetaMem:    raw paragraph storage + intent-aware RRF retrieval (best round config)
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable

# ── SimpleMem path ─────────────────────────────────────────────────────────────
_SIMPLEMEM = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..")
)
if _SIMPLEMEM not in sys.path:
    sys.path.insert(0, _SIMPLEMEM)

from .hotpotqa import (
    HotpotSample,
    load_hotpotqa,
    token_f1,
    retrieval_recall,
    _answer,
    _build_sample_store,
)
from ..retriever import RetrievalConfig, RetrievalEngine, format_context
from ..store import MemoryStore
from ..models import MemoryType, MemoryUnit


# ── Shared LLM (Anthropic via OpenAI-compat) ──────────────────────────────────

def _make_anthropic_llm(model: str = "claude-haiku-4-5-20251001") -> Callable:
    import anthropic
    client = anthropic.Anthropic()

    def llm_call(messages, max_tokens=512, temperature=0.0):
        system, user_msgs = "", []
        for m in messages:
            if m.get("role") == "system":
                system = m["content"]
            else:
                user_msgs.append({"role": m["role"], "content": m["content"]})
        if not user_msgs:
            return ""
        kw = dict(model=model, max_tokens=min(max_tokens, 4096), messages=user_msgs)
        if system:
            kw["system"] = system
        for attempt in range(3):
            try:
                r = client.messages.create(**kw)
                return (r.content[0].text or "").strip()
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                else:
                    return ""
        return ""

    return llm_call


# ── SimpleMem runner ───────────────────────────────────────────────────────────

def _run_simplemem(
    samples: list[HotpotSample],
    api_key: str,
    model: str = "claude-haiku-4-5-20251001",
    enable_planning: bool = True,
    enable_reflection: bool = True,
) -> list[dict]:
    """Run SimpleMem pipeline on HotpotQA samples. Returns per-sample results."""
    if _SIMPLEMEM not in sys.path:
        sys.path.insert(0, _SIMPLEMEM)
    from main import SimpleMemSystem  # noqa: PLC0415
    from simplemem.core.models.memory_entry import Dialogue  # noqa: PLC0415

    rows: list[dict] = []
    for idx, sample in enumerate(samples):
        db_dir = tempfile.mkdtemp(prefix="sm_hpqa_")
        system = SimpleMemSystem(
            api_key=api_key,
            base_url="https://api.anthropic.com/v1/",
            model=model,
            db_path=db_dir + "/mem.db",
            clear_db=True,
            enable_thinking=False,
            enable_planning=enable_planning,
            enable_reflection=enable_reflection,
            max_reflection_rounds=1,
        )

        # Feed all 10 paragraphs as dialogues
        dialogues = [
            Dialogue(dialogue_id=i, speaker="doc",
                     content=f"{p['title']}: {p['text']}")
            for i, p in enumerate(sample.paragraphs)
        ]
        system.add_dialogues(dialogues)
        system.finalize()

        # Ask the question — SimpleMem does retrieval + answering in one call
        t0 = time.time()
        raw_answer = system.ask(sample.question)
        latency = time.time() - t0

        # Parse answer (SimpleMem returns plain text, not JSON)
        prediction = raw_answer.strip()

        # For retrieval recall we check which supporting paragraph texts appear
        # in the memories retrieved (SimpleMem doesn't expose retrieved IDs easily,
        # so we check if supporting paragraph content is referenced in the answer
        # or use a heuristic based on memory content)
        all_mems = system.get_all_memories()
        retrieved_titles = []
        for mem in all_mems:
            mem_text = (getattr(mem, "lossless_restatement", "") or
                        getattr(mem, "content", "") or "").lower()
            for sup in sample.supporting_titles:
                if sup.lower() in mem_text:
                    retrieved_titles.append(sup)
                    break

        f1 = token_f1(prediction, sample.answer)
        rec = retrieval_recall(retrieved_titles, sample.supporting_titles)

        rows.append(dict(
            question=sample.question, answer=sample.answer,
            prediction=prediction, f1=f1, recall=rec,
            qtype=sample.qtype, level=sample.level,
            latency=round(latency, 2),
        ))

        if (idx + 1) % 10 == 0:
            avg_f1 = sum(r["f1"] for r in rows) / len(rows)
            print(f"  SimpleMem [{idx+1}/{len(samples)}] avg_f1={avg_f1:.3f}")

    return rows


# ── MetaMem runner (best config from evolution) ────────────────────────────────

def _run_metamem(
    samples: list[HotpotSample],
    embedder,
    llm_call: Callable,
    config: RetrievalConfig | None = None,
) -> list[dict]:
    """Run MetaMem pipeline on HotpotQA samples."""
    if config is None:
        config = RetrievalConfig(
            semantic_top_k=5, keyword_top_k=5, structured_top_k=0,
            max_context=5, fusion_mode="rrf",
            enable_intent_routing=False, enable_entity_graph=False,
            confidence_boost_weight=0.0, enable_result_feedback=False,
        )

    tmp_root = tempfile.mkdtemp(prefix="mm_hpqa_")
    rows: list[dict] = []

    for idx, sample in enumerate(samples):
        store, id_to_title = _build_sample_store(sample, embedder, tmp_root)
        retriever = RetrievalEngine(store, config)

        q_emb = embedder.encode(sample.question) if embedder else None
        t0 = time.time()
        retrieved = retriever.search(sample.question, config=config, query_embedding=q_emb)
        context = format_context(retrieved, max_tokens=1500)
        prediction = _answer(sample.question, context, llm_call)
        latency = time.time() - t0

        retrieved_titles = [id_to_title.get(rm.memory.id, "") for rm in retrieved]
        f1 = token_f1(prediction, sample.answer)
        rec = retrieval_recall(retrieved_titles, sample.supporting_titles)

        rows.append(dict(
            question=sample.question, answer=sample.answer,
            prediction=prediction, f1=f1, recall=rec,
            qtype=sample.qtype, level=sample.level,
            latency=round(latency, 2),
        ))
        store.close()

        if (idx + 1) % 10 == 0:
            avg_f1 = sum(r["f1"] for r in rows) / len(rows)
            print(f"  MetaMem  [{idx+1}/{len(samples)}] avg_f1={avg_f1:.3f}")

    return rows


# ── Aggregation ────────────────────────────────────────────────────────────────

def _agg(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {}
    f1 = sum(r["f1"] for r in rows) / n
    rec = sum(r["recall"] for r in rows) / n
    lat = sum(r["latency"] for r in rows) / n
    by_type: dict[str, dict] = {}
    for r in rows:
        b = by_type.setdefault(r["qtype"], {"f1": 0.0, "recall": 0.0, "n": 0})
        b["f1"] += r["f1"]; b["recall"] += r["recall"]; b["n"] += 1
    for b in by_type.values():
        b["f1"] = round(b["f1"] / b["n"], 4)
        b["recall"] = round(b["recall"] / b["n"], 4)
    return {"f1": round(f1, 4), "recall": round(rec, 4),
            "avg_latency_s": round(lat, 2), "n": n, "by_type": by_type}


# ── Main ───────────────────────────────────────────────────────────────────────

def run_head_to_head(
    n_samples: int = 50,
    model: str = "claude-haiku-4-5-20251001",
):
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    print("=" * 70)
    print(f"MetaMem vs SimpleMem  ·  HotpotQA  ·  n={n_samples}  ·  {model}")
    print("=" * 70)

    # Load shared samples (same for both systems)
    print(f"\nLoading {n_samples} HotpotQA examples...")
    samples = load_hotpotqa(n_samples=n_samples)
    type_dist = {}
    for s in samples:
        type_dist[s.qtype] = type_dist.get(s.qtype, 0) + 1
    print(f"Distribution: {type_dist}")

    # Shared embedder for MetaMem (CPU)
    embedder = None
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("MetaMem embedder: all-MiniLM-L6-v2")
    except Exception as e:
        print(f"MetaMem embedder unavailable: {e}")

    llm_call = _make_anthropic_llm(model)

    # ── Run MetaMem ────────────────────────────────────────────────────────
    print(f"\n{'─'*35} MetaMem {'─'*35}")
    t0 = time.time()
    mm_rows = _run_metamem(samples, embedder, llm_call)
    mm_time = time.time() - t0
    mm = _agg(mm_rows)
    print(f"Done in {mm_time:.0f}s")

    # ── Run SimpleMem ──────────────────────────────────────────────────────
    print(f"\n{'─'*33} SimpleMem {'─'*33}")
    t0 = time.time()
    sm_rows = _run_simplemem(samples, api_key, model,
                              enable_planning=True, enable_reflection=True)
    sm_time = time.time() - t0
    sm = _agg(sm_rows)
    print(f"Done in {sm_time:.0f}s")

    # ── Report ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("HEAD-TO-HEAD RESULTS")
    print("=" * 70)
    header = f"{'Metric':<22} {'MetaMem':>10} {'SimpleMem':>12} {'Delta':>10}"
    print(header)
    print("-" * 60)

    def row(label, mm_val, sm_val, fmt=".4f"):
        delta = mm_val - sm_val
        sign = "+" if delta >= 0 else ""
        print(f"{label:<22} {mm_val:>10{fmt}} {sm_val:>12{fmt}} {sign}{delta:>{fmt}}")

    row("Answer F1", mm["f1"], sm["f1"])
    row("Retrieval Recall", mm["recall"], sm["recall"])
    row("Avg latency (s)", mm["avg_latency_s"], sm["avg_latency_s"], ".2f")

    print("\nBy question type:")
    for qtype in ["bridge", "comparison"]:
        mm_t = mm["by_type"].get(qtype, {})
        sm_t = sm["by_type"].get(qtype, {})
        if mm_t and sm_t:
            print(f"  {qtype}:")
            row(f"    F1", mm_t["f1"], sm_t["f1"])
            row(f"    Recall", mm_t["recall"], sm_t["recall"])

    # Save
    run_id = time.strftime("h2h_%Y%m%d_%H%M%S")
    out_dir = f"benchmark_results/head_to_head/{run_id}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump({
            "metamem": mm, "simplemem": sm,
            "metamem_rows": mm_rows, "simplemem_rows": sm_rows,
        }, f, indent=2)
    print(f"\nArtifacts: {out_dir}/")

    return mm, sm
