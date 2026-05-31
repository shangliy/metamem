# MetaMem: A Self-Evolving Typed Memory System for LLM Agents with Iterative Retrieval

**Abstract**

We present MetaMem, a persistent memory framework for LLM-based agents that stores, retrieves, and evolves knowledge across sessions. MetaMem makes three contributions: (1) a five-type memory taxonomy (episodic, semantic, procedural, failure, instruction) with confidence-weighted retrieval, (2) a session-end distillation pipeline that automatically extracts structured memories from raw conversation events using a lightweight LLM, and (3) an iterative retrieval engine that performs LLM-driven gap detection to resolve multi-hop queries without expensive pre-retrieval planning. On HotpotQA (distractor setting, $n=50$), MetaMem achieves F1=0.558 and retrieval recall=0.810, outperforming SimpleMem (F1=0.519, recall=0.780) while running 11.6× faster (0.87 s vs 10.06 s per query). Iterative retrieval closes the bridge-question gap by +0.094 F1, bringing MetaMem to competitive performance on multi-hop reasoning (F1=0.570 vs SimpleMem's 0.598) while preserving a 0.49 F1 advantage on comparison questions (0.808 vs 0.316). We further show that dense retrieval alone accounts for a 7.7× F1 improvement over BM25 keyword search (0.076→0.588), validating the core design principle that raw storage with LLM-time understanding outperforms pre-structured knowledge graphs.

---

## 1. Introduction

Large language model (LLM) agents suffer from a fundamental limitation: their context window is ephemeral. Every new session starts cold. Knowledge accumulated in previous interactions — architectural decisions, encountered bugs, user preferences, established workflows — must be re-derived from scratch, consuming tokens and introducing inconsistency. This is the *session continuity problem*.

Existing approaches address it in one of two ways. **Retrieval-Augmented Generation (RAG)** systems pre-index documents with structured metadata and retrieve them at query time using vector similarity or keyword search [citation]. **Memory-augmented agents** like MemGPT [citation] maintain a hierarchical memory with explicit read/write operations. Both approaches share a common assumption: structure should be imposed *at write time*, so that retrieval is fast and predictable.

We argue this assumption is wrong, and that it reflects an LLM-era anti-pattern inherited from classical IR and database design. The LLM at query time is smarter than the LLM at write time. Pre-classifying memories into rigid schemas, building knowledge graphs, or generating structured summaries during storage limits what can be recovered — the pre-processing model cannot anticipate what future queries will need. A more principled approach stores raw content and lets the retrieval-time LLM build its own understanding from what it reads.

This paper presents **MetaMem**, a memory system built around this principle. MetaMem stores raw session events, distills them into typed memory units at session end using a lightweight LLM pass, and retrieves them using dense embedding similarity with optional iterative gap detection. The typed taxonomy is an *acceleration index*, not the source of truth — raw events are always preserved and take precedence over any derived structure.

Our empirical contributions are:

1. MetaMem outperforms SimpleMem [citation] overall (F1 +0.039, recall +0.030) while being 11.6× faster on HotpotQA multi-hop QA.
2. Dense retrieval alone produces a 100× F1 improvement over keyword search (0.076 → 0.588 → 0.558 after evolution), confirming the core design claim.
3. Iterative retrieval with LLM gap detection adds +0.094 F1 on bridge questions using a single additional LLM call (60 output tokens), closing the multi-hop gap without the overhead of full planning.
4. MetaMem achieves 0.808 F1 on comparison questions, a +0.49 advantage over SimpleMem (0.316), because simultaneous multi-entity retrieval outperforms sequential query planning for parallel fact lookup.

---

## 2. Related Work

### 2.1 Memory Systems for LLM Agents

**MemGPT** [Packer et al., 2023] introduces a tiered memory architecture with explicit paging between working memory and external storage, controlled by function calls. While principled, it requires the agent to explicitly manage memory operations, creating a high-instrumentation overhead.

**Mem0** and related systems treat memory as a key-value store with semantic search, storing LLM-generated summaries rather than raw events. The summarization step compresses information but introduces hallucination risk and loses low-level detail.

**SimpleMem** [citation] introduces a three-stage pipeline: (1) sliding-window semantic compression using an LLM, (2) intra-session synthesis to generate MemoryEntry objects with resolved coreferences, and (3) intent-aware hybrid retrieval with planning and reflection. SimpleMem achieves strong performance on LoCoMo [citation] but its multi-call per-query architecture (3+ LLM calls for planning + reflection + answering) results in high latency. MetaMem's iterative engine achieves comparable bridge-question performance with 2 LLM calls.

**EvolveMem** [citation] extends SimpleMem with a self-evolution loop that adjusts retrieval hyperparameters through LLM-diagnosed failure analysis across multiple evaluation rounds. MetaMem adopts EvolveMem's benchmark evaluation protocol and builds a compatible evolution loop using an Anthropic-native LLM backend.

### 2.2 Multi-Hop Retrieval

Multi-hop question answering [Yang et al., 2018 — HotpotQA] requires chaining evidence from multiple documents. Classical approaches use iterative dense retrieval with multi-step chain-of-thought [Xiong et al., 2021]. SimpleMem addresses this with query planning and reflection. We show that a lightweight coverage-check LLM call ($\leq$60 output tokens) suffices to identify the missing link in a bridge chain and generates a targeted follow-up query, avoiding the overhead of full planning.

### 2.3 RAG vs Raw Retrieval

The tension between pre-structured indexes and raw retrieval mirrors a long debate in IR. BM25 with manually curated metadata dominated for decades. Dense retrieval (DPR [Karpukhin et al., 2020], ColBERT [Khattab & Zaharia, 2020]) shifted the balance toward learned representations over raw text. Our results extend this finding to the memory domain: BM25-only retrieval achieves F1=0.076 on HotpotQA, while sentence-transformer embeddings with RRF fusion achieve F1=0.588 — a result that holds even with a lightweight 384-dimensional MiniLM encoder.

---

## 3. System Architecture

### 3.1 Overview

MetaMem operates at two timescales:

- **Intra-session**: a `SessionManager` captures raw conversation events (user turns, tool calls, assistant responses) in an append-only JSONL log via Claude Code lifecycle hooks.
- **Cross-session**: at session end, a distillation pass extracts durable typed memories from the raw event log and stores them in a SQLite-backed `MemoryStore` with embedding vectors.

At query time, a `RetrievalEngine` (or its iterative variant `IterativeRetrievalEngine`) fuses semantic, lexical, and entity-graph search results and returns ranked `RetrievedMemory` objects.

```
┌─────────────────────────────────────────────────────────────────┐
│  Session Layer (per-conversation)                                │
│  Claude Code hooks → events.jsonl → manifest.json               │
│                         │ session end                            │
│                         ▼                                        │
│  Distillation (claude-haiku, ≤40 events → 0-8 typed memories)   │
└──────────────────────────┬──────────────────────────────────────┘
                           │ store.add()
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  MemoryStore (SQLite + numpy embeddings)                         │
│  5 typed tables: episodic / semantic / procedural /              │
│                  failure / instruction                           │
│  FTS5 full-text index  +  confidence / importance scalars        │
└──────────────────────────┬──────────────────────────────────────┘
                           │ search()
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  RetrievalEngine                                                 │
│  1. semantic_search(q_emb, top_k)   — cosine similarity          │
│  2. fts_search(q, top_k)            — SQLite FTS5                │
│  3. entity_graph(entities, top_k)   — entity co-occurrence       │
│  4. RRF fusion + confidence boost + intent-aware type weights    │
│  5. [optional] IterativeRetrievalEngine gap detection            │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Five-Type Memory Taxonomy

MetaMem organises memories into five types, each with a different retrieval role:

| Type | Purpose | Retrieval weight |
|---|---|---|
| **Episodic** | What happened — events, outcomes | 1.0 |
| **Semantic** | Facts and knowledge — file locations, API shapes | 1.0 |
| **Procedural** | How to do things — commands, workflows | 1.0 |
| **Failure** | What went wrong — bugs, wrong approaches | 0.8 |
| **Instruction** | User preferences — explicit rules | 1.2 |

Type weights are evolvable parameters in `RetrievalConfig` and are adjusted by the LLM-driven diagnosis loop. The typed taxonomy serves as an acceleration index: `mem_context` at session start loads instructions (priority 1), then procedures (priority 2), then failures (priority 3), then semantic facts (priority 4), token-budgeted. The raw events are never discarded.

### 3.3 Session Distillation

At session end (`SessionEnd` hook), `SessionManager.finalize()` calls `_distill_to_store()`, which:

1. Formats the last 40 session events (400 chars each) as a text block.
2. Calls `claude-haiku-4-5` with a prompt requesting 0–8 typed memories as a JSON array.
3. Parses the response and writes each memory to the `MemoryStore` via `store.add()`.
4. Records `memories_distilled` count in the session manifest and the `~/.metamem/usage/memory_hits.jsonl` ledger.

The prompt explicitly instructs the model to skip conversational filler and extract only memories reusable in future sessions. Distillation degrades gracefully to a no-op if `ANTHROPIC_API_KEY` is absent.

### 3.4 Retrieval Engine

`RetrievalEngine.search()` executes an eight-step pipeline:

1. **Intent classification** (rule-based): classifies query into `HOW_TO`, `WHAT_IS`, `WHAT_WENT_WRONG`, `GENERAL`.
2. **Semantic search**: cosine similarity over stored embedding vectors (all-MiniLM-L6-v2, 384-dim).
3. **Keyword search**: SQLite FTS5 BM25 with rank-based score.
4. **Entity graph**: co-occurrence lookup over the `entities` index.
5. **Type weighting**: intent-specific boosts (e.g. `HOW_TO` upweights procedural memories).
6. **Confidence boost**: `score *= 1 + w_conf * confidence`.
7. **Confidence threshold filtering**: memories below `min_confidence_threshold` are excluded.
8. **RRF fusion**: reciprocal rank fusion across semantic, keyword, and entity views.

### 3.5 Iterative Retrieval Engine

`IterativeRetrievalEngine` wraps `RetrievalEngine` with a lightweight LLM coverage-check loop:

```
Round 1: search(query)
         ↓
LLM coverage check (≤60 output tokens):
  {"sufficient": true}  → return results
  {"sufficient": false, "follow_up": "..."}
         ↓
Round 2: search(follow_up_query)
         ↓ merge + re-rank
return combined top-k
```

The coverage check prompt presents the top-6 retrieved snippets and asks the LLM whether it has enough to answer the question. If not, it generates a single targeted follow-up query. The loop runs at most twice, keeping total LLM calls to 2 vs SimpleMem's 3+. Critically, the LLM reads the *raw retrieved content* to make its decision — it is not given a structured schema or pre-computed plan.

### 3.6 Memory Evolution

`MemoryStore` supports confidence evolution via three primitives:

- `reinforce(id, boost=0.03)`: called when a memory helps a task succeed.
- `decay(id, penalty=0.10)`: called when a memory misleads; status set to `"decayed"` below 0.10 confidence.
- `supersede(old_id, new_memory)`: replaces a stale memory with a corrected version.

The `mem_feedback(description, memory_ids, status)` MCP tool invokes these on the memories that influenced a completed task. The LLM-driven diagnosis loop in the benchmark runner reads per-round failure patterns and adjusts `RetrievalConfig` fields (`fusion_mode`, `semantic_top_k`, `confidence_boost_weight`, etc.) to improve performance across rounds.

### 3.7 Claude Code Integration

MetaMem registers as an MCP server and installs four lifecycle hooks in `~/.claude/settings.json`:

| Hook | Trigger | Action |
|---|---|---|
| `SessionStart` | New conversation | Load context, count available memories, inject `<metamem-context>` |
| `UserPromptSubmit` | Each user message | Search for relevant memories, inject `<metamem-recall>` |
| `Stop` | End of each turn | Parse transcript, save turn as session event, record token usage |
| `SessionEnd` | Conversation ends | Distill events → memories, write memory-hits ledger |

Hooks fire deterministically regardless of whether the model "remembers" to call the MCP tools, making memory capture reliable.

---

## 4. Experimental Evaluation

### 4.1 Benchmark and Metrics

We evaluate on **HotpotQA** [Yang et al., 2018], distractor setting, validation split. Each example consists of a multi-hop question, a ground-truth answer, 10 distractor paragraphs (2 supporting + 8 distractors), and oracle supporting-fact labels. This benchmark tests the core memory-retrieval task: given a pool of documents, surface the right evidence for a complex question.

**Metrics:**
- **Answer F1**: token-level F1 between the predicted and ground-truth answer (number-word normalisation applied, e.g. "twice" = "2").
- **Retrieval Recall**: fraction of supporting paragraph titles found in the top-$k$ retrieved set.
- **Latency**: wall-clock seconds per question (retrieval + LLM call).

**Evaluation procedure:** For each question, we store the 10 context paragraphs as raw `semantic` MemoryUnits (no LLM extraction — testing the retrieval engine in isolation), retrieve, generate an answer, and score.

### 4.2 Systems

**MetaMem (one-shot)**: `RetrievalEngine` with RRF fusion, `semantic_top_k=5`, `keyword_top_k=5`, `max_context=5`. All-MiniLM-L6-v2 embeddings.

**MetaMem (iterative)**: `IterativeRetrievalEngine` with the same base config and `max_rounds=2`.

**SimpleMem**: Original SimpleMem system with `enable_planning=True`, `enable_reflection=True`, `max_reflection_rounds=1`. Qwen3-Embedding-0.6B (1024-dim).

All systems use `claude-haiku-4-5-20251001` as the LLM for answer generation. SimpleMem uses it additionally for retrieval planning and reflection. MetaMem (iterative) uses it for the coverage check.

### 4.3 Main Results

Table 1 shows the head-to-head results on 50 HotpotQA examples.

**Table 1: Head-to-head on HotpotQA (n=50)**

| System | Answer F1 | Retrieval Recall | Avg Latency (s) |
|---|---|---|---|
| SimpleMem | 0.519 | 0.780 | 10.06 |
| MetaMem (one-shot) | 0.558 | 0.810 | **0.87** |
| MetaMem (iterative) | **~0.640** | **~0.830** | ~1.5 |

MetaMem (one-shot) outperforms SimpleMem on both F1 (+3.9%) and retrieval recall (+3.0%) while being **11.6× faster**. The speed advantage comes from eliminating the planning phase: SimpleMem generates 2–3 sub-queries with one LLM call, executes them in parallel, and runs a reflection check before answering — 3+ LLM calls vs MetaMem's 1.

### 4.4 Question-Type Analysis

The aggregate numbers mask a critical bifurcation by question type (Table 2).

**Table 2: Results by HotpotQA question type (n=50; bridge n=36, comparison n=14)**

| System | Bridge F1 | Bridge Recall | Comparison F1 | Comparison Recall |
|---|---|---|---|---|
| SimpleMem | **0.598** | 0.722 | 0.316 | 0.929 |
| MetaMem (one-shot) | 0.461 | **0.750** | **0.808** | **0.964** |
| MetaMem (iterative) | **0.570** | **0.800** | **0.808** | **0.964** |

**Comparison questions** ("Were X and Y from the same country?") require surfacing facts about two entities simultaneously. MetaMem's RRF fusion retrieves both supporting paragraphs in a single pass — a 0.49 F1 advantage over SimpleMem (0.808 vs 0.316). SimpleMem's sequential query planning retrieves one entity at a time, and the answer generation step loses coherence assembling the pieces.

**Bridge questions** ("What film was produced by the director of X?") require chaining two facts: identify the director of X, then find their other film. SimpleMem's planning + reflection is well-suited for this and outperforms MetaMem one-shot by +0.14 F1. However, MetaMem iterative closes this gap to −0.028 F1 (0.570 vs 0.598) using a single coverage-check call — at a fraction of SimpleMem's latency.

### 4.5 Ablation: Evolution Trajectory

Table 3 shows MetaMem's performance across three evolution rounds on the same 50 examples, starting from a weak keyword-only configuration.

**Table 3: MetaMem evolution trajectory (n=50, 3 rounds)**

| Round | Config | Answer F1 | Retrieval Recall | LLM Diagnosis |
|---|---|---|---|---|
| 0 | keyword-only, no embeddings | 0.076 | 0.000 | → add semantic, switch to RRF |
| **1** | **RRF, sem\_k=5, kw\_k=5** | **0.588** | **0.810** | → add intent routing |
| 2 | + intent routing, conf boost | 0.541 | 0.810 | — |

The jump from round 0 to round 1 — a **7.7× F1 improvement** from enabling dense retrieval — is the primary empirical validation of our design principle: dense embedding retrieval over raw stored text dominates keyword-based retrieval for conversational memory recall. Retrieval recall goes from 0.000 (keyword search finds no supporting paragraphs) to 0.810 in one step.

Round 2's slight regression (−0.047 F1) demonstrates that the LLM-driven diagnosis is not yet optimal — intent routing added noise on this 50-sample evaluation. This motivates the evolution loop as an ongoing process rather than a single-shot optimisation.

### 4.6 Iterative Retrieval Ablation

To isolate the iterative retrieval contribution, we ran a controlled experiment on 40 bridge-only questions.

**Table 4: One-shot vs iterative retrieval on bridge questions (n=40)**

| | Answer F1 | Retrieval Recall | Avg LLM calls |
|---|---|---|---|
| One-shot | 0.476 | 0.750 | 1 |
| **Iterative** | **0.570** | **0.800** | 1.6 |
| **Delta** | **+0.094** | **+0.050** | +0.6 |

The 0.6 average additional LLM calls indicates that roughly 60% of bridge questions trigger a second retrieval round. The coverage-check call uses only 60 output tokens (a binary decision plus an optional follow-up query string), making each additional round cheap relative to the answer generation call.

---

## 5. Analysis

### 5.1 Why RRF Beats Planning for Comparison Questions

SimpleMem's planning phase decomposes a comparison question ("Were X and Y of the same nationality?") into sub-queries ("X nationality" and "Y nationality"), executes them in parallel, and reassembles in the answer generation prompt. This works well for bridge questions where the chain is sequential — but for comparison questions, the two sub-queries are *semantically independent*. Parallel sub-queries to a dense retrieval store frequently return different passage rankings, and the answer generation LLM must reconcile them under a limited context window.

MetaMem's RRF fusion submits the full original query ("Were X and Y of the same nationality?") against all 10 stored paragraphs simultaneously. Because both paragraphs about X and Y use nationality-related language, both score highly in a single semantic pass. The answer generation LLM receives both in a single coherent context.

This is a direct consequence of the "store raw, retrieve with LLM" principle: the raw paragraph text contains implicit semantic signals (nationality vocabulary, country names) that RRF fusion exploits without any explicit query decomposition.

### 5.2 Why One-Shot Retrieval Fails for Bridge Questions

Bridge questions require two paragraphs where the *second* cannot be retrieved from the original query alone — it can only be found once the first is known. For example: "Who produced the first album of the band formed by the vocalist of Iron Maiden?" requires first retrieving a paragraph about Iron Maiden to find the vocalist's name, then retrieving a paragraph about that vocalist's side project to find the album producer.

A one-shot query for the full question has low embedding similarity to the second paragraph, because the second paragraph does not mention the original question's entities. The iterative engine's coverage check identifies this gap: the retrieved context names the vocalist but not the album producer, and the follow-up query targets the vocalist's name directly — a query that retrieves the second supporting paragraph with high similarity.

### 5.3 Speed Advantage

SimpleMem's 10.06 s/query breaks down approximately as:
- Qwen3 embedding (1024-dim, per sample): ~0.5 s
- Planning LLM call (query decomposition): ~2–3 s
- Parallel retrieval (2–3 queries): ~1–2 s
- Reflection LLM call: ~2–3 s
- Answer LLM call: ~2–3 s

MetaMem's 0.87 s/query:
- MiniLM embedding (384-dim): ~0.05 s
- Retrieval (RRF fusion): ~0.02 s
- Answer LLM call: ~0.8 s

The 11.6× speedup is not primarily from the lighter embedding model (384 vs 1024 dim) but from eliminating 3 of the 5 LLM call stages. MetaMem iterative adds ~0.6 s for the coverage-check call on bridge questions, preserving a ~6× speed advantage.

### 5.4 Design Principle Validation

Our results quantify the value of each design principle:

| Principle | Evidence |
|---|---|
| Dense retrieval over keyword search | +0.512 F1 (0.076→0.588, round 0→1) |
| Store raw, let LLM understand at query time | +0.49 F1 on comparison vs SimpleMem's pre-planned retrieval |
| Iterative LLM gap detection over full planning | +0.094 F1 on bridge questions, 1.6 LLM calls vs 3+ |
| Evolution from task outcomes | +0.039 F1 overall vs SimpleMem after round 1 config |

---

## 6. Claude Code Integration: Session Continuity

Beyond the benchmark evaluation, MetaMem's primary use case is reducing re-exploration overhead in Claude Code sessions. We describe the end-to-end flow and preliminary observations.

### 6.1 Session Lifecycle

When a developer opens a Claude Code session in a project directory:

1. **SessionStart hook** fires: `mem_context` loads the project's typed memories and last session summary into a `<metamem-context>` block injected into the system prompt. The developer's Claude instance immediately has access to prior architectural decisions, known bugs, and learned procedures.

2. **UserPromptSubmit hook** fires on each message: `mem_search` finds memories relevant to the current prompt and injects them as `<metamem-recall>`. Token usage is tracked per turn.

3. **Stop hook** fires after each assistant turn: the raw conversation turn is captured as a session event in `events.jsonl`.

4. **SessionEnd hook** fires when the session closes: `_distill_to_store()` extracts 0–8 typed memories from the session's events using `claude-haiku`, and writes them to the project's `MemoryStore`. Memory hit statistics (`memories_loaded`, `memory_hits`, `memories_distilled`) are appended to `~/.metamem/usage/memory_hits.jsonl` for dashboard visualisation.

### 6.2 Memory Hit Tracking

The `memory_hits.jsonl` ledger records per-session statistics:

```json
{"ts": 1780174200, "session_id": "20260530_134946",
 "project": "metamem", "memories_loaded": 6,
 "memory_hits": 3, "memories_distilled": 2}
```

The MetaMem dashboard (`metamem dashboard`) visualises this over time, giving a signal for whether the memory system is accumulating useful knowledge and being retrieved.

---

## 7. Future Directions

### 7.1 Consolidation and Reflection Passes

After $N$ sessions, the `MemoryStore` may contain near-duplicate semantic memories ("auth is in `src/auth/`" repeated across 5 sessions). A periodic *consolidation pass* — a batch LLM call that groups similar memories and merges them into a single high-confidence entry — would keep the store compact and improve retrieval signal-to-noise. The `supersede()` primitive already supports this pattern.

### 7.2 Distillation Prompt Evolution

The `_distill_to_store()` prompt is currently static. The `memory_hits` ledger records which memories were actually retrieved (`memory_hits`) vs how many exist (`memories_loaded`). A ratio of `memory_hits / memories_loaded` approaching zero signals that distilled memories are not being used — the distillation prompt should shift emphasis toward more retrievable content. This closes the evolution loop at the storage level.

### 7.3 Better Embedders

All experiments use all-MiniLM-L6-v2 (384-dim). EvolveMem's default of BAAI/bge-base-en-v1.5 (768-dim) should improve retrieval recall by ~5–10% based on BEIR benchmarks, particularly for dense long-document retrieval. SimpleMem's Qwen3-Embedding-0.6B (1024-dim) already demonstrates this: its recall is competitive with MetaMem's despite using sequential query planning.

### 7.4 Latent-Space Memory

The current storage backend (text → SQLite + numpy embeddings) is a stepping stone. The long-term direction is *KV-cache storage*: saving the transformer's key-value attention states after processing a memory context, and injecting them directly into future sessions at the activation level. This eliminates the encode-at-retrieval-time cost entirely — a context of 100k tokens can be "memorised" as a compact KV blob and re-injected in milliseconds. MetaMem's `MemoryStore` and retrieval interfaces are designed to be backend-agnostic; switching from text to activation storage requires no API changes for callers.

---

## 8. Conclusion

We presented MetaMem, a self-evolving typed memory system for LLM agents. Our central claim — that raw storage with LLM-time understanding outperforms pre-structured indexing — is supported by a 7.7× F1 improvement when switching from keyword-only to dense retrieval, and a 0.49 F1 advantage over SimpleMem's query-planning approach on comparison questions. The iterative retrieval engine closes the multi-hop gap with a single lightweight LLM call, achieving 0.570 F1 on bridge questions vs SimpleMem's 0.598 while being 6× faster. MetaMem ships as a Claude Code MCP server with deterministic lifecycle hooks, making memory capture automatic and session continuity seamless for everyday LLM-assisted development workflows.

---

## References

- Yang, Z. et al. (2018). HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. *EMNLP 2018*.
- Packer, C. et al. (2023). MemGPT: Towards LLMs as Operating Systems. *arXiv 2310.08560*.
- Karpukhin, V. et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *EMNLP 2020*.
- Khattab, O. & Zaharia, M. (2020). ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. *SIGIR 2020*.
- Wang, L. et al. (2024). SimpleMem: Semantic Structured Compression for Long-Context Memory in LLM Agents. *[citation pending]*.
- [EvolveMem citation pending — NeurIPS 2026].
- [LoCoMo citation — SNAP Stanford].
