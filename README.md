# 🧠 MetaMem

**Unified lifelong memory for LLM agents** — typed stores, evolution from task results, MCP integration for Claude Code.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-≥3.10-brightgreen.svg)](https://python.org)

---

## What is MetaMem?

MetaMem is a memory system that **gets smarter over time**. Unlike simple memory stores that just record observations, MetaMem:

1. **Types memories** — separates facts, skills, failures, preferences, and events into specialized stores with different retrieval behaviors
2. **Evolves from results** — when a recalled memory helps you succeed, its confidence increases; when it misleads, it gets corrected or deprecated
3. **Routes by intent** — "how to deploy" retrieves procedures; "what went wrong" retrieves failure cases
4. **Works with Claude Code** — exposes memory as MCP tools: search, store, feedback

---

## Quick Start

### Install

```bash
pip install -e .
```

### Register with Claude Code

```bash
metamem install
```

This registers MetaMem as an MCP server using Claude Code's native CLI
(`claude mcp add metamem --scope user -- <python> -m metamem.mcp_server`). If the
`claude` CLI isn't on your PATH, it falls back to writing `~/.claude/.mcp.json` and
`~/.claude/claude_desktop_config.json` directly. It also injects MetaMem instructions
into your project's `CLAUDE.md`.

Install variants:

```bash
metamem install --project-only        # register only for the current project (scope: project)
metamem install --project-dir /path   # write CLAUDE.md to a specific directory
```

Then verify and approve in Claude Code:

```bash
claude mcp list
```

If MetaMem shows **"Pending approval"**, launch `claude` and approve it. Restart Claude Code to activate.

### Automatic capture (hooks)

By default, `metamem install` also registers **Claude Code lifecycle hooks** into
`~/.claude/settings.json` (or `.claude/settings.json` with `--project-only`). Unlike
the `CLAUDE.md` instructions — which the model may skip — hooks fire **deterministically**
at each lifecycle event, so memory capture happens whether or not the model "remembers":

| Event | What MetaMem does |
|-------|-------------------|
| `SessionStart` | Injects prior project context into the conversation |
| `UserPromptSubmit` | Searches memory for the prompt and injects relevant hits |
| `Stop` | Captures the completed turn as a session event |
| `SessionEnd` | Finalizes + summarizes the session into project memory |

Skip hooks (MCP tools only) with:

```bash
metamem install --no-hooks
```

### Use from CLI

```bash
# Store a memory
metamem store "Docker rate limit is 100 pulls/6h for anonymous users" -t semantic

# Save a preference
metamem instruct "Always use poetry, never pip install directly"

# Search
metamem search "docker rate limit"

# View stats
metamem stats

# View token usage tracked from Claude Code sessions
metamem usage

# Launch the local dashboard (memories + token usage)
metamem dashboard            # → http://127.0.0.1:8765
```

### Dashboard

`metamem dashboard` launches a local, read-only web UI (FastAPI, no build step,
binds to `127.0.0.1` only) to browse:

- **Memories** — by type, summary, and confidence, filterable per project
- **Token usage** — totals, cache-hit ratio, and per-project breakdown, sourced
  from the separate usage ledger at `~/.metamem/usage/token_usage.jsonl`

Token usage is captured automatically by the `Stop` hook (from each turn's
`message.usage`) and stored separately from the memory store, so it's easy to
analyze cost/benefit over time.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      MetaMem                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────┐        │
│  │   Session Memory (per conversation folder)  │        │
│  │   • Working Memory • Topics • Event Log     │        │
│  └──────────────────┬──────────────────────────┘        │
│                     │ absorption                         │
│                     ▼                                    │
│  ┌─────────────────────────────────────────────┐        │
│  │   Global Memory (5 typed stores)            │        │
│  │   📖 Episodic  🧩 Semantic  ⚙️ Procedural   │        │
│  │   ❌ Failures  📋 Instructions              │        │
│  └──────────────────┬──────────────────────────┘        │
│                     │                                    │
│                     ▼                                    │
│  ┌─────────────────────────────────────────────┐        │
│  │   Evolution Engine                          │        │
│  │   Task result → reinforce/decay/supersede   │        │
│  └─────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────┘
```

## Memory Types

| Type | What it stores | Example |
|------|---------------|---------|
| **Episodic** | Events that happened | "Deployed v2 on May 28" |
| **Semantic** | Facts and knowledge | "API uses OAuth2, rate limit 60/min" |
| **Procedural** | Skills and how-to | "Deploy: build → push → restart" |
| **Failure** | What went wrong | "OOM from unbounded cache" |
| **Instruction** | User preferences | "Always use pnpm" |

## MCP Tools (for Claude Code)

| Tool | Layer | Purpose |
|------|-------|---------|
| `mem_search` | 1 (index) | Search memories (~50 tokens/result) |
| `mem_timeline` | 2 | Chronological context |
| `mem_get` | 3 (full) | Complete details (~500 tokens/result) |
| `mem_store` | — | Store a new typed memory |
| `mem_instruct` | — | Save a preference |
| `mem_feedback` | — | Report task result for evolution |
| `mem_stats` | — | System statistics |

### Progressive Disclosure (Token-Efficient)

```
1. mem_search("deploy production") → compact index with IDs
2. mem_get(["id1", "id2"])         → full details only for relevant ones
   
   ~10x token savings vs loading everything
```

## Evolution Loop

```
Retrieve → Act → Observe Result → Evolve
                                    ↓
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
               Reinforce        Refine         Deprecate
               (conf ↑)      (add caveats)   (supersede)
```

When you use `mem_feedback` after a task:
- **Success** → memories that helped get confidence boost (+0.03)
- **Failure** → memories that misled get decayed (-0.10), failure case created
- **Partial** → caveats added to procedural memories
- **Contradiction** → old memory superseded by corrected version

## Benchmark: MetaMem vs SimpleMem

Head-to-head evaluation on **HotpotQA** (distractor split) — a public multi-hop QA
benchmark. Each question has 10 paragraphs (2 supporting + 8 distractors). The task:
retrieve the 2 right paragraphs and answer correctly.

Same LLM (`claude-haiku-4-5-20251001`), same 50 questions, measured independently.

### Overall results

| Metric | MetaMem | SimpleMem | Δ |
|---|---|---|---|
| **Answer F1** | **0.558** | 0.519 | **+0.039** |
| **Retrieval Recall** | **0.810** | 0.780 | **+0.030** |
| Avg latency | **0.87 s** | 10.06 s | **11.6× faster** |

### By question type

| Type | MetaMem F1 | SimpleMem F1 | Winner |
|---|---|---|---|
| **Comparison** | **0.808** | 0.316 | MetaMem **+0.49** |
| Bridge | 0.461 | **0.598** | SimpleMem +0.14 |

### What the numbers mean

**Comparison questions** (two-entity lookup, e.g. "Were X and Y from the same country?"):
MetaMem's RRF fusion surfaces both entity paragraphs simultaneously. SimpleMem's
sequential planning generates sub-queries one at a time and loses coherence reassembling
the answer — a 0.49 F1 deficit.

**Bridge questions** (chain A → B, e.g. "Who directed the film produced by X?"):
SimpleMem's multi-query planning + reflection is purpose-built for this: retrieve fact A,
identify the missing link, retrieve fact B. MetaMem's one-shot retrieval often gets only
one of the two supporting paragraphs — a 0.14 F1 gap that iterative retrieval closes
(see [Design principles](#design-principles)).

**Speed**: MetaMem is 11.6× faster because SimpleMem's planning + reflection adds
3 LLM calls per question before the answer call. MetaMem retrieves once and answers.

### Evolution trajectory (MetaMem, 3 rounds, weak → optimised)

| Round | Config | F1 | Recall |
|---|---|---|---|
| 0 | keyword-only | 0.076 | 0.000 |
| **1** | **RRF + semantic (k=5)** | **0.588** | **0.810** |
| 2 | + intent routing | 0.541 | 0.810 |

The 100× F1 jump from round 0→1 confirms the core principle: **dense retrieval over
raw text dominates keyword search for memory recall.**

### Running the benchmark

```bash
export ANTHROPIC_API_KEY=sk-ant-...

# HotpotQA head-to-head (public dataset, no download needed)
python -c "
from metamem.benchmarks.head_to_head import run_head_to_head
run_head_to_head(n_samples=50)
"

# HotpotQA evolution benchmark (3 rounds, weak → optimised)
python -c "
from metamem.benchmarks.hotpotqa import run_hotpotqa
run_hotpotqa(n_samples=100, max_rounds=3, initial='weak')
"
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev,benchmark]"

# Run tests
pytest tests/ -v

# Lint
ruff check metamem/
```

## Configuration

Settings in `~/.metamem/settings.json` (auto-created):

```json
{
  "embed_model": "all-MiniLM-L6-v2",
  "context_budget_tokens": 4000,
  "evolution": {
    "reinforce_boost": 0.03,
    "decay_penalty": 0.10,
    "auto_extract_skills": true,
    "consolidation_interval": "daily"
  },
  "retrieval": {
    "fusion_mode": "rrf",
    "enable_intent_routing": true,
    "confidence_threshold": 0.1
  }
}
```

## Design Principles

These principles drive every architectural decision in MetaMem:

**1. Store raw, retrieve raw.**
Pre-classifying, pre-summarising, or building knowledge graphs at write time is an
LLM-era anti-pattern. The LLM at retrieval time is smarter than the LLM at write time.
Store raw events and paragraphs. Let the LLM build its own understanding when it reads
them back.

**2. Dense retrieval over keyword search.**
BM25/FTS retrieval gets 0.076 F1 on HotpotQA. RRF + dense embeddings gets 0.588 — a
100× improvement. Keyword overlap is a poor proxy for semantic relevance. Typed memories
and FTS are acceleration indexes only, never the source of truth.

**3. Iterative retrieval for multi-hop.**
For questions that chain two facts together, one retrieval pass is not enough. The right
architecture retrieves once, identifies what's missing, generates a targeted follow-up
query, and retrieves again. This is the bridge between MetaMem's comparison strength
and SimpleMem's bridge-question strength.

**4. Evolution from signal, not from schedule.**
Memory confidence should update from actual task outcomes — reinforce on success, decay
on failure. Periodic consolidation should merge high-similarity memories into stronger
ones. The system should get better by being used, not by being manually curated.

**5. Latent-space endgame.**
The current text → typed schema → SQLite architecture is a stepping stone. The long-term
direction is KV-cache storage: inject past context directly at the activation level
without re-encoding. Design retrieval interfaces to be stable across text → activation
backends.

## Comparison with Claude-Mem

| | Claude-Mem | MetaMem |
|-|-----------|---------|
| **Metaphor** | 📹 Recording | 🧠 Learning brain |
| **Memory model** | Flat observations | 5 typed stores |
| **Evolution** | None (write-once) | Task-result feedback loop |
| **Retrieval** | FTS + vector | Intent-aware + entity graph + typed routing |
| **Integration** | Claude Code hooks | MCP server (any client) |
| **Token efficiency** | Progressive disclosure ✓ | Progressive disclosure ✓ |

## License

Apache License 2.0
