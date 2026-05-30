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

This adds MetaMem as an MCP server in `~/.claude/claude_desktop_config.json`. Restart Claude Code to activate.

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
```

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

## Running Benchmarks

MetaMem is compatible with EvolveMem's evaluation protocol:

```bash
# Set your API key
export OPENAI_API_KEY=sk-...

# Run LoCoMo benchmark
metamem benchmark locomo --data data/locomo10.json --max-rounds 5 --initial weak

# Run MemBench
metamem benchmark membench --data data/membench/repo/MemData --max-rounds 3
```

The benchmark runner:
1. Extracts typed memories from session data
2. Evaluates QA pairs using retrieved context
3. Evolves retrieval config based on failures (LLM-diagnosed)
4. Reports per-round metrics and improvement trajectory

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
