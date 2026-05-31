"""DCI-style retrieval engine for Mem-Engram.

Implements the Direct Corpus Interaction (DCI) paradigm from:
  "Beyond Semantic Similarity: Rethinking Retrieval for Agentic Search
   via Direct Corpus Interaction" (DCI-Agent, 2025)

Instead of pre-computing embeddings and doing cosine similarity, the LLM
searches a raw plaintext corpus directly with ripgrep — composing its own
search strategies and iterating until it has enough evidence.

This directly implements Mem-Engram's design principle:
  "grep-style present raw data, let LLM build its own index — that is the
   scalable approach. The LLM at retrieval time is smarter than any
   pre-computed index."

Two modes:
  DCIRetriever       — pure DCI: only rg, no embeddings
  HybridDCIRetriever — embedding narrows candidates, DCI refines within them
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


# ── Corpus file store ─────────────────────────────────────────────────────────

class CorpusStore:
    """Stores text documents as plaintext files for rg-based search.

    Layout:
        corpus_dir/
            doc_000.txt   ← paragraph / memory content
            doc_001.txt
            ...
        index.json        ← {filename: {id, title, ...}}
    """

    def __init__(self, corpus_dir: str):
        self.root = Path(corpus_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, dict] = {}
        self._counter = 0
        index_file = self.root / "index.json"
        if index_file.exists():
            self._index = json.loads(index_file.read_text())
            self._counter = len(self._index)

    def add(self, doc_id: str, content: str, meta: dict | None = None) -> str:
        fname = f"doc_{self._counter:04d}.txt"
        self._counter += 1
        (self.root / fname).write_text(content, encoding="utf-8")
        self._index[fname] = {"id": doc_id, "meta": meta or {}}
        (self.root / "index.json").write_text(json.dumps(self._index), encoding="utf-8")
        return fname

    def get_meta(self, fname: str) -> dict:
        return self._index.get(fname, {})

    def all_files(self) -> list[str]:
        return [str(self.root / f) for f in sorted(self._index)]


# ── Search primitive ─────────────────────────────────────────────────────────

_RG_CANDIDATES = [
    "rg",
    "/home/shawnyang/.vscode/extensions/openai.chatgpt-26.5527.31454-linux-x64/bin/linux-x86_64/rg",
    "/usr/bin/rg",
    "/usr/local/bin/rg",
]


def _rg_bin() -> str:
    """Find a working ripgrep binary."""
    import shutil
    hit = shutil.which("rg")
    if hit:
        return hit
    for path in _RG_CANDIDATES:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError("ripgrep (rg) not found — install with: apt install ripgrep")


def rg_search(
    pattern: str,
    corpus_dir: str,
    context_lines: int = 2,
    max_results: int = 30,
    case_insensitive: bool = True,
) -> list[dict]:
    """Run ripgrep on corpus_dir and return structured matches.

    Returns list of {file, line_number, text, context_before, context_after}.
    """
    cmd = [_rg_bin(), "--json", f"--context={context_lines}",
           f"--max-count={max_results}"]
    if case_insensitive:
        cmd.append("--ignore-case")
    cmd += [pattern, corpus_dir]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []

    matches: list[dict] = []
    current: dict | None = None

    for line in result.stdout.splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = obj.get("type")
        if t == "match":
            data = obj["data"]
            file_path = data["path"]["text"]
            fname = Path(file_path).name
            match_text = data["lines"]["text"].rstrip("\n")
            matches.append({
                "file": fname,
                "path": file_path,
                "line": data["line_number"],
                "text": match_text,
            })
            current = matches[-1]
        elif t == "context" and current:
            pass  # context lines available if needed

    # Deduplicate by file — return at most one match entry per file
    seen: set[str] = set()
    deduped = []
    for m in matches:
        if m["file"] not in seen:
            seen.add(m["file"])
            deduped.append(m)
    return deduped


def read_full_doc(path: str) -> str:
    """Read a full document from the corpus."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""


# ── DCI retrieval loop ────────────────────────────────────────────────────────

_SEARCH_TOOL_SCHEMA = {
    "name": "search_corpus",
    "description": (
        "Search the raw document corpus using ripgrep. "
        "Returns matching document excerpts. "
        "Use this tool iteratively — compose specific patterns, "
        "read results, and search again if you need more evidence."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Ripgrep regex pattern to search for."
            },
            "case_insensitive": {
                "type": "boolean",
                "description": "Whether to ignore case (default true).",
                "default": True,
            },
        },
        "required": ["pattern"],
    },
}

_FINISH_TOOL_SCHEMA = {
    "name": "finish",
    "description": "Return when you have gathered enough evidence to answer the question.",
    "parameters": {
        "type": "object",
        "properties": {
            "relevant_docs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of document filenames that contain relevant evidence.",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of what you found.",
            },
        },
        "required": ["relevant_docs"],
    },
}


@dataclass
class DCIResult:
    """Result from a DCI retrieval session."""
    relevant_files: list[str]
    search_history: list[dict]   # [{pattern, matches_found}, ...]
    reasoning: str = ""
    n_searches: int = 0


class DCIRetriever:
    """Pure DCI retrieval: LLM searches raw corpus with rg, no embeddings.

    The LLM is given two tools: search_corpus(pattern) and finish(relevant_docs).
    It iterates searches until it decides it has enough evidence.
    """

    def __init__(
        self,
        llm_call: Callable,
        corpus_store: CorpusStore,
        max_searches: int = 5,
        model: str | None = None,
    ):
        self.llm_call = llm_call
        self.store = corpus_store
        self.max_searches = max_searches
        self.model = model or os.environ.get("LLM_MODEL", "claude-haiku-4-5-20251001")

    def retrieve(self, query: str) -> DCIResult:
        """Run the DCI loop: LLM searches corpus until it finds enough evidence."""
        import anthropic

        client = anthropic.Anthropic()
        tools = [_SEARCH_TOOL_SCHEMA, _FINISH_TOOL_SCHEMA]

        system = (
            "You are a precise research assistant searching a document corpus to answer a question. "
            "Use search_corpus to find relevant documents. "
            "Start with specific entity names or key phrases from the question. "
            "For multi-hop questions: search for the first entity, read the result, "
            "extract the linking concept, then search for that. "
            "Call finish() when you have identified the relevant documents."
        )
        messages = [{"role": "user", "content": f"Question: {query}\n\nSearch the corpus to find the relevant documents."}]

        search_history: list[dict] = []
        found_files: list[str] = []
        reasoning = ""
        n_searches = 0

        for _ in range(self.max_searches + 1):
            try:
                resp = client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    system=system,
                    tools=tools,
                    messages=messages,
                )
            except Exception as e:
                break

            # Process tool calls
            tool_calls = [b for b in resp.content if b.type == "tool_use"]
            text_blocks = [b.text for b in resp.content if b.type == "text"]

            if not tool_calls:
                break

            # Build assistant message
            messages.append({"role": "assistant", "content": resp.content})

            tool_results = []
            done = False
            for tc in tool_calls:
                if tc.name == "finish":
                    found_files = tc.input.get("relevant_docs", [])
                    reasoning = tc.input.get("reasoning", "")
                    done = True
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": "Done.",
                    })
                elif tc.name == "search_corpus":
                    pattern = tc.input.get("pattern", "")
                    ci = tc.input.get("case_insensitive", True)
                    matches = rg_search(pattern, str(self.store.root),
                                        case_insensitive=ci, max_results=20)
                    n_searches += 1
                    search_history.append({"pattern": pattern, "matches": len(matches)})

                    if matches:
                        # Return file content for each matched file
                        result_parts = []
                        for m in matches[:8]:
                            content = read_full_doc(m["path"])[:600]
                            result_parts.append(f"[{m['file']}]\n{content}")
                        result_text = "\n\n---\n\n".join(result_parts)
                    else:
                        result_text = f"No matches found for pattern: {pattern!r}"

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": result_text,
                    })

            messages.append({"role": "user", "content": tool_results})

            if done:
                break

        return DCIResult(
            relevant_files=found_files,
            search_history=search_history,
            reasoning=reasoning,
            n_searches=n_searches,
        )

    def retrieve_as_context(self, query: str) -> tuple[str, list[str]]:
        """Returns (context_text, list_of_doc_ids) for answer generation."""
        result = self.retrieve(query)
        parts = []
        doc_ids = []
        for fname in result.relevant_files:
            path = self.store.root / fname
            if path.exists():
                content = path.read_text(encoding="utf-8")
                parts.append(content)
                meta = self.store.get_meta(fname)
                doc_ids.append(meta.get("id", fname))
        context = "\n\n".join(parts) if parts else ""
        return context, doc_ids


# ── Hybrid: embedding narrows, DCI refines ───────────────────────────────────

class HybridDCIRetriever:
    """Hybrid: dense retrieval narrows to top-N candidates, DCI searches within them.

    Combines the recall of embedding search with DCI's precision — the LLM
    searches a smaller, pre-filtered corpus rather than the full store.
    """

    def __init__(
        self,
        llm_call: Callable,
        embedder,
        mem_store,               # MemoryStore instance (for embedding search)
        retrieval_config,        # RetrievalConfig
        dci_max_searches: int = 3,
        pre_filter_k: int = 15,
    ):
        from .retriever import RetrievalEngine
        self.base_engine = RetrievalEngine(mem_store, retrieval_config)
        self.embedder = embedder
        self.llm_call = llm_call
        self.dci_max_searches = dci_max_searches
        self.pre_filter_k = pre_filter_k

    def retrieve_as_context(
        self,
        query: str,
        config=None,
    ) -> tuple[str, list[str]]:
        """Two-stage: embedding top-k → write to temp corpus → DCI search."""
        from .retriever import format_context

        cfg = config or self.base_engine.config
        q_emb = self.embedder.encode(query) if self.embedder else None

        # Stage 1: embedding retrieval — get top-k candidates
        candidates = self.base_engine.search(query, cfg, q_emb)[:self.pre_filter_k]
        if not candidates:
            return "", []

        # Stage 2: write candidates to temp corpus for DCI
        with tempfile.TemporaryDirectory(prefix="mem_engram_hybrid_") as tmp:
            cs = CorpusStore(tmp)
            for rm in candidates:
                cs.add(rm.memory.id, rm.memory.content, {"title": rm.memory.summary})

            dci = DCIRetriever(self.llm_call, cs, max_searches=self.dci_max_searches)
            context, doc_ids = dci.retrieve_as_context(query)

        # Fallback: if DCI found nothing, return embedding top-5
        if not context:
            context = format_context(candidates[:5], max_tokens=1500)
            doc_ids = [rm.memory.id for rm in candidates[:5]]

        return context, doc_ids
