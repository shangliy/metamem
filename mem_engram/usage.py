"""Token usage tracking — a separate, append-only ledger.

Claude Code transcripts record per-turn token usage at ``message.usage``. The
Stop hook already parses the transcript to capture turns, so we extract usage in
the same pass and append it to a dedicated JSONL ledger — kept OUT of the memory
store so the memory DB stays clean and the usage data is easy to analyze later.

Ledger location: ``<data_dir>/usage/token_usage.jsonl`` (default ``~/.mem-engram``).
One JSON object per captured turn.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

# Token fields we extract from a transcript's ``message.usage`` object.
_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def _ledger_path(data_dir: str) -> Path:
    """Path to the token usage ledger under the data dir."""
    return Path(data_dir) / "usage" / "token_usage.jsonl"


def extract_usage(lines: list[dict]) -> dict[str, int] | None:
    """Extract token usage from the last assistant line carrying a ``usage`` block.

    Returns a dict with the four token fields (missing ones default to 0), or
    ``None`` if no usage data is present (e.g. older transcripts).
    """
    for ln in reversed(lines):
        msg = ln.get("message") or {}
        usage = msg.get("usage")
        if isinstance(usage, dict):
            return {field: int(usage.get(field, 0) or 0) for field in _USAGE_FIELDS}
    return None


def record_usage(data_dir: str, record: dict[str, Any]) -> None:
    """Append a usage record to the ledger (append-only, crash-safe)."""
    path = _ledger_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def build_record(
    session_id: str,
    project: str,
    usage: dict[str, int],
    ts: float | None = None,
) -> dict[str, Any]:
    """Build a normalized usage record for the ledger."""
    record = {
        "ts": ts if ts is not None else time.time(),
        "session_id": session_id,
        "project": project,
    }
    record.update({field: int(usage.get(field, 0) or 0) for field in _USAGE_FIELDS})
    return record


def _hits_ledger_path(data_dir: str) -> Path:
    return Path(data_dir) / "usage" / "memory_hits.jsonl"


def record_memory_hits(data_dir: str, record: dict[str, Any]) -> None:
    """Append a memory-hit record to the hits ledger (append-only)."""
    path = _hits_ledger_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_memory_hits(data_dir: str) -> list[dict]:
    """Load all memory-hit records from the hits ledger."""
    path = _hits_ledger_path(data_dir)
    if not path.exists():
        return []
    records: list[dict] = []
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        return []
    return records


def summarize_memory_hits(records: list[dict]) -> dict[str, Any]:
    """Aggregate memory-hit records into totals and per-project view."""
    totals = {"memories_loaded": 0, "memory_hits": 0, "memories_distilled": 0}
    by_project: dict[str, dict[str, int]] = {}

    for rec in records:
        proj = rec.get("project", "unknown")
        prj = by_project.setdefault(proj, {"memories_loaded": 0, "memory_hits": 0, "memories_distilled": 0, "sessions": 0})
        prj["sessions"] += 1
        for field in ("memories_loaded", "memory_hits", "memories_distilled"):
            val = int(rec.get(field, 0) or 0)
            totals[field] += val
            prj[field] += val

    return {
        "totals": totals,
        "sessions": len(records),
        "by_project": by_project,
    }


def load_usage(data_dir: str) -> list[dict]:
    """Load all usage records from the ledger, skipping malformed lines."""
    path = _ledger_path(data_dir)
    if not path.exists():
        return []
    records: list[dict] = []
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        return []
    return records


def summarize(records: list[dict]) -> dict[str, Any]:
    """Aggregate usage records into totals, per-session, and per-project views."""
    totals = {field: 0 for field in _USAGE_FIELDS}
    by_session: dict[str, dict[str, int]] = {}
    by_project: dict[str, dict[str, int]] = {}

    for rec in records:
        sid = rec.get("session_id", "unknown")
        proj = rec.get("project", "unknown")
        sess = by_session.setdefault(sid, {f: 0 for f in _USAGE_FIELDS} | {"turns": 0})
        prj = by_project.setdefault(proj, {f: 0 for f in _USAGE_FIELDS} | {"turns": 0})
        sess["turns"] += 1
        prj["turns"] += 1
        for field in _USAGE_FIELDS:
            val = int(rec.get(field, 0) or 0)
            totals[field] += val
            sess[field] += val
            prj[field] += val

    total_in = totals["input_tokens"]
    cache_read = totals["cache_read_input_tokens"]
    # Cache hit ratio = cached input / (cached input + fresh input).
    denom = total_in + cache_read
    cache_hit_ratio = round(cache_read / denom, 4) if denom else 0.0

    return {
        "totals": totals,
        "turns": len(records),
        "cache_hit_ratio": cache_hit_ratio,
        "by_session": by_session,
        "by_project": by_project,
    }
