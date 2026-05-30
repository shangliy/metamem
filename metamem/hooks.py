"""Claude Code hooks layer — deterministic memory capture.

Claude Code runs these at lifecycle events (SessionStart, UserPromptSubmit,
Stop, SessionEnd), piping a JSON payload via stdin and reading a JSON response
from stdout. Unlike the CLAUDE.md prompt instructions (which the model may skip),
hooks fire regardless of whether the model "remembers" — making memory capture
deterministic.

Design rules:
- Reuse the existing session/store layer; no new memory logic here.
- Pin the SessionManager to Claude Code's `session_id` so every hook invocation
  for one conversation appends to the SAME session folder.
- NEVER block: every handler returns ``{"continue": True}`` even on error.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from .session import SessionManager, detect_project

# Turn boundary marker in Claude Code transcripts (the ONLY reliable boundary).
_TURN_DURATION = ("system", "turn_duration")
_TRANSCRIPT_RETRIES = 5
_TRANSCRIPT_RETRY_DELAY = 0.1  # seconds


def _safe_session_id(raw: str) -> str:
    """Sanitize a Claude session id into a filesystem-safe folder name."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", raw)[:64] if raw else ""


def _make_session(payload: dict) -> SessionManager:
    """Build a SessionManager pinned to Claude Code's session id + cwd."""
    data_dir = os.environ.get("METAMEM_DATA_DIR", os.path.expanduser("~/.metamem"))
    cwd = payload.get("cwd") or os.getcwd()
    project = os.environ.get("METAMEM_PROJECT", "") or detect_project(cwd)
    session_id = _safe_session_id(str(payload.get("session_id", "")))
    return SessionManager.start(
        project=project,
        cwd=cwd,
        data_dir=data_dir,
        session_id=session_id,
    )


# ── Transcript parsing (Stop hook) ──


def _read_transcript_with_retry(path: Path) -> list[dict]:
    """Read transcript JSONL, retrying until the turn is complete.

    The Stop hook can fire before Claude Code flushes the closing
    ``turn_duration`` line. Retry a few times so we capture the full turn.
    """
    for _ in range(_TRANSCRIPT_RETRIES):
        lines = _parse_jsonl(path)
        if lines:
            last = lines[-1]
            if (last.get("type"), last.get("subtype")) == _TURN_DURATION:
                return lines
        time.sleep(_TRANSCRIPT_RETRY_DELAY)
    return _parse_jsonl(path)


def _parse_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file into a list of dicts, skipping malformed lines."""
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        return []
    return out


def _extract_last_turn(lines: list[dict]) -> tuple[str, str]:
    """Extract (user_text, assistant_text) for the last complete turn.

    - Turn boundary = ``turn_duration`` lines (NOT ``file-history-snapshot``).
    - User text = original ``user`` messages, skipping ``tool_result`` blocks.
    - Assistant text = all ``text`` blocks, skipping ``thinking`` / ``tool_use``.
    """
    # Find boundary indices of turn_duration markers.
    boundaries = [
        i for i, ln in enumerate(lines)
        if (ln.get("type"), ln.get("subtype")) == _TURN_DURATION
    ]
    if boundaries:
        end = boundaries[-1]
        start = boundaries[-2] + 1 if len(boundaries) >= 2 else 0
    else:
        start, end = 0, len(lines)

    turn = lines[start:end]
    user_parts: list[str] = []
    assistant_parts: list[str] = []

    for ln in turn:
        msg = ln.get("message") or {}
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            user_parts.extend(_collect_user_text(content))
        elif role == "assistant":
            assistant_parts.extend(_collect_assistant_text(content))

    return "\n\n".join(user_parts).strip(), "\n\n".join(assistant_parts).strip()


def _collect_user_text(content: Any) -> list[str]:
    """User text = plain strings or ``text`` blocks; skip ``tool_result``."""
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return [p for p in parts if p]


def _collect_assistant_text(content: Any) -> list[str]:
    """Assistant text = ``text`` blocks only; skip ``thinking`` / ``tool_use``."""
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return [p for p in parts if p]


# ── Hook handlers ──


def handle_session_start(payload: dict) -> dict:
    """Inject prior project context into Claude's context window."""
    sm = _make_session(payload)
    context = sm.get_context_injection()
    if not context:
        return {"continue": True}
    sessions = sm.list_sessions(limit=1)
    last = sessions[0] if sessions else {}
    msg = f"🧠 MetaMem: context loaded ({last.get('event_count', 0)} events from last session)"
    return {
        "continue": True,
        "systemMessage": msg,
        "systemPrompt": f"<metamem-context>\n{context}\n</metamem-context>",
    }


def handle_user_prompt_submit(payload: dict) -> dict:
    """Search memory for the prompt and inject relevant hits as context."""
    prompt = (payload.get("prompt") or "").strip()
    # Skip trivially short prompts (cheap + avoids noise), matching EverOS.
    if len(prompt.split()) < 3:
        return {"continue": True}

    from .mcp_server import mem_search
    result = mem_search(query=prompt, limit=5)
    hits = result.get("results", [])
    if not hits:
        return {"continue": True}

    lines = [f"[{h['score']:.2f}] ({h['type']}) {h['summary']}" for h in hits]
    injected = "\n".join(lines)
    return {
        "continue": True,
        "systemMessage": f"📝 MetaMem: {len(hits)} relevant memories",
        "systemPrompt": f"<metamem-recall>\n{injected}\n</metamem-recall>",
    }


def handle_stop(payload: dict) -> dict:
    """Capture the last completed turn as a lightweight session event."""
    transcript = payload.get("transcript_path")
    if not transcript:
        return {"continue": True}

    path = Path(os.path.expanduser(transcript))
    lines = _read_transcript_with_retry(path)
    user_text, assistant_text = _extract_last_turn(lines)
    if not user_text and not assistant_text:
        return {"continue": True}

    sm = _make_session(payload)
    summary = (user_text or assistant_text)[:200]
    sm.add_event(
        "message",
        f"User: {user_text}\n\nAssistant: {assistant_text}",
        metadata={"source": "claude-code-hook", "summary": summary},
    )
    return {"systemMessage": f"💾 MetaMem: turn saved ({len(sm.session.events)} events)"}


def handle_session_end(payload: dict) -> dict:
    """Finalize the session — generate a summary and absorb into project memory."""
    sm = _make_session(payload)
    if not sm.session.events:
        return {"continue": True}
    sm.finalize()
    return {
        "systemMessage": f"📦 MetaMem: session finalized ({len(sm.session.events)} events)"
    }


# ── Dispatch ──

_HANDLERS = {
    "session-start": handle_session_start,
    "user-prompt-submit": handle_user_prompt_submit,
    "stop": handle_stop,
    "session-end": handle_session_end,
}


def dispatch(event: str, payload: dict) -> dict:
    """Route an event to its handler, never raising (returns continue:true)."""
    handler = _HANDLERS.get(event)
    if handler is None:
        return {"continue": True}
    try:
        return handler(payload)
    except Exception:
        # Graceful degradation — a hook must never block Claude Code.
        return {"continue": True}


def run_hook(event: str, stdin_text: str) -> dict:
    """Parse stdin JSON and dispatch. Used by the CLI hook commands."""
    try:
        payload = json.loads(stdin_text) if stdin_text.strip() else {}
        if not isinstance(payload, dict):
            payload = {}
    except json.JSONDecodeError:
        payload = {}
    return dispatch(event, payload)
