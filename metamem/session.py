"""Session Manager — Per-project, per-session memory with auto-continuation.

When you open a new Claude session in a project folder:
1. Detects the project (git repo root or folder name)
2. Loads previous session context from that project's memory
3. Tracks all events during the session in a dedicated folder
4. At session end, absorbs important memories into the project store

Folder layout:
    ~/.metamem/
    ├── projects/
    │   ├── my-web-app/              # One folder per project
    │   │   ├── project_memory.db    # Project-scoped persistent memory
    │   │   ├── embeddings/
    │   │   ├── sessions/
    │   │   │   ├── 20260529_143022/ # One folder per session
    │   │   │   │   ├── manifest.json
    │   │   │   │   ├── events.jsonl
    │   │   │   │   ├── topics/
    │   │   │   │   └── summary.md
    │   │   │   └── 20260528_091500/
    │   │   └── context_injection.md # Auto-generated for next session
    │   └── api-service/
    └── global/
        └── memory.db               # Cross-project global memory
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .models import (
    Event,
    MemoryType,
    MemoryUnit,
    SessionMemory,
    Topic,
)
from .store import MemoryStore

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = os.path.expanduser("~/.metamem")


def detect_project(cwd: str | None = None) -> str:
    """Detect the current project from git repo or folder name.

    Priority:
    1. Git remote URL (normalized to repo name)
    2. Git repo root folder name
    3. Current working directory name
    """
    cwd = cwd or os.getcwd()

    # Try git repo root
    try:
        repo_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd, stderr=subprocess.DEVNULL, text=True,
        ).strip()

        # Try to get a meaningful name from remote
        try:
            remote_url = subprocess.check_output(
                ["git", "remote", "get-url", "origin"],
                cwd=cwd, stderr=subprocess.DEVNULL, text=True,
            ).strip()
            # Extract repo name: git@github.com:user/repo.git → repo
            name = remote_url.rstrip("/").split("/")[-1]
            if name.endswith(".git"):
                name = name[:-4]
            return name
        except (subprocess.CalledProcessError, IndexError):
            pass

        # Fallback: repo root folder name
        return os.path.basename(repo_root)

    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Fallback: current directory name
    return os.path.basename(os.path.abspath(cwd))


@dataclass
class SessionConfig:
    """Configuration for a session."""
    project: str = ""
    data_dir: str = DEFAULT_DATA_DIR
    auto_inject_context: bool = True
    context_budget_tokens: int = 3000
    topic_segmentation: bool = True


class SessionManager:
    """Manages per-project, per-session memory.

    Usage:
        sm = SessionManager.start(project="my-app")
        # ... session runs, events are captured ...
        sm.add_event("message", "User asked about deployment")
        sm.add_event("tool_call", "Ran pytest")
        # ... at session end ...
        sm.finalize()
    """

    def __init__(self, config: SessionConfig | None = None):
        self.config = config or SessionConfig()
        if not self.config.project:
            self.config.project = detect_project()

        self.data_dir = Path(self.config.data_dir)
        self.project_dir = self.data_dir / "projects" / self.config.project
        self.sessions_dir = self.project_dir / "sessions"

        # Create project directory structure
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(exist_ok=True)
        (self.project_dir / "embeddings").mkdir(exist_ok=True)

        # Current session
        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.sessions_dir / self.session_id
        self.session_dir.mkdir(exist_ok=True)
        (self.session_dir / "topics").mkdir(exist_ok=True)

        # Session state
        self.session = SessionMemory(
            session_id=self.session_id,
            project=self.config.project,
        )
        self._events_file = open(self.session_dir / "events.jsonl", "a")

        # Project-scoped memory store
        self._store: MemoryStore | None = None

        # Write manifest
        self._write_manifest()

        logger.info(
            "Session started: project=%s session=%s",
            self.config.project, self.session_id,
        )

    @classmethod
    def start(cls, project: str | None = None, cwd: str | None = None, **kwargs) -> "SessionManager":
        """Start a new session, auto-detecting project if not specified."""
        config = SessionConfig(
            project=project or detect_project(cwd),
            **kwargs,
        )
        return cls(config)

    @property
    def store(self) -> MemoryStore:
        """Lazy-load the project memory store."""
        if self._store is None:
            self._store = MemoryStore(data_dir=str(self.project_dir))
        return self._store

    # ── Event Capture ──

    def add_event(self, event_type: str, content: str, metadata: dict | None = None):
        """Record an event in the session log.

        Args:
            event_type: "message" | "tool_call" | "tool_result" | "error" | "observation"
            content: Event content
            metadata: Optional extra data
        """
        event = Event(
            type=event_type,
            content=content,
            metadata=metadata or {},
        )
        self.session.events.append(event)

        # Append to JSONL file
        record = {
            "id": event.event_id,
            "type": event.type,
            "content": event.content,
            "timestamp": event.timestamp,
            "metadata": event.metadata,
        }
        self._events_file.write(json.dumps(record) + "\n")
        self._events_file.flush()

    # ── Context Injection ──

    def get_context_injection(self) -> str:
        """Generate context to inject at session start.

        Loads from the project's previous sessions and memories to create
        a token-budgeted context bundle.
        """
        sections: list[str] = []
        token_budget = self.config.context_budget_tokens
        tokens_used = 0

        # 1. Instructions (always loaded, highest priority)
        instructions = self.store.get_by_type(MemoryType.INSTRUCTION)
        if instructions:
            inst_lines = ["## Project Instructions"]
            for mem in sorted(instructions, key=lambda m: -m.confidence)[:10]:
                line = f"- {mem.content}"
                inst_lines.append(line)
                tokens_used += len(line.split()) * 1.3
            sections.append("\n".join(inst_lines))

        # 2. Last session summary (what was I doing?)
        last_summary = self._get_last_session_summary()
        if last_summary and tokens_used < token_budget * 0.3:
            sections.append(f"## Last Session\n{last_summary}")
            tokens_used += len(last_summary.split()) * 1.3

        # 3. Recent procedural memories (skills for this project)
        procedures = self.store.get_by_type(MemoryType.PROCEDURAL)
        if procedures and tokens_used < token_budget * 0.6:
            proc_lines = ["## Project Skills"]
            for mem in sorted(procedures, key=lambda m: -m.confidence)[:5]:
                line = f"- [{mem.confidence:.0%}] {mem.content[:100]}"
                proc_lines.append(line)
                tokens_used += len(line.split()) * 1.3
                if tokens_used > token_budget * 0.6:
                    break
            sections.append("\n".join(proc_lines))

        # 4. Active failure warnings
        failures = self.store.get_by_type(MemoryType.FAILURE)
        if failures and tokens_used < token_budget * 0.8:
            fail_lines = ["## Warnings (known issues)"]
            for mem in sorted(failures, key=lambda m: -m.confidence)[:5]:
                line = f"- ⚠️ {mem.content[:100]}"
                fail_lines.append(line)
                tokens_used += len(line.split()) * 1.3
                if tokens_used > token_budget * 0.8:
                    break
            sections.append("\n".join(fail_lines))

        # 5. Key facts about the project
        semantics = self.store.get_by_type(MemoryType.SEMANTIC)
        if semantics and tokens_used < token_budget:
            fact_lines = ["## Project Knowledge"]
            for mem in sorted(semantics, key=lambda m: -m.importance)[:10]:
                line = f"- {mem.content[:80]}"
                fact_lines.append(line)
                tokens_used += len(line.split()) * 1.3
                if tokens_used > token_budget:
                    break
            sections.append("\n".join(fact_lines))

        if not sections:
            return ""

        header = f"# Memory Context: {self.config.project}\n\n"
        context = header + "\n\n".join(sections)

        # Save for reference
        injection_path = self.project_dir / "context_injection.md"
        injection_path.write_text(context)

        return context

    def _get_last_session_summary(self) -> str:
        """Get the summary from the most recent previous session."""
        session_dirs = sorted(
            [d for d in self.sessions_dir.iterdir() if d.is_dir() and d.name != self.session_id],
            reverse=True,
        )
        if not session_dirs:
            return ""

        last_dir = session_dirs[0]
        summary_file = last_dir / "summary.md"
        if summary_file.exists():
            return summary_file.read_text().strip()

        # Try to reconstruct from events
        events_file = last_dir / "events.jsonl"
        if events_file.exists():
            events = []
            for line in events_file.read_text().strip().split("\n")[-20:]:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            if events:
                last_msgs = [e["content"][:100] for e in events if e.get("type") == "message"]
                if last_msgs:
                    return "Recent activity: " + "; ".join(last_msgs[-5:])

        return ""

    # ── Topic Segmentation ──

    def segment_topic(self, title: str = ""):
        """Mark the start of a new topic in the session."""
        # Finalize previous topic
        if self.session.topics:
            self.session.topics[-1].end_time = time.time()

        topic = Topic(
            title=title or f"Topic {len(self.session.topics) + 1}",
            start_time=time.time(),
        )
        self.session.topics.append(topic)

    # ── Session Finalization ──

    def finalize(self, summary: str = "", llm_call=None):
        """Finalize the session — generate summary, absorb into project memory.

        Args:
            summary: Optional manual summary. If not provided and llm_call
                     is available, generates one automatically.
            llm_call: Optional LLM function for auto-summarization.
        """
        self.session.status = "finalized"

        # Generate summary
        if not summary and llm_call and self.session.events:
            summary = self._auto_summarize(llm_call)
        elif not summary and self.session.events:
            # Simple heuristic summary
            messages = [e for e in self.session.events if e.type == "message"]
            if messages:
                summary = f"Session with {len(self.session.events)} events. " \
                          f"Last activity: {messages[-1].content[:100]}"

        self.session.summary = summary

        # Save summary
        (self.session_dir / "summary.md").write_text(summary)

        # Save topics
        for i, topic in enumerate(self.session.topics):
            topic_file = self.session_dir / "topics" / f"topic_{i}_{topic.title[:30].replace(' ', '_')}.md"
            topic_events = [
                e for e in self.session.events
                if topic.start_time <= e.timestamp <= (topic.end_time or time.time())
            ]
            content = f"# {topic.title}\n\n"
            for evt in topic_events:
                content += f"- [{evt.type}] {evt.content[:200]}\n"
            topic_file.write_text(content)

        # Update manifest
        self._write_manifest()

        # Close files
        self._events_file.close()

        logger.info("Session finalized: %s (%d events)", self.session_id, len(self.session.events))

    def _auto_summarize(self, llm_call) -> str:
        """Use LLM to summarize the session."""
        events_text = "\n".join([
            f"[{e.type}] {e.content[:150]}"
            for e in self.session.events[-30:]
        ])
        messages = [
            {"role": "system", "content": "Summarize this work session in 2-3 sentences. Focus on what was accomplished and what's pending."},
            {"role": "user", "content": events_text},
        ]
        try:
            return llm_call(messages, max_tokens=200, temperature=0.1)
        except Exception:
            return ""

    def _write_manifest(self):
        """Write session manifest."""
        manifest = {
            "session_id": self.session_id,
            "project": self.config.project,
            "created_at": self.session.created_at,
            "status": self.session.status,
            "event_count": len(self.session.events),
            "topic_count": len(self.session.topics),
        }
        (self.session_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # ── Project History ──

    def list_sessions(self, limit: int = 10) -> list[dict]:
        """List recent sessions for this project."""
        session_dirs = sorted(
            [d for d in self.sessions_dir.iterdir() if d.is_dir()],
            reverse=True,
        )[:limit]

        sessions = []
        for d in session_dirs:
            manifest_file = d / "manifest.json"
            if manifest_file.exists():
                sessions.append(json.loads(manifest_file.read_text()))
            else:
                sessions.append({"session_id": d.name, "status": "unknown"})
        return sessions

    def get_project_stats(self) -> dict[str, Any]:
        """Get project-level statistics."""
        sessions = list(self.sessions_dir.iterdir()) if self.sessions_dir.exists() else []
        return {
            "project": self.config.project,
            "total_sessions": len(sessions),
            "memory_stats": self.store.stats(),
            "project_dir": str(self.project_dir),
        }
