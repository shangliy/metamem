"""MetaMem dashboard — a local, read-only web UI for memories + token usage.

Launched via ``metamem dashboard``. Builds a FastAPI app that serves a single
self-contained HTML page (no build step) plus read-only JSON endpoints backed by
the existing memory store and the token usage ledger.

Local-only by design: bind to 127.0.0.1, no auth.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_ASSETS_DIR = Path(__file__).parent / "assets"


def _data_dir() -> str:
    return os.environ.get("METAMEM_DATA_DIR", os.path.expanduser("~/.metamem"))


def list_projects(data_dir: str) -> list[str]:
    """List project names under ``<data_dir>/projects``."""
    projects_dir = Path(data_dir) / "projects"
    if not projects_dir.exists():
        return []
    return sorted(p.name for p in projects_dir.iterdir() if p.is_dir())


def _open_store(data_dir: str, project: str):
    """Open a project-scoped MemoryStore, or None if it doesn't exist."""
    from .store import MemoryStore
    project_dir = Path(data_dir) / "projects" / project
    if not project_dir.exists():
        return None
    return MemoryStore(data_dir=str(project_dir))


def collect_stats(data_dir: str, project: str | None) -> dict[str, Any]:
    """Aggregate store stats across one or all projects."""
    from .evolution import EvolutionConfig, EvolutionEngine

    projects = [project] if project else list_projects(data_dir)
    total = {"total": 0, "active": 0, "embeddings": 0}
    confidences: list[float] = []
    by_type: dict[str, int] = {}
    evo_actions = 0

    for proj in projects:
        store = _open_store(data_dir, proj)
        if store is None:
            continue
        s = store.stats()
        total["total"] += s.get("total", 0)
        total["active"] += s.get("active", 0)
        total["embeddings"] += s.get("embeddings", 0)
        if s.get("avg_confidence"):
            confidences.append(s["avg_confidence"])
        for t, c in (s.get("by_type") or {}).items():
            by_type[t] = by_type.get(t, 0) + c
        try:
            evo_actions += EvolutionEngine(store, EvolutionConfig()).get_stats().get("total_actions", 0)
        except Exception:
            pass

    avg_conf = round(sum(confidences) / len(confidences), 2) if confidences else 0
    return {
        "store": {**total, "avg_confidence": avg_conf, "by_type": by_type},
        "evolution": {"total_actions": evo_actions},
    }


def collect_memories(data_dir: str, project: str | None, mem_type: str | None = None) -> dict[str, Any]:
    """List memories (compact) across one or all projects."""
    from .models import MemoryType

    projects = [project] if project else list_projects(data_dir)
    memories: list[dict] = []

    for proj in projects:
        store = _open_store(data_dir, proj)
        if store is None:
            continue
        types = [MemoryType(mem_type)] if mem_type else list(MemoryType)
        for mt in types:
            for mem in store.get_by_type(mt):
                memories.append({
                    "id": mem.id,
                    "type": mem.type.value,
                    "summary": mem.summary or mem.content[:120],
                    "confidence": round(mem.confidence, 2),
                    "project": proj,
                })

    memories.sort(key=lambda m: -m["confidence"])
    return {"memories": memories, "count": len(memories)}


def collect_usage(data_dir: str, project: str | None) -> dict[str, Any]:
    """Summarize token usage, optionally filtered to one project."""
    from . import usage as _usage
    records = _usage.load_usage(data_dir)
    if project:
        records = [r for r in records if r.get("project") == project]
    return _usage.summarize(records)


def create_app():
    """Build the FastAPI app for the dashboard."""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="MetaMem Dashboard", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index():
        html = (_ASSETS_DIR / "dashboard.html").read_text()
        return HTMLResponse(html)

    @app.get("/api/projects")
    def api_projects():
        return JSONResponse({"projects": list_projects(_data_dir())})

    @app.get("/api/stats")
    def api_stats(project: str | None = None):
        return JSONResponse(collect_stats(_data_dir(), project))

    @app.get("/api/memories")
    def api_memories(project: str | None = None, type: str | None = None):
        return JSONResponse(collect_memories(_data_dir(), project, type))

    @app.get("/api/usage")
    def api_usage(project: str | None = None):
        return JSONResponse(collect_usage(_data_dir(), project))

    return app


def serve_dashboard(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Launch the dashboard with uvicorn (blocking)."""
    import uvicorn
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")
