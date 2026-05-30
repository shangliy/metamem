"""Tests for the dashboard module (metamem/dashboard.py).

Uses FastAPI's TestClient — no live server needed.
"""

import pytest

from metamem import dashboard, usage

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


# ── Pure helpers ──


def test_list_projects_empty(tmp_path):
    assert dashboard.list_projects(str(tmp_path)) == []


def test_list_projects(tmp_path):
    (tmp_path / "projects" / "alpha").mkdir(parents=True)
    (tmp_path / "projects" / "beta").mkdir(parents=True)
    assert dashboard.list_projects(str(tmp_path)) == ["alpha", "beta"]


def test_collect_usage_filters_by_project(tmp_path):
    data_dir = str(tmp_path)
    usage.record_usage(data_dir, usage.build_record("s1", "alpha", {
        "input_tokens": 100, "output_tokens": 50,
        "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}))
    usage.record_usage(data_dir, usage.build_record("s2", "beta", {
        "input_tokens": 200, "output_tokens": 80,
        "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}))

    all_usage = dashboard.collect_usage(data_dir, None)
    assert all_usage["totals"]["input_tokens"] == 300

    alpha = dashboard.collect_usage(data_dir, "alpha")
    assert alpha["totals"]["input_tokens"] == 100


def test_collect_stats_no_projects(tmp_path):
    stats = dashboard.collect_stats(str(tmp_path), None)
    assert stats["store"]["total"] == 0
    assert stats["evolution"]["total_actions"] == 0


def test_collect_memories_no_projects(tmp_path):
    result = dashboard.collect_memories(str(tmp_path), None)
    assert result["count"] == 0
    assert result["memories"] == []


# ── API endpoints ──


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("METAMEM_DATA_DIR", str(tmp_path))
    return TestClient(dashboard.create_app())


def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "MetaMem" in r.text
    assert "<html" in r.text.lower()


def test_api_projects(client):
    r = client.get("/api/projects")
    assert r.status_code == 200
    assert r.json() == {"projects": []}


def test_api_stats(client):
    r = client.get("/api/stats")
    assert r.status_code == 200
    body = r.json()
    assert "store" in body and "evolution" in body


def test_api_memories(client):
    r = client.get("/api/memories")
    assert r.status_code == 200
    assert r.json()["count"] == 0


def test_api_usage(client, tmp_path):
    usage.record_usage(str(tmp_path), usage.build_record("s1", "p1", {
        "input_tokens": 100, "output_tokens": 50,
        "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}))
    r = client.get("/api/usage")
    assert r.status_code == 200
    assert r.json()["totals"]["input_tokens"] == 100
