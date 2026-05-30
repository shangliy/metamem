"""Tests for the Claude Code hooks layer (metamem/hooks.py)."""

import json

import pytest

from metamem import hooks


# ── Transcript parsing ──


def _line(type_, **kw):
    d = {"type": type_}
    d.update(kw)
    return d


def _user(text):
    return _line("user", message={"role": "user", "content": text})


def _assistant_text(text):
    return _line("assistant", message={
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
    })


def _assistant_thinking(text):
    return _line("assistant", message={
        "role": "assistant",
        "content": [{"type": "thinking", "thinking": text}],
    })


def _assistant_tool_use(name):
    return _line("assistant", message={
        "role": "assistant",
        "content": [{"type": "tool_use", "name": name, "input": {}}],
    })


def _tool_result():
    return _line("user", message={
        "role": "user",
        "content": [{"type": "tool_result", "content": "file contents"}],
    })


def _turn_duration():
    return _line("system", subtype="turn_duration", durationMs=123)


def test_extract_last_turn_basic():
    lines = [
        _user("How do I deploy?"),
        _assistant_thinking("internal reasoning"),
        _assistant_tool_use("Read"),
        _tool_result(),
        _assistant_text("Run build then push."),
        _turn_duration(),
    ]
    user, assistant = hooks._extract_last_turn(lines)
    assert user == "How do I deploy?"
    assert assistant == "Run build then push."


def test_extract_last_turn_skips_thinking_and_tool_use():
    lines = [
        _user("question"),
        _assistant_thinking("should not appear"),
        _assistant_tool_use("Bash"),
        _assistant_text("answer"),
        _turn_duration(),
    ]
    _, assistant = hooks._extract_last_turn(lines)
    assert "should not appear" not in assistant
    assert "answer" in assistant


def test_extract_last_turn_only_last_turn():
    lines = [
        _user("first question"),
        _assistant_text("first answer"),
        _turn_duration(),
        _user("second question"),
        _assistant_text("second answer"),
        _turn_duration(),
    ]
    user, assistant = hooks._extract_last_turn(lines)
    assert user == "second question"
    assert assistant == "second answer"


def test_extract_last_turn_skips_tool_result_as_user():
    lines = [
        _user("real question"),
        _tool_result(),
        _assistant_text("answer"),
        _turn_duration(),
    ]
    user, _ = hooks._extract_last_turn(lines)
    assert user == "real question"


def test_parse_jsonl_skips_malformed(tmp_path):
    f = tmp_path / "t.jsonl"
    f.write_text('{"type":"user"}\nNOT JSON\n{"type":"system"}\n')
    out = hooks._parse_jsonl(f)
    assert len(out) == 2


def test_read_transcript_with_retry_returns_complete(tmp_path):
    f = tmp_path / "t.jsonl"
    lines = [_user("q"), _assistant_text("a"), _turn_duration()]
    f.write_text("\n".join(json.dumps(x) for x in lines))
    out = hooks._read_transcript_with_retry(f)
    assert (out[-1]["type"], out[-1]["subtype"]) == ("system", "turn_duration")


# ── Handlers never raise ──


def test_dispatch_unknown_event():
    assert hooks.dispatch("nonexistent", {}) == {"continue": True}


def test_run_hook_bad_json():
    # Malformed stdin must not raise.
    assert hooks.run_hook("session-start", "{not json") == {"continue": True} or \
        "continue" in hooks.run_hook("session-start", "{not json")


def test_handle_user_prompt_submit_short_prompt_skipped():
    # Prompts under 3 words are skipped without touching the store.
    assert hooks.handle_user_prompt_submit({"prompt": "hi"}) == {"continue": True}


def test_handle_stop_no_transcript():
    assert hooks.handle_stop({}) == {"continue": True}


def test_handle_session_end_no_events(tmp_path, monkeypatch):
    monkeypatch.setenv("METAMEM_DATA_DIR", str(tmp_path))
    result = hooks.handle_session_end({"session_id": "test123", "cwd": str(tmp_path)})
    assert result.get("continue") is True


def test_safe_session_id_sanitizes():
    assert hooks._safe_session_id("abc/../etc") == "abc_.._etc"
    assert hooks._safe_session_id("") == ""


# ── End-to-end: stop captures, session-end finalizes ──


def test_stop_then_session_end_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("METAMEM_DATA_DIR", str(tmp_path))
    transcript = tmp_path / "transcript.jsonl"
    lines = [
        _user("How do I configure logging?"),
        _assistant_text("Set LOG_LEVEL in the env."),
        _turn_duration(),
    ]
    transcript.write_text("\n".join(json.dumps(x) for x in lines))

    payload = {
        "session_id": "roundtrip1",
        "cwd": str(tmp_path),
        "transcript_path": str(transcript),
    }

    stop_result = hooks.handle_stop(payload)
    assert "turn saved" in stop_result.get("systemMessage", "")

    end_result = hooks.handle_session_end(payload)
    assert "finalized" in end_result.get("systemMessage", "")
