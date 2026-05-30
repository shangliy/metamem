"""Tests for token usage tracking (metamem/usage.py)."""

from metamem import usage


def _assistant_with_usage(in_t, out_t, cache_read=0, cache_create=0):
    return {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "answer"}],
            "usage": {
                "input_tokens": in_t,
                "output_tokens": out_t,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_create,
            },
        },
    }


def _assistant_no_usage():
    return {
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
    }


# ── extract_usage ──


def test_extract_usage_present():
    lines = [_assistant_with_usage(100, 50, cache_read=80, cache_create=10)]
    u = usage.extract_usage(lines)
    assert u == {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 80,
        "cache_creation_input_tokens": 10,
    }


def test_extract_usage_absent_returns_none():
    lines = [_assistant_no_usage()]
    assert usage.extract_usage(lines) is None


def test_extract_usage_picks_last_with_usage():
    lines = [
        _assistant_with_usage(10, 5),
        _assistant_no_usage(),
        _assistant_with_usage(200, 80),
    ]
    u = usage.extract_usage(lines)
    assert u["input_tokens"] == 200
    assert u["output_tokens"] == 80


def test_extract_usage_missing_fields_default_zero():
    lines = [{"type": "assistant", "message": {"role": "assistant", "usage": {"input_tokens": 42}}}]
    u = usage.extract_usage(lines)
    assert u["input_tokens"] == 42
    assert u["output_tokens"] == 0
    assert u["cache_read_input_tokens"] == 0


# ── record / load roundtrip ──


def test_record_and_load_roundtrip(tmp_path):
    data_dir = str(tmp_path)
    rec = usage.build_record("sess1", "proj1", {
        "input_tokens": 100, "output_tokens": 50,
        "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
    }, ts=123.0)
    usage.record_usage(data_dir, rec)
    usage.record_usage(data_dir, usage.build_record("sess1", "proj1", {
        "input_tokens": 200, "output_tokens": 60,
        "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
    }, ts=124.0))

    loaded = usage.load_usage(data_dir)
    assert len(loaded) == 2
    assert loaded[0]["session_id"] == "sess1"
    assert loaded[0]["input_tokens"] == 100


def test_load_usage_missing_file_returns_empty(tmp_path):
    assert usage.load_usage(str(tmp_path)) == []


def test_load_usage_skips_malformed(tmp_path):
    path = tmp_path / "usage" / "token_usage.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"input_tokens": 1}\nNOT JSON\n{"input_tokens": 2}\n')
    loaded = usage.load_usage(str(tmp_path))
    assert len(loaded) == 2


# ── summarize ──


def test_summarize_totals_and_breakdowns():
    records = [
        usage.build_record("s1", "p1", {"input_tokens": 100, "output_tokens": 50,
                                          "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}),
        usage.build_record("s1", "p1", {"input_tokens": 100, "output_tokens": 30,
                                          "cache_read_input_tokens": 100, "cache_creation_input_tokens": 0}),
        usage.build_record("s2", "p2", {"input_tokens": 200, "output_tokens": 80,
                                          "cache_read_input_tokens": 0, "cache_creation_input_tokens": 20}),
    ]
    summary = usage.summarize(records)
    assert summary["turns"] == 3
    assert summary["totals"]["input_tokens"] == 400
    assert summary["totals"]["output_tokens"] == 160
    assert summary["by_session"]["s1"]["turns"] == 2
    assert summary["by_project"]["p2"]["input_tokens"] == 200
    # cache hit ratio = cache_read / (input + cache_read) = 100 / (400 + 100)
    assert summary["cache_hit_ratio"] == 0.2


def test_summarize_empty():
    summary = usage.summarize([])
    assert summary["turns"] == 0
    assert summary["totals"]["input_tokens"] == 0
    assert summary["cache_hit_ratio"] == 0.0
