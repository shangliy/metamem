# Feature Plan: Automatic Memory Capture via Claude Code Hooks

- **Date:** 2026-05-30
- **Status:** POC / Proposal (awaiting confirmation)
- **Author:** shangliy (with AdaL)

## TL;DR

Today MetaMem only updates memory if Claude *chooses* to follow the `CLAUDE.md`
instructions (`mem_context`, `mem_search`, `mem_event`, `mem_feedback`). This is
best-effort and silently skips when the model forgets.

This plan adds a **Claude Code hooks layer** (the EverOS pattern) so memory
capture becomes **deterministic** — it fires on lifecycle events regardless of
whether the model "remembers". We reuse the existing MCP/session functions, so
this is additive and low-risk.

## Why hooks (vs. prompt instructions)

| | Prompt instructions (today) | Hooks (proposed) |
|---|---|---|
| Trigger | Model decides | Claude Code lifecycle event |
| Reliability | Best-effort | Deterministic |
| Latency cost | Model tokens | Local subprocess |
| Failure mode | Silent skip | `continue: true` fallback |

Hooks are shell commands Claude Code runs at events, piping JSON via stdin and
reading JSON from stdout. They run "whether or not the model remembers the rule."

## Design principles

1. **Reuse, don't rewrite.** Hooks call the *existing* session/store layer
   (`SessionManager.add_event`, `get_context_injection`, `mem_search`,
   `mem_feedback`). No new memory logic.
2. **One entrypoint, not N scripts.** EverOS ships separate `.js` files. We are
   a Python package — expose a single hidden CLI: `metamem hook <event>` that
   reads stdin JSON and writes stdout JSON. Cleaner, no Node dependency.
3. **Local-first.** No cloud API (unlike EverOS). All data stays in `~/.metamem`.
4. **Never block.** Every hook returns `continue: true` even on error, matching
   EverOS's graceful-degradation rule.

## Hook → MetaMem mapping

| Claude event | Hook action | MetaMem call |
|---|---|---|
| `SessionStart` | Inject prior context as `systemPrompt` | `get_context_injection()` |
| `UserPromptSubmit` | Search memory for the prompt, inject hits | `mem_search(prompt)` |
| `Stop` | Record the completed turn from transcript | `add_event("message", ...)` |
| `SessionEnd` | Finalize + summarize the session | `SessionManager.finalize()` |

### Hook I/O contract (from Claude Code docs + EverOS)

- **stdin:** `{ session_id, cwd, transcript_path?, prompt?, hook_event_name }`
- **stdout (SessionStart/UserPromptSubmit):**
  ```json
  { "continue": true, "systemMessage": "...", "systemPrompt": "<context>...</context>" }
  ```
- **stdout (Stop/SessionEnd):** `{ "systemMessage": "..." }`

### Transcript parsing (Stop hook)

Claude writes a JSONL transcript. Per EverOS findings:
- A turn ends at a `system / turn_duration` line (the **only** turn boundary).
- The Stop hook may fire *before* `turn_duration` is written → **retry read**
  (max 5 attempts, 100ms) until the last line is `turn_duration`.
- Collect the user's original text (skip `tool_result`) + all assistant `text`
  blocks (skip `thinking` / `tool_use`), merge with `\n\n`.

## Proposed changes (files)

1. **`metamem/hooks.py`** (new)
   - `handle_session_start(payload) -> dict`
   - `handle_user_prompt_submit(payload) -> dict`
   - `handle_stop(payload) -> dict`
   - `handle_session_end(payload) -> dict`
   - `_read_transcript_with_retry(path)` + `_extract_last_turn(lines)` helpers
   - All wrapped so any exception → `{"continue": true}`.

2. **`metamem/cli.py`** (edit)
   - Add a hidden group `hook` with subcommands `session-start`,
     `user-prompt-submit`, `stop`, `session-end`. Each reads `sys.stdin`,
     dispatches to `hooks.py`, prints JSON to stdout.
   - Extend `install` to write the hooks block into `~/.claude/settings.json`
     (user scope) — additive merge, never clobbering existing hooks.
   - Add `--no-hooks` flag to `install` for users who want MCP-only.

3. **`tests/test_hooks.py`** (new)
   - Unit-test each handler with synthetic stdin payloads.
   - Test transcript parsing with a fixture JSONL (turn boundary, retry,
     text/thinking/tool_use filtering).
   - Test that handlers never raise (return `continue: true` on bad input).

4. **`README.md`** (edit)
   - New "Automatic capture (hooks)" subsection explaining what fires when.

## What we are NOT doing (scope control)

- No cloud API / Memory Hub dashboard (EverOS-specific).
- No `PreToolUse`/`PostToolUse` guardrails (out of scope for memory).
- No change to the evolution engine or store schema.

## Settings.json hooks block (shape)

```json
{
  "hooks": {
    "SessionStart": [{ "hooks": [{ "type": "command",
      "command": "<python> -m metamem hook session-start", "timeout": 15 }] }],
    "UserPromptSubmit": [{ "hooks": [{ "type": "command",
      "command": "<python> -m metamem hook user-prompt-submit", "timeout": 15 }] }],
    "Stop": [{ "hooks": [{ "type": "command",
      "command": "<python> -m metamem hook stop", "timeout": 20 }] }],
    "SessionEnd": [{ "hooks": [{ "type": "command",
      "command": "<python> -m metamem hook session-end", "timeout": 15 }] }]
  }
}
```

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Hook latency on every prompt | Reuse warm session; keep search `limit` small; 15s timeout |
| Transcript race (turn not flushed) | Retry-read until `turn_duration` |
| Clobbering user's existing hooks | Additive merge into `settings.json`, dedup by command |
| Double capture (hooks + CLAUDE.md) | Trim CLAUDE.md mandatory rules to avoid redundant `mem_event` |
| `sentence-transformers` import cost per hook | Lazy-init; SessionStart/Stop tolerate cold start within timeout |

## Verification plan

- `pytest tests/ -q` (existing 21 + new hook tests)
- Manual: `echo '{"prompt":"deploy"}' | python -m metamem hook user-prompt-submit`
- Manual: `echo '{"transcript_path":"<fixture>.jsonl"}' | python -m metamem hook stop`
- `claude mcp list` + confirm hooks appear in `~/.claude/settings.json`

## Open questions for you

1. **Capture granularity on `Stop`:** store every turn as an `episodic`/`message`
   event (verbose, full history) — or only summarize at `SessionEnd` (cheaper,
   less granular)? I lean **both**: lightweight `mem_event` per turn + a
   `finalize()` summary at session end.
2. **Hooks scope:** user-wide (all projects) or project-only? Default **user**,
   matching the MCP install.
3. **Trim `CLAUDE.md`?** Once hooks auto-capture, the mandatory `mem_event`
   instruction becomes redundant. Keep, or slim it down to avoid double writes?
