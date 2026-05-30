"""MetaMem CLI — Install, manage, and query memory from the command line.

Usage:
    metamem install          # Register MCP server with Claude Code
    metamem search <query>   # Search memories
    metamem store <content>  # Store a memory
    metamem stats            # Show store statistics
    metamem serve            # Start MCP server (stdio)
    metamem benchmark        # Run benchmarks
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import click


@click.group()
@click.version_option(version="0.1.0")
def main():
    """MetaMem — Unified lifelong memory for LLM agents."""
    pass


CLAUDE_MD_MEMORY_SECTION = """\

## Memory (MetaMem)

You have persistent project memory via MCP tools. **These are mandatory behaviors, not optional.**

### At Session Start (ALWAYS do this first)
- Call `mem_context` to load previous work context BEFORE responding to the first user message.
- This gives you continuity from previous sessions (what was done, what's pending, warnings).

### During Session
- Call `mem_search` before starting tasks to check for relevant knowledge, procedures, or warnings.
- Call `mem_event(event_type, content)` to record important observations, decisions, and task outcomes.
- When you learn something new about the project, call `mem_store` to save it.

### After Completing or Failing a Task
- Call `mem_feedback(description, memories_used, status)` to report the outcome.
  - "success" → memories that helped get confidence boost
  - "failure" → memories that misled get corrected, failure case is created
  - "partial" → caveats are added
- This is how the memory system learns and improves. NEVER skip this step.

### When User Says "Remember..." or States a Preference
- Call `mem_instruct(rule, scope)` to save the rule/preference permanently.

### Rules
- ALWAYS call `mem_context` at session start — this is non-negotiable.
- ALWAYS call `mem_search` before deployments, debugging, or infrastructure changes.
- ALWAYS call `mem_feedback` after task completion — even for small tasks.
- When you encounter an error, `mem_search` for similar past failures before debugging from scratch.
"""


def _safe_load_json(path: Path) -> dict:
    """Load JSON file, returning {} if missing or malformed."""
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return data
    except (json.JSONDecodeError, OSError):
        # Backup corrupted file and start fresh
        backup = path.with_suffix(".mcp.json.bak")
        path.rename(backup)
        click.echo(f"  ⚠ Corrupted {path.name} backed up to {backup.name}")
        return {}


def _verify_server_starts() -> bool:
    """Quick sanity check that the MCP server module can be imported."""
    result = subprocess.run(
        [sys.executable, "-c", "from metamem.mcp_server import serve; print('OK')"],
        capture_output=True, text=True, timeout=30,
    )
    return result.returncode == 0 and "OK" in result.stdout


def _register_with_claude_cli(scope: str, data_dir: str) -> bool:
    """Register MetaMem via Claude Code's native `claude mcp add` command.

    Returns True if registration succeeded (or the server already exists),
    False if the `claude` CLI is unavailable or the command failed — in which
    case the caller should fall back to writing config files directly.
    """
    claude_bin = shutil.which("claude")
    if not claude_bin:
        return False

    # Remove any stale entry first so re-running install is idempotent.
    subprocess.run(
        [claude_bin, "mcp", "remove", "--scope", scope, "metamem"],
        capture_output=True, text=True,
    )

    # claude mcp add <name> -e KEY=val --scope <scope> -- <command> [args...]
    cmd = [
        claude_bin, "mcp", "add", "metamem",
        "-e", f"METAMEM_DATA_DIR={data_dir}",
        "--scope", scope,
        "--", sys.executable, "-m", "metamem.mcp_server",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return True

    click.echo(f"  ⚠ `claude mcp add` failed: {result.stderr.strip() or result.stdout.strip()}")
    return False


@main.command()
@click.option("--project-dir", default=None, help="Project directory to add CLAUDE.md to (default: cwd)")
@click.option("--project-only", is_flag=True, help="Only install in project dir, skip global")
def install(project_dir: str | None, project_only: bool):
    """Register MetaMem MCP server with Claude Code + inject CLAUDE.md instructions.

    Installs globally (~/.claude/.mcp.json) so it works in ALL projects.
    Also adds CLAUDE.md instructions to the current project directory.
    """
    data_dir = os.path.expanduser("~/.metamem")
    mcp_entry = {
        "command": sys.executable,
        "args": ["-m", "metamem.mcp_server"],
        "env": {
            "METAMEM_DATA_DIR": data_dir,
        },
    }

    # ── 1. Global MCP registration — works for all projects ──
    if not project_only:
        # Prefer Claude Code's native CLI so it owns config + approval lifecycle.
        if _register_with_claude_cli("user", data_dir):
            click.echo("✓ MCP server registered via `claude mcp add` (scope: user)")
            click.echo("  → MetaMem will be available in ALL projects (no per-project setup needed)")
        else:
            # Fallback: write Claude config files directly.
            claude_dir = Path.home() / ".claude"
            claude_dir.mkdir(parents=True, exist_ok=True)
            global_mcp = claude_dir / ".mcp.json"

            global_config = _safe_load_json(global_mcp)

            if "mcpServers" not in global_config:
                global_config["mcpServers"] = {}
            global_config["mcpServers"]["metamem"] = mcp_entry
            with open(global_mcp, "w") as f:
                json.dump(global_config, f, indent=2)

            click.echo(f"✓ MCP server registered globally in {global_mcp}")
            click.echo("  → MetaMem will be available in ALL projects (no per-project setup needed)")

            # Also update claude_desktop_config.json for Claude Desktop app compatibility
            desktop_config_file = claude_dir / "claude_desktop_config.json"
            desktop_config = _safe_load_json(desktop_config_file)
            if "mcpServers" not in desktop_config:
                desktop_config["mcpServers"] = {}
            desktop_config["mcpServers"]["metamem"] = mcp_entry
            with open(desktop_config_file, "w") as f:
                json.dump(desktop_config, f, indent=2)

    # ── 2. Project-level registration (for project-specific overrides) ──
    target_dir = Path(project_dir) if project_dir else Path.cwd()
    if project_only:
        if _register_with_claude_cli("project", data_dir):
            click.echo("✓ MCP server registered via `claude mcp add` (scope: project)")
        else:
            mcp_json_path = target_dir / ".mcp.json"
            proj_config = _safe_load_json(mcp_json_path)
            if "mcpServers" not in proj_config:
                proj_config["mcpServers"] = {}
            proj_config["mcpServers"]["metamem"] = mcp_entry
            with open(mcp_json_path, "w") as f:
                json.dump(proj_config, f, indent=2)
            click.echo(f"✓ MCP server registered in {mcp_json_path}")

    # ── 3. Inject CLAUDE.md instructions ──
    if not project_only:
        claude_md = target_dir / "CLAUDE.md"

        if claude_md.exists():
            existing = claude_md.read_text()
            if "## Memory (MetaMem)" in existing:
                click.echo(f"✓ CLAUDE.md already has MetaMem section ({claude_md})")
            else:
                # Append to existing CLAUDE.md
                with open(claude_md, "a") as f:
                    f.write("\n" + CLAUDE_MD_MEMORY_SECTION)
                click.echo(f"✓ Appended memory instructions to {claude_md}")
        else:
            # Create new CLAUDE.md
            claude_md.write_text(CLAUDE_MD_MEMORY_SECTION)
            click.echo(f"✓ Created {claude_md} with memory instructions")

    # ── 4. Global CLAUDE.md (optional backup) ──
    claude_dir = Path.home() / ".claude"
    global_claude_md = claude_dir / "CLAUDE.md"
    if not global_claude_md.exists():
        global_claude_md.write_text(CLAUDE_MD_MEMORY_SECTION)
        click.echo(f"✓ Created global {global_claude_md}")
    elif "## Memory (MetaMem)" not in global_claude_md.read_text():
        with open(global_claude_md, "a") as f:
            f.write("\n" + CLAUDE_MD_MEMORY_SECTION)
        click.echo(f"✓ Updated global {global_claude_md}")

    # ── 5. Verify server starts correctly ──
    click.echo()
    click.echo("  Verifying MCP server...")
    if _verify_server_starts():
        click.echo("  ✓ MCP server verified — starts successfully")
    else:
        click.echo("  ⚠ MCP server failed to start. Try running:")
        click.echo(f"    {sys.executable} -m metamem.mcp_server")
        click.echo("  to see the error details.")
        return

    click.echo()
    click.echo("  ✅ Installation complete! Restart Claude Code to activate.")
    click.echo()
    click.echo("  Next step — verify + approve in Claude Code:")
    click.echo("    claude mcp list")
    click.echo("    → If MetaMem shows \"Pending approval\", launch `claude` and approve it.")
    click.echo()
    click.echo("  What happens now:")
    click.echo("    1. Claude sees CLAUDE.md → knows to use memory tools")
    click.echo("    2. At session start → auto-calls mem_context (previous work)")
    click.echo("    3. During work → searches/stores/tracks events")
    click.echo("    4. After tasks → reports results (evolution feedback)")
    click.echo()
    click.echo("  Memory tools:")
    click.echo("    • mem_context  — Load previous session context (auto at start)")
    click.echo("    • mem_search   — Search memory index")
    click.echo("    • mem_get      — Full memory details")
    click.echo("    • mem_store    — Store a typed memory")
    click.echo("    • mem_instruct — Save a preference")
    click.echo("    • mem_feedback — Report task results for evolution")
    click.echo("    • mem_event    — Track session events")


@main.command()
@click.argument("query")
@click.option("--type", "-t", "mem_type", default=None,
              type=click.Choice(["episodic", "semantic", "procedural", "failure", "instruction"]))
@click.option("--limit", "-n", default=10)
def search(query: str, mem_type: str | None, limit: int):
    """Search memory store."""
    from .mcp_server import mem_search
    result = mem_search(query=query, type=mem_type, limit=limit)
    for item in result["results"]:
        conf = item["confidence"]
        click.echo(f"  [{item['type'][:4]}|{conf:.2f}] {item['id']}: {item['summary']}")
    click.echo(f"\n  Total matches: {result['total']}")


@main.command()
@click.argument("content")
@click.option("--type", "-t", "mem_type", default="semantic",
              type=click.Choice(["episodic", "semantic", "procedural", "failure", "instruction"]))
@click.option("--summary", "-s", default="")
def store(content: str, mem_type: str, summary: str):
    """Store a new memory."""
    from .mcp_server import mem_store
    result = mem_store(content=content, type=mem_type, summary=summary)
    click.echo(f"✓ Stored memory {result['id']} (type={result['type']})")


@main.command()
@click.argument("rule")
@click.option("--scope", default="global")
def instruct(rule: str, scope: str):
    """Save a user preference/instruction."""
    from .mcp_server import mem_instruct
    result = mem_instruct(rule=rule, scope=scope)
    click.echo(f"✓ Instruction stored: {result['id']} (scope={result['scope']})")


@main.command()
def stats():
    """Show memory store statistics."""
    from .mcp_server import mem_stats
    result = mem_stats()
    s = result["store"]
    click.echo("Memory Store Statistics")
    click.echo("=" * 40)
    click.echo(f"  Total memories:    {s['total']}")
    click.echo(f"  Active:            {s['active']}")
    click.echo(f"  Avg confidence:    {s['avg_confidence']}")
    click.echo(f"  Embeddings:        {s['embeddings']}")
    click.echo()
    click.echo("  By type:")
    for t, count in s["by_type"].items():
        click.echo(f"    {t:12s}: {count}")
    click.echo()
    e = result["evolution"]
    click.echo(f"  Evolution actions: {e['total_actions']}")
    if e.get("actions_by_type"):
        for a, count in e["actions_by_type"].items():
            click.echo(f"    {a:12s}: {count}")


@main.command()
def serve():
    """Start MCP server (stdio transport)."""
    from .mcp_server import serve as _serve
    _serve()


@main.command()
@click.argument("benchmark", type=click.Choice(["locomo", "membench", "longmemeval"]))
@click.option("--data", default=None, help="Data path")
@click.option("--max-rounds", default=5, help="Max evolution rounds")
@click.option("--initial", default="weak", type=click.Choice(["weak", "strong"]))
@click.option("--sample", default=0, type=int, help="Sample index (LoCoMo)")
def benchmark(benchmark: str, data: str | None, max_rounds: int, initial: str, sample: int):
    """Run EvolveMem-compatible benchmarks with MetaMem retrieval."""
    from .benchmarks.runner import run_benchmark
    run_benchmark(
        benchmark_name=benchmark,
        data_path=data,
        max_rounds=max_rounds,
        initial=initial,
        sample_index=sample,
    )


if __name__ == "__main__":
    main()
