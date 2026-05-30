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


@main.command()
@click.option("--claude-config", default=None, help="Path to Claude config directory")
@click.option("--project-dir", default=None, help="Project directory to add CLAUDE.md to (default: cwd)")
@click.option("--global-only", is_flag=True, help="Only install globally, skip project CLAUDE.md")
def install(claude_config: str | None, project_dir: str | None, global_only: bool):
    """Register MetaMem MCP server with Claude Code + inject CLAUDE.md instructions."""
    # ── 1. Register MCP Server ──
    if claude_config:
        config_dir = Path(claude_config)
    else:
        config_dir = Path.home() / ".claude"

    config_file = config_dir / "claude_desktop_config.json"
    if not config_dir.exists():
        config_dir.mkdir(parents=True, exist_ok=True)

    # Load or create config
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
    else:
        config = {}

    if "mcpServers" not in config:
        config["mcpServers"] = {}

    # Add metamem server
    config["mcpServers"]["metamem"] = {
        "command": sys.executable,
        "args": ["-m", "metamem.mcp_server"],
        "env": {
            "METAMEM_DATA_DIR": os.path.expanduser("~/.metamem"),
        },
    }

    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    click.echo(f"✓ MCP server registered in {config_file}")

    # ── 2. Inject CLAUDE.md instructions ──
    if not global_only:
        target_dir = Path(project_dir) if project_dir else Path.cwd()
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

    # ── 3. Global CLAUDE.md (optional backup) ──
    global_claude_md = config_dir / "CLAUDE.md"
    if not global_claude_md.exists():
        global_claude_md.write_text(CLAUDE_MD_MEMORY_SECTION)
        click.echo(f"✓ Created global {global_claude_md}")
    elif "## Memory (MetaMem)" not in global_claude_md.read_text():
        with open(global_claude_md, "a") as f:
            f.write("\n" + CLAUDE_MD_MEMORY_SECTION)
        click.echo(f"✓ Updated global {global_claude_md}")

    click.echo()
    click.echo("  ✅ Installation complete! Restart Claude Code to activate.")
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
