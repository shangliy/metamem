"""Allow running the MetaMem CLI via `python -m mem_engram`.

For the raw MCP server (stdio), use `python -m mem_engram.mcp_server` or `metamem serve`.
"""
from .cli import main

if __name__ == "__main__":
    main()
