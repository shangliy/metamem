"""Allow running the MetaMem CLI via `python -m metamem`.

For the raw MCP server (stdio), use `python -m metamem.mcp_server` or `metamem serve`.
"""
from .cli import main

if __name__ == "__main__":
    main()
