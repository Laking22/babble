"""Babble for AI agents (MCP over stdio):  .venv/bin/python babble_mcp.py

An AI app (Claude, Cursor, …) starts this to use Babble for you; Babble must be open, with AI agents turned on in
Settings → General. See narrator/mcp_server.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from narrator.mcp_server import main  # noqa: E402

if __name__ == "__main__":
    main()
