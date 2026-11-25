"""AI agents: other AI apps (Claude, Cursor, …) using Babble for you through MCP (babble_mcp.py, mcp_server.py).

The MCP server is a small program the AI app starts; it talks to this engine like the app's window does, with this
run's key, and says who it is (the X-Babble-Client header). The engine then:
- refuses it unless the owner turned AI agents on (Settings → General, settings agents_on);
- keeps keys, sign-ins and the spending limits the owner's own (BLOCKED, LOCKED_SETTINGS);
- marks the work it starts (VIA, Task.via), so paid work counts against the agents' own day limit
  (settings agent_spend_day_limit, $0 at first: spend.allow refuses it) as well as the owner's.

A program running as the owner can read the key and leave the header out: this keeps an AI agent the owner set up
within what they allowed, it isn't a wall against other programs (guard.py is that).
"""
import contextvars
import re
import sys
from pathlib import Path

from . import settings

HEADER = b"x-babble-client"
VIA = contextvars.ContextVar("agent", default=None)  # "mcp" while an agent's request (and the work it starts) runs

# what an agent may not do, whatever the method: keys, sign-ins, the spending limits, and turning agents on or off
BLOCKED = re.compile(
    r"^/api/(?:spending/limit|keychain|director/(?:key|forget)|sounds/online/[^/]+/key"
    r"|signing|connections/(?:kaggle/(?:signin|token|signout)|modal/(?:signin|signout)|ssh|orpheus|huggingface|[^/]+/key))$")
SETUP_ONLY = {"/api/agents", "/api/voices-setup"}  # an agent may read these, not change them (setting up voices installs software)
LOCKED_SETTINGS = {"agents_on", "agent_spend_day_limit", "spend_day_limit", "orpheus_url", "orpheus_key",
                   "pack_catalog_url", "pack_signer_github"}
ROOT = Path(__file__).resolve().parent.parent


def on():
    return bool(settings.load().get("agents_on"))


def current():
    """The agent the running code works for (its request, or the task it started), else None."""
    from . import tasks
    return VIA.get() or getattr(tasks.CURRENT.get(), "via", None)


def refused(path, method):
    """Why an agent's request is refused (None: it may go on)."""
    if not on():
        return "AI agents are turned off in Babble. Turn them on in Settings → General."
    if BLOCKED.match(path) or (path in SETUP_ONLY and method != "GET"):
        return "Only you can do that in Babble (keys, sign-ins, spending limits and the AI agents switch)."
    return None


def locked_setting(name):
    return name in LOCKED_SETTINGS or name.startswith("ssh_") or settings.is_secret(name)


def setup():
    """What an AI app needs to start the MCP server (Settings → General shows it to copy)."""
    if sys.platform == "win32":
        python = str(ROOT / ".venv" / "Scripts" / "python.exe")
    else:
        python = str(ROOT / ".venv" / "bin" / "python")
    if not Path(python).exists():
        python = sys.executable
    script = str(ROOT / "babble_mcp.py")
    cmd_str = (f'claude mcp add babble -- "{python}" "{script}"' if sys.platform == "win32"
               else f"claude mcp add babble -- '{python}' '{script}'")
    return {"command": python, "args": [script],
            "json": {"mcpServers": {"babble": {"command": python, "args": [script]}}},
            "claude_code": cmd_str}
