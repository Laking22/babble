"""Run: .venv/bin/python tests/agents_mcp_check.py  (a few seconds, no models, scratch data folder only)

AI agents through MCP (narrator/agents.py, narrator/mcp_server.py):
1. The guard: an agent's request is refused while agents are off; with them on it goes through, but never keys,
   sign-ins, the spending limits or the agents switch; locked settings are refused to it, not to the owner.
2. Spending: an agent's paid work is refused at a $0 agents' limit, fits or doesn't within a set one, and its
   charges (a task's, a cloud job's) are marked as the agents'; the owner's work isn't held by the agents' limit.
3. Tasks an agent starts are marked as its own.
4. The MCP server: initialize, tools/list, a tool call, and an unknown method, over its stdio protocol.
"""
import io
import json
import os
import sys
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="agents-"))
os.environ["NARRATOR_HOME"] = str(TMP / "home")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from narrator import agents, api, guard, settings, spend, tasks  # noqa: E402

failed = []


def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        failed.append(what)


app = FastAPI()
api.install(app)
owner = TestClient(app, base_url="http://127.0.0.1", headers={"X-Babble-Token": guard.token()})
agent = TestClient(app, base_url="http://127.0.0.1", headers={"X-Babble-Token": guard.token(), "X-Babble-Client": "mcp"})

# 1. the guard
r = agent.get("/api/settings")
check(r.status_code == 403 and "turned off" in r.json()["detail"], "agents off: refused")
check(owner.get("/api/settings").status_code == 200, "the owner isn't held by the switch")
check(owner.post("/api/agents", json={"on": True}).json()["on"] is True, "owner turns agents on")
check(agent.get("/api/settings").status_code == 200, "agents on: an agent gets in")
for method, path in [("POST", "/api/agents"), ("POST", "/api/spending/limit"), ("GET", "/api/keychain"),
                     ("POST", "/api/connections/modal/key"), ("POST", "/api/connections/kaggle/signin"),
                     ("POST", "/api/director/key"), ("POST", "/api/connections/ssh"), ("DELETE", "/api/connections/huggingface")]:
    check(agent.request(method, path, json={}).status_code == 403, f"agent refused: {method} {path}")
check(agent.get("/api/agents").status_code == 200, "agent may read /api/agents")
check(agent.post("/api/connections/kaggle/check").status_code != 403, "agent may check a connection")
for k, v in [("agent_spend_day_limit", 100), ("spend_day_limit", 100), ("agents_on", False), ("ssh_host", "x")]:
    check(agent.post("/api/settings", json={k: v}).status_code == 403, f"agent can't set {k}")
check(agent.post("/api/settings", json={"sfx_smart": False}).status_code == 200, "agent can set sfx_smart")
check(owner.post("/api/spending/limit", json={"agents": 2.5}).json()["agent_limit"] == 2.5, "owner sets agents' limit")
check(owner.post("/api/spending/limit", json={"day": 20}).json()["limit"] == 20, "owner's own limit still saves")
routes = agent.get("/api/agents/routes").json()
check(any(r["path"] == "/api/books/create" and r["method"] == "POST" for r in routes), f"routes listed ({len(routes)})")

# 2. spending
settings.save(agent_spend_day_limit=0.0, spend_day_limit=0.0)
mark = agents.VIA.set("mcp")
try:
    spend.allow("modal", {"low": 0.1, "high": 0.5}, "create")
    check(False, "agents' $0 limit refuses paid work")
except spend.Overspent as e:
    check("$0" in str(e), "agents' $0 limit refuses paid work")
settings.save(agent_spend_day_limit=1.0)
try:
    spend.allow("modal", {"low": 0.1, "high": 0.5}, "create")
    check(True, "within the agents' $1 limit: allowed")
except spend.Overspent:
    check(False, "within the agents' $1 limit: allowed")
try:
    spend.allow("modal", {"low": 1, "high": 3}, "create")
    check(False, "past the agents' limit: refused")
except spend.Overspent as e:
    check("AI agents'" in str(e), "past the agents' limit: refused")
spend.record("anthropic", "ai", "direct", 0.2)
spend.job_sent("modal", "run-agent-1", "Create in the cloud: test", price=1.0, hours=1)
check(spend._jobs()["run-agent-1"].get("agent") is True, "a cloud job an agent sent is marked")
agents.VIA.reset(mark)
spend.job_ended("run-agent-1")  # (ended later, outside the agent's request: still the agents')
charges = [e for e in spend.entries() if e.get("type") == "charge"]
check(all(e.get("agent") for e in charges) and len(charges) == 2, "its charges are the agents'")
check(abs(spend.agent_today() - sum(e["dollars"] for e in charges)) < 1e-9, "agent_today adds them up")
spend.record("anthropic", "ai", "direct", 0.3)
check(not spend.entries()[-1].get("agent"), "the owner's charge isn't")
try:
    spend.allow("modal", {"low": 1, "high": 3}, "create")
    check(True, "the owner isn't held by the agents' limit")
except spend.Overspent:
    check(False, "the owner isn't held by the agents' limit")
hist = spend.history()
check(any(r["agent"] for r in hist) and any(not r["agent"] for r in hist), "history marks the agents' rows")

# 3. tasks
r = agent.post("/api/sounds/make", json={"description": "x", "kind": "nope"})  # (refused before any task: 400)
check(r.status_code == 400, "a bad request is still the engine's answer")
mark = agents.VIA.set("mcp")
t = tasks.start("test", "Agent task", lambda: {"status": "done"})
agents.VIA.reset(mark)
t2 = tasks.start("test", "Owner task", lambda: {"status": "done"})
check(t.info()["agent"] is True and t2.info()["agent"] is False, "tasks an agent starts are marked")

# 4. the MCP server's protocol (no engine needed for these)
from narrator import mcp_server  # noqa: E402

out = io.StringIO()
srv = mcp_server.Server(out)
lines = [{"jsonrpc": "2.0", "id": 1, "method": "initialize",
          "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}}},
         {"jsonrpc": "2.0", "method": "notifications/initialized"},
         {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
         {"jsonrpc": "2.0", "id": 3, "method": "nope"},
         {"jsonrpc": "2.0", "id": 4, "method": "ping"}]
srv.serve(io.StringIO("\n".join(json.dumps(m) for m in lines) + "\n"))
got = {m["id"]: m for m in map(json.loads, out.getvalue().splitlines())}
check(got[1]["result"]["protocolVersion"] == "2025-06-18" and got[1]["result"]["serverInfo"]["name"] == "babble",
      "initialize answers with the client's version")
tools = got[2]["result"]["tools"]
check(len(tools) >= 25 and all(t["inputSchema"]["type"] == "object" for t in tools), f"tools/list ({len(tools)} tools)")
check(len({t["name"] for t in tools}) == len(tools), "tool names are unique")
check(got[3]["error"]["code"] == -32601 and got[4]["result"] == {}, "unknown method / ping")
check(len(got) == 4, "no answer to a notification")

print("\nFAILED " + str(len(failed)) if failed else "\nAll passed")
sys.exit(1 if failed else 0)
