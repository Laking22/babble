"""Who may use the engine. It listens on 127.0.0.1 only, but that alone lets in every program on this Mac (other
users' and sandboxed apps' too) and, through the browser, web pages: a page can send simple requests to
127.0.0.1 (CSRF), or point its own domain at 127.0.0.1 and then read the answers (DNS rebinding). So:

- Host: only 127.0.0.1 / localhost / [::1]. A rebinding page's requests carry its own domain and are refused.
- Origin: a browser request must come from the app's own window (Tauri), its dev server, or /app on this engine.
- A secret for this run, made at start and kept in <data folder>/engine-token (only you can read it). Every /api
  call carries it: the X-Babble-Token header, or ?auth=… where a header can't be sent (audio and image sources,
  live task updates). The desktop app reads it through its own native side; the launcher opens /app with it.
  /api/health (no data: is it up, which version) is the one call without it.

Programs running as you can still read the data folder (the keys too, until they move to the Keychain); this
shuts out everything else.
"""
import hmac
import json
import os
import re
import secrets
from urllib.parse import parse_qs

from . import agents
from .paths import HOME

TOKEN_FILE = HOME / "engine-token"
HEADER = b"x-babble-token"
QUERY = "auth"
OPEN = {"/api/health"}  # (OPTIONS preflights too: the CORS layer outside answers them)
ORIGINS = re.compile(r"^(tauri://localhost|https?://tauri\.localhost|https?://(localhost|127\.0\.0\.1)(:\d+)?)$")
HOSTS = ("127.0.0.1", "localhost", "[::1]")
_token = None


def token():
    """This run's secret, written (only you can read it) the first time it's asked for."""
    global _token
    if _token is None:
        t = secrets.token_urlsafe(32)
        tmp = TOKEN_FILE.with_name(TOKEN_FILE.name + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(t)
        os.chmod(tmp, 0o600)
        os.replace(tmp, TOKEN_FILE)
        _token = t
    return _token


def _host_ok(host):
    name = host.rsplit(":", 1)[0] if not host.endswith("]") else host
    return name in HOSTS


_told = set()


def _deny(code, why, kind="http", origin="", path=""):
    if (code, origin) not in _told:  # one line in the log per kind of refusal and caller (no keys in it)
        _told.add((code, origin))
        print(f"[guard] refused {origin or 'a program on this Mac'} ({code}) on {path[:60]}", flush=True)
    body = json.dumps({"detail": why}).encode()

    async def respond(send):
        if kind == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        await send({"type": "http.response.start", "status": code,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})
    return respond


QUERY_PATHS = re.compile(r"^/api/(?:files(?:/.*)?|tasks/[^/]+/events|packs/preview)$")


class Guard:
    """ASGI middleware (inside the CORS layer, so the app's window can read a refusal)."""

    def __init__(self, app):
        self.app = app
        token()

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers") or [])
        host = headers.get(b"host", b"").decode("latin-1")
        origin = headers.get(b"origin", b"").decode("latin-1")
        path = scope.get("path", "")
        method = scope.get("method", "GET")
        if not _host_ok(host):
            return await _deny(421, "Babble only answers on 127.0.0.1.", scope["type"], origin or host, path)(send)
        if origin and not ORIGINS.match(origin):  # ("null": a sandboxed frame or a file:// page, never the app)
            return await _deny(403, "Not from Babble.", scope["type"], origin, path)(send)
        if path.startswith("/api") and path not in OPEN and method != "OPTIONS":
            given = headers.get(HEADER, b"").decode("latin-1")
            if not given and (scope["type"] == "websocket" or (method == "GET" and QUERY_PATHS.match(path))):
                given = (parse_qs(scope.get("query_string", b"").decode("latin-1")).get(QUERY) or [""])[0]
            if not given or not hmac.compare_digest(given.encode(), token().encode()):
                return await _deny(401, "Babble's engine needs the app's key for this run. Reopen Babble.", scope["type"],
                                   origin, path)(send)
            if ("ok", origin) not in _told:  # (and one line when a caller first gets in with the key)
                _told.add(("ok", origin))
                print(f"[guard] {origin or 'a program on this Mac'} has this run's key", flush=True)
        # an AI agent (agents.py: the MCP server says so): only when the owner turned agents on, never keys, sign-ins
        # or limits, and what it starts is marked as its own (its own spending limit)
        agent = headers.get(agents.HEADER, b"").decode("latin-1")[:40] if path.startswith("/api") else ""
        if agent and path not in OPEN and method != "OPTIONS":
            why = agents.refused(path, method)
            if why:
                return await _deny(403, why, scope["type"], f"agent {agent}", path)(send)

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                hdrs = list(message.get("headers") or [])
                if not any(k.lower() == b"referrer-policy" for k, _ in hdrs):
                    hdrs.append((b"referrer-policy", b"no-referrer"))
                if method == "OPTIONS" and headers.get(b"access-control-request-private-network") == b"true":
                    hdrs.append((b"access-control-allow-private-network", b"true"))
                message = {**message, "headers": hdrs}
            await send(message)

        mark = agents.VIA.set(agent) if agent else None
        try:
            await self.app(scope, receive, send_with_headers if scope["type"] == "http" else send)
        finally:
            if mark is not None:
                agents.VIA.reset(mark)
