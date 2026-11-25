"""Connection problems, for the Connections page: the last time a service outside the app turned Babble away (a
sign-in it no longer accepts, a key it refuses, a limit or empty balance, a connection that didn't go through) while
the app was really using it, not only when its Check button was pressed. A service's problem is cleared the next
time a request to it goes through, or when it's dismissed.

Kept in HOME/connection-issues.json: {service id: {"kind", "error", "where", "when"}}.
kind: "sign-in" (sign in again / check the key), "limit" (a limit, quota or balance ran out), "network" (couldn't
reach it: the connection, an address that didn't look up, a timeout).
"""
import errno
import json
import re
import threading
import time

from .paths import HOME

FILE = HOME / "connection-issues.json"
_lock = threading.Lock()

SIGN_IN = re.compile(r"sign in|signed in|sign-in|log ?in|unauthori[sz]ed|forbidden|invalid (api )?key|api key|"
                     r"permission|credentials?|\b401\b|\b403\b|not authenticated|authentication|token (is )?(invalid|expired)",
                     re.I)
LIMIT = re.compile(r"rate.?limit|quota|too many requests|\b429\b|limit reached|daily limit|credit|balance|"
                   r"insufficient|billing|payment required|\b402\b|out of (gpu )?hours|hours left", re.I)
NETWORK = re.compile(r"NameResolution|Failed to resolve|nodename nor servname|Name or service not known|Max retries exceeded|"
                     r"Connection (reset|refused|aborted|error|closed|lost)|timed out|Temporary failure|"
                     r"Network is (unreachable|down)|No route to host|SSL|getaddrinfo|ConnectError|RemoteDisconnected|"
                     # Modal: its SDK's words, and modal_jobs' own for its file storage
                     r"Could(n't| not) connect|Can't reach|cannot connect|unable to connect|StreamTerminated|UNAVAILABLE|"
                     r"ClientConnectorError|ServerDisconnected|EOF occurred|Broken pipe|IncompleteRead|"
                     r"Bad Gateway|Service Unavailable|Gateway Time-?out|"
                     r"\[Errno (50|51|54|60|61|64|65)\]", re.I)  # (network down / unreachable, reset, timed out, refused…)


def kind_of(e):
    """"sign-in", "limit", "network", or None for a failure that isn't about the connection (a bad reply, a bug)."""
    import socket
    x, seen = e, 0
    while x is not None and seen < 6:
        name = type(x).__name__
        if name == "AuthError" or name in ("AuthenticationError", "PermissionDeniedError"):
            return "sign-in"
        if name in ("RateLimitError", "LimitReached"):
            return "limit"
        if isinstance(x, (ConnectionError, TimeoutError, socket.gaierror, socket.timeout)) or name in (
                "APIConnectionError", "APITimeoutError", "ConnectTimeout", "ReadTimeout"):
            return "network"
        x, seen = x.__cause__ or x.__context__, seen + 1
    text = str(e)
    if SIGN_IN.search(text):
        return "sign-in"
    if LIMIT.search(text):
        return "limit"
    if NETWORK.search(text):
        return "network"
    return None


def is_network(e):
    """Whether a failure was the connection (it dropped, didn't go through, a network that blocks the service):
    worth trying again once it's back, not a real error."""
    return kind_of(e) == "network"


ONLINE = ("captive.apple.com",)  # (whether this Mac is online at all, for a service that names no servers)


def reachable(hosts=(), timeout=5.0):
    """Whether this Mac reaches these servers now: each one's name looks up and a secure connection to it opens
    (a network that blocks a service by name turns that away). Quick, and costs nothing."""
    import socket
    import ssl
    ctx = ssl.create_default_context()
    for host in hosts or ONLINE:
        try:
            with socket.create_connection((host, 443), timeout=timeout) as s:
                with ctx.wrap_socket(s, server_hostname=host):
                    pass
        except (OSError, ssl.SSLError):
            return False
    return True


def _read():
    try:
        return json.loads(FILE.read_text())
    except (OSError, ValueError):
        return {}


def _write(d):
    tmp = FILE.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(d, indent=1))
        tmp.replace(FILE)
    except OSError as e:
        if e.errno != errno.ENOSPC:
            raise
        print("[connections] disk full: couldn't save connection-issues.json", flush=True)
    finally:
        tmp.unlink(missing_ok=True)


def note(service, e, where=""):
    """A request to `service` failed: kept when it's about the connection. Returns the kind (or None)."""
    kind = kind_of(e)
    if not kind or not service:
        return None
    with _lock:
        d = _read()
        d[service] = {"kind": kind, "error": re.sub(r"\s+", " ", str(e)).strip()[:400], "where": where,
                      "when": time.time()}
        _write(d)
    print(f"[connections] {service}: {kind} problem ({where}): {str(e)[:160]}", flush=True)
    return kind


def ok(service):
    """A request to `service` went through: any problem it had is over."""
    if not service:
        return
    with _lock:
        d = _read()
        if service in d:
            d.pop(service)
            _write(d)


def watch(service, where, fn, *args, **kw):
    """fn(*args, **kw), noting a connection problem with `service` if it fails (and clearing it if it works)."""
    try:
        r = fn(*args, **kw)
    except Exception as e:
        note(service, e, where)
        raise
    ok(service)
    return r


def watch_gen(service, where, gen):
    """The same for a generator (a task's events): `yield from conn_issues.watch_gen(...)`."""
    try:
        r = yield from gen
    except GeneratorExit:
        raise
    except Exception as e:
        note(service, e, where)
        raise
    ok(service)
    return r


def issues():
    """{service: {"kind", "error", "where", "when"}}"""
    return _read()
