"""API keys kept in the macOS Keychain, by the Babble app (desktop/src-tauri/src/lib.rs), not in settings.json.

The Keychain gives a key only to the program that saved it, so the app keeps them: one item ("Babble", account
"api-keys") holding all of them, so after an update macOS asks once, not once per key. Other programs can't read
it without a macOS prompt naming them.

When the app starts the engine (BABBLE_VAULT=1 in its environment), it passes them over two pipes only it holds:
- stdin, app → engine: first {"vault": {...}, "ok": true} (or "ok": false when the Keychain couldn't be read: then
  nothing is written back this run, so a refused read can never wipe the saved keys), then {"saved": n} or
  {"failed": n, "why": "…"} for each change.
- fd 3 (or BABBLE_VAULT_FD), engine → app: {"seq": n, "vault": {...}}, all the keys after a change; the app writes the Keychain item.
The keys reach the engine only through those pipes: not in its environment, its command line, or a file.

Started another way (the launcher, a test), there's no vault: keys stay in settings.json as before.
"""
import json
import os
import signal
import socket
import sys
import threading
import time

ON = os.environ.get("BABBLE_VAULT") == "1"
WAIT = 8.0             # seconds to wait for the app to say a change is in the Keychain
_keys = {}
_state = {"ready": False, "ok": False, "why": "", "seq": 0}
_acks = {}
_cv = threading.Condition()
_out = None
_sock = None
_expected = False  # True once Tauri vault handshake starts (prevents silent plaintext fallback on Windows)


def _reader(f):
    global _out, _sock, _expected
    first = True
    for line in f:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if first and "vault" in msg:
            back_pair = None
            if msg.get("back_port") and msg.get("back_nonce"):
                back_pair = _connect_back(int(msg["back_port"]), str(msg["back_nonce"]))
            with _cv:
                _expected = True
                _keys.clear()
                _keys.update({k: v for k, v in (msg.get("vault") or {}).items() if isinstance(v, str) and v})
                if back_pair is not None:
                    _sock, _out = back_pair
                    _state.update(ready=True, ok=bool(msg.get("ok")), why=msg.get("why") or "")
                elif msg.get("back_port"):
                    _state.update(ready=True, ok=False, why="Couldn't connect to Windows Credential Manager back-channel.")
                else:
                    _state.update(ready=True, ok=bool(msg.get("ok")), why=msg.get("why") or "")
                first = False
                _cv.notify_all()
            continue
        with _cv:
            if "saved" in msg:
                _acks[int(msg["saved"])] = True
            elif "failed" in msg:
                _acks[int(msg["failed"])] = msg.get("why") or "failed"
            _cv.notify_all()
    with _cv:  # the app is gone: wipe decrypted keys from RAM and shut down cleanly
        _keys.clear()
        _state["ok"] = False
        _cv.notify_all()
    if ON and not os.environ.get("BABBLE_VAULT_NO_EXIT"):
        print("[vault] app pipe closed — clearing keys and shutting down engine", flush=True)
        if sys.platform == "win32":
            # On Windows, os.kill(..., SIGTERM) calls TerminateProcess immediately; give in-flight writes 2s.
            threading.Timer(2.0, lambda: os._exit(0)).start()
        else:
            threading.Timer(3.0, lambda: os._exit(0)).start()
            try:
                os.kill(os.getpid(), signal.SIGTERM)
            except (OSError, AttributeError):
                os._exit(0)


def _connect_back(port, nonce):
    """Windows back-channel: connect to the app's ephemeral 127.0.0.1 listener and authenticate with the stdin nonce."""
    last_err = None
    for attempt in range(3):
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=2.0)
            f = s.makefile("w", encoding="utf-8", buffering=1)
            f.write(json.dumps({"nonce": nonce}) + "\n")
            f.flush()
            return s, f
        except OSError as e:
            last_err = e
            time.sleep(0.1 * (attempt + 1))
    print(f"[vault] couldn't connect back-channel: {last_err}", flush=True)
    return None


def start():
    """Take the pipes away from fd 0 (and fd 3 on Unix) so child processes don't inherit them, and listen."""
    global _out
    if not ON or _out is not None:
        return
    if sys.platform == "win32":
        try:
            inp = os.dup(0)
            devnull = os.open(os.devnull, os.O_RDONLY)
            os.dup2(devnull, 0)
            os.close(devnull)
        except OSError as e:
            print(f"[vault] no stdin pipe from the app: {e}", flush=True)
            return
        threading.Thread(target=_reader, args=(os.fdopen(inp, "r", encoding="utf-8"),), daemon=True, name="vault").start()
    else:
        back = int(os.environ.get("BABBLE_VAULT_FD") or 3)  # (the app's pipe back: fd 3 unless it says otherwise)
        try:
            os.fstat(back)        # (not there: no vault, rather than writing to whatever got that number)
            out = os.dup(back)    # (os.dup: not inherited by child processes)
            inp = os.dup(0)
            devnull = os.open(os.devnull, os.O_RDONLY)
            os.dup2(devnull, 0)
            os.close(devnull)
            os.close(back)
        except OSError as e:
            print(f"[vault] no pipes from the app: {e}", flush=True)
            return
        _out = os.fdopen(out, "w", buffering=1)
        threading.Thread(target=_reader, args=(os.fdopen(inp, "r"),), daemon=True, name="vault").start()
    with _cv:
        _cv.wait_for(lambda: _state["ready"], timeout=3)  # (normally at once; longer only while macOS asks you)
    if _state["ready"]:
        print(f"[vault] {len(_keys)} key(s) from the Keychain" + ("" if _state["ok"] else f" — couldn't read it: {_state['why']}"),
              flush=True)


def active():
    return ON and (_out is not None or _expected)


def status():
    """For Connections: where the keys are, and whether the Keychain could be read."""
    with _cv:
        if not active():
            return {"where": "file"}
        return {"where": "keychain", "ready": _state["ready"], "ok": _state["ok"], "why": _state["why"], "keys": len(_keys)}


def keys():
    with _cv:
        return dict(_keys)


def put(changes):
    """Set (or, with "", remove) keys. True once the app has them in the Keychain."""
    if not active():
        return False
    with _cv:
        if not _state["ready"]:
            _cv.wait_for(lambda: _state["ready"], timeout=WAIT)
        if not _state["ok"]:
            return False
        new = dict(_keys)
        for k, v in changes.items():
            v = (v or "").strip() if isinstance(v, str) or v is None else v
            if v:
                new[k] = v
            else:
                new.pop(k, None)
        _state["seq"] += 1
        n = _state["seq"]
        try:
            _out.write(json.dumps({"seq": n, "vault": new}) + "\n")
            _out.flush()
        except OSError as e:
            print(f"[vault] couldn't reach the app: {e}", flush=True)
            return False
        end = time.time() + WAIT
        while n not in _acks and time.time() < end and _state["ok"]:
            _cv.wait(timeout=0.5)
        ok = _acks.pop(n, None)
        if ok is True:
            _keys.clear()
            _keys.update(new)
            return True
        print(f"[vault] the Keychain didn't take the change: {ok or 'no answer'}", flush=True)
        return False
