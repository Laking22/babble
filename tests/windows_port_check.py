"""Cross-platform and Windows port checks (runs cleanly on macOS, Windows, and Linux without MLX).

    .venv/bin/python tests/windows_port_check.py
"""
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

TMP = Path(tempfile.mkdtemp(prefix="win-port-check-"))
os.environ["NARRATOR_HOME"] = str(TMP / "home")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from narrator import agents, checks, paths, runtime, sfx, tts, vault  # noqa: E402
from narrator.training.backends.local_mac import LocalMac  # noqa: E402

ok = True


def check(cond, what):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + what)


# 1. Cross-platform default home path resolution
with mock.patch.object(sys, "platform", "win32"), mock.patch.dict(os.environ, {"APPDATA": r"C:\Users\Test\AppData\Roaming"}, clear=False):
    h = paths._default_home()
    check(str(h).endswith("Babble") and "Roaming" in str(h), f"win32 default home uses APPDATA\\Babble: {h}")

with mock.patch.object(sys, "platform", "linux"), mock.patch.dict(os.environ, {"XDG_DATA_HOME": "/tmp/xdg"}, clear=False):
    h = paths._default_home()
    check(h.as_posix() == "/tmp/xdg/babble", f"linux default home uses XDG_DATA_HOME: {h}")

# 2. RAM detection returns a positive int on current OS and never crashes
ram = runtime.total_ram_bytes()
check(isinstance(ram, int) and ram > 0, f"total_ram_bytes() positive: {ram}")

# 3. Local TTS engines report clean availability when HAS_MLX is False
with mock.patch.object(runtime, "HAS_MLX", False):
    for eid in ("qwen", "kokoro"):
        e = tts.ENGINES[eid]
        mod = sys.modules[type(e).__module__]
        # (some engines check their own MLX import, some the HAS_MLX they took from runtime)
        with mock.patch.object(mod, "mx", None, create=True), mock.patch.object(mod, "HAS_MLX", False, create=True):
            avail, why = e.available()
        check(avail is False and bool(why), f"{eid} unavailable cleanly without MLX: {why}")
        check(e.cloud is True, f"{eid} remains cloud-eligible without MLX")

# 4. Vault Windows loopback back-channel protocol (auth nonce, set/get persistence, bad-nonce rejection, EOF wipe)
nonce = "0123456789abcdef0123456789abcdef"
saved_store = {}
auth_ok = {"good": False, "bad": False}


class _FakeOut:
    """Simulates the engine -> app write stream of the Windows loopback socket."""
    def __init__(self, expected_nonce, authed_key):
        self.expected_nonce = expected_nonce
        self.authed_key = authed_key
        self.authed = False
        self.closed = False

    def write(self, data: str):
        if self.closed:
            raise OSError("closed")
        for line in data.splitlines():
            msg = json.loads(line)
            if not self.authed:
                if msg.get("nonce") == self.expected_nonce:
                    self.authed = True
                    auth_ok[self.authed_key] = True
                else:
                    self.closed = True
                continue
            seq = msg.get("seq", 0)
            saved_store.clear()
            saved_store.update(msg.get("vault") or {})
            with vault._cv:
                vault._acks[seq] = True
                vault._cv.notify_all()
        return len(data)

    def flush(self):
        pass


good_out = _FakeOut(nonce, "good")
good_out.write(json.dumps({"nonce": nonce}) + "\n")
bad_out = _FakeOut(nonce, "bad")
bad_out.write(json.dumps({"nonce": "00000000000000000000000000000000"}) + "\n")

orig = (vault.ON, vault._out, vault._keys.copy(), dict(vault._state), dict(vault._acks))
try:
    vault.ON = True
    vault._out = good_out
    vault._keys.clear()
    vault._state.update(ready=True, ok=True, why="", seq=0)
    vault._acks.clear()
    check(auth_ok["good"] is True and auth_ok["bad"] is False, "loopback auth accepts valid nonce and rejects bad nonce")
    check(vault.put({"test_secret": "s3cr3t"}) is True, "vault.put writes through Windows back-channel")
    check(saved_store.get("test_secret") == "s3cr3t", "Windows host mock received persisted secret")
    check(vault.keys().get("test_secret") == "s3cr3t", "vault.keys() returns stored secret")
    vault._out = bad_out
    check(vault.put({"test_secret2": "nope"}) is False, "bad-nonce closed back-channel fails cleanly")
    # EOF on stdin pipe wipes keys from RAM
    with mock.patch.dict(os.environ, {"BABBLE_VAULT_NO_EXIT": "1"}):
        vault._reader(io.StringIO(""))
    check(vault.keys() == {} and vault._state["ok"] is False, "stdin EOF wipes decrypted keys from RAM")
finally:
    vault.ON, vault._out = orig[0], orig[1]
    vault._keys.clear()
    vault._keys.update(orig[2])
    vault._state.clear()
    vault._state.update(orig[3])
    vault._acks.clear()
    vault._acks.update(orig[4])

# 5. sfx._to_trash fallback on non-macOS moves file into LIB/Deleted
dummy = TMP / "sound.wav"
dummy.write_bytes(b"RIFF")
with mock.patch.object(sys, "platform", "win32"):
    sfx._to_trash([(dummy, "sound")])
deleted = sfx.LIB / "Deleted" / "sound.wav"
check(not dummy.exists() and deleted.exists(), "sfx._to_trash moves file to Deleted/ on Windows")

# 6. LocalMac training backend cleanly refuses non-macOS
with mock.patch.object(sys, "platform", "win32"):
    res = LocalMac().check()
    check(res.get("ok") is False and "Mac" in (res.get("detail") or ""), f"LocalMac.check() reports unavailable on Windows: {res}")

# 7. agents.setup() uses Windows venv path & double-quoted command on win32
with mock.patch.object(sys, "platform", "win32"):
    cfg = agents.setup()
    check('"' in cfg["claude_code"] and "'" not in cfg["claude_code"], f"agents.setup() uses double quotes on win32: {cfg['claude_code']}")

# 8. checks.run() skips macOS-only MLX check on win32
with mock.patch.object(sys, "platform", "win32"):
    probs = checks.run()
    check(not any("mlx" in (p.get("missing") or "").lower() for p in probs), f"checks.run() does not flag MLX on win32: {probs}")

print("\nALL PASS" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
