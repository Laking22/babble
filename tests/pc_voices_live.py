"""Run against a running Babble on Windows or Linux (GitHub's build machines, on request: build.yml input `voices`):

    python tests/pc_voices_live.py

Sets up voices (PyTorch) the way the Models page does, then speaks one line with each engine that runs on the computer
itself and checks each is real speech: not silent, about as long as the line. Downloads several GB of models and,
without an NVIDIA card, makes the lines on the processor (minutes each).
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:7895"
HOME = Path(os.environ.get("NARRATOR_HOME") or (
    Path(os.environ.get("APPDATA", "")) / "Babble" if sys.platform == "win32"
    else Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "babble"))
TOKEN = (HOME / "engine-token").read_text().strip()
LINE = "The lighthouse keeper climbed the stairs one more time, counting every step."
VOICES = ["builtin:ryan", "kokoro:af_heart", "orpheus:tara"]
failed = []


def call(path, body=None, timeout=60):
    req = urllib.request.Request(BASE + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"X-Babble-Token": TOKEN, "Content-Type": "application/json"},
                                 method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"null")


def wait(task, minutes):
    end, last = time.time() + minutes * 60, ""
    while time.time() < end:
        t = call(f"/api/tasks/{task['id']}")
        if t["status"] != last:
            last = t["status"]
            print(f"   {t['state']}: {last}", flush=True)
        if t["state"] in ("done", "failed", "stopped"):
            return t
        time.sleep(5)
    return {"state": "timeout", "error": f"not done after {minutes} min"}


def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what, flush=True)
    if not ok:
        failed.append(what)


s = call("/api/voices-setup")
print("voices setup:", s, flush=True)
if s.get("needed"):
    t = wait(call("/api/voices-setup", {}), 30)
    check(t["state"] == "done", f"Set up voices ({t.get('error') or t['state']})")
models = {m["id"]: m for m in call("/api/models")}
for v in VOICES:
    m = models[v.partition(":")[0] if not v.startswith("builtin:") else "qwen"]
    check(m["ok"], f"{m['name']} can run here ({m['why'] or 'ok'})")
    if not m["ok"]:
        continue
    started = time.time()
    t = wait(call("/api/speech", {"text": LINE, "voice": v}), 45)
    took = time.time() - started
    if t["state"] != "done":
        check(False, f"{v}: {t.get('error') or t['state']}")
        continue
    import numpy as np
    import soundfile as sf
    res = t["result"] or {}
    path = res.get("audio") or ""
    audio, sr = sf.read(path, dtype="float32") if path and Path(path).exists() else (np.zeros(0), 24000)
    audio = audio.mean(axis=1) if audio.ndim > 1 else audio
    seconds = len(audio) / sr
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    check(2.5 < seconds < 15 and peak > 0.05, f"{v}: {seconds:.1f} s of audio, peak {peak:.2f}, made in {took:.0f} s")

print(f"\n{'all passed' if not failed else f'{len(failed)} failed: ' + '; '.join(failed)}")
sys.exit(1 if failed else 0)
