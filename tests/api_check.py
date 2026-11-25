"""Check the local API (and that the Gradio page still works next to it) on an isolated copy of the app.

    .venv/bin/python tests/api_check.py

Starts the app on port 7897 with an empty data folder (your library and books are never touched), uses only light
models (Kokoro, and Stable Audio for one effect), and stops it at the end. A watchdog stops everything if free
memory falls under 8%.
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = 7897
BASE = f"http://127.0.0.1:{PORT}"
HOME = Path(tempfile.mkdtemp(prefix="api-check-"))
ok = True


def check(cond, what):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + what, flush=True)


def key():
    """The engine's key for this run (narrator/guard.py): every /api call but /api/health needs it."""
    f = HOME / "engine-token"
    return f.read_text().strip() if f.exists() else ""


def http(method, path, body=None, headers=None, raw=False):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json", "X-Babble-Token": key(), **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            out = r.read()
            return r.status, (out if raw else json.loads(out or b"null")), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null"), dict(e.headers)


def wait(task, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        _, info, _ = http("GET", f"/api/tasks/{task['id']}")
        if info["state"] != "running":
            return info
        time.sleep(0.5)
    return info


def free_memory():
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    line = next((x for x in out.splitlines() if "free percentage" in x), "100%")
    return int(line.rsplit(" ", 1)[-1].strip("%"))


env = dict(os.environ, NARRATOR_HOME=str(HOME))
app = subprocess.Popen([sys.executable, "run.py", "--port", str(PORT)], cwd=ROOT, env=env,
                       stdout=open(HOME / "app.log", "w"), stderr=subprocess.STDOUT)


def watchdog():
    while app.poll() is None:
        if free_memory() < 8:
            print("STOPPED: free memory under 8%", flush=True)
            app.kill()
            os._exit(2)
        time.sleep(1)


threading.Thread(target=watchdog, daemon=True).start()
try:
    for _ in range(120):
        try:
            urllib.request.urlopen(BASE + "/api/health", timeout=2)
            break
        except OSError:
            time.sleep(1)

    # ---------- the page and the API side by side ----------
    status, health, _ = http("GET", "/api/health")
    check(status == 200 and health["ok"] and health["api"] == 1, f"health: {health}")
    status, page, _ = http("GET", "/", raw=True)
    check(status == 200 and b'id="root"' in page, "/ opens the app's interface (the old Gradio page is gone)")
    if (ROOT / "desktop" / "dist" / "index.html").exists():
        status, ui, _ = http("GET", "/app/", raw=True)
        check(status == 200 and b'id="root"' in ui, "the desktop interface is served at /app/")
    status, docs, _ = http("GET", "/api/docs", raw=True)
    check(status == 200, "API docs at /api/docs")

    _, engs, _ = http("GET", "/api/engines")
    _, voices, _ = http("GET", "/api/voices")
    check({"kokoro", "orpheus", "qwen"} <= {e["id"] for e in engs}, f"engines: {[e['id'] for e in engs]}")
    check(any(v["value"] == "kokoro:bm_george" and v["available"] for v in voices), f"{len(voices)} voices listed")
    _, st, _ = http("GET", "/api/settings")
    check("orpheus_batch" in st and not any("key" in k for k in st), "settings, without API keys")
    status, _, _ = http("POST", "/api/settings", {"no_such_setting": 1})
    check(status == 400, "unknown settings are refused")

    # ---------- speech: a task, its progress and its file ----------
    _, task, _ = http("POST", "/api/speech", {"text": "The rain had stopped.\n\nMarcus knocked twice.",
                                              "voice": "kokoro:bm_george"})
    info = wait(task)
    audio = info["result"].get("audio")
    check(info["state"] == "done" and audio and info["progress"] == 1.0, f"speech task: {info['state']} · {info['status'][:60]!r}")
    status, wav, _ = http("GET", f"/api/files?path={urllib.request.quote(audio)}", raw=True)
    check(status == 200 and wav[:4] == b"RIFF", f"the audio file can be fetched ({len(wav)} bytes)")
    _, hist, _ = http("GET", "/api/history")
    check(hist and hist[0]["path"] == audio and hist[0]["kind"] == "speech", "history lists it first")
    status, _, _ = http("GET", "/api/files?path=/etc/hosts")
    check(status == 403, "files outside the app's folders are refused")
    status, bad, _ = http("POST", "/api/speech", {"text": "  ", "voice": "kokoro:bm_george"})
    info = wait(bad)
    check(info["state"] == "failed" and "Type or paste" in (info["error"] or ""), f"a user problem comes back as its message: {info['error']!r}")

    long = "\n\n".join(f"This is line number {n} of a long reading." for n in range(1, 15))
    _, task, _ = http("POST", "/api/speech", {"text": long, "voice": "kokoro:bm_george"})
    time.sleep(3)
    http("POST", f"/api/tasks/{task['id']}/stop")
    info = wait(task)
    check(info["state"] == "stopped", f"a task can be stopped: {info['state']} · {info['status'][:50]!r}")

    # ---------- an audiobook through the event stream ----------
    book = {"story": "## Chapter 1: The Door\n\nThe door **slammed** shut. {sfx: heavy wooden door slamming @slammed}\n\n"
                     "“Who’s there?” she whispered.",
            "cast": [{"character": "Narrator", "voice": "kokoro:bm_george"}], "title": "API Book"}
    _, task, _ = http("POST", "/api/books/create", book)
    events, end = [], None
    with urllib.request.urlopen(BASE + f"/api/tasks/{task['id']}/events?auth={key()}", timeout=900) as r:
        kind = None
        for line in r:
            line = line.decode().strip()
            if line.startswith("event: end"):
                kind = "end"
            elif line.startswith("data: "):
                (events.append(json.loads(line[6:])) if kind != "end" else None)
                end = json.loads(line[6:]) if kind == "end" else end
    statuses = [e.get("status", "") for e in events]
    check(end and end["state"] == "done" and end["result"]["audio"].endswith(".wav"),
          f"book task done: {end and end['status'][:70]!r}")
    check(any("Making the missing sounds" in s for s in statuses) and any("Line 1 of" in s for s in statuses),
          f"the stream had every step ({len(events)} events)")
    _, lib, _ = http("GET", "/api/sounds")
    check(len(lib) == 1 and lib[0]["kind"] == "effect" and Path(lib[0]["path"]).exists(),
          f"the missing sound was made, with its path: {[e['name'] for e in lib]}")
    _, bks, _ = http("GET", "/api/books")
    check(any(b["title"] == "API Book" and b["file"] for b in bks), "the book is listed with its file")
    t0 = time.time()
    _, again, _ = http("POST", "/api/books/create", book)
    info = wait(again)
    check(info["state"] == "done" and time.time() - t0 < 15, f"the same book again reuses its chapter ({time.time() - t0:.0f} s)")

    # ---------- the Audiobook page's helpers ----------
    _, opts, _ = http("GET", "/api/books/options")
    check("WAV" in opts["formats"] and "english" in opts["languages"] and opts["models"][0]["value"] == "cast",
          "book options: formats, languages, voices for the whole book")
    _, task, _ = http("POST", "/api/books/hear", {"story": book["story"], "character": "Narrator", "voice": "kokoro:bm_george"})
    info = wait(task)
    check(info["state"] == "done" and info["result"].get("audio", "").endswith(".wav"), "hear a character's voice")
    _, task, _ = http("POST", "/api/books/suggest", {"story": "He slammed the door. Then a heavy wooden door slammed again.", "method": "quick"})
    info = wait(task)
    check(info["state"] == "done", f"quick suggestions: {info['status'][:70]!r}")
    status, err, _ = http("POST", "/api/books/suggest/apply", {"story": "changed", "planned_on": "original", "placements": []})
    check(status == 400 and "changed" in err["detail"], "suggestions can't be added to a story that changed since")
    _, same, _ = http("POST", "/api/books/suggest/apply", {"story": "same", "planned_on": "same", "placements": []})
    check(same["story"] == "same" and "Nothing selected" in same["message"], "adding none leaves the story as it was")
    boundary = "----up"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"my cover.png\"\r\n"
            f"Content-Type: image/png\r\n\r\n").encode() + b"\x89PNG fake" + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(BASE + "/api/files/upload", data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "X-Babble-Token": key()})
    up = json.loads(urllib.request.urlopen(req, timeout=60).read())
    status, got, _ = http("GET", f"/api/files?path={urllib.request.quote(up['path'])}", raw=True)
    check(status == 200 and got == b"\x89PNG fake", "an uploaded cover can be shown again")

    # ---------- import, detect, Kaggle ----------
    txt = HOME / "Little.txt"
    txt.write_text("The Little Lighthouse\n\nMara lived by the sea.\n\n“Hello,” said Mara.")
    import mimetypes  # noqa: F401
    boundary = "----api-check"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"Little.txt\"\r\n"
            f"Content-Type: text/plain\r\n\r\n").encode() + txt.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(BASE + "/api/books/import", data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "X-Babble-Token": key()})
    imported = json.loads(urllib.request.urlopen(req, timeout=60).read())
    check(imported["text"].startswith("The Little Lighthouse") and "## " not in imported["text"], f"import: {imported['message']!r}")
    _, task, _ = http("POST", "/api/books/detect", {"story": imported["text"], "method": "quick",
                                                   "cast": [{"character": "Narrator", "voice": "kokoro:bm_george"}]})
    info = wait(task)
    check(info["state"] == "done" and "[Mara]" in info["result"]["story"] and any(r[0] == "Mara" for r in info["result"]["cast"]),
          f"speaker detection: {info['status'][:60]!r}")
    status, kz, _ = http("POST", "/api/books/kaggle", book)
    check(status == 200 and Path(kz["path"]).exists(), "Kaggle bundle")

    # ---------- only this app's web view may call it from a browser ----------
    _, _, h = http("GET", "/api/health", headers={"Origin": "https://example.com"})
    _, _, h2 = http("GET", "/api/health", headers={"Origin": "tauri://localhost"})
    check("access-control-allow-origin" not in {k.lower() for k in h} and
          h2.get("access-control-allow-origin") == "tauri://localhost", "CORS: the desktop app yes, other sites no")
finally:
    app.terminate()
    try:
        app.wait(10)
    except subprocess.TimeoutExpired:
        app.kill()
    import shutil
    shutil.rmtree(HOME, ignore_errors=True)

print("\nALL PASS" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
