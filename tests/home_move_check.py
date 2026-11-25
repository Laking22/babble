"""Run: .venv/bin/python tests/home_move_check.py  (about 20 s, no models, no network, scratch HOME only)

The Mac data folder's new name (narrator/move_home.py): ~/Library/Application Support/Narrator Studio → …/Babble.
1. A new Mac uses Babble; an old folder is used as it is until the engine moves it (importing paths creates nothing).
2. The move: the folder is renamed, records naming files by full path point to the new place, nothing else changes
   (other paths, "Narrator Studio 2", Python environments in tools/), settings.json stays yours only, links still work.
3. It waits when an engine is running, keeps an existing Babble folder that has files, clears an empty one, ignores
   NARRATOR_HOME setups, and finishes a move that was cut off.
4. The engine itself (run.py) moves it at start and writes its key in the new folder.
Every case runs in a scratch HOME: the real data folder is never touched.
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

if sys.platform != "darwin":
    print("skipped: the move is for Macs")
    sys.exit(0)

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="homemove-"))
failed = []


def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        failed.append(what)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def env(home):
    e = {k: v for k, v in os.environ.items() if k != "NARRATOR_HOME"}
    e["HOME"] = str(home)
    return e


def py(home, code, extra=None):
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env={**env(home), **(extra or {})},
                       capture_output=True, text=True, timeout=120)
    if r.returncode:
        print(r.stderr[-2000:])
    return r.stdout.strip().splitlines()


def where(home):
    return py(home, "from narrator import paths; print(paths.HOME)")[-1]


def move(home, port):
    return py(home, f"from narrator import move_home; print(move_home.move({port}))")


def scratch(name):
    home = TMP / name
    base = home / "Library" / "Application Support"
    base.mkdir(parents=True)
    return home, base / "Narrator Studio", base / "Babble"


def old_folder(old):
    """A small copy of a real data folder, with records that name files by their full path."""
    o = str(old)
    (old / "outputs" / "books" / "book-1").mkdir(parents=True)
    (old / "outputs" / "books" / "book-1" / "ch1.mp3").write_bytes(b"ID3")
    (old / "outputs" / "books" / "book-1" / "chapters.json").write_text(json.dumps(
        [{"file": f"{o}/outputs/books/book-1/ch1.mp3"}, {"file": "/Users/someone/Music/x.mp3"},
         {"file": f"{o} 2/outputs/other.mp3"}]))
    (old / "runs.json").write_text(json.dumps({"r1": {"installed": f"{o}/models/v1"}}, indent=1))
    (old / "activity.json").write_text(json.dumps([{"status": f"Done\nSaved to {o}/outputs/books/book-1"}]))
    (old / "spending.jsonl").write_text(json.dumps({"file": f"{o}/outputs/a.wav"}) + "\n" + json.dumps({"n": 1}) + "\n")
    (old / "sfx").mkdir()
    (old / "sfx" / "index.json").write_text(json.dumps([{"source": "Stable Audio 3 Small SFX, made in Narrator Studio"}]))
    (old / "settings.json").write_text(json.dumps({"theme": "dark"}))
    os.chmod(old / "settings.json", 0o600)
    (old / "engine-token").write_text("old-key")
    (old / "training" / "v" / "dataset").mkdir(parents=True)
    (old / "training" / "datasets").mkdir()
    os.symlink("../v/dataset", old / "training" / "datasets" / "v")
    (old / "tools" / "env" / "bin").mkdir(parents=True)
    (old / "tools" / "env" / "bin" / "info.json").write_text(json.dumps({"path": f"{o}/tools/env"}))
    (old / "models" / "v1").mkdir(parents=True)


# 1. which folder
home, old, new = scratch("fresh")
check(where(home) == str(new), "a new Mac uses …/Babble")
home, old, new = scratch("before")
old_folder(old)
check(where(home) == str(old) and not new.exists(), "before the move the old folder is used, and nothing new is made")

# 2. the move
port = free_port()
out = move(home, port)
n = (Path(new) / "outputs" / "books" / "book-1" / "chapters.json")
check(not old.exists() and new.is_dir() and out and out[-1] == "5", f"the folder is renamed and 5 records change (got {out[-1:]})")
check(where(home) == str(new), "after the move …/Babble is used")
ch = json.loads(n.read_text())
check(ch[0]["file"] == f"{new}/outputs/books/book-1/ch1.mp3" and Path(ch[0]["file"]).is_file(),
      "a chapter's file points to the new place, and is there")
check(ch[1]["file"] == "/Users/someone/Music/x.mp3" and ch[2]["file"] == f"{old} 2/outputs/other.mp3",
      "other paths (and a different folder that starts the same) are left alone")
check(json.loads((new / "runs.json").read_text())["r1"]["installed"] == f"{new}/models/v1", "installed voices point to the new place")
check(f"Saved to {new}/outputs" in json.loads((new / "activity.json").read_text())[0]["status"], "history text too")
lines = (new / "spending.jsonl").read_text().splitlines()
check(json.loads(lines[0])["file"] == f"{new}/outputs/a.wav" and json.loads(lines[1]) == {"n": 1}, "line-by-line records too")
check(json.loads((new / "sfx" / "index.json").read_text())[0]["source"].endswith("made in Babble"), "sounds made here say Babble")
check((new / "settings.json").stat().st_mode & 0o777 == 0o600, "settings.json stays readable by you only")
check((new / "training" / "datasets" / "v").resolve() == (new / "training" / "v" / "dataset").resolve(), "links inside still work")
check(str(old) in (new / "tools" / "env" / "bin" / "info.json").read_text(), "Python environments in tools/ aren't touched")
check(not (new / ".moved-from-narrator-studio").exists() and not list(new.rglob(".moving-*")), "nothing is left behind")
check(move(home, port)[-1] == "None", "running again does nothing")

# 3. when it waits or keeps things
home, old, new = scratch("running")
old_folder(old)
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    s.listen()
    busy = s.getsockname()[1]
    out = move(home, busy)
check(out[-1] == "None" and old.is_dir() and not new.exists() and "already running" in " ".join(out),
      "an engine already running: not moved (it could be using the old folder)")

home, old, new = scratch("empty-new")
old_folder(old)
(new / "voices" / "x").mkdir(parents=True)
move(home, free_port())
check(not old.exists() and (new / "runs.json").is_file() and not (new / "voices" / "x").exists(),
      "an empty Babble folder (folders only) is cleared and the move happens")

home, old, new = scratch("files-in-new")
old_folder(old)
new.mkdir()
(new / "note.txt").write_text("mine")
out = move(home, free_port())
check(out[-1] == "None" and old.is_dir() and (new / "note.txt").is_file() and where(home) == str(old),
      "a Babble folder with files in it is kept; the old folder stays in use")

home, old, new = scratch("used-new")
old_folder(old)
new.mkdir()
(new / "engine-token").write_text("k")
check(move(home, free_port())[-1] == "None" and where(home) == str(new) and old.is_dir(),
      "a Babble folder already in use is used, and the old one is left as it is")

home, old, new = scratch("env")
old_folder(old)
r = py(home, f"from narrator import move_home; print(move_home.move({free_port()}))", {"NARRATOR_HOME": str(TMP / "elsewhere")})
check(r[-1] == "None" and old.is_dir(), "NARRATOR_HOME set: nothing is moved")

home, old, new = scratch("cut-off")
old_folder(old)
os.rename(old, new)
(new / ".moved-from-narrator-studio").write_text(str(old))
move(home, free_port())
check(json.loads((new / "runs.json").read_text())["r1"]["installed"] == f"{new}/models/v1"
      and not (new / ".moved-from-narrator-studio").exists(), "a move that was cut off is finished at the next start")

# 4. the engine does it at start, when the app asks (an older app reads the key from the old folder)
def engine(home, ask):
    port = free_port()
    log = open(TMP / "engine.log", "w")
    e = {**env(home), **({"BABBLE_MOVE_DATA": "1"} if ask else {})}
    eng = subprocess.Popen([sys.executable, "run.py", "--port", str(port)], cwd=ROOT, env=e, stdout=log, stderr=log)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1)
                return True
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        return False
    finally:
        eng.terminate()
        eng.wait(10)


def key(folder):
    return (folder / "engine-token").read_text().strip() if (folder / "engine-token").exists() else ""


home, old, new = scratch("engine-old-app")
old_folder(old)
check(engine(home, ask=False) and old.is_dir() and not new.exists() and key(old) not in ("", "old-key"),
      "started by an older app: not moved, and the key is where that app looks (the old folder)")
home, old, new = scratch("engine")
old_folder(old)
check(engine(home, ask=True) and not old.exists() and key(new) not in ("", "old-key"),
      "started by this app: the folder is moved, and the engine's key is in the new one")
check("moved your data" in (TMP / "engine.log").read_text(), "…and it says so in its log")

print(f"\n{'all passed' if not failed else f'{len(failed)} failed'}")
sys.exit(1 if failed else 0)
