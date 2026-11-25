"""The Mac data folder's new name: ~/Library/Application Support/Narrator Studio becomes …/Babble.

move() runs once, when the engine starts (run.py) for an app that knows the new folder (BABBLE_MOVE_DATA=1: an
older app reads the engine's key from the old folder), before anything opens the folder. It renames the folder (one step,
nothing is copied) and then changes saved records that name files by their full path (book chapters, history, installed
voices, training jobs) to the new place. It does nothing when NARRATOR_HOME points elsewhere, when Babble's folder is
already in use, or when an engine is already running (it could be using the old folder).
"""
import os
import re
import socket
import sys
import tempfile
from pathlib import Path

MARK = ".moved-from-narrator-studio"   # left in the folder until every record is changed (a cut-off move finishes)
RECORDS = (".json", ".jsonl")
SKIP = {"tools", ".venv"}              # Python environments: nothing in them is a record
MAX_RECORD = 64 << 20


def folders():
    base = Path.home() / "Library" / "Application Support"
    return base / "Narrator Studio", base / "Babble"


def in_use(folder):
    """Whether a data folder has been used (the engine leaves its key there each start)."""
    return (folder / "engine-token").exists() or (folder / "settings.json").exists()


def _real_dir(p):
    return p.is_dir() and not p.is_symlink()


def old_home():
    """The old folder while it hasn't been moved yet (paths.py keeps using it until then), else None."""
    old, new = folders()
    return old if _real_dir(old) and not in_use(new) else None


def _engine_running(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def _has_files(folder):
    return any(files for _, _, files in os.walk(folder))


def move(port=7895, log=print):
    """Rename the old folder to Babble and change the records. Returns how many records changed (None: not moved)."""
    if sys.platform != "darwin" or (os.environ.get("NARRATOR_HOME") or "").strip():
        return None
    old, new = folders()
    if _real_dir(new) and (new / MARK).exists():
        return _finish(old, new, log)       # a move that was cut off
    if old_home() is None:
        return None
    if _engine_running(port):
        log(f"[babble] not moving {old} to {new} yet: an engine is already running")
        return None
    if new.exists() or new.is_symlink():
        if not _real_dir(new) or _has_files(new):
            log(f"[babble] not moving {old}: {new} is already there")
            return None
        for d, _, _ in sorted(os.walk(new), key=lambda w: -len(w[0])):   # (empty folders only)
            os.rmdir(d)
    (old / MARK).write_text(str(old))
    try:
        os.rename(old, new)
    except OSError as e:
        (old / MARK).unlink(missing_ok=True)
        log(f"[babble] couldn't move {old} to {new}: {e}")
        return None
    log(f"[babble] moved your data from {old} to {new}")
    return _finish(old, new, log)


def _finish(old, new, log):
    """Change every record under new that names a file under old. Safe to run again."""
    was = re.compile(re.escape(str(old)).encode() + rb"(?=[/\"\\]|$)", re.M)
    changed = failed = 0
    for d, dirs, files in os.walk(new):
        if Path(d) == new:
            dirs[:] = [x for x in dirs if x not in SKIP]
        for f in files:
            p = Path(d) / f
            if p.suffix not in RECORDS or p.is_symlink():
                continue
            try:
                if p.stat().st_size > MAX_RECORD:
                    continue
                data = p.read_bytes()
                out = was.sub(str(new).encode(), data).replace(b"made in Narrator Studio", b"made in Babble")
                if out != data:
                    _replace(p, out)
                    changed += 1
            except OSError as e:
                failed += 1
                log(f"[babble] couldn't update {p}: {e}")
    if not failed:
        (new / MARK).unlink(missing_ok=True)
    log(f"[babble] updated {changed} saved record(s) to the new folder" + (f"; {failed} couldn't be updated" if failed else ""))
    return changed


def _replace(path, data):
    """Write a file in one step, keeping its permissions (settings.json is yours only)."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".moving-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.chmod(tmp, path.stat().st_mode & 0o7777)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
