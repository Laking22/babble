"""Local Project Time Machine: content-addressed rolling JSON snapshots of a book's script, cast,
pronunciations, and sound settings so any AI pass can be reversed in one click without duplicating audio.
"""
import hashlib
import json
import os
import threading
import time

from . import projects

_lock = threading.Lock()
SNAP_DIR = projects.PROJECTS / "snapshots"
MAX_SNAPSHOTS = 15
SNAP_FIELDS = ("title", "author", "story", "cast", "pronunciations", "director_notes", "chapter_music", "sound", "bed")


def _book_dir(pid):
    projects._path(pid)  # validates ID format
    d = SNAP_DIR / pid
    d.mkdir(parents=True, exist_ok=True)
    return d


def take(pid, label="Snapshot"):
    """Save a compact content-addressed snapshot of project `pid`. Deduplicates identical back-to-back states."""
    with _lock:
        p = projects.get(pid)
        payload = {k: p.get(k) for k in SNAP_FIELDS}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
        d = _book_dir(pid)
        existing = sorted(d.glob("*.json"))
        if existing:
            try:
                last = json.loads(existing[-1].read_text(encoding="utf-8"))
                if last.get("hash") == digest:
                    return {"id": last["id"], "label": last["label"], "time": last["time"], "hash": digest, "deduped": True}
            except (OSError, ValueError):
                pass
        now = time.time()
        sid = f"{int(now * 1000)}-{digest}"
        rec = {"id": sid, "label": (label or "Snapshot").strip()[:80], "time": now, "hash": digest, "data": payload}
        tmp = d / f"{sid}.tmp"
        with tmp.open("w", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, d / f"{sid}.json")
        # Prune oldest beyond MAX_SNAPSHOTS
        all_files = sorted(d.glob("*.json"))
        for old in all_files[:-MAX_SNAPSHOTS]:
            old.unlink(missing_ok=True)
        return {"id": sid, "label": rec["label"], "time": now, "hash": digest, "deduped": False}


def list_for(pid):
    d = _book_dir(pid)
    out = []
    for f in sorted(d.glob("*.json"), reverse=True):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
            data = r.get("data") or {}
            out.append({
                "id": r["id"], "label": r.get("label") or "Snapshot", "time": r.get("time", 0),
                "hash": r.get("hash", ""), "chars": len(data.get("story") or ""),
                "cast": len(data.get("cast") or []), "pronunciations": len(data.get("pronunciations") or []),
            })
        except (OSError, ValueError):
            continue
    return out


def diff(pid, sid):
    """Return a plain-language summary of what changed between snapshot `sid` and the current project."""
    d = _book_dir(pid)
    path = d / f"{sid}.json"
    if not path.is_file():
        raise ValueError("Snapshot not found.")
    snap = json.loads(path.read_text(encoding="utf-8")).get("data") or {}
    cur = projects.get(pid)
    changes = []
    char_delta = len(cur.get("story") or "") - len(snap.get("story") or "")
    if char_delta:
        changes.append(f"Script {'+' if char_delta > 0 else ''}{char_delta} chars")
    cast_delta = len(cur.get("cast") or []) - len(snap.get("cast") or [])
    if cast_delta:
        changes.append(f"Cast {'+' if cast_delta > 0 else ''}{cast_delta} voice(s)")
    elif (cur.get("cast") or []) != (snap.get("cast") or []):
        changes.append("Cast assignments changed")
    if (cur.get("pronunciations") or []) != (snap.get("pronunciations") or []):
        changes.append("Pronunciations changed")
    if (cur.get("chapter_music") or None) != (snap.get("chapter_music") or None):
        changes.append("Chapter music changed")
    return {"id": sid, "changes": changes or ["No differences"]}


def restore(pid, sid):
    """Restore project `pid` to snapshot `sid` (automatically taking a safety snapshot of current state first)."""
    d = _book_dir(pid)
    path = d / f"{sid}.json"
    if not path.is_file():
        raise ValueError("Snapshot not found.")
    take(pid, label="Before restore")
    snap = json.loads(path.read_text(encoding="utf-8")).get("data") or {}
    return projects.save(pid, {k: snap[k] for k in SNAP_FIELDS if k in snap})
