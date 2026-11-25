"""Series-Wide Character & Pronunciation Bible: links sequel books so returning characters inherit identical
voices, speeds, emotions, and pronunciations automatically.
"""
import json
import os
import re
import secrets
import threading
import time

from .paths import HOME

_lock = threading.Lock()
SERIES_DIR = HOME / "series"
SERIES_DIR.mkdir(parents=True, exist_ok=True)
ID = re.compile(r"[a-f0-9]{8}")


def _path(sid):
    if not ID.fullmatch(sid or ""):
        raise ValueError("No such series.")
    return SERIES_DIR / f"{sid}.json"


def _write(s):
    path = _path(s["id"])
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(json.dumps(s, ensure_ascii=False, indent=1))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read(path):
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) and d.get("id") else None
    except (OSError, ValueError):
        return None


def list_series():
    out = [s for s in (_read(f) for f in SERIES_DIR.glob("*.json")) if s]
    return sorted(out, key=lambda s: s.get("updated", 0), reverse=True)


def get(sid):
    s = _read(_path(sid))
    if s is None:
        raise ValueError("That series isn't here any more.")
    return s


def create(name, cast=None, pronunciations=None):
    with _lock:
        sid = secrets.token_hex(4)
        while _path(sid).exists():
            sid = secrets.token_hex(4)
        now = time.time()
        s = {
            "id": sid,
            "name": (name or "").strip() or "Untitled Series",
            "cast": list(cast or []),
            "pronunciations": list(pronunciations or []),
            "created": now,
            "updated": now,
        }
        _write(s)
        return s


def save(sid, changes):
    with _lock:
        s = get(sid)
        for k in ("name", "cast", "pronunciations"):
            if k in changes:
                s[k] = changes[k]
        s["updated"] = time.time()
        _write(s)
        return s


def delete(sid):
    with _lock:
        p = _path(sid)
        if not p.exists():
            raise ValueError("That series isn't here any more.")
        p.unlink()


def sync_from_book(sid, project):
    """Merge a book's cast and pronunciations into its Series Bible (updating existing characters by name)."""
    with _lock:
        s = get(sid)
        cast_by_name = {str(r[0]).strip().lower(): r for r in (s.get("cast") or []) if r}
        for r in (project.get("cast") or []):
            if r and str(r[0]).strip():
                cast_by_name[str(r[0]).strip().lower()] = r
        s["cast"] = list(cast_by_name.values())
        pron_by_word = {str(p.get("word", "")).strip().lower(): p for p in (s.get("pronunciations") or []) if isinstance(p, dict) and p.get("word")}
        for p in (project.get("pronunciations") or []):
            if isinstance(p, dict) and str(p.get("word", "")).strip():
                pron_by_word[str(p["word"]).strip().lower()] = p
        s["pronunciations"] = list(pron_by_word.values())
        s["updated"] = time.time()
        _write(s)
        return s


def apply_to_book(sid, project):
    """Return merged (cast, pronunciations) inheriting from the Series Bible while keeping book-specific overrides."""
    s = get(sid)
    cast_by_name = {str(r[0]).strip().lower(): r for r in (s.get("cast") or []) if r}
    for r in (project.get("cast") or []):
        if r and str(r[0]).strip():
            cast_by_name[str(r[0]).strip().lower()] = r
    pron_by_word = {str(p.get("word", "")).strip().lower(): p for p in (s.get("pronunciations") or []) if isinstance(p, dict) and p.get("word")}
    for p in (project.get("pronunciations") or []):
        if isinstance(p, dict) and str(p.get("word", "")).strip():
            pron_by_word[str(p["word"]).strip().lower()] = p
    return list(cast_by_name.values()), list(pron_by_word.values())
