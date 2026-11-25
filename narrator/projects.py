"""Audiobook projects: each book's text, cast and settings, kept by the app (one JSON file per book), so the
Audiobook page can hold many books at once instead of one form.

A project's audio goes to its own folder in outputs/books (its "folder", fixed when it's made), so renaming a
book or leaving it untitled never mixes its chapters up with another book's. An untitled book is shown as
"Untitled-N": N is given when it's made, one more than the highest untitled book there is.
"""
import json
import os
import re
import secrets
import shutil
import threading
import time

from .paths import HOME

_lock = threading.Lock()
PROJECTS = HOME / "projects"
PROJECTS.mkdir(parents=True, exist_ok=True)
ID = re.compile(r"[a-f0-9]{8}")
COLORS = ["#cc4a0b", "#1c7ed6", "#7048e8", "#2b8a3e", "#c2255c", "#e67700", "#0c8599", "#5f3dc4"]

# what a project keeps, with the values a new book starts with (the same as the old one-book form)
DEFAULTS = {
    "title": "", "author": "", "cover": None, "story": "", "original": "",
    "cast": [], "models": "cast", "quotes_only": True, "out_format": "WAV",
    "speed": 1.0, "stability": 55, "language": "english", "seed": 0, "batch": 2, "line_pause": 0.2, "para_pause": 0.7,
    "parallel": True,
    "make_on": "mac",  # where Create makes it: "mac", "kaggle" or "modal" (the lines in the cloud, then finished here)
    "cloud_sounds": True,  # with a cloud: its missing AI sounds are made there too (else on this Mac)
    "cloud_gpu": "A10G",  # the Modal GPU
    "cloud_detect": False,  # Detect speakers (AI) on that cloud GPU too, not this Mac
    "cloud_hours": None,  # the most one cloud run may take: 3 (none), 6 or 12 h (cloud_book.hour_cap)
    "cloud_live": True,  # bring the lines back as they're made, where the service allows it (Backend.live)
    "pronunciations": [],  # this book's own [{"word", "say"}] (on top of the every-book list)      # make several lines at once (batch of them) with the models that can: Qwen and Orpheus
    "sound": True, "bed": 1.0, "intro": "",
    "fx_level": None, "amb_level": None, "music_level": None,
    "chapter_music": None, "intro_level": None,  # each chapter's intro and outro (chapter_music.py), their volume
    "detect": "ai",
    # the book's own sound settings (None: the app-wide one it had before they moved into each book)
    "sfx_exact": None, "sfx_auto_make": None, "director_where": None, "director_models": None, "director_checks": None, "director_scenes": None, "sound_libraries": None, "director_intros": None,
    "director_notes": "",  # notes for the sound director, for the whole book ("dark gloomy spooky")
    "new_sounds": "all",   # sounds made for this book: "book" keeps them in it, "all" shares them with every book
    "series_id": None,     # optional Series Bible ID linking sequels (shared cast + pronunciations)
    "file": None,          # the last finished audiobook file
}
META = ("id", "untitled", "folder", "color", "created", "updated")


def _path(pid):
    if not ID.fullmatch(pid or ""):
        raise ValueError("No such audiobook.")
    return PROJECTS / f"{pid}.json"


def _read(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return {**DEFAULTS, **data} if isinstance(data, dict) and data.get("id") else None


def _write(p):
    path = _path(p["id"])
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(json.dumps(p, ensure_ascii=False))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)  # crash-safe atomic replace, never half a file


def name_of(p):
    return (p.get("title") or "").strip() or f"Untitled-{p.get('untitled') or 1}"


def everything():
    out = [p for p in (_read(f) for f in PROJECTS.glob("*.json")) if p]
    return sorted(out, key=lambda p: p.get("updated", 0), reverse=True)


def summary(p):
    """What the shelf shows: no story text, just its size."""
    from . import book
    story = p.get("story") or ""
    words = len(re.sub(r"\[[^\]]*\]|\{[^{}]*\}", " ", story).split())
    chapters = sum(1 for t, body in book.split_chapters(story) if body.strip() or (t and t not in book.UNSPOKEN_TITLES))
    file = p.get("file")
    return {**{k: p.get(k) for k in META}, "name": name_of(p), "title": p.get("title") or "",
            "author": p.get("author") or "", "cover": p.get("cover"), "words": words, "chapters": chapters,
            "cast": len(p.get("cast") or []), "file": file if file and os.path.exists(file) else None,
            "format": p.get("out_format"), "cloud": _cloud(p)}


def _cloud(p):
    """The book's run in the cloud, while it's out there (for the shelf): {"state", "progress", "eta", "stitch",
    "stitch_progress"} (stitch: what putting it together here is doing, while it is)."""
    try:
        from . import books, cloud_book
        where = books.BOOKS / (p.get("folder") or "")
        rec = cloud_book.read(where) if p.get("folder") else None
        if not rec or rec.get("state") not in cloud_book.WAITING + ("downloading", "ready"):
            return None
        out = {"state": rec["state"], **cloud_book._pace(rec)}
        t = cloud_book.stitching(where)
        if t is not None:
            out["stitch"] = (t.status or "").split("\n")[0]
            out["stitch_progress"] = t.progress
            out["stitch_eta"] = t.time_left()
        return out
    except Exception:  # noqa: BLE001 - the shelf shows it without
        return None


def next_untitled():
    return max([p.get("untitled") or 0 for p in everything() if not (p.get("title") or "").strip()] + [0]) + 1


def create(data=None):
    """A new book from `data` (any of DEFAULTS; "folder" to use an existing book folder, e.g. one made before
    projects existed, so its chapters are reused)."""
    from .books import FOLDER_NAME
    with _lock:
        data = dict(data or {})
        pid = secrets.token_hex(4)
        while _path(pid).exists():
            pid = secrets.token_hex(4)
        folder = data.pop("folder", None)
        if data.pop("legacy", False):  # the one book the Audiobook page kept before: its folder, so its chapters are reused
            from .books import book_name
            folder = book_name(data.get("title"), data.get("story"))
        if not (folder and FOLDER_NAME.fullmatch(folder)):
            folder = f"book-{pid}"
        now = time.time()
        p = {**DEFAULTS, **{k: v for k, v in data.items() if k in DEFAULTS}, "id": pid, "untitled": next_untitled(),
             "folder": folder, "color": COLORS[len(everything()) % len(COLORS)], "created": now, "updated": now}
        _write(p)
        return p


def get(pid):
    p = _read(_path(pid))
    if p is None:
        raise ValueError("That audiobook isn't here any more.")
    return p


def save(pid, changes):
    """Merge changes (only the fields a project keeps) and save. Returns the project."""
    with _lock:
        p = get(pid)
        bad = sorted(set(changes) - set(DEFAULTS) - {"color"})
        if bad:
            raise ValueError(f"Unknown audiobook fields: {', '.join(bad)}")
        p.update(changes)
        p["updated"] = time.time()
        _write(p)
        return p


def delete(pid, delete_audio=False):
    """Remove a book's text and settings. If delete_audio is True, also removes its generated audio folder
    (outputs/books/<folder>) when no other project shares it. Stops any active cloud run for that folder."""
    with _lock:
        path = _path(pid)
        if not path.exists():
            raise ValueError("That audiobook isn't here any more.")
        folder = (_read(path) or {}).get("folder")
        path.unlink()
        shared = bool(folder and any(p.get("folder") == folder for p in everything()))
    if folder and not shared:
        from . import books, cloud_book
        where = books.BOOKS / folder
        try:
            if cloud_book.read(where):
                cloud_book.forget(where)
        except Exception as e:  # noqa: BLE001 - the book is deleted either way
            print(f"[cloud] couldn't stop the deleted book's cloud run: {e}", flush=True)
        if delete_audio and books.FOLDER_NAME.fullmatch(folder):
            root = books.BOOKS.resolve()
            target = where.resolve()
            if target.parent == root and target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
