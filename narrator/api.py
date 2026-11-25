"""The local API for the desktop app (and anything else on this Mac): http://127.0.0.1:7895/api/…

Served by the same process as the Gradio page, so there's still one voice model in memory at a time. Long work
(speech, books, sounds, speaker detection) starts a task and returns it at once:

  POST /api/books/create {…}        -> {"id": …, "state": "running", …}
  GET  /api/tasks/<id>              -> state, status text, progress 0..1, result (audio path, story, cast…)
  GET  /api/tasks/<id>/events       -> the same as a stream (server-sent events), from ?after=<n>
  POST /api/tasks/<id>/stop
  GET  /api/files?path=<audio path> -> the file (only from the app's own output, sound and voice folders)

Problems the user can fix come back as a task error (or HTTP 400) with a message meant for them.
Only listens on 127.0.0.1, and only answers the app: guard.py (this run's secret, the host and the origin).
"""
import asyncio
import json
import mimetypes
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from . import books, projects, settings, tasks
from . import engine as en
from . import tts as engines
from . import voices as vl

API_VERSION = 1
# Python doesn't know .m4b, so the finished book went out as application/octet-stream, which the app's web view
# (WebKit) won't play
mimetypes.add_type("audio/mp4", ".m4b")
mimetypes.add_type("audio/mp4", ".m4a")
mimetypes.add_type("audio/flac", ".flac")
mimetypes.add_type("audio/ogg", ".ogg")
mimetypes.add_type("audio/webm", ".webm")
mimetypes.add_type("audio/opus", ".opus")
mimetypes.add_type("audio/aac", ".aac")
router = APIRouter(prefix="/api")
# the desktop app's web view (Tauri) and its dev server; nothing else may call the API from a browser
ORIGINS = r"^(tauri://localhost|https?://tauri\.localhost|https?://(localhost|127\.0\.0\.1)(:\d+)?)$"


WEB_UI = Path(__file__).resolve().parent.parent / "desktop" / "dist"  # the desktop app's interface, once built


def install(app: FastAPI):
    from .guard import Guard
    app.add_middleware(Guard)  # (added first, so it runs inside CORS: the app's window can read a refusal)
    app.add_middleware(CORSMiddleware, allow_origin_regex=ORIGINS, allow_methods=["*"], allow_headers=["*"])
    from . import spend
    tasks.recover()   # tasks a crash or a quit cut off: noted in Activity, with how to carry on
    spend.on_start()  # cloud jobs a crash or a quit left running with nothing to come back for: stopped, and noted
    from . import api_manage
    app.include_router(router)
    app.include_router(api_manage.router)  # Voices, Library, Models, Connections and Train
    from . import cloud_book
    cloud_book.watch()  # cloud jobs are followed (and their lines brought back) even with no book page open
    from . import updater
    if settings.load().get("auto_update", False):
        updater.watch()  # new versions looked for every 6 hours, downloaded, installed when Babble quits

    def carry_on():  # training data that was being made when Babble quit (a few seconds in: the app opens first)
        from .dataset import pipeline
        pipeline.resume_interrupted(log=lambda m: print(m, flush=True))
    import threading
    threading.Timer(3, carry_on).start()
    if WEB_UI.is_dir():  # the same interface in any browser: http://127.0.0.1:7895/app/
        from fastapi.staticfiles import StaticFiles
        app.mount("/app", StaticFiles(directory=WEB_UI, html=True), name="web-ui")


def _start(kind, title, make_events):
    return tasks.start(kind, title, make_events).info()


@router.get("/update")
def update_status(check: bool = False):
    """Settings → Updates: this version, the newest release, and whether one is downloaded and ready."""
    from . import updater
    return updater.status(force=check)


@router.post("/update/download")
def update_download():
    from . import updater
    return _start("update", "Downloading an update", lambda: updater.download())


class AutoUpdate(BaseModel):
    on: bool


@router.post("/update/auto")
def update_auto(req: AutoUpdate):
    from . import updater
    updater.set_auto(req.on)
    return updater.status()


@router.get("/checks")
def checks():
    """What's missing for Babble to work fully (the app shows it when it opens)."""
    from . import checks as c
    return {"problems": c.run()}


def _ok(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


# ---------- about this app and Mac ----------

_seen_origins = set()


@router.get("/health")
def health(request: Request):
    origin = request.headers.get("origin") or "same page"
    if origin not in _seen_origins:  # one line in the log per kind of client, e.g. the desktop app's window
        _seen_origins.add(origin)
        print(f"[api] connected: {origin}", flush=True)
    from .tts.base import ram_gb
    return {"ok": True, "app": "Babble", "api": API_VERSION, "ram_gb": round(ram_gb())}


@router.get("/engines")
def list_engines():
    from . import library
    from .api_manage import GROUP
    cached = library._cached_repos()
    out = []
    for e in engines.ENGINES.values():
        ok, why = e.available()
        out.append({"id": e.id, "name": e.name, "short": e.short, "tier": e.tier, "license": e.license,
                    "min_ram_gb": e.min_ram_gb, "available": ok, "why": why, "note": e.note(),
                    "locked": not ok and e.min_ram_gb > 8, "group": GROUP.get(e.id, "standard"),
                    "downloaded": all(m in cached for m in e.hf_models)})
    return out


@router.get("/voices")
def list_voices():
    """Every voice, including ones this Mac can't run (locked: show them faded, with why)."""
    status = {e.id: e.available() for e in engines.ENGINES.values()}
    cloud = {e.id for e in engines.ENGINES.values() if e.cloud}
    out = []
    for v in engines.all_voices(only_available=False):
        if engines.is_off(v.engine):
            continue  # (turned off on the Models page: not shown at all)
        ok, why = status.get(v.engine, (True, ""))
        # cloud: this Mac can't run it, but Create in the cloud can make its lines (the cast can still pick it)
        out.append({"value": v.value, "name": v.name, "label": vl.short_name(v.value), "gender": v.gender,
                    "description": v.description, "engine": v.engine, "tags": list(v.tags),
                    "available": ok, "why": why, "cloud": not ok and v.engine in cloud})
    return out


@router.get("/usage")
def get_usage():
    """Memory and disk Babble is using (the sidebar's meters), in bytes."""
    from . import usage
    return {"ram": usage.memory(), "storage": usage.storage()}


def _dir_size(path: Path) -> int:
    import os as _os
    if not path.exists():
        return 0
    total = 0
    stack = [str(path)]
    while stack:
        cur = stack.pop()
        try:
            with _os.scandir(cur) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def _orphaned_book_folders():
    """Book output folders not referenced by any saved project and without an active cloud run."""
    from . import cloud_book
    active_folders = {p.get("folder") for p in projects.everything() if p.get("folder")}
    orphans = []
    if books.BOOKS.exists():
        for d in sorted(books.BOOKS.iterdir()):
            if not d.is_dir() or d.name.startswith("_") or d.name in active_folders:
                continue
            rec = cloud_book.read(d) or {}
            if rec.get("state") in cloud_book.WAITING + ("downloading",):
                continue  # never treat an active cloud run folder as an orphan
            orphans.append(d)
    return orphans


_STORAGE_CACHE = {"at": 0.0, "data": None}


@router.get("/storage")
async def storage_breakdown():
    """Detailed disk usage breakdown for Settings -> Storage."""
    import time as _time
    now = _time.monotonic()
    if _STORAGE_CACHE["data"] is not None and now - _STORAGE_CACHE["at"] < 5.0:
        return _STORAGE_CACHE["data"]
    data = await asyncio.to_thread(_storage_breakdown_sync)
    _STORAGE_CACHE["at"] = _time.monotonic()
    _STORAGE_CACHE["data"] = data
    return data


def _storage_breakdown_sync():
    from . import sfx, usage
    previews_dir = en.OUTPUTS / "voice-previews"
    uploads_dir = en.OUTPUTS / "uploads"
    history_files = list(en.OUTPUTS.glob("*.wav"))
    orphans = _orphaned_book_folders()
    orphan_names = {d.name for d in orphans}
    active_books_bytes = sum(
        _dir_size(d) for d in books.BOOKS.iterdir()
        if d.is_dir() and not d.name.startswith("_") and d.name not in orphan_names
    ) if books.BOOKS.exists() else 0
    orphan_bytes = sum(_dir_size(d) for d in orphans)
    disk = shutil.disk_usage(str(en.OUTPUTS))
    return {
        "books": active_books_bytes,
        "orphans": {"bytes": orphan_bytes, "count": len(orphans), "folders": [d.name for d in orphans]},
        "previews": {"bytes": _dir_size(previews_dir), "count": len(list(previews_dir.glob("*.wav"))) if previews_dir.exists() else 0},
        "uploads": {"bytes": _dir_size(uploads_dir), "count": sum(1 for f in uploads_dir.rglob("*") if f.is_file()) if uploads_dir.exists() else 0},
        "history": {"bytes": sum(f.stat().st_size for f in history_files if f.is_file()), "count": len(history_files)},
        "sounds": _dir_size(sfx.LIB),
        "models": usage._models_used(),
        "free": disk.free,
        "total": disk.total,
    }


class CleanStorageReq(BaseModel):
    target: str  # "previews" | "uploads" | "orphans" | "history"


@router.post("/storage/clean")
def storage_clean(req: CleanStorageReq):
    """Reclaim disk space for a specific safe cache category."""
    import time
    freed = 0
    removed = 0
    if req.target == "previews":
        d = en.OUTPUTS / "voice-previews"
        if d.exists():
            for f in d.glob("*.wav"):
                try:
                    freed += f.stat().st_size
                    f.unlink()
                    removed += 1
                except OSError:
                    pass
    elif req.target == "uploads":
        d = en.OUTPUTS / "uploads"
        # Keep files referenced as covers by existing projects, saved/active STT transcripts, or modified in the last 10 minutes
        from . import stt
        kept = set(stt.protected_uploads())
        for p in projects.everything():
            if p.get("cover"):
                try:
                    kept.add(Path(p["cover"]).resolve())
                except OSError:
                    pass
        cutoff = time.time() - 600
        if d.exists():
            for f in d.rglob("*"):
                if not f.is_file():
                    continue
                try:
                    if f.resolve() in kept or f.stat().st_mtime > cutoff:
                        continue
                    freed += f.stat().st_size
                    f.unlink()
                    removed += 1
                except OSError:
                    pass
    elif req.target == "orphans":
        for d in _orphaned_book_folders():
            freed += _dir_size(d)
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    elif req.target == "history":
        for f in en.OUTPUTS.glob("*.wav"):
            try:
                freed += f.stat().st_size
                f.unlink()
                removed += 1
            except OSError:
                pass
    else:
        raise HTTPException(400, "Unknown cleanup target.")
    from . import usage
    usage._storage["at"] = 0.0  # refresh sidebar storage meter immediately
    return {"target": req.target, "removed": removed, "freed": freed}


@router.get("/activity")
def get_activity():
    """Recent activity, newest first: what's running now (from the tasks), then what was made, cloud runs (with
    their GPU time and cost) and failures."""
    from . import activity, tasks
    now = []
    for t in tasks.all_tasks():
        if t.state != "running":
            continue
        kind = activity._kind(t)
        if kind == "other" or t.title.startswith(activity.PREVIEW):
            continue
        now.append({"id": t.id, "kind": kind, "title": activity.title_of(t), "state": "running", "status": t.status,
                    "started": t.started, "progress": t.progress, "eta": t.time_left()})
    return {"running": now, "entries": activity.entries()}


@router.delete("/activity")
def clear_activity():
    from . import activity
    activity.clear()
    return {"cleared": True}


@router.get("/settings")
def get_settings():
    return settings.public()  # never hand out API keys or tokens


@router.post("/settings")
def set_settings(changes: dict):
    known = set(settings.DEFAULTS)
    bad = sorted(set(changes) - known)
    if bad:
        raise HTTPException(400, f"Unknown settings: {', '.join(bad)}")
    from . import agents
    if agents.current():  # (an AI agent: the switch, the limits and connection settings stay the owner's)
        locked = sorted(k for k in changes if agents.locked_setting(k))
        if locked:
            raise HTTPException(403, f"Only you can change these in Babble: {', '.join(locked)}.")
    validated = {}
    for k, v in changes.items():
        default = settings.DEFAULTS[k]
        if isinstance(default, bool):
            if not isinstance(v, bool):
                raise HTTPException(400, f"Setting '{k}' must be a boolean.")
        elif isinstance(default, (int, float)) and not isinstance(default, bool):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise HTTPException(400, f"Setting '{k}' must be a number.")
            if k in ("spend_day_limit", "agent_spend_day_limit") and not (0 <= float(v) <= 10000):
                raise HTTPException(400, f"{k} must be between 0 and 10000.")
        elif isinstance(default, str) and v is not None and not isinstance(v, str):
            raise HTTPException(400, f"Setting '{k}' must be a string.")
        validated[k] = v
    settings.save(**validated)
    return get_settings()


@router.get("/keychain")
def keychain():
    """Where the API keys are kept (vault.py): the Keychain when the app started the engine."""
    from . import vault
    return vault.status()


# ---------- Spending (spend.py): what paid work cost, the day limit, stop everything ----------

@router.get("/spending")
def spending():
    from . import spend
    return spend.summary()


@router.get("/spending/history")
def spending_history(limit: int = 100):
    """Everything that cost money, newest first: one row per paid job (Settings → Spending)."""
    from . import spend
    return spend.history(max(1, min(int(limit), 500)))


class SpendLimit(BaseModel):
    day: Optional[float] = None     # dollars a day across paid services; 0: no limit
    agents: Optional[float] = None  # what AI agents' paid work may cost a day (agents.py); 0: none at all


@router.post("/spending/limit")
def spending_limit(req: SpendLimit):
    from . import spend
    for v in (req.day, req.agents):
        if v is not None and not 0 <= v <= 10000:
            raise HTTPException(400, "Pick a limit between $0 and $10,000 a day.")
    if req.day is not None:
        settings.save(spend_day_limit=round(max(req.day, 0.01), 2) if req.day > 0 else 0.0)  # (never rounded to 0: none)
    if req.agents is not None:
        settings.save(agent_spend_day_limit=round(req.agents, 2))
    return spend.summary()


# ---------- AI agents (agents.py, mcp_server.py): the switch, how to set one up, what the engine can do ----------

class AgentsSwitch(BaseModel):
    on: bool


@router.get("/agents")
def agents_info():
    from . import agents, spend
    return {"on": agents.on(), "limit": spend.agent_limit(), "setup": agents.setup()}


@router.post("/agents")
def agents_switch(req: AgentsSwitch):  # (the guard keeps agents themselves from it)
    settings.save(agents_on=bool(req.on))
    return agents_info()


@router.get("/agents/routes")
def agents_routes():
    """Every engine route with its first doc line (for an agent's call_api)."""
    from . import api_manage
    out = []
    for r in [*router.routes, *api_manage.router.routes]:
        path, methods = getattr(r, "path", ""), sorted(getattr(r, "methods", None) or [])
        if not path.startswith("/api") or path in ("/api/agents/routes",):
            continue
        doc = ((getattr(r, "endpoint", None) and r.endpoint.__doc__) or "").strip().split("\n")[0]
        for m in methods:
            if m not in ("HEAD", "OPTIONS"):
                out.append({"method": m, "path": path, "about": doc[:160]})
    return sorted(out, key=lambda x: (x["path"], x["method"]))


@router.post("/spending/stop-all")
def spending_stop_all():
    """Stop every task that spends money, every cloud job, and every book's cloud run."""
    from . import spend
    return {"stopped": spend.stop_all(), **spend.summary()}


# ---------- text to speech ----------

class Speech(BaseModel):
    text: str
    voice: str
    style: str = ""
    speed: float = 1.0
    stability: float = 55
    language: str = "english"
    seed: int = 0
    exact: bool = False
    max_chars: int = 250
    pause: float = 0.5
    batch: int = 2
    preview: bool = False
    parallel: Optional[bool] = None  # several lines at once (the page's switch); None: as each model's default


@router.post("/speech")
def speech(req: Speech):
    a = req.model_dump()
    t = tasks.start("speech", "Preview" if req.preview else "Speech", lambda: books.speak(**a))
    t.detail = " ".join((req.text or "").split())[:90]  # (the Activity page shows the words)
    return t.info()


PREVIEW_LINE = "Hi, welcome to Babble. My name's {name}. What can I do for you?"


class VoicePreview(BaseModel):
    voice: str
    text: Optional[str] = None
    language: str = "english"


def _preview_file(voice, text, language="english"):
    import hashlib
    return en.OUTPUTS / "voice-previews" / f"{hashlib.sha1(f'{voice}|{language}|{text}'.encode()).hexdigest()[:16]}.wav"


@router.post("/voices/preview")
def voice_preview(req: VoicePreview):
    """A voice reading the highlighted paragraph (when text is given) or saying hello: made once per
    (voice, language, text) and cached. Returns {"audio": path} when cached, else a task."""
    v = next((x for x in engines.all_voices(only_available=False) if x.value == req.voice), None)
    if v is None:
        raise HTTPException(404, "No such voice.")
    custom = books.clean_sample_text(req.text, max_chars=3000) if req.text else ""
    text = custom or PREVIEW_LINE.format(name=v.name.split(" (")[0])
    lang = req.language or "english"
    out = _preview_file(req.voice, text, lang)
    if out.exists():
        try:
            import os as _os
            _os.utime(out)
        except OSError:
            pass
        return {"audio": str(out)}

    def work():
        made = None
        for ev in books.speak(text, req.voice, language=lang, preview=not bool(custom)):
            made = ev.get("audio") or made
            yield {k: x for k, x in ev.items() if k != "audio"}
        if not made or not Path(made).exists():
            raise ValueError("The preview didn't come out. Try again.")
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(made, out)  # (kept for next time, and out of Text to Speech's history)
        _prune_previews(out.parent)
        yield {"audio": str(out), "status": "Done"}

    return _start("speech", f"Preview {v.name}", work)


def _prune_previews(folder: Path, max_files: int = 100, max_bytes: int = 200 * 1024 * 1024):
    """Keep voice-previews bounded by removing oldest files once count or total size exceeds limits."""
    try:
        files = sorted((f for f in folder.glob("*.wav") if f.is_file()), key=lambda f: f.stat().st_mtime, reverse=True)
        total = 0
        for i, f in enumerate(files):
            sz = f.stat().st_size
            if i >= max_files or total + sz > max_bytes:
                f.unlink(missing_ok=True)
            else:
                total += sz
    except OSError:
        pass


@router.get("/history")
def history(limit: int = 40):
    """Recent speech, previews and voice checks (newest first); finished books are in /api/books."""
    import datetime as dt
    out = []
    for f in sorted(en.OUTPUTS.glob("*.wav"), key=lambda f: f.stat().st_mtime, reverse=True)[:limit]:
        try:
            seconds = round(books.sf_duration(f), 3)  # the app rounds down, like a player
        except Exception:  # noqa: BLE001 - a half-written file
            seconds = None
        out.append({"path": str(f), "kind": f.stem.split("-")[0], "partial": f.stem.endswith(("-partial", "-gaps")),
                    "time": dt.datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="seconds"),
                    "seconds": seconds})
    return out


@router.delete("/history")
def delete_history(path: str):
    """Delete a single recent speech/preview WAV directly inside outputs/ to reclaim disk space."""
    root = en.OUTPUTS.resolve()
    p = Path(path).resolve()
    if p.parent != root or p.suffix.lower() != ".wav":
        raise HTTPException(403, "Only recent WAV files in outputs can be deleted here.")
    if not p.is_file():
        raise HTTPException(404, "That file is already gone.")
    p.unlink(missing_ok=True)
    return {"deleted": str(p)}


# ---------- speech to text (stt.py) ----------

class TranscribeReq(BaseModel):
    path: str
    model: str = "whisper-large-v3-turbo"
    language: str = "auto"
    timestamps: bool = True
    filename: str = ""


@router.get("/stt/models")
def stt_models():
    from . import stt
    return {"models": stt.list_models(), "languages": stt.list_languages()}


@router.delete("/stt/models/{model_id}")
def stt_delete_model(model_id: str):
    from . import stt
    return _ok(stt.delete_model, model_id)


@router.get("/stt/mics")
def stt_mics():
    from . import stt
    return stt.list_mics()


class RecordStartReq(BaseModel):
    device: Optional[int] = None


@router.post("/stt/record/start")
def stt_record_start(req: RecordStartReq):
    from . import stt
    return _ok(stt.record_start, req.device)


@router.get("/stt/record/status")
def stt_record_status():
    from . import stt
    return stt.record_status()


@router.post("/stt/record/stop")
def stt_record_stop():
    from . import stt
    return _ok(stt.record_stop)


@router.get("/stt/history")
def stt_history(limit: int = 30):
    from . import stt
    return stt.list_history(limit=limit)


@router.delete("/stt/history/{tid}")
def stt_delete_history(tid: str):
    from . import stt
    return {"deleted": _ok(stt.delete_history, tid)}


class ExportReq(BaseModel):
    fmt: str = "txt"


@router.post("/stt/history/{tid}/export")
def stt_export_history(tid: str, req: ExportReq):
    from . import stt
    return {"path": _ok(stt.export_transcript, tid, req.fmt)}


@router.post("/stt/transcribe")
def stt_transcribe(req: TranscribeReq):
    from . import stt
    if req.model not in stt.MODELS_BY_ID:
        raise HTTPException(400, f"Unknown speech-to-text model: {req.model}")
    p = Path(req.path).resolve()
    if not any(p == r or r in p.parents for r in _roots()):
        raise HTTPException(403, "Only uploaded or app audio files can be transcribed.")
    if not p.is_file():
        raise HTTPException(404, "Audio file not found.")
    label = (req.filename or p.name)[:60]
    t = tasks.start(
        "speech",
        f"Transcribe {label}",
        lambda: stt.transcribe(str(p), req.model, req.language, req.timestamps, req.filename or p.name),
    )
    return t.info()


@router.get("/books")
def list_books():
    """Audiobook folders, newest first, with the combined file (the chapters are NNN-key.wav beside it)."""
    import re
    out = []
    for d in sorted((p for p in books.BOOKS.glob("*") if p.is_dir() and not p.name.startswith("_")),
                    key=lambda p: p.stat().st_mtime, reverse=True):
        final = sorted((f for f in d.iterdir() if f.suffix in (".m4b", ".mp3", ".wav")
                        and not re.match(r"\d{3}-", f.name)), key=lambda f: f.stat().st_mtime, reverse=True)
        named = final[0].stem if final else d.name
        out.append({"title": named.replace("-", " "), "folder": str(d), "name": d.name,
                    "file": str(final[0]) if final else None,
                    "chapters": len([f for f in d.glob("[0-9][0-9][0-9]-*.wav")])})
    return out


@router.delete("/books/{name}")
def delete_book_folder(name: str):
    """Delete an audiobook output folder inside outputs/books/ to reclaim disk space."""
    if not books.FOLDER_NAME.fullmatch(name):
        raise HTTPException(400, "Invalid book folder name.")
    root = books.BOOKS.resolve()
    d = (root / name).resolve()
    if d.parent != root or not d.is_dir():
        raise HTTPException(404, "That audiobook folder is already gone.")
    shutil.rmtree(d)
    return {"deleted": name}


# ---------- audiobooks ----------

class CastRow(BaseModel):
    character: str
    voice: str            # a voice label as /api/voices gives it ("Tara (Orpheus)") or its value
    style: str = ""
    speed: float = 1.0
    stability: Optional[float] = None  # this character's own stability, 0-100 (else the book's)


class Book(BaseModel):
    story: str
    cast: list[CastRow]
    models: str = "cast"
    quotes_only: bool = True
    speed: float = 1.0
    stability: float = 55
    language: str = "english"
    seed: int = 0
    batch: int = 2
    parallel: Optional[bool] = None  # several lines at once (the book's switch); None: as each model's default
    pronunciations: list[dict] = []  # this book's own words said another way [{"word", "say"}]
    line_pause: float = 0.2
    para_pause: float = 0.7
    out_format: str = "WAV"
    title: str = ""
    author: str = ""
    cover: Optional[str] = None
    sound: bool = True
    bed: float = 1.0
    fx_level: Optional[float] = None     # the volume sliders: dB under the narrator while they speak (None: as always)
    amb_level: Optional[float] = None
    music_level: Optional[float] = None
    intro: str = ""                      # (books from before chapter_music: one intro for every chapter)
    chapter_music: Optional[dict] = None  # each chapter's intro and outro (chapter_music.py; the Intros tab)
    intro_level: Optional[float] = None  # the intros' and outros' volume: dB under the narrator (None: as always)
    automake: bool = True
    preview: bool = False
    preview_text: Optional[str] = None     # optional selected text to audition in Preview
    preview_chapter: Optional[int] = None  # optional chapter index to preview when no text is selected
    folder: Optional[str] = None  # an Audiobook project's own folder in outputs/books
    make_on: str = "mac"          # Create in the cloud: "kaggle" or "modal" (cloud_book)
    cloud_sounds: bool = True     # ...and its missing AI sounds made there too (else on this Mac)
    cloud_gpu: str = "A10G"       # the Modal GPU
    cloud_live: bool = True       # copy the lines back while the cloud works, where the service allows it
    cloud_hours: Optional[float] = None  # the most one run may take (3, 6 or 12 h; none: 3), not on Kaggle
    name: str = ""                # the combined file's name (an untitled project's "Untitled-3")
    book: Optional[str] = None    # the project id: tags can use that book's own sounds
    new_sounds: str = "all"       # sounds made for the book: "book" keeps them in it, "all" shares them


def _book_args(req: Book, cloud=False):
    a = req.model_dump()
    if not cloud:  # where it's made only matters to Create in the cloud
        a.pop("make_on", None)
        a.pop("cloud_sounds", None)
        a.pop("cloud_gpu", None)
        a.pop("cloud_live", None)
        a.pop("cloud_hours", None)
    cast = a.pop("cast")
    a["bed"] = {"bed": a["bed"], "sfx": a.pop("fx_level"), "amb": a.pop("amb_level"), "music": a.pop("music_level")}
    music, level = a.pop("chapter_music") or {}, a.pop("intro_level")
    if music.get("chapters") or a.get("intro"):
        a["intro"] = {"chapters": music.get("chapters") or [], "id": a.get("intro") or "", "level": level}
    else:
        a["intro"] = ""
    a["table"] = [[r["character"], vl.short_name(r["voice"]) if ":" in r["voice"] else r["voice"], r["style"], r["speed"]]
                  for r in cast]
    a["voice_stability"] = {r["character"]: r["stability"] for r in cast if r.get("stability") is not None}
    return a


@router.get("/books/options")
def book_options():
    from . import export
    return {"models": [{"label": label, "value": v} for label, v in books.model_choices()],
            "formats": export.FORMATS, "languages": books.LANGUAGES}


class Hear(BaseModel):
    story: str = ""
    character: str
    voice: str
    style: str = ""
    speed: float = 1.0
    stability: float = 55
    language: str = "english"
    cast: list[CastRow] = []
    models: str = "cast"
    sample_text: Optional[str] = None
    pronunciations: list[dict] = []


@router.post("/books/hear")
def hear(req: Hear):
    """Read the highlighted paragraph (when sample_text is given) or a character's own line from the story."""
    voice = vl.short_name(req.voice) if ":" in req.voice else req.voice
    table = [[r.character, vl.short_name(r.voice) if ":" in r.voice else r.voice, r.style, r.speed] for r in req.cast]
    return _start("speech", f"Hear {req.character}", lambda: books.hear_character(
        req.story, req.character, voice, req.style, req.speed, req.stability, req.language, table, req.models,
        sample_text=req.sample_text, pronunciations=req.pronunciations))


class Direct(BaseModel):
    story: str
    where: str = "mac"   # "mac" (the local model) or an AI service: anthropic, openai, gemini, openrouter, …
    model: str = ""
    checks: int | None = None  # 1 loose … 5 strict, as director.CHECKS (the slider shows it reversed: more sounds to the right)
    notes: str = ""            # the book's notes for the director ("dark gloomy spooky")
    book: str = ""             # the audiobook it's for (its own sounds, and its "Make the sounds it asks for")
    scenes: str | None = None  # scene size on a service: small (more requests, more detail) … large (director.SCENE_SIZES)


@router.post("/books/direct")
def direct_sounds(req: Direct):
    from . import director
    return _start("direct", "Direct sounds", lambda: director.direct(req.story, req.where, req.model, req.checks, req.notes, req.book,
                                                                          req.scenes))


@router.post("/director/estimate")
def director_estimate(req: Direct):
    from . import director
    return director.estimate(req.story, req.where, req.model, req.scenes)


@router.get("/director/providers")
def director_providers():
    from . import director
    return director.providers()


@router.get("/director/models")
def director_models(provider: str):
    from . import director
    return _ok(director.models, provider)


class AiKey(BaseModel):
    provider: str
    key: str = ""


@router.post("/director/key")
def director_key(req: AiKey):
    from . import director
    return _ok(director.save_key, req.provider, req.key)


@router.post("/director/forget")
def director_forget(req: AiKey):
    from . import director
    director.forget_key(req.provider)
    return {"ok": True}


class Apply(BaseModel):
    story: str
    planned_on: str
    placements: list[dict]


@router.post("/books/suggest/apply")
def apply_suggestions(req: Apply):
    story, message = _ok(books.apply_suggestions, req.story, req.planned_on, req.placements)
    return {"story": story, "message": message}


@router.post("/books/made")
def book_made(req: Book):
    """Whether every chapter is made already (Create would only put the book together again)."""
    return books.made(**_book_args(req))


@router.post("/books/remake")
def book_remake(req: Book):
    """The book's current take goes to the Trash so Create speaks it again: {"moved", "seed": the seed to use}."""
    a = _book_args(req)
    return books.remake(title=a.get("title", ""), story=a.get("story", ""), folder=a.get("folder"), seed=a.get("seed", 0))


_making = {}  # project id (or folder) -> the Create task making it
_making_lock = threading.Lock()


@router.post("/books/create")
def create_book(req: Book):
    a = _book_args(req)
    title = "Book preview" if req.preview else f"Audiobook: {req.title or 'Story'}"
    key = None if req.preview else (req.book or req.folder)
    if not key:
        return _start("book", title, lambda: books.create(**a))
    with _making_lock:  # one Create per book: a second click while it's being made gets the same run
        t = tasks.get(_making.get(key, ""))
        if t and t.state == "running":
            return t.info()
        out = _start("book", title, lambda: books.create(**a))
        _making[key] = out["id"]
        return out


@router.post("/books/kaggle")
def kaggle_bundle(req: Book):
    a = _book_args(req)
    for k in ("voice_stability", "book", "new_sounds"):  # not used on Kaggle
        a.pop(k, None)
    path, message = _ok(books.kaggle_bundle, **a)
    return {"path": path, "message": message}


# ---------- Create in the cloud (cloud_book): the lines are spoken on Kaggle, the book is put together here ----------

@router.post("/books/cloud/send")
def cloud_send(req: Book):
    from . import cloud_book
    from .api_manage import _logged
    if not req.folder:
        raise HTTPException(400, "Only saved Audiobook projects can be made in the cloud.")
    a = _book_args(req, cloud=True)
    return _start("cloud", f"Send to the cloud: {req.title or 'Story'}",
                  _logged(cloud_book.send, done=lambda rec: {"value": "sent", "cloud": rec}, **a))


@router.post("/books/cloud/estimate")
def cloud_estimate(req: Book):
    """Before sending: about how long, and what it may cost (low to worst case)."""
    from . import cloud_book
    if not req.folder:
        raise HTTPException(400, "Only saved Audiobook projects can be made in the cloud.")
    out = _ok(cloud_book.estimate, **_book_args(req, cloud=True))
    out["busy"] = cloud_book.busy(req.folder)  # already out there (a double click, or sent before): asked first
    return out


class StripTags(BaseModel):
    story: str
    keep: list[str] = []


@router.post("/books/strip-tags")
def strip_speaker_tags(req: StripTags):
    """Clear cast: the story without the speaker tags of anyone not kept (sound tags stay)."""
    from .cast import strip_tags
    return {"story": strip_tags(req.story, req.keep)}


def _cloud_folder(folder: str):
    if not folder:
        raise HTTPException(400, "Which book?")
    return _ok(books.book_folder, "", "", folder)


@router.get("/books/cloud")
def cloud_status(folder: str):
    from . import cloud_book
    return cloud_book.view(_cloud_folder(folder))


@router.post("/books/cloud/check")
def cloud_check(folder: str):
    from . import cloud_book
    where = _cloud_folder(folder)
    cloud_book.check(where, force=True)
    return cloud_book.view(where)


@router.delete("/books/cloud")
def cloud_forget(folder: str):
    """Stop the cloud job, or stop waiting for it (the lines already brought back stay): {"stopped", "back", "lines"}."""
    from . import cloud_book
    return cloud_book.forget(_cloud_folder(folder))


@router.post("/books/cloud/matches")
def cloud_matches(req: Book):
    """How the book compares with its last cloud run's settings (asked before Finish and Create in the cloud)."""
    from . import cloud_book
    if not req.folder:
        return {}
    return _ok(cloud_book.matches, **_book_args(req))


@router.post("/books/cloud/start-again")
def cloud_start_again(folder: str):
    """After a run is stopped: its lines and the chapters made from them go to the Trash, to be made again."""
    from . import cloud_book
    return _ok(cloud_book.start_again, _cloud_folder(folder))


# ---------- Check lines (line_check): cloud lines that came out wrong, set aside to be made again ----------

@router.post("/books/lines/check")
def lines_check(req: Book):
    from . import line_check
    if not req.folder:
        raise HTTPException(400, "Only saved Audiobook projects have cloud lines to check.")
    a = _book_args(req)
    return _start("lines", f"Check lines: {req.title or 'Story'}", lambda: line_check.check(**a))


@router.get("/books/lines/check")
def lines_checked(folder: str):
    """The last check, and how many lines from the cloud the book has (0: nothing to check)."""
    from . import line_check
    where = _cloud_folder(folder)
    return dict(line_check.view(where), cloud_lines=sum(1 for _ in (where / books.CLOUD_LINES).glob("*.flac")))


class LineKeys(BaseModel):
    keys: list[str]


@router.post("/books/lines/aside")
def lines_aside(folder: str, req: LineKeys):
    from . import line_check
    return _ok(line_check.set_aside, _cloud_folder(folder), req.keys)


@router.post("/books/lines/put-back")
def lines_put_back(folder: str):
    from . import line_check
    return _ok(line_check.put_back, _cloud_folder(folder))


# ---------- Two-Stage Immersion QA (immersion_qa.py) ----------

class QaRun(Book):
    stage2: bool = False
    where: str = ""
    model: str = ""


class QaEstimate(BaseModel):
    story: str
    where: str = ""
    model: str = ""


@router.get("/books/qa")
def qa_get(folder: str):
    from . import immersion_qa
    return immersion_qa.read(_cloud_folder(folder))


@router.post("/books/qa/estimate")
def qa_estimate(req: QaEstimate):
    from . import immersion_qa
    return _ok(immersion_qa.estimate_stage2, req.story, req.where, req.model)


@router.post("/books/qa/run")
def qa_run(req: QaRun):
    from . import immersion_qa
    a = _book_args(req)
    s2, where, model = a.pop("stage2", False), a.pop("where", ""), a.pop("model", "")
    return _start("lines", f"Immersion QA: {req.title or 'Story'}",
                  lambda: immersion_qa.run_qa(stage2=s2, where=where, model=model, **a))


class Chapters(BaseModel):
    story: str
    folder: Optional[str] = None
    title: str = ""
    quotes_only: bool = True


@router.post("/books/chapters")
def chapters(req: Chapters):
    """The story's chapters now, and which have audio for their current words (from the last Create)."""
    return _ok(books.chapters_status, req.story, req.folder, req.title, req.quotes_only)


class RemoveChapter(BaseModel):
    story: str
    index: int
    title: str
    quotes_only: bool = True


@router.post("/books/chapters/remove")
def remove_chapter(req: RemoveChapter):
    """The story without one of its chapters (or one part of a long chapter)."""
    return {"story": _ok(books.without_chapter, req.story, req.index, req.title, req.quotes_only)}


class ChapterTimings(BaseModel):
    story: str
    chapter: int
    file: str
    quotes_only: bool = True


@router.post("/books/timings")
def chapter_timings(req: ChapterTimings):
    """Per-paragraph start/end timestamps for a rendered chapter WAV (exact sidecar or proportional estimate)."""
    return _ok(books.chapter_timings, req.story, req.chapter, req.file, req.quotes_only)


class RegenParagraph(Book):
    chapter: int
    para: int
    seed_bump: int = 1


@router.post("/books/regen-paragraph")
def regen_paragraph(req: RegenParagraph):
    """Regenerate one paragraph in place, re-stitch its chapter, and return a surrounding-context preview clip."""
    a = _book_args(req)
    ch, pa, bump = a.pop("chapter"), a.pop("para"), a.pop("seed_bump", 1)
    return _start("speech", f"Regenerate paragraph {pa + 1}", lambda: books.regen_paragraph(ch, pa, bump, **a))


class StorySounds(BaseModel):
    story: str
    book: Optional[str] = None


@router.post("/books/sounds")
def story_sounds(req: StorySounds):
    """Every sound tag in the story with the sound it plays (the shared library's or the book's own), or none yet."""
    from . import sfx
    token = sfx.BOOK.set(req.book) if req.book else None
    try:
        out, seen = [], set()
        for m in sfx.CUE.finditer(req.story or ""):
            for kind, cue in sfx.split_cues(m.group()):
                if kind != "cue" or cue["type"] in ("room", "pause") or not cue.get("name"):
                    continue
                k = (cue["type"], cue["name"].lower())
                if k in seen:
                    continue
                seen.add(k)
                e = sfx.find_for(cue)
                out.append({"type": cue["type"], "name": cue["name"], "tag": f"{{{cue['type']}: {cue['name']}}}",
                            "sound": _with_path(e) if e else None})
        return out
    finally:
        if token is not None:
            sfx.BOOK.reset(token)


@router.get("/books/palette")
def sound_palette(book: Optional[str] = None):
    """The sounds an audiobook can use (every shared one plus its own), each with the tag that plays exactly it:
    what the book page lets you drag into the text."""
    from . import sfx
    token = sfx.BOOK.set(book) if book else None
    try:
        usable = sfx.pool()
        out = [dict(_with_path(e), tag=sfx.tag_for(e, usable), mine=bool(book and e.get("book") == book)) for e in usable]
        return sorted(out, key=lambda x: (not x["mine"], x["name"].lower()))
    finally:
        if token is not None:
            sfx.BOOK.reset(token)


class StoryCast(BaseModel):
    story: str
    cast: list[CastRow] = []
    method: str = "ai"     # "ai" or "quick"
    title: str = ""
    book: Optional[str] = None  # the project id: sounds made are kept in that book (scope "book") or shared
    scope: str = "all"
    make_on: str = "mac"   # Detect speakers on this Mac, or on a cloud GPU (cloud_ai)
    cloud_gpu: str = ""
    chapter_plan: Optional[dict] = None  # Make missing sounds: also the director's chapter music (chapter_music.plan)
    online_only: bool = False  # Make missing sounds: only the sound libraries' recorded sounds (AI ones go to the cloud)


@router.post("/books/detect")
def detect(req: StoryCast):
    table = [[r.character, r.voice, r.style, r.speed] for r in req.cast]
    return _start("detect", "Detect speakers", lambda: books.detect_speakers(req.story, table, req.method, req.make_on, req.cloud_gpu))


@router.post("/books/detect/estimate")
def detect_estimate(req: StoryCast):
    """Detect speakers on a cloud GPU: what it would cost and take (the confirmation before it's sent)."""
    from . import ai_detect, cloud_ai, detect_core
    chapters, texts = ai_detect.chapters_of(req.story)
    _, tokens = detect_core.prompt_tokens(texts, starts=ai_detect.starts_of(chapters))
    return _ok(cloud_ai.estimate, req.make_on, req.cloud_gpu, tokens)


@router.post("/books/stats")
def story_stats(req: StoryCast):
    return {"text": books.story_stats(req.story)}


@router.post("/books/missing-sounds")
def missing_sounds(req: StoryCast):
    return _start("sounds", "Make missing sounds",
                  lambda: books.make_missing(req.story, req.book, req.scope, req.chapter_plan, req.online_only))


# ---------- Intros (chapter_music.py): each chapter's intro and outro ----------

@router.post("/books/chapter-titles")
def chapter_titles(req: StoryCast):
    """The chapters Create makes, in order: the Intros tab's rows."""
    from . import chapter_music
    return {"chapters": chapter_music.titles(req.story)}


class DesignMusic(BaseModel):
    description: str
    kind: str = "intro"        # "intro", "sting" (a short outro) or "outro" (a soft closing phrase)
    book: Optional[str] = None
    scope: str = "all"


@router.post("/books/chapter-music/design")
def design_chapter_music(req: DesignMusic):
    from . import chapter_music
    return _start("sounds", f"Design {'an intro' if req.kind == 'intro' else 'an outro'}",
                  lambda: chapter_music.design(req.description, req.kind, req.book, req.scope))


MAX_IMPORT_BYTES = 100 * 1024 * 1024   # 100 MB
MAX_UPLOAD_BYTES = 500 * 1024 * 1024   # 500 MB


def _copy_capped(src, dst, max_bytes: int):
    total = 0
    while True:
        chunk = src.read(1 << 20)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(413, f"File exceeds maximum upload size ({max_bytes // (1024 * 1024)} MB).")
        dst.write(chunk)


@router.post("/books/import")
def import_book(file: UploadFile = File(...)):
    with tempfile.TemporaryDirectory(prefix="babble-import-") as td:
        tmp = Path(td) / (Path(file.filename or "book.txt").name or "book.txt")
        with open(tmp, "wb") as f:
            _copy_capped(file.file, f, MAX_IMPORT_BYTES)
        return _ok(books.import_file, tmp)


@router.get("/pronunciations")
def pronunciations():
    """The words every book and Text to Speech say another way."""
    from . import pronounce
    return pronounce.everyone()


class TrickyReq(BaseModel):
    story: str
    cast: list[str] = []
    existing: list[dict] = []


@router.post("/books/pronunciations/suggest")
def suggest_pronunciations(req: TrickyReq):
    """Scan the story for unusual names, recurring proper nouns, and homographs worth checking before render."""
    from . import pronounce
    return pronounce.find_tricky(req.story, req.cast, req.existing)


class Pronunciations(BaseModel):
    rows: list[dict] = []


@router.put("/pronunciations")
def save_pronunciations(req: Pronunciations):
    from . import pronounce
    return pronounce.save_everyone(req.rows)


@router.post("/files/upload")
def upload(file: UploadFile = File(...)):
    """A file for the app to use (e.g. a book cover). Kept in the outputs folder, so /api/files can show it."""
    import re
    import secrets
    import time
    folder = en.OUTPUTS / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(file.filename or "file").name)[-80:] or "file"
    dest = folder / f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}-{name}"
    try:
        with open(dest, "wb") as f:
            _copy_capped(file.file, f, MAX_UPLOAD_BYTES)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    return {"path": str(dest)}


@router.get("/books/intro-choices")
def intro_choices(book: Optional[str] = None):
    return books.intro_choices(book)


# ---------- audiobook projects (the Audiobook page's books) ----------

@router.get("/projects")
def list_projects():
    return [projects.summary(p) for p in projects.everything()]


@router.post("/projects")
def new_project(data: dict):
    return _ok(projects.create, data)


@router.get("/projects/{pid}")
def get_project(pid: str):
    return _ok(projects.get, pid)


@router.put("/projects/{pid}")
def save_project(pid: str, changes: dict):
    return projects.summary(_ok(projects.save, pid, changes))


@router.delete("/projects/{pid}")
def delete_project(pid: str, delete_audio: bool = False):
    _ok(projects.delete, pid, delete_audio=delete_audio)
    return {"deleted": pid, "deleted_audio": delete_audio}


# ---------- sound library ----------

def _with_path(entry):
    from . import sfx
    return dict(entry, path=str(sfx.LIB / entry["file"]))  # the full path, for /api/files


@router.get("/sounds")
def sounds():
    from . import sfx
    return [_with_path(e) for e in sfx.index()]


class MakeSound(BaseModel):
    description: str
    kind: str = "effect"          # effect, ambience, music
    quality: Optional[str] = None  # music: "small" or "medium"
    seconds: Optional[float] = None
    book: Optional[str] = None    # made in an audiobook (its project id)…
    scope: str = "all"            # …and kept in it ("book") or shared with every book ("all")


@router.post("/sounds/make")
def make_sound(req: MakeSound):
    from . import sfxgen
    if req.kind not in sfxgen.KINDS:
        raise HTTPException(400, f"kind must be one of {', '.join(sfxgen.KINDS)}")

    def work():
        e = books.keep_in(sfxgen.make(req.description.strip(), req.kind, seconds=req.seconds, quality=req.quality,
                                      log=tasks.note), req.book, req.scope)
        where = " to this book" if e.get("book") else ""
        return {"sound": _with_path(e), "status": f"Added {e['name']} ({e['kind']}, {e['duration']} s){where}."}
    return _start("sounds", f"Make “{req.description[:40]}”", work)


class Scope(BaseModel):
    book: Optional[str] = None  # a project id keeps the sound in that book; none shares it with every book


# ---------- sound packs and online sound libraries ----------

class Pack(BaseModel):
    path: str
    name: str = ""
    license: str = ""


@router.get("/sounds/packs")
def sound_packs():
    from . import sfx
    return sfx.packs()


@router.post("/sounds/packs")
def import_sound_pack(req: Pack):
    from . import sfx
    return _start("sounds", f"Add sound pack “{(req.name or Path(req.path).stem)[:40]}”",
                  lambda: sfx.import_pack(req.path, req.name, req.license))


class Pick(BaseModel):
    zip: bool = False


@router.post("/sounds/packs/pick")
def pick_pack(req: Pick):
    """Native chooser for a pack's folder (or .zip): {"path"}, or {"path": None} when cancelled."""
    import subprocess
    import sys
    if sys.platform == "darwin":
        what = ('choose file with prompt "Pick a sound pack (.zip)" of type {"zip", "public.zip-archive"}' if req.zip
                else 'choose folder with prompt "Pick a folder of sounds"')
        r = subprocess.run(["osascript", "-e", f"POSIX path of ({what})"], capture_output=True, text=True)
        return {"path": r.stdout.strip() or None}
    # Run tkinter in an isolated subprocess so Tk() always runs on its own main thread (never a FastAPI worker thread).
    code = (
        "import tkinter as tk; from tkinter import filedialog; "
        "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True); "
        + ("p = filedialog.askopenfilename(title='Pick a sound pack (.zip)', filetypes=[('ZIP archive', '*.zip')]); "
           if req.zip else "p = filedialog.askdirectory(title='Pick a folder of sounds'); ")
        + "r.destroy(); print(p or '', end='')"
    )
    kw = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)} if sys.platform == "win32" else {}
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", timeout=300, **kw)
        return {"path": r.stdout.strip() or None}
    except Exception:  # noqa: BLE001
        return {"path": None}


@router.delete("/sounds/packs/{name}")
def remove_sound_pack(name: str):
    from . import sfx
    return {"removed": sfx.remove_pack(name)}


@router.get("/sounds/online/sources")
def online_sources():
    from . import online_sounds
    return online_sounds.sources()


class SoundKey(BaseModel):
    key: str


@router.post("/sounds/online/{sid}/key")
def online_save_key(sid: str, req: SoundKey):
    from . import online_sounds
    return _ok(online_sounds.save_key, sid, req.key)


@router.delete("/sounds/online/{sid}/key")
def online_forget_key(sid: str):
    from . import online_sounds
    online_sounds.forget_key(sid)
    return {"ok": True}


@router.get("/sounds/online/search")
def online_search(q: str, kind: str = "effect"):
    from . import online_sounds
    if not online_sounds.ready():
        raise HTTPException(400, "Add a Freesound or Lots of Sounds key in Connections first.")
    found, errors = online_sounds.search(q.strip(), kind, n=12)
    if not found and errors:
        raise HTTPException(400, errors[0])
    return {"results": found, "errors": errors}


class OnlineAdd(BaseModel):
    sound: dict
    kind: str = "effect"
    description: str = ""


@router.post("/sounds/online/add")
def online_add(req: OnlineAdd):
    from . import online_sounds
    return _with_path(_ok(online_sounds.add_by_hand, req.sound, req.kind, req.description))


@router.post("/sounds/{sound_id}/scope")
def sound_scope(sound_id: str, req: Scope):
    from . import sfx
    return _with_path(_ok(sfx.set_scope, sound_id, req.book))


class SoundIds(BaseModel):
    ids: list[str]


@router.post("/sounds/delete")
def delete_sounds(req: SoundIds):
    """Delete several sounds at once (Sounds → select mode)."""
    from . import sfx
    return {"deleted": sfx.delete_many(req.ids)}


@router.delete("/sounds/{sound_id}")
def delete_sound(sound_id: str):
    from . import sfx
    if not any(e["id"] == sound_id for e in sfx.index()):
        raise HTTPException(404, "No such sound.")
    sfx.delete(sound_id)
    return {"deleted": sound_id}


# ---------- tasks ----------

def _task(task_id):
    t = tasks.get(task_id)
    if t is None:
        raise HTTPException(404, "No such task (tasks are forgotten when the app restarts).")
    return t


@router.get("/tasks")
def list_tasks():
    return [t.info() for t in tasks.all_tasks()]


@router.get("/tasks/{task_id}")
def task_info(task_id: str):
    return _task(task_id).info()


@router.post("/tasks/{task_id}/stop")
def stop_task(task_id: str):
    t = _task(task_id)
    t.stop_requested.set()
    return t.info()


@router.get("/tasks/{task_id}/events")
async def task_events(task_id: str, request: Request, after: int = 0):
    """Server-sent events: each event as `data: {"n": …, …}`, then a final `event: end` with the task's info."""
    t = _task(task_id)

    async def stream():
        n = after
        while True:
            if await request.is_disconnected():
                return
            for k, ev in tasks.events_after(t, n):
                n = k
                yield f"data: {json.dumps(dict(ev, n=k, progress=t.progress, eta=t.time_left(), updated=t.updated), default=str)}\n\n"
            if t.state != "running" and not tasks.events_after(t, n):
                yield f"event: end\ndata: {json.dumps(t.info(), default=str)}\n\n"
                return
            await asyncio.sleep(0.25)
    return StreamingResponse(stream(), media_type="text/event-stream")


# ---------- pre-flight mastering validator (preflight.py) ----------

class PreflightReq(BaseModel):
    target: str = "acx"  # "acx" | "inaudio" | "apple"


@router.post("/projects/{pid}/preflight")
def project_preflight(pid: str, req: PreflightReq):
    from . import preflight
    p = _ok(projects.get, pid)
    folder = p.get("folder")
    if not folder:
        raise HTTPException(400, "This book hasn't been built yet.")
    where = _ok(books.book_folder, "", "", folder)
    chapters = sorted(where.glob("[0-9][0-9][0-9]-*.wav"))
    if not chapters:
        raise HTTPException(400, "No rendered chapters found for this book.")
    return _ok(preflight.validate, chapters, req.target)


# ---------- series bible (series.py) ----------

class SeriesCreateReq(BaseModel):
    name: str
    cast: list = []
    pronunciations: list = []


@router.get("/series")
def list_series():
    from . import series
    return series.list_series()


@router.post("/series")
def create_series(req: SeriesCreateReq):
    from . import series
    return _ok(series.create, req.name, req.cast, req.pronunciations)


@router.get("/series/{sid}")
def get_series(sid: str):
    from . import series
    return _ok(series.get, sid)


@router.put("/series/{sid}")
def save_series(sid: str, changes: dict):
    from . import series
    return _ok(series.save, sid, changes)


@router.delete("/series/{sid}")
def delete_series(sid: str):
    from . import series
    _ok(series.delete, sid)
    return {"deleted": sid}


@router.post("/series/{sid}/sync/{pid}")
def sync_series_from_book(sid: str, pid: str):
    from . import series
    p = _ok(projects.get, pid)
    return _ok(series.sync_from_book, sid, p)


@router.post("/series/{sid}/apply/{pid}")
def apply_series_to_book(sid: str, pid: str):
    from . import series
    p = _ok(projects.get, pid)
    cast, pronunciations = _ok(series.apply_to_book, sid, p)
    return projects.summary(_ok(projects.save, pid, {"series_id": sid, "cast": cast, "pronunciations": pronunciations}))


# ---------- project time machine snapshots (snapshots.py) ----------

class SnapshotReq(BaseModel):
    label: str = "Snapshot"


@router.get("/projects/{pid}/snapshots")
def list_snapshots(pid: str):
    from . import snapshots
    return _ok(snapshots.list_for, pid)


@router.post("/projects/{pid}/snapshots")
def take_snapshot(pid: str, req: SnapshotReq):
    from . import snapshots
    return _ok(snapshots.take, pid, req.label)


@router.get("/projects/{pid}/snapshots/{sid}/diff")
def diff_snapshot(pid: str, sid: str):
    from . import snapshots
    return _ok(snapshots.diff, pid, sid)


@router.post("/projects/{pid}/snapshots/{sid}/restore")
def restore_snapshot(pid: str, sid: str):
    from . import snapshots
    return projects.summary(_ok(snapshots.restore, pid, sid))


# ---------- acoustic space presets (acoustics.py) ----------

@router.get("/acoustics/presets")
def acoustics_presets():
    from . import acoustics
    return [{"id": k, "label": v["label"], "rt60": v["rt60"], "wet": v["wet"]} for k, v in acoustics.PRESETS.items()]


# ---------- dataset health scorecard & auto-cure (dataset/health.py) ----------

def _dataset_dir(voice: str) -> Path:
    from .paths import TRAINING
    if not voice or Path(voice).name != voice or voice in (".", ".."):
        raise HTTPException(400, "Invalid dataset name.")
    root = TRAINING.resolve()
    target = (root / voice).resolve()
    if target.parent != root or not target.is_dir():
        raise HTTPException(404, "Dataset folder not found.")
    return target


@router.get("/train/dataset/{voice}/health")
def dataset_health(voice: str):
    from .dataset import health
    return _ok(health.score_dataset, _dataset_dir(voice))


@router.post("/train/dataset/{voice}/auto-cure")
def dataset_auto_cure(voice: str):
    from .dataset import health
    return _ok(health.auto_cure, _dataset_dir(voice))


# ---------- files ----------

def _roots():
    from . import sfx
    from .paths import MODELS, TRAINING, WORK
    # outputs, sounds and saved voices; plus training clips and results, and the packs' voice previews
    return [Path(r).resolve() for r in (en.OUTPUTS, sfx.LIB, vl.VOICES, TRAINING, WORK / "results", MODELS)]


INLINE_MEDIA_EXTS = {".wav", ".mp3", ".m4b", ".m4a", ".flac", ".ogg", ".webm", ".opus", ".aac", ".png", ".jpg", ".jpeg", ".webp", ".gif"}


@router.get("/files")
def files(path: str):
    p = Path(path).resolve()
    if not any(p == r or r in p.parents for r in _roots()):
        raise HTTPException(403, "Only files the app made (outputs, sounds, voices, training) can be fetched.")
    if not p.is_file():
        raise HTTPException(404, "No such file.")
    headers = {
        "X-Content-Type-Options": "nosniff",
        "Content-Security-Policy": "default-src 'none'; sandbox",
    }
    if p.suffix.lower() not in INLINE_MEDIA_EXTS:  # (Starlette writes the name so any characters work: — ’ é)
        return FileResponse(p, headers=headers, filename=p.name, content_disposition_type="attachment")
    return FileResponse(p, headers=headers)
