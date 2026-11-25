"""Babble for AI agents: an MCP server (stdio) that uses the running Babble engine for the owner.

Start it from an AI app (Claude, Cursor, …):  <repo>/.venv/bin/python <repo>/babble_mcp.py
(Settings → General shows the setup to copy.) It speaks MCP's JSON-RPC over stdin/stdout, one message a line, with
no packages beyond the app's own, and calls the engine on 127.0.0.1 with this run's key (<data folder>/engine-token)
and X-Babble-Client: mcp. The engine then applies the agents' rules (agents.py): only while the owner has AI agents
on; no keys, sign-ins or limits; paid work within the agents' own day limit as well as the owner's.

Logs go to stderr: stdout is the protocol.
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

from .paths import HOME, OUTPUTS

VERSION = "1.0"
PORT = int(os.environ.get("BABBLE_PORT") or 7895)
BASE = f"http://127.0.0.1:{PORT}"
PROTOCOLS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
APPS = [Path.home() / "Desktop" / "Babble.app", Path("/Applications/Babble.app"), Path.home() / "Applications" / "Babble.app"]
NOT_OPEN = "Babble isn't open. Call open_babble, or ask the owner to open it."
WAIT = 50          # seconds a tool waits for its work by default before handing back the task to wait for
MAX_TEXT = 60000   # characters a tool's answer may have

INSTRUCTIONS = """Babble is the owner's audiobook and voice studio on this Mac. It makes speech, voices, sound \
effects and whole audiobooks with local AI models (free), or on cloud GPUs and AI services (paid, from the owner's \
own accounts).

A book: create_book or import_book → detect_speakers (finds who says what, picks voices) → check the cast with \
get_book and list_voices, change it with update_book → direct_sounds (optional: sound effects, ambience, music) → \
make_book with preview=true first, then the whole book → wait_for_task until it's done. Long work runs as a task: \
tools hand back a task_id when it isn't done within wait_seconds; call wait_for_task again, or stop_task.

Files are paths on this Mac. Paid work (cloud GPUs, AI services) counts against the AI agents' own daily limit, \
which starts at $0: estimate first (estimate_cloud) and tell the owner the cost. If a limit refuses it, only the \
owner can raise it in Babble. Keys, sign-ins and limits are the owner's: ask them to do those in the app."""


class Fail(Exception):
    """A tool's error, for the agent to read."""


# ---------- the engine ----------

class Engine:
    def __init__(self):
        self.c = httpx.Client(base_url=BASE, timeout=60)

    @staticmethod
    def _key():
        try:
            return (HOME / "engine-token").read_text().strip()
        except OSError:
            raise Fail(NOT_OPEN) from None

    def up(self):
        try:
            return self.c.get("/api/health", timeout=3).status_code == 200
        except httpx.HTTPError:
            return False

    def call(self, method, path, body=None, params=None, files=None, timeout=60):
        r = None
        for attempt in (0, 1):
            headers = {"X-Babble-Token": self._key(), "X-Babble-Client": "mcp"}
            try:
                r = self.c.request(method, path, json=body if files is None else None, params=params, files=files,
                                   headers=headers, timeout=timeout)
            except httpx.ConnectError:
                raise Fail(NOT_OPEN) from None
            except httpx.TimeoutException:
                raise Fail("Babble didn't answer in time. It may be busy: try again.") from None
            if r.status_code != 401 or attempt:  # (401: Babble restarted with a new key; read it again)
                break
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail")
            except ValueError:
                detail = r.text[:400]
            if isinstance(detail, list):  # (a request that didn't fit: FastAPI's list of problems)
                detail = "; ".join(f"{'.'.join(str(x) for x in d.get('loc', [])[1:])}: {d.get('msg', '')}" for d in detail)
            raise Fail(f"{detail or r.reason_phrase} (HTTP {r.status_code})")
        return r.json() if "json" in r.headers.get("content-type", "") else r.text


E = Engine()


# ---------- tools ----------

TOOLS = {}


def tool(name, description, props=None, required=(), read_only=False, destructive=False, idempotent=False,
         open_world=False):
    def wrap(fn):
        TOOLS[name] = {"fn": fn, "spec": {
            "name": name, "description": description,
            "inputSchema": {"type": "object", "properties": props or {}, "required": list(required)},
            "annotations": {"readOnlyHint": read_only, "destructiveHint": destructive, "idempotentHint": idempotent,
                            "openWorldHint": open_world}}}
        return fn
    return wrap


def s(kind, about, **more):
    return {"type": kind, "description": about, **more}


WAIT_ARG = s("integer", f"Seconds to wait for it to finish before handing back the task (default {WAIT}, up to 600).",
             minimum=0, maximum=600)
BOOK_ARG = s("string", "The audiobook's id (list_books).")


class Ctx:
    """One tool call: progress notes to the agent while it waits, and whether the agent gave up on it."""

    def __init__(self, server, token):
        self.server, self.token, self.cancelled = server, token, threading.Event()
        self.n = 0

    def progress(self, fraction, message=""):
        if self.token is None:
            return
        self.n += 1
        params = {"progressToken": self.token, "progress": round(float(fraction or 0) * 100, 1), "total": 100}
        if message:
            params["message"] = str(message)[:200]
        self.server.send({"jsonrpc": "2.0", "method": "notifications/progress", "params": params})


def _brief(task, keep=("audio", "file", "path", "takes", "sound", "text", "transcript", "message", "value",
                       "cloud", "chapters", "placements_count", "made", "seconds")):
    """A task as the agent needs it: no story text or long logs."""
    result = task.get("result") or {}
    out = {"task_id": task.get("id"), "title": task.get("title"), "state": task.get("state"),
           "status": (task.get("status") or "").split("\n")[0][:300]}
    if task.get("progress") is not None and task.get("state") == "running":
        out["progress"] = round(task["progress"], 3)
    if task.get("eta"):
        out["eta_seconds"] = round(task["eta"])
    if task.get("error"):
        out["error"] = task["error"]
    kept = {k: result[k] for k in keep if k in result}
    if isinstance(kept.get("transcript"), dict):  # (the words, without every segment's timings)
        tr = kept["transcript"]
        kept["transcript"] = {k: tr[k] for k in ("text", "language", "duration") if k in tr}
    if kept:
        out["result"] = kept
    if task.get("state") == "running":
        out["next"] = "Still running: call wait_for_task with this task_id (or stop_task)."
    return out


def _wait(task, ctx, seconds):
    """Follow a task until it ends or `seconds` pass (the agent's progress shows its status)."""
    seconds = WAIT if seconds is None else max(0, min(int(seconds), 600))
    end = time.time() + seconds
    while task.get("state") == "running" and time.time() < end and not ctx.cancelled.is_set():
        ctx.progress(task.get("progress"), (task.get("status") or "").split("\n")[0])
        time.sleep(1.5)
        task = E.call("GET", f"/api/tasks/{task['id']}")
    return task


def _local(path, under=None):
    """A file on this Mac the engine can use: kept where it is when the app made it, else uploaded first."""
    p = Path(os.path.expanduser(str(path or ""))).resolve()
    if not p.is_file():
        raise Fail(f"No file at {p}.")
    if under is not None and (p == under.resolve() or under.resolve() in p.parents):
        return str(p)
    with open(p, "rb") as f:
        return E.call("POST", "/api/files/upload", files={"file": (p.name, f)}, timeout=600)["path"]


def _project(book_id):
    return E.call("GET", f"/api/projects/{book_id}")


def _book_request(p, preview=False, **over):
    """What the app's Create sends for a saved book (BookEditor's request())."""
    st = E.call("GET", "/api/settings")
    parallel = p.get("parallel") if p.get("parallel") is not None else True
    automake = p.get("sfx_auto_make") if p.get("sfx_auto_make") is not None else st.get("sfx_auto_make", True)
    req = {k: p.get(k) for k in ("story", "cast", "models", "quotes_only", "speed", "stability", "language", "seed",
                                 "line_pause", "para_pause", "out_format", "title", "author", "cover", "sound", "bed",
                                 "intro", "fx_level", "amb_level", "music_level", "chapter_music", "intro_level",
                                 "folder", "new_sounds", "cloud_hours")}
    req.update(batch=max(2, int(p.get("batch") or 2)) if parallel else 1, parallel=parallel,
               pronunciations=p.get("pronunciations") or [], automake=bool(automake), preview=bool(preview),
               name=(p.get("title") or "").strip() or f"Untitled-{p.get('untitled') or 1}", book=p["id"],
               make_on=p.get("make_on") or "mac", cloud_gpu=p.get("cloud_gpu") or "A10G",
               cloud_live=p.get("cloud_live") if p.get("cloud_live") is not None else True,
               cloud_sounds=p.get("cloud_sounds") if p.get("cloud_sounds") is not None else True)
    req.update({k: v for k, v in over.items() if v is not None})
    return {k: v for k, v in req.items() if v is not None or k in ("cover", "chapter_music")}


def _story_summary(p, include_story):
    story = p.get("story") or ""
    out = {k: p.get(k) for k in ("id", "title", "author", "cover", "language", "out_format", "models", "quotes_only",
                                 "speed", "stability", "make_on", "cloud_gpu", "director_notes", "file", "folder")}
    out["cast"] = p.get("cast") or []
    out["story_chars"] = len(story)
    if include_story:
        out["story"] = story
    else:
        out["story_start"] = story[:600]
    return out


# --- status and the app ---

@tool("babble_status", "Whether Babble is open, if AI agents are on, today's spending and the agents' limit, and "
      "the work that's running.", read_only=True)
def babble_status(args, ctx):
    if not E.up():
        return {"open": False, "next": NOT_OPEN}
    out = {"open": True}
    try:
        out["agents"] = E.call("GET", "/api/agents")["on"]
        sp = E.call("GET", "/api/spending")
        out["spending"] = {k: sp.get(k) for k in ("today", "month", "limit", "agent_today", "agent_limit")}
        out["running"] = [_brief(t) for t in E.call("GET", "/api/tasks") if t.get("state") == "running"]
    except Fail as e:
        out["problem"] = str(e)
    return out


@tool("open_babble", "Open the Babble app on this machine (in the background) and wait until it's ready.",
      idempotent=True)
def open_babble(args, ctx):
    if E.up():
        return {"open": True, "message": "Babble is open."}
    if sys.platform == "darwin":
        app = next((a for a in APPS if a.exists()), None)
        if app is None:
            raise Fail("Couldn't find Babble.app (Desktop, Applications). Ask the owner to open it.")
        subprocess.run(["open", "-g", str(app)], check=False)
    elif sys.platform == "win32":
        import shutil
        local_app = Path((os.environ.get("LOCALAPPDATA") or "").strip() or (Path.home() / "AppData" / "Local"))
        prog_files = Path((os.environ.get("PROGRAMFILES") or "").strip() or r"C:\Program Files")
        prog_x86 = Path((os.environ.get("PROGRAMFILES(X86)") or "").strip() or r"C:\Program Files (x86)")
        which_exe = shutil.which("Babble.exe")
        win_apps = [
            local_app / "Babble" / "Babble.exe",
            local_app / "Programs" / "Babble" / "Babble.exe",
            prog_files / "Babble" / "Babble.exe",
            prog_x86 / "Babble" / "Babble.exe",
            Path.home() / "Desktop" / "Babble.exe",
            *([Path(which_exe)] if which_exe else []),
        ]
        app = next((a for a in win_apps if a.exists()), None)
        if app is None:
            raise Fail("Couldn't find Babble.exe. Ask the owner to open Babble.")
        subprocess.Popen([str(app)], creationflags=getattr(subprocess, "DETACHED_PROCESS", 0x00000008))
    else:
        raise Fail("Automatic launch isn't set up on this OS. Ask the owner to open Babble.")
    end = time.time() + 90
    while time.time() < end and not ctx.cancelled.is_set():
        ctx.progress((time.time() - end + 90) / 90, "Opening Babble…")
        if E.up():
            return {"open": True, "message": f"Opened {app}."}
        time.sleep(1)
    raise Fail("Babble didn't start within 90 s. Ask the owner to open it.")


# --- voices and models ---

@tool("list_voices", "Voices to speak with: the owner's saved voices and each model's built-in ones. Use a voice's "
      "`voice` value in speak, cast rows and previews.", {
          "query": s("string", "Words to match in the name, description, model or gender."),
          "model": s("string", "Only this model's voices (list_models: qwen, orpheus, kokoro…)."),
          "include_unavailable": s("boolean", "Also voices this Mac can't make now (default false)."),
      }, read_only=True)
def list_voices(args, ctx):
    words = (args.get("query") or "").lower().split()
    out = []
    for v in E.call("GET", "/api/voices"):
        if "saved" in v.get("tags", []) and v.get("engine") != "qwen":
            continue  # (another model's copy of a saved voice: listed once, as the app does)
        if args.get("model") and v.get("engine") != args["model"]:
            continue
        if not args.get("include_unavailable") and not v.get("available"):
            continue
        text = f"{v.get('name')} {v.get('description')} {v.get('engine')} {v.get('gender')} {' '.join(v.get('tags', []))}"
        if not all(w in text.lower() for w in words):
            continue
        row = {"voice": v["value"], "name": v.get("name"), "model": v.get("engine"), "gender": v.get("gender"),
               "about": (v.get("description") or "")[:140], "tags": v.get("tags", [])}
        if not v.get("available"):
            row["why_not"] = v.get("why")
        out.append(row)
    return {"count": len(out), "voices": out[:300]}


@tool("list_models", "The speech models Babble has, whether each can run on this Mac, and why not.", read_only=True)
def list_models(args, ctx):
    keep = ("id", "name", "short", "tier", "available", "why", "downloaded", "size_gb", "features", "license", "off")
    return [{k: e[k] for k in keep if k in e} for e in E.call("GET", "/api/engines")]


@tool("preview_voice", "Hear a voice: it says hello, or reads `text`. Hands back the audio file's path.", {
    "voice": s("string", "A voice value from list_voices."), "text": s("string", "What it reads (optional)."),
    "language": s("string", "Language (default english)."), "wait_seconds": WAIT_ARG}, ["voice"])
def preview_voice(args, ctx):
    r = E.call("POST", "/api/voices/preview", {"voice": args["voice"], "text": args.get("text"),
                                              "language": args.get("language") or "english"})
    if "audio" in r and "state" not in r:
        return {"audio": r["audio"]}
    return _brief(_wait(r, ctx, args.get("wait_seconds")))


@tool("design_voice", "Make a new voice from a description (Qwen voice design, on this Mac, free): three takes to "
      "pick from. Save the best with save_voice.", {
          "description": s("string", "The voice: age, tone, accent, pace (\"a warm, calm British woman in her 40s\")."),
          "sample_text": s("string", "What the takes say (optional)."), "language": s("string", "Default english."),
          "wait_seconds": WAIT_ARG}, ["description"])
def design_voice(args, ctx):
    t = E.call("POST", "/api/my-voices/design", {"description": args["description"], "text": args.get("sample_text") or "",
                                                "language": args.get("language") or "english"})
    return _brief(_wait(t, ctx, args.get("wait_seconds") or 240))


@tool("save_voice", "Save a designed take as one of the owner's voices.", {
    "take": s("string", "A take's path from design_voice."), "name": s("string", "The voice's name."),
    "gender": s("string", "female, male or unspecified."), "description": s("string", "What it sounds like."),
    "transcript": s("string", "What the take says (default: the design sample text).")}, ["take", "name"])
def save_voice(args, ctx):
    return E.call("POST", "/api/my-voices/save-take", {"path": args["take"], "name": args["name"],
                                                      "gender": args.get("gender") or "unspecified",
                                                      "description": args.get("description") or "",
                                                      "transcript": args.get("transcript") or ""})


@tool("clone_voice", "Save a voice copied from a recording (3–30 s of one person speaking clearly works best). Only "
      "clone a voice with that person's consent.", {
          "audio_path": s("string", "The recording on this Mac."), "name": s("string", "The voice's name."),
          "transcript": s("string", "What the recording says (made with speech-to-text when left out)."),
          "gender": s("string", "female, male or unspecified."), "description": s("string", "What it sounds like."),
      }, ["audio_path", "name"])
def clone_voice(args, ctx):
    path = _local(args["audio_path"], OUTPUTS)
    text = args.get("transcript") or ""
    if not text:
        t = _wait(E.call("POST", "/api/my-voices/transcribe", {"path": path}), ctx, 300)
        if t.get("state") != "done":
            raise Fail(t.get("error") or "Couldn't transcribe the recording. Pass its transcript.")
        text = (t.get("result") or {}).get("text") or ""
    return E.call("POST", "/api/my-voices/clone", {"path": path, "name": args["name"], "transcript": text,
                                                  "gender": args.get("gender") or "unspecified",
                                                  "description": args.get("description") or ""})


@tool("delete_voice", "Delete one of the owner's saved voices.", {"name": s("string", "The saved voice's name.")},
      ["name"], destructive=True)
def delete_voice(args, ctx):
    return E.call("DELETE", f"/api/my-voices/{args['name']}")


# --- speech and transcription ---

@tool("speak", "Turn text into speech with one voice (Text to Speech). Hands back the audio file's path.", {
    "text": s("string", "What to say."), "voice": s("string", "A voice value from list_voices."),
    "style": s("string", "How to say it (\"excited\", \"whispering\"), for models that take directions."),
    "speed": s("number", "1.0 is normal."), "language": s("string", "Default english."), "wait_seconds": WAIT_ARG},
      ["text", "voice"])
def speak(args, ctx):
    body = {"text": args["text"], "voice": args["voice"], "style": args.get("style") or "",
            "speed": args.get("speed") or 1.0, "language": args.get("language") or "english"}
    return _brief(_wait(E.call("POST", "/api/speech", body), ctx, args.get("wait_seconds")))


@tool("transcribe", "Speech to text for a recording on this Mac (Whisper, on this Mac, free).", {
    "audio_path": s("string", "The recording."), "language": s("string", "auto (default) or a language."),
    "wait_seconds": WAIT_ARG}, ["audio_path"])
def transcribe(args, ctx):
    path = _local(args["audio_path"], OUTPUTS)
    t = E.call("POST", "/api/stt/transcribe", {"path": path, "language": args.get("language") or "auto",
                                              "timestamps": False, "filename": Path(args["audio_path"]).name})
    return _brief(_wait(t, ctx, args.get("wait_seconds") or 300))


# --- sounds ---

@tool("list_sounds", "The sound library: effects, ambience and music the books can use.", {
    "query": s("string", "Words to match in the name or tags."),
    "kind": s("string", "effect, ambience or music."), "limit": s("integer", "Default 50.")}, read_only=True)
def list_sounds(args, ctx):
    words = (args.get("query") or "").lower().split()
    rows = []
    for e in E.call("GET", "/api/sounds"):
        if args.get("kind") and e.get("kind") != args["kind"]:
            continue
        text = f"{e.get('name')} {' '.join(e.get('tags') or [])} {e.get('description') or ''}".lower()
        if all(w in text for w in words):
            rows.append({k: e.get(k) for k in ("id", "name", "kind", "duration", "book", "path")})
    return {"count": len(rows), "sounds": rows[:int(args.get("limit") or 50)]}


@tool("make_sound", "Make a new sound effect, ambience or music clip from a description (on this Mac, free) and "
      "add it to the library.", {
          "description": s("string", "The sound (\"heavy rain on a tin roof\")."),
          "kind": s("string", "effect (default), ambience or music."), "seconds": s("number", "Length (optional)."),
          "book_id": s("string", "Keep it in this audiobook only (optional)."), "wait_seconds": WAIT_ARG},
      ["description"])
def make_sound(args, ctx):
    body = {"description": args["description"], "kind": args.get("kind") or "effect", "seconds": args.get("seconds"),
            "book": args.get("book_id"), "scope": "book" if args.get("book_id") else "all"}
    return _brief(_wait(E.call("POST", "/api/sounds/make", body), ctx, args.get("wait_seconds") or 180))


# --- audiobooks ---

@tool("list_books", "The owner's audiobooks.", read_only=True)
def list_books(args, ctx):
    return E.call("GET", "/api/projects")


@tool("get_book", "An audiobook: its settings, cast (who's read by which voice) and chapters (made or not).", {
    "book_id": BOOK_ARG, "include_story": s("boolean", "Also the whole text (default: its first 600 characters).")},
      ["book_id"], read_only=True)
def get_book(args, ctx):
    p = _project(args["book_id"])
    out = _story_summary(p, args.get("include_story"))
    if p.get("story"):
        try:
            out["chapters"] = E.call("POST", "/api/books/chapters", {"story": p["story"], "folder": p.get("folder"),
                                                                     "title": p.get("title") or "",
                                                                     "quotes_only": p.get("quotes_only", True)})
        except Fail as e:
            out["chapters_problem"] = str(e)
    return out


@tool("create_book", "Start a new audiobook (optionally with its text).", {
    "title": s("string", "Title."), "author": s("string", "Author."),
    "story": s("string", "The text. Chapters start with a line like \"## Chapter 1\".")})
def create_book(args, ctx):
    p = E.call("POST", "/api/projects", {k: args[k] for k in ("title", "author", "story") if args.get(k)})
    return _story_summary(p, False)


@tool("import_book", "Read a book file (EPUB, PDF, DOCX, TXT…) into an audiobook: a new one, or `book_id`'s text "
      "replaced.", {"file_path": s("string", "The file on this Mac."), "book_id": s("string", "Optional.")},
      ["file_path"])
def import_book(args, ctx):
    p = Path(os.path.expanduser(args["file_path"])).resolve()
    if not p.is_file():
        raise Fail(f"No file at {p}.")
    with open(p, "rb") as f:
        r = E.call("POST", "/api/books/import", files={"file": (p.name, f)}, timeout=900)
    if args.get("book_id"):
        old = _project(args["book_id"])
        changes = {"story": r["text"], "original": old.get("story") or "", "title": old.get("title") or r.get("title"),
                   "author": old.get("author") or r.get("author"), "cover": old.get("cover") or r.get("cover")}
        E.call("PUT", f"/api/projects/{args['book_id']}", changes)
        p2 = _project(args["book_id"])
    else:
        p2 = E.call("POST", "/api/projects", {"story": r["text"], "title": r.get("title") or "",
                                              "author": r.get("author") or "", "cover": r.get("cover")})
    return {**_story_summary(p2, False), "chapters": r.get("chapters"), "message": r.get("message")}


@tool("update_book", "Change an audiobook: its text, title, cast, language, output format, settings.", {
    "book_id": BOOK_ARG,
    "changes": s("object", "Fields to set: story, title, author, cover (an image path), language, out_format, speed, "
                 "stability (0–100), quotes_only, director_notes, make_on (mac/kaggle/modal…), cloud_gpu, and cast: "
                 "[{character, voice (a list_voices value), style, speed, stability}] (the whole cast, \"Narrator\" "
                 "first).")}, ["book_id", "changes"])
def update_book(args, ctx):
    changes = dict(args.get("changes") or {})
    if "story" in changes and "original" not in changes:  # (the app's Undo puts this back)
        changes["original"] = _project(args["book_id"]).get("story") or ""
    E.call("PUT", f"/api/projects/{args['book_id']}", changes)
    return _story_summary(_project(args["book_id"]), False)


@tool("delete_book", "Delete an audiobook (its audio is kept unless delete_audio).", {
    "book_id": BOOK_ARG, "delete_audio": s("boolean", "Also its made audio (default false).")}, ["book_id"],
      destructive=True)
def delete_book(args, ctx):
    return E.call("DELETE", f"/api/projects/{args['book_id']}",
                  params={"delete_audio": str(bool(args.get("delete_audio"))).lower()})


def _apply(book_id, task):
    """A finished Detect speakers or Direct sounds task's result, put into the book as the app does."""
    r, kind = task.get("result") or {}, task.get("kind")
    p = _project(book_id)
    if kind == "detect" and r.get("cast"):
        voices = E.call("GET", "/api/voices")
        value = lambda label: next((v["value"] for v in voices if label in (v["label"], v["value"])), label)  # noqa: E731
        old = {c["character"]: c for c in p.get("cast") or []}
        cast = [{"character": row[0], "voice": value(row[1]), "style": row[2] or "", "speed": float(row[3] or 1),
                 "stability": (old.get(row[0]) or {}).get("stability")} for row in r["cast"]]
        E.call("PUT", f"/api/projects/{book_id}", {"story": r.get("story") or p["story"],
                                                    "original": r.get("original") or p.get("story") or "", "cast": cast})
        return {"applied": "cast", "cast": cast}
    if kind == "direct" and r.get("placements") is not None:
        if not r["placements"]:
            return {"applied": "nothing", "message": "The director found no sounds to add."}
        a = E.call("POST", "/api/books/suggest/apply", {"story": p["story"], "planned_on": r.get("planned_on") or "",
                                                         "placements": r["placements"]})
        E.call("PUT", f"/api/projects/{book_id}", {"story": a["story"], "original": p.get("story") or ""})
        return {"applied": "sounds", "added": len(r["placements"]), "message": a.get("message"),
                "next": "The new sounds are made when the book is made."}
    raise Fail("That task has nothing to put into the book (it isn't a finished Detect speakers or Direct sounds).")


def _start_and_apply(book_id, task, ctx, seconds):
    task = _wait(task, ctx, seconds)
    out = _brief(task)
    if task.get("state") == "done":
        out["book"] = _apply(book_id, task)
    elif task.get("state") == "running":
        out["next"] = f"Still running: call apply_book_task with book_id {book_id} and this task_id when it's done."
    return out


@tool("detect_speakers", "Find who says each line and give every character a voice (the book's cast). On this "
      "Mac it's free; method \"quick\" is fast and rough, \"ai\" reads the whole book.", {
          "book_id": BOOK_ARG, "method": s("string", "ai (default) or quick."), "wait_seconds": WAIT_ARG},
      ["book_id"])
def detect_speakers(args, ctx):
    p = _project(args["book_id"])
    if not (p.get("story") or "").strip():
        raise Fail("The book has no text yet (update_book or import_book).")
    t = E.call("POST", "/api/books/detect", {"story": p["story"], "cast": p.get("cast") or [],
                                            "method": args.get("method") or p.get("detect") or "ai",
                                            "make_on": "mac", "cloud_gpu": ""})
    return _start_and_apply(args["book_id"], t, ctx, args.get("wait_seconds") or 300)


@tool("direct_sounds", "Let the sound director add sound effects, ambience and music cues to the book's text. "
      "where=\"mac\" (default) is free; an AI service (anthropic, openai, gemini, openrouter…) is paid.", {
          "book_id": BOOK_ARG, "where": s("string", "mac (default) or a connected AI service."),
          "model": s("string", "That service's model (optional)."), "wait_seconds": WAIT_ARG}, ["book_id"])
def direct_sounds(args, ctx):
    p = _project(args["book_id"])
    st = E.call("GET", "/api/settings")
    where = args.get("where") or p.get("director_where") or st.get("director_where") or "mac"
    model = args.get("model") or (p.get("director_models") or st.get("director_models") or {}).get(where, "")
    body = {"story": p["story"], "where": where, "model": model,
            "checks": p.get("director_checks") or st.get("director_checks") or 3, "notes": p.get("director_notes") or "",
            "book": p["id"], "scenes": p.get("director_scenes") or st.get("director_scenes")}
    return _start_and_apply(args["book_id"], E.call("POST", "/api/books/direct", body), ctx,
                            args.get("wait_seconds") or 300)


@tool("apply_book_task", "Put a finished detect_speakers or direct_sounds task's result into its book (when the "
      "tool handed back a task instead of finishing).", {"book_id": BOOK_ARG, "task_id": s("string", "The task.")},
      ["book_id", "task_id"])
def apply_book_task(args, ctx):
    t = E.call("GET", f"/api/tasks/{args['task_id']}")
    if t.get("state") != "done":
        return _brief(t)
    return _apply(args["book_id"], t)


@tool("make_book", "Make the audiobook on this Mac (free): every chapter spoken, its sounds mixed, one finished "
      "file. Hours for a long book, so it hands back a task. preview=true makes a short sample first.", {
          "book_id": BOOK_ARG, "preview": s("boolean", "A short sample instead of the book."),
          "preview_text": s("string", "With preview: the passage to hear."),
          "preview_chapter": s("integer", "With preview: the chapter (0 = the first)."),
          "wait_seconds": s("integer", "Seconds to wait (default 20; up to 600).", minimum=0, maximum=600)},
      ["book_id"])
def make_book(args, ctx):
    p = _project(args["book_id"])
    if not p.get("cast"):
        raise Fail("The book has no cast yet: detect_speakers, or set one with update_book.")
    body = _book_request(p, preview=bool(args.get("preview")), preview_text=args.get("preview_text"),
                         preview_chapter=args.get("preview_chapter"))
    body["make_on"] = "mac"
    return _brief(_wait(E.call("POST", "/api/books/create", body), ctx, args.get("wait_seconds") or 20))


@tool("estimate_cloud", "What making the book on a cloud GPU would take and cost (nothing is sent).", {
    "book_id": BOOK_ARG, "where": s("string", "kaggle (free weekly hours), modal, runpod… (Babble's connected clouds)."),
    "gpu": s("string", "The GPU (optional).")}, ["book_id", "where"], read_only=True)
def estimate_cloud(args, ctx):
    body = _book_request(_project(args["book_id"]), make_on=args["where"], cloud_gpu=args.get("gpu"))
    return E.call("POST", "/api/books/cloud/estimate", body)


@tool("send_to_cloud", "Make the book's lines on a cloud GPU (the book is finished on this Mac as they come back). "
      "Paid except Kaggle: run estimate_cloud first and tell the owner the cost.", {
          "book_id": BOOK_ARG, "where": s("string", "kaggle, modal, runpod…"), "gpu": s("string", "Optional."),
          "wait_seconds": WAIT_ARG}, ["book_id", "where"], open_world=True)
def send_to_cloud(args, ctx):
    body = _book_request(_project(args["book_id"]), make_on=args["where"], cloud_gpu=args.get("gpu"))
    return _brief(_wait(E.call("POST", "/api/books/cloud/send", body), ctx, args.get("wait_seconds")))


# --- tasks, spending, settings ---

@tool("list_tasks", "Babble's recent and running work.", read_only=True)
def list_tasks(args, ctx):
    return [_brief(t) for t in E.call("GET", "/api/tasks")]


@tool("wait_for_task", "Wait for a task to finish (or `wait_seconds`), then say how it went.", {
    "task_id": s("string", "The task."), "wait_seconds": WAIT_ARG}, ["task_id"], read_only=True)
def wait_for_task(args, ctx):
    return _brief(_wait(E.call("GET", f"/api/tasks/{args['task_id']}"), ctx, args.get("wait_seconds")))


@tool("stop_task", "Stop a running task (what it made so far is kept).", {"task_id": s("string", "The task.")},
      ["task_id"])
def stop_task(args, ctx):
    return _brief(E.call("POST", f"/api/tasks/{args['task_id']}/stop"))


@tool("get_spending", "What paid work cost today and this month, the limits, and the recent paid jobs.",
      read_only=True)
def get_spending(args, ctx):
    sp = E.call("GET", "/api/spending")
    return {**{k: sp.get(k) for k in ("today", "month", "limit", "agent_today", "agent_limit", "jobs", "alerts")},
            "recent": E.call("GET", "/api/spending/history", params={"limit": 20})}


@tool("stop_all_paid_work", "Stop every paid task and cloud job now.", destructive=True)
def stop_all_paid_work(args, ctx):
    r = E.call("POST", "/api/spending/stop-all")
    return {"stopped": r.get("stopped"), "today": r.get("today")}


@tool("get_settings", "Babble's settings (no keys).", read_only=True)
def get_settings(args, ctx):
    return E.call("GET", "/api/settings")


@tool("update_settings", "Change Babble's settings (get_settings shows them). Keys, limits and connections are "
      "the owner's.", {"changes": s("object", "Setting names and values.")}, ["changes"])
def update_settings(args, ctx):
    return E.call("POST", "/api/settings", args.get("changes") or {})


@tool("show_in_finder", "Show a file Babble made in Finder or Explorer for the owner.", {"path": s("string", "The file.")},
      ["path"])
def show_in_finder(args, ctx):
    p = Path(os.path.expanduser(args["path"])).resolve()
    if not p.exists() or HOME.resolve() not in p.parents:
        raise Fail("Only files in Babble's own folder can be shown.")
    if sys.platform == "darwin":
        subprocess.run(["open", "-R", str(p)], check=False)
    elif sys.platform == "win32":
        subprocess.run(["explorer.exe", f"/select,{p}"], check=False)
    else:
        subprocess.run(["xdg-open", str(p.parent)], check=False)
    return {"shown": str(p)}


# --- everything else ---

@tool("list_api_routes", "Every Babble engine route (method, path, what it does), for call_api.", {
    "filter": s("string", "Words to match in the path or description.")}, read_only=True)
def list_api_routes(args, ctx):
    words = (args.get("filter") or "").lower().split()
    return [r for r in E.call("GET", "/api/agents/routes")
            if all(w in f"{r['path']} {r['about']}".lower() for w in words)]


@tool("call_api", "Call any Babble engine route directly (list_api_routes), for what the other tools don't cover. "
      "The same rules apply: no keys, sign-ins or limits; paid work within the agents' limit.", {
          "method": s("string", "GET, POST, PUT or DELETE."), "path": s("string", "Like /api/books/stats."),
          "body": s("object", "JSON body (POST/PUT)."), "query": s("object", "Query parameters.")},
      ["method", "path"], destructive=True, open_world=True)
def call_api(args, ctx):
    method, path = (args.get("method") or "GET").upper(), args.get("path") or ""
    if not path.startswith("/api/") or ".." in path or method not in ("GET", "POST", "PUT", "DELETE", "PATCH"):
        raise Fail("path must start with /api/ (list_api_routes), method GET/POST/PUT/DELETE.")
    return E.call(method, path, args.get("body"), params=args.get("query"))


# ---------- the protocol (JSON-RPC 2.0 over stdio, one message a line) ----------

class Server:
    def __init__(self, out=sys.stdout):
        self.out, self.lock, self.calls = out, threading.Lock(), {}

    def send(self, msg):
        line = json.dumps(msg, ensure_ascii=False, default=str)
        with self.lock:
            self.out.write(line + "\n")
            self.out.flush()

    def reply(self, mid, result=None, error=None):
        msg = {"jsonrpc": "2.0", "id": mid}
        if error is not None:
            msg["error"] = error
        else:
            msg["result"] = result
        self.send(msg)

    def handle(self, msg):
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
            return self.reply(msg.get("id") if isinstance(msg, dict) else None,
                              error={"code": -32600, "message": "Invalid request"})
        method, mid, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
        if method is None:
            return  # (a reply to something we asked: we ask nothing)
        if mid is None:  # notifications
            if method == "notifications/cancelled":
                ctx = self.calls.get(params.get("requestId"))
                if ctx is not None:
                    ctx.cancelled.set()
            return
        if method == "initialize":
            asked = params.get("protocolVersion")
            return self.reply(mid, {"protocolVersion": asked if asked in PROTOCOLS else PROTOCOLS[0],
                                    "capabilities": {"tools": {"listChanged": False}},
                                    "serverInfo": {"name": "babble", "title": "Babble", "version": VERSION},
                                    "instructions": INSTRUCTIONS})
        if method == "ping":
            return self.reply(mid, {})
        if method == "tools/list":
            return self.reply(mid, {"tools": [t["spec"] for t in TOOLS.values()]})
        if method == "tools/call":
            return threading.Thread(target=self._call, args=(mid, params), daemon=True).start()
        if method in ("resources/list", "prompts/list"):
            return self.reply(mid, {method.split("/")[0]: []})
        self.reply(mid, error={"code": -32601, "message": f"Method not found: {method}"})

    def _call(self, mid, params):
        name, args = params.get("name"), params.get("arguments") or {}
        t = TOOLS.get(name)
        if t is None:
            return self.reply(mid, error={"code": -32602, "message": f"Unknown tool: {name}"})
        ctx = Ctx(self, (params.get("_meta") or {}).get("progressToken"))
        self.calls[mid] = ctx
        try:
            out = t["fn"](args, ctx)
            text = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False, indent=1, default=str)
            if len(text) > MAX_TEXT:
                text = text[:MAX_TEXT] + "\n… (cut short: ask for less, e.g. a filter)"
            self.reply(mid, {"content": [{"type": "text", "text": text}], "isError": False})
        except Fail as e:
            self.reply(mid, {"content": [{"type": "text", "text": str(e)}], "isError": True})
        except Exception as e:  # noqa: BLE001 - the agent reads it; the server keeps going
            print(f"[babble-mcp] {name} failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            self.reply(mid, {"content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}], "isError": True})
        finally:
            self.calls.pop(mid, None)

    def serve(self, stream=sys.stdin):
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                self.reply(None, error={"code": -32700, "message": "Parse error"})
                continue
            for m in msg if isinstance(msg, list) else [msg]:
                try:
                    self.handle(m)
                except Exception as e:  # noqa: BLE001
                    print(f"[babble-mcp] {type(e).__name__}: {e}", file=sys.stderr, flush=True)


def main():
    print(f"[babble-mcp] {len(TOOLS)} tools, engine {BASE}", file=sys.stderr, flush=True)
    Server().serve()
