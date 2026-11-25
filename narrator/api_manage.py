"""The local API, part 2: the desktop app's Voices, Library (voice packs), Models, Connections and Train pages.

The same work the classic page does (narrator/ui/*), without Gradio. Long work (designing voices, downloads,
installs, preparing and sending training) runs as a task (narrator/tasks.py) that the app follows live.
"""
import json
import queue
import re
import shutil
import threading
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import engine as en
from . import library, settings, signing, tasks
from . import tts as engines
from . import voices as vl
from .paths import HOME, OUTPUTS, TRAINING
from .status import Status

router = APIRouter(prefix="/api")
DESIGN_TEXT = "The village woke slowly that morning, wrapped in fog and the quiet sound of waves against the harbor wall."
PREVIEW = "The rain had stopped by the time Marcus reached the old lighthouse. He paused at the door, listening."
GROUP = {"kokoro": "light", "qwen": "standard", "orpheus": "standard", "vibevoice": "high", "higgs": "high",
         "drama": "high"}


def _ok(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


def _start(kind, title, make_events):
    return tasks.start(kind, title, make_events).info()


def _logged(fn, *args, done=None, **kwargs):
    """Events for a function that reports with log=…: each line as the status, the whole log so far as "log",
    then {"value": what it returned} (or done(value) for the last event's fields)."""
    import contextvars

    def events():
        q, out = queue.Queue(), {}
        ctx = contextvars.copy_context()

        def work():
            try:
                out["value"] = fn(*args, log=lambda m: q.put(m if isinstance(m, str) else str(m)), **kwargs)
            except BaseException as e:  # noqa: BLE001 - re-raised below for the task
                out["error"] = e
            finally:
                q.put(None)

        threading.Thread(target=lambda: ctx.run(work), daemon=True).start()
        lines = []
        while True:
            line = q.get()
            if line is None:
                break
            lines.append(str(line))
            # (a Status line says how far along it is: "Downloading the model · 45% · 1.6 of 3.6 GB")
            yield {"status": Status(line.splitlines()[0], getattr(line, "progress", None)) if line else "",
                   "log": "\n".join(lines[-120:])}
        if "error" in out:
            e = out["error"]
            if isinstance(e, tasks.Stopped):
                raise e
            raise ValueError(str(e) or type(e).__name__) from e
        extra = done(out.get("value")) if done else {"value": str(out.get("value")) if out.get("value") is not None else None}
        yield {"log": "\n".join(lines[-120:]), **extra}
    return events


def _inside(path, *roots):
    p = Path(path).resolve()
    if not any(p == r.resolve() or r.resolve() in p.parents for r in roots):
        raise HTTPException(403, "That file isn't one the app made.")
    if not p.is_file():
        raise HTTPException(404, "That file isn't there any more.")
    return p


# ---------- Voices: your saved voices, design and clone ----------

@router.get("/my-voices")
def my_voices():
    return [dict(v, file=str(vl.wav_path(v["name"]))) for v in vl.library()]


class VoiceMeta(BaseModel):
    gender: str = "unspecified"
    description: str = ""
    transcript: Optional[str] = None


@router.put("/my-voices/{name}")
def update_voice(name: str, req: VoiceMeta):
    if name not in {v["name"] for v in vl.library()}:
        raise HTTPException(404, "No such voice.")
    fields = {"gender": req.gender if req.gender in vl.GENDERS else "unspecified", "description": req.description}
    if req.transcript is not None:
        fields["transcript"] = req.transcript
    vl.write_meta(name, **fields)
    return dict(vl.info(name), file=str(vl.wav_path(name)))


@router.delete("/my-voices/{name}")
def delete_voice(name: str):
    if name not in {v["name"] for v in vl.library()}:
        raise HTTPException(404, "No such voice.")
    vl.delete(name)
    return {"deleted": name}


class Design(BaseModel):
    description: str
    text: str = ""
    language: str = "english"
    seed: int = 0


@router.post("/my-voices/design")
def design(req: Design):
    if not req.description.strip():
        raise HTTPException(400, "Describe the voice first.")

    def events():
        for takes in en.design_takes(req.description.strip(), (req.text or DESIGN_TEXT).strip(), req.language, 3,
                                     int(req.seed or 0)):
            yield {"takes": list(takes), "status": f"Made {len(takes)} of 3 takes…" if len(takes) < 3
                   else "Pick your favourite take and save it."}
    return _start("design", "Design a voice", events)


class SaveVoice(BaseModel):
    path: str                 # a designed take or an uploaded sample
    name: str
    gender: str = "unspecified"
    description: str = ""
    transcript: str = ""      # what the sample says (a take: the text it read)


def _save_voice(req, source):
    src = _inside(req.path, OUTPUTS)
    gender = req.gender if req.gender in vl.GENDERS else "unspecified"
    if req.name.strip() and vl.clean_name(req.name) in {v["name"] for v in vl.library()}:
        raise HTTPException(400, f"You already have a voice called {vl.clean_name(req.name)}. Pick another name.")
    saved = _ok(vl.save, req.name, src, req.transcript or (DESIGN_TEXT if source == "designed" else ""), source,
                req.description, gender)
    engines.refresh()
    return dict(vl.info(saved), file=str(vl.wav_path(saved)))


@router.post("/my-voices/save-take")
def save_take(req: SaveVoice):
    return _save_voice(req, "designed")


@router.post("/my-voices/clone")
def save_clone(req: SaveVoice):
    return _save_voice(req, "cloned")


class Transcribe(BaseModel):
    path: str


@router.post("/my-voices/transcribe")
def transcribe(req: Transcribe):
    src = _inside(req.path, OUTPUTS)

    def work():
        from . import stt
        text = ""
        for ev in stt.transcribe(str(src), model_id="whisper-large-v3-turbo", language="auto", timestamps=False,
                                 filename=src.name, save=False):
            tr = ev.get("transcript")
            if ev.get("text"):
                text = ev["text"]
            elif isinstance(tr, dict) and tr.get("text"):
                text = tr["text"]
            yield {"status": ev.get("status", "Transcribing…")}
        if tasks.stop_wanted():
            return
        yield {"text": text, "status": "Transcribed. Check the words match the clip."}
    return _start("transcribe", "Transcribe a sample", work)


# ---------- Library: voice packs ----------

def _pack_view(m):
    pack = m["pack"] or {}
    vals = {spk: v for v, spk in library.voices_of(m["folder"])}
    previews = {v.get("speaker"): v.get("preview") for v in pack.get("voices", [])}
    return {"id": m["id"], "folder": m["folder"], "name": pack.get("name") or m["id"], "genre": pack.get("genre", ""),
            "description": pack.get("description", ""), "version": pack.get("version", 1), "author": pack.get("author", ""),
            "engine": m["engine"], "size_gb": m["size_gb"], "signed": m.get("signed"),
            "voices": [{"speaker": spk, "name": v.get("name", spk), "gender": v.get("gender", "unspecified"),
                        "value": vals.get(spk), "preview": str(Path(m["folder"]) / previews[spk]) if previews.get(spk) else None}
                       for spk, v in m["voices"].items()]}


@router.get("/packs")
def packs():
    """Installed packs (Library) and the catalog's packs not installed yet. Models you trained are on Train."""
    installed = [_pack_view(m) for m in library.installed() if m["pack"]]
    catalog = [{k: e.get(k) for k in ("id", "name", "genre", "description", "version", "engine", "author", "size")}
               | {"voices": [{"speaker": v.get("speaker"), "name": v.get("name"), "gender": v.get("gender"),
                              "has_preview": bool(v.get("preview") and v.get("speaker"))} for v in e.get("voices", [])],
                  "where": "online" if e.get("where") == "online" else "this Mac",
                  "signed": _signed_view(e.get("signature"))}
               for e in library.catalog() if not e.get("installed")]
    return {"installed": installed, "catalog": catalog, "genres": library.GENRES}


def _signed_view(sig):
    res = signing.check(sig) if sig else None
    return {k: res.get(k) for k in ("github", "fingerprint", "verified", "why")} if res else None


# ---------- signing packs (Settings → General) ----------

def _signing_view():
    line = signing.public_line()
    user = signing.github_user()
    return {"github": user, "public_key": line, "fingerprint": signing.fingerprint(line) if line else "",
            "on_github": signing.on_github(user, line) if user and line else False}


@router.get("/signing")
def signing_get():
    return _signing_view()


class Signing(BaseModel):
    github: str = ""


@router.post("/signing")
def signing_set(req: Signing):
    """Save the GitHub account packs are signed as, and make this Mac's signing key the first time."""
    try:
        user = signing.set_github_user(req.github)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if user:
        signing.make_key()
    return _signing_view()


@router.get("/packs/preview")
def pack_preview(id: str, version: int, speaker: str):
    """A voice's preview from a pack that isn't installed yet (fetched from the catalog once, then kept)."""
    try:
        return FileResponse(library.preview_file(id, version, speaker))
    except LookupError as e:
        raise HTTPException(404, str(e))
    except Exception:  # noqa: BLE001 - offline, moved, or not audio
        raise HTTPException(502, "Couldn't get the preview. Check the internet connection.")


class PackPick(BaseModel):
    id: str
    version: int = 1


@router.post("/packs/install")
def install_pack(req: PackPick):
    entry = next((e for e in library.catalog() if e["id"] == req.id and int(e.get("version", 1)) == req.version), None)
    if not entry:
        raise HTTPException(404, "That pack isn't in the list any more.")
    return _start("pack", f"Install {entry['name']}", _logged(library.download_pack, entry, done=lambda d: {"folder": str(d)}))


@router.post("/packs/install-file")
def install_pack_file(file: UploadFile = File(...)):
    folder = OUTPUTS / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(file.filename or "pack.narrator-pack").name)[-80:]
    dest = folder / name
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f, 1 << 20)

    def install(log):
        try:
            return library.install_pack(dest, log=log)
        finally:
            dest.unlink(missing_ok=True)  # the pack is unpacked into its own folder; the upload isn't needed
    return _start("pack", "Install a pack file", _logged(install, done=lambda d: {"folder": str(d)}))


@router.delete("/packs")
def delete_pack(folder: str):
    if not any(m["folder"] == folder and m["pack"] for m in library.installed()):
        raise HTTPException(404, "That pack isn't installed.")
    key, _ = _run_for(folder)
    _ok(library.delete, folder)
    if key:  # (its run is kept: it can still be trained again)
        from .training.base import save_run
        save_run(key, state="deleted")
    return {"deleted": folder}


# ---------- Models: the speech engines ----------

@router.get("/models")
def models():
    out = []
    for e in library.engines():
        eng = engines.ENGINES[e["id"]]
        out.append({**{k: e[k] for k in ("id", "name", "tier", "license", "ok", "why", "note", "locked", "models", "size_gb",
                                          "missing", "pip", "off")},
                    "short": eng.short, "group": GROUP.get(e["id"], "standard"), "min_ram_gb": eng.min_ram_gb,
                    "downloaded": not e["missing"], "features": sorted(getattr(eng, "features", []) or [])})
    return out


@router.post("/models/{engine_id}/download")
def download_model(engine_id: str):
    if engine_id not in engines.ENGINES:
        raise HTTPException(404, "No such model.")
    e = engines.ENGINES[engine_id]
    ok, why = e.available()
    if not ok and e.min_ram_gb > library.ram_gb():
        raise HTTPException(400, why or f"{e.name} needs {e.min_ram_gb} GB of memory.")
    return _start("model", f"Download {e.name}", _logged(library.download_engine, engine_id))


# ---------- Windows and Linux: PyTorch for the voices (narrator/pc_setup.py) ----------

def _voices_setup():
    import sys
    from . import pc_setup
    from .runtime import HAS_MLX
    if HAS_MLX or sys.platform == "darwin":
        return {"needed": False}
    from .tts import torch_rt
    card = pc_setup.gpu()
    which = pc_setup.build(card)
    return {"needed": bool(torch_rt.missing()), "gpu": card["name"] if card else "", "build": which,
            "size_gb": pc_setup.SIZE_GB[which], "uv": bool(pc_setup.find_uv(HOME))}


@router.get("/voices-setup")
def voices_setup():
    return _voices_setup()


@router.post("/voices-setup")
def start_voices_setup():
    from . import pc_setup
    s = _voices_setup()
    if not s["needed"]:
        raise HTTPException(400, "Voices are already set up here.")
    uv = pc_setup.find_uv(HOME)
    if not uv:
        raise HTTPException(400, "Babble's installer tool (uv) is missing. Reinstall Babble.")

    def setup(log):
        which = pc_setup.install_torch(uv, which=s["build"], log=log)
        import importlib
        importlib.invalidate_caches()
        engines.refresh()
        return which
    return _start("model", "Set up voices", _logged(setup))


class ModelOff(BaseModel):
    off: bool


@router.post("/models/{engine_id}/off")
def model_off(engine_id: str, req: ModelOff):
    if engine_id not in engines.ENGINES:
        raise HTTPException(404, "No such model.")
    _ok(library.set_engine_off, engine_id, req.off)
    return {"id": engine_id, "off": req.off}


@router.delete("/models/{engine_id}")
def delete_engine_model(engine_id: str):
    if engine_id not in engines.ENGINES:
        raise HTTPException(404, "No such model.")
    return {"id": engine_id, "freed_gb": _ok(library.delete_engine, engine_id)}


@router.post("/models/{engine_id}/redownload")
def redownload_model(engine_id: str):
    if engine_id not in engines.ENGINES:
        raise HTTPException(404, "No such model.")
    if library._busy():
        raise HTTPException(400, "Something is being made right now. Download it again when that's done.")
    e = engines.ENGINES[engine_id]
    return _start("model", f"Download {e.name} again", _logged(library.redownload_engine, engine_id))


@router.post("/models/{engine_id}/extras")
def model_extras(engine_id: str):
    if engine_id not in engines.ENGINES:
        raise HTTPException(404, "No such model.")
    return _start("model", f"Extras for {engines.ENGINES[engine_id].name}", _logged(library.install_extras, engine_id))


# ---------- Connections: places outside the app ----------

SSH_FIELDS = {"host": "ssh_host", "port": "ssh_port", "key": "ssh_key", "dir": "ssh_dir", "python": "ssh_python"}


@router.get("/connections")
def connections():
    c = settings.load()
    try:
        from huggingface_hub import get_token
        hf = bool(get_token())
    except Exception:  # noqa: BLE001
        hf = False
    return {"ssh": {k: str(c.get(v) or "") for k, v in SSH_FIELDS.items()},
            "orpheus": {"where": c.get("orpheus_where", "mac"), "url": c.get("orpheus_url", ""),
                        "has_key": bool(c.get("orpheus_key")), "batch": bool(c.get("orpheus_batch"))},
            "huggingface": {"has_token": hf}}


@router.get("/connections/issues")
def connection_issues():
    """Problems services had while the app used them (sign-in, limits, connection), newest first."""
    from . import conn_issues, director, online_sounds
    from .training import BACKENDS
    names = {k: v["name"] for k, v in director.PROVIDERS.items()}
    names.update({k: b.name for k, b in BACKENDS.items()})
    names.update({k: v["name"] for k, v in online_sounds.SOURCES.items()})
    return [dict(v, service=k, name=names.get(k, k))
            for k, v in sorted(conn_issues.issues().items(), key=lambda kv: -kv[1].get("when", 0))]


@router.delete("/connections/issues/{service}")
def dismiss_connection_issue(service: str):
    from . import conn_issues
    conn_issues.ok(service)
    return connection_issues()


@router.post("/connections/{backend}/check")
def check_backend(backend: str):
    from .training import BACKENDS
    if backend == "huggingface":  # (this route would catch /connections/huggingface/check first)
        return check_hf()
    if backend not in BACKENDS:
        raise HTTPException(404, "No such place to train.")
    b = BACKENDS[backend]
    from . import conn_issues
    try:
        c = b.check()
    except Exception as e:  # noqa: BLE001 - shown to the user
        c = {"ok": False, "detail": str(e)}
    if not c.get("ok"):
        conn_issues.note(backend, RuntimeError(c.get("detail") or ""), "Check")
    return {"ok": bool(c.get("ok")), "detail": c.get("detail", ""), "quota": c.get("quota"), "user": c.get("user"),
            "balance": c.get("balance"), "readiness": b.readiness}


# ---------- signing in to the cloud GPUs, from the app (no Terminal) ----------

@router.post("/connections/kaggle/signin")
def kaggle_sign_in():
    """Kaggle's browser sign-in: the task's log has a 'URL <link>' line while it waits."""
    from .training.backends import kaggle
    return _start("signin", "Sign in to Kaggle", _logged(kaggle.sign_in))


class KaggleToken(BaseModel):
    token: str


@router.post("/connections/kaggle/token")
def kaggle_token(req: KaggleToken):
    from .training.backends import kaggle
    user = _ok(kaggle.save_token, req.token)
    return {"ok": True, "user": user}


@router.post("/connections/kaggle/signout")
def kaggle_sign_out():
    from .training.backends import kaggle
    kaggle.sign_out()
    return {"ok": True}


@router.post("/connections/modal/signout")
def modal_sign_out():
    from .training.backends.modal_cloud import step
    _ok(step, "signout", timeout=60)
    return {"ok": True}


@router.delete("/connections/ssh")
def forget_ssh():
    settings.save(**{v: settings.DEFAULTS[v] for v in SSH_FIELDS.values()})  # (back to how it starts)
    return connections()["ssh"]


@router.delete("/connections/orpheus")
def forget_orpheus():
    settings.save(orpheus_where="mac", orpheus_url="", orpheus_key="")
    return connections()["orpheus"]


@router.delete("/connections/huggingface")
def forget_hf_token():
    from huggingface_hub import get_token, logout
    if get_token():
        try:
            logout()
        except Exception as e:  # noqa: BLE001 - e.g. a token set in the environment, which isn't ours to remove
            raise HTTPException(400, f"Couldn't remove the token: {str(e)[:160]}") from e
    return {"has_token": False}


@router.post("/connections/modal/signin")
def modal_sign_in():
    from .training.backends import modal_cloud
    return _start("signin", "Sign in to Modal", _logged(modal_cloud.sign_in))


# ---------- the pay-as-you-go GPU services: an API key pasted in the app ----------

class CloudKey(BaseModel):
    key: str


def _paid(service):
    from .training import BACKENDS
    from .training.backends.machines import PaidCloud
    b = BACKENDS.get(service)
    if not isinstance(b, PaidCloud):
        raise HTTPException(404, "No such cloud GPU service.")
    return b


@router.get("/connections/clouds")
def cloud_services():
    """The pay-as-you-go GPU services for the Create menu and Connections: whether a key is saved (no network),
    their GPUs and list prices where known. The menu checks the ones with a key."""
    from . import settings
    from .training import BACKENDS
    from .training.backends import CLOUDS
    from .training.backends.machines import PaidCloud
    c = settings.load()
    out = []
    for k in CLOUDS:
        b = BACKENDS[k]
        if isinstance(b, PaidCloud):
            out.append({"id": k, "name": b.name, "pricing": b.pricing, "has_key": bool((c.get(b.key_setting) or "").strip()),
                        "default_gpu": b.default_gpu, "sends_models": b.sends_models, "live": bool(b.live),
                        "gpus": [{"id": g, "label": label, "price": getattr(b, "prices", {}).get(g)}
                                 for g, label in b.gpu_names.items()]})
    return out


@router.post("/connections/{service}/key")
def cloud_key(service: str, req: CloudKey):
    """Save a Runpod / Vast.ai / Thunder / Lambda / Lightning / Beam / Cerebrium / Hugging Face Jobs key, once
    the service has accepted it."""
    acc = _ok(_paid(service).save_key, req.key)
    return {"ok": True, **acc}


@router.delete("/connections/{service}/key")
def cloud_forget_key(service: str):
    _paid(service).forget_key()
    return {"ok": True}


@router.get("/connections/{service}/gpus")
def cloud_gpus(service: str):
    """The GPUs the Create menu offers for a service, with its price per hour right now where it says."""
    b = _paid(service)
    try:
        return b.gpus()
    except Exception as e:  # noqa: BLE001 - no key yet, or offline: the menu shows them without prices
        raise HTTPException(400, str(e)[:200]) from e


# ---------- trained voices on Modal (modal_models): uploaded so the cloud can make their lines ----------

@router.get("/connections/modal/models")
def modal_models_list():
    from . import modal_models
    return modal_models.installed()


class ModelFolder(BaseModel):
    folder: str


@router.post("/connections/modal/models/upload")
def modal_models_upload(req: ModelFolder):
    from . import modal_models
    return _start("modal-upload", f"Upload {req.folder} to Modal", _logged(modal_models.upload, req.folder))


@router.delete("/connections/modal/models")
def modal_models_remove(folder: str):
    from . import modal_models
    _ok(modal_models.remove, folder, log=lambda m: None)
    return modal_models.installed()


class SSHSettings(BaseModel):
    host: str = ""
    port: str = "22"
    key: str = ""
    dir: str = "~/narrator-train"
    python: str = "python3"


@router.post("/connections/ssh")
def save_ssh(req: SSHSettings):
    settings.save(**{v: getattr(req, k) for k, v in SSH_FIELDS.items()})
    return connections()["ssh"]


class OrpheusServer(BaseModel):
    where: str = "mac"             # "mac" or "server"
    url: str = ""
    key: Optional[str] = None      # None keeps the saved key
    test: bool = False


@router.post("/connections/orpheus")
def save_orpheus(req: OrpheusServer):
    fields = {"orpheus_where": "server" if req.where == "server" else "mac", "orpheus_url": req.url}
    if req.key is not None:
        fields["orpheus_key"] = req.key
    settings.save(**fields)
    detail = ""
    if req.test and fields["orpheus_where"] == "server":
        try:
            detail = en.orpheus_check()
        except ValueError as e:
            detail = f"Not working: {e}"
    return {**connections()["orpheus"], "detail": detail}


class Token(BaseModel):
    token: str


@router.post("/connections/huggingface")
def save_hf_token(req: Token):
    """Save a Hugging Face access token where Hugging Face's own tools look for it (downloads get faster and
    steadier with one). It's checked first."""
    from huggingface_hub import login, whoami
    token = req.token.strip()
    if not token:
        raise HTTPException(400, "Paste a token first.")
    try:
        who = whoami(token=token)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Hugging Face didn't accept that token: {str(e)[:160]}") from e
    login(token=token, add_to_git_credential=False)
    return {"has_token": True, "user": who.get("name")}


@router.post("/connections/huggingface/check")
def check_hf():
    from huggingface_hub import get_token, whoami
    if not get_token():
        return {"ok": False, "detail": "No token saved. Downloads still work, just slower at busy times."}
    try:
        return {"ok": True, "detail": f"Signed in as {whoami().get('name')}."}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "detail": f"The saved token didn't work: {str(e)[:160]}"}


# ---------- Train a voice ----------

STATE = {"sent": "Sent", "queued": "Waiting for a GPU", "running": "Training", "complete": "Ready to install",
         "installed": "Installed", "error": "Failed", "packaged": "Packaged (test)", "stopped": "Stopped",
         "unknown": "Can't tell", "deleted": "Model deleted"}


def _runs():
    """The training runs; a run whose installed model has since been deleted (here, or its folder removed) says so."""
    from .training import runs
    from .training.base import save_run
    data = runs()
    for key, r in data.items():
        if r.get("state") == "installed" and r.get("installed") and not Path(r["installed"]).exists():
            r.update(save_run(key, state="deleted"))
    return data


def _run_for(folder):
    for key, r in _runs().items():
        if r.get("installed") and Path(r["installed"]).resolve() == Path(folder).resolve():
            return key, r
    return None, None


@router.get("/train/options")
def train_options():
    from .training import BACKENDS, RECIPES
    from .training.backends import CLOUDS
    from .training.backends.modal_cloud import GPUS as MODAL_GPUS
    notes = {"qwen": "More expressive in our tests, and faster on a Mac: it reads the mood from the words. "
                     "Recommended. Fine-tuned voices don't take style instructions.",
             "orpheus": "Understands tags like <laugh> and <sigh>. Slower on a Mac."}
    prep = [{"id": "mac", "name": "This Mac", "pricing": "free, runs locally", "readiness": "ready",
             "default_gpu": "", "gpus": []}]
    for cid in CLOUDS:
        b = BACKENDS.get(cid)
        if not b or b.readiness not in ("ready", "beta"):
            continue
        if cid == "modal":
            gpus = [{"id": g, "label": label, "price": price} for g, (label, price) in MODAL_GPUS.items() if g != "H100"]
            default_gpu = "L4"
        elif hasattr(b, "gpu_names"):
            gpus = [{"id": g, "label": label, "price": getattr(b, "prices", {}).get(g)} for g, label in b.gpu_names.items()]
            default_gpu = getattr(b, "default_gpu", "")
        else:
            gpus, default_gpu = [], ""
        prep.append({"id": b.id, "name": b.name, "pricing": b.pricing, "readiness": b.readiness,
                     "default_gpu": default_gpu, "gpus": gpus})
    return {"recipes": [{"id": r.id, "name": r.name.replace(" LoRA", ""), "ready": r.readiness == "ready",
                         "max_voices": r.max_voices, "note": notes.get(r.id, "")} for r in RECIPES.values()],
            "backends": [{"id": b.id, "name": b.name, "pricing": b.pricing, "readiness": b.readiness}
                         for b in BACKENDS.values() if getattr(b, "training", True)],  # (Runpod: books only)
            "prepare_backends": prep,
            "languages": ["english", "auto", "chinese", "german", "italian", "portuguese", "spanish", "japanese",
                          "korean", "french", "russian"]}


@router.get("/train/models")
def train_models():
    """Models you trained (installed packs are in the Library): each with its voices, and whether more can be
    added (a voice is trained into the model with the ones it has)."""
    from .training import RECIPES
    out = []
    for m in sorted(library.installed(), key=lambda m: m["installed"], reverse=True):
        if m["pack"]:
            continue
        key, run = _run_for(m["folder"])
        recipe = (run or {}).get("recipe") or m["engine"]
        cap = RECIPES[recipe].max_voices if recipe in RECIPES else 8
        vals = {spk: v for v, spk in library.voices_of(m["folder"])}
        voices = [{"speaker": spk, "name": v.get("name", spk), "gender": v.get("gender", "unspecified"),
                   "value": vals.get(spk)} for spk, v in m["voices"].items()]
        why = ""
        if not run:
            why = "This model was made before training runs were saved, so voices can't be added to it."
        elif recipe == "orpheus" and not run.get("chain"):
            why = "This model's saved adapter isn't here, so voices can't be added to it."
        elif len(voices) >= cap:
            why = f"A model holds up to {cap} voices."
        title = m["id"].replace("-orpheus-4bit", "").replace("-qwen-6bit", "").replace("-", " ").title()
        out.append({"id": m["id"], "folder": m["folder"], "title": title, "engine": m["engine"], "recipe": recipe,
                    "run": key, "voices": voices, "max_voices": cap, "can_add": not why, "why": why,
                    "size_gb": m["size_gb"], "hours": m.get("hours")})
    return out


@router.delete("/train/models")
def delete_model(folder: str):
    if not any(m["folder"] == folder for m in library.installed()):
        raise HTTPException(404, "That model isn't installed.")
    key, _ = _run_for(folder)
    _ok(library.delete, folder)
    if key:  # (its run is kept: it can still be trained again)
        from .training.base import save_run
        save_run(key, state="deleted")
    return {"deleted": folder}


@router.get("/train/runs")
def train_runs():
    from .training import BACKENDS, RECIPES
    out = []
    for key, r in sorted(_runs().items(), key=lambda kv: kv[1].get("sent", ""), reverse=True):
        names = r.get("names") or {}
        out.append({"key": key, "model": key.split("/", 1)[-1], "recipe": RECIPES[r["recipe"]].name.replace(" LoRA", "")
                    if r.get("recipe") in RECIPES else r.get("recipe", "?"),
                    "voices": [names.get(s, s.title()) for s in (r.get("voices") or {}).values()],
                    "where": BACKENDS[r["backend"]].name if r.get("backend") in BACKENDS else r.get("backend", "?"),
                    "state": r.get("state", "unknown"), "label": STATE.get(r.get("state"), r.get("state", "?")),
                    "url": r.get("url", ""), "sent": r.get("sent", ""), "hours": r.get("hours"),
                    "gpu_hours": r.get("gpu_hours"), "installed": r.get("installed"), "error": r.get("error"),
                    "can_retry": bool(r.get("voices")) and not r.get("start_from") and r.get("recipe") in RECIPES})
    return out


@router.post("/train/runs/check")
def check_run(key: str):
    from . import training
    try:
        st = training.check(key)
    except Exception as e:  # noqa: BLE001 - shown to the user
        raise HTTPException(400, f"Couldn't check {key.split('/', 1)[-1]}: {e}") from e
    return {"state": st.state, "label": STATE.get(st.state, st.state), "detail": st.detail}


@router.post("/train/runs/retry")
def retry_run(key: str):
    """Send a run that failed again, with the same voices, model and settings (and the current training script)."""
    from . import training
    r = _runs().get(key)
    if not r:
        raise HTTPException(404, "No such run.")
    if r.get("state") == "installed":
        raise HTTPException(400, "That model is already installed.")
    if r.get("start_from") or not r.get("voices"):
        raise HTTPException(400, "Add these voices again from Train a voice: this run built on an earlier one.")
    specs = [f"{k}={s}" for k, s in r["voices"].items()]
    kw = {"max_clips": int(r.get("max_clips") or 0)}
    if r.get("epochs"):
        kw["epochs"] = float(r["epochs"])
    return _start("train", f"Train {r['model']} again", _logged(
        training.send, specs, r["model"], recipe=r["recipe"], backend=r["backend"], names=r.get("names") or {},
        genders=r.get("genders") or {}, consent=bool(r.get("consent")), **kw, done=lambda _: {"run": key}))


@router.post("/train/runs/cancel")
def cancel_run(key: str):
    from . import training
    if key not in _runs():
        raise HTTPException(404, "No such run.")
    try:
        training.cancel(key)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # noqa: BLE001 - the service's own error, shown to the user
        raise HTTPException(400, f"Couldn't cancel it: {str(e)[:300]}") from e
    return {"cancelled": key}


@router.delete("/train/runs")
def forget_run(key: str):
    from . import training
    if key not in _runs():
        raise HTTPException(404, "No such run.")
    _ok(training.forget, key)
    return {"forgot": key}


@router.post("/train/runs/finish")
def finish_run(key: str):
    from . import training
    if key not in _runs():
        raise HTTPException(404, "No such run.")
    if training.installing(key):
        raise HTTPException(400, "This model is already downloading and installing.")
    r = _runs()[key]
    if r.get("state") == "installed" and r.get("installed") and Path(r["installed"]).exists():
        raise HTTPException(400, "Already installed. Delete it on Train a voice first to install it again.")
    return _start("train", f"Install {key.split('/', 1)[-1]}",
                  _logged(training.finish, key, done=lambda d: {"folder": str(d)}))


@router.get("/train/runs/samples")
def run_samples(key: str):
    from . import training
    try:
        d = training.results_dir(key) / "samples"
    except KeyError:
        return []
    return [{"file": str(f), "name": f.stem} for f in sorted(d.glob("*.wav"))[:6]] if d.exists() else []


# preparing a voice (step 1): an audiobook and the same book's text become clean, matched clips

def _pipeline():
    from .dataset import pipeline
    return pipeline


@router.get("/train/voices")
def prepared_voices():
    """Voices being prepared or ready to train: [{name, clips, hours, ready, running, stage, progress}]."""
    from . import training
    p = _pipeline()
    ready = {v: (n, h) for v, n, h in training.ready_voices()}
    out = []
    for name in sorted(set(p.voices()) | set(ready)):
        job = p._jobs.get(name)
        running = bool(job and job["thread"].is_alive())
        n, h = ready.get(name, (0, 0.0))
        out.append({"name": name, "clips": n, "hours": round(h, 2), "ready": name in ready, "running": running,
                    "stage": job["stage"] if job else "", "progress": job.get("progress") if job else None,
                    "where": job.get("where", "this Mac") if job else "this Mac",
                    "unfinished": bool(p.unfinished(name)) and not running})
    return out


# training tools: merge several book files into one text, or several audio files into one recording

def _keep_uploads(files):
    """Save uploads in the order they came (001-<name>, …) to a folder of their own; returns (folder, paths)."""
    import tempfile
    import time
    from .training_tools import TOOLS
    if not files:
        raise HTTPException(400, "Add the files to merge first.")
    (TOOLS / "_in").mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=time.strftime("%Y%m%d-%H%M%S-"), dir=TOOLS / "_in"))  # one per merge
    paths = []
    for i, f in enumerate(files, 1):
        name = re.sub(r"[^\w.\- ]+", "-", Path(f.filename or "file").name)[-120:] or "file"
        dest = folder / f"{i:03d}-{name}"
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        paths.append(dest)
    return folder, paths


@router.post("/train/tools/merge-text")
def merge_text(files: list[UploadFile] = File(default=[]), name: str = Form(""), headings: bool = Form(True)):
    from . import training_tools as tt
    folder, paths = _keep_uploads(files)
    return _start("tools", f"Merge {len(paths)} text files", _logged(
        tt.merge_text, paths, name=name, headings=headings, cleanup=folder, done=lambda r: r))


@router.post("/train/tools/merge-audio")
def merge_audio(files: list[UploadFile] = File(default=[]), name: str = Form(""), format: str = Form("flac")):
    from . import training_tools as tt
    folder, paths = _keep_uploads(files)
    return _start("tools", f"Merge {len(paths)} audio files", _logged(
        tt.merge_audio, paths, name=name, fmt=format, cleanup=folder, done=lambda r: r))


class PrepareEstimate(BaseModel):
    backend: str
    cloud_gpu: str = ""
    audio_seconds: float = 3600.0
    voice: str = ""             # carrying on: what's left of this voice's unfinished book


def _left_to_transcribe(voice):
    """Seconds of an unfinished voice's audio not transcribed yet (None: its audio isn't converted yet)."""
    from . import cloud_stt
    p = _pipeline()
    pending = p.unfinished(voice)
    if not pending:
        return None
    work = TRAINING / voice / "books" / p.slug(pending[0]["title"], "book")
    wavs = sorted((work / "audio").glob("*.wav"))
    return cloud_stt.remaining_seconds(wavs, work / "stt") if wavs else None


def _sent_to(p, voice):
    """Where an unfinished voice was being transcribed (Carry on asks before sending it there again)."""
    pending = p.unfinished(voice)
    o = (pending[0].get("opts") or {}) if pending else {}
    return {"backend": str(o.get("backend") or "mac"), "cloud_gpu": str(o.get("cloud_gpu") or "")}


@router.post("/train/prepare/estimate")
def prepare_estimate(req: PrepareEstimate):
    from . import cloud_stt
    seconds = float(req.audio_seconds or 3600.0)
    if req.voice:
        left = _left_to_transcribe(_pipeline().slug(req.voice, ""))
        if left is not None:
            seconds = left
    if seconds < 1:  # nothing left to send: carrying on costs nothing
        return {"cost": "nothing: it's all transcribed", "time": "", "free": True, "where": cloud_stt.where_name(req.backend),
                "gpu": "", "hours": 0.0, "budget": 0.0, "needed": False}
    return {**_ok(cloud_stt.estimate, req.backend, req.cloud_gpu, max(60.0, seconds)), "needed": True}


@router.post("/train/prepare")
def prepare(name: str = Form(...), title: str = Form(""), language: str = Form("english"),
            options: str = Form("{}"), backend: str = Form("mac"), cloud_gpu: str = Form(""),
            files: list[UploadFile] = File(default=[])):
    from .training.backends import CLOUDS
    p = _pipeline()
    voice = p.slug(name or "", "")
    if not voice:
        raise HTTPException(400, "Give the voice a name first.")
    where = (backend or "").strip()  # (empty when carrying on: the place it was sent to before)
    if where and where != "mac" and where not in CLOUDS:
        raise HTTPException(400, f"No such cloud GPU: {where}.")
    try:
        opts = {k: float(v) for k, v in json.loads(options or "{}").items()}
    except (ValueError, TypeError) as e:
        raise HTTPException(400, "Those options aren't numbers.") from e
    defaults = {"min_seconds": 2.0, "max_seconds": 15.0, "target_seconds": 8.0, "min_match": 0.9, "min_snr": 20.0,
                "split_narrators": 0.0,  # 1: more than one narrator, each becomes a voice of their own
                "dialogue_only": 0.0,    # 1: only the words in quotation marks (a voice that performs lines)
                "narrators": 0.0}        # how many (0: find out)
    opts = {**defaults, **{k: v for k, v in opts.items() if k in defaults}, "backend": where or "mac",
            "cloud_gpu": (cloud_gpu or "").strip()}
    if not files:  # carry on where it left off
        pending = p.unfinished(voice)
        if not pending:
            raise HTTPException(400, "Add the audiobook and the book. (Nothing unfinished to carry on for this voice.)")
        j = pending[0]
        missing = [x for x in [*j["audio"], j["book"]] if not Path(x).exists()]
        done_audio = (TRAINING / voice / "books" / p.slug(j["title"], "book") / "audio").exists()
        if missing and not done_audio:
            raise HTTPException(400, "The original files are gone: add them again to carry on (finished parts are reused).")
        resumed_opts = dict(j["opts"])
        if where:  # (told where this time; else where it was sent before)
            resumed_opts["backend"] = where
            resumed_opts["cloud_gpu"] = (cloud_gpu or "").strip()
        _ok(p.start, voice, [Path(x) for x in j["audio"]], Path(j["book"]), j["title"], j["language"], resumed_opts)
        return {"voice": voice, "title": j["title"], "resumed": True}
    up = TRAINING / voice / "uploads" / "incoming"
    up.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in files:
        dest = up / re.sub(r"[^\w.\- ]+", "-", Path(f.filename or "file").name)[-120:]
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out, 1 << 20)
        paths.append(str(dest))
    audio, books = p.sort_inputs(paths, TRAINING / voice / "uploads")
    if not audio:
        raise HTTPException(400, "Add the audiobook's audio files.")
    if not books:
        raise HTTPException(400, "Add the book's text (EPUB, PDF, TXT, DOCX…), the same edition as the audio.")
    title = (title or "").strip() or p.guess_title(audio, books[0])
    _ok(p.start, voice, audio, books[0], title, language, opts)
    return {"voice": voice, "title": title, "resumed": False}


def _safe_voice(voice: str) -> str:
    """Validate a training voice slug and verify it resolves directly under TRAINING."""
    p = _pipeline()
    cleaned = p.slug(voice or "", "")
    if not cleaned or cleaned != voice:
        raise HTTPException(400, "Invalid voice name.")
    target = (TRAINING / cleaned).resolve()
    if target.parent != TRAINING.resolve():
        raise HTTPException(400, "Invalid voice name.")
    return cleaned


@router.get("/train/prepare/{voice}")
def prepare_status(voice: str):
    voice = _safe_voice(voice)
    p = _pipeline()
    job = p._jobs.get(voice)
    md, log = p.status(voice)
    stats_path = TRAINING / voice / "dataset" / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else None
    import time
    return {"voice": voice, "running": bool(job and job["thread"].is_alive()), "stage": job["stage"] if job else "",
            "progress": job.get("progress") if job else None, "error": job.get("error") if job else None,
            "done": bool(job and job.get("done")), "elapsed": round(time.time() - job["started"]) if job else None,
            "title": job["title"] if job else "", "where": job.get("where", "this Mac") if job else "this Mac",
            **_sent_to(p, voice),
            "log": log or "",
            "books": [{"title": r[0], "audio_h": r[1], "kept_h": r[2], "clips": r[3], "matched": r[4]}
                      for r in p.books_table(voice)],
            "dataset": {"clips": stats["clips"], "hours": round(stats["hours"], 2)} if stats else None,
            "unfinished": bool(p.unfinished(voice))}


@router.post("/train/prepare/{voice}/stop")
def prepare_stop(voice: str):
    voice = _safe_voice(voice)
    return {"stopping": _pipeline().stop(voice)}


@router.get("/train/prepare/{voice}/clips")
def prepare_clips(voice: str):
    voice = _safe_voice(voice)
    return [{"file": f, "text": t} for f, t in _pipeline().samples(voice, 4)]


# training a model (step 2)

class TrainVoice(BaseModel):
    dataset: str        # a prepared voice's name
    name: str = ""      # shown as (e.g. "Helen")
    gender: str = "unspecified"


class Train(BaseModel):
    voices: list[TrainVoice]
    model: str = ""             # a new model's name
    recipe: str = "orpheus"
    backend: str = "kaggle"
    epochs: Optional[float] = None
    max_clips: int = 0
    add_to: Optional[str] = None  # an installed model's folder: the voices are trained into it, with the ones it has
    consent: bool = False


def _speaker(text):
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower()) or "voice"


def _train_plan(req: Train):
    """(specs, model, recipe, kwargs) for training.send / estimate, adding to an existing model if asked."""
    from .training import RECIPES
    if not req.voices:
        raise HTTPException(400, "Prepare a voice first.")
    new = {}
    for v in req.voices:
        spk = _speaker(v.name or v.dataset.split("-")[0])
        if spk in new:
            raise HTTPException(400, "Each voice needs its own name.")
        new[spk] = v
    names = {s: (v.name.strip() or s.title()) for s, v in new.items()}
    genders = {s: v.gender if v.gender in ("female", "male") else "unspecified" for s, v in new.items()}
    specs = [f"{v.dataset}={s}" for s, v in new.items()]
    kw = {"max_clips": int(req.max_clips or 0)}
    if req.epochs:
        kw["epochs"] = float(req.epochs)
    recipe, model = req.recipe, re.sub(r"[^a-z0-9]+", "", (req.model or "").lower())
    if req.add_to:
        key, run = _run_for(req.add_to)
        if not run:
            raise HTTPException(400, "Voices can't be added to this model (it was made before training runs were saved).")
        recipe, model = run["recipe"], run["model"]
        old = dict(run.get("voices") or {})  # dataset key -> speaker
        if set(new) & set(old.values()):
            raise HTTPException(400, f"The model already has a voice called {', '.join(sorted(set(new) & set(old.values())))}.")
        names = {**(run.get("names") or {}), **names}
        genders = {**(run.get("genders") or {}), **genders}
        if recipe == "orpheus":
            kw["start_from"] = model  # trains on top of its saved adapters, keeping the old voices sharp
        else:  # Qwen: the model is trained again with every voice
            specs = [f"{k}={s}" for k, s in old.items()] + specs
        total = len(set(old.values()) | set(new))
    else:
        if not model:
            raise HTTPException(400, "Give the model a name.")
        total = len(new)
    cap = RECIPES[recipe].max_voices if recipe in RECIPES else 8
    if total > cap:
        raise HTTPException(400, f"A model holds up to {cap} voices; this would make it {total}.")
    return specs, model, recipe, kw, names, genders


@router.post("/train/estimate")
def train_estimate(req: Train):
    specs, _, recipe, kw, _, _ = _train_plan(req)
    return _estimate_answer(specs, recipe, req.backend, kw.get("epochs"), kw.get("max_clips", 0))


@router.post("/train/runs/retry-estimate")
def retry_estimate(key: str):
    """What sending this run again would take and cost (what Train again asks before it sends)."""
    r = _runs().get(key)
    if not r or not r.get("voices"):
        raise HTTPException(404, "No such run.")
    specs = [f"{k}={s}" for k, s in r["voices"].items()]
    return _estimate_answer(specs, r["recipe"], r["backend"], float(r["epochs"]) if r.get("epochs") else None,
                            int(r.get("max_clips") or 0))


def _estimate_answer(specs, recipe, backend, epochs=None, max_clips=0):
    from . import training
    from .training import BACKENDS
    e = training.estimate(specs, recipe, backend, epochs, max_clips)
    text = f"About {e['hours']} h of audio, roughly {e['gpu_hours']} GPU hours on {e['backend']}."
    if backend == "kaggle":  # the week's GPU hours: a run that doesn't fit trains for what's left, then saves
        q = BACKENDS["kaggle"].check().get("quota")
        if q and q["left"] < e["gpu_hours"]:
            budget = min(11.0, q["left"] - 0.3)
            text += (f" You have {q['left']:.1f} left this week (it resets {q['resets']}), so it would train for about "
                     f"{max(budget, 0):.1f} h and save what it learned by then." if budget >= 1 else
                     f" You have {q['left']:.1f} left this week: not enough to train. It resets {q['resets']}.")
        elif q:
            text += f" You have {q['left']:.1f} left this week."
    from . import pricing, spend
    price = spend.gpu_price(backend, "T4")  # (training asks for the smallest GPU that fits)
    limit = max(1.0, e["gpu_hours"] * 2.5)
    r = pricing.gpu_range(backend, price, e["gpu_hours"], limit, "train")
    cost = pricing.say(r)
    if price:
        text += f" About {cost}."
    return {**e, "text": text, "cost": cost, "free": backend in ("kaggle", "local-mac", "ssh"), "paid": bool(price),
            "backend_id": backend}


@router.post("/train/send")
def train_send(req: Train):
    from . import training
    from .training import BACKENDS, RECIPES
    if not req.consent:
        raise HTTPException(400, "Tick the permission box first.")
    specs, model, recipe, kw, names, genders = _train_plan(req)
    if RECIPES[recipe].readiness != "ready" or BACKENDS[req.backend].readiness not in ("ready", "beta"):
        raise HTTPException(400, "That model type or place to train isn't available yet.")
    return _start("train", f"Train {model}", _logged(training.send, specs, model, recipe=recipe, backend=req.backend,
                                                     names=names, genders=genders, consent=True, **kw,
                                                     done=lambda r: {"run": f"{recipe}/{model}"}))


class Export(BaseModel):
    folder: str
    name: str
    genre: str = "General"
    version: int = 1
    description: str = ""
    author: str = ""
    rights: dict = {}           # {speaker: {"source", "license", "consent"}}
    previews: bool = True
    sign: bool = False          # sign it as the GitHub account in Settings → General (signing.py)


@router.post("/train/export")
def export_pack(req: Export):
    if not any(m["folder"] == req.folder for m in library.installed()):
        raise HTTPException(404, "That model isn't installed.")
    if not req.name.strip():
        raise HTTPException(400, "Give the pack a name.")
    rights = {str(s): {k: str((r or {}).get(k, "")).strip() for k in ("source", "license", "consent")}
              for s, r in (req.rights or {}).items()}
    sign_as = None
    if req.sign:
        from . import agents
        if agents.current():
            raise HTTPException(403, "Only you can sign a pack with your name.")
        sign_as = signing.github_user()
        if not sign_as or not signing.public_line():
            raise HTTPException(400, "Set up signing in Settings → General first.")

    def work(log):
        prev = {}
        if req.previews:
            for value, speaker in library.voices_of(req.folder):
                log(f"Recording a preview of {vl.short_name(value)}…")
                for p, _ in en.render([dict(text=PREVIEW, voice=value, pause=0.2)], kind="preview", language="english",
                                      temperature=0.8, batch=1):
                    if p:
                        prev[speaker] = p
        return library.export_pack(req.folder, req.name.strip(), req.genre, req.description, rights, req.author,
                                   int(req.version or 1), prev, log=log, sign_as=sign_as)
    return _start("pack", f"Export {req.name.strip()}", _logged(work, done=lambda p: {"file": str(p)}))
