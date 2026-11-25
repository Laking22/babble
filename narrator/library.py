"""Library: installed voice models, voice packs (install / export), and engine downloads.

A voice pack is a zip ("<id>.narrator-pack") holding one installed model folder plus:
  pack.json   {"format": 1, "id", "name", "genre", "description", "version", "engine", "author",
               "voices": [{"speaker", "name", "gender", "description", "preview": "previews/<speaker>.wav",
                           "source", "license", "consent"}]}
  previews/   one short wav per voice
Packs are installed into paths.MODELS/<id>/ with a voice.json generated from pack.json, so every engine lists
their voices like any trained model. A catalog is a JSON list of pack entries with "url", "sha256" and "size".
The built-in catalog is catalog.json in the public babble-voices repo (CATALOG_URL; the pack_catalog_url setting
overrides it, "" turns it off). Its "url" and each voice's "preview" may be relative to the catalog. Previews are
fetched one at a time when someone presses play, so a pack can be heard before it's downloaded.
Built-in packs must only contain voices we have the right to share (see PLAN.md: free-to-use voices only).
"""
import hashlib
import json
import re
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

from . import settings, signing, tts
from .paths import HOME, MODELS, OUTPUTS

PACKS = HOME / "packs"
PACKS.mkdir(parents=True, exist_ok=True)
PREVIEWS = HOME / "cache" / "pack-previews"
CATALOG_URL = "https://raw.githubusercontent.com/Laking22/babble-voices/main/catalog.json"
MAX_CATALOG = 2 << 20
MAX_PREVIEW = 15 << 20
PREVIEW_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
CATALOG_TTL = 600  # seconds the online list is kept; a failed fetch is tried again after a minute
GENRES = ["Fantasy", "Sci-Fi", "Thriller & Mystery", "Romance", "Horror", "Literary", "Non-fiction", "Kids",
          "Comedy", "Historical", "General"]
FORMAT = 1


def _size(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


def _slug(text):
    return re.sub(r"[^a-z0-9-]+", "-", (text or "").lower()).strip("-")[:60]


# ---------- installed models ----------

def installed():
    """Every model folder with a voice.json: [{folder, id, engine, voices, size_gb, source, pack}]."""
    out = []
    for meta in sorted(MODELS.glob("*/voice.json")):
        try:
            data = json.loads(meta.read_text())
        except ValueError:
            continue
        pack = None
        if (meta.parent / "pack.json").exists():
            try:
                pack = json.loads((meta.parent / "pack.json").read_text())
            except ValueError:
                pass
        signed = _signed(meta.parent) if pack else None
        source = (f"Pack: {pack.get('name')} v{pack.get('version', 1)}" if pack else
                  f"Trained ({data.get('trained_on') or 'unknown'}, {data.get('hours') or '?'} h)")
        st = meta.parent.stat()
        out.append({"folder": str(meta.parent), "id": meta.parent.name, "engine": data.get("engine", "orpheus"),
                    "voices": data.get("voices", {}), "size_gb": round(_size(meta.parent) / 1e9, 2),
                    "source": source, "pack": pack, "signed": signed, "linked": meta.parent.is_symlink(),
                    "target": str(meta.parent.resolve()) if meta.parent.is_symlink() else "",
                    "installed": getattr(st, "st_birthtime", st.st_mtime), "hours": data.get("hours"),
                    "trained_on": data.get("trained_on") or ("kaggle" if data.get("kaggle") else None)})
    return out


def _signed(folder):
    """An installed pack's signature check (done at install; checked again when GitHub couldn't be reached then)."""
    sig_file, check_file = Path(folder) / signing.FILE, Path(folder) / "signature-check.json"
    if not sig_file.exists():
        return None
    try:
        res = json.loads(check_file.read_text()) if check_file.exists() else {"verified": None}
        if res.get("verified") is None and time.time() - float(res.get("checked") or 0) > 60:
            res = dict(signing.check(json.loads(sig_file.read_text())), checked=time.time())
            check_file.write_text(json.dumps(res))
        return {k: res.get(k) for k in ("github", "fingerprint", "verified", "why")}
    except (OSError, ValueError):
        return None


def delete(folder):
    """Remove an installed model. A model linked in from Qwen TTS Studio only loses its link here."""
    folder = Path(folder)
    if folder.parent.resolve() != MODELS.resolve() or not (folder / "voice.json").exists():
        raise ValueError("That isn't an installed model folder.")
    if folder.is_symlink():
        folder.unlink()
    else:
        shutil.rmtree(folder)
    tts.refresh()


def voices_of(folder):
    """[(voice value, trained speaker name)] for an installed model folder, e.g. ("orpheus:helen-narrators", "helen")."""
    tts.refresh()
    prefix = {"qwen": "qwen", "orpheus": "orpheus"}
    out = []
    for e in tts.ENGINES.values():
        for key, row in (getattr(e, "tuned", None) or {}).items():
            if Path(row[3]).resolve() == Path(folder).resolve():
                out.append((f"{prefix.get(e.id, e.prefixes[0])}:{key}", row[4]))
    return out


# ---------- packs ----------

def export_pack(folder, name, genre, description, rights, author="", version=1, previews=None, log=print, sign_as=None):
    """Zip an installed model as a pack. rights: {speaker: {"source", "license", "consent"}}.
    previews: {speaker: wav path}. sign_as: a GitHub username to sign it as (signing.py). Returns the zip path."""
    folder = Path(folder)
    vj = json.loads((folder / "voice.json").read_text())
    missing = [s for s in vj["voices"] if not (rights.get(s) or {}).get("source") or not (rights.get(s) or {}).get("license")]
    if missing:
        raise ValueError(f"Fill in the source and license for: {', '.join(missing)}. Packs need a rights record per voice.")
    pid = _slug(name) or folder.name
    pack = {"format": FORMAT, "id": pid, "name": name, "genre": genre, "description": description,
            "version": int(version), "engine": vj.get("engine", "orpheus"), "author": author, "voices": []}
    for s, v in vj["voices"].items():
        pack["voices"].append({"speaker": s, "name": v.get("name", s), "gender": v.get("gender", "unspecified"),
                               "description": v.get("description", ""),
                               "preview": f"previews/{s}.wav" if previews and previews.get(s) else None,
                               **{k: (rights.get(s) or {}).get(k, "") for k in ("source", "license", "consent")}})
    out_dir = OUTPUTS / "packs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{pid}-v{int(version)}.narrator-pack"
    log(f"Packing {folder.name} ({_size(folder) / 1e9:.1f} GB)…")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:  # weights don't compress
        z.writestr("pack.json", json.dumps(pack, indent=1, ensure_ascii=False))
        for f in folder.rglob("*"):
            if f.is_file() and f.name not in ("voice.json", "pack.json"):
                z.write(f, f"model/{f.relative_to(folder).as_posix()}")
        for s, p in (previews or {}).items():
            if p:
                z.write(p, f"previews/{s}.wav")
    sig = None
    if sign_as:
        log(f"Signing it as {sign_as}…")
        sig = signing.sign_zip(out, sign_as)
    h = hashlib.sha256()
    with open(out, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    sha = h.hexdigest()
    entry = {k: pack[k] for k in ("id", "name", "genre", "description", "version", "engine", "author")}
    entry.update(voices=[{k: v[k] for k in ("speaker", "name", "gender", "license")} for v in pack["voices"]],
                 file=out.name, size=out.stat().st_size, sha256=sha, **({"signature": sig} if sig else {}))
    (out_dir / f"{pid}-v{int(version)}.json").write_text(json.dumps(entry, indent=1))
    log(f"Pack written: {out} (sha256 {sha[:12]}…). A catalog entry is next to it.")
    return out


MAX_PACK_FILES = 2000
MAX_PACK_UNCOMPRESSED = 25 * (1 << 30)  # 25 GiB cap against zip bombs
UNSAFE_PACK_EXTS = {".pt", ".pth", ".bin", ".pkl", ".pickle", ".py", ".pyc", ".sh", ".command", ".so", ".dylib", ".exe"}


def _safe_remote_url(url: str) -> str:
    """Ensure remote pack URLs are HTTPS and do not resolve to loopback/private/link-local hosts."""
    import ipaddress
    import socket
    from urllib.parse import urlparse
    u = urlparse((url or "").strip())
    if u.scheme != "https" or not u.hostname:
        raise ValueError("Remote voice pack URLs must use https://.")
    host = u.hostname.strip().lower()
    if host in ("localhost", "localhost.localdomain") or host.endswith(".local"):
        raise ValueError("Refusing to download from a local host.")
    try:
        infos = socket.getaddrinfo(host, u.port or 443, type=socket.SOCK_STREAM)
    except OSError as e:
        raise ValueError(f"Could not resolve pack host {host}: {e}") from e
    for _, _, _, _, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ValueError(f"Refusing to download from non-public address ({ip}).")
    return url


def install_pack(path, expected_sha256=None, log=print):
    """Install a .narrator-pack zip into paths.MODELS/<id>/. Returns the folder."""
    import stat
    path = Path(path)
    if expected_sha256:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != expected_sha256:
            raise ValueError("The pack file doesn't match its checksum (damaged or changed download).")
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        if len(infos) > MAX_PACK_FILES:
            raise ValueError(f"Voice pack has too many entries ({len(infos)} > {MAX_PACK_FILES}).")
        names = [i.filename for i in infos]
        if "pack.json" not in names:
            raise ValueError("Not a voice pack (no pack.json).")
        total_uncompressed = 0
        for info in infos:
            n = info.filename
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"Unsafe symlink in pack: {n}")
            if n.startswith("/") or "\\" in n or ".." in Path(n).parts:
                raise ValueError("Unsafe path in pack.")
            if Path(n).suffix.lower() in UNSAFE_PACK_EXTS:
                raise ValueError(
                    f"Unsafe file format in pack ({Path(n).name}): only safetensors weights and metadata are allowed."
                )
            total_uncompressed += max(0, int(info.file_size))
            if total_uncompressed > MAX_PACK_UNCOMPRESSED:
                raise ValueError("Voice pack exceeds maximum uncompressed size.")
        sig = signing.read(z)
        signed = None
        if sig:
            log("Checking its signature…")
            signed = signing.check(sig, signing.manifest(z))
            if signed["verified"] is False and signed["why"] in ("The signature doesn't match.",
                                                                 "The pack was changed after it was signed."):
                raise ValueError(f"Not installed: {signed['why'][0].lower()}{signed['why'][1:]}")
        pack = json.loads(z.read("pack.json"))
        if int(pack.get("format", 0)) > FORMAT:
            raise ValueError("This pack needs a newer version of the app.")
        pid = _slug(pack.get("id"))
        if not pid:
            raise ValueError("Voice pack id must contain ASCII letters or numbers.")
        dest = (MODELS / pid).resolve()
        if dest.parent != MODELS.resolve():
            raise ValueError("Unsafe pack destination path.")
        if dest.exists() and not (dest / "pack.json").exists():
            raise ValueError(f"A trained model named '{pid}' already exists; refusing to overwrite it with a pack.")
        tmp = Path(tempfile.mkdtemp(dir=MODELS))
        tmp_resolved = tmp.resolve()
        try:
            log(f"Installing {pack['name']} ({len(pack['voices'])} voices)…")
            extracted = 0
            for info in infos:
                n = info.filename
                if n.endswith("/"):
                    continue
                if n.startswith("model/"):
                    target = (tmp / n[len("model/"):]).resolve()
                elif n.startswith("previews/"):
                    target = (tmp / n).resolve()
                else:
                    continue
                if not target.is_relative_to(tmp_resolved):
                    raise ValueError("Unsafe path escape in pack.")
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(target, "wb") as dst:
                    while True:
                        buf = src.read(1 << 20)
                        if not buf:
                            break
                        extracted += len(buf)
                        if extracted > MAX_PACK_UNCOMPRESSED:
                            raise ValueError("Voice pack exceeded maximum extracted size.")
                        dst.write(buf)
            (tmp / "pack.json").write_text(json.dumps(pack, indent=1, ensure_ascii=False))
            if sig:
                (tmp / signing.FILE).write_text(json.dumps(sig, indent=1))
                (tmp / "signature-check.json").write_text(json.dumps(dict(signed, checked=time.time())))
            voices = {v["speaker"]: {"name": v["name"], "gender": v.get("gender", "unspecified"),
                                     "description": v.get("description") or pack.get("genre", "")} for v in pack["voices"]}
            (tmp / "voice.json").write_text(json.dumps({"engine": pack.get("engine", "orpheus"), "voices": voices,
                                                        "pack": pack["id"], "trained_on": "pack"}, indent=1))
            cfg = tmp / "config.json"
            if cfg.exists():  # models load their tokenizer from their own folder
                c = json.loads(cfg.read_text())
                if "tokenizer_name" in c:
                    c["tokenizer_name"] = str(dest)
                    cfg.write_text(json.dumps(c, indent=2))
            if dest.exists():
                shutil.rmtree(dest)
            tmp.rename(dest)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
    tts.refresh()
    log(f"Installed {pack['name']} in {dest}")
    return dest


def catalog_url():
    return (settings.load().get("pack_catalog_url", CATALOG_URL) or "").strip()


def _fetch(url, cap):
    """GET a small public https file (redirects checked hop by hop), at most cap bytes."""
    import httpx
    from urllib.parse import urljoin
    url = _safe_remote_url(url)
    with httpx.Client(timeout=20, follow_redirects=False) as client:
        for _ in range(6):
            with client.stream("GET", url) as r:
                if r.is_redirect and r.headers.get("location"):
                    url = _safe_remote_url(urljoin(str(r.url), r.headers["location"]))
                    continue
                r.raise_for_status()
                buf = bytearray()
                for chunk in r.iter_bytes(1 << 16):
                    buf += chunk
                    if len(buf) > cap:
                        raise ValueError("The file is larger than expected.")
                return bytes(buf)
    raise ValueError("Too many redirects.")


_online_cache = {"url": None, "at": 0.0, "entries": []}


def _online():
    """The online catalog's entries, with their pack and preview links made absolute."""
    from urllib.parse import urljoin
    url = catalog_url()
    if not url:
        return []
    c = _online_cache
    if c["url"] == url and time.time() - c["at"] < CATALOG_TTL:
        return c["entries"]
    try:
        data = json.loads(_fetch(url, MAX_CATALOG))
        rows = data.get("packs", []) if isinstance(data, dict) else data
        entries = []
        for e in rows if isinstance(rows, list) else []:
            if not isinstance(e, dict) or not e.get("id") or not e.get("url"):
                continue
            voices = [dict(v, preview=urljoin(url, v["preview"]) if v.get("preview") else None)
                      for v in e.get("voices") or [] if isinstance(v, dict)]
            entries.append(dict(e, url=urljoin(url, e["url"]), voices=voices, where="online"))
        c.update(url=url, at=time.time(), entries=entries)
    except Exception:  # noqa: BLE001 - offline or bad catalog: keep the last good list, try again in a minute
        keep = c["entries"] if c["url"] == url else []
        c.update(url=url, at=time.time() - CATALOG_TTL + 60, entries=keep)
    return c["entries"]


def catalog():
    """Pack catalog entries: the online catalog, plus local *.json next to downloaded/exported packs."""
    entries = {}
    for e in _online():
        entries[(e["id"], e.get("version", 1))] = dict(e)
    for f in sorted((OUTPUTS / "packs").glob("*.json")) + sorted(PACKS.glob("*.json")):
        try:
            e = json.loads(f.read_text())
            where = f.parent / e["file"]
        except (ValueError, KeyError, TypeError):
            continue
        key = (e["id"], e.get("version", 1))
        if key in entries:
            continue
        try:
            with zipfile.ZipFile(where) as z:
                names = set(z.namelist())
        except (OSError, zipfile.BadZipFile):
            names = set()
        voices = [dict(v, preview=f"previews/{v.get('speaker')}.wav" if f"previews/{v.get('speaker')}.wav" in names else None)
                  for v in e.get("voices") or [] if isinstance(v, dict)]
        entries[key] = dict(e, voices=voices, where=str(where))
    have = {(m["pack"] or {}).get("id"): (m["pack"] or {}).get("version") for m in installed() if m["pack"]}
    out = []
    for (pid, ver), e in entries.items():
        e["installed"] = have.get(pid) == ver
        out.append(e)
    return out


def preview_file(pid, version, speaker):
    """A catalog voice's preview on this Mac (fetched once, then kept in the cache). For packs not installed yet."""
    from urllib.parse import urlparse
    entry = next((e for e in catalog() if e["id"] == pid and int(e.get("version", 1)) == int(version)), None)
    if not entry:
        raise LookupError("That pack isn't in the list any more.")
    voice = next((v for v in entry.get("voices") or [] if v.get("speaker") == speaker), None)
    if not voice or not voice.get("preview"):
        raise LookupError("This voice has no preview.")
    folder = PREVIEWS / f"{_slug(pid) or 'pack'}-v{int(version)}"
    name = _slug(speaker) or "voice"
    for f in folder.glob(f"{name}.*"):
        if f.suffix in PREVIEW_EXTS:
            return f
    if entry.get("where") == "online":
        ext = Path(urlparse(voice["preview"]).path).suffix.lower()
        if ext not in PREVIEW_EXTS:
            raise ValueError("That preview isn't an audio file.")
        data = _fetch(voice["preview"], MAX_PREVIEW)
    else:  # a pack file on this Mac
        ext = ".wav"
        with zipfile.ZipFile(entry["where"]) as z:
            info = z.getinfo(voice["preview"])
            if info.file_size > MAX_PREVIEW:
                raise ValueError("The preview is larger than expected.")
            data = z.read(info)
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{name}{ext}"
    part = folder / f"{name}{ext}.part"
    part.write_bytes(data)
    part.rename(out)
    return out


def download_pack(entry, log=print):
    """Download a catalog entry (resumable, SSRF-checked, size-capped) and install it."""
    import httpx
    from urllib.parse import urljoin
    where = entry.get("where", "")
    if where and where != "online" and Path(where).exists():
        return install_pack(where, entry.get("sha256"), log=log)
    if not entry.get("sha256"):
        raise ValueError("Remote voice packs must include a sha256 checksum to be installed safely.")
    raw_name = Path(entry.get("file") or f"{entry['id']}-v{entry.get('version', 1)}.narrator-pack").name
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", raw_name).strip(".-") or "pack.narrator-pack"
    dest = (PACKS / safe_name).resolve()
    if dest.parent != PACKS.resolve():
        raise ValueError("Unsafe pack filename.")
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    log(f"Downloading {entry['name']} ({entry.get('size', 0) / 1e9:.1f} GB)…")
    from . import progress
    url = _safe_remote_url(entry["url"])
    with httpx.Client(timeout=60, follow_redirects=False) as client:
        for _ in range(6):
            with client.stream("GET", url, headers=headers) as r:
                if r.is_redirect and r.headers.get("location"):
                    url = _safe_remote_url(urljoin(str(r.url), r.headers["location"]))
                    continue
                r.raise_for_status()
                resumed = bool(have and r.status_code == 206)
                size = int(r.headers.get("content-length") or 0) or None
                if size and (size + (have if resumed else 0)) > MAX_PACK_UNCOMPRESSED:
                    raise ValueError("Remote voice pack exceeds maximum download size.")
                written = have if resumed else 0
                with open(part, "ab" if resumed else "wb") as f, \
                        progress.downloading(PACKS, f"Downloading {entry['name']}", log, total=size):
                    for chunk in r.iter_bytes(1 << 20):
                        written += len(chunk)
                        if written > MAX_PACK_UNCOMPRESSED:
                            raise ValueError("Remote voice pack exceeded maximum download size.")
                        f.write(chunk)
                break
        else:
            raise ValueError("Too many redirects while downloading voice pack.")
    part.rename(dest)
    (PACKS / f"{dest.stem}.json").write_text(json.dumps({k: v for k, v in entry.items() if k != "where"}))
    return install_pack(dest, entry.get("sha256"), log=log)


# ---------- engines ----------

def _cached_repos():
    try:
        from huggingface_hub import scan_cache_dir
        return {r.repo_id: r.size_on_disk for r in scan_cache_dir().repos}
    except Exception:  # noqa: BLE001
        return {}


def ram_gb():
    from .tts.base import ram_gb as _ram
    return _ram()


def engines():
    cached = _cached_repos()
    rows = []
    for e in tts.ENGINES.values():
        ok, why = e.can_run()  # (on or off: the page shows the switch apart)
        have = [m for m in e.hf_models if m in cached]
        rows.append({"id": e.id, "name": e.name, "tier": e.tier, "license": e.license, "ok": ok, "why": why,
                     "off": tts.is_off(e.id),
                     "note": e.note() if ok else "", "locked": not ok and e.min_ram_gb > ram_gb(),
                     "models": f"{len(have)}/{len(e.hf_models)} downloaded",
                     "size_gb": round(sum(cached.get(m, 0) for m in e.hf_models) / 1e9, 1),
                     "missing": [m for m in e.hf_models if m not in cached], "pip": list(e.pip)})
    return rows


def set_engine_off(engine_id, off):
    """Turn a speech model off (its voices leave the lists, nothing is made with it) or on again."""
    e = tts.ENGINES[engine_id]
    now = [x for x in (settings.load().get("engines_off") or []) if x in tts.ENGINES and x != engine_id]
    if off:
        if not [x for x in tts.ENGINES.values() if x.id not in now + [engine_id] and x.can_run()[0]]:
            raise ValueError(f"{e.name} is the only speech model left on: turn another one on first.")
        now.append(engine_id)
        _release(engine_id)
    settings.save(engines_off=now)
    return not off


def _release(engine_id):
    """Out of memory, if it's the model loaded now."""
    from .runtime import slot
    if (slot.key or "").startswith(engine_id):
        slot.release()


def _busy():
    from . import tasks
    return [t for t in tasks.all_tasks() if t.state == "running" and t.kind not in ("model",)]


def delete_engine(engine_id):
    """Delete a speech model's downloaded files from this Mac (it downloads again when it's needed, or from
    the Models page). Files another model uses too are kept. Returns the GB freed."""
    from huggingface_hub import scan_cache_dir
    e = tts.ENGINES[engine_id]
    if _busy():
        raise ValueError("Something is being made right now. Delete it when that's done (or stop it first).")
    others = {m for x in tts.ENGINES.values() if x.id != engine_id for m in x.hf_models}
    mine = [m for m in e.hf_models if m not in others]
    _release(engine_id)
    freed, hashes = 0, []
    for repo in scan_cache_dir().repos:
        if repo.repo_id in mine:
            freed += repo.size_on_disk
            hashes += [r.commit_hash for r in repo.revisions]
    if hashes:
        scan_cache_dir().delete_revisions(*hashes).execute()
    for d in getattr(e, "local_dirs", ()):
        d = Path(d)
        if d.exists():
            freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            shutil.rmtree(d, ignore_errors=True)
    return round(freed / 1e9, 2)


def redownload_engine(engine_id, log=print):
    """Delete it, then download it again (for a download that's gone wrong)."""
    log("Deleting the old download…")
    delete_engine(engine_id)
    download_engine(engine_id, log=log)


def download_engine(engine_id, log=print):
    from huggingface_hub import snapshot_download
    e = tts.ENGINES[engine_id]
    from . import progress
    from .status import Status
    repos, setup = list(e.hf_models), hasattr(e, "install")  # (setup: e.g. VibeVoice's 4-bit copy)
    cache = Path(progress.hf_cache())
    todo = []  # bytes still to come, per model (None: can't tell)
    for repo in repos:
        need = progress.hf_size(repo)
        have = progress.size_of(cache / f"models--{repo.replace('/', '--')}")
        todo.append(need - have if need and need > have else (0 if need else None))
    share = 0.9 if setup else 1.0  # of the bar, for the downloads (by size when known)
    known = sum(t or 0 for t in todo)
    shares = [share * (t or 0) / known if known and None not in todo else share / max(len(repos), 1) for t in todo]
    steps = [(f"Downloading {repo.split('/')[-1]}", s) for repo, s in zip(repos, shares)]
    if setup:
        steps.append(("Setting it up for this Mac (one time)", 1 - share))
    stage = progress.stages(log, steps)
    for n, repo in enumerate(repos):
        say = stage(n)
        snapshot_download(repo, tqdm_class=progress.hf_bars(say, total=todo[n] or None))
        say(Status("done", 1.0))
    if setup:
        e.install(log=stage(len(repos)))
    log(f"{e.name} is ready.")


def install_extras(engine_id, log=print):
    import subprocess
    import sys
    e = tts.ENGINES[engine_id]
    if not e.pip:
        return
    uv = shutil.which("uv")
    cmd = ([uv, "pip", "install", "--python", sys.executable] if uv else [sys.executable, "-m", "pip", "install"]) + list(e.pip)
    log(" ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    log((r.stdout + r.stderr)[-1500:])
    if r.returncode:
        raise RuntimeError("Install failed; see above.")
    log("Installed. Restart the app to use it.")
