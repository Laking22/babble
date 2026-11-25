"""Updates: newer versions of Babble from its GitHub releases (Settings → General → Updates).

The engine asks GitHub for the newest release (when asked, and every 6 hours while "Update automatically" is on),
downloads the installer for this computer from it, checks it against the checksum GitHub lists for that file, and
keeps it in <data>/updates with ready.json next to it. The app installs it (desktop/src-tauri/src/lib.rs,
apply_update): it quits, puts the new version where it is, and opens again. With "Update automatically" on, a
downloaded update is installed when Babble quits (ready.json "on_quit"), so work is never interrupted.

The app tells the engine its version and how it was installed (BABBLE_APP_VERSION; APPIMAGE on Linux). Run from a
checkout without the app, there's no version and nothing is offered.
"""
import hashlib
import json
import os
import platform
import re
import sys
import threading
import time
import urllib.request
from urllib.parse import urlparse

from . import settings, tasks
from .paths import HOME

REPO = "Laking22/babble"
API = f"https://api.github.com/repos/{REPO}/releases?per_page=20"
UPDATES = HOME / "updates"
READY = UPDATES / "ready.json"
EVERY = 6 * 3600
HOSTS = ("github.com", "api.github.com", "objects.githubusercontent.com", "release-assets.githubusercontent.com")
_cache = {"at": 0.0, "release": None, "error": ""}
_lock = threading.Lock()
_watching = False


def current():
    return os.environ.get("BABBLE_APP_VERSION", "").strip()


def version_key(v):
    """'v0.2.0' > '0.2.0-beta.2' > '0.2.0-beta.1' > '0.1.9'."""
    v = str(v or "").strip().lstrip("vV")
    main, _, pre = v.partition("-")
    nums = tuple(int(x) for x in re.findall(r"\d+", main)[:3])
    nums += (0,) * (3 - len(nums))
    pre_nums = tuple(int(x) for x in re.findall(r"\d+", pre))
    return nums + ((1,) if not pre else (0,) + pre_nums)


def system():
    """'mac', 'windows', 'appimage' or 'deb'."""
    if sys.platform == "darwin":
        return "mac"
    if sys.platform == "win32":
        return "windows"
    return "appimage" if os.environ.get("APPIMAGE") else "deb"


def asset_for(assets, where=None):
    """The release file this computer installs."""
    where = where or system()
    arm = platform.machine().lower() in ("arm64", "aarch64")
    for a in assets:
        n = a.get("name", "").lower()
        if where == "mac" and n.endswith(".zip") and ("mac" in n or "darwin" in n) and (("x86_64" in n or "x64" in n) != arm):
            return a
        if where == "windows" and n.endswith("-setup.exe"):
            return a
        if where == "appimage" and n.endswith(".appimage"):
            return a
        if where == "deb" and n.endswith(".deb"):
            return a
    return None


def _fetch_releases():
    req = urllib.request.Request(API, headers={"Accept": "application/vnd.github+json", "User-Agent": "Babble-updater"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read(4 << 20))


def latest(force=False):
    """The newest published release (pre-releases too: the betas are), from GitHub at most every 10 minutes."""
    with _lock:
        if not force and time.time() - _cache["at"] < 600 and (_cache["release"] or _cache["error"]):
            return _cache["release"]
    try:
        rels = [r for r in _fetch_releases() if not r.get("draft")]
        best = max(rels, key=lambda r: version_key(r.get("tag_name"))) if rels else None
        with _lock:
            _cache.update(at=time.time(), release=best, error="")
        return best
    except Exception as e:  # noqa: BLE001 - offline, GitHub down, rate-limited: say so, keep the last answer
        with _lock:
            _cache.update(at=time.time(), error=f"Couldn't reach GitHub ({e.__class__.__name__}).")
            return _cache["release"]


def ready():
    try:
        r = json.loads(READY.read_text())
        return r if (UPDATES / r["file"]).is_file() and version_key(r["version"]) > version_key(current()) else None
    except Exception:  # noqa: BLE001
        return None


def busy():
    """What's running that an update shouldn't cut off (for the Restart button and installing at quit)."""
    out = [t["title"] for t in (x.info() for x in tasks.all_tasks()) if t.get("state") == "running"]
    try:
        from .dataset import pipeline
        out += [f"Preparing {v}" for v, j in list(pipeline._jobs.items()) if j.get("thread") and j["thread"].is_alive()]
    except Exception:  # noqa: BLE001
        pass
    return out


def status(force=False):
    cur = current()
    rel = latest(force) if cur else None
    asset = asset_for(rel.get("assets", [])) if rel else None
    newer = bool(rel and asset and version_key(rel["tag_name"]) > version_key(cur))
    r = ready()
    return {
        "current": cur, "auto": bool(settings.load().get("auto_update", False)),
        "available": newer, "ready": bool(r), "system": system(),
        "latest": {"version": rel["tag_name"].lstrip("vV"), "name": rel.get("name") or rel["tag_name"],
                   "notes": (rel.get("body") or "")[:4000], "page": rel.get("html_url"), "published": rel.get("published_at")}
        if rel else None,
        "size": asset.get("size") if asset else None,
        "error": _cache["error"] if cur else "This copy of Babble wasn't installed from a release, so it isn't updated.",
        "busy": busy(),
    }


def _digest(asset):
    d = str(asset.get("digest") or "")
    return d.split(":", 1)[1].lower() if d.startswith("sha256:") else ""


def download(log=print):
    """Task: the newest release's installer for this computer into <data>/updates, checked; ready.json says it's there."""
    rel = latest(force=True)
    if not rel or not current():
        raise ValueError(_cache["error"] or "There's no update for this copy of Babble.")
    if version_key(rel["tag_name"]) <= version_key(current()):
        raise ValueError("Babble is up to date.")
    asset = asset_for(rel.get("assets", []))
    if not asset:
        raise ValueError(f"{rel.get('name') or rel['tag_name']} has no installer for this computer yet.")
    url = asset["browser_download_url"]
    if urlparse(url).scheme != "https" or urlparse(url).hostname not in HOSTS:
        raise ValueError("The update isn't on GitHub, so it wasn't downloaded.")
    UPDATES.mkdir(parents=True, exist_ok=True)
    for old in UPDATES.iterdir():  # (an earlier, unfinished or older update)
        if old.is_file():
            old.unlink(missing_ok=True)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", asset["name"])
    dest, part = UPDATES / name, UPDATES / (name + ".part")
    total, done, h = int(asset.get("size") or 0), 0, hashlib.sha256()
    version = rel["tag_name"].lstrip("vV")
    log(f"Downloading Babble {version} ({total / 1e6:.0f} MB)…")
    req = urllib.request.Request(url, headers={"User-Agent": "Babble-updater", "Accept": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
        if urlparse(r.geturl()).hostname not in HOSTS:
            raise ValueError("The download was sent somewhere other than GitHub, so it was stopped.")
        last = 0.0
        while True:
            if tasks.stop_wanted():
                part.unlink(missing_ok=True)
                raise tasks.Stopped()
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            h.update(chunk)
            done += len(chunk)
            if total and time.time() - last > 0.5:
                last = time.time()
                yield {"status": f"Downloading Babble {version}… {done / 1e6:.0f} of {total / 1e6:.0f} MB", "progress": done / total}
    if total and done != total:
        part.unlink(missing_ok=True)
        raise ValueError("The download was cut off. Try again.")
    want = _digest(asset)
    if want and h.hexdigest() != want:
        part.unlink(missing_ok=True)
        raise ValueError("The download doesn't match its checksum on GitHub, so it was thrown away.")
    part.replace(dest)
    READY.write_text(json.dumps({"version": version, "file": name, "sha256": h.hexdigest(), "system": system(),
                                 "on_quit": bool(settings.load().get("auto_update", False))}))
    yield {"status": f"Babble {version} is ready to install.", "progress": 1.0, "ready": True}


def set_auto(on):
    settings.save(auto_update=bool(on))
    r = ready()
    if r:  # the downloaded update follows the switch: installed at quit only while it's on
        READY.write_text(json.dumps(dict(r, on_quit=bool(on))))
    if on:
        watch()


def watch():
    """While "Update automatically" is on: look every 6 hours, and download a newer version when there is one."""
    global _watching
    if _watching or not current():
        return
    _watching = True

    def loop():
        global _watching
        time.sleep(60)  # (Babble opens first)
        while settings.load().get("auto_update", False):
            try:
                s = status(force=True)
                if s["available"] and not s["ready"] and not any(t.info().get("kind") == "update" and t.info().get("state") == "running"
                                                                for t in tasks.all_tasks()):
                    tasks.start("update", "Downloading an update", lambda: download())
            except Exception as e:  # noqa: BLE001
                print(f"[update] {e}", flush=True)
            time.sleep(EVERY)
        _watching = False

    threading.Thread(target=loop, daemon=True, name="updates").start()
