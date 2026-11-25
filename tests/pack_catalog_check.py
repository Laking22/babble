"""Run: .venv/bin/python tests/pack_catalog_check.py  (a few seconds, no models, no network, scratch data folder only)

Voice packs you can hear before downloading (narrator/library.py catalog/preview_file, /api/packs/preview):
1. The online catalog (fetched, made absolute, kept for a while, last good list kept when offline, "" turns it off).
2. /api/packs lists its voices with has_preview; /api/packs/preview fetches a preview once, then serves it from the
   cache; unknown voices are 404, network trouble and non-audio links are 502.
3. A pack exported on this Mac: its previews play from the pack file before it's installed.
4. scripts/publish_pack.py turns that pack into a catalog entry the app can read back.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="packcat-"))
os.environ["NARRATOR_HOME"] = str(TMP / "home")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from narrator import api, guard, library, settings  # noqa: E402

failed = []


def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        failed.append(what)


app = FastAPI()
api.install(app)
owner = TestClient(app, base_url="http://127.0.0.1", headers={"X-Babble-Token": guard.token()})

MP3 = b"ID3" + b"\x00" * 200
CAT = [{"id": "starter", "name": "Starter", "genre": "General", "description": "Two voices.", "version": 2,
        "engine": "qwen", "author": "Babble", "size": 2_500_000_000, "sha256": "ab" * 32, "file": "starter-v2.narrator-pack",
        "url": "https://huggingface.co/someone/babble-voices/resolve/main/starter-v2.narrator-pack",
        "voices": [{"speaker": "ann", "name": "Ann", "gender": "female", "preview": "previews/starter-v2/ann.mp3"},
                   {"speaker": "bob", "name": "Bob", "gender": "male"},
                   {"speaker": "eve", "name": "Eve", "gender": "female", "preview": "previews/starter-v2/eve.txt"}]},
       {"name": "no id, skipped"}]
calls = []
serve = {"catalog": json.dumps(CAT).encode(), "down": False}


def fake_fetch(url, cap):
    calls.append(url)
    if serve["down"]:
        raise OSError("offline")
    if url.endswith("catalog.json"):
        return serve["catalog"]
    if url.endswith("ann.mp3"):
        return MP3
    raise OSError("404")


library._fetch = fake_fetch

# 1. the online catalog
check(library.catalog_url() == library.CATALOG_URL, "the built-in catalog is used by default")
cat = library.catalog()
st = next((e for e in cat if e["id"] == "starter"), None)
check(st is not None and len(cat) == 1, "the catalog's packs are listed (entries without an id are skipped)")
check(st and st["where"] == "online" and st["url"].startswith("https://huggingface.co/"), "pack links are kept as given")
check(st and st["voices"][0]["preview"] == "https://raw.githubusercontent.com/Laking22/babble-voices/main/previews/starter-v2/ann.mp3",
      "preview links are made absolute against the catalog")
n = len(calls)
library.catalog()
check(len(calls) == n, "the catalog isn't fetched again within a few minutes")
library._online_cache["at"] = 0
serve["down"] = True
cat = library.catalog()
check(any(e["id"] == "starter" for e in cat), "offline: the last good list is kept")
serve["down"] = False
settings.save(pack_catalog_url="")
check(not [e for e in library.catalog() if e.get("where") == "online"], "an empty catalog setting turns it off")
settings.save(pack_catalog_url=library.CATALOG_URL)
library._online_cache["at"] = 0

# 2. the API
r = owner.get("/api/packs").json()
row = next((p for p in r["catalog"] if p["id"] == "starter"), None)
check(row and [v["has_preview"] for v in row["voices"]] == [True, False, True] and row["voices"][0]["speaker"] == "ann",
      "/api/packs shows which voices can be heard")
check(row and row["where"] == "online" and "url" not in row and "sha256" not in row, "the list doesn't hand out links")
n = len(calls)
r = owner.get("/api/packs/preview", params={"id": "starter", "version": 2, "speaker": "ann"})
check(r.status_code == 200 and r.content == MP3 and r.headers["content-type"].startswith("audio/"), "a preview plays")
check(len(calls) == n + 1, "...fetched once")
r = owner.get("/api/packs/preview", params={"id": "starter", "version": 2, "speaker": "ann"})
check(r.status_code == 200 and len(calls) == n + 1, "...then served from the cache")
check(owner.get("/api/packs/preview", params={"id": "starter", "version": 2, "speaker": "bob"}).status_code == 404,
      "a voice without a preview is 404")
check(owner.get("/api/packs/preview", params={"id": "starter", "version": 9, "speaker": "ann"}).status_code == 404,
      "a pack that isn't listed is 404")
check(owner.get("/api/packs/preview", params={"id": "starter", "version": 2, "speaker": "eve"}).status_code == 502,
      "a link that isn't audio is refused")
check(TestClient(app, base_url="http://127.0.0.1").get("/api/packs/preview", params={"id": "starter", "version": 2, "speaker": "ann"}).status_code == 401,
      "the preview needs the engine key")
r = TestClient(app, base_url="http://127.0.0.1").get("/api/packs/preview", params={"id": "starter", "version": 2, "speaker": "ann", "auth": guard.token()})
check(r.status_code == 200 and r.content == MP3, "...which an audio player can pass in the link")
try:
    library._safe_remote_url("http://example.com/x.mp3")
    check(False, "plain http links are refused")
except ValueError:
    check(True, "plain http links are refused")

# 3. a pack exported on this Mac
model = library.MODELS / "tiny-qwen"
model.mkdir(parents=True)
(model / "voice.json").write_text(json.dumps({"engine": "qwen", "voices": {"mia": {"name": "Mia", "gender": "female"}}}))
(model / "model.safetensors").write_bytes(b"\x00" * 64)
wav = TMP / "mia.wav"
subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=220:d=1", "-ar", "24000", str(wav)], check=True)
pack = library.export_pack(model, "Tiny", "General", "One voice.", {"mia": {"source": "own recording", "license": "CC0"}},
                           previews={"mia": str(wav)}, log=lambda *a: None)
import shutil  # noqa: E402
shutil.rmtree(model)
row = next((p for p in owner.get("/api/packs").json()["catalog"] if p["id"] == "tiny"), None)
check(row and row["where"] == "this Mac" and row["voices"][0]["has_preview"], "an exported pack is listed with its preview")
r = owner.get("/api/packs/preview", params={"id": "tiny", "version": 1, "speaker": "mia"})
check(r.status_code == 200 and r.content == wav.read_bytes(), "its preview plays from the pack file")

# 4. publish_pack.py
voices = TMP / "babble-voices"
voices.mkdir()
(voices / "catalog.json").write_text("[]")
out = subprocess.run([sys.executable, str(ROOT / "scripts" / "publish_pack.py"), str(pack), str(voices), "--hf", "someone/babble-voices"],
                     capture_output=True, text=True)
check(out.returncode == 0 and "hf upload someone/babble-voices" in out.stdout, "publish_pack.py runs and prints the upload step")
pub = json.loads((voices / "catalog.json").read_text())
e = pub[0] if pub else {}
check(e.get("url") == f"https://huggingface.co/someone/babble-voices/resolve/main/{pack.name}" and len(e.get("sha256", "")) == 64,
      "...the entry links the pack on Hugging Face, with its checksum")
mp3 = voices / (e.get("voices") or [{}])[0].get("preview", "missing")
check(mp3.exists() and mp3.suffix == ".mp3" and e["voices"][0]["license"] == "CC0", "...with an MP3 preview and each voice's license")
serve["catalog"] = json.dumps(pub).encode()
library._online_cache["at"] = 0
got = next((x for x in library.catalog() if x["id"] == "tiny"), None)
check(got and got["where"] == "online" and got["voices"][0]["preview"].endswith(f"previews/tiny-v1/mia.mp3"),
      "...and the app reads it back from the online catalog")

print(f"\n{'all passed' if not failed else f'{len(failed)} failed'}")
sys.exit(1 if failed else 0)
