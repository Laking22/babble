"""Run: .venv/bin/python tests/pack_signing_check.py  (a few seconds, no models, no network, scratch data folder only)

Signed voice packs (narrator/signing.py), with GitHub faked:
1. Setting up: a GitHub username, a key made once (kept as a secret, never handed out), whether it's on GitHub.
2. Exporting signed: signature.json in the pack and its catalog entry; agents can't sign; signing needs setting up.
3. Installing: a confirmed signature shows the name; a pack changed after signing, or a broken signature, isn't
   installed; a key that isn't on that GitHub account (or no such account) installs but shows as not confirmed;
   GitHub unreachable shows as not checked yet; unsigned packs install as before.
4. publish_pack.py carries the signature into the online catalog, where the app checks it before download.
"""
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="packsign-"))
os.environ["NARRATOR_HOME"] = str(TMP / "home")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from narrator import api, guard, library, settings, signing  # noqa: E402

failed = []


def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        failed.append(what)


app = FastAPI()
api.install(app)
owner = TestClient(app, base_url="http://127.0.0.1", headers={"X-Babble-Token": guard.token()})
agent = TestClient(app, base_url="http://127.0.0.1", headers={"X-Babble-Token": guard.token(), "X-Babble-Client": "mcp"})

# a fake GitHub: {user: [public key lines]}; users not in it are 404
github = {"alice": []}
net = {"down": False, "catalog": b"[]"}


def fake_fetch(url, cap):
    if net["down"]:
        raise OSError("offline")
    if url.endswith("catalog.json"):
        return net["catalog"]
    for user, lines in github.items():
        if url == f"https://api.github.com/users/{user}/ssh_signing_keys":
            return json.dumps([{"id": i, "key": " ".join(k.split()[:2])} for i, k in enumerate(lines)]).encode()
        if url == f"https://github.com/{user}.keys":
            return b""
    req = httpx.Request("GET", url)
    raise httpx.HTTPStatusError("404", request=req, response=httpx.Response(404, request=req))


library._fetch = fake_fetch
settings.save(pack_catalog_url="https://raw.githubusercontent.com/someone/babble-voices/main/catalog.json")


def fresh():
    signing._keys_cache.clear()


def model(name, speakers=("mia",)):
    m = library.MODELS / name
    m.mkdir(parents=True)
    (m / "voice.json").write_text(json.dumps({"engine": "qwen", "voices": {s: {"name": s.title(), "gender": "female"} for s in speakers}}))
    (m / "model.safetensors").write_bytes(os.urandom(256))
    return m


def export(name, sign_as=None):
    m = model(f"{name}-src")
    out = library.export_pack(m, name, "General", "Test.", {"mia": {"source": "own recording", "license": "CC0"}},
                              log=lambda *a: None, sign_as=sign_as)
    shutil.rmtree(m)
    return out


def rezip(src, dst, change):
    """Copy a pack, changing entries: change(name, data) -> new data."""
    with zipfile.ZipFile(src) as a, zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as b:
        for info in a.infolist():
            b.writestr(info.filename, change(info.filename, a.read(info)))
    return dst


def installed(pid):
    return next((p for p in owner.get("/api/packs").json()["installed"] if p["id"] == pid), None)


# 1. setting up
s = owner.get("/api/signing").json()
check(s["github"] == "" and s["public_key"] is None and s["on_github"] is False, "nothing is set up at first")
check(owner.post("/api/signing", json={"github": "not a name!"}).status_code == 400, "a bad username is refused")
s = owner.post("/api/signing", json={"github": "@alice"}).json()
check(s["github"] == "alice" and (s["public_key"] or "").startswith("ssh-ed25519 ") and s["fingerprint"].startswith("SHA256:"),
      "setting a username makes a key (the @ is dropped)")
key1 = s["public_key"]
check(owner.post("/api/signing", json={"github": "alice"}).json()["public_key"] == key1, "the key is made only once")
check(signing.KEY_SETTING not in settings.public() and signing.KEY_SETTING not in owner.get("/api/settings").json(),
      "the private key is a secret: never handed out")
check(s["on_github"] is False, "the key isn't on GitHub yet")
github["alice"].append(key1)
check(owner.get("/api/signing").json()["on_github"] is True, "...and is found once it's added there")

# 2. exporting
signed_pack = export("Signed One", sign_as="alice")
with zipfile.ZipFile(signed_pack) as z:
    sig = signing.read(z)
check(sig and sig["github"] == "alice" and sig["key"] == key1, "a signed export carries signature.json")
row = next((p for p in owner.get("/api/packs").json()["catalog"] if p["id"] == "signed-one"), None)
check(row and row["signed"] and row["signed"]["verified"] is True and row["signed"]["github"] == "alice",
      "the pack shows as signed by alice before it's installed")
unsigned_pack = export("Plain One")
row = next((p for p in owner.get("/api/packs").json()["catalog"] if p["id"] == "plain-one"), None)
check(row and row["signed"] is None, "an unsigned pack shows no signature")
settings.save(agents_on=True)
m = model("agent-src")
body = {"folder": str(m), "name": "Agent", "rights": {"mia": {"source": "x", "license": "CC0"}}, "previews": False, "sign": True}
check(agent.post("/api/train/export", json=body).status_code == 403, "an AI agent can't sign a pack")
check(agent.post("/api/signing", json={"github": "mallory"}).status_code == 403, "...or change who packs are signed as")
r = agent.post("/api/settings", json={"pack_signer_github": "mallory"})
check(r.status_code == 403, "...even through settings")
settings.save(agents_on=False)
signing.set_github_user("")
check(owner.post("/api/train/export", json=body).status_code == 400, "signing needs a username set first")
signing.set_github_user("alice")
shutil.rmtree(m)

# 3. installing
fresh()
library.install_pack(signed_pack, log=lambda *a: None)
p = installed("signed-one")
check(p and p["signed"]["verified"] is True and p["signed"]["github"] == "alice", "an installed signed pack shows alice")

changed = rezip(signed_pack, TMP / "changed.narrator-pack",
                lambda n, d: (b"\x01" + d[1:]) if n.startswith("model/model.safetensors") else d)
try:
    library.install_pack(changed, log=lambda *a: None)
    check(False, "a pack changed after signing isn't installed")
except ValueError as e:
    check("changed after it was signed" in str(e), "a pack changed after signing isn't installed")


def bad_sig(n, d):
    if n != signing.FILE:
        return d
    j = json.loads(d)
    j["manifest_sha256"] = "0" * 64
    return json.dumps(j).encode()


try:
    library.install_pack(rezip(signed_pack, TMP / "badsig.narrator-pack", bad_sig), log=lambda *a: None)
    check(False, "a broken signature isn't installed")
except ValueError as e:
    check("doesn't match" in str(e), "a broken signature isn't installed")

# someone else's key, claiming to be alice
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402
from cryptography.hazmat.primitives import serialization  # noqa: E402
other = Ed25519PrivateKey.generate()
real = settings.load()[signing.KEY_SETTING]
settings.save(**{signing.KEY_SETTING: base64.b64encode(other.private_bytes(
    serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())).decode()})
forged = export("Forged One", sign_as="alice")
nobody = export("Nobody One", sign_as="bob")
settings.save(**{signing.KEY_SETTING: real})
fresh()
library.install_pack(forged, log=lambda *a: None)
p = installed("forged-one")
check(p and p["signed"]["verified"] is False and "isn't on github.com/alice" in p["signed"]["why"],
      "a key that isn't on that GitHub account installs, but shows as not confirmed")
library.install_pack(nobody, log=lambda *a: None)
p = installed("nobody-one")
check(p and p["signed"]["verified"] is False and "no GitHub account named bob" in p["signed"]["why"], "...so does an account that doesn't exist")

fresh()
net["down"] = True
offline_pack = export("Offline One", sign_as="alice")
library.install_pack(offline_pack, log=lambda *a: None)
p = installed("offline-one")
check(p and p["signed"]["verified"] is None, "GitHub unreachable: installed, shown as not checked yet")
net["down"] = False
fresh()
check_file = library.MODELS / "offline-one" / "signature-check.json"
c = json.loads(check_file.read_text())
c["checked"] = 0
check_file.write_text(json.dumps(c))
p = installed("offline-one")
check(p and p["signed"]["verified"] is True, "...and confirmed once GitHub is back")

library.install_pack(unsigned_pack, log=lambda *a: None)
p = installed("plain-one")
check(p and p["signed"] is None, "an unsigned pack installs as before")

# 4. the online catalog
voices = TMP / "babble-voices"
voices.mkdir()
out = subprocess.run([sys.executable, str(ROOT / "scripts" / "publish_pack.py"), str(signed_pack), str(voices), "--hf", "alice/babble-voices"],
                     capture_output=True, text=True)
pub = json.loads((voices / "catalog.json").read_text())
check(out.returncode == 0 and pub and pub[0].get("signature", {}).get("github") == "alice", "publish_pack.py carries the signature")
pub[0]["id"] = "online-one"
net["catalog"] = json.dumps(pub).encode()
library._online_cache["at"] = 0
fresh()
row = next((p for p in owner.get("/api/packs").json()["catalog"] if p["id"] == "online-one"), None)
check(row and row["where"] == "online" and row["signed"]["verified"] is True, "an online pack shows its signer before download")

print(f"\n{'all passed' if not failed else f'{len(failed)} failed'}")
sys.exit(1 if failed else 0)
