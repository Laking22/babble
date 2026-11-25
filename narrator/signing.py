"""Signed voice packs (optional). A signed pack shows its author's GitHub username wherever it's shared.

- Each author's Babble makes an Ed25519 key once (kept like any other key: the Keychain when the app runs). They add
  its public half to their GitHub account as a signing key (Settings → SSH and GPG keys → New SSH key).
- Signing a pack hashes every file in it (the manifest), signs that hash, and adds signature.json to the pack:
  {"format": 1, "github", "key": "ssh-ed25519 …", "manifest_sha256", "signature"}.
- Anyone's Babble checks it: the signature must match the key, the files must match the manifest (a pack changed
  after signing isn't installed), and the key must be one listed publicly on that GitHub account. GitHub can't be
  reached: the name shows as not checked yet (verified None).
"""
import base64
import hashlib
import json
import re
import time
import zipfile

from . import settings

FILE = "signature.json"
CONTEXT = b"babble-voice-pack-v1\n"
KEY_SETTING = "pack_signing_key"      # (a secret: the Keychain keeps it when the app runs)
USER_SETTING = "pack_signer_github"
USER = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}$")
KEYS_TTL = 3600
MAX_SIG = 16 << 10
_keys_cache = {}


class NotFound(Exception):
    pass


# ---------- this author's key ----------

def _private():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    raw = (settings.load().get(KEY_SETTING) or "").strip()
    return Ed25519PrivateKey.from_private_bytes(base64.b64decode(raw)) if raw else None


def make_key():
    """This Mac's signing key, made the first time."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    key = _private()
    if key is None:
        key = Ed25519PrivateKey.generate()
        raw = key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
        settings.save(**{KEY_SETTING: base64.b64encode(raw).decode()})
    return key


def public_line(key=None):
    """The public key as GitHub takes it ("ssh-ed25519 AAAA… babble"), or None without a key."""
    from cryptography.hazmat.primitives import serialization
    key = key or _private()
    if key is None:
        return None
    line = key.public_key().public_bytes(serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH).decode()
    return f"{line} babble"


def _blob(line):
    parts = (line or "").split()
    return parts[1] if len(parts) >= 2 and parts[0] == "ssh-ed25519" else None


def fingerprint(line):
    blob = _blob(line)
    if not blob:
        return ""
    return "SHA256:" + base64.b64encode(hashlib.sha256(base64.b64decode(blob)).digest()).decode().rstrip("=")


def github_user():
    return (settings.load().get(USER_SETTING) or "").strip()


def set_github_user(user):
    user = (user or "").strip().lstrip("@")
    if user and not USER.match(user):
        raise ValueError("That isn't a GitHub username.")
    settings.save(**{USER_SETTING: user})
    _keys_cache.pop(user.lower(), None)
    return user


# ---------- GitHub ----------

def github_keys(user, fresh=False):
    """The public SSH keys (signing and sign-in) on a GitHub account, as key blobs. NotFound: no such account."""
    import httpx
    from . import library
    name = user.lower()
    hit = _keys_cache.get(name)
    if hit and not fresh and time.time() - hit[0] < KEYS_TTL:
        return hit[1]
    blobs = set()
    try:
        rows = json.loads(library._fetch(f"https://api.github.com/users/{user}/ssh_signing_keys", 1 << 20))
        blobs |= {b for b in (_blob(r.get("key")) for r in rows if isinstance(r, dict)) if b}
        text = library._fetch(f"https://github.com/{user}.keys", 1 << 20).decode("utf-8", "replace")
        blobs |= {b for b in (_blob(x) for x in text.splitlines()) if b}
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise NotFound(user) from e
        raise
    _keys_cache[name] = (time.time(), blobs)
    return blobs


def on_github(user=None, line=None):
    """Whether this Mac's key is on the GitHub account (None: GitHub couldn't be reached)."""
    user, line = user or github_user(), line or public_line()
    if not user or not line:
        return False
    try:
        return _blob(line) in github_keys(user, fresh=True)
    except NotFound:
        return False
    except Exception:  # noqa: BLE001 - offline
        return None


# ---------- packs ----------

def manifest(z):
    """sha256 of every file in a pack (but the signature), as one hash."""
    files = {}
    for info in z.infolist():
        if info.is_dir() or info.filename == FILE:
            continue
        h = hashlib.sha256()
        with z.open(info) as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        files[info.filename] = h.hexdigest()
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sign_zip(path, user):
    """Add this Mac's signature to a pack file. Returns the signature."""
    key = _private()
    if key is None or not user:
        raise ValueError("Set up signing in Settings → General first.")
    with zipfile.ZipFile(path) as z:
        if FILE in z.namelist():
            raise ValueError("That pack is already signed.")
        m = manifest(z)
    sig = {"format": 1, "github": user, "key": public_line(key), "manifest_sha256": m,
           "signature": base64.b64encode(key.sign(CONTEXT + m.encode())).decode()}
    with zipfile.ZipFile(path, "a", zipfile.ZIP_STORED) as z:
        z.writestr(FILE, json.dumps(sig, indent=1))
    return sig


def read(z):
    """A pack's signature.json, or None for an unsigned pack."""
    if FILE not in z.namelist():
        return None
    info = z.getinfo(FILE)
    if info.file_size > MAX_SIG:
        raise ValueError("The pack's signature is too large.")
    return json.loads(z.read(info))


def check(sig, manifest_sha=None):
    """{"github", "fingerprint", "verified": True | False | None, "why"} for a signature (None: unsigned)."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.serialization import load_ssh_public_key
    if not sig:
        return None
    user = str(sig.get("github") or "") if isinstance(sig, dict) else ""
    out = {"github": user, "fingerprint": fingerprint(sig.get("key") if isinstance(sig, dict) else ""), "verified": False}
    try:
        if not USER.match(user) or not _blob(sig.get("key")):
            raise ValueError
        pub = load_ssh_public_key(" ".join(sig["key"].split()[:2]).encode())
        pub.verify(base64.b64decode(sig["signature"]), CONTEXT + str(sig["manifest_sha256"]).encode())
    except (InvalidSignature, ValueError, KeyError, TypeError):
        return dict(out, why="The signature doesn't match.")
    if manifest_sha is not None and manifest_sha != sig["manifest_sha256"]:
        return dict(out, why="The pack was changed after it was signed.")
    try:
        found = _blob(sig["key"]) in github_keys(user)
    except NotFound:
        return dict(out, why=f"There's no GitHub account named {user}.")
    except Exception:  # noqa: BLE001 - offline: the name can't be confirmed yet
        return dict(out, verified=None, why="Couldn't reach GitHub to check.")
    if not found:
        return dict(out, why=f"That key isn't on github.com/{user}.")
    return dict(out, verified=True, why="")
