"""Small settings file (readable only by you). API keys and tokens aren't kept in it when the Babble app started the
engine: the app keeps them in the macOS Keychain and hands them over at start (vault.py). load() and save() are the
same either way; only where a secret goes differs."""
import json
import os
import re
import threading

from .paths import SETTINGS as PATH

# settings that hold a secret (API keys, tokens): never handed out by the API, whatever the service calls them
SECRET = re.compile(r"key|token|secret|password", re.I)
NOT_SECRET = {"ssh_key"}  # (a path to your SSH key file, not a key)
_lock = threading.Lock()

DEFAULTS = {"engines_off": [],  # speech models turned off on the Models page (their voices are hidden)
            "orpheus_where": "mac", "orpheus_url": "", "orpheus_key": "", "orpheus_model": "orpheus",
            "orpheus_batch": False,  # make several Orpheus lines at once on this Mac (off unless turned on)
            # Settings → General: the longest line (characters) a trained voice is given; longer sentences are cut at
            # their best break (engine.chunk_limit). Lower: fewer skipped words; higher: fewer cuts, more flow. 0: never cut
            "longest_line": 240,
            "sfx_smart": True,  # match sound tags by meaning (CLAP) once its model is downloaded
            "music_quality": "auto",
            "auto_update": False,  # Settings → General → Updates: download new versions, install them at quit (updater.py)
            "file_quality": "standard",  # the finished audiobook file: small | standard | high | highest (export.QUALITY)
            # what a book starts with until it sets its own (each audiobook keeps these in its project): make tagged
            # sounds the library doesn't have before creating; the sound director's place, model per service and
            # checks (1 loose … 5 strict)
            "sfx_auto_make": True,
            "director_where": "mac", "director_models": {}, "director_checks": 3, "director_scenes": "small", "sound_libraries": True,
            "director_intros": False,  # the director makes every chapter's intro and outro (replacing the Intros tab's)
            # Settings → Spending (spend.py): the most paid services may cost in a day, in dollars (0: no limit)
            "spend_day_limit": 10.0,
            # Settings → General: AI agents may use Babble through MCP (agents.py); Settings → Spending: the most
            # their paid work may cost in a day (0: they can't start any)
            "agents_on": False, "agent_spend_day_limit": 0.0,
            # Settings → General: the GitHub account packs are signed as (signing.py; the key itself is a secret)
            "pack_signer_github": "",
            # Connections → My GPU machine: where SSH training runs (training/backends/ssh.py reads these)
            "ssh_host": "", "ssh_port": "22", "ssh_key": "", "ssh_dir": "~/narrator-train", "ssh_python": "python3"}  # music_quality: Stable Audio 3 Medium on 16 GB Macs and up ("auto"), or "small"


def is_secret(name):
    return bool(SECRET.search(name)) and name not in NOT_SECRET


def _file():
    data = dict(DEFAULTS)
    if PATH.exists():
        try:
            saved = json.loads(PATH.read_text(encoding="utf-8"))
            if saved.get("split_long_lines") is False and "longest_line" not in saved:
                saved["longest_line"] = 0  # (the switch this slider replaced was turned off: sentences stay whole)
            data.update(saved)
        except ValueError:
            pass
    return data


def load():
    data = _file()
    from . import vault
    if vault.active():
        data.update(vault.keys())  # (the Keychain's copy wins)
    return data


def public():
    """The settings without any secret (what the app may be shown)."""
    return {k: v for k, v in load().items() if not SECRET.search(k)}


def save(**fields):
    """Secrets go to the Keychain when the app keeps one (vault.py), the rest to the file. The file is written whole
    to a new file (only you can read it from the start) that then replaces the old one, so a crash or the power going
    off mid-save never leaves a half-written file."""
    from . import vault
    fields = {k: (v or "").strip() if isinstance(v, str) or v is None else v for k, v in fields.items()}
    with _lock:
        secret = {k: v for k, v in fields.items() if is_secret(k)}
        in_keychain = bool(secret) and vault.put(secret)  # (False without the app's vault: they stay in the file)
        data = _file()
        data.update(fields)
        if in_keychain:
            for k in secret:  # in the Keychain now: the file keeps at most an empty field
                data[k] = "" if k in DEFAULTS else None
        _write({k: v for k, v in data.items() if v is not None})
        return load()


def migrate():
    """Keys still in the file (from before, or saved while the Keychain couldn't be reached) move to the Keychain,
    and leave the file only once the app says they're in. Returns how many moved."""
    from . import vault
    if not vault.active():
        return 0
    with _lock:
        data = _file()
        found = {k: v for k, v in data.items() if is_secret(k) and isinstance(v, str) and v.strip()}
        if not found or not vault.put(found):
            return 0
        for k in found:
            if k in DEFAULTS:
                data[k] = DEFAULTS[k]
            else:
                data.pop(k)
        _write(data)
    print(f"[settings] moved {len(found)} key(s) to the Keychain", flush=True)
    return len(found)


def _write(data):
    tmp = PATH.with_name(PATH.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, indent=2))
        f.flush()
        os.fsync(f.fileno())
    os.chmod(tmp, 0o600)
    os.replace(tmp, PATH)
