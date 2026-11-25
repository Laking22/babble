"""Saved voices (designed or cloned): a wav, its transcript and a little metadata, in paths.VOICES."""
import json
import re
import shutil
from datetime import datetime

from .paths import VOICES

GENDERS = ["female", "male", "unspecified"]


def clean_name(name):
    return re.sub(r"[^A-Za-z0-9_-]", "", (name or "").replace(" ", "_"))


def wav_path(name):
    return VOICES / f"{name}.wav"


def fingerprint_path(name):
    return VOICES / f"{name}.npy"


def info(name):
    meta = {"name": name, "gender": "unspecified", "source": "saved", "description": "", "created": ""}
    path = VOICES / f"{name}.json"
    if path.exists():
        meta.update(json.loads(path.read_text(encoding="utf-8")))
    txt = VOICES / f"{name}.txt"
    meta["transcript"] = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
    return meta


def library():
    return [info(p.stem) for p in sorted(VOICES.glob("*.wav"), key=lambda p: p.stem.lower())]


def save(name, source_wav, transcript, source, description="", gender="unspecified"):
    name = clean_name(name)
    if not name:
        raise ValueError("Give the voice a name using letters, numbers, - or _.")
    shutil.copy(source_wav, wav_path(name))
    (VOICES / f"{name}.txt").write_text((transcript or "").strip(), encoding="utf-8")
    fingerprint_path(name).unlink(missing_ok=True)  # rebuilt from the new audio
    write_meta(name, gender=gender, source=source, description=description,
               created=datetime.now().strftime("%Y-%m-%d %H:%M"))
    return name


def write_meta(name, **fields):
    meta = {k: v for k, v in info(name).items() if k not in ("transcript", "name")}
    meta.update({k: v for k, v in fields.items() if k != "transcript"})
    (VOICES / f"{name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if "transcript" in fields:
        (VOICES / f"{name}.txt").write_text((fields["transcript"] or "").strip(), encoding="utf-8")


def delete(name):
    for ext in ("wav", "txt", "json", "npy"):
        (VOICES / f"{name}.{ext}").unlink(missing_ok=True)
