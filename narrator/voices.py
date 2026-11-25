"""Voice lookups for the UI across every engine: dropdown choices, names, genders, typed-name matching.

Saved voices themselves live in voices_lib; each engine lists its own voices.
"""
import re

from . import tts
from .voices_lib import GENDERS, clean_name, delete, info, library, save, wav_path, write_meta  # noqa: F401
from .paths import VOICES  # noqa: F401
from .tts.qwen import BUILTINS  # noqa: F401


def _all():
    """Voices this Mac can make, plus ones only Create in the cloud can (Higgs on an 8 GB Mac: Engine.cloud), so
    their names in a book's cast keep pointing at them."""
    return {v.value: v for e in tts.ENGINES.values() if e.available()[0] or e.cloud or tts.is_off(e.id)
            for v in e.voices()}


def _label(v):
    eng = tts.ENGINES[v.engine]
    kind = "saved" if "saved" in v.tags and v.engine == "qwen" else ("built-in" if v.engine == "qwen" else eng.short)
    extra = v.description if "saved" not in v.tags else v.description or "saved"
    return f"{v.name}  ·  {v.gender}, {extra} ({kind})"


def choices():
    """Dropdown entries (label, value): your voices first, then each available engine's voices."""
    return [(_label(v), v.value) for v in tts.all_voices()]


def by_engine(engine_id, gender=None):
    e = tts.ENGINES[engine_id]
    return [v.value for v in e.voices() if gender is None or v.gender == gender]


def gender_of(value):
    v = _all().get(value)
    return v.gender if v else "unspecified"


def short_name(value):
    kind, _, key = (value or "").partition(":")
    v = _all().get(value)
    if not v:
        return key or value
    if kind in ("my",):
        return v.name
    if kind == "builtin":
        return v.name
    return f"{v.name} ({tts.ENGINES[v.engine].short})"


def from_name(text):
    """Match a name typed in the cast table ('Nova', 'ryan', 'Tara (Orpheus)', 'George (Kokoro)') to a voice value."""
    raw = (text or "").strip()
    if not raw:
        return None
    if re.match(r"^[a-z]+:", raw) and any(e.owns(raw) for e in tts.ENGINES.values()):
        return raw
    voices = _all()
    low = raw.lower()
    for value in voices:  # exact short name match, e.g. "Tara (Orpheus)"
        if short_name(value).lower() == low:
            return value
    tagged = re.fullmatch(r"(.+?)\s*\((\w+)\)", low)
    if tagged:
        name, eng = tagged.group(1).replace(" ", "_"), tagged.group(2)
        for value, v in voices.items():
            if tts.ENGINES[v.engine].name.lower().startswith(eng) and \
                    name in (value.partition(":")[2].lower(), v.name.lower().replace(" ", "_")):
                return value
    norm = low.replace(" (saved)", "").replace(" ", "_")
    for v in library():
        if v["name"].lower() == norm:
            return f"my:{v['name']}"
    for key, (display, _, _) in BUILTINS.items():
        if norm in (key, display.lower().replace(" ", "_")):
            return f"builtin:{key}"
    prefix = [v["name"] for v in library() if v["name"].lower().startswith(norm)]
    return f"my:{prefix[0]}" if len(prefix) == 1 else None


def names_help():
    lines = [f"**Your voices:** {', '.join(v['name'] for v in library()) or 'none yet'}"]
    for e in tts.available():
        vs = [v for v in e.voices() if "saved" not in v.tags]
        if vs:
            lines.append(f"**{e.name}:** " + ", ".join(short_name(v.value) for v in vs))
    return "  \n".join(lines)
