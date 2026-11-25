"""Sound effects and ambience: a local library, tags in the script, and the mixer.

Tags (curly braces, so they never clash with cast tags like [Mara: whispering]):
  {sfx: door creak}          a one-off sound, placed where the tag is
  {sfx: thunder, loud}       volume: quiet / normal / loud
  {amb: rain}                background sound from here on, looped, quieter while anyone speaks
  {amb: tavern, quiet}       volume works here too
  {amb: fire, add}           layer another background on top; {amb: fire, stop} fades just that one out
  {amb: off}                 stop all background sounds (they also fade out at the end of a chapter)
  A new {amb: …} crossfades from the old background; every background is looped without a seam.
  {music: quiet waltz at a ball}   music from here on, a layer of its own next to the backgrounds: looped,
                             and further under the voices than backgrounds; same add / stop / off as {amb: …}
  {room: cave}              voices and effects echo like that space (room, bathroom, hall, cave, cathedral); {room: off}
  {pause: 1.5}               silence for that many seconds (0.1 to 10; {pause} alone is 1 s), sound on or off
  {sfx: gunshot, far}        far away: quieter, duller, echoey; {sfx: scream, muffled}: through a wall
  {sfx: door knock, left}    place it left or right (also "hard left" / "hard right"); mixes with sound are stereo

Library: paths.HOME/sfx/<id>.wav (24 kHz mono) with index.json entries
  {"id", "name", "kind": "effect" | "ambience" | "music", "tags": [...], "file", "duration", "license", "source", "credit"}
Chapter intros: the audiobook puts an "intro" cue at the start of each chapter (a library sound by id). A long theme
is cut into a different stretch per chapter (plan_intros): the first words start just after the stretch's loudest
moment, and it fades out under them. An "outro" cue after a chapter's last words plays the theme's ending. Neither
follows the Effects/Music/Background sliders: they have the book's own intro volume.
Only sounds that may be redistributed (CC0 or similar) belong in a pack that ships; imported sounds keep the
license the user gives them. The starter sounds are generated here, so they are ours (CC0).
"""
import contextvars
import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

_lock = threading.RLock()


def _ffmpeg():
    for c in (shutil.which("ffmpeg"), "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if c and Path(c).exists():
            return c
    return "ffmpeg"

import numpy as np
import soundfile as sf

from .paths import HOME
from .runtime import SR

LIB = HOME / "sfx"
LIB.mkdir(parents=True, exist_ok=True)
INDEX = LIB / "index.json"
CUE = re.compile(r"\{\s*(sfx|amb|room|music|pause|pace)\s*(?::\s*([^{}]*?))?\s*\}", re.I)
PACE_TAG = re.compile(r"\{\s*pace\s*(?::\s*([^{}]*?))?\s*\}", re.I)
PAUSE_DEFAULT, PAUSE_MAX = 1.0, 10.0
PACE_PRESETS = {"slower": 0.80, "slow": 0.88, "normal": 1.0, "fast": 1.12, "faster": 1.20}


def pause_seconds(text):
    """"1.5", "1.5s", "2 seconds", "" -> seconds of silence for a {pause: …} tag."""
    m = re.search(r"\d+(?:\.\d+)?", text or "")
    return round(min(PAUSE_MAX, max(0.1, float(m.group()))) if m else PAUSE_DEFAULT, 2)


def pace_factor(text):
    """"slow", "fast", "0.9" -> speed multiplier (0.70–1.35) for a {pace: …} tag."""
    raw = (text or "").strip().lower()
    if raw in PACE_PRESETS:
        return PACE_PRESETS[raw]
    m = re.search(r"\d+(?:\.\d+)?", raw)
    return round(min(1.35, max(0.70, float(m.group()))), 2) if m else 1.0
AUDIO_EXT = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aif", ".aiff", ".opus"}
VOLUME = {"quiet": 0.5, "normal": 1.0, "loud": 1.6}
EFFECT_RMS_DB, AMB_RMS_DB, DUCK_DB = -24.0, -33.0, -10.0
# music sits a little above a background when nobody speaks, and further under the voices when they do
MUSIC_RMS_DB, MUSIC_DUCK_DB = -30.0, -13.0
# Volumes, as dB under the narrator while they speak (every line is leveled to SPEECH_DB, audio.level): the book's
# Effects, Music and Background sliders. The defaults are the mix as it always was; 0 is the narrator's own level
# (effects go a little past it); OFF_DB and under is silent.
SPEECH_DB = -20.0
DEFAULT_LEVELS = {"sfx": EFFECT_RMS_DB - SPEECH_DB, "amb": AMB_RMS_DB + DUCK_DB - SPEECH_DB,
                  "music": MUSIC_RMS_DB + MUSIC_DUCK_DB - SPEECH_DB}  # -4, -23, -23
MAX_LEVELS = {"sfx": 0.0, "amb": 0.0, "music": 0.0}
# Effects whose point is being loud (an explosion, a gunshot, thunder) play LOUD_OVER dB past the Effects volume, and
# may go that far past the narrator; every other effect stays at or under the narrator, even tagged loud
LOUD_OVER = 3.0
LOUD_CORE = re.compile(r"\b(explo\w*|detonat\w*|blast\w*|boom\w*|bang\w*|gunshot\w*|gunfire|gun ?fire|shots?|"
                       r"cannon\w*|artillery|grenade\w*|bomb\w*|thunder\w*|lightning|crash\w*|slam\w*|shatter\w*|"
                       r"smash\w*|collaps\w*|roar\w*|siren\w*|scream\w*|shriek\w*|impact\w*|kaboom)\b", re.I)


def loud_core(cue, e=None):
    """Whether an effect's point is being loud: its tag or its library sound says so."""
    words = " ".join([cue.get("name") or ""] + ([e.get("name", ""), e.get("description") or ""] + list(e.get("tags") or [])
                                                 if e else []))
    return bool(LOUD_CORE.search(words))
OFF_DB = -41.0
KIND_OF = {"sfx": "effect", "amb": "ambience", "music": "music"}  # tag type -> library kind
LEVEL_OF = {"effect": EFFECT_RMS_DB, "ambience": AMB_RMS_DB, "music": MUSIC_RMS_DB}
INTRO_LEAD, INTRO_TAIL = 8.0, 4.0  # chapter intro: seconds on its own, then seconds fading out under the title
INTRO_GAIN = 2.0  # +6 dB: an intro plays on its own, so it can be nearer the voices' level than background music


# ---------- library ----------

def index():
    if not INDEX.exists():
        return []
    try:
        return json.loads(INDEX.read_text())
    except ValueError as e:
        bak = INDEX.with_suffix(".json.bak")
        if bak.exists():
            try:
                print(f"[sfx] index.json corrupt ({e}); recovering from backup", flush=True)
                return json.loads(bak.read_text())
            except ValueError:
                pass
        corrupt = INDEX.with_suffix(f".corrupt-{int(time.time())}.json")
        try:
            shutil.copy2(INDEX, corrupt)
        except OSError:
            pass
        raise RuntimeError(f"Sound library index.json is unreadable ({e}); preserved copy at {corrupt.name}.") from e


# Sounds can belong to one audiobook ("book": its project id) instead of the whole library. While a book is being
# made or edited, BOOK holds its id: matching tags then uses the library's shared sounds plus that book's own.
BOOK = contextvars.ContextVar("sfx_book", default=None)


def pool(book=None):
    """The sounds a tag can use: every shared sound, plus the book's own (book, else the BOOK being worked on)."""
    book = book if book is not None else BOOK.get()
    return [e for e in index() if not e.get("book") or (book and e.get("book") == book)]


def set_scope(sid, book=None, made_in=None):
    """Keep a sound in one book (book = its project id) or share it with every book (book = None)."""
    with _lock:
        entries = index()
        hit = None
        for e in entries:
            if e["id"] == sid:
                if book:
                    e["book"] = book
                else:
                    e.pop("book", None)
                if made_in:
                    e["made_in"] = made_in
                hit = e
        if hit is None:
            raise ValueError("That sound isn't in the library any more.")
        _save(entries)
        return hit


def annotate(sid, **fields):
    """Set extra fields on a library sound (where an online sound came from, its pack…). Returns the entry."""
    with _lock:
        entries = index()
        hit = next((e for e in entries if e["id"] == sid), None)
        if hit is None:
            raise ValueError("That sound isn't in the library any more.")
        hit.update(fields)
        _save(entries)
        return hit


def also_for(sid, desc):
    """A sound already in the library that now answers to another description too (one online sound found for
    "door slam" and "heavy door slamming"), instead of a second copy."""
    with _lock:
        entries = index()
        hit = next((e for e in entries if e["id"] == sid), None)
        if hit is None:
            raise ValueError("That sound isn't in the library any more.")
        desc = re.sub(r"\s+", " ", (desc or "").strip().lower())
        also = [d for d in hit.get("also") or [] if d != desc]
        if desc and desc != (hit.get("description") or "").strip().lower():
            hit["also"] = also + [desc]
            _save(entries)
        return hit


def _save(entries):
    tmp = INDEX.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(entries, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, INDEX)
    finally:
        tmp.unlink(missing_ok=True)


def _words(text):
    return [w for w in re.split(r"[^a-z]+", text.lower()) if len(w) > 1 and w not in {"sfx", "the", "and", "of", "a", "wav", "mp3", "ogg"}]


def _rms_db(a):
    return 20 * np.log10(np.sqrt(np.mean(a ** 2)) + 1e-9)


def _normalize(a, target_db):
    a = a * 10 ** ((target_db - _rms_db(a)) / 20)
    peak = np.abs(a).max() if len(a) else 0
    return a / peak * 0.95 if peak > 0.95 else a


def _to_24k_mono(src):
    proc = subprocess.run([_ffmpeg(), "-v", "error", "-i", str(src), "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"],
                          capture_output=True, check=True)
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def mono_or_stereo(a):
    """(n, 2) stays stereo when the two sides really differ; otherwise one channel (no point storing it twice)."""
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 2 and a.shape[1] >= 2:
        a = a[:, :2]
        if np.abs(a[:, 0] - a[:, 1]).max() > 1e-3 * max(1e-6, float(np.abs(a).max())):
            return a
        return a[:, 0].copy()
    return a.reshape(-1)


def _to_24k(src):
    """Any audio file -> 24 kHz, stereo if it is stereo, else mono."""
    proc = subprocess.run([_ffmpeg(), "-v", "error", "-i", str(src), "-ac", "2", "-ar", str(SR), "-f", "f32le", "pipe:1"],
                          capture_output=True, check=True)
    return mono_or_stereo(np.frombuffer(proc.stdout, dtype=np.float32).reshape(-1, 2))


# ---------- matching by meaning (CLAP) ----------
# Tags still match by words first. When no sound's name or tags share the tag's words, CLAP compares what the
# sound actually sounds like with the tag's description. Tested on the starter sounds: right matches scored
# 0.28-0.37 and unrelated ones 0.02-0.24, so 0.25 is the bar. Imported sounds also get a few "heard" tags.
CLAP_MIN = 0.32  # by meaning alone (no shared words): below this, a wrong sound wins too often (one hub sound, "the sea", scores 0.3 for a jet or a searchlight); a miss is made instead
HALF_MIN = 0.15  # a sound only half its words match must sound at least this much like the tag (by CLAP)


def smart():
    """Matching by meaning is on and its model is downloaded."""
    from . import clap, settings
    return bool(settings.load().get("sfx_smart", True)) and clap.available() and clap.downloaded()


def _emb(entry):
    """The sound's CLAP fingerprint, worked out once and kept next to it."""
    from . import clap
    p = LIB / f"{entry['id']}.clap.npy"
    if p.exists():
        return np.load(p)
    v = clap.audio_embedding(load(entry))
    np.save(p, v)
    return v


def heard_tags(entry):
    """Add CLAP's tags for a sound that doesn't have them yet (kept apart from the tags it was given)."""
    from . import clap
    if not smart() or entry.get("heard"):
        return entry
    tags = clap.auto_tags(_emb(entry))
    with _lock:
        entries = index()
        for e in entries:
            if e["id"] == entry["id"]:
                e["heard"] = entry["heard"] = tags
        _save(entries)
    return entry


def tag_library(log=print):
    """Fingerprint and tag every sound in the library (after turning matching by meaning on)."""
    n = 0
    for e in index():
        if not e.get("heard"):
            heard_tags(e)
            n += 1
    log(f"Tagged {n} sound(s).")
    return n


def add(audio, name, kind, tags=None, license="", source="", credit="", sid=None, description=""):
    """Add a sound (float array at 24 kHz) to the library. Returns its entry."""
    audio = _normalize(np.asarray(audio, dtype=np.float32), LEVEL_OF.get(kind, AMB_RMS_DB))
    with _lock:
        entries = [e for e in index() if e["id"] != sid]
        base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "sound"
        sid = sid or base
        n = 2
        while any(e["id"] == sid for e in entries):
            sid, n = f"{base}-{n}", n + 1
        sf.write(LIB / f"{sid}.wav", audio, SR)
        entry = {"id": sid, "name": name, "kind": kind, "tags": sorted(set(_words(name) + [t.lower() for t in tags or []])),
                 "file": f"{sid}.wav", "duration": round(len(audio) / SR, 2), "license": license, "source": source,
                 "credit": credit}
        if description:
            entry["description"] = description
        entries.append(entry)
        _save(entries)
    if smart():
        try:
            heard_tags(entry)
        except Exception:  # noqa: BLE001 - tagging is a bonus; the sound is already saved
            pass
    return entry


def import_folder(folder, kind="auto", license="", source="", credit="", log=print):
    """Import every audio file in a folder (and sub-folders). Names and tags come from the file and folder names.
    kind "auto": files named like music (music, song, theme, score, bgm) are music; ones that look like
    loops/backgrounds (amb, ambience, loop, room, background, or > 20 s) are ambience."""
    added = 0
    for f in sorted(Path(folder).rglob("*")):
        if f.suffix.lower() not in AUDIO_EXT or not f.is_file():
            continue
        try:
            a = _to_24k(f)
        except subprocess.CalledProcessError:
            log(f"  skipped {f.name} (couldn't read it)")
            continue
        name = re.sub(r"[_\-]+", " ", f.stem).strip()
        k = kind
        if kind == "auto":
            hint = f"{f.parent.name} {f.stem}".lower()
            k = ("music" if re.search(r"music|song|theme|score|bgm|melod", hint) else
                 "ambience" if re.search(r"amb|loop|room ?tone|background|atmos", hint) or len(a) > 20 * SR else "effect")
        add(a, name, k, tags=_words(f.parent.name), license=license, source=source or str(f.parent), credit=credit)
        added += 1
    log(f"Imported {added} sound(s) from {folder}")
    return added


# ---------- sound packs ----------
# A folder or .zip of sounds the user has (Sonniss's free GDC bundles, a purchased library, their own recordings),
# imported into the library under the pack's name, so tags can use them and the director's sounds are found there
# before anything is made. The files are copied in (24 kHz, levelled), so the pack can be moved or deleted after.
PACK_MAX_SECONDS = 15 * 60  # longer files (whole field recordings) are left out
_UCS = re.compile(r"^[A-Z]{2,}[A-Za-z]*$")  # Universal Category System names: "DOORWood_Creak Slow_ACME_Sonniss"


def _pack_name_of(stem):
    """A sound's name from its file name, without category codes, creator ids and take numbers."""
    parts = [x for x in re.split(r"_+", stem) if x]
    if len(parts) >= 3 and _UCS.match(parts[0]):  # UCS: CatID_FXName_CreatorID_SourceID
        stem = parts[1]
    name = re.sub(r"[_\-.]+", " ", stem)
    name = re.sub(r"\b(?:take|tk|v|var|version)?\s*0*\d{1,3}\s*$", "", name, flags=re.I).strip()
    name = re.sub(r"\s+", " ", name)[:60].strip()
    if name.islower():  # "glass break" -> "Glass Break", like the library's other names
        name = name.title()
    return name or "Sound"


def _pack_kind(rel, seconds):
    hint = rel.lower()
    if re.search(r"music|song|theme|score|bgm|melod|\bmus", hint):
        return "music"
    if re.search(r"\bamb|ambien|atmos|room ?tone|background|loop|\bair\b|soundscape", hint) or seconds > 20:
        return "ambience"
    return "effect"


def packs():
    """[{name, sounds, seconds, license}] for the packs in the library."""
    out = {}
    for e in index():
        if e.get("pack"):
            p = out.setdefault(e["pack"], {"name": e["pack"], "sounds": 0, "seconds": 0.0, "license": e.get("license", "")})
            p["sounds"] += 1
            p["seconds"] += e.get("duration") or 0
    return sorted(out.values(), key=lambda p: p["name"].lower())


def remove_pack(name):
    with _lock:
        entries = index()
        gone = [e for e in entries if e.get("pack") == name]
        for e in gone:
            (LIB / e["file"]).unlink(missing_ok=True)
            (LIB / f"{e['id']}.clap.npy").unlink(missing_ok=True)
        _save([e for e in entries if e.get("pack") != name])
        return len(gone)


def import_pack(path, name="", license=""):
    """✦ Import a sound pack (a folder or a .zip): events for a task, ending with what was added."""
    import shutil
    import tempfile
    import zipfile
    from . import tasks
    src = Path(path).expanduser()
    if not src.exists():
        raise ValueError("That folder or file isn't there any more.")
    name = (name or "").strip() or re.sub(r"[_\-]+", " ", src.stem if src.is_file() else src.name).strip()
    tmp = None
    try:
        if src.is_file():
            if src.suffix.lower() != ".zip":
                raise ValueError("Pick a folder of sounds, or a .zip file.")
            yield {"status": f"Opening {src.name}…"}
            tmp = Path(tempfile.mkdtemp(prefix="babble-pack-"))
            tmp_root = tmp.resolve()
            with zipfile.ZipFile(src) as z:
                members = [m for m in z.infolist() if not m.is_dir() and Path(m.filename).suffix.lower() in AUDIO_EXT
                           and not Path(m.filename).name.startswith("._") and "__MACOSX" not in m.filename]
                for n, m in enumerate(members):
                    if n % 50 == 0:
                        yield {"status": f"Unpacking {src.name} · {n:,} of {len(members):,} sounds", "progress": 0.1 * n / max(len(members), 1)}
                        if tasks.stop_wanted():
                            raise tasks.Stopped()
                    rel = Path(m.filename)
                    if rel.is_absolute() or ".." in rel.parts:
                        continue
                    if ((m.external_attr >> 16) & 0o170000) == 0o120000:  # reject symlinks
                        continue
                    target = (tmp_root / rel).resolve()
                    if tmp_root not in target.parents:
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(m) as zf, open(target, "wb") as out:
                        shutil.copyfileobj(zf, out, 1 << 20)
            root = tmp
        else:
            root = src
        files = sorted(f for f in root.rglob("*") if f.is_file() and f.suffix.lower() in AUDIO_EXT
                       and not f.name.startswith("._") and "__MACOSX" not in f.parts)
        if not files:
            raise ValueError("No sounds were found there (wav, mp3, flac, ogg, aiff, m4a).")
        have = {(e.get("pack"), e.get("pack_file")) for e in index()}
        added, skipped, long_ = 0, 0, 0
        for n, f in enumerate(files):
            rel = str(f.relative_to(root))
            if n % 5 == 0:
                yield {"status": f"Adding “{name}” · {n:,} of {len(files):,} sounds", "progress": 0.1 + 0.9 * n / len(files)}
            if tasks.stop_wanted():
                raise tasks.Stopped()
            if (name, rel) in have:
                continue  # imported before (the same pack again: only what's new is added)
            try:
                a = _to_24k(f)
            except subprocess.CalledProcessError:
                skipped += 1
                continue
            secs = len(a) / SR
            if secs < 0.05:
                skipped += 1
                continue
            if secs > PACK_MAX_SECONDS:
                long_ += 1
                continue
            kind = _pack_kind(rel, secs)
            folder_words = _words(" ".join(Path(rel).parent.parts))
            e = add(a, unique_name(_pack_name_of(f.stem)), kind, tags=folder_words, license=license,
                    source=f"{name}: {rel}", credit=name)
            annotate(e["id"], pack=name, pack_file=rel)
            added += 1
        note = []
        if skipped:
            note.append(f"{skipped} couldn't be read")
        if long_:
            note.append(f"{long_} longer than {PACK_MAX_SECONDS // 60} minutes were left out")
        yield {"status": f"Added {added:,} sound(s) from “{name}” to your library"
                         + (f" ({'; '.join(note)})" if note else "") + ".", "pack": name, "added": added}
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


def unique_name(name, taken=None):
    """'Missile Explosion' -> 'Missile Explosion 2' when the library already has one."""
    taken = {n.lower() for n in (taken if taken is not None else (e["name"] for e in index()))}
    if name.lower() not in taken:
        return name
    n = 2
    while f"{name} {n}".lower() in taken:
        n += 1
    return f"{name} {n}"


def shorten_names(longer_than=24, log=print):
    """Give long names a short one; the old name is kept as the description and all its words stay as tags, so tags
    in stories that use the old name still find the sound."""
    from . import naming
    entries = index()
    todo = [e for e in entries if len(e["name"]) > longer_than or re.match(r"(?i)(a|an|the)\s", e["name"])]
    if not todo:
        log("No long names to shorten.")
        return 0
    log(f"Naming {len(todo)} sound(s)…")
    short = naming.short_names([e.get("description") or e["name"] for e in todo])
    mapping = {e["id"]: name for e, name in zip(todo, short)}
    with _lock:
        entries = index()
        taken = [e["name"] for e in entries if e["id"] not in mapping]
        for e in entries:
            if e["id"] not in mapping:
                continue
            name = unique_name(mapping[e["id"]], taken)
            taken.append(name)
            e.setdefault("description", e["name"])
            e["tags"] = sorted(set(e["tags"]) | set(_words(e["name"])) | set(_words(name)))
            log(f"  {e['name']} -> {name}")
            e["name"] = name
        _save(entries)
    return len(todo)


def delete(sid):
    delete_many([sid])


def delete_many(sids):
    """Delete these sounds with one save of the index: their audio goes to the Trash under the sound's name (one
    sound: the file itself; more: a folder "Babble sounds (N, <time>)"), their meaning vectors are removed.
    Returns how many went."""
    gone = set(sids)
    with _lock:
        entries = index()
        hit = [e for e in entries if e["id"] in gone]
        if not hit:
            return 0
        _save([e for e in entries if e["id"] not in gone])
        _to_trash([(LIB / e["file"], e.get("name") or e["id"]) for e in hit])
        for e in hit:
            (LIB / f"{e['id']}.clap.npy").unlink(missing_ok=True)
        return len(hit)


def _free_name(folder, stem, ext):
    """stem + ext in folder, or "stem 2" + ext, "stem 3"… when that's taken."""
    stem = re.sub(r"[/:\x00]", "-", stem).strip() or "sound"
    n, name = 1, f"{stem}{ext}"
    while (folder / name).exists():
        n += 1
        name = f"{stem} {n}{ext}"
    return folder / name


def _to_trash(files):
    """Move [(path, name)] to the Trash (named by the sound, not its id). When there's no Trash to move them to,
    they go to the library's "Deleted" folder instead (not in the library any more, but not lost)."""
    files = [(Path(p), name) for p, name in files if Path(p).exists()]
    if not files:
        return
    import sys
    trash = Path.home() / ".Trash" if sys.platform == "darwin" else LIB / "Deleted"
    try:
        if len(files) > 1:
            trash = trash / f"Babble sounds ({len(files)}, {time.strftime('%Y-%m-%d %H.%M.%S')})"
        trash.mkdir(parents=True, exist_ok=True)
    except OSError:
        trash = LIB / "Deleted"
        trash.mkdir(parents=True, exist_ok=True)
    for p, name in files:
        try:
            shutil.move(str(p), str(_free_name(trash, name, p.suffix)))
        except OSError:  # macOS refused the Trash (privacy settings): kept in the library's "Deleted" folder
            (LIB / "Deleted").mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(_free_name(LIB / "Deleted", name, p.suffix)))


def _by_words(query, pool_):
    """The sound whose tags share most of the query's words, and its score (0.5 or more counts as a match)."""
    q = set(_words(query))
    best, score = None, 0.0
    for e in pool_:
        s = len(q & set(e["tags"])) / max(1, len(q)) + 0.5 * len(q & (set(e.get("heard", [])) - set(e["tags"]))) / max(1, len(q))
        s += 0.5 if e["id"] == query.lower().replace(" ", "-") else 0
        s += 0.01 if e.get("book") else 0  # a book's own sound wins a tie with a shared one
        if s > score:
            best, score = e, s
    return best, score


TAG_TYPE = {"effect": "sfx", "ambience": "amb", "music": "music"}  # library kind -> tag type


def tag_for(entry, sounds=None):
    """The shortest tag that plays exactly this sound: its name when that picks it, else its id.
    Checked by words only (the sounds the tag can use: shared ones plus the BOOK's own), so no CLAP is loaded."""
    t = TAG_TYPE.get(entry["kind"], "sfx")
    name = re.sub(r"\s+", " ", re.sub(r"[{}\[\],@]", " ", entry["name"])).strip().lower()
    pool_ = [e for e in (pool() if sounds is None else sounds) if e["kind"] == entry["kind"]]
    for text in (name, entry["id"]):
        best, score = _by_words(text, pool_) if text else (None, 0)
        if best and best["id"] == entry["id"] and score >= 0.5:
            return f"{{{t}: {text}}}"
    return f"{{{t}: {entry['id']}}}"


_EXACT_OF = {}  # book id -> (project file time, its "Make the sounds it asks for"), so a long book doesn't reread it per tag


def exact_sounds():
    """A tag plays only a sound made for it (or one whose own tags hold every word of it); anything else counts as
    missing, so it's made instead of a near miss. Always, now that the director makes the sounds it needs (the
    switch is gone); only a book that turned it off itself, when it was a switch, still takes near misses."""
    book = BOOK.get()
    if book:
        try:
            from . import projects
            f = projects._path(book)
            t = f.stat().st_mtime
            if _EXACT_OF.get(book, (None,))[0] != t:
                _EXACT_OF[book] = (t, projects.get(book).get("sfx_exact"))
            if _EXACT_OF[book][1] is False:
                return False
        except (OSError, ValueError):
            pass
    return True


def _exact(query, pool_):
    """A sound made for this description, named it, or whose own tags hold every word of it; never a near miss
    (the closest sound by meaning, or one sharing only some words: "children cheering" isn't "children shrieking")."""
    text = re.sub(r"\s+", " ", (query or "").strip().lower())
    words = set(_words(text))
    if not text:
        return None
    slug = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    made_for = lambda e: (e.get("description") or "").strip().lower() == text or text in (e.get("also") or [])  # noqa: E731
    hits = [e for e in pool_ if made_for(e) or e["name"].lower() == text
            or e["id"] == slug or (words and words <= set(e["tags"]))]
    # made for it first, then the one with the fewest extra words (the most specific), the book's own on a tie
    return min(hits, key=lambda e: (not made_for(e),
                                    len(set(e["tags"]) - words), not e.get("book")), default=None)


def find(query, kind=None):
    """Best library match for a tag like "door creak", or None. Words first (the sound's own tags count fully,
    tags CLAP heard count half); with no word match and matching by meaning on, the closest-sounding sound.
    Only an exact match counts while exact_sounds() (always, unless a book turned it off before it was removed)."""
    pool_ = [e for e in pool() if not kind or e["kind"] == kind]
    if exact_sounds():
        return _exact(query, pool_)
    best, score = _by_words(query, pool_)
    if score >= 0.75 or not pool_ or not query.strip():
        return best if score >= 0.5 else None
    try:
        if smart():
            from . import clap
            t = clap.text_embedding([f"the sound of {query}"])[0]
            if score >= 0.5:
                # half a match ("door creak" -> Door Slam, "pistol gunshot" -> a pistol pulled from a holster):
                # kept only if it also sounds like the query; otherwise the closest-sounding sound, if any
                if float(t @ _emb(best)) >= HALF_MIN:
                    return best
                score = 0
            sim, e = max(((float(t @ _emb(e)), e) for e in pool_), key=lambda x: x[0])
            if sim >= CLAP_MIN:
                return e
    except Exception:  # noqa: BLE001 - fall back to "no match" rather than failing a render
        pass
    return best if score >= 0.5 else None


def load(entry):
    """Mono (what matching and tagging use)."""
    a, sr = sf.read(LIB / entry["file"], dtype="float32", always_2d=True)
    return a.mean(axis=1)


def _st(a):
    """Any sound as (n, 2) stereo; mono goes to both sides equally."""
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 1:
        return np.stack([a, a], axis=1)
    return a[:, :2] if a.shape[1] >= 2 else np.repeat(a, 2, axis=1)


def load_st(entry):
    """Stereo (what the mixer uses): stereo files keep their width."""
    a, sr = sf.read(LIB / entry["file"], dtype="float32", always_2d=True)
    return _st(a)


def _col(x, like):
    """A per-sample curve shaped to multiply mono (n,) or stereo (n, 2) audio."""
    return x[:, None] if np.ndim(like) == 2 else x


# ---------- starter sounds (generated, so CC0) ----------

def _noise(n, seed):
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32)


def _lowpass(x, cutoff):
    from scipy.signal import butter, sosfilt
    return sosfilt(butter(4, cutoff, fs=SR, output="sos"), x).astype(np.float32)


def _bandpass(x, lo, hi):
    from scipy.signal import butter, sosfilt
    return sosfilt(butter(4, [lo, hi], btype="band", fs=SR, output="sos"), x).astype(np.float32)


def _env(n, attack, decay):
    t = np.arange(n) / SR
    return (np.minimum(1, t / max(attack, 1e-4)) * np.exp(-t / decay)).astype(np.float32)


def starter_sounds():
    """A small set of generated sounds so tags work out of the box. Replace them with real recordings."""
    out, secs = [], 30
    n = SR * secs
    t = np.arange(n) / SR
    rain = _bandpass(_noise(n, 1), 800, 9000) * 0.6
    drops = np.zeros(n, np.float32)
    rng = np.random.default_rng(2)
    for pos in rng.integers(0, n - 2000, 1800):
        drops[pos:pos + 2000] += _bandpass(_noise(2000, int(pos)), 2000, 8000) * _env(2000, 0.0005, 0.004) * rng.uniform(0.2, 1)
    out.append(("Rain", "ambience", ["rain", "storm", "drizzle", "downpour"], rain + drops))
    gust = 0.6 + 0.4 * np.sin(2 * np.pi * t / 7) * np.sin(2 * np.pi * t / 3.1)
    out.append(("Wind", "ambience", ["wind", "breeze", "gust", "storm"], _lowpass(_noise(n, 3), 600) * gust))
    swell = 0.5 + 0.5 * np.sin(2 * np.pi * t / 8) ** 2
    out.append(("Ocean waves", "ambience", ["ocean", "sea", "waves", "beach", "shore"], _lowpass(_noise(n, 4), 1500) * swell))
    fire = _lowpass(_noise(n, 5), 400) * 0.5
    for pos in rng.integers(0, n - 600, 900):
        fire[pos:pos + 600] += _bandpass(_noise(600, int(pos) + 7), 1500, 7000) * _env(600, 0.0002, 0.002) * rng.uniform(0.3, 1.5)
    out.append(("Fireplace", "ambience", ["fire", "fireplace", "hearth", "campfire", "crackle"], fire))
    out.append(("Room tone", "ambience", ["room", "quiet", "indoors", "silence"], _lowpass(_noise(n, 6), 300) * 0.3))
    m = SR * 4
    out.append(("Thunder", "effect", ["thunder", "storm", "lightning", "rumble"],
                _lowpass(_noise(m, 8), 180) * _env(m, 0.05, 1.2) + _lowpass(_noise(m, 9), 1200) * _env(m, 0.005, 0.15) * 0.5))
    knock = np.zeros(int(SR * 1.2), np.float32)
    for i in range(3):
        s = int(SR * (0.05 + 0.28 * i))
        k = int(SR * 0.12)
        knock[s:s + k] += _lowpass(_noise(k, 10 + i), 900) * _env(k, 0.001, 0.025) + \
            np.sin(2 * np.pi * 110 * np.arange(k) / SR).astype(np.float32) * _env(k, 0.001, 0.03)
    out.append(("Door knock", "effect", ["knock", "knocking", "door", "knocked"], knock))
    hb = np.zeros(int(SR * 2.4), np.float32)
    for beat in (0.1, 0.35, 1.3, 1.55):
        s, k = int(SR * beat), int(SR * 0.15)
        hb[s:s + k] += np.sin(2 * np.pi * 55 * np.arange(k) / SR).astype(np.float32) * _env(k, 0.005, 0.04)
    out.append(("Heartbeat", "effect", ["heartbeat", "heart", "pulse"], hb))
    k = int(SR * 0.5)
    out.append(("Footstep", "effect", ["footstep", "footsteps", "step", "steps", "walk"],
                _lowpass(_noise(k, 20), 700) * _env(k, 0.002, 0.04)))
    return out


def install_starters(log=print):
    have = {e["id"] for e in index()}
    added = 0
    for name, kind, tags, audio in starter_sounds():
        sid = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        if sid in have:
            continue
        add(audio, name, kind, tags, license="CC0", source="generated by Babble", sid=sid)
        added += 1
    log(f"Added {added} starter sound(s). They're simple generated placeholders; import real recordings for a book.")
    return added


# ---------- tags in text ----------

def split_cues(text):
    """'He knocked. {sfx: knock} Nothing.' -> [("text", "He knocked."), ("cue", {...}), ("text", "Nothing.")]."""
    out, pos = [], 0
    for m in CUE.finditer(text or ""):
        before = text[pos:m.start()].strip()
        if before:
            out.append(("text", before))
        kind, body = m.group(1).lower(), [x.strip() for x in (m.group(2) or "").split(",")]
        if kind == "pause":
            out.append(("cue", {"type": "pause", "name": None, "seconds": pause_seconds(m.group(2))}))
            pos = m.end()
            continue
        if kind == "pace":
            pos = m.end()
            continue
        name = body[0] if body else ""
        anchor = re.search(r"@\s*([^\s@,]+)", name)  # {sfx: knock @knocked}: play it on that word of the line before
        name = re.sub(r"\s*@\s*[^\s@,]+", "", name).strip()
        opts = {x.lower() for x in body[1:]}
        vol = next((x.lower() for x in body[1:] if x.lower() in VOLUME), "normal")
        cue = {"type": kind, "name": None if kind in ("amb", "room", "music") and name.lower() in ("off", "stop", "none", "")
               else name, "volume": vol}
        if kind == "room" and cue["name"]:
            cue["name"] = name.lower() if name.lower() in ROOMS else next(
                (r for r in ROOMS if r in name.lower()), "room")  # {room: big cave} -> cave
        if kind == "sfx":
            cue["pan"] = next((PAN[o] for o in ("hard left", "hard right", "slight left", "slight right", "left", "right") if o in opts), 0.0)
            cue["distance"] = "far" if opts & {"far", "distant", "afar"} else \
                "muffled" if opts & {"muffled", "behind", "through wall", "wall"} else ""
        if kind in ("amb", "music") and cue["name"]:
            # {amb: rain} changes the scene; {amb: fire, add} layers on top; {amb: fire, stop} fades just that one out
            # (music works the same way, as its own layer: a new {amb: …} doesn't stop the music, or the other way)
            cue["op"] = "add" if opts & {"add", "layer", "also"} else "stop" if opts & {"stop", "off", "end"} else "set"
        if anchor and kind == "sfx":
            cue["at"] = anchor.group(1)
        out.append(("cue", cue))
        pos = m.end()
    rest = (text or "")[pos:].strip()
    if rest:
        out.append(("text", rest))
    return out


def find_for(cue):
    """The library sound for a tag: one of its own kind first. Effects and backgrounds fall back to any kind; music
    only to a background (so {music: ball waltz} never lands on a "Ball Bounce" effect)."""
    if not cue.get("name"):
        return None
    kind = KIND_OF.get(cue["type"], "effect")
    e = find(cue["name"], kind)
    if e:
        return e
    return find(cue["name"], "ambience") if kind == "music" else find(cue["name"])


def missing(text):
    """Tags in a text that no library sound matches: [(kind, name)] in order, without repeats."""
    out = []
    for m in CUE.finditer(text or ""):
        for kind, cue in split_cues(m.group()):
            if kind != "cue" or not cue.get("name") or cue["type"] in ("room", "pause", "pace"):
                continue
            k = KIND_OF[cue["type"]]
            if not find_for(cue) and (k, cue["name"]) not in out:
                out.append((k, cue["name"]))
    return out


def strip(text, keep_pauses=False):
    """The text without its tags (keep_pauses: {pause: …} stays, as it's not a sound)."""
    out = CUE.sub(lambda m: m.group() if keep_pauses and m.group(1).lower() == "pause" else " ", text or "")
    return re.sub(r"[ \t]{2,}", " ", out).strip()


def _fades(n):
    """Equal-power fade-in and fade-out curves (no dip in loudness halfway through a crossfade)."""
    t = np.linspace(0, np.pi / 2, n, dtype=np.float32)
    return np.sin(t), np.cos(t)


def _trim_quiet(a, below_db=12.0):
    """Cut a background's fade-in and fade-out: keep from the first to the last 100 ms that's within 12 dB of its
    typical loudness."""
    hop = SR // 10
    n = len(a) // hop
    if n < 10:
        return a
    m = a.mean(axis=1) if a.ndim == 2 else a
    db = 20 * np.log10(np.sqrt((m[: n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)
    ok = np.where(db > np.median(db) - below_db)[0]
    if not len(ok) or (ok[-1] - ok[0] + 1) < n * 0.5:  # keep it as it is if most of it would go
        return a
    return a[ok[0] * hop:(ok[-1] + 1) * hop]


def loopable(a, xfade=2 * SR):
    """Make a clip loop without a seam: its end is blended into its start, so repeat after repeat runs on smoothly.
    (A background made by AI or recorded once isn't cut to loop; a plain repeat jumps at every join.)"""
    a = _trim_quiet(a)  # AI-made clips often fade in and out; a quiet end blended into a loud start dips every loop
    xf = min(int(xfade), len(a) // 4)
    if xf < SR // 20:
        return a
    fin, fout = _fades(xf)
    out = a[: len(a) - xf].copy()
    out[:xf] = a[:xf] * _col(fin, a) + a[len(a) - xf:] * _col(fout, a)
    return out


def _loop(a, n, xfade=2 * SR):
    """Repeat a seamless version of a to length n."""
    a = loopable(a, xfade)
    if len(a) >= n:
        return a[:n].copy()
    reps = n // len(a) + 1
    return (np.tile(a, (reps, 1)) if a.ndim == 2 else np.tile(a, reps))[:n]


FADE_IN, FADE_OUT = int(1.5 * SR), int(2.0 * SR)


def _background_layers(amb, n):
    """[(sample, cue, entry)] -> [(start, end, entry, gain)]: which backgrounds play when. A scene change ends all
    current layers; 'add' layers on; 'stop' ends one; 'off' ends all; everything ends with the chapter."""
    layers, active = [], {}

    def close(key, at):
        start, e, g = active.pop(key)
        if at > start:
            layers.append((start, at, e, g))
    for at, c, e in amb:
        if c["name"] is None:
            for key in list(active):
                close(key, at)
            continue
        if e is None:
            continue
        op, g = c.get("op", "set"), VOLUME[c["volume"]]
        if op == "stop":
            if e["id"] in active:
                close(e["id"], at)
        elif op == "add":
            if e["id"] not in active:
                active[e["id"]] = (at, e, g)
        elif not (set(active) == {e["id"]}):  # a new scene (the same one again just carries on)
            for key in list(active):
                close(key, at)
            active[e["id"]] = (at, e, g)
    for key in list(active):
        close(key, n)
    return layers


def _smooth(x, attack, release):
    """One-pole envelope follower over a 0..1 gate (attack/release in samples)."""
    from scipy.signal import lfilter
    up = lfilter([1 - np.exp(-1 / attack)], [1, -np.exp(-1 / attack)], x)
    down = lfilter([1 - np.exp(-1 / release)], [1, -np.exp(-1 / release)], x)
    return np.maximum(up, down).astype(np.float32)


# ---------- rooms and distance ----------
# {room: cave} puts the voices (and effects) in a space: a short synthetic room response (decaying noise, darker as it
# fades) mixed in at that room's level. rt = seconds for the echo to die away, wet = how much of it you hear.
ROOMS = {"room": (0.45, 0.12), "bathroom": (0.8, 0.22), "hall": (1.6, 0.22), "cave": (2.6, 0.30),
         "cathedral": (3.4, 0.32)}


def _impulse(rt, seed=7):
    from scipy.signal import butter, sosfilt
    n = int(SR * min(rt, 3.5))
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    ir = rng.standard_normal(n).astype(np.float32) * np.exp(-6.9 * t / rt).astype(np.float32)  # -60 dB at rt
    ir = sosfilt(butter(2, 5000, "low", fs=SR, output="sos"), ir).astype(np.float32)  # rooms soak up the highs
    pre = int(SR * min(0.02 + rt * 0.01, 0.05))  # early gap before the first reflections
    ir = np.concatenate([np.zeros(pre, np.float32), ir])
    return ir / (np.sqrt(np.sum(ir ** 2)) + 1e-9)


def _reverb(x, room):
    """x plus its echo in that room, in stereo: each side gets a slightly different echo, so the room sounds wide
    (the tail runs past the end of x)."""
    from scipy.signal import oaconvolve
    rt, wet = ROOMS[room]
    x = _st(x)
    out = np.concatenate([x, np.zeros((int(SR * min(rt, 3.5)) + SR // 10, 2), np.float32)])
    for ch, seed in ((0, 7), (1, 11)):
        tail = oaconvolve(x[:, ch], _impulse(rt, seed)).astype(np.float32)
        out[: len(tail), ch] += tail[: len(out)] * wet
    return out


def _distance(a, how):
    """'far': quieter, duller, more echo; 'muffled': as if through a wall or door."""
    from scipy.signal import butter, sosfilt
    if how == "far":
        a = sosfilt(butter(2, 2800, "low", fs=SR, output="sos"), a, axis=0).astype(np.float32) * 0.45
        return _reverb(a, "hall")[: len(a) + SR] * 0.8
    if how == "muffled":
        return sosfilt(butter(4, 700, "low", fs=SR, output="sos"), a, axis=0).astype(np.float32) * 0.6
    return a


PAN = {"left": -0.7, "right": 0.7, "hard left": -1.0, "hard right": 1.0, "slight left": -0.35, "slight right": 0.35}


def _pan(a, p):
    """Place a sound left (-1) to right (+1); centre keeps both sides at full level (equal-power)."""
    if not p:
        return a
    ang = (p + 1) * np.pi / 4
    return a * np.array([np.cos(ang), np.sin(ang)], np.float32) * np.sqrt(2)


ROOM_TONE_DB = -66.0  # studio floor safely below ACX's -60 dB RMS ceiling (even after loudnorm lift) while avoiding digital zero
_ROOM_TILE = {}


def room_tone(n, db=ROOM_TONE_DB, stereo=True, seed=17):
    """Continuous, seamless warm room air (40-320 Hz) of length n, leveled to `db` RMS.
    Built from a cached 8 s loopable tile so even an hour-long chapter takes only milliseconds."""
    n = int(n)
    if n <= 0:
        return np.zeros((0, 2), np.float32) if stereo else np.zeros(0, np.float32)
    key = (round(float(db), 1), bool(stereo), int(seed))
    tile = _ROOM_TILE.get(key)
    if tile is None:
        tn = SR * 8
        ch = 2 if stereo else 1
        raw = np.stack([_bandpass(_noise(tn, seed + c * 31), 40, 320) for c in range(ch)], axis=1) if stereo \
            else _bandpass(_noise(tn, seed), 40, 320)
        raw = raw * (10 ** ((db - _rms_db(raw)) / 20))
        tile = loopable(raw, xfade=SR).astype(np.float32)
        _ROOM_TILE[key] = tile
    reps = n // len(tile) + 1
    out = (np.tile(tile, (reps, 1)) if stereo else np.tile(tile, reps))[:n].copy()
    fade = min(int(0.35 * SR), n // 4)
    if fade > 1:
        fin, fout = _fades(fade)
        out[:fade] *= _col(fin, out)
        out[-fade:] *= _col(fout, out)
    return out


WHISPER_DUCK_DB = 6.0  # extra ducking under hushed/whispered speech (RMS well below SPEECH_DB)


def _whisper_duck(dry, spans, n):
    """Per-sample extra ducking multiplier (1.0 normally, down to -6 dB under whispered/quiet lines)."""
    soft_gate = np.zeros(n, np.float32)
    any_soft = False
    for s, e in spans:
        seg = dry[s:e]
        if len(seg) < SR // 10:
            continue
        hop = max(1, SR // 20)
        frames = len(seg) // hop
        if not frames:
            continue
        rms = np.sqrt((seg[:frames * hop].reshape(frames, hop) ** 2).mean(1) + 1e-12)
        db = 20 * np.log10(rms)
        active = rms[db > db.max() - 25.0]
        if not len(active):
            continue
        span_db = float(20 * np.log10(np.sqrt(np.mean(active ** 2)) + 1e-12))
        # Speech leveled to -20 dB is normal; whispered lines (-26 dB) get proportional extra ducking
        quietness = min(1.0, max(0.0, (SPEECH_DB - 3.0 - span_db) / 4.0))
        if quietness > 0.05:
            soft_gate[s:e] = quietness
            any_soft = True
    if not any_soft:
        return None
    env = _smooth(soft_gate, int(0.08 * SR), int(0.45 * SR))
    return (10.0 ** (-WHISPER_DUCK_DB * env / 20.0)).astype(np.float32)


def _room_spans(rooms, n):
    """[(sample, room name or None)] -> [(start, end, room)]"""
    out, cur, start = [], None, 0
    for at, name in rooms + [(n, None)]:
        if cur and at > start:
            out.append((start, at, cur))
        cur, start = name, at
    return out


# ---------- mixing ----------

def levels_of(bed=1.0):
    """{"sfx", "amb", "music"}: dB under the narrator while they speak. bed is the book's mix settings: a number (the
    old Background and music volume, a multiplier) or {"bed", "sfx", "amb", "music"}, where a level left None is the
    default (backgrounds and music moved by bed)."""
    d = bed if isinstance(bed, dict) else {"bed": bed}
    b = float(d.get("bed", 1.0) if d.get("bed") is not None else 1.0)
    shift = 20 * np.log10(b) if b > 0 else -200.0
    out = {}
    for k, base in DEFAULT_LEVELS.items():
        v = d.get(k)
        v = base + (shift if k != "sfx" else 0.0) if v is None else float(v)
        out[k] = min(v, MAX_LEVELS[k])
    return out


def mix_key(bed):
    """The mix settings as they go into a chapter's key: the old number while no slider was moved (so chapters made
    before still count as made)."""
    d = bed if isinstance(bed, dict) else {"bed": bed}
    b = float(d.get("bed", 1.0) if d.get("bed") is not None else 1.0)
    moved = {k: round(float(d[k]), 1) for k in ("sfx", "amb", "music") if d.get(k) is not None}
    return [b, moved] if moved else b


def _layer_gains(level, base_db, duck_db, speaking):
    """(gain in the pauses, gain while speaking as an array) for a background or music layer whose sounds are
    leveled to base_db: while the narrator speaks it sits `level` dB under them; in the pauses it comes up by as
    much as it always did (duck_db), but never past the narrator's level."""
    if level <= OFF_DB:
        return 0.0, np.zeros_like(speaking)
    depth = max(0.0, min(-duck_db, -level))
    pause = 10 ** ((SPEECH_DB + level + depth - base_db) / 20)
    return pause, 1 - (1 - 10 ** (-depth / 20)) * speaking


def mix(speech, cues, spans, log=print, bed=1.0):
    """speech: mono float32 at 24 kHz. cues: [(sample, cue dict)]. spans: [(start, end)] samples where someone speaks.
    Returns the mix in stereo, (n, 2): voices centred, backgrounds and effects with their own width and position
    (same length as speech, plus any effect tail)."""
    if not cues:
        return speech
    total = len(speech)
    extra = 0
    events, rooms = [], []
    for at, c in cues:
        if c["type"] == "room":
            rooms.append((at, c["name"]))
            continue
        if c["type"] in ("intro", "outro"):  # a chapter intro (or outro) names its library sound by id
            e = next((x for x in index() if x["id"] == c.get("id")), None)
            if e:
                events.append((at, c, e))
            else:
                log(f"The chapter intro “{c.get('name') or c.get('id')}” isn't in the library any more: skipped.")
            continue
        e = find_for(c)
        if c["name"] and not e:
            log(f"No sound in the library for “{c['name']}”: skipped. Add one in the Sound Effects tab.")
            continue
        events.append((at, c, e))
        if c["type"] == "sfx" and e:
            ring = 3.5 if (rooms or c.get("distance") == "far") else 0.0  # echo carries on after the sound
            dur = min(e["duration"], 15.0) if e.get("kind") != "effect" else e["duration"]
            extra = max(extra, at + int((dur + ring) * SR) - total)
    dry = np.concatenate([np.asarray(speech, np.float32).reshape(-1), np.zeros(max(0, extra), np.float32)])
    n = len(dry)
    out = _st(dry)  # stereo from here: voices in the middle
    room_at = _room_spans(sorted(rooms, key=lambda r: r[0]), n)
    for start, end, room in room_at:  # each room's echo is worked out from the dry voice, so rooms never echo each other
        wet = _reverb(dry[start:end], room)  # the segment plus its echo tail
        seg, stop = end - start, min(n, start + len(wet))
        out[start:end] = wet[:seg]
        if stop > end:
            out[end:stop] += wet[seg:stop - start]  # the tail rings on past the change
    gate = np.zeros(n, np.float32)
    for s, e in spans:
        gate[s:e] = 1.0
    speaking = _smooth(gate, int(0.08 * SR), int(0.4 * SR))
    whisper = _whisper_duck(dry, spans, n)
    music_duck = 1 - (1 - 10 ** (MUSIC_DUCK_DB / 20)) * speaking  # (the chapter intro's, which no slider moves)
    if whisper is not None:
        music_duck = music_duck * whisper
    lv = levels_of(bed)
    # subtle continuous room tone under narration so pauses without an ambience bed never drop to dead air
    if spans:
        out += room_tone(n, stereo=True)
    # backgrounds and music are separate layers: each has its own scenes, crossfades and ducking
    amb = [(at, c, e) for at, c, e in events if c["type"] == "amb"]
    g, duck = _layer_gains(lv["amb"], AMB_RMS_DB, DUCK_DB, speaking)
    if whisper is not None:
        duck = duck * whisper
    if amb and g:
        out += _bed(amb, n, 2 * SR) * (duck * g)[:, None]
    music = [(at, c, e) for at, c, e in events if c["type"] == "music"]
    g, duck = _layer_gains(lv["music"], MUSIC_RMS_DB, MUSIC_DUCK_DB, speaking)
    if whisper is not None:
        duck = duck * whisper
    if music and g:  # music loops with a longer crossfade, so the join doesn't land on a jump in the tune
        out += _bed(music, n, 4 * SR) * (duck * g)[:, None]
    fx_off = lv["sfx"] <= OFF_DB
    for at, c, e in events:
        if c["type"] == "intro":
            out += _intro(load_st(e), at, c, n) * music_duck[:, None]
            continue
        if c["type"] == "outro":
            out += _outro(load_st(e), at, c, n)
            continue
        if c["type"] == "sfx" and e:
            if fx_off:
                continue
            over = LOUD_OVER if loud_core(c, e) else 0.0
            level = min(lv["sfx"] + 20 * np.log10(VOLUME[c["volume"]]) + over, over)  # dB under (or over) the narrator
            raw_sfx = load_st(e)
            if e.get("kind") != "effect" and len(raw_sfx) > int(15.0 * SR):
                raw_sfx = raw_sfx[:int(15.0 * SR)].copy()
                fo = min(SR, len(raw_sfx) // 4)
                _, fout = _fades(fo)
                raw_sfx[-fo:] *= fout[:, None]
            a = _st(_distance(raw_sfx, c.get("distance", ""))) * 10 ** ((SPEECH_DB + level - EFFECT_RMS_DB) / 20)
            room = next((r for s0, s1, r in room_at if s0 <= at < s1), None)
            if room:  # an effect in a room echoes like the voices do
                a = _reverb(a, room)
            a = _pan(a, c.get("pan", 0.0))
            out[at:at + len(a)] += a[: n - at]
    return _limit(out).astype(np.float32)


def _bed(events, n, xfade):
    """Background or music layers [(sample, cue, entry)] -> one stereo track of length n."""
    track = np.zeros((n, 2), np.float32)
    for start, end, e, g in _background_layers(events, n):
        # a layer that ends keeps going while it fades out, overlapping whatever starts there: a crossfade
        tail = min(FADE_OUT, n - end)
        bed = _loop(load_st(e), end - start + tail, xfade) * g
        fi = min(FADE_IN, (end - start) // 2)
        fin, _ = _fades(fi)
        bed[:fi] *= fin[:, None]
        fo = tail if tail else min(FADE_OUT, (end - start) // 3)
        _, fout = _fades(fo)
        bed[len(bed) - fo:] *= fout[:, None]
        track[start:start + len(bed)] += bed[: n - start]
    return track


def intro_lead(entry):
    """Seconds a chapter intro plays on its own before the chapter title (shorter sounds: most of their length)."""
    return round(max(0.5, min(INTRO_LEAD, entry["duration"] - 1.0)), 2)


# Chapter intros from a long theme (books.with_intro): each chapter gets its own stretch of it
INTRO_CUT = 12.0                        # seconds of a long theme a chapter's intro uses (its lead, then the fade)
INTRO_AFTER_PEAK = 0.8                  # the first words start this long after the stretch's loudest moment
INTRO_LEAD_MIN, INTRO_LEAD_MAX = 3.0, 9.0
OUTRO_PAUSE, OUTRO_SECONDS, LAST_OUTRO_SECONDS = 1.0, 5.0, 10.0  # after the last words; the theme's ending
OUTRO_WHOLE = 16.0                      # an outro sound this short plays whole (stings, closing phrases)
INTRO_DB = -4.0                         # intro volume default: dB under the narrator (the old INTRO_GAIN on music)
_CUTS = {}                              # sound id -> (file time, its stretches)


def _envelope(a, hop=0.25):
    """Loudness (dB) every `hop` seconds, smoothed over a second."""
    h = max(1, int(hop * SR))
    n = len(a) // h
    if not n:
        return np.zeros(1, np.float32)
    db = 20 * np.log10(np.sqrt((a[: n * h].reshape(n, h) ** 2).mean(1)) + 1e-6)
    k = max(1, int(1 / hop))
    padded = np.pad(db, (k // 2, k - 1 - k // 2), mode="edge")  # (zeros at the edges would read as loud)
    return np.convolve(padded, np.ones(k) / k, mode="valid")


def _sounding_end(env, hop=0.25, below=30.0):
    """Seconds where a sound has faded to silence (more than `below` dB under its loudest): its real ending."""
    loud = np.nonzero(env >= env.max() - below)[0]
    return ((loud[-1] + 1) * hop) if len(loud) else len(env) * hop


def intro_cuts(entry):
    """The stretches of an intro sound a chapter can open with: [{"start", "energy", "peak"}] (seconds; peak is the
    loudest moment, from the stretch's start, within the longest lead). One stretch (the whole sound) when it's too
    short to cut."""
    f = LIB / entry["file"]
    t = f.stat().st_mtime if f.exists() else 0
    if _CUTS.get(entry["id"], (None,))[0] == t:
        return _CUTS[entry["id"]][1]
    a = load(entry)
    env, hop = _envelope(a), 0.25
    dur = min(len(a) / SR, _sounding_end(env, hop))  # not its fade into silence
    env = env[: int(dur / hop)]
    width, reach = int(INTRO_CUT / hop), int(INTRO_LEAD_MAX / hop)
    if dur < 2 * INTRO_CUT:
        cuts = [{"start": 0.0, "energy": float(env.mean()), "peak": float(np.argmax(env[:reach]) * hop)}]
    else:
        cuts = []
        for k in range(0, len(env) - width, int(1 / hop)):
            w = env[k:k + width]
            cuts.append({"start": k * hop, "energy": float(w.mean()), "peak": float(np.argmax(w[:reach]) * hop)})
    _CUTS[entry["id"]] = (t, cuts)
    return cuts


def _apart(cuts, n):
    """Up to n stretches that don't overlap, in the order given."""
    out = []
    for c in cuts:
        if all(abs(c["start"] - o["start"]) >= INTRO_CUT for o in out):
            out.append(c)
        if len(out) == n:
            break
    return out


def plan_intros(entry, n, moods=None):
    """Each of n chapters' intro from one sound: {"start", "lead", "fade_in"}. The first opens with the sound's own
    start; a long sound gives the others a strong stretch (the loudest parts) or a soft one (quieter, not silent),
    alternating (or by moods[i]: "strong" / "soft"), never the same stretch twice in a row. A short sound (a chapter
    intro made for it) is one stretch: its start."""
    cuts = intro_cuts(entry)
    by_energy = sorted(cuts, key=lambda c: -c["energy"])
    strong = _apart(by_energy, 3)
    quiet = sorted(cuts, key=lambda c: c["energy"])
    mid = quiet[len(quiet) // 4: max(len(quiet) // 4 + 1, int(len(quiet) * 0.6))]  # soft: quieter, but not the silence
    soft = _apart(sorted(mid, key=lambda c: -c["energy"]), 3) or strong
    out, last = [], None
    for i in range(n):
        mood = (moods[i] if moods and i < len(moods) else None) or ("strong" if i % 2 else "soft")
        if i == 0 or len(cuts) == 1:
            c = cuts[0]
        else:
            pool = strong if mood == "strong" else soft
            c = next((pool[(i + j) % len(pool)] for j in range(len(pool)) if pool[(i + j) % len(pool)] is not last),
                     pool[0])
        last = c
        lead = c["peak"] + INTRO_AFTER_PEAK
        lead = min(max(lead, INTRO_LEAD_MIN), INTRO_LEAD_MAX, max(0.5, entry["duration"] - c["start"] - 0.5))
        out.append({"start": round(c["start"], 2), "lead": round(lead, 2), "fade_in": 0.0 if c["start"] == 0 else 1.0})
    return out


def outro_of(entry, last=False):
    """The theme's ending as a chapter's outro: {"start", "len"} (seconds into the sound): the stretch before it
    fades into silence, and a little of that fade."""
    end = min(entry["duration"], _sounding_end(_envelope(load(entry))))
    length = min(LAST_OUTRO_SECONDS if last else OUTRO_SECONDS, end)
    start = max(0.0, end - length)
    return {"start": round(float(start), 2), "len": round(float(min(length + 2.0, entry["duration"] - start)), 2)}


def outro_cut(entry, last=False):
    """A chapter's outro from this sound: a short one (a sting, a closing phrase made for it) plays whole; a long
    one plays its ending (outro_of)."""
    if entry["duration"] <= OUTRO_WHOLE:
        return {"start": 0.0, "len": round(float(entry["duration"]), 2)}
    return outro_of(entry, last)


def _intro_gain(cue):
    db = cue.get("gain_db", INTRO_DB)
    db = INTRO_DB if db is None else float(db)
    return 0.0 if db <= OFF_DB else 10 ** ((SPEECH_DB + db - MUSIC_RMS_DB) / 20) * VOLUME.get(cue.get("volume", "normal"), 1.0)


def _intro(a, at, cue, n):
    """A chapter intro: its stretch of the sound (cue start, from the theme) on its own for its lead time, then
    fading out under the first words. Its own level (the book's intro volume); the caller ducks it under speech."""
    lead = int(cue.get("lead", INTRO_LEAD) * SR)
    s0 = int(float(cue.get("start") or 0) * SR)
    seg = a[s0: s0 + lead + int(INTRO_TAIL * SR)].copy() * _intro_gain(cue)
    fi = min(int(max(0.05, float(cue.get("fade_in") or 0)) * SR), len(seg) // 4)
    if fi:
        seg[:fi] *= np.linspace(0, 1, fi, dtype=np.float32)[:, None]
    fo = max(1, len(seg) - lead if len(seg) > lead else len(seg) // 3)  # fade over the part under the first words
    _, fout = _fades(fo)
    seg[len(seg) - fo:] *= fout[:, None]
    track = np.zeros((n, 2), np.float32)
    track[at:at + len(seg)] = seg[: n - at]
    return track


def _outro(a, at, cue, n):
    """A chapter outro: the theme's ending, after a short pause once the last words are done, fading to silence."""
    s0, ln = int(float(cue.get("start") or 0) * SR), int(float(cue.get("len") or OUTRO_SECONDS) * SR)
    seg = a[s0: s0 + ln].copy() * _intro_gain(cue)
    if not len(seg):
        return np.zeros((n, 2), np.float32)
    fi = min(int(0.5 * SR), len(seg) // 4)
    seg[:fi] *= np.linspace(0, 1, fi, dtype=np.float32)[:, None]
    fo = max(1, int(len(seg) * 0.6))
    _, fout = _fades(fo)
    seg[len(seg) - fo:] *= fout[:, None]
    start = at + int(OUTRO_PAUSE * SR)
    track = np.zeros((n, 2), np.float32)
    if start < n:
        track[start:start + len(seg)] = seg[: n - start]
    return track

def _limit(out, knee=0.8, ceiling=0.98):
    """Tame only the loudest moments: everything under the knee is untouched, peaks above it bend smoothly up to the
    ceiling and never past it. (The old curve could reach 1.22, which clipped when saved.)"""
    a = np.abs(out)
    over = a > knee
    if over.any():
        out = out.copy()
        out[over] = np.sign(out[over]) * (knee + (ceiling - knee) * np.tanh((a[over] - knee) / (ceiling - knee)))
    return out


# Text before a tag that ends a sentence or a clause: a tag after it plays between the two parts. A tag inside a
# sentence ("He opened {sfx: door creak} the door.") leaves the sentence whole and starts on the word after it.
_ENDS_PART = re.compile(r"[.!?…:;—–\-][\"”’')\]]*$")
_WORD = re.compile(r"[\w'’]+")


def _norm_word(w):
    return re.sub(r"[^a-z0-9']", "", w.lower().replace("’", "'"))


def cue_items(text, base):
    """Split one render item's text on tags: returns items with text plus cue items ({"cue": {...}}).
    A tag at a sentence's end or start splits the line there. A tag inside a sentence doesn't: the line is spoken
    whole and the sound gets the word after the tag ("at", and "nth" for which time that word comes in the line),
    which the engine places with word timings."""
    parts = split_cues(text)
    if not any(k == "cue" for k, _ in parts):
        return [base]
    out, line = [], None  # line: the text item being built, which inline tags don't break
    for i, (kind, val) in enumerate(parts):
        if kind == "text":
            if line is not None and out and out[-1].get("inline"):
                line["text"] += " " + val
                continue
            line = dict(base, text=val)
            out.append(line)
            continue
        after = parts[i + 1][1] if i + 1 < len(parts) and parts[i + 1][0] == "text" else ""
        first = _WORD.match(after)
        inside = (line is not None and val["type"] in ("sfx", "amb", "music") and not val.get("at") and first
                  and not _ENDS_PART.search(line["text"].rstrip()))
        if inside:
            word = _norm_word(first.group())
            nth = sum(1 for w in _WORD.findall(line["text"]) if _norm_word(w) == word)
            out.append({"cue": dict(val, at=first.group(), nth=nth), "pause": 0, "inline": True})
            continue
        out.append({"cue": val, "pause": 0})
        line = None
    # a line keeps its own pause unless something other than its inline tags comes after it
    for i, it in enumerate(out):
        if "cue" not in it and any(not x.get("inline") for x in out[i + 1:]):
            it["pause"] = min(base.get("pause", 0.3), 0.25)
    for it in out:
        it.pop("inline", None)
    return out


def copy_library_to(dest):
    """For packaging sound packs later."""
    shutil.copytree(LIB, dest, dirs_exist_ok=True)
