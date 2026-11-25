"""Check lines: find the cloud lines that came out wrong (garbled, cut short, said twice) and set them aside, so
the next Create in the cloud makes just those again.

Hearing every line back would take hours on a Mac, so a quick screen comes first: a line much longer (or much
shorter) than its words should take, against that voice's own pace in this book. Only those are heard back with
the speech-to-text model (a few seconds each, on this Mac) and their words compared with the script. Tested on a
5,853-line book (November 2026): 53 lines screened, 24 wrong; a random 150 found nothing the screen had missed.

Lines set aside move to cloud-lines/set-aside/ (kept: Put back returns them), the chapters that use them are
made again, and the next Create in the cloud makes them with another seed under the same name (cloud_book plan,
cloud-lines/retries.json), so Finish uses the new ones. Lines made on this Mac aren't checked (they go straight
into their chapter).
"""
import difflib
import json
import re
import shutil
import statistics
import threading
import time
from pathlib import Path

import soundfile as sf

from . import books
from . import engine as en
from .audio import MIN_SPEECH, is_silent, voiced_seconds

_running, _lock = set(), threading.Lock()  # books being checked now: one check per book at a time
RESULT = "line-check.json"  # in the book's folder: the last check
ASIDE = "set-aside"         # in cloud-lines: lines set aside (to put back)
PACE = 14.6                 # characters a second, when a voice has too few long lines to measure its own
LONG, SHORT = 2.5, 0.35     # x the time its words should take: screened as possibly wrong
MIN_LONG = 3.0              # (a line under 3 s is never "too long": "Mmm." can take a while)
SHORT_LINE = 30             # characters: a line shorter than this is screened by the sound in it instead
SLOW, MIN_SLOW = 3.0, 1.5   # ...as too long: sound for over 3x the time its words take, and over 1.5 s
COVERED, SIMILAR = 0.8, 0.7  # heard back: at least this much of the script's words, and this similar overall
REPEATED = 1.35              # heard this much more than the script (+8 letters): said twice, or babbling


def _words(text):
    """The words of a line as a script or as heard: no directions ([whispering], <laugh>), no punctuation."""
    t = re.sub(r"\[[^\]]*\]|<[^>]*>", " ", (text or "").lower().replace("’", "'"))
    return " ".join(re.sub(r"[^\w' ]", " ", t).replace("_", " ").split())


def judge(script, heard):
    """(wrong, how much of the script was heard 0-1): heard back as something else, cut short or said twice."""
    a, h = _words(script), _words(heard)
    if not a:
        return False, 1.0
    m = difflib.SequenceMatcher(None, a, h, autojunk=False)
    covered = sum(b.size for b in m.get_matching_blocks()) / len(a)
    wrong = covered < COVERED or m.ratio() < SIMILAR or len(h) > len(a) * REPEATED + 8
    return wrong, round(covered, 2)


def _seconds(path):
    try:
        return sf.info(str(path)).duration
    except RuntimeError:
        return 0.0


def book_lines(**kw):
    """(book folder, [line]): every line of the book made in the cloud and back here, once each:
    {"key", "chapters", "text", "voice", "engine", "path", "seconds"}."""
    from . import cloud_book
    where, chapters, gen = books.chapters_to_make(all_chapters=True, **kw)
    cache = where / books.CLOUD_LINES
    opts = {"language": gen["language"], "temperature": gen["temperature"]}
    lines = {}
    for index, _, items in chapters:
        for it in en.spoken(items, gen["sound"]):
            if "cue" in it:
                continue
            spec = cloud_book.line_spec(it, opts, gen["seed"])
            if spec is None:
                continue
            k = cloud_book.key_of(spec)
            if k in lines:
                lines[k]["chapters"] = sorted(set(lines[k]["chapters"]) | {index})
                continue
            path = cache / f"{k}.flac"
            if path.exists():
                lines[k] = {"key": k, "chapters": [index], "text": spec["text"], "voice": it.get("voice") or "",
                            "engine": spec.get("engine", ""), "path": str(path), "seconds": round(_seconds(path), 2)}
    return where, list(lines.values())


def screen(lines):
    """The lines to hear back: much longer or shorter than their words should take at their voice's pace (a short
    line by the sound in it, since its file's length says little)."""
    pace = {}
    for voice in {ln["voice"] for ln in lines}:
        mine = [len(_words(ln["text"])) / ln["seconds"] for ln in lines
                if ln["voice"] == voice and ln["seconds"] > 0 and len(_words(ln["text"])) >= 40]
        pace[voice] = statistics.median(mine) if len(mine) >= 5 else PACE
    out = []
    for ln in lines:
        n = len(_words(ln["text"]))
        should = n / pace[ln["voice"]]
        if ln["seconds"] > max(MIN_LONG, should * LONG) or (n >= SHORT_LINE and ln["seconds"] < should * SHORT):
            out.append(ln)
        elif n < SHORT_LINE and _short_take_off(ln["path"], should):
            out.append(ln)
    return out


def _short_take_off(path, should):
    """Whether a short line's take looks wrong by the sound in it (its file's length says little: a model leaves
    silence around a line): silent, far less sound than its words take, or drawn out far longer."""
    try:
        wav, _ = sf.read(str(path), dtype="float32", always_2d=False)
    except (OSError, RuntimeError):  # (a file that can't be read is made again on this Mac: nothing to hear)
        return False
    wav = wav.mean(axis=1) if wav.ndim > 1 else wav
    if is_silent(wav):
        return True
    sound = voiced_seconds(wav)
    return sound < max(MIN_SPEECH, should * SHORT) or sound > max(MIN_SLOW, should * SLOW)


def read(where):
    try:
        return json.loads((Path(where) / RESULT).read_text())
    except (OSError, ValueError):
        return {}


def _atomic_json(path, data):
    import os as _os
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(data, indent=1, ensure_ascii=False))
            fh.flush()
            _os.fsync(fh.fileno())
        _os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _save(where, result):
    _atomic_json(Path(where) / RESULT, result)


def _stamp(path):
    st = Path(path).stat()
    return f"{st.st_size}-{int(st.st_mtime)}"


def check(**kw):
    """Events while lines are heard back ("Line 3 of 53"), then {"check": the result} (also saved). Lines heard
    before and unchanged since aren't heard again."""
    yield {"status": "Finding the lines to check…"}
    where, lines = book_lines(**kw)
    if not lines:
        raise ValueError("This book has no lines from the cloud to check (lines made on this Mac aren't checked).")
    with _lock:
        if str(where) in _running:
            raise ValueError("This book's lines are already being checked.")
        _running.add(str(where))
    try:
        yield {"status": "Finding the lines that look wrong…", "folder": where.name}  # (so the page can find it)
        yield from _check(where, lines)
    finally:
        _running.discard(str(where))


def _keep_heard(where, heard):
    """What was heard so far, kept with the last check (a check that's stopped or fails carries on from here)."""
    r = read(where)
    r["heard"] = dict(r.get("heard") or {}, **heard)
    _save(where, r)


def _check(where, lines):
    before = read(where).get("heard") or {}  # "key:stamp" -> what was heard
    suspects = screen(lines)
    tags = {ln["key"]: f"{ln['key']}:{_stamp(ln['path'])}" for ln in suspects}
    heard = {tags[ln["key"]]: before[tags[ln["key"]]] for ln in suspects if tags[ln["key"]] in before}
    to_hear = [ln for ln in suspects if tags[ln["key"]] not in heard]
    done = len(suspects) - len(to_hear)
    try:
        # the speech-to-text model is loaded once for them all (not once a line)
        for n, (path, said) in enumerate(en.transcribe_many([ln["path"] for ln in to_hear]), 1):
            heard[tags[to_hear[n - 1]["key"]]] = said
            yield {"status": f"Line {done + n} of {len(suspects)}: hearing back the lines that look wrong"}
            if n % 10 == 0:
                _keep_heard(where, heard)
    finally:
        if len(heard) < len(suspects):  # stopped or failed part way: the next check starts from here
            _keep_heard(where, heard)
    wrong = []
    for ln in suspects:
        said = heard[tags[ln["key"]]]
        bad, covered = judge(ln["text"], said)
        if bad:
            wrong.append(dict(ln, heard=said, covered=covered))
    result = {"checked": time.time(), "lines": len(lines), "screened": len(suspects), "wrong": wrong,
              "heard": heard, "aside": read(where).get("aside") or []}
    _save(where, result)
    n = len(wrong)
    yield {"check": view(where), "status": f"Done · checked {len(lines):,} cloud lines: "
                                           + (f"{n} sound wrong." if n else "none sound wrong.")}


def view(where):
    """The last check, for the app (without what was heard of the lines that were fine)."""
    r = read(where)
    if not r:
        return {}
    cache = Path(where) / books.CLOUD_LINES
    aside = [dict(ln, remade=(cache / f"{ln['key']}.flac").exists()) for ln in r.get("aside") or []]
    return dict({k: r.get(k) for k in ("checked", "lines", "screened", "wrong")}, aside=aside)


def _chapter_files(where, chapters):
    for i in sorted(set(chapters)):
        for f in Path(where).glob(f"{i:03d}-*.wav"):
            f.unlink(missing_ok=True)


def set_aside(where, keys):
    """Move these lines out of the way (to cloud-lines/set-aside/), so Create in the cloud makes them again with
    another seed; the chapters that use them are made again too. Returns the view."""
    from . import cloud_book
    where = Path(where)
    cache = where / books.CLOUD_LINES
    r = read(where)
    if not r:
        raise ValueError("Check the lines first.")
    wanted = set(keys)
    chosen = [ln for ln in r.get("wrong") or [] if ln["key"] in wanted]
    if not chosen:
        raise ValueError("None of those lines are in the last check.")
    aside = cache / ASIDE
    aside.mkdir(parents=True, exist_ok=True)
    retries = cloud_book._retries(cache)
    chapters = []
    for ln in chosen:
        src = cache / f"{ln['key']}.flac"
        if src.exists():
            shutil.move(str(src), str(aside / src.name))
        retries[ln["key"]] = retries.get(ln["key"], 0) + 1
        chapters += ln["chapters"]
    _atomic_json(cache / cloud_book.RETRIES, retries)
    _chapter_files(where, chapters)
    r["wrong"] = [ln for ln in r["wrong"] if ln["key"] not in wanted]
    r["aside"] = (r.get("aside") or []) + [dict(ln, set_aside=time.time()) for ln in chosen]
    _save(where, r)
    return view(where)


def put_back(where):
    """Return the lines set aside that haven't been made again yet (the old take is used). Returns the view."""
    from . import cloud_book
    where = Path(where)
    cache = where / books.CLOUD_LINES
    r = read(where)
    retries = cloud_book._retries(cache)
    back, chapters = [], []
    for ln in r.get("aside") or []:
        old, now = cache / ASIDE / f"{ln['key']}.flac", cache / f"{ln['key']}.flac"
        if old.exists() and not now.exists():
            shutil.move(str(old), str(now))
            retries[ln["key"]] = max(0, retries.get(ln["key"], 1) - 1)
            back.append(ln["key"])
            chapters += ln["chapters"]
    _atomic_json(cache / cloud_book.RETRIES, {k: v for k, v in retries.items() if v})
    _chapter_files(where, chapters)  # (made without them: made again with them)
    if r:
        r["aside"] = [ln for ln in r.get("aside") or [] if ln["key"] not in back]
        _save(where, r)
    return dict(view(where), put_back=len(back))
