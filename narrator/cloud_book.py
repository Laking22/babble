"""Create in the cloud: a book's lines are spoken on a cloud GPU (Kaggle, with the user's own account), then
this Mac puts the book together as usual (sound effects, music, pauses, chapters, M4B).

The cloud only speaks. Each line it can make (Qwen's speakers, saved voices and trained Qwen voice packs, stock and
trained Orpheus) is described by a spec: the voice, the words as that model reads them and the sampling
settings. The spec's hash names the line's file. render() looks every line up by the same spec in the book's
cloud-lines folder before making it, so a line that came back from the cloud is never made again, and anything
else (Kokoro, other models, lines changed since) is made here.

One job per book, kept in <book folder>/cloud.json:
  {"backend", "run", "url", "sent", "lines", "local", "seconds", "budget", "state", "checked", "made", "error"}
state: "running" → "downloading" → "ready" (lines are in cloud-lines), or "failed".
"""
import errno
import functools
import os
import subprocess
import sys
import hashlib
import json
import math
import re
import shutil
import tarfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from . import books
from . import engine as en
from . import voices_lib as lib
from .runtime import SR

RUNNER_DIR = Path(__file__).resolve().parent / "kaggle_render"
RECORD = "cloud.json"
WORDS_PER_SECOND = 2.6  # narration pace, for the estimate
WAITING = ("sent", "queued", "running")
_lock = threading.Lock()
_downloads = set()
_collecting = {}  # book folder -> its lock: one copy of its lines at a time (while it works, or the last one)
KEYS = "sent-keys.json"  # in <book>/cloud: the lines the job out there makes
REQUEST = "request.json"  # in <book>/cloud: what the book was sent with (to put it together as lines come back)
_sending = set()  # book folders being sent right now: a second send of the same book is refused
_stitchers = {}  # book folder -> (the task putting its chapters together as their lines come back, the run it's for)
_stitch_lock = threading.Lock()  # (the watcher and a send can start one at the same moment)
FINISH_MISSING = 20  # a run back with at most this many lines missing is still finished by itself (they're made here)
LINE = re.compile(r"[0-9a-f]{20}\.flac")


# ---------- lines ----------

@functools.lru_cache(maxsize=64)
def _trained_source(folder):
    """The Kaggle notebook a fine-tuned Orpheus model was trained in, and its merged model's folder name there."""
    try:
        meta = json.loads((Path(folder) / "voice.json").read_text())
    except (OSError, ValueError):
        return None
    # newer models say "trained_on": "kaggle" and "url"; the first ones only "kaggle": <notebook url>
    url = meta.get("url") if meta.get("trained_on") == "kaggle" else meta.get("kaggle")
    m = re.search(r"kaggle\.com/code/([\w-]+/[\w-]+)", url or "")
    return m.group(1) if m else None


def _sample_stamp(name):
    p = lib.wav_path(name)
    if not p.exists():
        return None
    st = p.stat()
    return f"{st.st_size}-{int(st.st_mtime)}"


def line_spec(item, opts, seed=0):
    """What the cloud is asked to make for one spoken line, or None when only this Mac can make it."""
    from .tts import orpheus as o
    from .tts.base import line_temperature
    kind, _, key = (item.get("voice") or "").partition(":")
    text = item.get("text") or ""
    temp = item.get("temperature", opts.get("temperature"))
    common = {"language": opts.get("language", "english"), "seed": int(seed or 0)}
    if kind == "builtin":
        return {"engine": "qwen", "speaker": key, "text": o.TAGS.sub("", text).strip(),
                "instruct": item.get("instruct") or "", "temperature": temp, **common}
    if kind == "my":
        stamp = _sample_stamp(key)
        if stamp is None:
            return None
        return {"engine": "qwen-clone", "voice": key, "sample": stamp, "text": o.TAGS.sub("", text).strip(),
                "instruct": item.get("instruct") or "", "temperature": temp, **common}
    if kind == "qwen":  # a speaker of a trained Qwen voice pack: the pack goes with the job (or is on Modal)
        from . import tts
        from .modal_models import _stamp, cloud_name
        ft = tts.ENGINES["qwen"].tuned.get(key)
        if not ft:
            return None
        return {"engine": "qwen-tuned", "model": {"kernel": None, "dir": cloud_name(ft[3]), "stamp": _stamp(ft[3])},
                "speaker": ft[4], "text": o.TAGS.sub("", text).strip(), "temperature": temp, "seed": int(seed or 0)}
    if kind == "orpheus":
        if key in o.STOCK_VOICES:
            model, speaker = None, key
        else:
            from . import tts
            ft = tts.ENGINES["orpheus"].tuned.get(key)
            if not ft:
                return None
            from .modal_models import cloud_name
            # kernel: the Kaggle notebook it was trained in (None for one trained on this Mac); dir: its name in
            # the cloud (Kaggle's training output, or uploaded to Modal)
            model, speaker = {"kernel": _trained_source(ft[3]), "dir": cloud_name(ft[3])}, ft[4]
        return {"engine": "orpheus", "model": model, "speaker": speaker, "text": o._text(text, item.get("instruct")),
                "temperature": line_temperature(item, opts, 0.6), "seed": int(seed or 0)}
    if kind == "drama":  # Drama 3: a saved voice cloned from its clip and transcript; the emotion as a direction
        stamp, said = _sample_stamp(key), lib.info(key).get("transcript", "") if lib.wav_path(key).exists() else ""
        if stamp is None or not said:
            return None
        how = " ".join((item.get("instruct") or "").replace("[", "").replace("]", "").split()).rstrip(".")
        words = o.TAGS.sub("", text).strip()
        return {"engine": "drama", "voice": key, "sample": stamp,
                "transcript": hashlib.sha1(said.encode()).hexdigest()[:12],
                "text": f"[{how}] {words}" if how else words}
    if kind == "higgs":  # a saved voice cloned by Higgs Audio from its clip and the clip's transcript
        stamp, said = _sample_stamp(key), lib.info(key).get("transcript", "") if lib.wav_path(key).exists() else ""
        if stamp is None or not said:
            return None
        return {"engine": "higgs", "voice": key, "sample": stamp,
                "transcript": hashlib.sha1(said.encode()).hexdigest()[:12], "text": o.TAGS.sub("", text).strip(),
                "temperature": line_temperature(item, opts, 0.7), "seed": int(seed or 0)}
    return None


def key_of(spec):
    return hashlib.sha1(json.dumps(spec, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]


def local_spec(item, opts, seed=0):
    """A line's label among the lines kept on this Mac (books.LINES): the cloud's label, else one for the voices only
    this Mac makes; with the model's own file time for fine-tuned voices, so a voice trained again is spoken again.
    None: the line isn't kept (its voice can't be told apart from a later one of the same name)."""
    from .tts import orpheus as o
    spec = line_spec(item, opts, seed)
    kind, _, key = (item.get("voice") or "").partition(":")
    if spec is not None:
        if kind == "orpheus" and spec.get("model"):
            from . import tts
            ft = tts.ENGINES["orpheus"].tuned.get(key)
            stamp = _dir_stamp(ft[3]) if ft else None
            return dict(spec, local=stamp) if stamp else None
        return spec
    base = {"engine": kind, "voice": key, "text": o.TAGS.sub("", item.get("text") or "").strip(),
            "instruct": item.get("instruct") or "", "temperature": item.get("temperature", opts.get("temperature")),
            "language": opts.get("language", "english"), "seed": int(seed or 0)}
    if kind == "kokoro":
        return base
    if kind == "vibevoice":
        stamp = _sample_stamp(key)
        return dict(base, sample=stamp) if stamp else None
    if kind == "qwen":
        from . import tts
        ft = tts.ENGINES["qwen"].tuned.get(key)
        stamp = _dir_stamp(ft[3]) if ft else None
        return dict(base, model=stamp) if stamp else None
    return None


def _dir_stamp(folder):
    """When a model folder's files last changed (training it again changes it)."""
    try:
        return max(int(f.stat().st_mtime) for f in Path(folder).iterdir() if f.is_file())
    except (OSError, ValueError):
        return None


def kept_line(folder, item, opts, seed=0):
    """A line made before on this Mac with the same words, voice and settings (books.LINES), or None. Reusing it
    marks it used, so lines no book needs any more can be cleared out later (books.prune_lines)."""
    spec = local_spec(item, opts, seed)
    if spec is None:
        return None
    f = Path(folder) / f"{key_of(spec)}.flac"
    wav = _read_line(f)
    if wav is not None:
        try:
            os.utime(f)
        except OSError:
            pass
    return wav


def keep_line(folder, item, opts, seed, wav):
    """Keep a line made on this Mac, so a chapter whose sounds change is mixed again without speaking it again."""
    spec = local_spec(item, opts, seed)
    if spec is None or wav is None or not len(wav):
        return
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    f = folder / f"{key_of(spec)}.flac"
    tmp = f.with_suffix(".tmp.flac")
    try:
        sf.write(str(tmp), np.asarray(wav, dtype=np.float32), SR, format="FLAC")
        os.replace(tmp, f)
    except (OSError, RuntimeError) as e:
        print(f"[lines] couldn't keep a line: {e}", flush=True)
        tmp.unlink(missing_ok=True)


def premade_line(folder, item, opts, seed=0):
    """A line's audio from the cloud, if it was made there; else None."""
    spec = line_spec(item, opts, seed)
    if spec is None:
        return None
    return _read_line(Path(folder) / f"{key_of(spec)}.flac")


def _read_line(f):
    f = Path(f)
    if not f.exists():
        return None
    try:
        wav, sr = sf.read(f, dtype="float32", always_2d=False)
    except (OSError, RuntimeError):
        return None
    wav = wav.mean(axis=1) if wav.ndim > 1 else wav
    if sr != SR:
        from .audio import resample
        wav = resample(wav, sr, SR)
    return np.asarray(wav, dtype=np.float32) if len(wav) else None


# ---------- the job record ----------

def read(where):
    try:
        return json.loads((Path(where) / RECORD).read_text())
    except (OSError, ValueError):
        return None


def _write(where, rec):
    """Write the record whole (under _lock): to another name first, then renamed over it. Read without the lock
    (the stitcher, the watcher), a record written in place can be found half written, and read as no record at all:
    the stitcher then stopped as if the run were let go. (A crash part way can't leave a broken one either.)"""
    f = Path(where) / RECORD
    tmp = f.with_name(f".{RECORD}.part")
    try:
        tmp.write_text(json.dumps(rec, indent=1))
        os.replace(tmp, f)
    finally:
        tmp.unlink(missing_ok=True)


def _save(where, rec):
    with _lock:
        _write(where, rec)


def _update(where, **changes):
    """Change some of the record's fields (re-read under the lock, so two threads don't undo each other's
    changes). Nothing is written when the record is gone (the job was forgotten meanwhile)."""
    with _lock:
        rec = read(where)
        if rec is None:
            return None
        rec.update(changes)
        _write_or_note(where, rec)
    return rec


DISK_FULL = {}  # book folder -> when its record last couldn't be saved because this Mac's disk was full


def _write_or_note(where, rec):
    """_write(), except that a full disk doesn't stop whatever asked (the book page asking how the run is going,
    the watcher): the record keeps its last saved version, the change is handed back as if saved, and the book
    page is told the disk is full (view()). It's saved again with the next change once there's room."""
    try:
        _write(where, rec)
    except OSError as e:
        if e.errno != errno.ENOSPC:
            raise
        if str(where) not in DISK_FULL:
            print(f"[cloud] this Mac's disk is full: {Path(where).name}'s record can't be saved until there's room",
                  flush=True)
        DISK_FULL[str(where)] = time.time()
        return
    DISK_FULL.pop(str(where), None)


def _update_run(where, run, **changes):
    """_update(), only while the record is still that run's (a run sent again since has a record of its own)."""
    with _lock:
        rec = read(where)
        if rec is None or rec.get("run") != run:
            return None
        rec.update(changes)
        _write_or_note(where, rec)
    return rec


def _noted(rec, state):
    """On the Activity page: the run's GPU time and cost."""
    try:
        from . import activity
        activity.cloud_ended(rec, state)
    except Exception as e:  # noqa: BLE001 - never stops the run's own work
        print(f"[activity] couldn't note the cloud run: {e}", flush=True)


LAST_COPY_SECONDS = 90  # before a run is stopped, this long at most to bring back the lines it made since the last copy


def forget(where):
    """Stop waiting for the job. One still working out there is stopped first (a rented machine deleted, a Modal call
    cancelled), after the lines it made since the last copy are brought back (they're paid for). Putting its
    chapters together here stops too (chapters already made stay). Returns {"stopped": True when the run out there
    was stopped, False when the service couldn't be told (Kaggle: it carries on to its time limit), None when
    nothing was running; "back", "lines": the run's lines on this Mac, and all it was sent}."""
    t = stitching(where)
    if t is not None:
        t.stop_requested.set()
    rec = read(where) or {}
    stopped = None
    if rec.get("state") in WAITING and rec.get("run"):
        _last_copy(where, rec)
        from . import spend
        try:  # a machine is deleted, a Modal call cancelled (before, Modal kept going to its time limit)
            stopped = spend.stop_remote(rec.get("backend", "kaggle"), rec["run"])
        except Exception as e:  # noqa: BLE001 - it deletes itself later anyway
            stopped = False
            print(f"[cloud] couldn't stop {rec.get('run')}: {e}", flush=True)
    if rec.get("state") in WAITING + ("downloading",):
        _noted(rec, "stopped")
    elif rec.get("run"):
        from . import spend
        spend.job_ended(rec["run"], charge=False)
    with _lock:  # (not between another thread's read and write of it: that would write it back)
        (Path(where) / RECORD).unlink(missing_ok=True)
    return {"stopped": stopped, "back": _back(where) or 0, "lines": rec.get("lines") or 0}


def _last_copy(where, rec, wait=None):
    """Bring back the lines a run made since the last copy, before it's stopped (a rented machine's go with it).
    Only for runs whose lines come back as they're made (live); never longer than `wait` (LAST_COPY_SECONDS), so
    Stop isn't held up by a connection that's down (the copy then carries on by itself in the background)."""
    from .training import BACKENDS
    wait = LAST_COPY_SECONDS if wait is None else wait
    if not rec.get("live"):
        return
    b = BACKENDS.get(rec.get("backend", "kaggle"))
    if b is None or not getattr(b, "live", False):
        return
    lock = _collecting.setdefault(str(where), threading.Lock())
    done = threading.Event()

    def copy():
        try:
            if not lock.acquire(timeout=wait):  # (a copy already going: it brings them)
                return
            try:
                n = _collect(where, rec, b)
                if n:
                    print(f"[cloud] {n} more line(s) back from the cloud for {where.name} before it was stopped",
                          flush=True)
            finally:
                lock.release()
        except Exception as e:  # noqa: BLE001 - what's back is kept; the rest goes with the run
            print(f"[cloud] couldn't bring the last lines back for {where.name}: {e}", flush=True)
        finally:
            done.set()
    threading.Thread(target=copy, daemon=True, name="cloud-last-copy").start()
    done.wait(wait)


def start_again(where):
    """After a run is stopped, to make its lines again rather than keep them: the lines it made, and the chapters
    put together from them, go to the Trash (as Remake does with a whole take), and they're made with another seed
    next time (retries.json), so a book with a fixed seed gets new takes too. Returns {"moved": lines, "chapters"}."""
    where = Path(where)
    rec = read(where)
    if rec and rec.get("state") in WAITING + ("downloading",):
        raise ValueError("This book is still being made in the cloud. Stop it first.")
    t = stitching(where)
    if t is not None:  # (told to stop when the run was: let it finish saving what it was on)
        t.stop_requested.set()
        end = time.time() + 15
        while stitching(where) is not None and time.time() < end:
            time.sleep(0.2)
    try:
        keys = json.loads((where / "cloud" / KEYS).read_text())
    except (OSError, ValueError):
        raise ValueError("There's no cloud run to start again for this book.")
    return _trash_lines(where, keys, "lines of a stopped cloud run")


def _trash_lines(where, keys, what):
    """Move these cloud lines (those that are here), and the chapters put together from them, to the Trash (a
    folder "<book> (<what> <time>)"), and have them made with another seed next time (retries.json).
    Returns {"moved": lines, "chapters"}."""
    where = Path(where)
    cache = where / books.CLOUD_LINES
    back = [k for k in keys if (cache / f"{k}.flac").exists()]
    chapters = set()
    try:  # the chapters that use them, from what the run was sent with
        from . import line_check
        kw = json.loads((where / "cloud" / REQUEST).read_text())
        mine = set(back)
        chapters = {i for ln in line_check.book_lines(**kw)[1] if ln["key"] in mine for i in ln["chapters"]}
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"[cloud] couldn't tell which chapters used the lines ({e}): all of them are made again", flush=True)
        chapters = None
    files = [f for f in where.iterdir() if books.CHAPTER_FILE.fullmatch(f.name)
             and (chapters is None or int(f.name[:3]) in chapters)]
    if back or files:
        stamp = time.strftime("%Y-%m-%d %H.%M.%S")
        bin_ = Path.home() / ".Trash" / f"{where.name} ({what} {stamp})"
        try:
            (bin_ / books.CLOUD_LINES).mkdir(parents=True)
        except OSError:  # no Trash to put it in: kept in the book's folder instead (an earlier one replaced)
            bin_ = where / what
            shutil.rmtree(bin_, ignore_errors=True)
            (bin_ / books.CLOUD_LINES).mkdir(parents=True)
        for k in back:
            shutil.move(str(cache / f"{k}.flac"), str(bin_ / books.CLOUD_LINES / f"{k}.flac"))
        for f in files:
            shutil.move(str(f), str(bin_ / f.name))
        retries = _retries(cache)
        for k in back:
            retries[k] = retries.get(k, 0) + 1
        if cache.exists():
            (cache / RETRIES).write_text(json.dumps(retries, indent=1))
    return {"moved": len(back), "chapters": len(files)}


def _wrong_voice(where, wrong):
    """The cloud made a voice wrong (voice_check): stop the run at once, so nothing more is made or charged, and
    keep that voice's lines out of the book (to the Trash: Finish and Create in the cloud then make them again)."""
    from . import voice_check
    where = Path(where)
    rec = read(where) or {}
    run = rec.get("run")
    t = stitching(where)
    if t is not None:
        t.stop_requested.set()
    stopped = None
    if rec.get("state") in WAITING and run:
        _last_copy(where, rec)
        from . import spend
        try:
            stopped = spend.stop_remote(rec.get("backend", "kaggle"), run)
        except Exception as e:  # noqa: BLE001 - it ends at its time limit anyway
            stopped = False
            print(f"[cloud] couldn't stop {run}: {e}", flush=True)
    names, moved = _trash_wrong(where, wrong)
    error = (f"Stopped: the cloud made {names}'s lines in a different voice ({wrong[0][1]}). Babble checks the first "
             f"lines of each voice against that voice on this Mac, and stopped the run"
             + (" so nothing more is made or charged" if stopped else
                "; the service couldn't be told, so it runs on to its time limit: stop it on its site" if stopped is False
                else "")
             + f". Its {moved['moved']:,} lines in that voice are in the Trash, kept out of the book.")
    _update_run(where, run, state="failed", error=error, stitch_end="failed", note="")
    _noted(rec, "failed")
    print(f"[cloud] {where.name}: wrong voice ({names}); run stopped: {stopped}; {moved['moved']} lines to the Trash",
          flush=True)


def _trash_wrong(where, wrong):
    """The lines of the voices the check found wrong, and the chapters made from them, to the Trash: (their names,
    {"moved", "chapters"})."""
    from . import voice_check
    data = voice_check.read(where)
    bad = {vid for vid, v in data.get("voices", {}).items() if v.get("state") == "wrong"}
    try:
        jobs = json.loads((Path(where) / "cloud" / "job" / "data" / "lines.json").read_text())
        keys = [j["key"] for j in jobs if voice_check.voice_id(j) in bad]
    except (OSError, ValueError):
        keys = json.loads((Path(where) / "cloud" / KEYS).read_text())
    return " and ".join(sorted({label for label, _ in wrong})), _trash_lines(where, keys, "lines in the wrong voice")


def _check_voice(where, final=False):
    """Check the voice of the cloud lines that are back (voice_check); a wrong one stops the run. In the
    background (it uses the Mac's speech model: after whatever it's doing)."""
    from . import voice_check
    try:
        wrong = voice_check.check(where, final=final)
        if wrong:
            _wrong_voice(where, wrong)
        return wrong
    except Exception as e:  # noqa: BLE001 - a check that can't be made doesn't stop a run
        print(f"[voice check] {Path(where).name}: {e}", flush=True)
        return []


def done_with(where):
    """The book was made with the lines that came back: the job is over (Finish stops showing). A job still
    out in the cloud is left alone."""
    rec = read(where) or {}
    if rec.get("state") == "ready":
        with _lock:  # (as forget: not between another thread's read and write of it)
            (Path(where) / RECORD).unlink(missing_ok=True)


_watching = []
_awake = []  # the caffeinate process keeping this Mac awake while a book is out in the cloud
_busy = set()  # other cloud jobs being followed right now (cloud_ai: Detect speakers)


def stay_awake():
    """While any book is out in the cloud, keep this Mac from sleeping (caffeinate, for as long as the engine runs),
    so the app is there to bring the lines back and free the GPU. A rented machine left uncollected deletes itself,
    lines and all, 90 minutes after it's done. Closing the lid or shutting down still stops it. Returns whether it's on."""
    out = bool(_busy) or any((read(f.parent) or {}).get("state") in WAITING + ("downloading",) for f in books.BOOKS.glob(f"*/{RECORD}"))
    p = _awake[0] if _awake else None
    if out and (p is None or (p != "win32" and p.poll() is not None)):
        if sys.platform == "darwin" and shutil.which("caffeinate"):
            # -i: no idle sleep; -s: no system sleep on mains power; -w: ends with the engine
            _awake[:] = [subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(os.getpid())])]
            print("[cloud] keeping this Mac awake until the cloud lines are back", flush=True)
        elif sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
                _awake[:] = ["win32"]
                print("[cloud] keeping this PC awake until the cloud lines are back", flush=True)
            except Exception:  # noqa: BLE001
                pass
    elif not out and p is not None:
        if p == "win32":
            try:
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)  # ES_CONTINUOUS
            except Exception:  # noqa: BLE001
                pass
        elif p.poll() is None:
            p.terminate()
        _awake.clear()
        print("[cloud] nothing in the cloud: this computer may sleep again", flush=True)
    return bool(_awake) and (_awake[0] == "win32" or _awake[0].poll() is None)


def watch(every=120):
    """Follow every book's cloud job in the background (every two minutes): ask how it's doing, and bring the
    lines back when it's done, even when no book page is open. For Runpod that's also what deletes the machine
    promptly, so it doesn't keep costing money."""
    if _watching:
        return

    def loop():
        while True:
            try:
                stay_awake()
            except Exception as e:  # noqa: BLE001
                print(f"[cloud] keep awake: {e}", flush=True)
            time.sleep(every)
            try:
                for f in books.BOOKS.glob(f"*/{RECORD}"):
                    rec = read(f.parent) or {}
                    if rec.get("state") in WAITING + ("downloading",):
                        check(f.parent)
                    # (after a restart it carries on; stopped, failed or waiting for Finish: not by itself)
                    if (rec.get("state") in WAITING + ("downloading", "ready") and not rec.get("stitch_end")
                            and not stitching(f.parent)):
                        start_stitch(f.parent)
                strays()
            except Exception as e:  # noqa: BLE001 - try again next time
                print(f"[cloud] watcher: {e}", flush=True)
    t = threading.Thread(target=loop, daemon=True, name="cloud-watch")
    t.start()
    _watching.append(t)
    # (and every few seconds, for a book waiting on a lost connection: carried on from as soon as it's back)
    threading.Thread(target=_reconnect_loop, daemon=True, name="cloud-reconnect").start()


def strays(min_age=600):
    """Book runs written down as out in the cloud that no book follows any more (a second send that replaced the
    first, or one stopped on the service's own site): asked how they are, and noted as over (Activity, Spending)
    once the service says they've ended. Still going: left, and asked again next time."""
    from . import spend
    from .training import BACKENDS
    followed = {(read(f.parent) or {}).get("run") for f in books.BOOKS.glob(f"*/{RECORD}")}
    for run, j in list(spend.jobs_running().items()):
        if (run in followed or not str(j.get("what", "")).startswith("Create in the cloud")
                or time.time() - (j.get("sent") or 0) < min_age):  # (one being sent isn't followed yet)
            continue
        b = BACKENDS.get(j.get("backend", ""))
        try:
            st = b.status(run) if b is not None else None
        except Exception as e:  # noqa: BLE001 - asked again next time
            print(f"[cloud] couldn't ask about {run}: {e}", flush=True)
            continue
        if st is not None and st.state in ("queued", "running"):
            spend.job_seen(run)
        elif st is not None and st.state in ("complete", "error", "cancelled"):
            print(f"[cloud] {run} isn't followed by any book and has ended ({st.state}): noted", flush=True)
            _noted({"run": run, "backend": j.get("backend"), "sent": j.get("sent"), "price": j.get("price"),
                    "finished_at": ended_about(j.get("sent"), j.get("seen"), j.get("expect")),
                    "max_cost": round(j["price"] * j["hours"], 2) if j.get("price") and j.get("hours") else None},
                   "stopped")


SEEN_GAP = 600  # longer than this between two looks at a run: Babble was closed (or the Mac asleep) meanwhile


def ended_about(sent, seen, expect, now=None):
    """When a run found to be over most likely ended (what its GPU time is charged up to). Looked at all along:
    about now. Not (it ended while Babble was closed): when it was thought to end, but not before it was last
    seen going, nor after now. (Not known how long it was thought to take: now, as before.)"""
    now = now or time.time()
    last = seen or sent
    if not (sent and last and expect) or now - last <= SEEN_GAP:
        return now
    return min(now, max(last, sent + expect * 3600))


# ---------- sending ----------

def _sends_models(backend):
    """Whether the Mac's own copy of a trained voice goes with the job (the pay-as-you-go services, mostly)."""
    from .training import BACKENDS
    return bool(getattr(BACKENDS.get(backend), "sends_models", False))


def _can_make(backend, model, on_modal, engine=""):
    """Whether the cloud can make a voice: a stock one always; a trained one wherever its model can get to: Kaggle
    attaches the notebook it was trained in, or the Mac's copy goes with the job; Modal has it uploaded (send()
    uploads it first when it isn't); the pay-as-you-go services take the Mac's copy with each job (see
    _pack_trained). Higgs needs a 24 GB GPU: everywhere but Kaggle (a 16 GB T4)."""
    if engine == "higgs":
        return backend != "kaggle"
    if engine == "drama":  # an H100 on Modal: the only place it's been tested
        return backend == "modal"
    if not model or backend in ("kaggle", "modal"):
        return True
    return _sends_models(backend)


def _trained_folders(jobs):
    """{cloud name: the Mac's folder} of the trained models a job's lines use."""
    from .modal_models import cloud_name
    from .paths import MODELS
    names = {s["model"]["dir"] for s in jobs.values() if s.get("model")}
    return {cloud_name(m): m for m in MODELS.glob("*") if cloud_name(m) in names and (m / "config.json").exists()}


def _to_send(backend, jobs):
    """{cloud name: Mac folder} of the trained models that have to be uploaded for this job: on Kaggle the ones not
    trained there (the rest come from their training notebook), on Modal the ones not uploaded yet (or changed
    since), and every one on the pay-as-you-go services."""
    folders = _trained_folders(jobs)
    if backend == "kaggle":
        attached = {s["model"]["dir"] for s in jobs.values() if (s.get("model") or {}).get("kernel")}
        return {n: f for n, f in folders.items() if n not in attached}
    if backend == "modal":
        from .modal_models import uploaded
        on = uploaded()
        return {n: f for n, f in folders.items() if n not in on}
    return folders if _sends_models(backend) else {}


def _upload_gb(folders, backend):
    """How much a job uploads for its trained models (Modal: the model's own files, once)."""
    files = [f for m in folders.values() for f in Path(m).iterdir()
             if f.is_file() and f.suffix in (".json", ".safetensors", ".jinja") + ((".txt",) if backend == "modal" else ())]
    return sum(f.stat().st_size for f in files) / 1e9


# ---------- the book's missing sounds, made in the cloud too ("Make sounds in the cloud") ----------

# GPU time for sounds (sound_worker.py): installing MLX for CUDA and downloading the generator once per run, then
# each take (the script loads its models for every take, which is most of the time for short effects). Measured on
# Kaggle's T4: install 9 s, first take with the download 42 s, effects and backgrounds 5-6 s a take,
# 60 s of Small Music 7.5 s. Medium isn't measured yet (a guess, with room)
SOUND_SETUP_HOURS = 0.02
SOUND_MEDIUM_SETUP_HOURS = 0.03  # (the Medium music model: 4.6 GB more)
SOUND_TAKE_SECONDS = {"sm-sfx": 7.0, "sm-music": 10.0, "medium": 30.0}
SOUND_CODE = Path(__file__).resolve().parent / "vendor" / "stable_audio_3"


def sound_jobs(story="", sound=True, automake=True, cloud_sounds=True, book=None, **_):
    """The book's tagged sounds that no library has and AI has to make: [{"key", "name", "prompt", "kind",
    "seconds", "model", "takes"}]. None go to the cloud unless the book plays its sounds and Make sounds in the cloud
    is on. (The sound libraries' recorded sounds are looked up on the Mac first, see make_missing.)"""
    from . import sfx, sfxgen
    if not (sound and automake and cloud_sounds) or not sfxgen.available():
        return []
    token = sfx.BOOK.set(book) if book else None
    try:
        missing = sfx.missing(story)
    finally:
        books._reset(sfx.BOOK, token)
    out = []
    for kind, name in missing:
        model = sfxgen.cloud_model(kind)
        seconds = min(sfxgen.length_for(name, kind), sfxgen.MAX_SECONDS[model])
        prompt = sfxgen.prompt_for(name, kind)
        key = hashlib.sha1(json.dumps([prompt, model, seconds]).encode()).hexdigest()[:16]
        out.append({"key": key, "name": name, "prompt": prompt, "kind": kind, "seconds": seconds, "model": model,
                    "takes": sfxgen.CLOUD_TAKES})
    return out


def sound_hours(sounds):
    """The GPU time a job's sounds take (on top of its lines)."""
    if not sounds:
        return 0.0
    takes = sum(SOUND_TAKE_SECONDS[s["model"]] * s["takes"] for s in sounds) / 3600
    medium = any(s["model"] == "medium" for s in sounds)
    return SOUND_SETUP_HOURS + (SOUND_MEDIUM_SETUP_HOURS if medium else 0.0) + takes


_plan_tls = threading.local()  # per-thread last plan(): chapters still to make, {voice model: lines only the Mac makes}


def _last_plan():
    d = getattr(_plan_tls, "data", None)
    if d is None:
        d = {"chapters": 0, "mac": {}}
        _plan_tls.data = d
    return d


def nothing_to_make(local):
    """Why the cloud has nothing to do, after plan(): every chapter is made, or their voices only run on the Mac."""
    lp = _last_plan()
    if not lp["chapters"]:
        return "Every chapter is made already."
    mac = lp["mac"]
    names = " and ".join(sorted(mac)) or "Their"
    return (f"{names} voices can't be made in the cloud: Create makes {'its' if local == 1 else 'their'} "
            f"{local:,} line{'s' if local != 1 else ''} on this Mac.")


def plan(backend="kaggle", **kw):
    """(book folder, {key: spec}, lines only this Mac makes, seconds of speech) for the chapters still to make.
    Trained voices go to Kaggle when they were trained there, and to Modal once they've been uploaded to it
    (Connections → Modal); otherwise they're made on this Mac."""
    where, todo, gen = books.chapters_to_make(**kw)
    lp = _last_plan()
    if not todo:
        lp.update(chapters=0, mac={})
        return where, {}, 0, 0.0
    cache = where / books.CLOUD_LINES
    opts = {"language": gen["language"], "temperature": gen["temperature"]}
    jobs, local, words = {}, 0, 0
    retries = _retries(cache)  # lines Check lines set aside: made again with another seed, under the same name
    cloud_only = set()  # voices whose lines neither this cloud nor this Mac can make (Higgs on Kaggle, on 8 GB)
    on_modal = set()
    lp.update(chapters=len(todo), mac={})
    for chapter, _, items in todo:
        for it in en.spoken(items, gen["sound"]):
            if "cue" in it:
                continue
            spec = line_spec(it, opts, gen["seed"])
            k = key_of(spec) if spec is not None else None
            if k and (cache / f"{k}.flac").exists():  # made in the cloud before
                continue
            if spec is None or not _can_make(backend, spec.get("model"), on_modal, spec.get("engine", "")):
                local += 1
                who = _engine_name(it.get("voice"))
                lp["mac"][who] = lp["mac"].get(who, 0) + 1
                if not _runs_here(it.get("voice")):
                    cloud_only.add(_engine_name(it.get("voice")))
                continue
            if k in jobs:
                continue
            if retries.get(k):
                spec = dict(spec, seed=int(spec.get("seed") or 0) + RETRY_SEED * int(retries[k]))
            # (the cloud makes the lines chapter by chapter, so chapters can be put together as they come back;
            # "_voice": the voice in the app, for the voice check: not sent)
            jobs[k] = dict(spec, chapter=chapter, _voice=it.get("voice") or "", _instruct=it.get("instruct") or "")
            words += len(spec["text"].split())
    if cloud_only:
        names = " and ".join(sorted(cloud_only))
        where = "Modal" if "Drama 3" in cloud_only else "Modal or a pay-as-you-go GPU"
        raise ValueError(f"{names} voices can't be made here: this Mac can't run them, and "
                         + ("Kaggle's GPU is too small for them" if backend == "kaggle" else "this cloud doesn't make them")
                         + f". Pick {where} in the ≡ menu next to Create.")
    if jobs:  # each voice's first lines, made before the rest and checked against the voice on this Mac
        from . import voice_check
        voice_check.pick(jobs)
    return where, jobs, local, words / WORDS_PER_SECOND


RETRIES = "retries.json"  # in cloud-lines: {line key: times Check lines set it aside}
RETRY_SEED = 7919  # a set-aside line is made again with its seed + this x the times it was set aside


def _retries(cache):
    try:
        return {k: int(v) for k, v in json.loads((Path(cache) / RETRIES).read_text()).items()}
    except (OSError, ValueError, AttributeError):
        return {}


def _runs_here(voice):
    """Whether this Mac can make a voice's lines itself (Higgs can't on a Mac with under 16 GB)."""
    from . import tts
    try:
        return tts.engine_for(voice or "").available()[0]
    except ValueError:
        return True


def _engine_name(voice):
    from . import tts
    try:
        return tts.engine_for(voice or "").short or "Some"
    except ValueError:
        return "Some"


# Modal GPUs: rough speed (x real time, once loaded) for the estimate and the spending cap. Not measured yet
# except the T4 class; newer cards read memory 2-5x faster, which is what speech generation waits on.
MODAL_SPEED = {"L4": 1.5, "A10G": 2.5, "L40S": 4.0, "A100-40GB": 5.0, "H100": 6.0}
MODAL_BATCH = {"L4": 16, "A10G": 16, "L40S": 32, "A100-40GB": 32, "H100": 32}
# Drama 3 (Ultra) runs on a Modal H100 only: a book with its voices uses one for the whole run. Measured 25x real
# time with 8 lines at once (16 go at once); the estimate uses less. Each run installs vLLM-Omni and downloads the
# 37 GB model first.
DRAMA_SPEED, DRAMA_SETUP_HOURS = 15.0, 0.25


def _drama_seconds(jobs):
    return sum(len(s["text"].split()) for s in jobs.values() if s.get("engine") == "drama") / WORDS_PER_SECOND
MODAL_MAX_HOURS = 3.0  # never more than this per run, whatever the estimate (Modal and the paid services)...
HOUR_LIMITS = (3.0, 6.0, 12.0)  # ...unless the book's "time limit per run" says 6 or 12 (Modal allows 24 h at most)


def hour_cap(cloud_hours):
    """The most one run may take: the book's own limit when it's one of HOUR_LIMITS, else 3 h. A longer limit means
    fewer runs for a long book, and also more that a stuck run can cost before it's stopped."""
    try:
        h = float(cloud_hours)
    except (TypeError, ValueError):
        return MODAL_MAX_HOURS
    return h if h in HOUR_LIMITS else MODAL_MAX_HOURS
# The pay-as-you-go services' speeds and batch sizes are on each backend (machines.PaidCloud.speed / .batch).
# What the cloud installs before making lines (its image has PyTorch): qwen-tts pins this transformers, which
# also runs Orpheus and works with the PyTorch 2.4 in the rented machines' images (5.x needs 2.6).
CLOUD_PIP = ["numpy", "soundfile", "transformers==4.57.3", "accelerate"]


def _pack_trained(folders, data, log):
    """Put the Mac's copy of each trained voice ({cloud name: Mac folder}) into data/models-mlx/<name> (as hard
    links: instant, no extra space). The machine unpacks them before it makes their lines."""
    import os
    for name, src in sorted(folders.items()):
        dst = data / "models-mlx" / name
        dst.mkdir(parents=True, exist_ok=True)
        for f in Path(src).iterdir():
            if f.is_file() and f.suffix in (".json", ".safetensors", ".jinja"):
                try:
                    os.link(f, dst / f.name)
                except OSError:
                    shutil.copy2(f, dst / f.name)
    if folders:
        gb = sum(f.stat().st_size for f in (data / "models-mlx").rglob("*") if f.is_file()) / 1e9
        log(f"Sending {len(folders)} trained voice model(s) with it ({gb:.1f} GB).")


def overhang(backend):
    """The hours a run can last past its time budget in the worst case: until its hard stop kills a stuck job and,
    on a rented machine, the wait for the app to collect the lines before the machine deletes itself."""
    from .training import BACKENDS
    from .training.backends.machines import GRACE_MINUTES, RentedMachine
    if isinstance(BACKENDS.get(backend), RentedMachine):
        return 1.0 + GRACE_MINUTES / 60
    return {"kaggle": 0.5, "modal": 0.5, "hfjobs": 1.25, "lightning": 1.2, "beam": 1.5, "cerebrium": 1.0}.get(backend, 1.0)


KAGGLE_SPEED = 1.0  # a T4, x real time (roughly what Kaggle's runs have measured)


def estimate(make_on="kaggle", cloud_gpu="A10G", cloud_hours=None, **kw):
    """How long the cloud would take and what it would cost, from a realistic low to the worst case (the job's own
    time limit, which it can never pass). Kaggle is free, so it's in GPU hours of the weekly allowance."""
    from .training import BACKENDS
    from .training.backends import CLOUDS
    backend = make_on if make_on in CLOUDS else "kaggle"
    try:
        where, jobs, local, seconds = plan(backend, **kw)
    except ValueError as e:  # keep it to one line: Create itself shows the whole message
        msg = str(e)
        if msg.startswith("Add these to the cast:"):
            n = len(msg.split(":", 1)[1].split(", "))
            msg = (f"Can't estimate yet: {n} speaker name{'s' if n != 1 else ''} in the text "
                   f"{'aren' if n != 1 else 'isn'}'t in the cast. Add them, or fix the lines they come from.")
        raise ValueError(msg[:200]) from e
    out = {"backend": backend, "lines": len(jobs), "local": local, "seconds": round(seconds),
           "low_hours": 0.0, "high_hours": 0.0, "price": None, "low_cost": None, "high_cost": None, "free": False,
           "made_all": not _last_plan()["chapters"]}
    if not jobs:
        out["why"] = nothing_to_make(local)
        return out
    free = free_bytes(where)
    if free is not None:  # room on this Mac for what comes back: the lines, the chapters made from them, the book
        out["disk"] = {"need": int(seconds * disk_per_second(kw.get("sound"))), "free": free}
    sounds = sound_jobs(**dict(kw, book=kw.get("book")))
    if sounds:
        out["sounds"] = len(sounds)
    elif kw.get("sound") and kw.get("automake"):
        from . import sfx
        token = sfx.BOOK.set(kw.get("book")) if kw.get("book") else None
        try:
            mac = len(sfx.missing(kw.get("story", "")))
        finally:
            books._reset(sfx.BOOK, token)
        if mac:
            out["mac_sounds"] = mac  # made on this Mac (free), while the lines are made or when they're back
    sending = _to_send(backend, jobs)
    if sending:  # trained voices that go up with it (Modal: once, then kept there)
        out["upload"] = {"models": len(sending), "gb": round(_upload_gb(sending, backend), 1)}
    if backend == "kaggle":
        speed, start, price, cap = KAGGLE_SPEED, 0.25, None, 11.0
        out["free"] = True
    elif backend == "modal":
        from .training.backends.modal_cloud import GPUS
        gpu = cloud_gpu if cloud_gpu in GPUS else "A10G"
        if _drama_seconds(jobs):
            gpu = "H100"
            out["gpu_note"] = "Drama 3 voices need an H100 ($3.95/h): this run uses one for all its lines."
        speed, start, price, cap = MODAL_SPEED[gpu], 0.15, GPUS[gpu][1], hour_cap(cloud_hours)
    else:
        b = BACKENDS[backend]
        gpu = cloud_gpu if cloud_gpu in b.gpu_names else b.default_gpu
        try:
            price = b.price(gpu)
        except Exception:  # noqa: BLE001
            price = None
        speed, start, cap = b.speed.get(gpu, 2.0), 0.25, hour_cap(cloud_hours)
    drama = _drama_seconds(jobs) if backend == "modal" else 0.0
    work = (seconds - drama) / 3600 / speed + drama / 3600 / DRAMA_SPEED + sound_hours(sounds) / min(1.0, speed)
    start += DRAMA_SETUP_HOURS if drama else 0.0
    hours = work + start
    low = work / 1.3 + start * 0.5                                # a fast GPU that starts at once
    floor = 0.3 if backend == "kaggle" else 0.4 if backend == "modal" else 0.5
    budget = min(cap, max(floor, hours * 2.5))                    # the run stops making lines after this
    high = budget + overhang(backend)                             # stuck, then never collected: the most it can bill
    runs = 1
    if hours * 1.3 > budget:  # too long for one run: Create in the cloud again carries on where the last one stopped
        runs = math.ceil(hours * 1.3 / budget)
        high = runs * (budget + overhang(backend))
    out.update(low_hours=round(low, 2), high_hours=round(high, 2), budget=round(budget, 1), price=price, runs=runs)
    if price:
        out.update(low_cost=round(low * price, 2), high_cost=round(high * price, 2),
                   run_max=round((cap + overhang(backend)) * price, 2))  # the most one stuck run could cost
    out["cap"] = cap
    from . import pricing
    r = pricing.gpu_range(backend, price, low, high, "create") if price else pricing.gpu_range(backend, None, 0, 0, "create")
    out.update(cost=pricing.say(r), cost_basis=r["basis"])
    return out


def send(log=print, make_on="kaggle", cloud_gpu="A10G", cloud_live=True, cloud_hours=None, **kw):
    """_send(), noting a connection problem with the cloud on Connections."""
    from . import conn_issues
    from .training.backends import CLOUDS
    key = str(kw.get("folder") or "")
    with _lock:  # (two sends at once, a double click: both would pass the check below before either is saved)
        if key and key in _sending:
            raise ValueError("This book is already being sent to the cloud.")
        _sending.add(key)
    try:
        return conn_issues.watch(make_on if make_on in CLOUDS else "kaggle", "Create in the cloud", _send, log, make_on,
                                 cloud_gpu, cloud_live, cloud_hours, **kw)
    finally:
        with _lock:
            _sending.discard(key)


def busy(folder):
    """Whether the book is out in the cloud already (or being sent): {"state", "sent", "back", "lines", "backend"},
    else None. Asked before Create in the cloud, so a second run isn't started by accident."""
    if str(folder or "") in _sending:
        return {"state": "sending"}
    where = books.book_folder("", "", folder)
    rec = read(where) or {}
    if rec.get("state") not in WAITING + ("downloading",):
        return None
    return {k: rec.get(k) for k in ("state", "sent", "back", "lines", "backend")}


# ---------- a book changed since its cloud run (its lines then don't match: they'd all be made again) ----------

def _book_keys(**kw):
    """The cloud names of every line the book has now, with these settings."""
    where, chapters, gen = books.chapters_to_make(all_chapters=True, **kw)
    opts = {"language": gen["language"], "temperature": gen["temperature"]}
    keys = set()
    for _, _, items in chapters:
        for it in en.spoken(items, gen["sound"]):
            if "cue" not in it:
                spec = line_spec(it, opts, gen["seed"])
                if spec is not None:
                    keys.add(key_of(spec))
    return where, keys


def matches(**kw):
    """How the book now compares with what its last cloud run was sent with (asked before Finish and Create in the
    cloud, so a setting changed since doesn't quietly make every line again): {"run_lines": that run's lines on this
    Mac, "used": how many of them the book uses now, "changed": [{"what", "then", "now"}] (what makes lines
    differ), "restore": {the settings to put back: book fields, and "cast": {character: {field: value}}}}; {} when
    there's no run to compare with."""
    from . import voices as vl
    where = books.book_folder(kw.get("title", ""), kw.get("story", ""), kw.get("folder"))
    try:
        sent = json.loads((where / "cloud" / REQUEST).read_text())
        run = json.loads((where / "cloud" / KEYS).read_text())
    except (OSError, ValueError):
        return {}
    cache = where / books.CLOUD_LINES
    back = {k for k in run if (cache / f"{k}.flac").exists()}
    if not back:
        return {}
    _, now = _book_keys(**kw)
    used = len(back & now)
    out = {"run_lines": len(back), "used": used, "changed": [], "restore": {}}
    if used == len(back):
        return out
    def num(v):  # (25 and 25.0 and "25" are the same setting)
        try:
            return None if v is None else float(v)
        except (TypeError, ValueError):
            return v
    # the book's own settings that go into a line (speed doesn't: it's applied when the chapter is put together)
    for k, what in (("stability", "the book's stability"), ("language", "the language"), ("seed", "the seed"),
                    ("quotes_only", "splitting quotes"), ("models", "the voice models"),
                    ("pronunciations", "the pronunciations")):
        a, b = sent.get(k), kw.get(k)
        if k in ("stability", "seed"):
            a, b = num(a), num(b)
        if a != b and not (k == "pronunciations" and not a and not b):
            out["changed"].append({"what": what, "then": sent.get(k), "now": kw.get(k)})
            out["restore"][k] = sent.get(k)
    then_rows = {r[0]: r for r in sent.get("table") or []}
    now_rows = {r[0]: r for r in kw.get("table") or []}
    then_stab, now_stab = sent.get("voice_stability") or {}, kw.get("voice_stability") or {}
    cast = {}
    for name in now_rows:
        if name not in then_rows:
            continue
        t, n = then_rows[name], now_rows[name]
        fix = {}
        if num(then_stab.get(name)) != num(now_stab.get(name)):
            out["changed"].append({"what": f"{name}'s stability", "then": then_stab.get(name), "now": now_stab.get(name)})
            fix["stability"] = then_stab.get(name)
        if t[1] != n[1]:
            out["changed"].append({"what": f"{name}'s voice", "then": t[1], "now": n[1]})
            value = vl.from_name(t[1])
            if value is not None:
                fix["voice"] = value
        if (t[2] or "") != (n[2] or ""):
            out["changed"].append({"what": f"{name}'s style", "then": t[2], "now": n[2]})
            fix["style"] = t[2] or ""
        if fix:
            cast[name] = fix
    if cast:
        out["restore"]["cast"] = cast
    if (sent.get("story") or "").strip() != (kw.get("story") or "").strip():
        out["changed"].append({"what": "the text", "then": None, "now": None})  # (edits aren't put back)
    return out


def _send(log=print, make_on="kaggle", cloud_gpu="A10G", cloud_live=True, cloud_hours=None, **kw):
    """Send the book's lines to the cloud (make_on: one of training.backends.CLOUDS). Returns the job record.
    cloud_live: copy the lines back while the job works, where the service allows it (Backend.live)."""
    from .training import BACKENDS
    from .training.backends import CLOUDS
    from .training.backends.kaggle import _slug
    from .training.base import Package
    backend = make_on if make_on in CLOUDS else "kaggle"
    if not kw.get("folder"):
        raise ValueError("Only saved Audiobook projects can be made in the cloud.")
    log("Working out the lines to make…")
    where, jobs, local, seconds = plan(backend, **kw)
    rec = read(where)
    if rec and rec.get("state") in WAITING + ("downloading",):
        raise ValueError("This book is already being made in the cloud. Wait for it to come back, or forget it first.")
    if not jobs:
        raise ValueError("Nothing to make in the cloud. " + nothing_to_make(local))
    b = BACKENDS[backend]
    c = b.check()
    if not c.get("ok"):
        raise ValueError(c.get("detail") or f"{b.name} isn't signed in.")
    sounds = []
    if kw.get("cloud_sounds", True) and kw.get("sound") and kw.get("automake") and sound_jobs(**kw):
        # as this Mac would: the sound libraries' recorded sounds first (when the book's Director tab allows them),
        # then only what still needs AI goes up with the lines
        if books.use_sound_libraries(kw.get("book")):
            from . import online_sounds
            if online_sounds.ready():
                gen = books.make_missing_sounds(kw.get("story", ""), kw.get("book"), kw.get("new_sounds", "all"),
                                                online_only=True)
                try:
                    while True:
                        log(next(gen))
                except StopIteration:
                    pass
        sounds = sound_jobs(**kw)
        if sounds:
            log(f"{len(sounds)} missing sound(s) are made in the cloud too, after the lines.")
    if backend == "modal":  # trained voices it doesn't have yet: uploaded once, kept there for later books
        from . import modal_models
        for name, folder in sorted(_to_send("modal", jobs).items()):
            log(f"{name} isn't on your Modal storage yet: uploading it first (once: later books use it from there).")
            modal_models.upload(folder.name, log=log)
    engines = {s["engine"] for s in jobs.values()}
    pip = (["qwen-tts"] if engines & {"qwen", "qwen-clone", "qwen-tuned"} else []) + (["snac"] if "orpheus" in engines else [])
    slug = (_slug(where.name)[:34].strip("-") or "book") + "-book-lines"
    log(f"{len(jobs):,} lines to make in the cloud (about {en.clock(seconds)} of speech)"
        + (f"; {local:,} more are made on this Mac" if local else "") + ".")
    kernels = []
    if backend == "kaggle":
        q = c.get("quota")
        left = q["left"] if q else 11.0
        budget = round(min(11.0, left - 0.3), 1)
        if budget < 0.3:
            raise ValueError(f"Kaggle has {left:.1f} GPU hours left this week, not enough to start. They come back "
                             f"{(q or {}).get('resets', 'next week')[:16].replace('T', ' ')} UTC.")
        log(f"Kaggle: {left:.1f} GPU hours left this week; this run stops after {budget} h at most.")
        args, gpu, cost, price = f"--max-hours {budget}", "T4", 0.0, 0.0
        kernels = sorted({s["model"]["kernel"] for s in jobs.values() if (s.get("model") or {}).get("kernel")})
    elif backend != "modal":  # a pay-as-you-go service (machines.PaidCloud)
        gpu = cloud_gpu if cloud_gpu in b.gpu_names else b.default_gpu
        label = b.gpu_names[gpu]
        try:
            price = b.price(gpu)
        except Exception:  # noqa: BLE001 - only for the estimate
            price = None
        if c.get("balance") is not None and c["balance"] < 1.0:
            raise ValueError(f"Your {b.name} account has ${c['balance']:.2f} of credit. Add some on {b.name}'s "
                             "billing page first: a GPU can't start without it.")
        hours = seconds / 3600 / b.speed.get(gpu, 2.0) + 0.25 + sound_hours(sounds)  # + starting up, + sounds
        budget = round(min(hour_cap(cloud_hours), max(0.5, hours * 2.5)), 1)
        cost = (budget + overhang(backend)) * price if price else None
        rate = f" (${price:.2f}/h)" if price else ""
        log(f"{b.name} {label}{rate}: about {hours:.1f} hours" + (f", roughly ${hours * price:.2f}" if price else "")
            + f". It stops after {budget} h at most, and what it used is freed as soon as the lines are back."
            + (f" Credit now: ${c['balance']:.2f}." if c.get("balance") is not None else ""))
        args = f"--max-hours {budget} --dtype bfloat16 --batch {b.batch.get(gpu, 16)}"
        pip = CLOUD_PIP + list(getattr(b, "extra_pip", [])) + pip
    else:
        from .training.backends.modal_cloud import GPUS
        gpu = cloud_gpu if cloud_gpu in GPUS else "A10G"
        drama = _drama_seconds(jobs)
        if drama:
            gpu = "H100"
            log("Drama 3 voices need an H100 ($3.95/h): this run uses one for all its lines.")
        price = GPUS[gpu][1]
        hours = ((seconds - drama) / 3600 / MODAL_SPEED[gpu] + drama / 3600 / DRAMA_SPEED + 0.15  # + starting up
                 + (DRAMA_SETUP_HOURS if drama else 0.0) + sound_hours(sounds))
        budget = round(min(hour_cap(cloud_hours), max(0.4, hours * 2.5)), 1)  # room for the guess being off
        cost = (budget + overhang("modal")) * price
        log(f"Modal {GPUS[gpu][0]} (${price:.2f}/h): about {hours:.1f} GPU hours, roughly ${hours * price:.2f}. "
            f"It stops after {budget} h, and even stuck it's ended by {budget + overhang('modal'):.1f} h, so never "
            f"more than ${cost:.2f}. Modal bills by the second.")
        args = f"--max-hours {budget} --dtype bfloat16 --batch {MODAL_BATCH[gpu]}"
        # Modal's image starts empty (Kaggle's comes with these): what the runner and Orpheus need on their own
        pip = ["numpy", "soundfile", "transformers", "accelerate"] + pip

    job = where / "cloud" / "job"
    shutil.rmtree(job, ignore_errors=True)
    data = job / "data"
    (data / "voices").mkdir(parents=True)
    for f in ("lines_runner.py", "runner.py", "unpack_mlx.py", "higgs_worker.py", "drama_worker.py"):
        shutil.copy2(RUNNER_DIR / f, data / f)
    clones = sorted({s["voice"] for s in jobs.values() if s["engine"] in ("qwen-clone", "higgs", "drama")})
    for name in clones:
        shutil.copy2(lib.wav_path(name), data / "voices" / f"{name}.wav")
        if any(s["engine"] in ("higgs", "drama") and s["voice"] == name for s in jobs.values()):  # clip + transcript
            (data / "voices" / f"{name}.txt").write_text(lib.info(name)["transcript"], encoding="utf-8")
    (data / "lines.json").write_text(json.dumps([{"key": key, **{f: x for f, x in s.items() if not f.startswith("_")}}
                                                 for key, s in jobs.items()], ensure_ascii=False))
    if sounds:  # the generator's own script (its weights are downloaded there)
        (data / "sounds.json").write_text(json.dumps(sounds, ensure_ascii=False))
        shutil.copytree(SOUND_CODE, data / "stable_audio_3", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copy2(RUNNER_DIR / "sound_worker.py", data / "sound_worker.py")
    sending = _to_send(backend, jobs)
    if sending and backend != "modal":  # trained voices go with the job as the Mac keeps them (unpacked on the GPU)
        _pack_trained(sending, data, log)
    live = bool(cloud_live) and bool(getattr(b, "live", False))
    if live:
        args += " --no-pack"  # the lines stay separate files, so the new ones can be copied as they're made
    (where / "cloud" / KEYS).write_text(json.dumps(sorted(jobs)))  # which lines this job makes (for "N back")
    pkg = Package(model=where.name, recipe="book", data_dir=data, script="lines_runner.py", args=args, pip=pip,
                  gpu=gpu, hours=round(seconds / 3600, 2), slug=slug, title=slug.replace("-", " "),
                  kernel_sources=kernels, meta={"timeout_hours": budget + 0.5, "live": live})
    from . import spend, tasks
    if tasks.stop_wanted():
        raise tasks.Stopped()
    if price:  # within today's spending limit (the most it can bill is its time limit × the price)
        spend.allow(backend, {"low": round(hours * price * 0.75, 2), "high": round(cost, 2) if cost else None}, "create")
    if tasks.stop_wanted():
        raise tasks.Stopped()
    run = b.submit(pkg, log=log)
    expect = round(hours, 2) if price else None  # (how long it's thought to take: its cost if it ends unseen)
    spend.job_sent(backend, run, f"Create in the cloud: {where.name}", price=price or None,
                   hours=budget + overhang(backend), collect=True, expect=expect)  # (followed again after a restart)
    label = getattr(b, "gpu_names", {}).get(gpu) or gpu
    rec = {"backend": backend, "gpu": gpu, "gpu_label": label, "run": run, "url": b.url(run), "sent": time.time(), "lines": len(jobs),
           # the sounds made with it, kept as this Mac keeps its own (in the book or shared, as its settings say)
           "sounds": {x["key"]: {k: x[k] for k in ("name", "kind", "model")} for x in sounds},
           "book": kw.get("book"), "scope": kw.get("new_sounds", "all"),
           "local": local, "seconds": round(seconds), "budget": budget,
           "max_cost": round(cost, 2) if cost is not None else None, "price": price,  # (US$ an hour: Activity's cost)
           "expect": expect,
           "state": "running", "checked": time.time(), "live": live, "back": 0,
           # its missing sounds are made on this Mac (while the cloud speaks the lines)
           "mac_sounds": bool(not sounds and sound_jobs(**dict(kw, cloud_sounds=True)))}
    _save(where, rec)
    (where / "cloud" / REQUEST).write_text(json.dumps(kw, ensure_ascii=False, default=str))
    from . import voice_check
    voice_check.start(where, jobs, kw)  # (before the stitcher: it waits for the voices to be checked)
    stay_awake()
    start_stitch(where)
    log(f"Done · Sent {len(jobs):,} lines to {b.name}. The app checks on it every couple of minutes and brings the "
        + ("lines back as they're made" if live else "lines back when it's finished")
        + "; then press Finish on this Mac. Keep this Mac on and the app open until then: "
        "it stays awake by itself, but closing the lid or shutting down stops it.")
    return rec


# ---------- putting the book together as its lines come back ----------

def stitching(where):
    """The task putting the book's chapters together as their lines come back, while it runs (else None)."""
    t = (_stitchers.get(str(where)) or (None, None))[0]
    return t if t is not None and t.state == "running" else None


def start_stitch(where):
    """While the book is out in the cloud: make its missing sounds here right away (when this Mac makes them), put
    each chapter together as soon as all its lines are back, and the whole book when the run is back. One per book,
    for the run its record is for now (one still going for a run let go is stopped)."""
    from . import tasks
    where = Path(where)
    with _stitch_lock:
        run = (read(where) or {}).get("run")
        t = stitching(where)
        if t is not None:
            if _stitchers[str(where)][1] == run:
                return None
            t.stop_requested.set()  # (that run was let go, and a new one sent)
        try:
            kw = json.loads((where / "cloud" / REQUEST).read_text())
        except (OSError, ValueError):
            return None
        t = tasks.start("book", f"Audiobook: {kw.get('title') or 'Story'}", lambda: _stitch(where, kw, run))
        _stitchers[str(where)] = (t, run)
    _update_run(where, run, stitch=t.id)
    return t


def _create_args(kw):
    import inspect
    names = set(inspect.signature(books.create).parameters)
    return {k: v for k, v in kw.items() if k in names}


def ready_chapters(where, kw):
    """(chapters still to make whose every line is back and whose sounds are all in the library, chapters still to
    make, chapters)."""
    from . import sfx
    _, todo, gen = books.chapters_to_make(**kw)
    cache = Path(where) / books.CLOUD_LINES
    opts = {"language": gen["language"], "temperature": gen["temperature"]}
    ready = []
    token = sfx.BOOK.set(kw["book"]) if kw.get("book") else None
    try:
        for i, _, items in todo:
            whole = True
            for it in en.spoken(items, gen["sound"]):
                if "cue" in it:
                    continue
                spec = line_spec(it, opts, gen["seed"])
                if spec is None or not (cache / f"{key_of(spec)}.flac").exists():
                    whole = False
                    break
            if whole and gen["sound"] and any(x[2] is None for x in books.chapter_sounds(items)):
                whole = False  # (a sound still being made: the chapter waits for it)
            if whole:
                ready.append(i)
    finally:
        books._reset(sfx.BOOK, token)
    return ready, len(todo), gen["chapters"]


def _wait(seconds):
    from . import tasks
    end = time.time() + seconds
    while time.time() < end:
        if tasks.stop_wanted():
            raise tasks.Stopped()
        time.sleep(1)


def _stitch(where, kw, run):
    """_stitching(), noting on the run's record when it ends by Stop or an error, so the watcher doesn't start it
    again by itself (Finish does it then)."""
    from . import tasks
    try:
        yield from _stitching(where, kw, run)
    except GeneratorExit:  # (closed: by Stop, or as the engine ends)
        if tasks.stop_wanted():
            _update_run(where, run, stitch_end="stopped")
        raise
    except tasks.Stopped:
        _update_run(where, run, stitch_end="stopped")
        raise
    except Exception as e:  # noqa: BLE001 - noted, and reported as the task's error
        _update_run(where, run, stitch_end="failed", stitch_error=str(e)[:300])
        raise


def _stitching(where, kw, run):
    from . import projects
    args = _create_args(kw)
    rec = read(where) or {}
    if rec.get("mac_sounds") and rec.get("run") == run:
        yield {"status": "Generating sounds…"}
        yield from books._relay(books.make_missing_sounds(kw.get("story", ""), kw.get("book"), kw.get("new_sounds", "all")))
    put = 0
    tried = set()  # chapters put together already (one saved with lines left out is made again by Finish, not here)
    while True:
        rec = read(where)
        if not rec or rec.get("run") != run:
            yield {"status": "Done · cloud run stopped"}
            return
        state = rec.get("state")
        if state == "failed":
            raise ValueError(rec.get("error") or "The cloud run failed.")
        from . import voice_check
        if not voice_check.cleared(where):  # nothing is put together before its voices are checked right
            if state == "ready" and not _check_voice(where, final=True) and not voice_check.cleared(where):
                # (back, and a voice still can't be told apart: Finish decides; this doesn't guess)
                _update_run(where, run, stitch_end="partial")
                yield {"status": "Done · couldn't verify voices automatically: press Finish to build the book."}
                return
            yield {"status": "Verifying voices…"}
            _wait(30)
            continue
        lines = rec.get("lines") or 0
        back = _back(where) if state == "ready" else None
        if state == "ready" and (back is None or lines - back <= FINISH_MISSING):  # (all back: Finish, as the button)
            _, left_now, total_now = ready_chapters(where, kw)
            assembled_now = max(0, total_now - left_now)
            p_start = (assembled_now / total_now) * 0.85 if total_now else 0.0
            audio = None
            for ev in books.create(**dict(args, automake=bool(kw.get("automake")), progress_start=p_start)):
                audio = ev.get("audio") or audio
                yield ev
            if audio and kw.get("book"):
                projects.save(kw["book"], {"file": audio})
            return
        ready, left, total = ready_chapters(where, kw)
        ready = [i for i in ready if i not in tried]
        if state == "ready" and not ready:  # the run is back without many of its lines: Finish, or send them again
            _update_run(where, run, stitch_end="partial")
            yield {"status": f"Done · {back:,}/{lines:,} lines back · {total - left}/{total} chapters assembled"}
            return
        if ready:
            done_before = max(0, total - left)
            batch_n = len(ready)
            for ev in books.create(**dict(args, automake=False), only=set(ready)):
                if "audio" in ev or str(ev.get("status", "")).startswith("Done"):
                    continue
                ev = dict(ev)
                if ev.get("progress") is not None and total:
                    ev["progress"] = min(0.85, ((done_before + ev["progress"] * batch_n) / total) * 0.85)
                yield ev
            put += len(ready)
            tried.update(ready)
            continue
        lines, back = rec.get("lines") or 0, rec.get("back") or 0
        assembled = max(0, total - left)
        line_ratio = (back / lines) if lines else 0.0
        if total == 1:
            p = line_ratio * 0.85 if lines else 0.0
        elif total > 1:
            ch_ratio = assembled / total
            p = (ch_ratio if assembled > 0 else line_ratio / total) * 0.85
        else:
            p = line_ratio * 0.85 if lines else None
        ch_txt = f"Assembled {assembled}/{total} chapters · " if total else ""
        yield {"status": f"{ch_txt}waiting on cloud ({back:,}/{lines:,})",
               "progress": p, "stitch": {"chapters": total, "ready": assembled}}
        _wait(60)


# ---------- checking and bringing it back ----------

def check(where, force=False):
    """The job, after asking Kaggle how it's doing (at most every two minutes unless forced). When it's finished,
    its lines are downloaded in the background."""
    rec = read(where)
    if not rec:
        return None
    if rec.get("state") in WAITING and (force or time.time() - rec.get("checked", 0) > 110):
        from .training import BACKENDS
        from . import conn_issues
        try:
            st = conn_issues.watch(rec.get("backend", "kaggle"), "Checking a book in the cloud",
                                   BACKENDS[rec.get("backend", "kaggle")].status, rec["run"])
        except Exception as e:  # noqa: BLE001 - shown to the user, checked again later
            if conn_issues.is_network(e):  # (asked again as soon as the connection is back: _reconnect)
                _net_fail(where, "status", e)
            return _update(where, checked=time.time(), note=str(e)[:200]) or rec
        _net_ok(where, "status")
        changes = {"checked": time.time(), "note": st.detail or ""}
        if st.state in ("queued", "running"):
            changes.update(state=st.state, seen=time.time())
        elif st.state in ("complete", "error", "cancelled"):
            changes.update(state="downloading", ended=st.state,  # an error can still leave lines made before it
                           # (the GPU's time ends here, for Activity's cost; earlier when it ended while Babble was closed)
                           finished_at=ended_about(rec.get("sent"), rec.get("seen") or rec.get("checked"), rec.get("expect")))
        rec = _update(where, **changes) or rec
    lock = _collecting.setdefault(str(where), threading.Lock())
    if (rec.get("live") and rec.get("state") == "running" and not lock.locked()
            and (force or time.time() - rec.get("live_at", 0) > 100)):
        rec = _update(where, live_at=time.time()) or rec
        threading.Thread(target=_live, args=(Path(where),), daemon=True).start()
    if (rec.get("state") == "downloading" and str(where) not in _downloads
            and (force or time.time() >= rec.get("retry_at", 0))):
        _downloads.add(str(where))
        threading.Thread(target=_download, args=(Path(where),), daemon=True).start()
    return rec


def _back(where):
    """How many of the job's lines are on this Mac now (None when it isn't known which lines it makes)."""
    try:
        keys = json.loads((Path(where) / "cloud" / KEYS).read_text())
    except (OSError, ValueError):
        return None
    cache = Path(where) / books.CLOUD_LINES
    return sum(1 for k in keys if (cache / f"{k}.flac").exists())


def _whole(f):
    """Whether a line's file is all there: its last tenth of a second reads back. (A file cut off on the way still
    says at its start how long it should be, so its header alone can't tell.)"""
    try:
        with sf.SoundFile(str(f)) as s:
            if s.frames <= 0:
                return False
            start = max(0, s.frames - s.samplerate // 10)
            s.seek(start)
            return len(s.read(dtype="float32")) == s.frames - start
    except Exception:  # noqa: BLE001 - not a whole audio file
        return False


def _put_line(cache, name, write):
    """Put one line into the book's cloud lines only when its file is whole: write(path) makes it under another
    name first, so a download cut off part way never leaves a broken line that counts as made."""
    tmp = cache / f".{name}.part"
    try:
        write(tmp)
        if not _whole(tmp):
            return False
        os.replace(tmp, cache / name)
        return True
    finally:
        tmp.unlink(missing_ok=True)


def _collect(where, rec, b, final=False):
    """Copy the job's lines made since the last time into the book's cloud lines (only whole files). A copy the
    connection cuts off part way keeps the lines that came down whole (the error is raised after): the next one
    carries on from there instead of starting again."""
    cache = where / books.CLOUD_LINES
    cache.mkdir(parents=True, exist_ok=True)
    tmp = where / "cloud" / "live"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    moved = 0
    try:
        b.fetch_new(rec["run"], tmp, {f.name for f in cache.glob("*.flac")}, final=final)
    except BaseException:
        moved = _keep_whole(tmp, cache)
        if moved:
            print(f"[cloud] the copy for {where.name} was cut off: kept the {moved} line(s) that came back whole",
                  flush=True)
        raise
    else:
        moved = _keep_whole(tmp, cache)
        return moved
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _keep_whole(tmp, cache):
    """Move the whole line files that came down into the book's cloud lines: how many."""
    moved = 0
    for f in Path(tmp).iterdir():
        if LINE.fullmatch(f.name) and f.is_file() and _whole(f):
            shutil.move(str(f), Path(cache) / f.name)
            moved += 1
    return moved


def _live(where):
    """While the job works: bring back the lines it has made so far, so they're safe on this Mac."""
    from .training import BACKENDS
    lock = _collecting.setdefault(str(where), threading.Lock())
    if not lock.acquire(blocking=False):
        return
    try:
        rec = read(where) or {}
        if rec.get("state") not in WAITING or not rec.get("live"):
            return
        try:
            n = _collect(where, rec, BACKENDS[rec.get("backend", "kaggle")])
            back = _back(where)
            # when lines first came back, and how many: the pace from then on gives the time left
            first = {} if rec.get("first_back") or not back else {"first_back": [time.time(), back]}
            _update(where, back=back, live_at=time.time(), live_error="", **first)
            _net_ok(where, "copy")
            if n:
                print(f"[cloud] {n} more line(s) back from the cloud for {where.name}", flush=True)
        except Exception as e:  # noqa: BLE001 - tried again next time; the last copy at the end gets them anyway
            from . import conn_issues
            if conn_issues.is_network(e):  # (tried again as soon as the connection is back: _reconnect)
                _net_fail(where, "copy", e)
            _update(where, back=_back(where), live_at=time.time(), live_error=str(e)[:200])
    finally:
        lock.release()
    from . import voice_check
    if not voice_check.cleared(where):  # the first lines of a voice are back: are they in the right voice?
        threading.Thread(target=_check_voice, args=(Path(where),), daemon=True, name="voice-check").start()


# ---------- a lost connection: noticed, and carried on from as soon as it's back ----------

RECONNECT_SECONDS = 15  # while a book waits on a lost connection: how often it's looked at whether it's back
RECONNECT_GAP = 60      # after trying again (the connection looked back), at least this long before the next try


def _net_fail(where, kind, e):
    """What the book was doing ("status", "copy" or "download") couldn't get through: waiting for the connection."""
    with _lock:
        rec = read(where)
        if rec is None:
            return
        wait = rec.get("net_wait") or {}
        was = wait.get(kind) or {}
        wait[kind] = {"since": was.get("since") or time.time(), "tried": was.get("tried", 0), "error": str(e)[:200]}
        rec["net_wait"] = wait
        _write_or_note(where, rec)
    if not was:
        print(f"[cloud] lost the connection for {Path(where).name} ({kind}): {str(e)[:160]}", flush=True)


def _net_ok(where, kind):
    """That got through: no longer waiting on the connection for it."""
    rec = read(where) or {}
    if kind not in (rec.get("net_wait") or {}):
        return
    with _lock:
        rec = read(where)
        if rec is None:
            return
        wait = rec.get("net_wait") or {}
        gone = wait.pop(kind, None)
        rec["net_wait"] = wait
        if not wait:
            rec.pop("net_wait", None)
        _write_or_note(where, rec)
    if gone:
        print(f"[cloud] the connection is back for {Path(where).name} ({kind}, after "
              f"{en.clock(time.time() - gone.get('since', time.time()))})", flush=True)


def _reconnect(where, rec, now=None):
    """A book waiting on a lost connection: when the service can be reached again, carry on at once (ask how
    the run is doing, copy its lines back, download them) instead of at the next round. True when it did."""
    from . import conn_issues
    from .training import BACKENDS
    now = now or time.time()
    wait = rec.get("net_wait") or {}
    if not wait or now - max(w.get("tried", 0) for w in wait.values()) < RECONNECT_GAP:
        return False
    b = BACKENDS.get(rec.get("backend", "kaggle"))
    if not conn_issues.reachable(getattr(b, "hosts", ()) or ()):
        return False
    print(f"[cloud] {getattr(b, 'name', 'the cloud')} can be reached again: carrying on with {Path(where).name}",
          flush=True)
    with _lock:
        rec = read(where)
        if rec is None:
            return False
        for w in (rec.get("net_wait") or {}).values():
            w["tried"] = now
        rec.update(live_at=0, retry_at=0)
        _write_or_note(where, rec)
    check(where, force=True)
    return True


def _reconnect_loop(every=None):
    while True:
        time.sleep(RECONNECT_SECONDS if every is None else every)
        try:
            for f in books.BOOKS.glob(f"*/{RECORD}"):
                rec = read(f.parent) or {}
                if rec.get("net_wait"):
                    _reconnect(f.parent, rec)
        except Exception as e:  # noqa: BLE001 - looked at again next time
            print(f"[cloud] reconnect: {e}", flush=True)


def _bring_sounds(rec, k, out):
    """The sounds made with the lines: each one's best take, kept in the library as this Mac keeps the sounds it
    makes (the same check, trim and naming; in the book or shared as its settings say). How many came back. Any
    that didn't are made on this Mac when the book is finished."""
    from . import sfx, sfxgen
    try:
        k.fetch(rec["run"], out, r"(^|/)sounds/[^/]+\.wav$")
    except Exception as e:  # noqa: BLE001 - Finish makes them here instead
        print(f"[cloud] couldn't bring the sounds back: {e}", flush=True)
        return 0
    type_of = {v: t for t, v in sfx.KIND_OF.items()}
    book, scope = rec.get("book"), rec.get("scope") or "all"
    token = sfx.BOOK.set(book) if book else None
    kept = 0
    try:
        for key, x in rec["sounds"].items():
            takes = sorted(f for f in out.rglob(f"{key}-*.wav") if not f.name.startswith("."))
            if not takes:
                continue
            if sfx.find_for({"type": type_of.get(x["kind"], "sfx"), "name": x["name"]}):
                kept += 1  # (made meanwhile)
                continue
            try:
                audio = sfxgen.best_take(x["name"], x["kind"], takes)
                books.keep_in(sfxgen.keep(x["name"], x["kind"], audio, x["model"], where="made in Babble on a cloud GPU"),
                              book, scope)
                kept += 1
            except Exception as e:  # noqa: BLE001 - this one's made on the Mac instead
                print(f"[cloud] sound “{x['name']}”: {e}", flush=True)
    finally:
        books._reset(sfx.BOOK, token)
    print(f"[cloud] {kept} of {len(rec['sounds'])} sound(s) back from the cloud", flush=True)
    return kept


RETRY_SECONDS = 120  # a download that couldn't reach the cloud is tried again after this (and as soon as it's back)…
RETRY_HOURS = 24      # …for this long after the run ended (the service's own keeps_hours where it says)


def _network_error(e):
    """Whether a download failed on the way (the connection dropped or didn't go through, a network that blocks
    the service), not on what came back: then it's worth trying again. (conn_issues decides, for every service.)"""
    from . import conn_issues
    return conn_issues.is_network(e)


def _download(where):
    lock = _collecting.setdefault(str(where), threading.Lock())
    with lock:  # (after a copy that's still going)
        _download_now(where)


def _download_now(where):
    from .training import BACKENDS
    rec = read(where) or {}
    out = where / "cloud" / "out"
    cache = where / books.CLOUD_LINES
    try:
        shutil.rmtree(out, ignore_errors=True)
        k = BACKENDS[rec.get("backend", "kaggle")]
        if rec.get("live") and getattr(k, "live", False):  # most lines are here already: just the rest
            k.fetch(rec["run"], out, r"(progress\.json|run\.log)$")
            _collect(where, rec, k, final=True)
        else:
            k.fetch(rec["run"], out, r"(lines\.tar|progress\.json|run\.log)$")
            if not any(out.rglob("lines.tar")):  # the run was cut off before it packed its lines
                k.fetch(rec["run"], out, r"lines/.*\.flac$")
        cache.mkdir(parents=True, exist_ok=True)
        broken = 0  # lines whose file came back cut off: left out, so they're made again
        for tar in out.rglob("lines.tar"):
            with tarfile.open(tar) as t:
                for m in t.getmembers():
                    name = Path(m.name).name
                    if m.isfile() and LINE.fullmatch(name):
                        def unpack(p, m=m):
                            with t.extractfile(m) as src, open(p, "wb") as dst:
                                shutil.copyfileobj(src, dst)
                        broken += not _put_line(cache, name, unpack)
        for f in out.rglob("*.flac"):
            if LINE.fullmatch(f.name):
                broken += not _put_line(cache, f.name, lambda p, f=f: shutil.move(str(f), p))
        if broken:
            print(f"[cloud] {broken} line(s) for {where.name} came back cut off: left out, so they're made again "
                  "(on this Mac at Finish)", flush=True)
        made = len(list(cache.glob("*.flac")))
        if rec.get("sounds"):
            rec["sounds_made"] = _bring_sounds(rec, k, out)
        progress = next(iter(out.rglob("progress.json")), None)
        info = json.loads(progress.read_text()) if progress else {}
        back = _back(where)  # the job's lines that are on this Mac now
        if back is None:  # (a job sent before the app kept its list of lines)
            back = int(info.get("made", 0)) or made
        log_tail = ""
        logs = list(out.rglob("run.log"))
        if logs:
            log_tail = "\n".join(logs[0].read_text(errors="replace").splitlines()[-6:])
        from . import voice_check
        wrong = voice_check.check(where, final=True) if not voice_check.cleared(where) else []
        if wrong:  # (a run whose lines only come back at the end: checked now, before Finish could use them)
            names, moved = _trash_wrong(where, wrong)
            back = _back(where) or 0
            rec.update(state="failed", made=back, back=back, stitch_end="failed", note="",
                       error=f"The cloud made {names}'s lines in a different voice ({wrong[0][1]}). Babble checked them "
                             f"against that voice on this Mac: its {moved['moved']:,} lines in that voice are in the "
                             "Trash, kept out of the book.")
        elif not info and not made:
            rec.update(state="failed", error="Nothing came back from the cloud. " + (log_tail[-400:] or
                       f"Open {rec.get('url')} to see what happened."))
        else:
            rec.update(state="ready", made=back, back=back, speed=info.get("speed"), audio=info.get("audio_seconds"),
                       took=info.get("seconds"), stopped=info.get("stopped"), error="", note="")
            if back < rec.get("lines", 0) and not info.get("stopped"):
                text = logs[0].read_text(errors="replace") if logs else ""
                why = [ln.strip() for ln in text.splitlines() if re.search(r"(Error|FAILED):", ln)]
                rec["error"] = (f"{rec['lines'] - back} line(s) didn't work in the cloud"
                                + (f" ({why[-1][:160]})" if why else "") + "; Finish makes them on this Mac.")
        rec.pop("net_wait", None)  # (whatever it waited on the connection for is done)
    except Exception as e:  # noqa: BLE001 - shown to the user
        from . import conn_issues
        conn_issues.note(rec.get("backend", "kaggle"), e, "Bringing a book's lines back")
        back = _back(where)
        ended = rec.get("finished_at") or rec.get("sent") or time.time()
        b = BACKENDS.get(rec.get("backend", "kaggle"))
        keeps = getattr(b, "keeps_hours", RETRY_HOURS)
        if _network_error(e) and time.time() - ended < keeps * 3600:  # the lines wait in the cloud: again soon
            name = getattr(b, "name", "the cloud")
            wait = rec.get("net_wait") or {}
            was = wait.get("download") or {}
            wait["download"] = {"since": was.get("since") or time.time(), "tried": was.get("tried", 0),
                                "error": str(e)[:200]}
            rec.update(state="downloading", retry_at=time.time() + RETRY_SECONDS, fetch_tries=rec.get("fetch_tries", 0) + 1,
                       back=back, net_wait=wait,
                       note=f"Couldn't reach {name} to bring the rest of the lines back ({str(e)[:120].rstrip('.')}). "
                            "It carries on by itself as soon as the connection is back.")
        elif back:  # what came back while it worked is safe here
            rec.pop("net_wait", None)
            rec.update(state="ready", made=back, back=back, error=f"The cloud run ended before the rest could be "
                       f"collected ({str(e)[:160]}); {back} of {rec.get('lines', 0)} lines came back while it worked. "
                       "Finish makes the rest on this Mac, or Create in the cloud again sends them.")
        else:
            rec.pop("net_wait", None)
            rec.update(state="failed", error=f"Couldn't bring the lines back: {str(e)[:300]}")
    finally:
        _downloads.discard(str(where))
        shutil.rmtree(out, ignore_errors=True)
        still_mine = False
        with _lock:
            cur = read(where)
            if cur is not None and cur.get("run") == rec.get("run"):
                _write_or_note(where, rec)
                still_mine = True
        if still_mine and rec.get("state") in ("ready", "failed"):
            _noted(rec, "done" if rec["state"] == "ready" else "failed")
        done = getattr(BACKENDS.get(rec.get("backend", "kaggle")), "finished", None)
        if done and rec.get("run") and rec.get("state") != "downloading":  # (not while it's tried again)
            try:
                done(rec["run"])  # Runpod: the machine is deleted now that its lines are back
            except Exception as e:  # noqa: BLE001 - it deletes itself later anyway
                print(f"[cloud] couldn't finish {rec.get('run')}: {e}", flush=True)


def view(where):
    """The job as the book page shows it."""
    rec = check(where)
    if not rec:
        return {}
    keys = ("state", "backend", "gpu", "gpu_label", "max_cost", "url", "sent", "lines", "local", "seconds", "budget", "made", "error", "note", "speed",
            "stopped", "checked", "live", "back", "live_error", "sounds_made", "stitch_end", "stitch_error")
    out = {k: rec.get(k) for k in keys if k in rec}
    if rec.get("net_wait"):  # waiting on a lost connection since then (carried on from as soon as it's back)
        out["offline_since"] = min(w.get("since") or time.time() for w in rec["net_wait"].values())
    if rec.get("sounds"):
        out["sound_count"] = len(rec["sounds"])
    t = stitching(where)
    if t is not None:
        out["stitch"] = {"id": t.id, "status": (t.status or "").split("\n")[0], "progress": t.progress,
                         "eta": t.time_left(), "updated": t.updated}
    if rec.get("state") in WAITING + ("downloading",):
        out["awake"] = stay_awake()
    from . import voice_check
    out["voice_check"] = voice_check.summary(where)
    out.update(_pace(rec))
    free = free_bytes(where)
    if str(where) in DISK_FULL or (free is not None and free < DISK_LOW):
        out["disk_full"] = True  # lines can't come back, and chapters can't be put together, until there's room
    return out


def disk_per_second(sound):
    """Bytes on this Mac per second of a book's audio once it's back and made: its lines as FLAC (~24 KB/s measured),
    its chapters as 16-bit WAV (mono 48 KB/s; stereo 96 KB/s with sounds mixed in) and the audiobook
    (~8 KB/s as M4B)."""
    return 25_000 + (96_000 if sound else 48_000) + 8_000


DISK_LOW = 500 * 2 ** 20  # under this much free space, the book page says the disk is full


def free_bytes(where):
    """Free space on the disk a book's folder is on (None when it can't be told)."""
    try:
        return shutil.disk_usage(where).free
    except OSError:
        return None


def _pace(rec, now=None):
    """How far along a job out in the cloud is, for the book page: {"progress": 0..1, "eta": seconds left} from the
    lines back so far and how fast they've come (lines brought back as they're made), else {"elapsed", "limit"}
    (seconds since it was sent, and its time limit) so the page can say how long it has been going."""
    now = now or time.time()
    if rec.get("state") not in WAITING or not rec.get("sent"):
        return {}
    out = {"elapsed": now - rec["sent"], "limit": (rec.get("budget") or 0) * 3600 or None}
    lines, back, first = rec.get("lines") or 0, rec.get("back") or 0, rec.get("first_back")
    if rec.get("live") and lines:
        out["progress"] = min(back / lines, 1.0)
        at = rec.get("live_at") or now
        if first and back > first[1] and at - first[0] > 60:
            rate = (back - first[1]) / (at - first[0])  # lines a second, since lines first came back
            out["eta"] = max(0.0, (lines - back) / rate - (now - at))
    return out
