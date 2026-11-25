"""Voice check: are the lines made in the cloud in the voice the book asked for?

A cloud run once made a whole book (6,128 lines, $3.60 on Modal, November 2026) in another voice, with no error
anywhere: the stock Qwen model's weights had got into the download cache and were loaded in the voice pack's place
(lines_runner.qwen_folder and pack_loaded prevent that now). This is the net for anything like it. Each voice's first lines in the cloud ("probes", made before the rest)
are compared with that voice as this Mac makes it, by voice fingerprint (Qwen Base's speaker encoder, as the Voices
tab's fingerprints); a run whose voice is wrong is stopped within minutes, and its lines are kept out of the book.

A fingerprint has a large part every voice shares, so it's measured against the average of Qwen's own speakers. A
take passes when it's at least PASS alike to its voice and closer to it than to any other voice known (Qwen's
speakers, and a trained pack's other voices). Tested on Hive Minds: right takes 0.71 to 0.92, and closer to their
voice than to any other in 24 of 24; the wrong ones -0.21 to 0.36, and 0 of 12.

Kept in <book>/cloud/voice-check.json; fingerprints are cached in HOME/voice-check (made once per voice).
"""
import hashlib
import json
import threading
import time
from pathlib import Path

import numpy as np

from . import books
from .paths import HOME

FILE = "voice-check.json"  # in <book>/cloud
FINGERPRINTS = HOME / "voice-check"
REF_TEXT = ("The lighthouse stood at the end of the harbour, and every evening its keeper climbed the stairs to "
            "light the lamp.")
PASS = 0.3   # a take at least this alike to its voice (and closer to it than to any other voice)…
NEED = 3     # …in most of this many takes of the voice: it's the right voice
PROBES = 3   # lines of each voice made first in the cloud, to be checked
_locks = {}


def voice_id(spec):
    """One voice of a run, whatever its lines' stability or direction: its engine, trained model and speaker."""
    return "|".join((spec.get("engine", ""), (spec.get("model") or {}).get("dir") or "",
                     spec.get("speaker") or spec.get("voice") or ""))


def pick(jobs):
    """Mark each voice's probes in a run's jobs ({key: spec, with "_voice": its voice in the app}): a few lines of
    an ordinary length from across the book, made before the rest. Returns {voice id: {"label", "voice",
    "probes", "more"}} (more: other lines of the voice, to look at when a probe didn't come back)."""
    from . import voices as vl
    by = {}
    for k, s in jobs.items():
        by.setdefault(voice_id(s), []).append(k)
    out = {}
    for vid, keys in by.items():
        keys.sort(key=lambda k: jobs[k].get("chapter", 0))
        def plain(t):  # (ordinary speech: no web addresses, mostly letters)
            return 40 <= len(t) <= 220 and "http" not in t and "www." not in t and \
                sum(c.isalpha() or c.isspace() for c in t) >= 0.85 * len(t)
        fair = [k for k in keys if plain(jobs[k].get("text") or "")] or keys
        probes = [fair[round(i * (len(fair) - 1) / max(PROBES - 1, 1))] for i in range(min(PROBES, len(fair)))]
        probes = list(dict.fromkeys(probes))
        for k in probes:
            jobs[k]["probe"] = True
        rest = [k for k in fair if k not in probes]
        voice = jobs[keys[0]].get("_voice") or ""
        out[vid] = {"label": vl.short_name(voice) if voice else vid, "voice": voice,
                    "instruct": jobs[keys[0]].get("_instruct") or "", "temperature": jobs[keys[0]].get("_temperature"),
                    "probes": probes, "more": rest[::max(1, len(rest) // 12)][:12], "state": "waiting"}
    return out


def start(where, jobs, kw):
    """At send: write the check down and make the fingerprints it needs in the background. (plan() already marked
    the probes in jobs; pick() picks the same ones again from a copy.)"""
    voices = pick({k: {f: x for f, x in s.items() if f != "probe"} for k, s in jobs.items()})
    _save(where, {"voices": voices, "ready": False, "language": kw.get("language", "english"), "made": time.time()})
    threading.Thread(target=prepare, args=(Path(where),), daemon=True, name="voice-check").start()


def read(where):
    try:
        return json.loads((Path(where) / "cloud" / FILE).read_text())
    except (OSError, ValueError):
        return {}


def _save(where, data):
    f = Path(where) / "cloud" / FILE
    tmp = f.with_name(f".{FILE}.part")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(f)


# ---------- fingerprints ----------

def _fp_file(key):
    return FINGERPRINTS / f"{hashlib.sha1(key.encode()).hexdigest()[:16]}.npy"


def _embed(wav):
    from .runtime import on_mlx_thread

    @on_mlx_thread
    def run(w):
        from .runtime import HAS_MLX
        if not HAS_MLX:  # (Windows/Linux: the same Base model's speaker encoder, through PyTorch)
            from .tts import qwen_torch
            item = qwen_torch.model("clone").create_voice_clone_prompt(
                ref_audio=(np.asarray(w, dtype=np.float32), 24000), ref_text=None, x_vector_only_mode=True)[0]
            return item.ref_spk_embedding.detach().float().cpu().numpy().reshape(-1)
        import mlx.core as mx
        from .tts.qwen import model
        e = model("clone").extract_speaker_embedding(mx.array(np.asarray(w, dtype=np.float32)), sr=24000)
        return np.array(e, dtype=np.float32).reshape(-1)
    return run(wav)


def _voice_stamp(voice):
    """What a voice's fingerprint is kept under: its name, and its trained model's files' time (trained again,
    it's measured again)."""
    from . import tts
    from .cloud_book import _dir_stamp
    kind, _, key = voice.partition(":")
    e = tts.ENGINES.get(kind) if kind in ("qwen", "orpheus") else None
    ft = getattr(e, "tuned", {}).get(key) if e else None
    return f"{voice}@{_dir_stamp(ft[3])}" if ft else voice


def fingerprint(voice, language="english", instruct=""):
    """A voice's fingerprint as this Mac makes it (REF_TEXT spoken in it, at the default stability: a calm take
    is its plainest), cached; or the fingerprint of its saved clip when this Mac can't make it. None when neither
    is possible."""
    from . import engine as en
    from . import voices_lib as lib
    from .audio import read_mono
    from .cloud_book import _runs_here
    from .books import temp_from_stability
    key = f"{_voice_stamp(voice)}|{instruct}"
    f = _fp_file(key)
    if f.exists():
        return np.load(f)
    FINGERPRINTS.mkdir(parents=True, exist_ok=True)
    kind, _, name = voice.partition(":")
    if _runs_here(voice):
        out = FINGERPRINTS / f"{f.stem}.wav"
        item = dict(text=REF_TEXT, voice=voice, instruct=instruct, speed=1.0, pause=0.0)
        for path, _ in en.render([item], kind="voice-check", language=language,
                                 temperature=temp_from_stability(55), path=str(out), sound=False):
            pass
        wav = read_mono(str(out))
        out.unlink(missing_ok=True)
    elif kind in ("my", "higgs", "drama") and lib.wav_path(name).exists():  # (a cloned voice: its own clip)
        wav = read_mono(lib.wav_path(name), max_seconds=20)
    else:
        return None
    e = _embed(wav)
    np.save(f, e)
    return e


def background():
    """Fingerprints of Qwen's own speakers: what every voice is measured against."""
    from . import tts
    voices = sorted(v.value for v in tts.ENGINES["qwen"].voices() if v.value.startswith("builtin:"))
    return {v: fingerprint(v) for v in voices}


def siblings(voice):
    """The other voices of a trained pack (a take closer to one of them than to its own is in the wrong voice)."""
    from . import tts
    kind, _, key = voice.partition(":")
    e = tts.ENGINES.get(kind) if kind in ("qwen", "orpheus") else None
    tuned = getattr(e, "tuned", {}) if e else {}
    ft = tuned.get(key)
    return [f"{kind}:{k}" for k, x in tuned.items() if ft and x[3] == ft[3] and k != key]


def prepare(where):
    """Make the fingerprints the check needs (in the background, while the cloud starts): each voice's own as
    this Mac makes it, and the voices it's compared with."""
    data = read(where)
    if not data:
        return
    try:
        bg = background()
        for vid, v in data["voices"].items():
            if not v.get("voice"):
                v.update(state="unchecked", why="the voice isn't known on this Mac")
                continue
            target = fingerprint(v["voice"], data.get("language", "english"), v.get("instruct") or "")
            if target is None:
                v.update(state="unchecked", why="this Mac can't make this voice to compare with")
                continue
            for s in siblings(v["voice"]):
                fingerprint(s, data.get("language", "english"))
            v["fingerprint"] = _voice_stamp(v["voice"]) + "|" + (v.get("instruct") or "")
            v["others"] = [_voice_stamp(o) + "|" for o in list(bg) + siblings(v["voice"]) if o != v["voice"]]
        data["background"] = [_voice_stamp(o) + "|" for o in bg]
        data["ready"] = True
    except Exception as e:  # noqa: BLE001 - a check that can't be made doesn't hold the book up
        print(f"[voice check] couldn't get ready for {Path(where).name}: {e}", flush=True)
        for v in data.get("voices", {}).values():
            if v.get("state") == "waiting":
                v.update(state="unchecked", why=f"the check couldn't be made ({str(e)[:120]})")
    fresh = read(where)  # (a check that ran meanwhile keeps its results)
    for vid, v in data.get("voices", {}).items():
        if fresh.get("voices", {}).get(vid, {}).get("state") not in (None, "waiting"):
            data["voices"][vid] = dict(v, **{k: fresh["voices"][vid][k] for k in ("state", "why", "scores")
                                             if k in fresh["voices"][vid]})
    _save(where, data)
    print(f"[voice check] ready for {Path(where).name}: " + ", ".join(
        f"{v['label']} ({v['state']})" for v in data.get("voices", {}).values()), flush=True)


# ---------- checking ----------

def _load(key):
    f = _fp_file(key)
    return np.load(f) if f.exists() else None


def score(wav, target, others, center):
    """(how alike a take is to its voice, how alike to the closest other voice), both measured from the average."""
    unit = lambda e: (e - center) / (np.linalg.norm(e - center) + 1e-9)  # noqa: E731
    e = unit(_embed(wav))
    return float(e @ unit(target)), max((float(e @ unit(o)) for o in others), default=-1.0)


def check(where, final=False):
    """Look at the cloud's takes that are back: each voice with NEED of its takes here is decided (right or wrong).
    final: the run is over, decide with what's here. Returns [(voice label, why)] for the voices found wrong."""
    from .audio import read_mono
    where = Path(where)
    lock = _locks.setdefault(str(where), threading.Lock())
    if not lock.acquire(blocking=False):
        return []
    try:
        data = read(where)
        if not data.get("ready"):
            return []
        bg = [x for x in (_load(k) for k in data.get("background") or []) if x is not None]
        if not bg:
            return []
        center = np.mean(bg, axis=0)
        cache = where / books.CLOUD_LINES
        wrong, changed = [], False
        for vid, v in data["voices"].items():
            if v.get("state") != "waiting":
                continue
            target = _load(v.get("fingerprint") or "")
            if target is None:
                continue
            back = [k for k in v["probes"] + v.get("more", []) if (cache / f"{k}.flac").exists()][:NEED]
            if len(back) < (1 if final else NEED):
                continue
            others = [x for x in (_load(k) for k in v.get("others") or []) if x is not None]
            scores = [score(read_mono(str(cache / f"{k}.flac")), target, others, center) for k in back]
            good = sum(s >= PASS and s > o for s, o in scores)
            v["scores"] = [[round(s, 2), round(o, 2)] for s, o in scores]
            if good * 2 > len(scores):
                v["state"] = "ok"
            else:
                v["state"] = "wrong"
                v["why"] = (f"{len(scores) - good} of the first {len(scores)} lines don't sound like {v['label']} "
                            f"(alike {min(s for s, _ in scores):.2f} to {max(s for s, _ in scores):.2f}; the right "
                            f"voice is {PASS} or more)")
                wrong.append((v["label"], v["why"]))
            changed = True
            print(f"[voice check] {where.name}: {v['label']} is {v['state']} ({v['scores']})", flush=True)
        if changed:
            _save(where, data)
        return wrong
    finally:
        lock.release()


def cleared(where):
    """Whether the book's cloud lines may be put together: no check, or every voice checked right (or one this
    Mac can't check). A voice still waiting holds the chapters back; a wrong one never clears."""
    data = read(where)
    return all(v.get("state") in ("ok", "unchecked") for v in data.get("voices", {}).values())


def summary(where):
    """For the book page: {"state": "checking" / "ok" / "wrong" / "unchecked", "voices": [...]} or None."""
    data = read(where)
    vs = data.get("voices") or {}
    if not vs:
        return None
    states = {v.get("state") for v in vs.values()}
    state = ("wrong" if "wrong" in states else "checking" if "waiting" in states
             else "ok" if states <= {"ok"} else "unchecked" if states <= {"unchecked"} else "ok")
    return {"state": state, "voices": [{"label": v["label"], "state": v.get("state"), "why": v.get("why", "")}
                                       for v in vs.values()]}
