"""Make sound effects, backgrounds and music from a description with Stable Audio 3 (Stability AI, MLX).

Runs Stability's own MLX script (vendor/stable_audio_3, MIT) in a separate process, after the voice model is let
go, so the two never share memory and everything is freed when it finishes.

Models (all share the T5Gemma text encoder):
  effects and backgrounds  Small SFX    (0.92 GB + SAME-S decoder) - 8 GB Mac: 2.6-4.6 s per sound, peak 1.3 GB
  music                    Small Music  (0.92 GB + SAME-S decoder) - 8 GB Mac: 45 s of music in 8 s, peak 1.4 GB
  music, 16 GB Macs and up Medium       (2.9 GB + SAME-L decoder 1.7 GB, ~5.9 GB in all) - better and longer;
                                        not tested here (an 8 GB Mac can't hold it next to everything else)
The Small SFX model plays music poorly (it's made for effects), which is why music has its own models.

Weights are downloaded on first use and are under the Stability AI Community License (free under USD 1M annual
revenue) and T5Gemma under the Gemma Terms of Use; generated sounds are labelled with that in the library.
"""
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .paths import HOME
from .runtime import SR, free, on_mlx_thread, slot

VENDOR = Path(__file__).resolve().parent / "vendor" / "stable_audio_3"
WEIGHTS_HOME = HOME / "sound-generator"
REPO = "stabilityai/stable-audio-3-optimized"
T5 = "MLX/t5gemma_f16.npz"
# model -> (DiT weights, decoder name, decoder weights, label)
MODELS = {
    "sm-sfx": ("MLX/dit_sm-sfx_f16.npz", "same-s", "MLX/same_s_decoder_f32.npz", "Stable Audio 3 Small SFX"),
    "sm-music": ("MLX/dit_sm-music_f16.npz", "same-s", "MLX/same_s_decoder_f32.npz", "Stable Audio 3 Small Music"),
    "medium": ("MLX/dit_medium_f16.npz", "same-l", "MLX/same_l_decoder_f32.npz", "Stable Audio 3 Medium"),
}
MEDIUM_RAM_GB = 16
MAX_SECONDS = {"sm-sfx": 120.0, "sm-music": 120.0, "medium": 380.0}
# Developer testing only: force a music model ("sm-music" or "medium") whatever the Mac. Not set for normal use.
DEV_MUSIC_MODEL = "NARRATOR_MUSIC_MODEL"
LICENSE = "Generated with Stable Audio 3 (Stability AI Community License)"
FILES = [MODELS["sm-sfx"][0], T5, MODELS["sm-sfx"][2]]  # the effects model (what the Library page shows)
KINDS = ("effect", "ambience", "music")


def available():
    return (VENDOR / "scripts" / "sa3_mlx.py").exists()


def medium_ok():
    from .tts.base import ram_gb
    return bool(os.environ.get(DEV_MUSIC_MODEL)) or ram_gb() >= MEDIUM_RAM_GB


def music_model(quality=None):
    """"medium" (better) on Macs with 16 GB or more unless Small is picked, else "sm-music"."""
    from . import settings
    forced = os.environ.get(DEV_MUSIC_MODEL)
    if forced in MODELS:
        return forced
    quality = quality or settings.load().get("music_quality", "auto")
    if quality == "small" or not medium_ok():
        return "sm-music"
    return "medium"


def model_for(kind, quality=None):
    return music_model(quality) if kind == "music" else "sm-sfx"


def _files(model):
    dit, _, decoder, _ = MODELS[model]
    return [dit, T5, decoder]


def downloaded(model="sm-sfx"):
    from huggingface_hub import try_to_load_from_cache
    return all(isinstance(try_to_load_from_cache(REPO, f), str) for f in _files(model))


def download(model="sm-sfx", log=print):
    from huggingface_hub import hf_hub_download
    size = {"sm-sfx": "1.7 GB", "sm-music": "0.9 GB more", "medium": "4.6 GB more"}[model]
    log(f"Downloading {MODELS[model][3]} ({size}, once)…")
    for f in _files(model):
        hf_hub_download(REPO, f)
    log("Sound generator ready.")


def length_for(desc, kind):
    """Backgrounds loop, so 20 s is plenty; music loops too, but a longer piece repeats less often; effects get a
    little longer when they describe something drawn out."""
    if kind == "music":
        return 60.0
    if kind == "ambience":
        return 20.0
    long = re.search(r"roll|rumbl|storm|crowd|engine|running|footsteps|stairs|march|gallop|applause|siren", desc, re.I)
    return 6.0 if long else 3.0


def _trim(a, floor_db=-50.0):
    """Cut silence after an effect ends (the model fills the whole length), keeping a short natural tail."""
    hop = SR // 100
    n = len(a) // hop
    if n < 3:
        return a
    m = a.mean(axis=1) if a.ndim == 2 else a
    db = 20 * np.log10(np.sqrt((m[: n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)
    loud = np.where(db > max(db.max() - 45, floor_db))[0]
    if not len(loud):
        return a
    start = max(0, loud[0] * hop - hop * 2)
    end = min(len(a), (loud[-1] + 1) * hop + int(SR * 0.25))
    out = a[start:end].copy()
    fade = min(int(SR * 0.05), len(out) // 4)
    if fade:
        ramp = np.linspace(1, 0, fade, dtype=np.float32)
        out[-fade:] *= ramp[:, None] if out.ndim == 2 else ramp
    return out


def prompt_for(desc, kind):
    if kind == "ambience":
        return desc.strip() + ", ambience, continuous background"
    if kind == "music":
        return desc.strip() + ", instrumental"
    return desc.strip()


@on_mlx_thread
def generate(desc, kind="effect", seconds=None, seed=None, quality=None, log=print):
    """Description -> float32 audio at the app's sample rate (stereo when the sound is)."""
    if not available():
        raise RuntimeError("The sound generator isn't installed with this copy of the app.")
    model = model_for(kind, quality)
    if not downloaded(model):
        download(model, log=log)
    slot.release()  # let the voice model go first: the generator runs in its own process
    free()
    from .tasks import Stopped, stop_wanted
    if stop_wanted():
        raise Stopped()
    seconds = min(float(seconds or length_for(desc, kind)), MAX_SECONDS[model])
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "sound.wav"
        env = dict(os.environ, SA3_HOME=str(WEIGHTS_HOME))
        args = [sys.executable, str(VENDOR / "scripts" / "sa3_mlx.py"), "--prompt", prompt_for(desc, kind),
                "--dit", model, "--decoder", MODELS[model][1], "--seconds", f"{seconds:g}", "--out", str(out)]
        if seed is not None:
            args += ["--seed", str(int(seed))]
        log(f"Making “{desc}” ({seconds:g} s, {MODELS[model][3]})…")
        proc = subprocess.Popen(args, cwd=VENDOR, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, stdin=subprocess.DEVNULL)
        waited = 0.0
        while True:  # Stop kills the generator; it gives up after 30 minutes
            try:
                stdout, stderr = proc.communicate(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                waited += 0.5
                if stop_wanted() or waited > 1800:
                    proc.kill()
                    proc.communicate()
                    if stop_wanted():
                        raise Stopped() from None
                    raise RuntimeError("The sound generator took over 30 minutes and was stopped.") from None
        if proc.returncode != 0 or not out.exists():
            tail = re.sub(r"\x1b\[[0-9;]*m", "", (stderr or stdout or "")).strip().splitlines()[-3:]
            raise RuntimeError("The sound generator failed: " + " / ".join(tail))
        return processed(out, kind)


def processed(path, kind):
    """A sound as the generator wrote it (44.1 kHz stereo) -> the app's sample rate, mono when it's really mono,
    an effect's silent end cut. (Also for sounds made in the cloud by the same generator.)"""
    import soundfile as sf
    from .audio import resample
    from .sfx import mono_or_stereo
    raw, sr = sf.read(str(path), dtype="float32", always_2d=True)  # Stable Audio makes 44.1 kHz stereo: keep it
    audio = mono_or_stereo(np.stack([resample(raw[:, ch], sr) for ch in range(min(2, raw.shape[1]))], axis=1))
    return _trim(audio) if kind == "effect" else audio


# The check on a sound just made: CLAP listens to it and scores how well it matches its description. Under the bar it's
# made again with a new random start (TRIES in all) and the best one is kept. The bars are low on purpose: CLAP is
# rough with long descriptions (a fine music cue can score 0.15), so only clearly wrong sounds are made again (a
# "water splashing" that scored -0.05, a "flame igniting" at 0.03)
CHECK_MIN = {"effect": 0.12, "ambience": 0.12, "music": 0.10}
TRIES = 3


def score(desc, audio):
    """How much a sound sounds like its description (CLAP, about -0.1 to 0.5), or None without the model."""
    from . import clap
    if not (clap.available() and clap.downloaded()):
        return None
    a = np.asarray(audio, dtype=np.float32)
    a = a.mean(axis=1) if a.ndim == 2 else a
    t = clap.text_embedding([f"the sound of {desc}"])[0]
    return float(clap.audio_embedding(a, SR) @ t)


def _checked(desc, kind, audio, seconds, quality, log):
    """The sound, or a better try when it doesn't sound like its description."""
    from . import tasks
    try:
        best = score(desc, audio)
    except Exception as e:  # noqa: BLE001 - the check is a bonus; the sound is kept as made
        print(f"[sfxgen] couldn't check “{desc}”: {e}", flush=True)
        return audio
    bar = CHECK_MIN.get(kind, 0.12)
    tries = 1
    while best is not None and best < bar and tries < TRIES:
        tries += 1
        msg = f"“{desc}” didn't sound right, so it's being made again (try {tries} of {TRIES})…"
        log(msg)
        tasks.note(msg)
        again = generate(desc, kind, seconds, None, quality, log=log)
        s = score(desc, again)
        if s is not None and s > best:
            audio, best = again, s
    if best is not None:
        print(f"[sfxgen] “{desc}”: {best:.2f} after {tries} try(s)", flush=True)
    return audio


def make(desc, kind="effect", seconds=None, seed=None, quality=None, log=print):
    """Generate a sound and add it to the library (made again when it doesn't sound like its description, unless a
    seed was asked for). Returns the library entry."""
    audio = generate(desc, kind, seconds, seed, quality, log=log)
    if seed is None:
        audio = _checked(desc, kind, audio, seconds, quality, log)
    return keep(desc, kind, audio, model_for(kind, quality))


def keep(desc, kind, audio, model, where="made in Babble"):
    """Add a made sound to the library, named from its description."""
    from . import sfx
    from . import naming
    desc = re.sub(r"\s+", " ", desc).strip()
    name = sfx.unique_name(naming.short_name(desc))  # "A quick … explosion with shattering glass" -> "Explosion Shatter"
    return sfx.add(audio, name, kind, tags=sfx._words(desc), license=LICENSE, description=desc,
                   source=f"{MODELS[model][3]}, {where}", credit="Stability AI")


def best_take(desc, kind, paths):
    """Of a sound's takes made in the cloud, the one CLAP says sounds most like its description (processed), or
    the first when there's no CLAP."""
    best, best_score = None, None
    for p in paths:
        audio = processed(p, kind)
        try:
            sc = score(desc, audio)
        except Exception:  # noqa: BLE001 - the check is a bonus
            sc = None
        if best is None or (sc is not None and (best_score is None or sc > best_score)):
            best, best_score = audio, sc
    return best


# On a cloud GPU (Create in the cloud with "Make sounds in the cloud"): exactly what this Mac would make, so the
# settings made here hold there too: the same model for each kind (Music quality, as this Mac picks it), the same
# length and wording, and as many tries as the check here allows; the Mac keeps the take that sounds most like its
# description, as it does with its own tries
CLOUD_TAKES = TRIES


def cloud_model(kind, quality=None):
    return model_for(kind, quality)
