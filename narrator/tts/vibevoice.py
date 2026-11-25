"""VibeVoice 1.5B (Microsoft): clones any saved voice from its sample clip alone (no transcript), 4-bit on this Mac.

Tested on an 8 GB Mac with Ray's voice: Whisper heard every word (3/3 lines), voice match 0.987-0.988 (the same as
the Ray fine-tuned Orpheus model, from one 8-10 s clip and no training), loads at 1.8 GB, peaks at 3.4 GB, and the
first line ran at 0.89x real time. Later lines slowed to ~0.15x on that Mac because it was already swapping.

Setup (Library → Engines → Download): there's no ready-made MLX copy of the 1.5B model, so the app downloads
Microsoft's original (5.4 GB), gives the language model its own output layer (the 1.5B model shares it with the input
embeddings, which mlx-audio's converter doesn't handle), and converts it to 4-bit (1.7 GB) in the data folder.

License: MIT (weights). Microsoft's model card says it's intended for research and does not recommend commercial or
real-world use without further testing; their own app adds an audible AI notice and a watermark, this port doesn't.
"""
import json
import os
import shutil

import numpy as np

from .. import voices_lib as lib
from ..audio import read_mono, resample
from ..paths import HOME
from ..runtime import slot
from .base import Engine, Voice, ram_gb

SOURCE = "microsoft/VibeVoice-1.5B"
MODEL_DIR = HOME / "engines" / "vibevoice-1.5b-4bit"
CLIP_SECONDS = (3.0, 30.0)
REF_SECONDS = 8  # 8 s cloned Ray perfectly (voice match 0.987); longer samples only cost memory (4 s: 2.6 GB, 8 s: 3.2 GB)


def installed():
    return (MODEL_DIR / "config.json").exists() and any(MODEL_DIR.glob("*.safetensors"))


def install(log=print):
    """Download Microsoft's VibeVoice 1.5B and make a 4-bit MLX copy in the data folder (about a minute)."""
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    if installed():
        log("VibeVoice is already set up.")
        return MODEL_DIR
    log("Downloading VibeVoice 1.5B from Microsoft (5.4 GB, once)…")
    src = snapshot_download(SOURCE)
    staging = MODEL_DIR.parent / "vibevoice-1.5b-source"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for f in os.listdir(src):
        if f != "model.safetensors.index.json":
            os.symlink(os.path.join(src, f), staging / f)
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    key = "model.language_model.embed_tokens.weight"
    w = mx.load(os.path.join(src, idx["weight_map"][key]))[key]
    mx.save_safetensors(str(staging / "lm_head.safetensors"), {"lm_head.weight": w})
    idx["weight_map"]["lm_head.weight"] = "lm_head.safetensors"
    json.dump(idx, open(staging / "model.safetensors.index.json", "w"), indent=1)
    del w
    log("Converting to 4-bit for this Mac (about a minute)…")
    from mlx_audio.convert import convert
    convert(hf_path=str(staging), mlx_path=str(MODEL_DIR), quantize=True, q_bits=4)
    shutil.rmtree(staging, ignore_errors=True)
    log(f"VibeVoice ready ({sum(p.stat().st_size for p in MODEL_DIR.glob('*')) / 1e9:.1f} GB).")
    return MODEL_DIR


def _clip_seconds(name):
    import soundfile as sf
    try:
        return sf.info(str(lib.wav_path(name))).duration
    except RuntimeError:
        return 0.0


class VibeVoice(Engine):
    id, name, tier = "vibevoice", "VibeVoice 1.5B", "standard"
    short = "VibeVoice"
    license = "MIT (Microsoft says research use; not recommended for commercial use without more testing)"
    min_ram_gb = 8
    prefixes = ("vibevoice",)
    features = frozenset({"clone"})
    hf_models = (SOURCE,)
    local_dirs = (MODEL_DIR,)  # its 4-bit copy (deleted with it)

    def install(self, log=print):
        return install(log=log)

    def note(self):
        # Measured on an 8 GB Mac: works (0.73x real time) but free memory dipped to 7% while generating
        if ram_gb() < 16:
            return f"Uses most of this Mac's {ram_gb():.0f} GB of memory: close other apps (especially browsers) if it slows down."
        return ""

    def available(self):
        from ..runtime import HAS_MLX
        if not HAS_MLX:
            return False, "Runs on Macs only for now."
        if not installed():
            return False, "Not set up yet: Library → Engines → VibeVoice → Download models."
        return True, ""

    def voices(self):
        return [Voice(f"vibevoice:{v['name']}", v["name"], v["gender"], "cloned by VibeVoice from your saved voice",
                      self.id, ["saved"])
                for v in lib.library() if CLIP_SECONDS[0] <= _clip_seconds(v["name"]) <= CLIP_SECONDS[1]]

    def model_key(self, value):
        return str(MODEL_DIR)

    def prepare(self, items, opts):
        ok, why = self.available()
        if not ok:
            raise ValueError(f"VibeVoice: {why}")
        for it in items:
            name = it["voice"].partition(":")[2]
            if not lib.wav_path(name).exists():
                raise ValueError(f"VibeVoice: there's no saved voice called “{name}”.")

    def speak(self, items, opts, stop=None):
        import mlx.core as mx
        from mlx_audio.tts.utils import load_model
        m = slot.get(f"vibevoice:{MODEL_DIR}", lambda: load_model(str(MODEL_DIR)))
        sr = int(getattr(m, "sample_rate", 24000))
        refs = {}
        strip = opts.get("strip_tags")
        for i, it in enumerate(items):
            if stop is not None and stop.is_set():
                return
            name = it["voice"].partition(":")[2]
            if name not in refs:
                # the first 8 s is plenty to clone from; a longer sample makes every line's prompt (and memory) bigger
                refs[name] = read_mono(str(lib.wav_path(name)), 24000)[: 24000 * REF_SECONDS]
            text = strip.sub("", it["text"]) if strip is not None else it["text"]
            text = " ".join(text.split())  # one speaker turn: a line break would read as a new turn
            parts = [np.array(mx.array(r.audio).astype(mx.float32)).flatten()
                     for r in m.generate(text=text, ref_audio=refs[name], verbose=False)]
            audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
            yield i, resample(audio, sr) if sr != 24000 else audio
