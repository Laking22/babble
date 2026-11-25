"""Higgs Audio v2 (Boson AI, 3B): the high tier, for Macs with 16 GB of memory or more.

Clones any saved voice from its sample clip and transcript (best with 5-15 s of clean speech), so every voice in
Voices → My voices with a 3-20 s clip is offered as "<name> · Higgs".

Memory on an 8 GB Mac (where it's blocked): the 6-bit model (4.75 GB) loads at 5.5 GB and then swapped so
hard it never finished a line; a 4-bit version loads at 4.05 GB and still ran out while generating. So it needs
16 GB: 6-bit on 16-31 GB Macs, 8-bit (6.2 GB, a little better) on 32 GB and up.

License: Boson Higgs Audio 2 Community License (based on the Llama 3 license). Commercial use is fine up to 100,000
annual active users (more needs a licence from Boson); products must show "Built with Higgs Materials licensed from
Boson AI USA, Inc." and ship the license text.
"""
import os

import numpy as np

from .. import voices_lib as lib
from ..audio import read_mono, resample
from ..runtime import slot
from .base import Engine, Voice, line_temperature, ram_gb

REPOS = {"q8": "mlx-community/higgs-audio-v2-3B-mlx-q8", "q6": "mlx-community/higgs-audio-v2-3B-mlx-q6"}
CODEC = "mlx-community/higgs-audio-v2-tokenizer"
MIN_RAM_GB = 16
CLIP_SECONDS = (3.0, 20.0)  # sample clips outside this make poor (or very slow) clones
# Developer testing only: point at a local model folder to run it on a smaller Mac. Not set for normal use.
DEV_MODEL = "NARRATOR_HIGGS_MODEL"


def model_repo():
    return os.environ.get(DEV_MODEL) or (REPOS["q8"] if ram_gb() >= 32 else REPOS["q6"])


def _clip_ok(name):
    import soundfile as sf
    try:
        return CLIP_SECONDS[0] <= sf.info(str(lib.wav_path(name))).duration <= CLIP_SECONDS[1]
    except RuntimeError:
        return False


class Higgs(Engine):
    id, name, tier = "higgs", "Higgs Audio v2 3B", "pro"
    short = "Higgs"
    license = ("Boson Higgs Audio 2 Community License (Llama 3 based): free up to 100,000 annual active users; "
               "credit “Built with Higgs Materials licensed from Boson AI USA, Inc.”")
    min_ram_gb = MIN_RAM_GB
    prefixes = ("higgs",)
    features = frozenset({"clone"})
    hf_models = (model_repo(), CODEC)
    cloud = True  # Create in the cloud makes its lines on a 24 GB GPU (kaggle_render/higgs_worker.py)

    def available(self):
        from ..runtime import HAS_MLX
        if not HAS_MLX:  # (its PyTorch code needs an older transformers than Qwen's: in the cloud it runs apart)
            return False, "On Windows and Linux, Higgs voices are made in the cloud for now (Create in the cloud)."
        if os.environ.get(DEV_MODEL):
            return True, ""
        have = ram_gb()
        if have < MIN_RAM_GB:
            return False, f"Needs a Mac with {MIN_RAM_GB} GB of memory or more (this one has {have:.0f} GB)."
        return True, ""

    def voices(self):
        return [Voice(f"higgs:{v['name']}", v["name"], v["gender"], "cloned by Higgs from your saved voice", self.id,
                      ["saved"])
                for v in lib.library() if v["transcript"] and _clip_ok(v["name"])]

    def model_key(self, value):
        return model_repo()

    def prepare(self, items, opts):
        ok, why = self.available()
        if not ok:
            raise ValueError(f"Higgs Audio: {why} On this Mac its voices are cloud only: choose a cloud GPU other "
                             "than Kaggle in the ≡ menu next to Create, then Create in the cloud.")
        for it in items:
            name = it["voice"].partition(":")[2]
            if not lib.wav_path(name).exists():
                raise ValueError(f"Higgs Audio: there's no saved voice called “{name}”.")
            if not lib.info(name)["transcript"]:
                raise ValueError(f"Higgs Audio needs the transcript of “{name}”'s sample. Add it in Voices → My voices.")

    def speak(self, items, opts, stop=None):
        from mlx_audio.tts.utils import load_model
        path = model_repo()
        m = slot.get(f"higgs:{path}", lambda: load_model(path))
        sr = int(getattr(m, "sample_rate", 24000))
        refs = {}
        strip = opts.get("strip_tags")
        for i, it in enumerate(items):
            if stop is not None and stop.is_set():
                return
            name = it["voice"].partition(":")[2]
            if name not in refs:
                refs[name] = (read_mono(str(lib.wav_path(name)), 24000), lib.info(name)["transcript"])
            text = strip.sub("", it["text"]) if strip is not None else it["text"]
            # Boson's recommended sampling (temperature 0.7, top_p 0.95); ~25 frames a second, so allow about
            # twice the frames a narrator would need for the line (min 10 s, max 80 s)
            frames = min(2000, max(250, int(len(text) / 14 * 25 * 2)))
            parts = [np.array(r.audio, dtype=np.float32).flatten()
                     for r in m.generate(text=text, ref_audio=refs[name][0], ref_text=refs[name][1],
                                         temperature=line_temperature(it, opts, 0.7), top_p=0.95, max_new_frames=frames,
                                         verbose=False)]
            audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
            yield i, resample(audio, sr) if sr != 24000 else audio
