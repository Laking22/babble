"""Kokoro 82M: small and fast (about 3x faster than real time on an 8 GB M-series Mac). Built-in voices only.

Voice keys follow Kokoro's names: first letter = language/accent (a = American, b = British, …),
second = gender (f / m). Needs the `misaki` text package. On Windows and Linux it runs through PyTorch with Kokoro's own
package (the same model, hexgrad/Kokoro-82M).
"""
import importlib.util

import numpy as np

from ..runtime import HAS_MLX, slot
from .base import Engine, Voice
from . import torch_rt

REPO = "mlx-community/Kokoro-82M-bf16"
TORCH_REPO = "hexgrad/Kokoro-82M"
_ACCENT = {"a": "American English", "b": "British English", "e": "Spanish", "f": "French", "h": "Hindi",
           "i": "Italian", "j": "Japanese", "p": "Brazilian Portuguese", "z": "Mandarin"}
# English voices from Kokoro v1.0. Others are listed from the model folder once it's downloaded.
_KNOWN = ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "af_aoede", "af_kore", "af_nova", "af_river",
          "af_alloy", "af_jessica", "am_adam", "am_michael", "am_eric", "am_liam", "am_onyx", "am_puck", "am_fenrir",
          "am_echo", "bf_emma", "bf_isabella", "bf_alice", "bf_lily", "bm_george", "bm_lewis", "bm_daniel", "bm_fable"]


def _downloaded_voices():
    try:
        from huggingface_hub import scan_cache_dir
        for repo in scan_cache_dir().repos:
            if repo.repo_id == (REPO if HAS_MLX else TORCH_REPO):
                for rev in repo.revisions:
                    names = sorted(f.file_name.rsplit(".", 1)[0] for f in rev.files
                                   if f.file_path.parent.name == "voices")
                    if names:
                        return names
    except Exception:  # noqa: BLE001 - cache layout differs between hub versions; fall back to the known list
        pass
    return []


class Kokoro(Engine):
    id, name, tier, license = "kokoro", "Kokoro 82M", "lite", "Apache-2.0"
    short = "Kokoro"
    min_ram_gb = 4
    prefixes = ("kokoro",)
    features = frozenset({"batch"})
    hf_models = (REPO,) if HAS_MLX else (TORCH_REPO,)
    cloud = True
    pip = ("misaki==0.9.4", "num2words", "spacy", "phonemizer-fork", "espeakng-loader")  # misaki[en] would pull in torch

    def available(self):
        if not HAS_MLX:
            why = torch_rt.why_not("kokoro", "misaki")
            return (False, why) if why else (True, "")
        if importlib.util.find_spec("misaki") is None:
            return False, "needs the 'misaki' text package"
        return True, ""

    def voices(self):
        keys = _downloaded_voices() or _KNOWN
        out = []
        for k in keys:
            if len(k) < 4 or k[2] != "_":
                continue
            gender = {"f": "female", "m": "male"}.get(k[1], "unspecified")
            out.append(Voice(f"kokoro:{k}", k[3:].replace("_", " ").title(), gender, _ACCENT.get(k[0], k[0]), self.id))
        return out

    def model_key(self, value):
        return "kokoro"

    def note(self):
        return torch_rt.note() if not HAS_MLX else ""

    def speak(self, items, opts, stop=None):
        if not HAS_MLX:
            yield from self._torch_speak(items, stop)
            return
        from mlx_audio.tts.utils import load_model
        m = slot.get("kokoro", lambda: load_model(REPO))
        for i, it in enumerate(items):
            if stop is not None and stop.is_set():
                return
            key = it["voice"].partition(":")[2]
            parts = [np.asarray(r.audio, dtype=np.float32).ravel()
                     for r in m.generate(text=it["text"], voice=key, lang_code=key[0], speed=1.0)]
            yield i, np.concatenate(parts) if parts else np.zeros(0, np.float32)

    def _torch_speak(self, items, stop):
        """Kokoro's own package: one model, a text pipeline per language (the voice name's first letter)."""
        def load():
            from kokoro import KModel
            return {"model": KModel(repo_id=TORCH_REPO).to(torch_rt.device()).eval(), "pipes": {}}
        k = slot.get("kokoro-torch", load)
        for i, it in enumerate(items):
            if stop is not None and stop.is_set():
                return
            key = it["voice"].partition(":")[2]
            if key[0] not in k["pipes"]:
                import spacy.util
                if key[0] in "ab" and not spacy.util.is_package("en_core_web_sm"):
                    # (misaki would try to install it with pip, then exit the thread: the line would never finish)
                    raise ValueError("Kokoro's English text model (en_core_web_sm) isn't installed. "
                                     "Restart Babble to finish setting up.")
                from kokoro import KPipeline
                k["pipes"][key[0]] = KPipeline(lang_code=key[0], repo_id=TORCH_REPO, model=k["model"])
            parts = []
            for r in k["pipes"][key[0]](it["text"], voice=key, speed=1.0):
                audio = getattr(r, "audio", None)
                audio = r[2] if audio is None else audio
                if audio is not None:
                    parts.append(np.asarray(audio.detach().float().cpu() if hasattr(audio, "detach") else audio,
                                            dtype=np.float32).ravel())
            yield i, np.concatenate(parts) if parts else np.zeros(0, np.float32)
