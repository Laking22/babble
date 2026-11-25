"""Fish Audio Drama 3: the Ultra tier, cloud only (Create in the cloud on Modal, an H100).

The most expressive model the app offers: clones any saved voice from its sample clip and transcript (3-60 s), and
follows directions in brackets anywhere in a line ("[whispering] Don't move.", "[choked up, crying quietly]").
The app puts a line's emotion there. 35B parameters (3B used per step, ~37 GB), so no Mac runs it: its lines are
made by kaggle_render/drama_worker.py with vLLM-Omni on a Modal H100. Tested November 2026: about 6.7x real time
one line at a time, 25x eight at once.

License: Fish Audio Community License v1.0 (the model card's tag says fish-audio-research-license: check which
applies before release). Free for commercial use under US$500,000 annual revenue; not to be offered as a TTS /
voice-cloning service or API without Fish Audio's licence; no cloning a real person's voice without consent.
"""
from .. import voices_lib as lib
from .base import Engine, Voice

CLIP_SECONDS = (3.0, 60.0)  # Drama 3 takes up to 60 s of reference audio


def _clip_ok(name):
    import soundfile as sf
    try:
        return CLIP_SECONDS[0] <= sf.info(str(lib.wav_path(name))).duration <= CLIP_SECONDS[1]
    except RuntimeError:
        return False


class Drama(Engine):
    id, name, tier = "drama", "Fish Audio Drama 3", "ultra"
    short = "Drama 3"
    license = ("Fish Audio Community License: free under US$500,000 annual revenue; not as a TTS service; "
               "no cloning real people without consent")
    min_ram_gb = 1000  # (no Mac: cloud only)
    prefixes = ("drama",)
    features = frozenset({"clone", "emotion"})
    cloud = True  # Create in the cloud makes its lines (Modal, an H100)

    def available(self):
        return False, "Ultra: runs only in the cloud. Create in the cloud on Modal makes its lines (on an H100)."

    def voices(self):
        return [Voice(f"drama:{v['name']}", v["name"], v["gender"], "cloned by Drama 3 from your saved voice",
                      self.id, ["saved"])
                for v in lib.library() if v["transcript"] and _clip_ok(v["name"])]

    def prepare(self, items, opts):
        raise ValueError("Drama 3 voices are cloud only: choose Modal in the ≡ menu next to Create, then Create in "
                         "the cloud. (Lines it already made in the cloud are used as they are.)")

    def speak(self, items, opts, stop=None):
        raise NotImplementedError("Drama 3 runs only in the cloud")
