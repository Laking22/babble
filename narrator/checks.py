"""What Babble needs to run, checked when the app opens: GET /api/checks.

Each problem names the parts of the app it affects, then the technical thing that's missing, for the app to show
as "Saving audiobooks, … (ffmpeg not found)". Only things that won't fix themselves are reported: models that
download on first use aren't.
"""
import importlib.util
import os
import shutil
import tempfile

from .paths import HOME, OUTPUTS

MIN_FREE = 2e9  # bytes free on the data disk below which making audio may fail


def _tool(name, fallback=None):
    return shutil.which(name) or (fallback if fallback and os.path.exists(fallback) else None)


def _package(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _metal():
    try:
        import mlx.core as mx
        return mx.metal.is_available()
    except Exception:  # noqa: BLE001 - a broken MLX install is reported the same way
        return False


def _writable(folder):
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=folder):
            pass
        return True
    except OSError:
        return False


def run():
    """[{"area": what won't work, "missing": the technical thing}]"""
    from . import sfxgen
    from .book import CALIBRE
    problems = []

    def need(ok, area, missing):
        if not ok:
            problems.append({"area": area, "missing": missing})

    import sys
    if sys.platform == "darwin":
        need(_package("mlx") and _metal(), "Voices, Whisper and sound generation", "MLX can't use the Mac's GPU (Metal)")
        need(_package("mlx_audio"), "Voices and speech recognition", "Python package mlx-audio not installed")
    else:
        need(_package("torch"), "Voices", "PyTorch isn't set up: Models → Set up voices")
    need(_package("soundfile"), "Reading and saving audio", "Python package soundfile not installed")
    need(_package("huggingface_hub"), "Downloading models", "Python package huggingface_hub not installed")
    need(_tool("ffmpeg"), "Saving audiobooks, adding your own sound files, speed changes and training data",
         "ffmpeg not found")
    need(_tool("ffprobe"), "Training data from audiobooks", "ffprobe not found")
    need(sfxgen.available(), "Sound effect, background and music generation",
         "Stable Audio 3 script missing: narrator/vendor/stable_audio_3")
    need(_package("scipy"), "Training data from audiobooks", "Python package scipy not installed")
    need(_tool("ebook-convert", CALIBRE), "Importing ebooks other than EPUB and TXT",
         "calibre's ebook-convert not found")
    need(_writable(OUTPUTS), "Saving and playing audio", f"can't write to {OUTPUTS}")
    try:
        free = shutil.disk_usage(HOME).free
    except OSError:
        free = None
    need(free is None or free >= MIN_FREE, "Making audio and downloading models",
         f"only {free / 1e9:.1f} GB free on disk" if free is not None else "")
    return problems
