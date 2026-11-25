"""Offline checks for five fixes:
1. api.voice_preview starts a task for a preview that isn't made yet (it returned None)
2. export.combine sets the file's rate and bitrate from Settings → Audiobook file quality (loudnorm alone gave
   192 kHz WAV / 96 kHz M4B / 48 kHz MP3)
3. modal_jobs.status: Modal's own ConnectionError is a network error (raised), FunctionTimeoutError is the end
4. spend.stop_all stops training runs and runs no book follows (it skipped every collect job)
5. /train/runs/retry-estimate answers what Train again would cost (the UI asks before it sends)

    .venv/bin/python tests/top5_fixes_check.py
"""
import io
import json
import os
import subprocess
import sys
import tempfile
import types
from contextlib import redirect_stdout
from pathlib import Path

HOME = tempfile.mkdtemp(prefix="top5-check-")
os.environ["NARRATOR_HOME"] = HOME  # never touch the real data folder
os.environ["HOME"] = HOME
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf

for mod in ("mlx", "mlx.core", "mlx.nn", "mlx_audio", "mlx_audio.tts", "mlx_audio.tts.utils"):
    sys.modules[mod] = types.ModuleType(mod)
sys.modules["mlx.core"].set_cache_limit = lambda *a, **kw: None
sys.modules["mlx.core"].clear_cache = lambda *a, **kw: None
sys.modules["mlx_audio.tts.utils"].load_model = lambda *a, **kw: None

failed = []


def check(name, ok, detail=""):
    print(f"{'OK  ' if ok else 'FAIL'} {name}{(' · ' + detail) if detail else ''}")
    if not ok:
        failed.append(name)


# ---- 1. voice preview ----
from narrator import api  # noqa: E402

voice = types.SimpleNamespace(value="test:voice", name="Test Voice")
api.engines.all_voices = lambda only_available=False: [voice]
api._start = lambda kind, title, work: {"task": title}
got = api.voice_preview(api.VoicePreview(voice="test:voice"))
check("voice_preview starts a task when not cached", got == {"task": "Preview Test Voice"}, repr(got))

# ---- 2. export quality ----
from narrator import export, settings  # noqa: E402

work = Path(HOME) / "export"
work.mkdir()
sr = 24000
t = np.arange(sr * 2) / sr
for i in range(2):
    sf.write(work / f"{i:03d}.wav", (0.2 * np.sin(2 * np.pi * 220 * t)).astype("float32"), sr)
files = [work / "000.wav", work / "001.wav"]


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                          "stream=sample_rate,codec_name,bit_rate,bits_per_sample:format=bit_rate", "-of", "json",
                          str(path)], capture_output=True, text=True).stdout
    d = json.loads(out)
    return d["streams"][0], d.get("format", {})


want = {"small": 24000, "standard": 44100, "high": 44100, "highest": 48000}
for q, rate in want.items():
    for fmt in ("Audiobook (M4B with chapters)", "MP3", "WAV"):
        path = export.combine(files, ["One", "Two"], work / f"book-{q}", fmt, "T", "A", file_quality=q)
        s, f = probe(path)
        ok = int(s["sample_rate"]) == rate
        if fmt == "WAV":
            ok = ok and s["codec_name"] == export.QUALITY[q]["wav"]
        check(f"{q} {fmt.split()[0]}", ok, f"{s['sample_rate']} Hz {s['codec_name']}")
        Path(path).unlink()

settings.save(file_quality="highest")
path = export.combine(files, ["One", "Two"], work / "book-setting", "WAV", "T", "A")
s, _ = probe(path)
check("combine reads Settings when not told", int(s["sample_rate"]) == 48000, s["sample_rate"])
settings.save(file_quality="nonsense")
path = export.combine(files, ["One", "Two"], work / "book-bad", "WAV", "T", "A")
s, _ = probe(path)
check("unknown setting falls back to standard", int(s["sample_rate"]) == 44100, s["sample_rate"])

# ---- 3. modal status ----
from narrator import modal_jobs  # noqa: E402


class ModalConnectionError(Exception):
    pass


ModalConnectionError.__name__ = "ConnectionError"  # modal.exception.ConnectionError: not Python's


class FunctionTimeoutError(Exception):
    pass


class ExecutionError(Exception):
    pass


class Vol:
    def read_file(self, path):
        raise FileNotFoundError(path)

    def listdir(self, path):
        return [types.SimpleNamespace(path=f"{path}/log.txt")]


modal_jobs._volume = lambda create=True: Vol()
modal_jobs._exists = lambda vol, path: True
fake = types.ModuleType("modal")
sys.modules["modal"] = fake


def status_with(exc):
    class Call:
        def get(self, timeout=None):
            raise exc

    fake.FunctionCall = types.SimpleNamespace(from_id=lambda _id: Call())
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            modal_jobs.status({"run": "r1", "call": "fc-1"})
    except Exception as e:  # noqa: BLE001
        return "raised " + type(e).__name__
    line = [x for x in buf.getvalue().splitlines() if x.startswith("RESULT")]
    return json.loads(line[-1].split(" ", 1)[1])["state"] if line else buf.getvalue()


check("Modal ConnectionError is raised (a network error)", status_with(ModalConnectionError("Could not connect "
      "to the Modal server.")) == "raised ConnectionError")
check("FunctionTimeoutError is the end of the run", status_with(FunctionTimeoutError("timed out")) == "error")
check("Python TimeoutError: still running", status_with(TimeoutError()) == "running")
check("ExecutionError: failed", status_with(ExecutionError("oom")) == "error")
check("OSError is raised", status_with(OSError("Network is down")).startswith("raised"))

# ---- 4. stop all ----
from narrator import books, cloud_book, spend  # noqa: E402

book = books.BOOKS / "my-book"
book.mkdir(parents=True, exist_ok=True)
(book / cloud_book.RECORD).write_text(json.dumps({"run": "book-run", "backend": "modal", "state": "running"}))
spend._save_jobs({
    "train-run": {"backend": "modal", "what": "Training orpheus", "price": 1.0, "hours": 3, "collect": True, "sent": 0, "service": "modal"},
    "book-run": {"backend": "modal", "what": "Create in the cloud", "price": 1.0, "hours": 3, "collect": True, "sent": 0, "service": "modal"},
    "stray-run": {"backend": "modal", "what": "Create in the cloud", "price": 1.0, "hours": 3, "collect": True, "sent": 0, "service": "modal"},
    "detect-run": {"backend": "modal", "what": "Detect speakers", "price": 1.0, "hours": 1, "collect": False, "sent": 0, "service": "modal"},
})
stopped_remote, forgotten = [], []
spend.stop_remote = lambda backend, run: stopped_remote.append(run) or True
cloud_book.forget = lambda where: forgotten.append(Path(where).name)
spend.stop_all()
check("training run stopped", "train-run" in stopped_remote, str(stopped_remote))
check("stray collect run stopped", "stray-run" in stopped_remote)
check("plain job stopped", "detect-run" in stopped_remote)
check("book's run left to its book", "book-run" not in stopped_remote and forgotten == ["my-book"], str(forgotten))

# ---- 5. retry estimate ----
from narrator import api_manage, training  # noqa: E402

api_manage._runs = lambda: {"k1": {"voices": {"amy": "Amy"}, "recipe": "orpheus", "backend": "modal", "model": "m"}}
training.estimate = lambda specs, recipe, backend, epochs, max_clips: {"hours": 2.0, "gpu_hours": 1.5,
                                                                       "backend": "Modal"}
spend.gpu_price = lambda backend, gpu: 0.6
e = api_manage.retry_estimate("k1")
check("retry estimate names the backend and cost", e["backend_id"] == "modal" and e["paid"] and e["cost"], e["text"])
try:
    api_manage.retry_estimate("nope")
    check("unknown run refused", False)
except api_manage.HTTPException as x:
    check("unknown run refused", x.status_code == 404)

print("\nALL OK" if not failed else f"\n{len(failed)} FAILED: {failed}")
sys.exit(1 if failed else 0)
