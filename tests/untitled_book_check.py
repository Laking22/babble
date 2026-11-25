"""Run: .venv/bin/python tests/untitled_book_check.py (Kokoro, a few seconds).

Two untitled books get their own folders; making the first again reuses its chapter. Scratch data folder only."""
import os, sys, tempfile, pathlib
TMP = pathlib.Path(tempfile.mkdtemp(prefix="untitled-"))
os.environ["NARRATOR_HOME"] = str(TMP / "home")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from narrator import books, runtime, tts  # noqa: E402
if not runtime.HAS_MLX:
    import numpy as _np
    def _stub_speak(self, items, opts, stop=None):
        t = _np.arange(int(runtime.SR * 0.5)) / runtime.SR
        w = (0.3 * _np.sin(2 * _np.pi * 220 * t)).astype(_np.float32)
        for i, _ in enumerate(items):
            yield i, w
    tts.ENGINES["kokoro"].speak = _stub_speak.__get__(tts.ENGINES["kokoro"], type(tts.ENGINES["kokoro"]))

table = [["Narrator", "Heart (Kokoro)", "", 1.0]]
A = "## The Harbour\n\nThe tavern by the harbour was nearly empty."
B = "## Chapter 1\n\nA different story about a lighthouse keeper."
fails = 0


def check(ok, what):
    global fails
    fails += not ok
    print(("ok  " if ok else "FAIL"), what)


def make(story):
    last = None
    statuses = []
    for ev in books.create(story, table, models="kokoro", sound=False, out_format="WAV"):
        statuses.append(ev.get("status", ""))
        last = ev.get("audio") or last
    return pathlib.Path(last), "\n".join(statuses)


a, _ = make(A)
b, _ = make(B)
check(a.parent.name == "The-Harbour" and a.exists(), f"untitled A -> {a.parent.name}/{a.name}")
check(b.parent.name == "A-different-story-about-a-lighthouse" and b.exists(), f"untitled B -> {b.parent.name}/{b.name}")
check(a.exists() and a.stat().st_size > 10000, "A is still there after B")
a2, status = make(A)
check(a2 == a, "A made again lands in the same folder")
check("already made, reusing it" in status, "and reuses its chapter")
print(sorted(p.name for p in books.BOOKS.iterdir()))
print("FAILED" if fails else "ALL OK", fails)
