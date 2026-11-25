"""Run: .venv/bin/python tests/book_chapters_check.py (Kokoro, under a minute). Scratch data folder only.

An Audiobook project's book: its own folder and file name, a report as each chapter finishes, the chapters
manifest (made / edited since), a character with their own stability, and chapters reused when made again."""
import os, sys, tempfile, pathlib, json
TMP = pathlib.Path(tempfile.mkdtemp(prefix="chapters-"))
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

table = [["Narrator", "Heart (Kokoro)", "", 1.0], ["Tom", "Michael (Kokoro)", "", 1.0]]
STORY = ("## The Harbour\n\nThe tavern was nearly empty.\n\n[Tom] \"You're late,\" said Tom.\n\n"
         "## The Crossing\n\nThe sea was rough that night.")
fails = 0


def check(ok, what):
    global fails
    fails += not ok
    print(("ok  " if ok else "FAIL"), what)


def make(story, **kw):
    events = list(books.create(story, table, models="kokoro", sound=False, out_format="WAV", folder="book-test1",
                               name="Untitled-4", voice_stability={"Tom": 30}, **kw))
    return events


events = make(STORY)
final = pathlib.Path(events[-1]["audio"])
check(final.parent.name == "book-test1" and final.name == "Untitled-4.wav" and final.exists(), f"own folder and name: {final}")
reports = [e["chapters"] for e in events if "chapters" in e]
check(any(r[0]["state"] == "making" for r in reports), "chapter 1 is reported while it's made")
check(any(r[0]["state"] == "done" and r[1]["state"] in ("waiting", "making") for r in reports),
      "chapter 1 is done (with a file) while chapter 2 is still to come")
check(all(pathlib.Path(r[0]["file"]).exists() for r in reports if r[0]["state"] == "done"), "its file is there to play")
st = books.chapters_status(STORY, "book-test1")
check([c["state"] for c in st] == ["done", "done"] and all(c["seconds"] for c in st), f"manifest: {st}")
st = books.chapters_status(STORY.replace("rough", "calm"), "book-test1")
check([c["state"] for c in st] == ["done", "edited"], f"editing chapter 2 marks it edited: {[c['state'] for c in st]}")
again = make(STORY)
check(sum("already made, reusing it" in e.get("status", "") for e in again) == 2, "made again: both chapters reused")
check(not any(e.get("chapters") and any(c["state"] == "making" for c in e["chapters"]) for e in again),
      "and nothing is reported as being made")
manifest = json.loads((books.BOOKS / "book-test1" / books.MANIFEST).read_text())
check(sorted(manifest["chapters"]) == ["0", "1"], "manifest has both chapters")
print("FAILED" if fails else "ALL OK", fails)
