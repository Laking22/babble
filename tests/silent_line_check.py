"""Check the guards against silent lines, without loading any real model (stand-in engines only).

    .venv/bin/python tests/silent_line_check.py

1. A line that comes back silent is made again one at a time; one that's silent twice is left out, named in the
   status, and the file gets "-gaps" so an audiobook chapter is made again next time. NaN never reaches the file.
2. Orpheus batching: after a GPU timeout, the batched model is let go before the one-line model loads.
3. Orpheus batching: a silent sentence inside a line hands the line back empty (so the renderer remakes it).
4. A .txt without chapter headings imports with no "## Full text", and made-up titles aren't read out.
5. Markdown marks (**bold**, *italic*, _italic_) are never read, and a line of nothing but marks isn't spoken.
"""
import gc
import os
import sys
import tempfile
import types
import weakref
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="silent-check-"))
os.environ["NARRATOR_HOME"] = str(TMP / "home")  # never touch the real data folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from narrator import book, tts  # noqa: E402
from narrator import engine as en  # noqa: E402
from narrator.runtime import SR, slot  # noqa: E402
from narrator.tts import orpheus as orp  # noqa: E402
from narrator.tts.base import Engine, Voice  # noqa: E402

ok = True


def check(cond, what):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + what)


def tone(seconds=1.0):
    t = np.arange(int(SR * seconds)) / SR
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


# ---------- 1. renderer ----------
class Fake(Engine):
    id, name, short, prefixes = "fake", "Fake engine", "Fake", ("fake",)
    calls = []

    def voices(self):
        return [Voice("fake:a", "A", "female", "", self.id)]

    def model_key(self, value):
        return "fake"

    def prepare(self, items, opts):
        pass

    def speak(self, items, opts, stop=None):
        single = bool(opts.get("one_at_a_time"))
        Fake.calls.append((single, [it["text"] for it in items]))
        for i, it in enumerate(items):
            t = it["text"]
            if "ALWAYS" in t:
                yield i, np.zeros(SR * 3, np.float32)               # silent every time
            elif "ONCE" in t and not single:
                yield i, np.zeros(SR * 3, np.float32)               # silent in the batch, fine one at a time
            elif "NAN" in t:
                w = tone()
                w[100] = np.nan                                     # one broken sample
                yield i, w
            else:
                yield i, tone()


tts.ENGINES["fake"] = Fake()
items = [dict(text=t, voice="fake:a", who=w, pause=0.2) for t, w in
         [("Fine line one.", "Narrator"), ("Silent ONCE line.", "Bell"), ("Silent ALWAYS line.", "Bell"),
          ("NAN line here.", "Narrator"), ("Fine line five.", "Narrator")]]
out = TMP / "chapter.wav"
final = None
for path, status in en.render(items, kind="chapter", language="english", temperature=0.6, path=out):
    if path:
        final = (path, status)
check(Fake.calls[1:] == [(True, ["Silent ONCE line.", "Silent ALWAYS line."])],
      f"silent lines made again one at a time: {Fake.calls[1:]}")
check(final and final[0].endswith("chapter-gaps.wav") and Path(final[0]).exists(), f"saved as -gaps: {final and final[0]}")
check(final and "line 3 (Bell: “Silent ALWAYS line.”)" in final[1] and "Left out 1 line" in final[1],
      "status names the line left out")
a, _ = sf.read(final[0], dtype="float32")
check(np.isfinite(a).all(), "no NaN in the saved file")
# 4 lines of 1 s kept (the always-silent one dropped), plus pauses: well under the 3 s a dead line would add
check(4.0 <= len(a) / SR <= 5.5, f"length {len(a) / SR:.2f} s: the dead line is left out, the remade one kept")
z = (np.abs(a) < 1e-6).astype(np.int8)
d = np.diff(np.r_[0, z, 0])
longest = max((e - s for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0])), default=0) / SR
check(longest < 1.0, f"no long silent stretch (longest {longest:.2f} s)")

# all lines fine: normal name, no note
Fake.calls.clear()
for path, status in en.render(items[:1] + items[4:], kind="chapter", language="english", temperature=0.6,
                              path=TMP / "clean.wav"):
    if path:
        final = (path, status)
check(final[0].endswith("clean.wav") and "Left out" not in final[1] and len(Fake.calls) == 1,
      "clean chapter: normal name, no retry, no note")

# ---------- 2. Orpheus timeout fallback lets go of the batched model ----------
state = {}


class FakeModel:
    pass


class FakeTok:
    def __call__(self, text):
        return types.SimpleNamespace(input_ids=[1, 2, 3])


def fake_load_batched(path):
    m = FakeModel()
    state["ref"] = weakref.ref(m)
    return m, FakeTok()


class TimeoutGen:
    def __init__(self, model, **kw):
        self.model = model

    def insert(self, prompts, max_tokens, logits_processors=None):
        return list(range(len(prompts)))

    def next_generated(self):
        raise RuntimeError("[METAL] Command buffer execution failed: Caused GPU Timeout Error")

    def close(self):
        pass


def fake_load_model(path):
    gc.collect()
    state["batch_alive_at_load"] = state["ref"]() is not None
    return "one-line model"


import importlib  # noqa: E402

try:
    gen_mod = importlib.import_module("mlx_lm.generate")  # (mlx_lm.generate is also a function name)
    _stubbed_mlx = False
except ImportError:
    _stubbed_mlx = True
    mlx_lm_pkg = types.ModuleType("mlx_lm")
    gen_mod = types.ModuleType("mlx_lm.generate")
    gen_mod.BatchGenerator = None
    sample_mod = types.ModuleType("mlx_lm.sample_utils")
    sample_mod.make_sampler = lambda **kw: None
    sample_mod.make_logits_processors = lambda **kw: []
    sys.modules["mlx_lm"] = mlx_lm_pkg
    sys.modules["mlx_lm.generate"] = gen_mod
    sys.modules["mlx_lm.sample_utils"] = sample_mod

real = (orp._load_batched, gen_mod.BatchGenerator, orp._load_model, orp.Orpheus._mac_speak, orp.time, orp._codes_audio)
orp._load_batched, gen_mod.BatchGenerator, orp._load_model = fake_load_batched, TimeoutGen, fake_load_model
orp.Orpheus._mac_speak = lambda self, m, it, temp=0.6: tone(0.5)
orp.time = types.SimpleNamespace(sleep=lambda s: None)
if _stubbed_mlx:
    orp._codes_audio = lambda toks: tone(0.5) if any(t >= orp.CODE0 for t in toks) else np.zeros(0, dtype=np.float32)
o = orp.Orpheus()
lines = [dict(text=f"Line {n}.", voice="orpheus:tara") for n in range(5)]
got = list(o._batch_speak("fake/path", lines))
check(len(got) == 5, f"fallback finished every line ({len(got)} of 5)")
check(state.get("batch_alive_at_load") is False, "batched model was let go before the one-line model loaded")
slot.release()

# ---------- 3. a silent sentence inside a line ----------
class PieceGen(TimeoutGen):
    """Two units (one line split into two sentences): the second comes back with no audio codes."""
    def __init__(self, model, **kw):
        super().__init__(model)
        self.sent = False

    def next_generated(self):
        if self.sent:
            return []
        self.sent = True
        R = types.SimpleNamespace
        codes = [orp.CODE0 + k * 4096 + 5 for k in range(7)] * 30
        return [R(uid=0, token=t, finish_reason=None) for t in codes] + [R(uid=0, token=orp.EOS_SPEECH, finish_reason="stop"),
                                                                        R(uid=1, token=orp.EOS_SPEECH, finish_reason="stop")]


gen_mod.BatchGenerator = PieceGen
got = list(o._batch_speak("fake/path", [dict(text="First sentence here. Second one there.", voice="orpheus:tara")]))
check(len(got) == 1 and len(got[0][1]) == 0, "line with a silent sentence comes back empty (renderer remakes it)")
orp._load_batched, gen_mod.BatchGenerator, orp._load_model, orp.Orpheus._mac_speak, orp.time, orp._codes_audio = real
slot.release()

# ---------- 4. no "Full text" ----------
txt = TMP / "The_Little_Lighthouse.txt"
txt.write_text("The Little Lighthouse\n\nMara lived by the sea.\n\nOne windy afternoon, the light went out.\n")
b = book.import_book(txt, TMP / "imp")
check("## " not in b["text"] and b["text"].startswith("The Little Lighthouse"), "headingless .txt: no ## heading added")
from narrator import books  # noqa: E402

table = [["Narrator", "Tara (Orpheus)", "", 1.0]]
for story, label in [("## Full text\n\nMara lived by the sea.", "old ## Full text"),
                     ("## Opening\n\nMara lived by the sea.\n\n## Chapter 1\n\nShe woke.", "Opening"),
                     ("## Chapter 1\n\nShe woke.", "real chapter")]:
    built, _, _ = books.build_book(story, table, True, 1.0, 0.2, 0.7)
    spoken = [it["text"] for _, its in built for it in its]
    if label == "real chapter":
        check("Chapter 1" in spoken, "a real chapter title is still read out")
    else:
        check(not any(t in ("Full text", "Opening") for t in spoken), f"{label}: title not read out ({spoken})")

# ---------- 5. Markdown marks ----------
Fake.calls.clear()
md = [dict(text="**", voice="fake:a", who="Narrator", pause=0.2),
      dict(text='"Because he was _relocated_,"', voice="fake:a", who="Elena", pause=0.1),
      dict(text="** she said, **sharp** as *glass*.", voice="fake:a", who="Narrator", pause=0.2),
      dict(text="**", voice="fake:a", who="Narrator", pause=0.7),   # the paragraph's end: its pause must stay
      dict(text="* * *", voice="fake:a", who="Narrator", pause=0.7),
      dict(text="Next scene.", voice="fake:a", who="Narrator", pause=0.7)]
kept = en._speakable(md)
check([it["text"] for it in kept] == ['"Because he was relocated,"', "she said, sharp as glass.", "Next scene."],
      f"marks stripped and mark-only lines dropped: {[it['text'] for it in kept]}")
check(kept[1]["pause"] == 0.7, "a dropped line's paragraph pause moves to the line before")
for path, status in en.render(md, kind="chapter", language="english", temperature=0.6, path=TMP / "md.wav"):
    pass
check(Fake.calls == [(False, [it["text"] for it in kept])], "the engine only gets the clean lines")

print("\nALL PASS" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
