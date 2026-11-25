"""Check the guards against lines that leave words out, without loading any real model (stand-in engine only).

    .venv/bin/python tests/rushed_line_check.py

1. Trained Qwen voices (like Orpheus) get lines of at most 190 characters; a longer sentence is cut at its clause
   breaks, and no word is lost.
2. A take far quicker than the rest of its voice's (words left out) is made again one at a time, and the new take
   is used only when it has more sound in it; one that stays rushed isn't kept for next time.
3. A rushed line back from the cloud is made again here while Settings → "Remake cloud lines that skip words" is on
   (the default), and kept as it came back when it's off.
"""
import os
import sys
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="rushed-check-"))
os.environ["NARRATOR_HOME"] = str(TMP / "home")  # never touch the real data folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from narrator import tts  # noqa: E402
from narrator import engine as en  # noqa: E402
from narrator.audio import rushed  # noqa: E402
from narrator.runtime import SR  # noqa: E402
from narrator.tts.base import Engine, Voice  # noqa: E402

ok = True


def check(cond, what):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + what)


# ---------- 1. line length ----------
long = ("Of such great powers or beings there may be conceivably a survival, a survival of a hugely remote period "
        "when consciousness was manifested, perhaps, in shapes and forms long since withdrawn before the tide of "
        "advancing humanity; forms of which poetry and legend alone have caught a flying memory and called them gods.")
check(en.chunk_limit("qwen:clarke", 250) == 240 and en.chunk_limit("orpheus:tara", 250) == 190
      and en.chunk_limit("builtin:ryan", 250) == 250, "trained Qwen <= 240, Orpheus <= 190, built-in <= 250")
parts = en.split_chunks(long, 240)
check(len(parts) == 2 and max(map(len, parts)) <= 240, f"a long sentence cut in fewest pieces: {[len(p) for p in parts]}")
check(max(map(len, parts)) <= 190, f"avoids lopsided strong break (30-70% balance): {[len(p) for p in parts]}")
mid_semi = ("Of such great powers or beings there may be conceivably a survival, a survival of a hugely remote period; "
            "when consciousness was manifested, perhaps, in shapes and forms long since withdrawn before the tide of "
            "advancing humanity and legend alone caught a flying memory.")
check(en.split_chunks(mid_semi, 240)[0].rstrip().endswith(";"), "prefers balanced semicolon over comma")
early_colon = ("The house stood empty for a decade and more: nobody came near it, nobody spoke its name, and the children "
               "of the village, who feared it more than anything, crossed the road rather than walk beneath its blind "
               "windows at dusk.")
check(en.split_chunks(early_colon, 200)[0].endswith("more:"),
      "a strong break outside 30-70% still wins when both pieces are under 200 (no comma reset)")
lopsided = ("He had come at last to the old house; and there, beneath the crooked elms that leaned over the broken wall "
            "like tired sentinels, with the wind moving through the dead grass and the moon hanging low and red above "
            "the hills, he waited for the stranger to appear.")
check(max(map(len, en.split_chunks(lopsided, 240))) <= 200,
      f"but not one leaving a long tail (37+222): {[len(p) for p in en.split_chunks(lopsided, 240)]}")
check(en.split_chunks("He got an A. Then he left the room.", 20)[0] == "He got an A.",
      "a grade (an A.) ends a sentence; it isn't an initial")
check("".join("".join(parts).split()) == "".join(long.split()), "no word lost or changed")
check(all(p.rstrip()[-1] in ",;." for p in parts), f"cut at clause breaks: {[p[-12:] for p in parts]}")
check(en.split_chunks("One. Two three.", 190) == ["One. Two three."], "short lines as before")
check(en.split_chunks("Dr. Angell met H. P. Lovecraft at St. Mary's.", 190) == ["Dr. Angell met H. P. Lovecraft at St. Mary's."],
      "abbreviations and initials aren't split")
check(en.split_chunks("x" * 300, 190) == ["x" * 300], "one long word is left whole")
tagged = ("{amb: quiet room with a slow ticking clock and faint wind against window glass} {music: Brooding dread with "
          "low sustained cello drone, distant muted timpani, slow tempo} The most merciful thing in the world, I think, "
          "is the inability of the human mind to correlate all its contents [softly] and so on, {sfx: door, far} we live.")
nums = ("In 1,000,000 years the 3:45 train from Arkham, which ran at 12,500 feet above the old sea, would stop at "
        "every station between here and the coast, 1,234 of them in all, and nobody would remember the 2,000 who rode it")
cuts = [en.split_chunks(nums, n) for n in (190, 120, 80)]
check(all(len(c) > 1 and not any(p[-1].isdigit() and q[0].isdigit() for p, q in zip(c, c[1:])) for c in cuts),
      f"never cut inside a number (1,000 or 3:45): {[p[-8:] for p in cuts[1]]}")
parts = en.split_chunks(tagged, 190)
check(all(p.count("{") == p.count("}") and p.count("[") == p.count("]") for p in parts),
      f"never cut inside a sound tag or direction: {[p[:30] for p in parts]}")
from narrator import settings as _s  # noqa: E402
_s.save(longest_line=0)
check(en.chunk_limit("qwen:clarke", 250) == 250 and en.chunk_limit("orpheus:tara", 250) == 250
      and en.split_chunks(long, 250) == [long], "slider at Whole sentences (0) keeps long sentences whole")
_s.save(longest_line=200)
check(en.chunk_limit("qwen:clarke", 250) == 200 and en.chunk_limit("orpheus:tara", 250) == 190
      and max(map(len, en.split_chunks(long, en.chunk_limit("qwen:clarke", 250)))) <= 200,
      "slider at 200: Qwen <= 200, Orpheus stays <= 190")
_s.save(longest_line=150)
check(en.chunk_limit("orpheus:tara", 250) == 150, "slider at 150 lowers Orpheus too")
_s.save(longest_line=240)
import json as _json  # noqa: E402
from narrator.paths import SETTINGS as _SP  # noqa: E402
_raw = _json.loads(_SP.read_text())
_raw.pop("longest_line", None)
_SP.write_text(_json.dumps(dict(_raw, split_long_lines=False)))
check(en.longest_line() == 0, "the old switch turned off becomes Whole sentences")
_SP.write_text(_json.dumps(dict(_raw, split_long_lines=True)))
check(en.longest_line() == 240, "the old switch left on keeps the default 240")
h = en._halves("ALWAYS " + "the dreaming city lay beneath the waves and waited for the stars to come right again")
check(h and len(h) == 2 and "".join(h).replace(" ", "") == ("ALWAYS the dreaming city lay beneath the waves and waited "
      "for the stars to come right again").replace(" ", ""), f"a line cut in two halves: {h}")
check(en._halves("He spoke softly to her, and she did not answer him. Then, without another word, he turned and left.")
      [0] == "He spoke softly to her, and she did not answer him.", "halves prefer a sentence break near the middle")
check(en._halves("short line") is None, "short lines aren't halved")


# ---------- 2. renderer ----------
def speech(letters, rate):
    """A tone as long as `letters` take at `rate` letters a second."""
    t = np.arange(int(SR * letters / rate)) / SR
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


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
        Fake.calls.append((single, [it["text"][:6] for it in items]))
        for i, it in enumerate(items):
            n = sum(c.isalnum() for c in it["text"])
            if it["text"].startswith("SKIPS") and not single:
                yield i, speech(n, 30)        # words left out in the batch (far quicker than the voice's 15)
            elif it["text"].startswith("ALWAYS"):
                yield i, speech(n, 30 if not single else 32)  # rushed every time (the retry even quicker)
            elif "LONG" in it["text"] and len(it["text"]) > 60:
                yield i, speech(n, 30)        # rushed whenever the whole line is said at once (halves are fine)
            else:
                yield i, speech(n, 15)


tts.ENGINES["fake"] = Fake()
from narrator import cloud_book  # noqa: E402
cloud_book.local_spec = lambda item, opts, seed=0: {"text": item["text"]}  # (the stand-in voice's lines are kept)
words = "the dreaming city lay beneath the waves and waited for the stars to come right again "
texts = [f"Line {k} " + words for k in range(22)]
texts[5] = "SKIPS " + words
texts[9] = "ALWAYS " + words
texts[12] = "LONG " + words
items = [dict(text=t, voice="fake:a", who="Narrator", pause=0.2) for t in texts]
keep = TMP / "lines"
final = None
for path, status in en.render(items, kind="chapter", language="english", temperature=0.6, path=TMP / "c.wav",
                              keep=keep):
    if path:
        final = path
check(Fake.calls[1] == (True, ["SKIPS ", "ALWAYS", "LONG t"]), f"the rushed takes made again one at a time: {Fake.calls[1]}")
check(len(Fake.calls) == 3 and Fake.calls[2][0] and len(Fake.calls[2][1]) == 4
      and Fake.calls[2][1][0] == "ALWAYS" and Fake.calls[2][1][2] == "LONG t",
      f"the two still rushed are made again in halves: {Fake.calls[2:]}")
a, _ = sf.read(final, dtype="float32")
n = sum(c.isalnum() for c in texts[5])
check(final and len(a) / SR > 22 * n / 15 * 0.9, f"the slower retakes were used ({len(a) / SR:.1f} s of audio)")
check(len(list(keep.glob("*.flac"))) == 21,
      f"the halved line is kept, the one that stayed rushed isn't: {len(list(keep.glob('*.flac')))} kept")
# 3. lines back from the cloud: a rushed one is made here again
from narrator import settings  # noqa: E402
cloud_dir = TMP / "cloud-lines"
cloud_dir.mkdir()
for k, t in enumerate(texts):
    if not t.startswith("ALWAYS"):
        n = sum(c.isalnum() for c in t)
        sf.write(str(cloud_dir / f"{abs(hash(t.strip()))}.flac"), speech(n, 30 if t.startswith("SKIPS") else 15), SR)
cloud_book.premade_line = lambda folder, it, opts, seed=0: (  # (the renderer strips each line's text first)
    sf.read(str(Path(folder) / f"{abs(hash(it['text'].strip()))}.flac"), dtype="float32")[0]
    if (Path(folder) / f"{abs(hash(it['text'].strip()))}.flac").exists() else None)
cloud_book._runs_here = lambda voice: True
Fake.calls = []
for path, status in en.render(items, kind="chapter", language="english", temperature=0.6,
                              path=TMP / "cloud.wav", premade=cloud_dir):
    pass
made = Fake.calls[0][1]
check(sorted(set(made)) == ["ALWAYS", "SKIPS "], f"rushed cloud lines remade on this Mac: {sorted(set(made))}")
check(settings.DEFAULTS["longest_line"] == 240 and "split_long_lines" not in settings.DEFAULTS
      and "remake_rushed_cloud" not in settings.DEFAULTS, "longest_line 240 by default; old switches removed")

takes = {k: ("v", texts[k], speech(sum(c.isalnum() for c in texts[k]), 15)) for k in range(22)}
check(rushed(takes) == set(), "steady takes aren't flagged")
check(rushed(dict(list(takes.items())[:3])) == set(), "short steady run isn't flagged")
check(rushed(dict(list(takes.items())[:3], x=("v", texts[0], speech(70, 28)))) == {"x"},
      "rushed long line is caught even on a short (<20 lines) Text-to-Speech run")

print("ALL OK" if ok else "SOME FAILED")
