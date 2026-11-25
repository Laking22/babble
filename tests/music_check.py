"""Check music in the sound system without loading any big model (stand-in sounds and a stand-in voice engine).

    .venv/bin/python tests/music_check.py

1. {music: …} tags parse like backgrounds; a music tag never lands on an effect; missing music is listed as music.
2. Music is its own layer: a new {amb: …} doesn't stop it, {music: off} does, and it sits ~13 dB lower under speech.
3. Chapter intro: plays alone for its lead time, the voice starts after it, and it fades out under the title.
4. AI suggestions: a "music" pick becomes {music: …} at the start of its paragraph.
5. The music model: Small on this Mac, Medium only with 16 GB (or the developer override).
6. The audiobook puts the intro at the start of every chapter (and not in previews).
"""
import os
import sys
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="music-check-"))
os.environ["NARRATOR_HOME"] = str(TMP / "home")  # never touch the real data folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from narrator import settings, sfx, sfxgen, tts  # noqa: E402
from narrator import engine as en  # noqa: E402
from narrator.runtime import SR  # noqa: E402
from narrator.tts.base import Engine, Voice  # noqa: E402

settings.save(sfx_smart=False)  # match by words only: no CLAP model needed
ok = True


def check(cond, what):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + what)


def tone(seconds, hz, amp=0.3):
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def band_db(a, lo, hi, hz):
    """Level of one frequency (a stand-in sound's own tone) between two times, in dB."""
    m = a.mean(1) if a.ndim == 2 else a
    seg = m[int(lo * SR):int(hi * SR)]
    t = np.arange(len(seg)) / SR
    amp = 2 * abs(np.mean(seg * np.exp(-2j * np.pi * hz * t)))
    return 20 * np.log10(amp + 1e-9)


# stand-in library: music = 440 Hz, background = 300 Hz, an effect called "Ball Bounce", an intro = 660 Hz
mus = sfx.add(tone(20, 440), "Ballroom Waltz", "music", tags=["waltz", "ball", "strings"])
amb = sfx.add(tone(20, 300), "Crowd Murmur", "ambience", tags=["crowd"])
amb2 = sfx.add(tone(20, 350), "Rain", "ambience", tags=["rain"])
sfx.add(tone(1, 900), "Ball Bounce", "effect", tags=["ball", "bounce"])
intro = sfx.add(tone(24, 660), "Seaside Theme", "music", tags=["theme"])

# ---------- 1. tags ----------
cues = [c for k, c in sfx.split_cues("{music: quiet waltz, quiet} a {music: harp, add} b {music: off}") if k == "cue"]
check([(c["type"], c["name"], c.get("op"), c["volume"]) for c in cues] ==
      [("music", "quiet waltz", "set", "quiet"), ("music", "harp", "add", "normal"), ("music", None, None, "normal")],
      "music tags parse like backgrounds (scene, add, off, volume)")
check((sfx.find_for({"type": "music", "name": "ball"}) or {}).get("id") == mus["id"], "{music: ball} finds the waltz")
check(sfx.find_for({"type": "music", "name": "bounce"}) is None, "a music tag never lands on an effect")
check(sfx.missing("{music: church organ} {sfx: ball bounce}") == [("music", "church organ")],
      "missing music is listed as music (so it's made with the music model)")

# ---------- 2. the mixer ----------
speech = np.zeros(SR * 30, np.float32)
spans = [(SR * 12, SR * 16)]  # someone speaks from 12 to 16 s
speech[SR * 12:SR * 16] = tone(4, 1000, 0.1)
cues = [(0, {"type": "amb", "name": "crowd", "volume": "normal", "op": "set"}),
        (SR * 2, {"type": "music", "name": "waltz", "volume": "normal", "op": "set"}),
        (SR * 8, {"type": "amb", "name": "rain", "volume": "normal", "op": "set"}),
        (SR * 22, {"type": "music", "name": None, "volume": "normal"})]
out = sfx.mix(speech, cues, spans, log=print)
check(band_db(out, 9, 11, 440) > -40, f"music plays on after a new background ({band_db(out, 9, 11, 440):.0f} dB)")
check(band_db(out, 9.5, 11, 300) < band_db(out, 4, 6, 300) - 20, "the background did change (crowd → rain)")
check(band_db(out, 26, 29, 440) < -60, f"{{music: off}} stops it ({band_db(out, 26, 29, 440):.0f} dB)")
duck = band_db(out, 9, 11, 440) - band_db(out, 13, 15, 440)
check(11 <= duck <= 15, f"music dips {duck:.1f} dB under speech (backgrounds dip 10)")

# ---------- 3. chapter intro through the renderer ----------
class Fake(Engine):
    id, name, short, prefixes = "fake", "Fake engine", "Fake", ("fake",)

    def voices(self):
        return [Voice("fake:a", "A", "female", "", self.id)]

    def model_key(self, value):
        return "fake"

    def prepare(self, items, opts):
        pass

    def speak(self, items, opts, stop=None):
        for i, _ in enumerate(items):
            yield i, tone(2, 1000, 0.2)


tts.ENGINES["fake"] = Fake()
lead = sfx.intro_lead(intro)
items = [{"cue": {"type": "intro", "id": intro["id"], "name": intro["name"], "lead": lead, "volume": "normal"}, "pause": 0},
         dict(text="Chapter One.", voice="fake:a", pause=1.0), dict(text="It was a calm night.", voice="fake:a", pause=0.5)]
for kind_sound in (True, False):
    path = None
    for p, status in en.render(items, kind="chapter", language="english", temperature=0.6, sound=kind_sound,
                               path=TMP / f"intro-{kind_sound}.wav"):
        path = p or path
    a, _ = sf.read(path, dtype="float32")
    alone, under, after = band_db(a, 1, lead - 0.5, 660), band_db(a, lead + 0.3, lead + 1.5, 660), \
        band_db(a, lead + 4.5, lead + 6, 660)
    voice_before = band_db(a, 0.5, lead - 0.3, 1000)
    voice_after = band_db(a, lead + 0.3, lead + 1.5, 1000)
    check(alone > -30 and voice_before < -60 and voice_after > -30,
          f"intro plays alone for {lead} s, then the title (sound tags {'on' if kind_sound else 'off'})")
    check(under < alone - 10 and after < -60, f"intro dips under the title ({under - alone:.0f} dB) and is gone "
          f"by {lead + 4.5:.1f} s ({after:.0f} dB)")

# ---------- 4. AI suggestion placement ----------
from narrator import ai_sfx  # noqa: E402

text = "The orchestra struck up a waltz as they entered the ballroom.\n\nOutside, the rain had started."
parts = __import__("re").split(r"(\n\s*\n)", text)
paras = [(0, parts[0]), (2, parts[2])]
placements = ai_sfx._plan(parts, paras, [[0, 1]], [(0, 0, 0, "music", "waltz", "", "quiet"),
                                                   (0, 1, 0, "amb", "rain", "", "normal")])
music_p = [p for p in placements if p["kind"] == "music"]
check(len(music_p) == 1 and music_p[0]["name"] == "Ballroom Waltz" and not music_p[0]["missing"],
      "a music pick matches the library's music")
tagged = ai_sfx.insert(text, placements)
check(tagged.startswith("{music: Ballroom Waltz, quiet} The orchestra"), f"…and goes at the paragraph start: {tagged[:45]!r}")
check(ai_sfx.label_of(music_p[0]).startswith("Music · Ballroom Waltz (quiet)"), "review list says Music")

# ---------- 5. which music model ----------
from narrator.tts.base import ram_gb  # noqa: E402

big = ram_gb() >= 16
check(sfxgen.model_for("effect") == "sm-sfx" and sfxgen.model_for("ambience") == "sm-sfx", "effects use Small SFX")
check(sfxgen.model_for("music") == ("medium" if big else "sm-music"), f"music on this {ram_gb():.0f} GB Mac: "
      f"{sfxgen.model_for('music')}")
check(sfxgen.model_for("music", "small") == "sm-music", "Small can always be picked")
os.environ[sfxgen.DEV_MUSIC_MODEL] = "medium"
check(sfxgen.model_for("music") == "medium", "developer override reaches Medium")
del os.environ[sfxgen.DEV_MUSIC_MODEL]
check(sfxgen.prompt_for("a waltz", "music") == "a waltz, instrumental", "music prompts ask for instrumental")

from narrator import ai_music  # noqa: E402

check(ai_music._clean('Theme: "gentle seaside folk, harp and flute, calm, slow tempo."') ==
      "gentle seaside folk, harp and flute, calm, slow tempo", "theme description is cleaned")
check(ai_music._clean("ok") == ai_music.FALLBACK, "a useless reply falls back to a plain theme")

# ---------- 6. the audiobook adds the intro to every chapter ----------
from narrator import books  # noqa: E402


def perform(*a, **k):
    """books.create as (audio, status) steps, plus the story and Undo text when the story changed."""
    for ev in books.create(*a, **k):
        if "status" not in ev:  # chapter progress for the desktop app's list
            continue
        yield ((ev.get("audio"), ev["status"], {"value": ev["story"]}, ev["original"]) if "story" in ev
               else (ev.get("audio"), ev["status"]))

seen = []


def fake_render(items, kind, path=None, **kw):
    seen.append((kind, items))
    sf.write(path or TMP / "prev.wav", tone(1, 1000, 0.2), SR)
    yield str(path or TMP / "prev.wav"), "Done"


real = en.render
en.render = fake_render
story = "## Chapter 1\n\nShe woke.\n\n## Chapter 2\n\nShe slept."
table = [["Narrator", "Tara (Orpheus)", "", 1.0]]
args = [story, table, "cast", True, 1.0, 55, "english", 0, 2, 0.2, 0.7, "WAV", "Intro test", "", None, True, 1.0]
list(perform(*args, intro["id"], False, preview=False))
chapters = [items for kind, items in seen if kind == "chapter"]
check(len(chapters) == 2 and all(it[0].get("cue", {}).get("type") == "intro" for it in chapters),
      "every chapter starts with the intro")
seen.clear()
list(perform(*args, intro["id"], False, preview=True))
check(seen and not any("cue" in it for _, items in seen for it in items), "previews have no intro")

# ---------- 7. a chapter is made again once its missing sounds exist; the final status says what wasn't heard ----------
story = "## Chapter 1\n\n{amb: harbour waves at night} She waited.\n\n## Chapter 2\n\nShe slept."
args = [story, table, "cast", True, 1.0, 55, "english", 0, 2, 0.2, 0.7, "WAV", "Missing test", "", None, True, 1.0]
seen.clear()
done = list(perform(*args, "", False, preview=False))[-1][1]
check(len(seen) == 2 and "Not heard: 1 tag(s)" in done and "“harbour waves at night”" in done,
      "the final status names the tag that wasn't heard")
seen.clear()
list(perform(*args, "", False, preview=False))
check(seen == [], "nothing changed: both chapters are reused")
sfx.add(tone(20, 250), "Harbour Waves", "ambience", tags=["harbour", "waves", "night"])
seen.clear()
done = list(perform(*args, "", False, preview=False))[-1][1]
check(len(seen) == 1 and any("harbour" in it.get("text", "") for it in seen[0][1]) and "Not heard" not in done,
      "after the sound is made, only the chapter that uses it is made again")
# ---------- 8. "Make missing sounds automatically" ----------
from narrator import sfxgen as _gen  # noqa: E402

asked = []


def fake_make(desc, kind="effect", **kw):
    asked.append((kind, desc))
    return sfx.add(tone(20, 270), desc.title(), kind, tags=sfx._words(desc))


real_make, _gen.make = _gen.make, fake_make
story = ("## Chapter 1\n\n{music: lute serenade under a balcony} She listened.\n\n"
         "## Chapter 2\n\n{amb: wind over the moors} She walked home.")
args = [story, table, "cast", True, 1.0, 55, "english", 0, 2, 0.2, 0.7, "WAV", "Auto test", "", None, True, 1.0]
seen.clear()
done = list(perform(*args, "", False, preview=False))[-1][1]  # automatic off: nothing made
check(asked == [] and "Not heard: 2 tag(s)" in done, "toggle off: nothing is made, the status says what wasn't heard")
seen.clear()
list(perform(*args, "", True, preview=True))  # preview with it on: only the preview paragraph's sound
check(asked == [("music", "lute serenade under a balcony")], f"preview makes only the sounds it plays: {asked}")
seen.clear()
steps = list(perform(*args, "", True, preview=False))
done = steps[-1][1]
check(asked[1:] == [("ambience", "wind over the moors")], f"create makes the rest, each with its own kind: {asked[1:]}")
check(any("Making the missing sounds with AI · 1 of 1" in (st or "") for _, st in steps), "it says what it's making")
check("Made 1 missing sound(s) first" in done and "Not heard" not in done and len(seen) == 2,
      "both chapters are made with their sounds, and the status says so")
_gen.make = real_make
en.render = real

# ---------- 9. "How many sound effects" levels ----------
sfx.add(tone(1, 800), "Door Slam", "effect", tags=["door", "slam"])
sfx.add(tone(1, 820), "Glass Break", "effect", tags=["glass", "break", "shatter"])
text = ("The door slammed. She ran. A glass shattered. He shouted. The door slammed again. Rain fell. "
        "Footsteps echoed. The door slammed once more.")
parts = [text]
paras = [(0, text)]
# the model's picks (window, paragraph, sentence, kind, sound, word, volume): door 3 times, glass, footsteps (new),
# and the glass one sentence off (on "He shouted.", next to the glass sentence)
picks = [(0, 0, 0, "sfx", "door slam", "slammed", "normal"), (0, 0, 2, "sfx", "glass break", "shattered", "normal"),
         (0, 0, 4, "sfx", "door slam", "slammed", "normal"), (0, 0, 6, "sfx", "footsteps", "Footsteps", "normal"),
         (0, 0, 7, "sfx", "door slam", "slammed", "normal")]
def effects_at(level, picks=picks):
    return [(p["s"], p["name"]) for p in ai_sfx._plan(parts, paras, [[0]], picks, level) if p["kind"] == "sfx"]
# (a sound is only used once per paragraph at every level, so the later door slams always go)
lv1, lv2, lv3 = effects_at(1), effects_at(2), effects_at(3)
check(lv1 == [(0, "Door Slam")], f"level 1: library sounds only, 6+ sentences apart: {lv1}")
check(lv2 == [(0, "Door Slam"), (6, "footsteps")], f"level 2: 3+ sentences apart: {lv2}")
check(lv3 == [(0, "Door Slam"), (2, "Glass Break"), (6, "footsteps")], f"level 3: every fitting pick, new sounds too: {lv3}")
off = effects_at(3, [(0, 0, 3, "sfx", "glass break", "shouted", "normal")])
check(off == [(2, "Glass Break")], f"a sound one sentence off moves onto its sentence: {off}")

# ---------- 10. "Add sound effects automatically to stories without tags" ----------
from narrator import ai_sfx as _ai  # noqa: E402

calls = []


def fake_suggest(text, stop=None, density=3):
    calls.append(density)
    yield ("progress", 0, 1)
    yield ("progress", 1, 1)
    tagged = text.replace("She woke.", "She woke. {sfx: door slam @woke}")
    yield ("done", tagged, ["Effect · Door Slam"], [], [{"id": 0}])


real_suggest, _ai.suggest = _ai.suggest, fake_suggest
en.render = fake_render
plain = "## Chapter 1\n\nShe woke.\n\n## Chapter 2\n\nShe slept."
args = [plain, table, "cast", True, 1.0, 55, "english", 0, 2, 0.2, 0.7, "WAV", "Autotag test", "", None, True, 1.0]
seen.clear()
steps = list(perform(*args, "", False, True, 4, preview=False))
story_step = [st for st in steps if len(st) == 4]
check(calls == [4] and len(story_step) == 1 and "{sfx: door slam" in story_step[0][2]["value"]
      and story_step[0][3] == plain, "a story without tags gets the AI's tags (at the slider's level), shown in the "
      "story box, with the old text kept for Undo")
check(any("door slam" in it.get("text", "") for _, items in seen for it in items), "…and the book is made with them")
check("The AI added 1 sound tag(s)" in steps[-1][1], "the final status says so")
calls.clear()
list(perform(*args[:1] + args[1:], "", False, False, 4, preview=False))
list(perform(*args, "", False, True, 4, preview=True))
tagged_args = ["## Chapter 1\n\nShe woke. {sfx: door slam}"] + args[1:]
list(perform(*tagged_args, "", False, True, 4, preview=False))
check(calls == [], "not when it's off, not for previews, and not when the story already has tags")
_ai.suggest = real_suggest
en.render = real

print("\nALL PASS" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
