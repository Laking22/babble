"""Run: .venv/bin/python tests/dialogue_voice_check.py  (a few seconds, no models, scratch data folder only)

Prepare a voice with Dialogue only: the clips are the characters' lines (the words in quotation marks), never the
narration, never two characters in one clip, each with how the book says it was said. The audio is made up (a burst
of noise per word, pauses where a reader pauses) and the transcript is the book's own words with their times.
"""
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

TMP = Path(tempfile.mkdtemp(prefix="dialoguevoice-"))
os.environ["NARRATOR_HOME"] = str(TMP / "home")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
failed = []


def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what, flush=True)
    if not ok:
        failed.append(what)


from narrator.dataset import pipeline as p  # noqa: E402

# 1. finding the lines
pieces = p.dialogue_pieces(p.clean_text('“You’re late,” Sarah snapped. “The ship left an hour ago.”'))
check(pieces == [("You're late,", True), ("Sarah snapped.", False), ("The ship left an hour ago.", True)],
      f"quotes and narration come apart in order ({pieces})")
check(p.said_how(pieces, 0) == "snapped" and p.said_how(pieces, 2) == "snapped", "how it was said, from the tag next to it")
check(p.said_how(p.dialogue_pieces('He leaned closer and whispered softly, "Run."'), 1) == "whispered softly",
      "a tag before the line, with its adverb")
check(p.said_how(p.dialogue_pieces('"Fine."'), 0) == "", "no tag: nothing made up")
check(p.dialogue_pieces('"It began in the north. Nobody listened.') == [("It began in the north. Nobody listened.", True)],
      "speech running on into the next paragraph (no closing quote) is still speech")
check(p._spoken("You're late,") == "You're late.", "a line cut off at its comma ends as a sentence")
single = p.dialogue_pieces("'Go away,' she said. It wasn't Tom's fault.", single=True)
check(single == [("Go away,", True), ("she said. It wasn't Tom's fault.", False)],
      f"single-quoted speech, apostrophes left alone ({single})")
british = ["'Hello,' she said.", "'It's late,' he muttered.", "Tom's dog barked."] * 10
check(p._single_quoted(british) and not p._single_quoted(['"Hello," she said.'] * 30),
      "a book that uses single quotes for speech is recognised")

# 2. the whole pipeline on a made-up book and recording
BOOK = """## Chapter 1

The harbour was cold and grey when the ship came in.

"You're late," Sarah snapped. "The ship left an hour ago."

Tom dropped his bag on the wet stones and looked out at the empty water for a long time.

"I know," he whispered softly. "I ran the whole way here."

"Then you'll swim," she said.

"No chance."

The gulls cried over the harbour while the two of them stood there in the rain.

"There is another boat at noon," the old harbour master called out from the door of his hut.

"It goes the long way around the islands. You can sleep on the deck if you pay me now.

"And mind the captain, he bites."
"""
book = TMP / "harbour.txt"
book.write_text(BOOK, encoding="utf-8")
info, sentences, lines = p.book_sentences(book, TMP / "work", dialogue=True)
spoken = [t for (t, skip), ln in zip(sentences, lines) if not skip]
check("You're late." in spoken and "No chance." in spoken and not any("snapped" in t for t in spoken),
      "the book's lines are the quoted words only")

SR = p.SR
rng = np.random.default_rng(0)
words, audio, t = [], [np.zeros(int(0.5 * SR))], 0.5
for para in [x for x in info["text"].split("\n\n") if x.strip() and not x.startswith("## ")]:
    toks = p.clean_text(para).split()
    for k, w in enumerate(toks):
        d = 0.09 * len(w.strip('".,!?')) + 0.12
        env = np.sin(np.linspace(0, np.pi, int(d * SR))) ** 0.5
        audio.append(rng.normal(0, 0.12, len(env)).astype(np.float32) * env)
        words.append([w, t, t + d])
        t += d
        gap = 0.5 if w[-1] in '.,!?"' or k == len(toks) - 1 else 0.07
        audio.append(np.zeros(int(gap * SR)))
        t += gap
wav = np.concatenate(audio).astype(np.float32)
# a quiet room whose noise comes and goes, as a real recording's does (perfectly even silence makes every 10 ms of a
# pause equally quiet, and the quietest-point search then settles on the pause's edge, against a word)
hop = SR // 100
swell = (0.2 + 0.8 * rng.random(len(wav) // hop + 1)).repeat(hop)[:len(wav)]
wav += (rng.normal(0, 0.001, len(wav)) * swell).astype(np.float32)
rec = TMP / "reading.wav"
sf.write(str(rec), wav, SR)
p.transcribe_file = lambda voice, d24, cache, language, stop, progress: [list(w) for w in words]


def prepare(voice, dialogue):
    p._jobs[voice] = {"log": [], "stage": "", "progress": 0.0}
    opts = {"min_seconds": 1.0, "max_seconds": 15.0, "target_seconds": 8.0, "min_match": 0.9, "min_snr": 20.0,
            "dialogue_only": 1.0 if dialogue else 0.0, "backend": "mac", "keep_work": True}
    p.process(voice, [rec], book, "Harbour", "english", opts, threading.Event())
    rows = [json.loads(x) for x in (p.TRAINING / voice / "books" / "harbour" / "clips.jsonl").read_text().splitlines()]
    return rows, p._jobs[voice]["log"]


rows, log = prepare("tom-performer", True)
texts = [r["text"] for r in rows]
print("     clips:", texts)
LINES = {"You're late.", "The ship left an hour ago.", "I know.", "I ran the whole way here.", "Then you'll swim.",
         "No chance.", "There is another boat at noon.", "It goes the long way around the islands.",
         "You can sleep on the deck if you pay me now.", "And mind the captain, he bites.",
         "It goes the long way around the islands. You can sleep on the deck if you pay me now."}
check(len(rows) >= 6, f"most lines became clips ({len(rows)})")
check(all(x in LINES for x in texts), "every clip is one character's line: no narration, no two speakers together")
check(not any(w in " ".join(texts) for w in ("snapped", "whispered", "harbour master", "gulls")), "the narration is left out")
hows = {r["text"]: r.get("how") for r in rows}
check(hows.get("You're late.") == "snapped" and hows.get("I ran the whole way here.") == "whispered softly"
      and hows.get("Then you'll swim.") == "said" and hows.get("There is another boat at noon.") == "called"
      and hows.get("No chance.") is None, f"each clip keeps how it was said, and only when the book says ({hows})")
check(any("Dialogue only:" in str(x) for x in log) and any("How the lines were said" in str(x) for x in log),
      "the log says how much is dialogue and how the lines were said")
ds = p.TRAINING / "tom-performer" / "dataset"
check((ds / "metadata.jsonl").exists() and (ds / "ref.wav").exists(), "a normal training dataset is made from them")

rows, _ = prepare("tom-narrator", False)
check(any("snapped" in r["text"] or "harbour" in r["text"].lower() for r in rows),
      f"without Dialogue only the narration is kept, as before ({len(rows)} clips)")

# 3. where clips are cut: in the middle of a pause, also when the pause is perfectly even (digital silence or steady
# room tone), not at its edge against a word (np.convolve's zero padding made the edges look quietest)
word = (rng.normal(0, 0.1, SR // 2)).astype(np.float32)
for name, pause in (("digital silence", np.zeros(int(0.4 * SR), np.float32)),
                    ("steady room tone", rng.normal(0, 0.002, int(0.4 * SR)).astype(np.float32))):
    x = np.concatenate([word, pause, word])
    cut = p._cut_point(x, 0.5, 0.9, SR)
    check(0.6 <= cut <= 0.8, f"{name}: cut at {cut:.2f} s, inside the 0.50–0.90 s pause and away from both words")
x = np.concatenate([word, rng.normal(0, 0.004, int(0.2 * SR)), np.zeros(int(0.2 * SR)), word]).astype(np.float32)
check(0.7 <= p._cut_point(x, 0.5, 0.9, SR) <= 0.9, "a breath in the pause: the cut goes in the quiet part")
before = len(rows)
wav_even = np.concatenate(audio).astype(np.float32)  # the same reading with perfectly silent pauses
sf.write(str(rec), wav_even, SR)
rows, _ = prepare("tom-even", False)
check(len(rows) >= before - 1, f"perfectly silent pauses keep as many clips ({len(rows)}, with room tone {before})")

print("\nall passed" if not failed else f"\n{len(failed)} failed: " + "; ".join(failed))
sys.exit(1 if failed else 0)
