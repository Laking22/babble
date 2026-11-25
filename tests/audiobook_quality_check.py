"""Offline checks for the 6 audiobook generation quality improvements (no GPU/models required)."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "narrator" / "kaggle_render"))

from narrator import ai_detect, audio, books, detect_core, export, pronounce  # noqa: E402
import runner  # noqa: E402


def test_context_pauses():
    assert books._gap("Hello,", 0.2) == 0.06
    assert books._gap("Hello—", 0.2) == 0.06
    assert books._gap("Hello.", 0.2) == 0.2
    assert books._gap("*She was gone.*", 0.2) == 0.2
    assert books._gap("He laughed. [laughs]", 0.2) == 0.2
    assert books._gap("(He was gone.)", 0.2) == 0.2
    assert books._gap("He laughed. <laugh>", 0.2) == 0.2
    assert books._gap("and the house <sigh>", 0.2) == 0.06
    assert books._gap("Hello...", 0.2, para_pause=0.7) == 0.32
    assert books._gap("“Wait…”", 0.2, para_pause=0.7) == 0.32
    assert books._gap("“Stop.”", 0.2, speaker_change=True, para_pause=0.7) == 0.25
    assert books._gap("“Stop,”", 0.2, speaker_change=True, para_pause=0.7) == 0.06
    assert books._gap("Hello...", 0.2, para_pause=0.25) == 0.25
    cast = {"Narrator": ("builtin:ryan", "", 1.0), "Alice": ("builtin:aiden", "", 1.0)}
    paras = [[("Alice", "", "“Wait…”"), ("Narrator", "", "she whispered."), ("Alice", "", "“Did you hear that?”")]]
    items = books._items(paras, cast, 1.0, 0.2, 0.7)
    assert [it["pause"] for it in items] == [0.32, 0.25, 0.7]
    # Single-voice TTS tab (books.speak) also uses context-aware _gap
    assert books._gap("Hello...", 0.25, para_pause=0.5) == 0.4
    assert books._gap("Hello,", 0.25, para_pause=0.5) == 0.06


def test_stereo_m4b_bitrate():
    sr = 24000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False, dtype=np.float32)
    mono = 0.2 * np.sin(2 * np.pi * 440 * t)
    stereo = np.stack([mono, 0.2 * np.sin(2 * np.pi * 660 * t)], axis=1)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        sf.write(tmp / "m.wav", mono, sr)
        sf.write(tmp / "s.wav", stereo, sr)
        out_m = export.combine([tmp / "m.wav"], ["Ch 1"], tmp / "mono_book", "Audiobook (M4B with chapters)", "T", "A")
        out_s = export.combine([tmp / "s.wav"], ["Ch 1"], tmp / "stereo_book", "Audiobook (M4B with chapters)", "T", "A")

        def probe(p):
            r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                                "-show_entries", "stream=channels,bit_rate", "-of", "json", str(p)],
                               capture_output=True, text=True, check=True)
            st = json.loads(r.stdout)["streams"][0]
            return int(st["channels"]), int(st["bit_rate"])

        ch_m, br_m = probe(out_m)
        ch_s, br_s = probe(out_s)
        assert ch_m == 1 and br_m < 85000, f"mono={ch_m}ch @{br_m}"
        assert ch_s == 2 and br_s > 95000, f"stereo={ch_s}ch @{br_s}"


def test_cloud_rushed_parity():
    sr = 24000
    txt = "a" * 100
    norm_wav = 0.2 * np.sin(2 * np.pi * 200 * np.linspace(0, 6.0, sr * 6, endpoint=False, dtype=np.float32))
    rush_wav = 0.2 * np.sin(2 * np.pi * 200 * np.linspace(0, 3.5, int(sr * 3.5), endpoint=False, dtype=np.float32))
    local_takes = {f"k{i}": ("qwen:clarke", txt, norm_wav) for i in range(20)}
    local_takes["bad"] = ("qwen:clarke", txt, rush_wav)
    cloud_takes = {k: (v, tx, runner.voiced_seconds(w)) for k, (v, tx, w) in local_takes.items()}
    assert audio.rushed(local_takes) == runner.rushed_keys(cloud_takes) == {"bad"}


def test_whisper_leveling():
    sr = 24000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    whisper = 0.015 * np.sin(2 * np.pi * 220 * t)

    def rms_db(x):
        return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)

    assert audio.is_soft("angry whisper")
    assert audio.is_soft("", "[whispering] don't move")
    assert not audio.is_soft("warm, calm storytelling", "Hello.")
    assert rms_db(audio.level(whisper, soft=True)) < rms_db(audio.level(whisper, soft=False)) - 5.0


def test_text_normalization():
    cases = [
        ("Dr. Smith paid $1.50 for Chapter IV.", "Doctor Smith paid 1 dollar and 50 cents for Chapter 4."),
        ("Mr. and Mrs. Jones met at St. Jude vs. St. Louis.", "Mister and Missus Jones met at Saint Jude versus Saint Louis."),
        ("Cost was $5 million, £20, or €15.", "Cost was 5 million dollars, 20 pounds, or 15 euros."),
        ("I walked into the room with King Henry VIII.", "I walked into the room with King Henry the Eighth."),
        ("Henry V met Louis XIV in Chapter IV.", "Henry the Fifth met Louis the Fourteenth in Chapter 4."),
        ("{sfx: $50 door} [Dr. Evil] <laugh> Dr. Evil paid $1.", "{sfx: $50 door} [Dr. Evil] <laugh> Doctor Evil paid 1 dollar."),
        ("Use e.g. tea, etc. on St. street.", "Use for example tea, et cetera on St. street."),
        ("Pope John XXIII and Louis XVIII.", "Pope John the Twenty-Third and Louis the Eighteenth."),
        ("In Elizabeth I's reign, Peter I of Russia and Elizabeth I.", "In Elizabeth the First's reign, Peter the First of Russia and Elizabeth the First."),
        ("Queen Elizabeth I ruled.", "Queen Elizabeth the First ruled."),
        ("Mary I think so. Henry I'm here. King John I said no.", "Mary I think so. Henry I'm here. King John I said no."),
        ("Chapter I: World War II, Rocky II, Type II.", "Chapter 1: World War II, Rocky II, Type II."),
    ]
    for inp, want in cases:
        assert pronounce.normalize(inp) == want
    assert pronounce.apply("Dr. Doom", [{"word": "Dr.", "say": "Doktor"}]) == "Doktor Doom"


def test_multi_speaker_split():
    p1 = '"Wait," Alice said. "No way!" Bob yelled.'
    assert detect_core.split_multi_speaker(p1) == ['"Wait," Alice said.', '"No way!" Bob yelled.']
    p2 = '"Wait," Alice said, "don\'t go!"'
    assert detect_core.split_multi_speaker(p2) == [p2]
    p3 = '"Hello." "Hi there."'
    assert detect_core.split_multi_speaker(p3) == ['"Hello."', '"Hi there."']
    _, texts = ai_detect.chapters_of(f"## Ch 1\n\n{p1}\n\n{p2}")
    assert texts == ['"Wait," Alice said.', '"No way!" Bob yelled.', p2]


if __name__ == "__main__":
    for fn in (test_context_pauses, test_stereo_m4b_bitrate, test_cloud_rushed_parity,
               test_whisper_leveling, test_text_normalization, test_multi_speaker_split):
        fn()
        print(f"OK: {fn.__name__}")
    print("ALL OK")
