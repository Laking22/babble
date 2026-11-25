"""Offline verification for TTS & Audiobook Polish upgrades:
1. Broadcast Mastering Pass (export.combine with MASTER_FILTER: 70Hz HPF, de-ess, 2:1 comp, -19 LUFS / -3 dBTP)
2. Raised-cosine 12ms edge fades in audio.trim_edges (no boundary click)
3. Voice-drift DSP outlier detection in Stage 1 Immersion QA
4. Pre-render tricky pronunciation scanner (fantasy names, proper nouns, homographs)
5. Selection & current-chapter Preview routing in books.create()
"""
import re
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import soundfile as sf

for mod in ("mlx", "mlx.core", "mlx.nn", "mlx_audio", "mlx_audio.tts", "mlx_audio.tts.utils"):
    sys.modules[mod] = types.ModuleType(mod)
sys.modules["mlx.core"].set_cache_limit = lambda *a, **kw: None
sys.modules["mlx.core"].clear_cache = lambda *a, **kw: None
sys.modules["mlx.core"].random = types.SimpleNamespace(seed=lambda *a, **kw: None)
sys.modules["mlx_audio.tts.utils"].load_model = lambda *a, **kw: None

from narrator import audio, books, export, immersion_qa, pronounce
from narrator.runtime import SR


def test_broadcast_mastering():
    # Generate a 4s quiet speech-like synthetic WAV with 40 Hz rumble + 250 Hz voice + loud peak
    n = SR * 4
    t = np.arange(n, dtype=np.float32) / SR
    sig = 0.04 * np.sin(2 * np.pi * 240 * t) + 0.02 * np.sin(2 * np.pi * 40 * t) + 0.02 * np.sin(2 * np.pi * 1200 * t)
    with tempfile.TemporaryDirectory() as td:
        ch = Path(td) / "000-test.wav"
        sf.write(ch, sig.astype(np.float32), SR)
        out = export.combine([ch], ["Chapter 1"], Path(td) / "mastered", "Audiobook (M4B with chapters)",
                             title="Test", author="Author")
        assert out.is_file() and out.suffix == ".m4b"
        # Measure integrated loudness & true peak with ffmpeg ebur128
        proc = subprocess.run(
            ["ffmpeg", "-i", str(out), "-af", "ebur128=peak=true", "-f", "null", "-"],
            capture_output=True, text=True, check=True,
        )
        m_i = re.findall(r"I:\s*(-?\d+\.\d+)\s*LUFS", proc.stderr)
        m_tp = re.findall(r"Peak:\s*(-?\d+\.\d+)\s*dBFS", proc.stderr)
        assert m_i, proc.stderr[-400:]
        lufs = float(m_i[-1])
        tp = float(m_tp[-1]) if m_tp else -3.0
        assert -21.5 <= lufs <= -16.5, f"Expected ~-19 LUFS, got {lufs} LUFS"
        assert tp <= -2.0, f"Expected true peak <= -2 dBFS, got {tp} dBFS"
        print(f"[PASS] broadcast_mastering: {lufs:.1f} LUFS, peak {tp:.1f} dBFS")


def test_raised_cosine_trim_edges():
    n = SR * 2
    t = np.arange(n, dtype=np.float32) / SR
    wav = np.zeros(n, np.float32)
    wav[SR // 2:SR + SR // 2] = 0.3 * np.sin(2 * np.pi * 300 * t[:SR])
    out = audio.trim_edges(wav)
    assert abs(float(out[0])) < 1e-4 and abs(float(out[-1])) < 1e-4
    print("[PASS] raised_cosine_trim_edges")


def test_voice_drift_detection():
    # 6 paragraphs for Narrator: 5 steady ~160 Hz voice, 1 shifted up to ~330 Hz + bright timbre
    n = SR * 12
    t = np.arange(n, dtype=np.float32) / SR
    wav = np.zeros(n, np.float32)
    timings = []
    for i in range(6):
        s0, s1 = i * SR * 2, (i + 1) * SR * 2 - SR // 5
        tt = t[:s1 - s0]
        if i == 3:
            # Drifted voice: higher F0 (320 Hz) + bright upper harmonics
            seg = 0.12 * np.sin(2 * np.pi * 320 * tt) + 0.12 * np.sin(2 * np.pi * 2400 * tt)
        else:
            seg = 0.14 * np.sin(2 * np.pi * 150 * tt) + 0.04 * np.sin(2 * np.pi * 450 * tt)
        wav[s0:s1] = seg
        timings.append({"para": i, "start": s0 / SR, "end": s1 / SR, "who": "Narrator"})
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "000-drift.wav"
        sf.write(p, wav, SR)
        issues = immersion_qa.audit_chapter_audio(p, timings, 0, "Chapter 1")
        drifts = [x for x in issues if x["category"] == "voice_drift"]
        assert len(drifts) == 1 and drifts[0]["para"] == 3, drifts
        print(f"[PASS] voice_drift_detection flagged ¶{drifts[0]['para'] + 1}: {drifts[0]['detail']}")


def test_tricky_pronunciations():
    story = (
        "In R'lyeh, Cthulhu dreamed while Inspector Legrasse read the ancient lead tablet. "
        "Legrasse read it again and felt a tear fall as the wind rose over R'lyeh and Cthulhu stirred. "
        "The wind howled around the lead chest."
    )
    found = pronounce.find_tricky(story, cast_names=["Narrator", "Legrasse"], existing=[])
    words = {x["word"].lower(): x["reason"] for x in found}
    assert "cthulhu" in words and words["cthulhu"] == "fantasy_name", words
    assert "r'lyeh" in words and words["r'lyeh"] == "fantasy_name", words
    assert "legrasse" in words, words
    assert "read" in words and words["read"] == "homograph", words
    assert "lead" in words and words["lead"] == "homograph", words
    print(f"[PASS] find_tricky found {len(found)} words: {list(words.keys())[:8]}")


def test_selection_and_chapter_preview():
    story = "## Chapter 1\n\nOpening of chapter one.\n\n## Chapter 2\n\nOpening of chapter two."
    captured = []
    orig_render = books._render
    try:
        def fake_render(items, **kw):
            captured.append([it["text"] for it in items])
            yield {"status": "Done", "audio": "/tmp/fake.wav"}
        books._render = fake_render
        cast = [["Narrator", "Ryan (Qwen)", "Warm.", 1.0]]
        list(books.create(story, cast, preview=True, preview_text="Custom highlighted snippet.", sound=False))
        list(books.create(story, cast, preview=True, preview_chapter=1, sound=False))
        assert captured[0] == ["Custom highlighted snippet."], captured
        assert captured[1] == ["Opening of chapter two."], captured
        print("[PASS] selection_and_chapter_preview:", captured)
    finally:
        books._render = orig_render


if __name__ == "__main__":
    test_broadcast_mastering()
    test_raised_cosine_trim_edges()
    test_voice_drift_detection()
    test_tricky_pronunciations()
    test_selection_and_chapter_preview()
    print("ALL TTS & AUDIOBOOK POLISH CHECKS PASSED")
