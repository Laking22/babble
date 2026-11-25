"""Offline verification for Milestone 4 Two-Stage Immersion QA pipeline:
1. Stage 1a script audit catches unclosed braces, unknown speakers, and missing sounds
2. Stage 1b DSP audio audit catches loudness jumps (>= 7.5 dB), sustained clipping, splice clicks, and >= 4s dead air
3. Stage 2 cost estimator + cloud narrative continuity parser produce structured issues with chapter/para/start
"""
import json
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
sys.modules["mlx_audio.tts.utils"].load_model = lambda *a, **kw: None

from narrator import director, immersion_qa
from narrator.runtime import SR


def test_stage1_script_audit():
    story = (
        "## Chapter 1\n\n"
        "Normal opening line.\n\n"
        "{amb: unclosed rain tag here\n\n"
        "[Ghost] Who goes there?\n\n"
        "{sfx: nonexistent alien laser blaster xyz} Boom."
    )
    issues = immersion_qa.audit_script(story, cast_names=["Narrator"], quotes_only=True)
    cats = {i["category"] for i in issues}
    assert "tag_syntax" in cats, cats
    assert "unknown_speaker" in cats, cats
    assert "missing_sound" in cats, cats
    print(f"[PASS] stage1_script_audit caught {sorted(cats)}")


def test_stage1_audio_dsp_audit():
    # Build a synthetic 12s chapter with:
    # - para 0 (0.0-2.0s): normal -20 dB tone
    # - para 1 (2.0-4.0s): loud jump (-10 dB) + sustained clipping + sharp splice click at 2.0s
    # - 4.0-8.5s: 4.5s digital dead air (no {pause:} tag)
    # - para 2 (8.5-10.5s): normal -20 dB tone
    n = SR * 12
    t = np.arange(n, dtype=np.float32) / SR
    wav = np.zeros(n, dtype=np.float32)
    wav[:SR * 2] = 0.14 * np.sin(2 * np.pi * 220 * t[:SR * 2])
    # Sharp click + loud clipped segment
    seg1 = 0.45 * np.sin(2 * np.pi * 220 * t[SR * 2:SR * 4])
    seg1[0] = 0.85  # instantaneous splice click
    seg1[100:130] = 0.998  # sustained clipping
    wav[SR * 2:SR * 4] = seg1
    wav[int(8.5 * SR):int(10.5 * SR)] = 0.14 * np.sin(2 * np.pi * 220 * t[:SR * 2])

    timings = [
        {"para": 0, "start": 0.0, "end": 2.0, "who": "Narrator"},
        {"para": 1, "start": 2.0, "end": 4.0, "who": "Narrator"},
        {"para": 2, "start": 8.5, "end": 10.5, "who": "Narrator"},
    ]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "000-test.wav"
        sf.write(p, wav, SR, subtype="FLOAT")
        issues = immersion_qa.audit_chapter_audio(p, timings, 0, "Chapter 1", ["A", "B", "C"])
        cats = {i["category"] for i in issues}
        assert "loudness_jump" in cats, cats
        assert "clipping" in cats, cats
        assert "splice_click" in cats, cats
        assert "dead_air" in cats, cats
        print(f"[PASS] stage1_audio_dsp_audit caught {sorted(cats)}")


def test_stage2_estimate_and_mock_review():
    story = "## Chapter 1\n\n{amb: heavy rain} He stepped inside the quiet library and shut the door.\n\n[Sarah] \"I am glad to be dry,\" he said."
    est = immersion_qa.estimate_stage2(story, "anthropic", "claude-haiku-5-5")
    assert est["chapters"] == 1 and est["high"] is not None
    # Mock cloud call to verify Stage 2 parsing without network
    orig_key = director.key
    orig_claude = director._ask_claude
    try:
        director.key = lambda pid: "sk-test"
        director._ask_claude = lambda *a, **kw: (json.dumps({"issues": [
            {"para": 1, "severity": "medium", "category": "ambience_mismatch",
             "title": "Rain continues indoors", "detail": "Heavy rain ambience stays active after entering the quiet library."},
            {"para": 2, "severity": "high", "category": "speaker_mismatch",
             "title": "Wrong speaker tag", "detail": "Line attributed to 'he said' is tagged [Sarah]."},
        ]}), 0.001)
        s2 = immersion_qa.audit_narrative_cloud(story, [["Narrator", "Ryan (Qwen)", "", 1.0], ["Sarah", "Tara (Orpheus)", "", 1.0]], "anthropic", "claude-haiku-5-5")
        assert len(s2) == 2
        assert {i["category"] for i in s2} == {"ambience_mismatch", "speaker_mismatch"}
        assert s2[0]["para"] == 0 and s2[1]["para"] == 1
        print(f"[PASS] stage2_estimate ({est['cost']}) & mock review ({len(s2)} issues)")
    finally:
        director.key = orig_key
        director._ask_claude = orig_claude


if __name__ == "__main__":
    test_stage1_script_audit()
    test_stage1_audio_dsp_audit()
    test_stage2_estimate_and_mock_review()
    print("ALL MILESTONE 4 IMMERSION QA CHECKS PASSED")
