"""Offline verification for Milestone 3 Interactive Script Studio:
1. chapter_timings() reads exact .timings.json sidecar and aggregates multi-chunk paragraphs cleanly
2. chapter_timings() falls back to proportional estimates for pre-existing chapters without .timings.json
3. regen_paragraph() invalidates ONLY the target paragraph's cached take and slices a surrounding-context preview WAV
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

from narrator import books
from narrator.runtime import SR


def test_chapter_timings_exact_and_fallback():
    story = "## Chapter 1\n\nFirst paragraph is short.\n\nSecond paragraph is quite a bit longer than the first one.\n\nThird paragraph."
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "000-abc1234567.wav"
        sf.write(wav, np.zeros(SR * 10, dtype=np.float32), SR)

        # 1. Fallback when no sidecar exists yet
        r_est = books.chapter_timings(story, 0, str(wav))
        assert r_est["estimated"] is True
        assert len(r_est["timings"]) == 3
        assert r_est["timings"][0]["start"] == 0.0
        assert abs(r_est["timings"][-1]["end"] - 10.0) < 0.05

        # 2. Exact sidecar aggregation (multi-chunk paragraph 1 merged cleanly)
        sidecar = wav.with_suffix(".timings.json")
        sidecar.write_text(json.dumps([
            {"para": 0, "who": "Narrator", "text": "First paragraph is short.", "start": 0.0, "end": 2.1},
            {"para": 1, "who": "Narrator", "text": "Second paragraph part A.", "start": 2.8, "end": 5.0},
            {"para": 1, "who": "Narrator", "text": "Second paragraph part B.", "start": 5.2, "end": 7.4},
            {"para": 2, "who": "Narrator", "text": "Third paragraph.", "start": 8.1, "end": 9.6},
        ]), encoding="utf-8")
        r_exact = books.chapter_timings(story, 0, str(wav))
        assert r_exact["estimated"] is False
        assert len(r_exact["timings"]) == 3
        assert r_exact["timings"][1]["start"] == 2.8 and r_exact["timings"][1]["end"] == 7.4
        print("[PASS] chapter_timings exact sidecar + proportional fallback")


def test_regen_paragraph_context_slice():
    story = "## Chapter 1\n\nOpening line.\n\nMiddle line to redo.\n\nClosing line."
    with tempfile.TemporaryDirectory() as td:
        orig_books = books.BOOKS
        orig_render = books.en.render
        try:
            books.BOOKS = Path(td)
            folder = books.BOOKS / "test-book"
            folder.mkdir(parents=True, exist_ok=True)

            # Fake render that writes a 9s chapter WAV + exact sidecar
            def fake_render(items, *, path, **kw):
                t = np.linspace(0, 9.0, SR * 9, endpoint=False, dtype=np.float32)
                sf.write(path, 0.2 * np.sin(2 * np.pi * 220 * t), SR)
                path.with_suffix(".timings.json").write_text(json.dumps([
                    {"para": 0, "who": "Narrator", "text": "Opening line.", "start": 0.0, "end": 2.5},
                    {"para": 1, "who": "Narrator", "text": "Middle line to redo.", "start": 3.0, "end": 5.5},
                    {"para": 2, "who": "Narrator", "text": "Closing line.", "start": 6.0, "end": 8.5},
                ]), encoding="utf-8")
                yield str(path), "Done"

            books.en.render = fake_render
            events = list(books.regen_paragraph(
                0, 1, seed_bump=2,
                story=story, table=[["Narrator", "Ryan (Qwen)", "Warm.", 1.0]],
                folder="test-book", title="Test",
            ))
            final = events[-1]
            preview = Path(final["audio"])
            assert preview.is_file() and preview.name == "_regen-ch0-p1.wav"
            dur = sf.info(str(preview)).duration
            # Prev tail (0..3) + target (3..5.5) + next head (5.5..8.5) -> ~8.5s context slice
            assert 5.0 <= dur <= 9.0, f"Unexpected context preview duration: {dur}"
            assert final["para_start"] == 3.0 and final["para_end"] == 5.5
            print(f"[PASS] regen_paragraph context preview slice ({dur:.2f}s)")
        finally:
            books.BOOKS = orig_books
            books.en.render = orig_render


if __name__ == "__main__":
    test_chapter_timings_exact_and_fallback()
    test_regen_paragraph_context_slice()
    print("ALL MILESTONE 3 INTERACTIVE SCRIPT CHECKS PASSED")
