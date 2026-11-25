"""Offline verification for Milestone 2 audio immersion upgrades:
1. Room tone floor (continuous warm air around -66 dB, under ACX's -60 dB; never dead zeros, never loud)
2. Dynamic whisper-aware ducking (music/ambience duck ~5-6 dB deeper under whispered speech vs normal speech)
3. Subtle spatial panning (slight left/right, left/right, hard left/right equal-power balance)
4. Per-paragraph pacing ({pace: slow/fast/0.9} applied to item speed, stripped cleanly, cache-compatible chapter_file)

    .venv/bin/python tests/immersion_dsp_check.py
"""
import os
import sys
import tempfile
import types
from pathlib import Path

os.environ["NARRATOR_HOME"] = tempfile.mkdtemp(prefix="immersion-check-")  # never touch the real data folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import soundfile as sf

for mod in ("mlx", "mlx.core", "mlx.nn", "mlx_audio", "mlx_audio.tts", "mlx_audio.tts.utils"):
    sys.modules[mod] = types.ModuleType(mod)
sys.modules["mlx.core"].set_cache_limit = lambda *a, **kw: None
sys.modules["mlx.core"].clear_cache = lambda *a, **kw: None
sys.modules["mlx_audio.tts.utils"].load_model = lambda *a, **kw: None

from narrator import books, sfx
from narrator.runtime import SR


def rms_db(a):
    return float(20 * np.log10(np.sqrt(np.mean(np.asarray(a, np.float32) ** 2)) + 1e-12))


def test_room_tone():
    rt_st = sfx.room_tone(SR * 4, stereo=True)
    assert rt_st.shape == (SR * 4, 2)
    mid_db = rms_db(rt_st[SR:SR * 3])
    assert -69.0 <= mid_db <= -63.0, f"Unexpected stereo room tone level: {mid_db:.2f} dB"
    assert float(np.abs(rt_st[:, 0] - rt_st[:, 1]).max()) > 1e-6, "Stereo room tone should have decorrelated channels"
    rt_mono = sfx.room_tone(SR * 4, stereo=False)
    assert rt_mono.shape == (SR * 4,)
    assert -69.0 <= rms_db(rt_mono[SR:SR * 3]) <= -63.0
    print(f"[PASS] room_tone: stereo={mid_db:.1f} dB, mono={rms_db(rt_mono[SR:SR * 3]):.1f} dB")


def test_whisper_ducking():
    # 6s speech track: 0.5-2.0s normal (-20 dB), 2.5-3.5s silence, 3.5-5.0s whisper (-26 dB)
    n = SR * 6
    t = np.arange(n, dtype=np.float32) / SR
    tone = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    dry = np.zeros(n, np.float32)
    dry[int(0.5 * SR):int(2.0 * SR)] = tone[:int(1.5 * SR)] * (10 ** (-20 / 20) * np.sqrt(2))
    dry[int(3.5 * SR):int(5.0 * SR)] = tone[:int(1.5 * SR)] * (10 ** (-26 / 20) * np.sqrt(2))
    spans = [(int(0.5 * SR), int(2.0 * SR)), (int(3.5 * SR), int(5.0 * SR))]

    w = sfx._whisper_duck(dry, spans, n)
    assert w is not None
    normal_mult = float(np.mean(w[int(0.9 * SR):int(1.8 * SR)]))
    whisper_mult = float(np.mean(w[int(4.0 * SR):int(4.8 * SR)]))
    extra_db = 20 * np.log10(whisper_mult / normal_mult)
    assert abs(normal_mult - 1.0) < 0.02, f"Normal speech shouldn't trigger whisper duck: {normal_mult}"
    assert -6.5 <= extra_db <= -3.5, f"Expected ~-4.5 to -6 dB extra whisper ducking, got {extra_db:.2f} dB"
    print(f"[PASS] whisper_ducking: normal_mult={normal_mult:.3f}, whisper extra duck={extra_db:.2f} dB")


def test_spatial_panning():
    cues = [c for k, c in sfx.split_cues("{sfx: knock, slight left} {sfx: knock, slight right} {sfx: knock, hard right}") if k == "cue"]
    assert [c["pan"] for c in cues] == [-0.35, 0.35, 1.0], cues
    mono = np.ones((1000, 2), np.float32) * 0.5
    sl = sfx._pan(mono, -0.35)
    assert float(sl[:, 0].mean()) > float(sl[:, 1].mean()) > 0.25
    # equal power: L^2 + R^2 == 2 * 0.5^2 == 0.5
    power = float(sl[0, 0] ** 2 + sl[0, 1] ** 2)
    assert abs(power - 0.5) < 1e-4, power
    print(f"[PASS] spatial_panning: slight left L={sl[0,0]:.3f} R={sl[0,1]:.3f}, equal-power={power:.4f}")


def test_per_paragraph_pacing_and_cache_compat():
    cast = {"Narrator": ("builtin:ryan", "Warm.", 1.0)}
    paras = [
        [("Narrator", "", "{pace: slow} She opened the door carefully.")],
        [("Narrator", "", "{pace: fast} Run now!")],
        [("Narrator", "", "Normal speed paragraph.")],
    ]
    items = books._items(paras, cast, speed=1.0, line_pause=0.2, para_pause=0.7)
    assert len(items) == 3
    assert items[0]["speed"] == 0.88 and "{pace" not in items[0]["text"] and items[0]["para"] == 0
    assert items[1]["speed"] == 1.12 and "{pace" not in items[1]["text"] and items[1]["para"] == 1
    assert items[2]["speed"] == 1.0 and items[2]["para"] == 2
    assert sfx.strip("{pace: slow} Hello {pause: 1.5} world", keep_pauses=True) == "Hello {pause: 1.5} world"
    assert sfx.missing("{pace: slow} Hello") == []
    # chapter_file hash must ignore internal 'para' key for backwards compatibility
    with tempfile.TemporaryDirectory() as td:
        a = books.chapter_file(Path(td), 0, [{"text": "Hi", "voice": "builtin:ryan", "para": 0}], "english", 55, 0, True, 1.0)
        b = books.chapter_file(Path(td), 0, [{"text": "Hi", "voice": "builtin:ryan"}], "english", 55, 0, True, 1.0)
        assert a == b, f"Hash mismatch: {a} != {b}"
    print("[PASS] per_paragraph_pacing & cache_compat")


if __name__ == "__main__":
    test_room_tone()
    test_whisper_ducking()
    test_spatial_panning()
    test_per_paragraph_pacing_and_cache_compat()
    print("ALL MILESTONE 2 DSP CHECKS PASSED")
