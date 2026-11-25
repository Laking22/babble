"""Verification suite for full-app review hardening fixes and new features."""
import json
import os
import sys
import tempfile
import types
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Isolated temp home so no user data is touched
_tmp_home = tempfile.mkdtemp(prefix="babble-review-test-")
os.environ["NARRATOR_HOME"] = _tmp_home

# Stub MLX for headless sandbox execution
for mod_name in ("mlx", "mlx.core", "mlx.nn", "mlx_audio", "mlx_audio.tts", "mlx_audio.tts.utils"):
    if mod_name not in sys.modules:
        m = types.ModuleType(mod_name)
        if mod_name == "mlx.core":
            m.set_cache_limit = lambda *a, **k: None
            m.clear_cache = lambda *a, **k: None
        if mod_name == "mlx_audio.tts.utils":
            m.load_model = lambda *a, **k: None
        sys.modules[mod_name] = m

import numpy as np
import soundfile as sf

from narrator import acoustics, audio, export, library, preflight, projects, series, snapshots
from narrator.dataset import health


def check_export():
    assert "highpass=f=75" in export.MASTER_FILTER, export.MASTER_FILTER
    assert "TP=-3.5" in export.MASTER_FILTER, export.MASTER_FILTER
    src = (ROOT / "narrator" / "export.py").read_text(encoding="utf-8")
    assert export.QUALITY["standard"]["mp3"] == 192  # (192k MP3 at the standard file quality)
    assert "+faststart+use_metadata_tags" in src
    print("export: OK")


def check_audio_dsp():
    sr = 24000
    t = np.arange(sr, dtype=np.float32) / sr
    # High-pass: 40 Hz should drop sharply, 1 kHz should stay near 0 dB
    low = np.sin(2 * np.pi * 40 * t).astype(np.float32)
    mid = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    hp_low = audio.highpass(low, hz=75.0, sr=sr)
    hp_mid = audio.highpass(mid, hz=75.0, sr=sr)
    assert np.sqrt(np.mean(hp_low ** 2)) < 0.35 * np.sqrt(np.mean(low ** 2))
    assert np.sqrt(np.mean(hp_mid ** 2)) > 0.90 * np.sqrt(np.mean(mid ** 2))

    # De-ess: strong 7 kHz sibilant reduced, 500 Hz untouched
    sib = 0.6 * np.sin(2 * np.pi * 7000 * t).astype(np.float32)
    vox = 0.2 * np.sin(2 * np.pi * 500 * t).astype(np.float32)
    de_sib = audio.deess(sib, sr=sr)
    de_vox = audio.deess(vox, sr=sr)
    assert np.sqrt(np.mean(de_sib ** 2)) < np.sqrt(np.mean(sib ** 2))
    assert abs(np.sqrt(np.mean(de_vox ** 2)) - np.sqrt(np.mean(vox ** 2))) < 0.03

    # Room profile & match_room
    rng = np.random.default_rng(42)
    ref = rng.standard_normal(sr).astype(np.float32) * 0.1
    dull = audio.highpass(ref, hz=80.0, sr=sr) * 0.5
    prof = audio.room_profile(ref, sr=sr)
    matched = audio.match_room(dull, prof, sr=sr)
    assert matched.shape == dull.shape and np.all(np.isfinite(matched))
    print("audio_dsp: OK")


def check_library_pickle_guard():
    zpath = Path(_tmp_home) / "bad.narrator-pack"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("pack.json", json.dumps({"format": 1, "id": "evil", "name": "Evil", "voices": []}))
        z.writestr("model/weights.pt", b"pickle-payload")
    try:
        library.install_pack(zpath)
        raise AssertionError("Expected ValueError for .pt inside pack")
    except ValueError as e:
        assert "Unsafe" in str(e), str(e)
    print("library_pickle_guard: OK")


def check_preflight():
    sr = 24000
    t = np.arange(sr * 2, dtype=np.float32) / sr
    # Good ACX chapter: 1.5s speech at -20 dB RMS + 0.5s quiet room tone at -70 dB
    speech = (0.14 * np.sin(2 * np.pi * 220 * t[: int(sr * 1.5)])).astype(np.float32)
    quiet = (1e-4 * np.sin(2 * np.pi * 100 * t[: int(sr * 0.5)])).astype(np.float32)
    good = np.concatenate([speech, quiet])
    good_p = Path(_tmp_home) / "001-good.wav"
    sf.write(str(good_p), good, sr)

    bad = (0.99 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    bad_p = Path(_tmp_home) / "002-bad.wav"
    sf.write(str(bad_p), bad, sr)

    r_good = preflight.validate([good_p], target="acx")
    assert r_good["ok"], r_good
    r_bad = preflight.validate([bad_p], target="acx")
    assert not r_bad["ok"] and len(r_bad["chapters"][0]["issues"]) >= 2, r_bad
    print("preflight: OK")


def check_series_and_snapshots():
    assert "series_id" in projects.DEFAULTS
    p1 = projects.create({"title": "Book 1", "story": "Chapter 1\nHello world.", "cast": [["Hero", "Tara", "", 1.0]],
                          "pronunciations": [{"word": "Aethel", "say": "Ay-thel"}]})
    s = series.create("Chronicles")
    s = series.sync_from_book(s["id"], p1)
    assert len(s["cast"]) == 1 and len(s["pronunciations"]) == 1

    p2 = projects.create({"title": "Book 2", "story": "Chapter 1\nSequel.", "cast": [["Villain", "Leo", "", 1.0]]})
    merged_cast, merged_pron = series.apply_to_book(s["id"], p2)
    names = {c[0] for c in merged_cast}
    assert names == {"Hero", "Villain"}, names
    assert any(pr["word"] == "Aethel" for pr in merged_pron)

    snap1 = snapshots.take(p1["id"], "Initial")
    snap_dup = snapshots.take(p1["id"], "Duplicate")
    assert snap_dup["deduped"] is True

    projects.save(p1["id"], {"story": "Chapter 1\nChanged story text with more words."})
    d = snapshots.diff(p1["id"], snap1["id"])
    assert any("Script" in c for c in d["changes"]), d

    restored = snapshots.restore(p1["id"], snap1["id"])
    assert restored["story"] == "Chapter 1\nHello world."
    assert len(snapshots.list_for(p1["id"])) >= 2
    print("series_and_snapshots: OK")


def check_acoustics():
    sr = 24000
    t = np.arange(sr // 2, dtype=np.float32) / sr
    mono = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    stereo = np.stack([mono, mono], axis=1)
    for pid in acoustics.PRESETS:
        out_m = acoustics.apply_space(mono, pid, sr=sr)
        out_s = acoustics.apply_space(stereo, pid, sr=sr)
        assert out_m.shape == mono.shape and np.all(np.isfinite(out_m))
        assert out_s.shape == stereo.shape and np.all(np.isfinite(out_s))
    print(f"acoustics ({len(acoustics.PRESETS)} presets): OK")


def check_dataset_health():
    sr = 24000
    ds = Path(_tmp_home) / "ds_test" / "clips"
    ds.mkdir(parents=True, exist_ok=True)
    # 2 good clips (3s, clean, quiet pause)
    for i in range(2):
        sig = np.concatenate([
            (0.15 * np.sin(2 * np.pi * 250 * np.arange(int(sr * 2.4)) / sr)).astype(np.float32),
            np.zeros(int(sr * 0.6), dtype=np.float32),
        ])
        sf.write(str(ds / f"good_{i}.wav"), sig, sr)
    # 1 clipped clip
    sf.write(str(ds / "clipped.wav"), np.ones(sr * 2, dtype=np.float32), sr)
    # 1 too-short clip (0.4s)
    sf.write(str(ds / "short.wav"), np.zeros(int(sr * 0.4), dtype=np.float32), sr)

    r1 = health.score_dataset(ds.parent)
    assert r1["total_clips"] == 4 and r1["usable_clips"] == 2 and r1["clipped"] >= 1 and r1["bad_duration"] >= 1, r1
    r2 = health.auto_cure(ds.parent)
    assert r2["quarantined"] == 2 and r2["usable_clips"] == 2 and r2["total_clips"] == 2, r2
    assert (ds.parent / "quarantine" / "clipped.wav").is_file()
    print("dataset_health: OK")


def check_api_routes():
    from narrator import api
    paths = {r.path for r in api.router.routes}
    expected = {
        "/api/projects/{pid}/preflight",
        "/api/series",
        "/api/series/{sid}",
        "/api/series/{sid}/sync/{pid}",
        "/api/series/{sid}/apply/{pid}",
        "/api/projects/{pid}/snapshots",
        "/api/projects/{pid}/snapshots/{sid}/diff",
        "/api/projects/{pid}/snapshots/{sid}/restore",
        "/api/acoustics/presets",
        "/api/train/dataset/{voice}/health",
        "/api/train/dataset/{voice}/auto-cure",
    }
    missing = expected - paths
    assert not missing, f"Missing API routes: {missing}"
    print(f"api_routes ({len(expected)} routes): OK")


if __name__ == "__main__":
    check_export()
    check_audio_dsp()
    check_library_pickle_guard()
    check_preflight()
    check_series_and_snapshots()
    check_acoustics()
    check_dataset_health()
    check_api_routes()
    print("ALL OK")
