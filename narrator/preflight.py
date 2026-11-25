"""ACX / Voices by INaudio / Apple Books pre-flight mastering validator for built audiobook chapters.

Checks each chapter WAV against distributor technical requirements:
- Integrated RMS loudness (-23 to -18 dB for ACX / INaudio)
- Peak level (<= -3.0 dB)
- Quietest-window noise floor (<= -60 dB RMS)
- Maximum chapter duration (<= 120 min for ACX)
- Consistent sample rate and channel count across chapters
"""
from pathlib import Path

import numpy as np
import soundfile as sf

SPECS = {
    "acx": {
        "name": "Standard Audiobook Spec",
        "rms_min": -23.0, "rms_max": -18.0,
        "peak_max": -3.0, "floor_max": -60.0,
        "max_minutes": 120.0,
        "policy_note": "Audible doesn't take AI-narrated books.",
    },
    "inaudio": {
        "name": "Voices by INaudio",
        "rms_min": -23.0, "rms_max": -18.0,
        "peak_max": -3.0, "floor_max": -60.0,
        "max_minutes": 120.0,
    },
    "apple": {
        "name": "Apple Books",
        "rms_min": -24.0, "rms_max": -16.0,
        "peak_max": -1.0, "floor_max": -58.0,
        "max_minutes": 1380.0,
    },
}


def measure_file(path):
    """Measure RMS dB, peak dB, noise floor dB, duration, sample rate, and channels for one audio file."""
    p = Path(path)
    info = sf.info(str(p))
    data, sr = sf.read(str(p), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    n = len(mono)
    if n == 0:
        return {"file": p.name, "duration_s": 0.0, "sr": sr, "channels": info.channels,
                "rms_db": -120.0, "peak_db": -120.0, "floor_db": -120.0}
    peak = float(np.max(np.abs(data)))
    peak_db = round(20.0 * np.log10(peak + 1e-12), 2)
    rms = float(np.sqrt(np.mean(mono ** 2)))
    rms_db = round(20.0 * np.log10(rms + 1e-12), 2)
    # 200 ms windows: 5th-percentile non-zero window approximates background noise floor
    win = max(1, int(sr * 0.2))
    steps = n // win
    if steps >= 5:
        w = mono[:steps * win].reshape(steps, win)
        w_db = 10.0 * np.log10(np.mean(w ** 2, axis=1) + 1e-12)
        floor_db = round(float(np.percentile(w_db, 5)), 2)
    else:
        floor_db = rms_db
    return {
        "file": p.name,
        "duration_s": round(info.duration, 2),
        "sr": int(sr),
        "channels": int(info.channels),
        "rms_db": rms_db,
        "peak_db": peak_db,
        "floor_db": floor_db,
    }


def validate(chapter_files, target="acx"):
    """Validate a list of chapter audio paths against a distributor spec ('acx', 'inaudio', or 'apple')."""
    spec = SPECS.get(target, SPECS["acx"])
    rows = []
    channels_seen = set()
    srs_seen = set()
    passed = 0
    for f in chapter_files:
        m = measure_file(f)
        channels_seen.add(m["channels"])
        srs_seen.add(m["sr"])
        issues = []
        if m["rms_db"] < spec["rms_min"] or m["rms_db"] > spec["rms_max"]:
            issues.append(f"RMS {m['rms_db']:.1f} dB (target {spec['rms_min']:.0f} to {spec['rms_max']:.0f} dB)")
        if m["peak_db"] > spec["peak_max"]:
            issues.append(f"Peak {m['peak_db']:.1f} dB exceeds {spec['peak_max']:.1f} dB")
        if m["floor_db"] > spec["floor_max"]:
            issues.append(f"Noise floor {m['floor_db']:.1f} dB exceeds {spec['floor_max']:.0f} dB")
        mins = m["duration_s"] / 60.0
        if mins > spec["max_minutes"]:
            issues.append(f"Duration {mins:.1f} min exceeds {spec['max_minutes']:.0f} min limit")
        ok = not issues
        if ok:
            passed += 1
        rows.append({**m, "ok": ok, "issues": issues})
    book_issues = []
    if len(channels_seen) > 1:
        book_issues.append("Mixed mono and stereo chapters (export will upmix automatically).")
    if len(srs_seen) > 1:
        book_issues.append(f"Mixed sample rates across chapters: {sorted(srs_seen)} Hz.")
    return {
        "target": target,
        "spec": spec["name"],
        "policy_note": spec.get("policy_note", ""),
        "ok": passed == len(rows) and not book_issues,
        "passed": passed,
        "total": len(rows),
        "book_issues": book_issues,
        "chapters": rows,
    }
