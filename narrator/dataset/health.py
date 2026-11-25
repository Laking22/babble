"""Automated Dataset Health Scorecard & One-Click Auto-Cure for Train Voice datasets.

Audits prepared clips (WAV/FLAC + metadata.jsonl) before burning local MLX or cloud GPU hours:
- Usable minutes & clip count
- Clipping detection (peak >= -0.3 dBFS)
- ACX noise floor estimate (5th-percentile 100 ms window RMS vs -60 dB)
- Duration bounds (< 1.5 s or > 15.5 s outliers)
- Overall grade (A-F) + one-click auto_cure() to quarantine bad clips cleanly
"""
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf


def _clip_metrics(path):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    dur = len(mono) / max(sr, 1)
    if len(mono) == 0:
        return {"dur": 0.0, "peak_db": -120.0, "rms_db": -120.0, "floor_db": -120.0, "clipped": False}
    peak = float(np.max(np.abs(mono)))
    peak_db = 20.0 * np.log10(peak + 1e-12)
    rms_db = 20.0 * np.log10(float(np.sqrt(np.mean(mono ** 2))) + 1e-12)
    win = max(1, int(sr * 0.1))
    steps = len(mono) // win
    if steps >= 4:
        w_db = 10.0 * np.log10(np.mean(mono[:steps * win].reshape(steps, win) ** 2, axis=1) + 1e-12)
        floor_db = float(np.percentile(w_db, 5))
    else:
        floor_db = rms_db
    return {
        "dur": round(dur, 2),
        "peak_db": round(peak_db, 2),
        "rms_db": round(rms_db, 2),
        "floor_db": round(floor_db, 2),
        "clipped": peak_db >= -0.3,
    }


def score_dataset(dataset_dir, limit=200):
    """Score all clips in `dataset_dir` (folder with WAV/FLAC clips or clips/ subdir) and return a health report."""
    root = Path(dataset_dir)
    if not root.exists():
        raise ValueError("Dataset folder not found.")
    clips_dir = root / "clips" if (root / "clips").is_dir() else root
    files = sorted(p for p in clips_dir.iterdir() if p.suffix.lower() in (".wav", ".flac"))
    if not files:
        return {"grade": "F", "total_clips": 0, "usable_clips": 0, "usable_minutes": 0.0,
                "total_minutes": 0.0, "clipped": 0, "noisy": 0, "bad_duration": 0,
                "outliers": [], "summary": "No audio clips found."}

    total_s = 0.0
    usable_s = 0.0
    clipped = 0
    noisy = 0
    bad_dur = 0
    outliers = []
    for f in files:
        try:
            m = _clip_metrics(f)
        except Exception:
            outliers.append({"file": f.name, "reasons": ["unreadable"]})
            continue
        total_s += m["dur"]
        reasons = []
        if m["dur"] < 1.5:
            reasons.append(f"too short ({m['dur']}s)")
            bad_dur += 1
        elif m["dur"] > 15.5:
            reasons.append(f"too long ({m['dur']}s)")
            bad_dur += 1
        if m["clipped"]:
            reasons.append(f"clipped ({m['peak_db']} dB)")
            clipped += 1
        if m["floor_db"] > -52.0:
            reasons.append(f"high noise floor ({m['floor_db']} dB)")
            noisy += 1
        if reasons:
            outliers.append({"file": f.name, "reasons": reasons, **m})
        else:
            usable_s += m["dur"]

    usable_min = round(usable_s / 60.0, 2)
    bad_ratio = len(outliers) / max(len(files), 1)
    if usable_min >= 20 and bad_ratio <= 0.05:
        grade = "A"
    elif usable_min >= 10 and bad_ratio <= 0.12:
        grade = "B"
    elif usable_min >= 5 and bad_ratio <= 0.25:
        grade = "C"
    elif usable_min >= 2:
        grade = "D"
    else:
        grade = "F"
    return {
        "grade": grade,
        "total_clips": len(files),
        "usable_clips": len(files) - len(outliers),
        "usable_minutes": usable_min,
        "total_minutes": round(total_s / 60.0, 2),
        "clipped": clipped,
        "noisy": noisy,
        "bad_duration": bad_dur,
        "outliers": outliers if limit is None else outliers[:limit],
    }


def auto_cure(dataset_dir):
    """Move outlier clips into a reversible `quarantine/` subfolder (never deletes permanently) and return updated score."""
    root = Path(dataset_dir)
    clips_dir = root / "clips" if (root / "clips").is_dir() else root
    report = score_dataset(root, limit=None)
    qdir = root / "quarantine"
    moved = 0
    bad_names = {o["file"] for o in report["outliers"]}
    if bad_names:
        qdir.mkdir(parents=True, exist_ok=True)
        for name in bad_names:
            src = clips_dir / name
            if src.is_file():
                shutil.move(str(src), str(qdir / name))
                moved += 1
    after = score_dataset(root)
    after["quarantined"] = moved
    return after
