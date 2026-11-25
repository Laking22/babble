"""Two-stage Immersion QA ('Finish-up') pipeline for audiobooks:

Stage 1 (Local, Free, Offline):
  - Script & cue sanity: malformed/unclosed tags, unknown cast speakers, missing library sounds.
  - Acoustic continuity DSP: loudness jumps between adjacent paragraphs, sustained clipping, unnatural dead-air
    silences (>= 4 s where no {pause:} tag exists), and sharp splice clicks.
  - Optional local Whisper word-diff spot-check on DSP-flagged or pace-anomalous paragraphs.

Stage 2 (Optional Cloud Narrative & Immersion Reviewer):
  - Uses the user's already-configured Director AI service & model (Claude, Gemini, OpenAI, OpenRouter, etc.)
    with upfront cost estimation.
  - Checks narrative continuity: wrong speaker attribution, ambience/music contradicting the scene's location or
    mood, anachronistic SFX, or jarring transitions.

Every issue carries (chapter, para, start) so the UI can jump straight to that paragraph or regenerate it in place.
"""
import json
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from . import book, books, director, pricing, sfx
from .cast import parse, strip_tags
from .runtime import SR

QA_FILE = "immersion-qa.json"
LOUD_JUMP_DB = 7.5       # dB jump between adjacent paragraphs worth flagging
CLIP_PEAK = 0.992        # sustained clipping threshold
CLIP_SAMPLES = 12        # consecutive/total clipped samples in a paragraph
DEAD_AIR_SEC = 4.0       # uninterrupted digital silence (< -65 dB) without a {pause:} tag
DEAD_AIR_DB = -65.0
SPLICE_CLICK = 0.35      # instantaneous sample jump right at a paragraph boundary


def _rms_db(seg):
    if len(seg) == 0:
        return -120.0
    a = seg.mean(axis=1) if seg.ndim > 1 else seg
    return float(20 * np.log10(np.sqrt(np.mean(a.astype(np.float32) ** 2)) + 1e-12))


def _active_rms_db(seg, sr=SR):
    a = seg.mean(axis=1) if seg.ndim > 1 else seg
    hop = max(1, sr // 20)
    n = len(a) // hop
    if not n:
        return _rms_db(a)
    rms = np.sqrt((a[:n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms)
    active = rms[db > db.max() - 25.0]
    if not len(active):
        return _rms_db(a)
    return float(20 * np.log10(np.sqrt(np.mean(active ** 2)) + 1e-12))


def audit_script(story, cast_names=(), quotes_only=True, book_id=None):
    """Stage 1a: Fast deterministic script, cast, and sound-cue sanity check."""
    token = sfx.BOOK.set(book_id) if book_id else None
    issues = []
    try:
        known = {str(n).strip().lower() for n in cast_names if str(n).strip()} | {"narrator"}
        chapters = books.story_chapters(story or "", quotes_only)
        for ci, (title, body, paras) in enumerate(chapters):
            raw_paras = [p.strip() for p in re.split(r"\n\s*\n", body or "") if p.strip()]
            for pi, raw in enumerate(raw_paras):
                # 1. Unbalanced curly braces (could be spoken aloud as raw text!)
                if raw.count("{") != raw.count("}"):
                    issues.append({
                        "stage": 1, "severity": "high", "category": "tag_syntax",
                        "chapter": ci, "para": pi, "start": None, "can_regen": False,
                        "title": f"Unclosed sound tag in {title or f'Chapter {ci + 1}'} · ¶{pi + 1}",
                        "detail": "Mismatched '{' or '}' braces can cause the narrator to read raw tag text aloud.",
                    })
                # 2. Missing library sounds
                for kind, name in sfx.missing(raw):
                    issues.append({
                        "stage": 1, "severity": "medium", "category": "missing_sound",
                        "chapter": ci, "para": pi, "start": None, "can_regen": False,
                        "title": f"Missing {kind} “{name}” ({title or f'Chapter {ci + 1}'} · ¶{pi + 1})",
                        "detail": f"No sound in your library matches “{name}”, so this cue will be silent until created or added.",
                    })
            # 3. Unknown speaker tags not in cast
            for pi, para in enumerate(paras):
                for who, _, _ in para:
                    if who.strip().lower() not in known:
                        issues.append({
                            "stage": 1, "severity": "high", "category": "unknown_speaker",
                            "chapter": ci, "para": pi, "start": None, "can_regen": False,
                            "title": f"Uncast speaker [{who}] in {title or f'Chapter {ci + 1}'} · ¶{pi + 1}",
                            "detail": f"“{who}” isn't in your Characters cast yet.",
                        })
        return issues
    finally:
        if token is not None:
            try:
                sfx.BOOK.reset(token)
            except ValueError:
                pass


def audit_chapter_audio(wav_path, timings, chapter_index, chapter_title="", raw_paras=None):
    """Stage 1b: Pure-DSP acoustic continuity check on a rendered chapter WAV."""
    p = Path(wav_path)
    if not p.is_file():
        return []
    data, sr = sf.read(str(p), dtype="float32", always_2d=False)
    mono = data.mean(axis=1) if data.ndim > 1 else data
    issues = []
    label = chapter_title or f"Chapter {chapter_index + 1}"
    raw_paras = raw_paras or []

    # Check paragraph-by-paragraph loudness jumps, clipping, and splice clicks
    prev_db = None
    prev_para = None
    for row in timings:
        pi = int(row["para"])
        s0 = max(0, int(float(row["start"]) * sr))
        s1 = min(len(mono), int(float(row["end"]) * sr))
        if s1 - s0 < sr // 10:
            continue
        seg = data[s0:s1]
        seg_mono = mono[s0:s1]

        # 1. Sustained clipping check
        clipped = int(np.sum(np.abs(seg) >= CLIP_PEAK))
        if clipped >= CLIP_SAMPLES:
            issues.append({
                "stage": 1, "severity": "high", "category": "clipping",
                "chapter": chapter_index, "para": pi, "start": float(row["start"]), "can_regen": True,
                "title": f"Audio clipping in {label} · ¶{pi + 1}",
                "detail": f"{clipped} samples hit full scale (≥0.992), which can sound harsh or distorted.",
            })

        # 2. Loudness jump between adjacent paragraphs
        db = _active_rms_db(seg_mono, sr)
        raw_t = raw_paras[pi] if 0 <= pi < len(raw_paras) else ""
        is_whisper = bool(re.search(r"\b(whisper|hush|murmur|softly|quietly)\b", raw_t, re.I))
        if prev_db is not None and db > -50.0 and prev_db > -50.0:
            jump = abs(db - prev_db)
            thresh = LOUD_JUMP_DB + 3.0 if is_whisper else LOUD_JUMP_DB
            if jump >= thresh:
                issues.append({
                    "stage": 1, "severity": "medium", "category": "loudness_jump",
                    "chapter": chapter_index, "para": pi, "start": float(row["start"]), "can_regen": True,
                    "title": f"Volume jump ({jump:.1f} dB) at {label} · ¶{pi + 1}",
                    "detail": f"Active level shifted from {prev_db:.1f} dB (¶{prev_para + 1}) to {db:.1f} dB (¶{pi + 1}).",
                })
        prev_db, prev_para = db, pi

        # 3. Sharp splice click right at paragraph start boundary
        if 2 <= s0 < len(mono) - 2:
            edge_jump = float(np.max(np.abs(np.diff(mono[s0 - 2:s0 + 3]))))
            if edge_jump >= SPLICE_CLICK:
                issues.append({
                    "stage": 1, "severity": "medium", "category": "splice_click",
                    "chapter": chapter_index, "para": pi, "start": float(row["start"]), "can_regen": True,
                    "title": f"Possible splice click at {label} · ¶{pi + 1}",
                    "detail": f"Abrupt waveform jump ({edge_jump:.2f}) at the transition into paragraph {pi + 1}.",
                })

    # 4. Unnatural dead-air silence scan (100 ms windows)
    hop = max(1, sr // 10)
    frames = len(mono) // hop
    if frames > 0:
        rms = np.sqrt((mono[:frames * hop].reshape(frames, hop) ** 2).mean(1) + 1e-12)
        silent = (20 * np.log10(rms)) < DEAD_AIR_DB
        run_start = None
        for fi, is_s in enumerate(silent):
            if is_s and run_start is None:
                run_start = fi
            elif not is_s and run_start is not None:
                dur = (fi - run_start) * 0.1
                if dur >= DEAD_AIR_SEC and run_start > 0:  # ignore leading intro silence
                    t_at = round(run_start * 0.1, 2)
                    near = min(timings, key=lambda r: abs(float(r["start"]) - t_at), default=None)
                    pi = int(near["para"]) if near else 0
                    raw_t = raw_paras[pi] if 0 <= pi < len(raw_paras) else ""
                    if not re.search(r"\{\s*pause\b", raw_t, re.I):
                        issues.append({
                            "stage": 1, "severity": "high", "category": "dead_air",
                            "chapter": chapter_index, "para": pi, "start": t_at, "can_regen": True,
                            "title": f"{dur:.1f}s dead-air gap in {label} · ¶{pi + 1}",
                            "detail": f"Digital silence (<{DEAD_AIR_DB:.0f} dB) lasted {dur:.1f}s at {t_at:.1f}s with no {{pause:}} tag.",
                        })
                run_start = None

    # 5. Voice-drift / timbre outlier check per speaker across the chapter (pure DSP, zero model RAM)
    issues += _audit_voice_drift(mono, sr, timings, chapter_index, label)
    return issues


def _voice_signature(seg, sr):
    """Lightweight (spectral_centroid_hz, median_f0_hz) fingerprint over active 50 ms frames."""
    hop = max(1, sr // 20)
    n = len(seg) // hop
    if n < 6:
        return None
    frames = seg[:n * hop].reshape(n, hop)
    rms = np.sqrt((frames ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms)
    active = frames[db > max(db.max() - 22.0, -45.0)]
    if len(active) < 4:
        return None
    # Spectral centroid (Hz)
    win = np.hanning(hop).astype(np.float32)
    spec = np.abs(np.fft.rfft(active * win[None, :], axis=1)) + 1e-9
    freqs = np.fft.rfftfreq(hop, 1.0 / sr)
    centroid = float(np.median((spec * freqs[None, :]).sum(1) / spec.sum(1)))
    # Autocorrelation pitch proxy in 75–380 Hz
    lo, hi = max(1, int(sr / 380)), min(hop - 1, int(sr / 75))
    f0s = []
    for f in active[:40]:
        f = f - f.mean()
        if float(np.dot(f, f)) < 1e-7:
            continue
        ac = np.correlate(f, f, mode="full")[hop - 1:]
        if hi <= lo or hi >= len(ac):
            continue
        k = int(np.argmax(ac[lo:hi])) + lo
        if ac[k] > 0.35 * ac[0]:
            f0s.append(sr / k)
    f0 = float(np.median(f0s)) if len(f0s) >= 3 else None
    return centroid, f0


def _audit_voice_drift(mono, sr, timings, chapter_index, label):
    """Flag paragraphs where a character's voice timbre or pitch drifts sharply from their chapter median."""
    by_who = {}
    for row in timings:
        who = (row.get("who") or "").strip()
        if not who:
            continue
        s0 = max(0, int(float(row["start"]) * sr))
        s1 = min(len(mono), int(float(row["end"]) * sr))
        if s1 - s0 < int(0.4 * sr):
            continue
        sig = _voice_signature(mono[s0:s1], sr)
        if sig is not None:
            by_who.setdefault(who, []).append((int(row["para"]), float(row["start"]), sig[0], sig[1]))
    issues = []
    for who, rows in by_who.items():
        if len(rows) < 5:
            continue
        cents = np.array([r[2] for r in rows], dtype=np.float32)
        c_med = float(np.median(cents))
        c_mad = max(float(np.median(np.abs(cents - c_med))) * 1.4826, 40.0)
        f0_vals = np.array([r[3] for r in rows if r[3] is not None], dtype=np.float32)
        f0_med = float(np.median(f0_vals)) if len(f0_vals) >= 5 else None
        f0_st_mad = (max(float(np.median(np.abs(12.0 * np.log2(f0_vals / f0_med)))) * 1.4826, 0.8)
                     if f0_med is not None else None)
        for pi, start, c, f0 in rows:
            c_z = abs(c - c_med) / c_mad
            f0_st = abs(12.0 * np.log2(f0 / f0_med)) if (f0 is not None and f0_med is not None) else 0.0
            f0_z = (f0_st / f0_st_mad) if f0_st_mad else 0.0
            if (c_z >= 3.2 and abs(c - c_med) >= 280.0) or (f0_z >= 3.2 and f0_st >= 4.0):
                why = (f"pitch shifted {f0_st:.1f} semitones from {who}'s usual tone"
                       if f0_z >= c_z and f0_st >= 4.0
                       else f"timbre shifted sharply ({c:.0f} Hz vs {who}'s {c_med:.0f} Hz median)")
                issues.append({
                    "stage": 1, "severity": "medium", "category": "voice_drift",
                    "chapter": chapter_index, "para": pi, "start": start, "can_regen": True,
                    "title": f"Voice drift for {who} in {label} · ¶{pi + 1}",
                    "detail": f"{who}'s delivery {why}. Regenerating this paragraph can restore voice consistency.",
                })
    return issues


STAGE2_SYSTEM = """You are an audiobook immersion QA director. Review the chapter's tagged script and cast list for
ONLY clear, high-confidence immersion breaks:
1. speaker_mismatch: dialogue clearly attributed to the wrong character (or Narrator) based on context/tags.
2. ambience_mismatch: an {amb: ...} or {music: ...} cue that contradicts the scene's current setting or stays on after a scene cut.
3. anachronism: a sound effect that doesn't belong in the story's era/world.
4. mood_clash: music or sound design that clashes sharply with the emotional tone of the paragraph.

Be strict and concise: ignore minor stylistic preferences and return an empty list if nothing breaks immersion."""

STAGE2_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["issues"],
    "properties": {
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["para", "severity", "category", "title", "detail"],
                "properties": {
                    "para": {"type": "integer"},
                    "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                    "category": {"type": "string", "enum": ["speaker_mismatch", "ambience_mismatch", "anachronism", "mood_clash"]},
                    "title": {"type": "string"},
                    "detail": {"type": "string"},
                },
            },
        }
    },
}


def estimate_stage2(story, where, model):
    """Upfront cost estimate for Stage 2 Cloud Immersion QA using the same pricing engine as Director."""
    if not (story or "").strip() or where == director.MAC:
        return {"chapters": 0, "cost": "free", "free_tier": True, "low": 0.0, "high": 0.0}
    chapters = books.story_chapters(story, True)
    chars = sum(len(body or "") for _, body, _ in chapters)
    tokens = {
        "in": chars / 3.8 + len(chapters) * 600,
        "in_repeat": max(0, len(chapters) - 1) * 450,
        "out_low": len(chapters) * 200,
        "out_high": len(chapters) * 600,
    }
    r = pricing.ai_range(where, model, tokens, "qa", 1)
    return {"chapters": len(chapters), "cost": pricing.say(r), "free_tier": r["free_tier"],
            "low": r["low"], "high": r["high"], "basis": r["basis"]}


def audit_narrative_cloud(story, cast, where, model, stage1_issues=None, quotes_only=True):
    """Stage 2: Cloud LLM narrative & cue continuity review across chapters (reuses Director providers)."""
    from . import spend, tasks
    if where == director.MAC:
        raise ValueError("Pick a connected AI service (Claude, Gemini, OpenAI, etc.) for Stage 2 cloud review.")
    if where not in director.PROVIDERS or not director.key(where):
        raise ValueError("Connect an AI service in Connections first.")
    if where == "anthropic" and not model:
        model = director.CLAUDE[0]["id"]
    if not model:
        raise ValueError("Pick a model for Stage 2 review.")

    est = estimate_stage2(story, where, model)
    if est.get("high"):
        spend.allow(where, {"low": est["low"], "high": est["high"], "model": model}, "qa")

    chapters = books.story_chapters(story or "", quotes_only)
    cast_summary = ", ".join(f"{r[0]} ({r[1]})" for r in books.rows_of(cast)) or "Narrator"
    out = []
    for ci, (title, body, _) in enumerate(chapters):
        if tasks.stop_wanted():
            raise tasks.Stopped()
        raw_paras = [p.strip() for p in re.split(r"\n\s*\n", body or "") if p.strip()]
        if not raw_paras:
            continue
        # Chunk long chapters to ~20k chars so every model handles them comfortably
        chunk, chunk_len, start_pi = [], 0, 0
        groups = []
        for pi, p in enumerate(raw_paras):
            if chunk and chunk_len + len(p) > 20000:
                groups.append((start_pi, chunk))
                chunk, chunk_len, start_pi = [], 0, pi
            chunk.append((pi, p))
            chunk_len += len(p)
        if chunk:
            groups.append((start_pi, chunk))

        for _, grp in groups:
            if tasks.stop_wanted():
                raise tasks.Stopped()
            numbered = "\n\n".join(f"[{pi + 1}] {p}" for pi, p in grp)
            user = f"Chapter {ci + 1}: {title or 'Story'}\nCast: {cast_summary}\n\n{numbered}"
            try:
                if where == "anthropic":
                    reply, _ = director._ask_claude(model, STAGE2_SYSTEM, user, STAGE2_SCHEMA, 4000)
                else:
                    reply, _ = director._ask_openai(where, model, STAGE2_SYSTEM, user, STAGE2_SCHEMA)
                parsed = director._json(reply) or {}
            except tasks.Stopped:
                raise
            except spend.Overspent:
                raise
            except Exception as e:  # noqa: BLE001 - keep already-paid chapters instead of discarding all work
                spend.mark_partial()
                out.append({
                    "stage": 2,
                    "severity": "low",
                    "category": "mood_clash",
                    "chapter": ci,
                    "para": grp[0][0],
                    "start": None,
                    "can_regen": False,
                    "title": f"Stage 2 skipped part of {title or f'Chapter {ci + 1}'}",
                    "detail": f"Cloud reviewer reply failed ({str(e)[:180]}); earlier chapters were kept.",
                })
                continue
            valid_pis = {pi for pi, _ in grp}
            for item in parsed.get("issues") or []:
                pi = int(item.get("para", 1)) - 1
                if pi not in valid_pis:
                    pi = grp[0][0]
                out.append({
                    "stage": 2,
                    "severity": item.get("severity") if item.get("severity") in ("high", "medium", "low") else "medium",
                    "category": item.get("category") or "mood_clash",
                    "chapter": ci,
                    "para": pi,
                    "start": None,
                    "can_regen": False,
                    "title": str(item.get("title") or "Immersion issue")[:120],
                    "detail": str(item.get("detail") or "")[:300],
                })
    return out


def read(folder):
    try:
        return json.loads((Path(folder) / QA_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save(folder, data):
    Path(folder).mkdir(parents=True, exist_ok=True)
    (Path(folder) / QA_FILE).write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


def run_qa(story, table, folder=None, title="", quotes_only=True, book=None,
           stage2=False, where="", model="", **_):
    """Generator task running Stage 1 (local script + acoustic DSP) and optionally Stage 2 (cloud narrative QA)."""
    where_dir = books.book_folder(title, story, folder)
    cast_names = [r[0] for r in books.rows_of(table)]
    yield {"status": "Stage 1 · Checking script tags, cast, and sound cues…", "progress": 0.1}
    issues = audit_script(story, cast_names, quotes_only, book_id=book)

    ch_status = books.chapters_status(story, folder, title, quotes_only)
    st_ch = books.story_chapters(story or "", quotes_only)
    rendered = [c for c in ch_status if c.get("file") and Path(c["file"]).is_file()]
    for k, c in enumerate(rendered):
        ci = c["index"]
        yield {
            "status": f"Stage 1 · Auditing audio continuity ({k + 1} of {len(rendered)}: {c['title']})…",
            "progress": 0.15 + 0.45 * (k / max(len(rendered), 1)),
        }
        tm = books.chapter_timings(story, ci, c["file"], quotes_only)["timings"]
        raw_paras = [p.strip() for p in re.split(r"\n\s*\n", st_ch[ci][1] or "") if p.strip()] if ci < len(st_ch) else []
        issues += audit_chapter_audio(c["file"], tm, ci, c["title"], raw_paras)

    if stage2:
        yield {"status": "Stage 2 · Running cloud story & immersion continuity review…", "progress": 0.65}
        s2 = audit_narrative_cloud(story, table, where, model, issues, quotes_only)
        # Attach paragraph start timestamp if chapter is rendered
        timings_cache = {}
        for item in s2:
            ci = item["chapter"]
            ch_info = next((c for c in rendered if c["index"] == ci), None)
            if ch_info:
                if ci not in timings_cache:
                    timings_cache[ci] = {t["para"]: t["start"] for t in books.chapter_timings(story, ci, ch_info["file"], quotes_only)["timings"]}
                item["start"] = timings_cache[ci].get(item["para"])
        issues += s2

    sev_rank = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda x: (sev_rank.get(x["severity"], 1), x["chapter"], x["para"]))
    for idx, it in enumerate(issues):
        it["id"] = idx

    report = {
        "checked": time.time(),
        "stage2": bool(stage2),
        "rendered_chapters": len(rendered),
        "total_chapters": len(ch_status),
        "issues": issues,
    }
    _save(where_dir, report)
    n = len(issues)
    yield {
        "qa": report,
        "progress": 1.0,
        "status": f"Done · found {n} immersion issue{'' if n == 1 else 's'}." if n else "Done · no immersion issues found.",
    }
