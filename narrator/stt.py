"""Speech-to-text transcription using Apple Silicon MLX models (Whisper Large v3 Turbo & Qwen3-ASR 1.7B).

Runs on Babble's shared MLX thread and model slot so it never collides with TTS models in memory,
records natively from the Mac's microphone via sounddevice (no WKWebView getUserMedia limitation),
converts any audio or video input to 16 kHz mono WAV via ffmpeg when needed, exports TXT/SRT files
for native Finder saving, and keeps recent transcripts under outputs/transcripts/.
"""
from __future__ import annotations

from collections import Counter
import json
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from . import engine as en
from . import tasks
from .paths import OUTPUTS
from .runtime import on_mlx_thread, slot, streamed_on_mlx_thread

TRANSCRIPTS = OUTPUTS / "transcripts"
EXPORTS = TRANSCRIPTS / "exports"
UPLOADS = OUTPUTS / "uploads"
SAMPLE_RATE = 16000
MAX_REC_SECONDS = 7200  # 2-hour cap so a forgotten mic recording can't exhaust RAM

_active_lock = threading.Lock()
_active_audio: Counter[Path] = Counter()
_active_models: Counter[str] = Counter()
_deleting_models: set[str] = set()

MODELS = [
    {
        "id": "whisper-large-v3-turbo",
        "name": "Whisper Large v3 Turbo",
        "repo": en.STT_MODEL,  # mlx-community/whisper-large-v3-turbo-asr-8bit
        "family": "whisper",
        "size_gb": 0.9,
        "license": "MIT",
        "languages": "99+ languages",
        "summary": "Fast, reliable multilingual transcription with segment timestamps. Already built into Babble.",
        "timestamps": True,
        "core": True,  # also used internally by Babble's voice/alignment tools
    },
    {
        "id": "qwen3-asr-1.7b",
        "name": "Qwen3-ASR 1.7B",
        "repo": "mlx-community/Qwen3-ASR-1.7B-8bit",
        "family": "qwen3_asr",
        "size_gb": 2.0,
        "license": "Apache 2.0",
        "languages": "52 languages & dialects",
        "summary": "Alibaba's 1.7B speech model on Apple Silicon. Strong accuracy on accents and noisy audio (plain text only).",
        "timestamps": False,
        "core": False,
    },
]
MODELS_BY_ID = {m["id"]: m for m in MODELS}

# Map UI language choices to what each model expects (Whisper: ISO code; Qwen3-ASR: full English name)
LANGUAGES = [
    {"id": "auto", "label": "Auto-detect", "whisper": None, "qwen": None},
    {"id": "en", "label": "English", "whisper": "en", "qwen": "English"},
    {"id": "es", "label": "Spanish", "whisper": "es", "qwen": "Spanish"},
    {"id": "fr", "label": "French", "whisper": "fr", "qwen": "French"},
    {"id": "de", "label": "German", "whisper": "de", "qwen": "German"},
    {"id": "it", "label": "Italian", "whisper": "it", "qwen": "Italian"},
    {"id": "pt", "label": "Portuguese", "whisper": "pt", "qwen": "Portuguese"},
    {"id": "zh", "label": "Chinese", "whisper": "zh", "qwen": "Chinese"},
    {"id": "ja", "label": "Japanese", "whisper": "ja", "qwen": "Japanese"},
    {"id": "ko", "label": "Korean", "whisper": "ko", "qwen": "Korean"},
    {"id": "ru", "label": "Russian", "whisper": "ru", "qwen": "Russian"},
    {"id": "ar", "label": "Arabic", "whisper": "ar", "qwen": "Arabic"},
    {"id": "hi", "label": "Hindi", "whisper": "hi", "qwen": "Hindi"},
    {"id": "nl", "label": "Dutch", "whisper": "nl", "qwen": "Dutch"},
    {"id": "pl", "label": "Polish", "whisper": "pl", "qwen": "Polish"},
    {"id": "tr", "label": "Turkish", "whisper": "tr", "qwen": "Turkish"},
    {"id": "sv", "label": "Swedish", "whisper": "sv", "qwen": "Swedish"},
    {"id": "vi", "label": "Vietnamese", "whisper": "vi", "qwen": "Vietnamese"},
    {"id": "uk", "label": "Ukrainian", "whisper": "uk", "qwen": "Ukrainian"},
]
LANGS_BY_ID = {l["id"]: l for l in LANGUAGES}

_TID_RE = re.compile(r"^[A-Za-z0-9_-]{4,64}$")


def protected_uploads() -> set[Path]:
    """Return uploaded audio paths currently being transcribed or referenced by saved transcripts."""
    with _active_lock:
        kept = {p for p, n in _active_audio.items() if n > 0}
    if TRANSCRIPTS.is_dir():
        for f in TRANSCRIPTS.glob("*.json"):
            try:
                ap = json.loads(f.read_text(encoding="utf-8")).get("audio_path")
                if ap:
                    kept.add(Path(ap).resolve())
            except Exception:  # noqa: BLE001
                pass
    return kept


# ---------- native macOS microphone recorder (sounddevice) ----------

_rec_lock = threading.Lock()
_rec_state: dict = {
    "stream": None,
    "session": None,  # {"chunks": list, "samples": int, "sr": int, "active": Event, "lock": Lock, "level": float, "capped": bool}
    "starting": False,
    "started": 0.0,
    "error": None,
}


def list_mics() -> list[dict]:
    """List available macOS microphone input devices via sounddevice."""
    try:
        import sounddevice as sd
        devs = sd.query_devices()
        try:
            default_in = sd.default.device[0]
        except (TypeError, IndexError):
            default_in = sd.default.device
        out = []
        for idx, d in enumerate(devs):
            if d.get("max_input_channels", 0) > 0:
                out.append({
                    "id": idx,
                    "name": str(d.get("name") or f"Input {idx}"),
                    "default": idx == default_in,
                })
        out.sort(key=lambda x: (not x["default"], x["name"].lower()))
        return out
    except Exception:  # noqa: BLE001
        return []


def record_status() -> dict:
    with _rec_lock:
        stream = _rec_state["stream"]
        session = _rec_state["session"]
        started = _rec_state["started"]
        err = _rec_state["error"]

    if stream is None or session is None:
        return {"recording": False, "seconds": 0.0, "level": 0.0, "error": err}

    try:
        alive = bool(stream.active)
    except Exception:  # noqa: BLE001
        alive = False

    if not alive:
        disconnected = False
        cur_err = None
        with _rec_lock:
            if _rec_state["stream"] is stream:
                _rec_state["stream"] = None
                _rec_state["session"] = None
                _rec_state["started"] = 0.0
                _rec_state["error"] = "Microphone disconnected while recording."
                session["active"].clear()
                disconnected = True
            else:
                cur_err = _rec_state["error"]
        if disconnected:
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass
            return {"recording": False, "seconds": 0.0, "level": 0.0, "error": "Microphone disconnected while recording."}
        return {"recording": False, "seconds": 0.0, "level": 0.0, "error": cur_err}

    secs = round(max(0.0, time.time() - started), 1)
    lvl = round(float(session["level"]), 3)
    cap_err = "Reached 2-hour recording limit. Click Stop to save." if session.get("capped") else err
    return {"recording": True, "seconds": secs, "level": lvl, "error": cap_err}


def record_start(device: int | None = None) -> dict:
    """Start capturing mono audio from the Mac's microphone (16 kHz, falling back to device default rate)."""
    try:
        import sounddevice as sd
    except ImportError as e:
        raise ValueError("Microphone recording requires sounddevice, which isn't installed.") from e

    with _rec_lock:
        if _rec_state["stream"] is not None:
            session = _rec_state["session"]
            return {
                "recording": True,
                "seconds": round(max(0.0, time.time() - _rec_state["started"]), 1),
                "level": round(float(session["level"]) if session else 0.0, 3),
                "error": None,
            }
        if _rec_state["starting"]:
            raise ValueError("Microphone is already starting.")
        _rec_state["starting"] = True
        _rec_state["error"] = None

    try:
        rates = [SAMPLE_RATE]
        try:
            info = sd.query_devices(device, "input")
            dev_sr = int(round(float(info.get("default_samplerate") or 0)))
            if dev_sr > 0 and dev_sr not in rates:
                rates.append(dev_sr)
        except Exception:  # noqa: BLE001
            pass

        # Open and start InputStream outside _rec_lock so PortAudio's callback thread can never deadlock with us
        session = {
            "chunks": [],
            "samples": 0,
            "sr": SAMPLE_RATE,
            "level": 0.0,
            "capped": False,
            "active": threading.Event(),
            "lock": threading.Lock(),
        }
        session["active"].set()
        max_samples = SAMPLE_RATE * MAX_REC_SECONDS

        def _callback(indata, frames, time_info, status):  # noqa: ARG001
            if not session["active"].is_set():
                return
            arr = np.asarray(indata[:, 0], dtype=np.float32).copy()
            rms = float(np.sqrt(np.mean(arr * arr))) if arr.size else 0.0
            lvl = min(1.0, rms * 8.0)
            with session["lock"]:
                if not session["active"].is_set():
                    return
                if session["samples"] < max_samples:
                    session["chunks"].append(arr)
                    session["samples"] += arr.size
                else:
                    session["capped"] = True
                session["level"] = lvl

        stream = None
        last_err = None
        for sr in rates:
            session["sr"] = sr
            max_samples = sr * MAX_REC_SECONDS
            try:
                stream = sd.InputStream(
                    samplerate=sr,
                    channels=1,
                    dtype="float32",
                    device=device,
                    callback=_callback,
                )
                stream.start()
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:  # noqa: BLE001
                        pass
                    stream = None

        if stream is None:
            session["active"].clear()
            msg = str(last_err).strip() or (type(last_err).__name__ if last_err else "unknown error")
            raise ValueError(
                f"Couldn't open the microphone ({msg}). Check System Settings → Privacy & Security → Microphone and allow Babble."
            ) from last_err

        with _rec_lock:
            _rec_state["stream"] = stream
            _rec_state["session"] = session
            _rec_state["started"] = time.time()
            _rec_state["starting"] = False
        return {"recording": True, "seconds": 0.0, "level": 0.0, "error": None}
    except Exception:
        with _rec_lock:
            _rec_state["starting"] = False
        raise


def record_stop() -> dict:
    """Stop recording and save a mono WAV under outputs/uploads/."""
    with _rec_lock:
        stream = _rec_state["stream"]
        session = _rec_state["session"]
        _rec_state["stream"] = None
        _rec_state["session"] = None
        _rec_state["started"] = 0.0
        _rec_state["error"] = None
        if session is not None:
            session["active"].clear()

    if stream is None or session is None:
        raise ValueError("No microphone recording is in progress.")
    try:
        stream.stop()
        stream.close()
    except Exception:  # noqa: BLE001
        pass

    with session["lock"]:
        chunks = list(session["chunks"])
        sr = int(session["sr"] or SAMPLE_RATE)

    if not chunks:
        raise ValueError(
            "No audio was captured. Check System Settings → Privacy & Security → Microphone and make sure Babble is allowed."
        )

    audio = np.concatenate(chunks)
    if audio.size < int(sr * 0.15):
        raise ValueError("Recording was too short. Try holding for at least a second.")
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-5:
        raise ValueError(
            "The recording came out completely silent. Check your Mac's Microphone permission for Babble in System Settings → Privacy & Security → Microphone."
        )

    UPLOADS.mkdir(parents=True, exist_ok=True)
    name = f"recording-{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}.wav"
    dest = UPLOADS / name
    sf.write(str(dest), audio, sr, subtype="PCM_16", format="WAV")
    duration = round(float(audio.size) / sr, 2)
    return {"path": str(dest), "name": name, "duration": duration}


# ---------- models & languages ----------

def _is_downloaded(repo: str) -> bool:
    return en._local(repo) != repo


def list_models() -> list[dict]:
    return [{**m, "downloaded": _is_downloaded(m["repo"])} for m in MODELS]


@on_mlx_thread
def _evict_if_loaded(slot_key: str):
    if slot.key == slot_key:
        slot.release()


def delete_model(model_id: str) -> dict:
    """Delete a non-core STT model's cached Hugging Face weights to free disk space."""
    spec = MODELS_BY_ID.get(model_id)
    if spec is None:
        raise ValueError(f"Unknown speech-to-text model: {model_id}")
    if spec.get("core"):
        raise ValueError(f"{spec['name']} is built into Babble's voice tools and can't be removed here.")
    with _active_lock:
        if _active_models.get(model_id, 0) > 0:
            raise ValueError(f"{spec['name']} is currently transcribing. Stop or wait for it to finish first.")
        if model_id in _deleting_models:
            raise ValueError(f"{spec['name']} is already being deleted.")
        _deleting_models.add(model_id)

    try:
        # Evict from MLX slot before deleting memory-mapped weight files on disk
        _evict_if_loaded(f"stt:{model_id}")

        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir()
        freed = 0
        commits = []
        for repo in info.repos:
            if repo.repo_id == spec["repo"]:
                freed += repo.size_on_disk
                commits.extend(r.commit_hash for r in repo.revisions)
        if commits:
            info.delete_revisions(*commits).execute()
            from . import usage
            usage._storage["at"] = 0.0
        return {"id": model_id, "freed_gb": round(freed / 1e9, 2)}
    finally:
        with _active_lock:
            _deleting_models.discard(model_id)


def list_languages() -> list[dict]:
    return [{"id": l["id"], "label": l["label"]} for l in LANGUAGES]


# ---------- audio prep, transcription, history & export ----------

def _find_ffmpeg() -> str | None:
    for cand in (shutil.which("ffmpeg"), "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def _prepare_wav(src: Path) -> tuple[Path, bool, float | None]:
    """Ensure a 16 kHz mono 16-bit PCM WAV file for MLX STT models. Returns (wav_path, should_cleanup, duration_sec)."""
    if not src.is_file():
        raise ValueError("Couldn't find that audio file.")
    # Fast path: already a 16 kHz mono PCM_16 WAV (e.g. from native mic recording)
    if src.suffix.lower() == ".wav":
        try:
            info = sf.info(str(src))
            if (
                info.samplerate == SAMPLE_RATE
                and info.channels == 1
                and info.subtype == "PCM_16"
                and info.frames > 0
                and info.duration > 0
            ):
                return src, False, round(float(info.duration), 3)
        except Exception:  # noqa: BLE001
            pass

    ffmpeg = _find_ffmpeg()
    if ffmpeg:
        fd, tmp = tempfile.mkstemp(prefix="babble-stt-", suffix=".wav")
        os.close(fd)
        out = Path(tmp)
        proc = None
        # Redirect stderr to a temp file (not PIPE) so ffmpeg can never deadlock on a full pipe buffer
        with tempfile.TemporaryFile() as err_file:
            try:
                proc = subprocess.Popen(
                    [ffmpeg, "-nostdin", "-nostats", "-loglevel", "error", "-y", "-i", str(src),
                     "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(out)],
                    stdout=subprocess.DEVNULL,
                    stderr=err_file,
                )
                deadline = time.monotonic() + 300.0
                while proc.poll() is None:
                    if tasks.stop_wanted():
                        proc.terminate()
                        try:
                            proc.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                        out.unlink(missing_ok=True)
                        raise tasks.Stopped()
                    if time.monotonic() > deadline:
                        proc.kill()
                        proc.wait()
                        out.unlink(missing_ok=True)
                        raise ValueError(f"Timed out decoding audio from {src.name}.")
                    time.sleep(0.15)
                err_file.seek(0)
                stderr = err_file.read().decode("utf-8", errors="replace")
            except (tasks.Stopped, ValueError):
                raise
            except Exception:
                if proc is not None and proc.poll() is None:
                    proc.kill()
                    proc.wait()
                out.unlink(missing_ok=True)
                raise
        if proc.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
            out.unlink(missing_ok=True)
            err = stderr.strip().splitlines()[-1:] or ["ffmpeg could not decode this file"]
            raise ValueError(f"Couldn't read audio from {src.name}: {err[0]}")
        try:
            info = sf.info(str(out))
            if info.frames <= 0 or info.duration <= 0:
                out.unlink(missing_ok=True)
                raise ValueError(f"No audio track was found in {src.name}.")
            dur = round(float(info.duration), 3)
        except ValueError:
            raise
        except Exception as e:  # noqa: BLE001
            out.unlink(missing_ok=True)
            raise ValueError(f"Couldn't read decoded audio from {src.name}.") from e
        return out, True, dur

    raise ValueError(
        f"ffmpeg is required to convert {src.name} to 16 kHz mono WAV for transcription. Install ffmpeg (e.g. brew install ffmpeg)."
    )


def _normalize_segments(raw_segments, fallback_text: str, duration: float | None) -> list[dict]:
    out = []
    if raw_segments:
        for seg in raw_segments:
            if not isinstance(seg, dict):
                continue
            txt = str(seg.get("text") or "").strip()
            if not txt:
                continue
            start = seg.get("start")
            end = seg.get("end")
            out.append({
                "start": round(float(start), 2) if start is not None else 0.0,
                "end": round(float(end), 2) if end is not None else (round(duration, 2) if duration else 0.0),
                "text": txt,
            })
    if not out and fallback_text:
        out.append({"start": 0.0, "end": round(duration, 2) if duration else 0.0, "text": fallback_text})
    return out


def _normalize_language(raw_lang) -> str:
    if isinstance(raw_lang, list):
        raw_lang = next((x for x in raw_lang if x), "")
    if not raw_lang:
        return ""
    s = str(raw_lang).strip()
    if s.lower() in LANGS_BY_ID:
        return LANGS_BY_ID[s.lower()]["label"]
    return s[:1].upper() + s[1:]


def _save_transcript(record: dict) -> Path:
    TRANSCRIPTS.mkdir(parents=True, exist_ok=True)
    dest = TRANSCRIPTS / f"{record['id']}.json"
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(dest)
    return dest


def _load_transcript(tid: str) -> dict:
    if not _TID_RE.fullmatch(tid):
        raise ValueError("Invalid transcript id.")
    p = (TRANSCRIPTS / f"{tid}.json").resolve()
    if p.parent != TRANSCRIPTS.resolve() or not p.is_file():
        raise ValueError("That transcript isn't there any more.")
    return json.loads(p.read_text(encoding="utf-8"))


def _with_audio_flag(record: dict) -> dict:
    ap = record.get("audio_path")
    exists = bool(ap and Path(ap).is_file())
    return {**record, "audio_exists": exists}


def list_history(limit: int = 30) -> list[dict]:
    if not TRANSCRIPTS.is_dir():
        return []
    items = []
    for f in sorted(TRANSCRIPTS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("id"):
                items.append(_with_audio_flag(data))
        except (OSError, ValueError):
            continue
    return items


def delete_history(tid: str) -> str:
    if not _TID_RE.fullmatch(tid):
        raise ValueError("Invalid transcript id.")
    p = (TRANSCRIPTS / f"{tid}.json").resolve()
    if p.parent != TRANSCRIPTS.resolve() or not p.is_file():
        raise ValueError("That transcript is already gone.")
    audio = None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        audio = data.get("audio_path")
    except Exception:  # noqa: BLE001
        pass
    p.unlink(missing_ok=True)

    # Remove any exported TXT/SRT files created for this transcript
    if EXPORTS.is_dir():
        for exp in EXPORTS.glob(f"*-{tid[-8:]}.*"):
            exp.unlink(missing_ok=True)

    # Clean up uploaded audio only if not actively being transcribed and no other transcript references it
    if audio:
        try:
            ap = Path(audio).resolve()
            if ap.parent == UPLOADS.resolve() and ap.is_file():
                still_used = False
                for other in TRANSCRIPTS.glob("*.json"):
                    try:
                        od = json.loads(other.read_text(encoding="utf-8"))
                        if od.get("audio_path") and Path(od["audio_path"]).resolve() == ap:
                            still_used = True
                            break
                    except Exception:  # noqa: BLE001 - corrupt JSON: play safe and keep upload
                        still_used = True
                        break
                if not still_used:
                    with _active_lock:
                        if _active_audio.get(ap, 0) == 0:
                            ap.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
    return tid


def _fmt_srt_time(sec: float) -> str:
    ms = max(0, round(float(sec) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def export_transcript(tid: str, fmt: str = "txt") -> str:
    """Write a transcript to a .txt or .srt file under outputs/transcripts/exports/ for native saving."""
    fmt = (fmt or "txt").lower().strip()
    if fmt not in ("txt", "srt"):
        raise ValueError("Export format must be 'txt' or 'srt'.")
    rec = _load_transcript(tid)
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(rec.get("filename") or tid).stem).strip(".-")[:60] or tid
    out = EXPORTS / f"{base}-{tid[-8:]}.{fmt}"
    if fmt == "srt":
        segs = rec.get("segments") or []
        if not segs and rec.get("text"):
            segs = [{"start": 0.0, "end": rec.get("duration") or 1.0, "text": rec["text"]}]
        lines = []
        for i, seg in enumerate(segs, 1):
            start = float(seg.get("start") or 0.0)
            end = max(float(seg.get("end") or 0.0), start + 0.1)
            lines.append(f"{i}\n{_fmt_srt_time(start)} --> {_fmt_srt_time(end)}\n{seg.get('text', '').strip()}")
        content = "\n\n".join(lines) + "\n"
    else:
        content = (rec.get("text") or "").strip() + "\n"
    EXPORTS.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    return str(out)


@streamed_on_mlx_thread
def _run_model(wav_path: Path, spec: dict, lang: dict, timestamps: bool, duration: float | None,
               on_start=None, on_done=None, stop=None):
    """Load and run the MLX STT model on the MLX thread (audio prep already happened off-thread)."""
    if on_start is not None:
        on_start()
    acquired = False
    try:
        if (stop is not None and stop.is_set()) or tasks.stop_wanted():
            return
        from mlx_audio.stt.utils import load_model as load_stt

        cached = _is_downloaded(spec["repo"])
        action = "Loading" if cached else f"Downloading (~{spec['size_gb']:.1f} GB)"
        yield {"status": f"{action} {spec['name']}…", "progress": 0.25}
        if (stop is not None and stop.is_set()) or tasks.stop_wanted():
            return

        model = slot.get(f"stt:{spec['id']}", lambda: load_stt(en._local(spec["repo"])))
        acquired = True
        if (stop is not None and stop.is_set()) or tasks.stop_wanted():
            return
        dur_note = f" ({en.clock(duration)})" if duration else ""
        yield {"status": f"Transcribing{dur_note}…", "progress": 0.55}

        if spec["family"] == "whisper":
            kwargs = {"return_timestamps": bool(timestamps)}
            if lang["whisper"]:
                kwargs["language"] = lang["whisper"]
            out = model.generate(str(wav_path), **kwargs)
        else:
            kwargs = {}
            if lang["qwen"]:
                kwargs["language"] = lang["qwen"]
            out = model.generate(str(wav_path), **kwargs)
        yield {"_out": out}
    finally:
        if acquired:
            slot.release()
        if on_done is not None:
            on_done()


def transcribe(path: str, model_id: str = "whisper-large-v3-turbo", language: str = "auto",
               timestamps: bool = True, filename: str = "", save: bool = True):
    """Stream transcription progress and yield the completed transcript payload."""
    spec = MODELS_BY_ID.get(model_id)
    if spec is None:
        raise ValueError(f"Unknown speech-to-text model: {model_id}")
    lang = LANGS_BY_ID.get(language or "auto", LANGS_BY_ID["auto"])

    src = Path(path)
    resolved_src = src.resolve()
    display_name = Path(filename.strip()).name if filename.strip() else src.name

    # Pin the uploaded audio and model before yielding so delete_history/delete_model cannot race us
    with _active_lock:
        if spec["id"] in _deleting_models:
            raise ValueError(f"{spec['name']} is being deleted right now. Try again in a moment.")
        _active_audio[resolved_src] += 1
        _active_models[spec["id"]] += 1

    wav_path: Path | None = None
    cleanup = False
    mlx_started = False
    mlx_finished = False
    model_released = False
    audio_released = False

    def _release_model_and_wav():
        nonlocal model_released
        with _active_lock:
            if model_released:
                return
            model_released = True
            _active_models[spec["id"]] -= 1
            if _active_models[spec["id"]] <= 0:
                _active_models.pop(spec["id"], None)
        if cleanup and wav_path is not None:
            wav_path.unlink(missing_ok=True)

    def _release_audio():
        nonlocal audio_released
        with _active_lock:
            if audio_released:
                return
            audio_released = True
            _active_audio[resolved_src] -= 1
            if _active_audio[resolved_src] <= 0:
                _active_audio.pop(resolved_src, None)

    def _on_mlx_start():
        nonlocal mlx_started
        mlx_started = True

    defer_audio_to_mlx = False

    def _on_mlx_done():
        nonlocal mlx_finished
        mlx_finished = True
        _release_model_and_wav()
        if defer_audio_to_mlx:
            _release_audio()

    started = time.time()
    try:
        yield {"status": f"Preparing audio ({display_name})…", "progress": 0.08}
        if tasks.stop_wanted():
            return

        wav_path, cleanup, duration = _prepare_wav(src)
        if tasks.stop_wanted():
            return

        out = None
        use_ts = bool(timestamps) and bool(spec.get("timestamps", True))
        for ev in _run_model(wav_path, spec, lang, use_ts, duration, on_start=_on_mlx_start, on_done=_on_mlx_done):
            if "_out" in ev:
                out = ev["_out"]
            else:
                yield ev

        if out is None or tasks.stop_wanted():
            return

        text = (getattr(out, "text", None) or "").strip()
        raw_segments = (getattr(out, "segments", None) or []) if use_ts else []
        segments = _normalize_segments(raw_segments, text, duration) if use_ts else []
        if not text and segments:
            text = " ".join(s["text"] for s in segments).strip()

        detected_lang = _normalize_language(getattr(out, "language", None)) or (
            lang["label"] if lang["id"] != "auto" else ""
        )
        elapsed = round(max(time.time() - started, 0.01), 2)
        tid = f"stt-{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"

        record = {
            "id": tid,
            "filename": display_name,
            "audio_path": str(src),
            "model": spec["id"],
            "model_name": spec["name"],
            "language": detected_lang,
            "duration": duration,
            "elapsed": elapsed,
            "text": text,
            "segments": segments,
            "created": datetime.now().isoformat(timespec="seconds"),
        }
        if save:
            _save_transcript(record)

        yield {
            "status": f"Done · {len(text.split()):,} words in {elapsed:.1f}s",
            "progress": 1.0,
            "text": text,
            "transcript": _with_audio_flag(record),
        }
    finally:
        # Only unlink temp WAV / release model & audio if MLX never started or already finished
        if not mlx_started or mlx_finished:
            _release_model_and_wav()
            _release_audio()
        else:
            defer_audio_to_mlx = True
            if mlx_finished:
                _release_audio()
