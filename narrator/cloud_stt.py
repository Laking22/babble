"""Transcribe audiobook files for voice training on a cloud GPU (Kaggle, Modal, or any pay-as-you-go service).

Mirrors narrator/cloud_ai.py: encodes only uncached audio files to compact 16 kHz mono Opus (~11 MB/hour),
sends them with kaggle_render/stt_runner.py (faster-whisper large-v3-turbo), waits with reconnect & stop
handling, fetches stt/<k:03d>.json back into work/stt/, and tears down the rented GPU in finally.
"""
import json
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import soundfile as sf

from .paths import HOME

RUNNER = Path(__file__).resolve().parent / "kaggle_render" / "stt_runner.py"
RECORD = "cloud-stt.json"  # next to a book's stt/ folder while its run is out: Babble quitting picks it up again
PIP = ["faster-whisper"]
MODEL = "large-v3-turbo"
BASE_RTF = 12.0       # conservative real-time speed on a T4 with faster-whisper large-v3-turbo + word timestamps
START_HOURS = 0.06    # container boot + model download (~3.5 min)


def where_name(make_on):
    from .training import BACKENDS
    b = BACKENDS.get(make_on)
    return b.name if b else make_on


def _bin(name):
    for c in (shutil.which(name), f"/opt/homebrew/bin/{name}", f"/usr/local/bin/{name}"):
        if c and Path(c).exists():
            return c
    return name


def _encode_compact(src, dst_base):
    """Encode 24 kHz WAV to compact 16 kHz mono Opus (~11 MB/h), falling back to 32k MP3 if libopus is absent."""
    ogg = dst_base.with_suffix(".ogg")
    proc = subprocess.run(
        [_bin("ffmpeg"), "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "libopus", "-b:a", "24k", str(ogg)],
        capture_output=True, text=True,
    )
    if proc.returncode == 0 and ogg.exists() and ogg.stat().st_size > 100:
        return ogg
    ogg.unlink(missing_ok=True)
    mp3 = dst_base.with_suffix(".mp3")
    proc = subprocess.run(
        [_bin("ffmpeg"), "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "libmp3lame", "-b:a", "32k", str(mp3)],
        capture_output=True, text=True,
    )
    if proc.returncode == 0 and mp3.exists() and mp3.stat().st_size > 100:
        return mp3
    mp3.unlink(missing_ok=True)
    raise ValueError(f"Couldn't compress {src.name} for cloud upload: {proc.stderr.strip()[-200:]}")


def _plan(backend, cloud_gpu, audio_seconds):
    """(gpu, speed, price, hours, budget, worst_hours) for transcribing `audio_seconds` of audio."""
    from . import cloud_book as cb
    from .training import BACKENDS
    from .training.backends import CLOUDS
    if backend not in CLOUDS:
        raise ValueError(f"No such cloud GPU: {backend}.")
    b = BACKENDS[backend]
    if backend == "kaggle":
        gpu, speed, price, cap = "T4", 1.0, None, 11.0
    elif backend == "modal":
        from .training.backends.modal_cloud import GPUS
        gpu = cloud_gpu if cloud_gpu in GPUS else "L4"
        speed, price, cap = cb.MODAL_SPEED.get(gpu, 1.8), GPUS[gpu][1], cb.MODAL_MAX_HOURS
    else:
        gpu = cloud_gpu if cloud_gpu in b.gpu_names else b.default_gpu
        speed, cap = b.speed.get(gpu, 1.8), cb.MODAL_MAX_HOURS
        try:
            price = b.price(gpu)
        except Exception:  # noqa: BLE001
            price = None
    audio_h = max(0.05, float(audio_seconds or 0.0) / 3600.0)
    hours = audio_h / (BASE_RTF * max(0.5, speed)) + START_HOURS
    budget = round(min(cap, max(0.3, hours * 2.5)), 1)
    return gpu, speed, price, hours, budget, budget + cb.overhang(backend)


def _clock(hours):
    return f"{max(1, round(hours * 60))} min" if hours < 1 else f"{hours:.1f} h"


def estimate(make_on, cloud_gpu, audio_seconds):
    """Before sending: {"cost", "time", "free", "where", "gpu", "hours", "budget"} for confirmation UI."""
    from . import pricing
    gpu, _, price, hours, budget, worst = _plan(make_on, cloud_gpu, audio_seconds)
    if make_on == "kaggle":
        cost = f"free: {hours:.1f} to {budget + 0.5:.1f} of your weekly Kaggle GPU hours"
    else:
        cost = pricing.say(pricing.gpu_range(make_on, price, hours * 0.75, worst, "prepare")) if price else "price unknown"
    return {
        "cost": cost,
        "time": f"about {_clock(hours * 0.75)} to {_clock(worst)}",
        "free": make_on == "kaggle",
        "where": where_name(make_on),
        "gpu": gpu,
        "hours": round(hours, 2),
        "budget": budget,
    }


def _cached_complete(cache_path, duration):
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        wins = data.get("windows") or []
        if wins and float(wins[-1].get("end") or 0.0) >= max(0.0, duration - 1.0):
            return [w for win in wins for w in win.get("words") or []]
    except (OSError, ValueError, TypeError):
        pass
    return None


def _record(stt_dir):
    try:
        return json.loads((Path(stt_dir).parent / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def in_flight(work):
    """The cloud run a book's transcription is waiting on (sent before Babble quit), or None."""
    return _record(Path(work) / "stt")


def remaining_seconds(audio_files, stt_dir):
    """Seconds of a book's audio (its converted 24 kHz files) not transcribed yet."""
    left = 0.0
    for k, wav in enumerate(audio_files):
        dur = sf.info(str(wav)).duration
        if _cached_complete(Path(stt_dir) / f"{k:03d}.json", dur) is None:
            left += dur
    return left


def _through(fn, *args, name, deadline, stop, stage, hosts=()):
    """Call fn(*args), riding out transient network drops while honoring stop.is_set()."""
    from . import conn_issues
    lost = None
    while True:
        if stop.is_set():
            raise InterruptedError
        try:
            answer = fn(*args)
        except Exception as e:
            if not conn_issues.is_network(e) or time.time() > deadline:
                raise
            if lost is None:
                lost = time.time()
                print(f"[cloud stt] lost connection to {name}: {str(e)[:160]}", flush=True)
            stage(f"Lost connection to {name} ({_clock((time.time() - lost) / 3600)} ago) — waiting…", 0.5)
            for i in range(60):
                if stop.is_set():
                    raise InterruptedError
                time.sleep(1)
                if i % 10 == 9 and conn_issues.reachable(hosts):
                    break
            continue
        if lost is not None:
            print(f"[cloud stt] connection to {name} is back", flush=True)
        return answer


def transcribe_files(voice, files24, stt_dir, language, make_on, cloud_gpu, stop, stage, log):
    """Transcribe all uncached files in `files24` on a cloud GPU, write caches to `stt_dir`, and return
    [[word, start, end, file_idx], ...] across all files."""
    from . import cloud_book as cb
    from . import conn_issues, pricing, spend
    from .training import BACKENDS
    from .training.backends import CLOUDS
    from .training.base import Package

    stt_dir = Path(stt_dir)
    stt_dir.mkdir(parents=True, exist_ok=True)
    durations = [sf.info(str(p)).duration for p in files24]
    cached = {}
    todo = []
    for k, (wav24, dur) in enumerate(zip(files24, durations)):
        hit = _cached_complete(stt_dir / f"{k:03d}.json", dur)
        if hit is not None:
            cached[k] = hit
        else:
            todo.append((k, wav24, dur))

    if not todo:
        log("All audio files already transcribed (reusing cache).")
        return [[w, s, e, k] for k in range(len(files24)) for w, s, e in cached[k]]

    if make_on not in CLOUDS:
        raise ValueError(f"No such cloud GPU: {make_on}.")
    b = BACKENDS[make_on]
    name = b.name
    rec_path = stt_dir.parent / RECORD
    rec = _record(stt_dir)
    if rec and (rec.get("backend") != make_on or not rec.get("remote")
                or sorted(rec.get("todo") or []) != sorted(k for k, _, _ in todo)):
        rec = None  # (a run for other files or another service: not this one)
    if rec:
        return _wait_and_fetch(b, make_on, rec, todo, files24, cached, stt_dir, rec_path, stop, stage, log,
                               fresh=False)
    stage(f"Checking your {name} account…", 0.02)
    c = b.check()
    if not c.get("ok"):
        conn_issues.note(make_on, RuntimeError(c.get("detail") or ""), "Dataset transcription")
        raise ValueError(c.get("detail") or f"{name} isn't signed in.")

    todo_secs = sum(d for _, _, d in todo)
    gpu, _, price, hours, budget, worst = _plan(make_on, cloud_gpu, todo_secs)
    if make_on == "kaggle":
        left = (c.get("quota") or {}).get("left", 11.0)
        budget = round(min(budget, left - 0.3), 1)
        if budget < 0.3:
            raise ValueError(f"Kaggle has {left:.1f} GPU hours left this week, not enough to start.")
    elif c.get("balance") is not None and c["balance"] < 1.0:
        raise ValueError(f"Your {name} account has ${c['balance']:.2f} of credit. Add some on {name}'s billing page first.")

    est = pricing.gpu_range(make_on, price, hours * 0.75, worst, "prepare")
    spend.allow(make_on, est, "prepare")

    run_id = f"stt-{uuid.uuid4().hex[:8]}"
    folder = HOME / "cloud-stt" / run_id
    data = folder / "data"
    data.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RUNNER, data / RUNNER.name)

    try:  # (anything going wrong before the run is out: the compressed copies go)
        job_files = []
        for i, (k, wav24, dur) in enumerate(todo):
            if stop.is_set():
                raise InterruptedError
            stage(f"Compressing audio for {name} ({i + 1}/{len(todo)})", 0.05 + 0.15 * i / max(len(todo), 1))
            encoded = _encode_compact(wav24, data / f"{k:03d}")
            job_files.append({"index": k, "file": encoded.name, "duration": round(dur, 3)})

        upload_mb = sum((data / f["file"]).stat().st_size for f in job_files) / (1024 * 1024)
        (data / "stt-job.json").write_text(json.dumps({
            "language": language,
            "model": MODEL,
            "files": job_files,
        }, indent=1), encoding="utf-8")
        log(f"Sending {len(todo)} file(s) ({todo_secs / 3600:.2f} h, {upload_mb:.1f} MB compressed) to {name} ({gpu}).")

        pip = PIP + list(getattr(b, "extra_pip", []))
        slug = f"narrator-stt-{voice[:24]}"
        pkg = Package(
            model=slug,
            recipe="stt",
            data_dir=data,
            script=RUNNER.name,
            pip=pip,
            gpu=gpu,
            slug=slug,
            title=f"Transcribe {voice}",
            meta={"timeout_hours": budget + 0.5},
            args=f"--max-hours {budget}",
        )

        if stop.is_set():
            raise InterruptedError
        stage(f"Uploading to {name} ({upload_mb:.1f} MB)…", 0.22)
        remote = b.submit(pkg, log=log)
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    rec = {"backend": make_on, "remote": remote, "todo": [k for k, _, _ in todo], "sent": time.time(),
           "price": price, "hours": hours, "worst": worst, "gpu": gpu, "folder": str(folder)}
    rec_path.write_text(json.dumps(rec), encoding="utf-8")
    spend.job_sent(make_on, remote, f"Dataset transcription ({voice})", price=price, hours=worst, collect=False)
    return _wait_and_fetch(b, make_on, rec, todo, files24, cached, stt_dir, rec_path, stop, stage, log, fresh=True)


def _wait_and_fetch(b, make_on, rec, todo, files24, cached, stt_dir, rec_path, stop, stage, log, fresh):
    """Follow a run until it ends, bring its transcripts into stt_dir, and free the GPU. The run's record stays
    only while it's out (Babble quitting mid-run picks it up again instead of sending another)."""
    from . import cloud_book as cb
    from . import conn_issues, spend
    name, remote = b.name, rec["remote"]
    hours, worst = float(rec.get("hours") or 0.1), float(rec.get("worst") or 1.0)
    run_id = f"stt-{remote}"
    folder = Path(rec.get("folder") or HOME / "cloud-stt" / re.sub(r"[^A-Za-z0-9._-]+", "-", str(remote))[-80:])
    if folder.parent != HOME / "cloud-stt":  # (only ever a folder of its own under cloud-stt/)
        folder = HOME / "cloud-stt" / re.sub(r"[^A-Za-z0-9._-]+", "-", str(remote))[-80:]
    if not fresh:
        log(f"Carrying on with the transcription already running on {name}.")
    cb._busy.add(run_id)
    cb.stay_awake()
    done_ok, ended, leaving = False, False, False
    try:
        sent = float(rec.get("sent") or time.time())
        deadline, hosts = sent + (worst + 1.0) * 3600.0, getattr(b, "hosts", ())
        while True:
            st = _through(b.status, remote, name=name, deadline=deadline, stop=stop, stage=stage, hosts=hosts)
            if st.state in ("complete", "error", "cancelled"):
                break
            waited_h = (time.time() - sent) / 3600.0
            frac = min(0.88, 0.25 + 0.60 * (waited_h / max(hours, 0.05)))
            phase = "waiting for a GPU" if st.state == "queued" else "transcribing"
            stage(f"Transcribing on {name} ({phase}, {_clock(waited_h)})", frac)
            for _ in range(15):
                if stop.is_set():
                    raise InterruptedError
                time.sleep(1)
        ended = True
        if st.state == "cancelled":
            raise InterruptedError
        stage(f"Downloading transcripts from {name}…", 0.90)
        out = folder / "out"
        _through(b.fetch, remote, out, r"(stt/\d+\.json|progress\.json|run\.log)$", name=name,
                 deadline=max(deadline, time.time() + 3600.0), stop=stop, stage=stage, hosts=hosts)
        fetched = 0
        for k, _, dur in todo:
            hit = next(iter(out.rglob(f"{k:03d}.json")), None)
            if hit and hit.is_file():
                payload = json.loads(hit.read_text(encoding="utf-8"))
                dst = stt_dir / f"{k:03d}.json"
                tmp = dst.with_name(f".{dst.name}.tmp")
                try:
                    with open(tmp, "w", encoding="utf-8") as fh:
                        fh.write(json.dumps(payload, ensure_ascii=False))
                        fh.flush()
                        os.fsync(fh.fileno())
                    os.replace(tmp, dst)
                finally:
                    tmp.unlink(missing_ok=True)
                cached[k] = [w for win in (payload.get("windows") or []) for w in (win.get("words") or [])]
                fetched += 1
        if fetched < len(todo):
            logs = list(out.rglob("run.log"))
            tail = "\n".join(logs[0].read_text(errors="replace").splitlines()[-6:]) if logs else ""
            raise ValueError(
                f"{name} returned {fetched} of {len(todo)} transcript(s). "
                f"{tail[-300:] or (st.detail or 'Check the run log on the service.')}"
            )
        conn_issues.ok(make_on)
        done_ok = True
        stage(f"Transcribed on {name}", 1.0)
        return [[w, s, e, k] for k in range(len(files24)) for w, s, e in cached[k]]
    except Exception as e:
        if not isinstance(e, InterruptedError):
            conn_issues.note(make_on, e, "Dataset transcription")
        raise
    except BaseException:  # Babble shutting down: the run goes on, and is picked up again on the next start
        leaving = True
        raise
    finally:
        cb._busy.discard(run_id)
        cb.stay_awake()
        if not leaving:
            try:
                if done_ok and getattr(b, "finished", None):
                    b.finished(remote)
                elif not done_ok and not ended:
                    spend.stop_remote(make_on, remote)  # (stopped, or gave up waiting: it mustn't run on)
            except Exception as e:  # noqa: BLE001
                print(f"[cloud stt] couldn't free {remote}: {e}", flush=True)
            spend.job_ended(remote)
            rec_path.unlink(missing_ok=True)
            shutil.rmtree(folder, ignore_errors=True)
