"""Transcribe audiobook files with faster-whisper on a cloud GPU, for Babble's Train a voice dataset prep.

Reads stt-job.json (written by narrator/cloud_stt.py):
  {"language": "en" | null, "model": "large-v3-turbo",
   "files": [{"index": 0, "file": "000.ogg", "duration": 3600.0}, ...]}
and writes out/stt/<index:03d>.json in the exact window format narrator/dataset/pipeline.py caches:
  {"windows": [{"start": 0.0, "end": <duration>, "words": [[word, start, end], ...]}]}
plus progress.json and run.log. Stops starting new files after --max-hours so partial runs still return
every completed file.

Usage:  python stt_runner.py [--max-hours 3] [--out /kaggle/working]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

STARTED = time.time()


def say(msg, out):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(out / "run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def find_data():
    env = os.environ.get("NARRATOR_DATA")
    if env and (Path(env) / "stt-job.json").exists():
        return Path(env)
    here = Path(__file__).resolve().parent
    if (here / "stt-job.json").exists():
        return here
    hits = sorted(Path("/kaggle/input").rglob("stt-job.json")) if Path("/kaggle/input").exists() else []
    if not hits:
        sys.exit("FAILED: couldn't find stt-job.json")
    return hits[0].parent


def write_json(path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/kaggle/working" if Path("/kaggle/working").exists() else "out")
    ap.add_argument("--max-hours", type=float, default=3.0)
    args = ap.parse_args()
    out = Path(args.out)
    stt_out = out / "stt"
    stt_out.mkdir(parents=True, exist_ok=True)
    data = find_data()
    job = json.loads((data / "stt-job.json").read_text(encoding="utf-8"))

    try:
        import torch  # (first: it loads CUDA's libraries, which faster-whisper's CTranslate2 then finds)
        cuda = torch.cuda.is_available()
    except ImportError:  # (an image without torch)
        import ctranslate2
        cuda = ctranslate2.get_cuda_device_count() > 0
    from faster_whisper import WhisperModel

    device = "cuda" if cuda else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    model_name = job.get("model") or "large-v3-turbo"
    language = job.get("language") or None
    say(f"Loading faster-whisper {model_name} ({device}, {compute_type})…", out)
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    say(f"Loaded in {time.time() - STARTED:.0f} s.", out)

    files = job.get("files") or []
    limit = max(60.0, args.max_hours * 3600.0 - 180.0)
    done = 0
    stopped = False
    for item in files:
        if time.time() - STARTED > limit:
            stopped = True
            say(f"Time limit reached ({args.max_hours} h): saved {done} of {len(files)} file(s).", out)
            break
        idx = int(item["index"])
        audio_path = data / item["file"]
        dur = float(item.get("duration") or 0.0)
        t0 = time.time()
        say(f"Transcribing file {idx + 1} ({item['file']}, {dur / 60:.1f} min)…", out)
        segments, info = model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        words = []
        for seg in segments:
            for w in (seg.words or []):
                txt = (w.word or "").strip()
                if txt:
                    words.append([txt, round(float(w.start), 3), round(float(w.end), 3)])
        end_s = max(dur, words[-1][2] if words else 0.0, float(getattr(info, "duration", 0.0) or 0.0))
        write_json(stt_out / f"{idx:03d}.json", {"windows": [{"start": 0.0, "end": round(end_s, 3), "words": words}]})
        done += 1
        took = max(0.1, time.time() - t0)
        rtf = (end_s / took) if end_s > 0 else 0.0
        say(f"Finished file {idx + 1}: {len(words):,} words in {took:.1f} s ({rtf:.1f}x real time).", out)
        write_json(out / "progress.json", {"done": done, "total": len(files), "stopped": stopped,
                                           "seconds": round(time.time() - STARTED)})

    write_json(out / "progress.json", {"done": done, "total": len(files), "stopped": stopped,
                                       "seconds": round(time.time() - STARTED)})
    say(f"Done: {done}/{len(files)} file(s) in {(time.time() - STARTED) / 60:.1f} min.", out)


if __name__ == "__main__":
    main()
