"""Tools for preparing a voice (Train a voice → Training tools): put several book files together as one text, or
several audio files together as one recording, in the order given. Merged files go to TRAINING/tools."""
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from . import book
from .paths import TRAINING

TOOLS = TRAINING / "tools"
TEXT_EXT = {".txt", ".md", ".epub", ".pdf", ".docx", ".mobi", ".azw", ".azw3", ".fb2", ".rtf", ".odt", ".html", ".htm"}
AUDIO_EXT = {".mp3", ".m4a", ".m4b", ".aac", ".flac", ".wav", ".ogg", ".oga", ".opus", ".wma", ".aiff", ".aif", ".caf"}
FORMATS = {  # output: extension and ffmpeg encoder settings
    "flac": (".flac", ["-c:a", "flac"]),
    "mp3": (".mp3", ["-c:a", "libmp3lame", "-q:a", "2"]),
    "m4a": (".m4a", ["-c:a", "aac", "-b:a", "160k"]),
    "wav": (".wav", ["-c:a", "pcm_s16le"]),
}


def natural_key(name):
    """"Chapter 2" before "Chapter 10"."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _out_path(name, default, ext):
    stem = re.sub(r"[^\w\- ]+", "", (name or "").strip())[:80].strip() or default
    TOOLS.mkdir(parents=True, exist_ok=True)
    out, n = TOOLS / f"{stem}{ext}", 2
    while out.exists():  # never overwrite an earlier merge
        out, n = TOOLS / f"{stem} ({n}){ext}", n + 1
    return out


def merge_text(paths, name="", headings=True, log=print, cleanup=None):
    try:
        return _merge_text(paths, name, headings, log)
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)


def _merge_text(paths, name, headings, log):
    """Read each book file (any format the Audiobook import reads) and join their text, in order.
    headings: keep chapter titles as lines of their own (a narrator usually reads them), else leave them out."""
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("Add the files to merge first.")
    bad = [_label(p) for p in paths if p.suffix.lower() not in TEXT_EXT]
    if bad:
        raise ValueError(f"These aren't text or book files: {', '.join(bad[:5])}.")
    work = TOOLS / "_work" / time.strftime("%Y%m%d-%H%M%S")
    try:
        parts = []
        for i, p in enumerate(paths, 1):
            log(f"Reading {_label(p)} ({i} of {len(paths)})")
            text = book.import_book(p, work / str(i))["text"]
            text = re.sub(r"(?m)^## (.*)$", r"\1" if headings else "", text)
            parts.append(re.sub(r"\n{3,}", "\n\n", text).strip())
        out = _out_path(name, "Merged text", ".txt")
        joined = "\n\n".join(x for x in parts if x) + "\n"
        out.write_text(joined, encoding="utf-8")
        words = len(joined.split())
        log(f"Done · Merged {len(paths)} files into {out.name}: {words:,} words.")
        return {"file": str(out), "name": out.name, "words": words, "files": len(paths)}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _label(p):
    """The file's name as the user gave it (uploads are kept as 001-<name> to hold their order)."""
    return re.sub(r"^\d{3}-", "", Path(p).name)


def _probe(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                        "stream=channels,sample_rate:format=duration", "-of", "json", str(p)], capture_output=True, text=True)
    try:
        info = json.loads(r.stdout or "{}")
        stream = (info.get("streams") or [{}])[0]
        return float(info.get("format", {}).get("duration") or 0), int(stream.get("channels") or 0), int(stream.get("sample_rate") or 0)
    except (ValueError, TypeError):
        return 0.0, 0, 0


def merge_audio(paths, name="", fmt="flac", log=print, cleanup=None):
    try:
        return _merge_audio(paths, name, fmt, log)
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)


def _merge_audio(paths, name, fmt, log):
    """Join audio files end to end, in order, into one file (any mix of formats, rates and channels)."""
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("Add the files to merge first.")
    if fmt not in FORMATS:
        raise ValueError(f"Unknown format {fmt}.")
    bad = [_label(p) for p in paths if p.suffix.lower() not in AUDIO_EXT]
    if bad:
        raise ValueError(f"These aren't audio files: {', '.join(bad[:5])}.")
    info = []
    for i, p in enumerate(paths, 1):
        d, ch, sr = _probe(p)
        if not ch:
            raise ValueError(f"{_label(p)} has no audio in it (or isn't a file ffmpeg can read).")
        info.append((d, ch, sr))
    total = sum(d for d, _, _ in info)
    stereo = any(ch >= 2 for _, ch, _ in info)
    rate = min(48000, max(sr for _, _, sr in info)) or 44100
    layout = "stereo" if stereo else "mono"
    log(f"Joining {len(paths)} files ({_clock(total)} in all), {layout}, {rate // 1000} kHz…")
    ext, codec = FORMATS[fmt]
    out = _out_path(name, "Merged audio", ext)
    graph = "".join(f"[{i}:a:0]aresample={rate},aformat=sample_fmts=fltp:channel_layouts={layout}[a{i}];" for i in range(len(paths)))
    graph += "".join(f"[a{i}]" for i in range(len(paths))) + f"concat=n={len(paths)}:v=0:a=1[out]"
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-y"]
    for p in paths:
        cmd += ["-i", str(p)]
    cmd += ["-filter_complex", graph, "-map", "[out]", "-vn", *codec, "-progress", "pipe:1", "-nostats", str(out)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    shown = -1
    for line in proc.stdout:
        m = re.match(r"out_time_us=(\d+)", line.strip())
        if m and total:
            pct = min(99, int(int(m[1]) / 1e6 / total * 100))
            if pct >= shown + 2:
                shown = pct
                log(f"Joining the audio… {pct}%")
    err = proc.stderr.read()
    if proc.wait() != 0 or not out.exists():
        out.unlink(missing_ok=True)
        raise ValueError(f"ffmpeg couldn't join them: {err.strip()[-300:] or 'unknown error'}")
    d, _, _ = _probe(out)
    size = out.stat().st_size / 1e6
    log(f"Done · Merged {len(paths)} files into {out.name}: {_clock(d)}, {size:,.1f} MB.")
    return {"file": str(out), "name": out.name, "seconds": round(d, 1), "files": len(paths), "mb": round(size, 1)}


def _clock(s):
    s = int(s or 0)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"
