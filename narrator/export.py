"""Join chapter audio into one file: M4B audiobook (chapters, cover, metadata), MP3 or WAV."""
import subprocess
import tempfile
from pathlib import Path

import soundfile as sf

FORMATS = ["Audiobook (M4B with chapters)", "MP3", "WAV"]


def _escape(value):
    return str(value or "").replace("\\", "\\\\").replace("=", "\\=").replace(";", "\\;").replace("#", "\\#").replace("\n", " ")


# Broadcast/Audible-style vocal mastering chain:
# - highpass=70 Hz: removes sub-vocal rumble/plosives without thinning deep voices
# - equalizer 6.8 kHz (-2.5 dB, Q=2): gentle de-essing for TTS sibilance
# - acompressor (2:1): transparent vocal glue across speakers
# - loudnorm (-19 LUFS, -3 dBTP, LRA 11): Audible/ACX-grade loudness & true-peak ceiling
MASTER_FILTER = (
    "highpass=f=75,"
    "equalizer=f=6800:t=q:w=2.0:g=-2.5,"
    "acompressor=threshold=-20dB:ratio=2:attack=12:release=140:makeup=1,"
    "loudnorm=I=-19.5:TP=-3.5:LRA=11"
)

# The finished file's quality (Settings → General: settings.file_quality), lowest to highest. The rate is always
# set: loudnorm puts out 192 kHz otherwise (a WAV 8× too big, an M4B at 96 kHz). The voices are made at 24 kHz, so
# above that only the file grows; 44.1 kHz is what audiobook stores ask for. aac: kbps (mono, stereo).
QUALITY = {
    "small": {"rate": 24000, "aac": (48, 96), "mp3": 96, "wav": "pcm_s16le"},
    "standard": {"rate": 44100, "aac": (64, 128), "mp3": 192, "wav": "pcm_s16le"},
    "high": {"rate": 44100, "aac": (128, 192), "mp3": 256, "wav": "pcm_s24le"},
    "highest": {"rate": 48000, "aac": (192, 256), "mp3": 320, "wav": "pcm_s24le"},
}


def quality(name=None):
    if name is None:
        try:
            from . import settings
            name = settings.load().get("file_quality")
        except Exception:  # noqa: BLE001 - (no settings: the standard file)
            name = None
    return QUALITY.get(name or "standard", QUALITY["standard"])


def combine(chapter_files, titles, out_base, fmt, title="", author="", cover=None, progress=None, master=True,
            file_quality=None):
    """Write the combined file and return its path. progress(0..1) is called as ffmpeg goes (an 18-hour M4B
    takes minutes); when it returns False, ffmpeg is stopped and this raises ValueError. file_quality: a QUALITY
    name (else the one picked in Settings)."""
    q = quality(file_quality)
    ext = {"MP3": ".mp3", "WAV": ".wav"}.get(fmt, ".m4b")
    out = Path(out_base).with_suffix(ext)
    with tempfile.TemporaryDirectory() as tmp:
        # chapters with sound effects are stereo, plain ones mono: joining needs them all the same, so upmix the mono
        ch_counts = [sf.info(f).channels for f in chapter_files]
        channels = set(ch_counts)
        stereo = any(c > 1 for c in channels)
        upmix_share = 0.0
        if len(channels) > 1:
            mono = [i for i, c in enumerate(ch_counts) if c == 1]
            upmix_share = 0.08 if mono else 0.0
            fixed = []
            done_mono = 0
            for i, (f, c) in enumerate(zip(chapter_files, ch_counts)):
                if c == 1:
                    if progress is not None and progress(upmix_share * done_mono / max(len(mono), 1)) is False:
                        raise ValueError("Stopped before the book was put together.")
                    dst = Path(tmp) / f"stereo-{i:03d}.wav"
                    ch_us = max(int(sf.info(f).duration * 1_000_000), 1)
                    up_proc = subprocess.Popen(
                        ["ffmpeg", "-y", "-v", "error", "-i", str(f), "-ac", "2", "-progress", "pipe:1", "-nostats", str(dst)],
                        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    )
                    up_stopped = False
                    for line in up_proc.stdout:
                        k, _, v = line.strip().partition("=")
                        if k == "out_time_us" and progress is not None and v.isdigit():
                            frac = min(1.0, int(v) / ch_us)
                            if progress(upmix_share * (done_mono + frac) / max(len(mono), 1)) is False:
                                up_stopped = True
                                up_proc.kill()
                                break
                    _, up_err = up_proc.communicate()
                    if up_stopped:
                        raise ValueError("Stopped before the book was put together.")
                    if up_proc.returncode != 0:
                        raise ValueError(f"Couldn't prepare audio channels: {(up_err or '')[-200:]}")
                    done_mono += 1
                    f = dst
                fixed.append(f)
            chapter_files = fixed
            if progress is not None and upmix_share:
                if progress(upmix_share) is False:
                    raise ValueError("Stopped before the book was put together.")
        listing = Path(tmp) / "list.txt"
        listing.write_text("".join(f"file '{Path(f).as_posix()}'\n" for f in chapter_files), encoding="utf-8")
        meta = [";FFMETADATA1", f"title={_escape(title)}", f"artist={_escape(author)}", f"album={_escape(title)}",
                "genre=Audiobook"]
        start = 0
        for f, name in zip(chapter_files, titles):
            length = int(sf.info(f).duration * 1000)
            meta += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={start}", f"END={start + length}", f"title={_escape(name)}"]
            start += length
        meta_file = Path(tmp) / "meta.txt"
        meta_file.write_text("\n".join(meta) + "\n", encoding="utf-8")

        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-i", str(meta_file)]
        use_cover = bool(cover) and Path(cover).exists() and fmt != "WAV"
        if use_cover:
            cmd += ["-i", str(cover)]
        cmd += ["-map", "0:a", "-map_metadata", "1"]
        if fmt not in ("MP3", "WAV"):
            cmd += ["-map_chapters", "1"]
        if use_cover:
            cmd += ["-map", "2:v", "-c:v", "mjpeg", "-disposition:v:0", "attached_pic"]
        if master:
            cmd += ["-af", MASTER_FILTER]
        cmd += ["-ar", str(q["rate"])]
        if fmt == "MP3":
            cmd += ["-c:a", "libmp3lame", "-b:a", f"{q['mp3']}k", "-id3v2_version", "3"]
        elif fmt == "WAV":
            cmd += ["-c:a", q["wav"]]
        else:
            # Use +faststart and 64-bit chunk offsets for long (>12h) epics so seeking/chapters stay valid
            cmd += ["-c:a", "aac", "-b:a", f"{q['aac'][1 if stereo else 0]}k", "-movflags", "+faststart+use_metadata_tags",
                    "-f", "ipod"]
        total_us = max(start, 1) * 1000  # the book's length (ms -> microseconds, as ffmpeg reports it)
        encode_span = 1.0 - upmix_share
        with tempfile.TemporaryFile("w+") as err:
            proc = subprocess.Popen(cmd + ["-progress", "pipe:1", "-nostats", str(out)], stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=err, text=True)
            stopped = False
            for line in proc.stdout:
                key, _, value = line.strip().partition("=")
                if key == "out_time_us" and progress is not None and value.isdigit():
                    p = upmix_share + encode_span * min(1.0, int(value) / total_us)
                    if progress(min(1.0, p)) is False:
                        stopped = True
                        proc.kill()
                        break
            proc.wait()
            err.seek(0)
            message = err.read().strip()
        if stopped:
            out.unlink(missing_ok=True)
            raise ValueError("Stopped before the book was put together.")
        if proc.returncode != 0:
            raise ValueError(f"Couldn't create the {ext[1:].upper()} file: {message[-300:]}")
    return out
