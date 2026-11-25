"""Small audio helpers shared by engines and the renderer."""
import re
import subprocess

import numpy as np
import soundfile as sf

from .runtime import SR


def resample(audio, sr, to=SR):
    audio = np.asarray(audio, dtype=np.float32)
    if sr == to or len(audio) == 0:
        return audio
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(int(sr), int(to))
    return resample_poly(audio, to // g, sr // g).astype(np.float32)


def read_mono(path, to=SR, max_seconds=None):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = resample(audio.mean(axis=1), sr, to)
    return audio[: to * max_seconds] if max_seconds else audio


def flat(results):
    """mlx-audio yields result chunks; join their audio into one float32 array."""
    return np.concatenate([np.array(r.audio, dtype=np.float32).flatten() for r in results])


SILENT_PEAK = 10 ** (-50 / 20)  # a line that never gets louder than -50 dB has no speech in it


def finite(wav):
    """Mono float32 with any NaN or infinity (a model glitch) turned into silence, so it can't spread through the
    mix: an echo or filter run over a single NaN turns the whole stretch silent when it's saved."""
    a = np.asarray(wav, dtype=np.float32).reshape(-1)
    bad = ~np.isfinite(a)
    return np.where(bad, 0.0, a).astype(np.float32) if bad.any() else a


def is_silent(wav):
    """True when audio that should hold speech is empty, mostly broken (NaN) or never louder than -50 dB.
    (A model once returned a whole paragraph of exact silence, with no error.)"""
    a = np.asarray(wav, dtype=np.float32).reshape(-1)
    if len(a) < SR // 20:
        return True
    ok = np.isfinite(a)
    if ok.mean() < 0.98:
        return True
    return float(np.abs(a[ok]).max()) < SILENT_PEAK


# A line cut short: far less sound in it than its words take (cut off, or only a breath or a click). The cloud's
# runner (kaggle_render/runner.py) makes the same test: keep the two alike.
SPEECH_RATE = 45.0  # letters a second: about twice the fastest narration measured (9-24); faster than that, it was cut short
MIN_SPEECH = 0.15   # seconds of sound any line with words in it has at least
UNSAID = re.compile(r"\[[^\]]*\]|<[^>]*>")  # directions ([whispering]) and tags (<laugh>) aren't words


def voiced_seconds(wav):
    """How long a take has sound in it: 10 ms steps within 30 dB of its loudest (and louder than -50 dB)."""
    a = np.nan_to_num(np.asarray(wav, dtype=np.float32).reshape(-1))
    hop = SR // 100
    steps = len(a) // hop
    if steps == 0:
        return 0.0
    db = 10 * np.log10((a[:steps * hop].reshape(steps, hop) ** 2).mean(1) + 1e-12)
    return float((db > max(db.max() - 30.0, -50.0)).sum()) / 100


def cut_short(wav, text):
    """True when a take has far less sound in it than the words of its line take (and isn't silent: is_silent)."""
    letters = len(re.findall(r"\w", UNSAID.sub(" ", text or "")))
    return bool(letters) and not is_silent(wav) and voiced_seconds(wav) < max(MIN_SPEECH, letters / SPEECH_RATE)


# A take that skipped words: the voice's own pace gives it away. Each voice reads at its own steady rate, so a take
# far quicker than the voice's usual (more letters than its sound could say) left words out. Measured on an
# audiobook: good takes were at most 1.24x the voice's middle rate, takes that left words out up to 1.8x.
RUSH = 1.25         # this much quicker than the voice's middle rate: words were left out
RUSH_LETTERS = 60   # shorter lines vary too much in pace to tell
RUSH_TAKES = 20     # takes of a voice needed to know its pace
RUSH_ABS_RATE = 29.0  # letters/sec ceiling when < RUSH_TAKES (normal range 9-24; margin prevents false retries on fast voices)


def _letters(text):
    return len(re.findall(r"\w", UNSAID.sub(" ", text or "")))


def rushed(takes):
    """takes: {key: (voice, text, wav)}. The keys of takes much quicker than the rest of their voice's (see RUSH),
    or faster than RUSH_ABS_RATE on long lines when a short run has fewer than RUSH_TAKES takes."""
    by_voice = {}
    for k, (voice, text, wav) in takes.items():
        if wav is None or not len(wav) or is_silent(wav):
            continue
        n = _letters(text)
        if n:
            by_voice.setdefault(voice, []).append((k, n, n / max(voiced_seconds(wav), 0.1)))
    out = set()
    for rows in by_voice.values():
        if len(rows) < RUSH_TAKES:
            out |= {k for k, n, r in rows if n >= RUSH_LETTERS and r > RUSH_ABS_RATE}
            continue
        middle = float(np.median([r for _, _, r in rows]))
        out |= {k for k, n, r in rows if n >= RUSH_LETTERS and r > RUSH * middle}
    return out


def stretch(wav, speed):
    """Change speaking speed without changing pitch (ffmpeg atempo)."""
    if abs(speed - 1.0) < 0.01 or len(wav) == 0:
        return wav
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
         "-filter:a", f"atempo={speed:.3f}", "-f", "f32le", "-ar", str(SR), "-ac", "1", "pipe:1"],
        input=wav.astype(np.float32).tobytes(), capture_output=True, check=True)
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def clock(seconds):
    s = int(seconds)
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def trim_edges(wav, keep_lead=0.04, keep_tail=0.08, floor_db=-42.0):
    """Cut the silence a model leaves before and after a line down to a small, even margin.
    Orpheus adds about 0.5 s per line and Kokoro about 0.7 s, which piles up between cast lines."""
    if len(wav) < SR // 10:
        return wav
    hop = int(SR * 0.01)
    frames = len(wav) // hop
    e = np.sqrt((wav[:frames * hop].reshape(frames, hop) ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(e)
    voiced = np.where(db > max(db.max() - 40.0, floor_db))[0]
    if not len(voiced):
        return wav
    start = max(0, voiced[0] * hop - int(SR * keep_lead))
    end = min(len(wav), (voiced[-1] + 1) * hop + int(SR * keep_tail))
    out = wav[start:end].copy()
    fade = min(int(SR * 0.012), len(out) // 4)
    if fade:
        # raised-cosine edge taper eliminates slope discontinuities (clicks/pops) at line boundaries
        curve = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade, dtype=np.float32))).astype(np.float32)
        out[:fade] *= curve
        out[-fade:] *= curve[::-1]
    return out


SOFT_DELIVERY = re.compile(
    r"\b(whisper\w*|hush\w*|murmur\w*|softly|quietly|sotto\s+voce|breathless\w*|mutter\w*)\b", re.I
)


def is_soft(instruct="", text=""):
    """True when a line's direction or inline tag calls for whispered/quiet delivery."""
    if instruct and SOFT_DELIVERY.search(instruct):
        return True
    if text:
        for m in re.finditer(r"\[([^\]]+)\]", text):
            if SOFT_DELIVERY.search(m.group(1)):
                return True
    return False


def level(wav, target_db=-20.0, ceiling=0.89, soft=False):
    """Bring a line's speech to the same loudness as every other line (training clips were leveled to -20 dB),
    so cast members don't jump in volume. Voices ranged from -20 dB to -28 dB (stock Tara).
    When soft=True (whispered/quiet delivery), target -26 dB and cap boost at 2.5x so whispers stay hushed
    instead of being blown up to narrator volume."""
    if len(wav) < 2400:
        return wav
    hop = 1200  # 50 ms at 24 kHz
    frames = len(wav) // hop
    rms = np.sqrt((wav[:frames * hop].reshape(frames, hop) ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms)
    active = rms[db > db.max() - 30]
    if not len(active):
        return wav
    now = 20 * np.log10(np.sqrt(np.mean(active ** 2)))
    goal = target_db - 6.0 if soft else target_db
    max_gain = 2.5 if soft else 6.0
    out = wav * float(np.clip(10 ** ((goal - now) / 20), 0.25, max_gain))
    over = np.abs(out) > ceiling
    if over.any():
        a = np.abs(out[over])
        out[over] = np.sign(out[over]) * (ceiling + (1 - ceiling) * np.tanh((a - ceiling) / (1 - ceiling)))
    return out.astype(np.float32)


def highpass(wav, hz=80.0, sr=SR):
    """4th-order Butterworth high-pass filter to remove sub-vocal rumble and DC offset."""
    a = np.asarray(wav, dtype=np.float32)
    if len(a) < 64:
        return a
    from scipy.signal import butter, sosfilt
    sos = butter(4, hz / (sr * 0.5), btype="highpass", output="sos")
    return sosfilt(sos, a).astype(np.float32)


def deess(wav, low_hz=5500.0, high_hz=9000.0, threshold_db=-24.0, ratio=3.0, sr=SR):
    """Gentle split-band dynamic de-esser targeting harsh neural-TTS sibilance (5.5-9 kHz)."""
    a = np.asarray(wav, dtype=np.float32).reshape(-1)
    if len(a) < 256 or high_hz >= sr * 0.49:
        return a
    from scipy.signal import butter, sosfilt
    sos = butter(2, [low_hz / (sr * 0.5), high_hz / (sr * 0.5)], btype="bandpass", output="sos")
    sib = sosfilt(sos, a)
    hop = max(1, int(sr * 0.005))
    env = np.sqrt(np.convolve(sib ** 2, np.ones(hop, dtype=np.float32) / hop, mode="same") + 1e-12)
    env_db = 20.0 * np.log10(env)
    over = np.maximum(0.0, env_db - threshold_db)
    gain = 10.0 ** (-(over * (1.0 - 1.0 / ratio)) / 20.0)
    return (a - sib * (1.0 - gain)).astype(np.float32)


def room_profile(wav, sr=SR, n_fft=2048):
    """Compute a smoothed log-magnitude spectral profile (LTAS) for Same-Room acoustic matching."""
    a = np.asarray(wav, dtype=np.float32).reshape(-1)
    if len(a) < n_fft:
        return None
    hop = n_fft // 2
    frames = max(1, (len(a) - n_fft) // hop + 1)
    win = np.hanning(n_fft).astype(np.float32)
    acc = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    count = 0
    for i in range(frames):
        seg = a[i * hop:i * hop + n_fft]
        if float(np.sqrt(np.mean(seg ** 2))) < 10 ** (-45 / 20):
            continue
        acc += np.abs(np.fft.rfft(seg * win)) ** 2
        count += 1
    if count == 0:
        return None
    spec_db = 10.0 * np.log10(acc / count + 1e-12)
    # 15-bin moving average smooths narrow harmonic peaks into broad booth/mic coloration
    k = np.ones(15, dtype=np.float64) / 15.0
    return np.convolve(spec_db, k, mode="same").astype(np.float32)


def match_room(wav, target_profile, max_db=4.0, sr=SR, n_fft=2048):
    """Apply a gentle clamped match-EQ (±max_db) so a secondary voice sits in the same room as target_profile."""
    a = np.asarray(wav, dtype=np.float32).reshape(-1)
    if target_profile is None or len(a) < n_fft:
        return a
    src = room_profile(a, sr=sr, n_fft=n_fft)
    if src is None or len(src) != len(target_profile):
        return a
    diff_db = np.clip(target_profile - src, -max_db, max_db)
    # Keep DC and Nyquist flat so we don't boost sub-rumble or brickwall hiss
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    taper = np.where((freqs < 120.0) | (freqs > min(sr * 0.45, 9500.0)), 0.0, 1.0)
    gain = (10.0 ** ((diff_db * taper) / 20.0)).astype(np.float32)
    # Zero-phase FIR via windowed inverse FFT
    ir = np.fft.irfft(gain, n=n_fft).real
    ir = np.fft.fftshift(ir) * np.hanning(n_fft)
    from scipy.signal import oaconvolve
    return oaconvolve(a, ir.astype(np.float32), mode="same").astype(np.float32)

