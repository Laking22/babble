"""Acoustic Space & Room Presets: lightweight synthetic impulse-response convolution so dry TTS voices
and SFX share the same physical environment (Cavern, Stone Hall, Tavern, Forest, Helmet, Radio, Behind Door).
"""
import numpy as np

from .runtime import SR

PRESETS = {
    "cavern": {"label": "Cavern", "rt60": 2.4, "wet": 0.28, "lp_hz": 3200.0, "hp_hz": 120.0, "pre_ms": 22.0},
    "stone_hall": {"label": "Stone Hall", "rt60": 1.4, "wet": 0.20, "lp_hz": 4800.0, "hp_hz": 100.0, "pre_ms": 14.0},
    "tavern": {"label": "Tavern", "rt60": 0.55, "wet": 0.14, "lp_hz": 4200.0, "hp_hz": 110.0, "pre_ms": 8.0},
    "forest": {"label": "Forest", "rt60": 0.35, "wet": 0.08, "lp_hz": 3600.0, "hp_hz": 140.0, "pre_ms": 18.0},
    "helmet": {"label": "Inside Helmet", "rt60": 0.14, "wet": 0.35, "lp_hz": 2400.0, "hp_hz": 320.0, "pre_ms": 2.0, "bandpass": True},
    "radio": {"label": "Telephone / Radio", "rt60": 0.06, "wet": 0.25, "lp_hz": 3100.0, "hp_hz": 420.0, "pre_ms": 1.0, "bandpass": True},
    "behind_door": {"label": "Behind Closed Door", "rt60": 0.40, "wet": 0.22, "lp_hz": 900.0, "hp_hz": 80.0, "pre_ms": 4.0, "bandpass": True},
}
_IR_CACHE = {}


def _ir(preset_id, sr=SR):
    key = (preset_id, int(sr))
    if key in _IR_CACHE:
        return _IR_CACHE[key]
    p = PRESETS[preset_id]
    n = max(64, int(sr * min(p["rt60"], 3.0)))
    rng = np.random.default_rng(abs(hash(preset_id)) % (2 ** 32))
    t = np.arange(n, dtype=np.float32) / sr
    decay = np.exp(-6.9077 * t / max(p["rt60"], 0.05)).astype(np.float32)
    ir = rng.standard_normal(n).astype(np.float32) * decay
    from scipy.signal import butter, sosfilt
    lp = min(p["lp_hz"], sr * 0.45)
    hp = min(p["hp_hz"], lp * 0.5)
    ir = sosfilt(butter(2, [hp / (sr * 0.5), lp / (sr * 0.5)], btype="bandpass", output="sos"), ir)
    pre = int(sr * p["pre_ms"] / 1000.0)
    if pre > 0:
        ir = np.concatenate([np.zeros(pre, dtype=np.float32), ir])
    rms = float(np.sqrt(np.mean(ir ** 2)) + 1e-12)
    ir = (ir / (rms * np.sqrt(len(ir)))).astype(np.float32)
    _IR_CACHE[key] = ir
    return ir


def apply_space(wav, preset_id, wet=None, sr=SR):
    """Apply an acoustic space preset to mono or stereo float32 audio using FFT overlap-add convolution."""
    if not preset_id or preset_id not in PRESETS:
        return np.asarray(wav, dtype=np.float32)
    a = np.asarray(wav, dtype=np.float32)
    if a.size == 0:
        return a
    p = PRESETS[preset_id]
    mix = float(p["wet"] if wet is None else wet)
    ir = _ir(preset_id, sr=sr)
    from scipy.signal import butter, oaconvolve, sosfilt

    def proc_1d(x):
        dry = x
        if p.get("bandpass"):
            lp = min(p["lp_hz"], sr * 0.45)
            hp = min(p["hp_hz"], lp * 0.5)
            dry = sosfilt(butter(2, [hp / (sr * 0.5), lp / (sr * 0.5)], btype="bandpass", output="sos"), x).astype(np.float32)
        rev = oaconvolve(dry, ir, mode="full")[:len(x)].astype(np.float32)
        return ((1.0 - mix) * dry + mix * rev).astype(np.float32)

    if a.ndim == 1:
        return proc_1d(a)
    return np.stack([proc_1d(a[:, ch]) for ch in range(a.shape[1])], axis=1).astype(np.float32)
