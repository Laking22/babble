"""PyTorch for the voice engines on Windows and Linux. On a Mac the engines use MLX (runtime.py); everywhere else the
same models run through PyTorch, the way the cloud scripts already run them on NVIDIA GPUs (kaggle_render/runner.py,
lines_runner.py): Qwen3-TTS through its official package, Orpheus through transformers + SNAC, Kokoro through its own
package.

PyTorch isn't part of the first install (0.2–2.6 GB): torch_setup.py adds it from the Models page.

Trained voice packs are kept as the Mac makes them (MLX, 4- or 6-bit). `unpacked` turns one back into an ordinary
16-bit model once, in the data folder, with the cloud's own unpacker (kaggle_render/unpack_mlx.py) and the files
training doesn't change from the stock model, exactly as Create in the cloud does: so a pack sounds the same here.
"""
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path

from ..paths import HOME

CACHE = HOME / "cache" / "torch-models"
WEIGHTS = re.compile(r"(model|pytorch_model)\S*\.(safetensors|bin)(\.index\.json)?|\S+\.(pt|pth|ckpt)")
_dev = {}


def missing(*packages):
    """Which of these Python packages (and PyTorch) aren't installed."""
    importlib.invalidate_caches()
    return [p for p in ("torch", *packages) if importlib.util.find_spec(p) is None]


def why_not(*packages):
    """"" when PyTorch and these packages are there, else why the engine can't run (shown on the Models page)."""
    if not missing(*packages):
        return ""
    if sys.platform == "darwin":
        return "Local MLX runs on Apple Silicon Macs."
    return "Set up voices on this computer first (Models → Set up voices)."


def device():
    if "d" not in _dev:
        import torch
        _dev["d"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    return _dev["d"]


def dtype():
    """16-bit everywhere (a 3B model in 32-bit doesn't fit most PCs): bfloat16 where the GPU has it, else float16; the
    processor gets bfloat16 too."""
    import torch
    if device() == "cpu" or torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def on_gpu():
    return device() != "cpu"


def note():
    """The heads-up shown with the voices when there's no NVIDIA graphics card."""
    try:
        return "" if on_gpu() else "No NVIDIA graphics card found: voices are made on the processor, which is slow."
    except Exception:  # noqa: BLE001 - PyTorch missing or broken: the engine says why elsewhere
        return ""


def free():
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def seed(n):
    import torch
    torch.manual_seed(int(n))


# ---------- trained voice packs ----------

def packed(folder):
    """Whether a model folder is the Mac's packed (MLX) copy."""
    try:
        c = json.loads((Path(folder) / "config.json").read_text())
    except (OSError, ValueError):
        return False
    return bool(c.get("quantization") or c.get("quantization_config"))


def _stamp(src):
    files = sorted((f.name, f.stat().st_size, int(f.stat().st_mtime)) for f in src.glob("*.safetensors"))
    return hashlib.sha256(json.dumps(files).encode()).hexdigest()


def _place(src, dst):
    """A file or folder at dst, as a hard link where it can be (no extra space), else a copy."""
    if src.is_dir():
        shutil.copytree(src, dst, copy_function=_link_or_copy, dirs_exist_ok=True)
    else:
        _link_or_copy(src, dst)


def _link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def _unpacker():
    path = Path(__file__).resolve().parent.parent / "kaggle_render" / "unpack_mlx.py"
    spec = importlib.util.spec_from_file_location("babble_unpack_mlx", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def unpacked(folder, stock_repo, log=print):
    """A trained voice pack as PyTorch loads it. A folder that isn't packed is used as it is. Otherwise, made once in
    cache/torch-models/<pack>: the pack's weights unpacked to 16-bit and its settings files, plus whatever it doesn't
    have (tokenizer, speech tokenizer…) from the stock model it was trained from, but never the stock model's weights
    (once they were picked over the pack's and every line came out in another voice: lines_runner.qwen_folder)."""
    src = Path(folder)
    if not packed(src):
        return src
    dst = CACHE / src.name
    ready = dst / ".ready"
    stamp = _stamp(src)
    if ready.exists() and ready.read_text() == stamp:
        return dst
    from huggingface_hub import snapshot_download
    log(f"Getting {src.name} ready for this computer (once, a few minutes)…")
    CACHE.mkdir(parents=True, exist_ok=True)
    _unpacker().main(src, dst)
    stock = Path(snapshot_download(stock_repo, ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.pth", "*.md"]))
    for f in stock.iterdir():
        if not WEIGHTS.fullmatch(f.name) and not (dst / f.name).exists():
            _place(f.resolve(), dst / f.name)
    own = {f.name for f in dst.iterdir() if f.name.startswith("model-") and f.suffix == ".safetensors"}
    foreign = [f.name for f in dst.iterdir() if WEIGHTS.fullmatch(f.name) and f.name not in own
               and f.name != "model.safetensors.index.json"]
    if foreign or not own:
        raise RuntimeError(f"{src.name} didn't unpack cleanly ({', '.join(foreign) or 'no weights'})")
    ready.write_text(stamp)
    log(f"{src.name} is ready.")
    return dst
