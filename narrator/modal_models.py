"""Trained voices on Modal (Connections → Modal → Your trained voices): upload a fine-tuned Orpheus model or Qwen
voice pack so Create in the cloud can make its lines on a Modal GPU too. Create in the cloud on Modal uploads the
ones a book needs by itself, once.

The Mac keeps each trained model as an MLX copy (MODELS/<name>-orpheus-4bit, 1.8 GB; MODELS/<name>-qwen-6bit, 2 GB).
Upload sends that copy to the Modal volume (models-mlx/<name>-orpheus), and a small CPU job there unpacks it into an
ordinary 16-bit model (models/<name>-orpheus), which the GPU jobs load. It's the very model the Mac plays, so a voice sounds the same
wherever its lines are made. Uploads are listed in HOME/modal-models.json with a stamp of the Mac's file: a model
installed again since (retrained) shows as changed and isn't used until it's uploaded again.
"""
import json
import os
import re
import shutil
import time
from pathlib import Path

from .paths import HOME, MODELS

REGISTRY = HOME / "modal-models.json"
UNPACK = Path(__file__).resolve().parent / "kaggle_render" / "unpack_mlx.py"
def _read():
    try:
        return json.loads(REGISTRY.read_text())
    except (OSError, ValueError):
        return {}


def _write(reg):
    REGISTRY.write_text(json.dumps(reg, indent=1))


def cloud_name(folder):
    """The model's name in the cloud: myvoice-orpheus-4bit -> myvoice-orpheus (the name Kaggle training gives it too),
    myvoice-qwen-6bit -> myvoice-qwen."""
    return re.sub(r"-\d+bit$", "", Path(folder).name)


def _stamp(folder):
    files = sorted(Path(folder).glob("*.safetensors"))
    return "+".join(f"{f.stat().st_size}-{int(f.stat().st_mtime)}" for f in files)


def installed():
    """The trained models on this Mac (Orpheus and Qwen), with whether each is on Modal."""
    reg = _read()
    out = []
    for meta in sorted(MODELS.glob("*/voice.json")):
        try:
            data = json.loads(meta.read_text())
        except ValueError:
            continue
        if data.get("engine", "orpheus") not in ("orpheus", "qwen") or not any(meta.parent.glob("*.safetensors")):
            continue
        name = cloud_name(meta.parent)
        r = reg.get(name) or {}
        state = ("uploaded" if r.get("stamp") == _stamp(meta.parent) else "changed") if r else "not uploaded"
        out.append({"folder": meta.parent.name, "name": name,
                    "voices": [v.get("name") or k.title() for k, v in data.get("voices", {}).items()],
                    "mb": round(sum(f.stat().st_size for f in meta.parent.glob("*.safetensors")) / 1e6),
                    "state": state, "uploaded": r.get("uploaded")})
    return out


def uploaded():
    """Cloud names of the models whose current version is on Modal."""
    return {m["name"] for m in installed() if m["state"] == "uploaded"}


def _folder(folder):
    p = MODELS / folder
    if not re.fullmatch(r"[\w.-]+", folder or "") or not (p / "voice.json").exists():
        raise ValueError("That trained voice isn't installed on this Mac.")
    return p


def upload(folder, log=print):
    """Upload one trained model to Modal and unpack it there (only the files a GPU needs: a Qwen pack's speech
    tokenizer is the stock one, which the GPU job takes from Qwen's own copy)."""
    from .training.backends.modal_cloud import step
    src = _folder(folder)
    name = cloud_name(src)
    mb = sum(f.stat().st_size for f in src.glob("*.safetensors")) / 1e6
    log(f"Uploading {name} ({mb / 1000:.1f} GB) to your Modal storage…")
    started = time.time()
    step("rm", {"path": f"/models-mlx/{name}"}, timeout=300)
    stage = HOME / "modal-upload" / name  # the model's own files only (links: no extra space)
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    try:
        for f in src.iterdir():
            if f.is_file() and f.suffix in (".json", ".safetensors", ".jinja", ".txt"):
                try:
                    os.link(f, stage / f.name)
                except OSError:
                    shutil.copy2(f, stage / f.name)
        step("put", {"local": str(stage), "remote": f"/models-mlx/{name}"}, log=log, timeout=4 * 3600)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    took = time.time() - started
    log(f"Uploaded in {f'{took / 60:.0f} min' if took >= 90 else f'{took:.0f} s'}. Unpacking it on Modal (a minute or two, a cent or so)…")
    try:
        a = step("unpack", {"name": name, "script": str(UNPACK)}, log=log, timeout=3600)
    except RuntimeError as e:
        raise ValueError(f"Unpacking on Modal didn't work: {str(e)[-500:]}") from e
    step("rm", {"path": f"/models-mlx/{name}"}, timeout=300)  # the packed copy isn't needed now
    reg = _read()
    reg[name] = {"folder": src.name, "stamp": _stamp(src), "uploaded": time.time()}
    _write(reg)
    done = next((ln for ln in (a.get("out") or "").splitlines() if ln.startswith("DONE:")), "")
    log(f"Done · {name} is on Modal: Create in the cloud on Modal now makes its voices there. ({done[6:].split('→')[0].strip()})")
    return name


def remove(folder, log=print):
    """Take a trained model off Modal (the Mac's copy stays)."""
    from .training.backends.modal_cloud import step
    name = cloud_name(_folder(folder))
    step("rm", {"path": f"/models/{name}"}, timeout=300)
    reg = _read()
    reg.pop(name, None)
    _write(reg)
    log(f"Done · {name} is off Modal.")
    return name

