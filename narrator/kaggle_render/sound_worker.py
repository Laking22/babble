"""Make a book's missing sounds on a cloud GPU, for lines_runner.py (Create in the cloud, "Make sounds in the cloud").

The same generator the Mac uses: Stability's Stable Audio 3 MLX script (vendor/stable_audio_3, sent with the job)
with the same weights (stabilityai/stable-audio-3-optimized, not gated), on MLX's CUDA build. Each sound is made
in TAKES with different seeds; the Mac keeps the one CLAP says sounds most like its description, trims and
resamples it exactly as it does its own, and adds it to the library. If this GPU can't run MLX, the sounds are left
and the Mac makes them when the book is finished.

    python sound_worker.py <sounds.json> <out folder> <stop at (unix time)> <stable_audio_3 folder>

sounds.json: [{"key", "prompt", "kind", "seconds", "model": "sm-sfx" | "sm-music" | "medium", "takes"}].
Writes <out>/<key>-<take>.wav (44.1 kHz stereo, as the script makes it). Prints "MADE <key> <seconds>" per take,
"FAILED <key> <why>", and "STOPPED" at the time limit.
"""
import json
import os
import subprocess
import sys
import time
import zlib
from pathlib import Path

PACKAGES = ["mlx[cuda]", "sentencepiece", "tqdm", "huggingface_hub", "safetensors"]
DECODER = {"sm-sfx": "same-s", "sm-music": "same-s", "medium": "same-l"}
WEIGHTS = "/tmp/sa3-weights"


def setup(log=print):
    """Install MLX's CUDA build into this Python, and check it can use the GPU."""
    log("Installing the sound generator (MLX for CUDA)…")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PACKAGES], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError((r.stderr or r.stdout)[-400:])
    r = subprocess.run([sys.executable, "-c", "import mlx.core as mx; a = mx.ones((256, 256)); mx.eval(a @ a); "
                        "print(mx.default_device())"], capture_output=True, text=True, timeout=600)
    if r.returncode or "gpu" not in (r.stdout or "").lower():
        raise RuntimeError("MLX can't use this GPU: " + ((r.stderr or r.stdout) or "")[-300:])
    log(f"Sound generator ready ({r.stdout.strip()}).")


def real_copy(code):
    """The generator's code as real files in a folder of its own. On Kaggle the job's files are links into one flat
    folder, and the script finds its own modules (models/…) next to where its file really is: there, they aren't."""
    import shutil
    dst = Path("/tmp/sa3-code")
    if not (dst / "scripts" / "sa3_mlx.py").exists():
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(code, dst, symlinks=False)  # (links are followed: their contents are copied)
    return dst


def main():
    jobs = json.loads(Path(sys.argv[1]).read_text())
    out, stop_at, code = Path(sys.argv[2]).resolve(), float(sys.argv[3]), real_copy(Path(sys.argv[4]))
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, SA3_HOME=WEIGHTS)
    Path(WEIGHTS).mkdir(parents=True, exist_ok=True)
    for j in jobs:
        for take in range(int(j.get("takes") or 1)):
            if time.time() > stop_at:
                print("STOPPED", flush=True)
                return
            dst = out / f"{j['key']}-{take}.wav"
            if dst.exists():
                continue
            part = out / f".{j['key']}-{take}.part.wav"
            args = [sys.executable, str(code / "scripts" / "sa3_mlx.py"), "--prompt", j["prompt"], "--dit", j["model"],
                    "--decoder", DECODER[j["model"]], "--seconds", f"{float(j['seconds']):g}", "--out", str(part),
                    "--seed", str(1000 + 7919 * take + zlib.crc32(j["key"].encode()) % 997)]
            t0 = time.time()
            r = subprocess.run(args, cwd=code, env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL)
            if r.returncode or not part.exists():
                tail = " / ".join((r.stderr or r.stdout or "").strip().splitlines()[-3:])
                print(f"FAILED {j['key']} {tail[:300]}", flush=True)
                part.unlink(missing_ok=True)
                break  # (the other takes would fail the same way)
            os.replace(part, dst)
            print(f"MADE {j['key']} {time.time() - t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
