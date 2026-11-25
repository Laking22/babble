"""Train on this Mac with MLX (Orpheus only for now). BETA.

Free, private and slow: about 50 s per clip on an 8 GB Mac (it swaps), so a full book takes
days there. On 32 GB+ Max/Ultra chips it is several times faster. Runs as a separate process in the background,
writes its log to train-work/local-runs/<run>/out, and can be resumed (the adapter is saved as it goes).
Don't render big jobs in the app at the same time on 8–16 GB Macs: both need the memory.
"""
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ...paths import WORK
from ..base import Backend, RunStatus

RUNS = WORK / "local-runs"
REPO = Path(__file__).resolve().parents[3]


def _ram_gb():
    try:
        return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 2 ** 30
    except (ValueError, OSError):
        return 0


def _seconds_per_clip():
    ram = _ram_gb()
    return 50 if ram < 12 else 25 if ram < 24 else 8 if ram < 48 else 5


class LocalMac(Backend):
    id, name, readiness = "local-mac", "This Mac", "beta"
    pricing = "free and private; slow (days on 8–16 GB Macs, hours on 32 GB+ Max/Ultra)"

    def check(self):
        if sys.platform != "darwin":
            return {"ok": False, "quota": None, "detail": "Local MLX training requires an Apple Silicon Mac. Use Kaggle or a cloud GPU."}
        ram = _ram_gb()
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        busy = [p for p in RUNS.glob("*/pid") if not (p.parent / "exit_code").exists() and _alive(p)]
        note = " Another training run is already going." if busy else ""
        tip = "" if ram >= 32 else " With this much memory, expect it to take days for a whole book."
        return {"ok": True, "quota": None, "detail": f"{chip or 'Apple Silicon'}, {ram:.0f} GB.{tip}{note} Beta."}

    def estimate(self, package):
        clips = package.meta.get("max_clips") or package.meta.get("clips") or 0
        passes = float(re.search(r"--epochs ([\d.]+)", package.args).group(1)) if "--epochs" in package.args else 1.0
        hours = clips * passes * _seconds_per_clip() / 3600
        return {"gpu_hours": round(hours, 1), "cost": 0.0, "fits": True,
                "note": f"About {hours:.0f} hours on this Mac (rough; it keeps going in the background)."}

    def submit(self, package, log=print):
        if sys.platform != "darwin":
            raise ValueError("Local MLX training is only available on Apple Silicon Macs. Use Kaggle or a cloud GPU.")
        if package.recipe != "orpheus":
            raise ValueError("This Mac can only train Orpheus models for now. Use Kaggle or a GPU machine for Qwen.")
        run_id = f"{package.model}-{package.recipe}-{time.strftime('%Y%m%d-%H%M%S')}"
        rdir = RUNS / run_id
        (rdir / "out").mkdir(parents=True)
        epochs = re.search(r"--epochs ([\d.]+)", package.args)
        extra = (["--epochs", epochs.group(1)] if epochs else []) + \
                (["--max-clips", str(package.meta["max_clips"])] if package.meta.get("max_clips") else [])
        cmd = [sys.executable, "-m", "narrator.training.recipes.orpheus_mlx_train", "--data", str(package.data_dir),
               "--model", package.model, "--out", str(rdir / "out"), *extra]
        (rdir / "run.sh").write_text("#!/bin/bash\n"
                                     f"cd {shlex.quote(str(REPO))}\n"
                                     f"{' '.join(shlex.quote(c) for c in cmd)} > {shlex.quote(str(rdir / 'out' / 'run.log'))} 2>&1\n"
                                     f"echo $? > {shlex.quote(str(rdir / 'exit_code'))}\n")
        p = subprocess.Popen(["bash", str(rdir / "run.sh")], start_new_session=True, env=dict(os.environ),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        (rdir / "pid").write_text(str(p.pid))
        log(f"Training on this Mac in the background ({run_id}). It keeps going if you close this tab.")
        return run_id

    def url(self, run_id):
        return ""

    def status(self, run_id):
        rdir = RUNS / run_id
        if not rdir.exists():
            return RunStatus("unknown", "not found")
        last = ""
        logf = rdir / "out" / "train.log"
        if logf.exists():
            lines = logf.read_text().splitlines()
            last = lines[-1] if lines else ""
        if (rdir / "exit_code").exists():
            code = (rdir / "exit_code").read_text().strip()
            done = "Done. Merged model" in (logf.read_text() if logf.exists() else "")
            if code == "0" and done:
                return RunStatus("complete", last)
            return RunStatus("error" if code != "0" else "stopped", last or f"exit {code}")
        return RunStatus("running" if _alive(rdir / "pid") else "error", last)

    def fetch(self, run_id, dest, pattern, log=print):
        root, dest = RUNS / run_id / "out", Path(dest)
        rx = re.compile(pattern)
        for f in root.rglob("*"):
            rel = f.relative_to(root).as_posix()
            if f.is_file() and rx.search(rel):
                (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(f, dest / rel)
                except OSError:
                    shutil.copy2(f, dest / rel)
        return dest

    def cancel(self, run_id):
        pid = RUNS / run_id / "pid"
        if _alive(pid) and hasattr(os, "killpg"):
            os.killpg(int(pid.read_text()), 15)


def _alive(pid_file):
    try:
        os.kill(int(Path(pid_file).read_text()), 0)
        return True
    except (OSError, ValueError):
        return False
