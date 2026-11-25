"""Setting up Babble's Python on Windows and Linux. Standard library only: it runs before anything else is installed.

    python -m narrator.pc_setup base --uv UV     the app's packages (the app runs this at first start and after updates)
    python -m narrator.pc_setup torch --uv UV    PyTorch for the voices, built for this computer's graphics card
    python -m narrator.pc_setup mac --uv UV      the released Mac app's packages (requirements.txt as it is: MLX)

`base` installs requirements.txt without its Mac-only lines (MLX, and the transformers MLX needs) plus
requirements-pc.txt, the voice engines' packages. PyTorch is separate because it's large (about 0.3 GB for the processor,
about 3 GB with an NVIDIA card): it's set up when the owner asks, from the Models page (api_manage /voices-setup).
The Mac's voices run on MLX: `mac` installs requirements.txt as it is, plus spaCy's English model for Kokoro (from
requirements-pc.txt; without it Kokoro would try to download it with pip, which a uv Python hasn't got).
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TORCH = ("torch", "torchaudio")
TORCH_VERSION = "2.11.0"
INDEX = "https://download.pytorch.org/whl/{build}"
SIZE_GB = {"cu128": 3.0, "cu126": 3.0, "cpu": 0.3}   # about, for the Set up button (Linux CUDA includes NVIDIA's libraries)
MAC_ONLY = ("mlx", "transformers")                   # requirements.txt lines left out here
NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def _lines(path):
    out = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            out.append((line, "no-deps" in raw.partition("#")[2]))
    return out


def requirements(root=ROOT):
    """(packages, packages installed without their own dependency lists) for Windows and Linux."""
    base = [(r, n) for r, n in _lines(Path(root) / "requirements.txt")
            if not r.lower().replace("_", "-").startswith(MAC_ONLY)]
    pc = _lines(Path(root) / "requirements-pc.txt")
    rows = base + pc
    return [r for r, n in rows if not n], [r for r, n in rows if n]


def stamp(root=ROOT):
    """What the installed packages were made from: when it changes (an update), `base` runs again."""
    return "\n".join(Path(root, f).read_text(encoding="utf-8") for f in ("requirements.txt", "requirements-pc.txt"))


def _run(cmd, log):
    log("$ " + " ".join(str(c) for c in cmd))
    p = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", creationflags=NO_WINDOW)
    for line in p.stdout:
        if line.strip():
            log(line.rstrip())
    if p.wait():
        raise RuntimeError(f"{Path(str(cmd[0])).name} {cmd[1]} {cmd[2]} failed (exit {p.returncode}).")


def install_base(uv, python=sys.executable, root=ROOT, log=print):
    normal, nodeps = requirements(root)
    _run([uv, "pip", "install", "--python", python, *normal], log)
    if nodeps:
        _run([uv, "pip", "install", "--python", python, "--no-deps", *nodeps], log)


def install_mac(uv, python=sys.executable, root=ROOT, log=print):
    rows = _lines(Path(root) / "requirements.txt")
    rows += [(r, n) for r, n in _lines(Path(root) / "requirements-pc.txt") if r.lower().startswith("en_core_web_sm")]
    _run([uv, "pip", "install", "--python", python, *[r for r, n in rows if not n]], log)
    nodeps = [r for r, n in rows if n]
    if nodeps:
        _run([uv, "pip", "install", "--python", python, "--no-deps", *nodeps], log)


def gpu():
    """The NVIDIA graphics card PyTorch would use: {"name", "cap"} (compute capability), or None."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run([smi, "--query-gpu=name,compute_cap", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=20, creationflags=NO_WINDOW).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    cards = []
    for row in out.splitlines():
        name, _, cap = row.rpartition(",")
        try:
            cards.append({"name": name.strip(), "cap": float(cap)})
        except ValueError:
            continue
    return max(cards, key=lambda c: c["cap"]) if cards else None


def build(card="auto"):
    """Which PyTorch build: CUDA 12.8 for RTX 20-series cards and newer (the only one with RTX 50-series support),
    CUDA 12.6 for older NVIDIA cards (12.8 left them out), else the processor."""
    card = gpu() if card == "auto" else card
    if not card:
        return "cpu"
    return "cu128" if card["cap"] >= 7.5 else "cu126"


def torch_packages(which):
    return [f"{p}=={TORCH_VERSION}+{which}" for p in TORCH]


def install_torch(uv, python=sys.executable, which=None, log=print):
    which = which or build()
    _run([uv, "pip", "install", "--python", python, *torch_packages(which), "--index-url", INDEX.format(build=which)],
         log)
    return which


def find_uv(home=None):
    """uv: the one the app downloaded (data folder tools/uv), BABBLE_UV, or one on the PATH."""
    if os.environ.get("BABBLE_UV"):
        return os.environ["BABBLE_UV"]
    if home:
        exe = Path(home) / "tools" / "uv" / ("uv.exe" if sys.platform == "win32" else "uv")
        if exe.is_file():
            return str(exe)
    return shutil.which("uv")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m narrator.pc_setup")
    ap.add_argument("what", choices=["base", "torch", "mac"])
    ap.add_argument("--uv", default=None)
    ap.add_argument("--build", default=None, choices=sorted(SIZE_GB))
    a = ap.parse_args(argv)
    uv = a.uv or find_uv()
    if not uv:
        sys.exit("uv wasn't found (pass --uv).")
    if a.what == "base":
        install_base(uv)
    elif a.what == "mac":
        install_mac(uv)
    else:
        print(f"PyTorch {TORCH_VERSION} ({install_torch(uv, which=a.build)}) is set up.")


if __name__ == "__main__":
    main()
