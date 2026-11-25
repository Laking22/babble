"""Speak Fish Audio Drama 3 lines on a cloud GPU (an H100), for lines_runner.py (Create in the cloud).

Drama 3 (35B mixture of experts, 3B used per step, ~37 GB) runs in vLLM-Omni's speech server, which has its own
PyTorch and transformers, so lines_runner makes every other line first, installs it (setup()) and then runs this
file in a fresh Python. Each line clones a saved voice from its clip and that clip's transcript, and can carry a
direction in brackets ("[whispering] Don't move."). Several lines go at once: that's where Drama 3 is fast.

    python drama_worker.py <jobs.json> <lines folder> <stop at (unix time)> <voices folder>

Prints "MADE <key> <seconds of audio>" or "FAILED <key> <why>" per line, and "STOPPED" at the time limit.
Tested on Modal's H100 (November 2026): about 6.7x real time one line at a time, 25x eight at once.
License: Fish Audio Community License (free commercial use under US$500k a year; not as a TTS service).
"""
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MODEL = "fishaudio/drama-3"
REVISION = "78e5ab79e49aa9a25abe8799eff2c4d73a1c3216"  # the snapshot tested
VLLM = "vllm==0.33.0"
OMNI = "git+https://github.com/vllm-project/vllm-omni.git@4dee28cac13d972ada72b6196a984988ceb3a057"  # (tested)
PORT = 8011
AT_ONCE = 16
SR = 24000


def setup(log=print):
    """Install vLLM and vLLM-Omni into this Python (after the other lines are made: it brings its own torch)."""
    import shutil
    log("Installing Drama 3's speech server (vLLM-Omni)…")
    omni, env = OMNI, dict(os.environ)
    if not shutil.which("git"):  # vLLM-Omni comes from GitHub; the slim images have no git
        subprocess.run("apt-get update -qq && apt-get install -y -qq git", shell=True, capture_output=True)
    if not shutil.which("git"):  # (no root): the same commit as a zip, with the version git would have given it
        omni = f"vllm-omni @ https://github.com/vllm-project/vllm-omni/archive/{OMNI.rsplit('@', 1)[1]}.zip"
        env["SETUPTOOLS_SCM_PRETEND_VERSION"] = "0.33.0rc2.dev214"
    for cmd in ([sys.executable, "-m", "pip", "install", "-q", "uv"],
                [sys.executable, "-m", "uv", "pip", "install", "-q", "--system", "--python", sys.executable, VLLM,
                 "--torch-backend=cu130"],
                [sys.executable, "-m", "uv", "pip", "install", "-q", "--system", "--python", sys.executable, omni,
                 "soundfile", "requests"]):
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode:
            raise RuntimeError(f"{' '.join(cmd[2:5])}: {(r.stderr or r.stdout).strip()[-600:]}")


def to_24k(audio, sr):
    import numpy as np
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sr == SR or len(audio) == 0:
        return audio
    return np.interp(np.linspace(0, len(audio), int(len(audio) * SR / sr), endpoint=False),
                     np.arange(len(audio)), audio).astype(np.float32)


def main():
    import concurrent.futures as cf
    import io
    import numpy as np
    import requests
    import soundfile as sf
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from runner import TRIES, better, take_problem  # (lines_runner's test of a take)
    jobs = json.loads(Path(sys.argv[1]).read_text())
    lines, stop_at, voices = Path(sys.argv[2]), float(sys.argv[3]), Path(sys.argv[4])
    # FlashInfer's sampler compiles itself with nvcc, which the images don't have: vLLM's own sampler instead
    env = dict(os.environ, VLLM_USE_FLASHINFER_SAMPLER="0")
    # (vLLM-Omni's own memory split between its two stages, the speech model and the decoder, as tested: setting
    # --gpu-memory-utilization gives it all to the first and the decoder runs out)
    srv = subprocess.Popen(["vllm", "serve", MODEL, "--revision", REVISION, "--omni", "--port", str(PORT)], env=env)
    try:
        ready = time.time() + 25 * 60
        while time.time() < ready:
            if srv.poll() is not None:
                raise RuntimeError(f"the speech server stopped (exit {srv.returncode})")
            try:
                if requests.get(f"http://localhost:{PORT}/health", timeout=3).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(5)
        else:
            raise RuntimeError("the speech server didn't start in 25 minutes")
        print("LOADED", flush=True)
        refs = {}

        def ref(name):
            if name not in refs:
                refs[name] = ("data:audio/wav;base64," + base64.b64encode((voices / f"{name}.wav").read_bytes()).decode(),
                              (voices / f"{name}.txt").read_text().strip())
            return refs[name]

        def one(j):
            if time.time() > stop_at:
                return j, None, "stopped"
            audio, said = ref(j["voice"])
            best, why = None, ""
            for attempt in range(TRIES + 1):  # a take silent or cut short is asked for again
                if attempt and time.time() > stop_at:
                    break
                try:
                    r = requests.post(f"http://localhost:{PORT}/v1/audio/speech", timeout=600, json={
                        "model": MODEL, "input": j["text"], "voice": "default", "ref_audio": audio, "ref_text": said,
                        "response_format": "wav"})
                    if not r.ok:  # (an error isn't asked again: only a take that came back wrong)
                        why = f"{r.status_code}: {r.text[:200]}"
                        break
                    wav, sr = sf.read(io.BytesIO(r.content), dtype="float32", always_2d=False)
                    best = better(best, to_24k(wav.mean(axis=1) if wav.ndim > 1 else wav, sr), j["text"])
                except Exception as e:  # noqa: BLE001
                    why = f"{type(e).__name__}: {str(e)[:200]}"
                    break
                if not take_problem(best, j["text"]):
                    break
            problem = take_problem(best, j["text"])
            if problem in ("", "cut short"):  # (one still cut short is kept: the Mac decides at Finish)
                return j, best, ""
            return j, None, why or ("no audio came out" if problem in ("none", "broken") else f"it came out {problem}")

        stopped = False
        with cf.ThreadPoolExecutor(AT_ONCE) as ex:
            for j, wav, why in ex.map(one, jobs):
                if why == "stopped":
                    stopped = True
                    continue
                if wav is None:
                    print(f"FAILED {j['key']} {why}", flush=True)
                    continue
                part = lines / f".{j['key']}.part"
                sf.write(part, np.clip(wav, -1, 1), SR, format="FLAC", subtype="PCM_16")
                os.replace(part, lines / f"{j['key']}.flac")
                print(f"MADE {j['key']} {len(wav) / SR:.1f}", flush=True)
        if stopped:
            print("STOPPED", flush=True)
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=30)
        except subprocess.TimeoutExpired:
            srv.kill()


if __name__ == "__main__":
    main()
