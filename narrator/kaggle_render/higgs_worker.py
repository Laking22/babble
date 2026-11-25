"""Speak Higgs Audio v2 lines on a cloud GPU, for lines_runner.py (Create in the cloud).

Higgs's own code (Boson AI's boson_multimodal) needs an older transformers than Qwen-TTS, so lines_runner makes
every other line first, installs Higgs's packages (setup()) and then runs this file in a fresh Python, which sees
them. Each line clones a saved voice from its sample clip and that clip's transcript (voices/<name>.wav / .txt).

    python higgs_worker.py <jobs.json> <lines folder> <stop at (unix time)> <voices folder>

Prints "MADE <key> <seconds of audio>" or "FAILED <key> <why>" per line, and "STOPPED" at the time limit.
Same model as the Mac's MLX version (higgs-audio-v2 3B), full size.
"""
import json
import os
import sys
import time
from pathlib import Path

MODEL = "bosonai/higgs-audio-v2-generation-3B-base"
TOKENIZER = "bosonai/higgs-audio-v2-tokenizer"
# The last versions in Boson's original format (July 2025): on 2026-04-04 both repos moved to transformers' own
# Higgs support ("trfms-support"), which Boson's code (below) can't load
MODEL_REV = "10840182ca4ad5d9d9113b60b9bb3c1ef1ba3f84"
TOKENIZER_REV = "9d4988fbd4ad07b4cac3a5fa462741a41810dbec"
SR = 24000
# Boson's example scene for clean narration
SYSTEM = ("Generate audio following instruction.\n\n<|scene_desc_start|>\nAudio is recorded from a quiet room.\n"
          "<|scene_desc_end|>")
# Boson's code, at a fixed commit (their repo has no releases). It's used from the downloaded folder, not pip
# installed: some of its folders (serve/) have no __init__.py, so an install leaves them out
COMMIT = "05a145bb490501b534563bf51bf2f7aa2326b271"
SOURCE = f"https://github.com/boson-ai/higgs-audio/archive/{COMMIT}.zip"
SRC = f"/tmp/higgs-audio-{COMMIT}"  # where it's unpacked
# requirements.txt there, without torch (the image's own) and dev tools
PACKAGES = ["descript-audio-codec", "transformers>=4.45.1,<4.47.0", "librosa", "dacite", "boto3==1.35.36", "s3fs",
            "json_repair", "pandas", "pydantic", "vector_quantize_pytorch", "loguru", "pydub", "omegaconf", "click",
            "langid", "jieba", "accelerate>=0.26.0"]


def setup(log=print):
    """Install Higgs's packages into this Python (after the other lines are made: it changes transformers).
    torchaudio and torchvision are pinned to the installed torch, so pip doesn't swap torch itself."""
    import subprocess
    import torch
    v = torch.__version__.split("+")[0]
    major, minor, patch = (v.split(".") + ["0", "0"])[:3]
    pins = [f"torchaudio=={v}", f"torchvision==0.{int(minor) + 15}.{patch}"] if major == "2" else []
    import io
    import urllib.request
    import zipfile
    log("Downloading Higgs Audio's code…")
    with urllib.request.urlopen(SOURCE, timeout=300) as r:
        tmp_root = Path("/tmp").resolve()
        with zipfile.ZipFile(io.BytesIO(r.read())) as z:
            for m in z.infolist():
                rel = Path(m.filename)
                if rel.is_absolute() or ".." in rel.parts:
                    raise RuntimeError(f"Refused unsafe path in Higgs archive: {m.filename}")
                target = (tmp_root / rel).resolve()
                if tmp_root != target and tmp_root not in target.parents:
                    raise RuntimeError(f"Refused archive path escaping /tmp: {m.filename}")
            z.extractall("/tmp")  # -> /tmp/higgs-audio-<commit>
    if not Path(SRC, "boson_multimodal", "serve", "serve_engine.py").exists():
        raise RuntimeError("Higgs Audio's code isn't where it should be in the download")
    log("Installing Higgs Audio's packages…")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PACKAGES, *pins],
                       capture_output=True, text=True)
    if r.returncode and pins:
        # torchaudio stopped following torch's version numbers after 2.10: with a newer torch there's no exact
        # match, so let pip pick a torch/torchaudio/torchvision set that fits (Higgs runs last, in its own Python)
        log(f"No torchaudio {v}; installing a matching torch set for Higgs…")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PACKAGES, "torchaudio", "torchvision"],
                           capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"pip: {(r.stderr or r.stdout).strip()[-600:]}")


def to_24k(audio, sr):
    import numpy as np
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sr == SR or len(audio) == 0:
        return audio
    return np.interp(np.linspace(0, len(audio), int(len(audio) * SR / sr), endpoint=False),
                     np.arange(len(audio)), audio).astype(np.float32)


def main():
    jobs = json.loads(Path(sys.argv[1]).read_text())
    lines, stop_at, voices = Path(sys.argv[2]), float(sys.argv[3]), Path(sys.argv[4])
    import numpy as np
    import soundfile as sf
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from runner import TRIES, better, calmer, take_problem  # (lines_runner's test of a take, before Boson's code)
    sys.path.insert(0, SRC)  # Boson's code, unpacked by setup()
    from boson_multimodal.data_types import AudioContent, ChatMLSample, Message
    from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
    from huggingface_hub import snapshot_download
    model = snapshot_download(MODEL, revision=MODEL_REV)
    codec = snapshot_download(TOKENIZER, revision=TOKENIZER_REV)
    engine = HiggsAudioServeEngine(model, codec, device="cuda" if torch.cuda.is_available() else "cpu")
    print("LOADED", flush=True)
    for j in jobs:
        if time.time() > stop_at:
            print("STOPPED", flush=True)
            return
        try:
            wav, txt = voices / f"{j['voice']}.wav", voices / f"{j['voice']}.txt"
            if int(j.get("seed") or 0):
                torch.manual_seed(int(j["seed"]))
            messages = [Message(role="system", content=SYSTEM),
                        Message(role="user", content=txt.read_text().strip()),
                        Message(role="assistant", content=AudioContent(audio_url=str(wav))),
                        Message(role="user", content=j["text"])]
            # ~25 audio frames a second: about twice what the line should need (10 s to 80 s), as on the Mac
            frames = min(2000, max(250, int(len(j["text"]) / 14 * 25 * 2)))
            audio, base = None, dict(j, temperature=float(j.get("temperature") or 0.7))
            for attempt in range(TRIES + 1):  # a take silent or cut short is made again, a little calmer each time
                if attempt and time.time() > stop_at:
                    break
                out = engine.generate(chat_ml_sample=ChatMLSample(messages=messages), max_new_tokens=frames,
                                      temperature=calmer(base, attempt)["temperature"],
                                      top_p=0.95, top_k=50, stop_strings=["<|end_of_text|>", "<|eot_id|>"])
                audio = better(audio, to_24k(out.audio, int(out.sampling_rate or SR)), j["text"])
                if not take_problem(audio, j["text"]):
                    break
            problem = take_problem(audio, j["text"])
            if problem not in ("", "cut short"):  # (one still cut short is kept: the Mac decides at Finish)
                raise RuntimeError("no audio came out" if problem in ("none", "broken") else f"it came out {problem}")
            part = lines / f".{j['key']}.part"
            sf.write(part, np.clip(audio, -1, 1), SR, format="FLAC", subtype="PCM_16")
            os.replace(part, lines / f"{j['key']}.flac")
            print(f"MADE {j['key']} {len(audio) / SR:.1f}", flush=True)
        except Exception as e:  # noqa: BLE001 - the next line may still work
            print(f"FAILED {j['key']} {type(e).__name__}: {str(e)[:200]}", flush=True)


if __name__ == "__main__":
    main()
