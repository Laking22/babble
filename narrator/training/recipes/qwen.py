"""Qwen3-TTS 1.7B fine-tune (LoRA on a 16 GB GPU; several voices per model, each with its own speaker id).

Local: encode every clip to Qwen's 12 Hz audio codes with MLX (the Base model's speech tokenizer, kept at
full precision), so only codes, one reference clip per voice and a few check clips are uploaded.
GPU: qwen_train.py (adapted from Qwen's official sft_12hz.py). Install: convert the merged CustomVoice model
to 6-bit MLX; its voices show up in the Qwen engine.
"""
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from ...paths import MODELS, WORK
from ..base import Package, Recipe, run_key, runs, save_run

HERE = Path(__file__).resolve().parent
CODES = WORK / "codes" / "qwen"
ADAPTERS = WORK / "adapters" / "qwen"
PACKAGES = WORK / "packages"
CHECK_CLIPS = 12


def _clean(text):
    import re
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def encode(v, log=print):
    """Qwen 12 Hz codes for every clip of one voice (cached by v.key, resumable). Returns the .jsonl path."""
    out = CODES / f"{v.key}.jsonl"
    meta = Path(v.dataset) / "metadata.jsonl" if v.dataset else None
    if not (meta and meta.exists()):
        if out.exists():
            return out
        raise ValueError(f"No processed dataset for '{v.key}'. Let the Train tab finish it first.")
    import mlx.core as mx
    from ...audio import read_mono
    from ...runtime import on_mlx_thread
    from ...tts.qwen import model as qwen_model

    rows = [json.loads(line) for line in open(meta, encoding="utf-8") if line.strip()]
    key = lambda r: (r["audio"], r["text"], round(float(r.get("duration") or 0), 1))
    done = {}
    if out.exists():
        for line in open(out, encoding="utf-8"):
            r = json.loads(line)
            done[key(r)] = r
    wanted = {key(r) for r in rows}
    todo = [r for r in rows if key(r) not in done]
    log(f"{v.key}: {len(rows)} clips, {len(rows) - len(todo)} already encoded, {len(todo)} to go")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")

    def job():
        st = qwen_model("clone").speech_tokenizer if todo else None
        started, secs = time.time(), 0.0
        with open(tmp, "w", encoding="utf-8") as f:
            for k, r in done.items():
                if k in wanted:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            for n, r in enumerate(todo, 1):
                audio = read_mono(Path(v.dataset) / r["audio"])
                codes = st.encode(mx.array(audio)[None, None])
                c = np.array(codes)[0].T.astype(int).tolist()     # [frames][16]
                secs += len(audio) / 24000
                f.write(json.dumps({"audio": r["audio"], "text": r["text"], "duration": r.get("duration", len(audio) / 24000),
                                    "audio_codes": c}, ensure_ascii=False) + "\n")
                if n % 200 == 0 or n == len(todo):
                    mx.clear_cache()
                    log(f"  {n}/{len(todo)} clips ({secs / max(time.time() - started, 1e-6):.1f}x real time)")
        tmp.replace(out)
    on_mlx_thread(job)()
    return out


class QwenRecipe(Recipe):
    id, name, engine, readiness = "qwen", "Qwen3-TTS 1.7B LoRA", "qwen", "ready"
    clip_seconds = (2.0, 15.0)
    # per epoch on a T4: measured 0.035 (1.1 s a step, batch 4), with room for longer clips and setup
    gpu_hours_per_audio_hour = 0.05

    def prepare(self, voices, model, *, start_from=None, epochs=3.0, max_clips=0, log=print, **_):
        if start_from:
            raise ValueError("Updating a Qwen model with new voices isn't supported yet; train them together.")
        model = _clean(model)
        speakers = [v.speaker for v in voices]
        if len(set(speakers)) != len(speakers):
            raise ValueError("Each voice in a model needs its own speaker name.")
        if len(speakers) > self.max_voices:
            log(f"Note: more than {self.max_voices} voices in one model raises the risk of voices bleeding together.")
        version = (runs().get(run_key(self.id, model), {}).get("version") or 0) + 1
        pkg = PACKAGES / f"{model}-qwen"
        shutil.rmtree(pkg, ignore_errors=True)
        data = pkg / "dataset"
        (data / "refs").mkdir(parents=True)
        rows = 0
        hours = 0.0
        check = []
        with open(data / "qwen-codes.jsonl", "w", encoding="utf-8") as f:
            for v in voices:
                ref = Path(v.dataset or "") / "ref.wav"
                if not ref.exists():
                    raise ValueError(f"'{v.key}' has no ref.wav (the Train tab writes one). Rebuild its dataset.")
                info = sf.info(str(ref))
                if info.samplerate != 24000:
                    raise ValueError(f"{ref} must be 24 kHz (it's {info.samplerate}).")
                shutil.copy2(ref, data / "refs" / f"{v.speaker}.wav")
                lines = [json.loads(line) for line in open(encode(v, log), encoding="utf-8")]
                for r in lines:
                    f.write(json.dumps({"speaker": v.speaker, "text": r["text"], "duration": r["duration"],
                                        "audio_codes": r["audio_codes"]}, ensure_ascii=False) + "\n")
                    rows += 1
                    hours += r["duration"] / 3600
                if v.dataset:  # a few clips for the official-tokenizer check on the GPU
                    (data / "check" / v.speaker).mkdir(parents=True, exist_ok=True)
                    for r in random.Random(0).sample(lines, min(CHECK_CLIPS, len(lines))):
                        dst = Path("check") / v.speaker / Path(r["audio"]).name
                        shutil.copy2(Path(v.dataset) / r["audio"], data / dst)
                        check.append({"wav": str(dst), "audio_codes": r["audio_codes"]})
        if check:
            with open(data / "check.jsonl", "w", encoding="utf-8") as f:
                for c in check:
                    f.write(json.dumps(c) + "\n")
        shutil.copy2(HERE / "qwen_train.py", data / "train_qwen.py")
        used = min(hours, hours * max_clips / rows) if max_clips else hours
        args = f"--model {model} --epochs {epochs} --max-hours 11" + (f" --max-clips {max_clips}" if max_clips else "")
        return Package(model=model, recipe=self.id, data_dir=data, script="train_qwen.py", args=args,
                       pip=["qwen-tts", "librosa", "soundfile"], gpu="T4", hours=round(hours, 2),
                       gpu_hours=round(used * epochs * self.gpu_hours_per_audio_hour + 0.5, 1),
                       small_outputs=r"(samples/.*\.wav|.*\.log)$", big_outputs=rf"^({model}-qwen|{model}-lora)/",
                       meta={"voices": {v.key: v.speaker for v in voices}, "clips": rows, "version": version,
                             "epochs": epochs, "max_clips": max_clips})

    def after_fetch(self, run, results, log=print):
        model = run["model"]
        lora = Path(results) / f"{model}-lora"
        if (lora / "adapter.pt").exists():
            vid = f"{model}-v{run.get('version') or 1}"
            shutil.copytree(lora, ADAPTERS / vid, dirs_exist_ok=True)
            save_run(run_key(self.id, model), chain=[vid])
            log(f"Saved this version's adapter as {vid}.")

    def install(self, run, results, genders=None, names=None, log=print):
        model = run["model"]
        src = Path(results) / f"{model}-qwen"
        if not (src / "config.json").exists():
            raise ValueError(f"No merged model in {src}. Fetch the results first.")
        dest = MODELS / f"{model}-qwen-6bit"
        shutil.rmtree(dest, ignore_errors=True)
        log(f"Converting {model} to 6-bit MLX (a minute or two)…")
        subprocess.run([sys.executable, "-W", "ignore", "-m", "mlx_audio.convert", "--hf-path", str(src),
                        "--mlx-path", str(dest), "-q", "--q-bits", "6", "--q-group-size", "64", "--model-domain", "tts"],
                       check=True)
        cfg = json.loads((dest / "config.json").read_text())
        speakers = list((cfg.get("talker_config", {}).get("spk_id") or {}).keys())
        genders, names = genders or {}, names or {}
        voices = {s: {"name": names.get(s) or s.title(), "gender": genders.get(s, "unspecified"),
                      "description": "fine-tuned"} for s in speakers}
        (dest / "voice.json").write_text(json.dumps({"engine": "qwen", "voices": voices, "hours": run.get("hours"),
                                                     "trained_on": run.get("backend"), "url": run.get("url")}, indent=1))
        log(f"Installed {', '.join(x['name'] for x in voices.values())} in {dest}.")
        return dest
