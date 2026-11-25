"""Orpheus 3B LoRA fine-tune (several voices per model, each with its own speaker name).

Local: SNAC-encode the clips (side env with torch + snac). GPU: train_orpheus.py (Unsloth LoRA, merges the
result, writes test samples). Install: convert the merged model to 4-bit MLX and register its voices.
Updating a model (`start_from`) trains on top of its saved adapter chain and mixes in a share of the old
voices' clips so they stay sharp.
"""
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

from ...paths import MODELS, TOOLS, WORK
from ..base import Package, Recipe, run_key, runs, save_run

HERE = Path(__file__).resolve().parent
ENV = TOOLS / "orpheus-encode"            # torch + snac, created on first use
CODES = WORK / "codes" / "orpheus"
ADAPTERS = WORK / "adapters" / "orpheus"
PACKAGES = WORK / "packages"


def _ensure_env(log):
    py = ENV / "bin" / "python"
    if py.exists() and subprocess.run([str(py), "-c", "import torch, snac, soundfile"], capture_output=True).returncode == 0:
        return py
    log("Setting up the Orpheus encoder (one time: torch + SNAC, a few hundred MB)…")
    uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
    subprocess.run([uv, "venv", "-q", "--python", "3.12", str(ENV)], check=True)
    subprocess.run([uv, "pip", "install", "-q", "--python", str(py), "torch", "snac", "einops", "soundfile", "numpy"],
                   check=True)
    return py


def encode(v, log=print):
    """Encoded clips for one voice (cached by v.key). Returns the .jsonl path."""
    out = CODES / f"{v.key}.jsonl"
    if not (v.dataset and (Path(v.dataset) / "metadata.jsonl").exists()):
        if out.exists():
            return out  # encoded earlier; the clips themselves aren't needed again
        raise ValueError(f"No processed dataset for '{v.key}' ({v.dataset}). Let the Train tab finish it first.")
    py = _ensure_env(log)
    subprocess.run([str(py), "-W", "ignore", str(HERE / "orpheus_encode.py"), str(v.dataset), str(out)], check=True)
    return out


def _clean(text):
    import re
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


class OrpheusRecipe(Recipe):
    id, name, engine, readiness = "orpheus", "Orpheus 3B LoRA", "orpheus", "ready"
    clip_seconds = (2.0, 20.0)

    def prepare(self, voices, model, *, start_from=None, replay=0.25, epochs=2.0, max_clips=0, log=print, **_):
        model = _clean(model)
        speakers = [v.speaker for v in voices]
        if len(set(speakers)) != len(speakers):
            raise ValueError("Each voice in a model needs its own speaker name.")
        chain, old_voices = [], {}
        if start_from:
            base = runs().get(run_key(self.id, _clean(start_from))) or {}
            chain = list(base.get("chain") or [])
            if not chain:
                raise ValueError(f"'{start_from}' has no saved adapter yet. Fetch its results first.")
            missing = [c for c in chain if not (ADAPTERS / c / "adapter_config.json").exists()]
            if missing:
                raise ValueError(f"Missing saved adapters for '{start_from}': {', '.join(missing)}")
            old_voices = dict(base.get("voices") or {})
        if len(set(speakers) | set(old_voices.values())) > self.max_voices:
            log(f"Note: more than {self.max_voices} voices in one model raises the risk of voices bleeding into each other.")
        version = (runs().get(run_key(self.id, model), {}).get("version") or 0) + 1

        pkg = PACKAGES / model
        shutil.rmtree(pkg, ignore_errors=True)
        data = pkg / "dataset"
        data.mkdir(parents=True)
        rows = 0
        with open(data / "orpheus-codes.jsonl", "w", encoding="utf-8") as f:
            for v in voices:
                for line in open(encode(v, log), encoding="utf-8"):
                    r = json.loads(line)
                    r["voice"] = v.speaker
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    rows += 1
            new_keys = {v.key for v in voices}
            for key, name in old_voices.items():  # keep the earlier model's voices sharp
                if key in new_keys:
                    continue
                src = CODES / f"{key}.jsonl"
                if not src.exists():
                    raise ValueError(f"Encoded clips for '{key}' aren't on this machine, so it can't be kept sharp.")
                old = [line for line in open(src, encoding="utf-8") if line.strip()]
                keep = random.Random(version).sample(old, max(1, int(len(old) * replay)))
                for line in keep:
                    r = json.loads(line)
                    r["voice"] = name
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    rows += 1
                log(f"  + {len(keep):,} of {len(old):,} clips from '{name}' to keep it sharp")
        for c in chain:
            shutil.copytree(ADAPTERS / c, data / "adapters" / c)
        if chain:
            (data / "init.json").write_text(json.dumps({"adapters": chain}))
            log(f"Building on: {' + '.join(chain)}")
        shutil.copy2(HERE / "orpheus_train.py", data / "train_orpheus.py")
        hours = sum(json.loads(line)["duration"] for line in open(data / "orpheus-codes.jsonl", encoding="utf-8")) / 3600
        args = f"--voice {model} --max-hours 11 --epochs {epochs}"  # stops early if held-out loss rises
        return Package(model=model, recipe=self.id, data_dir=data, script="train_orpheus.py", args=args,
                       pip=["unsloth", "snac"], gpu="T4", hours=round(hours, 2),
                       gpu_hours=round(max(0.5, hours * epochs * self.gpu_hours_per_audio_hour) + 0.3, 1),
                       small_outputs=r"(samples/.*\.wav|.*\.log)$", big_outputs=rf"^({model}-orpheus|{model}-lora)/",
                       meta={"voices": {**old_voices, **{v.key: v.speaker for v in voices}}, "clips": rows,
                             "version": version, "start_from": start_from, "chain_base": chain,
                             "max_clips": max_clips})

    def after_fetch(self, run, results, log=print):
        model = run["model"]
        lora = Path(results) / f"{model}-lora"
        if (lora / "adapter_config.json").exists():
            vid = f"{model}-v{run.get('version') or 1}"
            shutil.copytree(lora, ADAPTERS / vid, dirs_exist_ok=True)
            save_run(run_key(self.id, model), chain=list(run.get("chain_base") or []) + [vid])
            log(f"Saved this version's adapter as {vid} (later updates build on it).")

    def install(self, run, results, genders=None, names=None, log=print):
        """genders / names: {speaker: value}. Converts to 4-bit MLX in paths.MODELS and writes voice.json."""
        model = run["model"]
        src = Path(results) / f"{model}-orpheus"
        if not (src / "config.json").exists():
            raise ValueError(f"No merged model in {src}. Fetch the results first.")
        dest = MODELS / f"{model}-orpheus-4bit"
        shutil.rmtree(dest, ignore_errors=True)
        if (src / "mlx.json").exists():  # trained on this Mac: already a 4-bit MLX model
            shutil.copytree(src, dest)
            return self._register(run, dest, genders, names, log)
        log(f"Converting {model} to 4-bit (about a minute)…")
        subprocess.run([sys.executable, "-W", "ignore", "-m", "mlx_audio.convert", "--hf-path", str(src),
                        "--mlx-path", str(dest), "-q", "--q-bits", "4", "--q-group-size", "64", "--model-domain", "tts"],
                       check=True)
        cfg = json.loads((dest / "config.json").read_text())
        # Newer transformers saves position settings as rope_parameters; MLX's Orpheus reads rope_theta/rope_scaling.
        # Without this the voice starts each line fine and then turns to gibberish.
        rp = cfg.get("rope_parameters") or {}
        if rp and not cfg.get("rope_scaling"):
            cfg["rope_theta"] = rp.get("rope_theta", 500000.0)
            cfg["rope_scaling"] = {k: v for k, v in rp.items() if k != "rope_theta"}
        cfg["tokenizer_name"] = str(dest)
        (dest / "config.json").write_text(json.dumps(cfg, indent=2))
        return self._register(run, dest, genders, names, log)

    def _register(self, run, dest, genders, names, log):
        model = run["model"]
        cfg = json.loads((dest / "config.json").read_text())
        cfg["tokenizer_name"] = str(dest)
        (dest / "config.json").write_text(json.dumps(cfg, indent=2))
        speakers = sorted(set((run.get("voices") or {model: model}).values()))
        genders, names = genders or {}, names or {}
        voices = {s: {"name": names.get(s) or s.title(), "gender": genders.get(s, "unspecified"),
                      "description": "fine-tuned"} for s in speakers}
        (dest / "voice.json").write_text(json.dumps({"engine": "orpheus", "voices": voices, "hours": run.get("hours"),
                                                     "trained_on": run.get("backend"), "url": run.get("url")}, indent=1))
        log(f"Installed {', '.join(x['name'] for x in voices.values())} in {dest}.")
        return dest
