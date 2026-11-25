"""Speak a book's lines on a cloud GPU, for Babble's Create in the cloud.

Reads lines.json ([{"key", "engine", "text", ...}], written by the app's cloud_book.py) and saves each line as
lines/<key>.flac (24 kHz mono). The app downloads them and puts the book together on the Mac (sound effects,
music, pauses, chapters), so nothing else happens here: no speed changes, no joining.

engine "qwen": a Qwen3-TTS speaker (speaker, instruct); "qwen-clone": a saved voice cloned from voices/<voice>.wav;
"qwen-tuned": a speaker of a trained Qwen voice pack (model["dir"]: sent from the Mac in models-mlx/, or on Modal);
"orpheus": stock Orpheus, or a fine-tuned one when "model" names the notebook it was trained in (attached as an
input; its merged model is the folder model["dir"]).

sounds.json (when the app sends the book's missing sounds too): made after the lines with sound_worker.py, into
sounds/<key>-<take>.wav; the Mac picks the best take of each. A GPU that can't run them leaves them for the Mac.

With two GPUs, each GPU gets every other line (in a process of its own). Stops starting new lines after
--max-hours, packs what it made into lines.tar (unless --no-pack) and writes progress.json either way. Each line is
written under a temporary name and renamed when it's whole, so lines can be copied back while it works.

Usage:  python lines_runner.py [--max-hours 11] [--out /kaggle/working]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import (QWEN_CUSTOM, SR, TRIES, Orpheus, Qwen, better, calmer, clock,  # noqa: E402
                    rushed_keys, take_problem, voiced_seconds)

BATCH = {"qwen": 12, "qwen-clone": 12, "qwen-tuned": 12, "orpheus": 12}  # lines at once on a 16 GB T4 (fp16 model + room for the longest lines)
MARGIN_HOURS = 0.15  # left for packing and saving
LOGFILE = None  # run.log, when this process writes it (a GPU process's lines go through the main one)
DATA = None  # the job's folder (lines.json, voices/, models-mlx/)


def say(msg, logfile=None):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    logfile = logfile or LOGFILE
    if logfile:
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def find_data():
    env = os.environ.get("NARRATOR_DATA")
    if env and (Path(env) / "lines.json").exists():
        return Path(env)
    here = Path(__file__).resolve().parent
    if (here / "lines.json").exists():
        return here
    hits = sorted(Path("/kaggle/input").rglob("lines.json")) if Path("/kaggle/input").exists() else []
    if not hits:
        sys.exit("FAILED: couldn't find lines.json")
    return hits[0].parent


def find_model(dirname):
    """A fine-tuned model's merged folder among the attached inputs."""
    roots = [Path(os.environ["NARRATOR_MODELS"])] if os.environ.get("NARRATOR_MODELS") else []  # (Modal: uploaded)
    for root in roots + [Path("/kaggle/input"), Path.cwd()]:
        if root.exists():
            for cfg in root.rglob("config.json"):
                if cfg.parent.name == dirname and any(cfg.parent.glob("*.safetensors")) and not packed_copy(cfg):
                    return cfg.parent
    return None


def packed_copy(cfg):
    """Whether a model folder is the Mac's packed (MLX) copy, which copy_models unpacks, not a model to load."""
    try:
        c = json.loads(Path(cfg).read_text())
    except (OSError, ValueError):
        return False
    return bool(c.get("quantization") or c.get("quantization_config"))


WEIGHTS = re.compile(r"(model|pytorch_model)\S*\.(safetensors|bin)(\.index\.json)?|\S+\.(pt|pth|ckpt)")
SPEAKERS = "talker.model.codec_embedding.weight"  # the pack's speakers: their rows (spk_id) of this table


def qwen_folder(src):
    """A trained Qwen voice pack as Qwen3TTSModel loads it: its own weights, config and tokenizer files, and the
    parts training doesn't change (the speech tokenizer, merges.txt…) from the stock CustomVoice model. Made of
    links in a folder of its own, so the unpacked model (on Modal's storage) is never written to.
    Never the stock model's own weights: once they were in the download cache, they were linked in next to the
    pack's 2 shards, and loading picks a single model.safetensors over shards, so every line came out of the stock
    model with the pack's speaker numbers: another voice (Modal, 17-18 November 2026)."""
    from huggingface_hub import snapshot_download
    src = Path(src)
    size = {"0b6": "0.6B", "1b7": "1.7B"}.get(json.loads((src / "config.json").read_text()).get("tts_model_size"), "1.7B")
    stock = Path(snapshot_download(QWEN_CUSTOM.format(size=size), ignore_patterns=["model.safetensors", "*.md"]))
    dst = Path("/tmp/narrator-qwen") / src.name
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():  # the pack's own files: always these (a link to anything else is replaced)
        link = dst / f.name
        if f.name == "voice.json" or (link.is_symlink() and link.resolve() == f.resolve()):
            continue
        if link.is_symlink() or link.is_file():
            link.unlink()
        link.symlink_to(f.resolve())
    for f in stock.iterdir():  # the rest from the stock model, but never its weights
        link = dst / f.name
        if not WEIGHTS.fullmatch(f.name) and not link.exists() and not link.is_symlink():
            link.symlink_to(f.resolve())
    for f in dst.iterdir():  # (and no weights left there that aren't the pack's)
        if WEIGHTS.fullmatch(f.name) and not (src / f.name).exists():
            f.unlink()
    own = {f.resolve() for f in src.iterdir()}
    foreign = [f.name for f in dst.iterdir() if WEIGHTS.fullmatch(f.name) and f.resolve() not in own]
    if foreign or not any(WEIGHTS.fullmatch(f.name) for f in dst.iterdir()):
        raise RuntimeError(f"the voice pack's folder doesn't hold only its own weights ({', '.join(foreign) or 'none'})")
    return dst


def pack_loaded(model, src):
    """Raises unless the loaded model is the voice pack itself: its speakers' rows as they are in the pack's own
    weight files (another model's weights, loaded by mistake, speak in other voices)."""
    import torch
    from safetensors import safe_open
    src = Path(src)
    index = src / "model.safetensors.index.json"
    file = src / (json.loads(index.read_text())["weight_map"].get(SPEAKERS, "") if index.exists() else "model.safetensors")
    if not file.is_file():
        raise RuntimeError(f"the pack's weights have no {SPEAKERS}")
    with safe_open(str(file), framework="pt") as f:
        if SPEAKERS not in f.keys():
            raise RuntimeError(f"the pack's weights have no {SPEAKERS}")
        want = f.get_tensor(SPEAKERS)
    m = getattr(model, "model", model)
    ids = sorted(int(i) for i in m.config.talker_config.spk_id.values())
    got = m.talker.get_input_embeddings().weight.detach()[ids].float().cpu()
    want = want[ids].to(got.dtype)
    if got.shape != want.shape or not torch.allclose(got, want, rtol=2e-2, atol=1e-3):
        off = float((got - want).abs().max()) if got.shape == want.shape else float("nan")
        raise RuntimeError(f"the loaded speakers aren't the pack's (they differ by up to {off:.3g})")
    return len(ids)


LOCAL_MODELS = Path("/tmp/narrator-models")


def copy_models(dirnames):
    """Copy the attached fine-tuned models to this machine's own disk, in the background while the GPUs make other
    lines. Loading straight from the attached input took 15 minutes for one 6.6 GB model (two GPUs reading it at
    once over Kaggle's network mount); a plain copy reads it once, in order."""
    import shutil
    import threading

    def run():
        for d in dirnames:
            dst = LOCAL_MODELS / d
            try:
                src = find_model(d)
                packed = DATA / "models-mlx" / d if DATA else None  # sent from the Mac as it keeps it (4-bit)
                t0 = time.time()
                if src is None and packed is not None and (packed / "config.json").exists():
                    say(f"Unpacking the trained model {d} sent from the Mac…")
                    sys.path.insert(0, str(Path(__file__).resolve().parent))
                    import unpack_mlx
                    unpack_mlx.main(packed, dst)
                    (dst / ".copied").touch()
                    say(f"Unpacked {d} in {time.time() - t0:.0f}s")
                    continue
                if src is None:
                    raise FileNotFoundError(f"{d} isn't attached")
                say(f"Copying the trained model {d} to local disk…")
                shutil.copytree(src, dst, dirs_exist_ok=True)
                (dst / ".copied").touch()
                gb = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1e9
                say(f"Copied {d} ({gb:.1f} GB) in {time.time() - t0:.0f}s")
            except Exception as e:  # noqa: BLE001 - the GPUs then read it from the input
                dst.mkdir(parents=True, exist_ok=True)
                (dst / ".failed").write_text(str(e))
                say(f"Couldn't copy {d}: {str(e)[:160]}")
    LOCAL_MODELS.mkdir(parents=True, exist_ok=True)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def trained_model(dirname, wait_minutes=45):
    """A fine-tuned model's folder: the local copy once it's made (waiting for it), else the attached input."""
    dst = LOCAL_MODELS / dirname
    started = time.time()
    while dst.parent.exists() and time.time() - started < wait_minutes * 60:
        if (dst / ".copied").exists():
            return dst
        if (dst / ".failed").exists():
            break
        time.sleep(5)
    return find_model(dirname)


def group_key(j):
    model = (j.get("model") or {}).get("dir") if j["engine"] in ("orpheus", "qwen-tuned") else None
    # Higgs and Drama 3 last, each in a Python of its own with its own packages (see run_worker)
    order = {"qwen": 0, "qwen-clone": 1, "qwen-tuned": 2, "orpheus": 3, "higgs": 4, "drama": 5}[j["engine"]]
    return (order, model or "", round(float(j.get("temperature") or 0), 3), j.get("language", ""))


def language_of(j):
    lang = (j.get("language") or "auto").lower()
    return "Auto" if lang in ("auto", "") else lang.capitalize()


BLOCK = 2400  # lines: Qwen lines are batched by length a block of chapters at a time (chapters still come in order)


def by_length(items, n, length, say=print):
    """Batches of at most n lines of about the same prompt length (most of them exactly the same): the longest
    lines of a block together, and so on down, so a batch is hardly padded and takes no longer than its lines.
    items come in chapter order: they're batched a block of chapters at a time, so chapters still come back in
    order. (Tested November 2026: even widely mixed lengths keep the voice, but a little less steadily; batching
    only equal lengths left many lines on their own and took about 35% longer.) A line whose prompt length can't
    be told is placed by its letters."""
    out, block = [], []

    def flush():
        sized = []
        for j in block:
            try:
                size = int(length(j))
            except Exception as e:  # noqa: BLE001 - placed by its letters, then
                say(f"couldn't tell a line's prompt length ({type(e).__name__}: {str(e)[:120]}): placed by its letters")
                size = len(j.get("text") or "") // 3 + 8
            sized.append((size, j))
        sized.sort(key=lambda x: -x[0])  # (stable: lines of one length stay in chapter order)
        out.extend([j for _, j in sized[i:i + n]] for i in range(0, len(sized), n))
        block.clear()

    for j in items:
        if block and len(block) >= BLOCK and j.get("chapter") != block[-1].get("chapter"):
            flush()
        block.append(j)
    flush()
    return out


def prompt_length(qwen, engine, j):
    """A Qwen line's prompt length, as by_length sorts them."""
    tuned = engine == "qwen-tuned"  # (a trained voice pack is played without directions)
    return qwen.prompt_len(j["text"], "" if tuned else (j.get("instruct") or ""), clone=engine == "qwen-clone")


# ---------------- one GPU ----------------

def work(args, data, jobs, out):
    import torch
    started = time.time()
    shard = args.shard or 0
    lines = out / "lines"
    lines.mkdir(parents=True, exist_ok=True)
    progress = out / f"progress-{shard}.json"
    todo = [j for j in jobs if not (lines / f"{j['key']}.flac").exists()]
    todo.sort(key=lambda j: (group_key(j), j.get("chapter", 0), -len(j["text"])))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = (args.dtype or "float16") if device != "cpu" else "float32"  # bfloat16 on newer GPUs (not the T4)
    seed = max((int(j.get("seed") or 0) for j in jobs), default=0)
    if seed:
        torch.manual_seed(seed + shard)
    say(f"GPU {shard}: {len(todo)} lines ({len(jobs) - len(todo)} made before) on "
        f"{torch.cuda.get_device_name(0) if device != 'cpu' else 'CPU'}")
    budget = max(0.05, args.max_hours - MARGIN_HOURS) * 3600
    made, audio, failed, stopped = 0, 0.0, 0, False
    redone = [0, 0]  # takes that came out silent or cut short and were made again: [how many, then fine]
    qwen = orpheus = None
    loaded = None

    def save():
        progress.write_text(json.dumps({"made": made, "total": len(todo), "failed": failed, "audio_seconds": round(audio, 1),
                                        "seconds": round(time.time() - started, 1), "stopped": stopped,
                                        "redone": redone[0], "redone_fine": redone[1]}))

    # Each voice's test lines ("probe": the app checks them against the voice on the Mac and stops a run that came
    # out in the wrong voice) are made first, for every voice, then the rest. (Higgs and Drama 3 keep theirs at the
    # front of their own group: their engines take minutes to start, so they're started once.)
    groups = []
    first = [j for j in todo if j.get("probe") and j["engine"] not in WORKERS]
    picked = {id(j) for j in first}
    for phase in (first, [j for j in todo if id(j) not in picked]):
        g = {}
        for j in sorted(phase, key=lambda j: not j.get("probe")):  # (stable: chapter order kept)
            g.setdefault(group_key(j), []).append(j)
        groups += list(g.items())
    for gk, items in groups:
        if stopped:
            break
        engine = items[0]["engine"]
        if engine in WORKERS:  # in a Python of its own, with its own packages (see run_worker)
            qwen = orpheus = None
            loaded = None
            torch.cuda.empty_cache() if device != "cpu" else None
            n, secs, bad, stopped = run_worker(engine, items, lines, data, started + budget, shard)
            made, audio, failed = made + n, audio + secs, failed + bad
            save()
            continue
        want = (engine, gk[1]) if engine in ("orpheus", "qwen-tuned") else ("qwen", "")
        if loaded != want:  # one big model on the GPU at a time
            qwen = orpheus = None
            torch.cuda.empty_cache() if device != "cpu" else None
            if engine == "orpheus":
                repo = None
                if gk[1]:
                    repo = trained_model(gk[1])
                    if repo is None:
                        say(f"GPU {shard}: the trained model {gk[1]} isn't attached; its {len(items)} lines are left for the Mac")
                        failed += len(items)
                        continue
                orpheus = Orpheus(device, dtype, str(repo) if repo else None)
            elif engine == "qwen-tuned":  # a trained voice pack: its speakers, like Qwen's own
                repo = trained_model(gk[1])
                if repo is None:
                    say(f"GPU {shard}: the trained model {gk[1]} didn't arrive; its {len(items)} lines are left for the Mac")
                    failed += len(items)
                    continue
                qwen = Qwen(device, args.qwen_size, dtype, data / "voices", {})
                try:  # the pack itself, never another model's weights in its place (that's another voice)
                    qwen.custom = qwen._load(str(qwen_folder(repo)))
                    say(f"GPU {shard}: {gk[1]} loaded as itself ({pack_loaded(qwen.custom, repo)} speakers checked)")
                except Exception as e:  # noqa: BLE001 - its lines are left, never made in the wrong voice
                    say(f"GPU {shard}: the trained model {gk[1]} didn't load as itself ({type(e).__name__}: "
                        f"{str(e)[:200]}); its {len(items)} lines are left for the Mac")
                    qwen = None
                    failed += len(items)
                    continue
            else:
                qwen = Qwen(device, args.qwen_size, dtype, data / "voices", {})
            loaded = want
        n = args.batch or BATCH[engine]
        if engine in ("qwen", "qwen-clone", "qwen-tuned"):  # lines of about one length together (hardly padded)
            batches = by_length(items, n, lambda j: prompt_length(qwen, engine, j), say=lambda m: say(f"GPU {shard}: {m}"))
        else:
            batches = [items[s:s + n] for s in range(0, len(items), n)]
        seen = {}  # key -> (voice_id, job, voiced_s) for group-end rushed-take retry while model is loaded
        for batch in batches:
            if time.time() - started > budget:
                stopped = True
                say(f"GPU {shard}: time's up ({args.max_hours} h); the rest are left for next time")
                break
            t0 = time.time()
            try:
                wavs = speak(engine, qwen, orpheus, batch)
            except Exception as e:  # noqa: BLE001 - try them one at a time (out of memory, a bad line)
                say(f"GPU {shard}: batch failed ({type(e).__name__}: {str(e)[:160]}); one at a time")
                torch.cuda.empty_cache() if device != "cpu" else None
                wavs = {}
                for j in batch:
                    try:
                        wavs.update(speak(engine, qwen, orpheus, [j]))
                    except Exception as e2:  # noqa: BLE001
                        say(f"GPU {shard}: line {j['key']} failed: {str(e2)[:160]}")
            # A take that came out silent, broken or cut short (a short line at low stability is the likeliest) is
            # made again now, one at a time and a little calmer each time, while the GPU is at hand.
            redo = [j for j in batch if take_problem(wavs.get(j["key"]), j["text"])]
            still = {}
            for j in redo:
                best = wavs.get(j["key"])
                for attempt in range(1, TRIES + 1):
                    if time.time() - started > budget:
                        break
                    try:
                        w = speak(engine, qwen, orpheus, [calmer(j, attempt)]).get(j["key"])
                    except Exception as e:  # noqa: BLE001
                        say(f"GPU {shard}: line {j['key']} failed again: {str(e)[:160]}")
                        w = None
                    best = better(best, w, j["text"])
                    if not take_problem(best, j["text"]):
                        break
                if best is not None:
                    wavs[j["key"]] = best
                problem = take_problem(best, j["text"])
                if problem:
                    still[problem] = still.get(problem, 0) + 1
            if redo:
                redone[0] += len(redo)
                redone[1] += len(redo) - sum(still.values())
                say(f"GPU {shard}: {len(redo)} line(s) came out silent or cut short; made again, "
                    f"{len(redo) - sum(still.values())} fine"
                    + "".join(f", {n} still {p}" for p, n in still.items()))
            for j in batch:
                w = wavs.get(j["key"])
                if take_problem(w, j["text"]) not in ("", "cut short"):  # nothing, or nothing in it: the Mac makes it
                    failed += 1
                    continue
                # (a take still cut short is kept: the Mac makes that line again when it puts the book together,
                # and falls back on this one only if it can't do better)
                # written under another name, then renamed: the app may copy lines back while the job works,
                # and must never get a half-written one
                part = lines / f".{j['key']}.part"
                sf.write(part, np.clip(w, -1, 1), SR, format="FLAC", subtype="PCM_16")
                os.replace(part, lines / f"{j['key']}.flac")
                made += 1
                audio += len(w) / SR
                vid = f"{engine}:{j.get('speaker') or j.get('voice') or ''}"
                seen[j["key"]] = (vid, j, voiced_seconds(w))
            took = time.time() - t0
            spoken = sum(len(wavs[j["key"]]) for j in batch if wavs.get(j["key"]) is not None) / SR
            say(f"GPU {shard}: {made}/{len(todo)} · {engine} · {spoken:.0f}s audio in {took:.0f}s "
                f"({spoken / max(took, 1e-6):.2f}x) · {clock(time.time() - started)} so far")
            save()
        # Catch rushed takes (skipped words) on the GPU while this voice group's model is still loaded
        if not stopped and seen:
            rush = rushed_keys({k: (v, j["text"], vs) for k, (v, j, vs) in seen.items()})
            fixed_rush = 0
            for k in rush:
                if time.time() - started > budget:
                    break
                _, j, old_vs = seen[k]
                try:
                    w = speak(engine, qwen, orpheus, [calmer(j, 1)]).get(k)
                except Exception as e:  # noqa: BLE001
                    say(f"GPU {shard}: rushed retry {k} failed: {str(e)[:160]}")
                    continue
                if not take_problem(w, j["text"]) and voiced_seconds(w) > old_vs:
                    part = lines / f".{k}.part"
                    sf.write(part, np.clip(w, -1, 1), SR, format="FLAC", subtype="PCM_16")
                    os.replace(part, lines / f"{k}.flac")
                    fixed_rush += 1
            if rush:
                redone[0] += len(rush)
                redone[1] += fixed_rush
                say(f"GPU {shard}: {len(rush)} rushed line(s) retried on GPU; {fixed_rush} improved")
                save()
    save()
    say(f"GPU {shard}: done · {made} lines, {clock(audio)} of audio in {clock(time.time() - started)}"
        + (f" · {failed} left for the Mac" if failed else ""))


WORKERS = {"higgs": ("higgs_worker", "Higgs Audio"), "drama": ("drama_worker", "Drama 3")}


def run_worker(engine, jobs, lines, data, stop_at, shard):
    """Make Higgs or Drama 3 lines with <engine>_worker.py: (made, seconds of audio, failed, stopped). Its packages
    are installed first, once per machine (they change transformers / torch, which is why these lines come last)."""
    import fcntl
    module, name = WORKERS[engine]
    here = Path(__file__).resolve().parent
    flag = Path(f"/tmp/narrator-{engine}-ready")
    with open(f"/tmp/narrator-{engine}.lock", "w") as lock:  # (one install when there are several GPU processes)
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not flag.exists():
            sys.path.insert(0, str(here))
            worker = __import__(module)
            try:
                worker.setup(lambda m: say(f"GPU {shard}: {m}"))
            except Exception as e:  # noqa: BLE001 - its lines are left for next time
                say(f"GPU {shard}: {name} couldn't be installed ({str(e)[:300]}); its {len(jobs)} lines are left")
                return 0, 0.0, len(jobs), False
            flag.write_text("ok")
    todo = Path(f"/tmp/narrator-{engine}-{shard}.json")
    todo.write_text(json.dumps(jobs))
    say(f"GPU {shard}: {len(jobs)} {name} lines")
    made, secs, bad, stopped, t0 = 0, 0.0, 0, False, time.time()
    p = subprocess.Popen([sys.executable, str(here / f"{module}.py"), str(todo), str(lines), str(stop_at),
                          str(data / "voices")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        line = line.rstrip()
        if line.startswith("MADE "):
            made += 1
            secs += float(line.split()[2])
            if made % 10 == 0:
                took = time.time() - t0
                say(f"GPU {shard}: {made}/{len(jobs)} · {engine} · {secs:.0f}s audio in {took:.0f}s "
                    f"({secs / max(took, 1e-6):.2f}x)")
        elif line.startswith("FAILED "):
            bad += 1
            say(f"GPU {shard}: line {line[7:]}")
        elif line == "STOPPED":
            stopped = True
            say(f"GPU {shard}: time's up; the rest of the {name} lines are left for next time")
        elif line:
            print(line, flush=True)  # the model's own messages (and any error) go to the log
    p.wait()
    if p.returncode and not made:
        say(f"GPU {shard}: {name} stopped with an error (exit {p.returncode}); its lines are left")
    bad += len(jobs) - made - bad if not stopped else 0
    return made, secs, bad, stopped


def run_sounds(data, out, stop_at, logfile):
    """The book's missing sounds (sounds.json), after its lines, on the first GPU: {"sounds", "sounds_made"}."""
    todo = data / "sounds.json"
    if not todo.exists():
        return {}
    jobs = json.loads(todo.read_text())
    if not jobs:
        return {}
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    import sound_worker
    t0 = time.time()
    try:
        sound_worker.setup(lambda m: say(m, logfile))
    except Exception as e:  # noqa: BLE001 - the Mac makes them instead
        say(f"Sounds: couldn't start the sound generator here ({str(e)[:300]}); the Mac makes the {len(jobs)} sounds", logfile)
        return {"sounds": len(jobs), "sounds_made": 0}
    say(f"Sounds: {len(jobs)} to make ({sum(int(j.get('takes') or 1) for j in jobs)} takes)", logfile)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    p = subprocess.Popen([sys.executable, str(here / "sound_worker.py"), str(todo), str(out / "sounds"), str(stop_at),
                          str(data / "stable_audio_3")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    made, takes = set(), 0
    for line in p.stdout:
        line = line.rstrip()
        if line.startswith("MADE "):
            takes += 1
            made.add(line.split()[1])
            say(f"Sounds: {takes} take(s) made · {line.split()[1]} in {line.split()[2]}s", logfile)
        elif line.startswith("FAILED "):
            say(f"Sounds: {line[7:]}", logfile)
        elif line == "STOPPED":
            say("Sounds: time's up; the rest are made on the Mac", logfile)
        elif line:
            print(line, flush=True)
    p.wait()
    say(f"Sounds: {len(made)} of {len(jobs)} made in {clock(time.time() - t0)}", logfile)
    return {"sounds": len(jobs), "sounds_made": len(made), "sound_seconds": round(time.time() - t0, 1)}


def speak(engine, qwen, orpheus, batch):
    """{key: 24 kHz audio}."""
    if engine == "orpheus":
        items = [{"voice": f"orpheus:{j['speaker']}", "text": j["text"], "key": j["key"]} for j in batch]
        got = orpheus.speak(items, temperature=float(batch[0]["temperature"]), top_p=0.95, repetition_penalty=1.1)
        return {it["key"]: got[id(it)] for it in items if id(it) in got}
    prefix = "my" if engine == "qwen-clone" else "builtin"  # (a trained pack's speakers are its own "builtin" ones)
    name = "voice" if engine == "qwen-clone" else "speaker"
    tuned = engine == "qwen-tuned"  # as the Mac plays it (tts/qwen.py): no direction, no language tag
    items = [{"voice": f"{prefix}:{j[name]}", "text": j["text"], "instruct": "" if tuned else j.get("instruct") or "",
              "key": j["key"]} for j in batch]
    sampling = {"temperature": float(batch[0]["temperature"] or 0.86), "top_p": 1.0, "top_k": 50,
                "repetition_penalty": 1.05, "do_sample": True, "max_new_tokens": 720}  # 720 codes: a minute
    if tuned:  # cut off after about twice the line's length, so a line it can't end doesn't run on for a minute
        sampling["max_new_tokens"] = min(720, max(150, int(1.8 * max(len(j["text"]) for j in batch))))
    got = qwen.speak(items, "Auto" if tuned else language_of(batch[0]), sampling)
    return {it["key"]: got[id(it)] for it in items if id(it) in got}


# ---------------- the whole run ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/kaggle/working" if Path("/kaggle/working").exists() else "out")
    ap.add_argument("--max-hours", type=float, default=11.0)
    ap.add_argument("--qwen-size", default="1.7B")
    ap.add_argument("--dtype", default="", help="float16 (T4, the default) or bfloat16 (A10, L4, L40S, A100…)")
    ap.add_argument("--batch", type=int, default=0, help="lines at once (default: by model, for a 16 GB GPU)")
    ap.add_argument("--shard", type=int, default=None, help="(internal) this process's share of the lines")
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--no-pack", action="store_true",
                    help="leave the lines as separate files (the app copies them back as they're made)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logfile = out / "run.log"
    data = find_data()
    global DATA
    DATA = data
    jobs = json.loads((data / "lines.json").read_text(encoding="utf-8"))
    jobs.sort(key=lambda j: (group_key(j), j.get("chapter", 0), -len(j["text"])))

    if args.shard is not None:  # one GPU's share
        work(args, data, jobs[args.shard::args.of], out)
        return

    started = time.time()
    import torch
    gpus = max(1, torch.cuda.device_count())
    say(f"{len(jobs)} lines on {gpus} GPU(s): " + ", ".join(sorted({j['engine'] for j in jobs})), logfile)
    global LOGFILE
    LOGFILE = logfile
    trained = sorted({j["model"]["dir"] for j in jobs if j["engine"] in ("orpheus", "qwen-tuned") and j.get("model")})
    copying = copy_models(trained) if trained else None  # (trained voices come last, so the copy is usually done)
    if gpus == 1:
        args.shard, args.of = 0, 1
        work(args, data, jobs, out)
    else:
        procs = []
        for i in range(gpus):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            cmd = [sys.executable, str(Path(__file__).resolve()), "--out", str(out), "--max-hours", str(args.max_hours),
                   "--qwen-size", args.qwen_size, "--shard", str(i), "--of", str(gpus),
                   "--dtype", args.dtype, "--batch", str(args.batch)]
            procs.append(subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
        import threading

        def relay(p):
            for line in p.stdout:
                line = line.rstrip()
                print(line, flush=True)
                with open(logfile, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        threads = [threading.Thread(target=relay, args=(p,), daemon=True) for p in procs]
        for t in threads:
            t.start()
        for p in procs:
            p.wait()
        for t in threads:
            t.join(timeout=5)
        bad = [i for i, p in enumerate(procs) if p.returncode]
        if bad:
            say(f"GPU process(es) {bad} stopped with an error; keeping what they made", logfile)
    if copying is not None:
        copying.join(timeout=1)
    sounds = run_sounds(data, out, started + max(0.05, args.max_hours - MARGIN_HOURS) * 3600, logfile)

    # pack the lines (one download instead of thousands) and sum up
    parts = [json.loads(f.read_text()) for f in sorted(out.glob("progress-*.json"))]
    audio = sum(p["audio_seconds"] for p in parts)
    took = time.time() - started
    files = sorted((out / "lines").glob("*.flac")) if (out / "lines").exists() else []
    if not args.no_pack:
        with tarfile.open(out / "lines.tar", "w") as t:
            for f in files:
                t.add(f, arcname=f"lines/{f.name}")
        for f in files:
            f.unlink()
    summary = {"made": len(files), "total": len(jobs), "failed": sum(p.get("failed", 0) for p in parts), **sounds,
               "audio_seconds": round(audio, 1), "seconds": round(took, 1),
               "speed": round(audio / max(took, 1e-6), 2), "stopped": any(p.get("stopped") for p in parts)}
    (out / "progress.json").write_text(json.dumps(summary))
    for f in out.glob("progress-*.json"):
        f.unlink()
    say(f"DONE: {len(files)} of {len(jobs)} lines, {clock(audio)} of audio in {clock(took)} "
        f"({summary['speed']:.2f}x real time overall)" + (" · stopped at the time limit" if summary["stopped"] else ""),
        logfile)
    if not files:
        say("FAILED: no lines were made", logfile)


if __name__ == "__main__":
    main()
