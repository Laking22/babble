"""Render a Babble audiobook project on a Kaggle (or any CUDA) GPU.

Reads project.json exported by Studio ("Export for Kaggle"), renders every chapter with
Qwen3-TTS (built-in speakers + your saved voices) and/or Orpheus, saves each chapter as it
finishes, and joins them into an M4B/MP3/WAV at the end.

Resuming: chapters already present in the output folder, or in any attached input dataset
under a "chapters" folder (e.g. a previous run's output), are reused automatically.

Usage:  python runner.py [path/to/project_folder] [--out /kaggle/working] [--max-hours 11]
"""
import argparse
import gc
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 24000
QWEN_CUSTOM = "Qwen/Qwen3-TTS-12Hz-{size}-CustomVoice"
QWEN_BASE = "Qwen/Qwen3-TTS-12Hz-{size}-Base"
ORPHEUS_MODEL = "unsloth/orpheus-3b-0.1-ft"  # open copy of canopylabs/orpheus-3b-0.1-ft (no access request needed)
SNAC_MODEL = "hubertsiuzdak/snac_24khz"
ORPHEUS_TAGS = re.compile(r"<(laugh|chuckle|sigh|cough|sniffle|groan|yawn|gasp)>\s*", re.I)
EMOTION_TO_TAG = [("laugh", "laugh"), ("giggl", "laugh"), ("chuckl", "chuckle"), ("sigh", "sigh"), ("weary", "sigh"),
                  ("gasp", "gasp"), ("shock", "gasp"), ("sob", "sniffle"), ("cry", "sniffle"), ("tearful", "sniffle"),
                  ("weep", "sniffle"), ("sniff", "sniffle"), ("groan", "groan"), ("yawn", "yawn"), ("sleepy", "yawn"),
                  ("cough", "cough")]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def clock(s):
    s = int(s)
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"


# ---------------- devices ----------------

def devices():
    import torch
    n = torch.cuda.device_count()
    if n == 0:
        log("WARNING: no GPU found, running on CPU (very slow). In Kaggle: Settings → Accelerator → GPU T4 x2.")
        return "cpu", "cpu"
    names = ", ".join(torch.cuda.get_device_name(i) for i in range(n))
    log(f"GPUs: {names}")
    return "cuda:0", ("cuda:1" if n > 1 else "cuda:0")


def to_24k(audio, sr):
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sr == SR or len(audio) == 0:
        return audio
    return np.interp(np.linspace(0, len(audio), int(len(audio) * SR / sr), endpoint=False),
                     np.arange(len(audio)), audio).astype(np.float32)


# ---------------- is a take of a line any good? (the app's audio.py has the same test: keep them alike) ----------------

SILENT_PEAK = 10 ** (-50 / 20)  # never louder than -50 dB: no speech in it
SPEECH_RATE = 45.0  # letters a second: about twice the fastest narration measured (9-24); faster than that, it was cut short
MIN_SPEECH = 0.15   # seconds of sound any line with words in it has at least
UNSAID = re.compile(r"\[[^\]]*\]|<[^>]*>")  # directions ([whispering]) and tags (<laugh>) aren't words
TRIES = 2  # a take that won't do is made again this many more times, a little calmer each time


def voiced_seconds(wav):
    """How long a take has sound in it: 10 ms steps within 30 dB of its loudest (and louder than -50 dB)."""
    a = np.nan_to_num(np.asarray(wav, dtype=np.float32).reshape(-1))
    hop = SR // 100
    steps = len(a) // hop
    if steps == 0:
        return 0.0
    db = 10 * np.log10((a[:steps * hop].reshape(steps, hop) ** 2).mean(1) + 1e-12)
    return float((db > max(db.max() - 30.0, -50.0)).sum()) / 100


def take_problem(wav, text):
    """What's wrong with a take of a line: "none" (nothing came out), "broken", "silent", "cut short" (far less sound
    than its words take: cut off, or only a breath or a click), or "" when it's fine."""
    if wav is None:
        return "none"
    a = np.asarray(wav, dtype=np.float32).reshape(-1)
    if len(a) < SR // 10 or not np.isfinite(a).all():
        return "broken"
    letters = len(re.findall(r"\w", UNSAID.sub(" ", text or "")))
    if not letters:
        return ""
    if float(np.abs(a).max()) < SILENT_PEAK:
        return "silent"
    if voiced_seconds(a) < max(MIN_SPEECH, letters / SPEECH_RATE):
        return "cut short"
    return ""


def better(a, b, text):
    """The better of two takes of a line: a fine one, else one cut short (the one with more sound) over a silent,
    broken or missing one."""
    pa, pb = take_problem(a, text), take_problem(b, text)
    if not pa or not pb:
        return a if not pa else b
    rank = {"none": 0, "broken": 0, "silent": 1, "cut short": 2}
    if rank[pa] != rank[pb]:
        return a if rank[pa] > rank[pb] else b
    return b if pa == "cut short" and voiced_seconds(b) > voiced_seconds(a) else a


RUSH = 1.25         # matches narrator/audio.py: quicker than 1.25x the voice's median rate -> words skipped
RUSH_LETTERS = 60
RUSH_TAKES = 20
RUSH_ABS_RATE = 24.0


def rushed_keys(takes):
    """takes: {key: (voice, text, voiced_s)}. Keys of takes much quicker than their voice's median pace,
    or faster than RUSH_ABS_RATE on long lines when a short run has fewer than RUSH_TAKES takes."""
    by_voice = {}
    for k, (voice, text, vs) in takes.items():
        n = len(re.findall(r"\w", UNSAID.sub(" ", text or "")))
        if n and vs > 0:
            by_voice.setdefault(voice, []).append((k, n, n / max(vs, 0.1)))
    out = set()
    for rows in by_voice.values():
        if len(rows) < RUSH_TAKES:
            out |= {k for k, n, r in rows if n >= RUSH_LETTERS and r > RUSH_ABS_RATE}
            continue
        middle = float(np.median([r for _, _, r in rows]))
        out |= {k for k, n, r in rows if n >= RUSH_LETTERS and r > RUSH * middle}
    return out


def calmer(job, attempt):
    """A line's job for another try: its temperature a little lower each time (a short line at low stability is the
    one most likely to come out wrong), never under 0.5 unless it started there."""
    t = float(job.get("temperature") or 0.86)
    return dict(job, temperature=round(max(min(t, 0.5), t * 0.85 ** attempt), 3))


def stretch(wav, speed):
    if abs(speed - 1.0) < 0.01 or len(wav) == 0:
        return wav
    if shutil.which("ffmpeg"):
        proc = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
                               "-filter:a", f"atempo={speed:.3f}", "-f", "f32le", "-ar", str(SR), "-ac", "1", "pipe:1"],
                              input=wav.astype(np.float32).tobytes(), capture_output=True, check=True)
        return np.frombuffer(proc.stdout, dtype=np.float32).copy()
    import librosa
    return librosa.effects.time_stretch(wav, rate=speed).astype(np.float32)


# ---------------- Qwen3-TTS (official package) ----------------

class Qwen:
    def __init__(self, device, size, dtype_name, voices_dir, voice_info):
        self.device, self.size, self.dtype_name = device, size, dtype_name
        self.voices_dir, self.voice_info = voices_dir, voice_info
        self.custom = self.base = None
        self.prompts = {}

    def _load(self, repo):
        import torch
        from qwen_tts import Qwen3TTSModel
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[self.dtype_name]
        log(f"Loading {repo} ({self.dtype_name}) on {self.device}…")
        return Qwen3TTSModel.from_pretrained(repo, device_map=self.device, dtype=dtype, attn_implementation="sdpa")

    def prompt_len(self, text, instruct="", clone=False):
        """How many tokens a line's prompt takes as Qwen3-TTS builds it (its words, and its direction where it has
        one): lines of about one length are batched together (lines_runner.by_length), so a batch is hardly padded."""
        if clone:
            if self.base is None:
                self.base = self._load(QWEN_BASE.format(size=self.size))
            m = self.base
        else:
            if self.custom is None:
                self.custom = self._load(QWEN_CUSTOM.format(size=self.size))
            m = self.custom
        tokens = lambda s: int(m.processor(text=s, return_tensors="pt")["input_ids"].shape[-1])  # noqa: E731
        n = tokens(f"<|im_start|>assistant\n{ORPHEUS_TAGS.sub('', text)}<|im_end|>\n<|im_start|>assistant\n")
        return n + (tokens(f"<|im_start|>user\n{instruct}<|im_end|>\n") if instruct else 0)

    def speak(self, items, language, sampling):
        builtin = [it for it in items if it["voice"].startswith("builtin:")]
        mine = [it for it in items if it["voice"].startswith("my:")]
        out = {}
        if builtin:
            if self.custom is None:
                self.custom = self._load(QWEN_CUSTOM.format(size=self.size))
            wavs, sr = self.custom.generate_custom_voice(
                text=[ORPHEUS_TAGS.sub("", it["text"]) for it in builtin],
                speaker=[it["voice"].split(":", 1)[1] for it in builtin],
                language=[language] * len(builtin),
                instruct=[it.get("instruct") or "" for it in builtin], **sampling)
            out.update({id(it): to_24k(w, sr) for it, w in zip(builtin, wavs)})
        if mine:
            if self.base is None:
                self.base = self._load(QWEN_BASE.format(size=self.size))
            for it in mine:
                name = it["voice"].split(":", 1)[1]
                if name not in self.prompts:
                    self.prompts[name] = self.base.create_voice_clone_prompt(
                        ref_audio=str(self.voices_dir / f"{name}.wav"), ref_text=None, x_vector_only_mode=True)
            wavs, sr = self.base.generate_voice_clone(
                text=[ORPHEUS_TAGS.sub("", it["text"]) for it in mine], language=[language] * len(mine),
                voice_clone_prompt=[self.prompts[it["voice"].split(":", 1)[1]][0] for it in mine], **sampling)
            out.update({id(it): to_24k(w, sr) for it, w in zip(mine, wavs)})
        return out


# ---------------- Orpheus (transformers + SNAC) ----------------

class Orpheus:
    START, END = [128259], [128009, 128260, 128261, 128257]
    STOP, PAD, AUDIO_START, OFFSET = 128258, 128263, 128257, 128266

    def __init__(self, device, dtype_name, repo=None):
        import torch
        from snac import SNAC
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_name]
        repo = repo or ORPHEUS_MODEL  # or a fine-tuned model's folder
        log(f"Loading {repo} ({dtype_name}) on {device}…")
        self.torch, self.device = torch, device
        # a fine-tuned model has the same tokenizer as stock Orpheus (its saved copy may not load on its own)
        self.tok = AutoTokenizer.from_pretrained(ORPHEUS_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=dtype).to(device).eval()
        self.snac = SNAC.from_pretrained(SNAC_MODEL).eval().to(device)

    def recent_penalty(self, penalty, size=20):
        """The repetition penalty as the app uses it on the Mac (mlx-lm): only on the last 20 tokens. Penalising
        every sound token made so far (transformers' own repetition_penalty) pushes long lines off course: a
        fine-tuned voice then rambles on into babble instead of stopping."""
        from transformers import LogitsProcessor, LogitsProcessorList
        t = self.torch

        class Recent(LogitsProcessor):
            def __call__(self, input_ids, scores):
                recent = input_ids[:, -size:]
                picked = t.gather(scores, 1, recent)
                picked = t.where(picked < 0, picked * penalty, picked / penalty)
                return scores.scatter(1, recent, picked)
        return LogitsProcessorList([Recent()])

    @staticmethod
    def tagged(text, emotion):
        if ORPHEUS_TAGS.search(text) or not emotion:
            return text
        tag = next((t for word, t in EMOTION_TO_TAG if word in emotion.lower()), None)
        return f"<{tag}> {text}" if tag else text

    def prompt_ids(self, voice, text):
        return self.START + self.tok(f"{voice}: {text}").input_ids + self.END

    def codes_to_audio(self, tokens):
        tokens = [t for t in tokens if t != self.STOP]
        if self.AUDIO_START in tokens:
            tokens = tokens[len(tokens) - tokens[::-1].index(self.AUDIO_START):]
        codes = [t - self.OFFSET for t in tokens[: len(tokens) // 7 * 7]]
        l1, l2, l3 = [], [], []
        for i in range(len(codes) // 7):
            f = codes[7 * i: 7 * i + 7]
            frame = (f[0], f[1] - 4096, f[2] - 2 * 4096, f[3] - 3 * 4096, f[4] - 4 * 4096, f[5] - 5 * 4096, f[6] - 6 * 4096)
            if all(0 <= c < 4096 for c in frame):  # skip corrupt frames instead of failing
                l1.append(frame[0]); l2 += [frame[1], frame[4]]; l3 += [frame[2], frame[3], frame[5], frame[6]]
        if not l1:
            return np.zeros(0, dtype=np.float32)
        t = self.torch
        layers = [t.tensor(x, dtype=t.int32, device=self.device).unsqueeze(0) for x in (l1, l2, l3)]
        with t.inference_mode():
            return self.snac.decode(layers).squeeze().float().cpu().numpy().astype(np.float32)

    def speak(self, items, temperature=0.2, top_p=0.9, repetition_penalty=1.1):
        t = self.torch
        out = {}
        pending = list(items)
        for attempt in range(3):
            if not pending:
                break
            texts = [self.tagged(it["text"], it.get("instruct")) for it in pending]
            ids = [self.prompt_ids(it["voice"].split(":", 1)[1], tx) for it, tx in zip(pending, texts)]
            width = max(len(x) for x in ids)
            input_ids = t.tensor([[self.PAD] * (width - len(x)) + x for x in ids], device=self.device)
            mask = t.tensor([[0] * (width - len(x)) + [1] * len(x) for x in ids], device=self.device)
            words = max(len(tx.split()) for tx in texts)
            max_new = int(min(4096, 300 + words / 2.2 * 86 * 1.8))
            with t.inference_mode():
                gen = self.model.generate(input_ids=input_ids, attention_mask=mask, max_new_tokens=max_new, do_sample=True,
                                          temperature=temperature * (1 + 0.15 * attempt), top_p=top_p,
                                          logits_processor=self.recent_penalty(repetition_penalty * (1 + 0.15 * attempt)),
                                          eos_token_id=self.STOP, pad_token_id=self.PAD)
            retry = []
            for it, tx, row in zip(pending, texts, gen[:, width:].tolist()):
                audio = self.codes_to_audio(row)
                seconds = len(audio) / SR
                expected = len(tx.split()) / 2.5
                if (seconds < 0.3 or seconds > expected * 2.5 + 4) and attempt < 2:
                    retry.append(it)  # runaway, looping or empty audio: try again with adjusted sampling
                else:
                    out[id(it)] = audio
            if retry:
                log(f"  Orpheus: retrying {len(retry)} line(s) that came out too long or empty")
            pending = retry
        return out


# ---------------- export ----------------

def _esc(v):
    return str(v or "").replace("\\", "\\\\").replace("=", "\\=").replace(";", "\\;").replace("#", "\\#").replace("\n", " ")


def combine(files, titles, out_base, fmt, title, author, cover):
    ext = {"MP3": ".mp3", "WAV": ".wav"}.get(fmt, ".m4b")
    out = Path(out_base).with_suffix(ext)
    with tempfile.TemporaryDirectory() as tmp:
        listing = Path(tmp) / "list.txt"
        listing.write_text("".join(f"file '{Path(f).as_posix()}'\n" for f in files), encoding="utf-8")
        meta = [";FFMETADATA1", f"title={_esc(title)}", f"artist={_esc(author)}", f"album={_esc(title)}", "genre=Audiobook"]
        start = 0
        for f, name in zip(files, titles):
            length = int(sf.info(str(f)).duration * 1000)
            meta += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={start}", f"END={start + length}", f"title={_esc(name)}"]
            start += length
        meta_file = Path(tmp) / "meta.txt"
        meta_file.write_text("\n".join(meta) + "\n", encoding="utf-8")
        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-i", str(meta_file)]
        use_cover = bool(cover) and Path(cover).exists() and fmt != "WAV"
        if use_cover:
            cmd += ["-i", str(cover)]
        cmd += ["-map", "0:a", "-map_metadata", "1"]
        if fmt not in ("MP3", "WAV"):
            cmd += ["-map_chapters", "1"]
        if use_cover:
            cmd += ["-map", "2:v", "-c:v", "mjpeg", "-disposition:v:0", "attached_pic"]
        if fmt == "MP3":
            cmd += ["-c:a", "libmp3lame", "-b:a", "96k", "-id3v2_version", "3"]
        elif fmt == "WAV":
            cmd += ["-c:a", "pcm_s16le"]
        else:
            cmd += ["-c:a", "aac", "-b:a", "64k", "-f", "ipod"]
        subprocess.run(cmd + [str(out)], check=True)
    return out


# ---------------- main ----------------

def find_project(arg):
    if arg:
        p = Path(arg)
        return p if (p / "project.json").exists() else p.parent
    for root in (Path("/kaggle/input"), Path.cwd()):
        hits = sorted(root.rglob("project.json")) if root.exists() else []
        if hits:
            return hits[0].parent
    sys.exit("Couldn't find project.json. Attach the exported project as a Kaggle dataset.")


def chapter_key(items, settings):
    return hashlib.sha1(json.dumps([items, settings], sort_keys=True).encode()).hexdigest()[:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project", nargs="?")
    ap.add_argument("--out", default="/kaggle/working" if Path("/kaggle/working").exists() else "out")
    ap.add_argument("--max-hours", type=float, default=11.0, help="stop starting new chapters after this long")
    ap.add_argument("--qwen-size", default="1.7B", help="1.7B (best) or 0.6B (faster, for testing)")
    ap.add_argument("--dtype", default="float16", help="float16 (T4), bfloat16 (newer GPUs) or float32")
    ap.add_argument("--batch", type=int, default=0, help="lines per batch (default: from the project)")
    args = ap.parse_args()

    started = time.time()
    project_dir = find_project(args.project)
    project = json.loads((project_dir / "project.json").read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    chapters_dir = out_dir / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)
    settings = {k: project[k] for k in ("language", "temperature", "seed")}
    batch = args.batch or int(project.get("batch", 4))
    log(f"Project “{project['title']}”: {len(project['chapters'])} chapters, "
        f"{sum(len(c['items']) for c in project['chapters'])} lines → {out_dir}")

    # Reuse chapters from previous runs (attached as input datasets)
    previous = {}
    for root in (Path("/kaggle/input"), chapters_dir):
        if root.exists():
            for f in root.rglob("*.wav"):
                previous.setdefault(f.name, f)

    first_gpu, second_gpu = devices()  # Orpheus on the first GPU, Qwen on the second (same GPU if only one)
    if first_gpu == "cpu":
        args.dtype = "float32"
    uses = {it["voice"].split(":", 1)[0] for c in project["chapters"] for it in c["items"]}
    qwen = None
    if uses & {"builtin", "my"}:
        qwen = Qwen(second_gpu if "orpheus" in uses else first_gpu, args.qwen_size, args.dtype,
                    project_dir / "voices", project.get("voices", {}))
    orpheus = None
    lang = project["language"]
    language = "Auto" if lang in ("auto", "") else lang.capitalize()
    sampling = {"temperature": float(project["temperature"]), "top_p": 1.0, "top_k": 50, "repetition_penalty": 1.05,
                "do_sample": True}
    if int(project.get("seed") or 0) > 0:
        import torch
        torch.manual_seed(int(project["seed"]))

    files, titles, done_audio, total = [], [], 0.0, len(project["chapters"])
    for index, chapter in enumerate(project["chapters"]):
        items = chapter["items"]
        name = f"{index:03d}-{chapter_key(items, settings)}.wav"
        target = chapters_dir / name
        label = chapter["title"] or "Story"
        if name in previous and previous[name] != target:
            shutil.copy(previous[name], target)
        if target.exists():
            log(f"[{index + 1}/{total}] {label}: already made, reusing it")
            files.append(target); titles.append(label)
            continue
        if time.time() - started > args.max_hours * 3600:
            log(f"Time limit reached ({args.max_hours} h). Run the notebook again with this output attached to continue.")
            break
        log(f"[{index + 1}/{total}] {label}: {len(items)} lines")
        t0 = time.time()
        wavs = {}
        for start in range(0, len(items), batch):
            group = items[start:start + batch]
            remote = [it for it in group if it["voice"].startswith("orpheus:")]
            local = [it for it in group if not it["voice"].startswith("orpheus:")]
            if remote:
                if orpheus is None:
                    orpheus = Orpheus(first_gpu, args.dtype)
                wavs.update(orpheus.speak(remote))
            if local:
                wavs.update(qwen.speak(local, language, sampling))
            spoken = sum(len(wavs[id(it)]) for it in group if id(it) in wavs) / SR
            log(f"   lines {start + 1}-{start + len(group)} of {len(items)} · {spoken:.1f}s audio")
        parts = []
        for it in items:
            parts += [stretch(wavs.get(id(it), np.zeros(0, np.float32)), float(it.get("speed") or 1.0)),
                      np.zeros(int(SR * float(it.get("pause", 0.3))), dtype=np.float32)]
        audio = np.concatenate(parts)
        sf.write(target, audio, SR, subtype="PCM_16")
        done_audio += len(audio) / SR
        took = time.time() - t0
        log(f"   saved {name}: {clock(len(audio) / SR)} of audio in {clock(took)} ({len(audio) / SR / took:.2f}× real time)")
        files.append(target); titles.append(label)
        gc.collect()

    if len(files) == total:
        cover = project_dir / project["cover"] if project.get("cover") else None
        out = combine(files, titles, out_dir / project["slug"], project.get("format", "Audiobook (M4B with chapters)"),
                      project["title"], project.get("author", ""), cover)
        log(f"DONE: {out}  ({clock(sum(sf.info(str(f)).duration for f in files))} of audio, total time {clock(time.time() - started)})")
    else:
        log(f"Finished {len(files)} of {total} chapters this session. Chapters are saved in {chapters_dir}.")


if __name__ == "__main__":
    main()
