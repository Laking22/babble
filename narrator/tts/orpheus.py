"""Orpheus 3B: the stock voices plus installed fine-tuned models (paths.MODELS/*/voice.json).

Runs on this Mac (4-bit MLX), on Windows and Linux through PyTorch (orpheus_torch.py), or through an OpenAI-style
API server (Orpheus-TTS-FastAPI compatible). Fine-tuned models run on the computer itself, not on a server.
"""
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import soundfile as sf


def _load_model(path):
    from mlx_audio.tts.utils import load_model
    return load_model(path)

from .. import settings
from ..audio import is_silent, level, resample, trim_edges
from ..paths import MODELS
from ..runtime import HAS_MLX, free, slot
from .base import Engine, Voice, line_temperature
from . import orpheus_torch, torch_rt

STOCK = "mlx-community/orpheus-3b-0.1-ft-4bit"
STOCK_VOICES = {
    "tara": ("Tara", "female", "neutral, balanced"),
    "leah": ("Leah", "female", "warm"),
    "jess": ("Jess", "female", "expressive"),
    "mia": ("Mia", "female", "young"),
    "zoe": ("Zoe", "female", "bright"),
    "leo": ("Leo", "male", "deep"),
    "dan": ("Dan", "male", "natural"),
    "zac": ("Zac", "male", "casual"),
}
TAGS = re.compile(r"<(laugh|chuckle|sigh|cough|sniffle|groan|yawn|gasp)>\s*", re.I)
_EMOTION_TO_TAG = [("laugh", "laugh"), ("giggl", "laugh"), ("chuckl", "chuckle"), ("sigh", "sigh"), ("weary", "sigh"),
                   ("gasp", "gasp"), ("shock", "gasp"), ("sob", "sniffle"), ("cry", "sniffle"), ("tearful", "sniffle"),
                   ("weep", "sniffle"), ("sniff", "sniffle"), ("groan", "groan"), ("yawn", "yawn"), ("sleepy", "yawn"),
                   ("cough", "cough")]



def _installed_order(meta):
    st = meta.parent.stat()
    return (getattr(st, "st_birthtime", st.st_mtime), meta.parent.name)


def finetuned():
    """key -> (display name, gender, description, model folder, trained speaker name)."""
    out = {}
    # oldest install first, so it keeps the plain key and a newer model can't take over an existing voice
    for meta in sorted(MODELS.glob("*/voice.json"), key=_installed_order):
        try:
            data = json.loads(meta.read_text())
        except ValueError:
            continue
        if data.get("engine", "orpheus") != "orpheus":
            continue
        for speaker, v in data.get("voices", {}).items():
            # Keys must be unique here, but the model is still prompted with the name it was trained on.
            key = speaker if speaker not in out and speaker not in STOCK_VOICES else f"{speaker}-{meta.parent.name.split('-')[0]}"
            out[key] = (v.get("name", speaker.title()), v.get("gender", "unspecified"),
                        v.get("description", "fine-tuned"), str(meta.parent), speaker)
    return out


def _text(text, emotion):
    """Orpheus understands sound tags like <laugh>; turn the line's emotion into one."""
    if TAGS.search(text) or not emotion:
        return text
    lowered = emotion.lower()
    tag = next((t for word, t in _EMOTION_TO_TAG if word in lowered), None)
    return f"<{tag}> {text}" if tag else text


BATCH = 4  # lines made at once when batching is on: 4 was ~2.2x faster on an 8 GB Mac; 8 was no faster
SOS, EOS_SPEECH, EOT, CODE0 = 128257, 128258, 128009, 128266  # Orpheus special tokens
# With fewer lines than BATCH (e.g. one short paragraph), slots would sit idle, so lines are split into sentences
# that run side by side and are joined back. Tested: one paragraph 3-6x faster, same Whisper match, voice and pace
# on average. With BATCH or more lines it's slower (more pieces), so it isn't done then.
SENTENCE_GAP = 0.5  # seconds between joined sentences; the model's own pause inside a line was 0.53 s (median)
_ABBREV = re.compile(r"(?:^|\s)(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Rev|Gen|Col|Lt|Sgt|Capt|Hon|vs|etc|[A-HJ-Z])\.$")
_NOT_INITIAL = re.compile(r"(?i:\b(?:a|an|plan|vitamin|grade|type|letter|class|row|section|block|gate)\s+)[A-Z]\.$")
_SENTENCE = re.compile(r".+?(?:(?:[.!?…][”\"’']?(?=\s+[“\"‘'A-Z\u00C0-\u024F\u0400-\u04FF\u3040-\u9FFF\uAC00-\uD7AF])|[。！？][」』”\"’']?)|$)", re.S)


def _sentences(text):
    out = []
    for s in _SENTENCE.findall(text or ""):
        s = s.strip()
        if not s:
            continue
        if out and _ABBREV.search(out[-1]) and not _NOT_INITIAL.search(out[-1]):
            out[-1] = f"{out[-1]} {s}"
        else:
            out.append(s)
    return out


def batching():
    return bool(settings.load().get("orpheus_batch"))


def _max_tokens(text):
    # mlx-audio stops each line at max_tokens (default 1200 ≈ 14.6 s) and silently drops the rest. Orpheus uses
    # ~84 tokens/s and narrators read ~14 chars/s (≈6 tokens per char): allow 12 per char, twice the need.
    return min(12000, max(1200, 12 * len(text)))


def _codes_audio(tokens):
    """Generated tokens -> audio, the same way mlx-audio's Orpheus decodes them."""
    from mlx_audio.tts.models.llama.llama import decode_audio_from_codes
    if SOS in tokens:
        tokens = tokens[len(tokens) - tokens[::-1].index(SOS):]
    tokens = [t for t in tokens if t != EOS_SPEECH]
    codes = [t - CODE0 for t in tokens[: len(tokens) // 7 * 7]]
    if not codes:
        return np.zeros(0, dtype=np.float32)
    return np.array(decode_audio_from_codes(codes)[0], dtype=np.float32).flatten()


def _load_batched(path):
    """The same weights loaded through mlx-lm, whose batch generator can run several lines in one pass."""
    from mlx_lm import load
    from transformers import AutoTokenizer
    model, _ = load(path)
    return model, AutoTokenizer.from_pretrained(path)


def on_mac():
    return settings.load().get("orpheus_where", "mac") != "server"


def server_base():
    url = settings.load()["orpheus_url"].strip().rstrip("/")
    if not url:
        raise ValueError("Orpheus is set to use a server but has no address. Add it in Voices → Orpheus, "
                         "or switch it to run on this Mac.")
    return url[:-3] if url.endswith("/v1") else url


def _join(parts):
    """Sentences made separately, back into one line with an even, natural pause between them."""
    if len(parts) == 1:
        return parts[0]
    gap = np.zeros(int(24000 * SENTENCE_GAP), dtype=np.float32)
    out = []
    for k, p in enumerate(parts):
        out += [level(trim_edges(p))] + ([gap] if k < len(parts) - 1 else [])
    return np.concatenate(out)


class Orpheus(Engine):
    id, name, tier = "orpheus", "Orpheus 3B", "standard"
    short = "Orpheus"
    license = "Llama 3.2 Community License (attribution: Built with Llama)"
    prefixes = ("orpheus",)
    features = frozenset({"tags", "finetune", "remote"})
    hf_models = (STOCK,) if HAS_MLX else (orpheus_torch.STOCK, orpheus_torch.SNAC)
    cloud = True

    def available(self):
        if not on_mac():
            return True, ""
        if not HAS_MLX:
            why = torch_rt.why_not("transformers", "snac")
            return (False, why) if why else (True, "")
        return True, ""

    def note(self):
        return torch_rt.note() if on_mac() and not HAS_MLX else ""

    def __init__(self):
        self.tuned = finetuned()

    def refresh(self):
        self.tuned = finetuned()

    def voices(self):
        stock = [Voice(f"orpheus:{k}", n, g, d, self.id) for k, (n, g, d) in STOCK_VOICES.items()]
        tuned = [Voice(f"orpheus:{k}", n, g, d, self.id, ["fine-tuned"]) for k, (n, g, d, _, _) in self.tuned.items()]
        return stock + tuned

    def model_key(self, value):
        key = (value or "").partition(":")[2]
        return self.tuned[key][3] if key in self.tuned else STOCK

    def prepare(self, items, opts):
        if not on_mac():
            server_base()
            tuned = sorted({self.tuned[k][0] for k in (it["voice"].partition(":")[2] for it in items) if k in self.tuned})
            if tuned:
                raise ValueError(f"{', '.join(tuned)} only run on this Mac. In Voices → Orpheus, choose This Mac.")

    def speak(self, items, opts, stop=None):
        if not on_mac():
            with ThreadPoolExecutor(max_workers=max(1, int(opts.get("batch", 2)))) as pool:
                for i, wav in enumerate(pool.map(self._server_speak, items)):
                    yield i, wav
            return
        path = self.model_key(items[0]["voice"])
        if not HAS_MLX:
            yield from self._torch_speak(path, items, opts, stop)
            return
        # batching also for single lines, so switching between preview and full runs doesn't reload; the renderer
        # asks for one at a time when it remakes a line that came back silent
        # "parallel" comes from the Audiobook and Text to Speech switch (with its "lines at once"); callers that
        # don't say use the old setting from the Models page
        parallel = opts.get("parallel")
        if (batching() if parallel is None else parallel) and not opts.get("one_at_a_time"):
            n = BATCH if parallel is None else max(2, min(BATCH, int(opts.get("batch") or 2)))
            # one batch is sampled at one temperature: lines in a row with the same stability go together
            temps = [line_temperature(it, opts, 0.6) for it in items]
            start = 0
            while start < len(items):
                end = start
                while end < len(items) and temps[end] == temps[start]:
                    end += 1
                for i, wav in self._batch_speak(path, items[start:end], stop, n, temps[start]):
                    yield start + i, wav
                start = end
            return
        m = slot.get(f"orpheus:{path}", lambda: _load_model(path))
        for i, it in enumerate(items):
            if stop is not None and stop.is_set():
                return
            yield i, self._mac_speak(m, it, line_temperature(it, opts, 0.6))

    def _torch_speak(self, path, items, opts, stop):
        """Windows/Linux: lines in reading order, several at once when batching is on (one batch, one temperature)."""
        m = orpheus_torch.model(orpheus_torch.STOCK if path == STOCK else path)
        parallel = opts.get("parallel")
        on = (batching() if parallel is None else parallel) and not opts.get("one_at_a_time")
        n = max(1, min(BATCH, int(opts.get("batch") or 2))) if on and torch_rt.on_gpu() else 1
        temps = [line_temperature(it, opts, 0.6) for it in items]
        start = 0
        while start < len(items):
            if stop is not None and stop.is_set():
                return
            end = start + 1
            while end < len(items) and end - start < n and temps[end] == temps[start]:
                end += 1
            group = items[start:end]
            wavs = m.speak([(self._speaker(it), _text(it["text"], it.get("instruct"))) for it in group], temps[start])
            for k, wav in enumerate(wavs):
                yield start + k, wav
            torch_rt.free()
            start = end

    def _speaker(self, item):
        key = item["voice"].partition(":")[2]
        return self.tuned[key][4] if key in self.tuned else key

    def _batch_speak(self, path, items, stop=None, n=BATCH, temp=0.6):
        """Up to n lines at a time (BATCH at most); a finished line's place goes to the next one straight away.
        Same sampling settings as one at a time. Yields (index, audio) as each line finishes."""
        from mlx_lm.generate import BatchGenerator
        from mlx_lm.sample_utils import make_logits_processors, make_sampler
        model, tok = slot.get(f"orpheus-batch:{path}", lambda: _load_batched(path))
        units = []  # (item index, text); an emotion tag from the style box goes on a line's first sentence only
        for i, it in enumerate(items):
            parts = _sentences(it["text"]) if len(items) < n else []
            parts = parts if len(parts) > 1 else [it["text"]]
            units += [(i, _text(p, it.get("instruct")) if k == 0 else p) for k, p in enumerate(parts)]
        need = {i: sum(1 for j, _ in units if j == i) for i in range(len(items))}
        pieces = {i: {} for i in range(len(items))}
        texts = [t for _, t in units]
        prompts = [[128259] + tok(f"{self._speaker(items[i])}: {t}").input_ids + [EOT, 128260] for i, t in units]
        gen = BatchGenerator(model, stop_tokens=[[EOS_SPEECH], [EOT]], sampler=make_sampler(temp=temp, top_p=0.95),
                             completion_batch_size=n, prefill_batch_size=n)
        done = set()
        try:
            uids = gen.insert(prompts, [_max_tokens(t) for t in texts],
                              logits_processors=[make_logits_processors(repetition_penalty=1.1, repetition_context_size=20)
                                                 for _ in units])
            order = {u: n for n, u in enumerate(uids)}
            tokens = {u: [] for u in uids}
            while responses := gen.next_generated():
                if stop is not None and stop.is_set():
                    return
                for r in responses:
                    if r.finish_reason != "stop":
                        tokens[r.uid].append(r.token)
                    if r.finish_reason is not None:
                        n = order[r.uid]
                        i = units[n][0]
                        toks = tokens.pop(r.uid)
                        pieces[i][n] = _codes_audio(toks)
                        if is_silent(pieces[i][n]):  # details for the log, to find out why if it happens again
                            print(f"[orpheus batch] silent: {len(toks)} tokens ({sum(t >= CODE0 for t in toks)} audio), "
                                  f"ended by {r.finish_reason}, start {toks[:6]} · {units[n][1][:50]!r}", flush=True)
                        if len(pieces[i]) == need[i]:
                            done.add(i)
                            got = [pieces[i][k] for k in sorted(pieces[i])]
                            # a silent sentence inside a line: hand the line back empty so the renderer remakes it
                            yield i, (np.zeros(0, np.float32) if len(got) > 1 and any(map(is_silent, got))
                                      else _join(got))
            return
        except RuntimeError as e:  # on 8 GB the GPU can time out when memory is tight: finish one at a time
            if "Timeout" not in str(e):
                raise
        finally:
            gen.close()
        # Out here, not in the except block: while the error is being handled its traceback still holds the batch
        # generator and so the model. Let go of the batched copy before loading the one-line model, or both
        # (about 2 GB each) sit in memory at once.
        gen = model = tok = None
        slot.release()
        time.sleep(3)
        m = slot.get(f"orpheus:{path}", lambda: _load_model(path))
        for i, it in enumerate(items):
            if i not in done and not (stop is not None and stop.is_set()):
                yield i, self._mac_speak(m, it, temp)

    def _mac_speak(self, m, item, temp=0.6):
        speaker = self._speaker(item)
        text = _text(item["text"], item.get("instruct"))
        max_tokens = _max_tokens(text)
        for attempt in range(2):
            try:
                return np.concatenate([np.array(r.audio, dtype=np.float32).flatten()
                                       for r in m.generate(text=text, voice=speaker, verbose=False, max_tokens=max_tokens,
                                                           # tested: 0.6 at the default stability; mlx-audio's own
                                                           # defaults were slower
                                                           temperature=temp, top_p=0.95, repetition_penalty=1.1)])
            except RuntimeError as e:  # on 8 GB the GPU can time out when memory is tight; free and retry once
                if attempt or "Timeout" not in str(e):
                    raise
                free()
                time.sleep(3)

    def _server_speak(self, item):
        cfg = settings.load()
        headers = {"Authorization": f"Bearer {cfg['orpheus_key']}"} if cfg["orpheus_key"] else {}
        payload = {"model": cfg.get("orpheus_model") or "orpheus", "input": _text(item["text"], item.get("instruct")),
                   "voice": item["voice"].partition(":")[2], "response_format": "wav", "speed": 1.0}
        last = None
        for attempt in range(3):
            try:
                r = httpx.post(f"{server_base()}/v1/audio/speech", json=payload, headers=headers, timeout=900)
                if r.status_code == 200:
                    audio, sr = sf.read(io.BytesIO(r.content), dtype="float32", always_2d=True)
                    return resample(audio.mean(axis=1), sr)
                last = f"server said {r.status_code}: {r.text[:200]}"
                if r.status_code in (400, 401, 403, 404, 422):
                    break
            except httpx.HTTPError as e:
                last = f"couldn't reach the server ({type(e).__name__})"
            time.sleep(2 * (attempt + 1))
        raise ValueError(f"Orpheus failed: {last}")
