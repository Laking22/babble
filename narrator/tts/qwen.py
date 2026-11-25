"""Qwen3-TTS: built-in speakers, saved (designed / cloned) voices, fine-tuned models, voice design, exact cloning.

MLX on a Mac; PyTorch on Windows and Linux (qwen_torch.py, the same models in their official form)."""
import json

import numpy as np
try:
    import mlx.core as mx
except ImportError:
    mx = None


def _load_model(path):
    from mlx_audio.tts.utils import load_model
    return load_model(path)

from .. import voices_lib as lib
from ..paths import MODELS as MODELS_DIR
from ..audio import flat, read_mono
from ..runtime import slot
from .base import Engine, Voice
from . import qwen_torch, torch_rt

MODELS = {
    "speakers": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-6bit",
    "clone": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-6bit",
    "design": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-6bit",
}

# key: (display name, gender, description)
BUILTINS = {
    "ryan": ("Ryan", "male", "English, dynamic"),
    "aiden": ("Aiden", "male", "American English, bright"),
    "eric": ("Eric", "male", "Chinese (Sichuan) accent, lively"),
    "dylan": ("Dylan", "male", "Chinese (Beijing) accent, youthful"),
    "uncle_fu": ("Uncle Fu", "male", "Chinese accent, deep and mature"),
    "serena": ("Serena", "female", "Chinese accent, warm and gentle"),
    "vivian": ("Vivian", "female", "Chinese accent, bright and young"),
    "sohee": ("Sohee", "female", "Korean accent, warm"),
    "ono_anna": ("Ono Anna", "female", "Japanese accent, playful"),
}

# Saved voices ride on the built-in speaker model: built-in speakers are 2048-number voice fingerprints,
# and the cloning model's speaker encoder makes fingerprints of the same size. As consistent as cloning,
# much faster, and it takes emotions.
_SENTINEL = 900_000
_saved = {"ids": {}, "vectors": {}}
slot.on_release(lambda: (_saved["ids"].clear(), _saved["vectors"].clear()))



def _installed_order(meta):
    st = meta.parent.stat()
    return (getattr(st, "st_birthtime", st.st_mtime), meta.parent.name)


def finetuned():
    """key -> (display name, gender, description, model folder, trained speaker name) for paths.MODELS/*/voice.json
    with engine "qwen" (installed by the Qwen training recipe)."""
    out = {}
    # oldest install first, so it keeps the plain key and a newer model can't take over an existing voice
    for meta in sorted(MODELS_DIR.glob("*/voice.json"), key=_installed_order):
        try:
            data = json.loads(meta.read_text())
        except ValueError:
            continue
        if data.get("engine") != "qwen":
            continue
        for speaker, v in data.get("voices", {}).items():
            key = speaker if speaker not in out else f"{speaker}-{meta.parent.name.split('-')[0]}"
            out[key] = (v.get("name", speaker.title()), v.get("gender", "unspecified"),
                        v.get("description", "fine-tuned"), str(meta.parent), speaker)
    return out


def model(kind):
    m = slot.get(f"qwen:{kind}", lambda: _load_model(MODELS[kind]))
    if kind == "speakers" and not getattr(m, "_saved_voice_support", False):
        _install_saved_voice_support(m)
    return m


def _install_saved_voice_support(m):
    table = m.talker.get_input_embeddings()

    class WithSavedVoices:
        def __call__(self, ids):
            if not _saved["vectors"] or ids.size > 64 or int(ids.max().item()) < _SENTINEL:
                return table(ids)
            out = table(mx.where(ids >= _SENTINEL, 0, ids))
            for sentinel, vec in _saved["vectors"].items():
                out = mx.where((ids == sentinel)[..., None], vec.astype(out.dtype), out)
            return out

        def __getattr__(self, attr):
            return getattr(table, attr)

    wrapped = WithSavedVoices()
    m.talker.get_input_embeddings = lambda: wrapped
    m._saved_voice_support = True
    _saved["ids"].clear()
    _saved["vectors"].clear()


def fingerprints(names):
    """Make sure every saved voice in `names` has a cached fingerprint (.npy)."""
    todo = []
    for name in names:
        cache, wav = lib.fingerprint_path(name), lib.wav_path(name)
        if not (cache.exists() and cache.stat().st_mtime >= wav.stat().st_mtime):
            todo.append(name)
    if todo:
        base = model("clone")
        for name in todo:
            vec = np.array(base.extract_speaker_embedding(mx.array(read_mono(lib.wav_path(name), max_seconds=20))))
            np.save(lib.fingerprint_path(name), vec.reshape(-1))


def speaker_key(m, value):
    """Turn 'builtin:ryan' / 'my:Nova' into a speaker the model accepts."""
    kind, _, key = (value or "").partition(":")
    if kind == "builtin" and key in BUILTINS:
        return key
    if kind == "my" and lib.wav_path(key).exists():
        spk = f"saved_{key.lower()}"
        if spk not in _saved["ids"]:
            _saved["ids"][spk] = _SENTINEL + len(_saved["ids"])
        sentinel = _saved["ids"][spk]
        _saved["vectors"][sentinel] = mx.array(np.load(lib.fingerprint_path(key))).reshape(1, -1)
        m.config.talker_config.spk_id[spk] = sentinel
        if spk not in m.supported_speakers:
            m.supported_speakers.append(spk)
        return spk
    raise ValueError(f"Unknown voice “{value}”.")


class Qwen(Engine):
    id, name, tier, license = "qwen", "Qwen3-TTS 1.7B", "standard", "Apache-2.0 (check each model card)"
    short = "Qwen"
    prefixes = ("builtin", "my", "qwen")   # qwen: = fine-tuned models
    features = frozenset({"emotion", "clone", "design", "batch", "finetune"})
    hf_models = tuple(MODELS.values()) if mx is not None else tuple(qwen_torch.MODELS.values())
    cloud = True

    def available(self):
        if mx is None:
            why = torch_rt.why_not("qwen_tts")
            return (False, why) if why else (True, "")
        return True, ""

    def note(self):
        return torch_rt.note() if mx is None else ""

    def __init__(self):
        self.tuned = finetuned()

    def refresh(self):
        self.tuned = finetuned()

    def voices(self):
        mine = [Voice(f"my:{v['name']}", v["name"], v["gender"], v.get("source", "saved"), self.id, ["saved"])
                for v in lib.library()]
        built = [Voice(f"builtin:{k}", n, g, d, self.id) for k, (n, g, d) in BUILTINS.items()]
        tuned = [Voice(f"qwen:{k}", n, g, d, self.id, ["fine-tuned"]) for k, (n, g, d, _, _) in self.tuned.items()]
        return mine + built + tuned

    def model_key(self, value):
        kind, _, key = (value or "").partition(":")
        if kind == "qwen":
            if key not in self.tuned:
                raise ValueError(f"Unknown voice “{value}”.")
            return f"qwen:{self.tuned[key][3]}"
        return "qwen:speakers"

    def prepare(self, items, opts):
        if opts.get("exact") or mx is None:  # (PyTorch clones saved voices from their clip: no fingerprints)
            return
        items = [it for it in items if not it["voice"].startswith("qwen:")]
        fingerprints(sorted({it["voice"][3:] for it in items if it["voice"].startswith("my:")}))

    def speak(self, items, opts, stop=None):
        tags = opts.get("strip_tags")
        clean = (lambda t: tags.sub("", t)) if tags else (lambda t: t)
        gen = dict(lang_code=opts.get("language", "auto"), temperature=opts.get("temperature", 0.9),
                   top_p=opts.get("top_p", 1.0), top_k=int(opts.get("top_k", 50)),
                   repetition_penalty=opts.get("repetition_penalty", 1.05), verbose=False)
        if mx is None:
            yield from qwen_torch.speak(self, items, opts, stop, clean, gen, _batches, max_tokens)
            return
        exact = opts.get("exact")
        if exact:  # full transcript cloning: slower, no emotions, one line at a time
            info = lib.info(exact)
            if not info["transcript"]:
                raise ValueError("Exact cloning needs the voice's transcript. Add it in the Voices tab.")
            m = model("clone")
            for i, it in enumerate(items):
                if stop is not None and stop.is_set():
                    return
                yield i, flat(m.generate(clean(it["text"]), ref_audio=str(lib.wav_path(exact)),
                                         ref_text=info["transcript"], max_tokens=max_tokens([clean(it["text"])]),
                                         **_with_temp(gen, [it])))
            return
        if items[0]["voice"].startswith("qwen:"):
            yield from self._speak_tuned(items, gen, clean, stop, opts)
            return
        m = model("speakers")
        keys = {it["voice"]: speaker_key(m, it["voice"]) for it in items}
        batch = 1 if opts.get("one_at_a_time") else max(1, int(opts.get("batch", 2)))
        for start, group in _batches(items, batch):
            if stop is not None and stop.is_set():
                return
            g_opts = dict(_with_temp(gen, group), max_tokens=max_tokens([clean(g["text"]) for g in group]))
            if len(group) == 1:
                g = group[0]
                yield start, flat(m.generate(clean(g["text"]), voice=keys[g["voice"]], instruct=g.get("instruct"), **g_opts))
                continue
            outs = {}
            for r in m.batch_generate(texts=[clean(g["text"]) for g in group], voices=[keys[g["voice"]] for g in group],
                                      instructs=[g.get("instruct") for g in group], **g_opts):
                outs.setdefault(r.sequence_idx, []).append(np.array(r.audio, dtype=np.float32).flatten())
            for n in range(len(group)):
                yield start + n, np.concatenate(outs[n])
            mx.clear_cache()


    def _speak_tuned(self, items, gen, clean, stop, opts):
        """Fine-tuned CustomVoice model: prompt with the trained speaker name. It was trained without a language
        tag or style instructions, so neither is passed (an instruction would be read as unfamiliar input)."""
        folder = self.model_key(items[0]["voice"]).partition(":")[2]
        m = slot.get(f"qwen:{folder}", lambda: _load_model(folder))
        gen = dict(gen, lang_code="auto")
        speaker = {it["voice"]: self.tuned[it["voice"].partition(":")[2]][4] for it in items}
        batch = 1 if opts.get("one_at_a_time") else max(1, int(opts.get("batch", 2)))
        for start, group in _batches(items, batch):
            if stop is not None and stop.is_set():
                return
            g_opts = dict(_with_temp(gen, group), max_tokens=max_tokens([clean(g["text"]) for g in group]))
            if len(group) == 1:
                yield start, flat(m.generate(clean(group[0]["text"]), voice=speaker[group[0]["voice"]], **g_opts))
                continue
            outs = {}
            for r in m.batch_generate(texts=[clean(g["text"]) for g in group], voices=[speaker[g["voice"]] for g in group],
                                      instructs=[None] * len(group), **g_opts):
                outs.setdefault(r.sequence_idx, []).append(np.array(r.audio, dtype=np.float32).flatten())
            for n in range(len(group)):
                yield start + n, np.concatenate(outs[n])
            mx.clear_cache()


def max_tokens(texts):
    """The most a line may take, so a model that can't stop (a fine-tune that never learned to end a line) is cut
    off after seconds, not after mlx-audio's 2048 steps (~2.7 min of audio, minutes to make on a small Mac).
    Qwen3-TTS makes 12.5 steps a second and narrators read ~14 characters a second (~0.9 steps a character):
    twice that, and never under 12 s (a short "Mmm…" can be drawn out)."""
    def _steps_for(text):
        cjk = sum(1 for ch in text if "\u3040" <= ch <= "\u9fff" or "\uac00" <= ch <= "\ud7af")
        other = len(text) - cjk
        return int(1.8 * other + 5.0 * cjk)

    longest = max((_steps_for(t) for t in texts), default=0)
    return min(2048, max(150, longest))


def _batches(items, size):
    """Lines in reading order, in groups of up to `size` that share a temperature: a character can have its own
    stability, and one batch is made with one temperature."""
    start = 0
    while start < len(items):
        t = items[start].get("temperature")
        end = start + 1
        while end < len(items) and end - start < size and items[end].get("temperature") == t:
            end += 1
        yield start, items[start:end]
        start = end


def _with_temp(gen, group):
    t = group[0].get("temperature")
    return dict(gen, temperature=t) if t is not None else gen


def design_takes(description, text, language, count=3, seed=0, stop=None, out_dir=None):
    """Voice Design: several takes of the same description. Yields the list of paths so far."""
    import soundfile as sf
    from datetime import datetime
    from ..paths import OUTPUTS
    from ..runtime import SR
    m = model("design") if mx is not None else None
    paths = []
    for i in range(count):
        if stop is not None and stop.is_set():
            break
        if seed:
            (mx.random.seed if mx is not None else torch_rt.seed)(int(seed) + i)
        if m is None:
            wav = qwen_torch.design(description, text, language, max_tokens([text]))
        else:
            wav = flat(m.generate_voice_design(text=text, instruct=description, language=language,
                                               max_tokens=max_tokens([text])))
        path = (out_dir or OUTPUTS) / f"design-take{i + 1}-{datetime.now():%H%M%S}.wav"
        sf.write(path, wav, SR)
        paths.append(str(path))
        yield paths
