"""Qwen3-TTS through PyTorch, for Windows and Linux (tts/qwen.py uses it when there's no MLX).

The official qwen-tts package, called as the cloud calls it (kaggle_render/runner.py Qwen, lines_runner.py): built-in
speakers with the CustomVoice model, saved voices cloned by the Base model from their sample clip (its voice
fingerprint only, so no style directions, as in the cloud), exact cloning from clip + transcript, Voice Design, and
trained voice packs (unpacked once: torch_rt.unpacked) played like the Mac plays them: no direction, no language tag.
"""
import numpy as np

from .. import voices_lib as lib
from ..audio import resample
from ..runtime import slot
from . import torch_rt

MODELS = {
    "speakers": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "clone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}
STOCK = "Qwen/Qwen3-TTS-12Hz-{size}-CustomVoice"   # what a trained pack takes its unchanged files from
_prompts = {}
slot.on_release(_prompts.clear)


def _load(repo):
    from qwen_tts import Qwen3TTSModel
    return Qwen3TTSModel.from_pretrained(str(repo), device_map=torch_rt.device(), dtype=torch_rt.dtype(),
                                         attn_implementation="sdpa")


def model(kind):
    return slot.get(f"qwen-torch:{kind}", lambda: _load(MODELS[kind]))


def language(opts):
    lang = (opts.get("language") or "auto").lower()
    return "Auto" if lang in ("auto", "") else lang.capitalize()


def sampling(gen, group, max_new):
    t = group[0].get("temperature")
    return {"temperature": float(t if t is not None else gen["temperature"]), "top_p": gen["top_p"],
            "top_k": gen["top_k"], "repetition_penalty": gen["repetition_penalty"], "do_sample": True,
            "max_new_tokens": max_new}


def _out(wavs, sr):
    return [resample(np.asarray(w, dtype=np.float32).reshape(-1), sr) for w in wavs]


def pack_folder(folder, log=print):
    import json
    from pathlib import Path
    size = {"0b6": "0.6B", "1b7": "1.7B"}.get(json.loads((Path(folder) / "config.json").read_text())
                                              .get("tts_model_size"), "1.7B")
    return torch_rt.unpacked(folder, STOCK.format(size=size), log)


def speak(eng, items, opts, stop, clean, gen, batches, max_tokens):
    """The same lines and order as Qwen.speak on the Mac; `batches` and `max_tokens` are its own helpers."""
    lang = language(opts)
    exact = opts.get("exact")
    if exact:
        info = lib.info(exact)
        if not info["transcript"]:
            raise ValueError("Exact cloning needs the voice's transcript. Add it in the Voices tab.")
        m = model("clone")
        for i, it in enumerate(items):
            if stop is not None and stop.is_set():
                return
            text = clean(it["text"])
            wavs, sr = m.generate_voice_clone(text=[text], language=[lang], ref_audio=str(lib.wav_path(exact)),
                                              ref_text=info["transcript"], x_vector_only_mode=False,
                                              **sampling(gen, [it], max_tokens([text])))
            yield i, _out(wavs, sr)[0]
        return
    voice = items[0]["voice"]
    batch = 1 if opts.get("one_at_a_time") else max(1, int(opts.get("batch", 2)))
    if voice.startswith("qwen:"):
        from .. import tasks
        folder = eng.model_key(voice).partition(":")[2]
        m = slot.get(f"qwen-torch:{folder}", lambda: _load(pack_folder(folder, tasks.note)))
        speaker = {it["voice"]: eng.tuned[it["voice"].partition(":")[2]][4] for it in items}
        for start, group in batches(items, batch):
            if stop is not None and stop.is_set():
                return
            texts = [clean(g["text"]) for g in group]
            wavs, sr = m.generate_custom_voice(text=texts, speaker=[speaker[g["voice"]] for g in group],
                                               language=["Auto"] * len(group), instruct=[""] * len(group),
                                               **sampling(gen, group, max_tokens(texts)))
            for n, w in enumerate(_out(wavs, sr)):
                yield start + n, w
            torch_rt.free()
        return
    if voice.startswith("my:"):
        m = model("clone")
        for start, group in batches(items, batch):
            if stop is not None and stop.is_set():
                return
            names = [g["voice"][3:] for g in group]
            for name in names:
                if name not in _prompts:
                    _prompts[name] = m.create_voice_clone_prompt(ref_audio=str(lib.wav_path(name)), ref_text=None,
                                                                 x_vector_only_mode=True)[0]
            texts = [clean(g["text"]) for g in group]
            wavs, sr = m.generate_voice_clone(text=texts, language=[lang] * len(group),
                                              voice_clone_prompt=[_prompts[n] for n in names],
                                              **sampling(gen, group, max_tokens(texts)))
            for n, w in enumerate(_out(wavs, sr)):
                yield start + n, w
            torch_rt.free()
        return
    m = model("speakers")
    for start, group in batches(items, batch):
        if stop is not None and stop.is_set():
            return
        texts = [clean(g["text"]) for g in group]
        wavs, sr = m.generate_custom_voice(text=texts, speaker=[g["voice"].partition(":")[2] for g in group],
                                           language=[lang] * len(group),
                                           instruct=[g.get("instruct") or "" for g in group],
                                           **sampling(gen, group, max_tokens(texts)))
        for n, w in enumerate(_out(wavs, sr)):
            yield start + n, w
        torch_rt.free()


def design(description, text, language_name, max_new):
    m = model("design")
    lang = "Auto" if (language_name or "auto").lower() == "auto" else language_name
    wavs, sr = m.generate_voice_design(text=[text], instruct=[description], language=[lang], max_new_tokens=max_new)
    return _out(wavs, sr)[0]
