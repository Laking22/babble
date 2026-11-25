"""CLAP: how well a sound fits a description, so tags like {sfx: door slam} find the right sound by meaning
(not just by file name), and imported sounds get searchable tags.

Model: LAION larger_clap_general (Apache-2.0), the ONNX export by Xenova, 8-bit (about 205 MB), run with
onnxruntime on the CPU, so no PyTorch and no GPU memory taken from the voice models.
"""
import threading

import numpy as np

from .audio import resample

REPO = "Xenova/larger_clap_general"
FILES = ["onnx/audio_model_quantized.onnx", "onnx/text_model_quantized.onnx", "tokenizer.json",
         "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "config.json"]
SR, MAX_SAMPLES, HOP, NFFT = 48000, 480000, 480, 1024
_lock = threading.Lock()
_m = {}

# Words a sound can be tagged with automatically (common audiobook sounds). The best matches become its tags.
VOCAB = sorted(set("""rain thunder storm wind gust breeze sea ocean waves surf river stream water drip splash fire
fireplace campfire crackle forest birds crickets night insects owl wolf howl dog bark cat meow horse hooves gallop
footsteps steps running walking gravel snow leaves door creak slam knock knocking lock key gate chain metal clang
sword clash blade shield arrow bow gunshot gun explosion blast cannon glass shatter break crash thud impact punch
hit fall scream shout yell gasp laugh crowd tavern chatter applause cheering market city traffic car engine
train horn bell church clock ticking heartbeat breathing whisper cough sneeze crying baby phone ring typing paper
pages book wood creaking floor stairs ship boat creaking rope sail cave echo dungeon drip magic spell whoosh swoosh
swish swing whip rumble earthquake machine hum electric buzz alarm siren helicopter plane jet space laser beep
kitchen cooking sizzle pour cup plate cutlery eating drinking bottle cork coins money cash register office room
tone silence ambience music piano guitar drum""".split()))


def available():
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def downloaded():
    from huggingface_hub import try_to_load_from_cache
    return all(isinstance(try_to_load_from_cache(REPO, f), str) for f in FILES)


def download(log=print):
    from huggingface_hub import hf_hub_download
    log("Downloading the sound-matching model (about 205 MB)…")
    for f in FILES:
        hf_hub_download(REPO, f)
    log("Sound-matching model ready.")


def _load():
    with _lock:
        if not _m:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from transformers import AutoTokenizer
            from transformers.audio_utils import mel_filter_bank, window_function
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2  # leave the CPU free for everything else
            _m["audio"] = ort.InferenceSession(hf_hub_download(REPO, FILES[0]), opts, providers=["CPUExecutionProvider"])
            _m["text"] = ort.InferenceSession(hf_hub_download(REPO, FILES[1]), opts, providers=["CPUExecutionProvider"])
            _m["tok"] = AutoTokenizer.from_pretrained(REPO)
            _m["mel"] = mel_filter_bank(num_frequency_bins=NFFT // 2 + 1, num_mel_filters=64, min_frequency=50,
                                        max_frequency=14000, sampling_rate=SR, norm="slaney", mel_scale="slaney")
            _m["win"] = window_function(NFFT, "hann")
        return _m


def _features(wav48):
    """CLAP's input: a 10 s log-mel spectrogram, short sounds repeated to fill it (as in ClapFeatureExtractor)."""
    from transformers.audio_utils import spectrogram
    m = _load()
    if len(wav48) < MAX_SAMPLES:
        wav48 = np.tile(wav48, MAX_SAMPLES // max(1, len(wav48)))
        wav48 = np.pad(wav48, (0, MAX_SAMPLES - len(wav48)))
    mel = spectrogram(wav48.astype(np.float64), m["win"], frame_length=NFFT, hop_length=HOP, power=2.0,
                      mel_filters=m["mel"], log_mel="dB").T
    return mel[None, None].astype(np.float32)  # (1, 1, frames, 64)


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def audio_embedding(wav, sr=24000):
    """One vector for a sound. Long sounds (ambience) are averaged over up to three 10 s windows."""
    m = _load()
    a = resample(np.asarray(wav, dtype=np.float32), sr, SR)
    if len(a) <= MAX_SAMPLES:
        windows = [a]
    else:
        starts = np.linspace(0, len(a) - MAX_SAMPLES, 3).astype(int)
        windows = [a[s:s + MAX_SAMPLES] for s in starts]
    vs = [m["audio"].run(None, {"input_features": _features(w)})[0][0] for w in windows]
    return _unit(np.mean([_unit(v) for v in vs], axis=0)).astype(np.float32)


def text_embedding(texts):
    m = _load()
    ids = m["tok"](list(texts), padding=True, return_tensors="np")["input_ids"].astype(np.int64)
    return _unit(m["text"].run(None, {"input_ids": ids})[0]).astype(np.float32)


_vocab = {}


def auto_tags(emb, k=4):
    """The k vocabulary words that best describe a sound."""
    if "v" not in _vocab:
        _vocab["v"] = text_embedding([f"the sound of {w}" for w in VOCAB])
    scores = _vocab["v"] @ emb
    return [VOCAB[i] for i in np.argsort(-scores)[:k]]
