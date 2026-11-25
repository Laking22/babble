"""Run: .venv/bin/python tests/pc_voices_check.py  (a few seconds, no models, no network, scratch data folder only)

The voices on Windows and Linux (PyTorch), checked on a Mac: MLX is hidden and PyTorch, qwen-tts, Kokoro and SNAC are
stand-ins that record how they're called (the real ones are tested on GitHub's Linux machines: tests/pc_voices_live.py).
1. pc_setup: Windows/Linux packages are requirements.txt without its Mac-only lines plus requirements-pc.txt, some
   without their dependency lists; the PyTorch build for the graphics card; the same stamp the app keeps.
2. Without PyTorch the engines say to set up voices; with it they can run, and say when there's no NVIDIA card.
3. Qwen: built-in speakers, saved voices, exact cloning, Voice Design and trained packs, called as the cloud calls them.
   Orpheus and Kokoro speak through their PyTorch code. Higgs and VibeVoice say where they run instead.
4. A trained pack is unpacked once, with the stock model's other files and never its weights.
5. Set up voices: installs the right PyTorch build with uv, from the Models page only (not by AI agents).
"""
import importlib.abc
import json
import os
import stat
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

TMP = Path(tempfile.mkdtemp(prefix="pcvoices-"))
os.environ["NARRATOR_HOME"] = str(TMP / "home")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
failed = []


def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        failed.append(what)


class NoMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in ("mlx", "mlx_audio", "mlx_lm"):
            raise ImportError(f"no {name} here (a Windows/Linux PC)")
        return None


sys.meta_path.insert(0, NoMLX())

# ---------- stand-ins ----------
FAKES = TMP / "fakes"
calls = []


def fake(name, code):
    d = FAKES / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "__init__.py").write_text(code)


def install_fakes():
    fake("torch", '''
bfloat16, float16, float32 = "bf16", "fp16", "fp32"
class cuda:
    @staticmethod
    def is_available(): return False
    @staticmethod
    def is_bf16_supported(): return False
    @staticmethod
    def empty_cache(): pass
def manual_seed(n): CALLS.append(("seed", n))
CALLS = []
''')
    fake("qwen_tts", '''
import numpy as np
CALLS = []
class _T:
    def __init__(self, v): self.v = v
    def detach(self): return self
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return np.asarray(self.v, dtype=np.float32)
class _Prompt:
    def __init__(self, src): self.ref_spk_embedding, self.src = _T([1.0, 2.0, 3.0]), src
def _wav(n): return [np.sin(np.linspace(0, 400, 48000)).astype(np.float32) * 0.5 for _ in range(n)], 24000
class Qwen3TTSModel:
    @classmethod
    def from_pretrained(cls, repo, **kw):
        CALLS.append(("load", repo, kw)); m = cls(); m.repo = repo; return m
    def generate_custom_voice(self, text, speaker, language=None, instruct=None, **kw):
        CALLS.append(("custom", self.repo, list(text), list(speaker), list(language), list(instruct), kw)); return _wav(len(text))
    def create_voice_clone_prompt(self, ref_audio, ref_text=None, x_vector_only_mode=False):
        CALLS.append(("prompt", ref_audio if isinstance(ref_audio, str) else "array", x_vector_only_mode)); return [_Prompt(ref_audio)]
    def generate_voice_clone(self, text, language=None, ref_audio=None, ref_text=None, x_vector_only_mode=False, voice_clone_prompt=None, **kw):
        CALLS.append(("clone", self.repo, list(text), ref_text, x_vector_only_mode, len(voice_clone_prompt or []), kw)); return _wav(len(text))
    def generate_voice_design(self, text, instruct, language=None, **kw):
        CALLS.append(("design", list(text), list(instruct), list(language), kw)); return _wav(len(text))
''')
    fake("snac", "class SNAC: pass\n")
    fake("kokoro", '''
import numpy as np
CALLS = []
class KModel:
    def __init__(self, repo_id=None): CALLS.append(("model", repo_id))
    def to(self, d): return self
    def eval(self): return self
class KPipeline:
    def __init__(self, lang_code, repo_id=None, model=None): CALLS.append(("pipe", lang_code, repo_id))
    def __call__(self, text, voice=None, speed=1.0):
        CALLS.append(("say", voice, text))
        yield ("g", "p", np.ones(12000, dtype=np.float32) * 0.2)
''')
    sys.path.insert(0, str(FAKES))


# 1. pc_setup
from narrator import pc_setup  # noqa: E402
normal, nodeps = pc_setup.requirements()
check(not any(r.startswith(("mlx", "transformers==5")) for r in normal) and "transformers==4.57.3" in normal,
      "Windows/Linux packages: no MLX, and the transformers qwen-tts is made for")
check({r.split("==")[0] for r in nodeps} == {"accelerate", "snac", "qwen-tts", "kokoro"},
      "packages that name torch go in without their dependency lists")
check(not any(r.startswith(("torch==", "torchaudio==")) for r in normal + nodeps), "PyTorch isn't among them")
check(any(r.startswith("en_core_web_sm @ https://") for r in normal),
      "Kokoro's English text model is among them (misaki can't install it without pip)")
check(pc_setup.build({"name": "RTX 4070", "cap": 8.9}) == "cu128" and pc_setup.build({"name": "GTX 1080", "cap": 6.1})
      == "cu126" and pc_setup.build(None) == "cpu", "the PyTorch build follows the graphics card")
check(pc_setup.torch_packages("cu128") == ["torch==2.11.0+cu128", "torchaudio==2.11.0+cu128"],
      "PyTorch is pinned with its build, so a different build isn't taken as already installed")
check(pc_setup.stamp() == (ROOT / "requirements.txt").read_text() + "\n" + (ROOT / "requirements-pc.txt").read_text(),
      "the stamp is the text the app keeps (lib.rs requirements_stamp)")

# 2. without PyTorch
from narrator import tts  # noqa: E402
from narrator.runtime import HAS_MLX  # noqa: E402
from narrator.tts import torch_rt  # noqa: E402
check(not HAS_MLX, "(MLX hidden: this is a Windows/Linux PC for the test)")
real_sys = torch_rt.sys
torch_rt.sys = types.SimpleNamespace(platform="win32", modules=sys.modules)
ok, why = tts.ENGINES["qwen"].available()
check(not ok and "Set up voices" in why, f"without PyTorch Qwen says to set up voices ({why})")
check(not tts.ENGINES["kokoro"].available()[0] and not tts.ENGINES["orpheus"].available()[0], "…so do Kokoro and Orpheus")
check("cloud" in tts.ENGINES["higgs"].available()[1] and "Macs only" in tts.ENGINES["vibevoice"].available()[1],
      "Higgs and VibeVoice say where they run")
check(tts.ENGINES["qwen"].hf_models[0].startswith("Qwen/") and tts.ENGINES["kokoro"].hf_models == ("hexgrad/Kokoro-82M",)
      and tts.ENGINES["orpheus"].hf_models[0] == "unsloth/orpheus-3b-0.1-ft",
      "the Models page lists the PyTorch models, not the MLX copies")

# 3. with PyTorch
install_fakes()
import qwen_tts  # noqa: E402
ok, why = tts.ENGINES["qwen"].available()
check(ok and all(tts.ENGINES[e].available()[0] for e in ("kokoro", "orpheus")), f"with PyTorch they can run ({why})")
check("No NVIDIA" in tts.ENGINES["qwen"].note(), "and say so when there's no NVIDIA card")
torch_rt.sys = real_sys

q = tts.ENGINES["qwen"]
items = [{"voice": "builtin:ryan", "text": "One.", "instruct": "warm"}, {"voice": "builtin:serena", "text": "Two."},
         {"voice": "builtin:ryan", "text": "Three.", "temperature": 0.7}]
out = list(q.speak(items, {"language": "english", "temperature": 0.9, "top_p": 1.0, "top_k": 50,
                           "repetition_penalty": 1.05, "batch": 2}))
custom = [c for c in qwen_tts.CALLS if c[0] == "custom"]
check([i for i, _ in out] == [0, 1, 2] and all(len(w) == 48000 for _, w in out), "built-in speakers: every line, in order")
check(custom[0][3] == ["ryan", "serena"] and custom[0][4] == ["English", "English"] and custom[0][5] == ["warm", ""]
      and custom[1][6]["temperature"] == 0.7 and custom[0][2] == ["One.", "Two."],
      "…batched by stability, with their directions and the language as Qwen names it")
check(any(c[0] == "load" and c[1] == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" and c[2]["device_map"] == "cpu" for c in
          qwen_tts.CALLS), "…with the official CustomVoice model on this computer's device")

from narrator import voices_lib as lib  # noqa: E402
import soundfile as sf  # noqa: E402
lib.VOICES.mkdir(parents=True, exist_ok=True)
sf.write(lib.wav_path("Nova"), np.zeros(24000, np.float32), 24000)
lib.wav_path("Nova").with_suffix(".txt").write_text("Hello there.")
qwen_tts.CALLS.clear()
q.prepare([{"voice": "my:Nova", "text": "x"}], {})
out = list(q.speak([{"voice": "my:Nova", "text": "Saved one."}, {"voice": "my:Nova", "text": "Saved two."}],
                   {"temperature": 0.9, "top_p": 1.0, "top_k": 50, "repetition_penalty": 1.05}))
clone = [c for c in qwen_tts.CALLS if c[0] == "clone"]
check(len(out) == 2 and [c[0] for c in qwen_tts.CALLS].count("prompt") == 1 and clone and clone[0][4] is False
      and clone[0][5] == 2 and clone[0][1].endswith("Base"),
      "saved voices: cloned from their clip by the Base model, the voice worked out once")
qwen_tts.CALLS.clear()
out = list(q.speak([{"voice": "builtin:ryan", "text": "Exact."}], {"exact": "Nova", "temperature": 0.9, "top_p": 1.0,
                                                                     "top_k": 50, "repetition_penalty": 1.05}))
clone = [c for c in qwen_tts.CALLS if c[0] == "clone"]
check(len(out) == 1 and clone[0][3] == "Hello there." and clone[0][4] is False, "exact cloning uses the clip's transcript")
from narrator.tts import qwen as qwen_mod  # noqa: E402
qwen_tts.CALLS.clear()
paths = list(qwen_mod.design_takes("a calm old sailor", "Ahoy.", "English", count=2, seed=5, out_dir=TMP))
design = [c for c in qwen_tts.CALLS if c[0] == "design"]
import torch  # noqa: E402
check(len(paths[-1]) == 2 and len(design) == 2 and design[0][2] == ["a calm old sailor"] and ("seed", 6) in torch.CALLS,
      "Voice Design: takes from the VoiceDesign model, seeded")

pack = lib.VOICES.parent / "models" / "crew-qwen-6bit"
pack.mkdir(parents=True)
(pack / "config.json").write_text(json.dumps({"quantization": {"bits": 6, "group_size": 64}, "tts_model_size": "1b7"}))
(pack / "voice.json").write_text(json.dumps({"engine": "qwen", "voices": {"arthur": {"name": "Arthur"}}}))
(pack / "model.safetensors").write_bytes(b"mlx")
q.refresh()
seen = {}
from narrator.tts import qwen_torch  # noqa: E402
real_unpacked = torch_rt.unpacked
torch_rt.unpacked = lambda folder, stock, log=print: seen.setdefault("args", (folder, stock)) and TMP / "unpacked"
qwen_tts.CALLS.clear()
out = list(q.speak([{"voice": "qwen:arthur", "text": "Packed.", "instruct": "angry"}],
                   {"language": "english", "temperature": 0.9, "top_p": 1.0, "top_k": 50, "repetition_penalty": 1.05}))
custom = [c for c in qwen_tts.CALLS if c[0] == "custom"]
check(seen["args"] == (str(pack), "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice") and custom and custom[0][3] == ["arthur"]
      and custom[0][4] == ["Auto"] and custom[0][5] == [""] and custom[0][1] == str(TMP / "unpacked"),
      "a trained pack: unpacked, its trained speaker, no direction or language tag (as the Mac plays it)")
torch_rt.unpacked = real_unpacked

from narrator.tts import orpheus_torch  # noqa: E402
said = []


class FakeOrpheus:
    def speak(self, prompts, temperature):
        said.append((prompts, temperature))
        return [np.ones(24000, np.float32) * 0.3 for _ in prompts]


orpheus_torch.model = lambda path: said.append(("model", path)) or FakeOrpheus()
o = tts.ENGINES["orpheus"]
out = list(o.speak([{"voice": "orpheus:tara", "text": "Ha, that's funny.", "instruct": "laughing"},
                    {"voice": "orpheus:leo", "text": "Not really."}], {"parallel": True, "batch": 2}))
check(len(out) == 2 and said[0] == ("model", "unsloth/orpheus-3b-0.1-ft") and said[1][0][0] == ("tara", "<laugh> Ha, that's funny.")
      and len(said[1][0]) == 1, "Orpheus: stock model, emotion as a sound tag, one line at a time on the processor")

import kokoro  # noqa: E402
import spacy.util  # noqa: E402
is_package = spacy.util.is_package
# (spaCy's English model comes from requirements-pc.txt: GitHub's test machines install only requirements.txt)
spacy.util.is_package = lambda name: True
try:
    out = list(tts.ENGINES["kokoro"].speak([{"voice": "kokoro:bf_emma", "text": "Lovely."}], {}))
    check(len(out) == 1 and len(out[0][1]) == 12000 and ("pipe", "b", "hexgrad/Kokoro-82M") in kokoro.CALLS,
          "Kokoro: its own package, the British pipeline for a British voice")
    spacy.util.is_package = lambda name: False
    try:
        list(tts.ENGINES["kokoro"].speak([{"voice": "kokoro:af_heart", "text": "Hi."}], {}))
        msg = ""
    except ValueError as e:
        msg = str(e)
finally:
    spacy.util.is_package = is_package
check("en_core_web_sm" in msg, "Kokoro says so when spaCy's English model is missing, instead of never finishing")

from narrator import voice_check  # noqa: E402
fp = voice_check._embed(np.zeros(24000, np.float32))
check(list(fp) == [1.0, 2.0, 3.0], "the cloud voice check gets voice fingerprints from the same Base model")

# 4. unpacking a trained pack
src = TMP / "packs" / "v-qwen-6bit"
src.mkdir(parents=True)
(src / "config.json").write_text(json.dumps({"quantization": {"bits": 6}}))
(src / "model.safetensors").write_bytes(b"packed")
stock = TMP / "stock"
(stock / "speech_tokenizer").mkdir(parents=True)
(stock / "speech_tokenizer" / "model.safetensors").write_bytes(b"codec")
(stock / "merges.txt").write_text("m")
(stock / "model.safetensors").write_bytes(b"STOCK WEIGHTS")
(stock / "tokenizer_config.json").write_text("{}")
runs = []


class Unpacker:
    @staticmethod
    def main(s, d):
        runs.append(s)
        d.mkdir(parents=True, exist_ok=True)
        (d / "model-00001-of-00001.safetensors").write_bytes(b"16bit")
        (d / "model.safetensors.index.json").write_text("{}")
        (d / "config.json").write_text("{}")


torch_rt._unpacker = lambda: Unpacker
import huggingface_hub  # noqa: E402
huggingface_hub.snapshot_download = lambda repo, **kw: str(stock)
d = torch_rt.unpacked(src, "Qwen/x", log=lambda *a: None)
names = sorted(f.name for f in d.iterdir())
check("model.safetensors" not in names and "model-00001-of-00001.safetensors" in names and "merges.txt" in names
      and (d / "speech_tokenizer" / "model.safetensors").read_bytes() == b"codec",
      "a pack is unpacked with the stock model's other files, never the stock model's weights")
torch_rt.unpacked(src, "Qwen/x", log=lambda *a: None)
check(len(runs) == 1, "…once")
plain = TMP / "plain"
plain.mkdir()
(plain / "config.json").write_text("{}")
check(torch_rt.unpacked(plain, "Qwen/x") == plain, "a folder that isn't packed is used as it is")

# 5. Set up voices
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from narrator import api, guard, settings, tasks  # noqa: E402
app = FastAPI()
api.install(app)
owner = TestClient(app, base_url="http://127.0.0.1", headers={"X-Babble-Token": guard.token()})
agent = TestClient(app, base_url="http://127.0.0.1", headers={"X-Babble-Token": guard.token(), "X-Babble-Client": "mcp"})
uv = TMP / "uv"
uv.write_text(f"#!{sys.executable}\nimport sys, json\nopen({str(TMP / 'uv.log')!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n")
uv.chmod(uv.stat().st_mode | stat.S_IEXEC)
os.environ["BABBLE_UV"] = str(uv)
pc_setup.gpu = lambda: {"name": "NVIDIA GeForce RTX 4070", "cap": 8.9}
torch_rt.missing = lambda *p: ["torch"]
sys_platform = sys.platform
sys.platform = "darwin"  # (this test also runs on GitHub's Windows and Linux machines)
try:
    s = owner.get("/api/voices-setup").json()
finally:
    sys.platform = sys_platform
check(s == {"needed": False}, "on a Mac there's nothing to set up")
sys.platform = "win32"
try:
    s = owner.get("/api/voices-setup").json()
    check(s["needed"] and s["build"] == "cu128" and s["gpu"].endswith("4070") and s["size_gb"] == 3.0 and s["uv"],
          f"on a PC it says what it'll download: {s}")
    settings.save(agents_on=True)
    check(agent.post("/api/voices-setup", json={}).status_code == 403, "AI agents can't start it (it installs software)")
    settings.save(agents_on=False)
    t = owner.post("/api/voices-setup", json={}).json()
    import time
    for _ in range(100):
        info = owner.get(f"/api/tasks/{t['id']}").json()
        if info["state"] != "running":
            break
        time.sleep(0.05)
    argv = json.loads((TMP / "uv.log").read_text().splitlines()[-1])
    check(info["state"] == "done" and "torch==2.11.0+cu128" in argv and "https://download.pytorch.org/whl/cu128" in argv
          and argv[:4] == ["pip", "install", "--python", sys.executable],
          f"Set up installs the CUDA 12.8 build with uv ({info.get('error') or info['state']})")
finally:
    sys.platform = sys_platform

print(f"\n{'all passed' if not failed else f'{len(failed)} failed'}")
sys.exit(1 if failed else 0)
