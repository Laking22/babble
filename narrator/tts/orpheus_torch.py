"""Orpheus 3B through PyTorch, for Windows and Linux (tts/orpheus.py uses it when there's no MLX).

The cloud's way (kaggle_render/runner.py Orpheus): the full model with transformers, SNAC to turn its codes into sound,
the repetition penalty only on the last 20 tokens as the Mac's mlx-lm does it, and a line that comes back silent made
again a little warmer. Trained Orpheus packs (4-bit MLX) are unpacked once (torch_rt.unpacked).
"""
import numpy as np

from ..runtime import slot
from . import torch_rt

STOCK = "unsloth/orpheus-3b-0.1-ft"   # an open copy of canopylabs/orpheus-3b-0.1-ft (no access request needed)
SNAC = "hubertsiuzdak/snac_24khz"
START, END = [128259], [128009, 128260, 128261, 128257]
STOP, PAD, AUDIO_START, OFFSET = 128258, 128263, 128257, 128266


class _Model:
    def __init__(self, path):
        import torch
        from snac import SNAC as Codec
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.device = torch, torch_rt.device()
        self.tok = AutoTokenizer.from_pretrained(STOCK)   # (a trained model's saved copy may not load on its own)
        self.model = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=torch_rt.dtype()).to(self.device).eval()
        self.snac = Codec.from_pretrained(SNAC).eval().to(self.device)

    def penalty(self, value, size=20):
        from transformers import LogitsProcessor, LogitsProcessorList
        t = self.torch

        class Recent(LogitsProcessor):
            def __call__(self, input_ids, scores):
                recent = input_ids[:, -size:]
                picked = t.gather(scores, 1, recent)
                picked = t.where(picked < 0, picked * value, picked / value)
                return scores.scatter(1, recent, picked)
        return LogitsProcessorList([Recent()])

    def audio(self, tokens):
        tokens = [t for t in tokens if t != STOP]
        if AUDIO_START in tokens:
            tokens = tokens[len(tokens) - tokens[::-1].index(AUDIO_START):]
        codes = [t - OFFSET for t in tokens[: len(tokens) // 7 * 7]]
        l1, l2, l3 = [], [], []
        for i in range(len(codes) // 7):
            f = codes[7 * i: 7 * i + 7]
            frame = (f[0], f[1] - 4096, f[2] - 2 * 4096, f[3] - 3 * 4096, f[4] - 4 * 4096, f[5] - 5 * 4096, f[6] - 6 * 4096)
            if all(0 <= c < 4096 for c in frame):  # (a corrupt frame is skipped, not fatal)
                l1.append(frame[0])
                l2 += [frame[1], frame[4]]
                l3 += [frame[2], frame[3], frame[5], frame[6]]
        if not l1:
            return np.zeros(0, dtype=np.float32)
        t = self.torch
        layers = [t.tensor(x, dtype=t.int32, device=self.device).unsqueeze(0) for x in (l1, l2, l3)]
        with t.inference_mode():
            return self.snac.decode(layers).squeeze().float().cpu().numpy().astype(np.float32)

    def speak(self, prompts, temperature, top_p=0.95, repetition_penalty=1.1):
        """prompts: [(speaker, text)] made together. Returns their audio in order (empty: the line came back silent
        three times)."""
        from ..audio import is_silent
        t = self.torch
        out = [None] * len(prompts)
        todo = list(range(len(prompts)))
        for attempt in range(3):
            if not todo:
                break
            ids = [START + self.tok(f"{prompts[i][0]}: {prompts[i][1]}").input_ids + END for i in todo]
            width = max(len(x) for x in ids)
            input_ids = t.tensor([[PAD] * (width - len(x)) + x for x in ids], device=self.device)
            mask = t.tensor([[0] * (width - len(x)) + [1] * len(x) for x in ids], device=self.device)
            longest = max(len(prompts[i][1]) for i in todo)
            with t.inference_mode():
                # (as the Mac allows a line: 12 tokens a character, twice what narration needs)
                gen = self.model.generate(input_ids=input_ids, attention_mask=mask,
                                          max_new_tokens=min(12000, max(1200, 12 * longest)),
                                          do_sample=True, temperature=temperature * (1 + 0.15 * attempt), top_p=top_p,
                                          logits_processor=self.penalty(repetition_penalty * (1 + 0.15 * attempt)),
                                          eos_token_id=STOP, pad_token_id=PAD)
            left = []
            for row, i in zip(gen[:, width:].tolist(), todo):
                wav = self.audio(row)
                if is_silent(wav):
                    left.append(i)
                else:
                    out[i] = wav
            todo = left
        return [w if w is not None else np.zeros(0, np.float32) for w in out]


def model(path):
    """Stock Orpheus (path STOCK) or a trained one (its folder: unpacked once)."""
    from .. import tasks
    return slot.get(f"orpheus-torch:{path}",
                    lambda: _Model(STOCK if path == STOCK else torch_rt.unpacked(path, STOCK, tasks.note)))
