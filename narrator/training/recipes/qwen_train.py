"""Fine-tune Qwen3-TTS-12Hz-1.7B-Base on one or more voices, on a single 16 GB GPU (Kaggle T4 and up).

Adapted from Qwen's official finetuning/sft_12hz.py and dataset.py (github.com/QwenLM/Qwen3-TTS,
Apache-2.0, Copyright 2026 The Alibaba Qwen team). Changes:
  - LoRA on the talker and code predictor, fp16 autocast: fits a T4, which has no bf16 and no
    FlashAttention 2 (the official script needs both). --full trains every weight instead (needs an
    A100-class GPU).
  - Memory, for a 15 GB T4: the frozen weights under LoRA are kept in 16 bits (the rest, and so the sum
    the layers add to, stays 32-bit), every layer is recomputed in the backward pass instead of being kept
    (gradient checkpointing), the longest batch goes first so a batch that can't fit shows up in the first
    minute, and running out of GPU memory halves the batch instead of ending the run. (With 32-bit weights
    and nothing recomputed, the first try ran out of memory on the first step.)
  - The text goes through the talker's text projection, as it does when the model speaks.
  - Time: after the first steps it measures its speed and trains only as many steps as fit before
    --max-hours (the app sets it from what's left of the Kaggle week), so the learning-rate schedule ends
    properly instead of being cut off.
  - Several speakers per model: each gets its own reference clip, speaker embedding and speaker id
    (3000, 3001, …), so one model can hold up to 8 voices like the Orpheus recipe.
  - Audio codes come pre-computed from the Mac (MLX); a sample of clips is re-encoded with the official
    tokenizer here and the agreement is logged.
  - Stops before Kaggle's 12-hour limit, merges LoRA into the weights and writes a CustomVoice model
    plus test samples.

Input (found under /kaggle/input or --data): qwen-codes.jsonl rows {"speaker", "text", "audio_codes" (T x 16),
"duration"}, refs/<speaker>.wav (24 kHz), check/<speaker>/*.wav + check.jsonl (optional).
Output (--out): <model>-qwen/ (merged CustomVoice model), <model>-lora/adapter.pt, samples/*.wav, train.log
"""
import argparse
import glob
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
TOKENIZER = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
SPEAKER_ID0 = 3000
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
SAMPLE_TEXTS = [
    "The rain had stopped by the time Marcus reached the old lighthouse. He paused at the door, listening.",
    "Somewhere above him, a shutter banged in the wind, slow and steady, like a heartbeat.",
]
LOG = None
STARTED = time.time()
OOM = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)


def log(msg):
    line = f"[{(time.time() - STARTED) / 60:6.1f} min] {msg}"
    print(line, flush=True)
    if LOG:
        with open(LOG, "a") as f:
            f.write(line + "\n")


def find(name, data=None):
    """A file in --data, else the run's data folder (NARRATOR_DATA, set by the notebook), else /kaggle/input."""
    data = data or os.environ.get("NARRATOR_DATA")
    if data:
        p = Path(data) / name
        if p.exists():
            return p
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    return Path(hits[0]) if hits else None


# ---------- LoRA ----------

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        self.A = nn.Parameter(torch.zeros(r, base.in_features, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.A.t().to(x.dtype) @ self.B.t().to(x.dtype)) * self.scale

    def merged(self):
        w = self.base.weight.data.float() + (self.B.float() @ self.A.float()) * self.scale
        return w.to(self.base.weight.dtype)


def add_lora(root, r, alpha, dropout):
    wrapped = []
    for name, module in list(root.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in TARGETS:
                setattr(module, child_name, LoRALinear(child, r, alpha, dropout))
                wrapped.append(f"{name}.{child_name}")
    return wrapped


def lora_modules(root):
    return {n: m for n, m in root.named_modules() if isinstance(m, LoRALinear)}


def merge_lora(root, originals=None):
    """Fold each adapter into its layer. originals: {name: the layer's weight as loaded}, so a weight kept in
    16 bits for training is merged from its exact original rather than the rounded copy."""
    for name, m in lora_modules(root).items():
        parent = root.get_submodule(name.rsplit(".", 1)[0]) if "." in name else root
        w = (originals or {}).get(name)
        w = (w if w is not None else m.base.weight.data).float().cpu()
        m.base.weight.data = w + (m.B.detach().float().cpu() @ m.A.detach().float().cpu()) * m.scale
        setattr(parent, name.rsplit(".", 1)[-1], m.base)


def native_bf16():
    """Ampere (A100, L4, …) and newer do bf16 in hardware. torch.cuda.is_bf16_supported() also says yes on a
    T4, where bf16 is emulated: slow, and it meant the T4 never got the fp16 path it was meant to use."""
    return torch.cuda.get_device_capability(0)[0] >= 8


def half_frozen(root, dtype):
    """Keep the frozen weight under each adapter in 16 bits: half the memory, and autocast multiplies in 16
    bits anyway. Returns {name: the original weight} (on the CPU) for merging later."""
    originals = {}
    for name, m in lora_modules(root).items():
        w = m.base.weight.data
        low = w.to(dtype)
        if not torch.isfinite(low).all():  # a weight too big for fp16: leave that layer at full precision
            continue
        originals[name] = w.to(torch.bfloat16) if w.dtype != torch.bfloat16 else w.clone()
        m.base.to(dtype)
    return originals


def use_checkpointing(model):
    """Recompute each transformer layer in the backward pass instead of keeping what it made (much less
    memory, about a third more time). Non-reentrant, so LoRA weights inside get their gradients even
    though the layers' inputs don't need any."""
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception:  # noqa: BLE001 - older transformers: switch the layers on by hand
        from functools import partial
        from torch.utils.checkpoint import checkpoint
        for m in model.modules():
            if isinstance(getattr(m, "gradient_checkpointing", None), bool):
                m.gradient_checkpointing = True
                m._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
    return sum(1 for m in model.modules() if getattr(m, "gradient_checkpointing", False) is True)


# ---------- data (from Qwen's finetuning/dataset.py) ----------

class Data(torch.utils.data.Dataset):
    def __init__(self, rows, processor, config, spk_index):
        self.rows, self.processor, self.config, self.spk_index = rows, processor, config, spk_index

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        text = f"<|im_start|>assistant\n{r['text']}<|im_end|>\n<|im_start|>assistant\n"
        ids = self.processor(text=text, return_tensors="pt", padding=True)["input_ids"]
        ids = ids.unsqueeze(0) if ids.dim() == 1 else ids
        return {"text_ids": ids[:, :-5], "audio_codes": torch.tensor(r["audio_codes"], dtype=torch.long),
                "spk": self.spk_index[r["speaker"]]}

    def collate(self, batch):
        c, tc = self.config, self.config.talker_config
        lens = [b["text_ids"].shape[1] + b["audio_codes"].shape[0] for b in batch]
        b, t = len(batch), max(lens) + 8
        input_ids = torch.zeros((b, t, 2), dtype=torch.long)
        codec_ids = torch.zeros((b, t, 16), dtype=torch.long)
        text_mask = torch.zeros((b, t), dtype=torch.bool)
        codec_emb_mask = torch.zeros((b, t), dtype=torch.bool)
        codec_mask = torch.zeros((b, t), dtype=torch.bool)
        attn = torch.zeros((b, t), dtype=torch.long)
        labels = torch.full((b, t), -100, dtype=torch.long)
        for i, d in enumerate(batch):
            tid, codes = d["text_ids"], d["audio_codes"]
            tl, cl = tid.shape[1], codes.shape[0]
            input_ids[i, :3, 0] = tid[0, :3]
            input_ids[i, 3:7, 0] = c.tts_pad_token_id
            input_ids[i, 7, 0] = c.tts_bos_token_id
            input_ids[i, 8:8 + tl - 3, 0] = tid[0, 3:]
            input_ids[i, 8 + tl - 3, 0] = c.tts_eos_token_id
            input_ids[i, 8 + tl - 2:8 + tl + cl, 0] = c.tts_pad_token_id
            text_mask[i, :8 + tl + cl] = True
            input_ids[i, 3:8, 1] = torch.tensor([tc.codec_nothink_id, tc.codec_think_bos_id, tc.codec_think_eos_id,
                                                 0, tc.codec_pad_id])
            input_ids[i, 8:8 + tl - 3, 1] = tc.codec_pad_id
            input_ids[i, 8 + tl - 3, 1] = tc.codec_pad_id
            input_ids[i, 8 + tl - 2, 1] = tc.codec_bos_id
            input_ids[i, 8 + tl - 1:8 + tl - 1 + cl, 1] = codes[:, 0]
            input_ids[i, 8 + tl - 1 + cl, 1] = tc.codec_eos_token_id
            labels[i, 8 + tl - 1:8 + tl - 1 + cl] = codes[:, 0]
            labels[i, 8 + tl - 1 + cl] = tc.codec_eos_token_id
            codec_ids[i, 8 + tl - 1:8 + tl - 1 + cl, :] = codes
            codec_emb_mask[i, 3:8 + tl + cl] = True
            codec_emb_mask[i, 6] = False  # the speaker embedding goes here
            codec_mask[i, 8 + tl - 1:8 + tl - 1 + cl] = True
            attn[i, :8 + tl + cl] = True
        return {"input_ids": input_ids, "attention_mask": attn, "text_embedding_mask": text_mask.unsqueeze(-1),
                "codec_embedding_mask": codec_emb_mask.unsqueeze(-1), "codec_0_labels": labels,
                "codec_ids": codec_ids, "codec_mask": codec_mask,
                "spk": torch.tensor([d["spk"] for d in batch], dtype=torch.long)}


def batches(rows, batch_size, seed):
    """Similar lengths together (less padding), batches in random order, except that the batch of longest
    clips goes first: if anything won't fit in the GPU's memory, it shows up (and is handled) at once."""
    idx = sorted(range(len(rows)), key=lambda i: len(rows[i]["audio_codes"]))
    groups = [idx[i:i + batch_size] for i in range(0, len(idx), batch_size)]
    longest = groups[-1] if groups else None
    random.Random(seed).shuffle(groups)
    if longest is not None:
        groups.remove(longest)
        groups.insert(0, longest)
    return groups


def ref_mel(path):
    import librosa
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
    wav, _ = librosa.load(str(path), sr=24000, mono=True)
    return mel_spectrogram(torch.from_numpy(wav.astype(np.float32)).unsqueeze(0), n_fft=1024, num_mels=128,
                           sampling_rate=24000, hop_size=256, win_size=1024, fmin=0, fmax=12000).transpose(1, 2)


# ---------- steps ----------

def check_codes(data_dir, data_arg):
    """Re-encode the check clips with the official tokenizer; log how many codes match the Mac's."""
    meta = find("check.jsonl", data_arg)
    if not meta:
        return
    try:
        from qwen_tts import Qwen3TTSTokenizer
        tok = Qwen3TTSTokenizer.from_pretrained(TOKENIZER, device_map="cuda:0")
        rows = [json.loads(line) for line in open(meta)]
        enc = tok.encode([str(meta.parent / r["wav"]) for r in rows])
        same = total = 0
        first = []
        for r, ref in zip(rows, enc.audio_codes):
            a, b = np.array(r["audio_codes"]), ref.cpu().numpy()
            n = min(len(a), len(b))
            same += int((a[:n] == b[:n]).sum())
            total += a[:n].size
            first.append(float((a[:n, 0] == b[:n, 0]).mean()))
        log(f"Code check: {same / max(total, 1):.1%} of codes match the official tokenizer "
            f"({len(rows)} clips; first codebook {np.mean(first):.1%})")
        del tok
        torch.cuda.empty_cache()
    except Exception as e:  # never stop training over the check
        log(f"Code check skipped: {e}")


def save_model(model, base_dir, out, speakers, spk_embeds, log_prefix=""):
    from safetensors.torch import save_file
    shutil.copytree(base_dir, out, dirs_exist_ok=True)
    cfg = json.loads((Path(base_dir) / "config.json").read_text())
    cfg["tts_model_type"] = "custom_voice"
    tc = cfg.get("talker_config", {})
    tc["spk_id"] = {s: SPEAKER_ID0 + i for i, s in enumerate(speakers)}
    tc["spk_is_dialect"] = {s: False for s in speakers}
    cfg["talker_config"] = tc
    (Path(out) / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    sd = {k: v.detach().to("cpu") for k, v in model.state_dict().items() if not k.startswith("speaker_encoder")}
    w = sd["talker.model.codec_embedding.weight"]
    for i, s in enumerate(speakers):
        w[SPEAKER_ID0 + i] = spk_embeds[i].to(w.dtype).cpu()
    sd = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in sd.items()}
    for f in Path(out).glob("model*.safetensors*"):
        f.unlink()
    save_file(sd, str(Path(out) / "model.safetensors"))
    log(f"{log_prefix}Saved {out} ({sum(v.numel() for v in sd.values()) / 1e9:.2f}B parameters)")


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model name; outputs are <model>-qwen and <model>-lora")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=None, help="default 1e-4 for LoRA, 2e-5 with --full")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--full", action="store_true", help="train every weight (A100-class GPU)")
    ap.add_argument("--max-hours", type=float, default=11.0)
    ap.add_argument("--max-clips", type=int, default=0, help="use only this many clips (quick test runs)")
    ap.add_argument("--data", help="folder with qwen-codes.jsonl (default: find it under /kaggle/input)")
    ap.add_argument("--out", default="/kaggle/working")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    LOG = out / "train.log"
    lr = args.lr or (2e-5 if args.full else 1e-4)

    codes_file = find("qwen-codes.jsonl", args.data)
    if not codes_file:
        sys.exit("qwen-codes.jsonl not found")
    data_dir = codes_file.parent
    rows = [json.loads(line) for line in open(codes_file)]
    random.Random(0).shuffle(rows)
    if args.max_clips:
        rows = rows[:args.max_clips]
    speakers = sorted({r["speaker"] for r in rows})
    hours = sum(r.get("duration", 0) for r in rows) / 3600
    log(f"{len(rows)} clips, {hours:.2f} h, speakers: {', '.join(speakers)}; GPU: {torch.cuda.get_device_name(0)}")

    check_codes(data_dir, args.data)

    from huggingface_hub import snapshot_download
    from qwen_tts import Qwen3TTSModel
    from transformers import AutoConfig
    base_dir = snapshot_download(BASE)
    bf16_ok = native_bf16()
    load_dtype = torch.bfloat16 if (args.full and bf16_ok) else torch.float32
    q = Qwen3TTSModel.from_pretrained(base_dir, dtype=load_dtype, attn_implementation="sdpa")
    model = q.model
    config = AutoConfig.from_pretrained(base_dir)
    amp = torch.bfloat16 if bf16_ok else torch.float16
    model.requires_grad_(False)
    originals = {}
    if not args.full:
        wrapped = add_lora(model.talker, args.rank, args.rank * 2, 0.05)
        originals = half_frozen(model.talker, amp)
        log(f"LoRA rank {args.rank} on {len(wrapped)} layers; {len(originals)} frozen weights kept in {amp}")
    model = model.to("cuda")
    layers = use_checkpointing(model.talker)
    log(f"Gradient checkpointing on {layers} parts of the model" if layers else
        "Gradient checkpointing isn't available here: smaller batches if memory runs short")
    log(f"Model on the GPU: {torch.cuda.memory_allocated() / 1e9:.1f} GB of "
        f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    enc_dtype = next(model.speaker_encoder.parameters()).dtype  # model.dtype is ambiguous with 16-bit layers
    with torch.no_grad():
        spk_embeds = []
        for s in speakers:
            ref = data_dir / "refs" / f"{s}.wav"
            spk_embeds.append(model.speaker_encoder(ref_mel(ref).to("cuda").to(enc_dtype))[0].float())
        spk_table = torch.stack(spk_embeds)
    log(f"Speaker embeddings from refs/: {tuple(spk_table.shape)}")

    if args.full:
        model.talker.requires_grad_(True)
        params = [p for p in model.talker.parameters() if p.requires_grad]
    else:
        params = [p for m in lora_modules(model.talker).values() for p in (m.A, m.B)]
        log(f"{sum(p.numel() for p in params) / 1e6:.1f}M trainable")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16 and load_dtype == torch.float32)
    data = Data(rows, q.processor, config, {s: i for i, s in enumerate(speakers)})

    steps_per_epoch = math.ceil(len(rows) / args.batch)
    total_steps = int(steps_per_epoch * args.epochs)
    plan = {"opt_steps": max(1, total_steps // args.accum)}  # the schedule's length, shortened if time is short
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, plan["opt_steps"] // 20)) * max(0.05, 1 - s / plan["opt_steps"]))
    deadline = STARTED + (args.max_hours - 0.75) * 3600  # leave time to merge, save and make samples
    log(f"{total_steps} steps ({args.epochs} epochs, batch {args.batch} x {args.accum}), lr {lr}, {amp}, "
        f"{args.max_hours:.1f} h at most")

    def train_on(idx, weight):
        """Forward and backward for some clips of a batch (weight: their share of it). Returns the loss."""
        batch = data.collate([data[i] for i in idx])
        batch = {k: v.to("cuda") for k, v in batch.items()}
        with torch.autocast("cuda", dtype=amp):
            spk = spk_table[batch["spk"]]
            ids = batch["input_ids"]
            # text goes through the talker's text projection, exactly as when it speaks (mlx-audio and qwen_tts both
            # do this); without it the model trained on text it never sees, learned to do without, and babbled
            emb = model.talker.text_projection(model.talker.model.text_embedding(ids[:, :, 0])) * batch["text_embedding_mask"]
            cemb = model.talker.model.codec_embedding(ids[:, :, 1]) * batch["codec_embedding_mask"]
            cemb[:, 6, :] = spk.to(cemb.dtype)
            emb = emb + cemb
            for i in range(1, 16):
                e = model.talker.code_predictor.get_input_embeddings()[i - 1](batch["codec_ids"][:, :, i])
                emb = emb + e * batch["codec_mask"].unsqueeze(-1)
            # Both losses are worked out here, lined up the way the model speaks. Qwen's sft_12hz.py (copied before)
            # had three alignment bugs (github.com/QwenLM/Qwen3-TTS issues #371, #384): the library's loss shifts the
            # labels again (so it learned the frame after next, and lost the end-of-speech code: it never stopped),
            # the code predictor got the hidden state that had already seen its frame, and its labels were shifted
            # once more inside forward_finetune.
            outs = model.talker(inputs_embeds=emb[:, :-1, :], attention_mask=batch["attention_mask"][:, :-1],
                                output_hidden_states=True, use_cache=False)
            target = batch["codec_0_labels"][:, 1:]  # logits at p predict the code given at p + 1
            main_loss = nn.functional.cross_entropy(outs.logits.float().reshape(-1, outs.logits.shape[-1]),
                                                    target.reshape(-1), ignore_index=-100)
            hidden = outs.hidden_states[0][-1]
            cmask = batch["codec_mask"]
            th = hidden[cmask[:, 1:]]  # the step before each frame: the one that predicts it
            tcodes = batch["codec_ids"][cmask]
            sub_logits, _ = model.talker.forward_sub_talker_finetune(tcodes, th)
            sub_loss = nn.functional.cross_entropy(sub_logits.float().reshape(-1, sub_logits.shape[-1]),
                                                   tcodes[:, 1:].reshape(-1))  # head i -> codebook i (1..15)
            loss = (main_loss + 0.3 * sub_loss) * weight / args.accum
        if not torch.isfinite(loss):
            return float("nan")  # no backward: a bad step would spoil the weights
        scaler.scale(loss).backward()
        return loss.item() * args.accum

    model.train()
    step, losses, stopped, micro, skipped, bad_run = 0, [], False, args.batch, 0, 0
    timed = None  # (step, time) when the speed check starts
    for epoch in range(math.ceil(args.epochs)):
        for group in batches(rows, args.batch, epoch):
            if step >= total_steps:
                break
            if time.time() > deadline:
                log("Stopping early to stay inside the time limit.")
                stopped = True
                break
            while True:
                oom = False
                try:
                    value = 0.0
                    for k in range(0, len(group), micro):
                        part = group[k:k + micro]
                        value += train_on(part, len(part) / len(group))
                except OOM:
                    oom = True
                if not oom:
                    break
                # out of the except block, so the failed step's tensors are gone: free them and try smaller
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if micro == 1:
                    skipped += 1
                    value = None
                    log(f"Skipped a batch whose longest clip doesn't fit in the GPU's memory ({skipped} so far).")
                    if skipped > 25:
                        sys.exit("Too many clips don't fit in this GPU's memory. Use shorter clips or a bigger GPU.")
                    break
                micro = max(1, micro // 2)
                log(f"GPU memory full: {micro} clip(s) at a time from now on (same effective batch).")
            step += 1
            if value is not None:
                losses.append(value)
                bad_run = bad_run + 1 if not math.isfinite(value) else 0
                if bad_run >= 20:
                    sys.exit("The loss stopped being a number 20 steps in a row; stopping.")
            if step == 1:
                log(f"First step done: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB of GPU memory at most, "
                    f"loss {value if value is not None else float('nan'):.3f}")
            if step % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            if step == 5:
                timed = (step, time.time())
            elif timed and step == timed[0] + 40:  # the speed now: train only the steps that fit in the time left
                per = (time.time() - timed[1]) / 40
                fit = step + int((deadline - time.time()) / per * 0.97)
                hours = per * total_steps / 3600
                if fit < total_steps:
                    log(f"{per:.1f} s a step: all {total_steps} steps would take {hours:.1f} h, so training {fit} "
                        f"(about {fit / steps_per_epoch:.1f} epochs) to finish in time")
                    total_steps = max(step + 1, fit)
                    plan["opt_steps"] = max(1, total_steps // args.accum)
                else:
                    log(f"{per:.1f} s a step: all {total_steps} steps fit (about {hours:.1f} h)")
            if step % 50 == 0:
                recent = [x for x in losses[-50:] if math.isfinite(x)]
                log(f"epoch {epoch} step {step}/{total_steps} loss {np.mean(recent) if recent else float('nan'):.4f} "
                    f"lr {sched.get_last_lr()[0]:.2e} mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        if not args.full:
            ad = out / f"{args.model}-lora"
            ad.mkdir(exist_ok=True)
            torch.save({n: {"A": m.A.detach().cpu(), "B": m.B.detach().cpu(), "scale": m.scale}
                        for n, m in lora_modules(model.talker).items()}, ad / "adapter.pt")
            (ad / "adapter.json").write_text(json.dumps({"base": BASE, "rank": args.rank, "speakers": speakers,
                                                         "epoch": epoch, "steps": step}))
        if stopped or step >= total_steps:
            break
    recent = [x for x in losses[-50:] if math.isfinite(x)]
    log(f"Trained {step} steps; last loss {np.mean(recent) if recent else float('nan'):.4f}")

    model.eval()
    del opt
    if not args.full:
        model.to("cpu")
        torch.cuda.empty_cache()
        merge_lora(model.talker, originals)
    final = out / f"{args.model}-qwen"
    save_model(model, base_dir, final, speakers, spk_embeds)
    del model, q
    torch.cuda.empty_cache()

    try:
        import soundfile as sf
        # fp32 on GPUs without bf16: fp16 inference can overflow in this model
        tts = Qwen3TTSModel.from_pretrained(str(final), dtype=torch.bfloat16 if bf16_ok else torch.float32,
                                            attn_implementation="sdpa", device_map="cuda:0")
        (out / "samples").mkdir(exist_ok=True)
        for s in speakers:
            for k, text in enumerate(SAMPLE_TEXTS, 1):
                # at most ~33 s: a model that can't stop would otherwise run for minutes per sample
                wavs, sr = tts.generate_custom_voice(text=text, speaker=s, max_new_tokens=400)
                sf.write(out / "samples" / f"{s}-{k}.wav", wavs[0], sr)
        log(f"Samples written for {', '.join(speakers)}")
    except Exception as e:
        log(f"Samples failed (the model is still saved): {e}")
    log(f"Done. Merged model in {final}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        if e.code not in (0, None):
            log(f"FAILED: {e.code}")
        raise
    except BaseException as e:
        log("FAILED: the GPU ran out of memory." if isinstance(e, OOM) else f"FAILED: {type(e).__name__}: {e}")
        raise
