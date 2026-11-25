"""Orpheus LoRA fine-tune on Apple Silicon with MLX (the "This Mac" back end).

Same data and token layout as orpheus_train.py (the GPU trainer), on the 4-bit MLX model with LoRA and
gradient checkpointing. Slow on 8–16 GB Macs; practical on 32 GB+ Max/Ultra chips.

    python -m narrator.training.recipes.orpheus_mlx_train --data <folder with orpheus-codes.jsonl> --model NAME --out DIR

Output: DIR/<model>-orpheus/ (an MLX 4-bit model, marked with mlx.json, ready to install without conversion),
DIR/<model>-mlx-lora/adapters.safetensors (resumable), DIR/samples/*.wav, DIR/train.log
"""
import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

STOCK = "mlx-community/orpheus-3b-0.1-ft-4bit"
START_HUMAN, END_HUMAN, START_AI, END_AI = 128259, 128260, 128261, 128262
START_SPEECH, END_SPEECH, END_TEXT = 128257, 128258, 128009
KEYS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
MAX_LEN = 2048
TEST_LINES = ["It was a bright, cold morning, and the whole house seemed to be holding its breath.",
              "He waited a long moment before he answered, and when he did, his voice was very quiet."]
LOG = None
STARTED = time.time()


def log(msg):
    line = f"[{(time.time() - STARTED) / 60:6.1f} min] {msg}"
    print(line, flush=True)
    if LOG:
        with open(LOG, "a") as f:
            f.write(line + "\n")


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-hours", type=float, default=48.0)
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=100, help="optimizer steps between adapter saves")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    LOG = out / "train.log"

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_lm import load
    from mlx_lm.tuner.trainer import grad_checkpoint
    from mlx_lm.tuner.utils import linear_to_lora_layers

    rows = [json.loads(line) for line in open(Path(args.data) / "orpheus-codes.jsonl", encoding="utf-8")]
    random.Random(0).shuffle(rows)
    if args.max_clips:
        rows = rows[:args.max_clips]
    model, tok = load(STOCK)
    seqs, skipped = [], 0
    for r in rows:
        ids = [START_HUMAN] + tok.encode(f"{r['voice']}: {r['text']}") + [END_TEXT, END_HUMAN, START_AI, START_SPEECH] + \
            r["codes"] + [END_SPEECH, END_AI]
        if len(ids) > MAX_LEN or len(r["codes"]) % 7:
            skipped += 1
            continue
        seqs.append(ids)
    voices = sorted({r["voice"] for r in rows})
    log(f"{len(seqs)} clips ({skipped} skipped), {sum(r['duration'] for r in rows) / 3600:.2f} h, voices: {', '.join(voices)}")

    model.freeze()
    linear_to_lora_layers(model, len(model.layers), {"rank": args.rank, "scale": 2.0, "dropout": 0.0, "keys": KEYS})
    grad_checkpoint(model.layers[0])
    adir = out / f"{args.model}-mlx-lora"
    adir.mkdir(exist_ok=True)
    state = json.loads((adir / "state.json").read_text()) if (adir / "state.json").exists() else {"step": 0}
    if (adir / "adapters.safetensors").exists():
        model.load_weights(str(adir / "adapters.safetensors"), strict=False)
        log(f"Resuming from step {state['step']}")
    n = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    log(f"LoRA rank {args.rank}: {n / 1e6:.1f}M trainable")

    total = int(len(seqs) * args.epochs)
    opt_steps = max(1, total // args.accum)
    sched = lambda s: args.lr * min(1.0, (s + 1) / max(1, opt_steps // 30)) * max(0.05, 1 - s / opt_steps)
    opt = optim.Adam(learning_rate=args.lr)

    def loss_fn(m, x):
        logits = m(x[:, :-1]).astype(mx.float32)
        return nn.losses.cross_entropy(logits, x[:, 1:]).mean()

    vg = nn.value_and_grad(model, loss_fn)
    order = [i for e in range(math.ceil(args.epochs)) for i in random.Random(e).sample(range(len(seqs)), len(seqs))][:total]
    deadline = STARTED + args.max_hours * 3600
    acc, losses, k = None, [], state["step"] * args.accum
    for k in range(k, total):
        if time.time() > deadline:
            log("Stopping at the time limit; run again to continue.")
            break
        loss, grads = vg(model, mx.array([seqs[order[k]]]))
        acc = grads if acc is None else tree_unflatten([(a, b + c) for (a, b), (_, c)
                                                        in zip(tree_flatten(acc), tree_flatten(grads))])
        losses.append(loss.item())
        if (k + 1) % args.accum == 0:
            step = (k + 1) // args.accum
            opt.learning_rate = sched(step)
            opt.update(model, tree_unflatten([(a, b / args.accum) for a, b in tree_flatten(acc)]))
            mx.eval(model.parameters(), opt.state)
            acc = None
            if step % 10 == 0:
                log(f"step {step}/{opt_steps} loss {sum(losses[-80:]) / len(losses[-80:]):.4f} "
                    f"peak {mx.get_peak_memory() / 1e9:.1f} GB")
            if step % args.save_every == 0:
                mx.save_safetensors(str(adir / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
                (adir / "state.json").write_text(json.dumps({"step": step}))
    mx.save_safetensors(str(adir / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    (adir / "state.json").write_text(json.dumps({"step": (k + 1) // args.accum}))
    if (k + 1) < total:
        return  # stopped early: resume later

    log("Merging the adapter into the model…")
    fused = [(name, m.fuse()) for name, m in model.named_modules() if hasattr(m, "fuse")]
    model.update_modules(tree_unflatten(fused))
    from huggingface_hub import snapshot_download
    stock = Path(snapshot_download(STOCK))
    final = out / f"{args.model}-orpheus"
    shutil.rmtree(final, ignore_errors=True)
    shutil.copytree(stock, final, ignore=shutil.ignore_patterns("*.safetensors", "*.safetensors.index.json"))
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(final / "model.safetensors"), weights, metadata={"format": "mlx"})
    cfg = json.loads((final / "config.json").read_text())
    cfg["tokenizer_name"] = str(final)
    (final / "config.json").write_text(json.dumps(cfg, indent=2))
    (final / "mlx.json").write_text(json.dumps({"format": "mlx", "base": STOCK, "voices": voices}))
    log(f"Saved {final}")
    del model
    mx.clear_cache()

    import gc
    gc.collect()
    mx.clear_cache()
    for attempt in range(2):  # on 8-16 GB Macs the GPU can time out under memory pressure: free and retry once
        try:
            import numpy as np
            import soundfile as sf
            from mlx_audio.tts.utils import load_model
            m = load_model(str(final))
            (out / "samples").mkdir(exist_ok=True)
            for v in voices:
                for i, text in enumerate(TEST_LINES, 1):
                    audio = np.concatenate([np.array(r.audio, dtype=np.float32).flatten()
                                            for r in m.generate(text=text, voice=v, verbose=False, max_tokens=1600)])
                    sf.write(out / "samples" / f"{v}-{i}.wav", audio, 24000)
            log(f"Samples written for {', '.join(voices)}")
            break
        except Exception as e:  # the model is saved either way
            log(f"Samples failed{' (retrying)' if not attempt else ''}: {e}")
            m = None
            gc.collect()
            mx.clear_cache()
            time.sleep(5)
    log(f"Done. Merged model in {final}")


if __name__ == "__main__":
    sys.exit(main())
