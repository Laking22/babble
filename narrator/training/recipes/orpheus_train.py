"""Fine-tune Orpheus 3B on one narrator (LoRA) on a Kaggle GPU.

Input  (a Kaggle Dataset): orpheus-codes.jsonl made on the Mac by training/encode_snac.py
Output (/kaggle/working):
    checkpoints/        saved every ~20 minutes; a later run picks up from here
    <voice>-lora/       the trained adapter (small)
    <voice>-orpheus/    the full merged model (about 6.6 GB), for converting to MLX on the Mac
    samples/*.wav       test lines in the new voice, to listen to before downloading anything big

Run:  python train_orpheus.py --voice mymodel [--epochs 1] [--max-hours 11]
GPU time is only used while this runs. It stops itself before Kaggle's 12-hour limit, saving a checkpoint.
To continue, add this notebook's output as an input to the next run and run it again.
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # one GPU: LoRA on a 3B model doesn't benefit from two, and splitting slows it
import argparse
import glob
import json
import math
import random
import re
import shutil
import sys
import time
from pathlib import Path

BASE = "unsloth/orpheus-3b-0.1-ft"
SNAC_MODEL = "hubertsiuzdak/snac_24khz"
START_HUMAN, END_HUMAN, START_AI, END_AI = 128259, 128260, 128261, 128262
START_SPEECH, END_SPEECH, END_TEXT, PAD, OFFSET = 128257, 128258, 128009, 128263, 128266
MAX_LEN = 2048
TEST_LINES = [
    "It was a bright, cold morning, and the whole house seemed to be holding its breath.",
    "“You can’t be serious,” she said, laughing in spite of herself. “Not again!”",
    "He waited a long moment before he answered, and when he did, his voice was very quiet.",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find(pattern):
    """Files under the run's data folder (NARRATOR_DATA, set by the notebook) or else /kaggle/input."""
    roots = [os.environ["NARRATOR_DATA"]] if os.environ.get("NARRATOR_DATA") else []
    for root in roots + ["/kaggle/input"]:
        hits = sorted(glob.glob(f"{root}/**/{pattern}", recursive=True))
        if hits:
            return hits
    return []


def build_examples(tok, rows, voice):
    examples, skipped = [], 0
    for r in rows:
        text_ids = tok(f"{r.get('voice') or voice}: {r['text']}", add_special_tokens=True).input_ids + [END_TEXT]
        ids = [START_HUMAN] + text_ids + [END_HUMAN, START_AI, START_SPEECH] + r["codes"] + [END_SPEECH, END_AI]
        if len(ids) > MAX_LEN or len(r["codes"]) % 7:
            skipped += 1
            continue
        examples.append({"input_ids": ids, "labels": ids, "attention_mask": [1] * len(ids),
                         "voice": r.get("voice") or voice})
    return examples, skipped


def pad_batch(features):
    """Clips differ in length: pad each batch to its longest (ignored by the loss and attention)."""
    import torch
    width = max(len(f["input_ids"]) for f in features)
    def padded(key, value):
        return torch.tensor([f[key] + [value] * (width - len(f[key])) for f in features])
    return {"input_ids": padded("input_ids", PAD), "labels": padded("labels", -100), "attention_mask": padded("attention_mask", 0)}


def split_holdout(examples, share):
    """Keep back a share of each voice's clips (10 to 60 per voice) to measure progress on unseen text."""
    if share <= 0:
        return [{k: v for k, v in e.items() if k != "voice"} for e in examples], []
    by_voice = {}
    for e in examples:
        by_voice.setdefault(e.get("voice", ""), []).append(e)
    train, held = [], []
    for v, xs in by_voice.items():
        n = min(60, max(10, int(len(xs) * share))) if len(xs) >= 100 else 0
        held += xs[:n]
        train += xs[n:]
    random.Random(7).shuffle(train)
    strip = lambda xs: [{k: v for k, v in e.items() if k != "voice"} for e in xs]  # the trainer only wants token columns
    return strip(train), strip(held)


def has_bnb():
    try:
        import bitsandbytes  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def latest_checkpoint(out):
    """Newest checkpoint here or in an attached earlier output; copied into ./checkpoints so training resumes."""
    mine = sorted(glob.glob(str(out / "checkpoints" / "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
    if mine:
        return mine[-1]
    earlier = sorted(find("checkpoints/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
    if not earlier:
        return None
    dest = out / "checkpoints" / Path(earlier[-1]).name
    shutil.copytree(earlier[-1], dest, dirs_exist_ok=True)
    return str(dest)


def _half(torch):
    """16-bit weights, in whichever spelling this transformers version wants (dtype= in v5, torch_dtype= before)."""
    import inspect
    from transformers import PreTrainedModel
    return {"dtype" if "dtype" in inspect.signature(PreTrainedModel.from_pretrained).parameters or
            int(__import__("transformers").__version__.split(".")[0]) >= 5 else "torch_dtype": torch.float16}


def start_chain(data_file):
    """Adapters from earlier runs to build on, oldest first (from init.json next to the data), or []."""
    init = Path(data_file).parent / "init.json"
    if not init.exists():
        return []
    ids = json.loads(init.read_text()).get("adapters", [])
    dirs = [Path(data_file).parent / "adapters" / i for i in ids]
    missing = [str(d) for d in dirs if not (d / "adapter_config.json").exists()]
    if missing:
        sys.exit(f"Missing earlier adapters in the dataset: {missing}")
    return dirs


def build_start_model(adapter_dirs, dest):
    """Stock Orpheus with earlier runs' adapters merged in, saved to dest: the model this run continues from."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if (Path(dest) / "config.json").exists():
        return str(dest)
    log(f"Rebuilding the previous model from {len(adapter_dirs)} adapter(s) (on the CPU, a few minutes)…")
    model = AutoModelForCausalLM.from_pretrained(BASE, **_half(torch), low_cpu_mem_usage=True)
    for d in adapter_dirs:
        model = PeftModel.from_pretrained(model, str(d)).merge_and_unload()
        log(f"  merged {Path(d).name}")
    model.save_pretrained(str(dest), safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE).save_pretrained(str(dest))
    del model
    return str(dest)


def load_model(use_unsloth, base=BASE):
    import torch
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel
            model, tok = FastLanguageModel.from_pretrained(base, max_seq_length=MAX_LEN, dtype=None, load_in_4bit=False)
            model = FastLanguageModel.get_peft_model(
                model, r=64, lora_alpha=64, lora_dropout=0, bias="none", random_state=3407,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                use_gradient_checkpointing="unsloth")
            log("Loaded with Unsloth (faster, less memory)")
            return model, tok, "unsloth"
        except Exception as e:  # noqa: BLE001 - fall back to plain transformers + peft
            log(f"Unsloth unavailable ({type(e).__name__}: {str(e)[:200]}); using transformers + peft")
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, **_half(torch), device_map={"": 0})
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=64, lora_alpha=64, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    return model, tok, "peft"


def speak(model, tok, snac, voice, text, device):
    import torch
    ids = [START_HUMAN] + tok(f"{voice}: {text}", add_special_tokens=True).input_ids + [END_TEXT, END_HUMAN, START_AI, START_SPEECH]
    x = torch.tensor([ids], device=device)
    with torch.inference_mode():
        out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=1800, do_sample=True,
                             temperature=0.6, top_p=0.95, repetition_penalty=1.1, eos_token_id=END_SPEECH, pad_token_id=PAD)
    toks = [t for t in out[0, len(ids):].tolist() if t >= OFFSET]
    codes = [t - OFFSET for t in toks[: len(toks) // 7 * 7]]
    l1, l2, l3 = [], [], []
    for i in range(len(codes) // 7):
        f = codes[7 * i: 7 * i + 7]
        frame = (f[0], f[1] - 4096, f[2] - 8192, f[3] - 12288, f[4] - 16384, f[5] - 20480, f[6] - 24576)
        if all(0 <= c < 4096 for c in frame):
            l1.append(frame[0]); l2 += [frame[1], frame[4]]; l3 += [frame[2], frame[3], frame[5], frame[6]]
    if not l1:
        return None
    layers = [torch.tensor(v, device=device).unsqueeze(0) for v in (l1, l2, l3)]
    with torch.inference_mode():
        return snac.decode(layers).squeeze().float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="voice", help="the model name (outputs are <name>-lora and <name>-orpheus)")
    ap.add_argument("--epochs", type=float, default=2.0, help="passes over the data (stops early if held-out loss rises)")
    ap.add_argument("--holdout", type=float, default=0.02, help="share of each voice's clips kept back to check progress")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-hours", type=float, default=11.0)
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--no-unsloth", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="check the data and formatting only, no GPU")
    ap.add_argument("--data", help="path to orpheus-codes.jsonl (default: find it under /kaggle/input)")
    args = ap.parse_args()
    started = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = [args.data] if args.data else find("orpheus-codes.jsonl")
    if not data:
        sys.exit("Couldn't find orpheus-codes.jsonl under /kaggle/input. Attach the dataset (Add Input) and run again.")
    rows = [json.loads(line) for line in open(data[0], encoding="utf-8")]
    hours = sum(r["duration"] for r in rows) / 3600
    log(f"Data: {data[0]} — {len(rows)} clips, {hours:.1f} hours")
    chain = start_chain(data[0])
    log("Starting from: " + (" + ".join(d.name for d in chain) + " (an earlier model)" if chain else "stock Orpheus"))

    if args.dry_run:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(BASE)
        examples, skipped = build_examples(tok, rows, args.voice)
        train_ex, held = split_holdout(examples, args.holdout)
        log(f"Held back {len(held)} clips to check progress; {len(train_ex)} to train on")
        lens = sorted(len(e["input_ids"]) for e in examples)
        log(f"Dry run: {len(examples)} examples ({skipped} too long), length median {lens[len(lens) // 2]}, max {lens[-1]}")
        first = examples[0]["input_ids"]
        log(f"First example starts {first[:12]} … ends {first[-4:]}")
        log("Decoded text part: " + repr(tok.decode(first[1:first.index(END_HUMAN)])))
        steps = math.ceil(len(examples) * args.epochs / 8)
        log(f"{steps} optimizer steps for {args.epochs} epoch(s) at 8 clips per step")
        return

    import torch
    if not torch.cuda.is_available():
        sys.exit("No GPU. In the notebook's settings, set Accelerator to GPU T4 x2 (or P100) and run again.")
    base = build_start_model(chain, "/tmp/start-model") if chain else BASE
    model, tok, kind = load_model(not args.no_unsloth, base)
    examples, skipped = build_examples(tok, rows, args.voice)
    random.Random(3407).shuffle(examples)
    examples, held = split_holdout(examples, args.holdout)
    log(f"{len(examples)} training examples ({skipped} skipped as too long), {len(held)} held back to check progress")

    from datasets import Dataset
    from transformers import Trainer, TrainerCallback, TrainingArguments

    class Budget(TrainerCallback):
        """Save and stop before Kaggle's time limit; log progress with an ETA."""
        def __init__(self):
            self.last_save = time.time()

        def on_step_end(self, a, state, control, **kw):
            used = (time.time() - started) / 3600
            if state.global_step % 25 == 0 and state.max_steps:
                per = (time.time() - started) / max(state.global_step - self.first, 1)
                left = (state.max_steps - state.global_step) * per / 3600
                log(f"step {state.global_step}/{state.max_steps} · about {left:.1f} h of training left")
            if time.time() - self.last_save > 20 * 60:
                control.should_save, self.last_save = True, time.time()
            if used > args.max_hours:
                log(f"Time budget ({args.max_hours} h) reached: saving and stopping. Run again to continue.")
                control.should_save, control.should_training_stop = True, True

        def on_train_begin(self, a, state, control, **kw):
            self.first = state.global_step

    class HeldOut(TrainerCallback):
        """Every so often, measure the loss on clips the model never trains on. Keep the best adapter, and stop
        early if it keeps getting worse (more passes would only memorize the training clips)."""
        def __init__(self, every, patience=3):
            self.every, self.patience, self.trainer = every, patience, None
            path = out / "heldout.json"
            self.history = json.loads(path.read_text()) if path.exists() else []
            self.best = min((h["loss"] for h in self.history), default=float("inf"))
            self.worse = 0

        def check(self, state):
            if not held or self.trainer is None:
                return None
            loss = float(self.trainer.evaluate(eval_dataset=Dataset.from_list(held))["eval_loss"])
            epoch = state.epoch or 0.0
            self.history.append({"step": state.global_step, "epoch": round(epoch, 2), "loss": round(loss, 4)})
            (out / "heldout.json").write_text(json.dumps(self.history, indent=1))
            if loss < self.best - 1e-4:
                self.best, self.worse = loss, 0
                self.trainer.model.save_pretrained(str(out / "best-lora"))
                (out / "best-lora" / "step.json").write_text(json.dumps(self.history[-1]))
                log(f"held-out loss {loss:.4f} at step {state.global_step} (pass {epoch:.2f}): new best, saved")
            else:
                self.worse += 1
                log(f"held-out loss {loss:.4f} at step {state.global_step} (pass {epoch:.2f}); best {self.best:.4f}")
            return loss

        def on_step_end(self, a, state, control, **kw):
            if state.global_step and state.global_step % self.every == 0:
                self.check(state)
                if self.worse >= self.patience:
                    log(f"Held-out loss got worse {self.patience} checks in a row: stopping early (saves GPU time).")
                    control.should_training_stop = True

    bf16 = torch.cuda.is_bf16_supported()
    total_steps = math.ceil(len(examples) * args.epochs / 8)
    wanted = dict(
        output_dir=str(out / "checkpoints"), per_device_train_batch_size=1, gradient_accumulation_steps=8,
        num_train_epochs=args.epochs, learning_rate=args.lr, warmup_steps=max(1, int(0.03 * total_steps)),
        lr_scheduler_type="cosine", weight_decay=0.01, optim="adamw_8bit" if has_bnb() else "adamw_torch",
        fp16=not bf16, bf16=bf16, logging_steps=25, save_strategy="no", save_total_limit=2, report_to="none",
        seed=3407, dataloader_num_workers=2,
        # held-out checks one clip at a time: Orpheus's ~156k-token vocabulary makes 8-clip eval batches too big for a T4
        per_device_eval_batch_size=1, prediction_loss_only=True)
    if os.environ.get("ORPHEUS_CPU_TEST"):  # local rehearsal only
        wanted.update(fp16=False, bf16=False, use_cpu=True, dataloader_num_workers=0)
    # Pass every setting; drop one only if this library version rejects it by name (libraries rename things).
    while True:
        try:
            targs = TrainingArguments(**wanted)
            break
        except TypeError as e:
            bad = re.search(r"unexpected keyword argument '(\w+)'", str(e))
            if not bad or bad.group(1) not in wanted or bad.group(1) in ("per_device_train_batch_size", "num_train_epochs", "learning_rate"):
                raise
            log(f"This transformers version doesn't take '{bad.group(1)}'; continuing without it")
            wanted.pop(bad.group(1))
    log(f"batch {targs.per_device_train_batch_size} x accumulation {targs.gradient_accumulation_steps}, "
        f"{targs.num_train_epochs} epoch(s), learning rate {targs.learning_rate}, fp16={targs.fp16} bf16={targs.bf16}")
    heldout = HeldOut(every=max(25, total_steps // 16))  # about 8 checks per pass
    trainer = Trainer(model=model, args=targs, train_dataset=Dataset.from_list(examples), data_collator=pad_batch,
                      callbacks=[Budget(), heldout])
    heldout.trainer = trainer
    if held and not heldout.history:
        log(f"held-out loss before training: {heldout.check(trainer.state):.4f}")
    resume = latest_checkpoint(out)
    if resume:
        log(f"Continuing from {resume}")
    result = trainer.train(resume_from_checkpoint=resume)
    stopped_early = heldout.worse >= heldout.patience
    finished = trainer.state.global_step >= trainer.state.max_steps or stopped_early
    if not finished:  # the Budget callback already saved a full checkpoint (model + optimizer) at this step
        log("Not finished yet. Save this version, then run again with its output attached to continue.")
        return
    log(f"Training finished: loss {result.training_loss:.3f}")
    if held:
        final = heldout.check(trainer.state)
        best_dir = out / "best-lora"
        if final is not None and final > heldout.best + 1e-4 and (best_dir / "adapter_model.safetensors").exists():
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            set_peft_model_state_dict(model, load_file(str(best_dir / "adapter_model.safetensors")))
            at = json.loads((best_dir / "step.json").read_text())
            log(f"Using the best version (held-out {heldout.best:.4f} at step {at['step']}, pass {at['epoch']}) "
                f"instead of the last one ({final:.4f}).")
        h = heldout.history
        if h:
            log(f"Held-out loss: {h[0]['loss']:.4f} at the start, best {heldout.best:.4f}, last {h[-1]['loss']:.4f}")

    adapter = out / f"{args.voice}-lora"
    model.save_pretrained(str(adapter))
    tok.save_pretrained(str(adapter))
    log(f"Adapter saved to {adapter}")

    from snac import SNAC
    import soundfile as sf
    snac = SNAC.from_pretrained(SNAC_MODEL).eval().to("cuda")
    if kind == "unsloth":
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
    model.eval()
    (out / "samples").mkdir(exist_ok=True)
    voices = sorted({r.get("voice") or args.voice for r in rows})
    for voice in voices:  # a few test lines in every trained voice
        for n, line in enumerate(TEST_LINES[: 3 if len(voices) == 1 else 2], 1):
            audio = speak(model, tok, snac, voice, line, "cuda")
            if audio is not None:
                sf.write(out / "samples" / f"{voice}-{n}.wav", audio, 24000)
                log(f"Sample {voice} {n}: {len(audio) / 24000:.1f}s")

    merged = out / f"{args.voice}-orpheus"
    if kind == "unsloth":
        model.save_pretrained_merged(str(merged), tok, save_method="merged_16bit")
    else:
        model.merge_and_unload().save_pretrained(str(merged), safe_serialization=True)
        tok.save_pretrained(str(merged))
    shutil.rmtree(out / "checkpoints", ignore_errors=True)  # keep the output small once it's finished
    shutil.rmtree(out / "best-lora", ignore_errors=True)  # already merged in if it was the best
    log(f"Done. Merged model in {merged}. Listen to samples/ first, then download it.")


if __name__ == "__main__":
    main()
