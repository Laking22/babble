"""Read a story with a language model on a cloud GPU, for Babble's Detect speakers.

Reads llm-job.json (written by the app's cloud_ai.py):
  {"kind": "detect", "model": ..., "texts": [paragraphs]}   who speaks each line (detect_core.py, sent with it)
  {"kind": "prompts", "model": ..., "prompts": [[prompt, max tokens], ...]}   one reply each (sound suggestions)
and writes llm.json after every batch, so a run cut short still brings back what it read, plus progress.json.
Stops starting new batches after --max-hours.

Usage:  python llm_runner.py [--max-hours 3] [--out /kaggle/working] [--dtype bfloat16] [--batch 8]
"""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

STARTED = time.time()


def say(msg, out):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(out / "run.log", "a") as f:
        f.write(line + "\n")


def find_data():
    env = os.environ.get("NARRATOR_DATA")
    if env and (Path(env) / "llm-job.json").exists():
        return Path(env)
    here = Path(__file__).resolve().parent
    if (here / "llm-job.json").exists():
        return here
    hits = sorted(Path("/kaggle/input").rglob("llm-job.json")) if Path("/kaggle/input").exists() else []
    if not hits:
        sys.exit("FAILED: couldn't find llm-job.json")
    return hits[0].parent


def write(out, result, done, total, stopped=False):
    tmp = out / "llm.json.tmp"
    tmp.write_text(json.dumps(result))
    tmp.replace(out / "llm.json")
    (out / "progress.json").write_text(json.dumps({"done": done, "total": total, "stopped": stopped,
                                                    "seconds": round(time.time() - STARTED)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/kaggle/working" if Path("/kaggle/working").exists() else "out")
    ap.add_argument("--max-hours", type=float, default=3.0)
    ap.add_argument("--dtype", default="", help="float16 (T4, the default) or bfloat16 (newer GPUs)")
    ap.add_argument("--batch", type=int, default=8, help="passages read at once")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = find_data()
    job = json.loads((data / "llm-job.json").read_text())
    sys.path.insert(0, str(data))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    say(f"Loading {job['model']} ({'bfloat16' if dtype == torch.bfloat16 else 'float16'})…", out)
    tok = AutoTokenizer.from_pretrained(job["model"], padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(job["model"], torch_dtype=dtype, device_map=device).eval()
    say(f"Loaded in {time.time() - STARTED:.0f} s.", out)

    def generate(prompts, max_tokens):
        chats = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
                 for p in prompts]
        enc = tok(chats, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            res = model.generate(**enc, max_new_tokens=max(max_tokens), do_sample=False, pad_token_id=tok.pad_token_id)
        return [tok.decode(r[enc["input_ids"].shape[1]:], skip_special_tokens=True) for r in res]

    limit = args.max_hours * 3600 - 300  # room to save
    stop = threading.Event()
    if job["kind"] == "detect":
        import detect_core
        total, labels, genders, known, done = 0, {}, {}, [], 0
        for ev in detect_core.run(job["texts"], generate, stop=stop, batch=args.batch, starts=job.get("starts") or ()):
            if ev[0] == "progress":
                done, total, (labels, genders, known) = ev[1], ev[2], ev[3]
                say(f"{done} of {total} dialogue lines", out)
            else:
                labels, genders, known = ev[1:]
            write(out, {"labels": {str(k): v for k, v in labels.items()}, "known": known,
                        "genders": {k: dict(v) for k, v in genders.items()}}, done, total, stop.is_set())
            if time.time() - STARTED > limit:
                stop.set()
        if stop.is_set():
            say(f"Time limit reached ({args.max_hours} h): saved what was read.", out)
    else:
        prompts = job["prompts"]
        replies = [None] * len(prompts)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i][0]))  # similar lengths batch better
        done = 0
        for b in range(0, len(order), args.batch):
            if time.time() - STARTED > limit:
                stop.set()
                say(f"Time limit reached ({args.max_hours} h): saved what was read.", out)
                break
            ids = order[b:b + args.batch]
            for i, r in zip(ids, generate([prompts[i][0] for i in ids], [prompts[i][1] for i in ids])):
                replies[i] = r
            done += len(ids)
            say(f"{done} of {len(prompts)} pages", out)
            write(out, {"replies": replies}, done, len(prompts))
        write(out, {"replies": replies}, done, len(prompts), stop.is_set())
    say(f"Done in {(time.time() - STARTED) / 60:.1f} min.", out)


if __name__ == "__main__":
    main()
