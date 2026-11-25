"""Encode a Train-tab dataset into Orpheus audio codes (SNAC, 7 per frame). Standalone on purpose:
it runs in a side environment that has torch + snac, not in the app's own environment.

    python orpheus_encode.py <dataset dir with metadata.jsonl> <out.jsonl>

Resumable: clips already encoded with the same file, text and length are kept.
"""
import json
import re
import sys
import time
from pathlib import Path

SNAC_MODEL = "hubertsiuzdak/snac_24khz"
OFFSET, BOOK = 128266, 4096

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
        "seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _words(n):
    return _ONES[n] if n < 20 else _TENS[n // 10] + ("-" + _ONES[n % 10] if n % 10 else "") if n < 100 else str(n)


def _roman(s):
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    return sum(-vals[a] if vals.get(b, 0) > vals[a] else vals[a] for a, b in zip(s, s[1:] + " "))


def spoken(text):
    """Chapter headings as they're read aloud: 'CHAPTER XII.' -> 'Chapter Twelve.'"""
    return re.sub(r"\b(CHAPTER|Chapter|BOOK|Book|PART|Part)\s+([IVXLC]+|\d+)\b",
                  lambda m: f"{m.group(1).title()} "
                            f"{_words(int(m.group(2)) if m.group(2).isdigit() else _roman(m.group(2))).title()}", text)


def encode(ds, out):
    import numpy as np
    import soundfile as sf
    import torch
    from snac import SNAC
    ds, out = Path(ds), Path(out)
    rows = [json.loads(line) for line in open(ds / "metadata.jsonl", encoding="utf-8") if line.strip()]
    key = lambda r: (r["audio"], spoken(r["text"]), round(float(r.get("duration") or 0), 1))
    done = {}
    if out.exists():
        for line in open(out, encoding="utf-8"):
            r = json.loads(line)
            done[key(r)] = r
    wanted = {key(r) for r in rows}
    todo = [r for r in rows if key(r) not in done]
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{ds.name}: {len(rows)} clips, {len(rows) - len(todo)} already encoded, {len(todo)} to go ({device})", flush=True)
    snac = SNAC.from_pretrained(SNAC_MODEL).eval().to(device) if todo else None
    started = time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for k, r in done.items():
            if k in wanted:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for n, r in enumerate(todo, 1):
            audio, sr = sf.read(ds / r["audio"], dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
            if sr != 24000:
                audio = np.interp(np.linspace(0, len(audio), int(len(audio) * 24000 / sr), endpoint=False),
                                  np.arange(len(audio)), audio).astype(np.float32)
            with torch.inference_mode():
                l1, l2, l3 = (c.squeeze(0).tolist() for c in snac.encode(torch.from_numpy(audio).to(device)[None, None]))
            codes = []
            for i in range(len(l1)):
                codes += [l1[i], l2[2 * i] + BOOK, l3[4 * i] + 2 * BOOK, l3[4 * i + 1] + 3 * BOOK,
                          l2[2 * i + 1] + 4 * BOOK, l3[4 * i + 2] + 5 * BOOK, l3[4 * i + 3] + 6 * BOOK]
            f.write(json.dumps({"audio": r["audio"], "text": spoken(r["text"]),
                                "duration": r.get("duration", len(audio) / 24000),
                                "codes": [c + OFFSET for c in codes]}, ensure_ascii=False) + "\n")
            if n % 500 == 0 or n == len(todo):
                secs = sum(x.get("duration", 0) for x in todo[:n])
                print(f"  {n}/{len(todo)} clips ({secs / max(time.time() - started, 1e-6):.0f}x real time)", flush=True)
    tmp.replace(out)


if __name__ == "__main__":
    encode(sys.argv[1], sys.argv[2])
