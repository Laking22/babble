"""Check one engine end to end through the real renderer, then measure speed, accuracy and pauses.

    .venv/bin/python tests/engine_check.py kokoro:bm_george [orpheus:tara builtin:ryan …] [--out DIR]
    .venv/bin/python tests/engine_check.py --mixed kokoro:bm_george orpheus:tara builtin:ryan

One voice per run keeps a single model in memory. --mixed renders a short cast across engines in one
render call, to check that lines are grouped by model and joined back in story order.
"""
import difflib
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from narrator import engine as en
from narrator.runtime import SR

TEXT = ("The rain had stopped by the time Marcus reached the old lighthouse. He paused at the door, listening. "
        "Somewhere above him, a shutter banged in the wind, slow and steady, like a heartbeat.")


def pauses(a, sr=SR):
    f = int(sr * 0.02)
    e = np.array([np.sqrt(np.mean(a[i:i + f] ** 2) + 1e-12) for i in range(0, len(a) - f, f)])
    db = 20 * np.log10(e)
    sil = db < np.percentile(db, 90) - 35
    runs, c, started = [], 0, False
    for s in sil:
        if s:
            c += 1
        else:
            if c and started:
                runs.append(c * 0.02)
            c, started = 0, True
    return float(sil.mean()), max(runs) if runs else 0.0


def whisper_match(path, text):
    from mlx_audio.stt.utils import load_model
    from narrator.audio import read_mono
    from narrator.runtime import on_mlx_thread, slot
    import mlx.core as mx

    def job():
        m = slot.get("stt", lambda: load_model(en.STT_MODEL))
        heard = m.generate(mx.array(read_mono(path, to=16000))).text
        slot.release()
        return heard
    heard = on_mlx_thread(job)()
    norm = lambda s: re.sub(r"[^a-z' ]", " ", s.lower()).split()
    return difflib.SequenceMatcher(None, norm(text), norm(heard)).ratio()


def render(items, out):
    started, path = time.time(), None
    for p, status in en.render(items, kind="check", language="english", temperature=0.9, batch=2, path=out):
        path = p or path
    return Path(path), time.time() - started


def main(args):
    out_dir = Path(args[args.index("--out") + 1]) if "--out" in args else Path("tests/out")
    out_dir.mkdir(parents=True, exist_ok=True)
    voices = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] != "--out")]
    if "--mixed" in args:
        lines = TEXT.split(". ")
        items = [dict(text=l.rstrip(".") + ".", voice=voices[i % len(voices)], pause=0.3, who=voices[i % len(voices)])
                 for i, l in enumerate(lines)]
        path, took = render(items, out_dir / "mixed.wav")
        a, _ = sf.read(path)
        print(f"mixed {', '.join(voices)}: {len(a) / SR:.1f}s in {took:.0f}s · {path}")
        print(f"  whisper match {whisper_match(path, ' '.join(it['text'] for it in items)):.0%}")
        return
    for v in voices:
        path, took = render([dict(text=TEXT, voice=v, pause=0.2)], out_dir / f"{v.replace(':', '-')}.wav")
        a, _ = sf.read(path)
        dur = len(a) / SR
        sil, longest = pauses(a)
        print(f"{v}: {dur:.1f}s audio in {took:.1f}s ({dur / took:.2f}x real time, includes loading) · "
              f"silence {sil:.0%} · longest pause {longest:.2f}s · {path}")
        print(f"  whisper match {whisper_match(path, TEXT):.0%}")


if __name__ == "__main__":
    main(sys.argv[1:])
