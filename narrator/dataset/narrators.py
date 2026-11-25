"""Books read by more than one narrator: sort the clips by who reads them, so each narrator becomes a voice of
their own (training one voice on two narrators blends them, and it switches between them unpredictably).

Each clip gets a voice fingerprint from SpeechBrain's ECAPA speaker model (VoxCeleb, 20M parameters, run with MLX;
about 80 MB, downloaded the first time). Clips that sound alike are grouped (spherical k-means, 1-4 groups, the
number picked by how clearly the groups separate). The narrator with the most clips keeps the voice's name; the
others become "<voice>-2", "<voice>-3"… Clips that sit between two narrators, and groups too small to train on
(an announcer reading the credits), are left out.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from ..paths import TRAINING
from ..runtime import on_mlx_thread

MODEL = "aufklarer/SpeechBrain-ECAPA-VoxCeleb-20M-MLX"
MAX_NARRATORS = 4
CLEAR = 0.12          # how clearly groups must separate (silhouette) to count as different narrators
SAME = 0.45           # groups whose centres are this alike are one narrator doing different voices (tested on
                      # this app's voices: two narrators' centres 0.11 alike; one narrator's own groups 0.66-0.79)
MIN_SHARE = 0.04      # a narrator must read at least this share of the clips…
MIN_MINUTES = 10.0    # …and at least this much audio, to become a voice
MARGIN = 0.04         # a clip must be this much closer to its own narrator than to the next one
BATCH = 40            # clips per turn on the MLX thread (a book being made in the meantime still gets its turns)

_model = {"enc": None}


def _encoder():
    if _model["enc"] is None:
        from mlx_audio.codec import EcapaSpeakerEmbeddingExtractor
        _model["enc"] = EcapaSpeakerEmbeddingExtractor.from_pretrained(MODEL)
    return _model["enc"]


def free():
    _model["enc"] = None
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:  # noqa: BLE001
        pass


def _fingerprints(paths):
    """Unit-length 192-number voice fingerprints for these clips (24 kHz wavs)."""
    from scipy.signal import resample_poly
    enc = _encoder()
    out = []
    for p in paths:
        a, sr = sf.read(str(p), dtype="float32")
        if a.ndim > 1:
            a = a.mean(1)
        a16 = resample_poly(a, 2, 3).astype(np.float32) if sr == 24000 else a  # (the model wants 16 kHz)
        out.append(np.array(enc.encode(a16), dtype=np.float32))
    return out


def fingerprints(paths, stop=None, progress=None):
    vecs = []
    for i in range(0, len(paths), BATCH):
        if stop is not None and stop.is_set():
            raise InterruptedError
        vecs += on_mlx_thread(_fingerprints)(paths[i:i + BATCH])
        if progress:
            progress(min(i + BATCH, len(paths)) / max(len(paths), 1))
    return np.stack(vecs) if vecs else np.zeros((0, 192), np.float32)


def _kmeans(E, k, seed=0, rounds=30):
    """Spherical k-means (cosine), started k-means++ style. Returns (labels, centres)."""
    rng = np.random.default_rng(seed)
    centres = [E[rng.integers(len(E))]]
    for _ in range(1, k):
        d = np.maximum(1 - np.max(E @ np.stack(centres).T, axis=1), 0) ** 2  # (k-means++: far clips likelier)
        centres.append(E[rng.choice(len(E), p=d / d.sum())] if d.sum() > 0 else E[rng.integers(len(E))])
    C = np.stack(centres)
    for _ in range(rounds):
        labels = np.argmax(E @ C.T, axis=1)
        new = np.stack([E[labels == j].mean(0) if np.any(labels == j) else C[j] for j in range(k)])
        new /= np.maximum(np.linalg.norm(new, axis=1, keepdims=True), 1e-9)
        if np.allclose(new, C, atol=1e-5):
            break
        C = new
    return np.argmax(E @ C.T, axis=1), C


def _closest_pair(E, labels, k):
    """How alike the two most alike groups are (cosine of their centres)."""
    C = np.stack([E[labels == j].mean(0) for j in range(k)])
    C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-9)
    S = C @ C.T
    return float(np.max(S[~np.eye(k, dtype=bool)]))


def _silhouette(E, labels, sample=1500, seed=0):
    """Mean silhouette with cosine distance (on a sample, for big books): near 1 = clearly separate groups."""
    idx = np.random.default_rng(seed).permutation(len(E))[:sample]
    X, L = E[idx], labels[idx]
    D = 1 - X @ X.T
    s = []
    for i in range(len(X)):
        own = L == L[i]
        if own.sum() < 2:
            s.append(0.0)
            continue
        a = D[i, own].sum() / (own.sum() - 1)
        b = min(D[i, L == j].mean() for j in set(L.tolist()) if j != L[i])
        s.append((b - a) / max(a, b, 1e-9))
    return float(np.mean(s))


def group(E, seconds, narrators=0):
    """(labels, how clear, how many): labels[i] is clip i's narrator (0 = the one with the most audio, …) or -1
    (left out: between two narrators, or in a group too small to train on). narrators: 0 = find out."""
    n = len(E)
    if n < 20:
        return np.zeros(n, int), 0.0, 1
    tries = [narrators] if narrators else range(2, MAX_NARRATORS + 1)
    best = (np.zeros(n, int), 0.0, 1, None)
    for k in tries:
        labels, C = max((_kmeans(E, k, seed=s) for s in range(3)),
                        key=lambda lc: float(np.sum(np.max(E @ lc[1].T, axis=1))))
        share = np.bincount(labels, minlength=k) / n
        if not narrators and share.min() < MIN_SHARE:
            continue  # (a "narrator" this small is noise, or the credits)
        if not narrators and _closest_pair(E, labels, k) >= SAME:
            continue  # (one narrator's character voices, not two narrators)
        sil = _silhouette(E, labels)
        if narrators or sil > max(best[1], CLEAR):
            best = (labels, sil, k, C)
    labels, sil, k, C = best
    if k == 1:
        return labels, sil, 1
    # biggest narrator first; clips too close to call, and narrators with too little audio, left out
    sims = E @ C.T
    top2 = np.sort(sims, axis=1)[:, -2:]
    labels = np.where(top2[:, 1] - top2[:, 0] < MARGIN, -1, labels)
    minutes = np.array([seconds[labels == j].sum() / 60 for j in range(k)])
    order = [j for j in np.argsort(-minutes) if minutes[j] >= MIN_MINUTES]
    remap = {j: r for r, j in enumerate(order)}
    labels = np.array([remap.get(int(x), -1) if x >= 0 else -1 for x in labels])
    return labels, sil, max(len(order), 1)


def split(voice, work, rows, stop=None, stage=None, log=print, narrators=0):
    """Sort this book's clips (rows from clips.jsonl, wavs in work/wavs) by narrator. Narrator 1 stays in this
    voice (its rows are returned); each other narrator becomes the voice "<voice>-N" with this book in it.
    Returns (rows for this voice, [other voice names])."""
    if stage:
        stage("Telling the narrators apart", 0.0)
    paths = [work / "wavs" / r["file"] for r in rows]
    cache = work / "narrators.npy"
    E = np.load(cache) if cache.exists() and len(np.load(cache)) == len(rows) else None
    if E is None:
        E = fingerprints(paths, stop, (lambda f: stage(f"Telling the narrators apart · clip "
                                                       f"{int(f * len(rows)):,} of {len(rows):,}", f))
                         if stage else None)
        np.save(cache, E)
    seconds = np.array([r["seconds"] for r in rows], dtype=float)
    labels, clear, k = group(E, seconds, narrators)
    (work / "clips-all.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                          encoding="utf-8")
    if k == 1:
        log(f"Heard one narrator in this book (no clear second voice), so all {len(rows):,} clips stay together.")
        (work / "narrators.json").write_text(json.dumps({"narrators": 1, "clear": round(clear, 3)}, indent=1))
        return rows, []
    names, summary = [], []
    for g in range(k):
        mine = [r for r, lab in zip(rows, labels) if lab == g]
        hours = sum(r["seconds"] for r in mine) / 3600
        name = voice if g == 0 else f"{voice}-{g + 1}"
        summary.append({"voice": name, "clips": len(mine), "hours": round(hours, 2),
                        "sample": mine[len(mine) // 2]["file"] if mine else ""})
        if g == 0:
            keep = mine
            continue
        names.append(name)
        bdir = TRAINING / name / "books" / work.name
        if bdir.exists():
            shutil.rmtree(bdir)
        (bdir / "wavs").mkdir(parents=True)
        for r in mine:
            dst = bdir / "wavs" / r["file"]
            try:
                dst.hardlink_to(work / "wavs" / r["file"])
            except OSError:
                shutil.copy2(work / "wavs" / r["file"], dst)
        (bdir / "clips.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in mine),
                                          encoding="utf-8")
        book = json.loads((work / "book.json").read_text()) if (work / "book.json").exists() else {}
        book.update(clips=len(mine), kept_seconds=round(hours * 3600, 1), narrator=g + 1, split_from=voice)
        (bdir / "book.json").write_text(json.dumps(book, indent=2))
        with open(TRAINING / name / "log.txt", "a", encoding="utf-8") as f:
            f.write(f"Narrator {g + 1} of {k} in “{book.get('title', work.name)}” (sorted out of “{voice}”): "
                    f"{len(mine):,} clips, {hours:.2f} h.\n")
    left = int(np.sum(labels < 0))
    (work / "narrators.json").write_text(json.dumps({"narrators": k, "clear": round(clear, 3), "left_out": left,
                                                     "voices": summary}, indent=1))
    parts = "; ".join(f"{s['voice']}: {s['clips']:,} clips ({s['hours']:.2f} h)" for s in summary)
    log(f"Heard {k} narrators: {parts}. Left out {left:,} clips that could be either.")
    return keep, names
