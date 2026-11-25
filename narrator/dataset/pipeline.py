"""Train Voice: turn any audiobook + its book text into a clean fine-tuning dataset.

Pipeline (runs in the background, resumable):
  1. Audio: every file (mp3/m4b/m4a/flac/wav/ogg/opus/zip…) → 24 kHz mono 16-bit, rumble filter.
  2. Transcribe with Whisper (word timestamps), 10-minute windows cut at silences, cached.
  3. Book: EPUB/PDF/TXT/DOCX/MOBI… → sentences (via book.py).
  4. Align transcript words to book words (unique n-gram anchors + difflib between them).
  5. Clips: whole book sentences that the narrator read exactly, 2–15 s, cut at the quietest
     point near each boundary, silence trimmed, loudness evened out.
  6. Filter: credits/intros not in the book, misreads, clipping, noise, too short/long.
  7. Write: wavs/, metadata.csv (LJSpeech), train.jsonl / val.jsonl (Qwen3-TTS format), ref.wav, README.
"""
import difflib
import json
import random
import re
import shutil
import subprocess
import threading
import time
import traceback
import unicodedata
import zipfile
from bisect import bisect_left
from pathlib import Path

import numpy as np
import soundfile as sf

from .. import book as bk
from ..paths import TRAINING
from ..runtime import on_mlx_thread, slot

SR = 24000
STT_SR = 16000
WINDOW = 600.0           # seconds of audio per Whisper call
AUDIO_EXT = {".mp3", ".m4b", ".m4a", ".aac", ".flac", ".wav", ".ogg", ".oga", ".opus", ".wma", ".aiff", ".aif",
             ".mp4", ".webm", ".mka", ".caf"}
BOOK_EXT = {".epub", ".pdf", ".txt", ".md", ".docx", ".mobi", ".azw", ".azw3", ".fb2", ".rtf", ".odt", ".html", ".htm"}
LANGS = {"english": "en", "auto": None, "chinese": "zh", "german": "de", "italian": "it", "portuguese": "pt",
         "spanish": "es", "japanese": "ja", "korean": "ko", "french": "fr", "russian": "ru"}

_jobs = {}               # voice -> {"thread", "stop", "log": [...], "stage", "progress", "done", "error"}
_lock = threading.Lock()
_stt = {"model": None}


# ---------- small utils ----------

def slug(text, fallback="item"):
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return text[:60] or fallback


def natural_key(p):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(p))]


def clock(s):
    s = int(s)
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"


def _log(voice, msg):
    job = _jobs.get(voice)
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    if job is not None:
        job["log"].append(line)
        del job["log"][:-200]
    d = TRAINING / voice
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _bin(name):
    for c in (shutil.which(name), f"/opt/homebrew/bin/{name}", f"/usr/local/bin/{name}"):
        if c and Path(c).exists():
            return c
    return name


def _ffprobe_duration(path):
    try:
        out = subprocess.run([_bin("ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                             capture_output=True, text=True, timeout=120).stdout.strip()
        return float(out)
    except (ValueError, OSError, subprocess.SubprocessError):
        return 0.0


def _audio_tag(path, tag):
    try:
        out = subprocess.run([_bin("ffprobe"), "-v", "error", "-show_entries", f"format_tags={tag}", "-of", "csv=p=0", str(path)],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        return out
    except (OSError, subprocess.SubprocessError):
        return ""


# ---------- step 1: audio ----------

def sort_inputs(paths, workdir):
    """Split uploaded files into (audio files in reading order, book file). Zips are unpacked."""
    audio, books = [], []
    for p in paths or []:
        p = Path(p)
        ext = p.suffix.lower()
        if ext == ".zip":
            dest = Path(workdir) / "unzipped" / slug(p.stem)
            dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(p) as z:
                for info in z.infolist():
                    name = Path(info.filename)
                    if info.is_dir() or ".." in name.parts or name.name.startswith("."):
                        continue
                    if name.suffix.lower() in AUDIO_EXT | BOOK_EXT:
                        target = dest / "__".join(name.parts)
                        with z.open(info) as src, open(target, "wb") as out:
                            shutil.copyfileobj(src, out)
            a, b = sort_inputs(sorted(dest.iterdir(), key=natural_key), workdir)
            audio += a
            books += b
        elif ext in AUDIO_EXT:
            audio.append(p)
        elif ext in BOOK_EXT:
            books.append(p)
    audio.sort(key=lambda q: natural_key(q.name))
    # prefer the richest book format
    rank = [".epub", ".docx", ".txt", ".md", ".fb2", ".azw3", ".mobi", ".azw", ".odt", ".rtf", ".html", ".htm", ".pdf"]
    books.sort(key=lambda q: rank.index(q.suffix.lower()) if q.suffix.lower() in rank else 99)
    return audio, books


def convert_audio(src, dst):
    """Decode anything ffmpeg reads to one 24 kHz mono 16-bit WAV (about 0.17 GB per hour), removing rumble/DC.

    Whisper reads 16 kHz slices resampled from this file on the fly, so no second copy is kept on disk.
    """
    dst.with_name(dst.stem + "-16k.wav").unlink(missing_ok=True)  # left over by earlier versions
    if dst.exists() and dst.stat().st_size > 1000:
        return
    tmp = dst.with_suffix(".part.wav")
    proc = subprocess.run([_bin("ffmpeg"), "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(SR),
                           "-af", "highpass=f=60", "-c:a", "pcm_s16le", str(tmp)], capture_output=True, text=True)
    if proc.returncode or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise ValueError(f"Couldn't decode {src.name}: {proc.stderr.strip()[-300:]}")
    tmp.rename(dst)


def read_slice(path, start_s, end_s, rate=SR):
    """Read only [start_s, end_s) seconds from disk (never the whole book)."""
    a = sf.read(str(path), start=max(0, int(start_s * rate)), stop=max(0, int(end_s * rate)), dtype="float32")[0]
    return a


# ---------- step 2: transcription ----------

def _quiet_point(audio, rate, around, span=8.0):
    """Index of the quietest 0.2 s within ±span seconds of `around` seconds."""
    lo, hi = int(max(0, around - span) * rate), int(min(len(audio) / rate, around + span) * rate)
    seg = audio[lo:hi].astype(np.float32)
    hop = int(0.05 * rate)
    if len(seg) < hop * 5:
        return int(around * rate)
    frames = len(seg) // hop
    e = np.sqrt((seg[:frames * hop].reshape(frames, hop) ** 2).mean(1) + 1e-12)
    e = _smooth(e, 4)
    return lo + int(np.argmin(e)) * hop


def _stt_window(audio16, language):
    import mlx.core as mx
    from mlx_audio.stt.utils import load_model as load_stt
    from ..engine import STT_MODEL
    model = slot.get("stt", lambda: load_stt(STT_MODEL))
    out = model.generate(mx.array(audio16), language=language, word_timestamps=True, verbose=None,
                                 condition_on_previous_text=False, hallucination_silence_threshold=2.0)
    words = []
    for seg in out.segments or []:
        for w in seg.get("words") or []:
            words.append([w["word"].strip(), float(w["start"]), float(w["end"])])
    return words


def free_stt():
    on_mlx_thread(slot.release)()


def transcribe_file(voice, wav24, cache, language, stop, progress):
    """Whisper the whole file in silence-cut windows; cache each window so a restart resumes.

    Holds at most one 10-minute window in memory, resampled from 24 kHz to Whisper's 16 kHz.
    """
    from scipy.signal import resample_poly
    data = json.loads(cache.read_text()) if cache.exists() else {"windows": []}
    total = sf.info(str(wav24)).frames / SR
    start = data["windows"][-1]["end"] if data["windows"] else 0.0
    while start < total - 0.5:
        if stop.is_set():
            raise InterruptedError
        if start + WINDOW >= total - 30:
            end = total
        else:
            a = read_slice(wav24, start + WINDOW - 10, start + WINDOW + 10)
            end = start + WINDOW - 10 + _quiet_point(a, SR, 10, 10) / SR
        audio = resample_poly(read_slice(wav24, start, end), 2, 3).astype(np.float32)  # 24k -> 16k
        words = on_mlx_thread(_stt_window)(audio, language)
        for w in words:
            w[1] = round(w[1] + start, 3)
            w[2] = round(w[2] + start, 3)
        data["windows"].append({"start": start, "end": end, "words": words})
        cache.write_text(json.dumps(data))
        start = end
        progress(end / total)
    return [w for win in data["windows"] for w in win["words"]]


# ---------- step 3/4: text + alignment ----------

_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
_TOKEN = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]|[^\W_]+(?:['’][^\W_]+)*", re.U)
_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
        "seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _num_words(n):
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + _num_words(n % 100))
    if n < 10000 and n % 1000 and 1100 <= n < 2100:  # years like 1984
        return _num_words(n // 100) + " " + (("oh " + _ONES[n % 100]) if n % 100 < 10 else _num_words(n % 100))
    if n < 1_000_000:
        return _num_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + _num_words(n % 1000))
    return str(n)


def norm_tokens(text, english=True):
    """Lowercased comparable tokens. Digits become words for English so '3' matches 'three'."""
    text = unicodedata.normalize("NFKC", text).replace("’", "'").lower()
    out = []
    for t in _TOKEN.findall(text):
        t = t.replace("'", "")
        if english and t.isdigit() and len(t) <= 6:
            out += _num_words(int(t)).split()
        elif t:
            out.append(t)
    return out


_ABBREV = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|St|Jr|Sr|Prof|Capt|Col|Gen|Lt|Sgt|Mt|vs|etc|No|Vol|Rev|Hon|i\.e|e\.g|[A-Z])\.$")
_SENT_END = re.compile(r"[.!?…][\"'”’»)\]]*(?=\s+[\"'“‘«(\[]?[A-Z0-9])")


def split_sentences(para):
    if _CJK.search(para):
        return [s for s in re.split(r"(?<=[。！？])", para) if s.strip()]
    out = []
    # never cut inside a quotation (clean_text made all double quotes straight)
    cuts = [0] + [m.end() for m in _SENT_END.finditer(para) if para[:m.end()].count('"') % 2 == 0] + [len(para)]
    for piece in (para[a:b].strip() for a, b in zip(cuts, cuts[1:])):
        if out and _ABBREV.search(out[-1]):
            out[-1] += " " + piece
        else:
            out.append(piece)
    return [s.strip() for s in out if s.strip()]


def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\[\d+\]|\[\w\]|(?<!\d[.,])(?<=[a-z.,;!?”\"])\d{1,3}(?=\s|$)", "", text)  # footnote markers
    text = text.replace("­", "").replace("​", "")
    text = re.sub(r"[“”„‟]", '"', text)
    text = re.sub(r"[‘’‚‛]", "'", text)
    text = re.sub(r"\s*—\s*", " — ", text)
    text = re.sub(r"\.{3,}|…", "...", text)
    text = re.sub(r"[*_#~|<>{}]", "", text)
    return " ".join(text.split())


def book_sentences(path, workdir, dialogue=False):
    """Book metadata, and the book as [(text, skip)] in reading order (skip: never made into a clip). With dialogue,
    the pieces are the characters' lines (the words in quotation marks, sentence by sentence) and, skipped, the
    narration between them. Also [(quote number, how it was said) or None] for each piece."""
    info = bk.import_book(path, Path(workdir) / "book")
    paras = [line.strip() for line in info["text"].split("\n\n") if line.strip()]
    single = dialogue and _single_quoted(paras)
    sentences, lines, quote = [], [], 0
    for line in paras:
        if line.startswith("## "):
            sentences.append((clean_text(line[3:]), True))
            lines.append(None)
            continue
        para = clean_text(line)
        if not dialogue:
            sentences += [(s, False) for s in split_sentences(para)]
            lines += [None] * (len(sentences) - len(lines))
            continue
        pieces = dialogue_pieces(para, single)
        for k, (text, said) in enumerate(pieces):
            if not said:
                sentences.append((text, True))
                lines.append(None)
                continue
            quote += 1
            how = said_how(pieces, k)
            for sent in split_sentences(text):
                sentences.append((_spoken(sent), False))
                lines.append((quote, how))
    return info, sentences, lines


# ---------- dialogue only: the characters' lines, without the narration ----------

_DQ = re.compile(r'"([^"]+)"')
# single-quoted speech ('Go away,' she said): opens after a space or the start, closes after punctuation
_SQ = re.compile(r"(?:^|(?<=[\s(\[—]))'(?=[^\s'])(.+?)(?<=[.,!?;:—-])'(?=[\s)\]—.,;:!?]|$)")
SPEECH_VERBS = (
    "said says asked replied answered added continued called cried shouted yelled screamed shrieked roared bellowed "
    "hollered whispered murmured muttered mumbled breathed hissed growled snarled snapped barked spat sneered "
    "laughed chuckled giggled sobbed wept wailed moaned groaned sighed gasped panted stammered stuttered blurted "
    "pleaded begged demanded insisted protested exclaimed declared announced warned teased joked grumbled "
    "complained whimpered squeaked purred drawled scoffed retorted interrupted urged suggested admitted agreed").split()
_HOW = re.compile(r"\b(?:(\w+ly)\s+)?(" + "|".join(SPEECH_VERBS) + r")\b(?:\s+(\w+ly)\b)?", re.I)


def _single_quoted(paras):
    """Whether the book marks speech with single quotes (as many British books do) instead of double ones."""
    sample = [clean_text(p) for p in paras[:2000] if not p.startswith("## ")]
    doubles = sum(len(_DQ.findall(p)) for p in sample)
    singles = sum(len(_SQ.findall(p)) for p in sample)
    return singles > 2 * doubles and singles >= 20


def dialogue_pieces(para, single=False):
    """[(text, said)] in order: the quoted words (said) and the narration around them. A quote still open at the end
    of the paragraph (speech running on into the next one) runs to its end."""
    if not single and para.count('"') % 2:
        para += '"'
    out, pos = [], 0
    for m in (_SQ if single else _DQ).finditer(para):
        before = para[pos:m.start()].strip()
        if before:
            out.append((before, False))
        if m.group(1).strip():
            out.append((m.group(1).strip(), True))
        pos = m.end()
    if para[pos:].strip():
        out.append((para[pos:].strip(), False))
    return out


def said_how(pieces, k):
    """How the line at pieces[k] was said, from the narration next to it ("she whispered angrily" → "whispered
    angrily"); "" when the book doesn't say."""
    for j in (k + 1, k - 1):
        if 0 <= j < len(pieces) and not pieces[j][1]:
            near = pieces[j][0]
            near = near[:80] if j > k else near[-80:]
            m = _HOW.search(near)
            if m:
                return " ".join(w for w in (m.group(2), m.group(1) or m.group(3)) if w).lower()
    return ""


def _spoken(text):
    """A quoted line as it's spoken on its own: "You're late," → "You're late." """
    text = text.strip()
    return text[:-1] + "." if text.endswith(",") else text


def _unique_ngrams(tokens, n):
    seen = {}
    for i in range(len(tokens) - n + 1):
        g = tuple(tokens[i:i + n])
        seen[g] = -1 if g in seen else i
    return {g: i for g, i in seen.items() if i >= 0}


def _lis(pairs):
    """Longest chain increasing in both coordinates (pairs sorted by first)."""
    tails, tails_idx, prev = [], [], [-1] * len(pairs)
    for k, (_, b) in enumerate(pairs):
        j = bisect_left(tails, b)
        if j == len(tails):
            tails.append(b)
            tails_idx.append(k)
        else:
            tails[j] = b
            tails_idx[j] = k
        prev[k] = tails_idx[j - 1] if j else -1
    out, k = [], tails_idx[-1] if tails_idx else -1
    while k >= 0:
        out.append(pairs[k])
        k = prev[k]
    return out[::-1]


def align(t_tokens, b_tokens, n=5):
    """Map transcript token index -> book token index for words that match."""
    tg, bg = _unique_ngrams(t_tokens, n), _unique_ngrams(b_tokens, n)
    anchors = sorted((i, bg[g]) for g, i in tg.items() if g in bg)
    chain = _lis(anchors)
    match = {}
    for i, j in chain:
        for k in range(n):
            match[i + k] = j + k
    # fill gaps between anchors with difflib
    keys = sorted(chain)
    for (i0, j0), (i1, j1) in zip(keys, keys[1:]):
        ts, te, bs, be = i0 + n, i1, j0 + n, j1
        if ts >= te or bs >= be or (te - ts) > 3000 or (be - bs) > 3000:
            continue
        sm = difflib.SequenceMatcher(None, t_tokens[ts:te], b_tokens[bs:be], autojunk=False)
        for a, b, size in sm.get_matching_blocks():
            for k in range(size):
                match[ts + a + k] = bs + b + k
    return match


# ---------- step 5/6: clips ----------

def _energy(audio, rate, hop=0.01):
    h = int(hop * rate)
    frames = len(audio) // h
    if frames == 0:
        return np.zeros(0)
    return np.sqrt((audio[:frames * h].astype(np.float32).reshape(frames, h) ** 2).mean(1) + 1e-12)


def _smooth(e, n):
    """Moving average over n frames, the ends padded with their own values. (Padding with zeros, as
    np.convolve(mode="same") does, makes the first and last frames look quieter than they are, so the quietest point
    was found at the edge of a pause, against a word, instead of inside it.)"""
    if len(e) < n:
        return e
    return np.convolve(np.pad(e, (n // 2, n - 1 - n // 2), mode="edge"), np.ones(n) / n, mode="valid")


CUT_EVEN_DB = 1.5  # frames this close to the quietest are as quiet: of those, the cut goes nearest the pause's middle


def _cut_point(audio, lo, hi, rate):
    """Where to cut between lo and hi seconds (a pause between words): in its quietest part, as far from the words on
    either side as it can be. Even room tone or digital silence is cut in the middle of the pause."""
    if hi - lo < 0.02:
        return (lo + hi) / 2
    e = _energy(audio[int(lo * rate):int(hi * rate)], rate)
    if len(e) == 0:
        return (lo + hi) / 2
    e = _smooth(e, 3)
    quiet = np.flatnonzero(e <= e.min() * 10 ** (CUT_EVEN_DB / 20))
    k = int(quiet[np.argmin(np.abs(quiet - (len(e) - 1) / 2))])
    return lo + (k + 0.5) * 0.01


def _trim(clip, rate, keep=0.12):
    e = _energy(clip, rate)
    if len(e) == 0:
        return clip
    thr = max(e.max() * 10 ** (-40 / 20), 10 ** (-60 / 20))
    loud = np.where(e > thr)[0]
    if len(loud) == 0:
        return clip[:0]
    s = max(0, int((loud[0] * 0.01 - keep) * rate))
    t = min(len(clip), int(((loud[-1] + 1) * 0.01 + keep) * rate))
    return clip[s:t]


def _quality(clip, rate):
    """(snr_db, clipped_fraction)."""
    e = _energy(clip, rate, 0.02)
    if len(e) < 10:
        return 0.0, 0.0
    db = 20 * np.log10(e)
    snr = float(np.percentile(db, 95) - np.percentile(db, 10))
    clipped = float(np.mean(np.abs(clip) > 0.995))
    return snr, clipped


def _level(clip, target_db=-20.0, peak=0.89):
    rms = np.sqrt(np.mean(clip ** 2) + 1e-12)
    clip = clip * (10 ** (target_db / 20) / rms)
    m = np.abs(clip).max()
    return clip * (peak / m) if m > peak else clip


# Pacing checks (plus pause limits). Long gaps and odd-speed clips teach a
# fine-tuned voice to stall mid-line and speed up / slow down at random.
PACE_RANGE = (0.75, 1.3)   # clip speed must be within this factor of the narrator's typical chars per second
MAX_SILENCE = 0.4          # reject clips that are more than 40% silence
MAX_PAUSE, PAUSE_TO = 0.8, 0.6   # pauses longer than 0.8 s inside a clip are shortened to 0.6 s
MAX_JOIN_GAP = 0.6         # only join sentences into one clip across a short pause


def silence_fraction(clip, rate=SR):
    hop = int(0.02 * rate)
    fr = clip[: len(clip) // hop * hop].reshape(-1, hop)
    if len(fr) == 0:
        return 1.0
    lv = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-9)
    return float((lv < lv.max() - 40).mean())


def shorten_pauses(clip, rate=SR, longest=MAX_PAUSE, to=PAUSE_TO):
    """Shorten silent runs inside the clip that are longer than `longest` to `to` seconds.

    Keeps the start and end of each pause (breath, decay) and crossfades 10 ms where the middle is removed.
    Returns (clip, number of pauses shortened).
    """
    hop = int(0.01 * rate)
    n = len(clip) // hop
    if n < 10:
        return clip, 0
    lv = 20 * np.log10(np.sqrt((clip[: n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)
    quiet = lv < lv.max() - 40
    loud = np.where(~quiet)[0]
    if len(loud) == 0:
        return clip, 0
    runs, i = [], loud[0]
    while i <= loud[-1]:
        if quiet[i]:
            j = i
            while j <= loud[-1] and quiet[j]:
                j += 1
            if (j - i) * 0.01 > longest:
                runs.append((i * hop, j * hop))
            i = j
        else:
            i += 1
    if not runs:
        return clip, 0
    half, fade = int(to / 2 * rate), int(0.01 * rate)
    ramp = np.linspace(1, 0, fade, dtype=np.float32)
    out, pos = [], 0
    for a, b in runs:
        head = clip[pos: a + half].copy()
        tail = clip[b - half: b].copy()
        head[-fade:] = head[-fade:] * ramp + tail[:fade] * ramp[::-1]
        out += [head, tail[fade:]]
        pos = b
    out.append(clip[pos:])
    return np.concatenate(out), len(runs)


def make_clips(sentences, words, b2t, matched, b_tok_spans, opts, lines=None):
    """Choose sentence groups that were read exactly.

    b2t: book token index -> transcript word index; matched: set of transcript words that matched the book.
    lines (dialogue only): per sentence, (quote number, how it was said) or None. Sentences are joined only within one
    quote, so a clip is one character's line; each clip ends with how it was said.
    """
    min_s, max_s, min_ratio = opts["min_seconds"], opts["max_seconds"], opts["min_match"]
    good = []  # per sentence: (file, t_first, t_last, start, end, text, ratio)
    for si, ((text, heading), (bs, bn)) in enumerate(zip(sentences, b_tok_spans)):
        if heading or bn == 0:
            good.append(None)
            continue
        hits = [b2t[b] for b in range(bs, bs + bn) if b in b2t]
        ratio = len(hits) / bn
        first, last = b2t.get(bs), b2t.get(bs + bn - 1)
        if ratio < min_ratio or first is None or last is None or last < first:
            good.append(None)
            continue
        # no extra speech inside: transcript words in span must be (nearly) all matched
        extra = sum(1 for t in range(first, last + 1) if t not in matched)
        span_ok = (last - first + 1) <= bn + 2 and extra <= max(1, bn // 15)
        fi = words[first][3]
        if not span_ok or words[last][3] != fi:
            good.append(None)
            continue
        good.append((fi, first, last, words[first][1], words[last][2], text, ratio, lines[si] if lines else None))

    clips, i = [], 0
    while i < len(good):
        g = good[i]
        if g is None:
            i += 1
            continue
        group = [g]
        j = i + 1
        while j < len(good) and good[j] is not None:
            nxt = good[j]
            dur = nxt[4] - group[0][3]
            gap = nxt[3] - group[-1][4]
            if nxt[0] != g[0] or nxt[1] != group[-1][2] + 1 or gap > opts.get("max_join_gap", MAX_JOIN_GAP) or dur > max_s:
                break
            if group[-1][4] - group[0][3] >= opts["target_seconds"]:
                break
            if lines and (nxt[7] or (None,))[0] != (g[7] or (None,))[0]:
                break  # (another character's line)
            group.append(nxt)
            j += 1
        fi, first, last = group[0][0], group[0][1], group[-1][2]
        prev_end = words[first - 1][2] if first > 0 and words[first - 1][3] == fi else max(0.0, words[first][1] - 0.5)
        next_start = words[last + 1][1] if last + 1 < len(words) and words[last + 1][3] == fi \
            else words[last][2] + 0.5
        text = " ".join(x[5] for x in group)
        score = sum(x[6] for x in group) / len(group)
        clips.append((fi, prev_end, words[first][1], words[last][2], next_start, text, score,
                      (g[7] or (0, ""))[1] if lines else None))
        i = j
    return [c for c in clips if min_s - 0.3 <= c[3] - c[2] <= max_s + 1]


# ---------- the job ----------

def process(voice, audio_paths, book_path, title, language, opts, stop):
    voice_dir = TRAINING / voice
    bslug = slug(title, "book")
    work = voice_dir / "books" / bslug
    (work / "audio").mkdir(parents=True, exist_ok=True)
    (work / "stt").mkdir(exist_ok=True)
    job = _jobs[voice]
    english = language in ("english", "auto")
    kept_book = work / f"source{Path(book_path).suffix.lower()}"
    if Path(book_path) != kept_book:
        shutil.copy2(book_path, kept_book)
        book_path = kept_book
    (work / "job.json").write_text(json.dumps({"audio": [str(p) for p in audio_paths], "book": str(book_path),
                                               "title": title, "language": language, "opts": opts}, indent=1))

    def stage(name, frac=0.0):
        job["stage"], job["progress"] = name, frac

    # 1. audio
    files24, total = [], 0.0
    for k, src in enumerate(audio_paths):
        if stop.is_set():
            raise InterruptedError
        stage(f"Converting audio {k + 1}/{len(audio_paths)}", k / len(audio_paths))
        d24 = work / "audio" / f"{k:03d}.wav"
        convert_audio(src, d24)
        files24.append(d24)
        total += sf.info(str(d24)).duration
    _log(voice, f"{title}: {len(files24)} audio file(s), {clock(total)} of audio.")

    # 2. book text
    stage("Reading the book", 0)
    dialogue = bool(opts.get("dialogue_only"))
    info, sentences, lines = book_sentences(book_path, work, dialogue)
    b_tokens, spans = [], []
    for text, _ in sentences:
        toks = norm_tokens(text, english)
        spans.append((len(b_tokens), len(toks)))
        b_tokens += toks
    _log(voice, f"Book text: {len(sentences)} sentences, {len(b_tokens):,} words.")
    if dialogue:
        said = [s for s, ln in zip(sentences, lines) if ln]
        quotes = len({ln[0] for ln in lines if ln})
        n_said = sum(len(norm_tokens(t, english)) for t, _ in said)
        _log(voice, f"Dialogue only: {quotes:,} lines in quotation marks, {n_said:,} words "
                    f"({n_said / max(1, len(b_tokens)):.0%} of the book). The narration is left out.")
        if not quotes:
            raise ValueError("No dialogue found: this book has no words in quotation marks. Turn off Performer.")

    # 3. transcribe (on this Mac via MLX Whisper, or on a cloud GPU via faster-whisper)
    backend = str(opts.get("backend") or "mac").strip() or "mac"
    cloud_gpu = str(opts.get("cloud_gpu") or "").strip()
    if backend != "mac":
        from .. import cloud_stt
        job["where"] = cloud_stt.where_name(backend)
        words = cloud_stt.transcribe_files(
            voice, files24, work / "stt", LANGS.get(language), backend, cloud_gpu,
            stop, stage, lambda m: _log(voice, m),
        )
    else:
        job["where"] = "this Mac"
        words = []
        done_before = 0.0
        for k, d24 in enumerate(files24):
            dur = sf.info(str(d24)).duration

            def prog(f, base=done_before, dur=dur):
                stage(f"Transcribing {clock(base + f * dur)} / {clock(total)}", (base + f * dur) / max(total, 1))
            ws = transcribe_file(voice, d24, work / "stt" / f"{k:03d}.json", LANGS.get(language), stop, prog)
            words += [[w, s, e, k] for w, s, e in ws]
            done_before += dur
    _log(voice, f"Transcribed {len(words):,} words.")

    # 4. align on token level (one transcript word can be several tokens, e.g. '1984')
    stage("Lining up audio with the book", 0)
    t_tokens, t_owner = [], []
    for wi, w in enumerate(words):
        for t in norm_tokens(w[0], english):
            t_tokens.append(t)
            t_owner.append(wi)
    tok_match = align(t_tokens, b_tokens)
    b2t = {}
    for ti, bi in tok_match.items():
        b2t.setdefault(bi, t_owner[ti])
    matched = {t_owner[ti] for ti in tok_match}
    covered = len(tok_match) / max(1, len(b_tokens))
    _log(voice, f"Matched {covered:.0%} of the book's words to the audio.")
    if covered < 0.2:
        raise ValueError("The audio and the book barely match (under 20% of words). Is this the same book "
                         "and edition, in the chosen language?")
    clips = make_clips(sentences, words, b2t, matched, spans, opts, lines if dialogue else None)

    # 5/6. cut, clean, filter
    out_dir = work / "wavs"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir()
    rows, rejected = [], {"noisy": 0, "clipped": 0, "length": 0, "cut": 0, "pace": 0, "silence": 0}
    typical = float(np.median([len(c[5]) / max(c[3] - c[2], 0.1) for c in clips])) if clips else 0.0
    for n, (fi, prev_end, s, e, next_start, text, score, how) in enumerate(clips):
        if stop.is_set():
            raise InterruptedError
        if n % 50 == 0:
            stage(f"Cutting clips {n}/{len(clips)}", n / max(1, len(clips)))
        # only this clip's stretch of audio (plus the pauses around it) is read from disk
        lo, hi = max(0.0, min(prev_end, s - 0.6)), max(next_start, e + 0.6)
        audio = read_slice(files24[fi], lo, hi)
        a = _cut_point(audio, max(prev_end, s - 0.6) - lo, s - lo, SR) + lo if s - prev_end > 0.02 else s
        b = _cut_point(audio, e - lo, min(next_start, e + 0.6) - lo, SR) + lo if next_start - e > 0.02 else e
        clip = _trim(audio[int((a - lo) * SR):int((b - lo) * SR)], SR)
        clip, _ = shorten_pauses(clip)
        dur = len(clip) / SR
        if not (opts["min_seconds"] <= dur <= opts["max_seconds"] + 0.5):
            rejected["length"] += 1
            continue
        body = np.sqrt(np.mean(clip ** 2) + 1e-12)
        edge = int(0.03 * SR)
        loud_edge = max(np.sqrt(np.mean(clip[:edge] ** 2)), np.sqrt(np.mean(clip[-edge:] ** 2))) if len(clip) > edge * 2 else body
        if 20 * np.log10(loud_edge / body + 1e-12) > -25:  # no pause at a boundary: a word would be cut
            rejected["cut"] += 1
            continue
        if silence_fraction(clip) > MAX_SILENCE:
            rejected["silence"] += 1
            continue
        if typical and not PACE_RANGE[0] * typical <= len(text) / dur <= PACE_RANGE[1] * typical:
            rejected["pace"] += 1  # likely a dramatic pause, a misread or extra words
            continue
        snr, clipped = _quality(clip, SR)
        if clipped > 0.001:
            rejected["clipped"] += 1
            continue
        if snr < opts["min_snr"]:
            rejected["noisy"] += 1
            continue
        clip = _level(clip)
        name = f"{bslug}-{len(rows) + 1:05d}.wav"
        sf.write(str(out_dir / name), clip, SR, subtype="PCM_16")
        rows.append({"file": name, "text": text, "seconds": round(dur, 2), "snr": round(snr, 1),
                     "match": round(score, 3), "source": f"{fi}:{a:.2f}-{b:.2f}", **({"how": how} if how else {})})
    with open(work / "clips.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    kept = sum(r["seconds"] for r in rows)
    meta = {"title": title, "author": info.get("author", ""), "language": language, "audio_seconds": total,
            "clips": len(rows), "kept_seconds": kept, "rejected": rejected, "matched_words": covered,
            "candidates": len(clips), "finished": time.strftime("%Y-%m-%d %H:%M"),
            **({"dialogue_only": True} if dialogue else {})}
    (work / "book.json").write_text(json.dumps(meta, indent=2))
    _log(voice, f"{title}: kept {len(rows)} clips ({kept / 3600:.2f} h of {total / 3600:.2f} h). "
                f"Dropped: {rejected['noisy']} noisy, {rejected['clipped']} clipped, {rejected['length']} wrong length, "
                f"{rejected['cut']} with no pause to cut at, {rejected['pace']} at an unusual pace, "
                f"{rejected['silence']} mostly silence.")
    if dialogue:
        from collections import Counter
        hows = Counter(r["how"] for r in rows if r.get("how"))
        if hows:
            _log(voice, "How the lines were said, from the book: "
                        + ", ".join(f"{h} {n}" for h, n in hows.most_common(12)) + ".")
    if opts.get("split_narrators"):  # more than one narrator: each becomes a voice of their own
        from . import narrators as nr
        try:
            mine, others = nr.split(voice, work, rows, stop, stage, lambda m: _log(voice, m),
                                    int(opts.get("narrators") or 0))
        finally:
            nr.free()
        if others:
            with open(work / "clips.jsonl", "w", encoding="utf-8") as f:
                for r in mine:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            meta.update(clips=len(mine), kept_seconds=round(sum(r["seconds"] for r in mine), 1), narrator=1)
            (work / "book.json").write_text(json.dumps(meta, indent=2))
            for other in others:
                build_dataset(other)
    build_dataset(voice)
    # the clips are written, so the full-length working audio and any unpacked uploads can go
    if not opts.get("keep_work"):
        freed = sum(f.stat().st_size for f in files24 if f.exists())
        shutil.rmtree(work / "audio", ignore_errors=True)
        shutil.rmtree(voice_dir / "uploads", ignore_errors=True)
        _log(voice, f"Deleted {freed / 1e9:.1f} GB of working audio (the transcript is kept).")
    (work / "job.json").unlink(missing_ok=True)


def clean_existing(voice, progress=None):
    """Apply the pacing checks to clips that were already made (no re-transcribing). Keeps a backup per book."""
    report = {"kept": 0, "dropped_pace": 0, "dropped_silence": 0, "shortened": 0, "before_h": 0.0, "after_h": 0.0}
    for bdir in sorted((TRAINING / voice / "books").glob("*")):
        cl = bdir / "clips.jsonl"
        if not cl.exists():
            continue
        rows = [json.loads(x) for x in cl.read_text(encoding="utf-8").splitlines() if x.strip()]
        backup = bdir / "before-cleanup"
        if not backup.exists():  # keep the original clips once; running again starts from them
            shutil.copytree(bdir / "wavs", backup / "wavs", copy_function=shutil.copy2)
            shutil.copy2(cl, backup / "clips.jsonl")
        else:
            rows = [json.loads(x) for x in (backup / "clips.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        report["before_h"] += sum(r["seconds"] for r in rows) / 3600
        # pass 1: shortened length of every clip, for the narrator's typical pace
        info = []
        for n, r in enumerate(rows):
            clip = sf.read(str(backup / "wavs" / r["file"]), dtype="float32")[0]
            short, k = shorten_pauses(clip)
            info.append((len(short) / SR, k, silence_fraction(short)))
            if progress and n % 200 == 0:
                progress(f"Checking {bdir.name}: {n}/{len(rows)}")
        typical = float(np.median([len(r["text"]) / d for r, (d, _, _) in zip(rows, info)]))
        # pass 2: write the kept clips
        kept = []
        for r, (dur, k, sil) in zip(rows, info):
            dst = bdir / "wavs" / r["file"]
            if sil > MAX_SILENCE or not PACE_RANGE[0] * typical <= len(r["text"]) / dur <= PACE_RANGE[1] * typical:
                report["dropped_silence" if sil > MAX_SILENCE else "dropped_pace"] += 1
                dst.unlink(missing_ok=True)
                continue
            if k:
                clip = sf.read(str(backup / "wavs" / r["file"]), dtype="float32")[0]
                sf.write(str(dst), shorten_pauses(clip)[0], SR, subtype="PCM_16")
                report["shortened"] += 1
            elif not dst.exists():
                shutil.copy2(backup / "wavs" / r["file"], dst)
            kept.append({**r, "seconds": round(dur, 2)})
        with open(cl, "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        report["kept"] += len(kept)
        report["after_h"] += sum(r["seconds"] for r in kept) / 3600
        meta_p = bdir / "book.json"
        if meta_p.exists():
            m = json.loads(meta_p.read_text())
            m.update(clips=len(kept), kept_seconds=sum(r["seconds"] for r in kept), cleaned=time.strftime("%Y-%m-%d %H:%M"))
            meta_p.write_text(json.dumps(m, indent=2))
    build_dataset(voice)
    _log(voice, f"Cleaned up: kept {report['kept']} clips ({report['after_h']:.2f} h of {report['before_h']:.2f} h), "
                f"dropped {report['dropped_pace']} at an unusual pace and {report['dropped_silence']} mostly silence, "
                f"shortened long pauses in {report['shortened']}. Originals are in books/*/before-cleanup/.")
    return report


# ---------- step 7: dataset files ----------

def build_dataset(voice, val_fraction=0.02):
    """Combine every processed book of this voice into wavs/ + metadata.csv + train/val jsonl + ref.wav."""
    vdir = TRAINING / voice
    ds = vdir / "dataset"
    rows = []
    for bdir in sorted((vdir / "books").glob("*")):
        cl = bdir / "clips.jsonl"
        if cl.exists():
            for line in cl.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                r["path"] = bdir / "wavs" / r["file"]
                rows.append(r)
    if not rows:
        return None
    if ds.exists():
        shutil.rmtree(ds)
    (ds / "wavs").mkdir(parents=True)
    for r in rows:
        dst = ds / "wavs" / r["file"]
        try:
            dst.hardlink_to(r["path"])
        except OSError:
            shutil.copy2(r["path"], dst)
    # reference clip: clean, 6–12 s, best SNR
    pool = [r for r in rows if 6 <= r["seconds"] <= 12] or rows
    ref = max(pool, key=lambda r: (r["snr"], r["match"]))
    shutil.copy2(ds / "wavs" / ref["file"], ds / "ref.wav")
    (ds / "ref.txt").write_text(ref["text"], encoding="utf-8")

    rnd = random.Random(0)
    order = rows[:]
    rnd.shuffle(order)
    n_val = max(1, int(len(order) * val_fraction)) if len(order) > 50 else 0
    val, train = order[:n_val], order[n_val:]
    with open(ds / "metadata.csv", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"{Path(r['file']).stem}|{r['text']}|{r['text']}\n")
    # Orpheus pipeline format (training/encode_snac.py reads datasets/<narrator>/metadata.jsonl)
    with open(ds / "metadata.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"audio": f"wavs/{r['file']}", "text": r["text"], "duration": r["seconds"]},
                               ensure_ascii=False) + "\n")
    link = TRAINING / "datasets" / voice
    link.parent.mkdir(exist_ok=True)
    if not link.exists() and not link.is_symlink():
        link.symlink_to(Path("..") / voice / "dataset")
    for name, part in (("train.jsonl", train), ("val.jsonl", val)):
        with open(ds / name, "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps({"audio": f"./wavs/{r['file']}", "text": r["text"], "ref_audio": "./ref.wav"},
                                   ensure_ascii=False) + "\n")
    hours = sum(r["seconds"] for r in rows) / 3600
    stats = {"voice": voice, "clips": len(rows), "hours": round(hours, 2), "train": len(train), "val": len(val),
             "sample_rate": SR, "ref": ref["file"],
             "books": [json.loads((b / "book.json").read_text()) for b in sorted((vdir / "books").glob("*"))
                       if (b / "book.json").exists()]}
    (ds / "stats.json").write_text(json.dumps(stats, indent=2))
    (ds / "README.md").write_text(README.format(voice=voice, clips=len(rows), hours=hours, train=len(train), val=len(val)))
    return stats


README = """# {voice}: fine-tuning dataset

{clips} clips, {hours:.2f} hours, 24 kHz mono 16-bit WAV. {train} train / {val} validation.

- `wavs/`: one sentence (or a few) per clip, 2–15 s, silence trimmed, levels evened out
- `train.jsonl`, `val.jsonl`: Qwen3-TTS format `{{"audio", "text", "ref_audio"}}`, paths relative to this folder
- `ref.wav` / `ref.txt`: the cleanest clip, used as the one shared speaker reference (24 kHz, as Qwen requires)
- `metadata.jsonl`: Orpheus pipeline format `{{"audio", "text", "duration"}}` (what `training/encode_snac.py` reads;
  this folder is also linked as `training/datasets/{voice}`)
- `metadata.csv`: LJSpeech format (`id|text|text`), for Coqui/Piper/StyleTTS2/F5 and others

## Fine-tune Qwen3-TTS (NVIDIA GPU with CUDA and FlashAttention 2; not on a Mac)

Checked against github.com/QwenLM/Qwen3-TTS/tree/main/finetuning (Nov 2026). Run from inside this folder, because
the jsonl paths are relative to the current directory:

```bash
pip install qwen-tts flash-attn
git clone https://github.com/QwenLM/Qwen3-TTS
python Qwen3-TTS/finetuning/prepare_data.py --device cuda:0 \\
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \\
  --input_jsonl train.jsonl --output_jsonl train_with_codes.jsonl
python Qwen3-TTS/finetuning/sft_12hz.py --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \\
  --output_model_path output --train_jsonl train_with_codes.jsonl \\
  --batch_size 2 --lr 2e-5 --num_epochs 3 --speaker_name {voice}
```
These are the script defaults (the README's other example uses batch 32, lr 2e-6, 10 epochs). Each epoch saves
`output/checkpoint-epoch-N`, a CustomVoice model with the speaker `{voice}`. Qwen's fine-tuning is single-speaker:
make one dataset per voice.
"""


# ---------- job control ----------

def start(voice, audio_paths, book_path, title, language, opts):
    with _lock:
        job = _jobs.get(voice)
        if job and job["thread"].is_alive():
            raise ValueError(f"“{voice}” is already processing. Wait for it to finish or stop it.")
        stop = threading.Event()
        backend = str((opts or {}).get("backend") or "mac").strip() or "mac"
        where_label = "this Mac"
        if backend != "mac":
            from .. import cloud_stt
            where_label = cloud_stt.where_name(backend)
        job = {"stop": stop, "log": [], "stage": "Starting", "progress": 0.0, "done": False, "error": None,
               "title": title, "started": time.time(), "where": where_label}
        _jobs[voice] = job

        def run():
            from ..runtime import LABEL
            LABEL.set(f"preparing training data for “{voice}”")  # for "Waiting its turn: Babble is busy with …"
            _mark(voice, True)
            try:
                process(voice, audio_paths, book_path, title, language, opts, stop)
                job["stage"], job["progress"], job["done"] = "Ready for fine-tuning", 1.0, True
                _mark(voice, False)
            except InterruptedError:
                job["stage"], job["error"] = "Stopped (start again to resume where it left off)", "stopped"
                _log(voice, "Stopped.")
                _mark(voice, False)  # stopped on purpose: not carried on when Babble opens
            except Exception as e:  # noqa: BLE001
                job["stage"], job["error"] = f"Failed: {e}", str(e)
                _log(voice, f"Failed: {e}\n{traceback.format_exc()[-1500:]}")
                _mark(voice, False)  # failed: carrying on would only fail again
            finally:
                if backend == "mac" and not any(j["thread"].is_alive() for v, j in _jobs.items() if v != voice and "thread" in j):
                    try:
                        free_stt()
                    except Exception:  # noqa: BLE001
                        pass

        job["thread"] = threading.Thread(target=run, daemon=True, name=f"train-{voice}")
        job["thread"].start()


PREPARING = "preparing.json"  # in a voice's folder while its training data is being made; left behind by a quit


def _mark(voice, on):
    f = TRAINING / voice / PREPARING
    try:
        if on:
            f.write_text(json.dumps({"since": time.time()}))
        else:
            f.unlink(missing_ok=True)
    except OSError:
        pass


def resume_interrupted(log=print):
    """When the engine starts: carry on the training data that was being made when Babble quit (voices whose
    marker is still there). Voices stopped on purpose, finished or failed have no marker."""
    for f in sorted(TRAINING.glob(f"*/{PREPARING}")) if TRAINING.exists() else []:
        voice = f.parent.name
        pending = unfinished(voice)
        if not pending:
            _mark(voice, False)
            continue
        j = pending[0]
        done_audio = (TRAINING / voice / "books" / slug(j["title"], "book") / "audio").exists()
        if any(not Path(x).exists() for x in [*j["audio"], j["book"]]) and not done_audio:
            log(f"[train] can't carry on preparing “{voice}”: its original files are gone")
            _mark(voice, False)
            continue
        where = str((j.get("opts") or {}).get("backend") or "mac")
        if where not in ("mac", "kaggle"):
            from .. import cloud_stt
            if not cloud_stt.in_flight(TRAINING / voice / "books" / slug(j["title"], "book")):
                # (its run isn't out: carrying on would rent a GPU again, which only the owner may start)
                log(f"[train] “{voice}” waits for Carry on: transcribing it on {where} again would cost money")
                _mark(voice, False)
                continue
        try:
            start(voice, [Path(x) for x in j["audio"]], Path(j["book"]), j["title"], j["language"], j["opts"])
            log(f"[train] carrying on preparing “{voice}” ({j['title']})")
        except Exception as e:  # noqa: BLE001 - one voice that can't carry on doesn't stop the others
            log(f"[train] couldn't carry on preparing “{voice}”: {e}")


def stop(voice):
    job = _jobs.get(voice)
    if job and job["thread"].is_alive():
        job["stop"].set()
        return True
    return False


def voices():
    return sorted(p.name for p in TRAINING.glob("*") if (p / "books").is_dir() or (p / "log.txt").exists()) \
        if TRAINING.exists() else []


def unfinished(voice):
    return [json.loads(j.read_text()) for j in sorted((TRAINING / voice / "books").glob("*/job.json"))] if voice else []


def status(voice):
    if not voice:
        return "", None
    job = _jobs.get(voice)
    lines = []
    if job:
        running = job["thread"].is_alive()
        lines.append(f"**{job['title']}**" + ("" if running else f" · {job['stage']}"))
    stats_path = TRAINING / voice / "dataset" / "stats.json"
    if stats_path.exists():
        s = json.loads(stats_path.read_text())
        lines.append(f"**Dataset:** {s['clips']:,} clips · **{s['hours']:.2f} h** · {len(s['books'])} book(s)  \n"
                     f"`{TRAINING / voice / 'dataset'}`")
        if s["hours"] < 1:
            lines.append("Tip: 1–2 h is enough for a solid single-speaker fine-tune; more books help.")
    log = TRAINING / voice / "log.txt"
    tail = "\n".join(log.read_text(encoding="utf-8").splitlines()[-12:]) if log.exists() else ""
    return "\n\n".join(lines) or "Nothing processed yet.", tail


def books_table(voice):
    rows = []
    for b in sorted((TRAINING / voice / "books").glob("*")) if voice else []:
        m = json.loads((b / "book.json").read_text()) if (b / "book.json").exists() else None
        if m:
            rows.append([m["title"], f"{m['audio_seconds'] / 3600:.2f}", f"{m['kept_seconds'] / 3600:.2f}",
                         m["clips"], f"{m['matched_words']:.0%}"])
        else:
            rows.append([b.name, "", "", "", "in progress"])
    return rows


def samples(voice, k=4):
    p = TRAINING / voice / "dataset" / "train.jsonl" if voice else None
    if not p or not p.exists():
        return []
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    pick = random.sample(rows, min(k, len(rows)))
    return [(str(TRAINING / voice / "dataset" / r["audio"][2:]), r["text"]) for r in pick]


def zip_dataset(voice):
    ds = TRAINING / voice / "dataset"
    if not ds.exists():
        raise ValueError("No dataset yet.")
    out = TRAINING / voice / f"{voice}-dataset.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(ds.rglob("*")):
            if f.is_file():
                z.write(f, f"{voice}/{f.relative_to(ds)}")
    return str(out)


def guess_title(audio, book_path):
    try:
        t = bk._metadata(book_path, TRAINING / "_meta")["title"]
        if t and t != Path(book_path).stem:
            return t
    except Exception:  # noqa: BLE001
        pass
    if audio:
        t = _audio_tag(audio[0], "album") or _audio_tag(audio[0], "title")
        if t:
            return t
    return Path(book_path).stem


# ---------- dashboard tab ----------
