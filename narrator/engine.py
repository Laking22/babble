"""Rendering across engines, plus small helpers the UI uses (transcribe, design takes, Orpheus check).

render() groups lines by engine and model, so each model loads once per render no matter how the cast
mixes Qwen, Orpheus (stock or any fine-tuned model) and Kokoro, then joins everything back in story order.
"""
import re
import time
from datetime import datetime
from pathlib import Path

try:
    import mlx.core as mx
except ImportError:
    mx = None
import numpy as np
import soundfile as sf

from . import settings, tts
from .audio import clock, cut_short, finite, is_silent, is_soft, level, rushed, stretch, trim_edges, voiced_seconds
from .cast import plain
from .paths import OUTPUTS
from .runtime import SR, free, on_mlx_thread, slot, streamed_on_mlx_thread
from .status import Status
from .tts import orpheus as _orpheus
from .tts import qwen as _qwen

STT_MODEL = "mlx-community/whisper-large-v3-turbo-asr-8bit"

__all__ = ["render", "spoken", "split_chunks", "clock", "OUTPUTS", "SR", "transcribe", "design_takes", "orpheus_check",
           "free", "on_mlx_thread", "streamed_on_mlx_thread", "release_models"]


def release_models():
    """Free whatever voice model is loaded (e.g. so a language model can use the memory)."""
    slot.release()


ORPHEUS_MAX_CHARS = 190  # 90% of the training clips are under ~196 characters; longer lines can drift or rush
# Trained Qwen voices (Settings → "Longest line", settings.longest_line): takes said whole that lost 3+ words in a row
# (one audiobook, checked with Whisper): 161-190 chars 1.6%, 191-215 4-6%, 216-240 15-17%, 241-300 29-33%. Each cut
# costs a pitch drop and reset (-1.9/+4.8 semitones at commas), so it's a trade-off: 240 by default.
TRAINED_MAX_CHARS = 240


def longest_line():
    """The longest line a trained voice is given (Settings slider), or 0: sentences are never cut."""
    try:
        return max(0, int(settings.load().get("longest_line", TRAINED_MAX_CHARS)))
    except (TypeError, ValueError):
        return TRAINED_MAX_CHARS


def chunk_limit(voice, max_chars):
    """Orpheus voices and trained Qwen voices get lines no longer than the Settings slider (Orpheus never longer than
    what it was trained on); 0 on the slider: never cut."""
    cap = longest_line()
    if not cap:
        return int(max_chars)
    v = voice or ""
    if v.startswith("orpheus:"):
        return min(int(max_chars), ORPHEUS_MAX_CHARS, cap)
    if v.startswith("qwen:"):
        return min(int(max_chars), cap)
    return int(max_chars)


TAG_SPANS = re.compile(r"\{[^{}]*\}|\[[^\]]*\]|<[^<>]*>")
# Strong clause breaks (dashes, semicolons, colons) cause far less mid-sentence cadence drop than commas
STRONG_BREAK = re.compile(r"(?<=[;:])\s+|(?<=[—–])\s*(?=\S)|\s+(?=[—–-]\s)")
WEAK_BREAK = re.compile(r"(?<=[,)])\s+")
ABBREV = re.compile(r"(?:^|\s)(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Rev|Gen|Col|Lt|Sgt|Capt|Hon|vs|etc|[A-HJ-Z])\.$")
# a lone capital after these words is a grade or a name for something, not an initial: "He got an A. Then…"
NOT_INITIAL = re.compile(r"(?i:\b(?:a|an|plan|vitamin|grade|type|letter|class|row|section|block|gate)\s+)[A-Z]\.$")
SAFE_CHARS = 200  # pieces this short rarely skip words (1.6% at 161-190 chars, 4-6% at 191-215, 15-17% at 216-240)


def _pieces(sentence, max_chars):
    """Cut a sentence longer than max_chars into the fewest pieces, preferring strong breaks (— ; :) over commas
    so pitch doesn't drop and reset mid-clause; never cuts inside tags or numbers."""
    if len(sentence) <= max_chars:
        return [sentence]
    inside = [(m.start(), m.end()) for m in TAG_SPANS.finditer(sentence)]
    free = lambda k: 0 < k < len(sentence) and not any(a < k < b for a, b in inside)  # noqa: E731
    mid = len(sentence) / 2

    def pick(rx, margin=25, lo=None, hi=None, fit=None):
        a = max(margin, int(lo)) if lo is not None else margin
        b = min(len(sentence) - margin, int(hi)) if hi is not None else len(sentence) - margin
        cuts = [m.start() for m in rx.finditer(sentence) if free(m.start()) and a <= m.start() <= b
                and (fit is None or (m.start() <= fit and len(sentence) - m.start() <= fit))]
        if not cuts:
            return None
        # Prefer a cut where both halves already fit under max_chars (fewest total pieces), else nearest the middle
        clean = [c for c in cuts if c <= max_chars and len(sentence) - c <= max_chars]
        return min(clean or cuts, key=lambda k: abs(k - mid))

    # A strong break wins anywhere when both pieces land in the low-skip band (no comma's pitch reset, no skip risk);
    # otherwise only one in the middle 30-70%, so a semicolon near char 35 doesn't leave a long, skip-prone tail
    c = pick(STRONG_BREAK, 25, fit=min(max_chars, SAFE_CHARS))
    if c is None:
        c = pick(STRONG_BREAK, 25, lo=0.3 * len(sentence), hi=0.7 * len(sentence))
    if c is None:
        c = pick(WEAK_BREAK, 15)
    if c is None:
        c = pick(STRONG_BREAK, 25)
    if c is None:
        c = pick(re.compile(r"\s+"), 15)
    if c is None:
        return [sentence]
    left, right = sentence[:c].strip(), sentence[c:].strip()
    if not left or not right:
        return [sentence]
    return _pieces(left, max_chars) + _pieces(right, max_chars)


SENTENCE_BREAK = re.compile(r"(?<=[.!?…])\s+|(?<=[.!?…][”\"’'])\s+")
HALF_GAP = 0.06  # seconds between the two halves of a line made again in two (as books._gap between cut pieces)


def _halves(text):
    """A line cut in two at its best break near the middle (between sentences, else ; : —, else a comma, else a
    space), never inside a tag or a number: for a long line that keeps leaving words out. None when it can't be."""
    text = (text or "").strip()
    n = len(text)
    if n < 80:
        return None
    inside = [(m.start(), m.end()) for m in TAG_SPANS.finditer(text)]
    for rx in (SENTENCE_BREAK, STRONG_BREAK, WEAK_BREAK, re.compile(r"\s+")):
        cuts = [m.start() for m in rx.finditer(text)
                if 0.3 * n <= m.start() <= 0.7 * n and not any(a < m.start() < b for a, b in inside)]
        if cuts:
            c = min(cuts, key=lambda k: abs(k - n / 2))
            left, right = text[:c].strip(), text[c:].strip()
            if re.search(r"\w", TAG_SPANS.sub("", left)) and re.search(r"\w", TAG_SPANS.sub("", right)):
                return [left, right]
    return None


def split_chunks(paragraph, max_chars):
    raw = re.findall(r".+?(?:(?:[.!?…][”\"’']?(?=\s+[“\"‘'A-Z\u00C0-\u024F\u0400-\u04FF\u3040-\u9FFF\uAC00-\uD7AF])|[。！？][」』”\"’']?)|$)", paragraph, flags=re.S)
    sentences = []
    for s in raw:
        s = s.strip()
        if not s:
            continue
        if sentences and ABBREV.search(sentences[-1]) and not NOT_INITIAL.search(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {s}"
        else:
            sentences.append(s)
    if longest_line():
        sentences = [p for s in sentences for p in _pieces(s, max_chars) if p]
    chunks, current = [], ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def spoken(items, sound=True, exact=None):
    """The items as render() speaks them: sound tags turned into cue items (or dropped when sound is off),
    markdown marks taken out, and every line in the exact voice when there is one."""
    from . import sfx
    if any("cue" not in it and sfx.CUE.search(it.get("text") or "") for it in items):
        if not sound:  # sound off: drop the sound tags so they're never read aloud (pauses aren't sounds: they stay)
            items = [dict(it, text=sfx.strip(it["text"], keep_pauses=True)) for it in items
                     if "cue" in it or sfx.strip(it["text"], keep_pauses=True)]
            items = [it for it in items if "cue" not in it or it["cue"]["type"] in ("intro", "outro")]  # chosen intros stay
        items = [x for it in items for x in (sfx.cue_items(it["text"], it) if "cue" not in it else [it])]
    items = _speakable(items)
    if exact:
        items = [it if "cue" in it else dict(it, voice=f"my:{exact}") for it in items]
    return items


def _plan(items, skip=()):
    """[(engine, model_key, [indexes])] with the model that's already loaded first. skip: indexes already made."""
    groups = {}
    for n, it in enumerate(items):
        if "cue" in it or n in skip:
            continue  # sound-effect markers take no voice
        e = tts.engine_for(it["voice"])
        if tts.is_off(e.id):
            raise ValueError(f"{e.name} is turned off (Models page), and a line uses its voice "
                             f"“{it['voice'].partition(':')[2] or it['voice']}”. Turn it on, or pick another voice.")
        groups.setdefault((e.id, e.model_key(it["voice"])), []).append(n)
    loaded = slot.key or ""
    order = sorted(groups.items(), key=lambda kv: (not loaded.startswith(kv[0][0]), kv[1][0]))
    return [(tts.ENGINES[eid], mk, idx) for (eid, mk), idx in order]


def _speakable(items):
    """Markdown marks (**bold**, *italic*, _italic_) are never read. A line left with no words at all ("**", "* * *")
    isn't spoken; its pause goes to the line before, so a paragraph break stays a paragraph break."""
    out = []
    for it in items:
        if "cue" in it:
            out.append(it)
            continue
        text = plain(it.get("text") or "")
        if re.search(r"\w", text):
            out.append(dict(it, text=text))
            continue
        prev = next((x for x in reversed(out) if "cue" not in x), None)
        if prev is not None:
            prev["pause"] = max(prev.get("pause", 0.3), it.get("pause", 0.3))
    return out


def _failed(wav, item):
    """A line with words in it that came back empty, silent or broken."""
    text = _orpheus.TAGS.sub("", item.get("text") or "")
    return wav is not None and bool(re.search(r"\w", text)) and is_silent(wav)


def _cut(wav, item):
    """A line that came back with far less sound in it than its words take: cut off, or only a breath or a click
    (a short line at low stability, most often)."""
    return wav is not None and cut_short(wav, item.get("text") or "")


def _better(new, old, item, quick=False):
    """Whether a new take of a line beats the one it has: never a silent one; else when the old one was silent or
    missing, the new one isn't cut short, or both are and the new one has more sound in it.
    quick: the old take left words out (audio.rushed): a new one must have more sound in it too (more words said)."""
    if new is None or _failed(new, item):
        return False
    if old is None or _failed(old, item):
        return True
    if not _cut(new, item):
        return _cut(old, item) or not quick or voiced_seconds(new) > voiced_seconds(old)
    return _cut(old, item) and voiced_seconds(new) > voiced_seconds(old)


def _log_line(n, total, e, item, wav, retry=False):
    """One line per spoken line in the app log (~/Library/Logs/Babble.log), so a bad line can be traced."""
    peak = float(np.abs(wav).max()) if len(wav) else 0.0
    who = f"{item['who']} · " if item.get("who") else ""
    flag = "  << SILENT" if _failed(wav, item) else "  << CUT SHORT" if _cut(wav, item) else ""
    print(f"[line {n}/{total}{' again' if retry else ''}] {e.short or e.id} · {who}{item['voice']} · "
          f"{len(wav) / SR:.1f} s · peak {20 * np.log10(peak + 1e-9):.0f} dB · {item['text'][:60]!r}{flag}", flush=True)


LINES_SHARE, STITCH_SHARE, MIX_SHARE = 0.85, 0.05, 0.05  # of a chapter's progress: its lines, stitching, sounds


@streamed_on_mlx_thread
def render(items, *, kind, language, temperature, top_p=1.0, top_k=50, repetition_penalty=1.05,
           seed=0, batch=2, exact=None, path=None, sound=True, bed=1.0, stop=None, parallel=None, premade=None,
           keep=None, cache_seed=None):
    """Generate and join items. Yields (None, status) while working, then (path, status).

    items: dicts with text, voice ('builtin:x' / 'my:x' / 'orpheus:x' / 'kokoro:x'), instruct, speed, pause, who.
    exact: a saved voice name to use with full transcript cloning instead (Qwen; slower, no emotions).
    batch / parallel: lines made at once. parallel False makes every model go one line at a time; True lets
    Orpheus make `batch` lines at once too (Qwen always uses batch); None leaves Orpheus to its Models setting.
    premade: a folder of lines already made elsewhere (in the cloud): those lines are taken from it, not made again.
    keep: a folder of lines made on this Mac (books.LINES): lines made before with the same words, voice and
    settings are taken from it, and new ones go in it, so a chapter whose sounds change is only mixed again.
    """
    yield None, Status("Preparing voices…", 0.0)
    from . import sfx
    items = spoken(items, sound, exact)
    opts = dict(language=language, temperature=temperature, top_p=top_p, top_k=int(top_k),
                repetition_penalty=repetition_penalty, batch=1 if parallel is False else batch, exact=exact,
                strip_tags=_orpheus.TAGS, parallel=parallel)
    cseed = seed if cache_seed is None else cache_seed
    if int(seed) > 0 and mx is not None:
        mx.random.seed(int(seed))
    ready = {}  # lines made in the cloud, or here before: index -> audio
    doubtful = {}  # lines that came back from the cloud silent or cut short: made here, these kept to fall back on
    hasty = set()  # lines whose take was far quicker than the rest of its voice's: words left out (audio.rushed)
    from_cloud, cloud_at = 0, set()
    total = sum(1 for it in items if "cue" not in it)
    # how far along a chapter is: its lines (brought in or made) are the first LINES_SHARE, then stitching,
    # sound effects and saving
    share = lambda n: LINES_SHARE * n / max(total, 1)  # noqa: E731
    if (premade or keep) and not exact:
        from . import cloud_book
        looked = 0
        for i, it in enumerate(items):
            if "cue" not in it:
                if stop is not None and stop.is_set():
                    break
                wav = cloud_book.premade_line(premade, it, opts, cseed) if premade else None
                if wav is not None and (_failed(wav, it) or _cut(wav, it)) and cloud_book._runs_here(it.get("voice")):
                    doubtful[i] = wav  # silent or cut short in the cloud: made here instead (kept to fall back on)
                    wav = None
                if wav is not None:
                    from_cloud += 1
                    cloud_at.add(i)
                elif keep:
                    wav = cloud_book.kept_line(keep, it, opts, cseed)
                if wav is not None:
                    ready[i] = wav
                looked += 1
                if looked % 200 == 0:  # (the bar counts the lines found: lines still to make come after)
                    yield None, Status(f"Loading saved audio ({looked:,}/{total:,})",
                                       share(len(ready)))
        # lines that left words out (far quicker than the rest of their voice's): made here again
        quick = rushed({i: (items[i]["voice"], items[i]["text"], w) for i, w in ready.items()
                        if cloud_book._runs_here(items[i].get("voice"))})
        for i in quick:
            doubtful[i] = ready.pop(i)
            hasty.add(i)
            if i in cloud_at:
                from_cloud -= 1
        if hasty:
            print(f"[lines] {len(hasty)} line(s) made before were too quick for their voice (words left out): "
                  "making them here", flush=True)
        if len(doubtful) > len(hasty):
            print(f"[lines] {len(doubtful) - len(hasty)} line(s) from the cloud came out silent or cut short: "
                  "making them here", flush=True)
    plan = _plan(items, skip=ready)
    for e in {e.id: e for e, _, _ in plan}.values():
        if stop is not None and stop.is_set():  # (stopped while the cloud lines came in: no model to load)
            break
        e.prepare([items[i] for ee, _, idx in plan if ee is e for i in idx], opts)

    path = Path(path) if path else OUTPUTS / f"{kind}-{datetime.now():%Y%m%d-%H%M%S}.wav"
    wavs = [None] * len(items)
    for i, wav in ready.items():
        wavs[i] = wav
    done, made, started = len(ready), 0, time.time()
    if ready:
        yield None, Status(f"Loaded {len(ready):,}/{total:,} lines" + (" · speaking remaining…" if plan else ""), share(done))
    line_no = {i: n + 1 for n, i in enumerate(i for i, it in enumerate(items) if "cue" not in it)}
    gaps = []  # lines that came out silent twice: left out, and the file is named so it gets made again
    heads_up = {e.id: e.note() for e, _, _ in plan if e.note()}  # e.g. VibeVoice on an 8 GB Mac
    closing = False
    try:
        for e, _, idx in plan:
            sub = [items[i] for i in idx]
            gen = e.speak(sub, opts, stop)
            k = 0
            while True:
                if stop is not None and stop.is_set():
                    break
                nxt = sub[min(k, len(sub) - 1)]
                who = f"{nxt['who']} · " if nxt.get("who") else ""
                note = f"\n{heads_up[e.id]}" if e.id in heads_up else ""
                yield None, Status(f"Speaking ({done + 1}/{total})\n\n{who}{e.name}: "
                                   f"{nxt['text']}{note}", share(done))
                try:
                    j, wav = next(gen)
                except StopIteration:
                    break
                wavs[idx[j]] = finite(wav)
                _log_line(line_no[idx[j]], total, e, items[idx[j]], wavs[idx[j]])
                done += 1
                made += 1
                k += 1
            gen.close()
            if mx is not None:
                mx.clear_cache()
            if stop is not None and stop.is_set():
                break
            # A model can hand back a line of pure silence without any error (it happened to a whole paragraph
            # with Orpheus batching on), or one cut short (a short line at low stability). Make those lines again,
            # one at a time, the long-tested way.
            bad = [i for i in idx if _failed(wavs[i], items[i]) or _cut(wavs[i], items[i])]
            # and takes far quicker than the rest of their voice's in this chapter: words left out (audio.rushed)
            mine = set(idx)
            quick = rushed({i: (it["voice"], it["text"], wavs[i]) for i, it in enumerate(items)
                            if "cue" not in it and wavs[i] is not None}) & mine
            quick -= set(bad)
            hasty |= quick
            bad += sorted(quick)
            if bad:
                print(f"[lines] {len(bad)} silent or cut short line(s) from {e.name}: making them again one at a time",
                      flush=True)
                again = e.speak([items[i] for i in bad], dict(opts, one_at_a_time=True), stop)
                try:
                    for i in bad:
                        if stop is not None and stop.is_set():
                            break
                        who = f"{items[i]['who']} · " if items[i].get("who") else ""
                        how = ("silent" if _failed(wavs[i], items[i]) else "cut short" if _cut(wavs[i], items[i])
                               else "missing words")
                        yield None, Status(f"Retrying line {line_no[i]} ({how})\n\n{who}{e.name}: "
                                           f"{items[i]['text']}", share(done))
                        try:
                            j, wav = next(again)
                        except StopIteration:
                            break
                        wav = finite(wav)
                        _log_line(line_no[bad[j]], total, e, items[bad[j]], wav, retry=True)
                        if _better(wav, wavs[bad[j]], items[bad[j]], bad[j] in hasty):
                            wavs[bad[j]] = wav
                finally:
                    again.close()
                    if mx is not None:
                        mx.clear_cache()
                # a long line still too quick after its retry: made once more as two halves cut at its best break
                # (so long lines keep their flow unless they actually leave words out)
                still = rushed({i: (it["voice"], it["text"], wavs[i]) for i, it in enumerate(items)
                                if "cue" not in it and wavs[i] is not None}) & set(bad) & hasty
                halves = {i: h for i in sorted(still) if (h := _halves(items[i]["text"]))}
                if halves and not (stop is not None and stop.is_set()):
                    print(f"[lines] {len(halves)} line(s) still missing words: making each again in two halves",
                          flush=True)
                    yield from _remake_in_halves(e, items, wavs, halves, opts, stop, line_no, share(done))
                for i in bad:
                    if i in doubtful and _better(doubtful[i], wavs[i], items[i], i in hasty):
                        wavs[i] = doubtful[i]  # the cloud's take was the better one after all
                    if _failed(wavs[i], items[i]):
                        gaps.append(i)
                        wavs[i] = None  # left out: better than a long stretch of dead air
            if stop is not None and stop.is_set():
                break
    except GeneratorExit:  # closed (the task was stopped): still saved below, but nothing more can be yielded
        closing = True
        raise
    finally:
        parts, audio_seconds, pos, cues, spans, notes, timings = [], 0.0, 0, [], [], [], []
        last, anchored = None, []  # the line before a cue, for effects placed on a word ({sfx: knock @knocked})
        # takes that still leave words out aren't kept, so they're tried again next time
        still_quick = rushed({i: (it["voice"], it["text"], wavs[i]) for i, it in enumerate(items)
                              if "cue" not in it and wavs[i] is not None}) if keep and hasty else set()
        for n, it in enumerate(items):
            wav, wavs[n] = wavs[n], None  # (let each line's audio go once it's stitched in: a long book is GBs)
            if (keep and not exact and wav is not None and n not in ready and "cue" not in it and not _failed(wav, it)
                    and not _cut(wav, it) and n not in still_quick):  # (one still cut short is tried again next time)
                cloud_book.keep_line(keep, it, opts, cseed, wav)  # (made here: kept for the next time)
            if not closing and n % 500 == 499:
                try:
                    yield None, Status("Assembling audio…",
                                       LINES_SHARE + STITCH_SHARE * n / max(len(items), 1))
                except GeneratorExit:  # stopped while stitching: the chapter is still saved (as -partial)
                    closing = True
            if "cue" in it and it["cue"]["type"] == "pause":  # {pause: 1.5}: that much silence, here
                gap = np.zeros(int(SR * it["cue"]["seconds"]), dtype=np.float32)
                parts.append(gap)
                pos += len(gap)
                audio_seconds += len(gap) / SR
                continue
            if "cue" in it:
                if it["cue"].get("at") and last is not None and sound:
                    anchored.append((len(cues), last))
                cues.append((pos, it["cue"]))
                if it["cue"]["type"] in ("intro", "outro"):  # on its own this long, before the first words (or at the end)
                    gap = np.zeros(int(SR * it["cue"].get("lead", 0)), dtype=np.float32)
                    parts.append(gap)
                    pos += len(gap)
                    audio_seconds += len(gap) / SR
                continue
            if wav is None:
                continue
            wav = stretch(level(trim_edges(wav), soft=is_soft(it.get("instruct") or "", it.get("text") or "")), float(it.get("speed") or 1.0))
            last = (pos, wav)
            pause = it.get("pause", 0.3)
            spans.append((pos, pos + len(wav)))
            timings.append({
                "para": it.get("para"),
                "who": it.get("who") or "",
                "text": it.get("text") or "",
                "start": round(pos / SR, 3),
                "end": round((pos + len(wav)) / SR, 3),
            })
            gap = np.zeros(int(SR * pause), dtype=np.float32)
            parts += [wav, gap]
            pos += len(wav) + len(gap)
            audio_seconds += len(wav) / SR + pause
        finished = done == total
        if gaps and finished:
            gaps.sort()
            listed = "; ".join(f"line {line_no[i]} ({items[i]['who'] + ': ' if items[i].get('who') else ''}"
                               f"“{items[i]['text'][:50]}{'…' if len(items[i]['text']) > 50 else ''}”)"
                               for i in gaps[:3]) + (f"; and {len(gaps) - 3} more" if len(gaps) > 3 else "")
            notes.append(f"Left out {len(gaps)} line(s) that came out silent twice: {listed}. "
                         "Make it again to try those lines once more.")
            path = path.with_name(path.stem + "-gaps.wav")  # an audiobook chapter with gaps is never reused as is
            print(f"[lines] still silent after a second try: {[line_no[i] for i in gaps]}", flush=True)
        if parts:
            if not finished:
                path = path.with_name(path.stem + "-partial.wav")
            after = LINES_SHARE + STITCH_SHARE
            if cues:  # the sounds are mixed into the whole chapter at once
                audio = np.concatenate(parts)
                parts = None
                if anchored and finished:
                    if not closing:
                        try:
                            yield None, Status("Syncing sound effects…", after)
                        except GeneratorExit:
                            closing = True
                    cues = _place_on_words(cues, anchored, notes.append)
                if not closing:
                    try:
                        yield None, Status("Mixing audio…", after)
                    except GeneratorExit:
                        closing = True
                audio = sfx.mix(audio, cues, spans, log=notes.append, bed=bed)
                chunk = SR * 600  # saved ten minutes at a time
                parts = [audio[k:k + chunk] for k in range(0, len(audio), chunk)]
                after += MIX_SHARE
            elif kind in ("chapter", "story-preview") and spans:
                # gentle continuous studio room tone on dry narration chapters so pauses never drop to digital dead air
                audio = np.concatenate(parts)
                audio = audio + sfx.room_tone(len(audio), stereo=False)
                chunk = SR * 600
                parts = [audio[k:k + chunk] for k in range(0, len(audio), chunk)]
            # saved piece by piece to a sibling temp file, then atomically renamed so a crash/ENOSPC never leaves a truncated chapter
            channels = parts[0].shape[1] if parts[0].ndim > 1 else 1
            tmp_wav = path.with_name(f".{path.name}.tmp")
            try:
                with sf.SoundFile(str(tmp_wav), "w", SR, channels, "PCM_16", format="WAV") as out:
                    for k, part in enumerate(parts):
                        out.write(part)
                        if not closing and k % 400 == 399:
                            try:
                                yield None, Status("Saving chapter…",
                                                   after + (1 - after) * k / max(len(parts), 1))
                            except GeneratorExit:
                                closing = True
                import os as _os
                _os.replace(tmp_wav, path)
            finally:
                tmp_wav.unlink(missing_ok=True)
            if kind == "chapter" and timings:
                try:
                    import json as _json
                    import os as _os
                    t_path = path.with_suffix(".timings.json")
                    t_tmp = t_path.with_name(f".{t_path.name}.tmp")
                    try:
                        t_tmp.write_text(_json.dumps(timings, ensure_ascii=False), encoding="utf-8")
                        _os.replace(t_tmp, t_path)
                    finally:
                        t_tmp.unlink(missing_ok=True)
                except OSError:
                    pass
    if closing:  # stopped: saved above, and a closed generator can't say so
        return
    took = max(time.time() - started, 1e-6)
    extra = ("\n" + "\n".join(dict.fromkeys(notes))) if notes else ""
    speed = (("every line made in the cloud" if from_cloud == len(ready) else "every line reused: only mixed again")
             if ready and not made else
             f"in {clock(took)} ({audio_seconds / took:.2f}× real time)")
    yield str(path), f"Done · {clock(audio_seconds)} of audio {speed}\n{path}{extra}"


def _remake_in_halves(e, items, wavs, halves, opts, stop, line_no, progress):
    """Make each line in `halves` ({index: [left, right]}) as two lines, one at a time, and join them with a short
    breath (HALF_GAP); the joined take replaces the line's take only when it has more sound in it (_better)."""
    parts = [(i, dict(items[i], text=t)) for i, pair in halves.items() for t in pair]
    got = [None] * len(parts)
    gen = e.speak([p for _, p in parts], dict(opts, one_at_a_time=True), stop)
    try:
        for k, (i, part) in enumerate(parts):
            if stop is not None and stop.is_set():
                break
            who = f"{items[i]['who']} · " if items[i].get("who") else ""
            yield None, Status(f"Retrying line {line_no[i]} in two halves ({k % 2 + 1}/2, missing words)\n\n"
                               f"{who}{e.name}: {part['text']}", progress)
            try:
                j, wav = next(gen)
            except StopIteration:
                break
            got[j] = finite(wav)
    finally:
        gen.close()
        if mx is not None:
            mx.clear_cache()
    for n, i in enumerate(halves):
        a, b = got[2 * n], got[2 * n + 1]
        it = items[i]
        if a is None or b is None or _failed(a, dict(it, text=parts[2 * n][1]["text"])) \
                or _failed(b, dict(it, text=parts[2 * n + 1][1]["text"])):
            continue
        soft = is_soft(it.get("instruct") or "", it.get("text") or "")
        joined = np.concatenate([level(trim_edges(a), soft=soft), np.zeros(int(SR * HALF_GAP), np.float32),
                                 level(trim_edges(b), soft=soft)]).astype(np.float32)
        took = _better(joined, wavs[i], it, quick=True)
        print(f"[line {line_no[i]} halves] {len(joined) / SR:.1f} s · {'used' if took else 'not better: kept the take'}",
              flush=True)
        if took:
            wavs[i] = joined


def _place_on_words(cues, anchored, log):
    """Move sounds tagged with a word onto the moment that word is spoken (Whisper word timings): effects written
    as {sfx: knock @knocked} after a sentence, and any tag dropped inside a sentence. Lines are never cut in half.
    Runs on the MLX thread."""
    from mlx_audio.stt.utils import load_model as load_stt
    from .audio import resample
    cues = list(cues)
    try:
        stt = slot.get("stt", lambda: load_stt(_local(STT_MODEL)))
        heard = {}
        for k, (start, wav) in anchored:
            if start not in heard:
                r = stt.generate(mx.array(resample(wav, SR, 16000)), word_timestamps=True)
                heard[start] = [(re.sub(r"[^a-z0-9']", "", w["word"].lower().replace("’", "'")), w["start"])
                                for s in r.segments for w in (s.get("words") or [])]
            want = re.sub(r"[^a-z0-9']", "", cues[k][1]["at"].lower().replace("’", "'"))
            hits = [t for w, t in heard[start] if w == want] or [t for w, t in heard[start] if want and w.startswith(want[:4])]
            nth = cues[k][1].get("nth")  # a tag inside the sentence: which time its word comes in the line
            if hits:
                at = hits[min(nth, len(hits) - 1)] if nth is not None else hits[-1]
                cues[k] = (start + int(at * SR), cues[k][1])
            else:
                what = cues[k][1].get("name") or {"amb": "background off", "music": "music off"}.get(cues[k][1]["type"], "sound")
                log(f"Couldn't hear “{cues[k][1]['at']}”, so “{what}” plays after the sentence.")
    except Exception as e:  # noqa: BLE001 - effects still play after their sentence
        log(f"Word timing skipped ({type(e).__name__}); effects play after their sentences.")
    finally:
        slot.release()
    return sorted(cues, key=lambda c: c[0])  # stable: tags at the same moment keep their order


def _local(repo):
    """A Hugging Face model's folder on this Mac when it's downloaded (no asking the internet whether it changed:
    a dropped connection then can't stop the work), else its name (downloaded now)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        # (its config is always downloaded with it; snapshot_download(local_files_only) refuses a model that was
        # downloaded without its README, as mlx-audio's are)
        hit = try_to_load_from_cache(repo, "config.json")
        return str(Path(hit).parent) if isinstance(hit, str) else repo
    except Exception:  # noqa: BLE001 - not downloaded yet
        return repo


@streamed_on_mlx_thread
def transcribe_many(paths, stop=None):
    """(path, what was said) for each audio file, with the speech-to-text model loaded once for them all."""
    from mlx_audio.stt.utils import load_model as load_stt
    stt = slot.get("stt", lambda: load_stt(_local(STT_MODEL)))
    try:
        for p in paths:
            if stop is not None and stop.is_set():
                return
            yield str(p), stt.generate(str(p)).text.strip()
    finally:
        slot.release()


@on_mlx_thread
def transcribe(path):
    from mlx_audio.stt.utils import load_model as load_stt
    stt = slot.get("stt", lambda: load_stt(_local(STT_MODEL)))
    text = stt.generate(path).text.strip()
    slot.release()
    return text


@streamed_on_mlx_thread
def design_takes(description, text, language, count=3, seed=0, stop=None):
    yield from _qwen.design_takes(description, text, language, count=count, seed=seed, stop=stop)
    free()


def orpheus_check():
    """A short status line about how Orpheus is set up."""
    import httpx
    e = tts.ENGINES["orpheus"]
    item = {"text": "Hello! This is a quick test.", "voice": "orpheus:tara"}
    if _orpheus.on_mac():
        started = time.time()
        audio = on_mlx_thread(lambda: next(iter(e.speak([item], {}))))()[1]
        took = time.time() - started
        return (f"Orpheus runs on this Mac. Test line: {len(audio) / SR:.1f}s of audio in {took:.1f}s "
                f"(includes loading the model the first time).")
    base = _orpheus.server_base()
    try:
        r = httpx.get(f"{base}/health", timeout=15)
        health = f"health check {r.status_code}"
    except httpx.HTTPError as ex:
        raise ValueError(f"Couldn't reach {base} ({type(ex).__name__}). Check the address and that the server is running.")
    started = time.time()
    audio = e._server_speak(item)
    took = time.time() - started
    return f"Connected to {base} ({health}). Test line: {len(audio) / SR:.1f}s of audio in {took:.1f}s ({len(audio) / SR / took:.2f}× real time)."
