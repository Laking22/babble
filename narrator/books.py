"""Text to speech and audiobooks, without any UI: casting, building chapters, creating and previewing books,
speaker detection, importing, sound tags (missing sounds, AI tags, chapter intros and outros) and the Kaggle bundle.

Both front ends use this: the Gradio page (narrator/ui/app.py) and the local API for the desktop app
(narrator/api.py). Long operations are generators of events, plain dicts:
  {"status": "Line 3 of 20 …"}                        progress (the text says how far, see status.fraction)
  {"status": "…", "audio": "/path/out.wav"}           a finished file
  {"status": "…", "story": tagged, "original": old}   the story text changed (sound tags added); old is for Undo
  {"status": "…", "story": tagged, "cast": rows}      speaker detection's result
Problems the user can fix raise ValueError with a message meant for them.
"""
import hashlib
import json
import math
import re
import shutil
import time
from pathlib import Path

from . import ai_detect, book, export, pronounce
from . import engine as en
from . import tts as engines
from . import voices as vl
from .cast import book_brackets, parse, strip_tags, unbracket
from .status import progress_of
from .detect import detect as quick_detect

BOOKS = en.OUTPUTS / "books"
LANGUAGES = ["english", "auto", "chinese", "german", "italian", "portuguese", "spanish", "japanese", "korean",
             "french", "russian"]
DETECT_METHODS = ["AI (most accurate)", "Quick (rules)"]  # the page's labels; the API also takes "ai" / "quick"
KAGGLE_DIR = Path(__file__).resolve().parent / "kaggle_render"


# ---------- small helpers ----------

def temp_from_stability(stability):
    """ElevenLabs-style stability: 0% is most expressive (temperature 1.3), 100% most stable (0.5)."""
    return round(1.3 - 0.8 * float(stability) / 100, 3)


def num(value, default):
    try:
        v = float(value)
        return default if math.isnan(v) else v
    except (TypeError, ValueError):
        return default


def rows_of(table):
    """A cast table (rows, a DataFrame, or dicts from the API) -> [[character, voice name, style, speed]]."""
    rows = table.values.tolist() if hasattr(table, "values") else (table or [])
    out = []
    for r in rows:
        if isinstance(r, dict):  # {"character", "voice", "style", "speed"}
            r = [r.get("character") or r.get("name"), r.get("voice"), r.get("style"), r.get("speed", 1.0)]
        r = list(r) + ["", "", "", 1.0]
        if str(r[0] or "").strip():
            out.append([str(r[0]).strip(), str(r[1] or "").strip(), str(r[2] or "").strip(), num(r[3], 1.0)])
    return out


def instruct(text):
    text = (text or "").strip()
    if not text:
        return None
    return text if text.endswith((".", "!", "?")) else text[0].upper() + text[1:] + "."


def default_voice():
    """Your first saved voice, else Qwen's Ryan, else the first voice of any available engine."""
    mine = vl.library()
    if mine:
        return f"my:{mine[0]['name']}"
    values = [v.value for v in engines.all_voices()]
    return "builtin:ryan" if "builtin:ryan" in values else (values[0] if values else "builtin:ryan")


def sf_duration(path):
    import soundfile as sf
    return sf.info(str(path)).duration


def slug_of(title):
    # letters of any script stay (a Japanese or Russian title has a name too); the rest becomes "-"
    return re.sub(r"[^\w]+|_+", "-", title or "story").strip("-")[:60] or "story"


# headings that don't tell one book from another
GENERIC_HEADING = re.compile(r"^(chapter|part|book|section|prologue|epilogue|introduction|preface|contents|foreword)?"
                             r"[\s.:#-]*([0-9]+|[ivxlc]+|one|two|three|first)?[\s.:-]*$"
                             r"|^第[0-9一二三四五六七八九十百]+[章部节節回]$", re.I)  # 第一章: Chapter 1


def book_name(title, story):
    """The folder and file name of a book: its title, or for an untitled story its first heading that names
    something, else its opening words. Two untitled stories no longer overwrite each other, and the same
    story made again still lands in its folder and reuses the chapters it made."""
    if (title or "").strip():
        return slug_of(title)
    for heading, body in book.split_chapters(story or ""):
        if heading and heading not in book.UNSPOKEN_TITLES and not GENERIC_HEADING.match(heading.strip()):
            return slug_of(heading)
        words = [w for w in (re.sub(r"[^\w'-]", "", w) for w in
                             re.sub(r"\[[^\]]*\]|\{[^{}]*\}", " ", body).split()) if w.strip("'-")]
        if words:
            return slug_of(" ".join(words[:6]))
    return slug_of("")


def _render(items, **kw):
    """en.render as events."""
    for path, status in en.render(items, **kw):
        yield {"status": status, "audio": path} if path else {"status": status}


# ---------- text to speech ----------

def speak(text, voice, style="", speed=1.0, stability=55, language="english", seed=0, exact=False, max_chars=250,
          pause=0.5, batch=2, preview=False, parallel=None, pronunciations=None):
    """Read a text in one voice. Paragraphs are split into lines the voice handles well."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    if not paragraphs:
        raise ValueError("Type or paste some text first.")
    if exact and not voice.startswith("my:"):
        raise ValueError("Exact cloning only works with your own voices, not built-in ones.")
    items = []
    for paragraph in paragraphs:
        chunks = en.split_chunks(paragraph, en.chunk_limit(voice, max_chars))
        items += [dict(text=c, voice=voice, instruct=instruct(style), speed=speed,
                       pause=_gap(c, 0.25, para_pause=pause) if i < len(chunks) - 1 else pause)
                  for i, c in enumerate(chunks)]
    if preview:
        items = items[:1]
    items = pronounce.apply_items(items, pronounce.everyone())  # the words you said to say another way
    yield from _render(items, kind="preview" if preview else "speech", language=language,
                       temperature=temp_from_stability(stability), seed=seed, batch=batch,
                       exact=voice[3:] if exact else None, parallel=parallel)


# ---------- casting ----------

def model_choices():
    """[(label, value)] for "Voices for the whole book": as cast, or every voice from one engine."""
    return [("As cast (each voice's own model)", "cast")] + [(f"{e.short} only", e.id) for e in engines.available()]


def auto_cast(characters, table):
    """Give new characters a voice of their gender (your saved voices first), different from the others."""
    rows = {r[0]: r for r in rows_of(table)}
    narrator = rows.get("Narrator") or ["Narrator", vl.short_name(default_voice()), "Warm, calm storytelling.", 1.0]
    used = {vl.from_name(narrator[1])}
    pools = {}
    for gender in ("male", "female"):
        mine = [f"my:{v['name']}" for v in vl.library() if v["gender"] == gender]
        built = [f"builtin:{k}" for k, (_, g, _) in vl.BUILTINS.items() if g == gender]
        pools[gender] = mine + built or vl.by_engine(engines.available()[0].id, gender)
    out, unknown = [narrator], 0
    for name, gender in characters.items():
        if name in rows:
            out.append(rows[name])
            used.add(vl.from_name(rows[name][1]))
            continue
        if gender not in pools:
            gender = ("female", "male")[unknown % 2]
            unknown += 1
        pick = next((v for v in pools[gender] if v not in used), pools[gender][0])
        used.add(pick)
        out.append([name, vl.short_name(pick), "", 1.0])
    return out


def switch_models(cast, models):
    """Give every character a voice from one model. Returns (new cast, readable list of swaps).

    Characters already on that model keep their voice; the rest get one of the same gender,
    different from the others where possible. Same cast in, same result out, so finished chapters are reused."""
    if models not in engines.ENGINES:
        return cast, []
    target = engines.ENGINES[models]
    on_model = target.owns
    # saved voices are Qwen too, but swaps go to the engine's own stock voices so results stay stable
    pool = {g: [v.value for v in target.voices() if v.gender == g and "saved" not in v.tags] or
            [v.value for v in target.voices() if "saved" not in v.tags] for g in ("female", "male")}
    used = {value for value, _, _ in cast.values() if on_model(value)}
    out, swaps, unknown = {}, [], 0
    for name, (value, style, spd) in cast.items():
        if on_model(value):
            out[name] = (value, style, spd)
            continue
        gender = vl.gender_of(value)
        if gender not in pool:
            gender = ("female", "male")[unknown % 2]
            unknown += 1
        pick = next((v for v in pool[gender] if v not in used), pool[gender][len(swaps) % len(pool[gender])])
        used.add(pick)
        out[name] = (pick, style, spd)
        swaps.append(f"{name}: {vl.short_name(value)} → {vl.short_name(pick)}")
    return out, swaps


def set_character(table, name, voice, style, speed):
    """Add or change one character. Returns (rows, message)."""
    name = (name or "").strip()
    if not name:
        raise ValueError("Pick or type a character first.")
    if not vl.from_name(voice or ""):
        raise ValueError("Pick a voice from the list.")
    rows = rows_of(table)
    for r in rows:
        if r[0] == name:
            r[1:] = [vl.short_name(voice), (style or "").strip(), float(speed)]
            break
    else:
        rows.append([name, vl.short_name(voice), (style or "").strip(), float(speed)])
    return rows, f"{name} will be voiced by {vl.short_name(voice)}."


def remove_character(table, name):
    if name == "Narrator":
        raise ValueError("The Narrator can't be removed, but you can change its voice.")
    return [r for r in rows_of(table) if r[0] != name], f"Removed {name} from the cast."


def clean_sample_text(text, max_chars=3000):
    """Strip markdown headings, sound cues, and [Speaker] tags so a single voice reads clean prose."""
    raw = (text or "").strip()
    if not raw:
        return ""
    cleaned = re.sub(r"\{[^{}]*\}", "", raw)
    cleaned = unbracket(strip_tags(cleaned))
    lines = [re.sub(r"[ \t]{2,}", " ", re.sub(r"^[ \t]*##+[ \t]+", "", ln)).strip() for ln in cleaned.splitlines()]
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return cleaned[:max_chars].strip()


def hear_character(story, name, voice, style, speed, stability, language, table=None, models="cast",
                   sample_text=None, pronunciations=None):
    """Read the highlighted paragraph (when sample_text is given) or one of the character's own lines from the
    story (or a stock line) in the voice picked for them."""
    value = vl.from_name(voice) or default_voice()
    if models in engines.ENGINES:
        cast = {n: (vl.from_name(v) or default_voice(), st, sp) for n, v, st, sp in rows_of(table)}
        cast[name] = (value, style, speed)
        value = switch_models(cast, models)[0][name][0]
    sample = clean_sample_text(sample_text, max_chars=3000)
    if sample:
        paras = [p.strip() for p in re.split(r"\n\s*\n", sample) if p.strip()]
        items = []
        for p in paras:
            chunks = en.split_chunks(p, en.chunk_limit(value, 250))
            items += [dict(text=c, voice=value, instruct=instruct(style), speed=float(speed),
                           pause=_gap(c, 0.2, para_pause=0.45) if i < len(chunks) - 1 else 0.45, who=name)
                      for i, c in enumerate(chunks)]
    else:
        line = None
        body = "\n\n".join(b for _, b in book.split_chapters(story or ""))
        for paragraph in parse(body, narrator="Narrator", split_quotes=True):
            line = next((t for w, _, t in paragraph if w == name), None)
            if line:
                break
        text = en.split_chunks(line, 200)[0] if line else "Hello. This is how I sound when I tell a story."
        items = [dict(text=text, voice=value, instruct=instruct(style), speed=float(speed), pause=0.1, who=name)]
    items = pronounce.apply_items(items, pronounce.merged(pronunciations))
    yield from _render(items, kind="voice-check", language=language, temperature=temp_from_stability(stability),
                       batch=2 if len(items) > 1 else 1)


# ---------- story text ----------

def story_stats(text):
    titles = [t for t, _ in book.split_chapters(text) if t]
    words = len(unbracket(re.sub(r"\{[^{}]*\}", " ", strip_tags(text))).split())  # tags aren't read aloud
    chapters = f"{len(titles)} chapter{'' if len(titles) == 1 else 's'} · " if titles else ""
    return f"{chapters}{words:,} words · about {en.clock(words / 2.6)} of audio"


def import_file(path):
    """An ebook or text file -> {"text", "title", "author", "cover", "chapters", "message"}."""
    b = book.import_book(path, BOOKS / "_imports" / re.sub(r"[^A-Za-z0-9]+", "-", Path(path).stem)[:60])
    b["text"] = book_brackets(b["text"])  # the book's own [brackets] are text, not cast tags
    b["chapters"] = [book_brackets(t) for t in b["chapters"]]
    by = f" by {b['author']}" if b["author"] else ""
    b["message"] = f"Imported “{b['title']}”{by}: {len(b['chapters'])} chapters. Next: Detect speakers."
    return b


def detect_speakers(story, table, method="ai", make_on="mac", cloud_gpu=""):
    """Tag who speaks each line ([Name] tags) and cast the new characters. Ends with
    {"story": tagged, "cast": rows, "original": story, "status": what was found}. The AI runs on a cloud GPU when
    make_on names one (cloud_ai), else on this Mac."""
    raw = before = story or ""
    if not raw.strip():
        raise ValueError("Paste or import a story first.")
    if strip_tags(raw) != raw:  # brackets that don't name anyone in the cast are the book's own (imported before)
        raw = book_brackets(raw, keep=[r[0] for r in rows_of(table) if r and r[0] != "Narrator"])
    if strip_tags(raw) != raw:
        raise ValueError("This story already has [tags]. Use Undo, or remove the tags to detect again.")
    if method in ("quick", DETECT_METHODS[1]):
        parts, characters = [], {}
        for title, body in book.split_chapters(raw):
            tagged, found = quick_detect(body) if body else ("", {})
            for name, gender in found.items():
                characters[name] = characters.get(name) or gender
            parts.append((f"## {title}\n\n" if title else "") + tagged)
        tagged = "\n\n".join(parts)
    elif make_on and make_on != "mac":
        from . import cloud_ai, detect_core  # (ai_detect: imported at the top; importing it here made it local, and this Mac path failed)
        from collections import Counter
        chapters, texts = ai_detect.chapters_of(raw)
        starts = ai_detect.starts_of(chapters)
        _, tokens = detect_core.prompt_tokens(texts, starts=starts)
        r = yield from cloud_ai.run("detect", {"texts": texts, "starts": starts}, make_on, cloud_gpu, "Detecting speakers", tokens)
        labels = {int(k): tuple(v) for k, v in (r.get("labels") or {}).items()}
        genders = {k: Counter(v) for k, v in (r.get("genders") or {}).items()}
        tagged, characters = ai_detect.assemble(chapters, labels, genders, r.get("known") or [])
        if (r.get("_progress") or {}).get("stopped"):
            p = r["_progress"]
            yield {"status": f"The cloud's time ran out after {p.get('done')} of {p.get('total')} dialogue lines; "
                             "the rest are left to the narrator."}
    else:
        yield {"status": "Loading the AI model…"}
        tagged = characters = None
        try:
            for event in ai_detect.detect(raw):
                if event[0] == "progress":
                    done, total = event[1], event[2]
                    yield {"status": f"AI is reading the story… {f'{done} of {total} dialogue lines' if total else 'starting'}"}
                else:
                    tagged, characters = event[1], event[2]
        except Exception as e:  # noqa: BLE001 - shown to the user
            raise ValueError(f"AI detection failed: {e}") from e
        if tagged is None:
            return
    found = ", ".join(f"{n} ({g or '?'})" for n, g in characters.items()) or "no dialogue found"
    yield {"story": tagged, "cast": auto_cast(characters, table), "original": before,
           "status": f"Found: {found}. Check the tags and voices, then create the audiobook."}


# ---------- building a book ----------

MID_SENTENCE = re.compile(r"[,;:—–-][\"”’']?\s*$|[^.!?…\"”’'\s]\s*$")
BEAT_END = re.compile(r"(?:\.{3}|…)[\"”’']?\s*$")


def _gap(text, line_pause, speaker_change=False, para_pause=None):
    """Pause after a cast line inside a paragraph:
    - Mid-sentence split or quote attribution (comma/semicolon/dash): tight 0.06 s breath (trim_edges adds 120 ms).
    - Dramatic trailing beat ('...' or '…'): longer beat (~1.6x line_pause, capped at para_pause).
    - Speaker switch after a finished sentence: slightly wider turn-taking beat (~1.25x line_pause).
    - Normal finished sentence: line_pause."""
    clean = en.TAG_SPANS.sub("", text).rstrip()  # {sounds}, [directions] and <laugh>-style tags aren't words
    clean = re.sub(r"[*_)]+$", "", clean).rstrip()
    if MID_SENTENCE.search(clean):
        return min(line_pause, 0.06)
    cap = para_pause if para_pause is not None else line_pause * 2.0
    if BEAT_END.search(clean):
        return round(min(cap, line_pause * 1.6), 3)
    if speaker_change:
        return round(min(cap, line_pause * 1.25), 3)
    return line_pause


def _items(paragraphs, cast, speed, line_pause, para_pause, temps=None):
    from . import sfx
    items = []
    for pi, paragraph in enumerate(paragraphs):
        pace = 1.0
        for _, _, raw_t in paragraph:
            m = sfx.PACE_TAG.search(raw_t or "")
            if m:
                pace = sfx.pace_factor(m.group(1))
                break
        for si, (who, emotion, text) in enumerate(paragraph):
            text = sfx.PACE_TAG.sub("", text or "").strip()
            if not text:
                continue
            value, style, spd = cast[who]
            chunks = en.split_chunks(text, en.chunk_limit(value, 250))
            next_who = paragraph[si + 1][0] if si + 1 < len(paragraph) else None
            for ci, c in enumerate(chunks):
                last_in_seg = ci == len(chunks) - 1
                last = si == len(paragraph) - 1 and last_in_seg
                speaker_change = last_in_seg and next_who is not None and next_who != who
                pause = para_pause if last else _gap(c, line_pause, speaker_change=speaker_change, para_pause=para_pause)
                item = dict(text=c, voice=value, instruct=instruct(emotion or style), who=who,
                            speed=round(spd * speed * pace, 3), pause=pause, para=pi)
                if temps and who in temps:  # this character's own stability (only set when it differs from the book's)
                    item["temperature"] = temps[who]
                items.append(item)
    return items


def voice_temps(voice_stability, stability):
    """{character: temperature} for characters whose stability differs from the book's. Characters left at the
    book's stability get no per-line value, so chapters made before keep matching and are reused."""
    book_t = temp_from_stability(stability)
    out = {}
    for name, value in (voice_stability or {}).items():
        if value is None:
            continue
        t = temp_from_stability(num(value, stability))
        if t != book_t:
            out[name] = t
    return out


def story_chapters(story, quotes_only=True, spans=False):
    """[(title, body, parsed paragraphs)] for the chapters that get read, the same ones build_book makes.
    spans: each with where it is in the story too, (start, end, at) as book.chapter_spans gives it (or None)."""
    out = []
    cut = book.in_parts(book.split_chapters(story))
    where = (book.chapter_spans(story) if spans else None) or [None] * len(cut)
    for (t, body), span in zip(cut, where):
        paras = parse(body, narrator="Narrator", split_quotes=quotes_only)
        if not paras and not (t and t not in book.UNSPOKEN_TITLES):
            continue
        out.append((t, body, paras, span) if spans else (t, body, paras))
    return out


def _shown_title(t, i):
    return t or ("Story" if i == 0 else f"Part {i + 1}")


def without_chapter(story, index, title, quotes_only=True):
    """The story without one of the chapters chapters_status lists: a whole chapter from its heading to the next
    one, a part of a long chapter only its own paragraphs (its chapter's heading stays)."""
    chapters = story_chapters(story, quotes_only, spans=True)
    if not 0 <= index < len(chapters) or _shown_title(chapters[index][0], index) != title or chapters[index][3] is None:
        raise ValueError("The chapters have changed since: try removing it again.")
    start, end, _ = chapters[index][3]
    before, after = story[:start].rstrip(), story[end:].lstrip()
    return before + ("\n\n" if before and after else "") + after


def text_key(title, body):
    """A chapter's words (what Chapters shows as made or edited since), without its voices or settings."""
    return hashlib.sha1(f"{title or ''}\n{(body or '').strip()}".encode()).hexdigest()[:10]


def build_book(story, table, quotes_only, speed, line_pause, para_pause, models="cast", temps=None):
    """Check the cast and turn the story into [(chapter title, items)] plus the parsed chapters."""
    chapters = [(t, parse(body, narrator="Narrator", split_quotes=quotes_only)) for t, body in book.in_parts(book.split_chapters(story))]
    chapters = [(t, paras) for t, paras in chapters if paras or (t and t != "Title page")]
    if not chapters:
        raise ValueError("Paste or import a story first.")
    cast, problems = {}, []
    for name, voice_text, style, spd in rows_of(table):
        value = vl.from_name(voice_text)
        if value is None:
            problems.append(f"{name} → “{voice_text}”")
        cast[name] = (value, style, spd)
    missing = sorted({w for _, paras in chapters for p in paras for w, _, _ in p if w not in cast})
    if missing:
        raise ValueError(f"Add these to the cast: {', '.join(missing)}.")
    if problems:
        raise ValueError("Unknown voice names: " + "; ".join(problems) + ". Use names from the voice list.")
    cast, _ = switch_models(cast, models)
    narrator_voice, narrator_style, narrator_speed = cast["Narrator"]
    built = []
    for chapter_title, paras in chapters:
        items = []
        if chapter_title and chapter_title not in book.UNSPOKEN_TITLES and not book.is_part(chapter_title):  # names the importer made up aren't read out
            items.append(dict(text=chapter_title, voice=narrator_voice, instruct=instruct(narrator_style), who="Narrator",
                              speed=narrator_speed * speed, pause=1.2,
                              **({"temperature": temps["Narrator"]} if temps and "Narrator" in temps else {})))
        items += _items(paras, cast, speed, line_pause, para_pause, temps)
        if items:
            built.append((chapter_title, items))
    return built, chapters, cast


# ---------- sounds ----------

def sfx_tags(text):
    from . import sfx
    return bool(sfx.CUE.search(text or ""))


def intro_choices(book=None):
    """Library sounds a chapter can open or close with (the Intros tab), music first: [{"id", "label", "path",
    "kind", "duration"}]. book: a project id, to include that book's own sounds."""
    from . import sfx
    order = {"music": 0, "ambience": 1, "effect": 2}
    lib = sorted(sfx.pool(book), key=lambda e: (order.get(e["kind"], 3), e["name"].lower()))
    return [{"id": e["id"], "label": f"{e['name']} ({e['kind']}, {e['duration']:.0f} s)", "path": str(sfx.LIB / e["file"]),
             "kind": e["kind"], "duration": e["duration"]} for e in lib]


def apply_suggestions(story_now, planned_on, chosen):
    """Add the director's kept sounds to the story they were planned on. Returns (tagged story, message)."""
    from . import ai_sfx
    if story_now != planned_on:
        raise ValueError("The story changed after the director planned it. Direct it again.")
    if not chosen:
        return planned_on, "Nothing selected, so nothing was added."
    return ai_sfx.insert(planned_on, chosen), f"Added {len(chosen)} sound(s). Undo puts the text back."


def keep_in(entry, book=None, scope="all"):
    """A sound made while working on a book: kept in that book only (scope "book") or shared with every book."""
    from . import sfx
    if not book:
        return entry
    return sfx.set_scope(entry["id"], book if scope == "book" else None, made_in=book)


def use_sound_libraries(book=None):
    """The book's "Query sound libraries" (Director tab): look for recorded sounds online before making them.
    On unless the book turned it off."""
    if not book:
        return True
    try:
        from . import projects
        return projects.get(book).get("sound_libraries") is not False
    except (OSError, ValueError):
        return True


def make_missing_sounds(text, book=None, scope="all", online_only=False):
    """Make every tagged sound the library doesn't have yet (effects, backgrounds and music, each with its own
    model). Yields status lines; returns ([names made], [(name, error)]). online_only: only the recorded sounds
    from the sound libraries; the ones that need AI are left (they're made in the cloud with the book's lines)."""
    from . import online_sounds, sfx, sfxgen, tasks
    todo = sfx.missing(text)
    made, failed = [], []
    type_of = {v: k for k, v in sfx.KIND_OF.items()}
    # a recorded sound from Freesound / Lots of Sounds first, when keys are set and the book's Director tab allows it
    online = bool(online_sounds.ready()) and use_sound_libraries(book)
    for n, (kind, name) in enumerate(todo):
        if sfx.find_for({"type": type_of.get(kind, "sfx"), "name": name}):
            continue  # made meanwhile (by another task making the same story's sounds)
        if online:
            yield f"Finding sound {n + 1}/{len(todo)} · {name}"
            try:
                hit = online_sounds.find(name, kind, log=tasks.note)
            except tasks.Stopped:
                raise
            except Exception as ex:  # noqa: BLE001 - the sound is made instead
                print(f"[online] “{name}”: {ex}", flush=True)
                hit = None
            if hit:
                if hit[1]:  # (a sound already in the library keeps where it is shared)
                    keep_in(hit[0], book, scope)
                made.append(name)
                continue
        if online_only:
            continue
        yield f"Generating sound {n + 1}/{len(todo)} · {name}"
        try:
            keep_in(sfxgen.make(name, kind), book, scope)
            made.append(name)
        except Exception as e:  # noqa: BLE001 - reported to the user
            failed.append((name, str(e)))
    return made, failed


def _wait_for_sounds(note="", book=None):
    """Create, while Make missing sounds runs (the director's sounds, after Add): wait for it to finish. Otherwise the
    voice model takes the MLX thread first and the sounds wait for the book, which is then made without them."""
    import time
    from . import tasks
    me, said = tasks.CURRENT.get(), None
    while True:
        busy = [t for t in tasks.all_tasks() if t.kind == "sounds" and t.state == "running" and t is not me
                and not (book and getattr(t, "book", None) not in (None, book))]  # another book's: not waited for
        if not busy:
            return
        if tasks.stop_wanted():
            raise tasks.Stopped()
        line = busy[0].status.split("\n")[0]
        if line != said:
            yield {"status": note + line + "\nThe chapters start once the sounds are made."}
            said = line
        time.sleep(0.5)


def make_missing(text, book=None, scope="all", chapter_plan=None, online_only=False):
    """The “Make missing sounds” button: events, ending with what was made. chapter_plan: the director's chapter
    music (chapter_music.plan), made too when its "Make chapter intros and outros" is on; the events then end with
    {"chapter_music": …} for the book."""
    from . import chapter_music, sfx, tasks
    if tasks.CURRENT.get() is not None:
        tasks.CURRENT.get().book = book  # (a Create of this book waits for it; another book's doesn't)
    token = sfx.BOOK.set(book) if book else None
    music = None
    try:
        made, failed = (yield from _relay(make_missing_sounds(text, book, scope, online_only))) if sfx.missing(text) else ([], [])
        if chapter_plan and chapter_plan.get("chapters"):
            music, bad = yield from _relay(chapter_music.make(chapter_plan, book, scope))
            failed += bad
    finally:
        _reset(sfx.BOOK, token)
    if failed and not made and not music:
        raise ValueError(f"Couldn't make “{failed[0][0]}”: {failed[0][1]}")
    left = len(sfx.missing(text)) if online_only else 0
    if not made and not music and not failed:
        yield {"status": f"{left} sound(s) are made in the cloud with the book's lines." if left
               else "Every tagged sound is already in your library."}
        return
    msg = (f"Made {len(made)} sound(s) and added them to your library: " + ", ".join(f"“{m}”" for m in made) + ". "
           if made else "")
    if music:
        n = sum(1 for r in music["chapters"] if r["intro"])
        msg += f"Every chapter's intro and outro is set ({n} with an intro): see Intros."
    if left:
        msg += f" The other {left} are made in the cloud with the book's lines."
    yield dict({"chapter_music": music} if music else {}, status=msg.strip() + _made_note([], failed))


def _reset(var, token):
    if token is not None:
        try:
            var.reset(token)
        except ValueError:  # set in another context (a generator closed from elsewhere): nothing to undo here
            pass


def _relay(steps, note=""):
    """Pass a status generator's lines on as events, and hand back what it returns."""
    while True:
        try:
            status = next(steps)
        except StopIteration as done:
            return done.value
        yield {"status": status + (f"\n{note.strip()}" if note.strip() else "")}  # what it's doing comes first


def _made_note(made, failed):
    note = (f"\nMade {len(made)} missing sound(s) first: " + ", ".join(f"“{m}”" for m in made) + ".") if made else ""
    if failed:
        note += f"\nCouldn't make “{failed[0][0]}”: {failed[0][1]}" + (f" (and {len(failed) - 1} more)" if len(failed) > 1 else "")
    return note


def chapter_sounds(items):
    """Which library sound each tag in a chapter plays: [(tag, sound id or None, file time)]."""
    from . import sfx
    out = []
    for it in items:
        cues = [it["cue"]] if "cue" in it else [c for m in sfx.CUE.finditer(it.get("text") or "")
                                                for k, c in sfx.split_cues(m.group()) if k == "cue"]
        for c in cues:
            if c["type"] == "room" or not (c.get("name") or c.get("id")):
                continue
            e = (next((x for x in sfx.index() if x["id"] == c.get("id")), None) if c["type"] in ("intro", "outro")
                 else sfx.find_for(c))
            f = sfx.LIB / e["file"] if e else None
            out.append((c["type"], c.get("name"), e["id"] if e else None,
                        int(f.stat().st_mtime) if f is not None and f.exists() else 0))
    return out


# ---------- creating a book ----------

FOLDER_NAME = re.compile(r"[^\W_][\w-]{0,79}")
MANIFEST = "chapters.json"  # in a book's folder: each chapter's words, file and length, from the last Create


def book_folder(title, story, folder=None):
    """Where a book's chapters go: its own folder (an Audiobook project has one that never changes), else one
    named after the book."""
    if folder:
        if not FOLDER_NAME.fullmatch(folder):
            raise ValueError("That book folder name isn't allowed.")
        return BOOKS / folder
    return BOOKS / book_name(title, story)


def _read_manifest(folder):
    try:
        return json.loads((folder / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def chapters_status(story, folder=None, title="", quotes_only=True):
    """Each chapter of the story as it is now, and whether audio for its current words was made:
    [{"index", "title", "state": "done" | "gaps" | "partial" | "edited" | "not made", "file", "seconds"}].
    "edited": made before, but its words changed since. Voices and settings aren't compared: Create remakes a
    chapter whose voices changed, and reuses the rest."""
    folder = book_folder(title, story, folder)
    made = {int(k): v for k, v in (_read_manifest(folder).get("chapters") or {}).items()}
    out = []
    for i, (t, body, _, span) in enumerate(story_chapters(story, quotes_only, spans=True)):
        m = made.get(i) or {}
        f = Path(m["file"]) if m.get("file") else None
        exists = f is not None and f.exists()
        if exists and m.get("text") == text_key(t, body):
            state = "gaps" if f.stem.endswith("-gaps") else "partial" if f.stem.endswith("-partial") else "done"
        elif exists:
            state = "edited"
        else:
            state = "not made"
        out.append({"index": i, "title": _shown_title(t, i), "state": state,
                    "file": str(f) if exists else None, "seconds": m.get("seconds") if exists else None,
                    "at": span[2] if span else None})  # (where it starts in the story: clicking it goes there)
    return out


CLOUD_LINES = "cloud-lines"  # in a book's folder: lines made in the cloud, by key (see cloud_book)
LINES = "lines"  # ...and lines made on this Mac, by the same kind of key: a chapter whose sounds change is mixed again
LINES_KEEP_DAYS = 14  # kept lines the book's text no longer has (an edited sentence's) go after this long unused


def prune_lines(folder, built, gen):
    """After Create: the lines the book's chapters use now stay (marked used); others unused for LINES_KEEP_DAYS go."""
    import os
    import time
    from . import cloud_book
    d = Path(folder) / LINES
    if not d.is_dir():
        return
    opts = {"language": gen.get("language"), "temperature": gen.get("temperature")}
    used = set()
    for _, items in built:
        for it in en.spoken(items, gen.get("sound", True)):
            if "cue" not in it:
                spec = cloud_book.local_spec(it, opts, gen.get("seed", 0))
                if spec is not None:
                    used.add(cloud_book.key_of(spec) + ".flac")
    old = time.time() - LINES_KEEP_DAYS * 86400
    for f in d.glob("*.flac"):
        try:
            if f.name in used:
                os.utime(f)
            elif f.stat().st_mtime < old:
                f.unlink()
        except OSError:
            pass


def with_intro(built, intro):
    """Each chapter with its own intro and outro (chapter_music, from the Intros tab or the director). intro: the
    book's settings as the API passes them, {"chapters": rows, "level": dB under the narrator or None, "id": the
    one intro books had before chapter_music (every chapter then opens with it, and ends with its ending)}.
    An intro plays its loudest moment before the first words (sfx.plan_intros); a sound several chapters share
    gives each of them a different stretch when it's long enough."""
    from . import chapter_music, sfx
    if not isinstance(intro, dict):
        intro = {"id": intro}
    level, rows = intro.get("level"), (intro.get("chapters") or [])
    names = [t or "Opening" for t, _ in built]
    if rows:
        per = chapter_music.match(names, rows)
    elif intro.get("id"):  # (a book from before: one intro everywhere, its ending after every chapter)
        per = [{"intro": intro["id"], "outro": intro["id"] if intro.get("outros", True) and t != "Title page" else ""}
               for t, _ in built]
    else:
        return built
    lib = {e["id"]: e for e in sfx.index()}
    gone = set()
    uses = {}
    for k, r in enumerate(per):
        if r and r.get("intro"):
            uses.setdefault(r["intro"], []).append(k)
    cuts = {}
    for sid, ks in uses.items():
        e = lib.get(sid)
        if e is None:
            gone.add(sid)
            continue
        for k, p in zip(ks, sfx.plan_intros(e, len(ks))):
            cuts[k] = (e, p)
    last = max((k for k, (t, _) in enumerate(built) if t != "Title page"), default=len(built) - 1)
    out = []
    for k, (t, items) in enumerate(built):
        r = per[k] or {}
        if k in cuts:
            e, p = cuts[k]
            items = [{"cue": {"type": "intro", "id": e["id"], "name": e["name"], "volume": "normal", "gain_db": level, **p},
                      "pause": 0}] + items
        e = lib.get(r.get("outro") or "")
        if r.get("outro") and e is None:
            gone.add(r["outro"])
        if e is not None:
            o = sfx.outro_cut(e, last=k == last)
            items = items + [{"cue": {"type": "outro", "id": e["id"], "name": e["name"], "volume": "normal",
                                      "gain_db": level, **o, "lead": round(sfx.OUTRO_PAUSE + o["len"], 2)}, "pause": 0}]
        out.append((t, items))
    if gone:
        print(f"[intros] {len(gone)} chapter intro/outro sound(s) aren't in the library any more: left out", flush=True)
    return out


def chapter_file(folder, index, items, language, stability, seed, sound, bed):
    """Where a chapter made from these items and settings goes. The same words, voices, settings and sounds give
    the same file, so a chapter made before is reused."""
    # the sounds its tags play are part of a chapter too: after Make missing sounds (or a new or deleted sound)
    # the chapter is made again instead of reusing the one that was made without them
    sounds = chapter_sounds(items) if sound else []  # (chapters without tags keep the key they had)
    # strip internal para index so existing chapter caches stay valid
    hash_items = [{k: v for k, v in it.items() if k != "para"} if isinstance(it, dict) else it for it in items]
    from .sfx import mix_key
    key = hashlib.sha1(json.dumps([hash_items, language, stability, int(seed or 0), bool(sound), mix_key(bed)]
                                  + ([sounds] if sounds else []),
                                  sort_keys=True, default=str).encode()).hexdigest()[:10]
    return folder / f"{index:03d}-{key}.wav"


def chapters_to_make(story, table, models="cast", quotes_only=True, speed=1.0, stability=55, language="english",
                     seed=0, line_pause=0.2, para_pause=0.7, title="", sound=True, bed=1.0, intro="",
                     voice_stability=None, folder=None, book=None, pronunciations=None, all_chapters=False, **_):
    """The chapters Create would make now (not the ones already made), as (index, title, items), with the
    book's folder and the render settings. What Create in the cloud sends. (The AI sound tagging that Create can
    do first isn't run here: lines it changes are made on this Mac.) all_chapters: every chapter, made or not
    (Check lines)."""
    from . import sfx
    token = sfx.BOOK.set(book) if book else None
    try:
        temps = voice_temps(voice_stability, stability)
        built, _, _ = build_book(story, table, quotes_only, speed, line_pause, para_pause, models, temps)
        if intro:
            built = with_intro(built, intro)
        say = pronounce.merged(pronunciations)
        built = [(t, pronounce.apply_items(items, say)) for t, items in built]
        where = book_folder(title, story, folder)
        todo = [(i, t, items) for i, (t, items) in enumerate(built)
                if all_chapters or not chapter_file(where, i, items, language, stability, seed, sound, bed).exists()]
        return where, todo, dict(language=language, temperature=temp_from_stability(stability), seed=int(seed or 0),
                                 sound=bool(sound), chapters=len(built))
    finally:
        _reset(sfx.BOOK, token)


def made(**kw):
    """Whether Create would only put the book together again: {"all": every chapter is made with these words, voices
    and settings, "chapters", "todo": how many still need making}."""
    _, todo, gen = chapters_to_make(**kw)
    return {"all": bool(gen["chapters"]) and not todo, "chapters": gen["chapters"], "todo": len(todo)}


CHAPTER_FILE = re.compile(r"\d{3}-[0-9a-f]{10}\.wav")


def remake(title="", story="", folder=None, seed=0, **_):
    """Remake: the book's current take (its chapters, the lines kept on this Mac and from the cloud, Check lines'
    notes) goes to the Trash, so the next Create speaks every line again. A fixed seed would give the same takes
    again: the seed after it is returned to use instead (0, random, stays). Returns {"moved", "seed"}."""
    from . import cloud_book
    where = book_folder(title, story, folder)
    rec = cloud_book.read(where) if where.exists() else None
    if rec and rec.get("state") in cloud_book.WAITING + ("downloading",):
        raise ValueError("This book is being made in the cloud. Wait for it to come back, or stop it first.")
    old = [f for f in where.iterdir() if CHAPTER_FILE.fullmatch(f.name)] if where.exists() else []
    old += [where / n for n in (LINES, CLOUD_LINES, "line-check.json", "cloud") if (where / n).exists()]
    if old:
        stamp = time.strftime("%Y-%m-%d %H.%M.%S")
        bin_ = Path.home() / ".Trash" / f"{where.name} (earlier take {stamp})"
        try:
            bin_.mkdir(parents=True)
        except OSError:  # no Trash to put it in: kept in the book's folder instead (an earlier one replaced)
            bin_ = where / "earlier take"
            shutil.rmtree(bin_, ignore_errors=True)
            bin_.mkdir()
        for f in old:
            shutil.move(str(f), str(bin_ / f.name))
    return {"moved": len(old), "seed": int(seed) + 1 if int(seed or 0) else 0}


def create(story, table, models="cast", quotes_only=True, speed=1.0, stability=55, language="english", seed=0,
           batch=2, line_pause=0.2, para_pause=0.7, out_format="WAV", title="", author="", cover=None, sound=True,
           bed=1.0, intro="", automake=True, preview=False, voice_stability=None,
           folder=None, name="", book=None, new_sounds="all", parallel=None, pronunciations=None, only=None,
           preview_text=None, preview_chapter=None, progress_start=0.0):
    """Create the audiobook (or, with preview, the selected text, current chapter opening, or first paragraph).
    Chapters already made with the same text, voices, settings and sounds are reused. Events end with
    {"audio": the book, "status": "Done · …"}."""
    from . import sfx
    token = sfx.BOOK.set(book) if book else None
    try:
        yield from _create(story, table, models, quotes_only, speed, stability, language, seed, batch, line_pause,
                           para_pause, out_format, title, author, cover, sound, bed, intro, automake,
                           preview, voice_temps(voice_stability, stability), folder, name, book, new_sounds, parallel,
                           pronounce.merged(pronunciations), only, preview_text, preview_chapter, progress_start)
    finally:
        _reset(sfx.BOOK, token)


def _resolve_cast(table, models="cast"):
    """Build {character: (voice_value, style, speed)} from cast rows without parsing the whole book."""
    cast, problems = {}, []
    for name, voice_text, style, spd in rows_of(table):
        value = vl.from_name(voice_text)
        if value is None:
            problems.append(f"{name} → “{voice_text}”")
        cast[name] = (value, style, spd)
    if problems:
        raise ValueError("Unknown voice names: " + "; ".join(problems) + ". Use names from the voice list.")
    if "Narrator" not in cast:
        cast["Narrator"] = (default_voice(), "", 1.0)
    return switch_models(cast, models)


def _create(story, table, models, quotes_only, speed, stability, language, seed, batch, line_pause, para_pause,
            out_format, title, author, cover, sound, bed, intro, automake, preview, temps, folder_name,
            name, book_id, new_sounds, parallel=None, say=None, only=None, preview_text=None, preview_chapter=None,
            progress_start=0.0):
    gen = dict(language=language, temperature=temp_from_stability(stability), seed=seed, batch=batch,
               sound=bool(sound), bed=bed, parallel=parallel)

    if preview:
        cast, swaps = _resolve_cast(table, models)
        note = ((f"{engines.ENGINES[models].short} only" if models in engines.ENGINES else "") + ": " + " · ".join(swaps) + "\n") if swaps else ""
        if sound:
            yield from _wait_for_sounds(note, book_id)
        sel_text = (preview_text or "").strip()
        if sel_text:
            sel_lines = [re.sub(r"^[ \t]*##+[ \t]+", "", ln) for ln in sel_text[:2500].splitlines()]
            paras = parse("\n".join(sel_lines), narrator="Narrator", split_quotes=quotes_only)
            paras = [[(w if w in cast else "Narrator", em, tx) for w, em, tx in p] for p in paras]
            pick = paras[:6]
        else:
            chapters = [(t, parse(body, narrator="Narrator", split_quotes=quotes_only)) for t, body in book.in_parts(book.split_chapters(story))]
            chapters = [(t, paras) for t, paras in chapters if paras or (t and t != "Title page")]
            if not chapters:
                raise ValueError("Paste or import a story first.")
            if preview_chapter is not None and 0 <= int(preview_chapter) < len(chapters) and chapters[int(preview_chapter)][1]:
                raw_pick = chapters[int(preview_chapter)][1][:1]
            else:
                _, paras = next(((t, p) for t, p in chapters if p and t != "Title page"), chapters[0])
                raw_pick = paras[:1]
            pick = [[(w if w in cast else "Narrator", em, tx) for w, em, tx in p] for p in raw_pick]
        items = pronounce.apply_items(_items(pick, cast, speed, line_pause, para_pause, temps), say)
        if not items:
            raise ValueError("There's no spoken text in that selection to preview.")
        if sound and automake:  # only the sounds this preview paragraph uses
            yield from _relay(make_missing_sounds(" ".join(it["text"] for it in items), book_id, new_sounds), note)
        for ev in _render(items, kind="story-preview", **gen):
            yield dict(ev, status=note + ev["status"])
        return

    built, chapters, cast = build_book(story, table, quotes_only, speed, line_pause, para_pause, models, temps)  # checks the cast
    swaps = switch_models({n: (vl.from_name(v), st, sp) for n, v, st, sp in rows_of(table)}, models)[1]
    note = ((f"{engines.ENGINES[models].short} only" if models in engines.ENGINES else "") + ": " + " · ".join(swaps) + "\n") if swaps else ""
    if sound:  # sounds still being made (the director's, after Add): the chapters wait for them, so they play
        yield from _wait_for_sounds(note, book_id)
    if intro and not preview:  # the chapter intro: a library sound at the start of every chapter
        built = with_intro(built, intro)
    # pronunciations go into the lines before each chapter's key is worked out, so a chapter whose words are
    # said differently now is made again (and only those)
    built = [(t, pronounce.apply_items(items, say)) for t, items in built]

    auto_made, auto_failed = [], []
    if sound and automake and only is None:  # first, so the chapters are made with them (and their keys include them)
        auto_made, auto_failed = yield from _relay(make_missing_sounds(story, book_id, new_sounds), note)

    slug = slug_of(name) if (name or "").strip() else book_name(title, story)
    folder = book_folder(title, story, folder_name)
    folder.mkdir(parents=True, exist_ok=True)
    # each chapter's words, for Chapters (made / edited since); the same chapters build_book made, in order
    words = [text_key(t, body) for t, body, _ in story_chapters(story, quotes_only)]
    words = words if len(words) == len(built) else [None] * len(built)
    manifest = _read_manifest(folder)
    done_before = {int(k): v for k, v in (manifest.get("chapters") or {}).items()}
    live = [{"title": t or "Story", "state": "waiting", "file": None} for t, _ in built]

    def save_manifest():
        import os as _os
        mpath = folder / MANIFEST
        mtmp = mpath.with_name(f".{mpath.name}.tmp")
        try:
            with open(mtmp, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"chapters": {str(k): v for k, v in done_before.items()}},
                                    indent=1, ensure_ascii=False))
                fh.flush()
                _os.fsync(fh.fileno())
            _os.replace(mtmp, mpath)
        finally:
            mtmp.unlink(missing_ok=True)

    files, titles = [], []
    todo = [i for i, (_, items) in enumerate(built)
            if (only is None or i in only) and not chapter_file(folder, i, items, language, stability, seed, sound, bed).exists()]
    # Full book build reserves [progress_start..0.85] for chapters and [chapter_share..1.0] for final export
    p_start = max(0.0, min(0.85, float(progress_start or 0.0))) if only is None else 0.0
    chapter_share = 1.0 if only is not None else (0.85 if todo else p_start)
    span_p = max(0.0, chapter_share - p_start)
    for index, (chapter_title, items) in enumerate(built):
        if only is not None and index not in only:
            continue
        # the sounds its tags play are part of a chapter too: after Make missing sounds (or a new or deleted sound)
        # the chapter is made again instead of reusing the one that was made without them
        path = chapter_file(folder, index, items, language, stability, seed, sound, bed)
        label = chapter_title or "Story"
        header = f"Chapter {index + 1}/{len(built)}" if len(built) > 1 else label
        made = path
        # the chapters being made share [p_start, chapter_share]
        k, n = sum(1 for i in todo if i < index), max(len(todo), 1)
        where_ = f"Chapter {index + 1}/{len(built)} · " if len(built) > 1 else ""
        if path.exists():
            if todo:
                yield {"status": f"{header} · Reusing saved chapter" + (f"\n{note.strip()}" if note else ""),
                       "progress": p_start + (k / n) * span_p}
        else:
            live[index]["state"] = "making"
            yield {"status": f"{where_}Preparing voices\n{label}" + (f"\n{note.strip()}" if note else ""),
                   "progress": p_start + (k / n) * span_p, "chapters": [dict(c) for c in live], "chapter": index}
            cloud = folder / CLOUD_LINES  # lines made in the cloud (Create in the cloud) are used, not made again
            last_inner = 0.0
            for out, status in en.render(items, kind="chapter", path=path, premade=cloud if cloud.is_dir() else None,
                                         keep=folder / LINES, **gen):
                made = Path(out) if out else made  # "…-gaps.wav" when lines were left out, so it's made again next time
                first, _, rest = str(status).partition("\n")
                inner = progress_of(status)
                if inner is not None:
                    last_inner = max(last_inner, min(max(inner, 0.0), 1.0))
                p = p_start + ((k + last_inner) / n) * span_p
                yield {"status": (f"{where_}{first}" if not out else f"{header} · {first}") + (f"\n{rest}" if rest else "")
                       + (f"\n{note.strip()}" if note else ""), "progress": p}
            for old in folder.glob(f"{index:03d}-*.wav"):
                if old != made:
                    old.unlink(missing_ok=True)
        files.append(made)
        titles.append(label)
        seconds = round(sf_duration(made), 3) if made.exists() else None
        done_before[index] = {"title": label, "text": words[index], "file": str(made), "seconds": seconds}
        save_manifest()
        live[index].update(state="gaps" if made.stem.endswith("-gaps") else "done", file=str(made), seconds=seconds)
        yield {"chapters": [dict(c) for c in live], "chapter": index}
    if only is not None:  # (put together ahead of Finish: the book isn't joined yet)
        yield {"status": f"Done · {len(files)} chapter(s) assembled", "made": len(files), "progress": 1.0}
        return
    for k in [k for k in done_before if k >= len(built)]:  # chapters taken out of the story
        done_before.pop(k)
    save_manifest()
    out = yield from _combine(files, titles, folder / slug, out_format, title or name or "Story", author, cover,
                              base=chapter_share, span=1.0 - chapter_share)
    seconds = sum(sf_duration(f) for f in files)
    gaps = [t for t, f in zip(titles, files) if f.stem.endswith("-gaps")]
    warn = (f"\nSome lines came out silent twice and were left out of: {', '.join(gaps)}. "
            "Press Create audiobook again to remake just those chapters.") if gaps else ""
    try:
        prune_lines(folder, built, gen)
    except Exception as e:  # noqa: BLE001 - tidying only; the book is made
        print(f"[lines] couldn't tidy {folder.name}'s kept lines: {e}", flush=True)
    warn += _made_note(auto_made, auto_failed)
    if sound:  # said again at the end: each chapter's own note is gone by the time the book is done
        from . import sfx
        skipped = sfx.missing(story)
        if skipped:
            names = ", ".join(f"“{n}”" for _, n in skipped[:6]) + (f" and {len(skipped) - 6} more" if len(skipped) > 6 else "")
            warn += (f"\nNot heard: {len(skipped)} tag(s) with no sound in your library ({names}). Press ✦ Make "
                     "missing sounds with AI (Sound tab), then Create audiobook again: only the chapters with "
                     "those tags are made again.")
    from . import cloud_book
    cloud_book.done_with(folder)  # the lines that came back from the cloud are in the book now
    yield {"audio": str(out), "status": f"Done · {len(files)} chapter(s) · {en.clock(seconds)} of audio\nSaved to {out}{warn}"}


def chapter_timings(story, chapter_index, file_path, quotes_only=True):
    """Per-paragraph playback timestamps for a rendered chapter: [{"para", "start", "end", "who", "estimated"}].
    Reads the exact .timings.json sidecar written by engine.render(); if the chapter was rendered before
    timings were saved, estimates proportional timestamps from paragraph character lengths."""
    p = Path(file_path)
    if not p.is_file():
        raise ValueError("That chapter audio file isn't there.")
    chapters = story_chapters(story or "", quotes_only)
    if not 0 <= int(chapter_index) < len(chapters):
        return {"timings": [], "estimated": True}
    ch_title, _, paras = chapters[int(chapter_index)]
    # in_parts() prepends a carried-over {amb}/{room} paragraph on part 2+ which is not in the user's raw story text
    carried_shift = (
        1 if book.is_part(ch_title) and paras and len(paras[0]) == 1
        and book._BED.sub("", paras[0][0][2] or "").strip() == ""
        else 0
    )
    sidecar = p.with_suffix(".timings.json")
    if sidecar.is_file():
        try:
            raw = json.loads(sidecar.read_text(encoding="utf-8"))
            by_para = {}
            for row in raw:
                pi = row.get("para")
                if pi is None:
                    continue
                pi = int(pi) - carried_shift
                if pi < 0:
                    continue
                cur = by_para.get(pi)
                if cur is None:
                    by_para[pi] = {"para": pi, "start": float(row["start"]), "end": float(row["end"]),
                                   "who": row.get("who") or "", "estimated": False}
                else:
                    cur["start"] = min(cur["start"], float(row["start"]))
                    cur["end"] = max(cur["end"], float(row["end"]))
            if by_para:
                return {"timings": [by_para[k] for k in sorted(by_para)], "estimated": False}
        except (OSError, ValueError, KeyError, TypeError):
            pass
    # Fallback for chapters made before sidecar timings existed
    dur = sf_duration(p)
    effective_paras = paras[carried_shift:] if carried_shift else paras
    spoken_title = bool(ch_title and ch_title not in book.UNSPOKEN_TITLES and not book.is_part(ch_title))
    title_w = (len(ch_title) + 18) if spoken_title else 0  # +18 chars ≈ 1.2s title pause
    weights = [max(1, sum(len(t) for _, _, t in para)) for para in effective_paras]
    total = (title_w + sum(weights)) or 1
    pos = dur * (title_w / total)
    out = []
    for pi, (para, w) in enumerate(zip(effective_paras, weights)):
        span = dur * (w / total)
        out.append({"para": pi, "start": round(pos, 3), "end": round(pos + span, 3),
                    "who": para[0][0] if para else "Narrator", "estimated": True})
        pos += span
    return {"timings": out, "estimated": True}


def regen_paragraph(chapter_index, para_index, seed_bump=1, **kw):
    """Regenerate ONLY one paragraph in a chapter with a fresh RNG seed, reusing 100% of all other cached
    lines in that chapter (never touching other chapters or the full book), and save a short neighboring-context
    preview clip (prev + new + next)."""
    import soundfile as sf
    from . import cloud_book, sfx
    book_id = kw.get("book")
    token = sfx.BOOK.set(book_id) if book_id else None
    try:
        where, chapters, gen = chapters_to_make(all_chapters=True, **kw)
        idx = int(chapter_index)
        pi = int(para_index)
        if not 0 <= idx < len(chapters):
            raise ValueError("That chapter isn't in the story.")
        _, title, items = chapters[idx]
        st_ch = story_chapters(kw.get("story") or "", kw.get("quotes_only", True))
        ch_paras = st_ch[idx][2] if idx < len(st_ch) else []
        carried_shift = (
            1 if book.is_part(title) and ch_paras and len(ch_paras[0]) == 1
            and book._BED.sub("", ch_paras[0][0][2] or "").strip() == ""
            else 0
        )
        target_pi = pi + carried_shift
        target_items = [it for it in en.spoken(items, gen["sound"]) if "cue" not in it and it.get("para") == target_pi]
        if not target_items:
            raise ValueError("That paragraph has no spoken lines to regenerate.")
        where.mkdir(parents=True, exist_ok=True)
        local_cache = where / LINES
        cloud_cache = where / CLOUD_LINES
        opts = {"language": gen["language"], "temperature": gen["temperature"]}
        seed = int(gen.get("seed") or 0)
        # Drop cached takes ONLY for the target paragraph so every other paragraph's lines stay cached
        for it in target_items:
            lspec = cloud_book.local_spec(it, opts, seed)
            if lspec is not None:
                (local_cache / f"{cloud_book.key_of(lspec)}.flac").unlink(missing_ok=True)
            cspec = cloud_book.line_spec(it, opts, seed)
            if cspec is not None:
                f = cloud_cache / f"{cloud_book.key_of(cspec)}.flac"
                if f.exists():
                    aside = cloud_cache / "set-aside"
                    aside.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(f), str(aside / f.name))
        # Use fresh RNG seed for the target paragraph ONLY, while cache_seed stays at the book's real seed
        # so 100% of untouched paragraphs hit the cache and are NOT re-synthesized.
        fresh_seed = (seed + int(seed_bump or 1)) if seed else (int(time.time()) % 100000 + 1)
        path = chapter_file(where, idx, items, kw.get("language", "english"), kw.get("stability", 55),
                            seed, kw.get("sound", True), kw.get("bed", 1.0))
        made = path
        yield {"status": f"Regenerating paragraph {pi + 1} of “{title or 'Story'}” ({len(target_items)} line(s))…", "progress": 0.1}
        for out, status in en.render(
            items, kind="chapter", path=path,
            premade=cloud_cache if cloud_cache.is_dir() else None,
            keep=local_cache,
            language=gen["language"], temperature=gen["temperature"],
            seed=fresh_seed, cache_seed=seed,
            batch=1, sound=gen["sound"], bed=kw.get("bed", 1.0), parallel=False,
        ):
            made = Path(out) if out else made
            yield {"status": str(status).split("\n")[0]}
        # Update chapter manifest
        manifest = _read_manifest(where)
        ch_map = manifest.get("chapters") or {}
        seconds = round(sf_duration(made), 3) if made.exists() else None
        st_ch = story_chapters(kw.get("story") or "", kw.get("quotes_only", True))
        tkey = text_key(st_ch[idx][0], st_ch[idx][1]) if idx < len(st_ch) else None
        ch_map[str(idx)] = {"title": title or "Story", "text": tkey, "file": str(made), "seconds": seconds}
        import os as _os
        mpath = where / MANIFEST
        mtmp = mpath.with_name(f".{mpath.name}.tmp")
        try:
            with open(mtmp, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"chapters": ch_map}, indent=1, ensure_ascii=False))
                fh.flush()
                _os.fsync(fh.fileno())
            _os.replace(mtmp, mpath)
        finally:
            mtmp.unlink(missing_ok=True)
        # Build a neighboring-context preview clip: [prev para tail (up to 4s) + target para + next para head (up to 4s)]
        tm = chapter_timings(kw.get("story") or "", idx, str(made), kw.get("quotes_only", True))["timings"]
        cur = next((x for x in tm if x["para"] == pi), None)
        preview_path = where / f"_regen-ch{idx}-p{pi}.wav"
        if cur and made.is_file():
            prev_row = next((x for x in reversed(tm) if x["para"] < pi), None)
            next_row = next((x for x in tm if x["para"] > pi), None)
            s0 = max(0.0, (max(prev_row["start"], cur["start"] - 4.0) if prev_row else cur["start"] - 0.5))
            s1 = (min(next_row["end"], cur["end"] + 4.0) if next_row else cur["end"] + 0.5)
            info = sf.info(str(made))
            i0 = max(0, int(s0 * info.samplerate))
            i1 = min(info.frames, int(s1 * info.samplerate))
            if i1 > i0:
                seg, sr = sf.read(str(made), start=i0, stop=i1, dtype="float32")
                sf.write(str(preview_path), seg, sr, subtype="PCM_16")
        yield {
            "audio": str(preview_path if preview_path.is_file() else made),
            "chapter_file": str(made),
            "chapter": idx,
            "para": pi,
            "para_start": cur["start"] if cur else 0.0,
            "para_end": cur["end"] if cur else 0.0,
            "status": f"Done · regenerated paragraph {pi + 1} (playing with surrounding context)",
        }
    finally:
        _reset(sfx.BOOK, token)


COMBINING = {"MP3": "Encoding MP3", "WAV": "Exporting WAV"}  # else "Finalizing M4B"


def _combine(files, titles, out_base, fmt, title, author, cover, base=0.0, span=1.0):
    """export.combine in a thread, with monotonic progress mapped onto [base, base + span]; returns its path.
    Stopping the task stops ffmpeg."""
    import queue
    import threading
    what = COMBINING.get(fmt, "Finalizing M4B")
    q, result, stopped = queue.Queue(), {}, threading.Event()

    def tick(p):
        q.put(("progress", p))
        return not stopped.is_set()

    def work():
        try:
            result["out"] = export.combine(files, titles, out_base, fmt, title, author, cover, progress=tick)
        except BaseException as e:  # noqa: BLE001 - raised again below
            result["error"] = e
        finally:
            q.put(("end", None))

    yield {"status": what, "progress": base}
    threading.Thread(target=work, daemon=True).start()
    last, shown = 0.0, -1
    try:
        while True:
            kind, p = q.get()
            if kind == "end":
                break
            last = max(last, p)
            pct = int(last * 100)
            if pct != shown:  # (ffmpeg reports many times a second)
                shown = pct
                yield {"status": what, "progress": min(1.0, base + span * last)}
    finally:
        stopped.set()
    if "error" in result:
        raise result["error"]
    return result["out"]


def kaggle_bundle(story, table, models="cast", quotes_only=True, speed=1.0, stability=55, language="english", seed=0,
                  batch=2, line_pause=0.2, para_pause=0.7, out_format="WAV", title="", author="", cover=None, *_,
                  folder=None, name="", **__):
    """Bundle the book, cast voices, cover and the Kaggle runner + notebook into one zip. Returns (zip, message)."""
    import zipfile
    from . import sfx
    built, _, _ = build_book(sfx.strip(story) if sfx.CUE.search(story or "") else story, table, quotes_only, speed,
                             line_pause, para_pause, models)  # the Kaggle runner doesn't mix sound effects
    slug = slug_of(name) if (name or "").strip() else book_name(title, story)
    folder = book_folder(title, story, folder)
    folder.mkdir(parents=True, exist_ok=True)
    used = sorted({it["voice"][3:] for _, items in built for it in items if it["voice"].startswith("my:")})
    project = {"title": title or name or "Story", "author": author or "", "slug": slug, "format": out_format,
               "language": language, "temperature": temp_from_stability(stability), "seed": int(seed or 0),
               "batch": max(int(batch), 4), "cover": "cover.jpg" if cover else None,
               "voices": {name: {"transcript": vl.info(name)["transcript"]} for name in used},
               "chapters": [{"title": t or "", "items": [{k: it.get(k) for k in ("text", "voice", "instruct", "speed", "pause", "who")}
                                                       for it in items]} for t, items in built]}
    out = folder / f"{slug}-kaggle.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("project.json", json.dumps(project, indent=1, ensure_ascii=False))
        z.write(KAGGLE_DIR / "runner.py", "runner.py")
        z.write(KAGGLE_DIR / "qwen-tts-studio-kaggle.ipynb", "qwen-tts-studio-kaggle.ipynb")
        for name in used:
            z.write(vl.wav_path(name), f"voices/{name}.wav")
        if cover and Path(cover).exists():
            z.write(cover, "cover.jpg")
    lines = sum(len(items) for _, items in built)
    engine_ids = sorted({it["voice"].split(":", 1)[0] for _, items in built for it in items})
    note = ("Saved voices (" + ", ".join(used) + ") are cloned from their samples on Kaggle and don't take emotions there. "
            if used else "")
    msg = (f"Kaggle bundle ready: {len(built)} chapter(s), {lines} lines, engines: {', '.join(engine_ids)}.\n{note}"
           f"Upload this zip as a Kaggle Dataset, then open the notebook inside it on Kaggle.\n{out}")
    return str(out), msg
