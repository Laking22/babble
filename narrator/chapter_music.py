"""Chapter intros and outros: each chapter's own, kept in the book's project as chapter_music.

  chapter_music = {"chapters": [{"title", "intro": sound id or "", "outro": sound id or "", "by": "director" | "you"}]}

Rows follow the book's chapters (titles(): the chapters Create makes, in order) and are matched by title, so a chapter
added or moved earlier in the story doesn't hand every later chapter the wrong music. The Intros tab sets them one by
one (pick a library sound or design one, hear it, confirm); the director's "Make chapter intros and outros" sets them
all, replacing what was there.

What the director does (plan() and make()): its scenes say how each chapter opens (MOODS) and how it ends (ENDINGS).
From the book's theme it makes up to MAX_VARIANTS versions in the moods the book uses most, so the chapters share one
score without the same intro every time: each chapter gets the version nearest its mood, never the same one as the
chapter before. Outros: a short sting after a cliffhanger, a soft closing phrase after a chapter that ends at rest,
none when the next chapter carries straight on.
"""
import collections
import re

TEMPO = re.compile(r",?\s*(?:at\s+)?(?:\d+\s*bpm|(?:a\s+)?(?:very\s+)?(?:slow|medium|moderate|fast|quick|steady|brisk)\s+tempo)", re.I)

MOODS = {  # a chapter's opening -> how its version of the theme is played
    "calm": "soft and gentle, sparse",
    "warm": "warm and hopeful",
    "sad": "slow and melancholy",
    "tense": "tense with low pulsing strings",
    "eerie": "eerie and hushed, distant",
    "action": "fast and driving with pounding drums",
}
NEAR = {  # the next best version when a mood has none of its own (or the chapter before used it)
    "calm": ["warm", "sad", "eerie", "tense", "action"], "warm": ["calm", "sad", "tense", "eerie", "action"],
    "sad": ["calm", "warm", "eerie", "tense", "action"], "tense": ["eerie", "action", "sad", "calm", "warm"],
    "eerie": ["tense", "sad", "calm", "action", "warm"], "action": ["tense", "eerie", "warm", "sad", "calm"],
}
ENDINGS = ("cliffhanger", "resolved", "continues")
MAX_VARIANTS = 5
INTRO_SECONDS, STING_SECONDS, SOFT_SECONDS = 20, 7, 12
DESIGN_SECONDS = {"intro": INTRO_SECONDS, "sting": STING_SECONDS, "outro": SOFT_SECONDS}
DEFAULT_MOOD = "calm"


def titles(story):
    """The chapters Create makes, in order (books.build_book), by title ("Opening" for untitled text first)."""
    from . import book
    from .cast import parse
    out = []
    for t, body in book.in_parts(book.split_chapters(story or "")):
        paras = parse(body, narrator="Narrator") if body else []
        if not (paras or (t and t != "Title page")):
            continue
        if not paras and (not t or t in book.UNSPOKEN_TITLES):
            continue  # (nothing would be read: no chapter)
        out.append(t or "Opening")
    return out


def match(names, rows):
    """For each chapter name, its row: the same title (the n-th of that title for repeated ones), else the row in
    the same place when that row's title isn't used by any chapter."""
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    by_title = collections.defaultdict(list)
    for r in rows:
        by_title[r.get("title") or ""].append(r)
    seen, out = collections.Counter(), []
    used_titles = set(names)
    for k, name in enumerate(names):
        same = by_title.get(name) or []
        r = same[seen[name]] if seen[name] < len(same) else None
        seen[name] += 1
        if r is None and k < len(rows) and (rows[k].get("title") or "") not in used_titles:
            r = rows[k]
        out.append(r)
    return out


# ---------- the director's chapter music ----------

def plan(theme, chapters):
    """From the director: the book's theme and [{"title", "mood", "ending"}] per chapter. Returns what to make and
    who gets what: {"theme", "variants": {mood: prompt}, "outros": {"sting"|"soft": prompt},
    "chapters": [{"title", "variant", "outro": "sting" | "soft" | ""}]}. Nothing without a theme."""
    theme = (theme or "").strip().rstrip(".")
    if not theme or not chapters:
        return None
    moods = [c.get("mood") if c.get("mood") in MOODS else DEFAULT_MOOD for c in chapters]
    common = [m for m, _ in collections.Counter(moods).most_common(MAX_VARIANTS)]
    if len(common) == 1 and len(chapters) > 1:  # one mood all through: a second version, so chapters can alternate
        common.append(NEAR[common[0]][0])
    rows, last = [], None
    for c, mood in zip(chapters, moods):
        choices = [mood] + NEAR[mood]
        pick = next((m for m in choices if m in common and m != last), None) or next(m for m in choices if m in common)
        ending = c.get("ending") if c.get("ending") in ENDINGS else "resolved"
        rows.append({"title": c.get("title") or "", "variant": pick,
                     "outro": {"cliffhanger": "sting", "resolved": "soft", "continues": ""}[ending]})
        last = pick
    if rows:  # the book's very end closes at rest unless it's a cliffhanger
        rows[-1]["outro"] = rows[-1]["outro"] or "soft"
    used = {r["variant"] for r in rows}
    outros = {r["outro"] for r in rows if r["outro"]}
    return {"theme": theme,
            # (each version's mood sets its pace: the theme's own tempo is left out, so "60 bpm" never meets "fast")
            "variants": {m: f"{TEMPO.sub('', theme).strip(' ,')}, {MOODS[m]}" for m in common if m in used},
            "outros": {k: f"{theme}, " + ("a short dramatic ending sting" if k == "sting" else "a gentle closing phrase that fades out")
                       for k in ("sting", "soft") if k in outros},
            "chapters": rows}


def clips(p):
    """How many pieces of music a plan makes."""
    return len((p or {}).get("variants") or {}) + len((p or {}).get("outros") or {})


def make(p, book=None, scope="all"):
    """Make a plan's music (yields status lines). Returns chapter_music, and a list of (description, error) for any that
    failed (those chapters get none)."""
    from . import books, sfx, sfxgen
    made, failed = {}, []
    todo = [(f"v:{m}", prompt, INTRO_SECONDS, f"{m} intro") for m, prompt in (p.get("variants") or {}).items()]
    todo += [(f"o:{k}", prompt, STING_SECONDS if k == "sting" else SOFT_SECONDS, "sting" if k == "sting" else "ending")
             for k, prompt in (p.get("outros") or {}).items()]
    for n, (key, prompt, seconds, label) in enumerate(todo, 1):
        yield f"Making chapter music {n} of {len(todo)}: “{prompt}” ({seconds} s)…"  # (status lines: books._relay)
        try:
            e = books.keep_in(sfxgen.make(prompt, "music", seconds=seconds), book, scope)
            made[key] = sfx.annotate(e["id"], name=f"{e['name']} · {label}")["id"]  # (which is which, in Intros)
        except Exception as e:  # noqa: BLE001 - the rest still get made
            failed.append((prompt, str(e)))
    rows = [{"title": r["title"], "intro": made.get(f"v:{r['variant']}", ""),
             "outro": made.get(f"o:{r['outro']}", "") if r["outro"] else "", "by": "director"} for r in p["chapters"]]
    return {"chapters": rows}, failed


def design(description, kind="intro", book=None, scope="all"):
    """✦ The Intros tab's Design: one piece of music from words, for an intro ("intro") or an outro ("sting" or
    "outro"). Ends with {"sound": the library entry}."""
    from . import books, sfxgen
    description = (description or "").strip()
    if not description:
        raise ValueError("Describe the music first.")
    seconds = DESIGN_SECONDS.get(kind, INTRO_SECONDS)
    yield {"status": f"Making “{description}” ({seconds} s)…"}
    e = books.keep_in(sfxgen.make(description, "music", seconds=seconds), book, scope)
    yield {"sound": e, "status": f"Made **{e['name']}** ({e['duration']:.0f} s)."}
