"""Sound tags in the story's text: the helpers the sound director (director.py) uses to put its picks in and to
list them for review. (This was Suggest sounds' module; the director replaced it.)

Tags go at the end of the sentence (so a line is never split mid-sentence) with the word the sound happens on;
when rendering, Whisper finds that word and the effect plays right on it. Only adds tags, never changes text.
"""
import re

# Sentence splitter that avoids breaking on decimals (3.50) or common abbreviations (Dr., Mr., Mrs., Ms., St., Jr., Sr.)
# and keeps trailing closing quotes/parentheses/brackets attached to the sentence.
_SENT = re.compile(
    r"(?:\b(?:Mr|Mrs|Ms|Dr|St|Jr|Sr)\.(?=\s)|\d\.\d|[^.!?…])+"
    r"(?:[.!?…]+[”\"’'\)\]]*|$)"
)


def _sentences(p):
    return [s.strip() for s in _SENT.findall(p) if s.strip()]


_STOP = set("""the and a an of to in on at for from with without its it is was be being been by as out up off into onto
but or so then than that this these those his her their our your my some any very just quick short long""".split())


def _stem(w):
    """Whole words with common endings removed: swords -> sword, shattered/shattering -> shatter (hearth stays hearth)."""
    for end in ("ings", "ing", "edly", "ed", "es", "s", "ly"):
        if w.endswith(end) and len(w) - len(end) >= 3:
            w = w[: -len(end)]
            break
    return w[:-1] if len(w) > 4 and w[-1] == w[-2] else w  # knocked -> knock, stopped -> stop


def _stems(text):
    return {_stem(w) for w in re.findall(r"[a-z]+", (text or "").lower()) if len(w) > 2 and w not in _STOP}


def _anchor(word, sentence):
    """The model's word if it's in the sentence, else the closest word that is ("blow" -> "blew")."""
    import difflib
    words = re.findall(r"[\w'’-]+", sentence)
    if not word:
        return ""
    exact = [w for w in words if w.lower() == word.lower()]
    if exact:
        return exact[0]
    near = difflib.get_close_matches(word.lower(), [w.lower() for w in words], n=1, cutoff=0.6)
    return next((w for w in words if near and w.lower() == near[0]), "")


def _insert_after(paragraph, sentence, tag, sent_idx=None):
    """Put a tag right after a sentence in the original paragraph (which may already contain tags).
    Prefers exact sentence-span indexing when sent_idx matches, avoiding substring collisions on short/repeated text."""
    if sent_idx is not None and sent_idx >= 0:
        spans = [m for m in _SENT.finditer(paragraph) if m.group().strip()]
        if sent_idx < len(spans):
            # Verify the indexed span ends similarly to the target sentence (guards against heading lines shifting index)
            tail_n = re.sub(r"\s+", " ", sentence.strip())[-20:]
            span_txt = re.sub(r"\s+", " ", spans[sent_idx].group().strip())
            if not tail_n or tail_n in span_txt or span_txt.endswith(tail_n[-10:]):
                i = spans[sent_idx].end()
                return paragraph[:i] + " " + tag + paragraph[i:]
    tail = sentence.strip()[-30:]
    i = paragraph.find(tail)
    if i < 0:
        return paragraph.rstrip() + " " + tag
    i += len(tail)
    return paragraph[:i] + " " + tag + paragraph[i:]


# clear speech words only: "whispering wind", "birds calling" and "teeth chattering" are real sounds
SPEECH = re.compile(r"\b(voices?|speak\w*|talking|speech|says|said|calling out|calls out|answer\w*|conversation|"
                    r"dialogue)\b", re.I)


def tag_of(p):
    if p["kind"] in ("amb", "music"):
        return ("{" + p["kind"] + ": " + p["name"] + (f", {p['volume']}" if p["volume"] != "normal" and p["name"] != "off"
                                                      else "") + "}")
    return ("{sfx: " + p["name"] + (f" @{p['anchor']}" if p["anchor"] else "")
            + (f", {p['volume']}" if p["volume"] != "normal" else "")
            + "".join(f", {o}" for o in p.get("opts") or []) + "}")  # far / muffled / left / right (the director)


def label_of(p):
    """One line for the review list: 'Effect · Explosion Shatter on “blast” · paragraph 3'."""
    if p["kind"] in ("amb", "music"):
        what = {"amb": "Background", "music": "Music"}[p["kind"]]
        what = f"{what} off" if p["name"] == "off" else f"{what} · {p['name']}"
    else:
        what = f"Effect · {p['name']}" + (f" on “{p['anchor']}”" if p["anchor"] else "")
    how = [p["volume"]] if p["volume"] != "normal" and p["name"] != "off" else []
    how += list(p.get("opts") or [])
    vol = f" ({', '.join(how)})" if how else ""
    return f"{what}{vol} · paragraph {p['para']}" + ("  ·  not in your library yet" if p["missing"] else "")


def insert(text, placements):
    """Add the chosen placements' tags to the text (the same text they were planned on)."""
    parts = re.split(r"(\n\s*\n)", text or "")
    for k in sorted({p["part"] for p in placements}):
        if not (0 <= k < len(parts)):
            continue
        body = parts[k]
        sfx_items = [(idx, p) for idx, p in enumerate(placements) if p["part"] == k and p["kind"] == "sfx"]
        for _, p in sorted(sfx_items, key=lambda x: (-x[1]["s"], -x[0])):
            body = _insert_after(body, p["sentence"], tag_of(p), sent_idx=p.get("s"))
        for kind in ("music", "amb"):  # at the start of the paragraph: background, then music
            first = [p for p in placements if p["part"] == k and p["kind"] == kind]
            if first:
                lead = len(body) - len(body.lstrip())
                # after chapter headings on the paragraph's first lines ("## Chapter 2"), which must stay first
                while body[lead:].lstrip(" \t").startswith("#") and "\n" in body[lead:]:
                    lead = body.index("\n", lead) + 1
                if body[lead:].lstrip(" \t").startswith("#"):  # only a heading: the tag goes on a line of its own after it
                    body = body.rstrip() + "\n"
                    lead = len(body)
                tail = body[lead:]
                body = body[:lead] + tag_of(first[0]) + (" " + tail if tail else "")
        parts[k] = body
    return "".join(parts)
