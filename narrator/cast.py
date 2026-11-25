"""Parse tagged scripts into (character, emotion, text) segments.

Syntax
  [Name] or [Name: emotion]   switches who speaks, and how, until the next tag
  A tag lasts until the end of its paragraph; untagged paragraphs go to the narrator.
  With split_quotes on, only the quoted words in a character's paragraph use their
  voice; the words around the quotes ("she said") stay with the narrator.
"""
import re

TAG = re.compile(r"\[([^\[\]:]+?)(?:\s*:\s*([^\[\]]*))?\]")
QUOTE = re.compile(r"(“[^”]*”|\"[^\"]*\")")
_UNDERLINED = re.compile(r"(?<!\w)_{1,2}([^_\n]+?)_{1,2}(?!\w)")


def plain(text):
    """Markdown emphasis is never read out: **bold**, *italic* and _italic_ keep their words, stray marks go
    (text pasted from a chat or a document often has them, and "**" was read as a line of its own)."""
    text = _UNDERLINED.sub(r"\1", text or "").replace("*", "")
    return re.sub(r"[ \t]{2,}", " ", text).strip()


_NOT_A_NAME = re.compile(r"[\d|/\\=+<>~#@%^&,;!?]")
_CUES = re.compile(r"\{[^{}]*\}")  # sound tags ({sfx: …}, {amb: …}, {room: …}): not words


def _is_tag(match, raw):
    """A [Name] or [Name: emotion] tag says who speaks the words after it. Brackets that aren't one (a game's
    status box like "[Name: Emerald Dragonfly]" or "[Strength: 2 / 1]" on a line of its own) are read as text:
    a tag starts its paragraph or follows the end of a sentence or quote, has words after it, and its name is a few
    words without digits or symbols."""
    name = match.group(1).strip()
    before = _CUES.sub("", TAG.sub("", raw[:match.start()])).rstrip()  # ({room}/{amb}/{music} tags may lead it)
    if before and not re.search(r"[.!?…:;”\"’'—–)\]]$", before):  # mid-sentence ("name it [Censored] anyway"): text
        return False
    rest = TAG.sub("", raw[match.end():]).strip()
    how = (match.group(2) or "").strip()  # an emotion is a few lowercase words ("angry", "softly"), not a value
    words = how.split()
    if len(how) > 60 or re.search(r"[\d|/\\=+<>~#@%^&]", how) or (len(words) >= 2 and all(w[0].isupper() for w in words)):
        return False
    return (bool(rest) and 0 < len(name) <= 32 and len(name.split()) <= 4 and not _NOT_A_NAME.search(name)
            and bool(re.search(r"[^\W\d_]", name)))  # a name has a letter: "[___]" is a form's blank


# A book's own brackets ("[Skill Acquired]", a form's "[___]") become these lookalikes when it's imported, so they're
# read as text and never mistaken for [cast tags]. They aren't read aloud either.
BOOK_BRACKETS = ("⟦", "⟧")
_BRACKETED = re.compile(r"\[([^\[\]]*)\]")


def book_brackets(text, keep=()):
    """The text with its [brackets] turned into ⟦ ⟧, except speaker tags naming someone in `keep` (the cast)."""
    keep = {k.strip().lower() for k in keep if k}
    out = []
    for raw in re.split(r"(\n\s*\n)", text or ""):
        for _ in range(4):  # inside out: "[[x]]" takes two passes
            kept = {m.start() for m in tags(raw) if m.group(1).strip().lower() in keep}
            new = _BRACKETED.sub(lambda m: m.group(0) if m.start() in kept else f"⟦{m.group(1)}⟧", raw)
            if new == raw:
                break
            raw = new
        out.append(raw)
    return "".join(out)


def unbracket(text):
    """Brackets and their lookalikes aren't read aloud: their words are."""
    for ch in "[]" + "".join(BOOK_BRACKETS):
        text = text.replace(ch, " ")
    return text


def tags(raw):
    """The speaker tags in one paragraph (see _is_tag)."""
    return [m for m in TAG.finditer(raw) if _is_tag(m, raw)]


def strip_tags(text, keep=()):
    """The text without its speaker tags (status boxes and other brackets stay), except those naming someone in
    `keep` (Clear cast keeps the Narrator's)."""
    keep = {k.strip().lower() for k in keep if k}
    out = []
    for raw in re.split(r"(\n\s*\n)", text or ""):
        for m in reversed([m for m in tags(raw) if m.group(1).strip().lower() not in keep]):
            raw = raw[:m.start()] + raw[m.end():].lstrip(" ")
        out.append(raw)
    return "".join(out)


def parse(script, narrator="Narrator", split_quotes=True):
    """Return a list of paragraphs, each a list of (character, emotion, text)."""
    paragraphs = []
    for raw in re.split(r"\n\s*\n", script or ""):
        raw = raw.strip()
        if not raw:
            continue
        segments, who, emotion, pos = [], narrator, "", 0
        for match in tags(raw) + [None]:
            end = match.start() if match else len(raw)
            _add(segments, who, emotion, raw[pos:end], narrator, split_quotes)
            if match:
                who = match.group(1).strip()
                emotion = (match.group(2) or "").strip()
                pos = match.end()
        if segments:
            paragraphs.append(segments)
    return paragraphs


def _add(segments, who, emotion, text, narrator, split_quotes):
    text = " ".join(unbracket(text).split())  # a status box's words, not its brackets
    if not text:
        return
    if who == narrator or not split_quotes or not QUOTE.search(text):
        _append(segments, who, emotion, text)
        return
    for piece in QUOTE.split(text):
        piece = piece.strip(" ,")
        if not piece:
            continue
        if QUOTE.fullmatch(piece):
            _append(segments, who, emotion, piece)
        else:
            _append(segments, narrator, "", piece)


def _append(segments, who, emotion, text):
    if segments and segments[-1][0] == who and segments[-1][1] == emotion:
        segments[-1] = (who, emotion, f"{segments[-1][2]} {text}")
    else:
        segments.append((who, emotion, text))


def characters(script, narrator="Narrator"):
    """Names used in the script, narrator first."""
    names = [narrator]
    for raw in re.split(r"\n\s*\n", script or ""):
        for match in tags(raw):
            name = match.group(1).strip()
            if name not in names:
                names.append(name)
    return names
