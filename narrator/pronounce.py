"""Pronunciations: say a word another way ("Kael" -> "Kale", "Nguyen" -> "Win") without changing the text.

Two lists: one for every book and Text to Speech (HOME/pronunciations.json), and each audiobook's own (its
project's "pronunciations"); a book's own wins for the same word. They're applied to the lines just before they're
spoken, so the story keeps its spelling. Whole words or phrases, any capitals; never inside {sound} or [speaker]
tags. A word at the start of a sentence keeps its capital.
"""
import json
import re

from .paths import HOME

FILE = HOME / "pronunciations.json"
TAGS = re.compile(r"\{[^{}]*\}|\[[^\]]*\]|<[^<>]*>")

_CURRENCY_NAMES = {"$": ("dollar", "dollars", "cent", "cents"),
                   "£": ("pound", "pounds", "penny", "pence"),
                   "€": ("euro", "euros", "cent", "cents")}
_CURRENCY = re.compile(r"([$£€])(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d{1,2}))?(?:\s+(thousand|million|billion|trillion))?\b", re.I)
_ABBREVS = [
    (re.compile(r"\bMr\.(?=\s|$)"), "Mister"),
    (re.compile(r"\bMrs\.(?=\s|$)"), "Missus"),
    (re.compile(r"\bMs\.(?=\s|$)"), "Miz"),
    (re.compile(r"\bDr\.(?=\s|$)"), "Doctor"),
    (re.compile(r"\bProf\.(?=\s|$)"), "Professor"),
    (re.compile(r"\bCapt\.(?=\s|$)"), "Captain"),
    (re.compile(r"\bLt\.(?=\s|$)"), "Lieutenant"),
    (re.compile(r"\bSgt\.(?=\s|$)"), "Sergeant"),
    (re.compile(r"\bGen\.(?=\s|$)"), "General"),
    (re.compile(r"\bCol\.(?=\s|$)"), "Colonel"),
    (re.compile(r"\bRev\.(?=\s|$)"), "Reverend"),
    (re.compile(r"\bSt\.(?=\s+[A-Z])"), "Saint"),
    (re.compile(r"\bvs\.(?=\s|$)", re.I), "versus"),
    (re.compile(r"\betc\.(?=\s+(?-i:[A-Z])|$)", re.I), "et cetera."),
    (re.compile(r"\betc\.(?=\s)", re.I), "et cetera"),
    (re.compile(r"\be\.g\.(?=\s|$)", re.I), "for example"),
    (re.compile(r"\bi\.e\.(?=\s|$)", re.I), "that is"),
]
_ROMAN = r"(X{0,3}(?:IX|IV|V?I{0,3}))\b(?![’'][A-Za-z]{2})"  # I to XXXIX ("I'm", "I've" are never numerals)
_ORD = ["", "First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth", "Tenth", "Eleventh",
        "Twelfth", "Thirteenth", "Fourteenth", "Fifteenth", "Sixteenth", "Seventeenth", "Eighteenth", "Nineteenth"]
_TENS = {2: ("Twenty", "Twentieth"), 3: ("Thirty", "Thirtieth")}


def _roman(s):
    """A Roman numeral's value (1-39), or 0 when it isn't one."""
    vals = {"I": 1, "V": 5, "X": 10}
    if not s or any(c not in vals for c in s):
        return 0
    total, prev = 0, 0
    for c in reversed(s):
        v = vals[c]
        total += -v if v < prev else v
        prev = max(prev, v)
    return total if 0 < total < 40 else 0


def _ordinal(n):
    """3 -> "the Third", 23 -> "the Twenty-Third"."""
    if n < 20:
        return f"the {_ORD[n]}"
    word, nth = _TENS[n // 10]
    return f"the {nth}" if n % 10 == 0 else f"the {word}-{_ORD[n % 10]}"


# a numeral after a structural word is a number ("Chapter IV" -> "Chapter 4"); after a monarch's or pope's name it's
# an ordinal ("Henry V" -> "Henry the Fifth", "Pope John XXIII" -> "Pope John the Twenty-Third")
_ROMAN_CARD = re.compile(r"\b(Chapter|Part|Book|Volume|Act|Scene|Section)\s+" + _ROMAN)
_TITLE = r"(?:King|Queen|Pope|Emperor|Empress|Tsar|Tsarina|Czar|Kaiser|Pharaoh|Sultan)"
_ROMAN_TITLED = re.compile(r"\b(" + _TITLE + r"\s+[A-Z][a-z]+)\s+" + _ROMAN)
_ROMAN_REGNAL = re.compile(
    r"\b(Henry|Louis|Charles|Edward|George|Richard|James|Philip|Elizabeth|Nicholas|Alexander|William|Mary|Anne|"
    r"Victoria|Peter|Catherine|Frederick|Napoleon|Ferdinand|Wilhelm|Ivan|Leopold|Gustav|Ramesses|Ptolemy|Pius|"
    r"Gregory|Benedict|Innocent|Clement|Boniface|Urban)\s+" + _ROMAN + r"(?!\.\s+[A-Z])")
_I_ENDS = re.compile(r"[’']s\b|\s*[,.;:!?)]|\s*$|\s+of\b")  # where a bare "I" after a name is a numeral: "Elizabeth I."
# after a title ("Queen Elizabeth I ruled") a bare "I" is a numeral unless a word only the pronoun takes follows
_I_PRONOUN = re.compile(r"\s+(?:am|think|guess|mean|know|said|say|want|need|feel|hope|wish|swear|promise|believe)\b")


def _numeral(m, as_word, lone_i):
    """Replace a matched name + numeral, or leave it: a bare "I" only where lone_i(rest of the text) says so."""
    n = _roman(m.group(2))
    if not n or (n == 1 and not lone_i(m.string[m.end():])):
        return m.group(0)
    return f"{m.group(1)} {as_word(n)}"


def _money(m):
    sym, whole, cents, scale = m.group(1), m.group(2).replace(",", ""), m.group(3), m.group(4)
    sing, plur, csing, cplur = _CURRENCY_NAMES[sym]
    if scale:
        num = f"{whole}.{cents}" if cents else whole
        return f"{num} {scale.lower()} {plur}"
    if cents and len(cents) == 1:
        return m.group(0)
    w = int(whole)
    out = f"{w} {sing if w == 1 else plur}"
    if cents and int(cents):
        c = int(cents)
        out += f" and {c} {csing if c == 1 else cplur}"
    return out


def _normalize_span(text):
    text = _CURRENCY.sub(_money, text)
    text = _ROMAN_CARD.sub(lambda m: _numeral(m, str, lambda rest: not re.match(r"\s+[a-z]", rest)), text)
    text = _ROMAN_TITLED.sub(lambda m: _numeral(m, _ordinal, lambda rest: not _I_PRONOUN.match(rest)), text)
    text = _ROMAN_REGNAL.sub(lambda m: _numeral(m, _ordinal, _I_ENDS.match), text)
    for rx, repl in _ABBREVS:
        text = rx.sub(repl, text)
    return text


def normalize(text):
    """Expand currency, common abbreviations and titled Roman numerals for TTS without touching tags."""
    if not text:
        return text
    pieces, pos = [], 0
    for t in TAGS.finditer(text):
        pieces.append(_normalize_span(text[pos:t.start()]))
        pieces.append(t.group())
        pos = t.end()
    pieces.append(_normalize_span(text[pos:]))
    return "".join(pieces)


def clean(rows):
    """[{"word", "say"}], trimmed, without empties or repeats (the last one of a word wins)."""
    out = {}
    for r in rows or []:
        word = re.sub(r"\s+", " ", str((r or {}).get("word", ""))).strip()[:60]
        say = re.sub(r"\s+", " ", str((r or {}).get("say", ""))).strip()[:120]
        if word and say and word.lower() != say.lower():
            out[word.lower()] = {"word": word, "say": say}
    return list(out.values())


def everyone():
    try:
        return clean(json.loads(FILE.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return []


def save_everyone(rows):
    import os
    rows = clean(rows)
    tmp = FILE.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(rows, indent=1, ensure_ascii=False))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, FILE)
    finally:
        tmp.unlink(missing_ok=True)
    return rows


def merged(book_rows=None):
    """Every-book pronunciations with the book's own on top."""
    return clean(everyone() + list(book_rows or []))


def _compile(rows):
    words = sorted((r["word"] for r in rows), key=len, reverse=True)  # "New York" before "York"
    say = {r["word"].lower(): r["say"] for r in rows}
    # whole words; "Kael's" counts as Kael, "Kael'x" doesn't
    pattern = re.compile(r"(?<![\w'’])(" + "|".join(re.escape(w) for w in words) + r")(?!\w)(?!['’](?!s\b)\w)", re.I)
    return pattern, say


def apply(text, rows):
    """The text as it should be said (custom pronunciations first, then conservative TTS normalization)."""
    if not text:
        return text
    if rows:
        pattern, say = _compile(rows)

        def swap(m):
            new = say[m.group(1).lower()]
            return new[0].upper() + new[1:] if m.group(1)[0].isupper() and new[0].islower() else new

        pieces, pos = [], 0
        for t in TAGS.finditer(text):
            pieces.append(pattern.sub(swap, text[pos:t.start()]))
            pieces.append(t.group())
            pos = t.end()
        pieces.append(pattern.sub(swap, text[pos:]))
        text = "".join(pieces)
    return normalize(text)


def apply_items(items, rows):
    """Render items with their text as it should be said (cue items are left alone)."""
    return [it if "cue" in it or not it.get("text") else dict(it, text=apply(it["text"], rows)) for it in items]


# ---------- tricky-word pre-render scanner ----------

_HOMOGRAPHS = {
    "read": "present 'reed' vs past 'red'",
    "lead": "metal 'led' vs verb 'leed'",
    "tear": "cry 'teer' vs rip 'tair'",
    "bow": "weapon 'boh' vs bend 'bau'",
    "wind": "air 'wind' vs twist 'wynd'",
    "live": "verb 'liv' vs adj 'lyv'",
    "bass": "music 'bays' vs fish 'bass'",
    "wound": "injury 'woond' vs coiled 'wownd'",
    "dove": "bird 'duv' vs past dive 'dohv'",
    "minute": "time 'min-it' vs tiny 'my-nyoot'",
    "row": "line/paddle 'roh' vs argument 'rau'",
    "sow": "plant 'soh' vs pig 'sau'",
    "polish": "shine 'pol-ish' vs nationality 'poh-lish'",
    "desert": "sand 'dez-ert' vs abandon 'di-zert'",
    "resume": "continue 'ri-zoom' vs CV 'rez-oo-may'",
}
_COMMON_CAPS = set("""the a an and or but if then else when while where what who whom whose which that this these those
he she it they we you i his her its their our your my him them us me not no yes all some any many few more most
one two three four five six seven eight nine ten first second third chapter part book prologue epilogue
monday tuesday wednesday thursday friday saturday sunday january february march april may june july august
september october november december mr mrs ms dr prof sir lord lady king queen captain doctor professor
here there now soon later again still just only even also well oh ah""".split())
_ODD_CLUSTER = re.compile(r"(?:^cth|^ny|^kh|^zh|^pt|^gn|^mn|^x[aeiou]|q(?!u)|[bcdfghjklmnpqrstvwxz]{4,})", re.I)


def find_tricky(story, cast_names=(), existing=None, limit=24):
    """Scan a story for words TTS often mispronounces before rendering:
    - unusual/fantasy names (internal apostrophes, odd consonant clusters)
    - recurring proper nouns & cast names
    - common English homographs ('read', 'lead', 'tear', 'bow', 'wind'…)
    Skips words already covered by existing or global pronunciation rules."""
    have = {r["word"].lower() for r in clean(list(everyone()) + list(existing or []))}
    plain = TAGS.sub(" ", story or "")
    counts = {}
    for w in re.findall(r"[A-Za-z][A-Za-z'’-]{2,}", plain):
        clean_w = w.strip("'-’")
        if len(clean_w) < 3:
            continue
        k = clean_w.lower()
        if k not in counts:
            counts[k] = [clean_w, 0]
        counts[k][1] += 1

    out, seen = [], set()

    def push(word, reason, count, note):
        k = word.lower()
        if k in have or k in seen or k in _COMMON_CAPS:
            return
        seen.add(k)
        out.append({"word": word, "reason": reason, "count": count, "note": note})

    for name in cast_names or ():
        for part in re.findall(r"[A-Za-z][A-Za-z'’-]{2,}", str(name or "")):
            if part.lower() != "narrator":
                c = counts.get(part.lower(), [part, 1])[1]
                odd = bool(_ODD_CLUSTER.search(part) or "'" in part or "’" in part)
                if odd or c >= 2:
                    push(part, "fantasy_name" if odd else "proper_noun", c, "Cast character name")

    for k, (w, c) in sorted(counts.items(), key=lambda kv: -kv[1][1]):
        if w.endswith(("'s", "’s", "n't", "n’t", "'ll", "'ve", "'re", "'d")):
            continue
        if ("'" in w or "’" in w or _ODD_CLUSTER.search(w)) and (w[0].isupper() or c >= 2):
            push(w, "fantasy_name", c, "Unusual spelling")

    mid_caps = {}
    for m in re.finditer(r"[a-z,;]\s+([A-Z][a-z'’]{2,})", plain):
        w = m.group(1).strip("'’")
        if not w.endswith(("'s", "’s")):
            mid_caps[w] = mid_caps.get(w, 0) + 1
    for w, c in sorted(mid_caps.items(), key=lambda kv: -kv[1]):
        if c >= 2:
            push(w, "proper_noun", c, f"Proper noun ({c}×)")

    for k, note in _HOMOGRAPHS.items():
        if k in counts and counts[k][1] >= 2:
            push(k, "homograph", counts[k][1], f"Homograph: {note}")

    return out[:limit]
