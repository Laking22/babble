"""Import books (EPUB, PDF, TXT, DOCX, MOBI…) as chapter-marked text.

Output text format: each chapter starts with a line "## Chapter title", paragraphs are
separated by blank lines. That is the format the Story tab reads.
"""
import html
import posixpath
import re
import shutil
import subprocess
import tempfile
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET

CALIBRE = shutil.which("ebook-convert") or "/Applications/calibre.app/Contents/MacOS/ebook-convert"
EBOOK_META = shutil.which("ebook-meta") or "/Applications/calibre.app/Contents/MacOS/ebook-meta"
# a chapter's number in words: "Thirteen", "Twenty-One", "One Hundred and Five", "the First", "the Last"
_NUM_WORD = (r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
             r"seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|first|"
             r"second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|last|final)\b")
_NUMBER = rf"(?:the\s+)?(?:\d+|[IVXLC]+\b|{_NUM_WORD}(?:[\s-]+(?:and\s+)?{_NUM_WORD})*)"
HEADING = re.compile(
    r"^\s*("
    rf"(?:chapter|part|book)(?:\s+{_NUMBER}|\s*[:—–-]).{{0,60}}"
    r"|(?:chapter|part|book)\s*$"
    r"|(?:prologue|epilogue|interlude|afterword|foreword)\b.{0,70}"
    r"|[IVXLC]{1,7}\.?"
    r")\s*$",
    re.I,
)
NO_HEADINGS = "Full text"  # the one "chapter" of a text file without chapter headings
# Chapter names the importer makes up itself: they mark chapters in the file but aren't read out as titles
UNSPOKEN_TITLES = ("Title page", NO_HEADINGS, "Opening")
BLOCK = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "br", "tr", "section", "article", "pre", "hr"}


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.headings, self._skip, self._heading = [], [], 0, None

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head", "title"):
            self._skip += 1
        if tag in BLOCK:
            self.parts.append("\n\n")
        if tag in ("h1", "h2", "h3"):
            self._heading = []

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head", "title"):
            self._skip = max(0, self._skip - 1)
        if tag in ("h1", "h2", "h3") and self._heading is not None:
            text = " ".join("".join(self._heading).split())
            if text:
                self.headings.append(text)
            self._heading = None
        if tag in BLOCK:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if self._skip:
            return
        self.parts.append(data)
        if self._heading is not None:
            self._heading.append(data)


def _paragraphs(text):
    out = []
    for block in re.split(r"\n\s*\n", text):
        block = " ".join(block.split())
        if block and not re.fullmatch(r"\d{1,4}", block):  # drop bare page numbers
            out.append(block)
    return out


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _epub(path):
    z = zipfile.ZipFile(path)
    container = ET.fromstring(z.read("META-INF/container.xml"))
    opf_path = next(e.get("full-path") for e in container.iter() if _local(e.tag) == "rootfile")
    base = posixpath.dirname(opf_path)
    opf = ET.fromstring(z.read(opf_path))
    manifest = {}
    for item in opf.iter():
        if _local(item.tag) == "item":
            manifest[item.get("id")] = (posixpath.normpath(posixpath.join(base, item.get("href", ""))), item.get("media-type", ""),
                                        item.get("properties", ""))
    spine = [i.get("idref") for i in opf.iter() if _local(i.tag) == "itemref"]

    # Chapter titles from the table of contents (EPUB 2 toc.ncx or EPUB 3 nav).
    toc = {}
    for _, (href, media, props) in manifest.items():
        if media == "application/x-dtbncx+xml" and href in z.namelist():
            ncx = ET.fromstring(z.read(href))
            for nav in ncx.iter():
                if _local(nav.tag) == "navPoint":
                    label = next((" ".join(t.text.split()) for t in nav.iter() if _local(t.tag) == "text" and t.text), "")
                    src = next((c.get("src") for c in nav if _local(c.tag) == "content"), None)
                    if src and label:
                        target = posixpath.normpath(posixpath.join(posixpath.dirname(href), src.split("#")[0]))
                        toc.setdefault(target, label)
        if "nav" in props and href in z.namelist() and not toc:
            for m in re.finditer(r'<a[^>]+href="([^"#]+)[^"]*"[^>]*>(.*?)</a>', z.read(href).decode("utf-8", "replace"), re.S):
                target = posixpath.normpath(posixpath.join(posixpath.dirname(href), m.group(1)))
                toc.setdefault(target, " ".join(html.unescape(re.sub("<[^>]+>", "", m.group(2))).split()))

    chapters = []
    for idref in spine:
        href, media, _ = manifest.get(idref, (None, "", ""))
        if not href or "html" not in media or href not in z.namelist():
            continue
        parser = _Text()
        parser.feed(z.read(href).decode("utf-8", "replace"))
        paras = _paragraphs("".join(parser.parts))
        if not paras:
            continue
        title = toc.get(href)
        if title or not chapters or (not toc and parser.headings):
            title = title or (parser.headings[0] if parser.headings else f"Section {len(chapters) + 1}")
            if paras and paras[0].lower() == title.lower():
                paras = paras[1:]
            chapters.append([title, paras])
        else:
            chapters[-1][1].extend(paras)  # file split mid-chapter: continue the previous chapter
    return [(t, p) for t, p in chapters if p]


NUMBERED = re.compile(r"^\s*(\d{1,3})[.:)]?\s+(\S[^\n]{0,70}?)\s*$")


NUMBERED_GAP = 2500  # characters: chapters are mostly longer than this; a page (between page numbers) is mostly not
_ENDED = re.compile(r"[.!?…\"”’)\]}*_—]$")  # a paragraph that ends a chapter ends like this


def _numbered(paras):
    """Paragraphs that are chapter headings written as a number and a title ("2 Overreacting", "14. Speak"): short,
    not ending like a sentence, and counting up through the book (at least 3, mostly 1, 2, 3…), so a line that just
    starts with a number isn't taken for one. Page numbers with the page's running head (every page of a converted
    PDF: "12 Hive Minds") aren't either: those come a page apart, mostly with the same words, often mid-sentence."""
    found = [(i, int(m.group(1)), m.group(2).strip().lower()) for i, p in enumerate(paras)
             if (m := NUMBERED.match(p)) and not re.search(r"[.!?,;:…\"”’]$", p.strip())]
    if len(found) < 3:
        return set()
    steps = sum(1 for (_, a, _), (_, b, _) in zip(found, found[1:]) if b == a + 1)
    if steps < 0.7 * (len(found) - 1):
        return set()
    same = max(sum(1 for *_, t in found if t == title) for *_, title in found)
    if same >= 3 and same >= 0.6 * len(found):  # the book's name over and over (two narrators' names take turns)
        return set()
    upto = [0]
    for p in paras:
        upto.append(upto[-1] + len(p))
    gaps = sorted(upto[j] - upto[i + 1] for (i, *_), (j, *_) in zip(found, found[1:]))
    if gaps[len(gaps) // 2] < NUMBERED_GAP:  # a page apart, not a chapter
        return set()
    cut = sum(1 for i, *_ in found[1:] if i + 1 < len(paras)  # (the first can follow a title page)
              and (not _ENDED.search(paras[i - 1].strip()) or paras[i + 1].strip()[:1].islower()))
    return {i for i, *_ in found} if cut <= 0.25 * (len(found) - 1) else set()


def _txt_chapters(text):
    chapters, title, paras = [], None, []
    all_paras = list(_paragraphs(text))
    numbered = _numbered(all_paras)
    for i, para in enumerate(all_paras):
        if (HEADING.match(para) and len(para) <= 80) or i in numbered:
            if paras or title:
                chapters.append((title or "Opening", paras))
            title, paras = para, []
        else:
            paras.append(para)
    if paras or title:
        chapters.append((title or NO_HEADINGS, paras))
    return [(t, p) for t, p in chapters if p]


def _metadata(path, workdir):
    meta = {"title": Path(path).stem, "author": "", "cover": None}
    try:
        cover = Path(workdir) / "cover.jpg"
        out = subprocess.run([EBOOK_META, str(path), f"--get-cover={cover}"], capture_output=True, text=True, timeout=120).stdout
        for line in out.splitlines():
            key, _, value = line.partition(":")
            key = key.strip().lower()
            if key == "title" and value.strip():
                meta["title"] = value.strip()
            elif key.startswith("author") and value.strip():
                meta["author"] = re.sub(r"\s*\[.*?\]", "", value.strip())
        if cover.exists() and cover.stat().st_size > 1000:
            meta["cover"] = str(cover)
    except (OSError, subprocess.SubprocessError):
        pass
    return meta


def import_book(path, workdir):
    """Return dict(title, author, cover, text, chapters) for an ebook file."""
    path, workdir = Path(path), Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    meta = _metadata(path, workdir)
    ext = path.suffix.lower()
    if ext == ".epub":
        chapters = _epub(path)
    elif ext in (".txt", ".md"):
        chapters = _txt_chapters(path.read_text(encoding="utf-8", errors="replace"))
    else:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "book.txt"
            proc = subprocess.run([CALIBRE, str(path), str(out), "--txt-output-formatting=plain"],
                                  capture_output=True, text=True, timeout=900)
            if not out.exists():
                raise ValueError(f"Couldn't read this file: {proc.stderr.strip()[-300:] or 'unknown error'}")
            chapters = _txt_chapters(out.read_text(encoding="utf-8", errors="replace"))
    if not chapters:
        raise ValueError("No readable text found in this book.")
    first_title, first_paras = chapters[0]
    if first_title in ("Section 1", "Opening") and len(" ".join(first_paras)) < 300:
        chapters[0] = ("Title page", first_paras)
    # a file without chapter headings goes in as plain text, with no made-up "## Full text" heading on top
    text = "\n\n".join(("" if title == NO_HEADINGS else f"## {title}\n\n") + "\n\n".join(paras)
                       for title, paras in chapters)
    return {**meta, "text": text, "chapters": [t for t, _ in chapters]}


PART_CHARS = 28000  # about 30 minutes of narration: longer chapters are made in parts of about this size
_PART = re.compile(r" · part \d+$")
_BED = re.compile(r"\{\s*(amb|room)\s*(?::[^{}]*)?\}", re.I)


def is_part(title):
    """A later part of a chapter that was too long to make in one piece (its title isn't read out)."""
    return bool(title and _PART.search(title))


def _cut(body):
    """Where in_parts cuts one very long chapter: [(part text, where the part starts in body)]."""
    sep = "\n\n" if len(re.findall(r"\n\s*\n", body)) > len(body) // (PART_CHARS // 4) else "\n"
    units, pos = [], 0  # [(paragraph or line, where it starts)]
    for m in re.finditer(r"\n\s*\n" if sep == "\n\n" else r"\n", body):
        units.append((body[pos:m.start()], pos))
        pos = m.end()
    units.append((body[pos:], pos))
    parts, cur, size = [], [], 0
    for u in units:
        cur.append(u)
        size += len(u[0]) + len(sep)
        if size >= PART_CHARS:
            parts.append([sep.join(x for x, _ in cur).strip(), cur[0][1]])
            cur, size = [], 0
    tail = sep.join(x for x, _ in cur).strip()
    if cur and tail:
        if parts and size < PART_CHARS // 4:  # (a short tail joins the part before it)
            parts[-1][0] = parts[-1][0] + sep + tail
        else:
            parts.append([tail, cur[0][1]])
    return [(t, at) for t, at in parts]


def chapter_spans(text):
    """Where each chapter of in_parts(split_chapters(text)) is in the text, in the same order: [(start, end, at)].
    start..end is its own text: a whole chapter from its heading to the next heading, a part of a long chapter only
    its paragraphs (the heading stays with the chapter). at: where it begins (a first part: at its heading). None when
    the text's line ends can't be followed (anything but plain newlines)."""
    text = text or ""
    lines = text.splitlines()
    if "\n".join(lines) not in (text, text[:-1]):
        return None
    segs, title, head_at, body_from, pos = [], None, 0, 0, 0  # (cut as split_chapters cuts it)
    for line in lines:
        m = re.match(r"^\s*##\s+(.+?)\s*$", line)
        if m:
            if title is not None or text[body_from:pos].strip():
                segs.append((head_at, body_from, pos))
            title, head_at, body_from = m.group(1), pos, pos + len(line) + 1
        pos += len(line) + 1
    if title is not None or text[body_from:].strip():
        segs.append((head_at, min(body_from, len(text)), len(text)))
    spans = []
    for head_at, body_from, end in segs:
        raw = text[body_from:end]
        body, body_at = raw.strip(), body_from + len(raw) - len(raw.lstrip())
        if len(body) <= PART_CHARS * 1.5:
            spans.append((head_at, end, head_at))
            continue
        starts = [body_at + at for _, at in _cut(body)]
        for k, start in enumerate(starts):
            stop = starts[k + 1] if k + 1 < len(starts) else end
            spans.append((start, stop, head_at if k == 0 else start))
    return spans


def in_parts(chapters):
    """[(title, body)] with each very long chapter cut into parts of about PART_CHARS, at paragraph ends (or line
    ends, for text without blank lines). A chapter is mixed in memory, so a book with no headings (18 hours as one
    chapter) needed more memory than a Mac has. Each part starts with the background ({amb}/{room}) that was playing
    where the last one stopped. Parts after the first are "<title> · part N"."""
    out = []
    for title, body in chapters:
        if len(body or "") <= PART_CHARS * 1.5:
            out.append((title, body))
            continue
        parts = [t for t, _ in _cut(body)]
        beds = {}
        for k, part in enumerate(parts):
            carried = " ".join(v for v in beds.values())
            for m in _BED.finditer(part):
                beds[m.group(1).lower()] = m.group(0)
            name = title if k == 0 else f"{title or 'Story'} · part {k + 1}"
            # (the background on a paragraph of its own: on the first one, it would change how its lines are cut)
            out.append((name, carried + "\n\n" + part if carried and k else part))
    return out


def split_chapters(text):
    """Split Story text into [(title or None, body)]. Text before the first '## ' heading has no title."""
    parts, title, body = [], None, []
    for line in (text or "").splitlines():
        m = re.match(r"^\s*##\s+(.+?)\s*$", line)
        if m:
            if title is not None or "".join(body).strip():
                parts.append((title, "\n".join(body).strip()))
            title, body = m.group(1), []
        else:
            body.append(line)
    if title is not None or "".join(body).strip():
        parts.append((title, "\n".join(body).strip()))
    return parts
