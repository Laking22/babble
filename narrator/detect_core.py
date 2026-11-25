"""Who speaks each line: the part of AI speaker detection that runs the language model, with no other imports, so
the same code runs on this Mac (ai_detect, with MLX) and on a cloud GPU (kaggle_render/llm_runner.py, with
transformers). Give it the story's paragraphs and a generate(prompts, max_tokens) -> replies function.

Reads the story in passages so each line of dialogue has its surrounding narration, and labels who speaks, their
gender and their tone. Character names are carried across passages so the same person keeps the same name for the
whole book. On a GPU several passages go at once (batch); they share the names found before them.
"""
import json
import re
from collections import Counter, defaultdict

QUOTE = re.compile(r"“[^”]*”|\"[^\"]*\"")
MAX_LINES, MAX_CHARS = 10, 2600
_SENT_END = re.compile(r"[.!?…][\"”’']?\s*$")


def split_multi_speaker(paragraph):
    """Split a paragraph containing 2+ separate dialogue turns into individual paragraphs so each speaker can be
    labelled independently. Keeps mid-sentence split quotes ('"Wait," she said, "don't go!"') intact."""
    matches = list(QUOTE.finditer(paragraph))
    if len(matches) < 2:
        return [paragraph]
    cuts = []
    for prev, nxt in zip(matches, matches[1:]):
        between = paragraph[prev.end():nxt.start()]
        # Split before the next quote when the previous quote+attribution finished a sentence (or quotes are adjacent)
        if not between.strip() or _SENT_END.search(between):
            cuts.append(nxt.start())
    if not cuts:
        return [paragraph]
    out, pos = [], 0
    for c in cuts:
        part = paragraph[pos:c].strip()
        if part:
            out.append(part)
        pos = c
    tail = paragraph[pos:].strip()
    if tail:
        out.append(tail)
    return out or [paragraph]

PROMPT = """You are labelling dialogue in a story so an audiobook can give each character their own voice.

Characters found so far: {known}

Below is an excerpt. Paragraphs marked [n] contain spoken dialogue; unmarked paragraphs are narration for context.
For EACH numbered paragraph, decide who speaks the quoted words and how they sound.

Rules:
- speaker: the character's name as used in the story. Reuse a name from the list above when it is the same person.
  Use "Narrator" only if the narrator is speaking aloud. Use "Unknown" only if it truly cannot be worked out.
- Speech tags and pronouns ("she said", "he whispered") refer to the nearest fitting character. In a back-and-forth
  conversation without tags, speakers usually alternate.
- gender: male, female, or unknown.
- emotion: 2 to 5 words describing the tone of voice, e.g. "excited, breathless", "calm and warm", "angry whisper".

Reply with exactly one JSON object per numbered paragraph, one per line, and nothing else:
{{"n": 1, "speaker": "Name", "gender": "female", "emotion": "excited, breathless"}}

Excerpt:
{excerpt}"""


def _canonical(name, known):
    name = " ".join((name or "").strip().strip('"').split())
    if not name or name.lower() in ("unknown", "none", "?"):
        return "Unknown"
    if name.lower() in ("narrator", "the narrator"):
        return "Narrator"
    for k in known:
        if k.lower() == name.lower():
            return k
    first = name.split()[0].lower()
    matches = [k for k in known if k.split()[0].lower() == first or k.split()[-1].lower() == name.split()[-1].lower()]
    return matches[0] if len(matches) == 1 else name


def _windows(paragraphs, starts=()):
    """Group paragraph indexes into excerpts holding up to MAX_LINES dialogue paragraphs. A new chapter (an index in
    `starts`) starts a new excerpt: a small model read past a chapter's last line as the end and stopped there."""
    windows, current, lines, chars = [], [], 0, 0
    starts = set(starts or ())
    for i, p in enumerate(paragraphs):
        if i in starts and current:
            windows.append(current)
            current, lines, chars = [], 0, 0
        current.append(i)
        chars += len(p)
        if QUOTE.search(p):
            lines += 1
        if lines >= MAX_LINES or chars >= MAX_CHARS:
            windows.append(current)
            current, lines, chars = [], 0, 0
    if current:
        windows.append(current)
    return [w for w in windows if any(QUOTE.search(paragraphs[i]) for i in w)]


def _known_text(known, genders):
    return ", ".join(f"{k} ({genders[k].most_common(1)[0][0] if genders[k] else 'unknown'})" for k in known) or "none yet"


def _excerpt(texts, window, ask=None):
    """(prompt lines, {n: paragraph index}) for one passage: two paragraphs before it for context. `ask`: the
    paragraphs to number (default: every one with dialogue); the others are there as context."""
    numbered, lines = {}, []
    for i in range(max(0, window[0] - 2), window[0]):
        lines.append(texts[i][:400])
    for i in window:
        if (i in ask) if ask is not None else QUOTE.search(texts[i]):
            numbered[len(numbered) + 1] = i
            lines.append(f"[{len(numbered)}] {texts[i]}")
        else:
            lines.append(texts[i][:600])
    return lines, numbered


def _read_reply(reply, numbered, labels, genders, known):
    for m in re.finditer(r"\{[^{}]*\}", reply or ""):
        try:
            item = json.loads(m.group())
            n = int(item.get("n"))
        except (ValueError, TypeError):
            continue
        if n not in numbered:
            continue
        speaker = _canonical(item.get("speaker"), known)
        if speaker not in ("Narrator", "Unknown") and speaker not in known:
            known.append(speaker)
        gender = str(item.get("gender", "")).lower()
        if speaker not in ("Narrator", "Unknown") and gender in ("male", "female"):
            genders[speaker][gender] += 1
        emotion = " ".join(str(item.get("emotion", "")).replace("[", "").replace("]", "").split())[:60]
        labels[numbered[n]] = (speaker, emotion)


def _skipped(texts, numbered, labels):
    """Excerpts asking again about the paragraphs a reply left out (a small model can stop a few lines short, and
    those lines then went to the narrator without a word): from the first skipped one to the last."""
    left = sorted(i for i in numbered.values() if i not in labels)
    return [_excerpt(texts, list(range(left[0], left[-1] + 1)), ask=set(left))] if left else []


def run(texts, generate, stop=None, batch=1, starts=()):
    """Yields ("progress", done, total, state) after each batch of passages, and finally ("result", *state), where
    state is (labels {paragraph index: (speaker, emotion)}, genders {name: Counter}, known names in order).
    generate(prompts, max_tokens) returns one reply per prompt. Stops early (with what it has) when stop is set.
    starts: the paragraph indexes that open a chapter (optional)."""
    windows = _windows(texts, starts)
    total = sum(1 for p in texts if QUOTE.search(p))
    labels, genders, known, done = {}, defaultdict(Counter), [], 0

    def ask(group):
        known_text = _known_text(known, genders)
        prompts = [PROMPT.format(known=known_text, excerpt="\n\n".join(lines)) for lines, _ in group]
        return generate(prompts, [40 * len(numbered) + 40 for _, numbered in group])

    for b in range(0, len(windows), max(1, batch)):
        if stop is not None and stop.is_set():
            break
        group = [_excerpt(texts, w) for w in windows[b:b + max(1, batch)]]
        for (_, numbered), reply in zip(group, ask(group)):
            _read_reply(reply, numbered, labels, genders, known)
            done += len(numbered)
        again = [e for _, numbered in group for e in _skipped(texts, numbered, labels)]
        if again and not (stop is not None and stop.is_set()):
            for (_, numbered), reply in zip(again, ask(again)):
                _read_reply(reply, numbered, labels, genders, known)
        yield ("progress", done, total, (labels, genders, known))
    yield ("result", labels, genders, known)


def prompt_tokens(texts, batch=1, starts=()):
    """(prompts, longest reply in tokens) roughly, for the time estimate before sending."""
    windows = _windows(texts, starts)
    return len(windows), sum(40 * sum(1 for i in w if QUOTE.search(texts[i])) + 40 for w in windows)
