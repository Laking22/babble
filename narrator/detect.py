"""Find who says each line of dialogue in a plain story, and how they say it.

Rule-based, no AI model needed. It reads speech tags ("…," Mara whispered / said Mr. Bell),
action beats (Mara grinned. "…"), pronouns (she said → the most recent fitting character),
and turn-taking for untagged lines. The result is a tagged script you can review and edit.
"""
import re

QUOTE = re.compile(r"“[^”]*”|\"[^\"]*\"")
TITLES = r"(?:Mr|Mrs|Ms|Miss|Dr|Sir|Lady|Lord|Captain|Professor|Aunt|Uncle|King|Queen|Prince|Princess|Father|Mother|Sister|Brother)"
NAME = rf"(?:{TITLES}\.?\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"
MALE_TITLES = {"mr", "sir", "lord", "uncle", "king", "prince", "father", "brother", "captain"}
FEMALE_TITLES = {"mrs", "ms", "miss", "lady", "aunt", "queen", "princess", "mother", "sister"}
STOP = {"The", "A", "An", "He", "She", "They", "I", "It", "We", "You", "His", "Her", "Their", "Then", "But", "And",
        "When", "As", "After", "Before", "Now", "There", "This", "That", "These", "Those", "One", "Every", "Far", "So",
        "Yes", "No", "Oh", "Well", "What", "Why", "How", "Who", "Where", "If", "In", "On", "At", "From", "With",
        "Together", "Click", "Nothing", "Still", "Just", "Even", "Only", "Once", "Maybe", "Perhaps", "Suddenly",
        "Later", "Soon", "All", "Some", "My", "Our", "Your", "Its", "Not", "Let", "Don", "Please", "Thank", "Hello",
        "Come", "Go", "Look", "Wait", "Stop", "Good", "Okay", "Hey", "Sorry", "Never", "Always", "Here", "God"}

EMOTION = {
    "whispered": "whispering", "murmured": "softly, murmuring", "breathed": "breathy, barely audible",
    "muttered": "muttering under breath", "mumbled": "mumbling", "shouted": "shouting", "yelled": "yelling",
    "screamed": "screaming", "shrieked": "shrieking", "roared": "roaring, furious", "bellowed": "bellowing",
    "cried": "crying out, emotional", "sobbed": "sobbing", "wept": "weeping", "laughed": "laughing",
    "chuckled": "chuckling", "giggled": "giggling", "sighed": "with a weary sigh", "gasped": "gasping, shocked",
    "growled": "growling, angry", "snapped": "snapping, irritated", "hissed": "hissing, angry",
    "snarled": "snarling, furious", "pleaded": "pleading", "begged": "begging, desperate",
    "demanded": "demanding, firm", "insisted": "insisting, firm", "stammered": "stammering, nervous",
    "stuttered": "stuttering, nervous", "exclaimed": "excited", "teased": "teasing, playful",
    "joked": "joking, playful", "groaned": "groaning", "grumbled": "grumbling", "protested": "protesting",
    "warned": "warning, serious", "barked": "barking an order", "called": "calling out", "admitted": "reluctant",
    "announced": "announcing, proud", "declared": "declaring, confident", "sniffed": "sniffing, upset",
}
NEUTRAL = ("said", "says", "asked", "asks", "replied", "answered", "added", "continued", "began", "told",
           "agreed", "explained", "offered", "suggested", "promised", "repeated", "wondered", "responded", "went on")
VERB = r"(?:" + "|".join(sorted(set(EMOTION) | set(NEUTRAL), key=len, reverse=True)) + r")"
ADV = r"(?:[a-z]+ly)"
SUBJECT = rf"(?P<who>{NAME}|he|she|they|I)"

AFTER_SUBJECT_VERB = re.compile(rf"^\s*,?\s*{SUBJECT}\s+(?:(?P<adv1>{ADV})\s+)?(?P<verb>{VERB})\b(?:\s+(?P<adv2>{ADV}))?")
AFTER_VERB_SUBJECT = re.compile(rf"^\s*,?\s*(?P<verb>{VERB})\s+(?P<who>{NAME}|he|she)\b(?:\s*,?\s*(?P<adv2>{ADV}))?")
BEFORE_TAG = re.compile(rf"{SUBJECT}\s+(?:(?P<adv1>{ADV})\s+)?(?P<verb>{VERB})\b(?:\s+(?P<adv2>{ADV}))?[^.“\"!?]*[,:]\s*$")
BEFORE_BEAT = re.compile(rf"(?:^|[.!?]\s+)(?P<who>{NAME})\s+[a-z]+[^.“\"!?]{{0,80}}[.:!?]\s*$")
HONORIFIC = re.compile(rf"\b({TITLES})\.?\s+([A-Z][a-z]+)")


def _candidates(text):
    """Likely character names: titled names, plus names seen mid-sentence or at least twice."""
    names, counts, midsentence = {}, {}, set()
    for title, surname in HONORIFIC.findall(text):
        full = f"{title}. {surname}" if title in ("Mr", "Mrs", "Ms", "Dr") else f"{title} {surname}"
        names[surname] = full
        names[full] = full
    for m in re.finditer(r"\b[A-Z][a-z]+\b", text):
        word = m.group()
        if word in STOP or re.fullmatch(TITLES, word):
            continue
        counts[word] = counts.get(word, 0) + 1
        before = text[max(0, m.start() - 2):m.start()]
        if not re.search(r"[.!?“\"]\s?$|^\s*$", before) and m.start() > 0:
            midsentence.add(word)
    for word, n in counts.items():
        if word not in names and (n >= 2 or word in midsentence):
            names[word] = word
    return names


def _gender_from_title(full):
    title = full.split()[0].rstrip(".").lower() if " " in full else ""
    return "male" if title in MALE_TITLES else "female" if title in FEMALE_TITLES else None


def _emotion(match):
    parts = [EMOTION.get(match.group("verb"), "")] if match.groupdict().get("verb") else []
    for key in ("adv1", "adv2"):
        adv = match.groupdict().get(key)
        if adv and adv not in ("only", "really", "early", "family", "reply"):
            parts.append(adv)
    return ", ".join(p for p in parts if p)


def detect(story):
    """Return (tagged_script, {character: gender or None})."""
    names = _candidates(story)
    genders = {full: _gender_from_title(full) for full in set(names.values())}
    recent = []          # characters by recency of mention, most recent first
    turns = []           # speakers of recent dialogue paragraphs
    out = []

    def mention(full):
        if full in recent:
            recent.remove(full)
        recent.insert(0, full)

    def resolve(who, last_speaker):
        if who is None:
            return None
        if who == "I":
            return "Narrator"
        if who in ("he", "she", "they"):
            want = {"he": "male", "she": "female"}.get(who)
            for full in recent:
                g = genders.get(full)
                if want is None or g == want or (g is None and full != last_speaker):
                    if g is None and want:
                        genders[full] = want
                    return full
            return None
        if who in STOP:
            return None
        full = names.get(who) or names.get(who.split()[-1])
        if full is None:
            full = who
            names[who] = who
            genders.setdefault(who, _gender_from_title(who))
        return full

    for para in re.split(r"\n\s*\n", story.strip()):
        para = " ".join(para.split())
        quotes = list(QUOTE.finditer(para))
        if not quotes:
            for short, full in names.items():
                if re.search(rf"\b{re.escape(short)}\b", para):
                    mention(full)
            out.append(para)
            continue

        pieces, pos, current, last_tag, para_speakers = [], 0, None, None, []
        for i, q in enumerate(quotes):
            narration_before = para[pos:q.start()]
            for short, full in names.items():
                if re.search(rf"\b{re.escape(short)}\b", narration_before):
                    mention(full)
            next_start = quotes[i + 1].start() if i + 1 < len(quotes) else len(para)
            after = para[q.end():next_start]
            speaker, emotion = None, ""
            for pattern, text in ((AFTER_SUBJECT_VERB, after), (AFTER_VERB_SUBJECT, after),
                                  (BEFORE_TAG, narration_before), (BEFORE_BEAT, narration_before)):
                m = pattern.search(text)
                if m:
                    speaker = resolve(m.group("who"), turns[-1] if turns else None)
                    emotion = _emotion(m) if pattern is not BEFORE_BEAT else ""
                    if speaker:
                        break
            if speaker is None and current:
                speaker, emotion = current, last_tag[1] if last_tag else ""  # same paragraph, same speaker
            if speaker is None and len(turns) >= 2 and turns[-1] != turns[-2]:
                speaker = turns[-2]  # back-and-forth conversation
            if speaker is None:
                speaker = "Unknown"
            current = speaker
            if speaker not in ("Narrator", "Unknown"):
                mention(speaker)
            tag = (speaker, emotion)
            pieces.append(narration_before)
            if tag != last_tag:
                pieces.append(f"[{speaker}: {emotion}] " if emotion else f"[{speaker}] ")
                last_tag = tag
            pieces.append(q.group())
            para_speakers.append(speaker)
            pos = q.end()
        pieces.append(para[pos:])
        for short, full in names.items():
            if re.search(rf"\b{re.escape(short)}\b", para[pos:]):
                mention(full)
        turns.append(para_speakers[-1])
        out.append(re.sub(r"\s+", " ", "".join(pieces)).strip())

    speakers = []
    for line in out:
        for m in re.finditer(r"\[([^\]:]+)", line):
            if m.group(1) not in speakers and m.group(1) != "Narrator":
                speakers.append(m.group(1))
    return "\n\n".join(out), {s: genders.get(s) for s in speakers}
