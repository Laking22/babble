"""The sound director: an AI reads the book, then each scene, and plans its sound design. It's supplemental only:
it never changes the words and never directs how anyone speaks.

  Pass 1 (once per book): the sound bible, meaning the story's world, its recurring places and how each sounds, and
          a small music palette, so the same place always sounds the same and the music sounds like one score.
  Pass 2 (per scene): a background for every place, effects on the words where sounds happen, and music over the
          stretches a mood carries. Code then checks every pick: the word must be in its sentence, effects are
          spaced out, a background already playing isn't restated, every piece of music ends, and nothing is a voice.

The picks go to a review list (keep or drop each, then Add, then Undo if you like).

Where it runs: an AI service with the user's own key. Claude goes through Anthropic's library. OpenAI, Google
Gemini, OpenRouter, xAI, DeepSeek, Mistral, Groq and Together go through their OpenAI-style APIs. It can also run
on this Mac with the small local model speaker detection uses, which is free but plans less well. Every prompt and
reply is kept on this Mac (directors/plans.jsonl): that's the data a local director can be trained on later.
"""
import concurrent.futures as cf
import contextvars
import hashlib
import json
import os
import re
import threading
import time

import numpy as np

from . import sfx
from . import settings
from .ai_sfx import SPEECH, _anchor, _insert_after, _sentences, _stem, _stems, label_of  # noqa: F401 - label_of for the review list
from .cast import strip_tags
from .paths import HOME

PLANS = HOME / "directors" / "plans.jsonl"

# ---------- where it runs ----------

# Anthropic's models, with prices in dollars per million tokens: input, output, cache read.
CLAUDE = [
    {"id": "claude-opus-5-5", "label": "Claude Opus 5.5 · best", "price": (4.0, 20.0, 0.20)},
    {"id": "claude-sonnet-5-5", "label": "Claude Sonnet 5.5 · cheaper", "price": (2.0, 10.0, 0.10)},
    {"id": "claude-haiku-5-5", "label": "Claude Haiku 5.5 · cheapest", "price": (0.10, 0.50, 0.01)},
]
FALLBACKS = {"claude-opus-5-5", "claude-sonnet-5-5"}  # Anthropic's server-side fallback after a refusal (not Haiku)

# Services: their name, where their API is (the OpenAI-style ones), where to make a key, and whether using them costs
# money ("paid", "free tier", or "free models": some of its models are free). Free tiers as described in Oct-Nov 2026;
# they change often, and their daily limits can stop a long book partway (the scenes done are kept).
PROVIDERS = {
    "anthropic": {"name": "Claude (Anthropic)", "keys": ["console.anthropic.com", "API Keys", "Create Key"],
                  "cost": "paid", "cost_note": "Billed per use. Best plans."},
    "openai": {"name": "OpenAI", "base": "https://api.openai.com/v1",
               "keys": ["platform.openai.com", "API keys", "Create new secret key"],
               "cost": "paid", "cost_note": "Billed per use."},
    "gemini": {"name": "Google Gemini", "base": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "keys": ["aistudio.google.com", "Get API key", "Create API key"],
               "cost": "free tier", "cost_note": "Free with daily limits; Google may use free-tier text to improve its models."},
    "openrouter": {"name": "OpenRouter", "base": "https://openrouter.ai/api/v1",
                   "keys": ["openrouter.ai", "Keys", "Create API Key"],
                   "cost": "free models", "cost_note": "Models marked free cost nothing (about 50 requests a day); the rest are billed per use."},
    "xai": {"name": "xAI Grok", "base": "https://api.x.ai/v1", "keys": ["console.x.ai", "API Keys", "Create API Key"],
            "cost": "paid", "cost_note": "Billed per use."},
    "deepseek": {"name": "DeepSeek", "base": "https://api.deepseek.com/v1",
                 "keys": ["platform.deepseek.com", "API keys", "Create new API key"],
                 "cost": "paid", "cost_note": "Billed per use, at low prices."},
    "mistral": {"name": "Mistral", "base": "https://api.mistral.ai/v1",
                "keys": ["console.mistral.ai", "API Keys", "Create new key"],
                "cost": "free tier", "cost_note": "Free Experiment plan with limits (it may train on your text); paid plans billed per use."},
    "groq": {"name": "Groq", "base": "https://api.groq.com/openai/v1", "keys": ["console.groq.com", "API Keys", "Create API Key"],
             "cost": "free tier", "cost_note": "Free with daily limits (too low for a whole long book); paid plans billed per use."},
    "together": {"name": "Together AI", "base": "https://api.together.xyz/v1",
                 "keys": ["api.together.ai", "Settings", "API Keys"],
                 "cost": "paid", "cost_note": "Prepaid credit (from $15)."},
}
# models that can't plan sounds (speech, images, embeddings…), left out of a service's list
NOT_CHAT = re.compile(r"embed|tts|whisper|transcri|audio|speech|image|dall-e|vision|moderation|rerank|guard|realtime|"
                      r"search|ocr|imagen|veo|lyria|sora|codex|instruct-v|-001$|:batch\b", re.I)

# per service, models its list has that can't plan through chat completions (a different API only, live or
# streaming only, images, retired); what's left is tried once a week anyway (_usable)
NOT_CHAT_AT = {
    "openai": re.compile(r"-pro\b|codex|deep-research|computer-use|gpt-image|chatgpt-image|babbage|davinci|-instruct\b", re.I),
    "gemini": re.compile(r"antigravity|\baqa\b|computer-use|live|translate|robotics|banana|decision|customtools", re.I),
    "groq": re.compile(r"orpheus|playai|distil", re.I),
    "xai": re.compile(r"imagine|image", re.I),
    "mistral": re.compile(r"ocr|moderation", re.I),
}
PROBE_DAYS = 7     # a service's models are tried again after this (or with a new key)
PROBE_MAX = 80     # services listing more than this (OpenRouter) aren't tried: their own list says what each does

MAC = "mac"
SCENE_CHARS = {MAC: 2400}  # this Mac's scenes: small (the local model's memory)
# On a service, the book's Scene size (Director tab): smaller scenes are more requests and more detail, bigger ones
# fewer requests (a free tier's daily allowance) and less. Tested November 2026 on 10,500 words with flash-lite:
# 9,000 → 8 requests, 38 sounds; 40,000 → 2 requests, 17 (the back of each big scene got few backgrounds and music).
SCENE_SIZES = {"small": 9000, "medium": 20000, "large": 40000}
DEFAULT_SIZE = "small"
DEFAULT_SCENE_CHARS = SCENE_SIZES[DEFAULT_SIZE]
LIMIT_WAITS = 8                             # per-minute limits waited out per request before giving up
BIBLE_CHARS = {MAC: 8000}  # how much of the book the bible pass reads (spread through it)
DEFAULT_BIBLE_CHARS = 200_000  # safe margin under 100k tokens even on dense dialogue (avoids Haiku 5.5 5x cliff)
PARALLEL = 4                    # scenes asked at once on a service


def key_setting(pid):
    return f"ai_key_{pid}"


def key(pid):
    return (settings.load().get(key_setting(pid)) or "").strip()


def _fmt_million(x):
    if x == 0:
        return "$0"
    if x >= 1 and abs(x - round(x)) < 1e-6:
        return f"${int(round(x))}"
    if x >= 0.01:
        return f"${x:.2f}"
    return f"${x:.3f}"


def _model_entry(pid, mid, label, per_million=None):
    """A model item for the Director picker, with its input/output price per million tokens."""
    from . import pricing
    p_in = p_out = None
    if per_million is not None:
        p_in, p_out = float(per_million[0]), float(per_million[1])
    else:
        p, _ = pricing.model_price(pid, mid)
        if p is not None:
            p_in, p_out = p[0] * 1e6, p[1] * 1e6
    free = mid.endswith(":free") or (p_in is not None and p_out is not None and p_in == 0 and p_out == 0)
    free_tier = pid in pricing.FREE_TIER
    if free:
        price_label = "Free"
    elif p_in is not None and p_out is not None:
        pair = f"{_fmt_million(p_in)} / {_fmt_million(p_out)} per 1M"
        price_label = f"Free tier · {pair}" if free_tier else pair
    elif free_tier:
        price_label = "Free tier"
    else:
        price_label = "Price unknown"
    return {
        "id": mid,
        "label": label,
        "price_in": round(p_in, 4) if p_in is not None else None,
        "price_out": round(p_out, 4) if p_out is not None else None,
        "free": free,
        "free_tier": free_tier,
        "price_label": price_label,
    }


def providers():
    """For the app: each service, whether it has a key, the model last used, and Claude's models."""
    s = settings.load()
    models = s.get("director_models") or {}
    out = []
    for pid, p in PROVIDERS.items():
        out.append({"id": pid, "name": p["name"], "keys": p["keys"], "has_key": bool(key(pid)),
                    "cost": p["cost"], "cost_note": p["cost_note"],
                    "model": models.get(pid) or (CLAUDE[0]["id"] if pid == "anthropic" else ""),
                    "models": [_model_entry("anthropic", m["id"], m["label"], m["price"]) for m in CLAUDE]
                    if pid == "anthropic" else None})
    return out


def _openai_client(pid, api_key=None):
    from openai import OpenAI
    return OpenAI(api_key=api_key or key(pid), base_url=PROVIDERS[pid]["base"], max_retries=3, timeout=600)


def _anthropic_client(api_key=None):
    import anthropic
    return anthropic.Anthropic(api_key=api_key or key("anthropic"), max_retries=3, timeout=900)


def _friendly(pid, e, checking_key=False):
    """A service's error in words: a refused key, no credit, or no connection (some Wi-Fi blocks these services).
    checking_key: a bare 400 while listing models is a refused key too (xAI answers a bad key that way)."""
    name = PROVIDERS[pid]["name"]
    text = f"{type(e).__name__}: {e}"
    if re.search(r"Authentication|401|403|invalid.{0,20}key|valid API key|API key not valid|Unauthorized|Permission", text, re.I) \
            or (checking_key and "400" in text):
        return f"{name} didn't accept that key. Make one on {' → '.join(PROVIDERS[pid]['keys'])}."
    if re.search(r"credit|billing|quota|insufficient|402", text, re.I):
        return f"{name} says the account has no credit left. Add some on {PROVIDERS[pid]['keys'][0]}."
    if re.search(r"Connection|Timeout|resolve|unreachable|SSL", text, re.I):
        return f"Couldn't reach {name}. Check the internet connection (some Wi-Fi networks block it) and try again."
    return f"{name}: {_detail(e, 300)}"


def models(pid):
    """A service's models that can plan (Claude's are listed here; the others are asked)."""
    if pid == "anthropic":
        return [_model_entry("anthropic", m["id"], m["label"], m["price"]) for m in CLAUDE]
    if not key(pid):
        raise ValueError(f"Add a {PROVIDERS[pid]['name']} key in Connections first.")
    try:
        found = list(_openai_client(pid).models.list())
    except Exception as e:  # noqa: BLE001 - shown to the user
        raise ValueError(_friendly(pid, e)) from e
    ids = []
    inline_prices = {}
    for m in found:
        mid = getattr(m, "id", "") or ""
        extra = getattr(m, "model_extra", None) or {}
        if not mid or NOT_CHAT.search(mid) or (pid in NOT_CHAT_AT and NOT_CHAT_AT[pid].search(mid)):
            continue
        if extra.get("type") and extra["type"] not in ("chat", "language", "code"):  # Together says what each is
            continue
        outs = ((extra.get("architecture") or {}).get("output_modalities")) if isinstance(extra.get("architecture"), dict) else None
        if outs is not None and "text" not in outs:  # OpenRouter: models that only make images or audio
            continue
        p = extra.get("pricing") if isinstance(extra.get("pricing"), dict) else None
        if p and p.get("prompt") is not None and p.get("completion") is not None:
            try:
                inline_prices[mid] = (float(p["prompt"]) * 1e6, float(p["completion"]) * 1e6)
            except (TypeError, ValueError):
                pass
        ids.append(mid)
    ids = _usable(pid, sorted(ids))
    return [_model_entry(pid, mid, mid.removeprefix("models/").removesuffix(":free"), inline_prices.get(mid)) for mid in ids]


def _probes(pid):
    """What trying the service's models found, for the key it has now: {"key", "at", "bad": {model: why}, "ok": [...]}."""
    kh = hashlib.sha256(key(pid).encode()).hexdigest()[:12]
    try:
        c = json.loads((HOME / "directors" / f"models-{pid}.json").read_text())
    except (OSError, ValueError):
        c = {}
    if c.get("key") != kh or time.time() - c.get("at", 0) > PROBE_DAYS * 86400:
        c = {"key": kh, "at": time.time(), "bad": {}, "ok": []}
    c.setdefault("busy", {})  # {model: when}: busy or slow when tried, tried again a day later
    return c


def _save_probes(pid, c):
    f = HOME / "directors" / f"models-{pid}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(c, indent=1), encoding="utf-8")
        os.replace(tmp, f)
    finally:
        tmp.unlink(missing_ok=True)


def _usable(pid, ids):
    """The models that answer: each is asked for a one-word reply (once a week per key, a few at a time). Ones that
    can't be used (a different API only, retired, not on the account's plan) are left out; busy or rate-limited ones
    stay, and are tried again next time."""
    with _PROBING.setdefault(pid, threading.Lock()):  # once at a time (the key was just saved, and the tab opened)
        return _usable_locked(pid, ids)


_PROBING = {}


def _usable_locked(pid, ids):
    c = _probes(pid)
    todo = [m for m in ids if m not in c["bad"] and m not in c["ok"] and time.time() - c["busy"].get(m, 0) > 86400]
    if todo and len(ids) <= PROBE_MAX:
        from openai import OpenAI
        client = OpenAI(api_key=key(pid), base_url=PROVIDERS[pid]["base"], max_retries=0, timeout=15)

        def one(m):
            try:
                client.chat.completions.create(model=m, messages=[{"role": "user", "content": "Reply with just: ok"}])
                return m, None
            except Exception as e:  # noqa: BLE001 - sorted below
                return m, _unusable(e)
        with cf.ThreadPoolExecutor(8) as pool:
            for m, why in pool.map(one, todo):
                if why is None:
                    c["ok"].append(m)
                elif why:
                    c["bad"][m] = why
                else:
                    c["busy"][m] = time.time()
        _save_probes(pid, c)
    return [m for m in ids if m not in c["bad"]]


def _unusable(e):
    """Why a request shows the model can't be used, or "" when it may work later (busy, a rate limit, safety filter, no connection)."""
    code, text = getattr(e, "status_code", None), _detail(e, 2000)
    if code == 429 and re.search(r"limit: 0\b", text):
        return _detail(e)
    if code in (400, 404) and re.search(
        r"not found|does not exist|not a valid model|invalid model|model.*not supported|deprecated|no endpoints|decommissioned|unknown model",
        text, re.I,
    ):
        return _detail(e)
    return ""


def _forget_model(pid, model, why):
    """A model that failed while directing for good: it's left out of the list from now on."""
    try:
        c = _probes(pid)
        c["bad"][model] = why
        c["ok"] = [m for m in c["ok"] if m != model]
        _save_probes(pid, c)
    except Exception as e:  # noqa: BLE001 - only the list is affected
        print(f"[director] couldn't note {model}: {e}", flush=True)


def _detail(e, n=220):
    """The service's own words for an error (the SDK's message is only "Error code: 400" for some)."""
    b = getattr(e, "body", None)
    if isinstance(b, list) and b:
        b = b[0]
    if isinstance(b, dict):
        b = b.get("error", b)
        if isinstance(b, dict):
            b = b.get("message") or b
    text = re.sub(r"\s+", " ", str(b or e)).strip()
    return text[:n] if n > 300 else text.split(" * Quota")[0].split(" For more information")[0][:n]


def save_key(pid, api_key):
    """Check a key by listing the service's models, then keep it (settings file, readable only by you)."""
    api_key = (api_key or "").strip()
    if pid not in PROVIDERS:
        raise ValueError("Unknown AI service.")
    if not api_key:
        raise ValueError("Paste the key first.")
    try:
        if pid == "anthropic":
            next(iter(_anthropic_client(api_key).models.list(limit=1)), None)
        elif pid == "openrouter":  # its model list is public, so the key is checked on its own page
            import httpx
            r = httpx.get("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
            if r.status_code in (401, 403):
                raise PermissionError("401 Unauthorized")
            r.raise_for_status()
        else:
            next(iter(_openai_client(pid, api_key).models.list()), None)
    except Exception as e:  # noqa: BLE001 - shown to the user
        raise ValueError(_friendly(pid, e, checking_key=True)) from e
    settings.save(**{key_setting(pid): api_key})
    from . import conn_issues
    conn_issues.ok(pid)  # (a new key that works: the old key's problem is over)
    if pid != "anthropic":  # try its models now, so the Director tab's list is ready when it's opened
        threading.Thread(target=_warm, args=(pid,), daemon=True).start()
    return {"ok": True}


def _warm(pid):
    try:
        models(pid)
    except Exception as e:  # noqa: BLE001 - the list is tried again when it's opened
        print(f"[director] couldn't try {pid}'s models: {e}", flush=True)


def forget_key(pid):
    settings.save(**{key_setting(pid): ""})


# ---------- the brief ----------

BRIEF = """You are the sound designer for an audiobook. The narration is already recorded. Your job is supplemental:
add background sounds, sound effects and music where they make a scene more vivid. You never change the words and
you never direct how the narrator or characters speak.

What you can add:
- background: a place's steady sound, from a paragraph on, until the place changes ("off" for silence).
- effect: a one-off sound on the word in a sentence where it happens.
- music: score under a range of paragraphs where a mood carries the scene.

Rules:
1. A background for every place the scene is in, set at the paragraph where the story arrives there. When the story
   cuts between places (or between one mind's different bodies), set it again where the focus moves. For a place the
   bible lists, use its background description word for word, so the same place always sounds the same.
2. Effects only for sounds that physically happen now, in the scene: not memories, plans, summaries, hypotheticals,
   descriptions of how something works, or figures of speech. Never a sound for someone speaking, calling or
   whispering: the narrator reads the voices. The word must be copied exactly from its sentence and be the moment
   the sound happens. For a sound the bible lists (a recurring thing that makes a sound), use its description word
   for word, so it's made once and sounds the same every time.
3. Restraint: about one effect per 900 characters of text on average, never two in one sentence or in sentences
   next to each other. Silence is fine; most paragraphs get nothing.
4. Music only where a mood carries a stretch: wonder, chase, tension, grief, triumph, romance. At most two or three
   cues in a long chapter, none under quiet serious dialogue unless the mood demands it. Each cue covers a range of
   paragraphs and ends where the mood ends. Use one of the bible's music descriptions, word for word, whenever its
   mood fits; write new music only for a mood the bible doesn't have.
5. Descriptions are prompts for a sound generator: name the source, the material and the manner ("small stones
   knocking together underwater", not "rocks"). Backgrounds are continuous textures with no one-off events. Music
   names the mood, the instruments and the tempo, with no vocals. Under 15 words and no commas: join parts with
   "with" and "and".
6. Fit the story's world: no sounds it can't have (no traffic in a fantasy village, no Earth birds on an alien world
   unless the text says so).
Include only picks that satisfy every rule above; omit any that don't."""

BIBLE_ASK = """Here is the book{sampled}. Write its sound bible: the story's world and era (and what that rules out),
every recurring place with its steady background sound, every thing that makes a sound more than once (a character's
car, a creaking door, a dog, a ship's engine, a weapon) with one effect description, two to four music moods with
their instruments, so the music sounds like one score, and the book's theme: one recognizable piece of music that
suits the whole story (its instruments, feel and tempo; versions of it in different moods open the chapters). Use the
rules for descriptions.

<book>
{book}
</book>"""

SCENE_ASK = """{before}Scene (mark only these paragraphs):
{scene}

Plan this scene's sound. backgrounds: where each place's background starts (always set one at paragraph 1 for the
place the scene opens in). effects: the paragraph, the last 3 to 8 words of the sentence copied exactly, the word
where the sound happens, the sound, and options (far, muffled, left, right, quiet, loud) only when they matter.
music: the first and last paragraph each cue covers, and the music. opening: how the scene's first paragraph feels:
"calm", "warm", "sad", "tense", "eerie" or "action". ending: how its last paragraph leaves the story: "cliffhanger"
(a shock, a threat, a question left hanging), "resolved" (at rest, a pause in the story) or "continues" (the action
runs straight on)."""

CHAPTERS_ASK = """

Paragraphs marked (new chapter) start a new chapter: set the background again there for the place it opens in, end
music at the chapter before it unless the mood truly carries on, give each one's opening (as opening) in
chapter_openings, and in chapter_endings how the paragraph just before each one ends its chapter (as ending)."""

LOCAL_FORMAT = """Reply with only one JSON object, nothing else:
{"backgrounds": [{"paragraph": 1, "sound": "..."}], "effects": [{"paragraph": 1, "sentence_end": "...", "word": "...",
"sound": "...", "options": []}], "music": [{"from_paragraph": 1, "to_paragraph": 2, "sound": "..."}], "opening": "calm",
"ending": "resolved"}"""

OPENINGS = ["calm", "warm", "sad", "tense", "eerie", "action"]  # how a chapter opens: its intro's version of the theme
ENDINGS = ["cliffhanger", "resolved", "continues"]  # how it ends: its outro (chapter_music.py)
OLD_OPENINGS = {"soft": "calm", "strong": "tense"}  # (replies kept from before: resume files)

BIBLE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["world", "places", "sounds", "music", "theme"],
    "properties": {
        "world": {"type": "string"},
        "places": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                              "required": ["place", "background"],
                                              "properties": {"place": {"type": "string"}, "background": {"type": "string"}}}},
        "sounds": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["thing", "sound"],
                                              "properties": {"thing": {"type": "string"}, "sound": {"type": "string"}}}},
        "music": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["mood", "music"],
                                             "properties": {"mood": {"type": "string"}, "music": {"type": "string"}}}},
        "theme": {"type": "string"},
    },
}
SCENE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["backgrounds", "effects", "music", "opening", "ending", "chapter_openings", "chapter_endings"],
    "properties": {
        "backgrounds": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["paragraph", "sound"],
            "properties": {"paragraph": {"type": "integer"}, "sound": {"type": "string"}}}},
        "effects": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["paragraph", "sentence_end", "word", "sound", "options"],
            "properties": {"paragraph": {"type": "integer"}, "sentence_end": {"type": "string"},
                           "word": {"type": "string"}, "sound": {"type": "string"},
                           "options": {"type": "array", "items": {
                               "type": "string", "enum": ["far", "muffled", "left", "right", "quiet", "loud"]}}}}},
        "music": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["from_paragraph", "to_paragraph", "sound"],
            "properties": {"from_paragraph": {"type": "integer"}, "to_paragraph": {"type": "integer"},
                           "sound": {"type": "string"}}}},
        "opening": {"type": "string", "enum": OPENINGS},
        "ending": {"type": "string", "enum": ENDINGS},
        "chapter_openings": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["paragraph", "opening"],
            "properties": {"paragraph": {"type": "integer"}, "opening": {"type": "string", "enum": OPENINGS}}}},
        "chapter_endings": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["paragraph", "ending"],
            "properties": {"paragraph": {"type": "integer"}, "ending": {"type": "string", "enum": ENDINGS}}}},
    },
}

# ---------- the book in scenes ----------

HEADING = re.compile(r"^\s*##\s+\S")  # a chapter heading line (book.split_chapters)


class _Scenes(list):
    """Scenes (lists of paragraph indexes), with .heads: the paragraphs that open a chapter."""
    heads = frozenset()


def scene_chars(where, size=None):
    """How many characters a scene holds; scenes bigger than a usual chapter run on across chapters."""
    return SCENE_CHARS.get(where) or SCENE_SIZES.get(size or DEFAULT_SIZE, DEFAULT_SCENE_CHARS)


def layout(text, where, size=None):
    """(parts, paras, scenes): the text split on blank lines (parts, so tags go back exactly where they belong),
    the paragraphs the director reads as (part index, text without tags), and scenes as lists of paragraph indexes."""
    parts = re.split(r"(\n\s*\n)", text or "")
    paras, heads = [], set()  # heads: paragraphs that open a chapter (a "## " heading before them)
    pending = False
    for k in range(0, len(parts), 2):
        lines = parts[k].splitlines()
        if any(HEADING.match(x) for x in lines):
            pending = True
        clean = strip_tags(sfx.strip("\n".join(x for x in lines if not x.lstrip().startswith("#")))).strip()
        if clean:
            if pending:
                heads.add(len(paras))
                pending = False
            paras.append((k, clean))
    limit = scene_chars(where, size)
    span = limit > DEFAULT_SCENE_CHARS
    scenes, cur, size = _Scenes(), [], 0
    scenes.heads = heads
    for i, (_, p) in enumerate(paras):
        # a scene doesn't run on into the next chapter, except big ones (each chapter's start is marked instead)
        if cur and (size + len(p) > limit or (i in heads and not span)):
            scenes.append(cur)
            cur, size = [], 0
        cur.append(i)
        size += len(p)
    if cur:
        scenes.append(cur)
    return parts, paras, scenes


def _book_text(paras, limit):
    """The book for the bible pass: all of it when it fits, else even slices from opening through ending."""
    whole = "\n\n".join(p for _, p in paras)
    if len(whole) <= limit:
        return whole, False
    n = 12
    size = max(500, limit // n)
    span = max(1, len(whole) - size)
    chunks = []
    for i in range(n):
        start = int(round(i * span / (n - 1)))
        if start > 0:
            nl = whole.find("\n\n", start, min(len(whole), start + 400))
            if nl != -1:
                start = nl + 2
        end = min(len(whole), start + size)
        if end < len(whole):
            nl = whole.rfind("\n\n", max(start + 200, end - 400), end)
            if nl != -1:
                end = nl
        chunks.append(whole[start:end].strip())
    return "\n\n[…]\n\n".join(c for c in chunks if c), True


def _scene_prompt(paras, scene, before=True, heads=()):
    """heads: paragraphs that open a chapter. Inside a scene they're marked "(new chapter)" (big scenes run on
    across chapters), and the reply gives each one's opening."""
    first = scene[0]
    lead = ""
    if before and first > 0:
        lead = f"Just before this scene (context only, don't mark it):\n{paras[first - 1][1][-700:]}\n\n"
    inner = [i for i in scene[1:] if i in heads]
    lines = "\n\n".join(f"[{n + 1}] {'(new chapter) ' if i in inner else ''}{paras[i][1]}" for n, i in enumerate(scene))
    ask = SCENE_ASK.format(before=lead, scene=lines)
    if inner:
        ask += CHAPTERS_ASK
    return ask


def _brief(guide=""):
    """The brief, with the book's own notes for the director (the Director tab's box: "dark gloomy spooky")."""
    guide = re.sub(r"\s+", " ", guide or "").strip()[:1500]
    if not guide:
        return BRIEF
    return BRIEF + ("\n\nThe listener's notes for this book's sound. Follow them for the mood, the music and the kinds "
                    "of sounds you pick, within the rules above (they never change the words or how anyone speaks):\n"
                    f"<notes>{guide}</notes>")


def _system(bible, brief=BRIEF):
    return brief + "\n\nThe book's sound bible:\n" + json.dumps(bible, ensure_ascii=False, indent=1)


# ---------- reading the replies ----------

def _json(reply):
    """The JSON object in a reply (a local model may wrap it in words, <think> blocks, or a code fence)."""
    text = re.sub(r"<think>.*?</think>", "", (reply or ""), flags=re.S | re.I).strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group())
        except ValueError:
            return {}
    return {}


def _desc(s):
    """A sound description as a tag can hold it: no braces, @ or brackets; commas become "and" (a comma starts an
    option in a tag); short."""
    s = re.sub(r"[{}@\[\]]", " ", str(s or ""))
    s = re.sub(r"\s*[,;]\s*(and\s+)?", " and ", s)
    s = re.sub(r"\s+", " ", s).strip(" .")
    return s[:110]


def read_scene(reply, scene, paras, here=False, checks=3):
    """A scene's picks: [(paragraph index, kind, sound, extra)], kind amb (extra None), music (extra = last paragraph
    index) or sfx (extra = (sentence index, word, options)).
    checks (the Director tab's slider) says whether an effect has to fit its sentence (see CHECKS); here (the small
    local model): left/right are left out, as it sprinkles them on everything."""
    d = _json(reply)
    if not isinstance(d, dict):
        return []
    picks = []
    match = CHECKS[_level(checks)][0]

    def para(v):
        if v is None:
            return None
        try:
            n = int(float(str(v).strip("[]() "))) - 1
        except (TypeError, ValueError):
            m_num = re.search(r"\d+", str(v))
            if not m_num:
                return None
            n = int(m_num.group()) - 1
        return scene[n] if 0 <= n < len(scene) else None

    for b in d.get("backgrounds") or []:
        if not isinstance(b, dict):
            continue
        i, s = para(b.get("paragraph")), _desc(b.get("sound"))
        if i is not None and s:
            picks.append((i, "amb", "off" if s.lower() in ("off", "none", "silence", "stop") else s, None))
    for m in d.get("music") or []:
        if not isinstance(m, dict):
            continue
        raw_to = m.get("to_paragraph")
        i, j, s = para(m.get("from_paragraph")), para(raw_to), _desc(m.get("sound"))
        if i is not None and s and s.lower() not in ("off", "none", "silence", "stop"):
            end_i = scene[-1] if raw_to is None else (j if j is not None and j >= i else i)
            picks.append((i, "music", s, end_i))
    for e in d.get("effects") or []:
        if not isinstance(e, dict):
            continue
        i, s = para(e.get("paragraph")), _desc(e.get("sound"))
        if i is None or not s or SPEECH.search(s):
            continue
        sents = _sentences(paras[i][1])
        k = _find_sentence(sents, str(e.get("sentence_end") or ""), str(e.get("word") or ""))
        if k is None:
            continue
        word = _anchor(str(e.get("word") or "").strip(), sents[k])
        if not word:
            continue  # the word has to be in the sentence: it's where the sound plays
        if match and not _fits(s, sents[k], match):
            continue
        raw_opts = e.get("options") or []
        if isinstance(raw_opts, str):
            raw_opts = [raw_opts]
        opts = [o for o in (str(x).strip().lower() for x in raw_opts)
                if o in ("far", "muffled", "quiet", "loud") or (not here and o in ("left", "right"))]
        picks.append((i, "sfx", s, (k, word, opts)))
    def opening(v):
        v = str(v or "").strip().lower()
        v = OLD_OPENINGS.get(v, v)
        return v if v in OPENINGS else None

    def ending(v):
        v = str(v or "").strip().lower()
        return v if v in ENDINGS else None
    if opening(d.get("opening")):  # how the scene opens: its chapter's intro
        picks.append((scene[0], "opening", opening(d["opening"]), None))
    if ending(d.get("ending")):    # how it ends: its chapter's outro, when a chapter ends there
        picks.append((scene[-1], "ending", ending(d["ending"]), None))
    for c in d.get("chapter_openings") or []:  # chapters that start inside a big scene
        if not isinstance(c, dict):
            continue
        i, o = para(c.get("paragraph")), opening(c.get("opening"))
        if i is not None and i != scene[0] and o:
            picks.append((i, "opening", o, None))
    for c in d.get("chapter_endings") or []:   # and the ones that end inside it
        if not isinstance(c, dict):
            continue
        i, o = para(c.get("paragraph")), ending(c.get("ending"))
        if i is not None and o:
            picks.append((i, "ending", o, None))
    return picks


# what makes a sentence one an effect can play in, for the "fits" check: sounds, and things that make them
_SOUNDY = """roar hiss crack crash bang slam thud thump knock rattle creak clatter clank clang clink jingle ring rang rung
bell chime toll buzz hum whistle howl bark growl snarl whinny neigh chirp caw squawk screech shriek scream splash drip
pour drench soak gush spray rumble thunder boom blast explode explosion shot gunshot gunfire fire crackle sizzle pop snap
crunch click tick tap beep honk engine motor sputter cough wheeze gasp footstep step stomp stamp kick punch slap hit
struck strike smash shatter break broke tear tore rip scrape scratch rustle whoosh swish whir whirr flutter flap squeak
squeal groan moan clap wave surf wind gust rain storm hammer drum bubble gurgle slosh swing slide slid thunk zip snore
splinter collapse fall fell drop clunk chug purr hoot cry crow quack bleat moo clop gallop hoof hooves trot bray blare
siren alarm horn trumpet bugle gong whip lash pound batter rap walk run running ran stumble trip slip land shut open
gull bird owl dog hound wolf horse cat cow sheep goat rooster cricket bee fly crowd sea ocean water river stream waterfall
clock door gate window glass car truck train ship boat ferry plane helicopter phone gun rifle pistol sword blade chain
key lock bottle plate cup pot pan kettle fireplace hearth flame match candle lantern wick typewriter keyboard rope wood
timber branch leaf leaves gravel snow ice mud puddle coin paper page bag zipper boot shoe heel""".split()


def _forms(w):
    out = {w, w + "s", w + "ed", w + "ing", w + w[-1] + "ed", w + w[-1] + "ing"}
    if w.endswith("e"):
        out |= {w + "d", w[:-1] + "ing"}
    return out


SOUND_STEMS = {_stem(f) for w in _SOUNDY for f in _forms(w)}


def _fits(sound, sentence, how):
    """Whether an effect fits its sentence: "names", it shares a word with it (a seagull's cry in "gulls wheeled"
    counts); "fits", that, or the sentence has a sound or a thing making one in it."""
    a, b = _stems(sound), _stems(sentence)
    words = lambda t: {w[:-1] if w.endswith("s") else w for w in re.findall(r"[a-z]+", t.lower()) if len(w) > 4}  # noqa: E731
    if a & b or any(x.endswith(y) or y.endswith(x) for x in words(sound) for y in words(sentence)):
        return True
    return how == "fits" and bool(b & SOUND_STEMS)


def _norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower().replace("’", "'")).strip()


def _find_sentence(sents, end, word):
    """The sentence the model meant: the one ending with its words, else holding them, else holding its word."""
    end_n = _norm(end)
    if end_n:
        for k, s in enumerate(sents):
            if _norm(s).endswith(end_n):
                return k
        for k, s in enumerate(sents):
            if end_n in _norm(s):
                return k
    w = _norm(word)
    hits = [k for k, s in enumerate(sents) if w and w in _norm(s).split()]
    return hits[0] if len(hits) == 1 else None


# ---------- the checks, and the review list ----------

# The checks, from the Director tab's slider (1 loose … 5 strict): whether an effect has to fit its sentence (None:
# any sound the AI picks; "fits": it shares a word with the sentence, or the sentence has a sound in it; "names": it
# shares a word), how many sentences apart effects are at least (counted across paragraphs), and how many effects
# one paragraph gets
CHECKS = {1: (None, 1, 99), 2: (None, 2, 99), 3: ("fits", 2, 2), 4: ("names", 2, 1), 5: ("names", 3, 1)}
MUSIC_JOIN = 5  # the same music starting again within this many paragraphs of ending just carries on


def _level(checks=None):
    """The checks level: the one asked for, else the Director tab's slider (3 when it was never moved)."""
    if checks is None:
        checks = settings.load().get("director_checks", 3)
    try:
        return min(5, max(1, int(checks)))
    except (TypeError, ValueError):
        return 3


def assemble(parts, paras, picks, checks=3):
    """Check the picks and turn them into the review list's placements."""
    sents_of = {i: _sentences(p) for i, (_, p) in enumerate(paras)}
    start, n = {}, 0
    for i in range(len(paras)):
        start[i], n = n, n + len(sents_of[i])
    # backgrounds and music: walk the book in order; a background already playing isn't set again, and every piece
    # of music ends ({music: off} after its last paragraph, unless the next cue starts right there)
    amb_at, music_at = {}, {}
    for i, kind, s, extra in sorted((p for p in picks if p[1] == "amb"), key=lambda p: p[0]):
        amb_at[i] = s  # the scene's own pick wins over an earlier one for the same paragraph
    cues = []
    for i, s, j in sorted(((i, s, j) for i, kind, s, j in picks if kind == "music"), key=lambda c: c[0]):
        if cues and cues[-1][1] == s and i <= cues[-1][2] + MUSIC_JOIN:
            cues[-1] = (cues[-1][0], s, max(j, cues[-1][2]))  # the same music again soon after: it keeps playing
        else:
            cues.append((i, s, j))
    accepted = []
    for i, s, j in cues:
        if any(a <= i <= b for a, _, b in accepted):
            continue  # inside an accepted cue already playing
        accepted.append((i, s, j))
        music_at[i] = s
    for i, s, j in accepted:
        end = j + 1
        if end < len(paras) and end not in music_at:
            music_at[end] = "off"
    placements, playing_amb, playing_music = [], None, None
    for i in range(len(paras)):
        k = paras[i][0]
        if parts[k].lstrip().startswith("## "):
            playing_amb, playing_music = None, None
        have = sfx.split_cues(" ".join(m.group() for m in sfx.CUE.finditer(parts[k])))
        has_amb = False
        has_music = False
        for t, c in have:
            if t != "cue":
                continue
            if c["type"] == "amb":
                has_amb = True
                playing_amb = None if c.get("name") == "off" else c.get("name")
            elif c["type"] == "music":
                has_music = True
                playing_music = None if c.get("name") == "off" else c.get("name")
        if i in amb_at and not has_amb:
            s = amb_at[i]
            if s != playing_amb and not (s == "off" and playing_amb is None):
                placements.append(_placement(k, i, "amb", s))
                playing_amb = None if s == "off" else s
        if i in music_at and not has_music:
            s = music_at[i]
            if s != playing_music and not (s == "off" and playing_music is None):
                placements.append(_placement(k, i, "music", s))
                playing_music = None if s == "off" else s
    # effects: one per sentence, a few sentences apart, a few to a paragraph (the earlier one stays)
    _, gap, per_para = CHECKS[_level(checks)]
    effects, last, in_para = [], None, {}
    for i, kind, s, (k, word, opts) in sorted((p for p in picks if p[1] == "sfx"), key=lambda p: (p[0], p[3][0])):
        at = start[i] + k
        if (last is not None and (at == last or at - last < gap)) or in_para.get(i, 0) >= per_para:
            continue
        effects.append((i, s, k, word, opts))
        last, in_para[i] = at, in_para.get(i, 0) + 1
    for i, s, k, word, opts in effects:
        if any(m.group(1).lower() == "sfx" for m in sfx.CUE.finditer(parts[paras[i][0]])):
            continue  # the paragraph has its effects already (yours, or the director's from an earlier run)
        placements.append(_placement(paras[i][0], i, "sfx", s, sents_of[i][k], k, word, opts))
    placements.sort(key=lambda p: (p["part"], p["s"], {"amb": 0, "music": 1, "sfx": 2}[p["kind"]]))
    for n_, p in enumerate(placements):
        p["id"] = n_
    return placements


MERGE_SAME = 0.85  # CLAP text likeness above which two descriptions are one sound ("distant dog barks")
MERGE_NEAR = 0.65  # …or above this when most of their main words are the same ("dog barking far off", "distant dog barking")
MERGE_LIB = 0.92   # a library sound standing in for a description: only when it's nearly the same
LIB_KIND = {"sfx": "effect", "amb": "ambience", "music": "music"}
_FILLER = set("""against being through over under onto across around behind inside outside near while after before
during each every other another there here some""".split())


def _merge(picks, listed=()):
    """One description for each sound the book uses. Picks of one kind that mean the same sound take one
    description: one the bible listed (listed), or used most, so each is made once and sounds the same every time.
    A library sound is taken only when it's nearly the same words (a near miss there plays the wrong sound).
    Returns (picks, how many descriptions were folded in)."""
    listed = set(listed)
    out_name, folded = {}, 0
    for kind in ("sfx", "amb", "music"):
        used = {}
        for n, (_, k, s, _) in enumerate(picks):
            if k == kind and s != "off":
                c, first = used.get(s, (0, n))
                used[s] = (c + 1, first)
        if not used:
            continue
        lib = list(dict.fromkeys(e.get("description") or e["name"] for e in sfx.pool() if e.get("kind") == LIB_KIND[kind]))
        order = sorted(used, key=lambda s: (not sfx._exact(s, sfx.pool()), s not in listed, -used[s][0], used[s][1]))
        texts = list(dict.fromkeys(order + lib))
        emb = _embed(texts)
        idx = {t: n for n, t in enumerate(texts)}
        sim = (lambda a, b: float(emb[idx[a]] @ emb[idx[b]])) if emb is not None else (lambda a, b: MERGE_NEAR)  # noqa: E731
        canon = []
        for s in order:
            if sfx._exact(s, sfx.pool()) or s in listed:
                canon.append(s)
                continue
            near = [(sim(s, c), c) for c in canon if _alike(s, c, sim(s, c), c in listed)]
            near += [(sim(s, t), t) for t in lib if t not in canon and emb is not None and sim(s, t) >= MERGE_LIB]
            best = max(near, default=None)
            if best:
                out_name[(kind, s)] = best[1]
                folded += 1
            else:
                canon.append(s)
    picks = [(i, k, out_name.get((k, s), s), x) for i, k, s, x in picks]
    return picks, folded


class _Listed(list):
    """The bible's descriptions, and its theme (the chapter intros' music)."""
    theme = ""


def _chapter_plan(text, parts, paras, scenes, picks, theme):
    """The director's chapter music (chapter_music.plan): each chapter Create makes (chapter_music.titles), how it
    opens (from the scene it starts in) and how it ends (from the scene, or the paragraph, its last words are in)."""
    from . import chapter_music
    opening = {i: s for i, k, s, _ in picks if k == "opening"}
    ending = {i: s for i, k, s, _ in picks if k == "ending"}
    open_of = {i: opening.get(i) or opening.get(sc[0]) for sc in scenes for i in sc}  # (a chapter inside a big
    end_of = {i: ending.get(i) or ending.get(sc[-1]) for sc in scenes for i in sc}     # scene has its own)
    text = text or ""
    starts, pos = [], 0
    for part in parts:  # where each part starts in the text
        starts.append(pos)
        pos += len(part)
    heads = [(m.start(), m.group(1)) for m in re.finditer(r"(?m)^\s*##\s+(.+?)\s*$", text)]
    spans = [(0, None)] + heads
    names = chapter_music.titles(text)
    rows = []
    for n, (a, title) in enumerate(spans):
        b = spans[n + 1][0] if n + 1 < len(spans) else len(text)
        inside = [k for k, (pk, _) in enumerate(paras) if a <= starts[pk] < b]
        name = title or "Opening"
        if not inside and name not in names:
            continue
        rows.append({"title": name, "mood": open_of.get(inside[0]) if inside else None,
                     "ending": end_of.get(inside[-1]) if inside else None})
    rows = [r for r in rows if r["title"] in names] or rows
    return chapter_music.plan(theme, rows)


def _listed(bible):
    """The bible's descriptions, as the scenes' picks come out once read (_desc)."""
    out = []
    for key_, field in (("places", "background"), ("sounds", "sound"), ("music", "music")):
        for x in bible.get(key_) or []:
            if isinstance(x, dict) and x.get(field):
                out.append(_desc(x[field]))
    return out


def _alike(a, b, likeness, b_listed=False):
    """Whether description a is the same sound as b. b_listed: b is the bible's, which the scenes were told to copy,
    so a near copy (three main words the same) is taken as it even when CLAP is less sure."""
    if likeness >= MERGE_SAME:
        return True
    sa, sb = _stems(a) - _FILLER, _stems(b) - _FILLER
    shared = sa & sb
    if b_listed and likeness >= 0.45 and len(shared) >= 3 and len(shared) >= 0.6 * min(len(sa), len(sb)):
        return True
    return likeness >= MERGE_NEAR and len(shared) >= 2 and len(shared) >= 0.6 * min(len(sa), len(sb))


def _embed(texts):
    """CLAP text vectors (unit length), or None when the sound-matching model isn't here (words alone decide then)."""
    try:
        from . import clap
        if not (clap.available() and clap.downloaded()):
            return None
        return clap.text_embedding(texts)
    except Exception as e:  # noqa: BLE001 - merging by words still works
        print(f"[director] sound matching unavailable: {e}", flush=True)
        return None


MATCH_MIN = 0.33  # CLAP text-to-sound likeness a library sound needs to be offered in place of making one


def _close_matches(placements, lib):
    """For each sound the library doesn't have: the library sound of the same kind that sounds most like its
    description (by listening, CLAP), when it's close enough, as "match" for the review list. Only offered: the
    sound is made unless you pick the match. Nothing when the sound-matching model isn't here."""
    todo = [p for p in placements if p["missing"]]
    if not todo:
        return
    try:
        from . import clap
        if not (clap.available() and clap.downloaded()):
            return
        texts = sorted({p["name"] for p in todo})
        T = dict(zip(texts, clap.text_embedding([f"the sound of {t}" for t in texts])))
        by_kind = {}
        for p in todo:
            kind = LIB_KIND[p["kind"]]
            if kind not in by_kind:
                pool_ = [e for e in lib if e.get("kind") == kind]
                by_kind[kind] = (pool_, np.stack([sfx._emb(e) for e in pool_]) if pool_ else None)
            pool_, A = by_kind[kind]
            if A is None:
                continue
            sims = A @ T[p["name"]]
            k = int(np.argmax(sims))
            if sims[k] >= MATCH_MIN:
                e = pool_[k]
                tag = sfx.tag_for(e, lib)
                p["match"] = {"id": e["id"], "name": e["name"], "tag_name": tag[tag.index(":") + 1:-1].strip(),
                              "path": str(sfx.LIB / e["file"]), "score": round(float(sims[k]), 2)}
    except Exception as e:  # noqa: BLE001 - matches are a bonus; the sounds are still made
        print(f"[director] couldn't look for close matches: {e}", flush=True)


def _placement(part, i, kind, name, sentence="", s=-1, anchor="", opts=()):
    vol = next((o for o in opts if o in ("quiet", "loud")), "normal")
    e = None if name == "off" else sfx.find_for({"type": kind, "name": name})
    return {"part": part, "para": i + 1, "s": s, "sentence": sentence, "kind": kind, "name": name, "anchor": anchor,
            "volume": vol, "opts": [o for o in opts if o not in ("quiet", "loud")],
            "sound_id": e["id"] if e else None, "missing": name != "off" and not e}


# ---------- asking the AI ----------

class Declined(Exception):
    """The service declined a scene (a safety filter)."""


class TruncatedReply(ValueError):
    """The model hit max_tokens or returned empty content (often due to invisible reasoning tokens)."""

    def __init__(self, msg, out_tokens=0, reasoning_tokens=0, finish_reason="length"):
        super().__init__(msg)
        self.out_tokens, self.reasoning_tokens, self.finish_reason = out_tokens, reasoning_tokens, finish_reason


class LimitReached(ValueError):
    """The service's rate or daily limit (a free tier's, usually), after the library's own retries. daily: the day's
    allowance is used up; else a per-minute limit, over in `wait` seconds."""

    def __init__(self, msg, daily=False, wait=None):
        super().__init__(msg)
        self.daily, self.wait = daily, wait


def _limit(pid, e):
    """LimitReached for a 429: is it the day's limit, and how long the service says to wait."""
    raw = f"{e} {json.dumps(getattr(e, 'body', None), default=str)}"
    daily = bool(re.search(r"per.?day|daily|RPD\b|PerDay", raw, re.I))
    wait = None
    m = re.search(r"(?:retry(?:Delay)?|try again)[\"':\s]*(?:in\s*)?([\d.]+)\s*(ms|s)\b", raw, re.I)
    if m:
        wait = float(m.group(1)) / (1000 if m.group(2).lower() == "ms" else 1)
    try:
        wait = wait or float(getattr(getattr(e, "response", None), "headers", {}).get("retry-after"))
    except (TypeError, ValueError):
        pass
    name = PROVIDERS[pid]["name"]
    if daily or (wait or 0) > 600:  # (a wait of hours is the day's limit too)
        when = ""
        if wait and wait > 600:
            h, m = divmod(int(wait) // 60, 60)
            when = f" It resets in about {h} h {m} min." if h else f" It resets in about {m} min."
        return LimitReached(f"{name} reached its daily limit (a free tier allows only so many requests a day).{when}",
                            daily=True)
    return LimitReached(f"{name} hit its per-minute limit.", wait=wait)


def _error(pid, e, model=""):
    """The error to raise for a failed request: LimitReached for a 429, else ValueError, both in words. A model that
    can't be used at all is left out of the list from now on."""
    why = _unusable(e) if model and pid != "anthropic" else ""
    if why:
        _forget_model(pid, model, why)
        if getattr(e, "status_code", None) == 429:
            return ValueError(f"“{model}” isn't in {PROVIDERS[pid]['name']}'s free tier. Pick another model.")
        return ValueError(f"“{model}” can't be used by the director ({PROVIDERS[pid]['name']}: {why}). Pick another model.")
    if getattr(e, "status_code", None) == 429 or re.search(r"\b429\b|rate.?limit|quota", str(e), re.I):
        return _limit(pid, e)
    return ValueError(_friendly(pid, e))


def _price(model):
    return next((m["price"] for m in CLAUDE if m["id"] == model), None)


_CLAUDE_LOCK = threading.Lock()
_CLAUDE_THINKING = {}  # model -> index into thinking/effort ladder that succeeded
_CLAUDE_LADDER = [
    ({"type": "disabled"}, "low"),        # Haiku 5.5, Sonnet 5, Opus 5, Claude 4.6-4.8
    ({"type": "between_tools"}, "low"),   # Sonnet 5.5 (rejects disabled)
    (None, "low"),                       # Opus 5.5 / Fable / Mythos (mandatory thinking)
    (None, None),                        # Older models without output_config.effort
]


def _claude_start_idx(model):
    with _CLAUDE_LOCK:
        if model in _CLAUDE_THINKING:
            return _CLAUDE_THINKING[model]
    m = (model or "").lower()
    if any(k in m for k in ("opus-5-5", "fable", "mythos")):
        return 2
    if "sonnet-5-5" in m:
        return 1
    if "-4-5" in m:
        return 3
    return 0


def _ask_claude(model, system, user, schema, max_tokens, use_cache=True):
    """(reply text, dollars). Streamed (a whole book in the bible pass takes a while), JSON shaped by the schema,
    thinking disabled or minimized to avoid hidden reasoning token burn, system prompt cached when reused."""
    import anthropic
    start = _claude_start_idx(model)
    msg = None
    last = None
    for idx in range(start, len(_CLAUDE_LADDER)):
        thinking_cfg, effort = _CLAUDE_LADDER[idx]
        out_cfg = {"format": {"type": "json_schema", "schema": schema}}
        if effort:
            out_cfg["effort"] = effort
        kw = {"model": model, "max_tokens": max_tokens,
              "messages": [{"role": "user", "content": user}],
              "output_config": out_cfg}
        if system:
            sys_block = {"type": "text", "text": system}
            if use_cache:
                sys_block["cache_control"] = {"type": "ephemeral"}
            kw["system"] = [sys_block]
        if thinking_cfg is not None:
            kw["thinking"] = thinking_cfg
        betas = []
        if model in FALLBACKS:
            betas.append("server-side-fallback-2026-07-01")
            kw["fallbacks"] = "default"
        try:
            client = _anthropic_client()
            stream_ctx = client.beta.messages.stream(**kw, betas=betas) if betas else client.messages.stream(**kw)
            with stream_ctx as s:
                msg = s.get_final_message()
            with _CLAUDE_LOCK:
                _CLAUDE_THINKING[model] = idx
            break
        except anthropic.BadRequestError as e:
            last = e
            detail = _detail(e, 500)
            if idx + 1 < len(_CLAUDE_LADDER) and re.search(r"thinking|between_tools|disabled|output_config\.effort|\beffort\b", detail, re.I):
                continue
            raise _error("anthropic", e) from e
        except Exception as e:  # noqa: BLE001 - shown to the user
            raise _error("anthropic", e) from e
    else:
        raise _error("anthropic", last)
    u = msg.usage
    actual_model = getattr(msg, "model", None) or model
    p_in, p_out, p_read = _price(actual_model) or _price(model) or (0, 0, 0)
    t_in = (u.input_tokens or 0) if u else 0
    t_cw = (getattr(u, "cache_creation_input_tokens", 0) or 0) if u else 0
    t_cr = (getattr(u, "cache_read_input_tokens", 0) or 0) if u else 0
    t_out = (u.output_tokens or 0) if u else 0
    mult = 5.0 if ("haiku-5-5" in actual_model and (t_in + t_cw + t_cr) > 100_000) else 1.0
    dollars = (t_in * p_in + t_cw * p_in * 1.25 + t_cr * p_read + t_out * p_out) * mult / 1e6
    _charge("anthropic", model, dollars, u, exact=bool(_price(actual_model) or _price(model)), user=user)
    if msg.stop_reason == "refusal":
        raise Declined()
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    if msg.stop_reason in ("max_tokens", "model_context_window_exceeded") or not text.strip():
        raise TruncatedReply(
            f"Claude ({model}) ran out of output tokens ({t_out}) before finishing the JSON plan.",
            out_tokens=t_out, finish_reason=msg.stop_reason or "empty",
        )
    return text, dollars


_FORMAT = {}     # (service, model) -> the reply format it accepted (0 schema, 1 JSON mode, 2 plain)
_REASONING = {}  # (service, model) -> index into _reasoning_ladder(pid) that succeeded
_REASONING_RE = re.compile(r"reasoning|thinking|effort|mandatory|extra_body|budget|thought", re.I)
_FORMAT_RE = re.compile(r"response_format|json_schema|structured|json_object", re.I)
_GENERIC_PARAM_RE = re.compile(r"unrecognized|unknown|unsupported", re.I)
_UNRELATED_PARAM_RE = re.compile(r"max_tokens|max_completion_tokens|cache_control|messages|model\b", re.I)


def _reasoning_ladder(pid):
    """Ordered reasoning-suppression options per provider (disable first, then minimal/low, then omit)."""
    if pid == "openrouter":
        return [
            {"extra_body": {"reasoning": {"effort": "none"}}},
            {"extra_body": {"reasoning": {"effort": "minimal", "exclude": True}}},
            {"extra_body": {"reasoning": {"effort": "low", "exclude": True}}},
            {},
        ]
    if pid == "deepseek":
        return [{"extra_body": {"thinking": {"type": "disabled"}}}, {}]
    if pid in ("openai", "gemini"):
        return [{"reasoning_effort": "none"}, {"reasoning_effort": "minimal"}, {"reasoning_effort": "low"}, {}]
    return [{}]


def _usage_reasoning(usage):
    if not usage:
        return 0
    cd = getattr(usage, "completion_tokens_details", None)
    if isinstance(cd, dict):
        return int(cd.get("reasoning_tokens") or 0)
    return int(getattr(cd, "reasoning_tokens", 0) or 0)


def _ask_openai(pid, model, system, user, schema, prices=None, max_tokens=6144, session_id=None, use_cache=True):
    """(reply text, dollars or None). Passes explicit max_tokens, disables/minimizes reasoning tokens with a
    cached 400-fallback ladder, steps down response_format on 400s, records spend, and rejects truncated/empty JSON."""
    from openai import BadRequestError
    client = _openai_client(pid)
    # Claude through OpenRouter caches only what's marked: attach cache_control only when reused across scenes
    cached = use_cache and pid == "openrouter" and model.lstrip("~").startswith("anthropic/")
    msgs = [{"role": "system", "content": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
             if cached else system},
            {"role": "user", "content": user + "\n\nReply with only the JSON object."}]
    last = None
    formats = [{"type": "json_schema", "json_schema": {"name": "plan", "schema": schema, "strict": True}},
               {"type": "json_object"}, None]
    rladder = _reasoning_ladder(pid)
    fmt_idx = _FORMAT.get((pid, model), 0)
    r_idx = min(_REASONING.get((pid, model), 0), len(rladder) - 1)
    token_kw = {"max_completion_tokens": max_tokens} if pid == "openai" else {"max_tokens": max_tokens}
    r = None
    for _ in range(len(formats) + len(rladder)):
        fmt = formats[fmt_idx]
        rcfg = rladder[r_idx]
        extra = {}
        body = {}
        if pid == "openrouter":
            body["usage"] = {"include": True}
            if use_cache and session_id:
                body["session_id"] = session_id
        if "extra_body" in rcfg:
            body.update(rcfg["extra_body"])
        if body:
            extra["extra_body"] = body
        if "reasoning_effort" in rcfg:
            extra["reasoning_effort"] = rcfg["reasoning_effort"]
        try:
            r = client.chat.completions.create(
                model=model, messages=msgs, **token_kw,
                **({"response_format": fmt} if fmt else {}), **extra,
            )
            _FORMAT[(pid, model)] = fmt_idx
            _REASONING[(pid, model)] = r_idx
            break
        except BadRequestError as e:
            last = e
            msg = _detail(e, 500)
            if r_idx + 1 < len(rladder) and (
                _REASONING_RE.search(msg)
                or (rcfg and _GENERIC_PARAM_RE.search(msg) and not _UNRELATED_PARAM_RE.search(msg))
            ):
                if re.search(r"(unrecognized|unsupported\s+parameter|unknown).*reasoning_effort|reasoning_effort.*not\s+supported", msg, re.I):
                    r_idx = len(rladder) - 1
                else:
                    r_idx += 1
                continue
            if fmt_idx + 1 < len(formats) and _FORMAT_RE.search(msg):
                fmt_idx += 1
                continue
            raise _error(pid, e, model) from e
        except Exception as e:  # noqa: BLE001 - shown to the user
            raise _error(pid, e, model) from e
    else:
        raise _error(pid, last, model)
    choices = getattr(r, "choices", None) or []
    if not choices:
        err_obj = getattr(r, "error", None) or "The AI service returned an empty response."
        raise _error(pid, ValueError(str(err_obj)), model)
    choice = choices[0]
    dollars = None
    u = r.usage
    billed = (getattr(u, "model_extra", None) or {}).get("cost") if u else None
    if billed is not None:  # OpenRouter's own figure
        dollars = float(billed)
    elif prices and u:
        dollars = (u.prompt_tokens or 0) * prices[0] + (u.completion_tokens or 0) * prices[1]
    _charge(pid, model, dollars, u, exact=billed is not None, user=user)
    finish = getattr(choice, "finish_reason", "") or ""
    if finish == "content_filter":
        raise Declined()
    text = choice.message.content or ""
    if finish == "length" or not text.strip():
        tout = (getattr(u, "completion_tokens", None) or 0) if u else 0
        treason = _usage_reasoning(u)
        detail = f"out={tout}, reasoning={treason}" if treason else f"out={tout}"
        raise TruncatedReply(
            f"{PROVIDERS[pid]['name']} ({model}) ran out of output tokens ({detail}) before finishing the JSON plan. Try a faster or non-reasoning model.",
            out_tokens=tout, reasoning_tokens=treason, finish_reason=finish or "empty",
        )
    return text, dollars


def _charge(pid, model, dollars, usage, exact=False, user=""):
    """The ledger (spend.py): what a request cost. A service with no price given gets its list price (pricing.py),
    marked as an estimate; a free tier is noted at $0 with its tokens (until its limit, that's what it costs)."""
    from . import pricing, spend
    prompt_toks = getattr(usage, "prompt_tokens", None) if usage else None
    tcache_read = int(getattr(usage, "cache_read_input_tokens", None) or 0) if usage else 0
    tcache_write = int(getattr(usage, "cache_creation_input_tokens", None) or 0) if usage else 0
    if prompt_toks is not None:
        tin = int(prompt_toks)
        details = getattr(usage, "prompt_tokens_details", None)
        tcached = (details.get("cached_tokens") if isinstance(details, dict) else getattr(details, "cached_tokens", 0)) or 0
        tcached = min(max(0, int(tcached)), tin)
    else:
        tuncached = int(getattr(usage, "input_tokens", None) or 0) if usage else 0
        tin = tuncached + tcache_read + tcache_write
        tcached = tcache_read
    tout = int((getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None) or 0) if usage else 0)
    treason = _usage_reasoning(usage)
    if dollars is None:
        price, _ = pricing.model_price(pid, model)
        free = pid in pricing.FREE_TIER or model.endswith(":free")
        if free or not price:
            dollars = 0.0
        else:
            p_cached = price[2] if price[2] is not None else price[0]
            dollars = (tin - tcached) * price[0] + tcached * p_cached + tout * price[1]
    units = {"in": tin, "out": tout, "cached": tcached}
    if treason:
        units["reasoning"] = treason
    spend.record(pid, "ai", "direct", dollars, exact=exact, model=model, units=units,
                 fingerprint=hashlib.sha1(f"{model}\n{user}".encode()).hexdigest()[:16] if user else None)
    return dollars


def _openrouter_prices(model):
    """OpenRouter lists each model's price per token, so its cost can be shown too."""
    try:
        for m in _openai_client("openrouter").models.list():
            if m.id == model:
                p = (getattr(m, "model_extra", None) or {}).get("pricing") or {}
                return float(p.get("prompt", 0)), float(p.get("completion", 0))
    except Exception:  # noqa: BLE001 - no price shown, that's all
        pass
    return None


def _keep(where, model, kind, system, user, reply):
    PLANS.parent.mkdir(parents=True, exist_ok=True)
    with PLANS.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "where": where, "model": model, "kind": kind,
                            "system": system if kind == "bible" else None, "user": user, "reply": reply},
                           ensure_ascii=False) + "\n")


# ---------- estimate ----------

ESTIMATE_VERSION = 4  # (v4: reasoning off + 200k bible cap + mandatory-reasoning headroom)
_MANDATORY_REASONING = re.compile(r"(opus-5[.-]5|sonnet-5[.-]5|fable|mythos|\bo[134](?:-|$)|gpt-6|gemini-2\.5-pro|gemini-[34]|deepseek-r1|thinking)", re.I)


def estimate(text, where, model, size=None):
    """Before directing: how many scenes, and roughly what it costs or how long it takes."""
    parts, paras, scenes = layout(text, where, size)
    chars = sum(len(p) for _, p in paras)
    if not paras:
        return {"scenes": 0, "dollars": None, "minutes": 0}
    if where == MAC:
        return {"scenes": len(scenes), "dollars": 0.0, "low": 0.0, "high": 0.0, "cost": "free",
                "minutes": round(1 + len(scenes) * 0.75)}
    from . import pricing
    # With reasoning disabled, visible JSON replies are ~250-350 tokens/scene; mandatory-reasoning models get
    # extra output headroom so spend.py's runaway guard doesn't falsely abort legitimate runs.
    bible_in = min(chars, BIBLE_CHARS.get(where, DEFAULT_BIBLE_CHARS)) / 3.8 + 4000
    system = 3100
    scene_in = chars / 3.8 + len(scenes) * 250
    mandatory = bool(_MANDATORY_REASONING.search(model or ""))
    out_low = len(scenes) * (800 if mandatory else 300) + (2000 if mandatory else 800)
    out_high = len(scenes) * (2000 if mandatory else 600) + (4500 if mandatory else 2000)
    tokens = {"in": bible_in + system + scene_in, "in_repeat": max(0, len(scenes) - 1) * system,
              "out_low": out_low, "out_high": out_high}
    r = pricing.ai_range(where, model, tokens, "direct", ESTIMATE_VERSION)
    return {"scenes": len(scenes), "requests": len(scenes) + 1, "dollars": r["mid"], "low": r["low"], "high": r["high"],
            "mid": r["mid"], "cost": pricing.say(r), "basis": r["basis"], "free_tier": r["free_tier"], "v": r.get("v", ESTIMATE_VERSION),
            "model": model, "minutes": round(1 + len(scenes) / PARALLEL * 0.5),
            "tokens": int(tokens["in"] + tokens["in_repeat"] + tokens["out_high"])}


# ---------- directing a book ----------

def direct(text, where=MAC, model="", checks=None, guide="", book="", size=None):
    """✦ Direct sounds for a book (book: its id, so its own sounds and settings count)."""
    token = sfx.BOOK.set(book) if book else None
    try:
        return (yield from _direct(text, where, model, checks, guide, size))
    finally:
        if token is not None:
            try:
                sfx.BOOK.reset(token)
            except ValueError:  # closed from another context: nothing to undo here
                pass


def _direct(text, where=MAC, model="", checks=None, guide="", size=None):
    """✦ Direct sounds: events for a task, ending with {"placements", "planned_on", "status"} for the review list."""
    from . import tasks
    if not (text or "").strip():
        raise ValueError("Paste or import a story first.")
    if where != MAC and where not in PROVIDERS:
        raise ValueError("Pick where the director runs.")
    if where != MAC and not key(where):
        raise ValueError(f"Add a {PROVIDERS[where]['name']} key in Connections first.")
    if where == "anthropic" and not model:
        model = CLAUDE[0]["id"]
    if where not in (MAC, "anthropic") and not model:
        raise ValueError("Pick a model for the director.")
    checks = _level(checks)
    parts, paras, scenes = layout(text, where, size)
    if not paras:
        raise ValueError("Paste or import a story first.")
    listed = _Listed()  # the bible's descriptions (places, recurring sounds, music): never merged into another
    if where != MAC:  # within today's spending limit, and remembered for the runaway guard
        from . import spend
        spend.allow(where, estimate(text, where, model, size), "direct")
    meta = {"partial": False, "scenes_done": len(scenes), "scenes_total": len(scenes), "stop_reason": None}
    if where == MAC:
        picks, notes = yield from _direct_here(paras, scenes, checks, _brief(guide), listed)
        spent = None
    else:
        picks, notes, spent, meta = yield from _direct_service(where, model, paras, scenes, tasks, checks, _brief(guide), listed)
        if meta["partial"]:
            from . import spend
            spend.mark_partial()
    yield {"status": "Joining sounds that are the same, so each is made once…", "progress": 0.96}
    picks, folded = _merge(picks, listed)
    placements = assemble(parts, paras, picks, checks)
    lib = sfx.pool()
    for p in placements:
        e = next((x for x in lib if x["id"] == p["sound_id"]), None)
        near = e and not sfx._exact(p["name"], [e])  # a near miss would play: say which sound
        p["label"] = label_of(p) + (f"  ·  plays “{e['name']}”" if near else "")
        p["sound_path"] = str(sfx.LIB / e["file"]) if e else None
    yield {"status": "Listening to your library for close matches…", "progress": 0.98}
    _close_matches(placements, lib)
    missing = sorted({p["name"] for p in placements if p["missing"]})
    status = f"{len(placements)} sound(s) to review." if placements else "The director added nothing to this text."
    if spent:
        status += f" Cost about ${spent:.2f}."
    if missing:
        status += f" {len(missing)} aren't in your library yet (made when you create)."
    if folded:
        status += f" {folded} look-alike sound(s) were joined to one already planned or in your library."
    if notes:
        status += " " + " ".join(notes)
    yield {
        "placements": placements, "planned_on": text, "status": status, "progress": 1.0,
        "theme": listed.theme, "chapter_plan": _chapter_plan(text, parts, paras, scenes, picks, listed.theme),
        "partial": meta["partial"], "scenes_done": meta["scenes_done"],
        "scenes_total": meta["scenes_total"], "stop_reason": meta["stop_reason"],
    }


RESUME = HOME / "directors" / "resume"  # scenes already directed in a run that stopped partway (a daily limit)
RESUME_DAYS = 14


def _resume_file(where, model, brief, paras, scenes):
    """Where a run's finished scenes are kept: the same book text, service, model, notes and scenes pick them up."""
    key = hashlib.sha1(json.dumps([where, model, brief, [list(sc) for sc in scenes],
                                   [p for _, p in paras]], ensure_ascii=False).encode()).hexdigest()[:20]
    return RESUME / f"{key}.json"


def _read_resume(f):
    try:
        if time.time() - f.stat().st_mtime > RESUME_DAYS * 86400:
            f.unlink(missing_ok=True)
            return {}
        d = json.loads(f.read_text())
        if not isinstance(d, dict):
            return {}
        bible = d.get("bible")
        if bible is not None and not (isinstance(bible, str) and (bible.strip() == "{}" or bool(_json(bible)))):
            bible = None
        raw_scenes = d.get("scenes")
        scenes = ({int(k): v for k, v in raw_scenes.items() if isinstance(v, str) and bool(_json(v))}
                  if isinstance(raw_scenes, dict) else {})
        return {"bible": bible, "scenes": scenes}
    except (OSError, ValueError):
        return {}


def _write_resume(f, bible, done):
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps({"bible": bible, "scenes": {str(k): v for k, v in done.items()}}, ensure_ascii=False))
        tmp.replace(f)
    except OSError as e:
        print(f"[director] couldn't keep the scenes done so far: {e}", flush=True)


def _direct_service(where, model, paras, scenes, tasks, checks=3, brief=BRIEF, listed=None):
    """Bible, then every scene (several at once). Returns (picks, notes, dollars).
    A per-minute limit is waited out (fewer scenes at once from then on); a daily limit stops the run, and the
    scenes done so far are kept, so directing the same text again carries on from there."""
    spent, notes = 0.0, []
    prices = _openrouter_prices(model) if where == "openrouter" else None
    name = PROVIDERS[where]["name"]
    heads = getattr(scenes, "heads", ())
    resume = _resume_file(where, model, brief, paras, scenes)
    kept = _read_resume(resume)
    import random
    replies = dict(kept.get("scenes") or {})  # scene number -> reply
    pause_lock = threading.Lock()
    pause = {"until": 0.0, "slow": False, "clean": 0}
    cancel_ev = threading.Event()
    session_id = resume.stem  # sticky prompt-cache key for OpenRouter Anthropic routing

    def ask(system, user, schema, max_tokens, use_cache=True):
        if cancel_ev.is_set() or tasks.stop_wanted():
            raise tasks.Stopped()
        from . import conn_issues, spend
        spend.before_request(where)  # (the rate guard: a runaway loop is slowed down, then stopped)
        try:
            if where == "anthropic":
                r = _ask_claude(model, system, user, schema, max_tokens, use_cache=use_cache)
            else:
                r = _ask_openai(where, model, system, user, schema, prices, max_tokens=max_tokens,
                                session_id=session_id, use_cache=use_cache)
        except LimitReached as e:
            if e.daily:  # (a per-minute limit is waited out: not a problem with the connection)
                conn_issues.note(where, e, "Directing")
            raise
        except Exception as e:  # noqa: BLE001 - a refused key or a dropped connection shows on Connections
            conn_issues.note(where, e, "Directing")
            raise
        conn_issues.ok(where)
        with pause_lock:
            pause["clean"] += 1
            if pause["slow"] and pause["clean"] >= 4:
                pause["slow"] = False
        return r

    def ask_waiting(system, user, schema, max_tokens, use_cache=True):
        """ask(), waiting out per-minute limits with jittered wakeup (in a scene's thread)."""
        for attempt in range(LIMIT_WAITS + 1):
            while True:
                with pause_lock:
                    until = pause["until"]
                if time.time() >= until:
                    break
                if cancel_ev.is_set() or tasks.stop_wanted():
                    raise tasks.Stopped()
                time.sleep(0.5)
            if attempt > 0:
                time.sleep(random.uniform(0.2, 1.0))  # stagger wakeups to avoid thundering herd
            try:
                return ask(system, user, schema, max_tokens, use_cache=use_cache)
            except LimitReached as e:
                if e.daily or attempt == LIMIT_WAITS:
                    raise
                with pause_lock:
                    pause["slow"] = True
                    pause["clean"] = 0
                    pause["until"] = max(pause["until"], time.time() + min(max(e.wait or 30, 5), 120) + 1)

    def waiting(status):
        with pause_lock:
            left = pause["until"] - time.time()
        return f"Waiting {left:.0f}s for {name}'s per-minute limit · {status}" if left > 0 else status

    bible_reply = kept.get("bible")
    if bible_reply is None:
        book, sampled = _book_text(paras, BIBLE_CHARS.get(where, DEFAULT_BIBLE_CHARS))
        yield {"status": "The director is reading the book for its places and music…", "progress": 0.02}
        user = BIBLE_ASK.format(sampled=" (excerpts spread through it)" if sampled else "", book=book)
        pool = cf.ThreadPoolExecutor(1)
        try:
            f = pool.submit(contextvars.copy_context().run, ask_waiting, brief, user, BIBLE_SCHEMA, 8192, False)
            while True:
                try:
                    bible_reply, dollars = f.result(timeout=1)
                    break
                except cf.TimeoutError:
                    with pause_lock:
                        in_pause = pause["until"] > time.time()
                    if in_pause:
                        yield {"status": waiting("reading the book"), "progress": 0.02}
                    if tasks.stop_wanted():
                        cancel_ev.set()
                        raise tasks.Stopped()
                except Declined:
                    bible_reply, dollars = "{}", 0
                    notes.append("The service declined the book overview, so scenes were planned without it.")
                    break
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        spent += dollars or 0
        _keep(where, model, "bible", brief, user, bible_reply)
        _write_resume(resume, bible_reply, replies)
    bible = _json(bible_reply) or {"world": "", "places": [], "sounds": [], "music": []}
    if listed is not None:
        listed += _listed(bible)
        listed.theme = _desc(bible.get("theme"))
    system = _system(bible, brief)
    picks, declined, failed = [], 0, []
    for n, r in replies.items():
        if 0 <= n < len(scenes):
            picks += read_scene(r, scenes[n], paras, checks=checks)
    before = done = len(replies)
    if before:
        notes.append(f"Carried on from the last run: {before} of {len(scenes)} scenes were directed already.")

    def one(n):
        user = _scene_prompt(paras, scenes[n], heads=heads)
        r, d = ask_waiting(system, user, SCENE_SCHEMA, 6144, use_cache=len(scenes) > 1)
        _keep(where, model, "scene", None, user, r)
        return n, r, d
    todo = [n for n in range(len(scenes)) if n not in replies]
    truncated = 0
    consecutive_truncated = 0
    cost = lambda: f" · ${spent:.2f} so far" if where == "anthropic" or prices else ""  # noqa: E731
    prog = lambda: round(0.05 + 0.90 * ((done + declined + truncated) / max(len(scenes), 1)), 4)  # noqa: E731
    yield {"status": f"Directing scene {done + 1} of {len(scenes)}{cost()}", "progress": prog()}
    # the first scene alone writes the cached system prompt; the rest then read it at the cached price
    pool = cf.ThreadPoolExecutor(PARALLEL)
    try:
        running = {pool.submit(contextvars.copy_context().run, one, todo.pop(0))} if todo else set()
        while running:
            finished, running = cf.wait(running, timeout=0.5, return_when=cf.FIRST_COMPLETED)
            for f in finished:
                try:
                    n, r, d = f.result()
                except Declined:
                    declined += 1
                    continue
                except TruncatedReply as e:
                    consecutive_truncated += 1
                    truncated += 1
                    if done == 0 and not running:
                        cancel_ev.set()
                        raise
                    if consecutive_truncated >= 3 or done == before:
                        if not failed:
                            failed.append(str(e))
                        cancel_ev.set()
                        todo.clear()
                    continue
                except (cf.CancelledError, tasks.Stopped):
                    if failed:
                        continue
                    cancel_ev.set()
                    raise
                except KeyboardInterrupt:
                    cancel_ev.set()
                    raise
                except Exception as e:  # noqa: BLE001 - a daily limit or a dropped connection partway: keep what's done
                    if done == 0 and not running and not isinstance(e, LimitReached):
                        cancel_ev.set()
                        raise
                    if not failed:
                        failed.append(str(e))
                    cancel_ev.set()
                    todo.clear()  # nothing more is sent; the scenes already asked still come back
                    continue
                spent += d or 0
                if not _json(r):
                    consecutive_truncated += 1
                    truncated += 1
                    msg = f"{name} ({model}) returned invalid/empty JSON on {consecutive_truncated} consecutive scene(s)."
                    if done == 0 and not running:
                        cancel_ev.set()
                        raise TruncatedReply(msg)
                    if consecutive_truncated >= 3 or done == before:
                        if not failed:
                            failed.append(msg)
                        cancel_ev.set()
                        todo.clear()
                    continue
                consecutive_truncated = 0
                replies[n] = r
                _write_resume(resume, bible_reply, replies)
                picks += read_scene(r, scenes[n], paras, checks=checks)
                done += 1
            status = f"Directing scenes · {done + declined + truncated} of {len(scenes)} done{cost()}"
            with pause_lock:
                in_pause = pause["until"] > time.time()
                slow = pause["slow"]
            if finished or in_pause:
                yield {"status": waiting(status), "progress": prog()}
            if tasks.stop_wanted():
                cancel_ev.set()
                raise tasks.Stopped()
            if (done > before or not running) and not cancel_ev.is_set():  # stay at 1 until first scene warms cache
                limit = 1 if (slow or done == before) else PARALLEL
                while todo and len(running) < limit:
                    running.add(pool.submit(contextvars.copy_context().run, one, todo.pop(0)))
    finally:
        cancel_ev.set()
        pool.shutdown(wait=False, cancel_futures=True)  # stopped or failed: scenes not started never are
    if declined:
        notes.append(f"The service declined {declined} scene(s); they got no sounds.")
    if truncated and not failed:
        notes.append(f"The service returned truncated or invalid output on {truncated} scene(s); they got no sounds.")
    if failed:
        notes.append(f"{failed[0]} It stopped after {done} of {len(scenes)} scenes: these are the sounds for those. "
                     "Direct again later (tomorrow for a daily limit), or with a paid key: it carries on from there.")
    elif not declined and not truncated:
        resume.unlink(missing_ok=True)  # all done: directing again plans afresh
    meta = {
        "partial": bool(failed or truncated),
        "scenes_done": done,
        "scenes_total": len(scenes),
        "stop_reason": failed[0] if failed else None,
    }
    return picks, notes, (spent if where == "anthropic" or prices else None), meta


def _direct_here(paras, scenes, checks=3, brief=BRIEF, listed=None):
    """On this Mac with the local model: a short bible from the start of the book, then scene by scene."""
    picks, notes = [], []
    for ev in _local(paras, scenes, brief):
        if ev[0] == "status":
            yield {"status": ev[1], **({"progress": ev[2]} if len(ev) > 2 else {})}
        elif ev[0] == "bible":
            if listed is not None:
                listed += _listed(ev[1])
                listed.theme = _desc(ev[1].get("theme"))
        else:
            n, reply = ev[1], ev[2]
            picks += read_scene(reply, scenes[n], paras, here=True, checks=checks)
    return picks, notes


def _local_impl(paras, scenes, brief=BRIEF, stop=None):
    from . import engine as en
    from .ai_detect import LLM_MODEL
    en.release_models()
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler
    yield ("status", "Loading the AI model…")
    model, tok = load(LLM_MODEL)
    sampler = make_sampler(temp=0.0)

    def run(system, user, max_tokens):
        chat = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                       add_generation_prompt=True, tokenize=False)
        return generate(model, tok, prompt=chat, max_tokens=max_tokens, sampler=sampler, verbose=False)
    yield ("status", "The director is reading the start of the book…")
    book, sampled = _book_text(paras, BIBLE_CHARS[MAC])
    user = BIBLE_ASK.format(sampled=" (excerpts)", book=book) + (
        '\n\nReply with only one JSON object: {"world": "...", "places": [{"place": "...", "background": "..."}], '
        '"sounds": [{"thing": "...", "sound": "..."}], "music": [{"mood": "...", "music": "..."}], "theme": "..."}')
    reply = run(brief, user, 700)
    _keep(MAC, LLM_MODEL, "bible", brief, user, reply)
    bible = _json(reply) or {"world": "", "places": [], "sounds": [], "music": []}
    yield ("bible", bible)
    system = _system(bible, brief)
    for n, scene in enumerate(scenes):
        if stop is not None and stop.is_set():
            break
        yield ("status", f"Directing scene {n + 1} of {len(scenes)} on this Mac", round(0.05 + 0.90 * (n / max(len(scenes), 1)), 4))
        user = _scene_prompt(paras, scene, before=False) + "\n\n" + LOCAL_FORMAT
        reply = run(system, user, 1500)
        _keep(MAC, LLM_MODEL, "scene", None, user, reply)
        yield ("reply", n, reply)
    del model
    en.free()


def _local(paras, scenes, brief=BRIEF):
    from . import engine as en
    return en.streamed_on_mlx_thread(_local_impl)(paras, scenes, brief)
