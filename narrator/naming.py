"""Short, readable names for sounds ("A quick and short but powerful explosion with shattering glass" ->
"Explosion Shatter"), made by a tiny local language model (Qwen3 1.7B, 4-bit, 0.97 GB, Apache-2.0;
0.6B gave names like "Puller" for a pistol)
that's loaded for a moment and let go. A plain rule is the fallback."""
import re

from .runtime import free, on_mlx_thread

NAME_MODEL = "mlx-community/Qwen3-1.7B-4bit"
PROMPT = ("Name each sound effect for a sound library in 2 to 4 words, Title Case. Name the thing making the sound and "
          "what it does; keep one important detail. Drop filler like quick, short, the sound of. Reply with the name only."
          "\n\nSound: A heavy wooden door slamming shut\nName: Wooden Door Slam"
          "\nSound: The sound of rain falling softly on a tin roof at night\nName: Rain On Tin Roof"
          "\nSound: A man laughing loudly in a crowded room\nName: Man Laughing"
          "\n\nSound: {desc}\nName:")
_FILLER = set("a an the sound sounds of being some very quite really just".split())


def fallback(desc):
    """First few meaningful words, Title Case (used when the model isn't available or answers oddly)."""
    words = [w for w in re.findall(r"[A-Za-z0-9'’-]+", desc or "") if w.lower() not in _FILLER]
    return " ".join(w[:1].upper() + w[1:] for w in words[:4]) or "Sound"


def _clean(reply, desc):
    reply = re.sub(r"<think>.*?</think>", "", reply or "", flags=re.S).strip().splitlines()
    name = re.sub(r"[\"“”*_#:]|^name\s*", "", reply[0] if reply else "", flags=re.I).strip(" .-")
    words = name.split()
    if not 1 <= len(words) <= 6 or len(name) > 40 or not re.search(r"[A-Za-z]", name):
        return fallback(desc)
    return " ".join(w[:1].upper() + w[1:].lower() if w.isupper() or w[1:2].isupper() else w[:1].upper() + w[1:]
                    for w in words)


@on_mlx_thread
def short_names(descriptions):
    """[description] -> [short name], one model load for all of them."""
    try:
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler
        model, tok = load(NAME_MODEL)
    except Exception:  # noqa: BLE001 - offline or not downloaded: plain names
        return [fallback(d) for d in descriptions]
    out = []
    try:
        for d in descriptions:
            msgs = [{"role": "user", "content": PROMPT.format(desc=d.strip())}]
            try:
                chat = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
            except TypeError:
                chat = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            out.append(_clean(generate(model, tok, prompt=chat, max_tokens=16, sampler=make_sampler(temp=0.0),
                                       verbose=False), d))
    finally:
        del model
        free()
    return out


_NAME_CACHE = {}


def short_name(description):
    key = (description or "").strip()
    if key in _NAME_CACHE:
        return _NAME_CACHE[key]
    res = short_names([description])[0]
    if len(_NAME_CACHE) > 512:
        _NAME_CACHE.clear()
    _NAME_CACHE[key] = res
    return res
