"""AI speaker and emotion detection with a small local language model (Qwen3 4B, MLX).

Reads the story in passages so each line of dialogue has its surrounding narration,
and labels who speaks, their gender and their tone. Character names are carried across
passages so the same person keeps the same name for the whole book.
"""
import re

from . import engine as en
from .book import split_chapters
from . import detect_core as core
from .cast import strip_tags
from .detect_core import QUOTE  # noqa: F401 - others read it from here

LLM_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
CLOUD_MODEL = "Qwen/Qwen3-4B-Instruct-2507"  # the same model, full size, on a cloud GPU
TAG = re.compile(r"\[[^\[\]]+\]\s*")


def chapters_of(text):
    """[(title, [paragraphs without speaker tags])], and every paragraph in order (what the model reads)."""
    chapters = []
    for title, body in split_chapters(text):
        paras = []
        for raw in re.split(r"\n\s*\n", body):
            clean = strip_tags(raw).strip()
            if clean:
                paras.extend(core.split_multi_speaker(clean))
        chapters.append((title, paras))
    return chapters, [p for _, paras in chapters for p in paras]


def starts_of(chapters):
    """The paragraph indexes (in chapters_of's flat list) that open a chapter."""
    out, n = [], 0
    for _, paras in chapters:
        out.append(n)
        n += len(paras)
    return out


def assemble(chapters, labels, genders, known):
    """The story with [Speaker: emotion] tags in front of the labelled paragraphs, and {character: gender}."""
    out, flat_i = [], 0
    for title, paras in chapters:
        block = [f"## {title}"] if title else []
        for p in paras:
            if flat_i in labels:
                speaker, emotion = labels[flat_i]
                p = (f"[{speaker}: {emotion}] " if emotion else f"[{speaker}] ") + p
            block.append(p)
            flat_i += 1
        out.append("\n\n".join(block))
    characters = {k: (genders[k].most_common(1)[0][0] if genders.get(k) else None) for k in known}
    if any(s == "Unknown" for s, _ in labels.values()):
        characters["Unknown"] = None
    return "\n\n".join(out), characters


@en.streamed_on_mlx_thread
def detect(text, stop=None):
    """Yields ("progress", done, total) and finally ("done", tagged_text, {character: gender})."""
    en.release_models()  # free the voice model while the AI reads
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    yield ("progress", 0, 0)
    model, tok = load(LLM_MODEL)
    sampler = make_sampler(temp=0.0)
    chapters, texts = chapters_of(text)

    def gen(prompts, max_tokens):
        return [generate(model, tok, prompt=tok.apply_chat_template([{"role": "user", "content": p}],
                                                                     add_generation_prompt=True, tokenize=False),
                         max_tokens=m, sampler=sampler, verbose=False) for p, m in zip(prompts, max_tokens)]

    labels, genders, known = {}, {}, []
    for ev in core.run(texts, gen, stop, starts=starts_of(chapters)):
        if ev[0] == "progress":
            yield ev[:3]
        else:
            labels, genders, known = ev[1:]
    del model
    en.free()
    yield ("done", *assemble(chapters, labels, genders, known))
