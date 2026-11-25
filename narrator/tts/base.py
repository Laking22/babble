"""The interface every TTS engine implements.

A voice is a string "<prefix>:<key>", e.g. "builtin:ryan", "my:Nova", "orpheus:tara", "kokoro:bm_george".
Each engine owns one or more prefixes. The renderer groups a story's lines by engine and by the model they
need, so each model is loaded once per render, and asks the engine to speak each group.
"""
import subprocess
from dataclasses import dataclass, field


def line_temperature(item, opts, tested, most=1.4, least=0.6):
    """A model's sampling temperature for one line, from the voice's stability (0% most expressive, 100% most
    stable). Lines carry Qwen's temperature for it (a character's own, else the page's in opts): 1.3 at 0% to 0.5
    at 100%. Other models keep their own tested value at the default 55%, and go from tested*most (0%) to
    tested*least (100%), so they stay inside a range that still sounds right."""
    t = item.get("temperature", opts.get("temperature"))
    if t is None:
        return tested
    s = max(0.0, min(100.0, (1.3 - float(t)) / 0.8 * 100))
    k = most - (most - 1) * s / 55 if s <= 55 else 1 - (1 - least) * (s - 55) / 45
    return round(tested * k, 3)


def ram_gb():
    """This machine's memory in GB (0 if it can't be read)."""
    from ..runtime import total_ram_bytes
    return total_ram_bytes() / 2 ** 30


@dataclass
class Voice:
    value: str              # "prefix:key"
    name: str               # shown in the UI
    gender: str = "unspecified"
    description: str = ""
    engine: str = ""
    tags: list = field(default_factory=list)   # e.g. ["fine-tuned"], ["saved"]


class Engine:
    id = ""                 # "qwen", "orpheus", "kokoro"…
    name = ""               # "Qwen3-TTS 1.7B"
    short = ""              # "Qwen": used in labels like "Tara (Orpheus)"
    tier = "standard"       # "lite" (8 GB, fast), "standard" (16 GB), "pro" (32 GB+)
    license = ""            # short license name; check the model card before shipping
    min_ram_gb = 8
    prefixes = ()
    # "emotion" (free-text style), "tags" (<laugh> etc.), "clone", "design", "batch", "finetune", "remote"
    features = frozenset()
    hf_models = ()          # Hugging Face repos it downloads on first use
    cloud = False           # its lines can be made by Create in the cloud even where this Mac can't run it
    pip = ()                # extra Python packages it needs beyond the base install

    def available(self):
        """(True, "") when it can run here, else (False, why)."""
        return True, ""

    def voices(self):
        """list[Voice]"""
        return []

    def note(self):
        """A short heads-up to show with this engine's voices (e.g. it's tight on this Mac's memory), or ""."""
        return ""

    def owns(self, value):
        return (value or "").partition(":")[0] in self.prefixes

    def model_key(self, value):
        """Which model a voice needs. Lines with the same key are made together."""
        return self.id

    def prepare(self, items, opts):
        """Called once per render, before any `speak`, with every item for this engine."""

    def speak(self, items, opts, stop=None):
        """Yield (index, float32 mono audio at 24 kHz) for each item. All items share one model_key.

        opts: language, temperature, top_p, top_k, repetition_penalty, batch, exact (engine-specific extras allowed).
        Runs on the MLX thread. Check `stop.is_set()` between lines.
        """
        raise NotImplementedError

    def describe(self):
        ok, why = self.available()
        return {"id": self.id, "name": self.name, "tier": self.tier, "license": self.license,
                "min_ram_gb": self.min_ram_gb, "features": sorted(self.features), "available": ok, "why": why}
