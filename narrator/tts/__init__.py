"""Engine registry. Add an engine by writing a class in this package and listing it in ENGINE_CLASSES."""
from .base import Engine, Voice
from .higgs import Higgs
from .vibevoice import VibeVoice
from .kokoro import Kokoro
from .orpheus import Orpheus
from .qwen import Qwen

# (drama.py, Fish Audio Drama 3, is kept but not listed)
ENGINE_CLASSES = [Qwen, Orpheus, Kokoro, VibeVoice, Higgs]
ENGINES = {cls.id: cls() for cls in ENGINE_CLASSES}


def is_off(engine_id):
    """Turned off on the Models page: its voices leave the voice lists and nothing is made with it."""
    from .. import settings
    return engine_id in (settings.load().get("engines_off") or [])


def _with_off_switch(e):
    can_run = e.available  # (whether it could run here at all, on or off)

    def available():
        if is_off(e.id):
            return False, f"{e.name} is turned off. Turn it on on the Models page to use its voices."
        return can_run()
    e.can_run, e.available = can_run, available


for _e in ENGINES.values():
    _with_off_switch(_e)


def engine_for(value):
    for e in ENGINES.values():
        if e.owns(value):
            return e
    raise ValueError(f"No engine for voice “{value}”.")


def available():
    return [e for e in ENGINES.values() if e.available()[0]]


def all_voices(only_available=True):
    return [v for e in (available() if only_available else ENGINES.values()) for v in e.voices()]


def refresh():
    """Pick up newly installed fine-tuned models."""
    for e in ENGINES.values():
        if hasattr(e, "refresh"):
            e.refresh()


__all__ = ["Engine", "Voice", "ENGINES", "engine_for", "available", "all_voices", "refresh", "is_off"]
