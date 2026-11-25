"""Where the app keeps its data. Nothing is written next to the code.

NARRATOR_HOME overrides the location (handy for development and tests).
Default: ~/Library/Application Support/Babble on a Mac (until the engine has moved it there from its old name,
Narrator Studio: narrator/move_home.py), %APPDATA%\Babble on Windows, ~/.local/share/babble on Linux.
"""
import os
import sys
from pathlib import Path


def _default_home() -> Path:
    if sys.platform == "win32":
        base = (os.environ.get("APPDATA") or "").strip() or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "Babble"
    if sys.platform != "darwin":
        base = (os.environ.get("XDG_DATA_HOME") or "").strip() or str(Path.home() / ".local" / "share")
        return Path(base) / "babble"
    from .move_home import folders, old_home
    return old_home() or folders()[1]


HOME = Path(os.environ.get("NARRATOR_HOME") or _default_home()).expanduser()

VOICES = HOME / "voices"          # saved (designed / cloned) voices
MODELS = HOME / "models"          # installed fine-tuned models, one folder each with voice.json
OUTPUTS = HOME / "outputs"        # rendered audio
TRAINING = HOME / "training"      # Train tab: one folder per voice (books, clips, dataset)
WORK = HOME / "train-work"        # recipe/back-end scratch: encoded codes, packages, results, adapters
TOOLS = HOME / "tools"            # side environments (e.g. torch + SNAC for Orpheus encoding)
SETTINGS = HOME / "settings.json"
RUNS = HOME / "runs.json"         # training runs sent to any back end

for _d in (VOICES, MODELS, OUTPUTS, TRAINING, WORK, TOOLS):
    _d.mkdir(parents=True, exist_ok=True)
