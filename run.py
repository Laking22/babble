"""Start the Babble engine:  .venv/bin/python run.py [--port 7895]"""
import os
import sys
from pathlib import Path

_path = os.environ.get("PATH", "").split(os.pathsep)
if sys.platform == "darwin":
    # Opened from the Dock or Finder, macOS gives the app only /usr/bin:/bin:/usr/sbin:/sbin, so tools installed with
    # Homebrew (ffmpeg, which joins the chapters into the book) aren't found. Look in Homebrew's folders too.
    os.environ["PATH"] = os.pathsep.join(_path + [d for d in ("/opt/homebrew/bin", "/usr/local/bin") if d not in _path])
elif sys.platform == "win32":
    from narrator.paths import HOME
    _ff = str(HOME / "tools" / "ffmpeg")
    if Path(_ff).is_dir() and _ff not in _path:
        os.environ["PATH"] = os.pathsep.join([_ff, *_path])

PORT = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 7895
if __name__ == "__main__" and sys.platform == "darwin" and os.environ.get("BABBLE_MOVE_DATA") == "1":
    # the data folder's new name (…/Narrator Studio → …/Babble), once, before anything opens it. Only when the app (or
    # the launcher) asks: an older app looks for the engine's key in the old folder, so it mustn't be moved under it.
    from narrator import move_home
    move_home.move(PORT)

from narrator.server import main  # noqa: E402

if __name__ == "__main__":
    main(PORT)
