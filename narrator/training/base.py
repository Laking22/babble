"""Training: recipes (one per TTS model) and back ends (where the GPU work runs).

    dataset (Train tab)  ->  Recipe.prepare  ->  Package  ->  Backend.submit/status/fetch  ->  Recipe.install

A Recipe knows one model: how to turn the Train tab's clips into its training format (encoding audio locally
where it can, so only small files are uploaded), which script trains it, and how to install the result as
voices the app can use. A Backend knows one place to run a Package's script: Kaggle, a rented GPU, this
machine. The two never need to know about each other.
"""
import contextlib
import json
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..paths import RUNS


@dataclass
class VoiceData:
    dataset: Path        # folder with metadata.jsonl + wavs/ (written by the Train tab), or None if already encoded
    key: str             # cache key for encoded clips (usually the Train tab's voice folder name)
    speaker: str         # the name the model is trained with (unique within the model)


@dataclass
class Package:
    model: str                    # model name, e.g. "narrators"
    recipe: str                   # recipe id
    data_dir: Path                # everything the script needs; uploaded as-is
    script: str                   # file name inside data_dir to run
    args: str = ""                # command-line arguments for the script
    pip: list = field(default_factory=list)   # packages to install on the GPU machine first
    gpu: str = "T4"               # smallest GPU it fits on
    hours: float = 0.0            # hours of audio
    gpu_hours: float = 0.0        # estimate
    small_outputs: str = ""       # regex of result files to fetch first (samples, logs)
    big_outputs: str = ""         # regex of the trained model files
    meta: dict = field(default_factory=dict)  # recipe-specific (voices, version, adapter chain…)
    slug: str = ""                # the cloud notebook's name (else from model and recipe), e.g. a book's lines
    title: str = ""
    kernel_sources: list = field(default_factory=list)  # Kaggle: earlier notebooks whose output it reads


@dataclass
class RunStatus:
    state: str                    # "queued" | "running" | "complete" | "error" | "cancelled" | "unknown"
    detail: str = ""
    url: str = ""


class Recipe:
    id = ""
    name = ""
    engine = ""                   # the tts engine id its models run on
    readiness = "planned"         # "ready" once prepare + install are implemented and tested
    sample_rate = 24000
    clip_seconds = (2.0, 15.0)
    max_voices = 8                # per model, to limit bleed between voices
    gpu_hours_per_audio_hour = 0.13

    def prepare(self, voices, model, *, start_from=None, replay=0.25, epochs=1.0, log=print):
        """Encode locally and write a Package. `voices`: list[VoiceData]."""
        raise NotImplementedError(f"{self.name} training isn't built yet.")

    def install(self, run, results, log=print):
        """Turn fetched results (a folder) into an installed model the engine lists. Returns its folder."""
        raise NotImplementedError

    def after_fetch(self, run, results, log=print):
        """Optional: keep what later runs build on (e.g. LoRA adapters)."""


class AuthError(RuntimeError):
    """The back end needs the user to sign in (again)."""


@contextlib.contextmanager
def have_file(names):
    """A temporary file listing the line files the Mac has already (a long list doesn't fit in a command line)."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(sorted(names), f)
    try:
        yield f.name
    finally:
        Path(f.name).unlink(missing_ok=True)


class Backend:
    id = ""
    name = ""
    readiness = "planned"
    pricing = ""                  # e.g. "free, 30 GPU hours a week"
    # Create in the cloud can copy the lines back while the job works (fetch_new); False where the service only
    # hands a run's files over when it ends (Kaggle), or where that isn't known to work yet
    live = False
    # the servers Babble talks to for it, to tell when a lost connection is back (conn_issues.reachable); empty:
    # whether this Mac is online at all
    hosts = ()
    # how long a finished run's files wait out there for a download that couldn't get through (Modal's volume
    # keeps them; a rented machine is kept, and costs, until they're back)
    keeps_hours = 24

    def check(self):
        """dict: ok (bool), user, detail (str), quota (str or None). Never raises for 'not signed in'."""
        return {"ok": False, "detail": "not built yet"}

    def estimate(self, package):
        return {"gpu_hours": package.gpu_hours, "cost": None}

    def submit(self, package, log=print):
        """Upload and start. Returns a run id the other calls take."""
        raise NotImplementedError

    def status(self, run_id):
        raise NotImplementedError

    def fetch(self, run_id, dest, pattern, log=print):
        raise NotImplementedError

    def fetch_new(self, run_id, dest, have, final=False):
        """Copy the finished lines (out/lines/<key>.flac) whose names aren't in `have` into dest; returns how many.
        final: the job has ended (a service may need to wake its storage for this). Only where live is True."""
        raise NotImplementedError

    def cancel(self, run_id):
        raise NotImplementedError


# ---------- run registry (one file for every back end) ----------

def run_key(recipe, model):
    """Runs are stored per recipe and model, so an Orpheus and a Qwen model can share a name."""
    return f"{recipe}/{model}"


def runs():
    return json.loads(RUNS.read_text()) if RUNS.exists() else {}


def save_run(model, /, **fields):
    data = runs()
    data.setdefault(model, {}).update(fields)
    RUNS.write_text(json.dumps(data, indent=1, default=str))
    return data[model]


def package_summary(p):
    d = asdict(p)
    d["data_dir"] = str(p.data_dir)
    d["when"] = time.strftime("%Y-%m-%d %H:%M")
    return d
