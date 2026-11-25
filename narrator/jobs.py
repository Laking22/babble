"""Background jobs with a log on disk (training sends, installs, downloads), so the UI can poll them and a
restart still shows what happened."""
import itertools
import json
import threading
import time
import traceback

from .paths import WORK

JOBS = WORK / "jobs"
JOBS.mkdir(parents=True, exist_ok=True)
_live = {}
_ids = itertools.count(1)
_lock = threading.Lock()


class Job:
    def __init__(self, kind, title):
        self.id = f"{time.strftime('%Y%m%d-%H%M%S')}-{next(_ids)}"
        self.kind, self.title = kind, title
        self.state, self.result, self.error = "running", None, None
        self.started, self.ended = time.time(), None
        self.path = JOBS / f"{self.id}.log"
        self.stop = threading.Event()

    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        with _lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def text(self, last=60):
        if not self.path.exists():
            return ""
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return "\n".join(lines[-last:])

    def save(self):
        (JOBS / f"{self.id}.json").write_text(json.dumps(
            {"id": self.id, "kind": self.kind, "title": self.title, "state": self.state, "error": self.error,
             "started": self.started, "ended": self.ended}))


def start(kind, title, fn, *args, **kwargs):
    """Run fn(*args, log=job.log, **kwargs) in a thread. Returns the Job."""
    job = Job(kind, title)
    _live[job.id] = job
    job.log(f"{title}: started")
    job.save()

    def run():
        try:
            job.result = fn(*args, log=job.log, **kwargs)
            job.state = "done"
            job.log("Finished.")
        except BaseException as e:  # noqa: BLE001 - reported in the log and the UI
            job.state, job.error = "failed", str(e)
            job.log(f"FAILED: {e}")
            job.log(traceback.format_exc(limit=4))
        finally:
            job.ended = time.time()
            job.save()

    threading.Thread(target=run, daemon=True, name=f"job-{job.id}").start()
    return job


def get(job_id):
    return _live.get(job_id)


def running(kind=None):
    return [j for j in _live.values() if j.state == "running" and (kind is None or j.kind == kind)]


def recent(n=10):
    return sorted(_live.values(), key=lambda j: j.started, reverse=True)[:n]
