"""Modal (modal.com): pay-per-second cloud GPUs with your own Modal account.

Talks to Modal through Modal's own Python library, installed with the app: each step (check, sign in, start a
job, status, download) runs as a short-lived process of the app's own Python (narrator/modal_jobs.py), so there's
nothing to install or type in Terminal. Sign in from Connections (Sign in with Modal: the browser opens and you
approve this computer).

Per run: the package goes to a Modal Volume ("narrator-train") under /<run>/data; a job runs the script on a GPU
with the outputs written to /<run>/out on the same volume, then writes /<run>/exit_code. Status reads that file
(and asks Modal whether the job itself failed); fetch downloads the matching output files.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ...paths import HOME
from ..base import AuthError, Backend, RunStatus
from ..base import have_file as _have_file

VOLUME = "narrator-train"
GPU = "L4"
# GPUs a job can ask for: Modal's name -> (label, US$ per hour, Modal's list price)
GPUS = {"L4": ("L4 · 24 GB", 0.80), "A10G": ("A10 · 24 GB", 1.10), "L40S": ("L40S · 48 GB", 1.95),
        "A100-40GB": ("A100 · 40 GB", 2.10), "H100": ("H100 · 80 GB", 3.95)}  # (H100: Drama 3 books only)
JOBS = Path(__file__).resolve().parents[2] / "modal_jobs.py"
CALLS = HOME / "modal-calls.json"  # run -> Modal's id for the job, so status can ask whether it failed
SIGN_IN_HELP = "Sign in to Modal on the Connections page."


def step(name, args=None, log=None, timeout=600, progress=None):
    """Run one Modal step (modal_jobs.py) and return its answer. Progress lines go to log; a URL line too, as
    'URL <link>'; a download's PROGRESS lines to progress (a dict). Raises AuthError when Modal isn't signed in,
    RuntimeError for anything else."""
    p = subprocess.Popen([sys.executable, str(JOBS), name, json.dumps(args or {})], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    answer, tail = None, []
    started = time.time()
    try:
        for line in p.stdout:
            line = line.rstrip("\n")
            if line.startswith("RESULT "):
                answer = json.loads(line[7:])
            elif line.startswith("PROGRESS "):
                if progress:
                    try:
                        progress(json.loads(line[9:]))
                    except Exception:  # noqa: BLE001 - telling must never stop the download
                        pass
            elif line.startswith(("LOG ", "URL ")):
                if log:
                    log(line[4:] if line.startswith("LOG ") else line)
            elif line.strip():
                tail = (tail + [line])[-20:]
            if time.time() - started > timeout:
                p.kill()
                raise RuntimeError(f"Modal took too long ({name}).")
        p.wait(timeout=30)
    finally:
        if p.poll() is None:
            p.kill()
    if answer is None:
        raise RuntimeError(f"Modal step {name} stopped: {' '.join(tail)[-400:] or 'no answer'}")
    if not answer.get("ok"):
        if answer.get("auth"):
            raise AuthError(f"{answer.get('error') or 'Modal isn’t signed in.'} {SIGN_IN_HELP}".strip())
        raise RuntimeError(f"Modal: {answer.get('error') or 'it didn’t work'}")
    return answer


def sign_in(log=print):
    """Modal's browser sign-in, from the app (what `modal token new` does)."""
    a = step("signin", {"wait_minutes": 10}, log=log, timeout=11 * 60)
    from ... import conn_issues
    conn_issues.ok("modal")
    log(f"Done · Signed in to Modal ({a.get('workspace') or 'your workspace'}).")
    return a.get("workspace")


def _calls():
    try:
        return json.loads(CALLS.read_text())
    except (OSError, ValueError):
        return {}


class Modal(Backend):
    id, name, readiness = "modal", "Modal", "beta"
    reports_progress = True  # fetch says how far along it is (no need to watch the folder fill)
    resumes = True           # fetch(resume=True) keeps the files that came whole last time
    live = True  # the job saves its storage every 2 minutes, so the lines can be copied back as it goes
    # its main server, and the file storage a volume's files come from (some networks block only the second)
    hosts = ("api.modal.com", "clobber.modal-storage.com")
    keeps_hours = 7 * 24  # a run's lines stay on the volume: a download is tried again for a week
    pricing = "pay per second with your Modal account (an L4 is about $0.80/h)"

    def check(self):
        try:
            a = step("check", timeout=90)
        except AuthError:
            return {"ok": False, "detail": f"Modal isn't signed in. {SIGN_IN_HELP}", "quota": None}
        except Exception as e:  # noqa: BLE001 - e.g. offline
            return {"ok": False, "detail": f"Couldn't reach Modal: {str(e)[:200]}", "quota": None}
        return {"ok": True, "user": a.get("workspace"), "quota": None,
                "detail": f"Signed in to Modal ({a.get('workspace') or 'your workspace'}). Pays by the second; "
                          "the free Starter plan includes $30 of GPU time a month."}

    def estimate(self, package):
        hours = max(0.3, package.gpu_hours * 0.6)  # an L4 is roughly 1.5-2x a T4 for these jobs
        return {"gpu_hours": round(hours, 1), "cost": round(hours * 0.8, 2), "fits": True,
                "note": f"Roughly ${hours * 0.8:.2f} on an L4 (estimate)."}

    def submit(self, package, log=print):
        run_id = f"{package.slug or package.model + '-' + package.recipe}-{time.strftime('%Y%m%d-%H%M%S')}"
        gpu = package.gpu if package.gpu in GPUS else GPU  # (training asks for "T4": the smallest that fits)
        a = step("submit", {"run": run_id, "gpu": gpu, "data_dir": str(package.data_dir), "script": package.script,
                            "args": package.args, "pip": list(package.pip), "live": bool(package.meta.get("live")),
                            "timeout": int(package.meta.get("timeout_hours", 12) * 3600)},
                 log=log, timeout=45 * 60)  # (the first run builds the software image: a few minutes)
        calls = _calls()
        calls[run_id] = a.get("call")
        tmp = CALLS.with_name(f".{CALLS.name}.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(calls, indent=1))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, CALLS)
        finally:
            tmp.unlink(missing_ok=True)
        log(f"Started {run_id} on Modal ({gpu})")
        return run_id

    def url(self, run_id):
        return "https://modal.com/apps"

    def cancel(self, run_id):
        call = _calls().get(run_id)
        if not call:
            raise NotImplementedError  # sent before calls were kept: stop it on modal.com
        step("cancel", {"call": call}, timeout=120)

    def status(self, run_id):
        from ... import conn_issues
        try:
            a = step("status", {"run": run_id, "call": _calls().get(run_id)}, timeout=120)
        except AuthError:
            raise
        except Exception as e:  # noqa: BLE001 - propagate network errors so reconnect & conn_issues fire
            if conn_issues.is_network(e):
                raise
            return RunStatus("unknown", str(e)[:200])
        return RunStatus(a.get("state", "unknown"), a.get("detail", ""))

    def fetch(self, run_id, dest, pattern, log=print, resume=False):
        """The files, saying which one and how far along: "file 3 of 14 (vocab.json) · 3.9 of 4.5 GB · …"."""
        from ...progress import Meter
        from ...status import Status
        log("Asking Modal for the files…")
        meter = Meter()

        def told(p):
            text, share = meter(p["done"], p["total"])
            name = p["file"].rsplit("/", 1)[-1]
            log(Status(f"file {p['i']} of {p['n']} ({name}) · {text}" if p["n"] > 1 else text, share))
        step("fetch", {"run": run_id, "dest": str(dest), "pattern": pattern, "resume": resume}, log=log,
             timeout=3 * 3600,
             progress=told)
        return Path(dest)

    def fetch_new(self, run_id, dest, have, final=False):
        with _have_file(have) as f:
            return step("fetch_new", {"run": run_id, "dest": str(dest), "have_file": f}, timeout=3600).get("files", 0)
