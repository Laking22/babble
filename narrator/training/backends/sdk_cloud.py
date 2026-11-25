"""Lightning AI and Beam: GPU services reached through their own Python libraries (installed with the app). Each
step runs as a short-lived process (narrator/sdk_jobs.py) with the key in its environment, like Modal's.

Lightning AI (lightning.ai): the job runs on a Lightning Studio started on a GPU; new accounts get free credits
(after a phone check). The Studio stops itself when the job ends; the app copies the lines back and deletes it.
Beam (beam.cloud): the job runs in a GPU sandbox that ends itself after the job's time limit; the app ends it as
soon as the lines are back.
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from ...paths import HOME
from ..base import AuthError, RunStatus
from ..base import have_file as _have_file
from .machines import PaidCloud, job_script, job_seconds

JOBS = Path(__file__).resolve().parents[2] / "sdk_jobs.py"
IMAGE = "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime"


class SDKCloud(PaidCloud):
    service = ""  # sdk_jobs.py's name for it

    def env(self, key):
        raise NotImplementedError

    def step(self, name, args=None, key=None, log=None, timeout=600):
        """Run one step and return its answer. Raises AuthError when the key isn't accepted."""
        import os
        # the key goes over stdin (the step puts it in its own environment for the service's library): a process's
        # starting environment can be listed by other programs, what it sets later can't
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "BABBLE_ENV_STDIN": "1"}
        p = subprocess.Popen([sys.executable, str(JOBS), self.service, name, json.dumps(args or {})],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.PIPE,
                             text=True, bufsize=1, env=env)
        p.stdin.write(json.dumps(self.env(key or self.key())) + "\n")
        p.stdin.close()
        answer, tail, started = None, [], time.time()
        try:
            for line in p.stdout:
                line = line.rstrip("\n")
                if line.startswith("RESULT "):
                    answer = json.loads(line[7:])
                elif line.startswith("LOG "):
                    if log:
                        log(line[4:])
                elif line.strip():
                    tail = (tail + [line])[-20:]
                if time.time() - started > timeout:
                    p.kill()
                    raise RuntimeError(f"{self.name} took too long ({name}).")
            p.wait(timeout=30)
        finally:
            if p.poll() is None:
                p.kill()
        if answer is None:
            raise RuntimeError(f"{self.name} step {name} stopped: {' '.join(tail)[-400:] or 'no answer'}")
        if not answer.get("ok"):
            if answer.get("auth"):
                raise AuthError(f"{self.name} didn't accept the key. {self.sign_in_help}")
            raise RuntimeError(f"{self.name}: {answer.get('error') or 'it didn’t work'}")
        return answer

    def account(self, key=None):
        a = self.step("check", key=key, timeout=120)
        return {"balance": None, "user": a.get("user")}

    def gpus(self):
        return [{"id": g, "label": label, "gb": None, "price": self.prices.get(g), "stock": None}
                for g, label in self.gpu_names.items()]

    prices: dict = {}


class Lightning(SDKCloud):
    id, name, service = "lightning", "Lightning AI", "lightning"
    pricing = "free credits for new accounts, then pay as you go (a T4 or L4 Studio)"
    key_setting, keys_page = "lightning_key", "lightning.ai → Settings → Keys"
    console = "https://lightning.ai"
    gpu_names, default_gpu = {"L4": "L4", "T4": "T4", "L40S": "L40S", "A100": "A100", "H100": "H100"}, "L4"
    prices = {"T4": 0.45, "L4": 0.80, "L40S": 1.85, "A100": 3.20, "H100": 4.40}
    speed = {"L4": 1.5, "T4": 1.2, "L40S": 3.5, "A100": 5.0, "H100": 6.0}
    batch = {"L40S": 32, "A100": 32, "H100": 32}
    billing_note = "Studios are billed from your Lightning credits while they run."

    def parse_key(self, text):
        """Lightning's Keys page shows a user ID and an API key: paste both (the lines it shows, or the two values)."""
        t = (text or "").strip()
        uid = re.search(r"LIGHTNING_USER_ID\W+([\w-]{8,})", t)
        key = re.search(r"LIGHTNING_API_KEY\W+([\w-]{8,})", t)
        if uid and key:
            return f"{uid.group(1)}:{key.group(1)}"
        parts = re.findall(r"[\w-]{8,}", t)
        if len(parts) == 2:
            return f"{parts[0]}:{parts[1]}"
        raise ValueError("Paste both values from lightning.ai → Settings → Keys: the user ID and the API key "
                         "(the two lines it shows are fine).")

    def env(self, key):
        uid, _, k = key.partition(":")
        return {"LIGHTNING_USER_ID": uid, "LIGHTNING_API_KEY": k}

    def submit(self, package, log=print):
        run_id = f"{package.slug or package.model}-{time.strftime('%Y%m%d-%H%M%S')}"
        gpu = package.gpu if package.gpu in self.gpu_names else self.default_gpu
        home = f"/teamspace/studios/this_studio/narrator/{run_id}"
        stop = 'python3 -c "from lightning_sdk import Studio; Studio().stop()" >/dev/null 2>&1'
        (Path(package.data_dir) / "job.sh").write_text(job_script(
            self.name, package, f"{home}/data", f"{home}/out",
            watchdog=f"(sleep {job_seconds(package) + 600}; {stop}) &", after=stop))
        studio = f"narrator-{run_id}"[:60].rstrip("-")
        a = self.step("submit", {"run": run_id, "gpu": gpu, "data_dir": str(package.data_dir), "studio": studio},
                      log=log, timeout=45 * 60)
        self._save_run(run_id, {"studio": a["studio"], "teamspace": a["teamspace"], "user": a["user"], "gpu": gpu,
                                "started": time.time()})
        log(f"Started on a Lightning {gpu} Studio")
        return run_id

    def _args(self, run_id, **kw):
        r = self._runs().get(run_id)
        if not r:
            raise RuntimeError("This job isn't known to the app any more.")
        return {"run": run_id, "studio": r["studio"], "teamspace": r["teamspace"], "user": r["user"], **kw}

    def url(self, run_id):
        r = self._runs().get(run_id) or {}
        return f"https://lightning.ai/{r['user']}/{r['teamspace']}/studios" if r.get("user") else self.console

    def status(self, run_id):
        try:
            a = self.step("status", self._args(run_id), timeout=180)
        except AuthError as e:
            return RunStatus("unknown", str(e))
        except Exception as e:  # noqa: BLE001 - asked again later
            return RunStatus("unknown", str(e)[:200])
        # stopped: it stopped itself at the end (or Lightning took the machine back); what it made is on its disk
        return RunStatus({"stopped": "complete"}.get(a["state"], a["state"]), a.get("detail", ""))

    live = True  # read from the Studio's disk while it runs

    def fetch(self, run_id, dest, pattern, log=print):
        log("Copying the lines back from Lightning…")
        # just the summary and log when the lines already came back while it worked (see fetch_new)
        small = "lines" not in pattern
        self.step("fetch", self._args(run_id, dest=str(dest), small=small), log=log, timeout=3 * 3600)
        return Path(dest)

    def fetch_new(self, run_id, dest, have, final=False):
        with _have_file(have) as f:
            return self.step("fetch_new", self._args(run_id, dest=str(dest), have_file=f, start=final),
                             timeout=3600).get("files", 0)

    def finished(self, run_id):
        if not (self._runs().get(run_id) or {}).get("deleted"):
            self.step("stop", self._args(run_id), timeout=300)
            self._save_run(run_id, {"deleted": time.time()})

    delete = finished


class Beam(SDKCloud):
    id, name, service = "beam", "Beam", "beam"
    pricing = "pay by the millisecond with your Beam account (an RTX 4090 is $0.69/h)"
    key_setting, keys_page = "beam_token", "platform.beam.cloud → Settings → API Keys"
    console = "https://platform.beam.cloud"
    gpu_names, default_gpu = {"RTX4090": "RTX 4090", "RTX5090": "RTX 5090", "H100": "H100"}, "RTX4090"
    prices = {"RTX4090": 0.69, "RTX5090": 1.09, "H100": 3.50}
    speed = {"RTX4090": 3.5, "RTX5090": 5.0, "H100": 6.0}
    batch = {"RTX5090": 24, "H100": 32}
    billing_note = "Billed by the millisecond while the sandbox runs."

    def env(self, key):
        return {"BEAM_TOKEN": key, "CONFIG_PATH": str(HOME / "beam-config.ini")}

    def submit(self, package, log=print):
        run_id = f"{package.slug or package.model}-{time.strftime('%Y%m%d-%H%M%S')}"
        gpu = package.gpu if package.gpu in self.gpu_names else self.default_gpu
        (Path(package.data_dir) / "job.sh").write_text(job_script(self.name, package, "/workspace/data", "/workspace/out"))
        ttl = job_seconds(package) + 30 * 60  # then Beam ends it whatever happens (time for the app to collect)
        a = self.step("submit", {"gpu": gpu, "data_dir": str(package.data_dir), "ttl": ttl, "image": IMAGE},
                      log=log, timeout=45 * 60)
        self._save_run(run_id, {"sandbox": a["sandbox"], "gpu": gpu, "started": time.time()})
        log(f"Started on a Beam {self.gpu_names[gpu]}")
        return run_id

    def _sandbox(self, run_id):
        r = self._runs().get(run_id)
        if not r:
            raise RuntimeError("This job isn't known to the app any more.")
        return r["sandbox"]

    def status(self, run_id):
        try:
            a = self.step("status", {"sandbox": self._sandbox(run_id)}, timeout=180)
        except AuthError as e:
            return RunStatus("unknown", str(e))
        except Exception as e:  # noqa: BLE001 - asked again later
            return RunStatus("unknown", str(e)[:200])
        if a["state"] == "gone":
            return RunStatus("error", "the Beam sandbox ended before its lines were collected")
        return RunStatus(a["state"], a.get("detail", ""))

    live = True  # read from the sandbox while it runs

    def fetch_new(self, run_id, dest, have, final=False):
        with _have_file(have) as f:
            return self.step("fetch_new", {"sandbox": self._sandbox(run_id), "dest": str(dest), "have_file": f},
                             timeout=3600).get("files", 0)

    def fetch(self, run_id, dest, pattern, log=print):
        log("Downloading from Beam…")
        self.step("fetch", {"sandbox": self._sandbox(run_id), "dest": str(dest), "pattern": pattern}, log=log,
                  timeout=3 * 3600)
        return Path(dest)

    def finished(self, run_id):
        if not (self._runs().get(run_id) or {}).get("deleted"):
            self.step("stop", {"sandbox": self._sandbox(run_id)}, timeout=180)
            self._save_run(run_id, {"deleted": time.time()})

    delete = finished
