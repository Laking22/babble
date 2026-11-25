"""Rented GPU machines (Runpod, Vast.ai, Thunder Compute, Lambda): a whole machine rented by the second through the
service's API with the user's own API key, for Create in the cloud.

How a job runs, whatever the service: the app rents a machine (an image with Python, PyTorch and SSH), connects
with its own SSH key (made once, kept in the app's folder), copies the job over, and starts it. The app asks how
it's doing over SSH, copies the results back, then deletes the machine. A machine costs money for as long as it's
on, so it also deletes itself: when the job is done and the app hasn't collected the results within
GRACE_MINUTES, and in any case after the job's time limit plus that grace (a watchdog started first thing).

A service is a subclass that knows its API: account() (is the key good, how much credit), gpus() (what can be
rented now, at what price), rent(), machine() (its state and SSH address), remove() and self_delete() (how the
machine deletes itself). Everything else is here. Nothing to install: the APIs are plain HTTPS and SSH comes
with macOS. The keys are kept in the app's settings file (only you can read it).
"""
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import threading
import time
from pathlib import Path

_runs_lock = threading.Lock()

from ... import settings
from ...paths import HOME
from ..base import AuthError, Backend, RunStatus
from .ssh import SSH

KEY_FILE = HOME / "ssh" / "cloud_ed25519"
LINE = re.compile(r"[0-9a-f]{20}\.flac")  # a finished line (lines_runner renames each one when it's whole)
GRACE_MINUTES = 90  # how long a finished machine waits for the app to collect the results before deleting itself


def ssh_key():
    """The app's own SSH key for its rented machines (made once; ssh-keygen comes with macOS)."""
    if not KEY_FILE.exists():
        KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "narrator-studio-cloud", "-f", str(KEY_FILE)],
                       check=True)
    return KEY_FILE, Path(str(KEY_FILE) + ".pub").read_text().strip()


def size_mb(folder):
    return sum(f.stat().st_size for f in Path(folder).rglob("*") if f.is_file()) / 1e6


class MachineSSH(SSH):
    """The SSH connection to one rented machine (the app's own key; no host-key prompts)."""

    def __init__(self, host, port, user="root", service="the rented machine"):
        self.host, self.port, self.user, self.service = host, str(port), user, service

    def _cfg(self):
        return {"host": f"{self.user}@{self.host}", "port": self.port, "key": str(KEY_FILE), "dir": "~/narrator",
                "python": "python3"}

    def _ssh_base(self):
        opts = ["-p", self.port, "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", "-o", "ServerAliveInterval=30",
                "-i", str(KEY_FILE)]
        return opts, f"{self.user}@{self.host}"

    # copying with tar over SSH: both ends have tar (rsync may not be on the rented machine)
    def put(self, local_dir, remote_dir):
        opts, host = self._ssh_base()
        qdir = shlex.quote(remote_dir)
        pack = subprocess.Popen(["tar", "-C", str(local_dir), "-cf", "-", "."], stdout=subprocess.PIPE)
        r = subprocess.run(["ssh", *opts, "--", host, f"mkdir -p {qdir} && tar -C {qdir} -xf -"],
                           stdin=pack.stdout, capture_output=True, text=True)
        pack.stdout.close()
        pack.wait()
        if r.returncode or pack.returncode:
            raise RuntimeError(f"Copying to {self.service} failed: {(r.stderr or '').strip()[-300:]}")

    def write(self, remote_path, text, private=False):
        """Write a small file on the machine (through stdin, so its contents never show in a command line)."""
        opts, host = self._ssh_base()
        cmd = f"{'umask 077; ' if private else ''}cat > {shlex.quote(remote_path)}"
        r = subprocess.run(["ssh", *opts, "--", host, cmd], input=text, capture_output=True, text=True, timeout=60)
        if r.returncode:
            raise RuntimeError(f"Writing to {self.service} failed: {(r.stderr or '').strip()[-300:]}")

    def get(self, remote_files, remote_root, dest):
        opts, host = self._ssh_base()
        dest_root = Path(dest).resolve()
        dest_root.mkdir(parents=True, exist_ok=True)
        names = " ".join(shlex.quote(f) for f in remote_files)
        fetch = subprocess.Popen(["ssh", *opts, "--", host, f"cd {shlex.quote(remote_root)} && tar -cf - {names}"],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        extract_err = ""
        try:
            with tarfile.open(fileobj=fetch.stdout, mode="r|*") as tf:
                for m in tf:
                    if not m.isfile():
                        continue
                    rel = Path(m.name)
                    if rel.is_absolute() or ".." in rel.parts:
                        raise RuntimeError(f"Refused unsafe tar path from {self.service}: {m.name}")
                    target = (dest_root / rel).resolve()
                    if dest_root != target and dest_root not in target.parents:
                        raise RuntimeError(f"Refused tar path escaping destination: {m.name}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    src = tf.extractfile(m)
                    if src is not None:
                        with src, open(target, "wb") as out:
                            shutil.copyfileobj(src, out, 1 << 20)
        except (tarfile.TarError, OSError) as e:
            extract_err = str(e)
        finally:
            if fetch.stdout:
                fetch.stdout.close()
        err = fetch.stderr.read().decode(errors="replace")
        fetch.wait()
        if extract_err or fetch.returncode:
            raise RuntimeError(f"Copying back from {self.service} failed: {(err or extract_err).strip()[-300:]}")


RUNNER = """#!/bin/bash
# A Babble job on a rented {service} machine. The machine deletes itself when it's no longer needed.
cd "$(dirname "$0")"
mkdir -p out
(sleep {limit}; python3 selfdelete.py) >/dev/null 2>&1 &   # never longer than this, whatever happens
{{
  echo "== $(date) on $(hostname)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  if python3 -m venv --system-site-packages venv >/dev/null 2>&1; then PY=venv/bin/python; else PY=python3; fi
  export PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore
  {pip_line}
  NARRATOR_DATA="$PWD/data" $PY data/{script} {args} --out "$PWD/out"
}} > out/job.log 2>&1
echo $? > exit_code
# wait for the app to copy the results back (it leaves a file called collected), then delete this machine
for i in $(seq 1 {grace}); do [ -f collected ] && break; sleep 60; done
python3 selfdelete.py >/dev/null 2>&1
"""

# The start of every machine's selfdelete.py: env() also reads the container's own environment (/proc/1/environ),
# which an SSH session doesn't always get.
SELF_DELETE_HEAD = '''import json, os, subprocess, urllib.request
def env(name):
    if os.environ.get(name):
        return os.environ[name]
    try:
        for kv in open("/proc/1/environ", "rb").read().split(b"\\0"):
            k, _, v = kv.partition(b"=")
            if k.decode(errors="replace") == name:
                return v.decode(errors="replace")
    except OSError:
        pass
    return ""
def call(method, url, headers=None, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json", **(headers or {})})
    urllib.request.urlopen(req, timeout=60).read()
'''


class PaidCloud(Backend):
    """A pay-as-you-go GPU service used with the user's own API key, for Create in the cloud: the key, the runs
    this Mac started, and the GPUs it offers. RentedMachine (below) and the job services (Hugging Face Jobs,
    Lightning, Beam, Cerebrium) build on it."""
    readiness = "beta"
    training = False              # Create in the cloud only
    key_setting = ""              # the settings key that holds the API key
    keys_page = ""                # where the user makes a key
    console = ""                  # the service's page for its machines or jobs
    gpu_names: dict = {}          # the service's GPU id -> short label, in the order the menu shows them
    default_gpu = ""
    speed: dict = {}              # x real time, once loaded (estimates: memory speed vs the T4 we measured)
    batch: dict = {}              # lines at once (default 16)
    sends_models = True           # the Mac's trained voices go with each job (unpacked on the GPU)

    # ---------- the service's API (each service) ----------
    def account(self, key=None):
        """{"balance": US$ of credit or None when the service doesn't say, "user": name or None}. Raises AuthError."""
        raise NotImplementedError

    def gpus(self):
        """[{"id", "label", "gb", "price" ($/h now, or None), "stock" ("none" when none are free)}]"""
        raise NotImplementedError

    def parse_key(self, text):
        """The key as saved, from what was pasted (a service with two parts parses both)."""
        key = (text or "").strip()
        if not re.fullmatch(r"[\w\-.:=+/]{16,}", key):
            raise ValueError(f"That doesn't look like a {self.name} API key. Make one on {self.keys_page}.")
        return key

    # ---------- the key ----------
    @property
    def sign_in_help(self):
        return f"Add your {self.name} API key on the Connections page."

    def key(self):
        k = (settings.load().get(self.key_setting) or "").strip()
        if not k:
            raise AuthError(f"No {self.name} API key yet. {self.sign_in_help}")
        return k

    def save_key(self, key):
        key = self.parse_key(key)
        try:
            acc = self.account(key)
        except AuthError:
            raise ValueError(f"{self.name} didn't accept that key. Make one on {self.keys_page}.") from None
        settings.save(**{self.key_setting: key})
        from ... import conn_issues
        conn_issues.ok(self.id)
        return acc

    def forget_key(self):
        settings.save(**{self.key_setting: ""})

    # ---------- the runs this Mac started ----------
    @property
    def _runs_file(self):
        return HOME / f"{self.id}-runs.json"

    def _runs(self):
        try:
            return json.loads(self._runs_file.read_text())
        except (OSError, ValueError):
            return {}

    def _save_run(self, run_id, info):
        with _runs_lock:
            runs = self._runs()
            runs[run_id] = {**runs.get(run_id, {}), **info}
            tmp = self._runs_file.with_suffix(".tmp")
            try:
                tmp.write_text(json.dumps(runs, indent=1))
                os.replace(tmp, self._runs_file)
            finally:
                tmp.unlink(missing_ok=True)

    # ---------- Backend ----------
    def check(self):
        try:
            acc = self.account()
        except AuthError as e:
            return {"ok": False, "detail": str(e), "quota": None}
        except Exception as e:  # noqa: BLE001 - e.g. offline
            return {"ok": False, "detail": f"Couldn't reach {self.name}: {str(e)[:200]}", "quota": None}
        credit = f" · ${acc['balance']:.2f} of credit" if acc.get("balance") is not None else ""
        running = f" · ${acc['spend_per_hour']:.2f}/h running now" if acc.get("spend_per_hour") else ""
        who = f" ({acc['user']})" if acc.get("user") else ""
        return {"ok": True, "quota": None, "balance": acc.get("balance"), "user": acc.get("user"),
                "detail": f"{self.name} key saved{who}{credit}{running}. {self.billing_note}"}

    billing_note = "Billed by the second while it works."

    def estimate(self, package):
        return {"gpu_hours": package.gpu_hours, "cost": None, "fits": True, "note": ""}

    def price(self, gpu):
        """$/h for this GPU right now (None when the service doesn't say)."""
        return next((g.get("price") for g in self.gpus() if g["id"] == gpu), None)

    def url(self, run_id):
        return self.console

    def finished(self, run_id):
        """The results are back: free what the job used (machines, stored files)."""

    def delete(self, run_id):
        """Stop the job now and free what it used (the book page's Stop)."""

    def cancel(self, run_id):
        self.delete(run_id)


class RentedMachine(PaidCloud):
    """A pay-by-the-second GPU machine rented through a service's API (see the module's notes)."""
    live = True  # the lines can be copied back over SSH while it works
    user, home = "root", "/root"  # SSH user on the machine, and the folder the jobs go in
    needs_key_on_machine = False  # self_delete() needs the API key on the machine (kept in a file only root reads)
    billing_note = "Machines are billed by the second."

    def rent(self, gpu, name, public_key, log):
        """Rent one machine. Returns {"machine": its id, "price": $/h or None}."""
        raise NotImplementedError

    def machine(self, mid):
        """{"state": "starting" | "running" | "gone", "host", "port"} (host and port once it's reachable)."""
        raise NotImplementedError

    def remove(self, mid):
        """Delete the machine (no error when it's gone already)."""
        raise NotImplementedError

    def self_delete(self, mid):
        """Python (after SELF_DELETE_HEAD) the machine runs to delete itself; the key is in ~/.narrator-key."""
        return ""

    def submit(self, package, log=print):
        run_id = f"{package.slug or package.model}-{time.strftime('%Y%m%d-%H%M%S')}"
        gpu = package.gpu if package.gpu in self.gpu_names else self.default_gpu
        label = self.gpu_names.get(gpu, gpu)
        limit_h = float(package.meta.get("timeout_hours", 3))
        _, public = ssh_key()
        log(f"Renting a {label} on {self.name}…")
        m = self.rent(gpu, f"narrator-{run_id}"[:60], public, log)
        self._save_run(run_id, {"machine": m["machine"], "price": m.get("price"), "gpu": gpu, "started": time.time()})
        try:
            host, port = self._wait_for_ssh(m["machine"], log)
            self._save_run(run_id, {"host": host, "port": port})
            box = self._ssh(host, port)
            rdir = f"{self.home}/narrator/{run_id}"
            log(f"Copying the job to the machine ({size_mb(package.data_dir):,.0f} MB)…")
            box.put(package.data_dir, f"{rdir}/data")
            if self.needs_key_on_machine:
                box.write(f"{self.home}/.narrator-key", self.key(), private=True)
            box.write(f"{rdir}/selfdelete.py", SELF_DELETE_HEAD + (self.self_delete(m["machine"]) or "pass\n"))
            pip_line = ('$PY -m pip install -q --upgrade pip && '
                        f'$PY -m pip install -q {" ".join(shlex.quote(p) for p in package.pip)}' if package.pip else "true")
            box.write(f"{rdir}/run.sh", RUNNER.format(
                service=self.name, limit=int((limit_h + 0.5) * 3600 + GRACE_MINUTES * 60), grace=GRACE_MINUTES,
                pip_line=pip_line, script=package.script, args=package.args))
            box.sh(f"cd {rdir} && chmod +x run.sh && (nohup ./run.sh >/dev/null 2>&1 & echo $! > pid)")
        except BaseException:
            self.delete(run_id)  # a machine that never got its job mustn't keep costing money
            raise
        price = m.get("price")
        log(f"Started on a {self.name} {label}" + (f" (${price:.2f}/h)" if price else ""))
        return run_id

    def _ssh(self, host, port):
        return MachineSSH(host, port, self.user, f"the {self.name} machine")

    def _wait_for_ssh(self, mid, log, minutes=20):
        ends, said = time.time() + minutes * 60, False
        while time.time() < ends:
            info = self.machine(mid)
            if info["state"] == "gone":
                raise RuntimeError(f"{self.name} took the machine back before it started.")
            if info.get("host") and info.get("port"):
                try:
                    if self._ssh(info["host"], info["port"]).sh("echo ok", check=False, timeout=20).stdout.strip() == "ok":
                        return info["host"], info["port"]
                except (AuthError, subprocess.TimeoutExpired):
                    pass
            if not said:
                log("Waiting for the machine to start (usually a minute or a few)…")
                said = True
            time.sleep(10)
        raise RuntimeError(f"The {self.name} machine didn't start within {minutes} minutes.")

    def _box(self, run_id):
        r = self._runs().get(run_id) or {}
        if not r.get("host"):
            raise RuntimeError("This job's machine isn't known to the app any more.")
        return self._ssh(r["host"], r["port"]), f"{self.home}/narrator/{run_id}"

    def status(self, run_id):
        from ... import conn_issues
        r = self._runs().get(run_id)
        if not r:
            return RunStatus("unknown", "not found")
        try:
            info = self.machine(r["machine"])
        except AuthError:
            raise
        except Exception as e:  # noqa: BLE001 - propagate network errors so reconnect & conn_issues fire
            if conn_issues.is_network(e):
                raise
            return RunStatus("unknown", str(e)[:200])
        if info["state"] == "gone":
            return RunStatus("error", f"the {self.name} machine is gone (deleted before its results were collected)")
        box, rdir = self._box(run_id)
        try:
            out = box.sh(f"cd {rdir} 2>/dev/null || {{ echo NOTFOUND; exit 0; }}; "
                         "if [ -f exit_code ]; then echo EXIT $(cat exit_code); "
                         "elif kill -0 $(cat pid 2>/dev/null) 2>/dev/null; then echo RUNNING; else echo DIED; fi; "
                         "tail -1 out/run.log 2>/dev/null", check=False, timeout=30).stdout.splitlines() or ["?"]
        except (AuthError, subprocess.TimeoutExpired) as e:
            return RunStatus("running", f"couldn't connect just now ({str(e)[:80]})")
        head, detail = out[0], (out[1] if len(out) > 1 else "")
        if head.startswith("EXIT"):
            return RunStatus("complete" if head.split()[-1] == "0" else "error", detail)
        return RunStatus({"RUNNING": "running", "DIED": "error", "NOTFOUND": "unknown"}.get(head, "unknown"), detail)

    def fetch(self, run_id, dest, pattern, log=print):
        box, rdir = self._box(run_id)
        r = box.sh(f"cd {rdir}/out && find . -type f | sed 's|^\\./||'", timeout=60)
        rx = re.compile(pattern)
        files = [f for f in r.stdout.splitlines() if rx.search(f)]
        log(f"Copying {len(files)} file(s) back…")
        if files:
            box.get(files, f"{rdir}/out", dest)
        return Path(dest)

    def fetch_new(self, run_id, dest, have, final=False):
        box, rdir = self._box(run_id)
        r = box.sh(f"ls {rdir}/out/lines 2>/dev/null || true", check=False, timeout=60)
        names = [n for n in r.stdout.split() if LINE.fullmatch(n) and n not in have]
        for i in range(0, len(names), 300):  # (a short command line each)
            box.get(names[i:i + 300], f"{rdir}/out/lines", dest)
        return len(names)

    def finished(self, run_id):
        """The results are back: delete the machine now (it would delete itself later, still costing money)."""
        self.delete(run_id)

    def delete(self, run_id):
        r = self._runs().get(run_id)
        if not r or r.get("deleted"):
            return
        try:
            if r.get("host"):
                self._ssh(r["host"], r["port"]).sh(f"touch {self.home}/narrator/{run_id}/collected; "
                                                   f"rm -f {self.home}/.narrator-key", check=False, timeout=20)
        except Exception:  # noqa: BLE001 - deleting it below is what matters
            pass
        try:
            self.remove(r["machine"])
        finally:
            self._save_run(run_id, {"deleted": time.time()})

    def cleanup(self):
        """Forget machines of finished or lost jobs that are gone (called now and then)."""
        for run_id, r in self._runs().items():
            if r.get("deleted"):
                continue
            try:
                if self.machine(r["machine"])["state"] == "gone":
                    self._save_run(run_id, {"deleted": time.time()})
            except Exception:  # noqa: BLE001
                continue


JOB_SCRIPT = """#!/bin/bash
# A Babble job on {service}: make a book's lines, then stop.
DATA={data}; OUT={out}
mkdir -p "$OUT"
{watchdog}
{{
  echo "== $(date) on $(hostname)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  export PIP_BREAK_SYSTEM_PACKAGES=1 PIP_ROOT_USER_ACTION=ignore
  PY=python3; command -v python3 >/dev/null 2>&1 || PY=python
  {pip_line}
  NARRATOR_DATA="$DATA" timeout {limit} $PY "$DATA/{script}" {args} --out "$OUT"
}} > "$OUT/job.log" 2>&1
echo $? > "$OUT/exit_code"
{after}
"""


def job_seconds(package, extra_hours=0.5):
    """The job's hard time limit: a little after the runner's own --max-hours."""
    return int((float(package.meta.get("timeout_hours", 3)) + extra_hours) * 3600)


def job_script(service, package, data, out, extra_hours=0.5, watchdog="", after=""):
    """The script a job service runs (HF Jobs, Lightning, Beam, Cerebrium): install, make the lines into out,
    write out/exit_code. `timeout` stops the runner at the job's limit; `watchdog` (started first, in the
    background) and `after` are shell lines for services where the machine has to stop itself."""
    pip_line = (f'$PY -m pip install -q {" ".join(shlex.quote(p) for p in package.pip)}' if package.pip else "true")
    return JOB_SCRIPT.format(service=service, data=data, out=out, pip_line=pip_line,
                             limit=job_seconds(package, extra_hours), script=package.script, args=package.args,
                             watchdog=watchdog, after=after)
