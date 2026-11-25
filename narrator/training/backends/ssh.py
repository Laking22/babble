"""Any Linux machine with an NVIDIA GPU that you can reach over SSH: your own PC, or a rented pod
(RunPod, Vast.ai, Lambda… they all give you an SSH address). Free on your own hardware.

Settings (Train tab → Train on → My GPU machine): ssh_host ("user@1.2.3.4"), ssh_port, ssh_key (optional path),
ssh_dir (remote working folder). The machine needs Python 3.10+, an NVIDIA driver and internet access for pip.
Host "local" runs everything on this computer instead (for a Linux PC running the app itself, and for tests).

Per run: the package is copied to <ssh_dir>/<run>/data, a run.sh installs the pip packages into a venv
(sharing the system's torch) and starts the script with nohup; status reads exit_code / pid; fetch copies
matching files back from <run>/out.
"""
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from ... import settings
from ..base import AuthError, Backend, RunStatus

RUNNER = """#!/bin/bash
cd "$(dirname "$0")"
mkdir -p out
{{
  echo "== $(date) on $(hostname)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)"
  VENV="{venv}"
  if [ ! -x "$VENV/bin/python" ]; then {python} -m venv --system-site-packages "$VENV"; fi
  "$VENV/bin/python" -m pip install -q --upgrade pip
  {pip_line}
  NARRATOR_DATA="$PWD/data" "$VENV/bin/python" data/{script} {args} --out "$PWD/out"
}} > out/run.log 2>&1
echo $? > exit_code
"""


class SSH(Backend):
    id, name, readiness = "ssh", "My GPU machine (SSH)", "ready"
    pricing = "your own NVIDIA PC, or any rented GPU with an SSH address (RunPod, Vast.ai…)"

    # ---------- transport ----------
    def _cfg(self):
        c = settings.load()
        return {"host": (c.get("ssh_host") or "").strip(), "port": str(c.get("ssh_port") or "22"),
                "key": (c.get("ssh_key") or "").strip(), "dir": (c.get("ssh_dir") or "~/narrator-train").strip(),
                "python": (c.get("ssh_python") or "python3").strip()}

    def _local(self):
        return self._cfg()["host"] == "local"

    def _ssh_base(self):
        c = self._cfg()
        host = c["host"]
        if not host:
            raise AuthError("Set the machine's SSH address first (user@host).")
        if host.startswith("-") or any(ch.isspace() for ch in host):
            raise ValueError("Invalid SSH host address.")
        try:
            port_num = int(c["port"])
            if not (1 <= port_num <= 65535):
                raise ValueError
        except ValueError:
            raise ValueError("SSH port must be a number between 1 and 65535.") from None
        opts = ["-p", str(port_num), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=accept-new"]
        if c["key"]:
            opts += ["-i", str(Path(c["key"]).expanduser())]
        return opts, host

    def sh(self, command, check=True, timeout=120):
        """Run a shell command on the machine. Returns CompletedProcess."""
        if self._local():
            r = subprocess.run(["bash", "-lc", command], capture_output=True, text=True, timeout=timeout)
        else:
            opts, host = self._ssh_base()
            r = subprocess.run(["ssh", *opts, "--", host, command], capture_output=True, text=True, timeout=timeout)
            if r.returncode == 255:
                raise AuthError(f"Couldn't connect over SSH: {(r.stderr or '').strip()[-300:]}")
        if check and r.returncode:
            raise RuntimeError(f"Remote command failed ({r.returncode}): {(r.stderr or r.stdout).strip()[-400:]}")
        return r

    def _remote_dir(self, run_id=""):
        base = self._cfg()["dir"]
        if self._local():
            base = str(Path(base).expanduser())
        return f"{base.rstrip('/')}/{run_id}" if run_id else base.rstrip("/")

    def _q(self, path):
        """Quote a remote path safely while still allowing a leading ~ to expand."""
        if path == "~":
            return "~"
        if path.startswith("~/"):
            return "~/" + shlex.quote(path[2:])
        return shlex.quote(path)

    def put(self, local_dir, remote_dir):
        local_dir = str(local_dir).rstrip("/") + "/"
        self.sh(f"mkdir -p {self._q(remote_dir)}")
        if self._local():
            shutil.copytree(local_dir, Path(remote_dir), dirs_exist_ok=True)
            return
        opts, host = self._ssh_base()
        if shutil.which("rsync"):
            subprocess.run(["rsync", "-az", "--partial", "-e", "ssh " + " ".join(shlex.quote(o) for o in opts),
                            local_dir, f"{host}:{remote_dir}/"], check=True)
        else:
            scp_opts = ["-P" if o == "-p" else o for o in opts]
            subprocess.run(["scp", "-r", *scp_opts, *[str(p) for p in Path(local_dir).iterdir()], f"{host}:{remote_dir}/"],
                           check=True)

    def get(self, remote_files, remote_root, dest):
        dest = Path(dest)
        if self._local():
            for rel in remote_files:
                (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(Path(remote_root) / rel, dest / rel)
            return
        opts, host = self._ssh_base()
        dest.mkdir(parents=True, exist_ok=True)
        listing = dest / ".fetch-list"
        listing.write_text("\n".join(remote_files) + "\n")
        subprocess.run(["rsync", "-az", "--partial", "--files-from", str(listing),
                        "-e", "ssh " + " ".join(shlex.quote(o) for o in opts), f"{host}:{remote_root}/", str(dest) + "/"],
                       check=True)
        listing.unlink()

    # ---------- Backend ----------
    def check(self):
        c = self._cfg()
        if not c["host"]:
            return {"ok": False, "detail": "Add the machine's SSH address (user@host) in the settings below.", "quota": None}
        try:
            r = self.sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null; "
                        f"{shlex.quote(c['python'])} --version 2>&1; df -h {self._q(self._remote_dir())} 2>/dev/null | tail -1 || df -h ~ | tail -1",
                        check=False, timeout=30)
        except (AuthError, ValueError) as e:
            return {"ok": False, "detail": str(e), "quota": None}
        lines = [x for x in r.stdout.splitlines() if x.strip()]
        gpu = next((x for x in lines if "MiB" in x), None)
        py = next((x for x in lines if x.startswith("Python")), "no python")
        where = "this computer" if self._local() else c["host"]
        if not gpu:
            return {"ok": False, "quota": None, "detail": f"Connected to {where} ({py}) but found no NVIDIA GPU."}
        return {"ok": True, "quota": None, "detail": f"{where}: {gpu.strip()}, {py}"}

    def estimate(self, package):
        return {"gpu_hours": package.gpu_hours, "cost": None, "fits": True,
                "note": "Times are for a T4; most desktop GPUs (RTX 3060 and up) are faster."}

    def submit(self, package, log=print):
        run_id = f"{package.model}-{package.recipe}-{time.strftime('%Y%m%d-%H%M%S')}"
        rdir = self._remote_dir(run_id)
        log(f"Copying {package.hours:.1f} h of training data to {self._cfg()['host']}…")
        self.put(package.data_dir, f"{rdir}/data")
        c = self._cfg()
        pip_line = (f'"$VENV/bin/python" -m pip install -q {" ".join(shlex.quote(p) for p in package.pip)}'
                    if package.pip else "true")
        script = RUNNER.format(venv=self._q(f"{self._remote_dir()}/venv-{package.recipe}"), python=shlex.quote(c["python"]),
                               pip_line=pip_line, script=shlex.quote(package.script), args=package.args)
        local_tmp = Path(package.data_dir).parent / "run.sh"
        local_tmp.write_text(script)
        if self._local():
            shutil.copy2(local_tmp, Path(rdir) / "run.sh")
        else:
            opts, host = self._ssh_base()
            scp_opts = ["-P" if o == "-p" else o for o in opts]
            subprocess.run(["scp", *scp_opts, "--", str(local_tmp), f"{host}:{rdir}/run.sh"], check=True)
        self.sh(f"cd {self._q(rdir)} && chmod +x run.sh && (setsid nohup ./run.sh >/dev/null 2>&1 < /dev/null & echo $! > pid)")
        log(f"Started {run_id}")
        return run_id

    def url(self, run_id):
        return ""

    def status(self, run_id):
        rdir = self._q(self._remote_dir(run_id))
        try:
            r = self.sh(f"cd {rdir} 2>/dev/null || {{ echo NOTFOUND; exit 0; }}; "
                        "if [ -f exit_code ]; then echo EXIT $(cat exit_code); "
                        "elif kill -0 $(cat pid 2>/dev/null) 2>/dev/null; then echo RUNNING; else echo DIED; fi; "
                        "tail -1 out/run.log 2>/dev/null", check=False, timeout=30)
        except AuthError as e:
            return RunStatus("unknown", str(e))
        lines = r.stdout.splitlines() or ["?"]
        head, detail = lines[0], (lines[1] if len(lines) > 1 else "")
        if head.startswith("EXIT"):
            return RunStatus("complete" if head.split()[-1] == "0" else "error", detail)
        return RunStatus({"RUNNING": "running", "DIED": "error", "NOTFOUND": "unknown"}.get(head, "unknown"), detail)

    def fetch(self, run_id, dest, pattern, log=print):
        root = self._remote_dir(run_id) + "/out"
        r = self.sh(f"cd {self._q(root)} && find . -type f | sed 's|^\\./||'", timeout=60)
        rx = re.compile(pattern)
        files = [f for f in r.stdout.splitlines() if rx.search(f)]
        log(f"Copying {len(files)} file(s) back…")
        if files:
            self.get(files, root, dest)
        return Path(dest)

    def cancel(self, run_id):
        self.sh(
            f"cd {self._q(self._remote_dir(run_id))} && "
            'p=$(cat pid 2>/dev/null); '
            '[ -n "$p" ] && kill -TERM -- -"$p" "$p" 2>/dev/null; '
            'sleep 1; '
            '[ -n "$p" ] && kill -KILL -- -"$p" "$p" 2>/dev/null; '
            "echo 130 > exit_code",
            check=False,
        )
