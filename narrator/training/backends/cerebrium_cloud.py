"""Cerebrium (cerebrium.ai): serverless GPUs billed by the second, with your own Cerebrium account and a service
account token, through Cerebrium's REST API (what its `cerebrium run` command uses).

The job (the runner, the lines and any cloned voices' samples: a few MB) is sent as the code of one run on a GPU
of the chosen type, in an app called narrator-lines that's made in your project the first time. The run writes
its lines to the project's persistent storage (/persistent-storage/narrator/<run>), which the app downloads,
then deletes. Voices trained on this Mac aren't sent (gigabytes per voice); Finish makes their lines here.

Written from Cerebrium's API reference; it hasn't been run on Cerebrium yet.
"""
import io
import tarfile
import time
from pathlib import Path

import httpx

from ..base import AuthError, RunStatus
from .machines import PaidCloud, job_script

API = "https://rest.cerebrium.ai"
APP = "narrator-lines"
# Cerebrium's compute types, with its list price per hour
GPUS = {"ADA_L4": "L4", "TURING_T4": "T4", "AMPERE_A10": "A10", "ADA_L40": "L40S", "AMPERE_A100_40GB": "A100",
        "HOPPER_H100": "H100"}
PRICES = {"ADA_L4": 0.80, "TURING_T4": 0.59, "AMPERE_A10": 1.10, "ADA_L40": 1.95, "AMPERE_A100_40GB": 2.00,
          "HOPPER_H100": 3.40}
MAIN = '''import subprocess

def run():
    """Make the book's lines (data/job.sh writes them to persistent storage)."""
    r = subprocess.run(["bash", "data/job.sh"])
    return {"exit_code": r.returncode}
'''


class Cerebrium(PaidCloud):
    id, name = "cerebrium", "Cerebrium"
    pricing = "pay by the second with your Cerebrium account (an L4 is about $0.80/h)"
    key_setting, keys_page = "cerebrium_token", "dashboard.cerebrium.ai → API Keys"
    console = "https://dashboard.cerebrium.ai"
    gpu_names, default_gpu = GPUS, "ADA_L4"
    speed = {"ADA_L4": 1.5, "TURING_T4": 1.2, "AMPERE_A10": 2.5, "ADA_L40": 3.5, "AMPERE_A100_40GB": 5.0,
             "HOPPER_H100": 6.0}
    batch = {"ADA_L40": 32, "AMPERE_A100_40GB": 32, "HOPPER_H100": 32}
    sends_models = False
    extra_pip = ["torch"]  # Cerebrium's standard image has Python, not PyTorch
    billing_note = "Billed by the second while the run works."

    def api(self, method, path, key=None, timeout=60, **kw):
        r = httpx.request(method, API + path, timeout=timeout,
                          headers={"Authorization": f"Bearer {key or self.key()}"}, **kw)
        if r.status_code in (401, 403):
            raise AuthError(f"Cerebrium didn't accept the token. {self.sign_in_help}")
        if r.status_code == 404:
            return None
        if r.status_code >= 400:
            raise RuntimeError(f"Cerebrium: {r.text.strip()[:400] or r.status_code}")
        return r.json() if r.content else {}

    def parse_key(self, text):
        key = (text or "").strip().removeprefix("Bearer ").strip()
        if len(key) < 40 or " " in key:
            raise ValueError("That doesn't look like a Cerebrium service account token. Make one at "
                             "dashboard.cerebrium.ai → API Keys → Create Service Account.")
        return key

    def _project(self, key=None):
        ps = self.api("GET", "/v2/projects", key=key) or []
        ps = ps.get("items", ps) if isinstance(ps, dict) else ps
        if not ps:
            raise RuntimeError("Your Cerebrium account has no project yet: open dashboard.cerebrium.ai once to make one.")
        return ps[0]

    def account(self, key=None):
        p = self._project(key)
        return {"balance": None, "user": p.get("name") or p.get("id")}

    def gpus(self):
        return [{"id": g, "label": label, "gb": None, "price": PRICES.get(g), "stock": None} for g, label in GPUS.items()]

    def _volume(self, pid):
        vs = self.api("GET", f"/v2/projects/{pid}/volumes") or []
        vs = vs.get("items", vs) if isinstance(vs, dict) else vs
        if not vs:
            raise RuntimeError("Cerebrium has no persistent storage in this project.")
        return vs[0]

    def submit(self, package, log=print):
        run_id = f"{package.slug or package.model}-{time.strftime('%Y%m%d-%H%M%S')}"
        gpu = package.gpu if package.gpu in GPUS else self.default_gpu
        pid = self._project()["id"]
        vol = self._volume(pid)
        out = f"/persistent-storage/narrator/{run_id}/out"
        data = Path(package.data_dir)
        (data / "job.sh").write_text(job_script(self.name, package, '"$PWD/data"', out))
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as t:
            t.add(data, arcname="data", filter=lambda m: None if "/models-mlx" in m.name else m)
            info = tarfile.TarInfo("main.py")
            info.size = len(MAIN.encode())
            t.addfile(info, io.BytesIO(MAIN.encode()))
        app_id = f"{pid}-{APP}"
        self.api("POST", f"/v3/projects/{pid}/apps/{app_id}/create-run-app")
        log(f"Sending the job to Cerebrium ({len(buf.getvalue()) / 1e6:,.1f} MB)…")
        r = self.api("POST", f"/v3/projects/{pid}/apps/{app_id}/run", timeout=600,
                     params={"appName": APP, "filename": "main.py", "functionName": "run", "computeType": gpu,
                             "gpuCount": 1, "cpu": 4, "memoryGb": 32},
                     files={"file": ("code.tar.gz", buf.getvalue(), "application/gzip")}, data={"data": "{}"}) or {}
        if not r.get("runId"):
            raise RuntimeError(f"Cerebrium didn't start the run: {str(r)[:200]}")
        self._save_run(run_id, {"run": r["runId"], "project": pid, "app": app_id, "volume": vol.get("id"),
                                "region": vol.get("region"), "gpu": gpu, "started": time.time()})
        log(f"Started on a Cerebrium {GPUS[gpu]}")
        return run_id

    def _r(self, run_id):
        r = self._runs().get(run_id)
        if not r:
            raise RuntimeError("This job isn't known to the app any more.")
        return r

    def _files(self, r, run_id, sub=""):
        """The run's output files (paths relative to its out folder)."""
        base = f"narrator/{run_id}/out"
        items = self.api("GET", f"/v2/projects/{r['project']}/volumes/{r['volume']}/ls",
                         params={"dir": f"{base}/{sub}".rstrip("/"), "region": r.get("region") or "us-east-1"}) or []
        items = items.get("items", items) if isinstance(items, dict) else items
        out = []
        for f in items:
            name = f"{sub}{f['name']}"
            out += self._files(r, run_id, name + "/") if f.get("is_folder") else [name]
        return out

    def _download(self, r, path, dest):
        u = self.api("GET", f"/v2/projects/{r['project']}/volumes/{r['volume']}/download",
                     params={"file_path": path, "region": r.get("region") or "us-east-1"}) or {}
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        with httpx.stream("GET", u["url"], timeout=600) as resp, open(dest, "wb") as f:
            resp.raise_for_status()
            for chunk in resp.iter_bytes():
                f.write(chunk)

    def status(self, run_id):
        try:
            r = self._r(run_id)
            item = (self.api("GET", f"/v2/projects/{r['project']}/apps/{r['app']}/runs/{r['run']}") or {}).get("item") or {}
        except AuthError as e:
            return RunStatus("unknown", str(e))
        except Exception as e:  # noqa: BLE001 - asked again later
            return RunStatus("unknown", str(e)[:200])
        st = str(item.get("status") or "").lower()
        if st in ("queued", "pending", "scheduled", "building"):
            return RunStatus("queued", st)
        if st in ("running", "processing", "in_progress", "started", ""):
            return RunStatus("running", st)
        if st in ("success", "succeeded", "completed", "complete", "done"):
            return RunStatus("complete", "")
        if st in ("cancelled", "canceled"):
            return RunStatus("cancelled", "")
        return RunStatus("error", f"Cerebrium says the run {st or 'stopped'}")

    def fetch(self, run_id, dest, pattern, log=print):
        import re
        r = self._r(run_id)
        rx = re.compile(pattern)
        files = [f for f in self._files(r, run_id) if rx.search(f)]
        log(f"Downloading {len(files)} file(s) from Cerebrium…")
        for f in files:
            self._download(r, f"narrator/{run_id}/out/{f}", Path(dest) / f)
        return Path(dest)

    def finished(self, run_id):
        r = self._runs().get(run_id)
        if not r or r.get("deleted"):
            return
        self.api("DELETE", f"/v2/projects/{r['project']}/volumes/{r['volume']}/rm",
                 params={"file_path": f"narrator/{run_id}", "region": r.get("region") or "us-east-1"})
        self._save_run(run_id, {"deleted": time.time()})

    def delete(self, run_id):
        r = self._runs().get(run_id)
        if not r:
            return
        try:
            self.api("DELETE", f"/v2/projects/{r['project']}/apps/{r['app']}/runs/{r['run']}")
        except Exception:  # noqa: BLE001 - finished already
            pass
        self.finished(run_id)
