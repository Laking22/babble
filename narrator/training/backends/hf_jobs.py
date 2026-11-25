"""Hugging Face Jobs (huggingface.co/docs/hub/jobs): run the book's lines on Hugging Face's GPUs, billed by the
minute from the credit on your Hugging Face account, with an access token that can write.

Through the huggingface_hub library the app already has: the job's files go to a private storage bucket in your
account ("jobs-artifacts", one folder per run), mounted read-only at /data; the job writes its lines to a second
folder mounted at /out, which the app downloads, then deletes both folders. Jobs need a positive credit balance
(Settings → Billing on huggingface.co); a PRO plan's monthly credit counts too.
"""
import time
from pathlib import Path

from ..base import AuthError, RunStatus
from .machines import PaidCloud, job_script

IMAGE = "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime"  # Python 3.11 and PyTorch
BUCKET = "jobs-artifacts"
GPUS = {"l4x1": "L4", "t4-small": "T4", "a10g-small": "A10G", "a10g-large": "A10G · more memory",
        "l40sx1": "L40S", "a100-large": "A100"}


class HFJobs(PaidCloud):
    id, name = "hfjobs", "Hugging Face Jobs"
    pricing = "pay by the minute from your Hugging Face credit (an L4 is $0.80/h)"
    key_setting, keys_page = "hf_jobs_token", "huggingface.co → Settings → Access Tokens (Write)"
    console = "https://huggingface.co/settings/jobs"
    gpu_names, default_gpu = GPUS, "l4x1"
    speed = {"l4x1": 1.5, "t4-small": 1.2, "a10g-small": 2.5, "a10g-large": 2.5, "l40sx1": 3.5, "a100-large": 5.0}
    batch = {"l40sx1": 32, "a100-large": 32}
    billing_note = "Jobs are billed by the minute from the credit on your Hugging Face account."

    def _api(self, key=None):
        from huggingface_hub import HfApi
        return HfApi(token=key or self.key())

    def _call(self, fn, *a, **kw):
        from huggingface_hub.errors import HfHubHTTPError
        try:
            return fn(*a, **kw)
        except HfHubHTTPError as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code in (401, 403):
                raise AuthError(f"Hugging Face didn't accept the token, or it can't run jobs (it needs Write). "
                                f"{self.sign_in_help}") from None
            if code == 402:
                raise RuntimeError("Hugging Face Jobs need credit on your account: add some at "
                                   "huggingface.co → Settings → Billing.") from None
            raise RuntimeError(f"Hugging Face: {str(e)[:300]}") from None

    def parse_key(self, text):
        key = (text or "").strip()
        if not key.startswith("hf_") or len(key) < 20:
            raise ValueError("That doesn't look like a Hugging Face token (they start with hf_). Make one with "
                             "Write access at huggingface.co → Settings → Access Tokens.")
        return key

    def account(self, key=None):
        me = self._call(self._api(key).whoami)
        role = ((me.get("auth") or {}).get("accessToken") or {}).get("role")
        if role == "read":
            raise AuthError("That's a read-only Hugging Face token; Jobs need one with Write access.")
        return {"balance": None, "user": me.get("name")}

    def gpus(self):
        from huggingface_hub import HfApi
        hw = {h.name: h for h in HfApi().list_jobs_hardware(token=False)}
        out = []
        for gid, label in GPUS.items():
            h = hw.get(gid)
            per = (h.unit_cost_usd or 0) * (60 if (h and h.unit_label == "minute") else 1) if h else None
            out.append({"id": gid, "label": label, "gb": None, "price": round(per, 2) if per else None,
                        "stock": None if h else "none"})
        return out

    def submit(self, package, log=print):
        api = self._api()
        ns = self._call(api.whoami)["name"]
        run_id = f"{package.slug or package.model}-{time.strftime('%Y%m%d-%H%M%S')}"
        gpu = package.gpu if package.gpu in GPUS else self.default_gpu
        data = Path(package.data_dir)
        (data / "job.sh").write_text(job_script(self.name, package, "/data", "/out"))
        empty = data.parent / "hf-out"
        empty.mkdir(exist_ok=True)
        log(f"Uploading the job to your Hugging Face storage ({sum(f.stat().st_size for f in data.rglob('*') if f.is_file()) / 1e6:,.0f} MB)…")
        vin = self._call(api.sync_job_volume, str(data), "/data", remote_name=f"narrator-{run_id}-data")
        vout = self._call(api.sync_job_volume, str(empty), "/out", remote_name=f"narrator-{run_id}-out", read_only=False)
        limit = int((float(package.meta.get("timeout_hours", 3)) + 0.75) * 3600)
        job = self._call(api.run_job, image=IMAGE, command=["bash", "/data/job.sh"], flavor=gpu, timeout=limit,
                         volumes=[vin, vout], labels={"app": "narrator-studio"})
        self._save_run(run_id, {"job": job.id, "ns": ns, "gpu": gpu, "url": job.url, "started": time.time()})
        log(f"Started on Hugging Face Jobs ({GPUS[gpu]})")
        return run_id

    def url(self, run_id):
        return (self._runs().get(run_id) or {}).get("url") or self.console

    def status(self, run_id):
        r = self._runs().get(run_id)
        if not r:
            return RunStatus("unknown", "not found")
        try:
            api = self._api()
            info = self._call(api.inspect_job, job_id=r["job"])
        except AuthError as e:
            return RunStatus("unknown", str(e))
        except Exception as e:  # noqa: BLE001 - asked again later
            return RunStatus("unknown", str(e)[:200])
        stage = str(getattr(info.status.stage, "value", info.status.stage))
        detail = info.status.message or ""
        if stage in ("SCHEDULING", "RUNNING"):
            try:
                tail = [ln for ln in api.fetch_job_logs(job_id=r["job"], tail=3) if ln.strip()]
                detail = tail[-1][:200] if tail else detail
            except Exception:  # noqa: BLE001 - only the detail line
                pass
            return RunStatus("queued" if stage == "SCHEDULING" else "running", detail)
        return RunStatus({"COMPLETED": "complete", "ERROR": "error"}.get(stage, "cancelled"), detail)

    def fetch(self, run_id, dest, pattern, log=print):
        r = self._runs()[run_id]
        log("Downloading from Hugging Face…")
        Path(dest).mkdir(parents=True, exist_ok=True)
        self._call(self._api().sync_bucket, f"hf://buckets/{r['ns']}/{BUCKET}/narrator-{run_id}-out", str(dest))
        return Path(dest)

    def _remove_files(self, run_id):
        r = self._runs().get(run_id) or {}
        if not r or r.get("cleaned"):
            return
        api = self._api()
        bucket = f"{r['ns']}/{BUCKET}"
        for part in ("data", "out"):
            paths = [f.path for f in api.list_bucket_tree(bucket, prefix=f"narrator-{run_id}-{part}/", recursive=True)
                     if getattr(f, "type", "file") == "file"]
            if paths:
                api.batch_bucket_files(bucket, delete=paths)
        self._save_run(run_id, {"cleaned": time.time()})

    def finished(self, run_id):
        self._remove_files(run_id)

    def delete(self, run_id):
        r = self._runs().get(run_id)
        if not r:
            return
        try:
            self._call(self._api().cancel_job, job_id=r["job"])
        except Exception:  # noqa: BLE001 - finished already
            pass
        self._remove_files(run_id)
