"""Lambda (lambda.ai): rented cloud GPUs (one A10, A6000, A100 or H100) with your own Lambda account and API key.

The app adds its SSH key to the account once, launches an instance of the chosen type in a region that has one
free (Lambda Stack: Ubuntu with PyTorch), and terminates it when the lines are back, through Lambda's Cloud
API. Lambda gives a machine no key of its own, so the app leaves the user's key on it (in a file only that
machine's user can read) for the machine to terminate itself; the file goes when the lines are collected.
Lambda bills the card on the account (there's no prepaid balance to show). The rest is machines.py.
"""
import httpx

from ..base import AuthError
from .machines import RentedMachine

API = "https://cloud.lambda.com/api/v1"
SSH_KEY_NAME = "narrator-studio"
# single-GPU x86 types (the GH200 is ARM: PyTorch wheels differ); prices and free regions come live from Lambda
GPUS = {"gpu_1x_a10": "A10", "gpu_1x_a6000": "RTX A6000", "gpu_1x_a100_sxm4": "A100 40 GB",
        "gpu_1x_h100_pcie": "H100"}


class Lambda(RentedMachine):
    id, name = "lambda", "Lambda"
    pricing = "pay by the minute with your Lambda account (an A10 is about $0.75/h)"
    key_setting, keys_page = "lambda_key", "cloud.lambda.com → API keys"
    console = "https://cloud.lambda.com/instances"
    user, home = "ubuntu", "/home/ubuntu"
    gpu_names, default_gpu = GPUS, "gpu_1x_a10"
    speed = {"gpu_1x_a10": 2.5, "gpu_1x_a6000": 2.3, "gpu_1x_a100_sxm4": 5.0, "gpu_1x_h100_pcie": 6.0}
    batch = {"gpu_1x_a6000": 32, "gpu_1x_a100_sxm4": 32, "gpu_1x_h100_pcie": 32}
    needs_key_on_machine = True

    def api(self, method, path, body=None, key=None, timeout=60):
        r = httpx.request(method, API + path, json=body, timeout=timeout,
                          headers={"Authorization": f"Bearer {key or self.key()}", "Content-Type": "application/json"})
        if r.status_code in (401, 403):
            raise AuthError(f"Lambda didn't accept the API key. {self.sign_in_help}")
        if r.status_code == 404:
            return None
        if r.status_code >= 400:
            try:
                msg = (r.json().get("error") or {}).get("message") or r.text
            except ValueError:
                msg = r.text
            raise RuntimeError(f"Lambda: {str(msg).strip()[:400]}")
        return (r.json() if r.content else {}).get("data")

    def account(self, key=None):
        self.api("GET", "/instance-types", key=key)  # a key check: Lambda has no balance to read
        return {"balance": None}

    def _types(self):
        return self.api("GET", "/instance-types") or {}

    def gpus(self):
        types = self._types()
        out = []
        for tid, label in GPUS.items():
            t = types.get(tid) or {}
            it = t.get("instance_type") or {}
            cents = it.get("price_cents_per_hour")
            out.append({"id": tid, "label": label, "gb": None, "price": cents / 100 if cents else None,
                        "stock": None if t.get("regions_with_capacity_available") else "none"})
        return out

    def rent(self, gpu, name, public_key, log):
        mine = next((k for k in self.api("GET", "/ssh-keys") or [] if k.get("name") == SSH_KEY_NAME), None)
        if mine and (mine.get("public_key") or "").split()[:2] != public_key.split()[:2]:  # an older key of the app's
            self.api("DELETE", f"/ssh-keys/{mine['id']}")
            mine = None
        if not mine:
            self.api("POST", "/ssh-keys", {"name": SSH_KEY_NAME, "public_key": public_key})
        t = self._types().get(gpu) or {}
        regions = [r["name"] for r in t.get("regions_with_capacity_available") or []]
        if not regions:
            raise RuntimeError(f"No {GPUS.get(gpu, gpu)} is free on Lambda right now. Try another GPU, or later.")
        r = self.api("POST", "/instance-operations/launch", {"region_name": regions[0], "instance_type_name": gpu,
                                                             "ssh_key_names": [SSH_KEY_NAME], "name": name})
        ids = (r or {}).get("instance_ids") or []
        if not ids:
            raise RuntimeError(f"Lambda didn't launch the machine: {str(r)[:200]}")
        cents = (t.get("instance_type") or {}).get("price_cents_per_hour")
        return {"machine": ids[0], "price": cents / 100 if cents else None}

    def machine(self, mid):
        m = self.api("GET", f"/instances/{mid}") or {}
        status = m.get("status")
        if not m or status in ("terminated", "terminating", "preempted"):
            return {"state": "gone"}
        if status == "active" and m.get("ip"):
            return {"state": "running", "host": m["ip"], "port": 22}
        return {"state": "starting"}

    def remove(self, mid):
        self.api("POST", "/instance-operations/terminate", {"instance_ids": [mid]})

    def self_delete(self, mid):
        return ('key = open(os.path.expanduser("~/.narrator-key")).read().strip()\n'
                f'call("POST", "{API}/instance-operations/terminate", {{"Authorization": "Bearer " + key}}, '
                f'{{"instance_ids": ["{mid}"]}})\n')
