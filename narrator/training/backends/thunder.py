"""Thunder Compute (thundercompute.com): cheap rented GPUs (an RTX A6000 for about $0.35 an hour) with your own
account and API token.

The app creates an instance from Thunder's "base" template (Ubuntu with PyTorch) with its own SSH key, through
Thunder's REST API. Thunder gives a machine no key of its own, so the app leaves the user's token on it (in a
file only that machine's user can read) for the machine to delete itself; the file goes when the lines are
collected. The prices and what's free come from Thunder's public endpoints. The rest is machines.py.

Thunder's non-"native" GPUs are reached over the network, which may make speech generation slower than the
same card in the machine; the speed figures below are guesses until measured.
"""
import httpx

from ..base import AuthError
from .machines import RentedMachine

ROOT = "https://api.thundercompute.com:8443"
API = ROOT + "/v1"
GPUS = {"a6000": "RTX A6000", "l40": "L40", "a100xl": "A100 80 GB", "h100": "H100"}
DISK_GB = 100  # Thunder's smallest


class Thunder(RentedMachine):
    id, name = "thunder", "Thunder Compute"
    pricing = "pay by the minute with your Thunder Compute account (an RTX A6000 is about $0.35/h)"
    key_setting, keys_page = "thunder_key", "console.thundercompute.com → Settings → Tokens"
    console = "https://console.thundercompute.com"
    user, home = "ubuntu", "/home/ubuntu"
    gpu_names, default_gpu = GPUS, "a6000"
    speed = {"a6000": 1.5, "l40": 2.0, "a100xl": 3.0, "h100": 4.0}  # guesses (GPU over the network)
    batch = {"a6000": 32, "l40": 32, "a100xl": 32, "h100": 32}
    needs_key_on_machine = True

    def api(self, method, path, body=None, key=None, timeout=60, root=API):
        headers = {"Content-Type": "application/json"}
        if key is not False:
            headers["Authorization"] = f"Bearer {key or self.key()}"
        r = httpx.request(method, root + path, json=body, timeout=timeout, headers=headers)
        if r.status_code in (401, 403):
            raise AuthError(f"Thunder Compute didn't accept the API token. {self.sign_in_help}")
        if r.status_code == 404:
            return None
        if r.status_code >= 400:
            raise RuntimeError(f"Thunder Compute: {r.text.strip()[:400] or r.status_code}")
        return r.json() if r.content else {}

    def account(self, key=None):
        self.api("GET", "/instances/list", key=key)  # (Thunder doesn't say the balance through its API)
        return {"balance": None}

    def gpus(self):
        prices = (self.api("GET", "/v2/pricing", key=False, root=ROOT) or {}).get("pricing") or {}
        specs = (self.api("GET", "/v2/specs", key=False, root=ROOT) or {}).get("specs") or {}
        free = (self.api("GET", "/v2/status", key=False, root=ROOT) or {}).get("specs") or {}
        out = []
        for gid, label in GPUS.items():
            spec = specs.get(f"{gid}_x1") or {}
            out.append({"id": gid, "label": label, "gb": spec.get("vramGB"),
                        "price": prices.get(f"{gid}_x1", prices.get(gid)),
                        "stock": "none" if free.get(f"{gid}_x1") == "unavailable" else None})
        return out

    def rent(self, gpu, name, public_key, log):
        spec = ((self.api("GET", "/v2/specs", key=False, root=ROOT) or {}).get("specs") or {}).get(f"{gpu}_x1") or {}
        cores = min(spec.get("vcpuOptions") or [8])
        disk = max(DISK_GB, (spec.get("storageGB") or {}).get("min") or DISK_GB)
        r = self.api("POST", "/instances/create", {"gpu_type": gpu, "num_gpus": 1, "cpu_cores": cores,
                                                   "disk_size_gb": disk, "template": "base", "public_key": public_key})
        if not r or r.get("identifier") is None:
            raise RuntimeError(f"Thunder Compute didn't create the machine: {str(r)[:200]}")
        price = None
        try:
            price = next((g["price"] for g in self.gpus() if g["id"] == gpu), None)
        except Exception:  # noqa: BLE001 - only for the log line
            pass
        return {"machine": str(r["identifier"]), "uuid": r.get("uuid"), "price": price}

    def _find(self, mid):
        items = self.api("GET", "/instances/list") or {}
        if str(mid) in items:
            return items[str(mid)]
        return next((v for v in items.values() if str(v.get("id")) == str(mid) or v.get("uuid") == mid), None)

    def machine(self, mid):
        m = self._find(mid)
        status = str((m or {}).get("status") or "").lower()
        if not m or status in ("deleted", "deleting", "terminated", "terminating", "failed", "stopped"):
            return {"state": "gone"}
        if m.get("ip") and m.get("port") and status in ("running", "ready", "active", ""):
            return {"state": "running", "host": m["ip"], "port": m["port"]}
        return {"state": "starting"}

    def remove(self, mid):
        self.api("POST", f"/instances/{mid}/delete")

    def self_delete(self, mid):
        return ('key = open(os.path.expanduser("~/.narrator-key")).read().strip()\n'
                f'call("POST", "{API}/instances/{mid}/delete", {{"Authorization": "Bearer " + key}})\n')
