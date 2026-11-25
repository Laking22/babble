"""Vast.ai (vast.ai): the cheapest rented GPUs, from a marketplace of hosts, with your own account and API key.

The app searches the marketplace for the chosen GPU (verified hosts, reliable, enough disk, a new enough CUDA),
rents the cheapest on demand (not an interruptible bid), with PyTorch's own image and SSH. A Vast machine has a
key for itself (CONTAINER_API_KEY) that it deletes itself with. Machines belong to individual hosts, so one can
drop out mid-book; the lines made before that still come back. The rest is machines.py.
"""
import httpx

from ..base import AuthError
from .machines import RentedMachine

API = "https://console.vast.ai/api/v0"
IMAGE = "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime"  # Python 3.11 and PyTorch
DISK_GB = 60
GPUS = {"RTX 4090": "RTX 4090", "RTX 3090": "RTX 3090", "RTX 5090": "RTX 5090", "RTX A5000": "RTX A5000",
        "RTX A6000": "RTX A6000", "L40S": "L40S", "A100 SXM4": "A100"}


def _offers_query(names, limit):
    return {"gpu_name": {"in": names}, "num_gpus": {"eq": 1}, "rentable": {"eq": True}, "verified": {"eq": True},
            "reliability": {"gte": 0.97}, "disk_space": {"gte": DISK_GB}, "cuda_max_good": {"gte": 12.4},
            "type": "ondemand", "order": [["dph_total", "asc"]], "limit": limit}


class Vast(RentedMachine):
    id, name = "vast", "Vast.ai"
    pricing = "pay by the second with your Vast.ai account (an RTX 4090 is about $0.30-0.55/h)"
    key_setting, keys_page = "vast_key", "cloud.vast.ai → Keys"
    console = "https://cloud.vast.ai/instances/"
    gpu_names, default_gpu = GPUS, "RTX 4090"
    speed = {"RTX 4090": 3.5, "RTX 3090": 2.8, "RTX 5090": 5.0, "RTX A5000": 2.0, "RTX A6000": 2.3, "L40S": 3.5,
             "A100 SXM4": 5.0}
    batch = {"RTX 5090": 24, "RTX A6000": 32, "L40S": 32, "A100 SXM4": 32}

    def api(self, method, path, body=None, key=None, timeout=60):
        headers = {"Content-Type": "application/json"}
        if key is not False:  # (the marketplace search needs no key)
            headers["Authorization"] = f"Bearer {key or self.key()}"
        r = httpx.request(method, API + path, json=body, timeout=timeout, headers=headers)
        if r.status_code in (401, 403):
            raise AuthError(f"Vast.ai didn't accept the API key. {self.sign_in_help}")
        if r.status_code == 404:
            return None
        if r.status_code >= 400:
            raise RuntimeError(f"Vast.ai: {r.text.strip()[:400] or r.status_code}")
        return r.json() if r.content else {}

    def account(self, key=None):
        u = self.api("GET", "/users/current/", key=key) or {}
        credit = u.get("credit", u.get("balance"))
        return {"balance": float(credit) if credit is not None else None, "user": u.get("username") or None}

    def gpus(self):
        """The cheapest reliable on-demand offer of each GPU right now."""
        offers = (self.api("POST", "/bundles/", _offers_query(list(GPUS), 400), key=False) or {}).get("offers") or []
        best = {}
        for o in offers:
            n = o.get("gpu_name")
            if n in GPUS and (n not in best or (o.get("dph_total") or 99) < (best[n].get("dph_total") or 99)):
                best[n] = o
        out = []
        for n, label in GPUS.items():
            o = best.get(n)
            out.append({"id": n, "label": label, "gb": round((o or {}).get("gpu_ram", 0) / 1024) or None,
                        "price": round(o["dph_total"], 3) if o and o.get("dph_total") else None,
                        "stock": None if o else "none"})
        return out

    def rent(self, gpu, name, public_key, log):
        # the account's SSH keys go into every new machine; the app's own key is added once
        try:
            self.api("POST", "/ssh/", {"ssh_key": public_key})
        except RuntimeError:  # (already added)
            pass
        offers = (self.api("POST", "/bundles/", _offers_query([gpu], 6)) or {}).get("offers") or []
        if not offers:
            raise RuntimeError(f"No {GPUS.get(gpu, gpu)} is free on Vast.ai right now. Try another GPU, or later.")
        last = None
        for o in offers:  # offers come and go: the next one when another renter was faster
            try:
                r = self.api("PUT", f"/asks/{o['id']}/", {
                    "image": IMAGE, "disk": DISK_GB, "runtype": "ssh_direct", "label": name, "cancel_unavail": True,
                    # makes the machine's own settings (its API key and id) visible to SSH sessions
                    "onstart": "env >> /etc/environment"}) or {}
            except RuntimeError as e:
                last = e
                continue
            mid = r.get("new_contract")
            if not mid:
                last = RuntimeError(f"Vast.ai: {str(r)[:200]}")
                continue
            try:
                self.api("POST", f"/instances/{mid}/ssh/", {"ssh_key": public_key})
            except RuntimeError:  # (the account's key is in it already)
                pass
            return {"machine": mid, "price": o.get("dph_total")}
        raise last or RuntimeError("Vast.ai didn't rent a machine.")

    def machine(self, mid):
        r = self.api("GET", f"/instances/{mid}/") or {}
        m = r.get("instances") or {}
        if isinstance(m, list):
            m = m[0] if m else {}
        if not m or m.get("actual_status") in ("exited", "offline") or m.get("intended_status") == "stopped":
            return {"state": "gone"}
        if m.get("actual_status") != "running":
            return {"state": "starting"}
        direct = ((m.get("ports") or {}).get("22/tcp") or [{}])[0].get("HostPort") if isinstance(m.get("ports"), dict) else None
        if m.get("public_ipaddr") and direct:
            return {"state": "running", "host": m["public_ipaddr"].strip(), "port": direct}
        if m.get("ssh_host") and m.get("ssh_port"):  # through Vast's SSH proxy
            return {"state": "running", "host": m["ssh_host"], "port": m["ssh_port"]}
        return {"state": "starting"}

    def remove(self, mid):
        self.api("DELETE", f"/instances/{mid}/")

    def self_delete(self, mid):
        return (f'call("DELETE", "{API}/instances/" + (env("CONTAINER_ID") or "{mid}") + "/", '
                '{"Authorization": "Bearer " + env("CONTAINER_API_KEY")})\n')
