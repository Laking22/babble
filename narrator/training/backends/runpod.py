"""Runpod (runpod.io): rent a GPU machine by the second with your own Runpod account and API key.

Runpod's PyTorch image has SSH; the app rents a pod through Runpod's REST API, and reads the balance and live GPU
prices from its GraphQL API. The machine deletes itself with runpodctl (in every pod, with a key for that pod
only). The rest (copying the job, following it, collecting the lines, deleting the machine) is machines.py.
"""
import re

import httpx

from ..base import AuthError
from .machines import GRACE_MINUTES, RentedMachine  # noqa: F401 - GRACE_MINUTES: cloud_book reads it from here

REST = "https://rest.runpod.io/v1"
GRAPHQL = "https://api.runpod.io/graphql"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"  # Python 3.11, PyTorch, SSH ready
# GPUs offered (Runpod's id -> short label): 24 GB and up, newer than the T4; prices come live from Runpod
GPUS = {"NVIDIA GeForce RTX 4090": "RTX 4090", "NVIDIA RTX A5000": "RTX A5000", "NVIDIA GeForce RTX 3090": "RTX 3090",
        "NVIDIA L4": "L4", "NVIDIA RTX A6000": "RTX A6000", "NVIDIA L40S": "L40S"}


class Runpod(RentedMachine):
    id, name = "runpod", "Runpod"
    pricing = "pay by the second with your Runpod account (an RTX 4090 is about $0.35-0.70/h)"
    key_setting, keys_page = "runpod_key", "console.runpod.io → Credentials → API Keys"
    console = "https://console.runpod.io/pods"
    gpu_names, default_gpu = GPUS, "NVIDIA GeForce RTX 4090"
    speed = {"NVIDIA GeForce RTX 4090": 3.5, "NVIDIA RTX A5000": 2.0, "NVIDIA GeForce RTX 3090": 2.8,
             "NVIDIA L4": 1.5, "NVIDIA RTX A6000": 2.3, "NVIDIA L40S": 3.5}
    batch = {"NVIDIA RTX A6000": 32, "NVIDIA L40S": 32}

    def _headers(self, key=None):
        return {"Authorization": f"Bearer {key or self.key()}", "Content-Type": "application/json"}

    def rest(self, method, path, body=None, key=None, timeout=60):
        r = httpx.request(method, REST + path, headers=self._headers(key), json=body, timeout=timeout)
        if r.status_code in (401, 403):
            raise AuthError(f"Runpod didn't accept the API key. {self.sign_in_help}")
        if r.status_code == 404:
            return None
        if r.status_code >= 400:
            raise RuntimeError(f"Runpod: {r.text.strip()[:400] or r.status_code}")
        return r.json() if r.content else {}

    def graphql(self, query, key=None):
        r = httpx.post(GRAPHQL, headers=self._headers(key), json={"query": query}, timeout=60)
        if r.status_code in (401, 403):
            raise AuthError(f"Runpod didn't accept the API key. {self.sign_in_help}")
        data = r.json() if r.content else {}
        if data.get("errors"):
            msg = data["errors"][0].get("message", "")
            if re.search(r"unauthori|api key|authenticat", msg, re.I):
                raise AuthError(f"Runpod didn't accept the API key. {self.sign_in_help}")
            raise RuntimeError(f"Runpod: {msg[:300]}")
        return data.get("data") or {}

    def account(self, key=None):
        me = self.graphql("query { myself { clientBalance currentSpendPerHr } }", key).get("myself") or {}
        return {"balance": float(me.get("clientBalance") or 0), "spend_per_hour": float(me.get("currentSpendPerHr") or 0)}

    def gpus(self):
        """Runpod's price per hour now (Secure Cloud, on demand) and whether any are free."""
        q = ("query { gpuTypes { id displayName memoryInGb securePrice communityPrice "
             "lowestPrice(input: {gpuCount: 1}) { uninterruptablePrice stockStatus } } }")
        out = []
        for g in self.graphql(q).get("gpuTypes") or []:
            if g.get("id") not in GPUS:
                continue
            low = g.get("lowestPrice") or {}
            price = g.get("securePrice") or low.get("uninterruptablePrice") or g.get("communityPrice")
            out.append({"id": g["id"], "label": GPUS[g["id"]], "gb": g.get("memoryInGb"), "price": price,
                        "stock": (low.get("stockStatus") or "").lower() or None})
        order = list(GPUS)
        return sorted(out, key=lambda g: order.index(g["id"]))

    def rent(self, gpu, name, public_key, log):
        body = {"name": name, "imageName": IMAGE, "gpuTypeIds": [gpu], "gpuCount": 1, "cloudType": "SECURE",
                "containerDiskInGb": 80, "volumeInGb": 0, "ports": ["22/tcp"], "supportPublicIp": True,
                "env": {"PUBLIC_KEY": public_key, "SSH_PUBLIC_KEY": public_key}}
        try:
            pod = self.rest("POST", "/pods", body)
        except RuntimeError:  # none free in Secure Cloud right now: try Community Cloud (other people's machines)
            log("None free in Runpod's Secure Cloud; trying Community Cloud…")
            pod = self.rest("POST", "/pods", dict(body, cloudType="COMMUNITY"))
        return {"machine": pod["id"], "price": pod.get("costPerHr")}

    def machine(self, mid):
        p = self.rest("GET", f"/pods/{mid}") or {}
        if not p or p.get("desiredStatus") == "TERMINATED":
            return {"state": "gone"}
        port = (p.get("portMappings") or {}).get("22")
        if p.get("publicIp") and port:
            return {"state": "running", "host": p["publicIp"], "port": port}
        return {"state": "starting"}

    def remove(self, mid):
        self.rest("DELETE", f"/pods/{mid}")

    def self_delete(self, mid):
        # runpodctl comes with every pod, with RUNPOD_API_KEY: a key for this pod only
        return (f'subprocess.run(["runpodctl", "remove", "pod", env("RUNPOD_POD_ID") or {mid!r}], '
                'env={**os.environ, "RUNPOD_API_KEY": env("RUNPOD_API_KEY")})\n')


_RUNPOD = Runpod()


# the Connections page and cloud_book call these
def account(key=None):
    return _RUNPOD.account(key)


def gpus():
    return _RUNPOD.gpus()


def save_key(key):
    return _RUNPOD.save_key(key)


def forget_key():
    _RUNPOD.forget_key()
