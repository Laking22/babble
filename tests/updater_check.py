"""Run: .venv/bin/python tests/updater_check.py  (a second, no network: GitHub is a stand-in, scratch data folder)

Updates (narrator/updater.py): which release is newer, which file this computer installs, the download checked
against GitHub's checksum (and thrown away when it doesn't match), ready.json for the app, and the
"Update automatically" switch deciding whether it's installed when Babble quits.
"""
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="updater-"))
os.environ["NARRATOR_HOME"] = str(TMP / "home")
os.environ["BABBLE_APP_VERSION"] = "0.1.0"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
failed = []


def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what, flush=True)
    if not ok:
        failed.append(what)


from narrator import settings, updater  # noqa: E402

k = updater.version_key
check(k("v0.2.0") > k("0.2.0-beta.2") > k("0.2.0-beta.1") > k("0.1.9") and k("v0.1.0") == k("0.1.0"),
      "versions: a release beats its betas, beta.2 beats beta.1")
assets = [{"name": n} for n in ("Babble_0.2.0_x64-setup.exe", "Babble_0.2.0_amd64.deb", "Babble_0.2.0_amd64.AppImage",
                                "Babble_0.2.0_mac_aarch64.zip", "Babble_0.2.0_mac_x86_64.zip")]
check(updater.asset_for(assets, "windows")["name"].endswith("setup.exe") and updater.asset_for(assets, "deb")["name"].endswith(".deb")
      and updater.asset_for(assets, "appimage")["name"].endswith(".AppImage"), "each system gets its own installer")
mac = updater.asset_for(assets, "mac")["name"]
import platform  # noqa: E402
check(mac.endswith("aarch64.zip") if platform.machine() == "arm64" else mac.endswith("x86_64.zip"), f"this Mac's build ({mac})")

PAYLOAD = b"new babble " * 5000
NAME = {"darwin": "Babble_0.2.0_mac_aarch64.zip" if platform.machine() == "arm64" else "Babble_0.2.0_mac_x86_64.zip",
        "win32": "Babble_0.2.0_x64-setup.exe"}.get(sys.platform, "Babble_0.2.0_amd64.deb")
digest = "sha256:" + hashlib.sha256(PAYLOAD).hexdigest()
URL = f"https://github.com/Laking22/babble/releases/download/v0.2.0/{NAME}"
releases = [
    {"tag_name": "v0.1.0", "name": "Babble Beta 1", "draft": False, "assets": []},
    {"tag_name": "v0.3.0", "name": "Draft", "draft": True, "assets": []},
    {"tag_name": "v0.2.0", "name": "Babble Beta 2", "draft": False, "body": "Better voices.", "html_url": "https://github.com/x",
     "assets": [{"name": NAME, "size": len(PAYLOAD), "digest": digest, "browser_download_url": URL}]},
]
updater._fetch_releases = lambda: releases


class Resp(io.BytesIO):
    def __init__(self, data, url):
        super().__init__(data)
        self.url = url

    def geturl(self):
        return self.url

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


served = {"data": PAYLOAD}
updater.urllib.request.urlopen = lambda req, timeout=0: Resp(served["data"], "https://objects.githubusercontent.com/x")

s = updater.status(force=True)
check(s["available"] and s["latest"]["version"] == "0.2.0" and s["latest"]["name"] == "Babble Beta 2" and not s["ready"],
      f"the newest published release is offered, drafts aren't ({s['latest']})")


def run(gen):
    out = None
    for ev in gen:
        out = ev
    return out


served["data"] = PAYLOAD[:-1] + b"X"  # same size, different bytes
try:
    run(updater.download(log=lambda m: None))
    check(False, "a download that doesn't match its checksum is refused")
except ValueError as e:
    check("checksum" in str(e) and not list(updater.UPDATES.glob("*.part")) and not updater.READY.exists(),
          "a download that doesn't match its checksum is thrown away")

served["data"] = PAYLOAD
last = run(updater.download(log=lambda m: None))
r = json.loads(updater.READY.read_text())
check(last.get("ready") and (updater.UPDATES / NAME).read_bytes() == PAYLOAD and r["version"] == "0.2.0" and r["on_quit"] is False,
      "a good download is kept, with ready.json (not installed at quit: the switch is off)")
check(updater.status()["ready"], "Settings shows it ready to install")
updater.watch = lambda: None  # (no background thread in the test)
updater.set_auto(True)
check(settings.load()["auto_update"] is True and json.loads(updater.READY.read_text())["on_quit"] is True,
      "turning on Update automatically: it's installed when Babble quits")
updater.set_auto(False)
check(json.loads(updater.READY.read_text())["on_quit"] is False, "and turning it off again: it isn't")

releases[2]["assets"][0]["browser_download_url"] = "https://example.com/evil.zip"
updater._cache["at"] = 0
try:
    run(updater.download(log=lambda m: None))
    check(False, "downloads only come from GitHub")
except ValueError as e:
    check("isn't on GitHub" in str(e), "an installer that isn't on GitHub isn't downloaded")

os.environ["BABBLE_APP_VERSION"] = "0.2.0"
check(not updater.status(force=True)["available"] and not updater.ready(), "once installed, nothing is offered and the old download doesn't count")
os.environ["BABBLE_APP_VERSION"] = ""
check(not updater.status()["available"] and "wasn't installed from a release" in updater.status()["error"],
      "run from a checkout (no app version): no updates")

print("\nall passed" if not failed else f"\n{len(failed)} failed: " + "; ".join(failed))
sys.exit(1 if failed else 0)
