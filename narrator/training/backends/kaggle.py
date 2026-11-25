"""Kaggle: the user's own account (free, about 30 GPU hours a week, 12 hours per session, T4).

Uploads the Package folder as a PRIVATE dataset, then pushes a private GPU notebook that installs the
Package's pip packages and runs its script. Talks to Kaggle through Kaggle's own Python library (in the app's
Python), so there's nothing to install or type in Terminal. Signing in happens from Connections: Sign in with
Kaggle (the browser opens; Kaggle's sign-in then renews itself), or paste an API token from kaggle.com/settings.
Both are kept where Kaggle's own tools look (~/.kaggle), so a sign-in made with Kaggle's command-line tool works too.
"""
import json
import os
import re
import shutil
import time
import webbrowser
from pathlib import Path

from ..base import AuthError, Backend, RunStatus

_GPU = {"T4": "NvidiaTeslaT4", "P100": "NvidiaTeslaP100"}
SIGN_IN_HELP = "Sign in to Kaggle on the Connections page."
TOKEN_FILE = Path.home() / ".kaggle" / "access_token"
SCOPES = ["resources.admin:*"]  # what Kaggle's own tool asks for: datasets and notebooks of your account


def _slug(text):
    return re.sub(r"[^a-z0-9-]+", "-", text.lower()).strip("-")[:50]


def _http_status(e):
    return getattr(getattr(e, "response", None), "status_code", None)


def api():
    """Kaggle's library, signed in (a fresh one each time, so a sign-in made or renewed since is picked up).
    Raises AuthError when there's no sign-in."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    a = KaggleApi()
    try:
        a.authenticate()
    except SystemExit:  # its way of saying "no credentials" (it prints help and exits)
        raise AuthError(f"Kaggle isn't signed in. {SIGN_IN_HELP}") from None
    except Exception as e:  # noqa: BLE001 - a sign-in that Kaggle no longer accepts
        if _http_status(e) in (400, 401, 403):
            raise AuthError(f"Kaggle didn't accept the saved sign-in. {SIGN_IN_HELP}") from e
        raise
    if not a.config_values.get("username"):
        raise AuthError(f"Kaggle isn't signed in. {SIGN_IN_HELP}")
    return a


def _call(fn, *args, **kw):
    """A library call, with Kaggle's errors said plainly."""
    try:
        return fn(*args, **kw)
    except Exception as e:  # noqa: BLE001
        code = _http_status(e)
        if code == 401:
            raise AuthError(f"Kaggle needs you to sign in again. {SIGN_IN_HELP}") from e
        raise RuntimeError(f"Kaggle error{f' ({code})' if code else ''}: {str(e)[:400]}") from e


# ---------- signing in (Connections) ----------

def sign_in(log=print, wait_minutes=10):
    """Sign in with Kaggle in the browser: Kaggle's own sign-in (OAuth, the same as its command-line tool's), whose
    answer comes back to this computer. The sign-in is kept in ~/.kaggle/credentials.json and renews itself."""
    import random
    import socketserver
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kaggle_oauth import KaggleOAuth
    with KaggleApi().build_kaggle_client() as client:
        oauth = KaggleOAuth(client=client)
        port = random.randint(8000, 9000)
        state = KaggleOAuth.OAuthState()
        url = client.http_client().build_start_oauth_url(
            client_id=KaggleOAuth.OAUTH_CLIENT_ID, redirect_uri=f"http://localhost:{port}", scope=SCOPES,
            state=state.state, code_challenge=state.code_challenge)
        got = {}

        def handler(*args, **kwargs):
            return KaggleOAuth.OAuthCallbackHandler(
                *args, oauth=oauth, oauth_state=state, logger=oauth._logger,
                on_success=lambda code: got.update(code=code), **kwargs)

        with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
            httpd.timeout = 2
            webbrowser.open(url)
            log(f"URL {url}")  # (the page shows it as a link, in case the browser didn't open)
            log("Waiting for you to sign in to Kaggle in your browser…")
            oauth._server_running = True
            ends = time.time() + wait_minutes * 60
            while oauth._server_running and time.time() < ends and "code" not in got:
                httpd.handle_request()
        if "code" not in got:
            raise ValueError("The Kaggle sign-in wasn't finished in time. Press Sign in to try again.")
        oauth._exchange_oauth_token(got["code"], SCOPES, state)
        user = oauth._ensure_creds_valid(oauth._creds)
        oauth._creds.save()
    from ... import conn_issues
    conn_issues.ok("kaggle")  # (a problem noted before is over)
    log(f"Done · Signed in to Kaggle as {user}.")
    return user


def save_token(token):
    """Sign in with an API token made at kaggle.com/settings (API Tokens → Generate New Token)."""
    token = (token or "").strip()
    if not re.fullmatch(r"[\w.\-~+/=]{20,}", token):
        raise ValueError("That doesn't look like a Kaggle API token. Copy it from kaggle.com/settings → API Tokens.")
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    old = TOKEN_FILE.read_text() if TOKEN_FILE.exists() else None
    TOKEN_FILE.write_text(token)
    os.chmod(TOKEN_FILE, 0o600)
    try:
        a = api()
        if a.config_values.get("auth_method", "").upper().endswith("OAUTH") or not a.config_values.get("username"):
            raise AuthError("Kaggle didn't accept that token.")
        from ... import conn_issues
        conn_issues.ok("kaggle")
        return a.config_values["username"]
    except AuthError:
        if old is None:
            TOKEN_FILE.unlink(missing_ok=True)
        else:
            TOKEN_FILE.write_text(old)
        raise ValueError("Kaggle didn't accept that token. Make a new one at kaggle.com/settings → API Tokens.") from None


def sign_out():
    """Forget the Kaggle sign-in on this computer (both kinds)."""
    for f in (Path.home() / ".kaggle" / "credentials.json", TOKEN_FILE):
        f.unlink(missing_ok=True)


# ---------- the backend ----------

class Kaggle(Backend):
    id, name, readiness = "kaggle", "Kaggle", "ready"
    pricing = "free with your Kaggle account, about 30 GPU hours a week"
    hosts = ("www.kaggle.com",)

    def user(self):
        return api().config_values["username"]

    def check(self):
        try:
            a = api()
        except AuthError as e:
            return {"ok": False, "detail": str(e), "quota": None}
        except Exception as e:  # noqa: BLE001 - e.g. offline
            return {"ok": False, "detail": f"Couldn't reach Kaggle: {str(e)[:200]}", "quota": None}
        user = a.config_values["username"]
        quota = None
        try:
            q = a.quota_view()
            g = q.gpu_quota
            used, total = g.time_used.total_seconds() / 3600, g.total_time_allowed.total_seconds() / 3600
            resets = q.quota_refresh_time.isoformat() + ("" if q.quota_refresh_time.utcoffset() else "+00:00")
            quota = {"used": round(used, 2), "left": round(max(0.0, total - used), 2), "total": round(total, 2), "resets": resets}
        except Exception as e:  # noqa: BLE001 - the quota is a nice-to-have, but a refused sign-in isn't
            if _http_status(e) in (401, 403):  # (the saved sign-in reads fine, but Kaggle no longer takes it)
                return {"ok": False, "user": user, "quota": None,
                        "detail": f"Kaggle didn't accept the saved sign-in. {SIGN_IN_HELP}"}
        left = f", {quota['left']:.1f} of {quota['total']:.0f} GPU hours left this week" if quota else ""
        return {"ok": True, "user": user, "quota": quota, "detail": f"Signed in as {user}{left}"}

    def estimate(self, package):
        """Fits or not, and budget_hours: how long the run may take (a session is at most 12 hours, and a run
        stopped by the weekly limit saves nothing, so the script is told to save and stop before either)."""
        c = self.check()
        q = c.get("quota")
        need = package.gpu_hours
        fits = q is None or q["left"] >= need
        budget = 11.0 if q is None else round(min(11.0, q["left"] - 0.3), 1)  # 0.3 h: setup, saving, samples
        resets = f"It resets {q['resets']}." if q and q.get("resets") else ""
        note = "" if fits else (f"Needs about {need} GPU hours; {q['left']:.1f} left this week, so it trains for about "
                                f"{max(budget, 0):.1f} h and saves what it has learned by then. {resets}".strip())
        return {"gpu_hours": need, "cost": 0.0, "fits": fits, "budget_hours": budget, "left": q["left"] if q else None,
                "resets": resets, "note": note}

    def submit(self, package, log=print):
        a = api()
        user = a.config_values["username"]
        ds_slug = f"{package.slug}-data" if package.slug else f"{_slug(package.model)}-{package.recipe}-training"
        kn_slug = package.slug or f"{_slug(package.model)}-{package.recipe}-finetune"
        what = package.title or f"{package.model} {package.recipe}"
        # Kaggle's dataset upload skips sub-folders and strips unusual characters from file names, so nested
        # files go up under safe flat names with a tree.json mapping them back; the notebook rebuilds the
        # tree in /tmp/data before the script runs.
        src = Path(package.data_dir)
        data = src.parent / "upload"
        shutil.rmtree(data, ignore_errors=True)
        data.mkdir(parents=True)
        tree = {}
        for i, f in enumerate(sorted(x for x in src.rglob("*") if x.is_file())):
            rel = f.relative_to(src)
            flat = rel.name if len(rel.parts) == 1 else f"n{i:05d}-{re.sub(r'[^A-Za-z0-9._-]', '-', rel.name)}"
            tree[flat] = rel.as_posix()
            try:
                os.link(f, data / flat)
            except OSError:
                shutil.copy2(f, data / flat)
        (data / "tree.json").write_text(json.dumps(tree))
        (data / "dataset-metadata.json").write_text(json.dumps(
            {"title": f"{what} data" if package.slug else f"{what} training data", "id": f"{user}/{ds_slug}",
             "licenses": [{"name": "CC0-1.0"}]}))
        size_mb = sum(f.stat().st_size for f in data.rglob("*") if f.is_file()) / 1e6
        log(f"Uploading {'the job' if package.slug else f'{package.hours:.1f} h of training data'} ({size_mb:.0f} MB), private…")
        try:
            a.dataset_status(f"{user}/{ds_slug}")
            exists = True
        except Exception:  # noqa: BLE001 - not there yet
            exists = False
        if exists:
            r = _call(a.dataset_create_version, str(data), "updated", quiet=True, convert_to_csv=False, dir_mode="skip")
        else:
            r = _call(a.dataset_create_new, str(data), public=False, quiet=True, convert_to_csv=False, dir_mode="skip")
        if getattr(r, "error", None):
            raise RuntimeError(f"Kaggle didn't take the upload: {r.error}")
        time.sleep(30)  # let Kaggle register the new version before the notebook mounts it
        for _ in range(60):
            try:
                if str(a.dataset_status(f"{user}/{ds_slug}")).strip() == "ready":
                    break
            except Exception:  # noqa: BLE001 - still being made
                pass
            time.sleep(20)

        cells = [
            ("markdown", f"# {what}\nRuns `{package.script}` from the attached dataset."),
            ("code", f"!pip install -q {' '.join(package.pip)} 2>&1 | tail -3\n"
                     "!nvidia-smi --query-gpu=name,memory.total --format=csv"),
            ("code", "import glob, os, shutil\n"
                     f"src = os.path.dirname(glob.glob('/kaggle/input/**/{package.script}', recursive=True)[0])\n"
                     "dst = '/tmp/data'\n"
                     "import json\n"
                     "tree = json.load(open(os.path.join(src, 'tree.json')))\n"
                     "for name, rel in tree.items():\n"
                     "    path = os.path.join(dst, *rel.split('/'))\n"
                     "    os.makedirs(os.path.dirname(path), exist_ok=True)\n"
                     "    if not os.path.exists(path):\n"
                     "        os.symlink(os.path.join(src, name), path)\n"
                     "os.environ['NARRATOR_DATA'] = dst\n"
                     "print(sorted(os.listdir(dst)))"),
            ("code", f"!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python /tmp/data/{package.script} {package.args}"),
            ("code", "!ls -la /kaggle/working /kaggle/working/samples 2>/dev/null; du -sh /kaggle/working/* 2>/dev/null"),
        ]
        nb = {"cells": [{"cell_type": t, "metadata": {}, "source": src.splitlines(True),
                         **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, src in cells],
              "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
              "nbformat": 4, "nbformat_minor": 5}
        kern = data.parent / "kernel"
        shutil.rmtree(kern, ignore_errors=True)
        kern.mkdir(parents=True)
        (kern / "train.ipynb").write_text(json.dumps(nb, indent=1))
        (kern / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/{kn_slug}", "title": package.title or f"{package.model} {package.recipe} finetune", "code_file": "train.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_internet": True, "dataset_sources": [f"{user}/{ds_slug}"], "kernel_sources": list(package.kernel_sources),
            "competition_sources": []}))
        r = _call(a.kernels_push, str(kern), acc=_GPU.get(package.gpu, "NvidiaTeslaT4"))
        if r is None or getattr(r, "error", None):
            raise RuntimeError(f"Kaggle didn't start the notebook: {getattr(r, 'error', None) or 'no answer'}")
        bad = getattr(r, "invalidKernelSources", None) or getattr(r, "invalid_kernel_sources", None)
        if bad:
            log(f"Kaggle couldn't attach {', '.join(bad)}; those voices are made on this Mac.")
        return f"{user}/{kn_slug}"

    def url(self, run_id):
        return f"https://www.kaggle.com/code/{run_id}"

    def status(self, run_id):
        from ... import conn_issues
        try:
            r = api().kernels_status(run_id)
        except AuthError:
            raise
        except Exception as e:  # noqa: BLE001
            if conn_issues.is_network(e):
                raise
            code = _http_status(e)
            gone = code in (403, 404) or re.search(r"denied|not found|cannot access", str(e), re.I)
            why = "not found or no access" if gone else f"couldn't ask Kaggle ({str(e)[:120]})"
            return RunStatus("unknown", why, self.url(run_id))
        text = str(getattr(r, "status", "") or "")
        m = re.search(r"(RUNNING|COMPLETE|ERROR|CANCEL\w*|QUEUED)", text.upper())
        state = {"RUNNING": "running", "COMPLETE": "complete", "ERROR": "error", "QUEUED": "queued"}.get(
            m.group(1) if m else "", "cancelled" if m else "unknown")
        detail = str(getattr(r, "failure_message", "") or "")
        return RunStatus(state, detail[:300], self.url(run_id))

    def output_size(self, run_id, pattern):
        """Bytes in the run's output files that match pattern (for a download's progress), or None. Kaggle's list
        has no sizes: each file is asked for and only its headers read."""
        try:
            import requests
            from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
            a, rx, total, token = api(), re.compile(pattern), 0, None
            owner, slug, _ = a.parse_kernel_string(run_id)
            for _ in range(50):
                with a.build_kaggle_client() as k:
                    req = ApiListKernelSessionOutputRequest()
                    req.user_name, req.kernel_slug = owner, slug
                    a._set_paging(req, 100, token)
                    resp = k.kernels.kernels_api_client.list_kernel_session_output(req)
                for item in resp.files or []:
                    if rx.search(item.file_name):
                        with requests.get(item.url, stream=True, timeout=30) as r:
                            total += int(r.headers.get("content-length") or 0)
                token = resp.next_page_token
                if not token:
                    break
            return total or None
        except Exception:  # noqa: BLE001 - only the progress bar needs it
            return None

    def fetch(self, run_id, dest, pattern, log=print):
        Path(dest).mkdir(parents=True, exist_ok=True)
        a = api()
        token, pages = None, 0
        while True:  # Kaggle hands the files over a page at a time
            _, token = _call(a.kernels_output, run_id, str(dest), file_pattern=pattern, quiet=True, page_token=token,
                             page_size=100)
            pages += 1
            if not token or pages > 500:
                break
        return Path(dest)
