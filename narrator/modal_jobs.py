"""Modal steps, through Modal's own Python library (installed with the app): run by the engine as a short-lived
process of the app's own Python, one step at a time:

    python modal_jobs.py <step> '<json arguments>'

so Modal's sign-in is read fresh every time and nothing of Modal stays in the engine. Lines it prints:
    LOG <text>      progress, for the task's log
    URL <link>      a page the person should open (sign-in)
    RESULT <json>   the answer, last; {"error": "...", "auth": true} when it went wrong (auth: not signed in)

Standalone on purpose (no imports from the app): it starts quickly and the GPU jobs it defines are sent to Modal
whole, by value.

Steps: check · signin · submit · status · fetch · fetch_new · put · rm · unpack
"""
import json
import re
import sys
import time
from pathlib import Path

VOLUME = "narrator-train"
PYTHON = "3.11"  # the image's Python; it has to be the app's for the jobs sent by value


def say(kind, text):
    print(f"{kind} {text}", flush=True)


def result(**kw):
    print("RESULT " + json.dumps(kw), flush=True)


def _volume(create=True):
    import modal
    return modal.Volume.from_name(VOLUME, create_if_missing=create)


def _workspace():
    import modal.config as mc
    return getattr(mc, "_profile", None) or ""


# ---------- signing in ----------

def check(_):
    import modal.config as mc
    if not mc.config.get("token_id"):
        return result(ok=False, auth=True, error="Modal isn't signed in.")
    _volume().hydrate()  # anything that needs a valid token
    result(ok=True, workspace=_workspace())


def signin(args):
    """Modal's own browser sign-in (what `modal token new` does): the page asks you to approve this computer, and
    the token is kept in ~/.modal.toml, where Modal's tools look."""
    import webbrowser
    from modal._utils.async_utils import synchronizer
    from modal.client import _Client
    from modal.config import config
    from modal.token_flow import _set_token, _TokenFlow
    minutes = float(args.get("wait_minutes", 10))

    @synchronizer.create_blocking
    async def flow():
        server_url = config.get("server_url")
        async with _Client.anonymous(server_url) as client:
            tf = _TokenFlow(client)
            async with tf.start("narrator-studio", None, None) as (_, web_url, code):
                say("URL", web_url)
                webbrowser.open(web_url)
                say("LOG", "Waiting for you to approve this computer in your browser…")
                ends, res = time.time() + minutes * 60, None
                while res is None and time.time() < ends:
                    res = await tf.finish(timeout=30)
                if res is None:
                    raise TimeoutError("The Modal sign-in wasn't finished in time. Press Sign in to try again.")
            server_url = client.server_url
        await _set_token(res.token_id, res.token_secret, server_url=server_url, verify=True)
        return res.workspace_username

    who = flow()
    result(ok=True, workspace=who or "")


def signout(_):
    """Forget the Modal sign-in on this computer: the active profile's token leaves ~/.modal.toml (the file Modal's
    own tools read too)."""
    import modal.config as mc
    cfg = mc._read_user_config()
    profile = mc._profile
    cfg.pop(profile, None)
    mc._write_user_config(cfg)
    result(ok=True, profile=profile)


# ---------- jobs ----------

def _job_function():
    """The GPU job, made here so it's sent to Modal whole (it imports nothing from the app)."""
    def job(run, script, args, volume, live=False):
        import os
        import shlex
        import subprocess
        import modal
        base = "/vol/" + run
        env = dict(os.environ, NARRATOR_DATA=base + "/data", HF_HOME="/vol/hf-cache",  # models downloaded once
                   NARRATOR_MODELS="/vol/models")  # trained voices uploaded from the Mac
        os.makedirs(base + "/out", exist_ok=True)
        with open(base + "/out/job.log", "w") as log:
            p = subprocess.Popen(["python", base + "/data/" + script, *shlex.split(args), "--out", base + "/out"],
                                 stdout=log, stderr=subprocess.STDOUT, env=env)
            while True:
                try:
                    p.wait(timeout=120 if live else None)
                    break
                except subprocess.TimeoutExpired:  # live: save the volume, so the app can copy the lines made so far
                    try:
                        modal.Volume.from_name(volume).commit()
                    except Exception:  # noqa: BLE001 - next time
                        pass
        with open(base + "/exit_code", "w") as f:
            f.write(str(p.returncode))
        modal.Volume.from_name(volume).commit()
        return p.returncode
    return job


def submit(args):
    """Upload the job's folder to the volume and start it on a GPU; returns at once (the job keeps going)."""
    import modal
    run, gpu = args["run"], args["gpu"]
    vol = _volume()
    say("LOG", "Uploading the job to your Modal storage…")
    with vol.batch_upload(force=True) as b:
        b.put_directory(args["data_dir"], f"/{run}/data")
    image = (modal.Image.debian_slim(python_version=PYTHON).apt_install("ffmpeg", "libsndfile1")
             .pip_install("torch", *args.get("pip", [])))
    app = modal.App("narrator-" + run)
    fn = app.function(gpu=gpu, image=image, volumes={"/vol": vol}, timeout=int(args["timeout"]),
                      serialized=True, name="job")(_job_function())
    say("LOG", f"Starting it on a Modal {gpu} (the first time, Modal builds the software for it: a few minutes)…")
    with app.run(detach=True):
        call = fn.spawn(run, args["script"], args.get("args", ""), VOLUME, bool(args.get("live")))
    result(ok=True, run=run, call=call.object_id)


STORAGE = "modal-storage.com"  # where Modal keeps the volume's files (listings come from its main server)
BLOCKED = ("Can't reach Modal's file storage (modal-storage.com) right now. Babble tries again as soon as it can; "
           "if it keeps happening, this network may block it: try a phone hotspot, home Wi-Fi or a VPN.")


def _blocked(e):
    """Modal's file storage turned this computer away (a network that filters it), not a missing file."""
    text = f"{type(e).__name__} {e}"
    return STORAGE in text or "ClientConnectorError" in text or "Connection reset" in text


def _read(vol, path):
    """A volume file's text; None when it isn't there. Raises ConnectionError when the storage can't be reached."""
    try:
        return b"".join(vol.read_file(path)).decode(errors="replace")
    except Exception as e:  # noqa: BLE001 - not there (yet)
        if _blocked(e):
            raise ConnectionError(BLOCKED) from e
        return None


def status(args):
    import modal
    vol = _volume(create=False)
    blocked = False
    try:
        code = _read(vol, f"/{args['run']}/exit_code")
    except ConnectionError:
        code, blocked = None, True
    if code is not None:
        code = code.strip()
        return result(ok=True, state="complete" if code == "0" else "error", detail=f"exit {code}")
    if args.get("call"):  # the job's own answer: its exit code (what it also writes to exit_code)
        try:
            rc = modal.FunctionCall.from_id(args["call"]).get(timeout=0)
            if isinstance(rc, int):
                return result(ok=True, state="complete" if rc == 0 else "error", detail=f"exit {rc}")
            if blocked:
                return result(ok=True, state="unknown", detail=BLOCKED)
            return result(ok=True, state="error", detail="the job ended without saving its results")
        except TimeoutError:  # (Python's own, from get(timeout=0): not done yet)
            pass
        except Exception as e:  # noqa: BLE001 - it failed on Modal (out of memory, timed out, stopped…)
            name = type(e).__name__
            # The connection, not the job (modal.exception.ConnectionError isn't Python's): raised, so the engine
            # sees a network error and asks again when it's back, instead of taking the run as ended and leaving
            # its GPU going.
            if name in ("ConnectionError", "StreamTerminatedError", "ClientConnectorError") or isinstance(e, OSError) \
                    or _blocked(e):
                raise
            if name == "FunctionTimeoutError":  # (modal's own TimeoutError: it hit its time limit and was stopped)
                return result(ok=True, state="error", detail="It ran past its time limit on Modal and was stopped.")
            if "timeout" not in name.lower():
                return result(ok=True, state="error", detail=str(e)[:300])
    if blocked:
        return result(ok=True, state="unknown", detail=BLOCKED)
    names = [e.path for e in vol.listdir(f"/{args['run']}")] if _exists(vol, f"/{args['run']}") else []
    result(ok=True, state="running" if names else "unknown", detail="" if names else "not found on Modal")


def cancel(args):
    """Stop a training job (its container is shut down; nothing more is billed)."""
    import modal
    modal.FunctionCall.from_id(args["call"]).cancel(terminate_containers=True)
    result(ok=True)


def _exists(vol, path):
    try:
        vol.listdir(path)
        return True
    except Exception:  # noqa: BLE001
        return False


def _is_file(entry):
    """A listing entry that's a file (Modal gives the type as a number: 1 file, 2 folder, 3 link…)."""
    from modal.volume import FileEntryType
    return entry.type == FileEntryType.FILE


def fetch(args):
    """Download the run's output files that match the pattern. Says how far along it is as it goes:
    PROGRESS {"file": name, "i": 3, "n": 14, "done": bytes so far, "total": bytes in all}."""
    vol = _volume(create=False)
    run, dest, rx = args["run"], Path(args["dest"]), re.compile(args["pattern"])
    dest.mkdir(parents=True, exist_ok=True)
    files = []
    for e in vol.listdir(f"/{run}/out", recursive=True):
        path = e.path.lstrip("/")
        rel = path.split(f"{run}/out/", 1)[-1]
        if _is_file(e) and rx.search(rel):  # (folders come too: their files are listed as well)
            files.append((path, rel, e.size or 0))
    total, done, said = sum(s for *_, s in files), 0, 0.0
    resume = bool(args.get("resume"))  # files already here whole (from a download that stopped) aren't fetched again

    def tell(i, rel, force=False):
        nonlocal said
        if force or time.time() - said >= 1.0:
            said = time.time()
            say("PROGRESS", json.dumps({"file": rel, "i": i, "n": len(files), "done": done, "total": total}))

    n = 0
    for i, (path, rel, size) in enumerate(files, 1):
        tell(i, rel, force=True)

        def advanced(advance=0, **_):
            nonlocal done
            done += advance
            tell(i, rel)

        target, part = dest / rel, dest / (rel + ".part")
        target.parent.mkdir(parents=True, exist_ok=True)
        start = done
        if not (resume and target.exists() and target.stat().st_size == size):
            # (into .part first: Modal writes a file's pieces at once, so its size says nothing until it's done)
            with open(part, "wb") as f:
                vol.read_file_into_fileobj("/" + path, f, progress_cb=advanced)
            part.replace(target)
        done = start + size  # (exact, whatever the reports added up to)
        n += 1
    if files:
        tell(len(files), files[-1][1], force=True)
    result(ok=True, files=n)


LINE = re.compile(r"[0-9a-f]{20}\.flac")


def fetch_new(args):
    """The lines made since the last copy (names not in have_file), while the job works or after it."""
    vol = _volume(create=False)
    run, dest = args["run"], Path(args["dest"])
    have = set(json.loads(Path(args["have_file"]).read_text()))
    dest.mkdir(parents=True, exist_ok=True)
    try:
        entries = vol.listdir(f"/{run}/out/lines")
    except Exception as e:  # noqa: BLE001 - none made yet, unless it's an auth/network failure
        msg = f"{type(e).__name__}: {e}"
        if _blocked(e) or re.search(r"authenticat|unauthori|token|connect|timeout|unavailable", msg, re.I):
            raise
        return result(ok=True, files=0)
    n = 0
    for e in entries:
        name = e.path.rstrip("/").rsplit("/", 1)[-1]
        if not LINE.fullmatch(name) or name in have:
            continue
        part = dest / (name + ".part")
        with open(part, "wb") as f:
            vol.read_file_into_fileobj("/" + e.path.lstrip("/"), f)
        part.replace(dest / name)
        n += 1
    result(ok=True, files=n)


def put(args):
    vol = _volume()
    with vol.batch_upload(force=True) as b:
        b.put_directory(args["local"], args["remote"])
    result(ok=True)


def rm(args):
    vol = _volume()
    try:
        vol.remove_file(args["path"], recursive=True)
    except Exception:  # noqa: BLE001 - already gone
        pass
    result(ok=True)


def unpack(args):
    """Unpack an uploaded Mac model (MLX 4-bit) into an ordinary one, on a small CPU machine."""
    import modal
    name, script = args["name"], args["script"]
    image = (modal.Image.debian_slim(python_version=PYTHON)
             .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
             .pip_install("numpy", "safetensors")
             .add_local_file(script, "/root/unpack_mlx.py"))
    vol = _volume()
    app = modal.App("narrator-unpack-" + name)

    def make():
        def run(name, volume):
            import subprocess
            import sys as _sys
            import modal
            v = modal.Volume.from_name(volume)
            v.reload()
            r = subprocess.run([_sys.executable, "/root/unpack_mlx.py", "/vol/models-mlx/" + name, "/vol/models/" + name],
                               capture_output=True, text=True)
            v.commit()
            return r.returncode, (r.stdout + r.stderr)[-3000:]
        return run
    fn = app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=16384, timeout=3600, serialized=True,
                      name="unpack")(make())
    with app.run():
        code, out = fn.remote(name, VOLUME)
    result(ok=code == 0, code=code, out=out, **({} if code == 0 else {"error": out[-400:]}))


STEPS = {f.__name__: f for f in (check, signin, signout, submit, status, cancel, fetch, fetch_new, put, rm, unpack)}


if __name__ == "__main__":
    step, raw = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "{}")
    try:
        STEPS[step](json.loads(raw))
    except Exception as e:  # noqa: BLE001 - reported to the engine
        name = type(e).__name__
        signed_out = r"authenticat|unauthori|\btoken (id|secret)\b|invalid token|\btoken\b.*\b(missing|expired|invalid)"
        auth = name in ("AuthError",) or bool(re.search(signed_out, str(e), re.I))  # (not "speech_tokenizer")
        if _blocked(e) and not auth:  # (the storage by name; else a connection that dropped, not a block)
            error = BLOCKED if STORAGE in f"{name} {e}" else f"Couldn't connect to Modal ({name}: {str(e)[:300]})"
        else:
            error = f"{str(e)[:500] or name}"
        result(ok=False, auth=auth, error=error)
