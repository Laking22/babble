"""Lightning AI and Beam steps, through their own Python libraries (installed with the app): run by the engine as
a short-lived process of the app's own Python, one step at a time, with the key on its stdin:

    python sdk_jobs.py <lightning|beam> <step> '<json arguments>'

so their libraries (and the keys) never stay in the engine. Lines it prints, like modal_jobs.py:
    LOG <text>      progress, for the task's log
    RESULT <json>   the answer, last; {"error": "...", "auth": true} when the key wasn't accepted

Steps: check · submit · status · fetch · stop
Standalone on purpose (no imports from the app).
"""
import json
import os
import re
import sys
import tarfile
import tempfile
import time
from pathlib import Path

LINE = re.compile(r"[0-9a-f]{20}\.flac")
CHUNK = 300  # lines per copy (one tar each)


def _new_lines(names, have_file):
    have = set(json.loads(Path(have_file).read_text()))
    return [n for n in names if LINE.fullmatch(n) and n not in have]


def _unpack_lines(tar, dest):
    """Only finished line files, flat into dest (nothing else from the tar is trusted)."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar) as t:
        for m in t.getmembers():
            name = Path(m.name).name
            if m.isfile() and LINE.fullmatch(name):
                with t.extractfile(m) as src, open(dest / (name + ".part"), "wb") as f:
                    f.write(src.read())
                (dest / (name + ".part")).replace(dest / name)


AUTH = re.compile(r"401|403|unauthori|unauthenti|invalid (api )?(key|token)|not authori|forbidden|authenticat", re.I)


def say(text):
    print(f"LOG {text}", flush=True)


def result(**kw):
    print("RESULT " + json.dumps(kw, default=str), flush=True)


# ---------- Lightning AI: a Studio (Lightning's cloud machine) started on a GPU for the job ----------
# The job's files go to the Studio's own disk (/teamspace/studios/this_studio); the job stops the Studio when
# it's done (Lightning's library is signed in inside every Studio), so the GPU stops costing. The app downloads
# the lines from it (starting it on a free CPU machine if needed), then deletes the Studio.

STUDIO_HOME = "/teamspace/studios/this_studio"


def _lt_where():
    from lightning_sdk.api.user_api import UserApi
    uid = os.environ["LIGHTNING_USER_ID"]
    api = UserApi()
    me = api._get_user_by_id(uid)
    ms = api._get_all_teamspace_memberships(uid)
    own = [m for m in ms if getattr(m, "owner_id", "") == uid] or ms
    if not own:
        raise RuntimeError("Your Lightning account has no teamspace yet: open lightning.ai once to make one.")
    return me.username, own[0].name


def _lt_studio(args, create=False):
    from lightning_sdk import Studio
    return Studio(name=args["studio"], teamspace=args["teamspace"], user=args["user"], create_ok=create)


def lt_check(_):
    user, ts = _lt_where()
    result(ok=True, user=user, teamspace=ts)


def lt_submit(args):
    from lightning_sdk import Machine
    user, ts = _lt_where()
    a = {"studio": args["studio"], "teamspace": ts, "user": user}
    say("Starting a Lightning Studio on the GPU…")
    s = _lt_studio(a, create=True)
    try:
        s.auto_sleep = False  # the job stops the Studio itself; sleeping mid-book would stop the job
    except Exception:  # noqa: BLE001 - older accounts: it still stops itself at the end
        pass
    s.start(getattr(Machine, args["gpu"]), interruptible=bool(args.get("interruptible")))
    try:
        say("Uploading the job to the Studio…")
        s.upload_folder(args["data_dir"], f"narrator/{args['run']}/data", progress_bar=False)
        d = f"{STUDIO_HOME}/narrator/{args['run']}/data"
        s.run(f"chmod +x {d}/job.sh && (setsid nohup bash {d}/job.sh >/dev/null 2>&1 < /dev/null &) && echo started")
    except BaseException:
        try:
            s.stop()
        except Exception:  # noqa: BLE001 - best-effort cleanup so GPU doesn't keep billing
            pass
        raise
    result(ok=True, **a)


def lt_status(args):
    s = _lt_studio(args)
    st = str(getattr(s.status, "value", s.status))
    if st.lower() not in ("running",):
        # it stops itself when the job's done (or it was taken back); the files are still on its disk
        return result(ok=True, state="stopped", detail=st)
    out = f"{STUDIO_HOME}/narrator/{args['run']}/out"
    text, _ = s.run_with_exit_code(f"cat {out}/exit_code 2>/dev/null || echo RUNNING; tail -1 {out}/run.log 2>/dev/null")
    lines = text.strip().splitlines() or ["RUNNING"]
    head, detail = lines[0].strip(), (lines[1] if len(lines) > 1 else "")
    if head == "RUNNING":
        return result(ok=True, state="running", detail=detail)
    result(ok=True, state="complete" if head == "0" else "error", detail=detail)


def lt_fetch(args):
    from lightning_sdk import Machine
    s = _lt_studio(args)
    if str(getattr(s.status, "value", s.status)).lower() != "running":
        say("Opening the Studio on a free CPU machine to copy the lines back…")
        s.start(Machine.CPU)
    if args.get("small"):  # only the job's summary and log (the lines came back while it worked)
        dest = Path(args["dest"])
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("progress.json", "run.log"):
            try:
                s.download_file(f"narrator/{args['run']}/out/{name}", str(dest / name))
            except Exception:  # noqa: BLE001 - not written (the job was cut off)
                pass
        return result(ok=True)
    s.download_folder(f"narrator/{args['run']}/out", args["dest"])
    result(ok=True)


def lt_fetch_new(args):
    """The lines made since the last copy. While the job works only when the Studio is running (it isn't started
    for this); with "start", also after it stopped itself (the end of the job)."""
    from lightning_sdk import Machine
    s = _lt_studio(args)
    if str(getattr(s.status, "value", s.status)).lower() != "running":
        if not args.get("start"):
            return result(ok=True, files=0, skipped="the Studio isn't running")
        say("Opening the Studio on a free CPU machine to copy the lines back…")
        s.start(Machine.CPU)
    out = f"{STUDIO_HOME}/narrator/{args['run']}/out"
    names = _new_lines(s.run(f"ls {out}/lines 2>/dev/null || true").split(), args["have_file"])
    dest = Path(args["dest"])
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(0, len(names), CHUNK):
            s.run(f"cd {out}/lines && tar -cf {out}/live-part.tar " + " ".join(names[i:i + CHUNK]))
            s.download_file(f"narrator/{args['run']}/out/live-part.tar", f"{tmp}/part.tar")
            _unpack_lines(f"{tmp}/part.tar", dest)
    result(ok=True, files=len(names))


def lt_stop(args):
    s = _lt_studio(args)
    try:
        if str(getattr(s.status, "value", s.status)).lower() in ("running", "pending"):
            s.stop()
    finally:
        s.delete()
    result(ok=True)


# ---------- Beam: a GPU sandbox for the job ----------
# The sandbox ends itself after `ttl` seconds whatever happens (Beam's keep-warm limit); the app ends it as soon
# as the lines are back.

def _beam():
    import beam  # noqa: F401 - loading it first makes Beam's library talk to Beam (not its open-source twin)
    return beam


def bm_check(_):
    _beam()
    from beta9.channel import Channel, ServiceClient
    from beta9.clients.gateway import AuthorizeRequest
    from beta9.config import get_settings
    s = get_settings()
    with ServiceClient.with_channel(Channel(addr=f"{s.gateway_host}:{s.gateway_port}", token=os.environ["BEAM_TOKEN"])) as c:
        res = c.gateway.authorize(AuthorizeRequest())
    if not res.ok:
        return result(ok=False, auth=True, error=res.error_msg or "Beam didn't accept the key.")
    result(ok=True, user=getattr(res, "workspace_id", "") or None)


def bm_submit(args):
    beam = _beam()
    say("Starting a Beam sandbox on the GPU…")
    sb = beam.Sandbox(gpu=args["gpu"], cpu=4, memory="24Gi", keep_warm_seconds=int(args["ttl"]),
                      image=beam.Image(base_image=args["image"], python_version="python3.11"),
                      name="narrator-studio").create()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            pack = Path(tmp) / "job.tar"
            with tarfile.open(pack, "w") as t:
                t.add(args["data_dir"], arcname="data")
            say(f"Uploading the job ({pack.stat().st_size / 1e6:,.0f} MB)…")
            sb.fs.upload_file(str(pack), "/workspace/job.tar")
        p = sb.process.exec("bash", "-c", "cd /workspace && tar -xf job.tar && rm job.tar && chmod +x data/job.sh && "
                            "(setsid nohup bash data/job.sh >/dev/null 2>&1 < /dev/null &) && echo started")
        p.wait()
    except BaseException:
        sb.terminate()
        raise
    result(ok=True, sandbox=sb.sandbox_id())


def _bm_box(args):
    return _beam().Sandbox().connect(args["sandbox"])


def bm_status(args):
    try:
        sb = _bm_box(args)
        if sb.poll() is not None:
            return result(ok=True, state="gone", detail="the sandbox ended")
    except Exception as e:  # noqa: BLE001 - ended/missing -> gone, but auth/network errors propagate
        msg = f"{type(e).__name__}: {e}"
        if AUTH.search(msg) or re.search(r"connect|timeout|timed out|unavailable|reset|refused|resolve|eof|502|503|504", msg, re.I):
            raise
        return result(ok=True, state="gone", detail=str(e)[:200])
    try:
        code = sb.fs.read_text("/workspace/out/exit_code").strip()
    except Exception:  # noqa: BLE001 - not written yet
        code = ""
    try:
        tail = sb.fs.read_text("/workspace/out/run.log").strip().splitlines()[-1:]
    except Exception:  # noqa: BLE001
        tail = []
    detail = tail[0][:200] if tail else ""
    if not code:
        return result(ok=True, state="running", detail=detail)
    result(ok=True, state="complete" if code == "0" else "error", detail=detail)


def bm_fetch(args):
    sb = _bm_box(args)
    rx = re.compile(args.get("pattern") or ".")
    dest = Path(args["dest"])

    def walk(remote, rel=""):
        for f in sb.fs.list_files(remote):
            path, name = f"{remote}/{f.name}", f"{rel}{f.name}"
            if f.is_dir:
                walk(path, name + "/")
            elif rx.search(name):
                (dest / name).parent.mkdir(parents=True, exist_ok=True)
                sb.fs.download_file(path, str(dest / name))
    walk("/workspace/out")
    result(ok=True)


def bm_fetch_new(args):
    """The lines made since the last copy, while the sandbox runs."""
    sb = _bm_box(args)
    try:
        names = [f.name for f in sb.fs.list_files("/workspace/out/lines") if not f.is_dir]
    except Exception as e:  # noqa: BLE001 - none made yet
        if AUTH.search(str(e)):
            raise
        return result(ok=True, files=0)
    names = _new_lines(names, args["have_file"])
    dest = Path(args["dest"])
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(0, len(names), CHUNK):
            p = sb.process.exec("bash", "-c", "cd /workspace/out/lines && tar -cf /workspace/live-part.tar "
                                + " ".join(names[i:i + CHUNK]))
            p.wait()
            sb.fs.download_file("/workspace/live-part.tar", f"{tmp}/part.tar")
            _unpack_lines(f"{tmp}/part.tar", dest)
    result(ok=True, files=len(names))


def bm_stop(args):
    try:
        _bm_box(args).terminate()
    except Exception as e:  # noqa: BLE001 - ended already
        if AUTH.search(str(e)):
            raise
    result(ok=True)


STEPS = {"lightning": {"check": lt_check, "submit": lt_submit, "status": lt_status, "fetch": lt_fetch,
                       "fetch_new": lt_fetch_new, "stop": lt_stop},
         "beam": {"check": bm_check, "submit": bm_submit, "status": bm_status, "fetch": bm_fetch,
                  "fetch_new": bm_fetch_new, "stop": bm_stop}}

if __name__ == "__main__":
    if os.environ.pop("BABBLE_ENV_STDIN", None):  # the key comes on stdin, not in the environment it was started
        os.environ.update(json.loads(sys.stdin.readline() or "{}"))  # with (which other programs can list)
    service, step = sys.argv[1], sys.argv[2]
    raw = sys.argv[3] if len(sys.argv) > 3 else "{}"
    try:
        STEPS[service][step](json.loads(raw))
    except Exception as e:  # noqa: BLE001 - reported to the app
        msg = f"{type(e).__name__}: {e}"
        result(ok=False, auth=bool(AUTH.search(msg)), error=msg[:600])
    sys.stdout.flush()
    time.sleep(0.1)
