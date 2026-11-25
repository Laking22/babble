"""Detect speakers on a cloud GPU (Kaggle, Modal or a pay-as-you-go service), the same AI as on
this Mac, so a long book doesn't tie the Mac up for hours.

The app writes the story's paragraphs (or the sound prompts) to a job, sends it with kaggle_render/llm_runner.py,
follows it until it's done, brings back llm.json and frees what it used. The task stays open while it works (the
Mac is kept awake meanwhile); stopping the task stops the cloud job too.
"""
import json
import shutil
import time
import uuid
from pathlib import Path

from .paths import HOME

RUNNER = Path(__file__).resolve().parent / "kaggle_render" / "llm_runner.py"
CORE = Path(__file__).resolve().parent / "detect_core.py"
PIP = ["transformers==4.57.3", "accelerate"]
TOKENS_PER_SECOND = 120  # replies a T4 writes per second, several passages at once (a guess until measured)
START_HOURS = 0.08       # starting the GPU and getting the model (8 GB): measured on Kaggle, 41 s of
                         # setup and 67 s to download and load it; a little more for the machine to start
BATCH = {"detect": 8, "prompts": 16}


def where_name(make_on):
    from .training import BACKENDS
    b = BACKENDS.get(make_on)
    return b.name if b else make_on


def _plan(backend, cloud_gpu, reply_tokens):
    """(gpu, args dtype, speed, price, hours, budget, worst hours) for a job writing about reply_tokens tokens."""
    from . import cloud_book as cb
    from .training import BACKENDS
    b = BACKENDS[backend]
    if backend == "kaggle":
        gpu, dtype, speed, price, cap = "T4", "", 1.0, None, 11.0
    elif backend == "modal":
        from .training.backends.modal_cloud import GPUS
        gpu = cloud_gpu if cloud_gpu in GPUS else "A10G"
        dtype, speed, price, cap = "bfloat16", cb.MODAL_SPEED[gpu], GPUS[gpu][1], cb.MODAL_MAX_HOURS
    else:
        gpu = cloud_gpu if cloud_gpu in b.gpu_names else b.default_gpu
        dtype, speed, cap = "bfloat16", b.speed.get(gpu, 2.0), cb.MODAL_MAX_HOURS
        try:
            price = b.price(gpu)
        except Exception:  # noqa: BLE001 - only for the estimate
            price = None
    hours = reply_tokens * 0.6 / (TOKENS_PER_SECOND * speed) / 3600 + START_HOURS  # replies use ~60% of their room
    budget = round(min(cap, max(0.5, hours * 2.5)), 1)
    return gpu, dtype, speed, price, hours, budget, budget + cb.overhang(backend)


def estimate(make_on, cloud_gpu, reply_tokens, kind="detect"):
    """Before sending: {"cost", "time", "free"} in words (what the confirmation shows)."""
    from . import pricing
    gpu, _, _, price, hours, budget, worst = _plan(make_on, cloud_gpu, reply_tokens)
    if make_on == "kaggle":
        cost = f"free: {hours:.1f} to {budget + 0.5:.1f} of your weekly Kaggle GPU hours"
    else:
        cost = pricing.say(pricing.gpu_range(make_on, price, hours * 0.75, worst, kind)) if price else "price unknown"
    return {"cost": cost, "time": f"about {_clock(hours * 0.75)} to {_clock(worst)}", "free": make_on == "kaggle",
            "where": where_name(make_on), "gpu": gpu}


def run(kind, job, make_on, cloud_gpu, what, reply_tokens):
    """_run(), noting a connection problem with the cloud on Connections."""
    from . import conn_issues
    return (yield from conn_issues.watch_gen(make_on, what, _run(kind, job, make_on, cloud_gpu, what, reply_tokens)))


def _run(kind, job, make_on, cloud_gpu, what, reply_tokens):
    """Send a job ({"texts": …} for kind "detect", {"prompts": …} for "prompts"), follow it, and return llm.json's
    contents. Yields {"status": …} on the way (a task's events)."""
    from . import cloud_book as cb
    from .ai_detect import CLOUD_MODEL
    from .training import BACKENDS
    from .training.backends import CLOUDS
    from .training.base import Package
    if make_on not in CLOUDS:
        raise ValueError(f"No such cloud GPU: {make_on}.")
    b = BACKENDS[make_on]
    name = b.name
    yield {"status": f"Checking your {name} account…"}
    c = b.check()
    if not c.get("ok"):
        raise ValueError(c.get("detail") or f"{name} isn't signed in.")
    gpu, dtype, _, price, hours, budget, worst = _plan(make_on, cloud_gpu, reply_tokens)
    if make_on == "kaggle":
        left = (c.get("quota") or {}).get("left", 11.0)
        budget = round(min(budget, left - 0.3), 1)
        if budget < 0.3:
            raise ValueError(f"Kaggle has {left:.1f} GPU hours left this week, not enough to start.")
        cost = f"free: {hours:.1f} to {budget + 0.5:.1f} of your weekly GPU hours"
    elif c.get("balance") is not None and c["balance"] < 1.0:
        raise ValueError(f"Your {name} account has ${c['balance']:.2f} of credit. Add some on {name}'s billing page first.")
    from . import pricing, spend
    est = pricing.gpu_range(make_on, price, hours * 0.75, worst, kind)
    if make_on != "kaggle":
        cost = pricing.say(est)
    label = getattr(b, "gpu_names", {}).get(gpu, gpu)
    note = f"{name} {label}: about {_clock(hours * 0.75)} to {_clock(worst)}, {cost}."
    what = "Speaker detection" if kind == "detect" else "Sound prompts"
    spend.allow(make_on, est, kind)

    run_id = f"ai-{uuid.uuid4().hex[:8]}"
    folder = HOME / "cloud-ai" / run_id
    data = folder / "data"
    data.mkdir(parents=True)
    shutil.copy2(RUNNER, data / RUNNER.name)
    shutil.copy2(CORE, data / CORE.name)
    (data / "llm-job.json").write_text(json.dumps(dict(job, kind=kind, model=CLOUD_MODEL)))
    pip = PIP + list(getattr(b, "extra_pip", [])) + (["numpy"] if make_on == "modal" else [])
    slug = f"narrator-{'speakers' if kind == 'detect' else 'sounds'}"
    pkg = Package(model=slug, recipe="llm", data_dir=data, script=RUNNER.name, pip=pip, gpu=gpu, slug=slug,
                  title=slug.replace("-", " "), meta={"timeout_hours": budget + 0.5},
                  args=f"--max-hours {budget} --batch {BATCH[kind]}" + (f" --dtype {dtype}" if dtype else ""))
    cb._busy.add(run_id)  # keeps this Mac awake while it works
    cb.stay_awake()
    remote, done_ok = None, False
    try:
        yield {"status": f"Sending it to {name}… {note}"}
        remote = b.submit(pkg, log=lambda m: None)
        sent = time.time()
        # written down: if Babble crashes or quits now, the next start stops it (nothing else would come back for it)
        spend.job_sent(make_on, remote, what, price=price, hours=worst, collect=False)
        # (a lost connection is waited out: the job carries on out there, and this carries on once it's back,
        # until a while after the job's own time limit; before, it ended here and the job was cancelled)
        deadline, hosts = sent + (worst + 1) * 3600, getattr(b, "hosts", ())
        while True:
            st = yield from _through(b.status, remote, name=name, deadline=deadline, hosts=hosts)
            if st.state in ("complete", "error", "cancelled"):
                break
            waited = _clock((time.time() - sent) / 3600)
            yield {"status": f"{what} on {name} ({'waiting for a GPU' if st.state == 'queued' else 'working'}, "
                             f"{waited} so far). {note} Keep this Mac on and the app open."
                             + (f" {st.detail}" if st.detail else "")}
            for _ in range(30):  # a check every 30 s; stopping the task stops waiting at once
                from . import tasks
                if tasks.stop_wanted():
                    raise tasks.Stopped()
                time.sleep(1)
        yield {"status": f"Done on {name}. Bringing it back…"}
        out = folder / "out"
        yield from _through(b.fetch, remote, out, r"(llm\.json|progress\.json|run\.log)$", name=name,
                            deadline=max(deadline, time.time() + 24 * 3600), hosts=hosts)  # (it waits out there)
        got = next(iter(out.rglob("llm.json")), None)
        if not got:
            logs = list(out.rglob("run.log"))
            tail = "\n".join(logs[0].read_text(errors="replace").splitlines()[-4:]) if logs else ""
            raise ValueError(f"Nothing came back from {name}. {tail[-300:] or 'Open it on the site to see what happened.'}")
        progress = next(iter(out.rglob("progress.json")), None)
        result = json.loads(got.read_text())
        result["_progress"] = json.loads(progress.read_text()) if progress else {}
        done_ok = True
        return result
    finally:
        cb._busy.discard(run_id)
        cb.stay_awake()
        if remote:
            try:
                if done_ok and getattr(b, "finished", None):
                    b.finished(remote)  # a rented machine is deleted now, so it stops costing
                elif not done_ok:  # stopped or failed: stop it out there too (a Modal call is cancelled, a
                    spend.stop_remote(make_on, remote)  # machine deleted; Kaggle can't be told, it's free)
            except Exception as e:  # noqa: BLE001 - it deletes itself later anyway
                print(f"[cloud ai] couldn't free {remote}: {e}", flush=True)
            spend.job_ended(remote)  # (its GPU time, charged at the list price)
        shutil.rmtree(folder, ignore_errors=True)


def _clock(hours):
    return f"{max(1, round(hours * 60))} min" if hours < 1 else f"{hours:.1f} h"


def _through(fn, *args, name, deadline, hosts=()):
    """fn(*args), riding out a lost connection (`yield from` it: it yields statuses while it waits, then returns
    fn's answer). Tried again within a minute, and at once when the service can be reached again; anything
    that isn't the connection, or a wait past `deadline`, is raised."""
    from . import conn_issues
    lost = None
    while True:
        try:
            answer = fn(*args)
        except Exception as e:
            if not conn_issues.is_network(e) or time.time() > deadline:
                raise
            if lost is None:
                lost = time.time()
                print(f"[cloud ai] lost the connection to {name}: {str(e)[:160]}", flush=True)
            yield {"status": f"Lost the connection to {name} {_clock((time.time() - lost) / 3600)} ago. The job carries "
                             "on out there, and this carries on as soon as the connection is back. Keep the app open."}
            from . import tasks
            for i in range(60):  # again within a minute; at once when it can be reached
                if tasks.stop_wanted():
                    raise tasks.Stopped()
                time.sleep(1)
                if i % 10 == 9 and conn_issues.reachable(hosts):
                    break
            continue
        if lost is not None:
            print(f"[cloud ai] the connection to {name} is back", flush=True)
        return answer
