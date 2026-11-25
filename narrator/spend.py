"""Spending: what the app's paid work cost, and the guards that stop runaway spending.

Paid work, always on the user's own accounts: AI services (the sound director), cloud GPUs (Create in the cloud,
cloud speaker detection, training; Kaggle's are free weekly hours) and paid sound libraries.

The ledger, spending.jsonl in the data folder, one line each:
  {"type": "charge", time, service, kind (ai | gpu | sounds), what, model, dollars, exact, units, task, book}
      exact: the service's own figure (OpenRouter says what each request cost; Anthropic's come from its prices
      and the tokens it counted); otherwise an estimate (GPU time × list price, a model's list price).
  {"type": "run", time, task, what, service, estimate: {low, high}, spent}: a finished task that spent, with its
      estimate, so the next estimates can learn (pricing.py).

The guards:
- A day limit (settings spend_day_limit, dollars, 0 = none): paid work whose estimate would pass what's left today
  isn't started, and a task stops once today's spend reaches the limit.
- Runaway: a task that has cost more than twice its estimate's high end (25¢ past it at least) is stopped; the same
  AI request billed again and again within a task (LOOP times) is stopped; more than RATE paid AI requests a
  minute across the app are slowed down.
- Cloud jobs are written down as they're sent (cloud-jobs.json). One that nothing will come back for (speaker
  detection: its task ends with the app) is stopped when its task ends, or, after a crash or a quit, when the app
  starts again; Activity says so. (Create in the cloud is followed again after a restart, so it's left running.)
- Stop all paid work: every running task that has spent or was allowed to, and every cloud job.
"""
import collections
import errno
import json
import threading
import time

from . import settings
from .paths import HOME

FILE = HOME / "spending.jsonl"
JOBS = HOME / "cloud-jobs.json"
RATE = 90             # paid AI requests a minute across the app before they're slowed down
LOOP = 3              # the same request billed this many times in one task: something is looping
RUNAWAY_TIMES = 2.0   # a task past this × its estimate's high end (or that + RUNAWAY_SLACK dollars, whichever is
RUNAWAY_SLACK = 0.25  # more: a few cents' estimate isn't stopped over rounding) is stopped
_lock = threading.Lock()
_recent = collections.deque()           # times of paid AI requests (the rate guard)
_tasks = {}                             # task id -> {"spent", "estimate", "service", "what", "prints": Counter}


class Overspent(ValueError):
    """Stops paid work: a limit was reached (its message is for the user)."""


# ---------- the ledger ----------

def _append(entry):
    with _lock:
        try:
            with FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as e:
            if e.errno != errno.ENOSPC:
                raise
            print("[spend] disk full: couldn't append to spending.jsonl", flush=True)


def entries(since=0.0):
    try:
        lines = FILE.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for ln in lines:
        try:
            e = json.loads(ln)
        except ValueError:
            continue  # (a line cut short by a crash)
        if e.get("time", 0) >= since:
            out.append(e)
    return out


def _day_start(now=None):
    t = time.localtime(now or time.time())
    return time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, -1))


def _month_start(now=None):
    t = time.localtime(now or time.time())
    return time.mktime((t.tm_year, t.tm_mon, 1, 0, 0, 0, 0, 0, -1))


def spent_since(since):
    return sum(e.get("dollars") or 0 for e in entries(since) if e.get("type") == "charge")


def today():
    return spent_since(_day_start())


def day_limit():
    try:
        return max(0.0, float(settings.load().get("spend_day_limit") or 0))
    except (TypeError, ValueError):
        return 0.0


def _current():
    from . import tasks
    return tasks.CURRENT.get()


# ---------- AI agents (agents.py): their own day limit, on top of the owner's ----------

def _agent(t=None):
    from . import agents
    return bool(getattr(t, "via", None) or agents.current())


def agent_limit():
    """Dollars a day AI agents may spend (settings agent_spend_day_limit; 0: none at all)."""
    try:
        return max(0.0, float(settings.load().get("agent_spend_day_limit") or 0))
    except (TypeError, ValueError):
        return 0.0


def agent_today():
    return sum(e.get("dollars") or 0 for e in entries(_day_start()) if e.get("type") == "charge" and e.get("agent"))


def record(service, kind, what, dollars, exact=False, model="", units=None, book="", fingerprint=None, task=None,
           agent=None):
    """A charge. Inside a task, also checks that task's guards (Overspent stops it). agent: an AI agent's (else
    as the task or request it's made in)."""
    t = task or _current()
    agent = _agent(t) if agent is None else bool(agent)
    entry = {"type": "charge", "time": time.time(), "service": service, "kind": kind, "what": what, "model": model,
             "dollars": round(float(dollars or 0), 6), "exact": bool(exact), "units": units or {},
             "task": getattr(t, "id", None), "book": book}
    if agent:
        entry["agent"] = True
    _append(entry)
    if t is None:
        return entry
    with _lock:
        s = _tasks.setdefault(t.id, {"spent": 0.0, "estimate": None, "service": service, "what": what,
                                     "prints": collections.Counter()})
        s["spent"] += entry["dollars"]
        if fingerprint:
            s["prints"][fingerprint] += 1
        looping = fingerprint and s["prints"][fingerprint] >= LOOP
        est = s.get("estimate") or {}
        high = est.get("high")
        runaway = high is not None and high > 0 and s["spent"] > max(high * RUNAWAY_TIMES, high + RUNAWAY_SLACK)
    limit = day_limit()
    if looping:
        _stop(t, "the same request was paid for again and again (stopped in case something is looping).")
    elif runaway:
        _stop(t, f"it had cost ${s['spent']:.2f}, more than twice its estimate (up to ${high:.2f}).")
    elif limit and today() >= limit:
        _stop(t, f"today's spending reached your limit (${limit:.2f}, Settings → Spending).")
    elif agent and (spent := agent_today()) > 0 and spent >= agent_limit():
        _stop(t, f"AI agents' spending reached their limit today (${agent_limit():.2f}, Settings → Spending).")
    return entry


def _stop(task, why):
    task.stop_reason = why
    task.stop_requested.set()
    _append({"type": "alert", "time": time.time(), "task": task.id, "title": task.title, "why": why})
    print(f"[spend] stopping {task.id}: {why}", flush=True)


def _in_flight_reserved(exclude_task_id=None, agents_only=False):
    """Remaining high-end estimate reserved by other currently running paid tasks (agents_only: AI agents' tasks)."""
    with _lock:
        total = 0.0
        for tid, s in _tasks.items():
            if tid == exclude_task_id or (agents_only and not s.get("agent")):
                continue
            high = (s.get("estimate") or {}).get("high")
            if high and high > 0:
                total += max(0.0, float(high) - float(s.get("spent") or 0.0))
        return total


def allow(service, estimate, what):
    """Before paid work starts: refuses it when its estimate's high end would pass today's limit (including other
    in-flight reserved runs), and remembers the estimate ({"low", "high"} dollars) for the runaway guard and for
    learning (pricing.py)."""
    high = (estimate or {}).get("high")
    limit = day_limit()
    t = _current()
    if limit and high:
        reserved = _in_flight_reserved(getattr(t, "id", None))
        left = limit - today() - reserved
        if high > left:
            raise Overspent(f"This could cost up to ${high:.2f}, and ${max(0.0, left):.2f} of today's ${limit:.2f} "
                            "spending limit is left. Raise the limit in Settings → Spending, or try a smaller "
                            "model or a cheaper GPU.")
    agent = _agent(t)
    if agent:  # an AI agent's paid work: within the agents' own limit too ($0 at first: none)
        alimit = agent_limit()
        if not alimit:
            raise Overspent("AI agents can't start paid work: their limit is $0 a day. The owner can raise it in "
                            "Babble → Settings → Spending.")
        if high:
            left = alimit - agent_today() - _in_flight_reserved(getattr(t, "id", None), agents_only=True)
            if high > left:
                raise Overspent(f"This could cost up to ${high:.2f}, and ${max(0.0, left):.2f} of AI agents' "
                                f"${alimit:.2f} a day is left (Babble → Settings → Spending).")
    if t is not None:
        with _lock:
            s = _tasks.setdefault(t.id, {"spent": 0.0, "estimate": None, "service": service, "what": what,
                                         "prints": collections.Counter()})
            s.update(estimate=dict(estimate or {}), service=service, what=what, agent=agent)


def before_request(service):
    """Before a paid AI request: waits while the app has sent more than RATE in the last minute."""
    from . import tasks
    while True:
        now = time.time()
        with _lock:
            while _recent and now - _recent[0] > 60:
                _recent.popleft()
            if len(_recent) < RATE:
                _recent.append(now)
                return
            wait = 60 - (now - _recent[0])
        tasks.note(f"Slowing down: over {RATE} paid requests a minute ({wait:.0f}s)…")
        end = time.time() + min(wait, 5)
        while time.time() < end:
            if tasks.stop_wanted():
                raise tasks.Stopped()
            time.sleep(0.25)


def mark_partial(task=None):
    """Flag the current task as partial so its incomplete spend doesn't skew pricing.correction()."""
    t = task or _current()
    if t is None:
        return
    with _lock:
        if t.id in _tasks:
            _tasks[t.id]["partial"] = True


def task_ended(task):
    """Called as every task ends: a task that spent (or was allowed to) leaves a "run" line for learning."""
    with _lock:
        s = _tasks.pop(task.id, None)
    if not s or (not s["spent"] and not s.get("estimate")):
        return
    est = s.get("estimate") or {}
    _append({"type": "run", "time": time.time(), "task": task.id, "title": task.title, "what": s["what"],
             "service": s["service"], "model": est.get("model", ""), "partial": bool(s.get("partial")),
             "state": task.state, "estimate": s.get("estimate"), "spent": round(s["spent"], 6)})


def spending_tasks():
    with _lock:
        return set(_tasks)


# ---------- cloud jobs ----------

def _jobs():
    try:
        return json.loads(JOBS.read_text())
    except (OSError, ValueError):
        return {}


def _save_jobs(jobs):
    tmp = JOBS.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(jobs, indent=1))
        tmp.replace(JOBS)
    except OSError as e:
        if e.errno != errno.ENOSPC:
            raise
        print("[spend] disk full: couldn't save cloud-jobs.json", flush=True)
    finally:
        tmp.unlink(missing_ok=True)


def job_sent(backend, run, what, price=None, hours=None, collect=False, service=None, expect=None):
    """A job is out in the cloud. collect: a restarted app follows it again (Create in the cloud); else nothing
    will, and it's stopped after a crash or a quit. hours: its time limit; expect: the hours it's thought to take."""
    from . import tasks
    t = tasks.CURRENT.get()
    with _lock:
        jobs = _jobs()
        jobs[run] = {"backend": backend, "what": what, "price": price, "hours": hours, "collect": collect,
                     "sent": time.time(), "task": getattr(t, "id", None), "service": service or backend,
                     "expect": expect}
        if _agent(t):
            jobs[run]["agent"] = True  # (its GPU time, charged when it ends, is the agents')
        _save_jobs(jobs)


def job_seen(run):
    """The service said the job is still going (when it last was: how long it ran, if it ends unseen)."""
    with _lock:
        jobs = _jobs()
        if run in jobs:
            jobs[run]["seen"] = time.time()
            _save_jobs(jobs)


def job_ended(run, charge=True):
    """The job is over (back, failed or stopped): forgotten, and its GPU time charged (list price × time)."""
    with _lock:
        jobs = _jobs()
        j = jobs.pop(run, None)
        _save_jobs(jobs)
    if j and charge and j.get("price"):
        hours = (time.time() - j["sent"]) / 3600
        if j.get("hours"):
            hours = min(hours, j["hours"])
        record(j["service"], "gpu", j["what"], hours * j["price"], units={"gpu_hours": round(hours, 3)},
               model=j["backend"], agent=bool(j.get("agent")))
    return j


def stop_remote(backend, run):
    """Stop a job out in the cloud, whatever the service calls it (delete a rented machine, cancel a Modal call).
    False when the service can't be told (Kaggle: it ends at its own time limit, and it's free)."""
    from .training import BACKENDS
    b = BACKENDS.get(backend)
    for name in ("delete", "cancel"):
        fn = getattr(b, name, None)
        if fn is None:
            continue
        try:
            fn(run)
            return True
        except NotImplementedError:
            continue
    return False


def gpu_price(backend, gpu):
    """A cloud GPU's list price an hour (None: free, or not known)."""
    from .training import BACKENDS
    if backend == "modal":  # (as Modal's submit picks it: the GPU asked for, else its default)
        from .training.backends.modal_cloud import GPU, GPUS
        return GPUS.get(gpu if gpu in GPUS else GPU, (None, None))[1]
    b = BACKENDS.get(backend)
    names = getattr(b, "gpu_names", None)
    if names is not None and gpu not in names:
        gpu = getattr(b, "default_gpu", gpu)
    try:
        return b.price(gpu) if b is not None and hasattr(b, "price") else None
    except Exception:  # noqa: BLE001 - only for the estimate
        return None


def jobs_running():
    return _jobs()


def on_start():
    """At start: cloud jobs a crash or a quit left out there with nothing to come back for are stopped (in the
    background: it's network), noted in Activity and charged."""
    left = {r: j for r, j in _jobs().items() if not j.get("collect")}
    if not left:
        return

    def sweep():
        from . import activity
        for run, j in left.items():
            try:
                stopped = stop_remote(j["backend"], run)
            except Exception as e:  # noqa: BLE001 - don't abort stopping remaining jobs
                stopped = False
                print(f"[spend] couldn't stop {run} on {j['backend']}: {e}", flush=True)
            j = job_ended(run) or j
            mins = max(1, round((time.time() - j["sent"]) / 60))
            where = _name(j["backend"])
            note = (f"{j['what']} was still running on {where} when Babble closed; it was stopped."
                    if stopped else f"{j['what']} was still running on {where} when Babble closed. It can't be "
                    "stopped from here and ends by itself at its time limit" + (" (free)." if not j.get("price") else "."))
            activity._put({"id": f"left-{run}", "kind": "cloud", "title": f"{j['what']} · left running", "state": "stopped",
                           "status": note, "error": None, "started": j["sent"], "ended": time.time(),
                           "backend": j["backend"], "gpu_hours": round(mins / 60, 2)})
            print(f"[spend] {note}", flush=True)
    threading.Thread(target=sweep, daemon=True, name="spend-sweep").start()


# ---------- stop everything ----------

def stop_all():
    """Stop every running task that spends, every cloud job, and every book's cloud run."""
    from . import books, cloud_book, tasks
    stopped = []
    paying = spending_tasks()
    for t in tasks.all_tasks():
        if t.state == "running" and (t.id in paying or t.kind in ("direct", "cloud", "detect")):
            t.stop_reason = "you stopped all paid work."
            t.stop_requested.set()
            stopped.append(t.title)
    followed = {rec.get("run") for f in books.BOOKS.glob(f"*/{cloud_book.RECORD}")
                if (rec := cloud_book.read(f.parent) or {}).get("run")}
    for run, j in list(_jobs().items()):
        if j.get("collect") and run in followed:
            continue  # a book's cloud run: stopped below with its book (training runs and strays are stopped here)
        try:
            stop_remote(j["backend"], run)
        except Exception as e:  # noqa: BLE001 - keep stopping the rest
            print(f"[spend] couldn't stop {run} on {j['backend']}: {e}", flush=True)
        job_ended(run)
        stopped.append(j["what"])
    for f in books.BOOKS.glob(f"*/{cloud_book.RECORD}"):
        rec = cloud_book.read(f.parent) or {}
        if rec.get("state") in cloud_book.WAITING:
            cloud_book.forget(f.parent)
            stopped.append(f"Cloud run for {f.parent.name}")
    _append({"type": "alert", "time": time.time(), "why": "You stopped all paid work.", "stopped": stopped})
    return stopped


# ---------- for the app ----------

SERVICE_NAMES = {"anthropic": "Claude (Anthropic)", "openai": "OpenAI", "gemini": "Google Gemini", "openrouter": "OpenRouter",
                 "xai": "xAI Grok", "deepseek": "DeepSeek", "mistral": "Mistral", "groq": "Groq", "together": "Together AI",
                 "lotsofsounds": "Lots of Sounds", "freesound": "Freesound"}


def _name(service):
    if service in SERVICE_NAMES:
        return SERVICE_NAMES[service]
    from .training import BACKENDS
    b = BACKENDS.get(service)
    return b.name if b else service


def summary(recent=25):
    now = time.time()
    month = entries(_month_start(now))
    charges = [e for e in month if e.get("type") == "charge"]
    day0 = _day_start(now)
    per = {}
    for e in charges:
        s = per.setdefault(e["service"], {"service": e["service"], "name": _name(e["service"]), "today": 0.0,
                                          "month": 0.0, "requests_today": 0})
        s["month"] += e.get("dollars") or 0
        if e["time"] >= day0:
            s["today"] += e.get("dollars") or 0
            s["requests_today"] += 1
    alerts = [e for e in month if e.get("type") == "alert" and e["time"] >= now - 86400][-5:]
    jobs = [{"run": r, **j, "name": _name(j["backend"])} for r, j in _jobs().items()]
    rows = [e for e in charges][-recent:][::-1]
    return {"today": round(sum(s["today"] for s in per.values()), 4), "month": round(sum(s["month"] for s in per.values()), 4),
            "limit": day_limit(), "agent_limit": agent_limit(),
            "agent_today": round(sum(e.get("dollars") or 0 for e in charges if e.get("agent") and e["time"] >= day0), 4),
            "services": sorted(per.values(), key=lambda s: -s["month"]),
            "recent": [{"time": e["time"], "service": _name(e["service"]), "what": e["what"], "model": e.get("model", ""),
                        "dollars": e.get("dollars") or 0, "exact": e.get("exact", False)} for e in rows],
            "alerts": alerts[::-1], "jobs": jobs}


WHAT = {"direct": "Sound director", "create": "Create in the cloud", "qa": "Immersion QA",
        "prepare": "Dataset transcription"}


def history(limit=100):
    """Everything that cost money, newest first (Settings → Spending): one row per paid job, a task's charges
    together (a sound director run's hundreds of requests are one row), with its total, requests and GPU hours."""
    groups, order, runs = {}, [], {}
    for e in entries():
        if e.get("type") == "run" and e.get("task"):
            runs[e["task"]] = e
        if e.get("type") != "charge":
            continue
        key = e.get("task") or f"charge-{len(order)}"  # (no task: a charge of its own)
        g = groups.get(key)
        if g is None:
            g = groups[key] = {"task": e.get("task"), "started": e["time"], "service": e["service"],
                               "kind": e.get("kind", ""), "what": e.get("what", ""), "models": collections.Counter(),
                               "dollars": 0.0, "exact": True, "requests": 0, "gpu_hours": 0.0, "book": "",
                               "agent": False}
            order.append(g)
        g["time"] = e["time"]
        g["dollars"] += e.get("dollars") or 0
        g["exact"] = g["exact"] and bool(e.get("exact"))
        g["requests"] += 1
        g["gpu_hours"] += (e.get("units") or {}).get("gpu_hours") or 0
        if e.get("model"):
            g["models"][e["model"]] += 1
        g["book"] = g["book"] or e.get("book") or ""
        g["agent"] = g["agent"] or bool(e.get("agent"))
    out = []
    for g in sorted((g for g in order if g["dollars"] > 0), key=lambda g: -g["time"])[:limit]:
        run = runs.get(g["task"]) or {}
        out.append({"time": g["time"], "started": g["started"],
                    "title": run.get("title") or WHAT.get(g["what"], g["what"]), "state": run.get("state", ""),
                    "service": _name(g["service"]), "kind": g["kind"],
                    "model": g["models"].most_common(1)[0][0] if g["models"] else "",
                    "dollars": round(g["dollars"], 4), "exact": g["exact"], "book": g["book"],
                    "requests": g["requests"] if g["kind"] == "ai" else 0, "gpu_hours": round(g["gpu_hours"], 2),
                    "agent": g["agent"]})
    return out
