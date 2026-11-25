"""Activity: what the app made lately, kept across restarts (the Activity page). Audiobooks, text to speech, sound
effects, backgrounds and music; cloud runs with their GPU time and what they cost; and anything that failed, with
its error. Saved in activity.json in the data folder, newest first, at most KEEP entries.

Entries come from tasks as they end (tasks.run), and from cloud_book for a run out in the cloud (sent, then
finished, failed or stopped). Cost is the GPU's price an hour times the time from sending to the run ending: an
estimate (the provider's bill is what counts), $0 on Kaggle.
"""
import json
import threading
import time

from .paths import HOME

FILE = HOME / "activity.json"
KEEP = 300
_lock = threading.Lock()

# what a task is, for the filters (kinds not listed are kept only when they fail: "other")
KINDS = {"book": "audiobook", "speech": "speech", "cloud": "cloud"}
SOUND = {"effect": "effect", "ambience": "background", "music": "music"}
PREVIEW = ("Preview", "Hear ")  # voice previews: quick checks, not something made (kept only when they fail)


def _read():
    try:
        return json.loads(FILE.read_text())
    except (OSError, ValueError):
        return []


def _write(entries):
    tmp = FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries[:KEEP], ensure_ascii=False, indent=0))
    tmp.replace(FILE)


def _put(entry):
    """Add an entry, or update the one with the same id (keeping its place)."""
    with _lock:
        entries = _read()
        for i, e in enumerate(entries):
            if e.get("id") == entry["id"]:
                entries[i] = {**e, **entry}
                break
        else:
            entries.insert(0, entry)
        _write(entries)


def entries():
    with _lock:
        return _read()


def clear():
    with _lock:
        _write([])


def _kind(task):
    if task.kind == "sounds":
        sound = task.result.get("sound") or {}
        if sound.get("kind") in SOUND:
            return SOUND[sound["kind"]]
        return "music" if "theme" in task.title.lower() else "effect"
    return KINDS.get(task.kind, "other")


def _file(result):
    """The file a task made, when it says (an audiobook's audio, a sound's file)."""
    sound = result.get("sound") or {}
    for v in (result.get("audio"), result.get("out"), sound.get("path")):
        if isinstance(v, str) and v:
            return v
    return None


def title_of(task):
    """What the page calls it: speech by its words."""
    if _kind(task) == "speech" and getattr(task, "detail", ""):
        return f"“{task.detail}”" + ("…" if len(task.detail) >= 90 else "")
    return task.title


def from_task(task):
    """Called as a task ends."""
    kind = _kind(task)
    failed = task.state == "failed"
    if not failed and (kind == "other" or task.title.startswith(PREVIEW)):
        return  # (only what was made is kept, and every failure)
    entry = {"id": task.id, "kind": kind, "title": title_of(task), "state": task.state,
             "status": (task.status or "")[:400], "error": (task.error or "")[:1000] or None,
             "started": task.started, "ended": task.ended or time.time(), "file": _file(task.result)}
    cloud = task.result.get("cloud") if kind == "cloud" else None
    if cloud and not failed and cloud.get("run"):
        name = task.title.split(": ", 1)[-1] if task.title.startswith("Send to the cloud") else task.title
        entry.update(_cloud_fields(cloud), id=f"cloud-{cloud['run']}", state="running", title=f"{name} · cloud run",
                     status=f"Out in the cloud: {cloud.get('lines', 0):,} lines on {_where(cloud)}")
        entry.pop("ended", None)
    _put(entry)


def _where(rec):
    from .training import BACKENDS
    b = BACKENDS.get(rec.get("backend", ""))
    return f"{b.name if b else rec.get('backend', 'the cloud')}" + (f" ({rec['gpu_label']})" if rec.get("gpu_label") else "")


def _cloud_fields(rec):
    return {"backend": rec.get("backend"), "gpu": rec.get("gpu_label") or rec.get("gpu"), "lines": rec.get("lines"),
            "price": rec.get("price"), "max_cost": rec.get("max_cost"), "started": rec.get("sent")}


def cloud_ended(rec, state, error=None, title=None):
    """A cloud run is over (its lines came back, it failed, or it was stopped): its GPU time and cost."""
    if not rec or not rec.get("run"):
        return
    now = time.time()
    ended = rec.get("finished_at") or now  # (when the cloud said it was done, before the download)
    hours = max(0.0, (ended - (rec.get("sent") or ended)) / 3600)
    price = rec.get("price")
    cost = round(hours * price, 2) if price is not None else None
    if cost is not None and rec.get("max_cost") is not None:
        cost = min(cost, rec["max_cost"])  # (it can't run past its time limit)
    made, lines = rec.get("made") or rec.get("back") or 0, rec.get("lines") or 0
    status = {"done": f"{made:,} of {lines:,} lines back from {_where(rec)}",
              "stopped": f"Stopped on {_where(rec)}: {made:,} of {lines:,} lines had come back",
              "failed": f"Failed on {_where(rec)}"}[state]
    entry = {"id": f"cloud-{rec['run']}", "kind": "cloud", "state": state, "status": status,
             "error": (error or rec.get("error") or None) if state != "done" or rec.get("error") else None,
             "ended": now, "gpu_hours": round(hours, 2), "cost": cost, **_cloud_fields(rec)}
    if title:
        entry["title"] = title
    try:  # the spending ledger: its GPU time at the list price (once: only while it's still written down as out)
        from . import spend
        job = spend.job_ended(rec["run"], charge=False)
        if job is not None and cost:
            spend.record(rec.get("backend", ""), "gpu", "create", cost, units={"gpu_hours": round(hours, 3)},
                         model=rec.get("gpu_label") or rec.get("gpu") or "", agent=bool(job.get("agent")))
    except Exception as e:  # noqa: BLE001 - never stops the run being noted
        print(f"[spend] couldn't note {rec.get('run')}: {e}", flush=True)
    with _lock:
        known = any(e.get("id") == entry["id"] for e in _read())
    if not known:
        entry.setdefault("title", f"Cloud run on {_where(rec)}")
    _put(entry)
