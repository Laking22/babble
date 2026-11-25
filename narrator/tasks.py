"""Background tasks for the local API: a long operation (a book, speech, a sound) runs in its own thread and its
events are kept, so the desktop app can follow it, stop it, or pick it up again after its window reloads.

The heavy work inside still goes through the one MLX thread (runtime.py), so tasks never run two models at once:
a second task waits its turn there.
"""
import contextvars
import itertools
import json
import threading
import time
import traceback

from .paths import HOME
from .status import progress_of

_tasks = {}
_ids = itertools.count(1)
_lock = threading.Lock()
CURRENT = contextvars.ContextVar("task", default=None)  # the task this code runs for (MLX-thread work too)
KEEP = 50  # finished tasks remembered (in memory; a restart forgets them, the files they made stay)
RUNNING = HOME / "tasks-running.json"  # tasks under way, so the next start can say which ones a crash or quit cut off
_file_lock = threading.Lock()
AGAIN = {"direct": "Direct again: the scenes already directed are kept, so it carries on from there.",
         "book": "Create it again: the lines already made are kept.",
         "detect": "Detect again."}


class Stopped(Exception):
    """Raised inside a task's work when Stop was pressed and the work could be cut short (it was still waiting
    its turn, or a sound generator process was killed)."""


def stop_wanted():
    """True when the task this code runs for (MLX-thread work too) has been asked to stop."""
    task = CURRENT.get()
    return task is not None and task.stop_requested.is_set()


class Task:
    def __init__(self, kind, title):
        self.id = f"{kind}-{time.strftime('%H%M%S')}-{next(_ids)}"
        self.kind, self.title = kind, title
        self.detail = ""                # a line saying what it's of (the words spoken), for the Activity page
        self.state = "running"          # running, done, failed, stopped
        self.status, self.progress = "Starting…", None
        self.result = {}                # the latest of each event field besides status: audio, story, original, cast…
        self.error = None
        self.events = []                # [(n, event)], for streaming
        self.started, self.ended = time.time(), None
        self.updated = self.started  # the last event: a long quiet stretch shows in the app ("last news 3 min ago")
        self.eta = None              # seconds left, as of the last event (from how fast the progress has moved)
        self._pace = None            # (time, progress) where the current stretch of progress began
        self.stop_requested = threading.Event()
        self.stop_reason = ""        # why it was stopped, when it wasn't the Stop button (spend.py's guards)
        from . import agents
        self.via = agents.VIA.get()  # an AI agent started it (agents.py): its own spending limit applies

    def add(self, event):
        with _lock:
            n = len(self.events) + 1
            self.events.append((n, dict(event)))
            p = event.get("progress")  # how far along this step is, when the event says (0..1)
            if event.get("status"):
                self.status = str(event["status"])
                if p is None:
                    p = progress_of(event["status"])
            now = self.updated = time.time()
            if p is not None:
                p = max(0.0, min(1.0, float(p)))
                self._time_left(now, p)
                self.progress = p
            self.result.update({k: v for k, v in event.items() if k not in ("status", "progress")})

    def _time_left(self, now, p):
        """Seconds left from how fast progress has moved since this stretch began. A step that starts again from
        0 (a book's M4B after its chapters) starts a new stretch; under 10 s or 1% in, there's no estimate yet."""
        if self._pace is None or p < (self.progress or 0) - 0.05:
            self._pace, self.eta = (now, p), None
            return
        t0, p0 = self._pace
        if p >= 1.0:
            self.eta = 0
        elif now - t0 >= 10 and p - p0 >= 0.01:
            self.eta = (now - t0) * (1 - p) / (p - p0)

    def time_left(self):
        """Seconds left now (the estimate at the last event, less the time since), or None."""
        if self.eta is None or self.state != "running":
            return None
        if self.progress is not None and self.progress >= 0.995:
            return None
        rem = self.eta - (time.time() - self.updated)
        return rem if rem > 0 else None

    def info(self):
        return {"id": self.id, "kind": self.kind, "title": self.title, "state": self.state, "status": self.status,
                "progress": 1.0 if self.state == "done" else self.progress, "result": self.result,
                "error": self.error, "started": self.started, "ended": self.ended, "events": len(self.events),
                "eta": self.time_left(), "updated": self.updated,
                "stopping": self.state == "running" and self.stop_requested.is_set(), "agent": bool(self.via)}


def start(kind, title, make_events):
    """Run make_events() (a generator of event dicts, or a function returning one dict) in a thread."""
    task = Task(kind, title)
    with _lock:
        _tasks[task.id] = task
        done = [t for t in _tasks.values() if t.state != "running"]
        for t in sorted(done, key=lambda t: t.started)[:-KEEP]:
            _tasks.pop(t.id, None)

    def run():
        gen = None
        CURRENT.set(task)
        from . import agents
        agents.VIA.set(task.via)
        _mark(task, True)
        try:
            out = make_events()
            if isinstance(out, dict):
                task.add(out)
            else:
                gen = out
                for event in gen:
                    task.add(event)
                    if task.stop_requested.is_set():
                        gen.close()  # stops the voice model after the line it's on
                        task.state = "stopped"
                        task.add({"status": _stopped(task)})
                        break
            if task.state == "running":
                task.state = "done"
        except Stopped:
            task.state = "stopped"
            task.add({"status": _stopped(task)})
        except ValueError as e:  # a problem the user can fix: its message is meant for them
            task.state, task.error = "failed", str(e)
        except Exception as e:  # noqa: BLE001 - reported to the app, and in the log
            task.state, task.error = "failed", f"{type(e).__name__}: {e}"
            print(f"[task {task.id}] failed\n{traceback.format_exc(limit=6)}", flush=True)
        finally:
            task.ended = time.time()
            _mark(task, False)
            try:  # (what it spent, against its estimate)
                from . import spend
                spend.task_ended(task)
            except Exception as e:  # noqa: BLE001 - never breaks the task
                print(f"[spend] couldn't note {task.id}: {e}", flush=True)
            try:  # (the Activity page: what was made, and what failed)
                from . import activity
                activity.from_task(task)
            except Exception as e:  # noqa: BLE001 - never breaks the task
                print(f"[activity] couldn't note {task.id}: {e}", flush=True)

    threading.Thread(target=run, name=f"task-{task.id}", daemon=True).start()
    return task


def _mark(task, running):
    if task.title.startswith(("Preview", "Hear ")):
        return
    with _file_lock:
        try:
            d = json.loads(RUNNING.read_text())
        except (OSError, ValueError):
            d = {}
        if running:
            d[task.id] = {"kind": task.kind, "title": task.title, "started": task.started}
        else:
            d.pop(task.id, None)
        try:
            tmp = RUNNING.with_suffix(".tmp")
            tmp.write_text(json.dumps(d))
            tmp.replace(RUNNING)
        except OSError as e:
            print(f"[tasks] couldn't note running tasks: {e}", flush=True)


def recover():
    """At start: tasks that were running when Babble last closed (a crash, a quit, the Mac going off) are noted in
    Activity, with how to carry on."""
    with _file_lock:
        try:
            left = json.loads(RUNNING.read_text())
        except (OSError, ValueError):
            return []
        RUNNING.unlink(missing_ok=True)
    from . import activity
    for tid, t in left.items():
        activity._put({"id": tid, "kind": activity.KINDS.get(t["kind"], "other"), "title": t["title"],
                       "state": "failed", "status": "Interrupted: Babble closed while this was running.",
                       "error": "Babble closed while this was running. " + AGAIN.get(t["kind"], "Start it again."),
                       "started": t["started"], "ended": time.time(), "file": None})
    return list(left)


def _stopped(task):
    why = f" It was stopped because {task.stop_reason}" if task.stop_reason else ""
    return f"Stopped.{why} Anything already made is saved."


def note(status):
    """A status for the task this runs for, from anywhere (e.g. a model downloading inside a step that can't
    yield). No task: nothing."""
    task = CURRENT.get()
    if task is not None and task.state == "running":
        task.add({"status": status, "progress": getattr(status, "progress", None)})


def get(task_id):
    return _tasks.get(task_id)


def all_tasks():
    return sorted(_tasks.values(), key=lambda t: t.started, reverse=True)


def events_after(task, n):
    with _lock:
        return [e for e in task.events if e[0] > n]
