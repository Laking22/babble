"""Progress for work that can't report its own, like a download filling a folder: how much so far, how fast, how
much is left ("Downloading the trained model · 1.6 of 3.6 GB · 5.0 MB/s · about 7 min left")."""
import contextlib
import contextvars
import os
import threading
import time

from .status import Status


def size_of(folder):
    """Bytes in a folder and everything under it (links not followed: Hugging Face's snapshots link to its blobs)."""
    total = 0
    for root, _, files in os.walk(folder):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def amount(n):
    n = max(0, n)
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB" if n >= 1e6 else f"{n / 1e3:.0f} KB"


def left(seconds):
    s = int(seconds)
    if s < 60:
        return "under a minute left"
    if s < 3600:
        return f"about {round(s / 60)} min left"
    return f"about {s // 3600} h {round(s % 3600 / 60)} min left"


def _join(*parts):
    return " · ".join(x for x in parts if x)


class Meter:
    """Bytes arriving, told by the download itself: "3.9 of 4.5 GB · 2.6 MB/s · about 4 min left" and how far along
    (0-1), with the speed smoothed over the last few seconds."""

    def __init__(self):
        self.t, self.n, self.speed = None, 0, None

    def __call__(self, done, total):
        t = time.time()
        if self.t is None or done < self.n:  # (timed from the first bytes, not from asking for them)
            self.t, self.n = t, done
        elif t - self.t >= 1.0:
            step = (done - self.n) / (t - self.t)
            self.speed = step if self.speed is None else 0.7 * self.speed + 0.3 * step
            self.t, self.n = t, done
        fast = self.speed is not None and self.speed >= 1e3
        rate = f"{amount(self.speed)}/s" if fast else ""
        eta = left((total - done) / self.speed) if fast and done < total else ""
        size = f"{amount(done)} of {amount(total)}" if total else f"{amount(done)} so far"
        return _join(size, rate, eta), (min(done / total, 1.0) if total else None)


def stages(log, steps):
    """stage(k) -> a log for step k of steps ([(what it's doing, its share of the whole bar)]):
    "Step 2 of 4 · Downloading the trained model · <what it says>", with the whole job's progress (the shares
    before it, plus how far this step is along when it says so: a Status with progress)."""
    def stage(k):
        what, share = steps[k]
        start, last = sum(s for _, s in steps[:k]), [0.0]

        def say(msg=""):
            p = getattr(msg, "progress", None)
            if p is not None:
                last[0] = max(last[0], min(float(p), 1.0))
            note = str(msg).rstrip(".…") if msg else ""
            head = f"Step {k + 1} of {len(steps)} · {what}" if len(steps) > 1 else what
            log(Status(head + (f" · {msg}" if note and note != what else ""), start + share * last[0]))
        say()
        return say
    return stage


@contextlib.contextmanager
def downloading(folder, what, report, total=None, every=2.0, quiet_below=10e6):
    """While the block runs, report(Status) every couple of seconds with what has arrived in folder since it
    began, how fast, and (with total, the bytes expected) how far along and how long is left. Says nothing until
    quiet_below bytes have come when the total isn't known: loading a model already on disk isn't a download."""
    folder = str(folder)
    os.makedirs(folder, exist_ok=True)
    start, began = size_of(folder), time.time()
    stop = threading.Event()

    def watch():
        last_t, last_n, speed = began, 0, None
        while not stop.wait(every):
            now, t = size_of(folder) - start, time.time()
            if (now < quiet_below and not total) or time.time() - _hf_told[0] < 5:
                continue  # (nothing to say yet, or Hugging Face's own count is saying it)
            step = (now - last_n) / max(t - last_t, 1e-6)
            speed = step if speed is None else 0.7 * speed + 0.3 * step  # smoothed
            last_t, last_n = t, now
            rate = f"{amount(speed)}/s" if speed and speed > 0 else "waiting for data"
            try:
                if total:
                    p = min(now / total, 0.99)
                    eta = (f" · {left((total - now) / speed)}" if speed and speed > 0 and now < total
                           and t - began >= 6 else "")  # (the first seconds are too uneven to tell from)
                    report(Status(_join(what, f"{amount(now)} of {amount(total)} · {rate}{eta}"), p))
                else:
                    report(Status(_join(what, f"{amount(now)} so far · {rate}")))
            except Exception:  # noqa: BLE001 - reporting must never break the download
                pass

    # (in this code's context: tasks.note, as report, finds the task it's for)
    th = threading.Thread(target=contextvars.copy_context().run, args=(watch,), name="download-progress", daemon=True)
    th.start()
    try:
        yield
    finally:
        stop.set()
        th.join(timeout=every + 1)


_hf_told = [0.0]  # when Hugging Face's own count last said something (the folder watcher keeps quiet meanwhile)


@contextlib.contextmanager
def hf_counting(report, what=""):
    """While the block runs, Hugging Face downloads that don't pass their own tqdm_class (a model loading for
    the first time) report through hf_bars too."""
    from huggingface_hub import _snapshot_download as snap
    before = snap.hf_tqdm
    snap.hf_tqdm = hf_bars(report, what)
    try:
        yield
    finally:
        snap.hf_tqdm = before


def hf_bars(report, what="", total=None):
    """A progress-bar class for Hugging Face's snapshot_download(tqdm_class=...): its own count of the bytes that
    have come (Hugging Face now holds a file in memory and writes it whole at the end, so watching the folder
    sees nothing until then). report(Status) about once a second: "1.2 of 3.8 GB · 5 MB/s · about 8 min left".
    total: the bytes expected, when known (else Hugging Face's own total, which grows as each file starts)."""
    from tqdm.std import tqdm as plain
    bars, meter, said = [], Meter(), [0.0]

    def tell(force=False):
        if not force and time.time() - said[0] < 1.0:
            return
        said[0] = time.time()
        counts = [b for b in bars if (b.unit or "") == "B"]  # (not the "Fetching 12 files" bar)
        done = max((b.n for b in counts), default=0)
        whole = total or max((b.total or 0 for b in counts), default=0)
        if not whole and done < 1e6:
            return
        text, share = meter(int(min(done, whole) if whole else done), int(whole))
        _hf_told[0] = time.time()
        try:
            report(Status(_join(what, text), min(share, 0.99) if share is not None else None))
        except Exception:  # noqa: BLE001 - reporting must never break the download
            pass

    class Bar(plain):
        def __init__(self, *args, **kwargs):
            kwargs.pop("name", None)
            kwargs.update(file=open(os.devnull, "w"), disable=False, mininterval=3600)  # (counts, draws nothing)
            super().__init__(*args, **kwargs)
            bars.append(self)

        def update(self, n=1):
            out = super().update(n)
            tell()
            return out
    return Bar


def hf_cache():
    try:
        from huggingface_hub import constants
        return constants.HF_HUB_CACHE
    except Exception:  # noqa: BLE001
        return os.path.expanduser("~/.cache/huggingface/hub")


def hf_size(repo):
    """Bytes in a Hugging Face model repo, or None (offline, private, …)."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo, files_metadata=True)
        return sum(s.size or 0 for s in info.siblings or []) or None
    except Exception:  # noqa: BLE001
        return None
