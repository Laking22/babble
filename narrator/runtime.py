"""One MLX thread and one model slot shared by every engine.

MLX work must stay on one thread, and on an 8 GB Mac only one big model fits at a time.
Every engine loads its models through `slot`, so loading a Kokoro model frees the Orpheus one,
loading Whisper frees Qwen, and so on.
"""
import contextvars
import functools
import gc
import queue
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

try:
    import mlx.core as mx
except ImportError:
    mx = None

HAS_MLX = mx is not None
SR = 24000
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")


def total_ram_bytes():
    """System RAM in bytes across macOS, Windows, and Linux (0 if unavailable)."""
    if sys.platform == "darwin":
        import subprocess
        try:
            return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
        except (ValueError, OSError):
            return 0
    if sys.platform == "win32":
        import ctypes
        class _MEM(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        try:
            m = _MEM(dwLength=ctypes.sizeof(_MEM))
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return int(m.ullTotalPhys)
        except Exception:  # noqa: BLE001
            return 0
        return 0
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _cache_limit_bytes():
    """MLX keeps memory it has finished with to reuse it, and by default that store can grow to most of the Mac's
    memory. With Orpheus batching on an 8 GB Mac it passed 1.8 GB after two lines and free memory fell to 4%
    (the app's footprint had reached 6.4 GB in real renders); capped at 1 GB, six lines ran with 18% still free.
    So: 1 GB per 8 GB of memory."""
    ram = total_ram_bytes() or (8 * 2 ** 30)
    return max(2 ** 29, ram // 8)


CACHE_LIMIT = _cache_limit_bytes()
if mx is not None:
    mx.set_cache_limit(CACHE_LIMIT)


# what the MLX thread is busy with, for "Waiting its turn…": a task's title, or LABEL for work that isn't a task
# (e.g. a voice's training data, which sets it for its own thread)
LABEL = contextvars.ContextVar("mlx_work", default=None)
_busy = None


def _label():
    from . import tasks
    t = tasks.CURRENT.get()
    return LABEL.get() or (f"“{t.title}”" if t else None)


def _submit(fn, *args, **kwargs):
    """Queue fn for the MLX thread (with the caller's context, so values like the book a sound belongs to
    carry over), noting what it is while it runs."""
    label = _label()

    def job():
        global _busy
        _busy = label or "another job"
        try:
            return fn(*args, **kwargs)
        finally:
            _busy = None
    return _pool.submit(contextvars.copy_context().run, job)


def _waiting(fut, told):
    """While queued: say once what the MLX thread is busy with, and give up the place when Stop is pressed.
    Returns whether it has said so."""
    from . import tasks
    if fut.running():
        return told
    if tasks.stop_wanted() and fut.cancel():
        raise tasks.Stopped()
    if not told:
        tasks.note(f"Waiting its turn: Babble is busy with {_busy or 'another job'}…")
    return True


def on_mlx_thread(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if threading.current_thread().name.startswith("mlx"):
            return fn(*args, **kwargs)  # already there: don't deadlock waiting on ourselves
        fut, told = _submit(fn, *args, **kwargs), False
        while True:
            try:
                return fut.result(timeout=0.5)
            except FutureTimeout:
                told = _waiting(fut, told)
    return wrapper


def streamed_on_mlx_thread(genfn):
    """Run a generator on the MLX thread; relay what it yields. Closing it asks the job to stop."""
    @functools.wraps(genfn)
    def wrapper(*args, **kwargs):
        q, stop = queue.Queue(), threading.Event()

        def job():
            try:
                if stop.is_set():
                    return
                for item in genfn(*args, stop=stop, **kwargs):
                    q.put(("item", item))
            except BaseException as e:  # noqa: BLE001 - re-raised in the caller's thread
                q.put(("error", e))
            finally:
                q.put(("end", None))

        fut, told = _submit(job), False
        try:
            while True:
                try:
                    kind, value = q.get(timeout=0.5)
                except queue.Empty:
                    from . import tasks
                    if tasks.stop_wanted():
                        stop.set()
                        if fut.cancel():
                            raise tasks.Stopped()
                    told = _waiting(fut, told)
                    continue
                if kind == "item":
                    yield value
                elif kind == "error":
                    raise value
                else:
                    return
        finally:
            stop.set()
            fut.cancel()
    return wrapper


def free():
    gc.collect()
    if mx is not None:
        mx.clear_cache()
    torch = sys.modules.get("torch")  # (Windows/Linux voices: tts/torch_rt.py; never imported just for this)
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


class Slot:
    """Holds at most one loaded model. `get(key, loader)` returns it, loading (and freeing the old one) if needed."""

    def __init__(self):
        self.key, self.model = None, None
        self._on_release = []

    def get(self, key, loader):
        if self.key != key or self.model is None:
            self.release()
            from . import progress, tasks
            # a model used for the first time downloads here (several GB): say so, with how far along it is
            what = "Downloading a model (the first time only)"
            with progress.downloading(progress.hf_cache(), what, tasks.note), progress.hf_counting(tasks.note, what):
                self.model, self.key = loader(), key
        return self.model

    def release(self):
        if self.model is not None or self.key is not None:
            self.model, self.key = None, None
            for cb in self._on_release:
                cb()
        free()

    def on_release(self, cb):
        """Engines that keep state tied to the loaded model (e.g. Qwen's saved voices) reset it here."""
        self._on_release.append(cb)


slot = Slot()
