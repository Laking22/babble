"""How much memory and disk Babble is using, for the meters in the sidebar.

Memory: the memory footprint (what Activity Monitor's Memory column shows, the voice models' GPU memory included)
of the engine, the processes it started, and the app's window process when the app started the engine.
Storage: the data folder (books, voices, trained models, sounds…) plus the downloaded models Babble uses (the
Hugging Face downloads named in its code; others in that shared folder aren't counted). Measured at most once a
minute, in the background, since it means walking the folders.
"""
import ctypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from .paths import HOME

EXTRA_MODELS = {"mlx-community/snac_24khz"}  # pulled in by Orpheus on the Mac without being named here
_storage = {"at": 0.0, "value": None, "busy": False}
_lock = threading.Lock()


class _Rusage(ctypes.Structure):  # rusage_info_v2 (sys/resource.h): the fields up to the footprint
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size", "resident_size",
        "phys_footprint", "proc_start_abstime", "proc_exit_abstime", "child_user_time", "child_system_time",
        "child_pkg_idle_wkups", "child_interrupt_wkups", "child_pageins", "child_elapsed_abstime",
        "diskio_bytesread", "diskio_byteswritten")]


_libproc = None


def _win_footprint(pid):
    class _PMC(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]
    try:
        k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        k32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        k32.OpenProcess.restype = ctypes.c_void_p
        k32.CloseHandle.argtypes = [ctypes.c_void_p]
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        h = k32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return 0
        try:
            pmc = _PMC(cb=ctypes.sizeof(_PMC))
            if psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
                return int(pmc.WorkingSetSize)
        finally:
            k32.CloseHandle(h)
    except Exception:  # noqa: BLE001
        pass
    return 0


def footprint(pid):
    """A process's memory footprint in bytes (0 if it can't be read)."""
    global _libproc
    if sys.platform == "win32":
        return _win_footprint(pid)
    if sys.platform != "darwin":
        try:
            for ln in Path(f"/proc/{int(pid)}/status").read_text().splitlines():
                if ln.startswith("VmRSS:"):
                    return int(ln.split()[1]) * 1024
        except Exception:  # noqa: BLE001
            pass
        return 0
    try:
        if _libproc is None:
            _libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
        info = _Rusage()
        if _libproc.proc_pid_rusage(int(pid), 2, ctypes.byref(info)) == 0:
            return int(info.phys_footprint)
    except Exception:  # noqa: BLE001 - not a Mac, or the process is gone
        pass
    return 0


def _ps(*args):
    try:
        return subprocess.run(["ps", *args], capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return ""


def processes():
    """The engine, everything it started, and the app window that started it."""
    me = os.getpid()
    if sys.platform == "win32":
        return [me]
    rows = [ln.split(None, 2) for ln in _ps("-axo", "pid=,ppid=,comm=").splitlines() if ln.strip()]
    kids = {}
    for r in rows:
        if len(r) >= 2:
            kids.setdefault(int(r[1]), []).append(int(r[0]))
    out, todo = [me], [me]
    while todo:
        for c in kids.get(todo.pop(), []):
            out.append(c)
            todo.append(c)
    parent = next((r for r in rows if len(r) == 3 and int(r[0]) == os.getppid()), None)
    if parent and re.search(r"babble|narrator-studio", parent[2], re.I):
        out.append(int(parent[0]))
    return out


def memory():
    from .tts.base import ram_gb
    return {"used": sum(footprint(p) for p in processes()), "total": int(ram_gb() * 1024 ** 3)}


def _models_used():
    """Bytes of the downloaded models Babble's code names."""
    from huggingface_hub import scan_cache_dir
    src = "\n".join(p.read_text(errors="ignore") for p in Path(__file__).parent.rglob("*.py"))
    named = set(re.findall(r"[\"']([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)[\"']", src)) | EXTRA_MODELS
    try:
        return sum(r.size_on_disk for r in scan_cache_dir().repos if r.repo_id in named)
    except Exception:  # noqa: BLE001 - no downloads yet
        return 0


def _measure():
    from .progress import size_of
    try:
        data, models = size_of(HOME), _models_used()
        disk = shutil.disk_usage(str(HOME))
        value = {"used": data + models, "data": data, "models": models, "free": disk.free, "total": disk.total}
        with _lock:
            _storage.update(at=time.time(), value=value)
    finally:
        _storage["busy"] = False


def storage(max_age=60):
    """The last measure (re-measured in the background when it's older than max_age seconds; the first time,
    right away)."""
    with _lock:
        stale = time.time() - _storage["at"] > max_age
        start = stale and not _storage["busy"]
        if start:
            _storage["busy"] = True
    if start:
        if _storage["value"] is None:
            _measure()
        else:
            threading.Thread(target=_measure, daemon=True, name="storage-meter").start()
    return _storage["value"]
