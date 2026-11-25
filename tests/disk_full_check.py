"""A full disk never hides a cloud run (without any cloud: stand-ins only).

    .venv/bin/python tests/disk_full_check.py

1. Saving a run's record on a full disk doesn't raise: the change is handed back, the last saved record stays whole,
   and the book page is told the disk is full; once there's room it's saved again and the note goes.
2. Other errors still raise.
3. The estimate's room check: the bytes a second of audio takes once back (more with sounds mixed in).
"""
import errno
import json
import os
import sys
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="disk-check-"))
os.environ["NARRATOR_HOME"] = str(TMP / "home")  # never touch the real data folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from narrator import cloud_book as cb  # noqa: E402

ok = True


def check(cond, what):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + what)


where = TMP / "book"
where.mkdir()
cb._save(where, {"run": "r1", "state": "running", "back": 10})
real = Path.write_text


def full(self, *a, **k):
    raise OSError(errno.ENOSPC, "No space left on device", str(self))


Path.write_text = full
try:
    rec = cb._update(where, back=20, checked=1.0)
    check(rec and rec["back"] == 20, f"a change on a full disk is handed back, not raised: {rec}")
    rec = cb._update_run(where, "r1", back=30)
    check(rec and rec["back"] == 30, "the same for the run's own changes")
    check(json.loads((where / cb.RECORD).read_text())["back"] == 10, "the last saved record stays whole")
    check(not list(where.glob(".*.part")), "no half-written file left")
    check(str(where) in cb.DISK_FULL, "noted as a full disk")
finally:
    Path.write_text = real
cb._update(where, back=40)
check(json.loads((where / cb.RECORD).read_text())["back"] == 40 and str(where) not in cb.DISK_FULL,
      "saved once there's room, and the note goes")


def denied(self, *a, **k):
    raise PermissionError(errno.EACCES, "Permission denied", str(self))


Path.write_text = denied
try:
    cb._update(where, back=50)
    check(False, "another error still raises")
except PermissionError:
    check(True, "another error still raises")
finally:
    Path.write_text = real

check(cb.free_bytes(where) > 0 and cb.free_bytes(TMP / "nope" / "x") is None, "free space is told (None for no folder)")
hour = 3600
check(abs(cb.disk_per_second(False) * hour / 1e6 - 291.6) < 1 and cb.disk_per_second(True) > cb.disk_per_second(False),
      f"an hour of audio takes {cb.disk_per_second(False) * hour / 1e6:.0f} MB, {cb.disk_per_second(True) * hour / 1e6:.0f} MB with sounds")
print("ALL OK" if ok else "SOME FAILED")
