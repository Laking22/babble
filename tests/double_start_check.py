"""Run: .venv/bin/python tests/double_start_check.py  (a second or two, no models, scratch data folder only)

A double click on Create audiobook makes the book once: while a book is being made, Create for the same book gets the
run that's already going. Previews, other books, and the book again once the first run ended each start their own.
"""
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="doublestart-"))
os.environ["NARRATOR_HOME"] = str(TMP / "home")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
failed = []


def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what, flush=True)
    if not ok:
        failed.append(what)


from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from narrator import api, books, guard, tasks  # noqa: E402

release = threading.Event()
made = []


def fake_create(**kw):  # stands in for making the book: runs until the test lets it end
    made.append(kw.get("title"))
    yield {"status": "Speaking"}
    release.wait(10)
    yield {"status": "Done"}


books.create = fake_create
app = FastAPI()
api.install(app)
owner = TestClient(app, base_url="http://127.0.0.1", headers={"X-Babble-Token": guard.token()})
body = {"title": "Hive Minds", "story": "It began.", "folder": "hive-minds", "book": "p1", "cast": []}

a = owner.post("/api/books/create", json=body).json()
b = owner.post("/api/books/create", json=body).json()
check(a["id"] == b["id"], "a second Create while the book is being made gets the same run")

results = []
threads = [threading.Thread(target=lambda: results.append(owner.post("/api/books/create", json=body).json()["id"]))
           for _ in range(5)]
[t.start() for t in threads]
[t.join() for t in threads]
check(set(results) == {a["id"]}, "five at once still make it once")

other = owner.post("/api/books/create", json={**body, "book": "p2", "folder": "other"}).json()
check(other["id"] != a["id"], "another book starts its own")
p1 = owner.post("/api/books/create", json={**body, "preview": True}).json()
p2 = owner.post("/api/books/create", json={**body, "preview": True}).json()
check(len({p1["id"], p2["id"], a["id"]}) == 3, "previews aren't held back")

release.set()
end = time.time() + 5
while time.time() < end and tasks.get(a["id"]).state == "running":
    time.sleep(0.05)
c = owner.post("/api/books/create", json=body).json()
check(c["id"] != a["id"], "once it's done, Create makes the book again")
time.sleep(0.2)
check(made.count("Hive Minds") == 5, f"the book was started 5 times in all: 1 + other + 2 previews + again ({len(made)})")

print("\nall passed" if not failed else f"\n{len(failed)} failed: " + "; ".join(failed))
sys.exit(1 if failed else 0)
