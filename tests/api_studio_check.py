"""Checks for the desktop app's new API (projects, chapters, book sounds, voices, packs, models, connections, train)
without loading any voice model: runs against a scratch NARRATOR_HOME.

    .venv/bin/python tests/api_studio_check.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

HOME = Path(tempfile.mkdtemp(prefix="narrator-api-check-"))
os.environ["NARRATOR_HOME"] = str(HOME)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from narrator import api, books, guard, sfx  # noqa: E402

app = FastAPI()
api.install(app)
c = TestClient(app, base_url="http://127.0.0.1", headers={"X-Babble-Token": guard.token()})
failed = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failed.append(name)


# ---------- projects ----------
r = c.post("/api/projects", json={})
p1 = r.json()
check("new project", r.status_code == 200 and p1["untitled"] == 1 and p1["folder"] == f"book-{p1['id']}", r.text)
p2 = c.post("/api/projects", json={"story": "## One\n\nHello there."}).json()
check("second untitled is Untitled-2", p2["untitled"] == 2)
s = c.put(f"/api/projects/{p1['id']}", json={"title": "The Harbour", "story": "## A\n\nOne.\n\n## B\n\nTwo."}).json()
check("save returns a summary", s["name"] == "The Harbour" and s["chapters"] == 2, json.dumps(s))
p3 = c.post("/api/projects", json={}).json()
check("next untitled skips titled books", p3["untitled"] == 3)
check("unknown fields refused", c.put(f"/api/projects/{p1['id']}", json={"nope": 1}).status_code == 400)
check("bad id refused", c.get("/api/projects/../../etc").status_code in (400, 404))
lst = c.get("/api/projects").json()
check("list has 3, newest first", len(lst) == 3 and lst[0]["id"] == p3["id"])
check("list has no story text", "story" not in lst[0])
mig = c.post("/api/projects", json={"folder": "story", "story": "x"}).json()
check("an existing folder can be kept", mig["folder"] == "story")
check("a bad folder isn't", c.post("/api/projects", json={"folder": "../x"}).json()["folder"].startswith("book-"))
check("delete", c.delete(f"/api/projects/{p3['id']}").status_code == 200 and c.get(f"/api/projects/{p3['id']}").status_code == 400)

# ---------- chapters ----------
story = "## The Harbour\n\nMara came in.\n\n## The Crossing\n\nThe sea was rough."
ch = c.post("/api/books/chapters", json={"story": story, "folder": p1["folder"]}).json()
check("chapters listed, none made", [x["title"] for x in ch] == ["The Harbour", "The Crossing"]
      and all(x["state"] == "not made" for x in ch), json.dumps(ch))
folder = books.BOOKS / p1["folder"]
folder.mkdir(parents=True, exist_ok=True)
import soundfile as sf  # noqa: E402
wav = folder / "000-abc.wav"
sf.write(wav, np.zeros(24000, dtype=np.float32), 24000)
chs = books.story_chapters(story)
(folder / books.MANIFEST).write_text(json.dumps({"chapters": {"0": {"title": "The Harbour", "file": str(wav), "seconds": 1.0,
                                                                     "text": books.text_key(chs[0][0], chs[0][1])}}}))
ch = c.post("/api/books/chapters", json={"story": story, "folder": p1["folder"]}).json()
check("made chapter is done", ch[0]["state"] == "done" and ch[0]["file"] == str(wav) and ch[1]["state"] == "not made", json.dumps(ch))
ch = c.post("/api/books/chapters", json={"story": story.replace("Mara came in.", "Mara ran in."), "folder": p1["folder"]}).json()
check("edited chapter is edited", ch[0]["state"] == "edited")
check("bad folder refused", c.post("/api/books/chapters", json={"story": story, "folder": "../../x"}).status_code == 400)

# ---------- per-character stability ----------
temps = books.voice_temps({"Tom": 40, "Mara": 55, "Ada": None}, 55)
check("only differing stabilities get a temperature", temps == {"Tom": books.temp_from_stability(40)}, str(temps))
built, _, _ = books.build_book("[Tom] \"Hi,\" said Tom.\n\nThe end.", [["Narrator", "Ryan", "", 1], ["Tom", "Ryan", "", 1]],
                               True, 1.0, 0.2, 0.7, "cast", temps)
items = built[0][1]
check("Tom's lines carry his temperature", any(it.get("who") == "Tom" and it.get("temperature") == temps["Tom"] for it in items)
      and all("temperature" not in it for it in items if it.get("who") == "Narrator"), json.dumps(items))
from narrator.tts.qwen import _batches  # noqa: E402
groups = [len(g) for _, g in _batches([{"temperature": None}] * 3 + [{"temperature": 0.9}] * 2 + [{"temperature": None}], 2)]
check("batches split where the temperature changes", groups == [2, 1, 2, 1], str(groups))

# ---------- sounds kept in a book ----------
sfx.LIB.mkdir(parents=True, exist_ok=True)
shared = sfx.add(np.random.default_rng(0).normal(0, .1, 24000).astype(np.float32), "Door Creak", "effect")
local = sfx.add(np.random.default_rng(1).normal(0, .1, 24000).astype(np.float32), "Tavern Bell", "effect")
sfx.set_scope(local["id"], p1["id"], made_in=p1["id"])
story2 = "{sfx: door creak} One. {sfx: tavern bell} Two. {sfx: dragon roar}"
other = c.post("/api/books/sounds", json={"story": story2, "book": p2["id"]}).json()
mine = c.post("/api/books/sounds", json={"story": story2, "book": p1["id"]}).json()
check("another book can't use a book's own sound",
      [bool(x["sound"]) for x in other] == [True, False, False], json.dumps(other))
check("its own book can", [bool(x["sound"]) for x in mine] == [True, True, False]
      and mine[1]["sound"]["book"] == p1["id"], json.dumps(mine))
check("no book set: shared sounds only", sfx.find("tavern bell") is None and sfx.find("door creak") is not None)
tok = sfx.BOOK.set(p1["id"])
check("while a book is worked on, its sounds match", (sfx.find("tavern bell") or {}).get("id") == local["id"])
from narrator.runtime import on_mlx_thread  # noqa: E402
check("…on the voice thread too", (on_mlx_thread(lambda: sfx.find("tavern bell"))() or {}).get("id") == local["id"])
sfx.BOOK.reset(tok)
r = c.post(f"/api/sounds/{local['id']}/scope", json={"book": None}).json()
check("share it with every book", "book" not in r and r["made_in"] == p1["id"] and sfx.find("tavern bell") is not None)
check("intro choices for a book", any(x["id"] == local["id"] for x in c.get(f"/api/books/intro-choices?book={p2['id']}").json()))

# ---------- dragging sounds into a story ----------
pal = c.get(f"/api/books/palette?book={p1['id']}").json()
check("palette: every sound with a tag", pal and all(x["tag"].startswith("{") for x in pal), json.dumps(pal)[:200])
check("palette tags play exactly that sound", all(
    (sfx.find_for(sfx.split_cues(x["tag"])[0][1]) or {}).get("id") == x["id"] for x in pal))
other = c.get(f"/api/books/palette?book={p2['id']}").json()
check("another book doesn't see this book's own sounds", all(not x["mine"] for x in other))
base = {"text": "", "voice": "v", "pause": 0.6}
its = sfx.cue_items("He opened {sfx: door knock} the door, and the door creaked.", dict(base))
check("a tag inside a sentence keeps it whole", [i.get("text") for i in its if "text" in i] == ["He opened the door, and the door creaked."]
      and its[1]["cue"]["at"] == "the" and its[1]["cue"]["nth"] == 0 and its[0]["pause"] == 0.6, json.dumps(its))
its = sfx.cue_items("He knocked on the door and {sfx: door knock} the door opened.", dict(base))
check("…and knows which time its word comes", its[1]["cue"]["at"] == "the" and its[1]["cue"]["nth"] == 1, json.dumps(its))
its = sfx.cue_items("The door slammed. {sfx: door knock} Nobody moved.", dict(base))
check("a tag between sentences still splits there", [i.get("text") for i in its if "text" in i] == ["The door slammed.", "Nobody moved."]
      and "at" not in its[1]["cue"] and its[0]["pause"] == 0.25, json.dumps(its))

# ---------- training runs that fail ----------
from narrator import training  # noqa: E402
from narrator.training.base import save_run  # noqa: E402
save_run("qwen/oomtest", model="oomtest", recipe="qwen", backend="kaggle", state="complete", run_id="me/x",
         voices={"ann": "ann"}, epochs=3.0, small_outputs=".*", big_outputs="x", sent="2026-01-01 12:00")
out = training.results_dir("qwen/oomtest")
out.mkdir(parents=True, exist_ok=True)
(out / "run.log").write_text(json.dumps([{"stream_name": "stderr", "data": "Traceback (most recent call last):\n"},
                                         {"stream_name": "stderr", "data": "torch.OutOfMemoryError: CUDA out of memory.\n"}]))
check("a run that ran out of GPU memory says so", training.why_failed(out) == "The GPU ran out of memory, so nothing was trained.")
(out / "run.log").write_text("[  1.0 min] FAILED: qwen-codes.jsonl not found\n")
check("…and other failures", "qwen-codes.jsonl not found" in (training.why_failed(out) or ""))
(out / "run.log").write_text("[ 90.0 min] Done. Merged model in /kaggle/working/x-qwen\n")
check("a finished run isn't a failure", training.why_failed(out) is None)
save_run("qwen/oomtest", state="error", error="The GPU ran out of memory, so nothing was trained.")
rr = next(x for x in c.get("/api/train/runs").json() if x["key"] == "qwen/oomtest")
check("failed run listed with its reason", rr["label"] == "Failed" and "memory" in rr["error"] and rr["can_retry"], json.dumps(rr))
check("remove a failed run", c.delete("/api/train/runs", params={"key": "qwen/oomtest"}).status_code == 200
      and not any(x["key"] == "qwen/oomtest" for x in c.get("/api/train/runs").json()))

# ---------- each voice's stability on every model that samples ----------
from narrator.tts.base import line_temperature  # noqa: E402
q = books.temp_from_stability
check("stability: the default keeps each model's tested setting", line_temperature({}, {"temperature": q(55)}, 0.6) == 0.6)
check("stability: a character's own beats the page's", line_temperature({"temperature": q(0)}, {"temperature": q(55)}, 0.6) == 0.84)
check("stability: 100% is steadier, within range", line_temperature({}, {"temperature": q(100)}, 0.7) == 0.42)

# ---------- pauses and pronunciations ----------
its = sfx.cue_items("The door slammed. {pause: 2} Nobody moved.", {"text": "", "voice": "v", "pause": 0.6})
check("a pause tag splits the line with that much silence", [i.get("cue", {}).get("seconds") for i in its] == [None, 2.0, None])
check("a pause isn't a missing sound", sfx.missing("{pause: 2} {pause}") == [])
check("pronunciations: empty to start", c.get("/api/pronunciations").json() == [])
c.put("/api/pronunciations", json={"rows": [{"word": "Kael", "say": "Kale"}, {"word": "", "say": "x"}]})
check("pronunciations: saved, blanks dropped", c.get("/api/pronunciations").json() == [{"word": "Kael", "say": "Kale"}])
from narrator import pronounce  # noqa: E402
check("pronunciations: whole words, not inside tags", pronounce.apply("Kael's {sfx: kael} Kaelan", pronounce.merged()) == "Kale's {sfx: kael} Kaelan")
check("a book keeps its own words", "pronunciations" in c.get(f"/api/projects/{p1['id']}").json())
c.put("/api/pronunciations", json={"rows": []})

# ---------- the rest: read-only pages ----------
check("my voices", c.get("/api/my-voices").json() == [])
pk = c.get("/api/packs").json()
check("packs", pk["installed"] == [] and "General" in pk["genres"], json.dumps(pk)[:200])
ms = c.get("/api/models").json()
check("models with groups", {m["id"]: m["group"] for m in ms}.get("higgs") == "high" and all("downloaded" in m for m in ms))
eng = c.get("/api/engines").json()
check("engines say if downloaded", all("downloaded" in e and "group" in e for e in eng))
cn = c.get("/api/connections").json()
check("connections", cn["ssh"]["port"] == "22" and cn["orpheus"]["where"] == "mac", json.dumps(cn))
c.post("/api/connections/ssh", json={"host": "me@box", "port": "2222"})
check("ssh saved", c.get("/api/connections").json()["ssh"]["host"] == "me@box")
check("train options", {r["id"] for r in c.get("/api/train/options").json()["recipes"]} >= {"orpheus", "qwen"})
check("train models", c.get("/api/train/models").json() == [])
check("train runs", c.get("/api/train/runs").json() == [])
check("prepared voices", c.get("/api/train/voices").json() == [])
check("send needs consent", c.post("/api/train/send", json={"voices": [{"dataset": "x"}], "model": "m"}).status_code == 400)
check("files outside the app refused", c.get("/api/files", params={"path": "/etc/hosts"}).status_code == 403)

print(f"\n{'All passed' if not failed else f'{len(failed)} failed: ' + ', '.join(failed)}  (scratch home: {HOME})")
sys.exit(1 if failed else 0)
