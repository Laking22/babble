"""Run: .venv/bin/python tests/pack_admin_check.py  (a few seconds; needs git and ffmpeg; no network)

Pack Admin (scripts/pack_admin.py) against a stand-in babble-voices repo (a bare repo in a scratch folder) and a
stand-in Hugging Face: uploading puts the pack file up first, then its previews and catalog entry; removing takes the
entry and previews off GitHub first, then the file off Hugging Face. A failed push leaves the clone as GitHub has it
and nothing deleted; files outside the previews folder, shared files and files elsewhere are left alone.
"""
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import wave
import zipfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="pack-admin-check-"))
os.environ["HOME"] = str(TMP / "home")  # (git and Hugging Face read nothing of this computer's own)
(TMP / "home").mkdir()
ROOT = Path(__file__).resolve().parent.parent
failed = []


def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what, flush=True)
    if not ok:
        failed.append(what)


def git(*a, cwd=None):
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


ORIGIN, SEED, VOICES = TMP / "origin.git", TMP / "seed", TMP / "voices"
git("init", "--bare", "-q", "-b", "main", str(ORIGIN))
git("clone", "-q", str(ORIGIN), str(SEED))
for d in (SEED,):
    git("config", "user.name", "Test", cwd=d)
    git("config", "user.email", "test@example.com", cwd=d)
    git("config", "commit.gpgsign", "false", cwd=d)
(SEED / "catalog.json").write_text("[]\n")
git("add", "-A", cwd=SEED)
git("commit", "-qm", "catalog", cwd=SEED)
git("push", "-q", "origin", "HEAD:main", cwd=SEED)
git("clone", "-q", str(ORIGIN), str(VOICES))
for k, v in (("user.name", "Test"), ("user.email", "test@example.com"), ("commit.gpgsign", "false")):
    git("config", k, v, cwd=VOICES)
os.environ["BABBLE_VOICES_DIR"] = str(VOICES)

import huggingface_hub  # noqa: E402
from huggingface_hub.utils import EntryNotFoundError  # noqa: E402

HF = {"files": set(), "deleted": [], "fail": None}


class FakeHf:
    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type=None, commit_message=None):
        assert Path(path_or_fileobj).is_file() and repo_id == "Laking22/babble-voices"
        HF["files"].add(path_in_repo)

    def delete_file(self, path_in_repo, repo_id, repo_type=None, commit_message=None):
        if HF["fail"]:
            raise HF["fail"]
        if path_in_repo not in HF["files"]:
            raise EntryNotFoundError("not there")
        HF["files"].discard(path_in_repo)
        HF["deleted"].append(path_in_repo)


huggingface_hub.HfApi = FakeHf
spec = importlib.util.spec_from_file_location("pack_admin", ROOT / "scripts" / "pack_admin.py")
pa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pa)
logs = []
log = logs.append


def wav():
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b"\0\0" * 12000)
    return b.getvalue()


def make_pack(pid, ver, speakers=("ann", "bob")):
    p = TMP / f"{pid}-v{ver}.narrator-pack"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("pack.json", json.dumps({
            "format": 1, "id": pid, "name": pid.title(), "genre": "Test", "description": "A test pack.", "version": ver,
            "engine": "qwen", "author": "Babble",
            "voices": [{"speaker": s, "name": s.title(), "gender": "female", "description": "", "source": "LibriVox: x",
                        "license": "Public domain"} for s in speakers]}))
        for s in speakers:
            z.writestr(f"previews/{s}.wav", wav())
        z.writestr("model/model.safetensors", b"0" * 100)
    return p


def on_github(path="catalog.json"):
    return git("--git-dir", str(ORIGIN), "show", f"main:{path}")


def tree():
    return set(git("--git-dir", str(ORIGIN), "ls-tree", "-r", "--name-only", "main").splitlines())


def ids():
    return [(e["id"], e["version"]) for e in json.loads(on_github())]


def push_entries(*entries):
    """Someone else adds catalog entries on GitHub (the admin pulls them in)."""
    git("pull", "-q", cwd=SEED)
    cat = json.loads((SEED / "catalog.json").read_text()) + list(entries)
    (SEED / "catalog.json").write_text(json.dumps(cat, indent=1) + "\n")
    git("commit", "-qam", "entries", cwd=SEED)
    git("push", "-q", cwd=SEED)


# ---------- upload
v1, v2 = make_pack("tiny", 1), make_pack("tiny", 2)
pack, sig, signed, problems, notes = pa.read_pack(v1)
check(not problems and any("isn't signed" in n for n in notes), f"a good pack passes the checks ({problems})")
pa.upload(v1, pack, log)
check(ids() == [("tiny", 1)] and {"previews/tiny-v1/ann.mp3", "previews/tiny-v1/bob.mp3"} <= tree()
      and "tiny-v1.narrator-pack" in HF["files"], "upload: the file on Hugging Face, previews + entry on GitHub")
check(git("--git-dir", str(ORIGIN), "log", "-1", "--format=%s", "main") == "Add Tiny v1", "…in one commit, \"Add Tiny v1\"")
pa.upload(v2, pa.read_pack(v2)[0], log)
check(ids() == [("tiny", 1), ("tiny", 2)], "a second version goes next to the first")

# ---------- remove
pa.remove("tiny", 1, log)
check(ids() == [("tiny", 2)] and not any(p.startswith("previews/tiny-v1") for p in tree())
      and "previews/tiny-v2/ann.mp3" in tree(), "remove: its entry and previews leave GitHub, the other version stays")
check(HF["deleted"] == ["tiny-v1.narrator-pack"] and "tiny-v2.narrator-pack" in HF["files"],
      "…and its file leaves Hugging Face (only that one)")
check(git("--git-dir", str(ORIGIN), "log", "-1", "--format=%s", "main") == "Remove Tiny v1", "…in one commit, \"Remove Tiny v1\"")

# a push that fails: nothing is deleted, and the clone is back to what GitHub has
head = git("rev-parse", "HEAD", cwd=VOICES)
git("remote", "set-url", "--push", "origin", str(TMP / "nowhere.git"), cwd=VOICES)
try:
    pa.remove("tiny", 2, log)
    check(False, "a failed push stops the removal")
except RuntimeError:
    check(git("rev-parse", "HEAD", cwd=VOICES) == head and not git("status", "--porcelain", cwd=VOICES)
          and (VOICES / "previews/tiny-v2/ann.mp3").is_file() and "tiny-v2.narrator-pack" in HF["files"]
          and ids() == [("tiny", 2)], "a failed push: the clone is as GitHub has it and nothing is deleted")
git("remote", "set-url", "--push", "origin", str(ORIGIN), cwd=VOICES)

# changes in the clone that aren't on GitHub: it won't start
(VOICES / "stray.txt").write_text("x")
try:
    pa.remove("tiny", 2, log)
    check(False, "a clone with its own changes is refused")
except RuntimeError as e:
    check("changes that aren't pushed" in str(e) and ids() == [("tiny", 2)], "a clone with its own changes is refused")
(VOICES / "stray.txt").unlink()

# entries added on GitHub by someone else are pulled in first; files that aren't ours or are shared stay
HF["files"].add("shared.narrator-pack")
push_entries(
    {"id": "elsewhere", "version": 1, "name": "Elsewhere", "file": "e.narrator-pack", "url": "https://example.com/e.narrator-pack", "voices": []},
    {"id": "shared", "version": 1, "name": "Shared", "file": "shared.narrator-pack",
     "url": "https://huggingface.co/Laking22/babble-voices/resolve/main/shared.narrator-pack", "voices": []},
    {"id": "shared", "version": 2, "name": "Shared", "file": "shared.narrator-pack",
     "url": "https://huggingface.co/Laking22/babble-voices/resolve/main/shared.narrator-pack", "voices": []},
    {"id": "../../outside", "version": 1, "name": "Outside", "file": "../x.narrator-pack",
     "url": "https://huggingface.co/Laking22/babble-voices/resolve/main/../x.narrator-pack",
     "voices": [{"speaker": "x", "preview": "previews/../../outside.txt"}, {"speaker": "y", "preview": "/etc/hosts"}]},
    {"id": "gone", "version": 1, "name": "Gone", "file": "gone.narrator-pack",
     "url": "https://huggingface.co/Laking22/babble-voices/resolve/main/gone.narrator-pack", "voices": []},
)
(TMP / "outside.txt").write_text("keep me")
(TMP / "outside-v1").mkdir()
deleted = len(HF["deleted"])
pa.remove("elsewhere", 1, log)
pa.remove("shared", 1, log)
check(("elsewhere", 1) not in ids() and ("shared", 1) not in ids() and ("shared", 2) in ids()
      and len(HF["deleted"]) == deleted and "shared.narrator-pack" in HF["files"],
      "a file that isn't on our Hugging Face, or that another entry uses, is left where it is")
pa.remove("../../outside", 1, log)
check(("../../outside", 1) not in ids() and (TMP / "outside.txt").read_text() == "keep me" and (TMP / "outside-v1").is_dir()
      and len(HF["deleted"]) == deleted, "an entry with odd paths: nothing outside the previews folder is touched")
pa.remove("gone", 1, log)
check(("gone", 1) not in ids() and logs[-2] == "It wasn't on Hugging Face.", "a file that's already gone from Hugging Face: fine")
HF["fail"] = RuntimeError("401 Unauthorized")
try:
    pa.remove("shared", 2, log)
    check(False, "a Hugging Face error is reported")
except RuntimeError as e:
    check("out of the catalog" in str(e) and "shared.narrator-pack" in str(e) and ("shared", 2) not in ids(),
          "Hugging Face refuses: it's out of the catalog and the message says the file is still there")
HF["fail"] = None
try:
    pa.remove("tiny", 9, log)
    check(False, "a version that isn't there")
except RuntimeError as e:
    check("isn't in the catalog" in str(e), "a version that isn't published: says so")

# ---------- the window (where there's a screen)
try:
    import tkinter as tk
    root = tk.Tk()
except Exception as e:  # noqa: BLE001 - no display (GitHub's Linux machines)
    print(f"skip the window ({e.__class__.__name__})")
else:
    pa.App.refresh = lambda self: None  # (no repo work from the window here)
    app = pa.App(root)
    app.show_catalog(json.loads(on_github()))
    root.update()
    check(str(app.remove_btn["state"]) == "disabled", "Remove waits for a pack to be picked")
    app.packs.selection_set("0")
    app.picked()
    root.update()
    check(str(app.remove_btn["state"]) == "normal" and "Ann" in app.detail.cget("text"), "picking one shows its voices and Remove")
    app.load(make_pack("tiny", 2))
    check("already published" in app.checks.get("1.0", "end") and str(app.go["state"]) == "disabled",
          "Upload: a version that's published already can't go up again")
    app.load(make_pack("tiny", 3))
    check("updates the published pack" in app.checks.get("1.0", "end") and str(app.go["state"]) == "normal",
          "Upload: a newer version can")
    root.destroy()

print("\nall passed" if not failed else f"\n{len(failed)} failed: " + "; ".join(failed))
sys.exit(1 if failed else 0)
