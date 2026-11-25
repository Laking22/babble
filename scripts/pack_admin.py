"""Babble Pack Admin: the voice packs every Babble can download. Upload new ones, remove published ones.

    .venv/bin/python scripts/pack_admin.py          (or double-click scripts/Pack Admin.command)

Published: the packs in the public catalog. Remove takes one down: its catalog entry and previews leave the
babble-voices repo on GitHub first, then its file leaves Hugging Face. People who installed it keep their copy.

Upload: pick a .narrator-pack (Train a voice → Export pack), look over its voices, press Upload. The pack file goes
to Hugging Face first, then its previews and catalog entry go to the babble-voices repo.

Babble's Library → Available reads that catalog (each Babble looks again every 10 minutes). It signs in with what
this computer already has: the Hugging Face login (`hf auth login`) and the GitHub CLI (`gh auth login`). It never
asks for or stores a key.
"""
import hashlib
import json
import os
import queue
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import zipfile
from pathlib import Path

# the pack checks below use Babble's own signature code; it must not touch the real Babble data folder
os.environ["NARRATOR_HOME"] = tempfile.mkdtemp(prefix="pack-admin-")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:  # (the checks use the functions without a window)
    tk = None

from publish_pack import add_to_catalog  # noqa: E402

HF_REPO = "Laking22/babble-voices"
GH_REPO = "Laking22/babble-voices"
VOICES_DIR = Path(os.environ.get("BABBLE_VOICES_DIR", Path.home() / "babble-voices"))
CATALOG_RAW = f"https://raw.githubusercontent.com/{GH_REPO}/main/catalog.json"

# the same limits Babble checks when it installs a pack (narrator/library.py)
MAX_FILES = 2000
MAX_UNCOMPRESSED = 25 * (1 << 30)
UNSAFE_EXTS = {".pt", ".pth", ".bin", ".pkl", ".pickle", ".py", ".pyc", ".sh", ".command", ".so", ".dylib", ".exe"}
FORMAT = 1


def run(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{' '.join(cmd[:3])}: {(r.stderr or r.stdout).strip()[-400:]}")
    return r.stdout.strip()


def read_pack(path):
    """(pack.json, signature or None, problems, notes, previews present) after Babble's own install checks."""
    problems, notes = [], []
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        names = {i.filename for i in infos}
        if "pack.json" not in names:
            raise ValueError("This isn't a voice pack: it has no pack.json. Export it from Babble (Train a voice → Export pack).")
        if len(infos) > MAX_FILES:
            problems.append(f"It has too many files ({len(infos)}; Babble takes up to {MAX_FILES}).")
        total = 0
        for i in infos:
            n = i.filename
            if stat.S_ISLNK(i.external_attr >> 16) or n.startswith("/") or "\\" in n or ".." in Path(n).parts:
                problems.append(f"Unsafe path in the pack: {n}")
            if Path(n).suffix.lower() in UNSAFE_EXTS:
                problems.append(f"Babble won't install a pack with {Path(n).name} in it (only safetensors weights).")
            total += max(0, int(i.file_size))
        if total > MAX_UNCOMPRESSED:
            problems.append("It's bigger than Babble installs (25 GB unpacked).")
        pack = json.loads(z.read("pack.json"))
        if int(pack.get("format", 0)) > FORMAT:
            problems.append("It was made by a newer Babble than people have.")
        sig = json.loads(z.read("signature.json")) if "signature.json" in names else None
        previews = {v.get("speaker"): f"previews/{v.get('speaker')}.wav" in names for v in pack.get("voices", [])}
        signed = None
        if sig:
            try:
                from narrator import signing
                signed = signing.check(sig, signing.manifest(z))
            except Exception as e:  # noqa: BLE001 - offline: say so, don't block
                signed = {"verified": None, "github": sig.get("github", ""), "why": f"couldn't check it ({e})"}
    if not pack.get("voices"):
        problems.append("It has no voices.")
    for v in pack.get("voices", []):
        who = v.get("name") or v.get("speaker") or "A voice"
        if not str(v.get("source", "")).strip():
            problems.append(f"{who} has no source (where the recording came from).")
        if not str(v.get("license", "")).strip():
            problems.append(f"{who} has no licence.")
        if not previews.get(v.get("speaker")):
            notes.append(f"{who} has no preview, so people can't hear it before downloading.")
    if not sig:
        notes.append("It isn't signed. People will see it as unsigned (sign it when exporting, in Babble).")
    elif signed and signed.get("verified") is False:
        problems.append(f"Its signature doesn't check out: {signed.get('why')}")
    elif signed and signed.get("verified") is None:
        notes.append(f"Its signature wasn't checked: {signed.get('why')}")
    return pack, sig, signed, problems, notes


def published():
    """The catalog as GitHub serves it to every Babble (it can lag a few minutes behind the repo)."""
    import urllib.request
    with urllib.request.urlopen(CATALOG_RAW, timeout=20) as r:
        return json.loads(r.read())


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sync(log):
    """The babble-voices clone, up to date with GitHub and with nothing of its own."""
    log("Getting the babble-voices repo up to date…")
    if not (VOICES_DIR / ".git").exists():
        run(["gh", "repo", "clone", GH_REPO, str(VOICES_DIR)])
    if run(["git", "status", "--porcelain"], cwd=VOICES_DIR):
        raise RuntimeError(f"{VOICES_DIR} has changes that aren't pushed. Commit or undo them first.")
    run(["git", "pull", "--ff-only"], cwd=VOICES_DIR)
    if run(["git", "rev-list", "--count", "@{u}..HEAD"], cwd=VOICES_DIR) != "0":
        raise RuntimeError(f"{VOICES_DIR} has commits that aren't on GitHub. Push or undo them first.")


def catalog_here():
    """The catalog in the clone (after sync)."""
    p = VOICES_DIR / "catalog.json"
    return json.loads(p.read_text()) if p.exists() else []


def publish(message, log):
    """Commit everything in the clone and push it. If the push fails, the clone goes back to what GitHub has."""
    before = run(["git", "rev-parse", "HEAD"], cwd=VOICES_DIR)
    try:
        run(["git", "add", "-A"], cwd=VOICES_DIR)
        run(["git", "commit", "-m", message], cwd=VOICES_DIR)
        log("Publishing the catalog on GitHub…")
        run(["git", "push"], cwd=VOICES_DIR)
    except Exception:
        run(["git", "reset", "--hard", before], cwd=VOICES_DIR)
        raise


def upload(path, pack, log):
    """Hugging Face first (so the catalog never points at a missing file), then previews + catalog on GitHub."""
    from huggingface_hub import HfApi
    path = Path(path)
    name, ver = pack.get("name") or pack["id"], int(pack.get("version", 1))
    sync(log)

    size = path.stat().st_size
    log(f"Uploading {path.name} ({size / 1e9:.2f} GB) to Hugging Face. Big packs take a while…")
    HfApi().upload_file(path_or_fileobj=str(path), path_in_repo=path.name, repo_id=HF_REPO, repo_type="model",
                        commit_message=f"Add {name} v{ver}")
    log("Uploaded to Hugging Face.")

    log("Writing its previews and catalog entry…")
    add_to_catalog(path, VOICES_DIR, HF_REPO, log=log)
    publish(f"Add {name} v{ver}", log)
    log(f"Done. {name} is in every Babble's Library → Available within about 10 minutes.")


def _inside(base, rel):
    """base/rel, only when it stays inside base."""
    p = (base / rel).resolve()
    return p if p != base and base in p.parents else None


def remove(pid, version, log):
    """GitHub first (so no Babble offers a file that's gone), then the pack file off Hugging Face."""
    version = int(version)
    sync(log)
    catalog = catalog_here()
    entry = next((e for e in catalog if e.get("id") == pid and int(e.get("version", 1)) == version), None)
    if not entry:
        raise RuntimeError("It isn't in the catalog any more.")
    name = entry.get("name") or pid
    rest = [e for e in catalog if e is not entry]
    log(f"Taking {name} v{version} out of the catalog…")
    (VOICES_DIR / "catalog.json").write_text(json.dumps(rest, indent=1, ensure_ascii=False) + "\n")

    # its previews: the files its entry names and its own folder, unless another entry still uses them
    base, previews = VOICES_DIR.resolve(), (VOICES_DIR / "previews").resolve()
    kept = {_inside(base, v["preview"]) for e in rest for v in e.get("voices", []) if v.get("preview")}
    folders = {_inside(previews, f"{pid}-v{version}")}
    for v in entry.get("voices", []):
        p = _inside(previews, Path(v.get("preview") or "").relative_to("previews")) \
            if str(v.get("preview") or "").startswith("previews/") else None
        if p and p not in kept and p.is_file():
            p.unlink()
            folders.add(p.parent)
    for d in folders:
        if d and d.is_dir() and d.parent == previews and not any(k and d in k.parents for k in kept):
            shutil.rmtree(d)
    publish(f"Remove {name} v{version}", log)

    file = str(entry.get("file") or "")
    if not file or Path(file).name != file:
        log("It has no pack file listed, so nothing was deleted from Hugging Face.")
    elif any(e.get("file") == file for e in rest):
        log(f"{file} stays on Hugging Face: another catalog entry uses it.")
    elif not str(entry.get("url", "")).startswith(f"https://huggingface.co/{HF_REPO}/"):
        log(f"Its file isn't in {HF_REPO}, so it was left where it is.")
    else:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import EntryNotFoundError
        log(f"Deleting {file} from Hugging Face…")
        try:
            HfApi().delete_file(path_in_repo=file, repo_id=HF_REPO, repo_type="model",
                                commit_message=f"Remove {name} v{version}")
        except EntryNotFoundError:
            log("It wasn't on Hugging Face.")
        except Exception as e:
            raise RuntimeError(f"{name} v{version} is out of the catalog, but its file is still on Hugging Face "
                               f"({e}). Delete {file} there by hand.") from e
    log(f"Done. {name} v{version} leaves every Babble's Library → Available within about 10 minutes.")


def size_text(n):
    n = int(n or 0)
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


class App:
    def __init__(self, root):
        self.root, self.q, self.busy = root, queue.Queue(), False
        self.path, self.pack, self.catalog, self.ready = None, None, None, False
        root.title("Babble Pack Admin")
        root.geometry("820x660")
        root.minsize(660, 520)
        f = ttk.Frame(root, padding=16)
        f.pack(fill="both", expand=True)
        self.tabs = ttk.Notebook(f)
        self.tabs.pack(fill="both", expand=True)
        pub, up = ttk.Frame(self.tabs, padding=12), ttk.Frame(self.tabs, padding=12)
        self.tabs.add(pub, text="Published")
        self.tabs.add(up, text="Upload")

        # ---- Published: what every Babble can download now
        cols = ("pack", "version", "voices", "size", "signed")
        self.packs = ttk.Treeview(pub, columns=cols, show="headings", height=8, selectmode="browse")
        for c, w in zip(cols, (300, 70, 70, 90, 170)):
            self.packs.heading(c, text=c.capitalize())
            self.packs.column(c, width=w, anchor="w", stretch=c == "pack")
        self.packs.pack(fill="x")
        self.packs.bind("<<TreeviewSelect>>", lambda e: self.picked())
        self.detail = ttk.Label(pub, text="", foreground="#777", wraplength=740, justify="left")
        self.detail.pack(anchor="w", fill="x", pady=(10, 0))
        prow = ttk.Frame(pub)
        prow.pack(fill="x", side="bottom")
        self.refresh_btn = ttk.Button(prow, text="Refresh", command=self.refresh)
        self.refresh_btn.pack(side="left")
        self.remove_btn = ttk.Button(prow, text="Remove", command=self.start_remove, state="disabled")
        self.remove_btn.pack(side="right")

        # ---- Upload: a pack exported from Babble
        top = ttk.Frame(up)
        top.pack(fill="x")
        ttk.Button(top, text="Choose a pack…", command=self.choose).pack(side="left")
        self.file = ttk.Label(top, text="No pack chosen", foreground="#777")
        self.file.pack(side="left", padx=10)
        self.title = ttk.Label(up, text="", font=("Helvetica", 16, "bold"))
        self.title.pack(anchor="w", pady=(14, 0))
        self.meta = ttk.Label(up, text="", foreground="#777")
        self.meta.pack(anchor="w")
        cols = ("name", "gender", "licence", "source")
        self.tree = ttk.Treeview(up, columns=cols, show="headings", height=6)
        for c, w in zip(cols, (150, 70, 120, 360)):
            self.tree.heading(c, text=c.capitalize())
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(fill="x", pady=10)
        self.checks = tk.Text(up, height=4, wrap="word", relief="flat", background=root.cget("background"))
        self.checks.pack(fill="x")
        self.checks.tag_configure("bad", foreground="#c0392b")
        self.checks.tag_configure("note", foreground="#b9770e")
        self.checks.tag_configure("ok", foreground="#1e8449")
        urow = ttk.Frame(up)
        urow.pack(fill="x", side="bottom")
        self.go = ttk.Button(urow, text="Upload", command=self.start_upload, state="disabled")
        self.go.pack(side="right")

        # ---- both: progress and what happened
        self.bar = ttk.Progressbar(f, mode="indeterminate")
        self.bar.pack(fill="x", pady=(10, 6))
        self.log = tk.Text(f, height=7, wrap="word", state="disabled")
        self.log.pack(fill="x")
        root.after(150, self.drain)
        root.after(200, self.refresh)

    # ---------- background work
    def say(self, text):
        self.q.put(("log", text))

    def work(self, fn, done_title, fail_title):
        """Run fn on a thread, one job at a time; the buttons wait for it."""
        self.busy = True
        self.buttons()
        self.bar.start(12)

        def go():
            try:
                msg = fn()
                self.q.put(("end", (True, done_title, msg)))
            except Exception as e:  # noqa: BLE001
                self.say(f"Stopped: {e}")
                self.q.put(("end", (False, fail_title, str(e))))

        threading.Thread(target=go, daemon=True).start()

    def drain(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "log":
                    self.log.configure(state="normal")
                    self.log.insert("end", val + "\n")
                    self.log.see("end")
                    self.log.configure(state="disabled")
                elif kind == "catalog":
                    self.show_catalog(val)
                elif kind == "end":
                    ok, title, msg = val
                    self.busy = False
                    self.bar.stop()
                    self.buttons()
                    if title:
                        (messagebox.showinfo if ok else messagebox.showerror)(title, msg)
                    if ok and title:  # an upload or a removal changed the catalog
                        self.refresh()
        except queue.Empty:
            pass
        self.root.after(150, self.drain)

    def buttons(self):
        idle = "disabled" if self.busy else "normal"
        self.refresh_btn.configure(state=idle)
        self.remove_btn.configure(state="normal" if not self.busy and self.packs.selection() else "disabled")
        self.go.configure(state="normal" if not self.busy and self.pack is not None and self.ready else "disabled")

    # ---------- Published
    def refresh(self):
        def fn():
            try:
                sync(self.say)
                cat = catalog_here()
            except Exception as e:  # noqa: BLE001 - still show what's published
                self.say(f"Couldn't update the babble-voices repo ({e}). Showing the catalog from GitHub.")
                cat = published()
            self.q.put(("catalog", cat))
            return None

        self.work(fn, None, "Couldn't read the catalog")

    def show_catalog(self, cat):
        self.catalog = cat
        self.packs.delete(*self.packs.get_children())
        for i, e in enumerate(cat):
            sig = (e.get("signature") or {}).get("github")
            self.packs.insert("", "end", iid=str(i), values=(
                e.get("name") or e.get("id"), e.get("version", 1), len(e.get("voices", [])), size_text(e.get("size")),
                f"signed by {sig}" if sig else "unsigned"))
        self.detail.configure(text="" if cat else "Nothing is published.")
        self.buttons()

    def selected(self):
        sel = self.packs.selection()
        return self.catalog[int(sel[0])] if sel and self.catalog else None

    def picked(self):
        e = self.selected()
        if e:
            voices = ", ".join(v.get("name") or v.get("speaker", "") for v in e.get("voices", []))
            self.detail.configure(text=f"{e.get('description', '')}\n{voices}\n{e.get('file', '')}")
        self.buttons()

    def start_remove(self):
        e = self.selected()
        if not e:
            return
        name, ver = e.get("name") or e.get("id"), int(e.get("version", 1))
        if not messagebox.askyesno("Remove for everyone?", f"Remove {name} version {ver}?\n\n"
                                   "It leaves every Babble's Library → Available, and its file is deleted from "
                                   "Hugging Face. People who installed it keep their copy.", icon="warning"):
            return
        self.work(lambda: (remove(e["id"], ver, self.say), f"{name} version {ver} is removed.")[1],
                  "Removed", "Not removed")

    # ---------- Upload
    def choose(self):
        p = filedialog.askopenfilename(title="Choose a voice pack", filetypes=[("Babble voice pack", "*.narrator-pack"), ("All files", "*")])
        if p:
            self.tabs.select(1)
            self.load(Path(p))

    def load(self, path):
        self.ready = False
        self.tree.delete(*self.tree.get_children())
        self.checks.delete("1.0", "end")
        try:
            pack, sig, signed, problems, notes = read_pack(path)
        except Exception as e:  # noqa: BLE001 - not a zip, no pack.json…
            self.pack = None
            self.file.configure(text=path.name)
            self.title.configure(text="")
            self.meta.configure(text="")
            self.checks.insert("end", f"✕ {e}\n", "bad")
            self.buttons()
            return
        try:
            live = self.catalog if self.catalog is not None else published()
            same = [e for e in live if e.get("id") == pack.get("id") and int(e.get("version", 1)) == int(pack.get("version", 1))]
            if same:
                problems.append(f"Version {pack.get('version', 1)} of this pack is already published. Export it again "
                                "with a higher version number to update it.")
            older = [int(e.get("version", 1)) for e in live if e.get("id") == pack.get("id")]
            if older and not same:
                notes.append(f"It updates the published pack (version {max(older)} now).")
        except Exception as e:  # noqa: BLE001
            notes.append(f"Couldn't read the published catalog ({e}).")
        self.path, self.pack = path, pack
        self.file.configure(text=f"{path.name} · {size_text(path.stat().st_size)}")
        self.title.configure(text=f"{pack.get('name') or pack.get('id')}  ·  version {pack.get('version', 1)}")
        who = f"signed by {sig.get('github')}" if sig else "unsigned"
        self.meta.configure(text=f"{pack.get('genre', '')} · {pack.get('engine', '')} · by {pack.get('author') or 'unknown'} · {who}\n"
                                 f"{pack.get('description', '')}")
        for v in pack.get("voices", []):
            self.tree.insert("", "end", values=(v.get("name", ""), v.get("gender", ""), v.get("license", ""), v.get("source", "")))
        for p in problems:
            self.checks.insert("end", f"✕ {p}\n", "bad")
        for n in notes:
            self.checks.insert("end", f"! {n}\n", "note")
        if not problems:
            self.checks.insert("end", "✓ Ready to upload. Only upload voices you have the right to share.\n", "ok")
            self.ready = True
        self.buttons()

    def start_upload(self):
        name = self.pack.get("name") or self.pack.get("id")
        if not messagebox.askyesno("Upload for everyone?", f"Upload {name} so every Babble user can download it?\n\n"
                                   "It goes on Hugging Face and in the public babble-voices catalog."):
            return
        path, pack = self.path, self.pack

        def fn():
            upload(path, pack, self.say)
            self.ready = False  # (that version is published now; a failed upload can be tried again)
            return f"{name} is published."

        self.work(fn, "Uploaded", "Not uploaded")


def main():
    for tool in ("gh", "git", "ffmpeg"):
        if not shutil.which(tool):
            print(f"{tool} isn't installed; Pack Admin needs it.", file=sys.stderr)
    root = tk.Tk()
    app = App(root)
    if len(sys.argv) > 1:  # a pack given on the command line opens on the Upload tab
        root.after(300, lambda: (app.tabs.select(1), app.load(Path(sys.argv[1]))))
    root.mainloop()


if __name__ == "__main__":
    main()
