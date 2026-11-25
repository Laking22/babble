"""Babble Pack Uploader: put a voice pack exported from Babble where every Babble can download it.

    .venv/bin/python scripts/pack_uploader.py          (or double-click scripts/Pack Uploader.command)

Pick a .narrator-pack (Train a voice → Export pack), look over its voices, press Upload. The pack file goes to the
Hugging Face repo, then its previews and catalog entry go to the babble-voices repo on GitHub. Babble's Library →
Available reads that catalog (each Babble looks again every 10 minutes).

It signs in with what this computer already has: the Hugging Face login (`hf auth login`) and the GitHub CLI
(`gh auth login`). It never asks for or stores a key.
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
os.environ["NARRATOR_HOME"] = tempfile.mkdtemp(prefix="pack-uploader-")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import tkinter as tk  # noqa: E402
from tkinter import filedialog, messagebox, ttk  # noqa: E402

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
    """The catalog as it is on GitHub now."""
    import urllib.request
    with urllib.request.urlopen(CATALOG_RAW, timeout=20) as r:
        return json.loads(r.read())


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def upload(path, pack, log):
    """Hugging Face first (so the catalog never points at a missing file), then previews + catalog on GitHub."""
    from huggingface_hub import HfApi
    path = Path(path)
    name, ver = pack.get("name") or pack["id"], int(pack.get("version", 1))

    log("Getting the babble-voices repo up to date…")
    if not (VOICES_DIR / ".git").exists():
        run(["gh", "repo", "clone", GH_REPO, str(VOICES_DIR)])
    if run(["git", "status", "--porcelain"], cwd=VOICES_DIR):
        raise RuntimeError(f"{VOICES_DIR} has changes that aren't pushed. Commit or undo them first.")
    run(["git", "pull", "--ff-only"], cwd=VOICES_DIR)

    size = path.stat().st_size
    log(f"Uploading {path.name} ({size / 1e9:.2f} GB) to Hugging Face. Big packs take a while…")
    HfApi().upload_file(path_or_fileobj=str(path), path_in_repo=path.name, repo_id=HF_REPO, repo_type="model",
                        commit_message=f"Add {name} v{ver}")
    log("Uploaded to Hugging Face.")

    log("Writing its previews and catalog entry…")
    add_to_catalog(path, VOICES_DIR, HF_REPO, log=log)
    run(["git", "add", "-A"], cwd=VOICES_DIR)
    run(["git", "commit", "-m", f"Add {name} v{ver}"], cwd=VOICES_DIR)
    log("Publishing the catalog on GitHub…")
    run(["git", "push"], cwd=VOICES_DIR)
    log(f"Done. {name} is in every Babble's Library → Available within about 10 minutes.")


class App:
    def __init__(self, root):
        self.root, self.path, self.pack, self.q, self.err = root, None, None, queue.Queue(), ""
        root.title("Babble Pack Uploader")
        root.geometry("760x560")
        root.minsize(620, 460)
        f = ttk.Frame(root, padding=16)
        f.pack(fill="both", expand=True)

        top = ttk.Frame(f)
        top.pack(fill="x")
        ttk.Button(top, text="Choose a pack…", command=self.choose).pack(side="left")
        self.file = ttk.Label(top, text="No pack chosen", foreground="#777")
        self.file.pack(side="left", padx=10)

        self.title = ttk.Label(f, text="", font=("Helvetica", 16, "bold"))
        self.title.pack(anchor="w", pady=(14, 0))
        self.meta = ttk.Label(f, text="", foreground="#777")
        self.meta.pack(anchor="w")

        cols = ("name", "gender", "licence", "source")
        self.tree = ttk.Treeview(f, columns=cols, show="headings", height=7)
        for c, w in zip(cols, (150, 70, 120, 360)):
            self.tree.heading(c, text=c.capitalize())
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(fill="x", pady=10)

        self.checks = tk.Text(f, height=5, wrap="word", relief="flat", background=root.cget("background"))
        self.checks.pack(fill="x")
        self.checks.tag_configure("bad", foreground="#c0392b")
        self.checks.tag_configure("note", foreground="#b9770e")
        self.checks.tag_configure("ok", foreground="#1e8449")

        row = ttk.Frame(f)
        row.pack(fill="x", pady=(8, 6))
        self.bar = ttk.Progressbar(row, mode="indeterminate")
        self.bar.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.go = ttk.Button(row, text="Upload", command=self.start, state="disabled")
        self.go.pack(side="right")

        self.log = tk.Text(f, height=8, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True)
        root.after(150, self.drain)

    def say(self, text):
        self.q.put(("log", text))

    def drain(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "log":
                    self.log.configure(state="normal")
                    self.log.insert("end", val + "\n")
                    self.log.see("end")
                    self.log.configure(state="disabled")
                elif kind == "end":
                    self.bar.stop()
                    self.go.configure(state="normal" if val is None else "disabled")
                    if val is None:
                        messagebox.showerror("Not uploaded", self.err)
                    else:
                        messagebox.showinfo("Uploaded", val)
        except queue.Empty:
            pass
        self.root.after(150, self.drain)

    def choose(self):
        p = filedialog.askopenfilename(title="Choose a voice pack", filetypes=[("Babble voice pack", "*.narrator-pack"), ("All files", "*")])
        if p:
            self.load(Path(p))

    def load(self, path):
        self.go.configure(state="disabled")
        self.tree.delete(*self.tree.get_children())
        self.checks.delete("1.0", "end")
        try:
            pack, sig, signed, problems, notes = read_pack(path)
        except Exception as e:  # noqa: BLE001 - not a zip, no pack.json…
            self.file.configure(text=path.name)
            self.title.configure(text="")
            self.meta.configure(text="")
            self.checks.insert("end", f"✕ {e}\n", "bad")
            return
        try:
            live = published()
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
        self.file.configure(text=f"{path.name} · {path.stat().st_size / 1e9:.2f} GB")
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
            self.go.configure(state="normal")

    def start(self):
        name = self.pack.get("name") or self.pack.get("id")
        if not messagebox.askyesno("Upload for everyone?", f"Upload {name} so every Babble user can download it?\n\n"
                                   "It goes on Hugging Face and in the public babble-voices catalog."):
            return
        self.go.configure(state="disabled")
        self.bar.start(12)

        def work():
            try:
                upload(self.path, self.pack, self.say)
                self.q.put(("end", f"{name} is published."))
            except Exception as e:  # noqa: BLE001
                self.err = str(e)
                self.say(f"Stopped: {e}")
                self.q.put(("end", None))

        threading.Thread(target=work, daemon=True).start()


def main():
    for tool in ("gh", "git", "ffmpeg"):
        if not shutil.which(tool):
            print(f"{tool} isn't installed; the uploader needs it.", file=sys.stderr)
    root = tk.Tk()
    app = App(root)
    if len(sys.argv) > 1:  # a pack given on the command line is opened straight away
        root.after(200, lambda: app.load(Path(sys.argv[1])))
    root.mainloop()


if __name__ == "__main__":
    main()
