"""Add an exported voice pack to the babble-voices catalog.

    .venv/bin/python scripts/publish_pack.py PACK.narrator-pack ../babble-voices --hf USER/babble-voices

Writes MP3 previews to <voices>/previews/<id>-v<version>/ and the pack's entry in <voices>/catalog.json, with its
download link on Hugging Face (and its signature, when it's signed). Uploads nothing: it prints the two commands
that publish the pack and the catalog.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def add_to_catalog(pack_path, voices_dir, hf, log=print):
    """Write the pack's MP3 previews and its catalog entry into the babble-voices folder. Returns the entry."""
    pack_path, voices_dir = Path(pack_path), Path(voices_dir)
    with zipfile.ZipFile(pack_path) as z:
        pack = json.loads(z.read("pack.json"))
        names = set(z.namelist())
        signature = json.loads(z.read("signature.json")) if "signature.json" in names else None
        pid, ver = pack["id"], int(pack.get("version", 1))
        folder = voices_dir / "previews" / f"{pid}-v{ver}"
        folder.mkdir(parents=True, exist_ok=True)
        voices = []
        for v in pack["voices"]:
            row = {k: v.get(k, "") for k in ("speaker", "name", "gender", "description", "source", "license")}
            src = f"previews/{v['speaker']}.wav"
            if src in names:
                out = folder / f"{v['speaker']}.mp3"
                with tempfile.TemporaryDirectory() as td:
                    tmp = Path(td) / "preview.wav"
                    tmp.write_bytes(z.read(src))
                    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(tmp), "-c:a", "libmp3lame",
                                    "-b:a", "128k", str(out)], check=True)
                row["preview"] = out.relative_to(voices_dir).as_posix()
            else:
                log(f"No preview for {v['name']} in the pack.")
            voices.append(row)

    entry = {k: pack.get(k, "") for k in ("id", "name", "genre", "description", "engine", "author")}
    entry.update(version=ver, file=pack_path.name, size=pack_path.stat().st_size, sha256=sha256(pack_path),
                 url=f"https://huggingface.co/{hf}/resolve/main/{pack_path.name}", voices=voices)
    if signature:
        entry["signature"] = signature
    cat_path = voices_dir / "catalog.json"
    catalog = json.loads(cat_path.read_text()) if cat_path.exists() else []
    catalog = [e for e in catalog if not (e.get("id") == pid and int(e.get("version", 1)) == ver)] + [entry]
    cat_path.write_text(json.dumps(catalog, indent=1, ensure_ascii=False) + "\n")
    log(f"Added {entry['name']} v{ver} ({len(voices)} voices, {entry['size'] / 1e9:.1f} GB) to {cat_path}.")
    return entry


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack", type=Path)
    ap.add_argument("voices", type=Path, help="the babble-voices folder")
    ap.add_argument("--hf", required=True, help="the Hugging Face repo that holds the pack files, e.g. USER/babble-voices")
    a = ap.parse_args()
    entry = add_to_catalog(a.pack, a.voices, a.hf, log=lambda m: print(m, file=sys.stderr))
    print("To publish:")
    print(f"  hf upload {a.hf} '{a.pack}' '{a.pack.name}' --repo-type model")
    print(f"  cd '{a.voices}' && git add -A && git commit -m 'Add {entry['name']} v{entry['version']}' && git push")


if __name__ == "__main__":
    main()
