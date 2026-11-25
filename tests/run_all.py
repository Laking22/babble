"""Run Babble's checks that need no speech models, each in its own scratch data folder.

    python tests/run_all.py            # the checks below; exit 1 if any of them fails
    python tests/run_all.py --all      # also the ones known to fail (shown, not counted)

Used by the GitHub build (.github/workflows/build.yml) on Windows and Linux, and fine on a Mac.
engine_check.py and api_check.py are left out: they load speech models.
"""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKS = [
    "agents_mcp_check.py", "api_studio_check.py", "audiobook_quality_check.py", "book_chapters_check.py",
    "cloud_dataset_check.py", "connections_security_check.py", "disk_full_check.py", "full_review_fixes_check.py",
    "immersion_dsp_check.py", "immersion_qa_check.py", "mastering_polish_check.py", "pack_catalog_check.py",
    "pack_signing_check.py", "rushed_line_check.py", "silent_line_check.py", "top5_fixes_check.py",
    "home_move_check.py", "pc_voices_check.py", "double_start_check.py", "dialogue_voice_check.py", "updater_check.py", "untitled_book_check.py", "windows_port_check.py",
]
KNOWN_FAILING = ["interactive_script_check.py", "music_check.py"]  # (failing before the cross-platform work)


def run(name):
    if not (ROOT / "tests" / name).exists():  # (a check from work not in this copy yet)
        print(f"skip {name} (not in this copy)", flush=True)
        return True
    home = tempfile.mkdtemp(prefix="babble-check-")
    env = dict(os.environ, NARRATOR_HOME=str(Path(home) / "data"), PYTHONPATH=str(ROOT), PYTHONUTF8="1")
    t = time.time()
    try:
        r = subprocess.run([sys.executable, str(ROOT / "tests" / name)], cwd=ROOT, env=env, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=900)
        ok, out = r.returncode == 0, (r.stdout + r.stderr)
    except subprocess.TimeoutExpired:
        ok, out = False, "timed out after 15 minutes"
    print(f"{'ok  ' if ok else 'FAIL'} {name} ({time.time() - t:.0f}s)", flush=True)
    if not ok:
        print("\n".join("    " + line for line in out.strip().splitlines()[-25:]), flush=True)
    return ok


def main():
    failed = [n for n in CHECKS if not run(n)]
    if "--all" in sys.argv:
        print("\nKnown to fail (not counted):")
        for n in KNOWN_FAILING:
            run(n)
    print(f"\n{len(CHECKS) - len(failed)} of {len(CHECKS)} passed" + (f"; failed: {', '.join(failed)}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
