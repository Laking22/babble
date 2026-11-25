"""Check cloud GPU dataset transcription (narrator/cloud_stt.py, stt_runner.py, pipeline.py, api_manage.py).

Run: .venv/bin/python tests/cloud_dataset_check.py  (seconds, no network, scratch data folder only)
"""
import json
import os
import sys
import tempfile
import threading
import types
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(tempfile.mkdtemp(prefix="cloud-dataset-"))
os.environ["NARRATOR_HOME"] = str(SCRATCH / "home")  # (never the real data folder)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Headless MLX stub so this runs even without a Metal device
if "mlx" not in sys.modules:
    mx = types.ModuleType("mlx.core")
    mx.array = type("array", (), {})
    mx.clear_cache = lambda: None
    mx.set_cache_limit = lambda *a, **k: None
    mx.set_wired_limit = lambda *a, **k: None
    mx.random = types.SimpleNamespace(seed=lambda *a, **k: None)
    mx.metal = types.SimpleNamespace(
        clear_cache=lambda: None, set_cache_limit=lambda *a, **k: None,
        get_active_memory=lambda: 0, get_peak_memory=lambda: 0, is_available=lambda: False,
    )
    pkg = types.ModuleType("mlx")
    nn = types.ModuleType("mlx.nn")
    nn.Module = object
    utils = types.ModuleType("mlx.utils")
    utils.tree_flatten = lambda x, *a, **k: []
    utils.tree_unflatten = lambda x, *a, **k: {}
    pkg.core, pkg.nn, pkg.utils = mx, nn, utils
    sys.modules["mlx"] = pkg
    sys.modules["mlx.core"] = mx
    sys.modules["mlx.nn"] = nn
    sys.modules["mlx.utils"] = utils

from narrator import api_manage, cloud_stt  # noqa: E402
from narrator.training import BACKENDS  # noqa: E402
from narrator.training.base import RunStatus  # noqa: E402


def test_estimates():
    e_k = cloud_stt.estimate("kaggle", "", 7200)
    assert e_k["free"] is True and e_k["gpu"] == "T4" and "free" in e_k["cost"]
    e_m = cloud_stt.estimate("modal", "L4", 7200)
    assert e_m["free"] is False and e_m["gpu"] == "L4" and "$" in e_m["cost"]
    print("OK: test_estimates")


def test_api_handlers():
    opts = api_manage.train_options()
    ids = [b["id"] for b in opts["prepare_backends"]]
    assert "mac" in ids and "kaggle" in ids and "modal" in ids, ids
    est = api_manage.prepare_estimate(api_manage.PrepareEstimate(backend="modal", cloud_gpu="L4", audio_seconds=3600))
    assert est["gpu"] == "L4" and est["where"] == "Modal"
    print("OK: test_api_handlers")


def test_transcribe_roundtrip_and_cache():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        wav = td / "000.wav"
        sf.write(str(wav), np.zeros(24000 * 3, dtype=np.float32), 24000, subtype="PCM_16")
        stt_dir = td / "stt"
        submits = []

        class FakeBackend:
            id, name = "modal", "Modal"
            hosts = ()
            extra_pip = []

            def check(self):
                return {"ok": True, "detail": "ok"}

            def submit(self, pkg, log=print):
                submits.append(pkg)
                job = json.loads((pkg.data_dir / "stt-job.json").read_text())
                assert job["model"] == "large-v3-turbo"
                assert len(job["files"]) == 1
                assert (pkg.data_dir / job["files"][0]["file"]).stat().st_size > 100
                return "fake-run-1"

            def status(self, run_id):
                return RunStatus("complete", "exit 0")

            def fetch(self, run_id, dest, pattern, log=print):
                (dest / "stt").mkdir(parents=True, exist_ok=True)
                (dest / "stt" / "000.json").write_text(json.dumps({
                    "windows": [{"start": 0.0, "end": 3.0, "words": [["hello", 0.1, 0.5], ["world", 0.6, 1.0]]}]
                }))
                return dest

        with mock.patch.dict(BACKENDS, {"modal": FakeBackend()}), \
             mock.patch("narrator.spend.allow"), \
             mock.patch("narrator.spend.job_sent"), \
             mock.patch("narrator.spend.job_ended"), \
             mock.patch("narrator.cloud_book.stay_awake"):
            words = cloud_stt.transcribe_files(
                "testvoice", [wav], stt_dir, "en", "modal", "L4",
                threading.Event(), lambda s, p: None, lambda m: None,
            )
            assert words == [["hello", 0.1, 0.5, 0], ["world", 0.6, 1.0, 0]], words
            assert len(submits) == 1
            assert (stt_dir / "000.json").exists()
            words2 = cloud_stt.transcribe_files(
                "testvoice", [wav], stt_dir, "en", "modal", "L4",
                threading.Event(), lambda s, p: None, lambda m: None,
            )
            assert words2 == words and len(submits) == 1
    print("OK: test_transcribe_roundtrip_and_cache")


class Fake:
    """A cloud service: runs stay "running" until told, then hand back one transcript per file."""
    id, name, hosts, extra_pip = "modal", "Modal", (), []

    def __init__(self):
        self.submits, self.state, self.stopped = [], "running", []

    def check(self):
        return {"ok": True, "detail": "ok"}

    def submit(self, pkg, log=print):
        self.submits.append(pkg)
        return f"run-{len(self.submits)}"

    def status(self, run_id):
        return RunStatus(self.state, "")

    def fetch(self, run_id, dest, pattern, log=print):
        (dest / "stt").mkdir(parents=True, exist_ok=True)
        (dest / "stt" / "000.json").write_text(json.dumps({"windows": [{"start": 0.0, "end": 3.0, "words": [["hi", 0.1, 0.4]]}]}))
        return dest


def _patched(fake):
    return [mock.patch.dict(BACKENDS, {"modal": fake}), mock.patch("narrator.spend.allow"),
            mock.patch("narrator.spend.job_sent"), mock.patch("narrator.spend.job_ended"),
            mock.patch("narrator.spend.stop_remote", side_effect=lambda b, r: fake.stopped.append(r)),
            mock.patch("narrator.cloud_book.stay_awake"), mock.patch("time.sleep")]


def test_quit_and_carry_on():
    """Babble quitting mid-run picks the same run up again; stopping frees it; nothing is sent twice."""
    import contextlib
    td = SCRATCH / "work-a"
    (td / "stt").mkdir(parents=True)
    wav = td / "000.wav"
    sf.write(str(wav), np.zeros(24000 * 3, dtype=np.float32), 24000, subtype="PCM_16")
    fake = Fake()
    stop = threading.Event()
    with contextlib.ExitStack() as st:
        for p in _patched(fake):
            st.enter_context(p)
        # 1. sent, then Babble "quits" while it runs: the record stays, the run isn't stopped
        quit_now = threading.Event()

        def stage(text, frac):
            if "Transcribing on" in text:
                quit_now.set()
                raise SystemExit  # (the engine going away: nothing after this runs in a real quit)
        try:
            cloud_stt.transcribe_files("v", [wav], td / "stt", "en", "modal", "L4", stop, stage, lambda m: None)
        except SystemExit:
            pass
        rec = cloud_stt.in_flight(td)
        assert quit_now.is_set() and rec and rec["remote"] == "run-1" and len(fake.submits) == 1, rec
        # 2. carrying on: the same run is followed, not a new one; it finishes and the record goes
        fake.state = "complete"
        words = cloud_stt.transcribe_files("v", [wav], td / "stt", "en", "modal", "L4", stop, lambda *a: None, lambda m: None)
        assert words == [["hi", 0.1, 0.4, 0]] and len(fake.submits) == 1, (words, fake.submits)
        assert cloud_stt.in_flight(td) is None and not fake.stopped
        print("OK: a quit mid-run carries on with the same run (nothing sent twice)")
        # 3. stopping while it runs frees the GPU and forgets the run
        (td / "stt" / "000.json").unlink()
        fake.state = "running"

        def stop_soon(text, frac):
            if "Transcribing on" in text:
                stop.set()
        try:
            cloud_stt.transcribe_files("v", [wav], td / "stt", "en", "modal", "L4", stop, stop_soon, lambda m: None)
            raise AssertionError("should have stopped")
        except InterruptedError:
            pass
        assert fake.stopped == ["run-2"] and cloud_stt.in_flight(td) is None, fake.stopped
        print("OK: stopping frees the cloud GPU")


def test_startup_and_carry_on_api():
    """At startup a paid cloud book whose run isn't out waits for the owner; Carry on keeps where it was sent."""
    from narrator.dataset import pipeline as p
    voice = "paidvoice"
    work = p.TRAINING / voice / "books" / "a-book"
    (work / "audio").mkdir(parents=True)
    audio = SCRATCH / "a.wav"
    sf.write(str(audio), np.zeros(24000 * 2, dtype=np.float32), 24000, subtype="PCM_16")
    book = SCRATCH / "a.txt"
    book.write_text("Hello there.")
    (work / "job.json").write_text(json.dumps({"audio": [str(audio)], "book": str(book), "title": "A Book",
                                               "language": "english", "opts": {"backend": "modal", "cloud_gpu": "L4"}}))
    p._mark(voice, True)
    with mock.patch.object(p, "start") as started:
        p.resume_interrupted(log=lambda m: None)
        assert not started.called and not (p.TRAINING / voice / p.PREPARING).exists()
        p._mark(voice, True)
        (work / "stt").mkdir()
        (work / cloud_stt.RECORD).write_text(json.dumps({"backend": "modal", "remote": "run-9", "todo": [0]}))
        p.resume_interrupted(log=lambda m: None)
        assert started.called, "a run that's out is picked up again"
    print("OK: startup never rents a GPU again by itself")
    st = api_manage.prepare_status(voice)
    assert st["backend"] == "modal" and st["cloud_gpu"] == "L4", st
    with mock.patch.object(p, "start") as started:
        api_manage.prepare(name=voice, title="", language="english", options="{}", backend="", cloud_gpu="", files=[])
        assert started.call_args[0][5]["backend"] == "modal", "Carry on keeps the place it was sent to"
        api_manage.prepare(name=voice, title="", language="english", options="{}", backend="mac", cloud_gpu="", files=[])
        assert started.call_args[0][5]["backend"] == "mac", "...unless told otherwise"
    sf.write(str(work / "audio" / "000.wav"), np.zeros(24000 * 2, dtype=np.float32), 24000, subtype="PCM_16")
    e = api_manage.prepare_estimate(api_manage.PrepareEstimate(backend="modal", cloud_gpu="L4", audio_seconds=0, voice=voice))
    assert e["needed"] is True and e["hours"] > 0, e
    (work / "stt" / "000.json").write_text(json.dumps({"windows": [{"start": 0, "end": 2.0, "words": []}]}))
    e = api_manage.prepare_estimate(api_manage.PrepareEstimate(backend="modal", cloud_gpu="L4", audio_seconds=0, voice=voice))
    assert e["needed"] is False, e
    print("OK: Carry on asks about what's left (nothing, once it's all transcribed)")


if __name__ == "__main__":
    test_estimates()
    test_api_handlers()
    test_transcribe_roundtrip_and_cache()
    test_quit_and_carry_on()
    test_startup_and_carry_on_api()
    print("ALL OK")
