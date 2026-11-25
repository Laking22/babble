"""Offline checks for the outside-connections and security audit fixes (no live API calls required)."""
import errno
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from narrator import cloud_ai, cloud_book, conn_issues, modal_jobs, online_sounds, sdk_jobs, spend, tasks  # noqa: E402
from narrator.training.backends import kaggle, machines, modal_cloud  # noqa: E402
from narrator.training.base import AuthError  # noqa: E402


def test_enospc_guards():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # conn_issues._write survives ENOSPC
        with mock.patch.object(conn_issues, "FILE", tmp / "issues.json"), \
             mock.patch.object(Path, "write_text", side_effect=OSError(errno.ENOSPC, "No space left")):
            conn_issues._write({"modal": {"kind": "network"}})

        # spend._append and _save_jobs survive ENOSPC
        with mock.patch.object(spend, "FILE", tmp / "spending.jsonl"), \
             mock.patch.object(spend, "JOBS", tmp / "jobs.json"):
            with mock.patch.object(Path, "open", side_effect=OSError(errno.ENOSPC, "No space left")):
                spend._append({"type": "charge", "dollars": 0.01})
            with mock.patch.object(Path, "write_text", side_effect=OSError(errno.ENOSPC, "No space left")):
                spend._save_jobs({"r1": {"backend": "modal"}})

        # cloud_book _net_fail / _net_ok / _download_now release lock & survive ENOSPC
        book_dir = tmp / "book-x"
        book_dir.mkdir()
        (book_dir / cloud_book.RECORD).write_text(json.dumps({"backend": "modal", "run": "r1", "state": "downloading"}))
        with mock.patch.object(cloud_book, "_write", side_effect=OSError(errno.ENOSPC, "No space left")):
            cloud_book._net_fail(book_dir, "status", ConnectionError("reset"))
            assert str(book_dir) in cloud_book.DISK_FULL
            cloud_book._downloads.add(str(book_dir))
            with mock.patch("narrator.training.BACKENDS", {"modal": mock.Mock(live=False, fetch=mock.Mock(side_effect=RuntimeError("boom")), keeps_hours=1)}):
                cloud_book._download_now(book_dir)
            assert str(book_dir) not in cloud_book._downloads, "_downloads lock leaked on ENOSPC"
        cloud_book.DISK_FULL.pop(str(book_dir), None)


def test_spend_stop_all_and_forget():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        jobs_file = tmp / "cloud-jobs.json"
        spend_file = tmp / "spending.jsonl"
        jobs_file.write_text(json.dumps({
            "r1": {"backend": "modal", "what": "Job 1", "sent": time.time() - 60, "collect": False},
            "r2": {"backend": "runpod", "what": "Job 2", "sent": time.time() - 60, "collect": False},
        }))
        calls = []

        def fake_stop(backend, run):
            calls.append((backend, run))
            if run == "r1":
                raise RuntimeError("network blip")
            return True

        with mock.patch.object(spend, "JOBS", jobs_file), mock.patch.object(spend, "FILE", spend_file), \
             mock.patch.object(spend, "stop_remote", side_effect=fake_stop), \
             mock.patch("narrator.books.BOOKS", tmp):
            stopped = spend.stop_all()
            assert ("modal", "r1") in calls and ("runpod", "r2") in calls, f"aborted early: {calls}"
            assert "Job 1" in stopped and "Job 2" in stopped

        # cloud_book.forget() calls spend.job_ended(run) once without double-charging
        book_dir = tmp / "book-y"
        book_dir.mkdir()
        (book_dir / cloud_book.RECORD).write_text(json.dumps({"backend": "modal", "run": "r99", "state": "running"}))
        with mock.patch.object(spend, "stop_remote", return_value=True), \
             mock.patch.object(spend, "job_ended") as ended:
            cloud_book.forget(book_dir)
            ended.assert_called_once_with("r99", charge=False)
        # Also cleans up stray run record when state is already ready/failed
        (book_dir / cloud_book.RECORD).write_text(json.dumps({"backend": "modal", "run": "r100", "state": "failed"}))
        with mock.patch.object(spend, "job_ended") as ended:
            cloud_book.forget(book_dir)
            ended.assert_called_once_with("r100", charge=False)


def test_freesound_header_and_forget_key():
    seen = {}

    class FakeClient:
        def get(self, url, params=None, headers=None):
            seen["url"] = url
            seen["params"] = params or {}
            seen["headers"] = headers or {}
            return mock.Mock(status_code=200)

    online_sounds._request(FakeClient(), "freesound", "rain", "effect", 3, api_key="SECRET_FS_KEY")
    assert "token" not in seen["params"], "Freesound key leaked into query params!"
    assert seen["headers"].get("Authorization") == "Token SECRET_FS_KEY"

    online_sounds._OFF["freesound"] = "401"
    with mock.patch("narrator.settings.save"), mock.patch.object(conn_issues, "ok") as ok:
        online_sounds.forget_key("freesound")
        assert "freesound" not in online_sounds._OFF
        ok.assert_called_once_with("freesound")


def test_backend_status_propagates_network_and_auth():
    # Kaggle
    k = kaggle.Kaggle()
    with mock.patch.object(kaggle, "api", side_effect=AuthError("bad creds")):
        try:
            k.status("u/s")
            raise AssertionError("kaggle.status swallowed AuthError")
        except AuthError:
            pass
    with mock.patch.object(kaggle, "api", side_effect=ConnectionError("reset")):
        try:
            k.status("u/s")
            raise AssertionError("kaggle.status swallowed ConnectionError")
        except ConnectionError:
            pass

    # Modal
    m = modal_cloud.Modal()
    with mock.patch.object(modal_cloud, "step", side_effect=ConnectionError("modal-storage.com reset")):
        try:
            m.status("r1")
            raise AssertionError("modal.status swallowed ConnectionError")
        except ConnectionError:
            pass

    # RentedMachine
    class DummyMachine(machines.RentedMachine):
        id, name = "dummy", "Dummy"
        def _runs(self): return {"r1": {"machine": "m1"}}
        def machine(self, mid): raise ConnectionError("timed out")

    try:
        DummyMachine().status("r1")
        raise AssertionError("machines.status swallowed ConnectionError")
    except ConnectionError:
        pass


def test_modal_and_beam_error_propagation():
    with tempfile.TemporaryDirectory() as tmp:
        have = Path(tmp) / "have.json"
        have.write_text("[]")
        fake_vol = mock.Mock(listdir=mock.Mock(side_effect=RuntimeError("ClientConnectorError modal-storage.com")))
        with mock.patch.object(modal_jobs, "_volume", return_value=fake_vol):
            try:
                modal_jobs.fetch_new({"run": "r1", "dest": tmp, "have_file": str(have)})
                raise AssertionError("modal_jobs.fetch_new swallowed network error")
            except RuntimeError:
                pass

    with mock.patch.object(sdk_jobs, "_bm_box", side_effect=ConnectionError("Connection timed out")):
        try:
            sdk_jobs.bm_status({"sandbox": "sb1"})
            raise AssertionError("sdk_jobs.bm_status swallowed network error")
        except ConnectionError:
            pass


def test_cloud_ai_interruptible_sleep():
    t0 = time.time()
    with mock.patch.object(tasks, "stop_wanted", return_value=True):
        gen = cloud_ai._through(mock.Mock(side_effect=ConnectionError("reset")), name="Modal", deadline=time.time() + 3600)
        next(gen)  # yields "Lost the connection..."
        try:
            next(gen)
            raise AssertionError("cloud_ai._through did not stop when stop_wanted=True")
        except tasks.Stopped:
            pass
    assert time.time() - t0 < 2.0, "sleep loop did not abort immediately"


if __name__ == "__main__":
    for fn in (test_enospc_guards, test_spend_stop_all_and_forget, test_freesound_header_and_forget_key,
               test_backend_status_propagates_network_and_auth, test_modal_and_beam_error_propagation,
               test_cloud_ai_interruptible_sleep):
        fn()
        print(f"OK: {fn.__name__}")
    print("ALL OK")
