"""Train voices: pick datasets, a recipe (which model) and a back end (where), then send / check / finish.

    from narrator import training
    run = training.send(["alice-book=alice", "marcus-book=marcus"], model="narrators", recipe="orpheus", backend="kaggle")
    training.check("narrators")      # RunStatus
    training.finish("narrators", genders={"alice": "female", "marcus": "male"})   # fetch + install

The same calls drive the CLI (python -m narrator.training) and, later, the Train tab.
"""
import json
import re
import shutil
import threading
from pathlib import Path

from ..paths import TRAINING, WORK
from .backends import BACKENDS
from .base import AuthError, Package, RunStatus, VoiceData, package_summary, run_key, runs, save_run  # noqa: F401
from .recipes import RECIPES

RESULTS = WORK / "results"


def ready_voices():
    """[(voice folder name, clips, hours)] for every finished Train-tab dataset."""
    import json
    out = []
    for meta in sorted(TRAINING.glob("*/dataset/metadata.jsonl")):
        rows = [json.loads(line) for line in open(meta, encoding="utf-8") if line.strip()]
        out.append((meta.parent.parent.name, len(rows), sum(r.get("duration", 0) for r in rows) / 3600))
    return out


def _voice(spec):
    """'alice-book=alice' -> VoiceData(dataset, key='alice-book', speaker='alice')."""
    src, _, speaker = spec.partition("=")
    ds = Path(src).expanduser()
    if not (ds / "metadata.jsonl").exists():
        ds = TRAINING / src / "dataset"
    key = ds.parent.name if ds.name == "dataset" else ds.name
    import re
    speaker = re.sub(r"[^a-z0-9]+", "", (speaker or key.split("-")[0]).lower()) or "voice"
    return VoiceData(dataset=ds if (ds / "metadata.jsonl").exists() else None, key=key, speaker=speaker)


def prepare(specs, model, recipe="orpheus", **kw):
    return RECIPES[recipe].prepare([_voice(s) for s in specs], model, **kw)


def resolve(name):
    """A run key ("qwen/raytest") from a key or a bare model name, if that name is unique."""
    data = runs()
    if name in data:
        return name
    hits = [k for k in data if k.split("/", 1)[-1] == name]
    if len(hits) == 1:
        return hits[0]
    raise KeyError(f"No run called '{name}'." if not hits else f"'{name}' is ambiguous: {', '.join(hits)}")


def estimate(specs, recipe="orpheus", backend="kaggle", epochs=None, max_clips=0):
    """Quick estimate from the datasets' hours, without encoding anything."""
    r, b = RECIPES[recipe], BACKENDS[backend]
    hours = sum(h for v, _, h in ready_voices() if v in {s.partition("=")[0] for s in specs})
    clips = sum(n for v, n, _ in ready_voices() if v in {s.partition("=")[0] for s in specs})
    if max_clips and clips:
        hours = hours * min(1.0, max_clips / clips)
    passes = epochs or (3.0 if recipe == "qwen" else 1.0)
    gpu = round(hours * r.gpu_hours_per_audio_hour * (passes if recipe == "qwen" else passes) + 0.5, 1)
    return {"hours": round(hours, 2), "gpu_hours": gpu, "backend": b.name}


def send(specs, model, recipe="orpheus", backend="kaggle", dry_run=False, log=print, genders=None, names=None,
         consent=False, **kw):
    """Send a training run (see _send), noting a connection problem with the service on Connections."""
    from .. import conn_issues
    return conn_issues.watch(backend, "Training", _send, specs, model, recipe, backend, dry_run, log, genders, names,
                             consent, **kw)


def _send(specs, model, recipe="orpheus", backend="kaggle", dry_run=False, log=print, genders=None, names=None,
          consent=False, **kw):
    r, b = RECIPES[recipe], BACKENDS[backend]
    if r.readiness != "ready":
        raise ValueError(f"{r.name} training isn't built yet.")
    if b.readiness not in ("ready", "beta"):
        raise ValueError(f"{b.name} isn't available yet.")
    pkg = r.prepare([_voice(s) for s in specs], model, log=log, **kw)
    key = run_key(r.id, pkg.model)
    est = b.estimate(pkg) if not dry_run else {"gpu_hours": pkg.gpu_hours}
    log(f"Packaged {pkg.meta.get('clips', 0):,} clips ({pkg.hours:.2f} h) for '{pkg.model}': "
        f"about {pkg.gpu_hours} GPU hours on {b.name}.")
    budget = est.get("budget_hours")  # GPU hours this run may use (e.g. what's left of the Kaggle week)
    if budget is not None and not dry_run:
        if budget < 1.0:
            raise ValueError(f"Only {max(budget, 0):.1f} GPU hours are left on {b.name} this week: not enough to train. "
                             f"{est.get('resets') or 'It resets weekly.'}")
        if re.search(r"--max-hours \S+", pkg.args):  # the script saves and stops before the budget runs out
            hours = min(budget, float(re.search(r"--max-hours (\S+)", pkg.args).group(1)))
            pkg.args = re.sub(r"--max-hours \S+", f"--max-hours {hours:.1f}", pkg.args)
    if est.get("note"):
        log(est["note"])
    common = {**pkg.meta, "model": pkg.model, "recipe": r.id, "backend": b.id, "hours": pkg.hours,
              "gpu_hours": pkg.gpu_hours, "genders": genders or {}, "names": names or {},
              "consent": bool(consent), "small_outputs": pkg.small_outputs, "big_outputs": pkg.big_outputs}
    if dry_run:
        return save_run(key, **common, state="packaged", package=str(pkg.data_dir))
    from .. import pricing, spend
    price = spend.gpu_price(b.id, pkg.gpu)
    limit_h = float(pkg.meta.get("timeout_hours") or 0) or max(1.0, float(pkg.gpu_hours or 1) * 2.5)
    if price:  # within today's spending limit (the most it can bill: its time limit × the price)
        spend.allow(b.id, pricing.gpu_range(b.id, price, float(pkg.gpu_hours or 0), limit_h, "train"), "train")
    run_id = b.submit(pkg, log=log)
    spend.job_sent(b.id, run_id, f"Training {pkg.model}", price=price, hours=limit_h, collect=True)
    url = b.url(run_id) if hasattr(b, "url") else ""
    log(f"Training started: {url or run_id}")
    return save_run(key, **common, run_id=run_id, url=url, state="sent", sent=package_summary(pkg)["when"])


def check(name):
    """The run's state. A run its back end calls finished is only "complete" if its log says the model was made:
    a notebook finishes even when the training script in it failed."""
    key = resolve(name)
    run = runs()[key]
    if run.get("state") in ("installed", "packaged"):
        return RunStatus(run["state"], "", run.get("url", ""))
    b = BACKENDS[run["backend"]]
    from .. import conn_issues
    st = conn_issues.watch(run["backend"], "Checking a training run", b.status, run["run_id"])
    if st.state in ("complete", "error"):
        out = results_dir(key)
        try:  # the logs and test samples: small, and the samples can be heard before installing
            if not (out / ".downloading").exists():  # (an install that stopped part way is kept, to carry on)
                shutil.rmtree(out, ignore_errors=True)
            b.fetch(run["run_id"], out, run["small_outputs"])
            why = why_failed(out)
        except Exception:  # noqa: BLE001 - can't tell from here: keep what the back end says
            why = None
        if why or st.state == "error":
            st = RunStatus("error", why or f"The run stopped with an error on {b.name}.", st.url)
    if st.state != "unknown":
        save_run(key, state=st.state, error=st.detail if st.state == "error" else None)
    if st.state in ("complete", "error", "cancelled"):
        from .. import spend
        spend.job_ended(run["run_id"])  # (its GPU time at the list price, once)
    return st


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _log_text(path):
    """A log file as plain text (Kaggle's notebook log is a JSON list of output chunks)."""
    raw = Path(path).read_text(errors="replace")
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            raw = "".join(x.get("data", "") for x in data if isinstance(x, dict))
    except ValueError:
        pass
    return _ANSI.sub("", raw)


def why_failed(out):
    """Why a run made no model, in plain words, from the logs it left; None if they show no failure."""
    logs = [_log_text(f) for f in sorted(Path(out).rglob("*.log"))] if Path(out).exists() else []
    text = "\n".join(logs)
    if re.search(r"^\[.*\] (Done\. Merged model|Saved .*-qwen|Training finished)", text, re.M):
        return None
    if re.search(r"OutOfMemoryError|CUDA out of memory|GPU ran out of memory", text):
        return "The GPU ran out of memory, so nothing was trained."
    m = re.findall(r"FAILED: (.+)", text)
    if m:
        return f"It stopped: {m[-1].strip()}"
    if "Traceback (most recent call last)" in text:
        last = [ln.strip() for ln in text.splitlines() if re.match(r"\s*[\w.]*(Error|Exception)\b", ln)]
        return f"It stopped with an error: {last[-1][:300]}" if last else "It stopped with an error."
    return None


def cancel(name):
    """Stop a run that's training (or waiting for a GPU) on its service. ValueError where the service can't be
    stopped from Babble, saying where to stop it instead."""
    key = resolve(name)
    run = runs()[key]
    b = BACKENDS[run["backend"]]
    from .. import spend
    if not spend.stop_remote(run["backend"], run["run_id"]):  # (cancel or delete, whichever the service has)
        where = run.get("url") or (b.url(run["run_id"]) if hasattr(b, "url") else "")
        raise ValueError(f"{b.name} runs can't be cancelled from Babble: stop it on {b.name}"
                         + (f" ({where})." if where else "."))
    spend.job_ended(run["run_id"])
    save_run(key, state="stopped", error=None)


def forget(name):
    """Take a run that never made a model off the list (its downloaded logs go too)."""
    key = resolve(name)
    run = runs()[key]
    if run.get("state") == "installed":
        raise ValueError("That run's model is installed: delete the model instead.")
    shutil.rmtree(results_dir(key), ignore_errors=True)
    data = runs()
    data.pop(key, None)
    from .base import RUNS
    RUNS.write_text(json.dumps(data, indent=1, default=str))


def results_dir(name):
    return RESULTS / resolve(name).replace("/", "-")


_installing, _installing_lock = set(), threading.Lock()


def installing(name):
    """True while finish() is downloading or installing this run."""
    return resolve(name) in _installing


def finish(name, genders=None, names=None, keep_download=True, log=print):
    """Fetch samples + log, then the model; keep the adapter; install. Returns the installed folder.
    Only one install per run at a time: two at once download into the same folder and overwrite each other."""
    key = resolve(name)
    with _installing_lock:
        if key in _installing:
            raise RuntimeError(f"{key} is already being installed. Wait for that to finish.")
        _installing.add(key)
    try:
        return _finish(key, genders, names, keep_download, log)
    finally:
        _installing.discard(key)


# Installing a trained model, step by step (what each step says, and its share of the whole progress bar)
FINISH_STEPS = (("Downloading the samples and the training log", 0.02), ("Downloading the trained model", 0.83),
                ("Keeping its adapter (so voices can be added later)", 0.02),
                ("Converting it for this Mac and installing it", 0.13))


def _finish(key, genders, names, keep_download, log):
    import contextlib
    from .. import progress
    run = runs()[key]
    r, b = RECIPES[run["recipe"]], BACKENDS[run["backend"]]
    out = results_dir(key)
    # a download that stopped part way carries on (back ends that can), if it was this same run's
    mark = out / ".downloading"
    resume = getattr(b, "resumes", False) and mark.exists() and mark.read_text().strip() == run["run_id"]
    if out.exists() and not resume:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    mark.write_text(run["run_id"])
    more = {"resume": True} if resume else {}
    stage = progress.stages(log, FINISH_STEPS)

    def watching(say, total=None):  # the folder filling, for back ends that can't say how far along they are
        if getattr(b, "reports_progress", False):
            return contextlib.nullcontext()
        return progress.downloading(out, "", say, total=total)

    say = stage(0)
    with watching(say):
        b.fetch(run["run_id"], out, run["small_outputs"], log=say)
    has_samples = (out / "samples").exists() and any((out / "samples").glob("*.wav"))
    say = stage(1)
    if not has_samples:
        say("No test samples came back (the run may have failed, or only the samples did). Checking for the model…")
    size = getattr(b, "output_size", lambda *_: None)(run["run_id"], run["big_outputs"])
    if resume:
        say("carrying on from where the last download stopped")
    with watching(say, total=size):
        b.fetch(run["run_id"], out, run["big_outputs"], log=say, **more)
    mark.unlink(missing_ok=True)
    if not any(p.name == "config.json" for p in out.rglob("config.json")):
        why = why_failed(out) or "it stopped before saving a model."
        save_run(key, state="error", error=why)
        raise RuntimeError(f"No trained model came back. {why} The logs are in {out}.")
    if list(out.rglob("*.kaggle-partial")):
        raise RuntimeError("The download didn't finish (partial files left). Run finish again.")
    r.after_fetch(runs()[key], out, log=stage(2))
    dest = r.install(runs()[key], out, genders=genders or run.get("genders"), names=names or run.get("names"),
                     log=stage(3))
    save_run(key, state="installed", installed=str(dest))
    if not keep_download:
        shutil.rmtree(out, ignore_errors=True)
    from .. import tts
    tts.refresh()
    return dest
