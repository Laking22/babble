"""What paid work costs before it's started: list prices, and estimates as a range that learns from real runs.

AI models: Claude's from director.CLAUDE. Every other service's from OpenRouter's public price list, which carries
most of them under the service's name (openai/…, google/…, x-ai/…, deepseek/…, mistralai/…): fetched at most once a
day into prices.json (no key needed). A model it can't match has no price, said as "price unknown".
Free tiers (Gemini, Groq, Mistral's Experiment plan, OpenRouter's :free models) cost nothing until their daily
limits, so their range starts at $0; the high end is the paid price, in case billing is on for that key.

A range: low is what a run usually costs, high is the most it should (the runaway guard stops a run well past it).
Every finished run is kept with its estimate (spend.py), and the median ratio of real cost to estimate over the
last runs of the same kind on the same service corrects the next estimate.
"""
import json
import re
import statistics
import threading
import time
import urllib.request

from .paths import HOME

CACHE = HOME / "prices.json"
DAY = 86400
LEARN_FROM = 8           # the last runs of the same kind and service the correction is the median of
_lock = threading.Lock()
# OpenRouter's name for a service's models (its list puts them under these)
VENDOR = {"openai": "openai", "gemini": "google", "xai": "x-ai", "deepseek": "deepseek", "mistral": "mistralai",
          "anthropic": "anthropic", "groq": None, "together": None}
FREE_TIER = {"gemini", "groq", "mistral"}


def _openrouter_list():
    """{model id: (dollars per input token, per output token, per cached input token or None)}, a day old at most."""
    with _lock:
        try:
            d = json.loads(CACHE.read_text())
            if time.time() - d.get("at", 0) < DAY and d.get("models"):
                return d["models"]
        except (OSError, ValueError):
            d = {}
        try:
            r = urllib.request.urlopen(urllib.request.Request("https://openrouter.ai/api/v1/models",
                                                              headers={"User-Agent": "Babble"}), timeout=20)
            models = {}
            for m in json.loads(r.read())["data"]:
                p = m.get("pricing") or {}
                try:
                    cached = p.get("input_cache_read")
                    models[m["id"]] = (float(p.get("prompt") or 0), float(p.get("completion") or 0),
                                       float(cached) if cached not in (None, "") else None)
                except (TypeError, ValueError):
                    continue
            tmp = CACHE.with_suffix(".tmp")
            try:
                tmp.write_text(json.dumps({"at": time.time(), "models": models}))
                tmp.replace(CACHE)
            except OSError:
                pass
            finally:
                tmp.unlink(missing_ok=True)
            return models
        except Exception as e:  # noqa: BLE001 - offline: the last list, however old
            print(f"[pricing] couldn't get OpenRouter's prices: {e}", flush=True)
            return d.get("models") or {}


def _norm(model):
    m = model.lower().split("/")[-1]
    m = re.sub(r"^models/", "", m)
    m = re.sub(r"-(latest|preview|exp)$|-\d{4}-?\d{2}-?\d{2}$|-\d{3,4}$", "", m)
    return m.replace(".", "-")


def model_price(service, model):
    """(in, out, cached in) dollars per token, and where it's from; (None, why) when unknown."""
    if not model:
        return None, "no model picked"
    if service == "anthropic":
        from .director import CLAUDE
        p = next((m["price"] for m in CLAUDE if m["id"] == model), None)
        return ((p[0] / 1e6, p[1] / 1e6, p[2] / 1e6), "Anthropic's prices") if p else (None, "price unknown")
    listed = _openrouter_list()
    if service == "openrouter":
        p = listed.get(model)
        return (tuple(p), "OpenRouter's prices") if p else (None, "price unknown")
    vendor = VENDOR.get(service)
    if vendor:
        exact = listed.get(f"{vendor}/{model.split('/')[-1]}")
        if exact:
            return tuple(exact), "list price (via OpenRouter's list)"
        want = _norm(model)
        for mid, p in listed.items():
            if mid.startswith(vendor + "/") and _norm(mid) == want and ":" not in mid:
                return tuple(p), "list price (via OpenRouter's list)"
    return None, "price unknown"


# ---------- learning from real runs ----------

def correction(what, service, version=1, model=None):
    """(factor, spread): the median of real cost ÷ estimate (the middle of its range) over the last runs of this
    kind on this service (and model, when specified) estimated the same way (version), and how far they varied.
    (1, 0) until there are two."""
    from . import spend
    ratios = []
    for e in reversed(spend.entries(time.time() - 120 * DAY)):
        if (e.get("type") != "run" or e.get("what") != what or e.get("service") != service
                or e.get("state") != "done" or e.get("partial")):
            continue
        est = e.get("estimate") or {}
        if est.get("v", 1) != version:
            continue
        if model and (e.get("model") or est.get("model")) != model:
            continue
        mid = est.get("mid") or ((est.get("low") or 0) + (est.get("high") or 0)) / 2
        if mid and e.get("spent"):
            ratios.append(e["spent"] / mid)
        if len(ratios) >= LEARN_FROM:
            break
    if len(ratios) < 2:
        return 1.0, 0.0
    f = statistics.median(ratios)
    spread = (max(ratios) - min(ratios)) / max(f, 1e-9) / 2
    return max(0.2, min(5.0, f)), min(1.0, spread)


def ai_range(service, model, tokens, what="direct", version=1):
    """An AI job's cost, from its token counts: {"in", "in_repeat" (the same prompt sent again: cached where the
    service caches), "out_low", "out_high"}. Returns {low, high, mid, basis, free_tier}."""
    price, basis = model_price(service, model)
    free = service in FREE_TIER or model.endswith(":free")
    if price is None:
        return {"low": 0.0 if free else None, "high": None, "mid": None, "basis": basis, "free_tier": free, "v": version, "model": model}
    p_in, p_out, p_cached = price
    cached = p_cached if p_cached is not None else p_in
    low = tokens["in"] * p_in + tokens.get("in_repeat", 0) * cached + tokens["out_low"] * p_out
    high = (tokens["in"] + tokens.get("in_repeat", 0)) * p_in + tokens["out_high"] * p_out
    f, spread = correction(what, service, version, model=model)
    low, high = low * f * (1 - spread / 2), high * f * (1 + spread)
    if f != 1.0:
        basis += f", corrected by your last runs (×{f:.2f})"
    if free:
        basis += "; free on a free tier until its daily limit"
    return {"low": 0.0 if free else round(low, 4), "high": round(max(high, low), 4), "mid": round((low + high) / 2, 4),
            "basis": basis, "free_tier": free, "v": version, "model": model}


def gpu_range(service, price_per_hour, hours_low, hours_high, what):
    """A cloud GPU job: list price an hour × its likely time (low) and its time limit (high)."""
    if price_per_hour is None:
        free = service == "kaggle"
        return {"low": 0.0 if free else None, "high": 0.0 if free else None, "mid": 0.0 if free else None,
                "basis": "free (weekly GPU hours)" if free else "price unknown", "free_tier": free}
    f, spread = correction(what, service)
    low = price_per_hour * hours_low * f * (1 - spread / 2)
    high = price_per_hour * hours_high  # the job's own time limit: it can't run longer
    return {"low": round(min(low, high), 2), "high": round(high, 2), "mid": round((min(low, high) + high) / 2, 2),
            "basis": f"${price_per_hour:.2f}/h list price" + (f", corrected by your last runs (×{f:.2f})" if f != 1 else ""),
            "free_tier": False}


def say(r):
    """A range in words: "$0.02–$0.05", "free", "up to $1.20", "price unknown"."""
    if r.get("high") is None:
        return "free tier" if r.get("free_tier") else "price unknown"
    if r["high"] == 0:
        return "free"
    lo = r.get("low") or 0

    def d(x):
        return f"${x:.2f}" if x >= 0.01 else "<$0.01"
    if r.get("free_tier") and lo == 0:
        return f"free tier (up to {d(r['high'])} if billed)"
    if r["high"] < 0.01:
        return "under $0.01"
    if lo < 0.01:
        return f"up to {d(r['high'])}"
    return d(r["high"]) if d(lo) == d(r["high"]) else f"{d(lo)}–{d(r['high'])}"
