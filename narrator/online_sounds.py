"""Recorded sounds from online libraries (Freesound, Lots of Sounds), with the user's own keys.

When a tag needs a sound the library doesn't have, the online libraries are searched first: a few candidates from
each are downloaded, CLAP scores how much each sounds like the description (sfxgen.score), and the best one is kept
in the library when it scores well enough. Otherwise the sound is made with Stable Audio as before.

Only CC0 (public domain) sounds are used, so nothing needs crediting and a book can be sold. Each sound keeps where it
came from (source, author, link) in the library.

Terms (checked November 2026):
  Freesound   free API key at freesound.org/apiv2/apply; the API is free for non-commercial use, and commercial
              apps need a licence from Freesound (contact form). Previews (~128 kbps MP3) need only the key.
  Lots of Sounds  CC0 catalog (many of its sounds come from Freesound's CC0 ones); the API needs its Pro plan
              (USD 15 a month, 2,500 requests a day).
"""
import concurrent.futures as cf
import re
import tempfile
from pathlib import Path

from . import settings
from . import conn_issues

SOURCES = {
    "freesound": {"name": "Freesound", "keys": ["freesound.org/apiv2/apply", "Log in", "Create new API credential"],
                  "cost": "free", "note": "Free for personal, non-commercial use. Only public-domain (CC0) sounds are used."},
    "lotsofsounds": {"name": "Lots of Sounds", "keys": ["lotsofsounds.com/dashboard/api-keys", "Create key"],
                     "cost": "paid", "note": "The API needs its Pro plan (about $15 a month). Every sound is public domain (CC0)."},
}
LICENSE = "CC0 1.0 (public domain)"
# a sound's length for each kind: effects are short; backgrounds and music need enough to loop
DURATION = {"effect": (0.2, 15.0), "ambience": (8.0, 600.0), "music": (15.0, 600.0)}
# CLAP likeness a recording needs to be used for a description. Generated sounds only need sfxgen.CHECK_MIN (0.12),
# since they're made for the words; a recording found by keywords can be something else entirely. Set from 24 real
# Freesound searches (Nov 2026): right sounds scored 0.33–0.46, wrong ones up to 0.29 (a loudspeaker van for rain, a
# drum kit for dripping water); music matched worse (a folk dance scored 0.34 for a taiko chase), so its bar is higher.
ONLINE_MIN = {"effect": 0.33, "ambience": 0.33, "music": 0.36}
PER_SOURCE = 5         # candidates tried from each library
MAX_BYTES = 60_000_000  # a file larger than this isn't downloaded (a long ambience preview is a few MB)
TIMEOUT = 45
SCORE_SECONDS = 30      # how much of a recording CLAP listens to when judging it
SR = 24000


def key_setting(sid):
    return f"sound_key_{sid}"


def key(sid):
    return (settings.load().get(key_setting(sid)) or "").strip()


_OFF = {}  # library -> why it was turned off until its key is saved again (refused: wrong key, no Pro plan)


def ready():
    """The libraries with a key saved (and not refused since)."""
    return [sid for sid in SOURCES if key(sid) and sid not in _OFF]


def sources():
    return [{"id": sid, "name": s["name"], "keys": s["keys"], "cost": s["cost"], "cost_note": s["note"],
             "has_key": bool(key(sid))} for sid, s in SOURCES.items()]


def _http():
    import httpx
    return httpx.Client(timeout=TIMEOUT, follow_redirects=True, headers={"User-Agent": "Babble audiobook studio"})


def _why(sid, r):
    """An error response in words."""
    name = SOURCES[sid]["name"]
    try:
        detail = r.json().get("error") or r.json().get("detail") or ""
    except ValueError:
        detail = ""
    if r.status_code in (401,):
        return f"{name} didn't accept the key. Check it and paste it again."
    if r.status_code == 403 and sid == "lotsofsounds":
        return "Lots of Sounds' API needs its Pro plan. Upgrade at lotsofsounds.com/pricing, then try again."
    if r.status_code == 403:
        return f"{name} refused the request: {detail or 'not allowed'}."
    if r.status_code == 429:
        return f"{name} hit its request limit for now. Try again later."
    return f"{name}: {detail or f'error {r.status_code}'}"


def save_key(sid, api_key):
    """Check a key with a tiny search, then keep it (settings file, readable only by you)."""
    api_key = (api_key or "").strip()
    if sid not in SOURCES:
        raise ValueError("Unknown sound library.")
    if not api_key:
        raise ValueError("Paste the key first.")
    try:
        with _http() as c:
            r = _request(c, sid, "rain", "effect", 1, api_key)
    except Exception as e:  # noqa: BLE001 - shown to the user
        raise ValueError(f"Couldn't reach {SOURCES[sid]['name']}: {e}") from e
    if r.status_code != 200:
        raise ValueError(_why(sid, r))
    settings.save(**{key_setting(sid): api_key})
    _OFF.pop(sid, None)
    conn_issues.ok(sid)
    return {"ok": True}


def forget_key(sid):
    settings.save(**{key_setting(sid): ""})
    _OFF.pop(sid, None)
    conn_issues.ok(sid)


# ---------- searching ----------

def _request(c, sid, query, kind, n, api_key=None):
    lo, hi = DURATION.get(kind, DURATION["effect"])
    if sid == "freesound":
        params = {"query": query, "page_size": n, "sort": "score", "group_by_pack": 1,
                  "filter": f'license:"Creative Commons 0" duration:[{lo} TO {hi}]',
                  "fields": "id,name,tags,duration,username,previews,url"}
        headers = {"Authorization": f"Token {api_key or key(sid)}"}
        r = c.get("https://freesound.org/apiv2/search/", params=params, headers=headers)
        if r.status_code == 404:  # (the older address, in case the new one isn't there)
            r = c.get("https://freesound.org/apiv2/search/text/", params=params, headers=headers)
        return r
    category = {"effect": "sfx", "ambience": "ambient,sfx", "music": "music,lofi"}.get(kind, "sfx")
    return c.get("https://api.lotsofsounds.com/api/v1/sounds",
                 params={"q": query, "limit": n, "min_duration": lo, "max_duration": hi, "category": category},
                 headers={"x-api-key": api_key or key(sid)})


def _candidates(sid, data):
    out = []
    if sid == "freesound":
        for s in data.get("results") or []:
            url = (s.get("previews") or {}).get("preview-hq-mp3")
            if url:
                out.append({"source": sid, "id": str(s["id"]), "name": s.get("name") or "", "tags": s.get("tags") or [],
                            "duration": float(s.get("duration") or 0), "credit": s.get("username") or "",
                            "link": s.get("url") or f"https://freesound.org/s/{s['id']}/", "play": url})
    else:
        for s in data.get("data") or []:
            url = s.get("stream_url") or s.get("download_url")
            if url and str(s.get("license") or "CC0").upper().startswith("CC0"):
                out.append({"source": sid, "id": str(s["id"]), "name": s.get("name") or "", "tags": s.get("tags") or [],
                            "duration": float(s.get("duration") or 0), "credit": "",
                            "link": f"https://www.lotsofsounds.com/sounds/{s['id']}", "play": url})
    return out


_FILLER = set("""a an the of and or with on in at to from into onto over under through by for its their his her
some few many very slightly softly gently faintly distant far near close nearby low high heavy light small large big
quick slow fast steady constant continuous occasional intermittent rhythmic quiet loud soft hard deep sharp""".split())


def _content(text):
    return [w for w in re.findall(r"[a-z]+", text.lower()) if w not in _FILLER and len(w) > 2]


def queries(desc):
    """Keyword searches for a description, most specific first. Both libraries need every word to match, so a
    director's long description ("heavy wooden door slamming shut with a rattle") finds nothing as it is. The sound's
    short name (naming.short_name: "Wooden Door Slam", the name it gets in the library anyway) gives the main words."""
    from . import naming
    try:
        short = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", naming.short_name(desc))  # "InsectoidLifeDrone" -> words
    except Exception:  # noqa: BLE001 - the description's own words still work
        short = ""
    named = _content(short)
    first = _content(re.split(r"\b(?:with|and|while|as)\b|[,;]", (desc or "").lower())[0])
    out = []
    if len(named) >= 2:
        out += [" ".join(named[-3:]), " ".join(named[-2:]), " ".join(named[:2])]
    if len(first) >= 2:
        out.append(" ".join(first[:2]))
    out += named[-1:] or first[:1]
    return list(dict.fromkeys(q for q in out if q))[:4]


def search(query, kind="effect", n=PER_SOURCE, only=None):
    """Candidates from every library with a key (or only one), for one keyword query."""
    found, errors = [], []
    with _http() as c:
        for sid in (only and [only]) or ready():
            try:
                r = _request(c, sid, query, kind, n)
            except Exception as e:  # noqa: BLE001 - another library may still answer
                errors.append(f"Couldn't reach {SOURCES[sid]['name']}: {e}")
                conn_issues.note(sid, e, "Looking for sounds")
                continue
            if r.status_code != 200:
                errors.append(_why(sid, r))
                if r.status_code in (401, 403):  # it won't work until the key or plan changes: not asked again
                    _OFF[sid] = errors[-1]
                if r.status_code in (401, 402, 403, 429):
                    conn_issues.note(sid, RuntimeError(f"{r.status_code}: {errors[-1]}"), "Looking for sounds")
                continue
            conn_issues.ok(sid)
            found += _candidates(sid, r.json())
    return found, errors


SAFE_AUDIO_SUFFIXES = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".opus"}


def _fetch(c, cand):
    """A candidate's audio (24 kHz), downloaded to a temporary file and read with ffmpeg."""
    from urllib.parse import urljoin
    import httpx
    from . import library, sfx
    url = library._safe_remote_url(cand["play"])
    with httpx.Client(timeout=TIMEOUT, follow_redirects=False, headers={"User-Agent": "Babble audiobook studio"}) as dl:
        for _ in range(6):
            with dl.stream("GET", url) as r:
                if r.is_redirect and r.headers.get("location"):
                    url = library._safe_remote_url(urljoin(str(r.url), r.headers["location"]))
                    continue
                r.raise_for_status()
                if int(r.headers.get("content-length") or 0) > MAX_BYTES:
                    raise ValueError("too large")
                raw_suffix = Path(str(r.url).split("?")[0]).suffix.lower()
                suffix = raw_suffix if raw_suffix in SAFE_AUDIO_SUFFIXES else ".mp3"
                with tempfile.NamedTemporaryFile(suffix=suffix) as f:
                    size = 0
                    for chunk in r.iter_bytes():
                        size += len(chunk)
                        if size > MAX_BYTES:
                            raise ValueError("too large")
                        f.write(chunk)
                    f.flush()
                    return sfx._to_24k(f.name)
    raise ValueError("too many redirects")


def _known(cand):
    from . import sfx
    tag = f"{cand['source']}:{cand['id']}"
    return next((e for e in sfx.pool() if e.get("online") == tag), None)  # (one kept in another book: a new copy)


def find(desc, kind="effect", log=print):
    """(entry, new): the best recorded sound online for a description, added to the library (new), or one already
    there that now also answers to this description; or None when nothing sounds enough like it (or no key)."""
    from . import sfx, sfxgen, tasks
    if not ready():
        return None
    cands, seen, errors = [], set(), []
    for q in queries(desc):
        found, errs = search(q, kind)
        errors += errs
        for cnd in found:
            if (cnd["source"], cnd["id"]) not in seen:
                seen.add((cnd["source"], cnd["id"]))
                cands.append(cnd)
        if len(cands) >= PER_SOURCE:
            break
        if tasks.stop_wanted():
            raise tasks.Stopped()
    for e in dict.fromkeys(errors):
        log(e)
    if not cands:
        return None

    def heard(cnd):
        known = _known(cnd)
        if known:
            return cnd, sfx.load(known), known
        with _http() as c:
            return cnd, _fetch(c, cnd), None
    got = []
    with cf.ThreadPoolExecutor(4) as pool:  # downloads at once; CLAP scores them one at a time below
        for fut in cf.as_completed([pool.submit(heard, c) for c in cands[:PER_SOURCE * 2]]):
            try:
                got.append(fut.result())
            except Exception as e:  # noqa: BLE001 - one bad file doesn't stop the rest
                print(f"[online] skipped a candidate: {e}", flush=True)
    scored = []
    for cnd, audio, known in got:
        if audio is None or not len(audio):
            continue
        s = sfxgen.score(desc, audio[:SCORE_SECONDS * SR])  # (a long background: its start says what it is)
        if s is None:  # no CLAP model: a keyword hit can't be told from a wrong sound, so nothing is used
            return None
        scored.append((s, cnd, audio, known))
    if not scored:
        return None
    s, cnd, audio, known = max(scored, key=lambda t: t[0])
    print(f"[online] “{desc}”: best {SOURCES[cnd['source']]['name']} “{cnd['name']}” {s:.2f} "
          f"(of {len(scored)})", flush=True)
    if s < ONLINE_MIN.get(kind, 0.25):
        return None
    if known:
        return sfx.also_for(known["id"], desc), False
    if cnd["source"] == "lotsofsounds":  # listened to through its stream link; kept through a proper download
        with _http() as c:
            audio = _fetch(c, dict(cnd, play=_los_download(c, cnd)))
    return keep(cnd, audio, desc, kind), True


def _los_download(c, cnd):
    r = c.get(f"https://api.lotsofsounds.com/api/v1/sounds/{cnd['id']}/download",
              headers={"x-api-key": key("lotsofsounds")})
    if r.status_code != 200:
        raise ValueError(_why("lotsofsounds", r))
    return r.json()["data"]["download_url"]


def keep(cnd, audio, desc, kind):
    """Add a recording to the library under the description it was found for."""
    from . import naming, sfx, sfxgen
    if kind == "effect":
        audio = sfxgen._trim(audio)
    name = sfx.unique_name(naming.short_name(desc) if desc else (cnd["name"] or "Sound"))
    tags = sfx._words(desc or cnd["name"]) + [str(t).lower() for t in cnd.get("tags") or [] if len(str(t)) < 24][:12]
    src = SOURCES[cnd["source"]]["name"]
    e = sfx.add(audio, name, kind, tags=tags, license=LICENSE, description=desc or cnd["name"],
                source=f"{src}: “{cnd['name']}” {cnd['link']}", credit=cnd.get("credit") or src)
    return sfx.annotate(e["id"], online=f"{cnd['source']}:{cnd['id']}")


def add_by_hand(cnd, kind, desc=""):
    """The Sounds page's online search: keep the one the user picked (its own name unless they gave a description)."""
    if not cnd.get("play") or cnd.get("source") not in SOURCES:
        raise ValueError("That sound can't be added.")
    known = _known(cnd)
    if known:
        return known
    with _http() as c:
        if cnd["source"] == "lotsofsounds":  # a kept sound is downloaded properly (a stream link is for listening)
            cnd = dict(cnd, play=_los_download(c, cnd))
        audio = _fetch(c, cnd)
    return keep(cnd, audio, (desc or "").strip() or re.sub(r"[_\-.]+", " ", cnd["name"]).strip(), kind)
