import { useRef, useState } from "react";
import { FolderOpen, FileArchive, Pause, Play, Plus, Search, Trash2 } from "lucide-react";
import { soundsApi, useLoad, useTask, type OnlineSound, type Sound } from "../api";
import { Segmented, TaskProgress, clock, useToast } from "../components/ui";

type Kind = Sound["kind"];
const SOURCE_NAME: Record<string, string> = { freesound: "Freesound", lotsofsounds: "Lots of Sounds" };

/** Sounds page → Find online: search Freesound / Lots of Sounds (public-domain recordings), listen, add to the library. */
export function FindOnline({ onAdded }: { onAdded: (s: Sound) => void }) {
  const toast = useToast();
  const sources = useLoad(soundsApi.sources);
  const [q, setQ] = useState("");
  const [kind, setKind] = useState<Kind>("effect");
  const [busy, setBusy] = useState(false);
  const [results, setResults] = useState<OnlineSound[] | null>(null);
  const [playing, setPlaying] = useState<string | null>(null);
  const [adding, setAdding] = useState<string | null>(null);
  const [added, setAdded] = useState<Set<string>>(new Set());
  const audio = useRef<HTMLAudioElement>(null);
  const ready = (sources.data || []).some((s) => s.has_key);
  const keyOf = (r: OnlineSound) => `${r.source}:${r.id}`;

  const search = async () => {
    if (!q.trim()) return;
    setBusy(true);
    try {
      const r = await soundsApi.search(q, kind);
      setResults(r.results);
      if (r.errors.length) toast(r.errors[0], true);
    } catch (e) { toast((e as Error).message, true); }
    finally { setBusy(false); }
  };
  const play = (r: OnlineSound) => {
    const el = audio.current;
    if (!el) return;
    if (playing === keyOf(r)) { el.pause(); setPlaying(null); return; }
    el.src = r.play;
    el.play().catch(() => toast("Couldn't play that sound.", true));
    setPlaying(keyOf(r));
  };
  const add = async (r: OnlineSound) => {
    setAdding(keyOf(r));
    try {
      const s = await soundsApi.add(r, kind);
      setAdded((a) => new Set(a).add(keyOf(r)));
      onAdded(s);
    } catch (e) { toast((e as Error).message, true); }
    finally { setAdding(null); }
  };

  return (
    <div className="card card-pad stack" style={{ marginBottom: 16 }}>
      <h3 className="card-title" style={{ margin: 0 }}>Find a recorded sound</h3>
      {sources.data && !ready ? (
        <p className="small muted" style={{ margin: 0 }}>Add a Freesound or Lots of Sounds key in Connections → Sound libraries first.</p>
      ) : (
        <>
          <div className="row wrap">
            <div className="search-box grow"><Search size={14} className="muted" />
              <input className="input" placeholder="door creak" value={q} onChange={(e) => setQ(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && !busy && search()} /></div>
            <Segmented<Kind> value={kind} onChange={setKind} options={[
              { value: "effect", label: "Effect" }, { value: "ambience", label: "Background" }, { value: "music", label: "Music" }]} />
            <button className="btn primary" disabled={busy || !q.trim()} onClick={search}>{busy ? "Searching…" : "Search"}</button>
          </div>
          {results && (
            <div className="list">
              {results.map((r) => (
                <div key={keyOf(r)} className="list-item">
                  <button className={`kind-icon ${kind}`} style={{ border: 0, cursor: "pointer" }} onClick={() => play(r)} aria-label={`Play ${r.name}`}>
                    {playing === keyOf(r) ? <Pause size={16} /> : <Play size={16} />}
                  </button>
                  <div className="grow">
                    <div className="title" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.name}</div>
                    <div className="small muted">{SOURCE_NAME[r.source] || r.source} · {clock(r.duration)}{r.credit ? ` · by ${r.credit}` : ""}</div>
                  </div>
                  <button className="btn sm" disabled={!!adding || added.has(keyOf(r))} onClick={() => add(r)}>
                    <Plus size={13} />{added.has(keyOf(r)) ? "Added" : adding === keyOf(r) ? "Adding…" : "Add"}</button>
                </div>
              ))}
              {!results.length && <div className="empty">Nothing found. Try fewer or simpler words.</div>}
            </div>
          )}
        </>
      )}
      <audio ref={audio} onEnded={() => setPlaying(null)} hidden />
    </div>
  );
}

/** Sounds page → Add sound pack: a folder or .zip of sounds you have (a free Sonniss GDC bundle, a library you
 *  bought, your own recordings), copied into the library. */
export function AddPack({ onAdded }: { onAdded: () => void }) {
  const toast = useToast();
  const packs = useLoad(soundsApi.packs);
  const [path, setPath] = useState("");
  const [name, setName] = useState("");
  const [license, setLicense] = useState("");
  const [confirm, setConfirm] = useState<string | null>(null);
  const { task, start, stop, running } = useTask((t) => { if (t.state === "done") { onAdded(); packs.reload(); setPath(""); } }, "sound-pack");

  const pick = async (zip: boolean) => {
    try {
      const r = await soundsApi.pick(zip);
      if (!r.path) return;
      setPath(r.path);
      const base = r.path.replace(/[\\/]+$/, "").split(/[\\/]/).pop() || "";
      setName(base.replace(/\.zip$/i, "").replace(/[_-]+/g, " ").trim());
    } catch (e) { toast((e as Error).message, true); }
  };
  const remove = async (n: string) => {
    try {
      const r = await soundsApi.removePack(n);
      toast(`Removed ${r.removed} sound(s) from “${n}”.`);
      packs.reload(); onAdded();
    } catch (e) { toast((e as Error).message, true); }
    setConfirm(null);
  };

  return (
    <div className="card card-pad stack" style={{ marginBottom: 16 }}>
      <h3 className="card-title" style={{ margin: 0 }}>Add a sound pack</h3>
      <div className="row wrap">
        <button className="btn" disabled={running} onClick={() => pick(false)}><FolderOpen size={14} />Choose a folder</button>
        <button className="btn" disabled={running} onClick={() => pick(true)}><FileArchive size={14} />Choose a .zip</button>
        {path && <span className="small muted" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 320 }}>{path}</span>}
      </div>
      {path && (
        <div className="row wrap">
          <input className="input grow" placeholder="Pack name" value={name} onChange={(e) => setName(e.target.value)} />
          <input className="input grow" placeholder="License (e.g. CC0, Sonniss GDC licence)" value={license} onChange={(e) => setLicense(e.target.value)} />
          <button className="btn primary" disabled={running || !name.trim()} onClick={() => start(() => soundsApi.importPack(path, name, license))}>Add pack</button>
        </div>
      )}
      {task && <TaskProgress task={task} onStop={stop} />}
      {!!packs.data?.length && (
        <div className="list">
          {packs.data.map((p) => (
            <div key={p.name} className="list-item">
              <div className="grow"><div className="title">{p.name}</div>
                <div className="small muted">{p.sounds.toLocaleString()} sounds · {clock(p.seconds)}{p.license ? ` · ${p.license}` : ""}</div></div>
              {confirm === p.name ? (
                <span className="row" style={{ gap: 4 }}>
                  <button className="btn sm danger" onClick={() => remove(p.name)}>Remove</button>
                  <button className="btn sm ghost" onClick={() => setConfirm(null)}>Keep</button>
                </span>
              ) : (
                <button className="btn icon sm ghost" onClick={() => setConfirm(p.name)} title="Remove pack" aria-label={`Remove ${p.name}`}><Trash2 size={14} /></button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
