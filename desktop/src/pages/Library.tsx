import { useMemo, useRef, useState } from "react";
import { BadgeCheck, Download, Package, Pause, Play, Search, ShieldAlert, Trash2, Upload } from "lucide-react";
import { fileUrl, libraryApi, packPreviewUrl, useLoad, useTask, type Pack, type PackSigned, type PackVoice } from "../api";
import { Orb, TaskProgress, useToast } from "../components/ui";

/** Who signed a pack (narrator/signing.py): confirmed on GitHub, not checked yet, or not confirmed (and why). */
function Signed({ s }: { s: PackSigned }) {
  const tip = s.fingerprint ? `Key ${s.fingerprint}` : undefined;
  if (s.verified) return <span className="small ok-text row" style={{ gap: 4 }} title={tip}><BadgeCheck size={13} />Signed by {s.github}</span>;
  if (s.verified === null) return <span className="small muted" title={tip}>Signed by {s.github} · not checked yet</span>;
  return <span className="small warn-text row" style={{ gap: 4 }} title={tip}><ShieldAlert size={13} />Not confirmed as {s.github}. {s.why}</span>;
}

/** Voice packs: ones you installed, and ones you can install. Models you trained are on Train a voice. */
export function Library({ onChanged }: { onChanged: () => void }) {
  const toast = useToast();
  const packs = useLoad(libraryApi.packs);
  const [q, setQ] = useState("");
  const [genre, setGenre] = useState("All");
  const [confirm, setConfirm] = useState<string | null>(null);
  const [playing, setPlaying] = useState<string | null>(null);
  const audio = useRef<HTMLAudioElement | null>(null);
  const fileIn = useRef<HTMLInputElement>(null);
  const job = useTask((t) => {
    if (t.state === "failed") toast(t.error || "Couldn't install the pack.", true);
    else if (t.state === "done") { toast("Installed. Its voices are in every voice list."); packs.reload(); onChanged(); }
  }, "library-pack-install");

  const match = (p: Pack) => {
    const words = q.toLowerCase().split(/\s+/).filter(Boolean);
    return (genre === "All" || p.genre === genre) && words.every((w) => `${p.name} ${p.genre} ${p.description} ${p.voices.map((v) => v.name).join(" ")}`.toLowerCase().includes(w));
  };
  const installed = useMemo(() => (packs.data?.installed || []).filter(match), [packs.data, q, genre]); // eslint-disable-line react-hooks/exhaustive-deps
  const catalog = useMemo(() => (packs.data?.catalog || []).filter(match), [packs.data, q, genre]); // eslint-disable-line react-hooks/exhaustive-deps
  const genres = useMemo(() => ["All", ...new Set([...(packs.data?.installed || []), ...(packs.data?.catalog || [])].map((p) => p.genre).filter(Boolean))], [packs.data]);

  const play = (key: string, url: string) => {
    if (!audio.current) { audio.current = new Audio(); audio.current.onended = () => setPlaying(null); }
    if (playing === key) { audio.current.pause(); setPlaying(null); return; }
    audio.current.src = url;
    setPlaying(key);
    audio.current.play().catch(() => { setPlaying(null); toast("Couldn't play the preview.", true); });
  };
  const hear = (p: Pack, v: PackVoice, kind: "installed" | "catalog") =>
    kind === "installed"
      ? v.preview ? { key: v.preview, url: fileUrl(v.preview) } : null
      : v.has_preview && v.speaker ? { key: `${p.id}-${p.version}-${v.speaker}`, url: packPreviewUrl(p.id, p.version, v.speaker) } : null;
  const remove = async (p: Pack) => {
    try { await libraryApi.remove(p.folder!); toast(`Removed ${p.name}.`); setConfirm(null); packs.reload(); onChanged(); }
    catch (e) { toast((e as Error).message, true); }
  };

  const card = (p: Pack, kind: "installed" | "catalog") => (
    <div key={`${p.id}-${p.version}`} className="pack-card">
      <div className="row" style={{ alignItems: "flex-start" }}>
        <div className="kind-icon effect"><Package size={16} /></div>
        <div className="grow">
          <b>{p.name}</b>
          <div className="small muted">{[p.genre, p.engine, `v${p.version}`, p.size_gb ? `${p.size_gb} GB` : p.size ? `${(p.size / 1e9).toFixed(1)} GB` : ""].filter(Boolean).join(" · ")}</div>
          {p.signed && <Signed s={p.signed} />}
        </div>
        {kind === "installed"
          ? <button className="btn icon sm ghost" title={`Remove ${p.name}`} onClick={() => setConfirm(p.folder!)}><Trash2 size={14} /></button>
          : <button className="btn sm primary" disabled={job.running} onClick={() => job.start(() => libraryApi.install(p.id, p.version))}><Download size={13} />Install</button>}
      </div>
      {p.description && <p className="small muted" style={{ margin: "8px 0 0" }}>{p.description}</p>}
      <div className="pill-voices" style={{ marginTop: 10 }}>
        {p.voices.map((v) => {
          const h = hear(p, v, kind);
          return (
            <span key={v.speaker || v.name} className="pill-voice">
              <Orb value={v.value || v.name} size={20} />{v.name}
              {h && <button className="btn icon sm ghost" title="Hear it" onClick={() => play(h.key, h.url)}>{playing === h.key ? <Pause size={12} /> : <Play size={12} />}</button>}
            </span>
          );
        })}
      </div>
      {confirm === p.folder && (
        <div className="confirm" role="alertdialog" aria-label={`Remove ${p.name}?`}>
          <span className="small">Remove {p.name} from this Mac? Its voices leave every voice list. You can install it again from its file.</span>
          <div className="row" style={{ gap: 6 }}>
            <button className="btn sm ghost" onClick={() => setConfirm(null)}>Cancel</button>
            <button className="btn sm danger-solid" onClick={() => remove(p)}><Trash2 size={13} />Remove</button>
          </div>
        </div>
      )}
    </div>
  );

  return (
    <div className="page">
      <div className="page-head">
        <div><h1>Library</h1><p>Voice packs: ready-made sets of voices for a kind of book. Packs you make are on Train a voice.</p></div>
        <div className="row">
          <input ref={fileIn} type="file" hidden accept=".narrator-pack,.zip"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) job.start(() => libraryApi.installFile(f)); e.target.value = ""; }} />
          <button className="btn" disabled={job.running} onClick={() => fileIn.current?.click()}><Upload size={15} />Install from a file</button>
        </div>
      </div>
      <div className="row wrap" style={{ marginBottom: 16 }}>
        <div className="search-box"><Search size={15} className="muted" />
          <input className="input" placeholder="Search packs" aria-label="Search packs" value={q} onChange={(e) => setQ(e.target.value)} /></div>
        <div className="chips-row">
          {genres.map((g) => <button key={g} className={`chip-btn ${genre === g ? "on" : ""}`} onClick={() => setGenre(g)}>{g}</button>)}
        </div>
      </div>
      {job.task && <div className="card card-pad" style={{ marginBottom: 16 }}><TaskProgress task={job.task} onStop={job.stop} /></div>}

      <div className="section-label">Installed</div>
      <div className="pack-grid">
        {installed.map((p) => card(p, "installed"))}
        {packs.data && !installed.length && <div className="card card-pad empty small">{q || genre !== "All" ? "No installed packs match." : "No packs installed yet. Install one below or from a .narrator-pack file."}</div>}
      </div>

      <div className="section-label" style={{ marginTop: 24 }}>Available</div>
      <div className="pack-grid">
        {catalog.map((p) => card(p, "catalog"))}
        {packs.data && !catalog.length && <div className="card card-pad empty small">{q || genre !== "All" ? "No packs match." : "No packs to install yet. Packs that come with the app will show up here, free to use."}</div>}
      </div>
      {packs.error && <div className="card card-pad empty">{packs.error}</div>}
    </div>
  );
}
