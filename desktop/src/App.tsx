import { useEffect, useState } from "react";
import { AlertTriangle, AudioLines, BookOpen, FileText, FolderOpen, GraduationCap, History, Loader2, Mic, Package, RotateCw, Settings as SettingsIcon, Settings2, Waves, X } from "lucide-react";
import { api, engineReady, onNewKey, updateApi, useLoad, type Usage } from "./api";
import { Speech } from "./pages/Speech";
import { Transcribe } from "./pages/Transcribe";
import { Sounds } from "./pages/Sounds";
import { Audiobook } from "./pages/Audiobook";
import { BookEditor } from "./pages/book/BookEditor";
import { Voices } from "./pages/Voices";
import { Library } from "./pages/Library";
import { Models } from "./pages/Models";
import { useConnIssues } from "./pages/ConnIssues";
import { Activity } from "./pages/Activity";
import { Train } from "./pages/Train";
import { Settings, type SettingsTab } from "./pages/Settings";
import { BookJobsHost, useBookJobs } from "./bookJobs";
import { CommandPalette } from "./components/CommandPalette";
import { engineStatus, isMac, isMacDesktop, openLogFolder, retryEngine, type EngineStatus } from "./native";

export const APP_NAME = "Babble";

export type PageId = "speech" | "transcribe" | "book" | "voices" | "sounds" | "library" | "train" | "activity" | "models" | "settings";
const CREATE_PAGES: { id: PageId; label: string; icon: typeof Mic }[] = [
  { id: "speech", label: "Text to Speech", icon: AudioLines },
  { id: "transcribe", label: "Speech to Text", icon: FileText },
  { id: "book", label: "Audiobooks", icon: BookOpen },
];
const STUDIO_PAGES: { id: PageId; label: string; icon: typeof Mic }[] = [
  { id: "voices", label: "Voices", icon: Mic },
  { id: "sounds", label: "Sounds", icon: Waves },
  { id: "library", label: "Library", icon: Package },
  { id: "train", label: "Train a voice", icon: GraduationCap },
];
const PAGE_IDS: PageId[] = [...CREATE_PAGES.map((p) => p.id), ...STUDIO_PAGES.map((p) => p.id), "activity", "models", "settings"];

const stored = (k: string) => { try { return localStorage.getItem(k); } catch { return null; } };
const store = (k: string, v: string | null) => { try { if (v == null) localStorage.removeItem(k); else localStorage.setItem(k, v); } catch { /* private window */ } };

export default function App() {
  const [up, setUp] = useState<{ ram_gb: number } | null>(null);
  const [, redraw] = useState(0);
  // the engine started again by itself: draw everything again, so audio and images on screen carry its new key
  useEffect(() => onNewKey(() => redraw((n) => n + 1)), []);
  useEffect(() => { // wait for the engine, then load its run token before rendering Shell
    let live = true;
    let timer: number | null = null;
    const check = async () => {
      try {
        const h = await api.health();
        await engineReady();
        if (live) setUp(h);
      } catch {
        if (live) timer = window.setTimeout(check, 1000);
      }
    };
    check();
    return () => { live = false; if (timer !== null) clearTimeout(timer); };
  }, []);
  if (!up) return <Splash />;
  return <BookJobsHost><Shell ramGb={up.ram_gb} /><Missing /></BookJobsHost>;
}

/** When the app opens: anything Babble needs that's missing, by what it affects (and the technical thing). */
function Missing() {
  const [problems, setProblems] = useState<{ area: string; missing: string }[]>([]);
  useEffect(() => { api.checks().then((r) => setProblems(r.problems)).catch(() => {}); }, []);
  if (!problems.length) return null;
  return (
    <div className="missing" role="alert">
      <AlertTriangle size={16} className="missing-icon" />
      <div className="stack" style={{ gap: 6, minWidth: 0, flex: 1 }}>
        <strong>Some parts of {APP_NAME} won't work</strong>
        <ul>{problems.map((p) => <li key={p.area + p.missing}>{p.area} <span className="muted">({p.missing})</span></li>)}</ul>
      </div>
      <button className="btn icon sm ghost" onClick={() => setProblems([])} aria-label="Close" title="Close"><X size={14} /></button>
    </div>
  );
}

function Splash() {
  const [secs, setSecs] = useState(0);
  const [st, setSt] = useState<EngineStatus | null>(null);
  useEffect(() => { const t = setInterval(() => setSecs((s) => s + 1), 1000); return () => clearInterval(t); }, []);
  useEffect(() => {
    let live = true;
    const poll = () => engineStatus().then((x) => live && x && setSt(x));
    poll();
    const t = setInterval(poll, 2000);
    return () => { live = false; clearInterval(t); };
  }, []);
  const failed = st?.state === "failed";
  const phase = st?.phase || "Starting the voice engine…";
  const bootstrapping = Boolean(st?.phase && st.phase !== "Starting the voice engine…");
  const slow = secs >= (bootstrapping ? 240 : 30);
  return (
    <div className="splash" data-tauri-drag-region>
      <div style={{ maxWidth: 460 }}>
        <div className="brand-mark"><AudioLines size={28} /></div>
        <h2 style={{ margin: "0 0 6px" }}>{APP_NAME}</h2>
        {failed ? (
          <div className="stack" style={{ gap: 10, alignItems: "center", marginTop: 8 }}>
            <p className="row" style={{ justifyContent: "center", color: "var(--danger-text, #ef4444)" }}>
              <AlertTriangle size={15} />{st?.error || "Couldn't start the voice engine."}
            </p>
            {st?.log_tail && (
              <pre className="small muted" style={{ textAlign: "left", maxWidth: 440, maxHeight: 160, overflow: "auto", padding: 10, borderRadius: 8, background: "var(--surface-2, rgba(255,255,255,0.05))", whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
                {st.log_tail}
              </pre>
            )}
            <div className="row" style={{ gap: 8, justifyContent: "center" }}>
              <button className="btn sm primary" onClick={() => { setSecs(0); setSt((prev) => prev ? { ...prev, state: "starting", error: null, phase: "Retrying setup…" } : prev); retryEngine().catch(() => {}); }}>
                <RotateCw size={14} />Retry setup
              </button>
              <button className="btn sm" onClick={() => openLogFolder().catch(() => {})}>
                <FolderOpen size={14} />Open log folder
              </button>
            </div>
          </div>
        ) : (
          <>
            <p className="muted row" style={{ justifyContent: "center" }}><Loader2 size={15} className="spin" />{phase}{secs >= 5 ? ` ${secs} s` : ""}</p>
            {bootstrapping && !slow && <p className="small muted">First-time setup downloads Python and audio tools automatically (about 1–2 minutes).</p>}
            {!bootstrapping && secs >= 10 && !slow && <p className="small muted">It usually takes 10–30 seconds, longer the first time after an update.</p>}
            {slow && (
              <div className="stack" style={{ gap: 8, alignItems: "center" }}>
                <p className="small muted" style={{ maxWidth: 380 }}>This is taking longer than usual. Check the end of
                  {st?.log_path ? ` ${st.log_path}` : (isMac() ? " ~/Library/Logs/Babble.log" : " %APPDATA%\\Babble\\logs\\Babble.log")}.</p>
                {st?.log_tail && (
                  <pre className="small muted" style={{ textAlign: "left", maxWidth: 440, maxHeight: 140, overflow: "auto", padding: 10, borderRadius: 8, background: "var(--surface-2, rgba(255,255,255,0.05))", whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
                    {st.log_tail}
                  </pre>
                )}
                <button className="btn sm" onClick={() => openLogFolder().catch(() => {})}>
                  <FolderOpen size={14} />Open log folder
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

const gib = (n: number) => `${(n / 2 ** 30).toFixed(1)} GB`;  // (memory: as the Mac counts it, 8 GB)
const gb = (n: number) => `${n >= 1e11 ? Math.round(n / 1e9) : (n / 1e9).toFixed(1)} GB`;  // (disk: as Finder counts it)

/** How much memory and disk Babble is using, above the engine's status. */
/** A line under the sidebar when a new version is out or downloaded (Settings → General → Updates). Looked for
 *  when Babble opens and every 10 minutes (the engine asks GitHub at most that often). */
function UpdateNote({ onOpen }: { onOpen: () => void }) {
  const [note, setNote] = useState("");
  useEffect(() => {
    const look = () => updateApi.status().then((u) => setNote(
      u.ready ? `Babble ${u.latest?.version} is ready` : u.available && !u.auto ? `Babble ${u.latest?.version} is out` : "")).catch(() => {});
    const first = window.setTimeout(look, 8000);
    const every = window.setInterval(look, 10 * 60 * 1000);
    return () => { window.clearTimeout(first); window.clearInterval(every); };
  }, []);
  if (!note) return null;
  return <button className="link small accent-link" style={{ textAlign: "left" }} onClick={onOpen}>{note}</button>;
}

function Meters() {
  const [u, setU] = useState<Usage | null>(null);
  useEffect(() => {
    let live = true;
    const tick = () => api.usage().then((x) => live && setU(x)).catch(() => {});
    tick();
    const t = setInterval(tick, 5000);
    return () => { live = false; clearInterval(t); };
  }, []);
  if (!u) return null;
  const ram = u.ram.total ? Math.min(u.ram.used / u.ram.total, 1) : 0;
  const s = u.storage;
  return (
    <div className="meters">
      <div className="usage-meter" title="Memory Babble is using now (the voice engine and the app), of this Mac's memory">
        <div className="usage-row"><span>RAM</span><span>{gib(u.ram.used)} of {gib(u.ram.total)}</span></div>
        <div className={`usage-bar ${ram > 0.85 ? "high" : ""}`}><i style={{ width: `${ram * 100}%` }} /></div>
      </div>
      {s && (
        <div className="usage-meter" title={`Babble's files: ${gb(s.data)} of books, voices and sounds, ${gb(s.models)} of downloaded models. ${gb(s.free)} free on this disk.`}>
          <div className="usage-row"><span>Storage</span><span>{gb(s.used)} used · {gb(s.free)} free</span></div>
          <div className={`usage-bar ${s.free < 10e9 ? "high" : ""}`}><i style={{ width: `${Math.min(s.used / s.total, 1) * 100}%` }} /></div>
        </div>
      )}
    </div>
  );
}

function Shell({ ramGb }: { ramGb: number }) {
  const problems = useConnIssues(60000).list.length;  // (a badge on Settings while a service has a problem)
  const [page, setPageState] = useState<PageId>(() => {
    const p = stored("page");
    if (p === "connections") return "settings";
    return p && PAGE_IDS.includes(p as PageId) ? (p as PageId) : "speech";
  });
  const [settingsTab, setSettingsTab] = useState<SettingsTab | undefined>(() =>
    stored("page") === "connections" ? "connections" : undefined,
  );
  const [connSection, setConnSection] = useState<string | undefined>(undefined);
  const [settingsTabNonce, setSettingsTabNonce] = useState(0);
  const [book, setBookState] = useState<string | null>(() => stored("open-book"));
  const setPage = (p: PageId) => { setPageState(p); store("page", p); };
  const openConnections = (section?: string) => {
    setSettingsTab("connections");
    setConnSection(section);
    setSettingsTabNonce((n) => n + 1);
    setPage("settings");
  };
  const openBook = (id: string | null) => { setBookState(id); store("open-book", id); if (id) setPage("book"); };
  const voices = useLoad(api.voices);
  const engines = useLoad(api.engines);
  const { jobs } = useBookJobs();
  const making = Object.values(jobs).some((t) => t.state === "running");
  const reloadVoices = () => { voices.reload(); engines.reload(); };

  const palette = <CommandPalette onPage={setPage} onOpenBook={(id) => openBook(id)} onConnections={openConnections} />;

  if (page === "book" && book) { // a book's own tools replace the app's sidebar, like the editor of one book
    return (
      <div className="shell book-mode">
        {isMacDesktop() && <div className="drag-strip" data-tauri-drag-region />}
        <BookEditor key={book} id={book} voices={voices.data || []} engines={engines.data || []} ramGb={ramGb}
          onClose={() => openBook(null)} />
        {palette}
      </div>
    );
  }

  return (
    <div className="shell">
      {isMacDesktop() && <div className="drag-strip" data-tauri-drag-region />}
      {palette}
      <nav className="sidebar" aria-label="Sections">
        <div className="brand" data-tauri-drag-region>
          <div className="brand-mark"><AudioLines size={17} /></div>
          <span className="brand-name">{APP_NAME}</span>
        </div>
        <div className="nav-section">Create</div>
        {CREATE_PAGES.map((p) => (
          <button key={p.id} className={`nav-item ${page === p.id ? "active" : ""}`} onClick={() => setPage(p.id)}
            aria-current={page === p.id ? "page" : undefined}>
            <p.icon size={17} />{p.label}
          </button>
        ))}
        <div className="nav-section">Studio</div>
        {STUDIO_PAGES.map((p) => (
          <button key={p.id} className={`nav-item ${page === p.id ? "active" : ""}`} onClick={() => setPage(p.id)}
            aria-current={page === p.id ? "page" : undefined}>
            <p.icon size={17} />{p.label}
          </button>
        ))}
        <div className="nav-section nav-bottom">System</div>
        <button className={`nav-item ${page === "activity" ? "active" : ""}`} onClick={() => setPage("activity")}
          aria-current={page === "activity" ? "page" : undefined}>
          <History size={17} />Activity
        </button>
        <button className={`nav-item ${page === "models" ? "active" : ""}`} onClick={() => setPage("models")}
          aria-current={page === "models" ? "page" : undefined}>
          <Settings2 size={17} />Models
        </button>
        <button className={`nav-item ${page === "settings" ? "active" : ""}`} onClick={() => setPage("settings")}
          aria-current={page === "settings" ? "page" : undefined}>
          <SettingsIcon size={17} />Settings
          {problems > 0 && <span className="nav-badge" title={`${problems} connection problem${problems === 1 ? "" : "s"}`}>{problems}</span>}
        </button>
        <div className="sidebar-foot">
          <UpdateNote onOpen={() => { setSettingsTab("general"); setSettingsTabNonce((n) => n + 1); setPage("settings"); }} />
          <Meters />
          <span><span className={`dot ${making ? "busy" : ""}`} />{making ? "Making an audiobook…" : `Engine ready · ${ramGb} GB ${isMac() ? "Mac" : "RAM"}`}</span>
        </div>
      </nav>
      <main className="main">
        {page === "speech" && <Speech voices={voices.data || []} engines={engines.data || []} />}
        {page === "transcribe" && (
          <Transcribe
            onSendToSpeech={(text) => {
              try {
                const prev = JSON.parse(localStorage.getItem("speech-form") || "{}");
                localStorage.setItem("speech-form", JSON.stringify({ ...prev, text }));
              } catch { /* ignore */ }
              setPage("speech");
            }}
            onOpenBook={(id) => openBook(id)}
          />
        )}
        {page === "book" && <Audiobook onOpen={(id) => openBook(id)} />}
        {page === "voices" && <Voices voices={voices.data || []} engines={engines.data || []} onSaved={reloadVoices} />}
        {page === "sounds" && <Sounds ramGb={ramGb} />}
        {page === "library" && <Library onChanged={reloadVoices} />}
        {page === "train" && <Train onChanged={reloadVoices} goConnections={() => openConnections("gpus")} />}
        {page === "activity" && <Activity />}
        {page === "models" && <Models onChanged={reloadVoices} goConnections={() => openConnections("speech")} />}
        {page === "settings" && <Settings initialTab={settingsTab} focusGroup={connSection} tabNonce={settingsTabNonce} />}
      </main>
    </div>
  );
}
