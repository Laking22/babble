import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { ArrowDown, ArrowLeft, ArrowUp, AudioLines, Check, ChevronDown, Download, ExternalLink, FileText, Merge, Package, Play, Plus, RefreshCw, Search, Square, Trash2, Upload, Wand2, Wrench, X } from "lucide-react";
import {
  connectionApi, fileUrl, libraryApi, signingApi, trainApi, useLoad, useTask,
  type Check as CheckResult, type PrepareStatus, type PreparedVoice, type Signing, type TrainedModel,
} from "../api";
import { Field, Orb, Slider, Starting, StopButton, Switch, TaskProgress, accepts, duration, timeLeft, useEta, useFileDrop, useStarting, useToast } from "../components/ui";
import { openExternal, saveFile } from "../native";
import { useCostConfirm } from "../components/CostConfirm";

type View = { at: "home" } | { at: "choose" } | { at: "prepare"; recipe: string; addTo: TrainedModel | null; voice?: string }
  | { at: "train"; recipe: string; addTo: TrainedModel | null; voices: string[] } | { at: "export" };
const MAX_VOICES = 8;  // voices in one model
const RECIPE_NAME: Record<string, string> = { orpheus: "Orpheus 3B", qwen: "Qwen3-TTS 1.7B" };

/** Models you trained, each with its voices (up to 8); Create trains a new voice into one, or a new model. */
export function Train({ onChanged, goConnections }: { onChanged: () => void; goConnections: () => void }) {
  const [view, setView] = useState<View>({ at: "home" });
  const models = useLoad(trainApi.models);
  const runs = useLoad(trainApi.runs);
  const home = () => { setView({ at: "home" }); models.reload(); runs.reload(); };
  if (view.at === "choose") return <Choose back={home} next={(recipe) => setView({ at: "prepare", recipe, addTo: null })} />;
  if (view.at === "prepare") return <Prepare key={view.voice || ""} view={view} back={home} next={(voices) => setView({ at: "train", recipe: view.recipe, addTo: view.addTo, voices })} goConnections={goConnections} />;
  if (view.at === "train") return <TrainStep view={view} back={() => setView({ at: "prepare", recipe: view.recipe, addTo: view.addTo })} home={home} goConnections={goConnections} />;
  if (view.at === "export") return <Export models={models.data || []} back={home} />;
  return <Home models={models} runs={runs} onChanged={onChanged} create={() => setView({ at: "choose" })}
    addTo={(m) => setView({ at: "prepare", recipe: m.recipe, addTo: m })} exportPack={() => setView({ at: "export" })}
    openPrepared={(voice) => setView({ at: "prepare", recipe: "qwen", addTo: null, voice })} />;
}

function Home({ models, runs, onChanged, create, addTo, exportPack, openPrepared }: {
  models: ReturnType<typeof useLoad<TrainedModel[]>>; runs: ReturnType<typeof useLoad<import("../api").Run[]>>; onChanged: () => void;
  create: () => void; addTo: (m: TrainedModel) => void; exportPack: () => void; openPrepared: (voice: string) => void;
}) {
  const toast = useToast();
  const prepared = useLoad(trainApi.voices);
  const preparing = (prepared.data || []).filter((v) => v.running);
  useEffect(() => { // what's training or transcribing: kept current (a voice can carry on, or a run end, any time)
    const t = setInterval(() => { prepared.reload(); runs.reload(); }, 5000);
    return () => clearInterval(t);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  const [cancelling, setCancelling] = useState<string | null>(null); // the run whose "Cancel it?" is showing
  const stopPreparing = async (voice: string) => {
    try { await trainApi.stop(voice); toast(`Stopping ${voice} after the current step. Carry on any time from its page.`); prepared.reload(); }
    catch (e) { toast((e as Error).message, true); }
  };
  const cancelRun = async (key: string) => {
    try { await trainApi.cancelRun(key); toast(`Cancelled ${key.split("/")[1]}.`); setCancelling(null); runs.reload(); }
    catch (e) { toast((e as Error).message, true); setCancelling(null); }
  };
  const [confirm, setConfirm] = useState<string | null>(null);
  const [checking, setChecking] = useState<string | null>(null);
  const install = useTask((t) => {
    if (t.state === "failed") toast(t.error || "Couldn't install it.", true);
    else if (t.state === "done") { toast("Installed. Its voices are in every voice list."); models.reload(); runs.reload(); onChanged(); }
  });
  const remove = async (m: TrainedModel) => {
    try { await trainApi.removeModel(m.folder); toast(`Deleted ${m.title}.`); setConfirm(null); models.reload(); onChanged(); }
    catch (e) { toast((e as Error).message, true); }
  };
  const check = async (key: string) => {
    setChecking(key);
    try { const r = await trainApi.check(key); toast(`${key.split("/")[1]}: ${r.label}${r.detail ? ` · ${r.detail}` : ""}`, r.state === "error"); runs.reload(); }
    catch (e) { toast((e as Error).message, true); }
    finally { setChecking(null); }
  };
  const again = useTask((t) => {
    if (t.state === "failed") toast(t.error || "Couldn't send it again.", true);
    else if (t.state === "done") toast("Sent again. Check its status here in a few minutes.");
    runs.reload();
  });
  const confirmCost = useCostConfirm();
  const retrying = useStarting();
  const sendAgain = async (key: string) => {
    try {
      const e = await trainApi.retryEstimate(key);
      if (!["local-mac", "ssh"].includes(e.backend_id)) {  // a cloud GPU: money, or Kaggle's weekly hours
        const ok = await confirmCost({ title: `Train on ${e.backend} again?`, confirm: "Train again",
          cost: e.paid ? e.cost || "price unknown" : e.backend_id === "kaggle" ? `free: about ${e.gpu_hours} of your weekly Kaggle GPU hours` : e.cost || "price unknown",
          free: !e.paid, details: [e.text] });
        if (!ok) return;
      }
    } catch (x) { return toast((x as Error).message, true); }
    again.start(() => trainApi.retry(key));
  };
  const forget = async (key: string) => {
    try { await trainApi.forget(key); runs.reload(); } catch (e) { toast((e as Error).message, true); }
  };
  const going = (runs.data || []).filter((r) => !["installed", "packaged", "deleted", "stopped"].includes(r.state));
  return (
    <div className="page">
      <div className="page-head">
        <div><h1>Train a voice</h1><p>Turn an audiobook into a voice of your own, trained on Kaggle's free GPUs or your own machine.</p>
          <p className="small muted" style={{ marginTop: 6 }}><b>Best results:</b> 1–2 hours of one narrator on clean audio (no music), in the style you want the voice to have.<br />
            Past about 4 hours per voice, more audio adds little. More books in the right style help more than more hours.</p></div>
        <div className="row">
          <button className="btn" onClick={exportPack} disabled={!models.data?.length}><Package size={15} />Export pack</button>
          <button className="btn primary" onClick={create}><Plus size={15} />Create</button>
        </div>
      </div>
      <TrainingTools />
      {models.error && <div className="card card-pad empty">{models.error}</div>}
      <div className="pills">
        {(models.data || []).map((m) => (
          <div key={m.folder} className="model-pill">
            <div className="row" style={{ alignItems: "flex-start" }}>
              <div className="grow">
                <h3 style={{ margin: 0 }}>{m.title}</h3>
                <div className="small muted">{RECIPE_NAME[m.recipe] || m.engine} · {m.size_gb} GB{m.hours ? ` · ${m.hours} h of audio` : ""}</div>
              </div>
              <button className="btn sm danger" onClick={() => setConfirm(m.folder)}><Trash2 size={13} />Delete</button>
              <button className="btn sm primary" disabled={!m.can_add} title={m.can_add ? `Train a new voice into ${m.title}` : m.why} onClick={() => addTo(m)}><Plus size={13} />Create</button>
            </div>
            <div className="meter" aria-label={`${m.voices.length} of ${m.max_voices} voices`}>
              <div className="bar"><i style={{ width: `${(m.voices.length / m.max_voices) * 100}%` }} /></div>
              <span className="small muted">{m.voices.length} of {m.max_voices} voices</span>
            </div>
            <div className="pill-voices">
              {m.voices.map((v) => (
                <span key={v.speaker} className="pill-voice"><Orb value={v.value || v.speaker} size={22} />{v.name}</span>
              ))}
            </div>
            {!m.can_add && m.why && <div className="small faint">{m.why}</div>}
            {confirm === m.folder && (
              <div className="confirm" role="alertdialog" aria-label={`Delete ${m.title}?`}>
                <span className="small">Delete {m.title} from this Mac? Its {m.voices.length} voice{m.voices.length === 1 ? "" : "s"} leave every voice list. This can't be undone.</span>
                <div className="row" style={{ gap: 6 }}>
                  <button className="btn sm ghost" onClick={() => setConfirm(null)}>Cancel</button>
                  <button className="btn sm danger-solid" onClick={() => remove(m)}><Trash2 size={13} />Delete</button>
                </div>
              </div>
            )}
          </div>
        ))}
        {models.data && !models.data.length && (
          <div className="card card-pad empty">
            <p style={{ marginTop: 0 }}>No trained models yet.</p>
            <button className="btn primary" onClick={create}><Plus size={14} />Create your first</button>
          </div>
        )}
      </div>

      {(going.length > 0 || preparing.length > 0) && (
        <div className="card" style={{ marginTop: 24 }}>
          <div className="card-pad" style={{ paddingBottom: 6 }}><h3 className="card-title" style={{ margin: 0 }}>Training now</h3></div>
          <div className="list">
            {preparing.map((v) => (
              <div key={`prep-${v.name}`} className="list-item">
                <div className="grow">
                  <div className="title">{v.name} <span className="chip">{v.stage || "Preparing"}{v.progress != null ? ` · ${Math.round(v.progress * 100)}%` : ""}</span></div>
                  <div className="small muted">Preparing training data on {v.where || "this Mac"}</div>
                </div>
                <button className="btn sm" onClick={() => openPrepared(v.name)}>Open</button>
                <button className="btn sm danger" onClick={() => stopPreparing(v.name)}><Square size={12} />Cancel</button>
              </div>
            ))}
            {going.map((r) => (
              <div key={r.key} className="list-item">
                <div className="grow">
                  <div className="title">{r.model} <span className={`chip ${r.state === "error" ? "warn" : r.state === "complete" ? "ok" : ""}`}>{r.label}</span></div>
                  <div className="small muted">{r.voices.join(", ")} · {r.recipe} on {r.where}{r.sent ? ` · sent ${r.sent}` : ""}
                    {["running", "queued", "sent"].includes(r.state) ? runFor(r.sent, r.gpu_hours) : ""}</div>
                  {r.state === "error" && r.error && <div className="small error-text" style={{ marginTop: 4 }}>{r.error}</div>}
                  {r.state === "error" && r.can_retry && (
                    <div className="small muted" style={{ marginTop: 2 }}>Train again sends it with the same voices and the current training settings.</div>
                  )}
                </div>
                {r.url && <button className="btn sm ghost" onClick={() => openExternal(r.url)}><ExternalLink size={13} />Open run</button>}
                {r.state === "error" ? (
                  <>
                    <button className="btn sm ghost" onClick={() => forget(r.key)} title="Take it off this list">Remove</button>
                    <button className="btn sm" disabled={checking === r.key} onClick={() => check(r.key)}
                      title="Ask again how the run went (a network problem can make a finished run look failed)">
                      <RefreshCw size={13} className={checking === r.key ? "spin" : ""} />Check status</button>
                    {r.can_retry && <button className="btn sm primary" disabled={again.running || retrying.starting} onClick={() => retrying.run(() => sendAgain(r.key))}>
                      {retrying.starting ? <Starting size={13} /> : <><RefreshCw size={13} />Train again</>}</button>}
                  </>
                ) : (
                  <button className="btn sm" disabled={checking === r.key} onClick={() => check(r.key)}><RefreshCw size={13} className={checking === r.key ? "spin" : ""} />Check status</button>
                )}
                {r.state === "complete" && <button className="btn sm primary" disabled={install.running} onClick={() => install.start(() => trainApi.finish(r.key))}>Finish &amp; install</button>}
                {["running", "queued", "sent", "unknown"].includes(r.state) && (cancelling === r.key ? (
                  <>
                    <span className="small">Cancel this run? What it trained so far is lost.</span>
                    <button className="btn sm ghost" onClick={() => setCancelling(null)}>Keep it</button>
                    <button className="btn sm danger-solid" onClick={() => cancelRun(r.key)}>Cancel it</button>
                  </>
                ) : <button className="btn sm danger" onClick={() => setCancelling(r.key)}><Square size={12} />Cancel</button>)}
              </div>
            ))}
          </div>
          {install.task && <div className="card-pad"><TaskProgress task={install.task} onStop={install.stop} /></div>}
          {again.task && again.task.state !== "done" && <div className="card-pad"><TaskProgress task={again.task} onStop={again.stop} /></div>}
        </div>
      )}
    </div>
  );
}

// ---------- training tools: merging files before a voice is prepared ----------

const natural = (a: string, b: string) => a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });
const TEXT_ACCEPT = ".txt,.md,.epub,.pdf,.docx,.mobi,.azw,.azw3,.fb2,.rtf,.odt,.html,.htm";
const AUDIO_ACCEPT = "audio/*,.m4b,.m4a,.mp3,.flac,.wav,.ogg,.opus,.aac,.aif,.aiff,.wma";
const PREP_AUDIO = "audio/*,.m4b,.m4a,.mp3,.flac,.wav,.ogg,.opus,.zip";
const PREP_TEXT = ".epub,.pdf,.txt,.md,.docx,.mobi,.azw,.azw3,.fb2,.rtf,.odt,.html,.htm";

/** A fold-out under the page title: tools that get files ready for Prepare a voice. Remembers if it's open. */
function TrainingTools() {
  const [open, setOpen] = useState(() => { try { return localStorage.getItem("train-tools") === "open"; } catch { return false; } });
  const toggle = () => { setOpen(!open); try { localStorage.setItem("train-tools", open ? "closed" : "open"); } catch { /* fine */ } };
  return (
    <div className={`tools-fold ${open ? "open" : ""}`}>
      <button className="tools-toggle" onClick={toggle} aria-expanded={open}>
        <Wrench size={15} /><b>Training tools</b>
        <span className="small muted">Merge text or audio files before you prepare a voice</span>
        <span className="spacer" /><ChevronDown size={17} className="chev" />
      </button>
      {open && (
        <div className="tools-grid">
          <Merger kind="text" />
          <Merger kind="audio" />
        </div>
      )}
    </div>
  );
}

/** Pick files, put them in order, merge: one text file from books in any format, or one recording from audio files. */
function Merger({ kind }: { kind: "text" | "audio" }) {
  const toast = useToast();
  const input = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [name, setName] = useState("");
  const [headings, setHeadings] = useState(true);
  const [format, setFormat] = useState("flac");
  const job = useTask((t) => { if (t.state === "failed") toast(t.error || "Couldn't merge them.", true); });
  const add = (list: FileList | File[] | null) => {
    if (!list?.length) return;
    const more = [...list].filter((f) => !files.some((x) => x.name === f.name && x.size === f.size));
    setFiles([...files, ...more].sort((a, b) => natural(a.name, b.name)));
  };
  const drop = useFileDrop((got) => {
    const fit = got.filter((f) => accepts(f, text ? TEXT_ACCEPT : AUDIO_ACCEPT));
    if (!fit.length) return toast(text ? "Those aren't text files." : "Those aren't audio files.", true);
    add(fit);
  });
  const move = (i: number, by: number) => {
    const next = [...files]; const [f] = next.splice(i, 1); next.splice(i + by, 0, f); setFiles(next);
  };
  const result = job.task?.state === "done" ? job.task.result : null;
  const text = kind === "text";
  return (
    <div className={`card card-pad merger${drop.over ? " file-over" : ""}`} {...drop.props}>
      <div className="row" style={{ gap: 10 }}>
        <div className={`kind-icon ${text ? "effect" : "ambience"}`}>{text ? <FileText size={16} /> : <AudioLines size={16} />}</div>
        <div className="grow"><b>{text ? "Merge text files" : "Merge audio files"}</b>
          <div className="small muted">{text ? "Books and text in any format (EPUB, PDF, Word, TXT…) into one text file."
            : "Audio files of any format into one recording, end to end."}</div></div>
      </div>
      <input ref={input} type="file" hidden multiple accept={text ? TEXT_ACCEPT : AUDIO_ACCEPT}
        onChange={(e) => { add(e.target.files); e.target.value = ""; }} />
      <div className="merge-list">
        {files.map((f, i) => (
          <div key={`${f.name}-${f.size}`} className="merge-row">
            <span className="merge-n small faint">{i + 1}</span>
            <span className="grow small merge-name" title={f.name}>{f.name}</span>
            <span className="small faint">{f.size > 1e6 ? `${(f.size / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(f.size / 1e3))} KB`}</span>
            <button className="btn icon sm ghost" disabled={i === 0} onClick={() => move(i, -1)} aria-label={`Move ${f.name} up`} title="Move up"><ArrowUp size={13} /></button>
            <button className="btn icon sm ghost" disabled={i === files.length - 1} onClick={() => move(i, 1)} aria-label={`Move ${f.name} down`} title="Move down"><ArrowDown size={13} /></button>
            <button className="btn icon sm ghost" onClick={() => setFiles(files.filter((_, k) => k !== i))} aria-label={`Remove ${f.name}`} title="Remove"><X size={13} /></button>
          </div>
        ))}
        <button className="merge-add" onClick={() => input.current?.click()}><Plus size={14} />{files.length ? "Add more files" : "Add files"}</button>
      </div>
      {files.length > 1 && <p className="small faint" style={{ margin: 0 }}>Sorted by name (Part 2 before Part 10). They're joined in this order, so it must match the audiobook's.</p>}
      <div className="row" style={{ gap: 10, alignItems: "flex-end" }}>
        <Field label="Name of the merged file">
          <input className="input" placeholder={text ? "Merged text" : "Merged audio"} value={name} onChange={(e) => setName(e.target.value)} />
        </Field>
        {!text && (
          <Field label="Format">
            <select className="select" value={format} onChange={(e) => setFormat(e.target.value)} style={{ width: 150 }}>
              <option value="flac">FLAC (lossless)</option>
              <option value="m4a">M4A (smaller)</option>
              <option value="mp3">MP3</option>
              <option value="wav">WAV</option>
            </select>
          </Field>
        )}
      </div>
      {text && <Switch label="Keep chapter titles" hint="As lines of their own, the way a narrator reads them. Off: they're left out." checked={headings} onChange={setHeadings} />}
      <div className="row">
        <span className="spacer" />
        <button className="btn primary" disabled={files.length < 2 || job.running}
          onClick={() => job.start(() => (text ? trainApi.mergeText(files, name, headings) : trainApi.mergeAudio(files, name, format)))}>
          <Merge size={14} />Merge {files.length > 1 ? `${files.length} files` : ""}
        </button>
      </div>
      {job.task && job.task.state !== "done" && <TaskProgress task={job.task} onStop={job.stop} />}
      {result?.file && (
        <div className="made-note">
          <span className="small grow">{job.task!.status.replace(/^Done · /, "")}</span>
          <button className="btn sm" onClick={() => saveFile(result.file).then((m) => toast(m), (e) => toast(String(e), true))}><Download size={13} />Save</button>
          {!text && <audio src={fileUrl(result.file)} controls preload="none" style={{ width: "100%" }} />}
        </div>
      )}
    </div>
  );
}

/** Voices prepared before, picked one or several at a time (they're trained into one model together); the same
 *  list as the Director's model picker. */
function PreparedPicker({ voices, value, onChange, max }: {
  voices: PreparedVoice[]; value: string[]; onChange: (v: string[]) => void; max: number;
}) {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const anchor = useRef<HTMLButtonElement>(null);
  const [pos, setPos] = useState({ top: 0, left: 0, width: 320 });

  useEffect(() => {
    if (!open || !anchor.current) return;
    setQ("");
    const place = () => {
      if (!anchor.current) return;
      const r = anchor.current.getBoundingClientRect();
      const width = Math.min(Math.max(r.width, 320), window.innerWidth - 24);
      const maxH = Math.min(420, window.innerHeight - 24);
      setPos({
        top: Math.max(12, Math.min(r.bottom + 6, window.innerHeight - maxH - 12)),
        left: Math.max(12, Math.min(r.left, window.innerWidth - width - 12)),
        width,
      });
    };
    place();
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    return () => {
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", place, true);
    };
  }, [open]);

  const filtered = useMemo(() => {
    const words = q.toLowerCase().split(/\s+/).filter(Boolean);
    return words.length ? voices.filter((v) => words.every((w) => v.name.toLowerCase().includes(w))) : voices;
  }, [voices, q]);
  const toggle = (name: string) => onChange(value.includes(name) ? value.filter((x) => x !== name) : [...value, name]);
  const full = value.length >= max;
  const first = voices.find((v) => v.name === value[0]);

  return (
    <>
      <button ref={anchor} type="button" className="voice-btn dir-btn" style={{ width: 280 }} onClick={() => setOpen(true)}>
        <span className="kind-icon sm ambience"><AudioLines size={14} /></span>
        <span className="voice-meta">
          <strong>{!value.length ? "Pick voices…" : value.length === 1 ? value[0] : `${value.length} voices`}</strong>
          <span>{!value.length ? `${voices.length} prepared` : value.length === 1 && first
            ? `${first.clips.toLocaleString()} clips · ${first.hours} h` : value.join(", ")}</span>
        </span>
        <span className="spacer" />
        <ChevronDown size={15} className="muted" />
      </button>
      {open && createPortal(
        <>
          <div className="popover-backdrop" onClick={() => setOpen(false)} />
          <div className="popover dir-pop" style={{ top: pos.top, left: pos.left, width: pos.width, position: "fixed" }} role="listbox" aria-multiselectable="true">
            {voices.length > 5 && (
              <div className="popover-search">
                <div className="row" style={{ position: "relative" }}>
                  <Search size={15} style={{ position: "absolute", left: 10 }} className="muted" />
                  <input autoFocus className="input" style={{ paddingLeft: 32 }} placeholder="Search voices" value={q}
                    onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Escape" && setOpen(false)} />
                </div>
              </div>
            )}
            <div className="dir-pop-head">
              <span>Voice</span>
              <span>{max > 1 ? `Up to ${max} in one model` : "One more fits this model"}</span>
            </div>
            <div className="popover-list">
              {filtered.map((v) => {
                const sel = value.includes(v.name);
                return (
                  <button key={v.name} type="button" className={`voice-opt dir-opt ${sel ? "sel" : ""}`} role="option" aria-selected={sel}
                    disabled={!sel && full} onClick={() => toggle(v.name)}>
                    <Orb value={v.name} size={22} />
                    <span className="voice-meta">
                      <strong>{v.name}</strong>
                      <span>{v.clips.toLocaleString()} clips · {v.hours} h</span>
                    </span>
                    <span className="spacer" />
                    {sel && <Check size={14} className="dir-check" />}
                  </button>
                );
              })}
              {!filtered.length && <div className="empty small">No voices match “{q}”.</div>}
            </div>
            <div className="popover-foot row">
              <span className="small muted grow">{value.length ? `${value.length} picked` : "Pick one or more"}</span>
              {value.length > 0 && <button type="button" className="btn sm ghost" onClick={() => onChange([])}>Clear</button>}
              <button type="button" className="btn sm primary" onClick={() => setOpen(false)}>Done</button>
            </div>
          </div>
        </>,
        document.body,
      )}
    </>
  );
}

function Back({ onClick, label }: { onClick: () => void; label: string }) {
  return <button className="link back" onClick={onClick}><ArrowLeft size={15} />{label}</button>;
}

/** " · going 1 h 20 min (about 3 h expected)" for a run out in the cloud: its own progress isn't reported. */
function runFor(sent?: string, gpuHours?: number | null) {
  const at = sent ? new Date(sent.replace(" ", "T")).getTime() : NaN;
  if (!isFinite(at)) return "";
  const s = (Date.now() - at) / 1000;
  return ` · going ${duration(s)}${gpuHours ? ` (about ${duration(gpuHours * 3600)} expected${s > gpuHours * 3600 * 1.5 ? ": taking longer than expected, check it on its page" : ""})` : ""}`;
}

function Choose({ back, next }: { back: () => void; next: (recipe: string) => void }) {
  const options = useLoad(trainApi.options);
  const [pick, setPick] = useState("qwen");
  return (
    <div className="page narrow">
      <Back onClick={back} label="All models" />
      <div className="page-head"><div><h1>Which model should it train?</h1><p>A model holds up to 8 voices, trained together. You can add more later.</p></div></div>
      <div className="stack">
        {(options.data?.recipes || []).filter((r) => r.ready).sort((a, b) => Number(b.id === "qwen") - Number(a.id === "qwen")).map((r) => (
          <label key={r.id} className={`choice ${pick === r.id ? "sel" : ""}`}>
            <input type="radio" name="recipe" checked={pick === r.id} onChange={() => setPick(r.id)} style={{ accentColor: "var(--accent)" }} />
            <span className="grow">
              <b>{r.name}</b>{r.id === "qwen" && <span className="chip accent" style={{ marginLeft: 8 }}>Recommended</span>}
              <span className="small muted" style={{ display: "block" }}>{r.note} Up to {r.max_voices} voices.</span>
            </span>
          </label>
        ))}
        <div className="row"><span className="spacer" /><button className="btn primary" onClick={() => next(pick)}>Continue</button></div>
      </div>
    </div>
  );
}

function Steps({ at, recipe, addTo }: { at: 1 | 2; recipe: string; addTo: TrainedModel | null }) {
  return (
    <div className="row steps">
      <span className={`step-n ${at === 1 ? "on" : "done"}`}>{at === 1 ? "1" : <Check size={12} />}</span><b className="small">Prepare a voice</b>
      <span className="step-line" />
      <span className={`step-n ${at === 2 ? "on" : ""}`}>2</span><b className="small">Train</b>
      <span className="spacer" />
      <span className="small muted">{addTo ? `Goes into ${addTo.title} (${addTo.voices.length} of ${addTo.max_voices})` : `New ${RECIPE_NAME[recipe] || recipe} model`}</span>
    </div>
  );
}

function Prepare({ view, back, next, goConnections }: { view: Extract<View, { at: "prepare" }>; back: () => void; next: (voices: string[]) => void; goConnections: () => void }) {
  const toast = useToast();
  const confirmCost = useCostConfirm();
  const options = useLoad(trainApi.options);
  const prepared = useLoad(trainApi.voices);
  const [name, setName] = useState("");
  const [audio, setAudio] = useState<File[]>([]);
  const [bookFile, setBookFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [language, setLanguage] = useState("english");
  const [backend, setBackend] = useState("mac");
  const [cloudGpu, setCloudGpu] = useState("");
  const [check, setCheck] = useState<CheckResult | null>(null);
  const [checking, setChecking] = useState(false);
  const [opts, setOpts] = useState({ min_seconds: 2, max_seconds: 15, target_seconds: 8, min_match: 90, min_snr: 20 });
  const [split, setSplit] = useState(false);   // more than one narrator: sort the clips into a voice each
  const [dialogue, setDialogue] = useState(false);  // only the characters' lines: a voice that performs, not narrates
  const [howMany, setHowMany] = useState(0);   // 0: find out
  const [voice, setVoice] = useState<string | null>(view.voice ?? null);
  const [status, setStatus] = useState<PrepareStatus | null>(null);
  const [clips, setClips] = useState<{ file: string; text: string }[]>([]);
  const [sending, setSending] = useState(false);
  const [pollNonce, setPollNonce] = useState(0);
  const [earlier, setEarlier] = useState<string[]>([]);
  // files dropped on either box go where they belong: audio to the audiobook, a text file to the book's text
  const dropped = (got: File[]) => {
    const sound = got.filter((f) => accepts(f, PREP_AUDIO));
    const book = got.filter((f) => accepts(f, PREP_TEXT));
    if (!sound.length && !book.length) return toast("Those aren't audio or book files.", true);
    if (sound.length) setAudio(sound);
    if (book.length) setBookFile(book[0]);
    if (book.length > 1) toast(`Used ${book[0].name} as the book's text. To use several, merge them first in Training tools.`);
  };
  const audioDrop = useFileDrop(dropped);
  const bookDrop = useFileDrop(dropped);
  const audioIn = useRef<HTMLInputElement>(null);
  const bookIn = useRef<HTMLInputElement>(null);

  const prepBackends = options.data?.prepare_backends || [{ id: "mac", name: "This Mac", pricing: "free, runs locally", readiness: "ready", default_gpu: "", gpus: [] }];
  const selBackend = prepBackends.find((b) => b.id === backend) || prepBackends[0];

  const runCheck = async (id: string) => {
    if (id === "mac") { setCheck(null); return; }
    setChecking(true); setCheck(null);
    try { setCheck(await connectionApi.check(id)); } catch (e) { setCheck({ ok: false, detail: (e as Error).message }); }
    finally { setChecking(false); }
  };
  useEffect(() => {
    setCloudGpu(selBackend?.default_gpu || "");
    runCheck(backend);
  }, [backend]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { // follow the voice being prepared
    if (!voice) return;
    let live = true;
    const tick = () => trainApi.status(voice).then((s) => { if (live) { setStatus(s); if (s.running) setTimeout(tick, 2500); } }).catch(() => live && setTimeout(tick, 4000));
    tick();
    return () => { live = false; };
  }, [voice, pollNonce]);

  const estAudioSeconds = () => {
    if (!audio.length) return 3600;
    let secs = 0;
    for (const f of audio) {
      const uncompressed = /\.(wav|aif|aiff|flac)$/i.test(f.name);
      secs += f.size / (uncompressed ? 48000 : 16000);
    }
    return Math.max(900, Math.min(40 * 3600, Math.round(secs)));
  };

  const process = async (resume = false) => {
    const n = resume ? voice! : name;
    if (!n.trim()) return toast("Give the voice a name first.", true);
    // carrying on goes back where it was sent (the server keeps it); a cloud GPU is asked about either way
    const where = resume ? status?.backend || "mac" : backend;
    const gpu = resume ? status?.cloud_gpu || "" : cloudGpu;
    if (where !== "mac") {
      try {
        const e = await trainApi.prepareEstimate(where, gpu, resume ? 0 : estAudioSeconds(), resume ? n : "");
        if (e.needed) {
          const ok = await confirmCost({
            title: `Transcribe on ${e.where}?`, confirm: resume ? "Carry on" : "Start processing",
            cost: e.cost, free: e.free, details: [`${e.gpu} · ${e.time}`],
          });
          if (!ok) return;
        }
      } catch (x) { return toast((x as Error).message, true); }
    }
    setSending(true);
    try {
      const r = await trainApi.prepare({ name: n, title, language, backend: resume ? undefined : backend, cloud_gpu: resume ? undefined : cloudGpu,
        files: resume ? [] : [...audio, ...(bookFile ? [bookFile] : [])],
        options: { ...opts, min_match: opts.min_match / 100, split_narrators: split ? 1 : 0, narrators: howMany, dialogue_only: dialogue ? 1 : 0 } });
      toast(r.resumed ? `Carrying on with ${r.title}.` : `Started ${r.title}.`);
      setStatus(null);
      setVoice(r.voice);
      setPollNonce((x) => x + 1);
      prepared.reload();
    } catch (e) { toast((e as Error).message, true); }
    finally { setSending(false); }
  };
  const ready = (prepared.data || []).filter((v) => v.ready && !v.running);
  const pct = status?.progress != null ? Math.round(status.progress * 100) : null;
  const prepLeft = timeLeft(useEta(status?.progress, !!status?.running));

  return (
    <div className="page narrow">
      <Back onClick={back} label="All models" />
      <div className="page-head"><div><h1>Prepare a voice</h1><p>An audiobook and the same book's text become clean clips with the book's exact words.</p></div></div>
      <Steps at={1} recipe={view.recipe} addTo={view.addTo} />

      {ready.length > 0 && !voice && (
        <div className="card card-pad row wrap" style={{ marginBottom: 16 }}>
          <span className="small grow">Or use voices you prepared before:</span>
          <PreparedPicker voices={ready} value={earlier} onChange={setEarlier} max={MAX_VOICES - (view.addTo?.voices.length || 0)} />
          <button className="btn" disabled={!earlier.length} onClick={() => next(earlier)}>Use {earlier.length > 1 ? `these ${earlier.length}` : "it"}</button>
        </div>
      )}

      {!voice ? (
        <div className="card card-pad stack">
          <Field label="Voice name" hint="The reader's name, e.g. helen. Add more books to the same name any time; they're merged.">
            <input className="input" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. helen" list="prepared-names" />
            <datalist id="prepared-names">{(prepared.data || []).map((v) => <option key={v.name} value={v.name} />)}</datalist>
          </Field>
          <input ref={audioIn} type="file" hidden multiple accept={PREP_AUDIO}
            onChange={(e) => { setAudio(Array.from(e.target.files || [])); e.target.value = ""; }} />
          <input ref={bookIn} type="file" hidden accept={PREP_TEXT}
            onChange={(e) => { setBookFile(e.target.files?.[0] || null); e.target.value = ""; }} />
          <div className="two">
            <button className={`drop${audioDrop.over ? " over" : ""}`} onClick={() => audioIn.current?.click()} {...audioDrop.props}>
              <Upload size={18} /><b>{audio.length ? `${audio.length} audio file${audio.length === 1 ? "" : "s"}` : "The audiobook"}</b>
              <span className="small muted">{audio.length ? audio.map((f) => f.name).slice(0, 3).join(", ") + (audio.length > 3 ? "…" : "") : "mp3, m4b, flac, wav… or a zip. Several parts are fine."}</span>
            </button>
            <button className={`drop${bookDrop.over ? " over" : ""}`} onClick={() => bookIn.current?.click()} {...bookDrop.props}>
              <Upload size={18} /><b>{bookFile ? bookFile.name : "The book's text"}</b>
              <span className="small muted">EPUB, PDF, TXT, DOCX, MOBI… the same edition as the audio.</span>
            </button>
          </div>
          <div className="row">
            <Field label="Book title (optional)"><input className="input" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Read from the files if left empty" /></Field>
            <Field label="Language">
              <select className="select" value={language} onChange={(e) => setLanguage(e.target.value)}>
                {(options.data?.languages || ["english"]).map((l) => <option key={l} value={l}>{l[0].toUpperCase() + l.slice(1)}</option>)}
              </select>
            </Field>
          </div>
          <Switch label="Performer (only trains on dialogue)" checked={dialogue}
            onChange={(v) => { setDialogue(v); if (opts.min_seconds === (v ? 2 : 1)) setOpts({ ...opts, min_seconds: v ? 1 : 2 }); }} />
          <Switch label="More than one narrator" checked={split} onChange={setSplit}
            hint={split
              ? `The clips are sorted by who reads them, and each narrator becomes a voice of their own: ${name.trim() || "the name above"} (the one who reads the most), then ${name.trim() || "name"}-2, ${name.trim() || "name"}-3…`
              : "For a book read by two or more people. One voice trained on several narrators blends them and switches between them."} />
          {split && (
            <Field label="How many narrators">
              <select className="select" style={{ width: 200 }} value={howMany} onChange={(e) => setHowMany(Number(e.target.value))}>
                <option value={0}>Find out by listening</option>
                {[2, 3, 4].map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </Field>
          )}
          <details className="advanced">
            <summary>Advanced</summary>
            <div className="stack">
              <Slider label="Shortest clip" value={opts.min_seconds} min={1} max={5} step={0.5} onChange={(v) => setOpts({ ...opts, min_seconds: v })} format={(v) => `${v} s`} />
              <Slider label="Longest clip" value={opts.max_seconds} min={8} max={30} step={1} onChange={(v) => setOpts({ ...opts, max_seconds: v })} format={(v) => `${v} s`} />
              <Slider label="Aim for" value={opts.target_seconds} min={3} max={15} step={1} onChange={(v) => setOpts({ ...opts, target_seconds: v })} format={(v) => `${v} s`} />
              <Slider label="Word match needed" value={opts.min_match} min={70} max={100} step={1} onChange={(v) => setOpts({ ...opts, min_match: v })} format={(v) => `${v}%`} left="Keeps more clips" right="Fewer misreads" />
              <Slider label="Voice to noise, at least" value={opts.min_snr} min={10} max={40} step={1} onChange={(v) => setOpts({ ...opts, min_snr: v })} format={(v) => `${v} dB`} />
            </div>
          </details>
          <Field label="Transcribe on">
            <div className="row">
              <select className="select" value={backend} onChange={(e) => setBackend(e.target.value)}>
                {prepBackends.map((b) => <option key={b.id} value={b.id}>{b.name}{b.readiness === "beta" ? " (beta)" : ""}</option>)}
              </select>
              {selBackend && selBackend.gpus.length > 0 && (
                <select className="select" style={{ width: 190 }} value={cloudGpu} onChange={(e) => setCloudGpu(e.target.value)}>
                  {selBackend.gpus.map((g) => <option key={g.id} value={g.id}>{g.label}{g.price != null ? ` · $${g.price.toFixed(2)}/h` : ""}</option>)}
                </select>
              )}
              {backend !== "mac" && <button className="btn sm" onClick={() => runCheck(backend)} disabled={checking}><RefreshCw size={13} className={checking ? "spin" : ""} />Check</button>}
            </div>
          </Field>
          {backend !== "mac" && check && <div className={`small ${check.ok ? "ok-text" : "error-text"}`}>{check.ok ? "Ready: " : "Not ready: "}{check.detail}</div>}
          {backend !== "mac" && <button className="link small accent-link" onClick={goConnections}>Set up Kaggle, Modal or cloud GPUs in Connections</button>}
          <p className="small muted" style={{ margin: 0 }}>{backend === "mac" ? "Transcribing is the slow part: about 10–12 minutes per hour of audio on this Mac. Only use recordings you have the right to train on." : "Audio is compressed to 16 kHz Opus (~11 MB/h) for cloud transcription; clip cutting stays lossless on this Mac. Only use recordings you have the right to train on."}</p>
          <div className="row">
            <span className="spacer" />
            <button className="btn primary" disabled={sending || !name.trim() || !audio.length || !bookFile} onClick={() => process()}>
              <Wand2 size={15} />{sending ? "Uploading…" : "Process audiobook"}</button>
          </div>
        </div>
      ) : (
        <div className="card card-pad stack">
          <div className="row">
            <h3 style={{ margin: 0 }} className="grow">{voice}{status?.title ? ` · ${status.title}` : ""}</h3>
            {status?.running && <StopButton className="btn sm" onStop={() => trainApi.stop(voice).then(() => toast("Stopping after the current step…"))}><Square size={12} />Stop</StopButton>}
          </div>
          {status && (
            <div className="progress">
              <div className="row small"><b className="grow">{status.running ? status.stage || "Working…" : status.error ? (status.error === "stopped" ? "Stopped" : "Couldn't finish") : status.dataset ? "Ready to train" : status.stage}</b>
                {pct != null && status.running && <span className="muted">{pct}%{prepLeft ? ` · ${prepLeft}` : ""}</span>}</div>
              <div className={`bar ${status.running && pct == null ? "indeterminate" : ""} ${status.dataset && !status.running ? "done" : ""}`}><i style={{ width: `${status.running ? pct ?? 0 : 100}%` }} /></div>
              {status.error && status.error !== "stopped" && <div className="small error-text">{status.error}</div>}
            </div>
          )}
          {status?.dataset && <div className="small">Dataset: <b>{status.dataset.clips.toLocaleString()} clips</b> · <b>{status.dataset.hours} h</b>{status.dataset.hours < 1 ? " · 1–2 h is enough for a solid voice; more books help." : ""}</div>}
          {!!status?.books.length && (
            <table className="table small">
              <thead><tr><th>Book</th><th>Audio h</th><th>Kept h</th><th>Clips</th><th>Matched</th></tr></thead>
              <tbody>{status.books.map((r) => <tr key={r.title}><td>{r.title}</td><td>{r.audio_h}</td><td>{r.kept_h}</td><td>{r.clips}</td><td>{r.matched}</td></tr>)}</tbody>
            </table>
          )}
          {status?.log && <pre className="log">{status.log}</pre>}
          {status?.dataset && (
            <details className="advanced" onToggle={(e) => (e.target as HTMLDetailsElement).open && trainApi.clips(voice).then(setClips).catch(() => {})}>
              <summary><Play size={13} />Listen to 4 random clips</summary>
              <div className="stack" style={{ gap: 8 }}>{clips.map((c) => <div key={c.file}><div className="small muted">{c.text}</div><audio src={fileUrl(c.file)} controls preload="none" /></div>)}</div>
            </details>
          )}
          <div className="row">
            {status && !status.running && status.unfinished && <button className="btn" disabled={sending} onClick={() => process(true)}>Carry on where it stopped</button>}
            <button className="btn ghost" disabled={status?.running} onClick={() => { setVoice(null); setStatus(null); }}>Prepare another</button>
            <span className="spacer" />
            <button className="btn primary" disabled={!status?.dataset || status.running} onClick={() => next([voice])}>Continue to step 2</button>
          </div>
        </div>
      )}
    </div>
  );
}

function TrainStep({ view, back, home, goConnections }: {
  view: Extract<View, { at: "train" }>; back: () => void; home: () => void; goConnections: () => void;
}) {
  const toast = useToast();
  const options = useLoad(trainApi.options);
  // each prepared voice: how it's shown and its gender (a model can hold several)
  const newRow = (dataset: string) => {
    const guess = dataset.split("-")[0];
    return { dataset, shown: guess.charAt(0).toUpperCase() + guess.slice(1), gender: "unspecified" };
  };
  const [rows, setRows] = useState(() => view.voices.map(newRow));
  const setRow = (i: number, change: Partial<(typeof rows)[number]>) => setRows(rows.map((r, k) => (k === i ? { ...r, ...change } : r)));
  // more prepared voices can join (or leave) before training: they're all trained into this one model
  const prepared = useLoad(trainApi.voices);
  const ready = (prepared.data || []).filter((v) => v.ready && !v.running);
  const pickVoices = (names: string[]) => {
    if (!names.length) return toast("A model needs at least one voice.", true);
    setRows(names.map((d) => rows.find((r) => r.dataset === d) || newRow(d)));
  };
  const [model, setModel] = useState("");
  const [backend, setBackend] = useState("kaggle");
  const confirmCost = useCostConfirm();
  const starting = useStarting();
  const startTraining = async () => {
    if (!["local-mac", "ssh"].includes(backend)) {  // a cloud GPU: money, or Kaggle's weekly hours
      try {
        const e = await trainApi.estimate(body());
        const ok = await confirmCost({ title: `Train on ${e.backend}?`, confirm: "Start training",
          cost: e.paid ? e.cost || "price unknown" : backend === "kaggle" ? `free: about ${e.gpu_hours} of your weekly Kaggle GPU hours` : e.cost || "price unknown",
          free: !e.paid, details: [e.text] });
        if (!ok) return;
      } catch (x) { return toast((x as Error).message, true); }
    }
    send.start(() => trainApi.send(body()));
  };
  const [check, setCheck] = useState<CheckResult | null>(null);
  const [checking, setChecking] = useState(false);
  const [epochs, setEpochs] = useState<number | "">("");
  const [maxClips, setMaxClips] = useState(0);
  const [consent, setConsent] = useState(false);
  const [estimate, setEstimate] = useState("");
  const send = useTask((t) => {
    if (t.state === "failed") toast(t.error || "Couldn't start training.", true);
    else if (t.state === "done") toast("Training started. Check its status on Train a voice.");
  });
  const body = () => ({ voices: rows.map((r) => ({ dataset: r.dataset, name: r.shown, gender: r.gender })), model, recipe: view.recipe, backend,
    epochs: epochs === "" ? null : epochs, max_clips: maxClips, add_to: view.addTo?.folder || null, consent });
  const runCheck = async (id: string) => {
    setChecking(true); setCheck(null);
    try { setCheck(await connectionApi.check(id)); } catch (e) { setCheck({ ok: false, detail: (e as Error).message }); }
    finally { setChecking(false); }
  };
  useEffect(() => { runCheck(backend); }, [backend]); // eslint-disable-line react-hooks/exhaustive-deps
  const backends = (options.data?.backends || []).filter((b) => ["ready", "beta"].includes(b.readiness));
  const sel = backends.find((b) => b.id === backend);

  return (
    <div className="page narrow">
      <Back onClick={back} label="Step 1" />
      <div className="page-head"><div><h1>Train a model</h1><p>Encoding runs on this Mac first, so only small files are uploaded.</p></div></div>
      <Steps at={2} recipe={view.recipe} addTo={view.addTo} />
      <div className="card card-pad stack">
        <div className="summary-row"><Orb value={rows[0]?.dataset || ""} size={30} /><b>{rows.map((r) => r.shown || r.dataset).join(", ")}</b>
          <span className="muted small">· {view.addTo ? `goes into ${view.addTo.title}, with ${view.addTo.voices.map((v) => v.name).join(", ")}` : `a new ${RECIPE_NAME[view.recipe] || view.recipe} model`}</span></div>
        {ready.length > 1 && (
          <Field label="Voices in this model">
            <PreparedPicker voices={ready} value={rows.map((r) => r.dataset)} onChange={pickVoices} max={MAX_VOICES - (view.addTo?.voices.length || 0)} />
          </Field>
        )}
        {rows.map((r, i) => (
          <div className="row" key={r.dataset}>
            <Field label={rows.length > 1 ? `Shown as · ${r.dataset}` : "Shown as"}><input className="input" value={r.shown} onChange={(e) => setRow(i, { shown: e.target.value })} /></Field>
            <Field label="Gender">
              <select className="select" value={r.gender} onChange={(e) => setRow(i, { gender: e.target.value })}>
                <option value="unspecified">Not set</option><option value="female">Female</option><option value="male">Male</option>
              </select>
            </Field>
          </div>
        ))}
        {!view.addTo && <Field label="Model name" hint="Letters and numbers, e.g. fantasy"><input className="input" value={model} onChange={(e) => setModel(e.target.value)} placeholder="e.g. fantasy" /></Field>}
        <Field label="Train on">
          <div className="row">
            <select className="select" value={backend} onChange={(e) => setBackend(e.target.value)}>
              {backends.map((b) => <option key={b.id} value={b.id}>{b.name}{b.readiness === "beta" ? " (beta)" : ""}</option>)}
            </select>
            <button className="btn sm" onClick={() => runCheck(backend)} disabled={checking}><RefreshCw size={13} className={checking ? "spin" : ""} />Check</button>
          </div>
        </Field>
        {sel && <div className="small muted">{sel.pricing}</div>}
        {check && <div className={`small ${check.ok ? "ok-text" : "error-text"}`}>{check.ok ? "Ready: " : "Not ready: "}{check.detail}</div>}
        <button className="link small accent-link" onClick={goConnections}>Set up Kaggle, a GPU machine or Modal in Connections</button>
        <details className="advanced">
          <summary>Advanced</summary>
          <div className="row">
            <Field label="Passes over the data" hint="Blank is the recommended number.">
              <input className="input" type="number" min={0.5} step={0.5} value={epochs} onChange={(e) => setEpochs(e.target.value === "" ? "" : Number(e.target.value))} />
            </Field>
            <Field label="Use only this many clips" hint="0 is all of them; fewer makes a quick test.">
              <input className="input" type="number" min={0} value={maxClips} onChange={(e) => setMaxClips(Number(e.target.value) || 0)} />
            </Field>
          </div>
        </details>
        <label className="row small" style={{ cursor: "pointer" }}>
          <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} style={{ accentColor: "var(--accent)" }} />
          I own these recordings or have the speaker's permission to train on them
        </label>
        <div className="row">
          <button className="btn" onClick={async () => { try { setEstimate((await trainApi.estimate(body())).text); } catch (e) { toast((e as Error).message, true); } }}>Estimate</button>
          <span className="small muted grow">{estimate}</span>
          <button className="btn primary" disabled={!consent || send.running || starting.starting || (!view.addTo && !model.trim())} onClick={() => starting.run(startTraining)}>
            {starting.starting ? <Starting size={15} /> : <><Wand2 size={15} />Start training</>}</button>
        </div>
        {send.task && <TaskProgress task={send.task} onStop={send.stop} />}
        {send.task?.result.log && <pre className="log">{send.task.result.log}</pre>}
        {send.task?.state === "done" && <div className="row"><span className="spacer" /><button className="btn" onClick={home}>Back to all models</button></div>}
      </div>
    </div>
  );
}

function Export({ models, back }: { models: TrainedModel[]; back: () => void }) {
  const toast = useToast();
  const packs = useLoad(libraryApi.packs);
  const [folder, setFolder] = useState(models[0]?.folder || "");
  const m = models.find((x) => x.folder === folder) || null;
  const [name, setName] = useState("");
  const [genre, setGenre] = useState("General");
  const [version, setVersion] = useState(1);
  const [desc, setDesc] = useState("");
  const [author, setAuthor] = useState("");
  const [previews, setPreviews] = useState(true);
  const [signer, setSigner] = useState<Signing | null>(null);
  const [sign, setSign] = useState(false);
  useEffect(() => { signingApi.get().then((x) => { setSigner(x); setSign(!!x.on_github); }).catch(() => {}); }, []);
  const [rights, setRights] = useState<Record<string, { source: string; license: string; consent: string }>>({});
  const job = useTask((t) => { if (t.state === "failed") toast(t.error || "Couldn't export it.", true); });
  useEffect(() => { if (m && !name) setName(m.title); }, [m?.folder]); // eslint-disable-line react-hooks/exhaustive-deps
  const row = (s: string) => rights[s] || { source: "", license: "", consent: "" };
  const setRow = (s: string, k: "source" | "license" | "consent", v: string) => setRights({ ...rights, [s]: { ...row(s), [k]: v } });
  const missing = useMemo(() => (m?.voices || []).filter((v) => !row(v.speaker).source.trim() || !row(v.speaker).license.trim()).length, [m, rights]); // eslint-disable-line react-hooks/exhaustive-deps
  const file = job.task?.state === "done" ? job.task.result.file : null;
  return (
    <div className="page narrow">
      <Back onClick={back} label="All models" />
      <div className="page-head"><div><h1>Export a voice pack</h1><p>Share a model you trained as one file that others install in their Library.</p></div></div>
      <div className="card card-pad stack">
        <div className="section-label">Which model</div>
        <div className="pill-pick" role="radiogroup">
          {models.map((x) => (
            <button key={x.folder} role="radio" aria-checked={folder === x.folder} className={folder === x.folder ? "on" : ""} onClick={() => setFolder(x.folder)}>
              <b>{x.title}</b><span className="small muted">{x.voices.map((v) => v.name).join(", ")}</span>
            </button>
          ))}
        </div>
        <div className="row">
          <Field label="Pack name"><input className="input" value={name} onChange={(e) => setName(e.target.value)} /></Field>
          <Field label="Genre">
            <select className="select" value={genre} onChange={(e) => setGenre(e.target.value)}>
              {(packs.data?.genres || ["General"]).map((g) => <option key={g}>{g}</option>)}
            </select>
          </Field>
          <Field label="Version"><input className="input" type="number" min={1} value={version} onChange={(e) => setVersion(Number(e.target.value) || 1)} style={{ width: 90 }} /></Field>
        </div>
        <Field label="Description"><textarea className="textarea" rows={2} value={desc} onChange={(e) => setDesc(e.target.value)} /></Field>
        <Field label="Made by"><input className="input" value={author} onChange={(e) => setAuthor(e.target.value)} /></Field>
        <div className="section-label">Where each voice came from</div>
        <p className="small muted" style={{ margin: 0 }}>Only export voices you have the right to share. Each needs its source and license; whoever installs the pack sees them.</p>
        <table className="table small rights">
          <thead><tr><th>Voice</th><th>Source</th><th>License</th><th>Consent</th></tr></thead>
          <tbody>{(m?.voices || []).map((v) => (
            <tr key={v.speaker}>
              <td><span className="row" style={{ gap: 6 }}><Orb value={v.value || v.speaker} size={18} />{v.name}</span></td>
              <td><input className="input" placeholder="e.g. my own recording" value={row(v.speaker).source} onChange={(e) => setRow(v.speaker, "source", e.target.value)} /></td>
              <td><input className="input" placeholder="e.g. CC BY 4.0" value={row(v.speaker).license} onChange={(e) => setRow(v.speaker, "license", e.target.value)} /></td>
              <td><input className="input" placeholder="e.g. emailed 2026-11-10" value={row(v.speaker).consent} onChange={(e) => setRow(v.speaker, "consent", e.target.value)} /></td>
            </tr>
          ))}</tbody>
        </table>
        <label className="row small" style={{ cursor: "pointer" }}>
          <input type="checkbox" checked={previews} onChange={(e) => setPreviews(e.target.checked)} style={{ accentColor: "var(--accent)" }} />
          Record a short preview of each voice (about a minute per voice)
        </label>
        <label className="row small" style={{ cursor: signer?.on_github ? "pointer" : "default" }}>
          <input type="checkbox" checked={sign} disabled={!signer?.on_github} onChange={(e) => setSign(e.target.checked)} style={{ accentColor: "var(--accent)" }} />
          {signer?.on_github ? `Sign it as ${signer.github}` : <span className="muted">Sign it with your GitHub name (set it up in Settings → General)</span>}
        </label>
        <div className="row">
          <span className="small muted grow">{missing ? `Fill in the source and license for ${missing} voice${missing === 1 ? "" : "s"}.` : ""}</span>
          <button className="btn primary" disabled={!m || !name.trim() || !!missing || job.running}
            onClick={() => m && job.start(() => trainApi.exportPack({ folder: m.folder, name, genre, version, description: desc, author, rights, previews, sign }))}><Package size={15} />Export pack</button>
        </div>
        {job.task && <TaskProgress task={job.task} onStop={job.stop} hideDone={!!file} />}
        {file && <div className="made-note"><span className="small grow">Pack written: <b>{file.split(/[\\/]/).pop()}</b></span>
          <button className="btn sm primary" onClick={() => saveFile(file).then((msg) => toast(msg), (e) => toast(String(e), true))}>Save the pack file</button></div>}
      </div>
    </div>
  );
}
