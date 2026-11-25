import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Loader2, Mic, Pause, Pencil, Play, Save, Search, Trash2, Upload, Wand2 } from "lucide-react";
import { bookApi, fileUrl, isCopy, useLoad, useTask, voiceApi, type Engine, type MyVoice, type Voice } from "../api";
import { useVoicePreview } from "../hooks/useAudioPreview";
import { Field, Orb, Starting, TaskProgress, useStarting, useToast } from "../components/ui";

type Tab = "mine" | "design" | "clone";
const GENDERS = ["female", "male", "unspecified"];

function GenderSelect({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <select className="select" value={value} onChange={(e) => onChange(e.target.value)}>
      {GENDERS.map((g) => <option key={g} value={g}>{g === "unspecified" ? "Not set" : g[0].toUpperCase() + g.slice(1)}</option>)}
    </select>
  );
}

/** Every voice, designing a voice, and cloning one. */
export function Voices({ voices, engines, onSaved }: { voices: Voice[]; engines: Engine[]; onSaved: () => void }) {
  const [tab, setTab] = useState<Tab>(() => { try { return (localStorage.getItem("voices-tab") as Tab) || "mine"; } catch { return "mine"; } });
  const pick = (t: Tab) => { setTab(t); try { localStorage.setItem("voices-tab", t); } catch { /* fine */ } };
  const mine = useLoad(voiceApi.mine);
  const saved = () => { mine.reload(); onSaved(); };
  return (
    <div className="page">
      <div className="page-head">
        <div><h1>Voices</h1></div>
      </div>
      <div className="tabs page-tabs" role="tablist">
        {([["mine", "Voices"], ["design", "Design a voice"], ["clone", "Clone a voice"]] as [Tab, string][]).map(([id, label]) => (
          <button key={id} role="tab" aria-selected={tab === id} className={tab === id ? "on" : ""} onClick={() => pick(id)}>{label}</button>
        ))}
      </div>
      {tab === "mine" && <AllVoices list={mine.data} reload={saved} voices={voices} engines={engines} goDesign={() => pick("design")} />}
      {tab === "design" && <Design onSaved={() => { saved(); pick("mine"); }} />}
      {tab === "clone" && <Clone onSaved={() => { saved(); pick("mine"); }} />}
    </div>
  );
}

/** Every voice, by the model it runs on, each with a preview. Your own voices (designed or cloned) open to edit. */
function AllVoices({ list, reload, voices, engines, goDesign }: {
  list: MyVoice[] | null; reload: () => void; voices: Voice[]; engines: Engine[]; goDesign: () => void;
}) {
  const [q, setQ] = useState("");
  const [editing, setEditing] = useState<MyVoice | null>(null);
  const preview = useVoicePreview();
  const engineName = (id: string) => engines.find((e) => e.id === id)?.short || id;
  const groups = useMemo(() => {
    const words = q.toLowerCase().split(/\s+/).filter(Boolean);
    const shown = voices.filter((v) => !isCopy(v) && words.every((w) =>
      `${v.name} ${v.description} ${v.gender} ${engineName(v.engine)} ${v.tags.join(" ")}`.toLowerCase().includes(w)));
    const order = engines.map((e) => e.id);
    const ids = [...new Set(shown.map((v) => v.engine))].sort((a, b) => order.indexOf(a) - order.indexOf(b));
    return ids.map((id) => ({ id, name: engines.find((e) => e.id === id)?.name || id, voices: shown.filter((v) => v.engine === id) }));
  }, [voices, engines, q]); // eslint-disable-line react-hooks/exhaustive-deps
  const mineByValue = (v: Voice) => (v.tags.includes("saved") ? (list || []).find((m) => `my:${m.name}` === v.value) || null : null);

  return (
    <div className="stack" style={{ gap: 16 }}>
      <div className="row wrap">
        <div className="search-box grow"><Search size={14} className="muted" />
          <input className="input" placeholder="Search voices" aria-label="Search voices" value={q} onChange={(e) => setQ(e.target.value)} /></div>
        <button className="btn" onClick={goDesign}><Wand2 size={14} />Design a voice</button>
      </div>
      {groups.map((g) => (
        <div key={g.id} className="card">
          <div className="card-pad row" style={{ paddingBottom: 6 }}>
            <h3 className="card-title" style={{ margin: 0 }}>{g.name}</h3>
            <span className="small muted">{g.voices.length}</span>
          </div>
          <div className="voice-grid">
            {g.voices.map((v) => {
              const mine = mineByValue(v);
              const busy = preview.making.has(v.value);
              const canPlay = v.available;
              return (
                <div key={v.value} className={`voice-cell ${v.available ? "" : "locked"} ${mine ? "editable" : ""}`}
                  title={v.available ? v.description : v.why}>
                  <button className="btn sm ghost icon voice-play" disabled={!canPlay || busy}
                    aria-label={preview.on === v.value ? `Pause ${v.name}` : `Preview ${v.name}`}
                    title={!canPlay ? v.why : undefined} onClick={() => preview.toggle(v.value)}>
                    {busy ? <Loader2 size={13} className="spin" /> : preview.on === v.value ? <Pause size={13} /> : <Play size={13} />}
                  </button>
                  <Orb value={v.value} size={28} />
                  <span className="voice-meta grow" onClick={mine ? () => setEditing(mine) : undefined} role={mine ? "button" : undefined}>
                    <strong>{v.name}</strong>
                    <span>{[v.gender !== "unspecified" && v.gender, v.tags.includes("fine-tuned") && "fine-tuned", mine && (mine.source || "yours")].filter(Boolean).join(" · ") || " "}</span>
                  </span>
                  {mine && <button className="btn sm ghost icon" aria-label={`Edit ${v.name}`} onClick={() => setEditing(mine)}><Pencil size={13} /></button>}
                </div>
              );
            })}
          </div>
        </div>
      ))}
      {!groups.length && <div className="card card-pad empty">{q ? "No voices match." : "No voices yet."}</div>}
      {editing && <EditVoice voice={editing} onClose={() => setEditing(null)} onChanged={() => { reload(); setEditing(null); }} />}
    </div>
  );
}

/** A voice of your own: its sample, gender, description and transcript, or delete it. */
function EditVoice({ voice, onClose, onChanged }: { voice: MyVoice; onClose: () => void; onChanged: () => void }) {
  const toast = useToast();
  const [form, setForm] = useState({ gender: voice.gender, description: voice.description, transcript: voice.transcript });
  const [confirm, setConfirm] = useState(false);
  const saving = useStarting();
  const save = async () => {
    try { await voiceApi.update(voice.name, form); toast(`Saved changes to ${voice.name}.`); onChanged(); }
    catch (e) { toast((e as Error).message, true); }
  };
  const remove = async () => {
    try { await voiceApi.remove(voice.name); toast(`Deleted ${voice.name}.`); onChanged(); }
    catch (e) { toast((e as Error).message, true); }
  };
  useEffect(() => {
    const key = (e: KeyboardEvent) => { if (e.key === "Escape") { e.stopPropagation(); onClose(); } };
    window.addEventListener("keydown", key, true);
    return () => window.removeEventListener("keydown", key, true);
  }, [onClose]);
  return createPortal(
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal" role="dialog" aria-modal="true" aria-labelledby="edit-voice-title">
        <div className="row"><Orb value={`my:${voice.name}`} size={40} />
          <div><h2 id="edit-voice-title" style={{ margin: 0 }}>{voice.name}</h2>
            <div className="small muted">{voice.source} · saved {voice.created || "—"}</div></div></div>
        <div className="modal-body stack">
          <audio key={voice.file} src={fileUrl(voice.file)} controls />
          <Field label="Gender"><GenderSelect value={form.gender} onChange={(gender) => setForm({ ...form, gender })} /></Field>
          <Field label="Description"><input className="input" value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} /></Field>
          <Field label="What the sample says" hint="Used when the voice is cloned exactly (Text to Speech → Advanced).">
            <textarea className="textarea" rows={3} value={form.transcript} onChange={(e) => setForm({ ...form, transcript: e.target.value })} />
          </Field>
          {confirm && (
            <div className="confirm" role="alertdialog" aria-label={`Delete ${voice.name}?`}>
              <span className="small">Delete {voice.name}? Books that use it will need another voice. This can't be undone.</span>
              <div className="row" style={{ gap: 6 }}>
                <button className="btn sm ghost" onClick={() => setConfirm(false)}>Cancel</button>
                <button className="btn sm danger-solid" onClick={remove}><Trash2 size={13} />Delete</button>
              </div>
            </div>
          )}
        </div>
        <div className="modal-actions">
          <button className="btn danger" onClick={() => setConfirm(true)}><Trash2 size={14} />Delete</button>
          <span className="spacer" />
          <button className="btn" onClick={onClose}>Close</button>
          <button className="btn primary" disabled={saving.starting} onClick={() => saving.run(save)}>
            {saving.starting ? <Starting label="Saving…" /> : <><Save size={14} />Save changes</>}</button>
        </div>
      </div>
    </div>,
    document.body,
  );
}

function Design({ onSaved }: { onSaved: () => void }) {
  const toast = useToast();
  const options = useLoad(bookApi.options);
  const [desc, setDesc] = useState("");
  const [text, setText] = useState("");
  const [language, setLanguage] = useState("english");
  const [seed, setSeed] = useState(0);
  const [pick, setPick] = useState(0);
  const [name, setName] = useState("");
  const [gender, setGender] = useState("unspecified");
  const takes = useTask((t) => t.state === "failed" && toast(t.error || "Couldn't make the takes.", true), "voices-design");
  const made: string[] = takes.task?.result.takes || [];
  const saving = useStarting();
  const save = async () => {
    if (!made[pick]) return;
    try {
      const v = await voiceApi.saveTake({ path: made[pick], name, gender, description: desc, transcript: text });
      toast(`Saved “${v.name}”. It's in every voice list now.`);
      onSaved();
    } catch (e) { toast((e as Error).message, true); }
  };
  return (
    <div className="split">
      <div className="card card-pad stack">
        <Field label="Describe the voice" hint="Age, accent, tone, pace: “a warm, low-voiced old sailor with a slight Irish accent”.">
          <textarea className="textarea" rows={3} value={desc} onChange={(e) => setDesc(e.target.value)} placeholder="e.g. a calm, clear young woman, softly spoken" />
        </Field>
        <Field label="What it reads (optional)">
          <textarea className="textarea" rows={2} value={text} onChange={(e) => setText(e.target.value)}
            placeholder="The village woke slowly that morning, wrapped in fog and the quiet sound of waves against the harbor wall." />
        </Field>
        <div className="row">
          <Field label="Language">
            <select className="select" value={language} onChange={(e) => setLanguage(e.target.value)}>
              {(options.data?.languages || ["english"]).map((l) => <option key={l} value={l}>{l[0].toUpperCase() + l.slice(1)}</option>)}
            </select>
          </Field>
          <Field label="Seed (0 is random)"><input className="input" type="number" min={0} value={seed} onChange={(e) => setSeed(Number(e.target.value) || 0)} /></Field>
        </div>
        <button className="btn primary" disabled={!desc.trim() || takes.running}
          onClick={() => { setPick(0); takes.start(() => voiceApi.design({ description: desc, text, language, seed })); }}><Wand2 size={15} />Make 3 takes</button>
        {takes.task && takes.task.state !== "done" && <TaskProgress task={takes.task} onStop={takes.stop} />}
      </div>
      <div className="card card-pad stack">
        <h3 className="card-title" style={{ margin: 0 }}>Takes</h3>
        {!made.length && <div className="empty small">Describe a voice and make three takes, then keep your favourite.</div>}
        {made.map((p, i) => (
          <label key={p} className={`take ${pick === i ? "sel" : ""}`}>
            <input type="radio" name="take" checked={pick === i} onChange={() => setPick(i)} style={{ accentColor: "var(--accent)" }} />
            <span className="small" style={{ width: 48 }}>Take {i + 1}</span>
            <audio src={fileUrl(p)} controls />
          </label>
        ))}
        {made.length > 0 && (
          <>
            <Field label="Name"><input className="input" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Old Sailor" /></Field>
            <Field label="Gender"><GenderSelect value={gender} onChange={setGender} /></Field>
            <button className="btn primary" disabled={!name.trim() || !made[pick] || saving.starting} onClick={() => saving.run(save)}>
              {saving.starting ? <Starting label="Saving…" /> : <><Save size={14} />Save take {pick + 1} to my voices</>}</button>
          </>
        )}
      </div>
    </div>
  );
}

function Clone({ onSaved }: { onSaved: () => void }) {
  const toast = useToast();
  const input = useRef<HTMLInputElement>(null);
  const [path, setPath] = useState<string | null>(null);
  const [fileName, setFileName] = useState("");
  const [transcript, setTranscript] = useState("");
  const [name, setName] = useState("");
  const [gender, setGender] = useState("unspecified");
  const [desc, setDesc] = useState("");
  const tr = useTask((t) => {
    if (t.state === "failed") toast(t.error || "Couldn't transcribe it.", true);
    else if (t.state === "done") {
      const txt = (t.result?.text || "").trim();
      setTranscript(txt);
      if (!txt) toast("No speech was detected in that sample.", true);
    }
  });
  const upload = async (f: File) => {
    if (tr.running) return;
    try { setPath((await bookApi.uploadFile(f)).path); setFileName(f.name); setTranscript(""); }
    catch (e) { toast((e as Error).message, true); }
  };
  const saving = useStarting();
  const save = async () => {
    if (!path) return;
    try {
      const v = await voiceApi.saveClone({ path, name, gender, description: desc, transcript });
      toast(`Saved “${v.name}”. It's in every voice list now.`);
      onSaved();
    } catch (e) { toast((e as Error).message, true); }
  };
  return (
    <div className="split">
      <div className="card card-pad stack">
        <input ref={input} type="file" hidden accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg"
          onChange={(e) => { const f = e.target.files?.[0]; if (f) upload(f); e.target.value = ""; }} />
        <div className="drop" onClick={() => !tr.running && input.current?.click()} role="button" tabIndex={0}
          onKeyDown={(e) => e.key === "Enter" && !tr.running && input.current?.click()}
          onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); const f = e.dataTransfer.files?.[0]; if (f) upload(f); }}>
          <Upload size={20} />
          <b>{fileName || "Choose a voice sample"}</b>
          <span className="small muted">5–15 seconds of one person speaking clearly, no music. Drop it here or click.</span>
        </div>
        {path && <audio key={path} src={fileUrl(path)} controls />}
        <Field label="What the sample says" hint="Exact words make the clone closer. Transcribe fills it in for you.">
          <textarea className="textarea" rows={3} value={transcript} onChange={(e) => setTranscript(e.target.value)} />
        </Field>
        <button className="btn" disabled={!path || tr.running} onClick={() => path && tr.start(() => voiceApi.transcribe(path))}><Mic size={14} />Transcribe automatically</button>
        {tr.task?.state === "running" && <TaskProgress task={tr.task} onStop={tr.stop} />}
      </div>
      <div className="card card-pad stack">
        <Field label="Name"><input className="input" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Grandpa" /></Field>
        <Field label="Gender"><GenderSelect value={gender} onChange={setGender} /></Field>
        <Field label="Description (optional)"><input className="input" value={desc} onChange={(e) => setDesc(e.target.value)} /></Field>
        <p className="small muted" style={{ margin: 0 }}>Only clone voices you have permission to use.</p>
        <button className="btn primary" disabled={!path || !name.trim() || saving.starting} onClick={() => saving.run(save)}>
          {saving.starting ? <Starting label="Saving…" /> : <><Save size={14} />Save to my voices</>}</button>
      </div>
    </div>
  );
}

