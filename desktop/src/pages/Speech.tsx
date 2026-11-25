import { useEffect, useMemo, useRef, useState } from "react";
import { Download, Eraser, FileUp, Loader2, Play, RotateCcw, Sparkles, Square, Trash2, Undo2, Wand2 } from "lucide-react";
import { api, bookApi, fileUrl, HistoryItem, useLoad, useTask, type Engine, type Voice } from "../api";
import { VoicePicker } from "../components/VoicePicker";
import { ConfirmDialog, Field, ParallelLines, Slider, TaskProgress, accepts, clock, useFileDrop, useToast } from "../components/ui";
import { saveFile } from "../native";

const LANGUAGES = ["english", "auto", "chinese", "german", "italian", "portuguese", "spanish", "japanese", "korean", "french", "russian"];
const TEXT_FILES = ".txt,.md,.epub,.pdf,.docx,.mobi,.azw3,.rtf,.odt,.fb2,.html";
const STYLES = ["warm and calm", "excited", "whispering", "sad", "angry", "cheerful"];
const DEFAULTS = { text: "", voice: "", style: "", speed: 1, stability: 55, language: "english", seed: 0, pause: 0.5, max_chars: 250, batch: 2, parallel: true };
type Form = typeof DEFAULTS;
const KEY = "speech-form";

function loadForm(): Form {
  try { return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(KEY) || "{}") }; } catch { return DEFAULTS; }
}

export function Speech({ voices, engines }: { voices: Voice[]; engines: Engine[] }) {
  const [form, setForm] = useState<Form>(loadForm);
  const set = <K extends keyof Form>(k: K, v: Form[K]) => setForm((f) => ({ ...f, [k]: v }));
  const toast = useToast();
  const history = useLoad(api.history);
  const { task, start, stop, running } = useTask((t) => {
    history.reload();
    if (t.state === "failed") toast(t.error || "Couldn't make the speech.", true);
  }, "speech");
  const [playing, setPlaying] = useState<string | null>(null);
  const [asking, setAsking] = useState(false); // the "clear all the text?" confirmation
  const [cleared, setCleared] = useState<string | null>(null); // what Clear took out, until something new is typed
  const save = (path: string) => saveFile(path).then((m) => toast(m), (e) => toast(String(e), true));
  const removeHistory = async (path: string) => {
    try {
      if (playing === path) setPlaying(null);
      await api.deleteHistory(path);
      history.reload();
      toast("Deleted audio file.");
    } catch (e) { toast((e as Error).message, true); }
  };

  useEffect(() => { try { localStorage.setItem(KEY, JSON.stringify(form)); } catch { /* private window */ } }, [form]);
  useEffect(() => { // a default voice: your first saved voice, else the first one this Mac can run
    if (!voices.length || voices.some((v) => v.value === form.voice && v.available)) return;
    const pick = voices.find((v) => v.tags.includes("saved") && v.available) || voices.find((v) => v.available);
    if (pick) set("voice", pick.value);
  }, [voices]); // eslint-disable-line react-hooks/exhaustive-deps

  const chars = form.text.length;
  const estimate = useMemo(() => clock(chars / 15), [chars]);
  const voice = voices.find((v) => v.value === form.voice);
  const note = engines.find((e) => e.id === voice?.engine)?.note;
  const result = task?.state === "done" ? task.result.audio : null;

  const fileIn = useRef<HTMLInputElement>(null);
  const [importing, setImporting] = useState(false);
  const importText = async (file: File) => {
    setImporting(true);
    try {
      const r = await bookApi.importFile(file);
      const text = r.text.trim();
      if (!text) return toast(`There's no text in ${file.name}.`, true);
      setForm((f) => ({ ...f, text: f.text.trim() ? `${f.text.trimEnd()}\n\n${text}` : text }));  // (added after what's there)
      setCleared(null);
      toast(`Imported ${file.name}${form.text.trim() ? " after your text" : ""}.`);
    } catch (e) { toast((e as Error).message, true); }
    finally { setImporting(false); }
  };
  // text files dragged in from Finder: each one's text goes in after what's there, in name order
  const drop = useFileDrop(async (got) => {
    const fit = got.filter((f) => accepts(f, TEXT_FILES));
    if (!fit.length) return toast("Those aren't text files.", true);
    for (const f of fit) await importText(f);
  }, importing);

  const go = (preview: boolean) => {
    if (!form.text.trim()) return toast("Type or paste some text first.", true);
    start(() => api.speech({ ...form, batch: form.parallel ? Math.max(2, form.batch) : 1, preview }));
  };

  return (
    <div className="page">
      <div className="page-head">
        <div><h1>Text to Speech</h1><p>Turn any text into speech in any voice.</p></div>
      </div>
      <div className="split">
        <div className="stack">
          <div className={`card${drop.over ? " file-over" : ""}`} {...drop.props}>
            <textarea className="editor" placeholder="Type or paste your text here…" value={form.text}
              onChange={(e) => { set("text", e.target.value); setCleared(null); }} />
            <div className="editor-foot">
              <span className="small muted stats">{chars.toLocaleString()} characters · about {estimate} of audio</span>
              {form.text && <button className="btn sm ghost" onClick={() => setAsking(true)} title="Remove all the text"><Eraser size={13} />Clear</button>}
              {!form.text && cleared && <button className="btn sm ghost" onClick={() => { set("text", cleared); setCleared(null); }}><Undo2 size={13} />Undo clear</button>}
              <input ref={fileIn} type="file" hidden accept={TEXT_FILES}
                onChange={(e) => { const f = e.target.files?.[0]; if (f) importText(f); e.target.value = ""; }} />
              <button className="btn sm ghost" disabled={importing} onClick={() => fileIn.current?.click()}>
                {importing ? <Loader2 size={13} className="spin" /> : <FileUp size={13} />}Import</button>
              <span className="spacer" />
              <button className="btn" disabled={running} onClick={() => go(true)}><Play size={15} />Preview</button>
              <button className="btn primary" disabled={running} onClick={() => go(false)}><Wand2 size={15} />Generate speech</button>
            </div>
          </div>

          {asking && (
            <ConfirmDialog title="Clear all the text?" confirm="Clear" danger onCancel={() => setAsking(false)}
              onConfirm={() => { setCleared(form.text); set("text", ""); setAsking(false); }}>
              Everything in the text box goes ({chars.toLocaleString()} characters). Undo clear brings it back until you type something new.
            </ConfirmDialog>
          )}

          {task && (
            <div className="card card-pad stack">
              <TaskProgress task={task} onStop={stop} />
              {result && (
                <div className="row">
                  <audio key={result} src={fileUrl(result)} controls autoPlay />
                  <button className="btn icon" onClick={() => save(result)} title="Save the file"><Download size={15} /></button>
                </div>
              )}
            </div>
          )}

          <div className="card">
            <div className="row card-pad" style={{ paddingBottom: 8 }}>
              <h3 className="card-title" style={{ margin: 0 }}>Recents</h3>
              <span className="spacer" />
              <button className="btn sm ghost" onClick={history.reload}><RotateCcw size={13} />Refresh</button>
            </div>
            <div className="list">
              {(history.data || []).slice(0, 12).map((h: HistoryItem) => (
                <div key={h.path} className="list-item">
                  <button className="btn icon sm" onClick={() => setPlaying(playing === h.path ? null : h.path)} aria-label={playing === h.path ? "Stop" : "Play"}>{playing === h.path ? <Square size={11} /> : <Play size={13} />}</button>
                  <div className="grow">
                    <div className="title" style={{ textTransform: "capitalize" }}>{h.kind.replace("-", " ")}{h.partial && <span className="chip" style={{ marginLeft: 8 }}>partial</span>}</div>
                    <div className="small muted">{new Date(h.time).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })} · {clock(h.seconds)}</div>
                    {playing === h.path && <audio src={fileUrl(h.path)} controls autoPlay style={{ marginTop: 8 }} />}
                  </div>
                  <button className="btn icon sm ghost" onClick={() => save(h.path)} title="Save the file"><Download size={13} /></button>
                  <button className="btn icon sm ghost" onClick={() => removeHistory(h.path)} title="Delete this file" aria-label="Delete this file"><Trash2 size={13} /></button>
                </div>
              ))}
              {history.data && !history.data.length && <div className="empty small">Nothing yet. What you make shows up here.</div>}
            </div>
          </div>
        </div>

        <div className="card card-pad stack">
          <Field label="Voice"><VoicePicker voices={voices} engines={engines} value={form.voice} onChange={(v) => set("voice", v)} /></Field>
          {note && <div className="small" style={{ color: "var(--warn)" }}>{note}</div>}
          <Field label="Style or emotion">
            <input className="input" placeholder="e.g. warm and calm" value={form.style} onChange={(e) => set("style", e.target.value)} />
          </Field>
          <div className="row wrap" style={{ gap: 6, marginTop: -4 }}>
            {STYLES.map((s) => <button key={s} className="chip" style={{ border: 0, cursor: "pointer" }} onClick={() => set("style", s)}><Sparkles size={10} />{s}</button>)}
          </div>
          <Slider label="Speed" value={form.speed} min={0.7} max={1.3} step={0.05} onChange={(v) => set("speed", v)}
            left="Slower" right="Faster" format={(v) => `${v.toFixed(2)}×`} />
          <Slider label="Stability" value={form.stability} min={0} max={100} step={5} onChange={(v) => set("stability", v)}
            left="More expressive" right="More stable" format={(v) => `${v}%`} />
          {voice && ["kokoro", "vibevoice"].includes(voice.engine) && (
            <p className="small muted" style={{ margin: "-6px 0 0" }}>{engines.find((e) => e.id === voice.engine)?.short || voice.engine} voices read the same way every time, so stability doesn't change them.</p>
          )}
          <ParallelLines on={form.parallel} count={form.batch} setOn={(v) => set("parallel", v)} setCount={(n) => set("batch", n)} />
          <details className="advanced">
            <summary>Advanced</summary>
            <div className="stack">
              <Field label="Language">
                <select className="select" value={form.language} onChange={(e) => set("language", e.target.value)}>
                  {LANGUAGES.map((l) => <option key={l} value={l}>{l[0].toUpperCase() + l.slice(1)}</option>)}
                </select>
              </Field>
              <Slider label="Pause between paragraphs" value={form.pause} min={0} max={2} step={0.1} onChange={(v) => set("pause", v)} format={(v) => `${v.toFixed(1)} s`} />
              <Slider label="Longest line" value={form.max_chars} min={100} max={400} step={10} onChange={(v) => set("max_chars", v)} format={(v) => `${v} chars`} />
              <Field label="Seed" hint="0 is random; the same seed and text give the same take.">
                <input className="input" type="number" min={0} value={form.seed} onChange={(e) => set("seed", Number(e.target.value) || 0)} />
              </Field>
              <button className="btn sm ghost" onClick={() => setForm((f) => ({ ...DEFAULTS, text: f.text, voice: f.voice }))}><RotateCcw size={13} />Reset values</button>
            </div>
          </details>
        </div>
      </div>
    </div>
  );
}
