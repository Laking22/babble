import { createContext, ReactNode, useCallback, useContext, useEffect, useRef, useState, type DragEvent } from "react";
import { createPortal } from "react-dom";
import { AlertCircle, CheckCircle2, Loader2, X } from "lucide-react";
import type { Task } from "../api";

/** Stop, which says it's stopping (and can't be pressed again) from the moment it's pressed until the work ends.
 *  It ignores clicks in its first half second: it often appears where the start button was, under the second click
 *  of a double click. */
export function StopButton({ onStop, stopping, className = "btn sm ghost", children }: {
  onStop: () => unknown; stopping?: boolean; className?: string; children?: ReactNode;
}) {
  const [pressed, setPressed] = useState(false);
  const shown = useRef(Date.now());
  if (pressed || stopping) {
    return <button className={className} disabled aria-busy="true"><Loader2 size={12} className="spin" />Stopping…</button>;
  }
  return <button className={className} onClick={() => {
    if (Date.now() - shown.current < 500) return;
    setPressed(true);
    Promise.resolve(onStop()).catch(() => setPressed(false));
  }}>{children ?? "Stop"}</button>;
}

/** Work that waits on something before it starts (an estimate, a check, an upload, a question): `starting` is on
 *  from the click until the work has started or was called off, and clicks in between do nothing. */
export function useStarting() {
  const busy = useRef(false);
  const [starting, setStarting] = useState(false);
  const run = useCallback(async (work: () => unknown) => {
    if (busy.current) return;
    busy.current = true;
    setStarting(true);
    try { await work(); } finally { busy.current = false; setStarting(false); }
  }, []);
  return { starting, run };
}

// ---------- files dragged in from Finder ----------

/** Does a file fit a file picker's `accept` list (".txt,.md", "audio/*")? */
export function accepts(f: File, accept: string) {
  const name = f.name.toLowerCase();
  return accept.split(",").map((a) => a.trim().toLowerCase()).filter(Boolean).some((a) =>
    a.startsWith(".") ? name.endsWith(a) : a.endsWith("/*") ? f.type.startsWith(a.slice(0, -1)) : f.type === a);
}

type Entry = {
  isFile: boolean; isDirectory: boolean; name: string;
  file?: (ok: (f: File) => void, bad: (e: unknown) => void) => void;
  createReader?: () => { readEntries: (ok: (e: Entry[]) => void, bad: (e: unknown) => void) => void };
};

async function entryFiles(e: Entry): Promise<File[]> {
  if (e.name.startsWith(".")) return [];
  if (e.isFile && e.file) return [await new Promise<File>((ok, bad) => e.file!(ok, bad))];
  if (!e.isDirectory || !e.createReader) return [];
  const reader = e.createReader();
  const all: Entry[] = [];
  for (;;) {  // a folder's entries come a batch at a time
    const batch = await new Promise<Entry[]>((ok, bad) => reader.readEntries(ok, bad));
    if (!batch.length) break;
    all.push(...batch);
  }
  return (await Promise.all(all.map(entryFiles))).flat();
}

const byName = (a: File, b: File) => a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: "base" });

/** The files in a drop, a dropped folder's files included (not hidden ones), sorted by name (Part 2 before Part 10). */
export async function droppedFiles(dt: DataTransfer): Promise<File[]> {
  // read everything before the first await: the drop's items are gone once the event is over
  const files = [...dt.files];
  const entries = [...dt.items].map((i) => (i.kind === "file"
    ? (i as DataTransferItem & { webkitGetAsEntry?: () => Entry | null }).webkitGetAsEntry?.() ?? null : null));
  if (!entries.some((e) => e?.isDirectory)) return files.filter((f) => !f.name.startsWith(".")).sort(byName);
  const found = await Promise.all(entries.map((e, k) => (e ? entryFiles(e) : Promise.resolve(files[k] ? [files[k]] : []))));
  return found.flat().sort(byName);
}

/** An element that takes files dragged in from Finder: spread `props` on it; `over` is on while files are held over it. */
export function useFileDrop(onFiles: (files: File[]) => void, disabled = false) {
  const [over, setOver] = useState(false);
  const depth = useRef(0);  // dragenter/dragleave also fire for the element's children
  const files = (e: DragEvent) => e.dataTransfer.types.includes("Files");
  const props = {
    onDragEnter: (e: DragEvent) => { if (!files(e) || disabled) return; e.preventDefault(); depth.current++; setOver(true); },
    onDragOver: (e: DragEvent) => { if (!files(e) || disabled) return; e.preventDefault(); e.dataTransfer.dropEffect = "copy"; },
    onDragLeave: (e: DragEvent) => {
      if (!files(e) || disabled) return;
      if (--depth.current <= 0) { depth.current = 0; setOver(false); }
    },
    onDrop: (e: DragEvent) => {
      if (!files(e)) return;
      e.preventDefault();
      e.stopPropagation();
      depth.current = 0;
      setOver(false);
      if (!disabled) droppedFiles(e.dataTransfer).then((f) => { if (f.length) onFiles(f); });
    },
  };
  return { over, props };
}

/** What a button says while its work is starting. */
export function Starting({ label = "Starting…", size = 14 }: { label?: string; size?: number }) {
  return <><Loader2 size={size} className="spin" />{label}</>;
}

export function Slider(props: {
  label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void;
  left?: string; right?: string; format?: (v: number) => string;
}) {
  const { label, value, min, max, step, onChange, left, right, format } = props;
  return (
    <label className="slider">
      <div className="slider-top"><b>{label}</b><em>{format ? format(value) : value}</em></div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
      {(left || right) && <div className="slider-ends"><span>{left}</span><span>{right}</span></div>}
    </label>
  );
}

export function Segmented<T extends string>(props: {
  value: T; options: { value: T; label: ReactNode; disabled?: boolean; title?: string }[]; onChange: (v: T) => void;
}) {
  return (
    <div className="segmented" role="radiogroup">
      {props.options.map((o) => (
        <button key={o.value} type="button" role="radio" aria-checked={props.value === o.value} title={o.title}
          className={props.value === o.value ? "on" : ""} disabled={o.disabled}
          style={o.disabled ? { opacity: 0.4, cursor: "not-allowed" } : undefined}
          onClick={() => !o.disabled && props.onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Field(props: { label: string; children: ReactNode; hint?: string }) {
  return (
    <label className="field">
      <span>{props.label}</span>
      {props.children}
      {props.hint && <small className="muted small">{props.hint}</small>}
    </label>
  );
}

/** A task's live progress: bar, status text, and a stop button while it runs. */
/** hideDone: the page shows its own result instead of the final status line. */
/** The time now (ms), updated every second while active: for "about 5 min left" counting down. */
export function useNow(active: boolean, ms = 1000) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    setNow(Date.now());
    const id = window.setInterval(() => setNow(Date.now()), ms);
    return () => window.clearInterval(id);
  }, [active, ms]);
  return now;
}

/** "~5m left" / "<1m left" from seconds (empty when unknown or done). */
export function timeLeft(s?: number | null) {
  if (s == null || !isFinite(s) || s <= 0) return "";
  if (s < 60) return "<1m left";
  const mins = Math.round(s / 60);
  if (mins < 60) return `~${Math.max(1, mins)}m left`;
  const h = Math.floor(mins / 60), m = mins % 60;
  return `~${h}h${m ? ` ${m}m` : ""} left`;
}

/** "4m" / "1h 20m": how long something has taken. */
export function duration(s: number) {
  const secs = Math.max(0, Math.round(s));
  if (secs < 60) return `${secs}s`;
  const mins = Math.round(secs / 60);
  if (mins < 60) return `${mins}m`;
  const h = Math.floor(mins / 60), m = mins % 60;
  return `${h}h${m ? ` ${m}m` : ""}`;
}

/** What a running task's line under its bar says: time left, and a note when it has been quiet a while. */
export function taskTiming(task: Task, now: number) {
  if (task.state !== "running") return { left: "", quiet: "" };
  const left = task.eta != null ? timeLeft(task.eta - (now - (task.etaAt ?? now)) / 1000) : "";
  const since = task.updated ? now / 1000 - task.updated : 0;
  const quiet = since > 90 ? `Still working (${duration(since)})…` : "";
  return { left, quiet };
}

/** Seconds left from how fast a polled progress (0..1) has moved since it was first seen on this page. */
export function useEta(progress: number | null | undefined, active: boolean) {
  const first = useRef<[number, number] | null>(null);
  const now = useNow(active);
  if (!active || progress == null) { first.current = null; return null; }
  if (!first.current || progress < first.current[1] - 0.05) first.current = [Date.now(), progress];
  const [t0, p0] = first.current, dt = (now - t0) / 1000, dp = progress - p0;
  return dt >= 10 && dp >= 0.01 ? (dt * (1 - progress)) / dp : null;
}

export function TaskProgress({ task, onStop, hideDone }: { task: Task | null; onStop?: () => void; hideDone?: boolean }) {
  const now = useNow(task?.state === "running");
  if (!task) return null;
  const { state, progress } = task;
  const { left, quiet } = taskTiming(task, now);
  const cls = state === "done" ? "done" : state === "failed" ? "failed" : progress == null ? "indeterminate" : "";
  const width = state === "done" ? 100 : Math.round((progress ?? 0) * 100);
  const text = state === "failed" ? task.error || "Something went wrong."
    : state === "done" ? (hideDone ? "" : task.status.replace(/^Done · /, "")) : task.status;
  return (
    <div className="progress">
      <div className="row">
        {state === "done" && <CheckCircle2 size={16} color="var(--ok)" />}
        {state === "failed" && <AlertCircle size={16} color="var(--danger)" />}
        <strong className="small">{state === "running" ? "Working…" : state === "done" ? "Done" : state === "stopped" ? "Stopped" : "Couldn't finish"}</strong>
        <span className="spacer" />
        {state === "running" && (progress != null || left) && <span className="small muted">{[progress != null ? `${width}%` : "", left].filter(Boolean).join(" · ")}</span>}
        {state === "running" && onStop && task.id && <StopButton onStop={onStop} stopping={task.stopping} />}
      </div>
      <div className={`bar ${cls}`}><i style={{ width: `${width}%` }} /></div>
      {text && <div className="status-text">{firstLines(text)}</div>}
      {quiet && <div className="small faint">{quiet}</div>}
    </div>
  );
}

// file paths aren't for reading: the page shows the result with a player and a download button
const firstLines = (s: string) => s.split("\n").filter((l) => !l.startsWith("/") && !l.startsWith("Saved to ")).slice(0, 4).join("\n");

/** A colored orb for a voice, its hue from the voice's id (the same voice always gets the same color). */
export function Orb({ value, size }: { value: string; size?: number }) {
  let h = 0;
  for (const c of value) h = (h * 31 + c.charCodeAt(0)) % 360;
  const style = { background: `radial-gradient(circle at 30% 30%, hsl(${h} 90% 75%), hsl(${(h + 40) % 360} 70% 50%))`, ...(size ? { width: size, height: size } : {}) };
  return <span className="orb" style={style} />;
}

/** "Make several lines at once" and how many: the same switch on Audiobook and Text to Speech. */
export function ParallelLines({ on, count, setOn, setCount }: { on: boolean; count: number; setOn: (v: boolean) => void; setCount: (n: number) => void }) {
  return (
    <div className="stack" style={{ gap: 10 }}>
      <Switch label="Make several lines at once" checked={on} onChange={setOn}
        hint={on ? "Faster, but uses more memory. Qwen and Orpheus voices make lines side by side; Kokoro and the others go one at a time."
          : "One line at a time with every voice: slower, and the least memory."} />
      {on && <Slider label="Lines at once" value={Math.max(2, count)} min={2} max={8} step={1} onChange={setCount}
        left="Less memory" right="Faster" format={(v) => `${v}`} />}
    </div>
  );
}

/** A confirmation in front of the page. Cancel has the focus; Escape or a click outside cancels too. */
export function ConfirmDialog(props: {
  title: string; children: ReactNode; confirm: string; danger?: boolean; onConfirm: () => void; onCancel: () => void;
  /** a second choice besides Confirm (Cancel still backs out) */
  other?: string; onOther?: () => void;
  /** what backing out is called ("Keep it going", "Not now"): Cancel by default */
  cancel?: string;
}) {
  const cancelBtn = useRef<HTMLButtonElement>(null);
  const cancel = useRef(props.onCancel);
  cancel.current = props.onCancel;
  useEffect(() => {
    const before = document.activeElement as HTMLElement | null;
    cancelBtn.current?.focus();
    const key = (e: KeyboardEvent) => { if (e.key === "Escape") { e.stopPropagation(); cancel.current(); } };
    window.addEventListener("keydown", key, true);
    return () => { window.removeEventListener("keydown", key, true); before?.focus?.(); };
  }, []);
  return createPortal(
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && props.onCancel()}>
      <div className="modal" role="alertdialog" aria-modal="true" aria-labelledby="confirm-title" aria-describedby="confirm-body">
        <h2 id="confirm-title">{props.title}</h2>
        <div id="confirm-body" className="modal-body">{props.children}</div>
        <div className="modal-actions">
          <button ref={cancelBtn} className="btn" onClick={props.onCancel}>{props.cancel || "Cancel"}</button>
          {props.other && <button className="btn" onClick={props.onOther}>{props.other}</button>}
          <button className={`btn ${props.danger ? "danger-solid" : "primary"}`} onClick={props.onConfirm}>{props.confirm}</button>
        </div>
      </div>
    </div>,
    document.body,
  );
}

type ToastMsg = { text: string; error?: boolean } | null;
const ToastCtx = createContext<(text: string, error?: boolean) => void>(() => {});
export const useToast = () => useContext(ToastCtx);

export function ToastHost({ children }: { children: ReactNode }) {
  const [msg, setMsg] = useState<ToastMsg>(null);
  const show = useCallback((text: string, error?: boolean) => {
    setMsg({ text, error });
    window.setTimeout(() => setMsg((m) => (m?.text === text ? null : m)), error ? 7000 : 3500);
  }, []);
  return (
    <ToastCtx.Provider value={show}>
      {children}
      {msg && (
        <div className={`toast ${msg.error ? "error" : ""}`} role="status">
          <span style={{ flex: 1 }}>{msg.text}</span>
          <button className="btn icon sm ghost" style={{ color: "inherit" }} onClick={() => setMsg(null)} aria-label="Close"><X size={14} /></button>
        </div>
      )}
    </ToastCtx.Provider>
  );
}

export function clock(seconds: number | null | undefined) {
  if (seconds == null) return "";
  const s = Math.floor(seconds); // like the engine's clock(): rounded down, hours past an hour
  const mm = (n: number) => String(n).padStart(2, "0");
  return s >= 3600 ? `${Math.floor(s / 3600)}:${mm(Math.floor(s / 60) % 60)}:${mm(s % 60)}` : `${Math.floor(s / 60)}:${mm(s % 60)}`;
}

export function Switch(props: { label: ReactNode; hint?: ReactNode; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="switch-row">
      <span className="grow">
        <span className="switch-label">{props.label}</span>
        {props.hint && <span className="small muted" style={{ display: "block" }}>{props.hint}</span>}
      </span>
      <input type="checkbox" className="switch" checked={props.checked} onChange={(e) => props.onChange(e.target.checked)} />
    </label>
  );
}
