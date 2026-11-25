import { useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, AudioLines, BookOpen, Cloud, Loader2, Music, Pause, Play, Sparkles, Trash2, Wind, Wrench } from "lucide-react";
import { activityApi, fileUrl, type ActivityEntry, type ActivityKind } from "../api";
import { ConfirmDialog, duration, useToast } from "../components/ui";

const KIND: Record<ActivityKind, { label: string; icon: typeof BookOpen }> = {
  audiobook: { label: "Audiobooks", icon: BookOpen },
  speech: { label: "Text to speech", icon: AudioLines },
  effect: { label: "Sound effects", icon: Sparkles },
  background: { label: "Backgrounds", icon: Wind },
  music: { label: "Music", icon: Music },
  cloud: { label: "Cloud", icon: Cloud },
  other: { label: "Other", icon: Wrench },
};
type Filter = "all" | ActivityKind | "failed";

/** "3 min ago", "yesterday 14:05", "12 Nov 09:30". */
function when(t?: number) {
  if (!t) return "";
  const s = Date.now() / 1000 - t;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  const d = new Date(t * 1000);
  const time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const today = new Date();
  if (d.toDateString() === today.toDateString()) return `today ${time}`;
  const y = new Date(today); y.setDate(today.getDate() - 1);
  if (d.toDateString() === y.toDateString()) return `yesterday ${time}`;
  return `${d.toLocaleDateString([], { day: "numeric", month: "short" })} ${time}`;
}

const money = (n?: number | null) => (n == null ? null : n === 0 ? "free" : `$${n.toFixed(2)}`);
const AUDIO = /\.(wav|mp3|m4a|m4b|flac|ogg|opus|aac)$/i;

/** Recent activity: audiobooks, text to speech, sounds and music, cloud runs (with what they cost) and failures. */
export function Activity() {
  const toast = useToast();
  const [data, setData] = useState<{ running: ActivityEntry[]; entries: ActivityEntry[] } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>("all");
  const [asking, setAsking] = useState(false);
  const load = () => activityApi.list().then((d) => { setData(d); setError(null); }, (e) => setError((e as Error).message));
  useEffect(() => {
    load();
    const tick = () => { if (!document.hidden) load(); };
    const t = setInterval(tick, 5000);
    const vis = () => { if (!document.hidden) load(); };
    document.addEventListener("visibilitychange", vis);
    return () => { clearInterval(t); document.removeEventListener("visibilitychange", vis); };
  }, []);

  // one player for every Play button
  const el = useRef<HTMLAudioElement | null>(null);
  const [on, setOn] = useState<string | null>(null);
  useEffect(() => () => el.current?.pause(), []);
  const play = (path: string) => {
    if (!el.current) { el.current = new Audio(); el.current.onended = () => setOn(null); }
    if (on === path) { el.current.pause(); setOn(null); return; }
    el.current.src = fileUrl(path);
    el.current.play().then(() => setOn(path), () => toast("Couldn't play that (the file may have been moved or deleted).", true));
  };

  const all = useMemo(() => {
    const saved = data?.entries ?? [];
    const ids = new Set(saved.map((e) => e.id));
    return [...(data?.running ?? []).filter((e) => !ids.has(e.id)), ...saved];
  }, [data]);
  const count = (f: Filter) => all.filter((e) => (f === "all" ? true : f === "failed" ? e.state === "failed" : e.kind === f)).length;
  const shown = all.filter((e) => (filter === "all" ? true : filter === "failed" ? e.state === "failed" : e.kind === filter));
  const filters: Filter[] = ["all", "audiobook", "speech", "effect", "background", "music", "cloud", "failed",
    ...(count("other") ? (["other"] as Filter[]) : [])];
  const spent = all.reduce((sum, e) => sum + (e.kind === "cloud" && e.cost ? e.cost : 0), 0);

  const clear = async () => {
    try { await activityApi.clear(); toast("Activity cleared. The files that were made are kept."); load(); }
    catch (e) { toast((e as Error).message, true); }
  };

  const row = (e: ActivityEntry) => {
    const K = KIND[e.kind] ?? KIND.other;
    const took = e.started && e.ended ? e.ended - e.started : null;
    const cost = e.kind === "cloud"
      ? e.state === "running"
        ? (e.price != null ? `${money(e.price)}${e.price ? "/h" : ""}${e.max_cost ? ` · at most ${money(e.max_cost)}` : ""}` : null)
        : money(e.cost) && `${money(e.cost)}${e.gpu_hours != null ? ` · ${duration(e.gpu_hours * 3600)} of GPU time` : ""}`
      : null;
    const audio = e.file && AUDIO.test(e.file) ? e.file : null;
    return (
      <div key={e.id} className={`activity-row ${e.state}`}>
        <div className={`activity-icon ${e.state}`}>
          {e.state === "running" ? <Loader2 size={15} className="spin" /> : e.state === "failed" ? <AlertTriangle size={15} /> : <K.icon size={15} />}
        </div>
        <div className="grow" style={{ minWidth: 0 }}>
          <div className="row" style={{ gap: 8 }}>
            <b className="activity-title">{e.title}</b>
            <span className="chip">{K.label}</span>
            {e.state === "failed" && <span className="chip warn">Failed</span>}
            {e.state === "stopped" && <span className="chip">Stopped</span>}
            {e.state === "running" && <span className="chip accent">{e.progress != null ? `${Math.round(e.progress * 100)}%` : "Working"}</span>}
          </div>
          {e.state === "failed" && e.error
            ? <div className="small error-text activity-error">{e.error}</div>
            : e.status && <div className="small muted activity-status">{e.status.split("\n")[0]}</div>}
          {e.state !== "failed" && e.error && <div className="small error-text activity-error">{e.error}</div>}
          <div className="small faint">
            {[when(e.ended ?? e.started), took != null && took >= 1 ? `took ${duration(took)}` : null,
              e.state === "running" && e.eta ? `about ${duration(e.eta)} left` : null, cost].filter(Boolean).join(" · ")}
          </div>
        </div>
        {audio && (
          <button className="btn sm ghost icon" onClick={() => play(audio)} aria-label={on === audio ? "Pause" : "Play"} title={audio}>
            {on === audio ? <Pause size={13} /> : <Play size={13} />}
          </button>
        )}
      </div>
    );
  };

  return (
    <div className="page narrow">
      <div className="page-head">
        <div><h1>Activity</h1></div>
        <button className="btn" disabled={!all.length} onClick={() => setAsking(true)}><Trash2 size={14} />Clear activity</button>
      </div>
      <div className="activity-filters" role="tablist" aria-label="Show">
        {filters.map((f) => (
          <button key={f} role="tab" aria-selected={filter === f} className={filter === f ? "on" : ""} onClick={() => setFilter(f)}>
            {f === "all" ? "All" : f === "failed" ? "Failed" : KIND[f].label}
            <span className="activity-count">{count(f)}</span>
          </button>
        ))}
      </div>
      {filter === "cloud" && spent > 0 && <p className="small muted" style={{ margin: "0 0 10px" }}>Cloud runs listed here cost about <b>{money(spent)}</b> in all (an estimate from each GPU's hourly price: your provider's bill is what counts).</p>}
      {error && <div className="card card-pad empty">{error}</div>}
      {!error && data && !shown.length && (
        <div className="card card-pad empty">{all.length ? "Nothing of this kind yet." : "Nothing yet. Audiobooks, speech, sounds and cloud runs show up here as you make them."}</div>
      )}
      {!data && !error && <div className="card card-pad empty"><Loader2 size={15} className="spin" /> Loading…</div>}
      {shown.length > 0 && <div className="card activity-list">{shown.map(row)}</div>}
      {asking && (
        <ConfirmDialog title="Clear activity?" confirm="Clear activity" danger onCancel={() => setAsking(false)}
          onConfirm={() => { setAsking(false); clear(); }}>
          <p>Empties this list. The audiobooks, speech and sounds that were made stay where they are, and anything running keeps going.</p>
        </ConfirmDialog>
      )}
    </div>
  );
}
