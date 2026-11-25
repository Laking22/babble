import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Check, ChevronDown, Clapperboard, Cloud, Cpu, Laptop, Loader2, Search, Sparkles } from "lucide-react";
import { directorApi, soundsApi, useLoad, type AiService, type DirectorEstimate, type DirectorModel } from "../../api";
import { Field, Slider, Starting, Switch, useStarting } from "../../components/ui";
import { useCostConfirm } from "../../components/CostConfirm";
import { ReviewBox, keptPlacements, type ReviewState } from "./SoundTools";

const CHECKS = ["Loosest", "Loose", "Balanced", "Strict", "Strictest"];
// scene size on a service (director.SCENE_SIZES), shown so more detail is to the right like the other sliders
const SIZES = ["large", "medium", "small"];
const SIZE_NAMES = ["Large", "Medium", "Small"];

function splitLabel(label: string): [string, string] {
  const i = label.indexOf(" · ");
  return i >= 0 ? [label.slice(0, i), label.slice(i + 3)] : [label, ""];
}

function shortPrice(m: DirectorModel): string {
  if (m.free) return "Free";
  if (m.price_in != null && m.price_out != null) {
    const fmt = (x: number) => (x === 0 ? "$0" : x >= 1 && Math.abs(x - Math.round(x)) < 1e-6 ? `$${Math.round(x)}` : x >= 0.01 ? `$${x.toFixed(2)}` : `$${x.toFixed(3)}`);
    return `${fmt(m.price_in)} / ${fmt(m.price_out)}`;
  }
  return m.free_tier ? "Free tier" : "—";
}

function costBadge(cost: AiService["cost"]): { text: string; cls: string } {
  if (cost === "free tier") return { text: "Free tier", cls: "chip ok" };
  if (cost === "free models") return { text: "Free models", cls: "chip ok" };
  return { text: "Paid API", cls: "chip" };
}

/** Custom Babble-styled dropdown for picking where the Director runs (this Mac or a connected AI service). */
function ProviderPicker({ value, ready, onChange }: {
  value: string; ready: AiService[]; onChange: (id: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const anchor = useRef<HTMLButtonElement>(null);
  const [pos, setPos] = useState({ top: 0, left: 0, width: 332 });
  const current = ready.find((s) => s.id === value);

  useEffect(() => {
    if (!open || !anchor.current) return;
    const place = () => {
      if (!anchor.current) return;
      const r = anchor.current.getBoundingClientRect();
      const width = Math.min(Math.max(r.width, 310), window.innerWidth - 24);
      const maxH = Math.min(360, window.innerHeight - 24);
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

  return (
    <>
      <button ref={anchor} type="button" className="voice-btn dir-btn" onClick={() => setOpen(true)}>
        <span className={`kind-icon sm ${current ? "effect" : ""}`}>
          {current ? <Cloud size={14} /> : <Laptop size={14} />}
        </span>
        <span className="voice-meta">
          <strong>{current ? current.name : "This Mac"}</strong>
          <span>{current ? (current.cost === "paid" ? "Billed per use" : current.cost === "free tier" ? "Free tier with daily limits" : "Free & paid models") : "Free · local, simpler plans"}</span>
        </span>
        <span className="spacer" />
        {current ? <span className={costBadge(current.cost).cls}>{costBadge(current.cost).text}</span>
          : <span className="chip ok">Free</span>}
        <ChevronDown size={15} className="muted" />
      </button>
      {open && createPortal(
        <>
          <div className="popover-backdrop" onClick={() => setOpen(false)} />
          <div className="popover dir-pop" style={{ top: pos.top, left: pos.left, width: pos.width, position: "fixed" }} role="listbox">
            <div className="popover-list">
              <div className="voice-group">Local</div>
              <button type="button" className={`voice-opt ${value === "mac" ? "sel" : ""}`}
                onClick={() => { onChange("mac"); setOpen(false); }}>
                <span className="kind-icon sm"><Laptop size={14} /></span>
                <span className="voice-meta">
                  <strong>This Mac</strong>
                  <span>Simpler plans · runs offline</span>
                </span>
                <span className="spacer" />
                <span className="chip ok">Free</span>
                {value === "mac" && <Check size={14} className="dir-check" />}
              </button>
              {ready.length > 0 && <div className="voice-group">Connected AI services</div>}
              {ready.map((s) => {
                const badge = costBadge(s.cost);
                const sel = value === s.id;
                return (
                  <button key={s.id} type="button" className={`voice-opt ${sel ? "sel" : ""}`}
                    onClick={() => { onChange(s.id); setOpen(false); }}>
                    <span className="kind-icon sm effect"><Cloud size={14} /></span>
                    <span className="voice-meta">
                      <strong>{s.name}</strong>
                      <span>{s.cost_note}</span>
                    </span>
                    <span className="spacer" />
                    <span className={badge.cls}>{badge.text}</span>
                    {sel && <Check size={14} className="dir-check" />}
                  </button>
                );
              })}
            </div>
          </div>
        </>,
        document.body,
      )}
    </>
  );
}

/** Custom Babble-styled searchable dropdown for Director models, showing each model's price per 1M tokens. */
function ModelPicker({ models, value, onChange }: {
  models: DirectorModel[] | null; value: string; onChange: (id: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const anchor = useRef<HTMLButtonElement>(null);
  const [pos, setPos] = useState({ top: 0, left: 0, width: 332 });
  const current = (models || []).find((m) => m.id === value);
  const [curName, curNote] = splitLabel(current?.label || value || "Choose a model");

  useEffect(() => {
    if (!open || !anchor.current) return;
    setQ("");
    const place = () => {
      if (!anchor.current) return;
      const r = anchor.current.getBoundingClientRect();
      const width = Math.min(Math.max(r.width, 336), window.innerWidth - 24);
      const maxH = Math.min(440, window.innerHeight - 24);
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
    const list = models || [];
    const words = q.toLowerCase().split(/\s+/).filter(Boolean);
    if (!words.length) return list;
    return list.filter((m) => words.every((w) => `${m.id} ${m.label} ${m.price_label || ""}`.toLowerCase().includes(w)));
  }, [models, q]);

  const loading = models === null;
  const showSearch = (models?.length || 0) > 5;

  return (
    <>
      <button ref={anchor} type="button" className="voice-btn dir-btn" disabled={loading} onClick={() => setOpen(true)}>
        <span className="kind-icon sm music">
          {loading ? <Loader2 size={14} className="spin" /> : <Sparkles size={14} />}
        </span>
        <span className="voice-meta">
          <strong>{loading ? "Loading models…" : curName}</strong>
          <span>
            {loading ? "Fetching available models" : current?.price_label
              ? (curNote ? `${curNote} · ${current.price_label}` : current.price_label)
              : curNote || "Select a model"}
          </span>
        </span>
        <span className="spacer" />
        {!loading && current && (
          <span className={`chip tnum ${current.free ? "ok" : ""}`}>{shortPrice(current)}</span>
        )}
        <ChevronDown size={15} className="muted" />
      </button>
      {open && createPortal(
        <>
          <div className="popover-backdrop" onClick={() => setOpen(false)} />
          <div className="popover dir-pop" style={{ top: pos.top, left: pos.left, width: pos.width, position: "fixed" }} role="listbox">
            {showSearch && (
              <div className="popover-search">
                <div className="row" style={{ position: "relative" }}>
                  <Search size={15} style={{ position: "absolute", left: 10 }} className="muted" />
                  <input autoFocus className="input" style={{ paddingLeft: 32 }} placeholder="Search models or “free”" value={q}
                    onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Escape" && setOpen(false)} />
                </div>
              </div>
            )}
            <div className="dir-pop-head">
              <span>Model</span>
              <span>In / out per 1M tokens</span>
            </div>
            <div className="popover-list">
              {filtered.map((m) => {
                const [name, note] = splitLabel(m.label);
                const sel = m.id === value;
                return (
                  <button key={m.id} type="button" className={`voice-opt dir-opt ${sel ? "sel" : ""}`}
                    onClick={() => { onChange(m.id); setOpen(false); }}>
                    <span className="kind-icon sm"><Cpu size={13} /></span>
                    <span className="voice-meta">
                      <strong>{name}</strong>
                      <span>{note || (m.free_tier && !m.free ? "Free tier available" : m.id !== name ? m.id : "")}</span>
                    </span>
                    <span className="spacer" />
                    <span className={`chip tnum ${m.free ? "ok" : sel ? "accent" : ""}`}>{shortPrice(m)}</span>
                    {sel && <Check size={14} className="dir-check" />}
                  </button>
                );
              })}
              {!filtered.length && <div className="empty small">No models match “{q}”.</div>}
            </div>
          </div>
        </>,
        document.body,
      )}
    </>
  );
}

/** The Director tab: an AI plans each scene's backgrounds, effects and music (narrator/director.py), on an AI service
 *  with your own key (Connections) or on this Mac with the small local model. Its picks come back here to review. */
export function DirectorTool({ story, server, saveServer, busy, onDirect, review, setReview, onAddKept, notes, setNotes }: {
  story: string; notes: string; setNotes: (v: string) => void; server: Record<string, any>; saveServer: (k: string, v: any) => void; busy: boolean;
  onDirect: (where: string, model: string, checks: number, scenes: string) => void;
  review: ReviewState; setReview: (r: ReviewState) => void; onAddKept: () => void;
}) {
  const services = useLoad(directorApi.providers);
  const confirmCost = useCostConfirm();
  const direct = useStarting();  // (the estimate and the question come before it starts)
  const libs = useLoad(soundsApi.sources);
  const libsReady = (libs.data || []).some((s) => s.has_key);
  const ready = (services.data || []).filter((s) => s.has_key);
  const saved = server.director_where as string | undefined;
  const where = saved === "mac" || ready.some((s) => s.id === saved) ? saved! : ready[0]?.id ?? "mac";
  const svc = ready.find((s) => s.id === where);
  const [modelState, setModelState] = useState<{ provider: string; list: DirectorModel[] } | null>(null);
  useEffect(() => {
    if (!svc) { setModelState(null); return; }
    if (svc.models) { setModelState({ provider: svc.id, list: svc.models }); return; }
    let live = true;
    const pid = svc.id;
    setModelState(null);
    directorApi.models(pid).then(
      (list) => { if (live) setModelState({ provider: pid, list }); },
      () => { if (live) setModelState({ provider: pid, list: [] }); },
    );
    return () => { live = false; };
  }, [svc?.id]); // eslint-disable-line react-hooks/exhaustive-deps
  const models = svc ? (svc.models ?? (modelState?.provider === svc.id ? modelState.list : null)) : null;
  const pref = (server.director_models || {})[where] || svc?.model || "";
  // a saved model the service no longer offers (or that can't be used) gives way to the first one listed
  const chosen = models ? (models.some((m) => m.id === pref) ? pref : models[0]?.id || "") : pref;
  const size = SIZES.includes(server.director_scenes) ? server.director_scenes as string : "small";
  const estKey = `${story.length}:${where}:${chosen}:${size}`;
  const [estState, setEstState] = useState<{ key: string; data: DirectorEstimate } | null>(null);
  const est = estState?.key === estKey ? estState.data : null;
  useEffect(() => {
    if (!story.trim()) { setEstState(null); return; }
    let live = true;
    const k = estKey;
    const t = setTimeout(() => directorApi.estimate(story, where, chosen, size).then(
      (r) => live && setEstState({ key: k, data: r }),
      () => live && setEstState(null),
    ), 700);
    return () => { live = false; clearTimeout(t); };
  }, [estKey, story, where, chosen, size]);
  const scenes = est ? `${est.scenes} scene${est.scenes === 1 ? "" : "s"}` : "";
  const cost = !est || !est.scenes ? "" : where === "mac" ? `${scenes} · about ${est.minutes} min`
    : est.cost && est.cost !== "price unknown" ? `${scenes} · ${est.cost}`
      : `${scenes} · about ${Math.round((est.tokens || 0) / 1000)}k tokens`;
  const checks = server.director_checks ?? 3;
  const mine = review?.source === "director" ? review : null;
  const toMake = mine ? new Set(keptPlacements(mine).filter((p) => p.missing).map((p) => p.name)).size
    + (server.director_intros && mine.chapterPlan ? Object.keys(mine.chapterPlan.variants).length + Object.keys(mine.chapterPlan.outros).length : 0) : 0;
  return (
    <div className="stack">
      <Field label="Runs on">
        <ProviderPicker value={where} ready={ready} onChange={(v) => saveServer("director_where", v)} />
      </Field>
      {svc && (
        <Field label="Model">
          <ModelPicker models={models} value={chosen}
            onChange={(v) => saveServer("director_models", { ...(server.director_models || {}), [where]: v })} />
        </Field>
      )}
      {services.data && !ready.length && <p className="small muted" style={{ margin: 0 }}>For better plans, add an AI key in Connections.</p>}
      <Field label="Notes for the director">
        <textarea className="input" rows={3} value={notes} maxLength={1500} placeholder="dark gloomy spooky"
          onChange={(e) => setNotes(e.target.value)} style={{ resize: "vertical", width: "100%" }} />
      </Field>
      {/* shown reversed (director.CHECKS is 1 loose … 5 strict) so more sounds is to the right, like the volumes */}
      <Slider label="Checks" value={6 - checks} min={1} max={5} step={1} format={(v) => CHECKS[5 - v]}
        onChange={(v) => saveServer("director_checks", 6 - v)} left="Only clear fits" right="More sounds" />
      {svc && <Slider label="Scene size" value={SIZES.indexOf(size) + 1} min={1} max={3} step={1}
        format={(v) => SIZE_NAMES[v - 1]} onChange={(v) => saveServer("director_scenes", SIZES[v - 1])}
        left="Fewer requests" right="More detail" />}
      <Switch label="Query sound libraries" checked={server.sound_libraries !== false}
        onChange={(v) => saveServer("sound_libraries", v)}
        hint={libs.data && !libsReady ? "Connect one in Connections → Sound libraries." : undefined} />
      <Switch label="Make chapter intros and outros" checked={!!server.director_intros}
        onChange={(v) => saveServer("director_intros", v)} hint={server.director_intros ? "Replaces what's set in Intros." : undefined} />
      <button className="btn primary" disabled={busy || direct.starting || (!!svc && !chosen)} onClick={() => direct.run(async () => {
        if (svc) {  // a service: money, or a free tier's daily requests
          const e = est || await directorApi.estimate(story, where, chosen, size).catch(() => null);
          const label = (models || []).find((m) => m.id === chosen)?.label || chosen;
          const ok = await confirmCost({
            title: `Direct with ${svc.name}?`, confirm: "Direct",
            cost: e?.cost || "price unknown",
            details: [label, e ? `${e.scenes} scene${e.scenes === 1 ? "" : "s"} · ${e.requests ?? e.scenes + 1} requests` : "",
              svc.cost === "free tier" ? "On a free tier these count toward its daily request limit." : ""],
          });
          if (!ok) return;
        }
        onDirect(where, chosen, checks, size);
      })}>
        {direct.starting ? <Starting /> : <><Clapperboard size={14} />Direct sounds</>}</button>
      {cost && <p className="small muted" style={{ margin: 0 }}>{cost}</p>}
      {mine && <ReviewBox review={mine} setReview={setReview} onAdd={onAddKept}
        addLabel={toMake ? `Add · makes ${toMake} sound${toMake === 1 ? "" : "s"}` : "Add"} />}
    </div>
  );
}
