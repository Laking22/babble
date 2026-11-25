import { useEffect, useRef, useState } from "react";
import { Check, Cloud, Laptop, Menu } from "lucide-react";
import { bookApi, connectionApi, type Check as Conn, type CloudGpu, type CloudEstimate, type CloudJob, type CloudService, type BookRequest } from "../../api";
import { openExternal } from "../../native";
import { StopButton, duration, timeLeft } from "../../components/ui";

export type MakeOn = string;
/** Modal GPUs: id, label and US$ an hour (Modal's list prices). */
export const MODAL_GPUS: [string, string, number][] = [
  ["L4", "L4", 0.8], ["A10G", "A10", 1.1], ["L40S", "L40S", 1.95], ["A100-40GB", "A100", 2.1]];
/** Every cloud Create can use, by id (the pay-as-you-go ones come with their GPUs from /connections/clouds). */
export const CLOUD_NAMES: Record<string, string> = {
  kaggle: "Kaggle", modal: "Modal", runpod: "Runpod", vast: "Vast.ai", thunder: "Thunder Compute", lambda: "Lambda",
  lightning: "Lightning AI", beam: "Beam", cerebrium: "Cerebrium", hfjobs: "Hugging Face Jobs" };
/** "NVIDIA GeForce RTX 4090" -> "RTX 4090" */
const shortGpu = (id?: string) => (id || "").replace(/^NVIDIA (GeForce )?/, "");
const usd = (n: number) => "$" + n.toFixed(2);
export const WAITING = ["sent", "queued", "running", "downloading"];

/** The ≡ next to Create: make the book on this Mac, or have its lines made on a cloud GPU first. Kaggle and Modal
 *  are always listed; a pay-as-you-go service once its key is saved on the Connections page. */
const hrs = (h: number) => h < 1 ? `${Math.max(5, Math.round(h * 60 / 5) * 5)} min` : `${h.toFixed(1)} h`;

/** "About 20 min to 1.5 h · $0.15 to $1.10" (Kaggle: hours of the free weekly allowance). */
const gb = (b: number) => `${(b / 1e9).toFixed(b < 1e10 ? 1 : 0)} GB`;

export function estimateText(e: CloudEstimate) {
  if (!e.lines) return e.why || "Every chapter is made already.";
  const time = `${hrs(e.low_hours)}–${hrs(e.high_hours)}`;
  const cost = e.free ? "free" : e.low_cost != null && e.high_cost != null ? `${usd(e.low_cost)}–${usd(e.high_cost)}` : "price unknown";
  const runs = (e.runs ?? 1) > 1 ? ` · ${e.runs} runs` : "";
  const snd = e.sounds || e.mac_sounds ? ` · ${e.sounds || e.mac_sounds} sounds` : "";
  const up = e.upload ? ` · ${e.upload.gb.toFixed(1)} GB upload` : "";
  return `${e.lines.toLocaleString()} lines${snd} · ${time} · ${cost}${runs}${up}`;
}

export function MakeOnMenu({ value, onChange, gpu, onGpu, request, ai, onAi, live, onLive, hours, onHours, sounds, onSounds }: {
  value: MakeOn; onChange: (v: MakeOn) => void; gpu: string; onGpu: (g: string) => void;
  /** the book as Create would send it, for the cost estimate (saved projects only) */
  request?: () => BookRequest | null;
  /** whether Detect speakers (AI) also runs on the chosen cloud GPU */
  ai?: { detect: boolean }; onAi?: (k: "detect", v: boolean) => void;
  /** bring the lines back as they're made (where the chosen cloud allows it) */
  live?: boolean; onLive?: (v: boolean) => void;
  /** the most one run may take (3, 6 or 12 h), on every cloud but Kaggle (it has its own weekly hours) */
  hours?: number; onHours?: (h: number) => void;
  /** where the book's missing AI sounds are made: with the lines in the cloud (true), or on this Mac */
  sounds?: boolean; onSounds?: (v: boolean) => void;
}) {
  const [est, setEst] = useState<{ key: string; e?: CloudEstimate; err?: string } | null>(null);
  const [open, setOpen] = useState(false);
  const [kaggle, setKaggle] = useState<Conn | null>(null);
  const [modal, setModal] = useState<Conn | null>(null);
  const [services, setServices] = useState<CloudService[] | null>(null);
  const [checks, setChecks] = useState<Record<string, Conn>>({});
  const [gpus, setGpus] = useState<Record<string, CloudGpu[]>>({});
  const fail = (e: unknown): Conn => ({ ok: false, detail: (e as Error).message });
  useEffect(() => {
    if (!open) return;
    if (!kaggle) connectionApi.check("kaggle").then(setKaggle).catch((e) => setKaggle(fail(e)));
    if (!modal) connectionApi.check("modal").then(setModal).catch((e) => setModal(fail(e)));
    if (!services) connectionApi.clouds().then((list) => {
      setServices(list);
      for (const sv of list) if (sv.has_key || sv.id === value)
        connectionApi.check(sv.id).then((c) => setChecks((x) => ({ ...x, [sv.id]: c })))
          .catch((e) => setChecks((x) => ({ ...x, [sv.id]: fail(e) })));
    }).catch(() => setServices([]));
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps
  const service = services?.find((sv) => sv.id === value);
  useEffect(() => {  // the chosen service's GPUs with today's prices, once its key is good
    if (open && service && checks[service.id]?.ok && !gpus[service.id])
      connectionApi.cloudGpus(service.id).then((g) => setGpus((x) => ({ ...x, [service.id]: g })))
        .catch(() => setGpus((x) => ({ ...x, [service.id]: service.gpus })));
  }, [open, service, checks]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {  // what the chosen cloud would take and cost for this book
    if (!open || value === "mac" || !request) return;
    const body = request();
    if (!body?.folder) { setEst(null); return; }
    const key = `${value}|${gpu}`;
    setEst({ key });
    const t = setTimeout(() => bookApi.cloudEstimate({ ...body, make_on: value, cloud_gpu: gpu })
      .then((e) => setEst((x) => x?.key === key ? { key, e } : x))
      .catch((err) => setEst((x) => x?.key === key ? { key, err: (err as Error).message } : x)), 250);
    return () => clearTimeout(t);
  }, [open, value, gpu, hours]); // eslint-disable-line react-hooks/exhaustive-deps
  const chosen = MODAL_GPUS.find((g) => g[0] === gpu) ?? MODAL_GPUS[1];
  useEffect(() => {
    if (!open) return;
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [open]);
  const q = kaggle?.quota;
  const pick = (v: MakeOn) => {
    onChange(v);
    // each cloud has its own GPUs: keep a choice that fits it
    if (v === "modal" && !MODAL_GPUS.some((g) => g[0] === gpu)) onGpu("A10G");
    const sv = services?.find((x) => x.id === v);
    if (sv && !sv.gpus.some((g) => g.id === gpu)) onGpu(sv.default_gpu);
    if (v === "mac") setOpen(false);
  };
  const listed = (services || []).filter((sv) => sv.has_key || sv.id === value);  return (
    <div className="make-on">
      <button className={`btn sm icon ${open ? "active" : ""}`} onClick={() => setOpen(!open)} aria-haspopup="menu" aria-expanded={open}
        title="Where to make it: this Mac or a cloud GPU" aria-label="Where to make it"><Menu size={14} /></button>
      {open && (
        <>
          <div className="popover-backdrop" onClick={() => setOpen(false)} />
          <div className="popover make-on-pop" role="menu">
            <div className="make-on-head small muted">Make it on</div>
            <button role="menuitemradio" aria-checked={value === "mac"} className={`make-on-item ${value === "mac" ? "on" : ""}`} onClick={() => pick("mac")}>
              <Laptop size={16} />
              <span className="grow"><b>This Mac</b><span className="small muted">Everything is made here, with this Mac's memory and speed.</span></span>
              {value === "mac" && <Check size={15} />}
            </button>
            <button role="menuitemradio" aria-checked={value === "kaggle"} className={`make-on-item ${value === "kaggle" ? "on" : ""}`} onClick={() => pick("kaggle")}>
              <Cloud size={16} />
              <span className="grow"><b>Cloud GPU · Kaggle</b>
                <span className="small muted">{!kaggle ? "Checking your Kaggle account…"
                  : kaggle.ok ? `Free with your account${q ? ` · ${q.left.toFixed(1)} of ${Math.round(q.total)} GPU hours left this week` : ""}`
                    : kaggle.detail}</span></span>
              {value === "kaggle" && <Check size={15} />}
            </button>
            <button role="menuitemradio" aria-checked={value === "modal"} className={`make-on-item ${value === "modal" ? "on" : ""}`} onClick={() => pick("modal")}>
              <Cloud size={16} />
              <span className="grow"><b>Cloud GPU · Modal</b>
                <span className="small muted">{!modal ? "Checking your Modal account…"
                  : modal.ok ? `Pay by the second with your account · ${chosen[1]} about ${usd(chosen[2])} an hour`
                    : modal.detail}</span></span>
              {value === "modal" && <Check size={15} />}
            </button>
            {value === "modal" && (
              <div className="make-on-gpus" role="radiogroup" aria-label="Modal GPU">
                {MODAL_GPUS.map(([id, label, price]) => (
                  <button key={id} role="radio" aria-checked={gpu === id} className={`chip-btn ${gpu === id ? "on" : ""}`}
                    onClick={() => onGpu(id)} title={`${label}: about ${usd(price)} an hour`}>{label} · {usd(price)}</button>
                ))}
              </div>
            )}
            {listed.map((sv) => {
              const c = checks[sv.id];
              const list = gpus[sv.id] ?? sv.gpus;
              const pickGpu = list.find((g) => g.id === gpu) ?? list.find((g) => g.id === sv.default_gpu);
              return (
                <div key={sv.id}>
                  <button role="menuitemradio" aria-checked={value === sv.id} className={`make-on-item ${value === sv.id ? "on" : ""}`} onClick={() => pick(sv.id)}>
                    <Cloud size={16} />
                    <span className="grow"><b>Cloud GPU · {sv.name}</b>
                      <span className="small muted">{!c ? `Checking your ${sv.name} account…`
                        : c.ok ? ["Pay as you go", c.balance != null ? `$${c.balance.toFixed(2)} of credit` : "",
                                   pickGpu?.price ? `${pickGpu.label} ${usd(pickGpu.price)} an hour` : ""].filter(Boolean).join(" · ")
                          : c.detail}</span></span>
                    {value === sv.id && <Check size={15} />}
                  </button>
                  {value === sv.id && c?.ok && (
                    <div className="make-on-gpus" role="radiogroup" aria-label={`${sv.name} GPU`}>
                      {!gpus[sv.id] ? <span className="small muted">Asking {sv.name} for today's prices…</span>
                        : list.map((g) => (
                          <button key={g.id} role="radio" aria-checked={gpu === g.id} className={`chip-btn ${gpu === g.id ? "on" : ""}`}
                            onClick={() => onGpu(g.id)} disabled={g.stock === "none"}
                            title={`${g.label}${g.gb ? `, ${g.gb} GB` : ""}${g.price ? `: ${usd(g.price)} an hour` : ""}${g.stock === "none" ? " (none free right now)" : ""}`}>
                            {g.label}{g.price ? ` · ${usd(g.price)}` : ""}
                          </button>
                        ))}
                    </div>
                  )}
                </div>
              );
            })}
            {value !== "mac" && value !== "kaggle" && onHours && (
              <div className="make-on-hours">
                <div className="row" style={{ gap: 6 }}>
                  <span className="small">Time limit per run</span>
                  <div className="make-on-gpus" role="radiogroup" aria-label="Time limit per run" style={{ margin: 0 }}>
                    {[3, 6, 12].map((h) => (
                      <button key={h} role="radio" aria-checked={(hours ?? 3) === h} className={`chip-btn ${(hours ?? 3) === h ? "on" : ""}`}
                        onClick={() => onHours(h)}>{h} h</button>
                    ))}
                  </div>
                </div>
              </div>
            )}
            {value !== "mac" && onSounds && (
              <div className="make-on-hours">
                <div className="row" style={{ gap: 6 }}>
                  <span className="small">Generate sounds on</span>
                  <div className="make-on-gpus" role="radiogroup" aria-label="Generate sounds on" style={{ margin: 0 }}>
                    {[[true, CLOUD_NAMES[value] ?? "the cloud"], [false, "This Mac"]].map(([on, label]) => (
                      <button key={String(on)} role="radio" aria-checked={(sounds ?? true) === on} className={`chip-btn ${(sounds ?? true) === on ? "on" : ""}`}
                        onClick={() => onSounds(on as boolean)}>{label as string}</button>
                    ))}
                  </div>
                </div>
              </div>
            )}
            {value !== "mac" && onLive && (() => {
              // Modal and the rented machines, Lightning and Beam can be read while they work; Kaggle can't
              const can = value === "modal" || !!services?.find((x) => x.id === value)?.live;
              const name = CLOUD_NAMES[value] ?? value;
              const why = value === "kaggle" ? "Kaggle only hands a run's files over when it ends"
                : `not available on ${name} yet: it isn't known to show files while a run works`;
              return (
                <label className={`small make-on-live ${can ? "" : "dim"}`} title={can ? "If the run breaks, or this Mac sleeps, the lines made so far are already here" : why}>
                  <input type="checkbox" disabled={!can} checked={can && live !== false} onChange={(e) => onLive(e.target.checked)} />
                  <span>Bring lines back as they're made</span>
                </label>
              );
            })()}
            {value !== "mac" && ai && onAi && (
              <div className="make-on-ai">
                <span className="small muted">Also on this GPU</span>
                <label className="small"><input type="checkbox" checked={ai.detect} onChange={(e) => onAi("detect", e.target.checked)} /> Detect speakers</label>
              </div>
            )}
            {value !== "mac" && est && (
              <div className="make-on-estimate small" role="status">
                <b>Estimate</b>{" "}
                <span className="muted">{est.e ? estimateText(est.e) : est.err ?? "Working it out…"}</span>
                {est.e?.disk && est.e.disk.need > est.e.disk.free && (
                  <span className="warn-text"> Not enough space on this Mac: the book needs about {gb(est.e.disk.need)}, and {gb(est.e.disk.free)} is free.</span>
                )}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

const time = (t?: number) => (t ? new Date(t * 1000).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) : "");

/** The book's cloud job, kept up to date: asked about every minute while it's in the cloud (the engine asks
 *  Kaggle at most every two minutes), every few seconds while its lines are coming back. */
export function useCloudJob(folder: string | undefined) {
  const [job, setJob] = useState<CloudJob | null>(null);
  const timer = useRef<number | undefined>(undefined);
  const fails = useRef(0);
  const load = () => {
    if (!folder) return;
    bookApi.cloud(folder).then((j) => {
      fails.current = 0;
      setJob(j.state ? j : null);
    }).catch(() => {
      fails.current += 1;
      const delay = Math.min(60000, 4000 * Math.pow(2, Math.min(fails.current - 1, 4)));
      window.clearTimeout(timer.current);
      timer.current = window.setTimeout(load, delay);
    });
  };
  useEffect(() => { fails.current = 0; setJob(null); load(); return () => window.clearTimeout(timer.current); }, [folder]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    window.clearTimeout(timer.current);
    // (every few seconds too while the book is being put together after the run: so it shows when that's over)
    const soon = job?.state === "downloading" || (!!job?.stitch && job.state === "ready");
    if (job?.state && (WAITING.includes(job.state) || job.stitch)) timer.current = window.setTimeout(load, soon ? 4000 : 60000);
    return () => window.clearTimeout(timer.current);
  }, [job]); // eslint-disable-line react-hooks/exhaustive-deps
  const checkNow = () => folder && bookApi.cloudCheck(folder).then((j) => { fails.current = 0; setJob(j.state ? j : null); }).catch(() => {});
  const forget = () => folder && bookApi.cloudForget(folder).then(() => setJob(null)).catch(() => {});
  /** Stop the run out there (the lines back are kept): {stopped, back, lines}. Throws when that didn't work. */
  const stop = async () => { const r = await bookApi.cloudForget(folder || ""); setJob(null); return r; };
  return { job, setJob, checkNow, forget, stop, reload: load };
}

/** What the book bar says about the cloud job. onStop: the Stop button (it comes back when that throws). */
export function CloudLine({ job, onCheck, onForget, onStop }: {
  job: CloudJob; onCheck: () => void; onForget: () => void; onStop: () => unknown;
}) {
  const where = CLOUD_NAMES[job.backend ?? "kaggle"] ?? job.backend ?? "the cloud";
  const open = job.url ? <button className="link-btn" onClick={() => openExternal(job.url!)}>Open on {where}</button> : null;
  if (job.state === "failed") return (
    <span className="small error-text status-line cloud-line">Cloud error: {job.error || "Run failed."} {open} <button className="link-btn" onClick={onForget}>Dismiss</button></span>
  );
  if (job.state === "ready") {
    const all = (job.made ?? 0) >= (job.lines ?? 0);
    const readyDetails = [
      all ? `All ${(job.lines ?? 0).toLocaleString()} lines back from ${where}` : `${(job.made ?? 0).toLocaleString()} of ${(job.lines ?? 0).toLocaleString()} lines back from ${where}`,
      job.sound_count ? `${job.sounds_made ?? 0} of ${job.sound_count} sounds back` : "",
      job.speed ? `${job.speed}× real time` : "",
      job.stitch?.status || "",
    ].filter(Boolean).join("\n");
    if (job.stitch) {
      const cleanStep = (job.stitch.status || "Assembling")
        .replace(/…$/, "")
        .replace(/\s*(?:·\s*)?\d+(?:\.\d+)?%$/, "")
        .trim() || "Assembling";
      const pct = job.stitch.progress != null ? `${Math.round(job.stitch.progress * 100)}%` : "";
      const left = job.stitch.eta != null ? timeLeft(job.stitch.eta) : "";
      return (
        <span className="small muted status-line cloud-line" title={readyDetails}>
          <Cloud size={12} /> {[cleanStep, pct, left].filter(Boolean).join(" · ")}
        </span>
      );
    }
    return (
      <span className="small muted status-line cloud-line" title={readyDetails}>
        <Cloud size={12} />{" "}
        {all ? "Cloud audio ready · Finish to build book"
          : job.stopped ? `${(job.made ?? 0).toLocaleString()}/${(job.lines ?? 0).toLocaleString()} lines back (time limit) · Finish or run again`
            : job.error || `${(job.made ?? 0).toLocaleString()}/${(job.lines ?? 0).toLocaleString()} lines back · Finish to build book`}
        {job.stitch_end === "failed" && job.stitch_error ? ` · ${job.stitch_error}` : ""}
      </span>
    );
  }
  if (job.state === "downloading") return <span className="small muted status-line cloud-line" title={job.note || ""}><Cloud size={12} /> {job.note || "Downloading audio…"}</span>;
  const gpu = job.backend !== "kaggle" && job.gpu ? ` (${job.gpu_label || shortGpu(job.gpu)})` : "";
  const pct = job.live && job.progress != null ? `${Math.round(job.progress * 100)}%` : "";
  const left = job.eta != null ? timeLeft(job.eta) : "";
  const trouble = job.disk_full ? "Disk full"
    : job.offline_since ? "Offline (will resume)"
    : job.live_error ? "Retrying download" : "";
  const details = [
    `${where}${gpu}${job.elapsed != null && job.state === "running" ? ` · ${duration(job.elapsed)} elapsed` : ""}`,
    job.live ? `${(job.back ?? 0).toLocaleString()} / ${(job.lines ?? 0).toLocaleString()} lines back` : `${(job.lines ?? 0).toLocaleString()} lines`,
    job.sent ? `Sent ${time(job.sent)}` : "", job.max_cost ? `Up to ${usd(job.max_cost)}` : "",
    job.budget ? `Time limit: ${job.budget}h` : "", job.note || "",
    job.stitch ? `Local: ${job.stitch.status}` : "",
    `Keep this Mac on and Babble open until lines finish downloading${job.awake ? " (awake lock active)" : ""}.`,
  ].filter(Boolean).join("\n");
  return (
    <span className="small muted status-line cloud-line" title={details}>
      <Cloud size={12} />{" "}
      {[job.state === "queued" ? `Queued on ${where}` : `Cloud (${where})`, pct, left, trouble].filter(Boolean).join(" · ")}{" "}
      {!job.live && <><button className="link-btn" onClick={onCheck}>Check</button>{" "}</>}{open}
      {" "}<StopButton className="link-btn" onStop={onStop}>Stop</StopButton>
    </span>
  );
}
