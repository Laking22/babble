import { useEffect, useState } from "react";
import { CircleAlert, Loader2, OctagonX } from "lucide-react";
import { spendingApi, type SpendRow, type Spending as Spend } from "../api";
import { useToast } from "../components/ui";

const usd = (d: number) => (d > 0 && d < 0.01 ? "<$0.01" : `$${d.toFixed(2)}`);
const ago = (t: number) => {
  const m = Math.round((Date.now() / 1000 - t) / 60);
  return m < 1 ? "now" : m < 60 ? `${m} min ago` : m < 1440 ? `${Math.round(m / 60)} h ago` : `${Math.round(m / 1440)} d ago`;
};
/** Within a day: "2 h ago"; older: the date ("Nov 21"). */
const when = (t: number) => (Date.now() / 1000 - t < 86400 ? ago(t)
  : new Date(t * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric" }));

/** Settings → Spending: what paid services cost, the day limit, cloud jobs out and Stop all, then everything
 *  that cost money (one row per paid job, newest first). */
export function SpendingTab() {
  const [spend, setSpend] = useState<Spend | null>(null);
  const [rows, setRows] = useState<SpendRow[] | null>(null);
  const [err, setErr] = useState("");
  const load = () => {
    spendingApi.get().then((s) => { setSpend(s); setErr(""); }).catch((e) => setErr((e as Error).message));
    spendingApi.history().then(setRows).catch(() => {});
  };
  useEffect(() => {
    load();
    const t = setInterval(load, 15000);
    return () => clearInterval(t);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="stack">
      <section className="card card-pad stack">
        <h3 className="card-title" style={{ margin: 0 }}>Spending &amp; limits</h3>
        {err && <div className="small error-text">{err}</div>}
        {!spend && !err && <div className="small muted row"><Loader2 size={14} className="spin" /> Loading…</div>}
        <SpendingCard data={spend} onChange={(s) => { setSpend(s); spendingApi.history().then(setRows).catch(() => {}); }} />
      </section>

      <section className="card card-pad stack">
        <h3 className="card-title" style={{ margin: 0 }}>Recent</h3>
        {rows && !rows.length && <div className="small muted">Nothing has cost money yet.</div>}
        {!!rows?.length && (
          <div className="list">
            {rows.map((r) => (
              <div key={`${r.started}-${r.title}`} className="list-item">
                <div className="grow">
                  <div className="title">{r.title}{r.book ? ` · ${r.book}` : ""}</div>
                  <div className="small muted">{[
                    r.service,
                    r.model.split("/").pop(),
                    r.requests ? `${r.requests} ${r.requests === 1 ? "request" : "requests"}` : "",
                    r.gpu_hours ? `${r.gpu_hours} GPU h` : "",
                    r.state === "stopped" || r.state === "failed" ? r.state : "",
                    r.agent ? "AI agent" : "",
                    when(r.time),
                  ].filter(Boolean).join(" · ")}</div>
                </div>
                <span title={r.exact ? "The service's own figure" : "Estimated from list prices"}>
                  {r.exact ? "" : "≈ "}{usd(r.dollars)}
                </span>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

/** What paid services cost today and this month (narrator/spend.py), the day limit, cloud jobs out, and a
 *  button that stops all paid work. */
export function SpendingCard({ data, onChange }: { data: Spend | null; onChange: (s: Spend) => void }) {
  const toast = useToast();
  const [limit, setLimit] = useState("");
  const [agentLimit, setAgentLimit] = useState("");
  const [confirm, setConfirm] = useState(false);
  useEffect(() => { if (data) setLimit(data.limit ? String(data.limit) : ""); }, [data?.limit]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (data) setAgentLimit(String(data.agent_limit ?? 0)); }, [data?.agent_limit]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!data) return null;

  const saveAgentLimit = async () => {
    const day = agentLimit.trim() === "" ? 0 : Number(agentLimit);
    if (Number.isNaN(day) || day < 0) { toast("Type an amount in dollars (0: agents can't spend).", true); return; }
    try { onChange(await spendingApi.setAgentLimit(day)); toast(day ? `AI agents: ${usd(day)} a day.` : "AI agents can't spend."); }
    catch (e) { toast((e as Error).message, true); }
  };

  const saveLimit = async () => {
    const day = limit.trim() === "" ? 0 : Number(limit);
    if (Number.isNaN(day) || day < 0) { toast("Type an amount in dollars, or leave it empty for no limit.", true); return; }
    try { onChange(await spendingApi.setLimit(day)); toast(day ? `Limit: ${usd(day)} a day.` : "No daily limit."); }
    catch (e) { toast((e as Error).message, true); }
  };
  const stopAll = async () => {
    setConfirm(false);
    try {
      const r = await spendingApi.stopAll();
      onChange(r);
      toast(r.stopped.length ? `Stopped: ${r.stopped.join(", ")}.` : "Nothing paid was running.");
    } catch (e) { toast((e as Error).message, true); }
  };

  return (
    <div className="stack">
      <div className="row wrap">
        <div className="grow"><b>{usd(data.today)}</b> <span className="muted">today</span> · <b>{usd(data.month)}</b> <span className="muted">this month</span></div>
        <label className="row small" style={{ gap: 6 }}>Day limit $
          <input className="input" style={{ width: 80 }} inputMode="decimal" placeholder="none" value={limit}
            onChange={(e) => setLimit(e.target.value)} onKeyDown={(e) => e.key === "Enter" && saveLimit()} aria-label="Daily spending limit in dollars" />
        </label>
        <button className="btn sm" onClick={saveLimit}>Save</button>
      </div>
      <div className="row wrap">
        <div className="grow"><b>{usd(data.agent_today ?? 0)}</b> <span className="muted">by AI agents today</span></div>
        <label className="row small" style={{ gap: 6 }}>AI agents $
          <input className="input" style={{ width: 80 }} inputMode="decimal" placeholder="0" value={agentLimit}
            onChange={(e) => setAgentLimit(e.target.value)} onKeyDown={(e) => e.key === "Enter" && saveAgentLimit()} aria-label="AI agents' daily spending limit in dollars" />
        </label>
        <button className="btn sm" onClick={saveAgentLimit}>Save</button>
      </div>

      {data.alerts.map((a, i) => (
        <div key={i} className="small row" style={{ gap: 6 }}><CircleAlert size={13} style={{ color: "var(--warn)", flex: "none" }} />
          <span>{a.title ? `${a.title}: ` : ""}{a.why} <span className="muted">{ago(a.time)}</span></span></div>
      ))}

      {!!data.services.length && (
        <div className="list">
          {data.services.map((s) => (
            <div key={s.service} className="list-item">
              <div className="grow"><div className="title">{s.name}</div>
                <div className="small muted">{s.requests_today} today</div></div>
              <div className="small" style={{ textAlign: "right" }}>{usd(s.today)} today<div className="muted">{usd(s.month)} this month</div></div>
            </div>
          ))}
        </div>
      )}

      {!!data.jobs.length && (
        <div className="list">
          {data.jobs.map((j) => (
            <div key={j.run} className="list-item"><div className="grow"><div className="title">{j.what}</div>
              <div className="small muted">On {j.name} · sent {ago(j.sent)}{j.price ? ` · $${j.price.toFixed(2)}/h` : " · free"}</div></div></div>
          ))}
        </div>
      )}

      <div className="row">
        {confirm ? (
          <>
            <span className="small grow">Stop every task and cloud job that costs money?</span>
            <button className="btn sm danger" onClick={stopAll}>Stop all</button>
            <button className="btn sm ghost" onClick={() => setConfirm(false)}>Keep going</button>
          </>
        ) : (
          <button className="btn sm danger" onClick={() => setConfirm(true)}><OctagonX size={14} />Stop all paid work</button>
        )}
      </div>
    </div>
  );
}
