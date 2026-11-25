import { useState } from "react";
import { CheckCircle2, KeyRound, Sparkles } from "lucide-react";
import { directorApi, type AiService } from "../api";
import { useToast } from "../components/ui";
import { HowTo } from "./HowTo";
import { Disconnect } from "./Disconnect";

/** Connections → an AI service the sound director can use with your own key. The key is checked when it's saved
 *  (the service lists its models) and kept in the app's settings file, readable only by you. */
export function AiKeyCard({ service, onChange }: { service: AiService; onChange: () => void }) {
  const toast = useToast();
  const [key, setKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState(false);
  const save = async () => {
    setBusy(true);
    try {
      await directorApi.saveKey(service.id, key);
      setKey(""); setOpen(false);
      toast(`${service.name} key saved.`);
      onChange();
    } catch (e) { toast((e as Error).message, true); }
    finally { setBusy(false); }
  };
  return (
    <div className="conn-card">
      <div className="row"><div className="kind-icon ambience"><Sparkles size={16} /></div>
        <div className="grow"><b>{service.name}</b>
          <span className={`chip ${service.cost === "paid" ? "" : "ok"}`} style={{ marginLeft: 8 }}>{service.cost === "paid" ? "Paid" : service.cost === "free tier" ? "Free tier" : "Free models"}</span>
          <div className="small muted">{service.cost_note}</div></div>
        {service.has_key ? <span className="chip ok"><CheckCircle2 size={11} />Key saved</span> : <span className="chip">Not set up</span>}
        {service.has_key && <Disconnect name={service.name} run={async () => { await directorApi.forget(service.id); onChange(); }} />}
        {!open && <button className="btn sm" onClick={() => setOpen(true)}>{service.has_key ? "Replace key" : "Set up"}</button>}
      </div>
      {open && (
        <div className="row" style={{ marginTop: 10 }}>
          <div className="search-box grow"><KeyRound size={14} className="muted" />
            <input className="input" type="password" autoComplete="off" placeholder="Paste an API key" aria-label={`${service.name} API key`}
              value={key} onChange={(e) => setKey(e.target.value)} onKeyDown={(e) => e.key === "Enter" && key.trim() && !busy && save()} /></div>
          <button className="btn ghost" onClick={() => { setOpen(false); setKey(""); }}>Cancel</button>
          <button className="btn primary" disabled={!key.trim() || busy} onClick={save}>{busy ? "Checking…" : "Save"}</button>
        </div>
      )}
      {open && <HowTo title={`How to get a ${service.name} key`} open steps={[<>Make a key on <b>{service.keys[0]}</b> and paste it here. The service bills you for what the director uses.</>]}
        path={[["API key", service.keys[0], ...service.keys.slice(1)]]} />}
    </div>
  );
}
