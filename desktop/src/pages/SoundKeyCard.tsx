import { useState } from "react";
import { AudioLines, CheckCircle2, KeyRound } from "lucide-react";
import { soundsApi, type SoundSource } from "../api";
import { useToast } from "../components/ui";
import { HowTo } from "./HowTo";
import { Disconnect } from "./Disconnect";

const STEPS: Record<string, React.ReactNode[]> = {
  freesound: [
    <>Sign in on <b>freesound.org</b> (or make a free account).</>,
    <>Open <b>freesound.org/apiv2/apply</b>, give it a name (Babble) and a short description, and accept the terms.</>,
    <>Copy the <b>Client secret / API key</b> it shows and paste it here.</>,
  ],
  lotsofsounds: [
    <>The API needs Lots of Sounds' <b>Pro</b> plan (lotsofsounds.com/pricing).</>,
    <>Open <b>lotsofsounds.com/dashboard/api-keys</b>, create a key (it starts with <code>los_</code>) and paste it here.</>,
  ],
};

/** Connections → an online sound library with your own key. Missing sounds are looked for there (public-domain
 *  recordings only) before they're made with AI; the Sounds page can search it too. */
export function SoundKeyCard({ source, onChange }: { source: SoundSource; onChange: () => void }) {
  const toast = useToast();
  const [key, setKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState(false);
  const save = async () => {
    setBusy(true);
    try {
      await soundsApi.saveKey(source.id, key);
      setKey(""); setOpen(false);
      toast(`${source.name} key saved.`);
      onChange();
    } catch (e) { toast((e as Error).message, true); }
    finally { setBusy(false); }
  };
  return (
    <div className="conn-card">
      <div className="row"><div className="kind-icon effect"><AudioLines size={16} /></div>
        <div className="grow"><b>{source.name}</b>
          <span className={`chip ${source.cost === "paid" ? "" : "ok"}`} style={{ marginLeft: 8 }}>{source.cost === "paid" ? "Paid" : "Free"}</span>
          <div className="small muted">{source.cost_note}</div></div>
        {source.has_key ? <span className="chip ok"><CheckCircle2 size={11} />Key saved</span> : <span className="chip">Not set up</span>}
        {source.has_key && <Disconnect name={source.name} run={async () => { await soundsApi.forget(source.id); onChange(); }} />}
        {!open && <button className="btn sm" onClick={() => setOpen(true)}>{source.has_key ? "Replace key" : "Set up"}</button>}
      </div>
      {open && (
        <div className="row" style={{ marginTop: 10 }}>
          <div className="search-box grow"><KeyRound size={14} className="muted" />
            <input className="input" type="password" autoComplete="off" placeholder="Paste an API key" aria-label={`${source.name} API key`}
              value={key} onChange={(e) => setKey(e.target.value)} onKeyDown={(e) => e.key === "Enter" && key.trim() && !busy && save()} /></div>
          <button className="btn ghost" onClick={() => { setOpen(false); setKey(""); }}>Cancel</button>
          <button className="btn primary" disabled={!key.trim() || busy} onClick={save}>{busy ? "Checking…" : "Save"}</button>
        </div>
      )}
      {open && <HowTo title={`How to get a ${source.name} key`} open steps={STEPS[source.id] || []}
        path={[["API key", ...source.keys]]} />}
    </div>
  );
}
