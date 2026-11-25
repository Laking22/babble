import { useEffect, useState } from "react";
import { CheckCircle2, CircleAlert, Cloud, KeyRound, RefreshCw } from "lucide-react";
import { connectionApi, type Check, type CloudService } from "../api";
import { useToast } from "../components/ui";
import { openExternal } from "../native";
import { HOWTO, HowTo, PATHS } from "./HowTo";
import { Disconnect } from "./Disconnect";

/** Per service: what to paste, and where to add credit. */
const KEYS: Record<string, { what: string; billing: string; howTo: string }> = {
  runpod: { what: "Runpod API key", billing: "https://console.runpod.io/user/billing", howTo: "How to get a Runpod API key" },
  vast: { what: "Vast.ai API key", billing: "https://cloud.vast.ai/billing/", howTo: "How to get a Vast.ai API key" },
  thunder: { what: "Thunder Compute token", billing: "https://console.thundercompute.com/settings/billing", howTo: "How to get a Thunder Compute token" },
  lambda: { what: "Lambda API key", billing: "https://cloud.lambda.com/", howTo: "How to get a Lambda API key" },
  lightning: { what: "Lightning user ID and API key", billing: "https://lightning.ai/account/settings", howTo: "How to get your Lightning keys" },
  beam: { what: "Beam API key", billing: "https://platform.beam.cloud/settings/billing", howTo: "How to get a Beam API key" },
  cerebrium: { what: "Cerebrium service account token", billing: "https://dashboard.cerebrium.ai/", howTo: "How to get a Cerebrium token" },
  hfjobs: { what: "Hugging Face token with Write access", billing: "https://huggingface.co/settings/billing", howTo: "How to get a Hugging Face Jobs token" },
};

/** Connections → a pay-as-you-go GPU service for Create in the cloud: paste an API key from the service's site
 *  (the steps are right under it); the app rents a GPU per book and frees it when the lines are back. Folded to
 *  one line until it's set up, so the page stays short. */
export function CloudKeyCard({ service, onChange }: { service: CloudService; onChange?: () => void }) {
  const toast = useToast();
  const info = KEYS[service.id] ?? { what: `${service.name} API key`, billing: "", howTo: `How to get a ${service.name} key` };
  const [check, setCheck] = useState<Check | null>(null);
  const [busy, setBusy] = useState(false);
  const [key, setKey] = useState("");
  const [setup, setSetup] = useState(false);
  const load = async () => {
    setBusy(true);
    try { setCheck(await connectionApi.check(service.id)); }
    catch (e) { setCheck({ ok: false, detail: (e as Error).message }); }
    finally { setBusy(false); }
  };
  useEffect(() => { if (service.has_key) load(); }, [service.id, service.has_key]); // eslint-disable-line react-hooks/exhaustive-deps
  const save = async () => {
    setBusy(true);
    try {
      const r = await connectionApi.cloudKey(service.id, key);
      setKey(""); setSetup(false);
      toast(`${service.name} key saved${r.balance != null ? ` · $${r.balance.toFixed(2)} of credit` : ""}.`);
      onChange?.();
      await load();
    } catch (e) { toast((e as Error).message, true); setBusy(false); }
  };
  const remove = async () => { await connectionApi.cloudForget(service.id); setCheck(null); onChange?.(); };
  const connected = !!check?.ok;
  const showForm = !connected && (setup || (service.has_key && !!check && !busy));

  return (
    <div className="conn-card">
      <div className="row"><div className="kind-icon music"><Cloud size={16} /></div>
        <div className="grow"><b>{service.name}</b><span className="chip" style={{ marginLeft: 8 }}>Beta</span>
          <div className="small muted">{service.pricing[0].toUpperCase() + service.pricing.slice(1)}</div></div>
        {busy ? <span className="chip"><RefreshCw size={11} className="spin" />Checking…</span>
          : connected ? <span className="chip ok"><CheckCircle2 size={11} />Connected</span>
            : service.has_key && check ? <span className="chip warn"><CircleAlert size={11} />Not connected</span>
              : <span className="chip">Not set up</span>}
        {service.has_key && <Disconnect name={service.name} run={remove} />}
        {service.has_key || connected
          ? <button className="btn sm" onClick={load} disabled={busy}>Check</button>
          : <button className="btn sm" onClick={() => setSetup(!setup)} aria-expanded={setup}>{setup ? "Close" : "Set up"}</button>}
      </div>
      {check && (service.has_key || connected) && <p className="small muted conn-detail">{check.detail.replace(/\s*Add your .+ API key on the Connections page\.?/, "")}</p>}
      {connected && (
        <p className="small conn-detail signin-links">
          What a book uses is freed as soon as its lines are back, so it only costs while it works.{" "}
          {info.billing && <button className="link-btn" onClick={() => openExternal(info.billing)}>Add credit</button>}

        </p>
      )}
      {showForm && (
        <div className="signin-box">
          <div className="row">
            <div className="search-box grow signin-token"><KeyRound size={14} className="muted" />
              <input className="input" type="password" autoComplete="off" placeholder={info.what} aria-label={info.what}
                value={key} onChange={(e) => setKey(e.target.value)} onKeyDown={(e) => e.key === "Enter" && key.trim() && save()} />
            </div>
            <button className="btn primary" disabled={!key.trim() || busy} onClick={save}>Save</button>
          </div>
          {service.has_key && <p className="small muted" style={{ margin: 0 }}>A key is saved but {service.name} didn't accept it. Paste a new one, or disconnect it.</p>}
        </div>
      )}
      {(showForm || connected) && HOWTO[service.id as keyof typeof HOWTO] && (
        <HowTo title={info.howTo} open={showForm} steps={HOWTO[service.id as keyof typeof HOWTO]} path={PATHS[service.id]} />
      )}
    </div>
  );
}
