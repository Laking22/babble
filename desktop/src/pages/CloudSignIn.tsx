import { useState } from "react";
import { KeyRound, LogIn } from "lucide-react";
import { connectionApi, useTask } from "../api";
import { useToast } from "../components/ui";
import { openExternal } from "../native";

const NAMES = { kaggle: "Kaggle", modal: "Modal" } as const;

/** Signing in to a cloud GPU from the app: the browser opens the service's own sign-in page and the app waits for
 *  it (nothing to install, no Terminal). Kaggle can also take an API token pasted from kaggle.com/settings. */
export function CloudSignIn({ service, signedIn, onDone }: { service: "kaggle" | "modal"; signedIn: boolean; onDone: () => void }) {
  const toast = useToast();
  const name = NAMES[service];
  const [token, setToken] = useState("");
  const task = useTask((t) => {
    if (t.state === "failed") toast(t.error || `The ${name} sign-in didn't finish.`, true);
    else toast((t.status || "").split("\n")[0].replace(/^Done · /, ""));
    onDone();
  }, `cloud-signin-${service}`);
  const waiting = task.task?.state === "running";
  const url = /^URL (\S+)/m.exec(String(task.task?.result?.log || ""))?.[1];
  const start = () => task.start(() => (service === "kaggle" ? connectionApi.kaggleSignIn() : connectionApi.modalSignIn()));
  const saveToken = async () => {
    try { const r = await connectionApi.kaggleToken(token); setToken(""); toast(`Signed in to Kaggle as ${r.user}.`); onDone(); }
    catch (e) { toast((e as Error).message, true); }
  };

  if (waiting) return (
    <p className="small conn-detail signin-wait">
      Waiting for you to sign in to {name} in your browser…{" "}
      {url && <button className="link-btn" onClick={() => openExternal(url)}>Open the sign-in page</button>}
    </p>
  );
  if (signedIn) return (
    <p className="small conn-detail signin-links">
      <button className="link-btn" onClick={start}>Use another {name} account</button>
    </p>
  );
  return (
    <div className="signin-box">
      <div className="row">
        <button className="btn primary" onClick={start}><LogIn size={14} />Sign in with {name}</button>
        <span className="small muted grow">
          {service === "kaggle" ? "Opens Kaggle in your browser. Free account, about 30 GPU hours a week."
            : "Opens Modal in your browser to approve this Mac. The free plan includes $30 of GPU time a month."}
        </span>
      </div>
      {service === "kaggle" && (
        <details className="advanced">
          <summary>Or paste an API token</summary>
          <div className="row" style={{ marginTop: 8 }}>
            <div className="search-box grow signin-token"><KeyRound size={14} className="muted" />
              <input className="input" type="password" autoComplete="off" placeholder="From kaggle.com/settings → API Tokens → Generate New Token"
                value={token} onChange={(e) => setToken(e.target.value)} onKeyDown={(e) => e.key === "Enter" && token.trim() && saveToken()} aria-label="Kaggle API token" />
            </div>
            <button className="btn" disabled={!token.trim()} onClick={saveToken}>Save</button>
          </div>
        </details>
      )}
    </div>
  );
}
