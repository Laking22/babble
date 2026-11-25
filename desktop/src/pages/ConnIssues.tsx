import { useEffect, useState } from "react";
import { CircleAlert, X } from "lucide-react";
import { connectionApi, type ConnIssue } from "../api";

/** What to do about each kind of problem (the service's own words are shown with it). */
const WHAT_TO_DO: Record<ConnIssue["kind"], string> = {
  "sign-in": "It no longer accepts the sign-in or key: sign in again, or paste a new key, below.",
  limit: "A limit, quota or balance ran out: it works again when it resets, or once you add credit.",
  network: "Babble couldn't reach it: check your internet connection, VPN or firewall. Jobs already sent keep going and are tried again.",
};

const ago = (t: number) => {
  const m = Math.round((Date.now() / 1000 - t) / 60);
  return m < 1 ? "just now" : m < 60 ? `${m} min ago` : m < 48 * 60 ? `${Math.round(m / 60)} h ago` : `${Math.round(m / 1440)} days ago`;
};

/** The problems services had while Babble used them, kept fresh (also for the sidebar's badge). */
export function useConnIssues(every = 15000) {
  const [list, setList] = useState<ConnIssue[]>([]);
  const load = () => connectionApi.issues().then(setList).catch(() => {});
  useEffect(() => {
    load();
    const t = setInterval(load, every);
    window.addEventListener("focus", load);
    return () => { clearInterval(t); window.removeEventListener("focus", load); };
  }, [every]); // eslint-disable-line react-hooks/exhaustive-deps
  const dismiss = async (service: string) => { try { setList(await connectionApi.dismissIssue(service)); } catch { /* kept */ } };
  const of = (service: string) => list.find((x) => x.service === service);
  return { list, of, dismiss, reload: load };
}

/** One problem: what went wrong, when and where, and what to do. */
export function IssueLine({ issue, onDismiss, named }: { issue?: ConnIssue; onDismiss: (service: string) => void; named?: boolean }) {
  if (!issue) return null;
  return (
    <div className="conn-issue small" role="alert">
      <CircleAlert size={14} />
      <div className="grow">
        <b>{named ? `${issue.name}: ` : ""}{issue.kind === "sign-in" ? "Sign-in refused" : issue.kind === "limit" ? "Limit reached" : "Couldn't connect"}</b>
        <span className="muted"> · {issue.where ? `${issue.where}, ` : ""}{ago(issue.when)}</span>
        <div>{WHAT_TO_DO[issue.kind]}</div>
        <div className="faint conn-issue-error">{issue.error}</div>
      </div>
      <button className="btn sm ghost icon" title="Dismiss" aria-label="Dismiss" onClick={() => onDismiss(issue.service)}><X size={13} /></button>
    </div>
  );
}
