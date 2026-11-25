import { useEffect, useRef, useState } from "react";
import { CheckCircle2, ChevronRight, CircleAlert, Cloud, Cpu, Download, KeyRound, RefreshCw, Server } from "lucide-react";
import { connectionApi, directorApi, keychainApi, soundsApi, trainApi, useLoad, type Check, type CloudService, type Connections as Conn } from "../api";
import { AiKeyCard } from "./AiKeyCard";
import { SoundKeyCard } from "./SoundKeyCard";
import { Field, Switch, useToast } from "../components/ui";
import { ModalVoices } from "./ModalVoices";
import { CloudSignIn } from "./CloudSignIn";
import { CloudKeyCard } from "./CloudKeyCard";
import { HOWTO, HowTo, PATHS } from "./HowTo";
import { Disconnect } from "./Disconnect";
import { IssueLine, useConnIssues } from "./ConnIssues";
import type { ConnIssue } from "../api";

/** A check's words on this page: "Sign in … on the Connections page" is said elsewhere (the button is right here). */
const here = (detail: string) => detail.replace(/\s*Sign in to \w+ on the Connections page\.?/, "");

/** Open the steps once a check says the service isn't set up yet (not while it's still checking). */
const needsSetup = (c: Check | undefined) => !!c && !c.ok;

function Status({ check, busy, issue }: { check: Check | null | undefined; busy: boolean; issue?: ConnIssue }) {
  if (busy) return <span className="chip"><RefreshCw size={11} className="spin" />Checking…</span>;
  if (issue) return <span className="chip warn"><CircleAlert size={11} />Problem</span>;
  if (!check) return <span className="chip">Not checked</span>;
  return check.ok ? <span className="chip ok"><CheckCircle2 size={11} />Connected</span> : <span className="chip warn"><CircleAlert size={11} />Not connected</span>;
}

/** One collapsible group of connections (AI APIs, Cloud GPUs, …), folded until opened; it stays as you left it. */
function Group({ id, title, summary, problems = 0, forceOpen, focusNonce, children }: {
  id: string; title: string; summary?: string; problems?: number; forceOpen?: boolean; focusNonce?: number; children: React.ReactNode;
}) {
  const key = `connections-open-${id}`;
  const ref = useRef<HTMLDetailsElement>(null);
  const [open, setOpen] = useState(() => {
    if (forceOpen) return true;
    try { return localStorage.getItem(key) === "1"; } catch { return false; }
  });
  useEffect(() => { if (problems) setOpen(true); }, [problems]);  // (a group with a problem opens to show it)
  useEffect(() => {
    if (!forceOpen) return;
    setOpen(true);
    requestAnimationFrame(() => ref.current?.scrollIntoView({ behavior: "smooth", block: "start" }));
  }, [forceOpen, focusNonce]);
  return (
    <details ref={ref} className="conn-group" open={open}
      onToggle={(e) => { const o = (e.currentTarget as HTMLDetailsElement).open; setOpen(o); try { localStorage.setItem(key, o ? "1" : "0"); } catch { /* not kept */ } }}>
      <summary><ChevronRight size={16} /><span className="conn-group-title">{title}</span>{summary && <span className="small muted">{summary}</span>}
        {problems > 0 && <span className="chip warn"><CircleAlert size={11} />{problems} problem{problems === 1 ? "" : "s"}</span>}</summary>
      <div className="conn-group-body">{children}</div>
    </details>
  );
}

/** Places outside the app: AI services, cloud GPUs, a speech server, and downloads (spending: Settings → Spending). */
export function Connections({ embedded = false, focusGroup, focusNonce }: { embedded?: boolean; focusGroup?: string; focusNonce?: number }) {
  const toast = useToast();
  const issues = useConnIssues(15000);
  const conn = useLoad(connectionApi.get);
  const options = useLoad(trainApi.options);
  const ai = useLoad(directorApi.providers);
  const libs = useLoad(soundsApi.sources);
  const vault = useLoad(keychainApi.get);
  const [checks, setChecks] = useState<Record<string, Check>>({});
  const [busy, setBusy] = useState<Record<string, boolean>>({});
  const [ssh, setSsh] = useState<Conn["ssh"] | null>(null);
  const [orpheus, setOrpheus] = useState({ where: "mac", url: "", key: "" });
  const [hf, setHf] = useState("");
  const [orpheusNote, setOrpheusNote] = useState("");
  const [clouds, setClouds] = useState<CloudService[]>([]);
  const loadClouds = () => connectionApi.clouds().then(setClouds).catch(() => setClouds([]));
  useEffect(() => { loadClouds(); }, []);

  useEffect(() => {
    if (!conn.data) return;
    setSsh(conn.data.ssh);
    setOrpheus({ where: conn.data.orpheus.where, url: conn.data.orpheus.url, key: "" });
  }, [conn.data]);

  const check = async (id: string) => {
    setBusy((b) => ({ ...b, [id]: true }));
    try { const c = await connectionApi.check(id); setChecks((x) => ({ ...x, [id]: c })); issues.reload(); }
    catch (e) { setChecks((x) => ({ ...x, [id]: { ok: false, detail: (e as Error).message } })); }
    finally { setBusy((b) => ({ ...b, [id]: false })); }
  };
  useEffect(() => { check("kaggle"); check("modal"); check("huggingface"); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const saveSsh = async () => {
    if (!ssh) return;
    try { await connectionApi.saveSsh(ssh); toast("Saved."); check("ssh"); } catch (e) { toast((e as Error).message, true); }
  };
  const saveOrpheus = async (test: boolean) => {
    try {
      const r = await connectionApi.saveOrpheus({ where: orpheus.where, url: orpheus.url, key: orpheus.key || null, test });
      setOrpheus((o) => ({ ...o, key: "" }));
      setOrpheusNote(test ? (r.detail || (r.where === "mac" ? "Orpheus runs on this Mac; there's no server to test." : "")) : "Saved.");
      conn.reload();
    } catch (e) { toast((e as Error).message, true); }
  };
  const saveHf = async () => {
    try { const r = await connectionApi.saveHf(hf); setHf(""); toast(`Saved. Signed in to Hugging Face as ${r.user}.`); conn.reload(); check("huggingface"); }
    catch (e) { toast((e as Error).message, true); }
  };

  const backends = options.data?.backends || [];
  const name = (id: string) => backends.find((b) => b.id === id)?.name || id;
  const pricing = (id: string) => backends.find((b) => b.id === id)?.pricing || "";
  const connected = ["kaggle", "modal", "ssh"].filter((k) => checks[k]?.ok).length + clouds.filter((c) => c.has_key).length;
  const gpuSummary = connected ? `${connected} connected` : "";

  const body = (
    <>
      {!embedded && <div className="page-head"><div><h1>Connections</h1><p>Places outside the app it can use: AI services, cloud GPUs, a speech server, and downloads.</p></div></div>}
      {vault.data?.where === "keychain" && !vault.data.ready && (
        <p className="small section-note" style={{ display: "flex", gap: 6 }}><CircleAlert size={14} style={{ color: "var(--warn)", flex: "none" }} />
          Waiting for macOS to let Babble read your keys from the Keychain. Look for its prompt, then <button className="btn sm ghost" onClick={vault.reload}>check again</button></p>
      )}
      {vault.data?.where === "keychain" && vault.data.ready && !vault.data.ok && (
        <p className="small section-note" style={{ display: "flex", gap: 6 }}><CircleAlert size={14} style={{ color: "var(--warn)", flex: "none" }} />
          Babble couldn't read your keys from the Keychain ({vault.data.why}). Keys you add now are kept in its settings file until it can; reopen Babble to try again.</p>
      )}

      {issues.list.length > 0 && (
        <div className="conn-issues">
          {issues.list.map((x) => <IssueLine key={x.service} issue={x} named onDismiss={issues.dismiss} />)}
        </div>
      )}

      <div className="conn-section-head">AI &amp; Media Services</div>
      <Group id="ai" title="AI APIs" problems={(ai.data || []).filter((s) => issues.of(s.id)).length} summary={ai.data ? `${ai.data.filter((s) => s.has_key).length} of ${ai.data.length} set up` : ""}>
        <p className="small muted section-note">For the sound director and Immersion QA, with your own key. Free tiers have daily limits, so a long book can stop partway; the scenes done so far are kept.</p>
        <div className="stack">
          {(ai.data || []).filter((s) => s.has_key).concat((ai.data || []).filter((s) => !s.has_key)).map((s) => (
            <div key={s.id} className={issues.of(s.id) ? "conn-with-issue" : undefined}>
              <AiKeyCard service={s} onChange={() => { ai.reload(); issues.reload(); }} />
              <IssueLine issue={issues.of(s.id)} onDismiss={issues.dismiss} />
            </div>
          ))}
        </div>
      </Group>

      <Group id="sound-libraries" title="Sound libraries" problems={(libs.data || []).filter((s) => issues.of(s.id)).length} summary={libs.data ? `${libs.data.filter((s) => s.has_key).length} of ${libs.data.length} set up` : ""}>
        <p className="small muted section-note">Real recordings for your stories' sounds. A missing sound is looked for here first, and made with AI only when nothing sounds right.</p>
        <div className="stack">
          {(libs.data || []).map((s) => (
            <div key={s.id} className={issues.of(s.id) ? "conn-with-issue" : undefined}>
              <SoundKeyCard source={s} onChange={() => { libs.reload(); issues.reload(); }} />
              <IssueLine issue={issues.of(s.id)} onDismiss={issues.dismiss} />
            </div>
          ))}
        </div>
      </Group>

      <div className="conn-section-head">Cloud Compute</div>
      <Group id="gpus" title="Cloud GPUs" summary={gpuSummary} forceOpen={focusGroup === "gpus"} focusNonce={focusNonce}
        problems={issues.list.filter((x) => backends.some((b) => b.id === x.service) || clouds.some((c) => c.id === x.service)).length}>
      <div className="section-label">Training and Create in the cloud</div>
      <div className="stack">
        <div className="conn-card">
          <div className="row"><div className="kind-icon effect"><Cloud size={16} /></div>
            <div className="grow"><b>Kaggle</b><div className="small muted">{pricing("kaggle")}</div></div>
            <Status check={checks.kaggle} busy={!!busy.kaggle} issue={issues.of("kaggle")} />
            {checks.kaggle?.ok && <Disconnect name="Kaggle" note="This also signs out Kaggle's own tools on this Mac."
              run={async () => { await connectionApi.kaggleSignOut(); await check("kaggle"); }} />}
            <button className="btn sm" onClick={() => check("kaggle")} disabled={busy.kaggle}>Check</button></div>
          {checks.kaggle && <p className="small muted conn-detail">{here(checks.kaggle.detail)}</p>}
          <IssueLine issue={issues.of("kaggle")} onDismiss={issues.dismiss} />
          {checks.kaggle && <CloudSignIn service="kaggle" signedIn={checks.kaggle.ok && !issues.of("kaggle")} onDone={() => check("kaggle")} />}
          <HowTo title="How to set up Kaggle (free)" open={needsSetup(checks.kaggle)} steps={HOWTO.kaggle} path={PATHS.kaggle} />
        </div>

        <div className="conn-card">
          <div className="row"><div className="kind-icon ambience"><Server size={16} /></div>
            <div className="grow"><b>{name("ssh")}</b><div className="small muted">{pricing("ssh")}</div></div>
            <Status check={checks.ssh} busy={!!busy.ssh} issue={issues.of("ssh")} />
            {!!conn.data?.ssh.host && <Disconnect name="your GPU machine"
              run={async () => { const s = await connectionApi.forgetSsh(); setSsh(s); setChecks(({ ssh: _, ...rest }) => rest); conn.reload(); }} />}</div>
          {ssh && (
            <div className="stack" style={{ marginTop: 12, gap: 10 }}>
              <div className="row">
                <Field label="Address"><input className="input" placeholder="user@host (or local)" value={ssh.host} onChange={(e) => setSsh({ ...ssh, host: e.target.value })} /></Field>
                <Field label="Port"><input className="input" style={{ width: 90 }} value={ssh.port} onChange={(e) => setSsh({ ...ssh, port: e.target.value })} /></Field>
              </div>
              <Field label="SSH key" hint="The key file's path. Leave empty to use your usual SSH setup.">
                <input className="input" placeholder="~/.ssh/id_ed25519" value={ssh.key} onChange={(e) => setSsh({ ...ssh, key: e.target.value })} />
              </Field>
              <details className="advanced">
                <summary>More</summary>
                <div className="row">
                  <Field label="Folder on the machine"><input className="input" value={ssh.dir} onChange={(e) => setSsh({ ...ssh, dir: e.target.value })} /></Field>
                  <Field label="Python"><input className="input" value={ssh.python} onChange={(e) => setSsh({ ...ssh, python: e.target.value })} /></Field>
                </div>
              </details>
              <div className="row"><span className="spacer" /><button className="btn primary" onClick={saveSsh} disabled={busy.ssh}>Save and connect</button></div>
              {checks.ssh && <p className="small muted" style={{ margin: 0 }}>{checks.ssh.detail}</p>}
            </div>
          )}
          <IssueLine issue={issues.of("ssh")} onDismiss={issues.dismiss} />
          <HowTo title="Where to find the address and key" steps={HOWTO.ssh} path={PATHS.ssh}
            note="This one is for people who already use SSH. Kaggle, Modal and Runpod need only an account." />
        </div>

        {backends.filter((b) => !["kaggle", "ssh"].includes(b.id)).map((b) => (
          <div key={b.id} className="conn-card">
            <div className="row"><div className="kind-icon music">{b.id === "local-mac" ? <Cpu size={16} /> : <Cloud size={16} />}</div>
              <div className="grow"><b>{b.name}</b>{b.readiness === "beta" && <span className="chip" style={{ marginLeft: 8 }}>Beta</span>}<div className="small muted">{b.pricing}</div></div>
              <Status check={checks[b.id]} busy={!!busy[b.id]} issue={issues.of(b.id)} />
              {b.id === "modal" && checks.modal?.ok && <Disconnect name="Modal"
                note="This also signs out Modal's own tools on this Mac. Anything running on Modal keeps going, but the app can't bring it back until you sign in again."
                run={async () => { await connectionApi.modalSignOut(); await check("modal"); }} />}
              <button className="btn sm" onClick={() => check(b.id)} disabled={busy[b.id]}>Check</button></div>
            {checks[b.id] && <p className="small muted conn-detail">{here(checks[b.id].detail)}</p>}
            <IssueLine issue={issues.of(b.id)} onDismiss={issues.dismiss} />
            {b.id === "modal" && checks.modal && <CloudSignIn service="modal" signedIn={checks.modal.ok && !issues.of("modal")} onDone={() => check("modal")} />}
            {b.id === "modal" && checks.modal?.ok && <ModalVoices />}
            {b.id === "modal" && <HowTo title="How to set up Modal" open={needsSetup(checks.modal)} steps={HOWTO.modal} path={PATHS.modal} />}
            {b.id !== "modal" && (
              <HowTo title="How to use it" steps={[<>Nothing to set up. When you train a model, choose <b>{b.name}</b> under <b>Train on</b>.</>]}
                path={[["Use it", "Train a voice", "Train a model", "Train on", b.name]]} />
            )}
          </div>
        ))}
      </div>

      <div className="section-label" style={{ marginTop: 18 }}>Pay-as-you-go · Create in the cloud</div>
      <p className="small muted section-note">Use credit you already have on any of these. Each rents a GPU for one book and frees it when the lines are back. Set one up and it appears in the Create menu.</p>
      <div className="stack">
        {clouds.filter((sv) => sv.has_key).concat(clouds.filter((sv) => !sv.has_key)).map((sv) => (
          <div key={sv.id} className={issues.of(sv.id) ? "conn-with-issue" : undefined}>
            <CloudKeyCard service={sv} onChange={() => { loadClouds(); issues.reload(); }} />
            <IssueLine issue={issues.of(sv.id)} onDismiss={issues.dismiss} />
          </div>
        ))}
      </div>
      </Group>

      <div className="conn-section-head">Speech Server &amp; Downloads</div>
      <Group id="speech" title="Speech server" forceOpen={focusGroup === "speech"} focusNonce={focusNonce} summary={conn.data?.orpheus.where === "server" ? "Orpheus server" : "Orpheus on this Mac"}>
      <div className="conn-card">
        <div className="row"><div className="kind-icon ambience"><Server size={16} /></div>
          <div className="grow"><b>Orpheus API server</b><div className="small muted">Runs Orpheus voices on a fast GPU instead of this Mac. Any server with an OpenAI-style speech endpoint, such as Orpheus-TTS-FastAPI.</div></div>
          {(!!conn.data?.orpheus.url || !!conn.data?.orpheus.has_key || conn.data?.orpheus.where === "server") && (
            <Disconnect name="the Orpheus server" note="Orpheus voices go back to running on this Mac."
              run={async () => { await connectionApi.forgetOrpheus(); setOrpheus({ where: "mac", url: "", key: "" }); setOrpheusNote(""); conn.reload(); }} />
          )}</div>
        <div className="stack" style={{ marginTop: 12, gap: 10 }}>
          <Switch label="Use the server for Orpheus voices" hint="Off: Orpheus runs on this Mac (free, slower)."
            checked={orpheus.where === "server"} onChange={(v) => setOrpheus({ ...orpheus, where: v ? "server" : "mac" })} />
          <Field label="Address"><input className="input" placeholder="https://my-gpu-box:8880" value={orpheus.url} onChange={(e) => setOrpheus({ ...orpheus, url: e.target.value })} /></Field>
          <Field label="API key" hint={conn.data?.orpheus.has_key ? "A key is saved. Type a new one to replace it." : "If the server needs one."}>
            <input className="input" type="password" autoComplete="off" placeholder={conn.data?.orpheus.has_key ? "••••••••" : ""} value={orpheus.key} onChange={(e) => setOrpheus({ ...orpheus, key: e.target.value })} />
          </Field>
          <div className="row">
            <span className="small muted grow">{orpheusNote}</span>
            <button className="btn" onClick={() => saveOrpheus(true)}>Test</button>
            <button className="btn primary" onClick={() => saveOrpheus(false)}>Save</button>
          </div>
        </div>
        <HowTo title="Where to find the address and key" steps={HOWTO.orpheus} path={PATHS.orpheus} />
      </div>
      </Group>

      <Group id="downloads" title="Downloads" summary={conn.data?.huggingface.has_token ? "Hugging Face token saved" : "No token"}>
      <div className="conn-card">
        <div className="row"><div className="kind-icon effect"><Download size={16} /></div>
          <div className="grow"><b>Hugging Face</b><div className="small muted">Optional. An access token makes model downloads faster and steadier at busy times.</div></div>
          <Status check={checks.huggingface} busy={!!busy.huggingface} />
          {!!conn.data?.huggingface.has_token && <Disconnect name="Hugging Face" note="This also removes the token for Hugging Face's own tools on this Mac."
            run={async () => { await connectionApi.forgetHf(); conn.reload(); await check("huggingface"); }} />}</div>
        {checks.huggingface && <p className="small muted conn-detail">{checks.huggingface.detail}</p>}
        <div className="row" style={{ marginTop: 10 }}>
          <div className="search-box grow"><KeyRound size={14} className="muted" />
            <input className="input" type="password" autoComplete="off" placeholder={conn.data?.huggingface.has_token ? "A token is saved. Paste a new one to replace it." : "Paste a read-only access token"}
              aria-label="Hugging Face access token" value={hf} onChange={(e) => setHf(e.target.value)} /></div>
          <button className="btn primary" disabled={!hf.trim()} onClick={saveHf}>Save</button>
        </div>
        <HowTo title="How to get a Hugging Face token" steps={HOWTO.huggingface} path={PATHS.huggingface} />
      </div>
      </Group>
    </>
  );
  return embedded ? <div className="conn-embedded">{body}</div> : <div className="page narrow">{body}</div>;
}
