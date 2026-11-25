import { useEffect, useState } from "react";
import { CheckCircle2, Download, Loader2, Lock, RotateCcw, Trash2, Wrench } from "lucide-react";
import { api, libraryApi, sttApi, useLoad, useTask, type ModelInfo, type SttModel } from "../api";
import { ConfirmDialog, Segmented, Switch, TaskProgress, useToast } from "../components/ui";

const GROUP_LABEL = { light: "Light", standard: "Standard", high: "High quality" };

/** All local AI models: speech synthesis (TTS) and speech-to-text (STT). Each downloads once. */
export function Models({ onChanged, goConnections }: { onChanged: () => void; goConnections: () => void }) {
  const toast = useToast();
  const models = useLoad(libraryApi.models);
  const stt = useLoad(sttApi.models);
  const setup = useLoad(libraryApi.voicesSetup);
  const [server, setServer] = useState<Record<string, any>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [deletingStt, setDeletingStt] = useState<string | null>(null);
  const job = useTask((t) => {
    setBusy(null);
    if (t.state === "failed") toast(t.error || "That didn't finish.", true);
    else if (t.state === "done") { toast(t.status || "Done."); models.reload(); setup.reload(); onChanged(); }
  });
  useEffect(() => { api.settings().then(setServer).catch(() => {}); }, []);
  const saveServer = (k: string, v: any) => { setServer((s) => ({ ...s, [k]: v })); api.saveSettings({ [k]: v }).catch((e) => toast(e.message, true)); };
  const run = (id: string, kind: "download" | "extras" | "again") => {
    setBusy(id);
    job.start(() => (kind === "download" ? libraryApi.download(id) : kind === "again" ? libraryApi.redownload(id) : libraryApi.extras(id)));
  };
  const [asking, setAsking] = useState<{ m: ModelInfo; what: "delete" | "again" } | null>(null);
  const setOff = async (m: ModelInfo, off: boolean) => {
    try {
      await libraryApi.setOff(m.id, off);
      toast(off ? `${m.name} is off: its voices are hidden until you turn it on.` : `${m.name} is on.`);
      models.reload(); onChanged();
    } catch (e) { toast((e as Error).message, true); }
  };
  const remove = async (m: ModelInfo) => {
    try {
      const r = await libraryApi.deleteModel(m.id);
      toast(`Deleted ${m.name}${r.freed_gb ? ` (${r.freed_gb} GB freed)` : ""}.`);
      models.reload(); onChanged();
    } catch (e) { toast((e as Error).message, true); }
  };
  const removeStt = async (m: SttModel) => {
    setDeletingStt(m.id);
    try {
      const r = await sttApi.deleteModel(m.id);
      toast(`Deleted ${m.name}${r.freed_gb ? ` (${r.freed_gb.toFixed(1)} GB freed)` : ""}.`);
      stt.reload();
    } catch (e) { toast((e as Error).message, true); }
    finally { setDeletingStt(null); }
  };

  const card = (m: ModelInfo) => {
    const ready = m.ok && m.downloaded;
    return (
      <div key={m.id} className={`model-card ${m.locked ? "locked" : ""}`}>
        <div className="row" style={{ alignItems: "flex-start" }}>
          <div className="grow">
            <b>{m.name}</b> <span className="chip">{GROUP_LABEL[m.group]}</span>
            <div className="small muted">{m.license} · {m.models}{m.size_gb ? ` · ${m.size_gb} GB on this Mac` : ""}</div>
          </div>
          {m.locked ? <span className="chip"><Lock size={11} />Needs {m.min_ram_gb} GB</span>
            : m.off ? <span className="chip">Off</span>
            : ready ? <span className="chip ok"><CheckCircle2 size={11} />Ready</span>
              : !m.downloaded ? <button className="btn sm primary" disabled={!!busy} onClick={() => run(m.id, "download")}><Download size={13} />Download</button>
                : <span className="chip">Needs setup</span>}
        </div>
        {m.locked && <p className="small muted" style={{ margin: "8px 0 0" }}>{m.why}</p>}
        {!m.locked && !m.ok && m.why && <p className="small muted" style={{ margin: "8px 0 0" }}>{m.why}</p>}
        {m.note && <p className="small muted" style={{ margin: "8px 0 0" }}>{m.note}</p>}
        {!m.locked && !m.downloaded && m.group !== "high" && <p className="small faint" style={{ margin: "8px 0 0" }}>Downloads by itself the first time you use one of its voices.</p>}
        {!m.locked && !m.ok && m.pip.length > 0 && (
          <button className="btn sm" style={{ marginTop: 8 }} disabled={!!busy} onClick={() => run(m.id, "extras")}><Wrench size={13} />Install what it needs</button>
        )}
        {busy === m.id && job.task && <div style={{ marginTop: 10 }}><TaskProgress task={job.task} onStop={job.stop} /></div>}
        {!m.locked && (
          <div className="stack" style={{ marginTop: 12, gap: 8 }}>
            <Switch label="Use this model" checked={!m.off} onChange={(on) => setOff(m, !on)}
              hint={m.off ? "Off: its voices are hidden from the voice lists, and nothing is made with it." : undefined} />
            {m.size_gb > 0 && (
              <div className="row" style={{ gap: 8 }}>
                <span className="spacer" />
                <button className="btn sm" disabled={!!busy} onClick={() => setAsking({ m, what: "again" })}
                  title="Delete the download and download it again (if it's not working right)"><RotateCcw size={13} />Download again</button>
                <button className="btn sm ghost" disabled={!!busy} onClick={() => setAsking({ m, what: "delete" })}
                  title="Free up its space on this Mac"><Trash2 size={13} />Delete</button>
              </div>
            )}
          </div>
        )}
        {m.id === "orpheus" && (
          <div className="stack" style={{ marginTop: 12, gap: 10 }}>
            <div className="row">
              <span className="small grow">Run Orpheus on</span>
              <Segmented<string> value={server.orpheus_where || "mac"} onChange={(v) => saveServer("orpheus_where", v)}
                options={[{ value: "mac", label: "This Mac" }, { value: "server", label: "API server" }]} />
            </div>
            {server.orpheus_where === "server" && <button className="link small accent-link" onClick={goConnections}>Set the server's address in Connections</button>}
            {(server.orpheus_where || "mac") === "mac" && (
              <p className="small muted" style={{ margin: 0 }}>Several lines at once: the switch in Text to Speech and in each audiobook's Output.</p>
            )}
          </div>
        )}
      </div>
    );
  };

  const list = models.data || [];
  const sttList = stt.data?.models || [];
  return (
    <div className="page narrow">
      <div className="page-head"><div><h1>Models</h1><p>The local speech synthesis and transcription models Babble uses. Each downloads once.</p></div></div>
      {setup.data?.needed && (
        <div className="model-card" style={{ marginBottom: 16 }}>
          <div className="row" style={{ alignItems: "center" }}>
            <div className="grow">
              <b>Set up voices</b>
              <div className="small muted">{setup.data.gpu || "No NVIDIA graphics card (slow)"} · PyTorch · about {setup.data.size_gb} GB</div>
            </div>
            <button className="btn sm primary" disabled={!!busy}
              onClick={() => { setBusy("voices-setup"); job.start(() => libraryApi.setUpVoices()); }}>
              <Download size={14} />Set up</button>
          </div>
          {busy === "voices-setup" && job.task && <div style={{ marginTop: 10 }}><TaskProgress task={job.task} onStop={job.stop} /></div>}
        </div>
      )}
      <div className="section-label">Speech synthesis · Ready to use</div>
      <div className="stack">{list.filter((m) => m.group !== "high").map(card)}</div>
      <div className="section-label" style={{ marginTop: 24 }}>Speech synthesis · High quality</div>
      <p className="small muted" style={{ margin: "0 0 10px" }}>Bigger and slower, and their voices only join the voice lists once downloaded.</p>
      <div className="stack">{list.filter((m) => m.group === "high").map(card)}</div>

      {sttList.length > 0 && (
        <>
          <div className="section-label" style={{ marginTop: 24 }}>Speech to Text</div>
          <p className="small muted" style={{ margin: "0 0 10px" }}>Used on the Speech to Text page and for voice-clone transcription. Downloads automatically on first use.</p>
          <div className="stack">
            {sttList.map((m) => (
              <div key={m.id} className="model-card">
                <div className="row" style={{ alignItems: "flex-start" }}>
                  <div className="grow">
                    <b>{m.name}</b> <span className="chip">{m.languages}</span>
                    <div className="small muted">{m.license} · ~{m.size_gb.toFixed(1)} GB{m.core ? " · Core" : ""}</div>
                  </div>
                  {m.downloaded ? <span className="chip ok"><CheckCircle2 size={11} />Downloaded</span>
                    : <span className="chip">Downloads on first use</span>}
                </div>
                <p className="small muted" style={{ margin: "8px 0 0" }}>{m.summary}</p>
                {m.downloaded && !m.core && (
                  <div className="row" style={{ marginTop: 10 }}>
                    <span className="spacer" />
                    <button className="btn sm ghost" disabled={deletingStt === m.id} onClick={() => removeStt(m)}
                      title={`Delete ${m.name} weights (~${m.size_gb.toFixed(1)} GB)`}>
                      {deletingStt === m.id ? <Loader2 size={13} className="spin" /> : <Trash2 size={13} />}Delete
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        </>
      )}

      {models.error && <div className="card card-pad empty">{models.error}</div>}
      {asking && (
        <ConfirmDialog
          title={asking.what === "delete" ? `Delete ${asking.m.name}?` : `Download ${asking.m.name} again?`}
          confirm={asking.what === "delete" ? `Delete ${asking.m.size_gb} GB` : "Download again"} danger={asking.what === "delete"}
          onCancel={() => setAsking(null)}
          onConfirm={() => { const a = asking; setAsking(null); a.what === "delete" ? remove(a.m) : run(a.m.id, "again"); }}>
          {asking.what === "delete"
            ? <p>Its {asking.m.size_gb} GB of downloaded files leave this Mac. Voices you trained or saved stay. It downloads
              again the first time you use one of its voices, or with Download on this page.</p>
            : <p>Deletes its download ({asking.m.size_gb} GB) and downloads it fresh: for a model that isn't working right.
              It needs the internet and takes a while.</p>}
        </ConfirmDialog>
      )}
    </div>
  );
}
