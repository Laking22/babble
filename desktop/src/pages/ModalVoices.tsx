import { useEffect, useState } from "react";
import { CheckCircle2, CloudUpload, Trash2 } from "lucide-react";
import { connectionApi, useTask, type ModalModel } from "../api";
import { useToast } from "../components/ui";

/** Connections → Modal: put your trained voices on Modal, so Create in the cloud makes their lines there too
 *  (otherwise they're made on this Mac when the book is finished). */
export function ModalVoices() {
  const toast = useToast();
  const [models, setModels] = useState<ModalModel[] | null>(null);
  const [which, setWhich] = useState<string | null>(null);
  const load = () => connectionApi.modalModels().then(setModels).catch((e) => toast((e as Error).message, true));
  useEffect(() => { load(); }, []); // eslint-disable-line react-hooks/exhaustive-deps
  const upload = useTask((t) => {
    setWhich(null);
    if (t.state === "failed") toast(t.error || "The upload didn't work.", true);
    else toast(t.status.split("\n")[0].replace(/^Done · /, ""));
    load();
  }, "modal-voices-upload");
  const busy = upload.task?.state === "running";
  const start = (m: ModalModel) => { setWhich(m.folder); upload.start(() => connectionApi.modalUpload(m.folder)); };
  const remove = async (m: ModalModel) => {
    try { setModels(await connectionApi.modalRemove(m.folder)); toast(`${m.name} is off Modal.`); }
    catch (e) { toast((e as Error).message, true); }
  };

  return (
    <div className="modal-voices">
      <div className="small"><b>Your trained voices on Modal</b></div>
      <p className="small muted" style={{ margin: "2px 0 8px" }}>
        Upload a trained voice once and Create in the cloud makes its lines on Modal too. It's the same model this Mac
        plays (1.9 GB each, uploaded from here), so the voice sounds the same. Kept in your Modal storage.
      </p>
      {!models ? <div className="small muted">Loading…</div> : !models.length ? <div className="small muted">No trained voices installed yet.</div> : (
        <div className="stack modal-voice-list" style={{ gap: 6 }}>
          {models.map((m) => (
            <div key={m.folder} className="modal-voice row">
              <div className="grow" style={{ minWidth: 0 }}>
                <b className="small">{m.voices.join(", ")}</b>
                <div className="small faint">{m.name} · {(m.mb / 1000).toFixed(1)} GB
                  {m.state === "changed" ? " · changed since it was uploaded" : ""}</div>
                {busy && which === m.folder && <div className="small muted status-line">{upload.task?.status.split("\n")[0]}</div>}
              </div>
              {m.state === "uploaded" ? (
                <>
                  <span className="chip ok"><CheckCircle2 size={11} />On Modal</span>
                  <button className="btn sm icon ghost" title="Take it off Modal (this Mac keeps its copy)" aria-label={`Take ${m.name} off Modal`}
                    disabled={busy} onClick={() => remove(m)}><Trash2 size={13} /></button>
                </>
              ) : (
                <button className="btn sm" disabled={busy} onClick={() => start(m)}>
                  <CloudUpload size={13} />{busy && which === m.folder ? "Uploading…" : m.state === "changed" ? "Upload again" : "Upload"}
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
