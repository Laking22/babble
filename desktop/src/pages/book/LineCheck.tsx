import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Pause, Play, Undo2 } from "lucide-react";
import { api, bookApi, fileUrl, useTask, type BookRequest, type CheckedLine, type LineCheck } from "../../api";
import { TaskProgress, useToast } from "../../components/ui";

/** How many cloud lines this book has and its last check (null while loading), reloaded on demand, and the
 * check running for it (kept here, not in the window: closing the window doesn't lose it, and a check started
 * before the page was opened is picked up again). */
export function useLineCheck(folder?: string | null) {
  const [info, setInfo] = useState<LineCheck | null>(null);
  const toast = useToast();
  const load = () => { if (folder) bookApi.linesChecked(folder).then(setInfo, () => setInfo(null)); };
  useEffect(load, [folder]);
  const checking = useTask((t) => {
    if (t.state === "failed") return toast(t.error || "Couldn't check the lines.", true);
    if (t.state === "done") { load(); toast(t.status.replace(/^Done · /, "").replace(/^c/, "C")); }
  });
  useEffect(() => { // a check already running for this book (the page was closed, or reloaded)
    if (!folder) return;
    api.tasks().then((ts) => {
      const t = ts.find((x) => x.kind === "lines" && x.state === "running" && x.result?.folder === folder);
      if (t) checking.resume(t);
    }).catch(() => {});
  }, [folder]); // eslint-disable-line react-hooks/exhaustive-deps
  return { info, setInfo, reload: load, checking };
}

const when = (t?: number) => (t ? new Date(t * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }) : "");
/** Where a line's audio is now: set aside lines wait in cloud-lines/set-aside until they're made again. */
const src = (l: CheckedLine) => (l.set_aside && !l.remade ? l.path.replace(/([\\/])([^\\/]+)$/, "$1set-aside$1$2") : l.path);
const who = (voice: string) => voice.split(":").slice(1).join(":") || voice;

/** Check lines: hear back the cloud lines that look wrong, and set the ones that are aside to be made again. */
export function LineCheckDialog(props: {
  folder: string; info: LineCheck | null; setInfo: (i: LineCheck) => void; request: () => BookRequest;
  checking: ReturnType<typeof useLineCheck>["checking"]; busy: boolean; onClose: () => void;
}) {
  const { folder, info, setInfo, checking } = props;
  const toast = useToast();
  const wrong = info?.wrong ?? [];
  const aside = info?.aside ?? [];
  const [ticked, setTicked] = useState<Set<string>>(() => new Set(wrong.map((l) => l.key)));
  useEffect(() => setTicked(new Set((info?.wrong ?? []).map((l) => l.key))), [info?.checked]);
  const running = checking.task?.state === "running";

  // one player for every Play button: playing another line stops the first
  const el = useRef<HTMLAudioElement | null>(null);
  const [on, setOn] = useState<string | null>(null);
  useEffect(() => () => el.current?.pause(), []);
  const play = (path: string) => {
    if (!el.current) { el.current = new Audio(); el.current.onended = () => setOn(null); }
    if (on === path) { el.current.pause(); setOn(null); return; }
    el.current.src = fileUrl(path);
    el.current.play().then(() => setOn(path), () => toast("Couldn't play that line.", true));
  };

  const close = useRef(props.onClose);
  close.current = props.onClose;
  useEffect(() => {
    const key = (e: KeyboardEvent) => { if (e.key === "Escape") { e.stopPropagation(); close.current(); } };
    window.addEventListener("keydown", key, true);
    return () => window.removeEventListener("keydown", key, true);
  }, []);

  const check = () => checking.start(() => bookApi.linesCheck(props.request()));
  const setAside = async () => {
    try {
      const r = await bookApi.linesAside(folder, [...ticked]);
      setInfo({ ...r, cloud_lines: (info?.cloud_lines ?? 0) - ticked.size });
      toast(`Set aside ${ticked.size} line(s). Create in the cloud makes them again, then Finish puts them in.`);
    } catch (e) { toast((e as Error).message, true); }
  };
  const putBack = async () => {
    try {
      const r = await bookApi.linesPutBack(folder);
      setInfo({ ...r, cloud_lines: (info?.cloud_lines ?? 0) + r.put_back });
      toast(r.put_back ? `Put back ${r.put_back} line(s): Finish uses the old takes.` : "Nothing to put back: those lines were made again already.");
    } catch (e) { toast((e as Error).message, true); }
  };
  const waiting = aside.filter((l) => !l.remade);

  const row = (l: CheckedLine, tick: boolean) => (
    <div key={l.key} className="line-check-row">
      {tick && <input type="checkbox" checked={ticked.has(l.key)} aria-label={`Make “${l.text.slice(0, 40)}” again`}
        onChange={(e) => setTicked((s) => { const n = new Set(s); e.target.checked ? n.add(l.key) : n.delete(l.key); return n; })} />}
      <button className="btn sm ghost icon" onClick={() => play(src(l))} aria-label={on === src(l) ? "Pause" : "Play"}
        title={l.set_aside && !l.remade ? "Play the old take" : "Play it"}>
        {on === src(l) ? <Pause size={13} /> : <Play size={13} />}
      </button>
      <div className="line-check-text">
        <div>{l.text}</div>
        <div className="small muted">
          {l.set_aside ? (l.remade ? "Made again: Finish uses the new take" : "Set aside: Create in the cloud makes it again")
            : <>Heard: “{l.heard || "nothing"}”</>}
          {" "}· {who(l.voice)} · {l.seconds}s · chapter {l.chapters.map((c) => c + 1).join(", ")}
        </div>
      </div>
    </div>
  );

  return createPortal(
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && props.onClose()}>
      <div className="modal line-check" role="dialog" aria-modal="true" aria-labelledby="line-check-title">
        <h2 id="line-check-title">Check lines</h2>
        <div className="modal-body">
          <p>
            Finds lines from the cloud that came out wrong: garbled, cut short or said twice. Lines much longer or
            shorter than their words should take are heard back on this Mac and compared with the script
            (a few seconds each). Lines made on this Mac aren't checked.
          </p>
          {running ? <TaskProgress task={checking.task} onStop={() => checking.task && api.stopTask(checking.task.id)} />
            : checking.task?.state === "failed" ? <TaskProgress task={checking.task} /> : null}
          {!running && info?.checked != null && (
            <p className="small">
              Checked {when(info.checked)}: {info.lines?.toLocaleString()} lines, {info.screened} heard back,{" "}
              <b>{wrong.length ? `${wrong.length} sound wrong` : "none sound wrong"}</b>.
            </p>
          )}
          {!running && wrong.length > 0 && (
            <>
              <div className="line-check-list">{wrong.map((l) => row(l, true))}</div>
              <p className="small muted">
                Untick any that sound fine to you (names are often misheard). Set aside: the ticked lines are made
                again with a new seed by your next Create in the cloud (only those lines, so it's quick), and the
                chapters that use them are put together again when you press Finish.
              </p>
            </>
          )}
          {!running && aside.length > 0 && (
            <>
              <p className="small"><b>Set aside</b> ({aside.length - waiting.length} of {aside.length} made again)</p>
              <div className="line-check-list">{aside.map((l) => row(l, false))}</div>
            </>
          )}
        </div>
        <div className="modal-actions">
          {waiting.length > 0 && !running && <button className="btn" onClick={putBack} title="Use the old takes after all"><Undo2 size={14} />Put back {waiting.length}</button>}
          <span className="spacer" />
          <button className="btn" onClick={props.onClose}>Close</button>
          {wrong.length > 0 && !running
            ? <button className="btn primary" onClick={setAside} disabled={!ticked.size || props.busy}
                title={props.busy ? "Wait until the book is made" : undefined}>Set aside {ticked.size}</button>
            : <button className="btn primary" onClick={check} disabled={running || props.busy || !info?.cloud_lines}
                title={props.busy ? "Wait until the book is made" : !info?.cloud_lines ? "This book has no lines from the cloud" : undefined}>
                {info?.checked ? "Check again" : "Check lines"}</button>}
        </div>
      </div>
    </div>,
    document.body,
  );
}
