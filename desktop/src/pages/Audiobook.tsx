import { useEffect, useMemo, useRef, useState } from "react";
import { BookOpen, Download, FileUp, LayoutGrid, Plus, Rows3, Search, Trash2 } from "lucide-react";
import { bookApi, fileUrl, projectApi, useLoad, type ProjectSummary } from "../api";
import { ConfirmDialog, Segmented, Starting, accepts, taskTiming, timeLeft, useFileDrop, useNow, useStarting, useToast } from "../components/ui";
import { useBookJobs } from "../bookJobs";
import { saveFile } from "../native";
import { SHELF_SIZES, useShelfPrefs } from "../uiPrefs";

type Filter = "all" | "making" | "finished" | "drafts";
const BOOK_FILES = ".txt,.md,.epub,.pdf,.docx,.mobi,.azw3,.rtf,.odt,.fb2,.html";

/** "Edited today", "Edited yesterday", "Edited Nov 3". */
function edited(ts: number) {
  const d = new Date(ts * 1000), now = new Date();
  const days = Math.round((new Date(now.toDateString()).getTime() - new Date(d.toDateString()).getTime()) / 86400000);
  return days <= 0 ? "Edited today" : days === 1 ? "Edited yesterday" : `Edited ${d.toLocaleDateString(undefined, { month: "short", day: "numeric" })}`;
}

const plural = (n: number, one: string) => `${n} ${one}${n === 1 ? "" : "s"}`;

/** Every audiobook as a card; opening one shows its own editor. The one book the page used to keep moves in here. */
export function Audiobook({ onOpen }: { onOpen: (id: string) => void }) {
  const toast = useToast();
  const { shape, size, setShape, setSize } = useShelfPrefs();
  const list = useLoad(projectApi.list);
  const folders = useLoad(bookApi.list);
  const options = useLoad(bookApi.options);
  const { jobs, onChange } = useBookJobs();
  const now = useNow(Object.values(jobs).some((j) => j?.state === "running"), 5000);
  const [q, setQ] = useState("");
  const [filter, setFilter] = useState<Filter>("all");
  const [confirmBook, setConfirmBook] = useState<ProjectSummary | null>(null);
  const [deleteBookAudio, setDeleteBookAudio] = useState(false);
  const [confirmRecent, setConfirmRecent] = useState<{ name: string; title: string } | null>(null);
  const importInput = useRef<HTMLInputElement>(null);
  const migrating = useRef(false);

  useEffect(() => onChange(() => list.reload()), [onChange]); // eslint-disable-line react-hooks/exhaustive-deps
  const anyInCloud = (list.data || []).some((p) => p.cloud);
  useEffect(() => { // books out in the cloud: their cards follow along
    if (!anyInCloud) return;
    const t = setInterval(() => list.reload(), 30000);
    return () => clearInterval(t);
  }, [anyInCloud]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { // the page's old single book (kept in this window) becomes the first project, once
    if (!list.data || list.data.length || migrating.current) return;
    let old: Record<string, any> | null = null;
    try {
      if (!localStorage.getItem("book-moved")) old = JSON.parse(localStorage.getItem("book-form") || "null");
    } catch { old = null; }
    if (!old || !String(old.story || "").trim()) return;
    migrating.current = true;
    const keep = ["title", "author", "cover", "story", "original", "cast", "models", "quotes_only", "out_format", "speed",
      "stability", "language", "seed", "batch", "line_pause", "para_pause", "sound", "bed", "intro", "detect"];
    projectApi.create({ ...Object.fromEntries(keep.filter((k) => k in old!).map((k) => [k, old![k]])), legacy: true })
      .then(() => { try { localStorage.setItem("book-moved", "1"); } catch { /* fine */ } list.reload(); })
      .catch((e) => toast(e.message, true));
  }, [list.data]); // eslint-disable-line react-hooks/exhaustive-deps

  const nextUntitled = useMemo(() => Math.max(0, ...(list.data || []).filter((p) => !p.title.trim()).map((p) => p.untitled)) + 1, [list.data]);
  const state = (p: ProjectSummary) => {
    const t = jobs[p.id];
    if (t?.state === "running" || p.cloud) return "making";
    return p.file ? "finished" : "drafts";
  };
  const shown = useMemo(() => {
    const words = q.toLowerCase().split(/\s+/).filter(Boolean);
    return (list.data || []).filter((p) => (filter === "all" || state(p) === filter)
      && words.every((w) => `${p.name} ${p.author}`.toLowerCase().includes(w)));
  }, [list.data, q, filter, jobs]); // eslint-disable-line react-hooks/exhaustive-deps
  const counts = useMemo(() => {
    const c = { all: 0, making: 0, finished: 0, drafts: 0 };
    (list.data || []).forEach((p) => { c.all++; c[state(p)]++; });
    return c;
  }, [list.data, jobs]); // eslint-disable-line react-hooks/exhaustive-deps
  const ownFolders = new Set((list.data || []).map((p) => p.folder));
  const earlier = (folders.data || []).filter((f) => f.file && !ownFolders.has(f.name));

  const opening = useStarting();  // New audiobook, Import a book: one book per click
  const create = () => opening.run(async () => {
    try { onOpen((await projectApi.create({})).id); } catch (e) { toast((e as Error).message, true); }
  });
  const importFile = (f: File) => opening.run(async () => {
    try {
      const r = await bookApi.importFile(f);
      const p = await projectApi.create({ story: r.text, title: r.title, author: r.author, cover: r.cover,
        out_format: options.data?.formats[0] || "WAV" });
      toast(r.message);
      onOpen(p.id);
    } catch (e) { toast((e as Error).message, true); }
  });
  // a book file dragged in from Finder becomes a new audiobook, like Import a book
  const drop = useFileDrop((got) => {
    const fit = got.filter((f) => accepts(f, BOOK_FILES));
    if (!fit.length) return toast("That isn't a book or text file.", true);
    if (fit.length > 1) toast(`Opening ${fit[0].name}. Drop one book at a time.`);
    importFile(fit[0]);
  }, opening.starting);
  const remove = async (p: ProjectSummary, deleteAudio: boolean) => {
    try {
      await projectApi.remove(p.id, deleteAudio);
      setConfirmBook(null);
      setDeleteBookAudio(false);
      toast(deleteAudio ? `Deleted “${p.name}” and its generated audio files.` : `Deleted “${p.name}”. Generated audio files were kept.`);
      list.reload();
      folders.reload();
    } catch (e) { toast((e as Error).message, true); }
  };
  const removeRecent = async (f: { name: string; title: string }) => {
    try { await bookApi.deleteFolder(f.name); setConfirmRecent(null); toast(`Deleted “${f.title}” and freed its audio files.`); folders.reload(); }
    catch (e) { toast((e as Error).message, true); }
  };
  const save = (path: string) => saveFile(path).then((m) => toast(m), (e) => toast(String(e), true));

  return (
    <div className={`page${drop.over ? " file-over" : ""}`} {...drop.props}>
      <div className="page-head">
        <div><h1>Audiobooks</h1><p>A book or story read by a full cast, with sound effects, music and chapter intros.</p></div>
        <div className="row">
          <input ref={importInput} type="file" hidden accept={BOOK_FILES}
            onChange={(e) => { const f = e.target.files?.[0]; if (f) importFile(f); e.target.value = ""; }} />
          <button className="btn" disabled={opening.starting} onClick={() => importInput.current?.click()}><FileUp size={15} />Import a book</button>
          <button className="btn primary" disabled={opening.starting} onClick={create}>
            {opening.starting ? <Starting size={15} label="Opening…" /> : <><Plus size={15} />New audiobook</>}</button>
        </div>
      </div>

      <div className="row wrap" style={{ marginBottom: 16 }}>
        <div className="search-box">
          <Search size={15} className="muted" />
          <input className="input" placeholder="Search audiobooks" aria-label="Search audiobooks" value={q} onChange={(e) => setQ(e.target.value)} />
        </div>
        <Segmented<Filter> value={filter} onChange={setFilter} options={[
          { value: "all", label: `All ${counts.all}` },
          ...(counts.making ? [{ value: "making" as Filter, label: `Making ${counts.making}` }] : []),
          { value: "finished", label: `Finished ${counts.finished}` },
          { value: "drafts", label: `Drafts ${counts.drafts}` },
        ]} />
        <span className="spacer" />
        <div className="shelf-view-ctl" aria-label="Shelf layout">
          <div className="segmented" role="group" aria-label="Card shape">
            <button type="button" className={shape === "vertical" ? "on" : ""} onClick={() => setShape("vertical")} title="Vertical cards" aria-label="Vertical cards"><LayoutGrid size={14} /></button>
            <button type="button" className={shape === "horizontal" ? "on" : ""} onClick={() => setShape("horizontal")} title="Horizontal rows" aria-label="Horizontal rows"><Rows3 size={14} /></button>
          </div>
          <div className="segmented" role="group" aria-label="Card size">
            {SHELF_SIZES.map((sz) => (
              <button key={sz} type="button" className={size === sz ? "on" : ""} onClick={() => setSize(sz)} title={`${sz[0].toUpperCase() + sz.slice(1)} cards`}>
                {sz[0].toUpperCase()}
              </button>
            ))}
          </div>
        </div>
      </div>

      {list.error && <div className="card card-pad empty">{list.error}</div>}
      <div className={`shelf shape-${shape} size-${size}`}>
        {shown.map((p) => {
          const t = jobs[p.id];
          const making = t?.state === "running";
          const inCloud = !making && p.cloud;
          const back = inCloud && p.cloud!.state === "ready";
          const cloudShare = !inCloud ? null : back ? (p.cloud!.stitch ? p.cloud!.stitch_progress ?? null : null) : p.cloud!.progress ?? null;
          const st = state(p);
          return (
            <div
              key={p.id}
              className="book-card clickable"
              role="button"
              tabIndex={0}
              aria-label={`Open ${p.name}`}
              onClick={() => onOpen(p.id)}
              onKeyDown={(e) => { if (e.target !== e.currentTarget) return; if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onOpen(p.id); } }}
            >
              <div className="book-cover" style={{ ["--c" as string]: p.color }}>
                {p.cover ? <img src={fileUrl(p.cover)} alt="" /> : <span className="cover-title">{p.name}</span>}
                {cloudShare != null && <span className="cover-bar"><i style={{ width: `${Math.round(cloudShare * 100)}%` }} /></span>}
                {making && <span className="cover-bar"><i style={{ width: `${Math.round((t.progress ?? 0) * 100)}%` }} /></span>}
              </div>
              <div className="book-card-body">
                <div className="book-card-top">
                  <span className="book-name">{p.name}</span>
                  <div className="small muted book-meta-line">
                    {p.words ? `${plural(p.chapters, "chapter")} · ${plural(p.cast, "character")}` : "No text yet"}
                    {p.file && p.format ? ` · ${p.file.split(".").pop()?.toUpperCase()}` : ""}
                  </div>
                </div>
                <div className="card-foot">
                  <div className="card-status">
                    {inCloud ? (() => {
                      const pct = cloudShare != null ? `${Math.round(cloudShare * 100)}%` : "";
                      const eta = back ? p.cloud!.stitch_eta : p.cloud!.eta;
                      const etaTxt = timeLeft(eta).replace(/ left$/, "");
                      const hasDetail = Boolean(pct || etaTxt);
                      const stLabel = back
                        ? (p.cloud!.stitch ? (/^(Finalizing|Encoding|Exporting)/i.test(p.cloud!.stitch) ? "Finalizing" : "Assembling") : "Ready to finish")
                        : p.cloud!.state === "downloading" ? (hasDetail ? "Downloading" : "Downloading…")
                        : p.cloud!.state === "queued" ? (hasDetail ? "Queued" : "Queued…") : "Cloud";
                      return <span className="chip accent" title={p.cloud!.stitch || ""}>{[stLabel, pct, etaTxt].filter(Boolean).join(" · ")}</span>;
                    })()
                      : st === "making" ? <span className="chip accent" title={(t.status || "").split("\n")[0]}>{["Making", t.progress != null ? `${Math.round(t.progress * 100)}%` : "", taskTiming(t, now).left.replace(/ left$/, "")].filter(Boolean).join(" · ")}</span>
                      : st === "finished" ? <span className="chip ok">Finished</span> : <span className="chip">Draft</span>}
                    <span className="small faint card-edited">{edited(p.updated)}</span>
                  </div>
                  <div className="card-actions" onClick={(e) => e.stopPropagation()}>
                    {p.file && <button type="button" className="btn icon sm ghost" title="Save the audiobook file" aria-label={`Save ${p.name}`} onClick={() => save(p.file!)}><Download size={14} /></button>}
                    <button type="button" className="btn icon sm ghost" title={`Delete ${p.name}`} aria-label={`Delete ${p.name}`} disabled={making} onClick={() => { setDeleteBookAudio(false); setConfirmBook(p); }}><Trash2 size={14} /></button>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
        {filter === "all" && !q && (
          <button type="button" className="book-card new-book" disabled={opening.starting} onClick={create}>
            <span className="book-cover new"><Plus size={26} /></span>
            <span className="book-card-body">
              <span className="book-name">New audiobook</span>
              <span className="small muted" style={{ display: "block" }}>Starts as Untitled-{nextUntitled}</span>
            </span>
          </button>
        )}
      </div>
      {list.data && !shown.length && (q || filter !== "all") && <div className="empty small">No audiobooks match.</div>}

      {confirmBook && (
        <ConfirmDialog
          title={`Delete “${confirmBook.name}”?`}
          confirm={deleteBookAudio ? "Delete book & audio" : "Delete"}
          danger
          onCancel={() => { setConfirmBook(null); setDeleteBookAudio(false); }}
          onConfirm={() => remove(confirmBook, deleteBookAudio)}
        >
          <p style={{ margin: "0 0 10px" }}>
            Removes the script and settings.
            {confirmBook.cloud && confirmBook.cloud.state !== "ready" ? " Its cloud run will also be stopped." : ""}
          </p>
          <label className="row small" style={{ gap: 8, cursor: "pointer" }}>
            <input type="checkbox" checked={deleteBookAudio} onChange={(e) => setDeleteBookAudio(e.target.checked)} style={{ accentColor: "var(--accent)" }} />
            <span>Also permanently delete generated audio files for this book</span>
          </label>
        </ConfirmDialog>
      )}

      {confirmRecent && (
        <ConfirmDialog
          title={`Delete “${confirmRecent.title}”?`}
          confirm="Delete files"
          danger
          onCancel={() => setConfirmRecent(null)}
          onConfirm={() => removeRecent(confirmRecent)}
        >
          Permanently deletes this audiobook folder and all its audio files to free up storage. This can't be undone.
        </ConfirmDialog>
      )}

      {earlier.length > 0 && (
        <div className="card" style={{ marginTop: 28 }}>
          <div className="card-pad" style={{ paddingBottom: 6 }}>
            <h3 className="card-title" style={{ margin: 0 }}>Recents</h3>
            <p className="small muted" style={{ margin: "2px 0 0" }}>Recent exports and earlier audiobook folders. Delete unwanted files to free up storage.</p>
          </div>
          <div className="list">
            {earlier.slice(0, 12).map((f) => (
              <div key={f.folder} className="list-item">
                <div className="kind-icon effect"><BookOpen size={16} /></div>
                <div className="grow">
                  <div className="title">{f.title}</div>
                  <div className="small muted">{plural(f.chapters, "chapter")} · {f.file!.split(".").pop()?.toUpperCase()}</div>
                </div>
                <audio className="recent-audio" src={fileUrl(f.file!)} controls preload="none" />
                <button type="button" className="btn icon sm ghost" onClick={() => save(f.file!)} title="Save the file" aria-label={`Save ${f.title}`}><Download size={13} /></button>
                <button type="button" className="btn icon sm ghost" onClick={() => setConfirmRecent({ name: f.name, title: f.title })} title={`Delete ${f.title} files`} aria-label={`Delete ${f.title}`}><Trash2 size={13} /></button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
