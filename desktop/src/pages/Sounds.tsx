import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { CheckSquare, Copy, Globe, Music2, Package, Pause, Play, Plus, Search, Square, Trash2, Wand2, Waves, X, Zap } from "lucide-react";
import { api, fileUrl, useLoad, useTask, type Sound } from "../api";
import { Segmented, Slider, TaskProgress, clock, useToast } from "../components/ui";
import { AddPack, FindOnline } from "./SoundSources";

type Kind = "all" | Sound["kind"];
const KIND_LABEL: Record<Sound["kind"], string> = { effect: "Effect", ambience: "Background", music: "Music" };
const TAG: Record<Sound["kind"], string> = { effect: "sfx", ambience: "amb", music: "music" };
const ICON = { effect: Zap, ambience: Waves, music: Music2 };
const HOLD_MS = 400;  // press and hold a sound this long to start selecting
const EDGE = 48;      // dragging this close to the top or bottom scrolls the list
export function Sounds({ ramGb }: { ramGb: number }) {
  const toast = useToast();
  const lib = useLoad(api.sounds);
  const [kind, setKind] = useState<Kind>("all");
  const [q, setQ] = useState("");
  const [panel, setPanel] = useState<"make" | "online" | "pack" | null>(null);
  const [playing, setPlaying] = useState<string | null>(null);
  const audio = useRef<HTMLAudioElement>(null);

  const sounds = useMemo(() => {
    const words = q.toLowerCase().split(/\s+/).filter(Boolean);
    return (lib.data || [])
      .filter((s) => !s.book) // a sound kept in one audiobook is in that book's tools, not here
      .filter((s) => kind === "all" || s.kind === kind)
      .filter((s) => words.every((w) => `${s.name} ${s.tags.join(" ")} ${s.description || ""}`.toLowerCase().includes(w)))
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [lib.data, kind, q]);

  const play = (s: Sound) => {
    const el = audio.current;
    if (!el) return;
    if (playing === s.id) { el.pause(); setPlaying(null); return; }
    el.src = fileUrl(s.path);
    el.play().catch(() => toast("Couldn't play that sound.", true));
    setPlaying(s.id);
  };

  const remove = async (s: Sound) => {  // (to the Trash, so no asking first)
    try { await api.deleteSound(s.id); toast(`Moved ${s.name} to the Trash.`); lib.reload(); }
    catch (e) { toast((e as Error).message, true); }
  };

  // ---- select mode: hold a sound to select it, keep holding and drag to select the ones you pass over ----
  const [selecting, setSelecting] = useState(false);
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [confirmMany, setConfirmMany] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);
  const hold = useRef<{ timer: number; x: number; y: number } | null>(null);
  const drag = useRef<{ anchor: number; base: Set<string>; add: boolean } | null>(null);
  const swallowClick = useRef(false);
  const pointer = useRef({ x: 0, y: 0 });
  const edgeTimer = useRef<number | null>(null);
  const live = useRef({ selecting, picked, sounds });
  live.current = { selecting, picked, sounds };

  const exitSelect = () => { setSelecting(false); setPicked(new Set()); setConfirmMany(false); };
  useEffect(() => {
    if (!selecting) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") exitSelect(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selecting]);
  useEffect(() => { // sounds that are gone (deleted, reloaded) leave the selection
    const ids = new Set((lib.data || []).map((s) => s.id));
    setPicked((p) => (Array.from(p).every((id) => ids.has(id)) ? p : new Set(Array.from(p).filter((id) => ids.has(id)))));
  }, [lib.data]);

  const rowAt = (x: number, y: number) => {
    const el = document.elementFromPoint(x, y)?.closest<HTMLElement>("[data-row]");
    return el ? Number(el.dataset.row) : -1;
  };
  const extendTo = (i: number) => {
    const d = drag.current;
    const list = live.current.sounds;
    if (!d || i < 0 || i >= list.length) return;
    const next = new Set(d.base);
    for (const s of list.slice(Math.min(d.anchor, i), Math.max(d.anchor, i) + 1)) {
      if (d.add) next.add(s.id); else next.delete(s.id);
    }
    setPicked(next);
  };
  const stopEdge = () => { if (edgeTimer.current != null) { clearInterval(edgeTimer.current); edgeTimer.current = null; } };
  const cancelHold = () => { if (hold.current) { clearTimeout(hold.current.timer); hold.current = null; } };
  const unlisten = useRef<(() => void) | null>(null);
  const keepUnder = useRef<{ row: number; top: number } | null>(null);  // the held row stays under the pointer

  /** Start picking from row i (the hold ended, or in select mode the press moved). */
  const beginDrag = (i: number) => {
    const { selecting: was, picked: had, sounds: list } = live.current;
    const id = list[i]?.id;
    if (!id) return;
    swallowClick.current = true;  // (the click that ends this press isn't a play or a toggle)
    drag.current = { anchor: i, base: was ? new Set(had) : new Set(), add: !(was && had.has(id)) };
    if (!was) {  // the bar that appears pushes the list down: scroll by as much so the held sound stays put
      const el = listRef.current?.querySelector<HTMLElement>(`[data-row="${i}"]`);
      if (el) keepUnder.current = { row: i, top: el.getBoundingClientRect().top };
    }
    setSelecting(true);
    setConfirmMany(false);
    extendTo(i);
    window.getSelection()?.removeAllRanges();
    const scroller = listRef.current?.closest<HTMLElement>(".main");
    stopEdge();
    edgeTimer.current = window.setInterval(() => { // near the top or bottom: scroll, and keep picking
      if (!scroller || !drag.current) return;
      const r = scroller.getBoundingClientRect();
      const { x, y } = pointer.current;
      const step = y < r.top + EDGE ? -14 : y > r.bottom - EDGE ? 14 : 0;
      if (step) { scroller.scrollTop += step; extendTo(rowAt(x, Math.min(Math.max(y, r.top + 1), r.bottom - 1))); }
    }, 30);
  };
  useLayoutEffect(() => {
    const k = keepUnder.current;
    if (!selecting || !k) return;
    keepUnder.current = null;
    const el = listRef.current?.querySelector<HTMLElement>(`[data-row="${k.row}"]`);
    const scroller = listRef.current?.closest<HTMLElement>(".main");
    if (el && scroller) scroller.scrollTop += el.getBoundingClientRect().top - k.top;
  }, [selecting]);

  const endPress = () => {
    cancelHold();
    stopEdge();
    drag.current = null;
    unlisten.current?.();
    unlisten.current = null;
  };
  const onRowDown = (e: React.PointerEvent, i: number) => {
    if (e.button !== 0) return;
    endPress();
    swallowClick.current = false;
    const start = { x: e.clientX, y: e.clientY };
    const touch = e.pointerType === "touch";
    pointer.current = start;
    hold.current = { ...start, timer: window.setTimeout(() => { hold.current = null; beginDrag(i); }, HOLD_MS) };
    // followed on the whole window (not the row), so the drag keeps working whatever is under the pointer
    const move = (ev: PointerEvent) => {
      pointer.current = { x: ev.clientX, y: ev.clientY };
      const far = Math.hypot(ev.clientX - start.x, ev.clientY - start.y) > 6;
      if (hold.current && far) {
        if (live.current.selecting) { cancelHold(); beginDrag(i); }  // already selecting: a press-and-drag picks at once
        else if (touch) cancelHold();  // a finger that moves first is scrolling (a mouse or trackpad keeps the hold)
      }
      if (drag.current) { ev.preventDefault(); extendTo(rowAt(ev.clientX, ev.clientY)); }
    };
    const up = () => endPress();
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", up);
    unlisten.current = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", up);
    };
  };
  useEffect(() => () => endPress(), []);

  const toggle = (id: string) => setPicked((p) => { const n = new Set(p); if (n.has(id)) n.delete(id); else n.add(id); return n; });
  const allShown = sounds.length > 0 && sounds.every((s) => picked.has(s.id));
  const selectAll = () => setPicked((p) => {
    const n = new Set(p);
    for (const s of sounds) { if (allShown) n.delete(s.id); else n.add(s.id); }
    return n;
  });
  const removePicked = async () => {
    setDeleting(true);
    try {
      const r = await api.deleteSounds(Array.from(picked));
      toast(`Moved ${r.deleted} sound${r.deleted === 1 ? "" : "s"} to the Trash.`);
      lib.reload();
      exitSelect();
    } catch (e) { toast((e as Error).message, true); }
    setDeleting(false);
  };

  const copyTag = (s: Sound) => {
    const tag = `{${TAG[s.kind]}: ${s.name.toLowerCase()}}`;
    navigator.clipboard?.writeText(tag).then(() => toast(`Copied ${tag}`), () => toast(tag));
  };

  const shared = useMemo(() => (lib.data || []).filter((s) => !s.book), [lib.data]);
  const counts = (k: Sound["kind"]) => shared.filter((s) => s.kind === k).length;

  return (
    <div className="page">
      <div className="page-head">
        <div><h1>Sounds</h1><p>Effects, backgrounds and music that your stories' tags play.</p></div>
        <div className="row">
          <button className="btn" onClick={() => setPanel((p) => p === "pack" ? null : "pack")}><Package size={15} />Add sound pack</button>
          <button className="btn" onClick={() => setPanel((p) => p === "online" ? null : "online")}><Globe size={15} />Find online</button>
          <button className="btn primary" onClick={() => setPanel((p) => p === "make" ? null : "make")}><Plus size={15} />Make with AI</button>
        </div>
      </div>
      {panel === "make" && <MakeSound ramGb={ramGb} onMade={(s) => { lib.reload(); toast(`Added ${s.name} to your library.`); }} />}
      {panel === "online" && <FindOnline onAdded={(s) => { lib.reload(); toast(`Added ${s.name} to your library.`); }} />}
      {panel === "pack" && <AddPack onAdded={lib.reload} />}

      <div className="card">
        {selecting && (
          <div className="select-bar" role="toolbar" aria-label="Selected sounds">
            <button className="btn icon sm ghost" onClick={exitSelect} title="Done (Esc)" aria-label="Leave select mode"><X size={15} /></button>
            <b>{picked.size} selected</b>
            <span className="spacer" />
            {confirmMany ? (
              <span className="row" style={{ gap: 6 }}>
                <span className="small">Move {picked.size} sound{picked.size === 1 ? "" : "s"} to the Trash?</span>
                <button className="btn sm danger" disabled={deleting} onClick={removePicked}>Delete</button>
                <button className="btn sm ghost" disabled={deleting} onClick={() => setConfirmMany(false)}>Keep</button>
              </span>
            ) : (
              <>
                <button className="btn sm" disabled={!sounds.length} onClick={selectAll}>{allShown ? "Select none" : "Select all"}</button>
                <button className="btn sm danger" disabled={!picked.size} onClick={() => setConfirmMany(true)}><Trash2 size={13} />Delete</button>
              </>
            )}
          </div>
        )}
        <div className="row wrap card-pad" style={{ paddingBottom: 12 }}>
          <Segmented<Kind> value={kind} onChange={setKind} options={[
            { value: "all", label: `All ${lib.data ? shared.length : ""}` },
            { value: "effect", label: `Effects ${counts("effect")}` },
            { value: "ambience", label: `Backgrounds ${counts("ambience")}` },
            { value: "music", label: `Music ${counts("music")}` },
          ]} />
          <span className="spacer" />
          <div className="row" style={{ position: "relative", width: 240 }}>
            <Search size={15} style={{ position: "absolute", left: 10 }} className="muted" />
            <input className="input" style={{ paddingLeft: 32 }} placeholder="Search sounds" value={q} onChange={(e) => setQ(e.target.value)} />
          </div>
        </div>
        <div className={`list sound-list ${selecting ? "selecting" : ""}`} ref={listRef}
          onClickCapture={(e) => { if (swallowClick.current) { swallowClick.current = false; e.preventDefault(); e.stopPropagation(); } }}>
          {sounds.map((s, i) => {
            const Icon = ICON[s.kind];
            const on = picked.has(s.id);
            return (
              <div key={s.id} className={`list-item ${on ? "picked" : ""}`} data-row={i}
                onPointerDown={(e) => onRowDown(e, i)} onClick={() => selecting && toggle(s.id)}
                aria-selected={selecting ? on : undefined}>
                <button className={`kind-icon ${s.kind}`} style={{ border: 0, cursor: "pointer" }}
                  onClick={(e) => { e.stopPropagation(); play(s); }} aria-label={`Play ${s.name}`}>
                  {playing === s.id ? <Pause size={16} /> : <Play size={16} />}
                </button>
                <div className="grow">
                  <div className="title">{s.name}</div>
                  <div className="small muted" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    <Icon size={11} style={{ verticalAlign: -1 }} /> {KIND_LABEL[s.kind]} · {clock(s.duration)}
                    {s.description && s.description !== s.name ? ` · ${s.description}` : ""}
                  </div>
                </div>
                {selecting ? (
                  <span className="sel-box" aria-hidden>{on ? <CheckSquare size={17} /> : <Square size={17} />}</span>
                ) : <button className="btn sm ghost" onClick={() => copyTag(s)} title="Copy its tag"><Copy size={13} />Tag</button>}
                <button className="btn icon sm sound-del" onClick={(e) => { e.stopPropagation(); remove(s); }}
                  title="Move to the Trash" aria-label={`Delete ${s.name}`}><Trash2 size={14} /></button>
              </div>
            );
          })}
          {lib.data && !sounds.length && (
            <div className="empty">{lib.data.length ? "No sounds match." : "Your library is empty. Make a sound with AI, or let a story's tags make them."}</div>
          )}
          {lib.error && <div className="empty">{lib.error}</div>}
        </div>
      </div>
      <audio ref={audio} onEnded={() => setPlaying(null)} hidden />
    </div>
  );
}

function MakeSound({ ramGb, onMade }: { ramGb: number; onMade: (s: Sound) => void }) {
  const [desc, setDesc] = useState("");
  const [kind, setKind] = useState<Sound["kind"]>("effect");
  const [seconds, setSeconds] = useState(60);
  const [quality, setQuality] = useState<"small" | "medium">(ramGb >= 16 ? "medium" : "small");
  const { task, start, stop, running } = useTask((t) => t.state === "done" && t.result.sound && onMade(t.result.sound), "sounds-make");
  const made: Sound | undefined = task?.state === "done" ? task.result.sound : undefined;

  return (
    <div className="card card-pad stack" style={{ marginBottom: 16 }}>
      <h3 className="card-title" style={{ margin: 0 }}>Make a sound with AI</h3>
      <textarea className="textarea" rows={2} placeholder={kind === "music" ? "a quiet waltz played by a string quartet at a grand ball" : kind === "ambience" ? "rain on a tin roof at night" : "a heavy wooden door slamming shut"}
        value={desc} onChange={(e) => setDesc(e.target.value)} />
      <div className="row wrap">
        <Segmented<Sound["kind"]> value={kind} onChange={setKind} options={[
          { value: "effect", label: "Effect" }, { value: "ambience", label: "Background" }, { value: "music", label: "Music" }]} />
        {kind === "music" && (
          <Segmented<"small" | "medium"> value={quality} onChange={setQuality} options={[
            { value: "small", label: "Small · fast" },
            { value: "medium", label: ramGb >= 16 ? "Medium · best" : "Medium · needs 16 GB", disabled: ramGb < 16, title: "Stable Audio 3 Medium, for Macs with 16 GB or more" }]} />
        )}
        <span className="spacer" />
        <button className="btn primary" disabled={running || !desc.trim()}
          onClick={() => start(() => api.makeSound({ description: desc, kind, ...(kind === "music" ? { seconds, quality } : {}) }))}>
          <Wand2 size={15} />Make it
        </button>
      </div>
      {kind === "music" && <Slider label="Length" value={seconds} min={10} max={120} step={5} onChange={setSeconds} format={(v) => `${v} s`} left="Loops in a scene" right="Repeats less often" />}
      <p className="small muted" style={{ margin: 0 }}>Made on this Mac with Stable Audio 3 (an effect takes about 5 s, a minute of music about 10 s). Sounds you make are under the Stability AI Community License.</p>
      {task && <TaskProgress task={task} onStop={stop} hideDone={!!made} />}
      {made && <MadeSound sound={made} />}
    </div>
  );
}

function MadeSound({ sound }: { sound: Sound }) {
  return <div className="small muted">Added <b>{sound.name}</b> ({KIND_LABEL[sound.kind].toLowerCase()}, {clock(sound.duration)}). Use it with <code>{`{${TAG[sound.kind]}: ${sound.name.toLowerCase()}}`}</code>.</div>;
}
