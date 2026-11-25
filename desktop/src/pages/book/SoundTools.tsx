import { useEffect, useMemo, useState } from "react";
import { CircleSlash, GripVertical, Timer, Music2, Pause, Play, Plus, Search, Wand2, Waves, X, Zap } from "lucide-react";
import { api, bookApi, useLoad, useTask, type ChapterPlan, type PaletteSound, type Placement, type Scope, type Sound, type StorySound } from "../../api";
import { Field, Segmented, Slider, Starting, Switch, TaskProgress, clock, useStarting, useToast } from "../../components/ui";
import { useFilePreview } from "../../hooks/useAudioPreview";
import { soundDrag } from "./drag";

const KIND = { sfx: "Effect", amb: "Background", music: "Music" } as const;
const ICON = { sfx: Zap, amb: Waves, music: Music2 };
const TAG_OF: Record<Sound["kind"], "sfx" | "amb" | "music"> = { effect: "sfx", ambience: "amb", music: "music" };
const title = (s: string) => s.replace(/\b\w/g, (c) => c.toUpperCase());

const usePreview = useFilePreview;

function ScopeChoice({ value, onChange }: { value: Scope; onChange: (s: Scope) => void }) {
  return <Segmented<Scope> value={value} onChange={onChange} options={[{ value: "book", label: "This book" }, { value: "all", label: "All books" }]} />;
}

/** A tag in the story and the sound it plays: made in this book (its scope can change), from Sounds, or not made yet. */
function SoundRow({ s, book, preview, onScope, onMake, busy }: {
  s: StorySound; book: string; preview: ReturnType<typeof usePreview>;
  onScope: (id: string, scope: Scope) => void; onMake: (s: StorySound) => void; busy: boolean;
}) {
  const Icon = ICON[s.type];
  const snd = s.sound;
  const here = !!snd && (snd.made_in === book || snd.book === book);
  const origin = !snd ? "not in your sounds yet" : here ? "made in this book" : "from your Sounds";
  return (
    <div className={`snd-row ${snd ? "can-drag" : ""}`} {...(snd ? soundDrag(s.tag, snd.name) : {})}>
      <div className="row" style={{ gap: 10 }}>
        <div className={`kind-icon ${s.type === "amb" ? "ambience" : s.type === "music" ? "music" : "effect"}`}><Icon size={15} /></div>
        <div className="grow" style={{ minWidth: 0 }}>
          <div className="title">{snd?.name || title(s.name)}</div>
          <div className="small muted">{KIND[s.type]} · {origin}{snd ? ` · ${clock(snd.duration)}` : ""}</div>
        </div>
        {snd ? (
          <button className="btn icon sm ghost" title={preview.on === snd.path ? "Stop" : "Play"} onClick={() => preview.play(snd.path)}>
            {preview.on === snd.path ? <Pause size={14} /> : <Play size={14} />}
          </button>
        ) : <button className="btn sm" disabled={busy} onClick={() => onMake(s)}><Wand2 size={13} />Make it</button>}
      </div>
      <div className="row" style={{ gap: 8, marginTop: 6 }}>
        <code className="small">{s.tag}</code>
        <span className="spacer" />
        {here && <ScopeChoice value={snd!.book ? "book" : "all"} onChange={(v) => onScope(snd!.id, v)} />}
      </div>
    </div>
  );
}

/** Make a sound or music for this book: kept in it, or shared with every book (it joins Sounds). */
function MakeSound(props: { kinds: ("effect" | "ambience" | "music")[]; book: string; scope: Scope; placeholder: string;
  onMade: (s: Sound) => void; insert: (tag: string) => void; extra?: React.ReactNode }) {
  const toast = useToast();
  const [desc, setDesc] = useState("");
  const [kind, setKind] = useState(props.kinds[0]);
  const [scope, setScope] = useState<Scope>(props.scope);
  const [made, setMade] = useState<Sound | null>(null);
  const [tag, setTag] = useState("");
  const make = useTask((t) => {
    if (t.state === "failed") return toast(t.error || "Couldn't make that sound.", true);
    if (t.state === "done" && t.result.sound) {
      const snd: Sound = t.result.sound;
      setMade(snd); props.onMade(snd); setDesc("");
      setTag(`{${TAG_OF[snd.kind]}: ${snd.name.toLowerCase()}}`);
      bookApi.palette(props.book).then((list) => { const p = list.find((x) => x.id === snd.id); if (p) setTag(p.tag); }).catch(() => {});
    }
  });
  return (
    <div className="stack" style={{ gap: 10 }}>
      <Field label={props.kinds[0] === "music" ? "Describe the music" : "Describe the sound"}>
        <input className="input" placeholder={props.placeholder} value={desc} onChange={(e) => setDesc(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && desc.trim() && !make.running && make.start(() => api.makeSound({ description: desc, kind, book: props.book, scope }))} />
      </Field>
      {props.kinds.length > 1 && (
        <Segmented<typeof kind> value={kind} onChange={setKind}
          options={props.kinds.map((k) => ({ value: k, label: k === "effect" ? "Effect" : k === "ambience" ? "Background" : "Music" }))} />
      )}
      {props.extra}
      <div className="row"><span className="small muted">Keep it in</span><span className="spacer" /><ScopeChoice value={scope} onChange={setScope} /></div>
      <button className="btn primary" disabled={!desc.trim() || make.running}
        onClick={() => make.start(() => api.makeSound({ description: desc, kind, book: props.book, scope }))}><Wand2 size={14} />Make it</button>
      <p className="small muted" style={{ margin: 0 }}>This book: only this audiobook can use it. All books: it joins your Sounds library.</p>
      {make.task && make.task.state !== "done" && <TaskProgress task={make.task} onStop={make.stop} />}
      {made && (
        <div className="made-note can-drag" {...soundDrag(tag, made.name)}>
          <GripVertical size={14} className="faint" />
          <span className="small grow">Made <b>{made.name}</b>. Drag it into the text, or: <code>{tag}</code></span>
          <button className="btn sm" onClick={() => props.insert(tag)}><Plus size={13} />Add at the cursor</button>
        </div>
      )}
    </div>
  );
}

const KIND_LABEL: Record<Sound["kind"], string> = { effect: "Effect", ambience: "Background", music: "Music" };

/** The sounds a book can use, to drag into the text where each should start (or add at the cursor). */
function SoundPalette({ list, kinds, preview, insert, stop, extra, empty, pauses }: {
  list: PaletteSound[] | null; kinds: Sound["kind"][]; preview: ReturnType<typeof usePreview>; insert: (tag: string) => void;
  stop: { tag: string; label: string; hint: string }; extra?: (s: PaletteSound) => React.ReactNode; empty: string; pauses?: boolean;
}) {
  const [q, setQ] = useState("");
  const [kind, setKind] = useState<"all" | Sound["kind"]>("all");
  const mine = (list || []).filter((s) => kinds.includes(s.kind));
  const shown = useMemo(() => {
    const words = q.toLowerCase().split(/\s+/).filter(Boolean);
    return mine.filter((s) => (kind === "all" || s.kind === kind) && words.every((w) => `${s.name} ${s.tags.join(" ")}`.toLowerCase().includes(w)));
  }, [mine, q, kind]);
  return (
    <div className="pal-wrap">
      <p className="small muted" style={{ margin: 0 }} title="Dropped inside a sentence, it starts on the word after it and the sentence stays whole.">Drag one onto the text where it should start.</p>
      {(mine.length > 6 || kinds.length > 1) && mine.length > 0 && (
        <div className="row" style={{ gap: 6 }}>
          {mine.length > 6 && <div className="search-box grow"><Search size={14} className="muted" />
            <input className="input" placeholder="Search" aria-label="Search your sounds" value={q} onChange={(e) => setQ(e.target.value)} /></div>}
          {kinds.length > 1 && (
            <Segmented<"all" | Sound["kind"]> value={kind} onChange={setKind}
              options={[{ value: "all", label: "All" }, ...kinds.map((k) => ({ value: k, label: k === "effect" ? "Effects" : k === "ambience" ? "Backgrounds" : "Music" }))]} />
          )}
        </div>
      )}
      <div className="palette">
        {list === null ? <div className="small muted">Loading your sounds…</div>
          : !mine.length ? <div className="small muted">{empty}</div>
            : !shown.length ? <div className="small muted">No sounds match.</div>
              : shown.map((s) => {
                const Icon = ICON[TAG_OF[s.kind]];
                return (
                  <div key={s.id} className="pal-row can-drag" {...soundDrag(s.tag, s.name)}>
                    <GripVertical size={14} className="grip" />
                    <div className={`kind-icon sm ${s.kind}`}><Icon size={13} /></div>
                    <div className="grow" style={{ minWidth: 0 }}>
                      <div className="title">{s.name}</div>
                      <div className="small muted">{KIND_LABEL[s.kind]} · {clock(s.duration)}{s.mine ? " · this book" : ""}</div>
                    </div>
                    {extra?.(s)}
                    <button className="btn icon sm ghost" title={preview.on === s.path ? "Stop" : "Hear it"} onClick={() => preview.play(s.path)}>
                      {preview.on === s.path ? <Pause size={13} /> : <Play size={13} />}
                    </button>
                    <button className="btn icon sm ghost" title="Add at the cursor" aria-label={`Add ${s.name} at the cursor`} onClick={() => insert(s.tag)}><Plus size={13} /></button>
                  </div>
                );
              })}
      </div>
      <div className="pal-row stop can-drag" {...soundDrag(stop.tag, stop.label.toLowerCase())}>
        <GripVertical size={14} className="grip" />
        <div className="kind-icon sm"><CircleSlash size={13} /></div>
        <div className="grow" style={{ minWidth: 0 }}><div className="title">{stop.label}</div><div className="small muted">{stop.hint}</div></div>
        <button className="btn icon sm ghost" title="Add at the cursor" aria-label={`Add “${stop.label}” at the cursor`} onClick={() => insert(stop.tag)}><Plus size={13} /></button>
      </div>
      {pauses && (
        <div className="pause-row">
          <Timer size={14} className="grip" />
          <span className="small grow">A pause</span>
          {[0.5, 1, 2, 3].map((n) => (
            <button key={n} className="chip-btn can-drag" {...soundDrag(`{pause: ${n}}`, `${n}-second pause`)}
              onClick={() => insert(`{pause: ${n}}`)} title={`Drag into the text, or click to add at the cursor: ${n} s of silence`}>{n === 0.5 ? "½" : n} s</button>
          ))}
        </div>
      )}
    </div>
  );
}

/** The director's sounds to keep or drop. */
export type ReviewState = {
  placements: Placement[]; planned_on: string; kept: Set<number>; title?: string; source?: "director";
  /** the director's chapter intros and outros: made on Add when its "Make chapter intros and outros" is on */
  chapterPlan?: ChapterPlan | null;
  /** placements that play their close library match instead of a new sound */
  useMatch?: Set<number>;
  /** set when the director stopped partway (e.g. out of credit or daily limit) */
  partial?: boolean; scenesDone?: number; scenesTotal?: number; stopReason?: string | null;
} | null;

/** The kept placements as Add sends them: one that takes its library match plays that sound by its own tag. */
export function keptPlacements(review: NonNullable<ReviewState>): Placement[] {
  return review.placements.filter((p) => review.kept.has(p.id)).map((p) => (p.match && review.useMatch?.has(p.id)
    ? { ...p, name: p.match.tag_name, missing: false, sound_id: p.match.id, opts: p.kind === "sfx" ? p.opts : [] } : p));
}

/** The review list: tick what to keep, then Add. `addLabel` can say what Add does besides ("Add · makes 2 sounds"). */
export function ReviewBox({ review, setReview, onAdd, addLabel }: {
  review: NonNullable<ReviewState>; setReview: (r: ReviewState) => void; onAdd: () => unknown; addLabel?: string;
}) {
  const adding = useStarting();
  const preview = usePreview();
  const [confirmDiscard, setConfirmDiscard] = useState(false);
  useEffect(() => { setConfirmDiscard(false); }, [review.planned_on, review.placements.length]);
  useEffect(() => {
    if (!confirmDiscard) return;
    const t = window.setTimeout(() => setConfirmDiscard(false), 4000);
    return () => window.clearTimeout(t);
  }, [confirmDiscard]);
  return (
    <div className="card-sub stack review-box" style={{ gap: 8 }}>
      <div className="row wrap" style={{ gap: 6 }}>
        <b className="small">{review.title || "The director's sounds"}</b>
        <span className="small muted">{review.kept.size} of {review.placements.length}</span>
        {review.placements.length > 5 && (
          <>
            <span className="spacer" />
            <button type="button" className="link-btn small"
              onClick={() => setReview({ ...review, kept: new Set(review.placements.map((p) => p.id)) })}>All</button>
            <button type="button" className="link-btn small"
              onClick={() => setReview({ ...review, kept: new Set() })}>None</button>
          </>
        )}
      </div>
      {review.partial && (
        <div className="small error-text" style={{ lineHeight: 1.4 }}>
          Stopped partway ({review.scenesDone ?? 0} of {review.scenesTotal ?? "?"} scenes)
          {review.stopReason ? `: ${review.stopReason}` : "."} Direct again to resume the remaining scenes.
        </div>
      )}
      <div className="row wrap review-actions">
        <button className="btn sm primary" onClick={() => adding.run(onAdd)} disabled={!review.kept.size || adding.starting}>
          {adding.starting ? <Starting label="Adding…" size={13} /> : addLabel || "Add"}</button>
        <button className={`btn sm ${confirmDiscard ? "danger" : "ghost"}`} onClick={() => (confirmDiscard ? setReview(null) : setConfirmDiscard(true))}>
          <X size={13} />{confirmDiscard ? "Discard picks?" : "Discard"}
        </button>
      </div>
      <div className="review">
        {review.placements.map((p) => (
          <label key={p.id}>
            <input type="checkbox" checked={review.kept.has(p.id)} style={{ accentColor: "var(--accent)" }}
              onChange={(e) => { const kept = new Set(review.kept); if (e.target.checked) kept.add(p.id); else kept.delete(p.id); setReview({ ...review, kept }); }} />
            <span className="grow small">{p.label}{p.match && (
              <span className="review-match">
                {review.useMatch?.has(p.id) ? "Plays" : "Close match:"} <b>{p.match.name}</b>
                <button className="btn icon sm ghost" title={`Hear ${p.match.name}`} onClick={(e) => { e.preventDefault(); preview.play(p.match!.path); }}><Play size={11} /></button>
                <button className="btn sm ghost" onClick={(e) => {
                  e.preventDefault();
                  const use = new Set(review.useMatch || []);
                  if (use.has(p.id)) use.delete(p.id); else use.add(p.id);
                  setReview({ ...review, useMatch: use });
                }}>{review.useMatch?.has(p.id) ? "Make it instead" : "Use it instead"}</button>
              </span>)}</span>
            {p.sound_path && <button className="btn icon sm ghost" title="Hear it" onClick={(e) => { e.preventDefault(); preview.play(p.sound_path!); }}><Play size={12} /></button>}
          </label>
        ))}
      </div>
    </div>
  );
}

/** Volumes are dB under the narrator while they speak (narrator/sfx.py levels_of); OFF and under is silent. */
export const OFF = -41;
export const dbText = (v: number) => (v <= OFF ? "Off" : v === 0 ? "Narrator level" : v > 0 ? `${v} dB above the narrator` : `${-v} dB under the narrator`);

export function SoundTool(props: {
  book: string; sounds: StorySound[] | null; reloadSounds: () => void; insert: (tag: string) => void;
  sound: boolean; setSound: (v: boolean) => void; newSounds: Scope; setNewSounds: (v: Scope) => void;
  server: Record<string, any>; saveServer: (k: string, v: any) => void;
  busy: boolean; onMissing: () => void;
  levels: { fx: number; amb: number; music: number }; setLevel: (k: "fx_level" | "amb_level" | "music_level", v: number) => void;
}) {
  const toast = useToast();
  const preview = usePreview();
  const palette = useLoad(() => bookApi.palette(props.book), [props.book, props.sounds]);
  const [view, setView] = useState<"library" | "story" | "make" | "settings">("library");
  const list = (props.sounds || []).filter((s) => s.type !== "music");
  const scope = async (id: string, v: Scope) => {
    try { await api.soundScope(id, v === "book" ? props.book : null); props.reloadSounds(); palette.reload(); } catch (e) { toast((e as Error).message, true); }
  };
  const makeOne = useTask((t) => {
    if (t.state === "failed") toast(t.error || "Couldn't make that sound.", true);
    else if (t.state === "done") { toast(t.status); props.reloadSounds(); palette.reload(); }
  });
  const makeMissing = (s: StorySound) => makeOne.start(() => api.makeSound({ description: s.name, kind: s.type === "amb" ? "ambience" : "effect",
    book: props.book, scope: props.newSounds }));
  const missing = list.filter((s) => !s.sound).length;
  return (
    <div className="tool-fit">
      <Segmented<typeof view> value={view} onChange={setView} options={[
        { value: "library", label: "Library" }, { value: "story", label: list.length ? `In story · ${list.length}` : "In story" },
        { value: "make", label: "Make" }, { value: "settings", label: "Settings" }]} />

      {view === "library" && (
        <SoundPalette list={palette.data} kinds={["effect", "ambience"]} preview={preview} insert={props.insert}
          stop={{ tag: "{amb: off}", label: "Stop the background", hint: "Backgrounds fade out from here" }}
          empty="No sound effects or backgrounds yet. Make one, or import some in Sounds." pauses />
      )}
      {view === "story" && (
        <div className="fill stack">
          {makeOne.task?.state === "running" && <TaskProgress task={makeOne.task} onStop={makeOne.stop} />}
          {missing > 0 && <button className="btn sm" disabled={props.busy} onClick={props.onMissing}><Wand2 size={13} />Make the {missing} missing sound{missing > 1 ? "s" : ""}</button>}
          {props.sounds === null ? <div className="small muted">Looking at the story's tags…</div>
            : list.length ? list.map((s) => <SoundRow key={s.tag} s={s} book={props.book} preview={preview} onScope={scope} onMake={makeMissing} busy={makeOne.running} />)
              : <div className="small muted">No sound tags yet. Direct some (Director tab), drag one in from Library, or type a tag like <code>{"{sfx: door creak}"}</code>.</div>}
        </div>
      )}
      {view === "make" && (
        <div className="fill">
          <MakeSound kinds={["effect", "ambience"]} book={props.book} scope={props.newSounds} placeholder="e.g. floorboards creaking upstairs"
            onMade={() => { props.reloadSounds(); palette.reload(); }} insert={props.insert} />
        </div>
      )}
      {view === "settings" && (
        <div className="fill stack">
          <Switch label="Play sound tags" hint="Off: the {sfx}, {amb} and {music} tags are skipped."
            checked={props.sound} onChange={props.setSound} />
          <Switch label="Make missing sounds automatically" hint="Tagged sounds you don't have are made with AI first."
            checked={props.server.sfx_auto_make ?? true} onChange={(v) => props.saveServer("sfx_auto_make", v)} />
          <Slider label="Effects volume" value={Math.min(0, props.levels.fx)} min={OFF} max={0} step={1} format={dbText}
            onChange={(v) => props.setLevel("fx_level", v)} left="Off" right="Narrator level" />
          <Slider label="Music volume" value={props.levels.music} min={OFF} max={0} step={1} format={dbText}
            onChange={(v) => props.setLevel("music_level", v)} left="Off" right="Narrator level" />
          <Slider label="Background volume" value={props.levels.amb} min={OFF} max={0} step={1} format={dbText}
            onChange={(v) => props.setLevel("amb_level", v)} left="Off" right="Narrator level" />
          <div className="row"><span className="small" style={{ flex: 1 }}>New sounds go to</span><ScopeChoice value={props.newSounds} onChange={props.setNewSounds} /></div>
          <details className="advanced">
            <summary>Tag cheat sheet</summary>
            <div className="cheats">
              <span><code>{"{sfx: door creak}"}</code> a sound right there · <code>{"{sfx: knock @knocked}"}</code> on that word</span>
              <span><code>{"{sfx: thunder, loud}"}</code> quiet / loud · <code>{"{sfx: scream, far}"}</code> far / muffled · <code>left</code> / <code>right</code></span>
              <span><code>{"{amb: rain}"}</code> a looping background · <code>{"{amb: fire, add}"}</code> layer · <code>{"{amb: off}"}</code></span>
              <span><code>{"{music: quiet waltz}"}</code> scene music · <code>{"{music: off}"}</code></span>
              <span><code>{"{room: cave}"}</code> echo (room, bathroom, hall, cave, cathedral) · <code>{"{room: off}"}</code></span>
              <span><code>{"{pause: 1.5}"}</code> that many seconds of silence (up to 10), even with sound off</span>
            </div>
          </details>
        </div>
      )}
    </div>
  );
}

export function MusicTool(props: {
  book: string; story: string; title: string; sounds: StorySound[] | null; reloadSounds: () => void; insert: (tag: string) => void;
  newSounds: Scope;
}) {
  const toast = useToast();
  const preview = usePreview();
  const palette = useLoad(() => bookApi.palette(props.book), [props.book, props.sounds]);
  const list = (props.sounds || []).filter((s) => s.type === "music");
  const scope = async (id: string, v: Scope) => {
    try { await api.soundScope(id, v === "book" ? props.book : null); props.reloadSounds(); palette.reload(); } catch (e) { toast((e as Error).message, true); }
  };
  const [view, setView] = useState<"library" | "story" | "make">("library");
  return (
    <div className="tool-fit">
      <Segmented<typeof view> value={view} onChange={setView} options={[
        { value: "library", label: "Library" }, { value: "story", label: list.length ? `In story · ${list.length}` : "In story" },
        { value: "make", label: "Make" }]} />

      {view === "library" && (
        <SoundPalette list={palette.data} kinds={["music"]} preview={preview} insert={props.insert}
          stop={{ tag: "{music: off}", label: "Stop the music", hint: "Music fades out from here" }}
          empty="No music yet. Make some, or import it in Sounds." />
      )}
      {view === "story" && (
        <div className="fill stack">
          {props.sounds === null ? <div className="small muted">Looking at the story's tags…</div>
            : list.length ? list.map((s) => <SoundRow key={s.tag} s={s} book={props.book} preview={preview} onScope={scope} onMake={() => {}} busy />)
              : <div className="small muted">No music tags yet. Drag some in from Library, or type one like <code>{"{music: quiet waltz}"}</code>.</div>}
        </div>
      )}
      {view === "make" && (
        <div className="fill">
          <MakeSound kinds={["music"]} book={props.book} scope={props.newSounds} placeholder="e.g. quiet ballroom waltz, far away"
            onMade={() => { props.reloadSounds(); palette.reload(); }} insert={props.insert} />
        </div>
      )}

    </div>
  );
}
