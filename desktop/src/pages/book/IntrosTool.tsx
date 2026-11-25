import { useEffect, useMemo, useRef, useState } from "react";
import { Check, Pause, Play, Sparkles, Undo2 } from "lucide-react";
import { bookApi, fileUrl, useTask, type ChapterMusic, type ChapterMusicRow, type IntroChoice, type Scope } from "../../api";
import { Segmented, Slider, TaskProgress, useToast } from "../../components/ui";
import { OFF, dbText } from "./SoundTools";

type Slot = "intro" | "outro";
type Pick = { intro: string; outro: string };
const OUTRO_WHOLE = 16; // (sfx.OUTRO_WHOLE) a longer outro sound plays its ending

/** For each chapter title, its saved row: the same title (the n-th of a repeated one), else the row in the same
 *  place when that row's title isn't any chapter's (chapter_music.match). */
function matchRows(names: string[], rows: ChapterMusicRow[]): (ChapterMusicRow | null)[] {
  const byTitle = new Map<string, ChapterMusicRow[]>();
  rows.forEach((r) => byTitle.set(r.title, [...(byTitle.get(r.title) || []), r]));
  const seen = new Map<string, number>();
  const used = new Set(names);
  return names.map((name, k) => {
    const n = seen.get(name) || 0;
    seen.set(name, n + 1);
    const same = byTitle.get(name) || [];
    if (n < same.length) return same[n];
    return k < rows.length && !used.has(rows[k].title) ? rows[k] : null;
  });
}

/** Intros: every chapter's intro and outro. Pick a library sound or design one, hear it, confirm. The director's
 *  "Make chapter intros and outros" fills them all (and replaces these). */
export function IntrosTool(props: {
  book: string; story: string; music: ChapterMusic | null | undefined; legacyIntro: string;
  setMusic: (m: ChapterMusic) => void; level: number; setLevel: (v: number) => void;
  choices: IntroChoice[]; reloadChoices: () => void; newSounds: Scope; directorOn: boolean;
}) {
  const toast = useToast();
  const [names, setNames] = useState<string[] | null>(null);
  const [draft, setDraft] = useState<Record<number, Pick>>({});
  const [designing, setDesigning] = useState<{ k: number; slot: Slot } | null>(null);
  const [words, setWords] = useState("");
  const [outroKind, setOutroKind] = useState<"sting" | "outro">("outro");
  const [all, setAll] = useState<Pick>({ intro: "", outro: "" });
  const audio = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState<string | null>(null);

  useEffect(() => {
    if (!props.story.trim()) { setNames([]); return; }
    const t = setTimeout(() => bookApi.chapterTitles(props.story).then((r) => setNames(r.chapters), () => setNames([])), 500);
    return () => clearTimeout(t);
  }, [props.story]);

  const saved: Pick[] = useMemo(() => {
    const list = names || [];
    const rows = props.music?.chapters || [];
    if (!rows.length && props.legacyIntro) return list.map(() => ({ intro: props.legacyIntro, outro: props.legacyIntro }));
    return matchRows(list, rows).map((r) => ({ intro: r?.intro || "", outro: r?.outro || "" }));
  }, [names, props.music, props.legacyIntro]);
  const byDirector = useMemo(() => matchRows(names || [], props.music?.chapters || []).map((r) => r?.by === "director"), [names, props.music]);
  const now = (k: number): Pick => draft[k] || saved[k] || { intro: "", outro: "" };
  const changed = (k: number) => !!draft[k] && (draft[k].intro !== saved[k]?.intro || draft[k].outro !== saved[k]?.outro);
  const pending = (names || []).map((_, k) => k).filter(changed);

  const choose = (k: number, slot: Slot, id: string) => setDraft((d) => ({ ...d, [k]: { ...now(k), [slot]: id } }));
  const confirm = (ks: number[]) => {
    const list = names || [];
    const old = matchRows(list, props.music?.chapters || []);
    props.setMusic({ chapters: list.map((title, k) => {
      const p = ks.includes(k) ? now(k) : (saved[k] || { intro: "", outro: "" });
      return { title, ...p, by: ks.includes(k) ? "you" : (old[k]?.by || "you") };
    }) });
    setDraft((d) => { const n = { ...d }; ks.forEach((k) => delete n[k]); return n; });
  };
  const undo = (k: number) => setDraft((d) => { const n = { ...d }; delete n[k]; return n; });

  const play = (id: string, slot: Slot) => {
    const c = props.choices.find((x) => x.id === id);
    if (!c) return;
    const key = `${slot}:${id}`;
    if (!audio.current) { audio.current = new Audio(); audio.current.onended = () => setPlaying(null); }
    const el = audio.current;
    if (playing === key) { el.pause(); setPlaying(null); return; }
    el.src = fileUrl(c.path);
    // an outro from a long sound is its ending: hear that part
    if (slot === "outro" && c.duration > OUTRO_WHOLE) el.onloadedmetadata = () => { el.currentTime = Math.max(0, c.duration - 12); };
    else el.onloadedmetadata = null;
    el.play().then(() => setPlaying(key), () => toast("Couldn't play that sound.", true));
  };

  const design = useTask((t) => {
    if (t.state === "failed") return toast(t.error || "That didn't work.", true);
    const s = t.result.sound;
    if (s && designing) { props.reloadChoices(); choose(designing.k, designing.slot, s.id); setDesigning(null); setWords(""); }
  });
  const startDesign = () => {
    if (!designing || !words.trim()) return;
    const kind = designing.slot === "intro" ? "intro" : outroKind;
    design.start(() => bookApi.designMusic(words, kind, props.book, props.newSounds));
  };

  const options = (
    <>
      <option value="">None</option>
      {props.choices.map((c) => <option key={c.id} value={c.id}>{c.label}</option>)}
    </>
  );
  const pickRow = (k: number, slot: Slot) => {
    const id = now(k)[slot];
    const missing = id && !props.choices.some((c) => c.id === id);
    return (
      <div className="row" style={{ gap: 6 }}>
        <span className="small muted" style={{ width: 40 }}>{slot === "intro" ? "Intro" : "Outro"}</span>
        <select className="select grow" value={id} onChange={(e) => choose(k, slot, e.target.value)} aria-label={`${slot} for chapter ${k + 1}`}>
          {options}
          {missing && <option value={id}>(not in the library any more)</option>}
        </select>
        <button className="btn icon sm ghost" disabled={!id || !!missing} onClick={() => play(id, slot)} aria-label={`Hear the ${slot}`} title="Hear it">
          {playing === `${slot}:${id}` ? <Pause size={14} /> : <Play size={14} />}</button>
        <button className="btn icon sm ghost" disabled={design.running} onClick={() => { setDesigning({ k, slot }); setWords(""); }}
          aria-label={`Design an ${slot}`} title="Design one"><Sparkles size={14} /></button>
      </div>
    );
  };

  if (names === null) return <div className="small muted">Reading the chapters…</div>;
  if (!names.length) return <div className="small muted">Paste or import a story first: its chapters are listed here.</div>;
  return (
    <div className="tool-fit">
      {props.directorOn && <p className="small muted" style={{ margin: 0 }}>The director's “Make chapter intros and outros” is on: directing again replaces these.</p>}
      <Slider label="Volume" value={props.level} min={OFF} max={0} step={1} format={dbText}
        onChange={props.setLevel} left="Off" right="Narrator level" />
      <details className="card-sub">
        <summary className="small" style={{ cursor: "pointer" }}>Same for every chapter</summary>
        <div className="stack" style={{ gap: 6, marginTop: 8 }}>
          <select className="select" value={all.intro} onChange={(e) => setAll({ ...all, intro: e.target.value })} aria-label="Intro for every chapter">{options}</select>
          <select className="select" value={all.outro} onChange={(e) => setAll({ ...all, outro: e.target.value })} aria-label="Outro for every chapter">{options}</select>
          <button className="btn sm" onClick={() => setDraft(Object.fromEntries(names.map((_, k) => [k, { ...all }])))}>Use for all chapters</button>
        </div>
      </details>
      {pending.length > 1 && (
        <div className="row">
          <span className="small grow">{pending.length} chapters changed</span>
          <button className="btn sm primary" onClick={() => confirm(pending)}><Check size={13} />Confirm all</button>
          <button className="btn sm ghost" onClick={() => setDraft({})}>Undo all</button>
        </div>
      )}
      <div className="fill stack" style={{ gap: 8 }}>
        {names.map((title, k) => (
          <div key={k} className="card-sub stack" style={{ gap: 6 }}>
            <div className="row">
              <b className="small grow" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{k + 1}. {title}</b>
              {byDirector[k] && !changed(k) && <span className="chip">Director</span>}
            </div>
            {pickRow(k, "intro")}
            {pickRow(k, "outro")}
            {designing?.k === k && (
              <div className="stack" style={{ gap: 6 }}>
                {designing.slot === "outro" && <Segmented<"sting" | "outro"> value={outroKind} onChange={setOutroKind}
                  options={[{ value: "outro", label: "Soft ending" }, { value: "sting", label: "Sting" }]} />}
                <div className="row" style={{ gap: 6 }}>
                  <input className="input grow" autoFocus placeholder={designing.slot === "intro" ? "e.g. eerie harp and low strings, slow" : "e.g. short brass stinger, ominous"}
                    value={words} onChange={(e) => setWords(e.target.value)} onKeyDown={(e) => e.key === "Enter" && startDesign()} />
                  <button className="btn sm primary" disabled={!words.trim() || design.running} onClick={startDesign}>Make</button>
                  <button className="btn sm ghost" onClick={() => setDesigning(null)}>Cancel</button>
                </div>
                {design.task && design.running && <TaskProgress task={design.task} onStop={design.stop} />}
              </div>
            )}
            {changed(k) && (
              <div className="row" style={{ gap: 6 }}>
                <span className="spacer" />
                <button className="btn sm ghost" onClick={() => undo(k)}><Undo2 size={13} />Undo</button>
                <button className="btn sm primary" onClick={() => confirm([k])}><Check size={13} />Confirm</button>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
