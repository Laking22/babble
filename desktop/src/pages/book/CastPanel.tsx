import { useState } from "react";
import { ChevronDown, ChevronRight, Plus, RotateCcw, Trash2, Users, Volume2 } from "lucide-react";
import { bookApi, fileUrl, useTask, type CastRow, type Engine, type Pronunciation, type Voice } from "../../api";
import { VoicePicker } from "../../components/VoicePicker";
import { Field, Orb, Segmented, Slider, Starting, Switch, useToast } from "../../components/ui";

/** Who reads what: one row per character, the narrator first. Each can have their own speed and stability. */
export function CastPanel(props: {
  cast: CastRow[]; setCast: (c: CastRow[]) => void; voices: Voice[]; engines: Engine[];
  models: string; setModels: (m: string) => void; modelChoices: { label: string; value: string }[];
  quotesOnly: boolean; setQuotesOnly: (v: boolean) => void; story: string; stability: number; language: string;
  detect: "ai" | "quick"; setDetect: (v: "ai" | "quick") => void; onDetect: () => void; busy: boolean; detecting?: boolean; detectOn?: string;
  sampleText?: string; sampleLabel?: string; pronunciations?: Pronunciation[];
  onCleared?: (keep: string[]) => void;  // Clear cast: the story's tags for them go too
}) {
  const { cast, setCast, voices, engines, onCleared } = props;
  const [open, setOpen] = useState<string | null>(null);
  const [newName, setNewName] = useState("");
  const toast = useToast();
  const hear = useTask((t) => t.state === "failed" && toast(t.error || "Couldn't make the sample.", true));
  const [hearing, setHearing] = useState<string | null>(null);
  const label = (value: string) => voices.find((v) => v.value === value)?.name || value || "No voice yet";

  const [clearing, setClearing] = useState(false);
  const others = cast.filter((r) => r.character !== "Narrator").length;
  const clear = () => {
    if (!clearing) { setClearing(true); setTimeout(() => setClearing(false), 4000); return; }
    setCast(cast.filter((r) => r.character === "Narrator"));
    setClearing(false); setOpen(null);
    onCleared?.(["Narrator"]);
    toast(`Removed ${others} character${others === 1 ? "" : "s"} and their tags in the story. The Narrator stays.`);
  };
  const update = (i: number, patch: Partial<CastRow>) => setCast(cast.map((r, k) => (k === i ? { ...r, ...patch } : r)));
  const add = () => {
    const name = newName.trim();
    if (!name) return;
    if (cast.some((r) => r.character.toLowerCase() === name.toLowerCase())) return toast(`${name} is already in the cast.`, true);
    const used = new Set(cast.map((r) => r.voice));
    const voice = voices.find((v) => v.available && !used.has(v.value))?.value || cast[0]?.voice || "";
    setCast([...cast, { character: name, voice, style: "", speed: 1 }]);
    setNewName("");
    setOpen(name);
  };
  const listen = (r: CastRow) => {
    setHearing(r.character);
    hear.start(() => bookApi.hear({
      story: props.story, character: r.character, voice: r.voice, style: r.style, speed: r.speed,
      stability: r.stability ?? props.stability, language: props.language, cast, models: props.models,
      sample_text: props.sampleText, pronunciations: props.pronunciations,
    }));
  };

  return (
    <div className="tool-fit">
      <div className="row">
        <span className="section-label" style={{ margin: 0 }}>Characters in this book</span>
        <span className="spacer" />
        <Segmented<"ai" | "quick"> value={props.detect} onChange={props.setDetect}
          options={[{ value: "ai", label: "AI" }, { value: "quick", label: "Quick" }]} />
      </div>
      <div className="row" style={{ gap: 6 }}>
        <button className="btn sm" disabled={props.busy} onClick={props.onDetect}>
          {props.detecting ? <Starting /> : <><Users size={14} />Detect speakers{props.detectOn ? ` · on ${props.detectOn}` : ""}</>}</button>
        <span className="spacer" />
        {others > 0 && (
          <button className={`btn sm ${clearing ? "danger" : "ghost"}`} onClick={clear}
            title="Remove every character except the Narrator, for a new story in this audiobook">
            <Trash2 size={13} />{clearing ? `Remove all ${others}? Click again` : "Clear cast"}</button>
        )}
      </div>
      {props.sampleText && (
        <div className="small muted" style={{ margin: "-2px 0 2px" }}>
          Voice previews read {props.sampleLabel || "the highlighted paragraph"}.
        </div>
      )}
      <div className="fill stack" style={{ gap: 6 }}>
      {cast.map((r, i) => {
        const v = voices.find((x) => x.value === r.voice);
        const isOpen = open === r.character;
        const sample = hearing === r.character && hear.task;
        const own = r.stability != null;
        return (
          <div key={r.character} className="cast-row">
            <div className="cast-head" onClick={() => setOpen(isOpen ? null : r.character)}>
              {isOpen ? <ChevronDown size={14} className="muted" /> : <ChevronRight size={14} className="muted" />}
              <Orb value={r.voice} size={26} />
              <span className="voice-meta" style={{ flex: 1 }}>
                <strong>{r.character}</strong>
                <span>{label(r.voice)}{v ? ` · ${engines.find((e) => e.id === v.engine)?.short || v.engine}` : ""}{r.style ? ` · ${r.style}` : ""}{own ? ` · ${r.stability}% stable` : ""}</span>
              </span>
              <button className="btn icon sm ghost"
                title={props.sampleText ? `Hear ${r.character} reading ${props.sampleLabel || "highlighted paragraph"}` : `Hear ${r.character}`}
                disabled={hear.running}
                onClick={(e) => { e.stopPropagation(); listen(r); }}><Volume2 size={14} /></button>
              {r.character !== "Narrator" && (
                <button className="btn icon sm ghost" title={`Remove ${r.character}`} aria-label={`Remove ${r.character}`}
                  onClick={(e) => { e.stopPropagation(); setCast(cast.filter((_, k) => k !== i)); if (isOpen) setOpen(null); }}><Trash2 size={13} /></button>
              )}
            </div>
            {sample && (sample.state === "running"
              ? <div className="small muted" style={{ padding: "0 12px 10px" }}>Making a sample…</div>
              : sample.result.audio && <audio src={fileUrl(sample.result.audio)} controls autoPlay style={{ padding: "0 10px 10px" }} />)}
            {isOpen && (
              <div className="cast-body">
                <Field label="Voice">
                  <VoicePicker voices={voices} engines={engines} value={r.voice} onChange={(voice) => update(i, { voice })} allowCloud
                    sampleText={props.sampleText} sampleLabel={props.sampleLabel} language={props.language} />
                </Field>
                <Field label="Default style">
                  <input className="input" placeholder={i === 0 ? "Warm, calm storytelling." : "e.g. gruff, kind old man"}
                    value={r.style} onChange={(e) => update(i, { style: e.target.value })} />
                </Field>
                <Slider label="Speed" value={r.speed} min={0.7} max={1.3} step={0.05} onChange={(speed) => update(i, { speed })} format={(x) => `${x.toFixed(2)}×`} />
                <Slider label={own ? "Stability" : "Stability (the book's)"} value={r.stability ?? props.stability} min={0} max={100} step={5}
                  onChange={(stability) => update(i, { stability })} format={(x) => `${x}%`} left="More expressive" right="More stable" />
                {own && (
                  <button className="link small muted row" style={{ gap: 4 }} onClick={() => update(i, { stability: null })}>
                    <RotateCcw size={12} />Use the book's stability ({props.stability}%)
                  </button>
                )}
                {v && ["kokoro", "vibevoice"].includes(v.engine) && (
                  <p className="small muted" style={{ margin: 0 }}>{engines.find((e) => e.id === v.engine)?.short || v.engine} voices read the same way every time, so stability doesn't change them.</p>
                )}
                {r.character !== "Narrator" && (
                  <button className="btn sm danger" onClick={() => setCast(cast.filter((_, k) => k !== i))}><Trash2 size={13} />Remove {r.character}</button>
                )}
              </div>
            )}
          </div>
        );
      })}
      </div>
      <div className="row">
        <input className="input" placeholder="Add a character" value={newName} onChange={(e) => setNewName(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && add()} />
        <button className="btn" onClick={add} disabled={!newName.trim()}><Plus size={15} />Add</button>
      </div>
      <p className="small muted" style={{ margin: 0 }}>Tag lines with <code>[Name]</code> or <code>[Name: angry]</code>.</p>
      <Field label="Voices for the whole book">
        <select className="select" value={props.models} onChange={(e) => props.setModels(e.target.value)}>
          {props.modelChoices.map((m) => <option key={m.value} value={m.value}>{m.label}</option>)}
        </select>
      </Field>
      <Switch label="Narrator reads the words around dialogue" hint="“I know,” she said: the quote in her voice, “she said” in the narrator's."
        checked={props.quotesOnly} onChange={props.setQuotesOnly} />
    </div>
  );
}
