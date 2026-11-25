import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { ChevronDown, Cloud, Loader2, Lock, Pause, Play, Search } from "lucide-react";
import { isCopy, type Engine, type Voice } from "../api";
import { useVoicePreview } from "../hooks/useAudioPreview";
import { Orb } from "./ui";

const ENGINE_ORDER = ["qwen", "orpheus", "kokoro", "vibevoice", "higgs"];

/** Pick a voice: searchable, grouped by engine, with inline voice preview play buttons. */
/** allowCloud: voices only a cloud GPU can make (cloud) can be picked too, marked Cloud only (a book's cast). */
export function VoicePicker(props: {
  voices: Voice[]; engines: Engine[]; value: string; onChange: (v: string) => void;
  allowCloud?: boolean; sampleText?: string; sampleLabel?: string; language?: string;
}) {
  const { voices, engines, value, onChange, allowCloud, sampleText, sampleLabel, language } = props;
  const ok = (v: Voice) => v.available || (!!allowCloud && !!v.cloud);
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const anchor = useRef<HTMLButtonElement>(null);
  const [pos, setPos] = useState({ top: 0, left: 0 });
  const current = voices.find((v) => v.value === value);
  const engineName = (id: string) => engines.find((e) => e.id === id)?.short || id;
  const preview = useVoicePreview({ active: open, sampleText, language });

  useEffect(() => {
    if (!open || !anchor.current) return;
    const place = () => {
      if (!anchor.current) return;
      const r = anchor.current.getBoundingClientRect();
      const popW = Math.min(360, window.innerWidth - 24);
      const popH = Math.min(460, window.innerHeight - 32);
      setPos({
        top: Math.max(12, Math.min(r.bottom + 6, window.innerHeight - popH - 12)),
        left: Math.max(12, Math.min(r.left, window.innerWidth - popW - 12)),
      });
    };
    place();
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    return () => {
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", place, true);
    };
  }, [open]);

  const groups = useMemo(() => {
    const words = q.toLowerCase().split(/\s+/).filter(Boolean);
    const match = (v: Voice) => words.every((w) => `${v.name} ${v.label} ${v.description} ${v.gender} ${v.tags.join(" ")} ${engineName(v.engine)}`.toLowerCase().includes(w));
    const hidden = new Set(engines.filter((e) => e.group === "high" && e.downloaded === false && !e.locked).map((e) => e.id));
    const listed = voices.filter((v) => (!hidden.has(v.engine) && !isCopy(v)) || v.value === value);
    const mine = listed.filter((v) => v.tags.includes("saved") && v.engine === "qwen" && match(v));
    const byEngine = ENGINE_ORDER.map((id) => ({
      title: engineName(id),
      items: listed.filter((v) => v.engine === id && !(id === "qwen" && v.tags.includes("saved")) && match(v)),
    }));
    return [{ title: "Your voices", items: mine }, ...byEngine].filter((g) => g.items.length);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [voices, engines, q]);

  return (
    <>
      <button ref={anchor} type="button" className="voice-btn" onClick={() => setOpen(true)}>
        <Orb value={value} />
        <span className="voice-meta">
          <strong>{current?.name || "Choose a voice"}</strong>
          <span>{current ? [engineName(current.engine), current.gender !== "unspecified" && current.gender, ...current.tags].filter(Boolean).join(" · ") : ""}</span>
        </span>
        <span className="spacer" />
        <ChevronDown size={16} className="muted" />
      </button>
      {/* in <body>, not here: the picker sits in a <label> (Field), which would pass a click on the
          backdrop on to the voice button and open the list again */}
      {open && createPortal(
        <>
          <div className="popover-backdrop" onClick={() => setOpen(false)} />
          <div className="popover" style={{ top: pos.top, left: pos.left, position: "fixed" }} role="listbox">
            <div className="popover-search">
              <div className="row" style={{ position: "relative" }}>
                <Search size={15} style={{ position: "absolute", left: 10 }} className="muted" />
                <input autoFocus className="input" style={{ paddingLeft: 32 }} placeholder="Search voices" value={q}
                  onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Escape" && setOpen(false)} />
              </div>
              {sampleText && (
                <div className="small muted" style={{ marginTop: 6, paddingLeft: 2 }}>
                  Play previews {sampleLabel ? `using ${sampleLabel}` : "using highlighted paragraph"}
                </div>
              )}
            </div>
            <div className="popover-list">
              {groups.map((g) => (
                <div key={g.title}>
                  <div className="voice-group">{g.title}</div>
                  {g.items.map((v) => {
                    const busy = preview.making.has(v.value);
                    const isPlaying = preview.on === v.value;
                    const playTip = !v.available ? v.why : isPlaying ? `Pause ${v.name}`
                      : sampleText ? `Preview ${v.name} reading ${sampleLabel || "highlighted paragraph"}` : `Preview ${v.name}`;
                    return (
                      <div
                        key={v.value}
                        role="option"
                        aria-selected={v.value === value}
                        tabIndex={ok(v) ? 0 : -1}
                        title={v.available ? v.description : ok(v) ? "This Mac can't make it, but Create in the cloud can (Modal or a pay-as-you-go GPU)" : v.why}
                        className={`voice-opt ${v.value === value ? "sel" : ""} ${ok(v) ? "" : "locked"}`}
                        onClick={() => { if (ok(v)) { onChange(v.value); setOpen(false); } }}
                        onKeyDown={(e) => { if ((e.key === "Enter" || e.key === " ") && ok(v)) { e.preventDefault(); onChange(v.value); setOpen(false); } }}
                      >
                        <button
                          type="button"
                          className="btn sm ghost icon voice-play"
                          disabled={!v.available || busy}
                          aria-label={playTip}
                          title={playTip}
                          onClick={(e) => { e.stopPropagation(); preview.toggle(v.value); }}
                        >
                          {busy ? <Loader2 size={13} className="spin" /> : isPlaying ? <Pause size={13} /> : <Play size={13} />}
                        </button>
                        <Orb value={v.value} />
                        <span className="voice-meta">
                          <strong>{v.name}</strong>
                          <span>{v.available ? (v.description || (v.gender !== "unspecified" ? v.gender : "")) : ok(v) ? "Cloud only on this Mac · made by Create in the cloud" : v.why}</span>
                        </span>
                        <span className="spacer" />
                        {!v.available && (ok(v) ? <span className="chip"><Cloud size={11} />Cloud only</span> : <Lock size={13} />)}
                        {v.tags.includes("fine-tuned") && <span className="chip accent">fine-tuned</span>}
                      </div>
                    );
                  })}
                </div>
              ))}
              {!groups.length && <div className="empty small">No voices match “{q}”.</div>}
            </div>
          </div>
        </>,
        document.body,
      )}
    </>
  );
}
