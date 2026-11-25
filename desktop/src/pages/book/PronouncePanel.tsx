import { useEffect, useMemo, useRef, useState } from "react";
import { Pause, Plus, Search, Volume2, X } from "lucide-react";
import { api, bookApi, fileUrl, useTask, type CastRow, type Pronunciation, type TrickyWord } from "../../api";
import { Segmented, useToast } from "../../components/ui";

/** Say a word another way ("Kael" as "Kale"): this book's own words, and the ones every book and Text to Speech
 *  use. The story keeps its spelling; the voices read the new way. */
export function PronouncePanel({ rows, setRows, cast, story }: {
  rows: Pronunciation[]; setRows: (r: Pronunciation[]) => void; cast: CastRow[]; story: string;
}) {
  const toast = useToast();
  const [scope, setScope] = useState<"book" | "all">("book");
  const [shared, setShared] = useState<Pronunciation[] | null>(null);
  const [word, setWord] = useState("");
  const [say, setSay] = useState("");
  const [tricky, setTricky] = useState<TrickyWord[] | null>(null);
  const [scanning, setScanning] = useState(false);
  const saveTimer = useRef<number | undefined>(undefined);
  useEffect(() => { api.pronunciations().then(setShared).catch(() => setShared([])); }, []);

  const list = scope === "book" ? rows : shared || [];
  const setList = (next: Pronunciation[]) => {
    if (scope === "book") { setRows(next); return; }
    setShared(next);
    window.clearTimeout(saveTimer.current);
    saveTimer.current = window.setTimeout(() => api.savePronunciations(next.filter((r) => r.word.trim() && r.say.trim()))
      .catch((e) => toast((e as Error).message, true)), 600);
  };
  const add = (w = word, s = say) => {
    const ww = w.trim(), ss = s.trim();
    if (!ww || !ss) return;
    setList([...list.filter((r) => r.word.toLowerCase() !== ww.toLowerCase()), { word: ww, say: ss }]);
    setWord(""); setSay("");
  };

  // hear a respelling in the narrator's voice (a short preview, like the voice samples)
  const narrator = cast.find((c) => c.character === "Narrator")?.voice || cast[0]?.voice || "";
  const audio = useRef<HTMLAudioElement | null>(null);
  const [hearing, setHearing] = useState<string | null>(null);
  const hear = useTask((t) => {
    if (t.state === "failed") { setHearing(null); toast(t.error || "Couldn't make the sample.", true); }
    else if (t.state === "done" && t.result.audio) {
      if (!audio.current) { audio.current = new Audio(); audio.current.onended = () => setHearing(null); }
      audio.current.src = fileUrl(t.result.audio);
      audio.current.play().catch(() => setHearing(null));
    }
  });
  const listen = (text: string) => {
    if (!narrator) return toast("Give the Narrator a voice in Characters first.", true);
    if (hearing === text) { audio.current?.pause(); setHearing(null); return; }
    setHearing(text);
    hear.start(() => api.speech({ text: `${text}.`, voice: narrator, preview: true }));
  };

  const scanTricky = async () => {
    if (!story.trim()) return toast("Paste or import a story first.", true);
    setScanning(true);
    try {
      const found = await bookApi.trickyWords(story, cast.map((c) => c.character), [...rows, ...(shared || [])]);
      setTricky(found);
      if (!found.length) toast("No unusual names or homographs found.");
    } catch (e) {
      toast((e as Error).message, true);
    } finally {
      setScanning(false);
    }
  };

  // names in the story worth checking: the cast, and capitalised words that come up often mid-sentence
  const listed = new Set(list.map((r) => r.word.toLowerCase()));
  const shownTricky = (tricky || []).filter((t) => !listed.has(t.word.toLowerCase()));
  const frequentWords = useMemo(() => {
    const counts = new Map<string, number>();
    const clean = story.replace(/\{[^{}]*\}|\[[^\]]*\]/g, " ");
    for (const m of clean.matchAll(/[a-z,;] ([A-Z][a-z’']{2,})/g)) {
      const w = m[1];
      counts.set(w, (counts.get(w) || 0) + 1);
    }
    return [...counts.entries()].filter(([, n]) => n >= 2).sort((a, b) => b[1] - a[1]).map(([w]) => w);
  }, [story]);
  const names = [...new Set([
    ...cast.map((c) => c.character).filter((n) => n !== "Narrator"),
    ...frequentWords,
  ])].filter((n) => !listed.has(n.toLowerCase())).slice(0, 10);

  return (
    <div className="tool-fit">
      <p className="small muted" style={{ margin: 0 }}>Say a word another way. The text keeps its spelling; the voices read it as you write it here.</p>
      <Segmented<"book" | "all"> value={scope} onChange={setScope} options={[
        { value: "book", label: `This book${rows.length ? ` · ${rows.length}` : ""}` },
        { value: "all", label: `Every book${shared?.length ? ` · ${shared.length}` : ""}` }]} />
      <div className="say-add">
        <input className="input" placeholder="Word, e.g. Kael" aria-label="Word" value={word} onChange={(e) => setWord(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && add()} />
        <input className="input" placeholder="Say it as, e.g. Kale" aria-label="Say it as" value={say} onChange={(e) => setSay(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && add()} />
        <button className="btn icon sm ghost" disabled={!say.trim()} onClick={() => listen(say.trim())} title="Hear it" aria-label="Hear it">
          {hearing === say.trim() && say.trim() ? <Pause size={13} /> : <Volume2 size={13} />}</button>
        <button className="btn sm primary" disabled={!word.trim() || !say.trim()} onClick={() => add()}><Plus size={13} />Add</button>
      </div>
      <div className="row" style={{ gap: 6 }}>
        <button type="button" className="btn sm" disabled={scanning || !story.trim()} onClick={scanTricky}
          title="Scan the story for unusual names, fantasy spellings, and homographs (read, lead, tear, bow…)">
          <Search size={12} />{scanning ? "Scanning…" : "Find tricky words"}
        </button>
        {tricky && <span className="small muted">{shownTricky.length} to check</span>}
      </div>
      {shownTricky.length > 0 ? (
        <div className="say-names">
          {shownTricky.map((t) => (
            <button key={t.word} className={`chip-btn ${t.reason === "fantasy_name" ? "on" : ""}`}
              title={`${t.note} (${t.count}× in story)`}
              onClick={() => { setWord(t.word); setSay(""); }}>
              {t.word}
            </button>
          ))}
        </div>
      ) : names.length > 0 && (
        <div className="say-names">
          <span className="small faint">In this story:</span>
          {names.map((n) => <button key={n} className="chip-btn" onClick={() => { setWord(n); setSay(""); }}>{n}</button>)}
        </div>
      )}
      <div className="fill say-list">
        {scope === "all" && shared === null ? <div className="small muted">Loading…</div>
          : !list.length ? <div className="small muted">{scope === "book" ? "No words for this book yet." : "No words for every book yet."} Add one above, like a name the voices get wrong.</div>
            : list.map((r, i) => (
              <div key={r.word} className="say-row">
                <span className="say-word" title={r.word}>{r.word}</span>
                <span className="faint">→</span>
                <input className="input" aria-label={`Say ${r.word} as`} value={r.say}
                  onChange={(e) => setList(list.map((x, k) => (k === i ? { ...x, say: e.target.value } : x)))} />
                <button className="btn icon sm ghost" onClick={() => listen(r.say)} title="Hear it" aria-label={`Hear ${r.say}`}>
                  {hearing === r.say ? <Pause size={13} /> : <Volume2 size={13} />}</button>
                <button className="btn icon sm ghost" onClick={() => setList(list.filter((_, k) => k !== i))} title="Remove" aria-label={`Remove ${r.word}`}><X size={13} /></button>
              </div>
            ))}
      </div>
      <p className="small faint" style={{ margin: 0 }}>{scope === "book" ? "This book's words win over the every-book ones." : "Used by every audiobook and by Text to Speech."} Chapters that use a changed word are made again next time.</p>
    </div>
  );
}
