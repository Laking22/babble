import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle, ArrowLeft, CheckCircle2, ChevronsLeft, Circle, Clapperboard, Clock, FileUp, ImagePlus, Music2, PencilLine, Play, Plus,
  Cloud, Eraser, Languages, ListMusic, ListOrdered, Search, ShieldCheck, SlidersHorizontal, Square, Trash2, Undo2, Users, Wand2, X, Zap, ListChecks, Loader2,
} from "lucide-react";
import {
  api, bookApi, fileUrl, projectApi, projectName, useLoad, useTask,
  type BookRequest, type Chapter, type CloudEstimate, type CloudMatch, type Engine, type ParaTiming, type Placement, type Project, type Scope, type StorySound, type Task, type Voice,
} from "../../api";
import { ConfirmDialog, Field, ParallelLines, Slider, Starting, StopButton, accepts, clock, taskTiming, useFileDrop, useNow, useStarting, useToast } from "../../components/ui";
import { useBookJobs } from "../../bookJobs";
import { saveFile } from "../../native";
import { CastPanel } from "./CastPanel";
import { MusicTool, SoundTool, keptPlacements, type ReviewState } from "./SoundTools";
import { DirectorTool } from "./DirectorTool";
import { useAsk, useCostConfirm } from "../../components/CostConfirm";
import { IntrosTool } from "./IntrosTool";
import { Player, type Track } from "./Player";
import { ScriptView, parseStoryBlocks } from "./ScriptView";
import { ImmersionQaDialog } from "./ImmersionQa";
import { placeTag } from "./drag";
import { CLOUD_NAMES, CloudLine, MakeOnMenu, WAITING, estimateText, useCloudJob } from "./CloudMenu";
import { LineCheckDialog, useLineCheck } from "./LineCheck";
import { PronouncePanel } from "./PronouncePanel";

const BOOK_FILES = ".txt,.md,.epub,.pdf,.docx,.mobi,.azw3,.rtf,.odt,.fb2,.html";

/** " · about 5 min left" (and a note when it has been quiet a while) after the book bar's status. Its own
 * component: it updates every second, and the whole editor (a long story) needn't. */
function MakingTiming({ job }: { job: Task }) {
  const { left, quiet } = taskTiming(job, useNow(job.state === "running"));
  return <>{left ? ` · ${left}` : ""}{quiet ? ` · ${quiet}` : ""}</>;
}

/** What a book being made is doing now: its status's first line ("Stitching the audio together · 2,500 of 7,821
 * pieces"), without a percentage of its own (the bar's is shown next to it). */
const stepOf = (status?: string) => (status || "").split("\n")[0].replace(/\s*·\s*\d+%$/, "").replace(/…$/, "").trim();

type Tool = "chapters" | "cast" | "say" | "sound" | "music" | "intros" | "director" | "output";
// the rail: an icon and a short name each; the panel beside it shows the tool picked
const TOOLS: { id: Tool; label: string; short: string; icon: typeof Users }[] = [
  { id: "chapters", label: "Chapters", short: "Chapters", icon: ListOrdered },
  { id: "cast", label: "Characters", short: "Characters", icon: Users },
  { id: "say", label: "Pronunciation", short: "Pronounce", icon: Languages },
  { id: "sound", label: "Sound effects", short: "Sounds", icon: Zap },
  { id: "music", label: "Music", short: "Music", icon: Music2 },
  { id: "intros", label: "Chapter intros and outros", short: "Intros", icon: ListMusic },
  { id: "director", label: "Director", short: "Director", icon: Clapperboard },
  { id: "output", label: "Output", short: "Output", icon: SlidersHorizontal },
];
// what the book page saves (the finished file is saved by the app when a book is done)
const SAVED: (keyof Project)[] = ["title", "author", "cover", "story", "original", "cast", "models", "quotes_only", "out_format",
  "speed", "stability", "language", "seed", "batch", "parallel", "pronunciations", "line_pause", "para_pause", "sound", "bed", "fx_level", "amb_level", "music_level", "intro", "chapter_music", "intro_level", "detect", "new_sounds", "make_on", "cloud_gpu", "cloud_detect", "cloud_live", "cloud_hours", "cloud_sounds",
  "director_notes", "sfx_auto_make", "director_where", "director_models", "director_checks", "director_scenes", "sound_libraries", "director_intros"];
// settings in the book's tabs that used to be app-wide: each book keeps its own (a book that never set one starts
// from the app-wide value it had)
const LOCAL = ["sfx_auto_make", "director_where", "director_models", "director_checks", "director_scenes", "sound_libraries", "director_intros"] as const;

/** "## Chapter" lines and words, like the classic page: about 2.6 words a second. */
function storyStats(text: string, chapters: number) {
  const words = text.replace(/\[[^\]]+\]|\{[^{}]*\}/g, " ").split(/\s+/).filter(Boolean).length;
  if (!words) return "No text yet";
  return `${chapters > 1 ? `${chapters} chapters · ` : ""}${words.toLocaleString()} words · about ${clock(words / 2.6)}`;
}

const tagCount = (story: string, types: string[]) =>
  new Set([...story.matchAll(/\{\s*(sfx|amb|music)\s*:\s*([^{},@]+)/gi)].filter((m) => types.includes(m[1].toLowerCase())).map((m) => `${m[1]}:${m[2].trim().toLowerCase()}`)).size;



const lineFraction = (status: string) => {
  const head = (status || "").split("\n", 1)[0];
  const m = /(?:Speaking \((\d+)\/(\d+)\)|Line (\d+) of (\d+))/.exec(head);
  const cur = Number(m?.[1] ?? m?.[3]), tot = Number(m?.[2] ?? m?.[4]);
  return m && tot ? Math.max(0, Math.min(1, (cur - 1) / tot)) : null;
};

export function BookEditor({ id, voices, engines, ramGb, onClose }: {
  id: string; voices: Voice[]; engines: Engine[]; ramGb: number; onClose: () => void;
}) {
  const toast = useToast();
  const confirmCost = useCostConfirm();
  const ask = useAsk();
  const cloudBusy = useRef(false);  // (a second click while the first is still asking: ignored)
  // from the click until the work has started (checks, estimates and questions come first): one start, and it shows
  const beginning = useStarting();  // Create audiobook, Finish on this Mac, Create in the cloud
  const bundling = useStarting();
  const importing = useStarting();
  // a book file dragged from Finder onto the text: the same as Import (importFile is set further down)
  const dropImport = useRef<(f: File) => void>(() => {});
  const textDrop = useFileDrop((got) => {
    const fit = got.filter((f) => accepts(f, BOOK_FILES));
    if (!fit.length) return toast("That isn't a book or text file.", true);
    if (fit.length > 1) toast(`Imported ${fit[0].name}. Drop one book at a time.`);
    dropImport.current(fit[0]);
  }, importing.starting);
  const detecting = useStarting();
  const [b, setB] = useState<Project | null>(null);
  const [tool, setToolState] = useState<Tool | null>(() => {
    try { const t = localStorage.getItem("book-tool"); return t === "none" ? null : TOOLS.some((x) => x.id === t) ? (t as Tool) : "chapters"; } catch { return "chapters"; }
  });
  const setTool = (t: Tool | null) => { setToolState(t); try { localStorage.setItem("book-tool", t || "none"); } catch { /* fine */ } };
  const FLYOUT_DEFAULT = 360, FLYOUT_MIN = 240, FLYOUT_MAX = 640;
  const clampFlyout = useCallback((w: number) => {
    const cap = Math.max(FLYOUT_MIN, Math.min(FLYOUT_MAX, window.innerWidth - 420));
    return Math.round(Math.max(FLYOUT_MIN, Math.min(cap, w)));
  }, []);
  const [flyoutW, setFlyoutW] = useState<number>(() => {
    try {
      const saved = Number(localStorage.getItem("book-flyout-width"));
      return Number.isFinite(saved) && saved >= FLYOUT_MIN ? Math.round(Math.max(FLYOUT_MIN, Math.min(FLYOUT_MAX, saved))) : FLYOUT_DEFAULT;
    } catch { return FLYOUT_DEFAULT; }
  });
  const [resizing, setResizing] = useState(false);
  useEffect(() => {
    const onResize = () => setFlyoutW((w) => clampFlyout(w));
    onResize();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [clampFlyout]);
  const startResize = (e: React.PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = flyoutW;
    let lastW = startW;
    setResizing(true);
    const onMove = (ev: PointerEvent) => {
      lastW = clampFlyout(startW + (ev.clientX - startX));
      setFlyoutW(lastW);
    };
    const onUp = () => {
      setResizing(false);
      try { localStorage.setItem("book-flyout-width", String(lastW)); } catch { /* fine */ }
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
  };
  const options = useLoad(bookApi.options);
  const intros = useLoad(() => bookApi.introChoices(id), [id]);
  const [server, setServer] = useState<Record<string, any>>({});
  const [review, setReview] = useState<ReviewState>(null);
  const [chapters, setChapters] = useState<Chapter[] | null>(null);
  const [sounds, setSounds] = useState<StorySound[] | null>(null);
  const [track, setTrack] = useState<Track>(null);
  const [timings, setTimings] = useState<ParaTiming[]>([]);
  const [timingsNonce, setTimingsNonce] = useState(0);
  const [playPos, setPlayPos] = useState({ time: 0, playing: false });
  const playingRef = useRef(false);
  const [focusedBlock, setFocusedBlock] = useState<{ chapter: number; para: number } | null>(null);
  const [previewTarget, setPreviewTarget] = useState<{ chapter: number; para: number } | null>(null);
  const [dragSelection, setDragSelection] = useState("");
  // Capture any dragged text selection inside the story editor so clicking Preview, Hear, or VoicePicker never loses it
  useEffect(() => {
    const onSel = () => {
      const s = window.getSelection();
      if (!s || !s.rangeCount) return;
      const node = s.anchorNode;
      const el = node instanceof Element ? node : node?.parentElement;
      if (!el?.closest(".el-editor")) return;
      setDragSelection(s.toString().trim());
    };
    document.addEventListener("selectionchange", onSel);
    return () => document.removeEventListener("selectionchange", onSel);
  }, []);
  const [scrollSignal, setScrollSignal] = useState(0);
  const [clearing, setClearing] = useState(false); // the "clear all the text?" confirmation
  const [removing, setRemoving] = useState<Chapter | null>(null); // the "remove this chapter?" confirmation
  const [findOpen, setFindOpen] = useState(false);
  const [findQ, setFindQ] = useState("");
  const [replaceQ, setReplaceQ] = useState("");
  const [findCase, setFindCase] = useState(false);
  const [findWord, setFindWord] = useState(false);
  const findInput = useRef<HTMLInputElement>(null);
  const [sel, setSel] = useState<number | null>(null);
  const coverInput = useRef<HTMLInputElement>(null);
  const importInput = useRef<HTMLInputElement>(null);
  const { jobs, start, stop, onChange } = useBookJobs();
  const job = jobs[id];
  const making = job?.state === "running";

  // ---------- loading and saving ----------
  useEffect(() => {
    projectApi.get(id).then((p) => {
      setB(p);
      if (p.file) setTrack({ file: p.file, label: "The audiobook", v: p.updated });
    }).catch((e) => { toast(e.message, true); onClose(); });
    api.settings().then(setServer).catch(() => {});
  }, [id]); // eslint-disable-line react-hooks/exhaustive-deps

  const latest = useRef<Project | null>(null);
  latest.current = b;
  const dirty = useRef(false);
  const timer = useRef<number | null>(null);
  const flush = useCallback(() => {
    if (timer.current) { clearTimeout(timer.current); timer.current = null; }
    const p = latest.current;
    if (!dirty.current || !p) return;
    dirty.current = false;
    projectApi.save(id, Object.fromEntries(SAVED.map((k) => [k, p[k]])) as Partial<Project>)
      .catch((e) => { dirty.current = true; toast(`Couldn't save the book: ${e.message}`, true); });
  }, [id]); // eslint-disable-line react-hooks/exhaustive-deps
  const edit = (patch: Partial<Project>) => {
    setB((x) => (x ? { ...x, ...patch } : x));
    dirty.current = true;
    if (timer.current) clearTimeout(timer.current);
    timer.current = window.setTimeout(flush, 700);
  };
  const set = <K extends keyof Project>(k: K, v: Project[K]) => edit({ [k]: v } as Partial<Project>);
  useEffect(() => {
    window.addEventListener("beforeunload", flush);
    return () => { window.removeEventListener("beforeunload", flush); flush(); };
  }, [flush]);

  useEffect(() => { // a narrator to start with: your first saved voice, else the first voice this Mac can run
    if (!b || !voices.length || b.cast.length) return;
    const pick = voices.find((v) => v.tags.includes("saved") && v.available) || voices.find((v) => v.available);
    if (pick) set("cast", [{ character: "Narrator", voice: pick.value, style: "Warm, calm storytelling.", speed: 1 }]);
  }, [voices, b?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const changeStory = (story: string, original?: string) => edit({ story, original: original !== undefined ? original : (b?.original || b?.story || "") });
  const detectBaseline = useRef<string | null>(null);
  const valueOf = (label: string) => voices.find((v) => v.label === label || v.value === label)?.value || label;

  // ---------- chapters and the story's sounds ----------
  const refreshChapters = useCallback(() => {
    const p = latest.current;
    if (p) bookApi.chapters(p.story, p.folder, p.title, p.quotes_only).then(setChapters).catch(() => {});
  }, []);
  useEffect(() => {
    if (!b) return;
    const t = setTimeout(refreshChapters, 900);
    return () => clearTimeout(t);
  }, [b?.story, b?.quotes_only, b?.id, refreshChapters]); // eslint-disable-line react-hooks/exhaustive-deps
  // the book against its last cloud run, while Chapters is open: a setting changed since that makes the run's lines
  // unused is shown there (and asked about before Finish / Create in the cloud: checkMatch)
  const [mismatch, setMismatch] = useState<CloudMatch | null>(null);
  useEffect(() => {
    if (!b?.folder || tool !== "chapters") return;
    const t = setTimeout(() => bookApi.cloudMatches(request(false))
      .then((m) => setMismatch(m?.run_lines && (m.used ?? 0) < m.run_lines / 2 ? m : null)).catch(() => {}), 1200);
    return () => clearTimeout(t);
  }, [tool, b?.folder, b?.cast, b?.stability, b?.language, b?.seed, b?.quotes_only, b?.models, b?.pronunciations, b?.story]); // eslint-disable-line react-hooks/exhaustive-deps

  const refreshSounds = useCallback(() => {
    const p = latest.current;
    if (p) bookApi.storySounds(p.story, id).then(setSounds).catch(() => setSounds([]));
  }, [id]);
  useEffect(() => {
    if (!b || (tool !== "sound" && tool !== "music")) return;
    const t = setTimeout(refreshSounds, 700);
    return () => clearTimeout(t);
  }, [b?.story, tool, refreshSounds]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => onChange((book, t) => { // the book being made: new sound tags in its text, or it's finished
    if (book !== id) return;
    const r = t.result;
    if (r.story) setB((x) => (x && x.story !== r.story ? { ...x, story: r.story, original: r.original ?? x.story } : x));
    if (t.state === "running") return;
    setPreviewTarget(null);
    if (t.state === "done" && r.audio) {
      const preview = t.title.startsWith("Book preview");
      if (!preview) setB((x) => (x ? { ...x, file: r.audio } : x));
      // a new version each time: the finished book keeps its file name, so the old copy must not be played
      setTrack({ file: r.audio, label: preview ? "Preview" : "The audiobook", autoplay: preview, v: Date.now() });
    }
    refreshChapters();
    setTimingsNonce((n) => n + 1);
    intros.reload();
  }), [onChange, id]); // eslint-disable-line react-hooks/exhaustive-deps

  // while a book is made, its own report says which chapters are done
  const live = making ? (job.result.chapters as { title: string; state: Chapter["state"]; file: string | null }[] | undefined) : undefined;
  const shown: Chapter[] = useMemo(() => {
    const base = chapters || [];
    if (!live) return base;
    if (!base.length) return live.map((c, i) => ({ index: i, title: c.title, state: c.state, file: c.file, seconds: null }));
    return base.map((c) => { const l = live[c.index]; return l ? { ...c, state: l.state, file: l.file || c.file } : c; });
  }, [chapters, live]);
  const makingIndex = live ? live.findIndex((c) => c.state === "making") : -1;
  const readyWhileMaking = live ? live.filter((c) => c.state === "done").length : 0;

  // ---------- helpers: detect, suggest, theme, missing sounds (one at a time) ----------
  const helper = useTask((t) => {
    if (t.state === "failed") return toast(t.error || "That didn't work.", true);
    const r = t.result;
    if (r.cast) {
      const cur = latest.current;
      const old = new Map((cur?.cast || []).map((c) => [c.character, c]));
      const editedWhileRunning = detectBaseline.current !== null && cur?.story !== detectBaseline.current;
      detectBaseline.current = null;
      const nextCast = r.cast.map((row: any[]) => ({ character: row[0], voice: valueOf(row[1]), style: row[2] || "", speed: Number(row[3]) || 1,
        stability: old.get(row[0])?.stability ?? null }));
      if (editedWhileRunning) {
        edit({ cast: nextCast });
        toast("Updated cast (kept your newer text edits).");
      } else {
        edit({ story: r.story, original: r.original ?? cur?.story ?? "", cast: nextCast });
      }
    } else if (r.story) changeStory(r.story, r.original);
    const directed = t.title.startsWith("Direct");
    if (r.placements) {
      setReview(r.placements.length ? {
        placements: r.placements, planned_on: r.planned_on, kept: new Set(r.placements.map((p: Placement) => p.id)),
        ...(directed ? {
          title: "The director's sounds", source: "director" as const, chapterPlan: r.chapter_plan || null,
          partial: !!r.partial, scenesDone: r.scenes_done, scenesTotal: r.scenes_total, stopReason: r.stop_reason || null,
        } : {}),
      } : null);
      if (r.partial && r.stop_reason) {
        toast(`Stopped after ${r.scenes_done ?? 0} of ${r.scenes_total ?? "?"} scenes: ${r.stop_reason}`, true);
      } else if (t.status) {
        toast(t.status.split("\n")[0]);
      }
    }
    if (r.chapter_music) { edit({ chapter_music: r.chapter_music, intro: "" }); intros.reload(); }  // the director's chapter music
    if (r.chapter_file && r.audio) {
      refreshChapters();
      setTimingsNonce((n) => n + 1);
      setTrack({ file: r.audio, label: `Context preview · ¶${(r.para ?? 0) + 1}`, autoplay: true, v: Date.now() });
    }
    if (t.status && !r.placements) toast(t.status.split("\n")[0].replace(/\*\*/g, ""));
    refreshSounds();
  }, `book-helper:${id}`);
  // the volume sliders (dB under the narrator while they speak): a book that never moved them shows the mix it has
  // (the old Background and music volume, bed, moved backgrounds and music together)
  const bedDb = b && (b.bed ?? 1) > 0 ? Math.round(20 * Math.log10(b.bed ?? 1)) : -41;
  const levels = { fx: b?.fx_level ?? -4, amb: b?.amb_level ?? Math.max(-41, -23 + bedDb), music: b?.music_level ?? Math.max(-41, -23 + bedDb) };
  const local: Record<string, any> = { ...server };
  for (const k of LOCAL) if (b && b[k] != null) local[k] = b[k];
  const setLocal = (k: string, v: any) => set(k as (typeof LOCAL)[number], v);
  const needStory = (fn: () => void) => () => (latest.current?.story.trim() ? fn() : toast("Paste or import a story first.", true));

  // ---------- Create in the cloud: the lines are spoken on Kaggle, then the book is put together here ----------
  const cloud = useCloudJob(b?.folder);
  // Check lines: the cloud lines that came out wrong, set aside to be made again
  const lineCheck = useLineCheck(b?.folder);
  const [checkingLines, setCheckingLines] = useState(false);
  const [showingQa, setShowingQa] = useState(false);
  useEffect(() => { if (!making) lineCheck.reload(); }, [making, cloud.job?.state]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (!making) cloud.reload(); }, [making]); // eslint-disable-line react-hooks/exhaustive-deps
  // Create in the cloud's sending (uploads can take a while): kept by book, so leaving the book and coming back shows
  // it still sending, and when it's over the run shows (the record as it is now: it may have moved on meanwhile)
  const sending = useTask((t) => {
    if (t.state === "failed") return toast(t.error || "Couldn't send it to the cloud.", true);
    if (t.state === "done" && t.result.cloud) {
      cloud.setJob(t.result.cloud);
      cloud.reload();
      toast(`Sent to ${({ modal: "Modal", runpod: "Runpod" } as Record<string, string>)[t.result.cloud.backend] || "Kaggle"}. The lines come back here when they're made.`);
    }
  }, `book-send:${id}`);
  useEffect(() => {
    if (!(making || helper.running || sending.running || review)) return;
    const warn = (e: BeforeUnloadEvent) => { e.preventDefault(); e.returnValue = ""; };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [making, helper.running, sending.running, review]);

  // paragraph timings for the active chapter (all hooks must stay before the early return below)
  const activeCh = sel ?? track?.chapter ?? 0;
  const activeChObj = shown[activeCh];
  useEffect(() => {
    if (!activeChObj?.file || !b) { setTimings([]); return; }
    let live = true;
    const { story, quotes_only } = b;
    const { index, file } = activeChObj;
    const timer = setTimeout(() => {
      bookApi.timings(story, index, file, quotes_only)
        .then((r) => { if (live) setTimings(r.timings || []); })
        .catch(() => { if (live) setTimings([]); });
    }, 450);
    return () => { live = false; clearTimeout(timer); };
  }, [activeChObj?.index, activeChObj?.file, b?.story, b?.quotes_only, timingsNonce]); // eslint-disable-line react-hooks/exhaustive-deps

  const playingPara = useMemo(() => {
    if (track?.chapter == null || track.chapter !== activeCh || !timings.length) return null;
    const t = playPos.time;
    const last = timings[timings.length - 1];
    if (t < timings[0].start - 0.05 || t > last.end + 1.5) return null;
    let cur: ParaTiming | null = null;
    for (const row of timings) {
      if (row.start <= t + 0.05) cur = row;
      else break;
    }
    return cur ? cur.para : null;
  }, [track?.chapter, activeCh, timings, playPos.time]);

  // Keep focusedBlock synced to live playback so pausing leaves the highlight on the current paragraph
  useEffect(() => {
    if (playPos.playing && playingPara != null) {
      setFocusedBlock((prev) => (prev?.chapter === activeCh && prev?.para === playingPara ? prev : { chapter: activeCh, para: playingPara }));
    }
  }, [playPos.playing, playingPara, activeCh]);

  // Resolved text of the currently highlighted passage (dragged text first, else clicked/highlighted paragraph)
  const highlightedSample = useMemo(() => {
    if (dragSelection) return { text: dragSelection, label: "selected text" };
    if (!b || !focusedBlock || focusedBlock.para < 0) return null;
    const blk = parseStoryBlocks(b.story, shown).find((x) => x.chapter === focusedBlock.chapter && x.para === focusedBlock.para);
    if (!blk || !blk.raw.trim()) return null;
    return { text: blk.raw, label: `¶${blk.para + 1}` };
  }, [dragSelection, b, focusedBlock, shown]);

  // Find & Replace regex + match count
  const findRegex = useMemo(() => {
    if (!findQ) return null;
    const esc = findQ.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const pat = findWord ? `(?<![\\p{L}\\p{N}_])${esc}(?![\\p{L}\\p{N}_])` : esc;
    return new RegExp(pat, findCase ? "gu" : "giu");
  }, [findQ, findCase, findWord]);
  const findCount = useMemo(() => {
    if (!findRegex || !b?.story) return 0;
    return (b.story.match(findRegex) || []).length;
  }, [findRegex, b?.story]);

  // Keyboard shortcuts: Cmd+F (Find), Cmd+S (Save), Cmd+Enter (Preview), Esc (Close Find)
  const previewRef = useRef<() => void>(() => {});
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;
      if (mod && e.key.toLowerCase() === "f") {
        e.preventDefault();
        setFindOpen(true);
        setTimeout(() => { findInput.current?.focus(); findInput.current?.select(); }, 0);
      } else if (mod && e.key.toLowerCase() === "s") {
        e.preventDefault();
        flush();
        toast("Saved.");
      } else if (mod && e.key === "Enter") {
        const el = document.activeElement;
        if (el?.closest(".find-bar") || el?.closest(".modal")) return;
        e.preventDefault();
        previewRef.current();
      } else if (e.key === "Escape" && findOpen) {
        setFindOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [flush, findOpen]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!b) return <div className="book-loading muted">Opening the book…</div>;
  const name = projectName(b);

  /** Where Detect speakers runs: the ≡ menu's cloud when its toggle is on (AI only), else this Mac. */
  const aiOn = (k: "cloud_detect", method: string) =>
    b && b[k] && method === "ai" && (b.make_on ?? "mac") !== "mac" ? (b.make_on as string) : "mac";
  const request = (preview: boolean): BookRequest => ({
    story: b.story, cast: b.cast, models: b.models, quotes_only: b.quotes_only, speed: b.speed, stability: b.stability,
    language: b.language, seed: b.seed, batch: (b.parallel ?? true) ? Math.max(2, b.batch) : 1, parallel: b.parallel ?? true, pronunciations: b.pronunciations || [],
    line_pause: b.line_pause, para_pause: b.para_pause,
    out_format: b.out_format, title: b.title, author: b.author, cover: b.cover, sound: b.sound, bed: b.bed, intro: b.intro,
    fx_level: b.fx_level ?? null, amb_level: b.amb_level ?? null, music_level: b.music_level ?? null,
    chapter_music: b.chapter_music ?? null, intro_level: b.intro_level ?? null,
    automake: local.sfx_auto_make ?? true, preview,
    folder: b.folder, name, book: id, new_sounds: b.new_sounds, make_on: b.make_on ?? "mac", cloud_gpu: b.cloud_gpu ?? "A10G", cloud_live: b.cloud_live ?? true, cloud_hours: b.cloud_hours ?? null,
    cloud_sounds: b.cloud_sounds ?? true,
  });
  /** Every chapter is made already: remake the book (its current take goes to the Trash), or on this Mac just put it
   *  together again. The request to make it with, "together", or null (Cancel). */
  const askRemake = async (body: BookRequest, onMac: boolean): Promise<BookRequest | "together" | null> => {
    const seed = body.seed || 0;
    const a = await ask({
      title: "Every chapter is made already", confirm: "Remake", other: onMac ? "Just put it together" : undefined,
      lines: ["Remake the audiobook? Every line is spoken again, and the current take goes to the Trash.",
        seed ? `Seed ${seed} would give the same takes, so it's remade with seed ${seed + 1}.` : ""],
    });
    if (a !== "confirm") return a === "other" ? "together" : null;
    try {
      const r = await bookApi.remake(body);
      if (r.seed !== seed) set("seed", r.seed);
      return { ...body, seed: r.seed };
    } catch (x) { toast((x as Error).message, true); return null; }
  };
  /** Before Finish or Create in the cloud: when the book changed since its cloud run so that most of the run's lines
   *  wouldn't be used (a stability changed, a voice…), say so, and offer to put the run's settings back. The request
   *  to go on with, or null (Cancel). */
  /** What changed since the cloud run, in words ("Narrator's stability 25 (now 30)"). */
  const changedSince = (m: CloudMatch) => {
    const fmt = (v: unknown) => (v == null || v === "" ? "none" : typeof v === "boolean" ? (v ? "on" : "off") : String(v));
    return (m.changed || []).map((c) => (c.what === "the text" ? "different text" : `${c.what} ${fmt(c.then)} (now ${fmt(c.now)})`)).join(", ");
  };
  /** Put the cloud run's settings back on the book: the fields changed. */
  const putRunSettingsBack = (m: CloudMatch): Partial<Project> => {
    const { cast: castFix, ...fields } = m.restore || {};
    const cast = castFix ? b.cast.map((c) => (castFix[c.character] ? { ...c, ...castFix[c.character] } : c)) : b.cast;
    const patch = { ...fields, ...(castFix ? { cast } : {}) } as Partial<Project>;
    edit(patch);
    setMismatch(null);
    toast(`Put back the cloud run's settings: ${changedSince(m)}.`);
    return patch;
  };
  const checkMatch = async (body: BookRequest): Promise<BookRequest | null> => {
    if (!b.folder) return body;
    const m = await bookApi.cloudMatches(body).catch(() => null);  // (can't tell: go on as before)
    if (!m?.run_lines || (m.used ?? 0) >= m.run_lines / 2) return body;
    const what = changedSince(m);
    const canRestore = Object.keys(m.restore || {}).length > 0;
    const unused = m.run_lines - (m.used ?? 0);
    const a = await ask({
      title: "The cloud lines don't match the book now",
      confirm: canRestore ? "Use the run's settings" : "Make them again",
      other: canRestore ? "Make them again" : undefined,
      lines: [`${m.run_lines.toLocaleString()} lines came back from the cloud, made with ${what || "other settings"}. `
        + `As the book is now, ${m.used ? `only ${m.used.toLocaleString()} of them are` : "none of them are"} used, `
        + `and ${unused.toLocaleString()} would be made again.`,
        canRestore ? "Use the run's settings puts those back, so the lines from the cloud are used."
          : "The text changed since, so those lines can't be used."],
    });
    if (!a) return null;
    if (!canRestore || a === "other") return body;
    return { ...body, ...putRunSettingsBack(m) } as BookRequest;
  };
  const go = async (preview: boolean) => {
    if (!b.story.trim()) return toast("Paste or import a story first.", true);
    flush();
    let body = request(preview);
    if (preview) {
      const s = window.getSelection();
      const node = s?.rangeCount ? s.anchorNode : null;
      const inEditor = (node instanceof Element ? node : node?.parentElement)?.closest(".el-editor");
      const liveSel = inEditor ? (s?.toString().trim() || "") : "";
      const pickText = liveSel || highlightedSample?.text || "";
      if (pickText) body = { ...body, preview_text: pickText };
      else if (focusedBlock?.chapter != null || sel != null) body = { ...body, preview_chapter: focusedBlock?.chapter ?? sel ?? 0 };
      if (!liveSel && focusedBlock && focusedBlock.para >= 0) setPreviewTarget(focusedBlock);
    } else {
      const fixed = await checkMatch(body);
      if (!fixed) return;
      body = fixed;
      const made = await bookApi.made(body).catch(() => null);  // (any error: Create shows it)
      if (made?.all) {
        const r = await askRemake(body, true);
        if (!r) return;
        if (r !== "together") body = r;
      }
    }
    start(id, preview ? "Book preview" : name, () => bookApi.create(body));
  };
  previewRef.current = () => { if (!making) go(true); };
  const goCloud = async () => {
    if (cloudBusy.current) return;
    cloudBusy.current = true;
    try { await goCloudOnce(); } finally { cloudBusy.current = false; }
  };
  /** Stop the book's run out there (its lines back are kept, the newest brought back first). null: it didn't. */
  const stopRun = async (backend?: string) => {
    toast("Stopping the cloud run…");
    try {
      const r = await cloud.stop();
      const where = CLOUD_NAMES[backend ?? "kaggle"] || "the cloud";
      if (r.stopped === false && (backend ?? "kaggle") === "kaggle")
        toast("Let go. It carries on on Kaggle until its time limit: stop it on kaggle.com to save your GPU hours.");
      else if (r.stopped === false)
        toast(`${where} couldn't be told to stop: it carries on until its time limit. You can stop it on ${where}'s site.`, true);
      else toast(r.stopped ? `Stopped. ${(r.back ?? 0).toLocaleString()} of ${(r.lines ?? 0).toLocaleString()} lines came back.` : "Stopped.");
      return r;
    } catch (x) { toast((x as Error).message, true); return null; }
  };
  const startAgain = async () => {
    try { await bookApi.cloudStartAgain(b.folder || ""); return true; } catch (x) { toast((x as Error).message, true); return false; }
  };
  /** The book bar's Stop: asked first, then whether to send the book again (continue, or start again). Throws when
   *  it's kept going, so the Stop button comes back. */
  const stopCloud = async () => {
    const j = cloud.job;
    if (!j) return;
    const kaggle = (j.backend ?? "kaggle") === "kaggle";
    const a = await ask({
      title: "Stop the cloud run?", confirm: "Stop", cancel: "Keep it going",
      lines: [`${(j.back ?? 0).toLocaleString()} of ${(j.lines ?? 0).toLocaleString()} lines are back. They're kept`
        + (kaggle ? "." : ", and the ones made since the last copy are brought back first."),
        ...(kaggle ? ["Kaggle can't be stopped from here: Babble lets it go, and it carries on until its time limit "
          + "(stop it on kaggle.com to save your GPU hours)."] : []),
        ...(j.live_error ? [`Lines can't be copied back right now (${j.live_error.replace(/\.$/, "")}). Stopping now `
          + "leaves the ones made since the last copy out of reach. Kept going, they come back once the connection works."] : [])],
    });
    if (a !== "confirm") throw new Error("kept going");
    const r = await stopRun(j.backend);
    if (!r) throw new Error("not stopped");
    flush();
    let e: CloudEstimate;
    try { e = await bookApi.cloudEstimate(request(false)); } catch (x) { return toast((x as Error).message, true); }
    const left = e.lines || 0, run = r.lines || j.lines || 0;
    const again = await ask({
      title: "Send the book to the cloud again?", cancel: "Not now",
      confirm: left ? `Continue (${left.toLocaleString()} left)` : "Start again",
      other: left && run ? `Start again (all ${run.toLocaleString()})` : undefined,
      lines: [left ? "Continue keeps the lines that came back and sends only the rest."
        : "Every line the book needs is back: Finish puts it together on this Mac.",
        "Start again puts this run's lines, and the chapters made from them, in the Trash and makes them all again."],
    });
    if (!again) return;
    if ((again === "other" || !left) && !(await startAgain())) return;
    await goCloud();
  };
  const goCloudOnce = async () => {
    if (!b.story.trim()) return toast("Paste or import a story first.", true);
    flush();
    let body = request(false);
    let e: CloudEstimate | null = null;
    try { e = await bookApi.cloudEstimate(body); } catch (x) { return toast((x as Error).message, true); }
    if (e.busy) {  // out there already (a double click, or sent before): one run per book
      if (e.busy.state === "sending") return toast("This book is already being sent to the cloud.");
      const mins = e.busy.sent ? Math.max(1, Math.round((Date.now() / 1000 - e.busy.sent) / 60)) : 0;
      const kaggle = (e.busy.backend ?? "kaggle") === "kaggle";
      const a = await ask({
        title: "Already being made in the cloud", confirm: "Continue", other: "Start again", cancel: "Keep it going",
        lines: [`This book was sent ${mins ? (mins < 90 ? `${mins} min` : `${Math.round(mins / 60)} h`) + " ago" : "already"}`
          + (e.busy.lines ? `; ${(e.busy.back || 0).toLocaleString()} of ${e.busy.lines.toLocaleString()} lines are back.` : "."),
          "Continue: the run out there is stopped, the lines already back are kept, and only the rest are sent again.",
          "Start again: its lines, and the chapters made from them, go to the Trash, and every line is made again.",
          ...(kaggle ? ["Kaggle can't be stopped from here: the run out there carries on until its time limit."] : [])],
      });
      if (!a) return;
      if (!(await stopRun(e.busy.backend))) return;
      if (a === "other" && !(await startAgain())) return;
      try { e = await bookApi.cloudEstimate(body); } catch (x) { return toast((x as Error).message, true); }
    }
    const fixed = await checkMatch(body);
    if (!fixed) return;
    if (fixed !== body) {  // (the run's settings put back: what's left to send changes)
      body = fixed;
      try { e = await bookApi.cloudEstimate(body); } catch (x) { return toast((x as Error).message, true); }
    }
    if (!e.lines && e.made_all) {
      const r = await askRemake(body, false);
      if (!r || r === "together") return;
      body = r;
      try { e = await bookApi.cloudEstimate(body); } catch (x) { return toast((x as Error).message, true); }
    }
    if (!e.lines) return toast(estimateText(e));
    const where = CLOUD_NAMES[e.backend] || e.backend;
    const ok = await confirmCost({
      title: `Create in the cloud on ${where}?`, confirm: "Send",
      cost: e.free ? `free · ${e.low_hours.toFixed(1)}–${e.high_hours.toFixed(1)} weekly GPU h`
        : e.low_cost != null && e.high_cost != null ? `$${e.low_cost.toFixed(2)}–$${e.high_cost.toFixed(2)}` : "price unknown",
      free: e.free, details: [estimateText(e)],
    });
    if (ok) sending.start(() => bookApi.cloudSend(body));
  };
  const detectNow = async () => {
    const on = aiOn("cloud_detect", b.detect);
    if (on !== "mac") {  // on a cloud GPU: money, or Kaggle's weekly hours
      try {
        const e = await bookApi.detectEstimate(b.story, on, b.cloud_gpu ?? "");
        const ok = await confirmCost({ title: `Detect speakers on ${e.where}?`, confirm: "Detect", cost: e.cost, free: e.free,
          details: [`${e.gpu} · ${e.time}`] });
        if (!ok) return;
      } catch (x) { return toast((x as Error).message, true); }
    }
    detectBaseline.current = b.story;
    helper.start(() => bookApi.detect(b.story, b.cast, b.detect, on, b.cloud_gpu ?? ""));
  };
  // the lines that came back are used and the rest made here. The cloud job is kept until the book is made (the
  // engine lets it go then), so stopping Finish halfway leaves Finish there to press again
  const finish = () => go(false);
  const kaggle = async () => {
    try { const r = await bookApi.kaggle(request(false)); toast(r.message.split("\n")[0]); await saveFile(r.path).then((m) => toast(m)); }
    catch (e) { toast((e as Error).message, true); }
  };
  const save = (path: string) => saveFile(path).then((m) => toast(m), (e) => toast(String(e), true));
  const uploadCover = async (f: File) => {
    try { set("cover", (await bookApi.uploadFile(f)).path); } catch (e) { toast((e as Error).message, true); }
  };
  const importFile = async (f: File) => {
    try {
      const r = await bookApi.importFile(f);
      edit({ story: r.text, original: b.story, title: b.title || r.title, author: b.author || r.author, cover: b.cover || r.cover,
        out_format: options.data?.formats[0] || b.out_format });
      toast(r.message);
    } catch (e) { toast((e as Error).message, true); }
  };
  dropImport.current = (f) => importing.run(() => importFile(f));
  const addKept = async () => {
    if (!review) return;
    try {
      const kept = keptPlacements(review);
      const r = await bookApi.apply(b.story, review.planned_on, kept);
      changeStory(r.story, b.story);
      setReview(null);
      toast(r.message);
      // the director's sounds are new descriptions, so they're made right away (Create would otherwise play
      // nothing for them when "Make missing sounds automatically" is off)
      // only the director's own new sounds (the story's other tags are made by Create or Make missing sounds)
      const toMake = [...new Set(kept.filter((p) => p.missing).map((p) => `{${p.kind}: ${p.name}}`))].join(" ");
      const plan = local.director_intros && review.chapterPlan ? review.chapterPlan : null;
      // with a cloud that makes the sounds too, only the sound libraries' recorded ones are fetched now; the AI ones
      // go up with the lines (Create in the cloud)
      const later = (b.make_on ?? "mac") !== "mac" && (b.cloud_sounds ?? true);
      if (review.source === "director" && (toMake || plan)) helper.start(() => bookApi.missing(toMake, id, b.new_sounds, plan, later));
    } catch (e) { toast((e as Error).message, true); }
  };
  /** Put a sound's tag in the story at `at` (moved out of a word or another tag). */
  const putTag = (at: number, tag: string) => {
    const r = placeTag(b.story, at, tag);
    changeStory(r.story, b.story);
    return r;
  };
  const insert = (tag: string) => {
    let at = b.story.length;
    if (focusedBlock && focusedBlock.para >= 0) {
      const blk = parseStoryBlocks(b.story, shown).find((x) => x.chapter === focusedBlock.chapter && x.para === focusedBlock.para);
      if (blk) at = blk.end;
    }
    putTag(at, tag);
  };
  const pickChapter = (c: Chapter) => {
    setSel(c.index);
    setScrollSignal((n) => n + 1);
  };
  const addChapter = () => {
    const n = (shown.length || 0) + 1;
    const story = b.story.replace(/\s*$/, "") + (b.story.trim() ? "\n\n" : "") + `## Chapter ${n}\n\n`;
    changeStory(story, b.story);
    setSel(n - 1);
    setScrollSignal((x) => x + 1);
  };
  const playChapter = (c: Chapter, seek?: number, autoplay = true) => c.file && setTrack({
    file: c.file, label: `${c.index + 1} · ${c.title}`, autoplay, v: track?.file === c.file ? track.v : Date.now(),
    seek, seekNonce: Date.now(), chapter: c.index,
  });

  const focusPara = async (chIdx: number, paraIdx: number, timing?: ParaTiming) => {
    setDragSelection("");
    setFocusedBlock({ chapter: chIdx, para: paraIdx });
    setSel(chIdx);
    const ch = shown[chIdx];
    if (!ch?.file) return;
    const shouldPlay = playingRef.current;
    if (timing) { playChapter(ch, timing.start, shouldPlay); return; }
    try {
      const r = await bookApi.timings(b.story, chIdx, ch.file, b.quotes_only);
      setTimings(r.timings || []);
      const hit = (r.timings || []).find((x) => x.para === paraIdx);
      playChapter(ch, hit?.start ?? 0, shouldPlay);
    } catch {
      playChapter(ch, 0, shouldPlay);
    }
  };

  const regenPara = (chIdx: number, paraIdx: number) => {
    flush();
    helper.start(() => bookApi.regenParagraph({ ...request(false), chapter: chIdx, para: paraIdx, seed_bump: Date.now() % 997 + 1 }));
  };

  const counts: Record<Tool, number | null> = {
    chapters: shown.length || null, cast: b.cast.length, say: b.pronunciations?.length || null, sound: tagCount(b.story, ["sfx", "amb"]) || null, music: tagCount(b.story, ["music"]) || null,
    intros: (b.chapter_music?.chapters || []).filter((r) => r.intro || r.outro).length || (b.intro ? 1 : null), director: null, output: null,
  };

  return (
    <>
      {/* the cover picker, here so the Chapters and Output panels' cover buttons both reach it */}
      <input ref={coverInput} type="file" hidden accept="image/*" onChange={(e) => { const f = e.target.files?.[0]; if (f) uploadCover(f); e.target.value = ""; }} />
      <nav className="book-rail" aria-label="This book's tools">
        <button className="rail-btn back" onClick={() => { flush(); onClose(); }} title="All audiobooks"><ArrowLeft size={18} /><span>Books</span></button>
        {TOOLS.map((t) => (
          <button key={t.id} className={`rail-btn ${tool === t.id ? "active" : ""}`} aria-expanded={tool === t.id} aria-label={t.label}
            title={t.label} onClick={() => setTool(tool === t.id ? null : t.id)}>
            <span className="rail-icon"><t.icon size={18} />{counts[t.id] ? <i className="rail-count">{counts[t.id]}</i> : null}</span>
            <span>{t.short}</span>
          </button>
        ))}
      </nav>

      {tool && (
        <section className={`flyout ${resizing ? "resizing" : ""}`} style={{ ["--flyout-w" as string]: `${flyoutW}px` }} aria-label={TOOLS.find((t) => t.id === tool)!.label}>
          <div className="fly-h">
            <h2>{TOOLS.find((t) => t.id === tool)!.label}</h2>
            <span className="spacer" />
            <button className="btn icon sm ghost" onClick={() => setTool(null)} aria-label="Close the panel" title="Close the panel"><ChevronsLeft size={16} /></button>
          </div>
          <div className="fly-body">
            {tool === "chapters" && (
              <div className="tool-fit">
        <div className="book-id">
                  <button className="mini-cover" style={{ ["--c" as string]: b.color }} onClick={() => coverInput.current?.click()} title={b.cover ? "Change the cover" : "Add a cover"}>
                    {b.cover ? <img src={fileUrl(b.cover)} alt="Cover" /> : <ImagePlus size={18} />}
                  </button>
                  <div className="stack" style={{ gap: 4, minWidth: 0, flex: 1 }}>
                    <input className="input bare title-in" aria-label="Title" placeholder={`Untitled-${b.untitled}`} value={b.title} onChange={(e) => set("title", e.target.value)} />
                    <input className="input bare small" aria-label="Author" placeholder="Add an author" value={b.author} onChange={(e) => set("author", e.target.value)} />
                  </div>
                </div>
        {mismatch && (
                  <p className="small cloud-mismatch">
                    <AlertTriangle size={12} /> The {mismatch.run_lines?.toLocaleString()} lines from the cloud were made with {changedSince(mismatch)},
                    so they aren't used as the book is now.{" "}
                    {Object.keys(mismatch.restore || {}).length > 0 && <button className="link-btn" onClick={() => putRunSettingsBack(mismatch)}>Use the run's settings</button>}
                  </p>
                )}
        <div className="chapters fill">
                  {shown.map((c) => {
                    const isMaking = c.state === "making";
                    const inner = isMaking ? lineFraction(job?.status || "") : null;
                    return (
                      <div key={c.index} className={`ch ${sel === c.index ? "sel" : ""}`}>
                        <ChapterIcon state={c.state} fraction={inner} />
                        <button className="ch-name" onClick={() => pickChapter(c)} title={c.title}>{c.index + 1} · {c.title}</button>
                        {c.file && ["done", "gaps", "partial", "edited"].includes(c.state)
                          ? <button className="btn icon sm ghost" onClick={() => playChapter(c)} title={c.state === "edited" ? "Play what was made (before the last edits)" : "Play this chapter"}><Play size={13} /></button>
                          : <span className="ch-state small faint">{isMaking ? (inner != null ? `${Math.round(inner * 100)}%` : "making") : c.state === "waiting" ? "waiting" : c.state === "not made" ? "not made" : ""}</span>}
                        <button className="btn icon sm ghost ch-del" onClick={() => setRemoving(c)} disabled={!!making}
                          title={making ? "Wait until the book is made" : `Remove “${c.title}”`} aria-label={`Remove chapter ${c.index + 1}`}><Trash2 size={13} /></button>
                      </div>
                    );
                  })}
                  <button className="ch add" onClick={addChapter}><Plus size={14} />Add a chapter</button>
                </div>
              </div>
            )}
            {tool === "say" && (
              <PronouncePanel rows={b.pronunciations || []} setRows={(r) => set("pronunciations", r)} cast={b.cast} story={b.story} />
            )}
            {tool === "cast" && (
              <CastPanel cast={b.cast} setCast={(c) => set("cast", c)}
                onCleared={async (keep) => {  // their [Name] tags go too (sound tags stay)
                  try { const r = await bookApi.stripTags(b.story, keep); if (r.story !== b.story) changeStory(r.story, b.story); }
                  catch (x) { toast((x as Error).message, true); }
                }} voices={voices} engines={engines}
                models={b.models} setModels={(m) => set("models", m)} modelChoices={options.data?.models || [{ label: "As cast", value: "cast" }]}
                quotesOnly={b.quotes_only} setQuotesOnly={(v) => set("quotes_only", v)} story={b.story} stability={b.stability} language={b.language}
                detect={b.detect} setDetect={(v) => set("detect", v)} busy={helper.running || detecting.starting} detecting={detecting.starting}
                detectOn={aiOn("cloud_detect", b.detect) !== "mac" ? CLOUD_NAMES[b.make_on ?? ""] : undefined}
                sampleText={highlightedSample?.text}
                sampleLabel={highlightedSample?.label}
                pronunciations={b.pronunciations}
                onDetect={() => detecting.run(needStory(detectNow))} />
            )}
            {tool === "sound" && (
              <SoundTool book={id} sounds={sounds} reloadSounds={refreshSounds} insert={insert}
                sound={b.sound} setSound={(v) => set("sound", v)} newSounds={b.new_sounds} setNewSounds={(v: Scope) => set("new_sounds", v)}
                server={local} saveServer={setLocal} busy={helper.running}
                levels={levels} setLevel={(k, v) => set(k, v)}
                onMissing={needStory(() => helper.start(() => bookApi.missing(b.story, id, b.new_sounds)))} />
            )}
            {tool === "director" && (
              <DirectorTool story={b.story} server={local} saveServer={setLocal} busy={helper.running}
                review={review} setReview={setReview} onAddKept={addKept}
                notes={b.director_notes || ""} setNotes={(v) => set("director_notes", v)}
                onDirect={(where, model, checks, scenes) => needStory(() => helper.start(() => bookApi.direct(b.story, where, model, checks, b.director_notes || "", id, scenes)))()} />
            )}
            {tool === "music" && (
              <MusicTool book={id} story={b.story} title={b.title} sounds={sounds} reloadSounds={refreshSounds} insert={insert}
                newSounds={b.new_sounds} />
            )}
            {tool === "intros" && (
              <IntrosTool book={id} story={b.story} music={b.chapter_music} legacyIntro={b.intro || ""}
                setMusic={(m) => edit({ chapter_music: m, intro: "" })} level={b.intro_level ?? -4} setLevel={(v) => set("intro_level", v)}
                choices={intros.data || []} reloadChoices={intros.reload} newSounds={b.new_sounds} directorOn={!!local.director_intros} />
            )}
            {tool === "output" && (
              <div className="stack">
                <Field label="Format">
                  <select className="select" value={b.out_format} onChange={(e) => set("out_format", e.target.value)}>
                    {(options.data?.formats || ["WAV"]).map((f) => <option key={f}>{f}</option>)}
                  </select>
                </Field>
                <div className="row">
                  {b.cover ? <img className="cover" src={fileUrl(b.cover)} alt="Cover" /> : <div className="cover"><ImagePlus size={20} /></div>}
                  <div className="stack" style={{ gap: 6 }}>
                    <button className="btn sm" onClick={() => coverInput.current?.click()}>{b.cover ? "Change cover" : "Add a cover"}</button>
                    {b.cover && <button className="btn sm ghost" onClick={() => set("cover", null)}>Remove</button>}
                  </div>
                </div>
                <Slider label="Speed" value={b.speed} min={0.7} max={1.3} step={0.05} onChange={(v) => set("speed", v)} format={(v) => `${v.toFixed(2)}×`} left="Slower" right="Faster" />
                <Slider label="Stability (the whole book)" value={b.stability} min={0} max={100} step={5} onChange={(v) => set("stability", v)} format={(v) => `${v}%`} left="More expressive" right="More stable" />
                <p className="small muted" style={{ margin: 0 }}>A character can have their own stability in Characters.</p>
                <ParallelLines on={b.parallel ?? true} count={b.batch} setOn={(v) => set("parallel", v)} setCount={(n) => set("batch", n)} />
                <details className="advanced">
                  <summary>Advanced</summary>
                  <div className="stack">
                    <Field label="Language">
                      <select className="select" value={b.language} onChange={(e) => set("language", e.target.value)}>
                        {(options.data?.languages || ["english"]).map((l) => <option key={l} value={l}>{l[0].toUpperCase() + l.slice(1)}</option>)}
                      </select>
                    </Field>
                    <Slider label="Pause between lines" value={b.line_pause} min={0} max={2} step={0.05} onChange={(v) => set("line_pause", v)} format={(v) => `${v.toFixed(2)} s`} />
                    <Slider label="Pause between paragraphs" value={b.para_pause} min={0} max={3} step={0.05} onChange={(v) => set("para_pause", v)} format={(v) => `${v.toFixed(2)} s`} />
                    <Field label="Seed" hint="0 is random; the same seed gives the same takes.">
                      <input className="input" type="number" min={0} value={b.seed} onChange={(e) => set("seed", Number(e.target.value) || 0)} />
                    </Field>
                    <button className="btn sm" disabled={bundling.starting} onClick={() => bundling.run(needStory(kaggle))}
                      title="Bundle the book to make it on Kaggle's free GPUs (no sound effects there)">
                      {bundling.starting ? <Starting label="Making the bundle…" /> : "Make a Kaggle bundle"}</button>
                  </div>
                </details>
                {b.file && <button className="btn" onClick={() => save(b.file!)}>Save the audiobook file</button>}
              </div>
            )}
          </div>
          <div
            className="flyout-resize"
            role="separator"
            aria-orientation="vertical"
            aria-label="Resize control panel"
            aria-valuemin={FLYOUT_MIN}
            aria-valuemax={FLYOUT_MAX}
            aria-valuenow={flyoutW}
            tabIndex={0}
            title="Drag to resize · double-click to reset"
            onPointerDown={startResize}
            onDoubleClick={() => {
              const reset = clampFlyout(FLYOUT_DEFAULT);
              setFlyoutW(reset);
              try { localStorage.setItem("book-flyout-width", String(reset)); } catch { /* fine */ }
            }}
            onKeyDown={(e) => {
              if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
                e.preventDefault();
                const next = clampFlyout(flyoutW + (e.key === "ArrowRight" ? 20 : -20));
                setFlyoutW(next);
                try { localStorage.setItem("book-flyout-width", String(next)); } catch { /* fine */ }
              }
            }}
          >
            <span className="flyout-grip" aria-hidden="true" />
          </div>
        </section>
      )}

      <main className="book-main">
        <div className="book-bar">
          <div className="bar-info">
            <b className="bar-title">{projectName(b)}</b>
            {sending.task?.state === "running" ? (
              <span className="small muted status-line"><Cloud size={12} /> {sending.task.status.split("\n")[0] || "Sending it to the cloud…"}</span>
            ) : !making && cloud.job ? (
              <CloudLine job={cloud.job} onCheck={cloud.checkNow} onForget={cloud.forget} onStop={stopCloud} />
            ) : making ? (
              <span className="small muted status-line" title={ramGb < 16 ? "Close other memory-heavy apps on an 8 GB Mac" : undefined}>
                {job.title.startsWith("Book preview") ? "Previewing"
                  : stepOf(job.status) || (makingIndex >= 0 && live ? `Chapter ${makingIndex + 1}/${live.length}` : "Starting")}
                {job.progress != null ? ` · ${Math.round(job.progress * 100)}%` : "…"}
                <MakingTiming job={job} />
                {readyWhileMaking ? ` · ${readyWhileMaking} ready` : ""}
              </span>
            ) : job?.state === "failed" ? <span className="small error-text status-line"><AlertTriangle size={12} /> {job.error}</span>
              : job?.state === "done" && !job.title.startsWith("Book preview")
                ? <span className="small muted status-line">{job.status.replace(/^Done · /, "Done: ").split("\n")[0]}</span>
                : <span className="small muted status-line">{storyStats(b.story, shown.length)}</span>}
            {making && <div className={`bar thin ${job.progress == null ? "indeterminate" : ""}`}><i style={{ width: `${Math.round((job.progress ?? 0) * 100)}%` }} /></div>}
            {!making && (sending.task?.state === "running" || (cloud.job?.state && WAITING.includes(cloud.job.state))) && (
              cloud.job?.progress != null && sending.task?.state !== "running"
                ? <div className="bar thin"><i style={{ width: `${Math.round(cloud.job.progress * 100)}%` }} /></div>
                : <div className="bar thin indeterminate"><i /></div>)}
            {!making && sending.task?.state !== "running" && cloud.job?.state === "ready" && cloud.job.stitch && (
              <div className={`bar thin ${cloud.job.stitch.progress == null ? "indeterminate" : ""}`}><i style={{ width: `${Math.round((cloud.job.stitch.progress ?? 0) * 100)}%` }} /></div>)}
          </div>
          {making ? <StopButton key={jobs[id]?.id} className="btn sm" onStop={() => stop(id)} stopping={jobs[id]?.stopping}><Square size={12} />Stop</StopButton> : (
            <>
              {!!(lineCheck.info?.cloud_lines || lineCheck.info?.aside?.length) && !(cloud.job?.state && WAITING.includes(cloud.job.state)) && (
                <button className="btn sm" onClick={() => setCheckingLines(true)}
                  title="Find lines from the cloud that came out garbled, cut short or said twice, and make them again">
                  <ListChecks size={14} />{lineCheck.checking.running
                    ? `Checking${lineCheck.checking.task?.progress != null ? ` ${Math.round(lineCheck.checking.task.progress * 100)}%` : "…"}`
                    : lineCheck.info?.wrong?.length ? `${lineCheck.info.wrong.length} wrong` : "Check lines"}</button>
              )}
              {b.folder && b.story.trim() && (
                <button className="btn sm" onClick={() => setShowingQa(true)}
                  title="Two-stage Immersion QA: check audio continuity, tags, and story flow">
                  <ShieldCheck size={14} />Finish-up QA
                </button>
              )}
              <button className="btn sm" onMouseDown={(e) => e.preventDefault()} onClick={() => go(true)}
                title={highlightedSample ? `Preview ${highlightedSample.label}` : "Preview the highlighted paragraph, selected chapter's opening, or first paragraph"}>
                <Play size={14} />{highlightedSample ? `Preview ${highlightedSample.label === "selected text" ? "selection" : highlightedSample.label}` : "Preview"}
              </button>
              {cloud.job?.stitch
                ? <>
                    <button className="btn sm primary" disabled title={cloud.job.stitch.status}><Loader2 size={14} className="spin" />{cloud.job.voice_check?.state === "checking" ? "Verifying voices…" : /^(Finalizing|Encoding|Exporting)/i.test(cloud.job.stitch.status || "") ? "Finalizing…" : "Assembling…"}</button>
                    {/* (Stop stops it all: the run out there too while it's still making lines, not only putting it together) */}
                    <StopButton key={cloud.job.stitch.id} className="btn sm"
                      onStop={() => WAITING.includes(cloud.job!.state ?? "") ? stopCloud()
                        : api.stopTask(cloud.job!.stitch!.id).then(() => { window.setTimeout(cloud.reload, 1500); })}><Square size={12} />Stop</StopButton>
                  </>
                : cloud.job?.state === "ready"
                ? <>
                    {cloud.job.stitch_end === "partial" && (b.make_on ?? "mac") !== "mac" && (
                      <button className="btn sm" onClick={() => beginning.run(goCloud)} disabled={beginning.starting || sending.task?.state === "running"}
                        title="Send the lines that didn't come back to the cloud again">
                        {beginning.starting || sending.task?.state === "running"
                          ? <Starting label={beginning.starting ? "Starting…" : "Sending…"} /> : <><Cloud size={14} />Create in the cloud</>}</button>
                    )}
                    <button className="btn sm primary" disabled={beginning.starting} onClick={() => beginning.run(finish)} title="Put the book together here with the lines that came back">
                      {beginning.starting ? <Starting /> : <><Wand2 size={14} />Finish on this Mac</>}</button>
                  </>
                : (b.make_on ?? "mac") !== "mac"
                  ? <button className="btn sm primary" onClick={() => beginning.run(goCloud)}
                      disabled={beginning.starting || sending.task?.state === "running" || !!(cloud.job?.state && WAITING.includes(cloud.job.state))}
                      title={`Make the lines on ${b.make_on === "kaggle" ? "Kaggle's" : `a ${CLOUD_NAMES[b.make_on ?? "kaggle"] ?? "cloud"}`} GPU, then put the book together on this Mac`}>
                      {beginning.starting || sending.task?.state === "running"
                        ? <Starting label={beginning.starting ? "Starting…" : "Sending…"} /> : <><Cloud size={14} />Create in the cloud</>}</button>
                  : <button className="btn sm primary" disabled={beginning.starting} onClick={() => beginning.run(() => go(false))}>
                      {beginning.starting ? <Starting /> : <><Wand2 size={14} />Create audiobook</>}</button>}
              <MakeOnMenu value={b.make_on ?? "mac"} onChange={(v) => set("make_on", v)} gpu={b.cloud_gpu ?? "A10G"} onGpu={(g) => set("cloud_gpu", g)} request={() => request(false)}
                ai={{ detect: !!b.cloud_detect }} onAi={(_, v) => set("cloud_detect", v)}
                live={b.cloud_live ?? true} onLive={(v) => set("cloud_live", v)}
                hours={b.cloud_hours ?? 3} onHours={(h) => set("cloud_hours", h)}
                sounds={b.cloud_sounds ?? true} onSounds={(v) => set("cloud_sounds", v)} />
            </>
          )}
        </div>
        <div className={`book-text${textDrop.over ? " file-over" : ""}`} {...textDrop.props}>
          {findOpen && (
            <div className="find-bar" role="search" aria-label="Find and replace">
              <Search size={14} className="muted" />
              <input ref={findInput} className="input sm" placeholder="Find in book…" value={findQ}
                onChange={(e) => setFindQ(e.target.value)} onKeyDown={(e) => e.key === "Escape" && setFindOpen(false)} />
              <input className="input sm" placeholder="Replace with…" value={replaceQ}
                onChange={(e) => setReplaceQ(e.target.value)} onKeyDown={(e) => e.key === "Escape" && setFindOpen(false)} />
              <label className="row small muted" style={{ gap: 4, cursor: "pointer", whiteSpace: "nowrap" }} title="Match case">
                <input type="checkbox" checked={findCase} onChange={(e) => setFindCase(e.target.checked)} style={{ accentColor: "var(--accent)" }} />Aa
              </label>
              <label className="row small muted" style={{ gap: 4, cursor: "pointer", whiteSpace: "nowrap" }} title="Whole word">
                <input type="checkbox" checked={findWord} onChange={(e) => setFindWord(e.target.checked)} style={{ accentColor: "var(--accent)" }} />Word
              </label>
              <span className="small muted tnum" style={{ minWidth: 68, textAlign: "right" }}>
                {findQ ? `${findCount} match${findCount === 1 ? "" : "es"}` : ""}
              </span>
              <button className="btn sm" disabled={!findCount} onClick={() => {
                if (!findRegex) return;
                const single = new RegExp(findRegex.source, findCase ? "" : "i");
                const next = b.story.replace(single, () => replaceQ);
                if (next !== b.story) { changeStory(next, b.story); toast("Replaced 1 occurrence."); }
              }}>Replace 1</button>
              <button className="btn sm primary" disabled={!findCount} onClick={() => {
                if (!findRegex) return;
                const next = b.story.replace(findRegex, () => replaceQ);
                if (next !== b.story) { changeStory(next, b.story); toast(`Replaced ${findCount} occurrence${findCount === 1 ? "" : "s"}. Undo brings it back.`); }
              }}>Replace all</button>
              <button className="btn icon sm ghost" onClick={() => setFindOpen(false)} aria-label="Close find" title="Close (Esc)"><X size={14} /></button>
            </div>
          )}
          <ScriptView
            story={b.story}
            chapters={shown}
            activeChapter={activeCh}
            scrollSignal={scrollSignal}
            timings={timings}
            playingPara={playingPara}
            isPlaying={playPos.playing}
            focusedBlock={focusedBlock}
            previewingPara={making ? previewTarget : null}
            busy={making || helper.running}
            onFocusPara={focusPara}
            onPreviewPara={(ch, pa, rawText) => {
              if (!rawText.trim()) return toast("That paragraph is empty.", true);
              setDragSelection("");
              setFocusedBlock({ chapter: ch, para: pa });
              setPreviewTarget({ chapter: ch, para: pa });
              setSel(ch);
              flush();
              toast(`Previewing ¶${pa + 1}…`);
              start(id, "Book preview", () => bookApi.create({ ...request(true), preview_text: rawText }));
            }}
            onRegenPara={regenPara}
            onChangeStory={(next) => changeStory(next)}
            onSelectChapter={(idx) => { setSel(idx); setFocusedBlock(null); setDragSelection(""); }}
            onTagDropped={(tag, label) => toast(tag.startsWith("{pause") ? `A ${label} added.` : `“${label}” added.`)}
          />
          <div className="editor-foot">
            <input ref={importInput} type="file" hidden accept={BOOK_FILES}
              onChange={(e) => { const f = e.target.files?.[0]; if (f) importing.run(() => importFile(f)); e.target.value = ""; }} />
            <button className="btn sm ghost" disabled={importing.starting} onClick={() => importInput.current?.click()} title="Put a book file's text into this audiobook">
              {importing.starting ? <Starting label="Importing…" size={13} /> : <><FileUp size={13} />Import</>}</button>
            {b.story && <button className="btn sm ghost" onClick={() => { setFindOpen((v) => !v); setTimeout(() => findInput.current?.focus(), 0); }} title="Find and replace (⌘F)"><Search size={13} />Find</button>}
            {b.story && <button className="btn sm ghost" onClick={() => setClearing(true)} title="Remove all of this book's text"><Eraser size={13} />Clear</button>}
            {b.original && b.original !== b.story && (
              <button className="btn sm ghost" onClick={() => changeStory(b.original, "")} title="Put the story back as it was before the last change"><Undo2 size={13} />Undo</button>
            )}
            <span className="spacer" />
            {helper.task?.state === "running" && <span className="small muted status-line">{helper.task.status.split("\n")[0]}</span>}
            {helper.running && <StopButton key={helper.task?.id} onStop={helper.stop} stopping={helper.task?.stopping} />}
          </div>
        </div>
        {removing && (
          <ConfirmDialog title={`Remove “${removing.title}”?`} confirm="Remove chapter" danger onCancel={() => setRemoving(null)}
            onConfirm={async () => {
              const ch = removing, story = b.story;
              setRemoving(null);
              try {
                const r = await bookApi.removeChapter(story, ch.index, ch.title, b.quotes_only);
                changeStory(r.story, story); setSel(null); toast(`Removed “${ch.title}”. Undo brings it back.`);
              } catch (x) { toast((x as Error).message, true); }
            }}>
            Its text goes from the story. Audio already made for it stays in the book's folder, and Undo brings the text back.
          </ConfirmDialog>
        )}
        {clearing && (
          <ConfirmDialog title="Clear all the text in this book?" confirm="Clear all text" danger onCancel={() => setClearing(false)}
            onConfirm={() => { changeStory("", b.story); setClearing(false); }}>
            Every chapter's text goes ({b.story.length.toLocaleString()} characters). The cast, settings and anything already
            made stay. Undo brings the text back.
          </ConfirmDialog>
        )}
        {checkingLines && b.folder && (
          <LineCheckDialog folder={b.folder} info={lineCheck.info} request={() => request(false)} busy={making} checking={lineCheck.checking}
            setInfo={(i) => { lineCheck.setInfo(i); refreshChapters(); }} onClose={() => setCheckingLines(false)} />
        )}
        {showingQa && b.folder && (
          <ImmersionQaDialog
            folder={b.folder}
            story={b.story}
            directorWhere={String(local.director_where || "mac")}
            directorModels={(local.director_models || {}) as Record<string, string>}
            request={() => request(false)}
            busy={making || helper.running}
            onJump={(ch, para, start) => {
              setSel(ch);
              setFocusedBlock({ chapter: ch, para });
              setScrollSignal((n) => n + 1);
              const c = shown[ch];
              if (c?.file) playChapter(c, start ?? 0, true);
            }}
            onRegen={regenPara}
            onClose={() => setShowingQa(false)}
          />
        )}
        <Player track={track} onSave={save} onTime={(time, playing) => { playingRef.current = playing; setPlayPos({ time, playing }); }} empty={shown.some((c) => c.file) ? "Pick a chapter to play" : "Nothing made yet: Preview or Create audiobook"} />
      </main>
    </>
  );
}

function ChapterIcon({ state, fraction }: { state: Chapter["state"]; fraction: number | null }) {
  if (state === "done") return <CheckCircle2 size={15} color="var(--ok)" aria-label="Made" />;
  if (state === "making") {
    const p = Math.max(0.04, fraction ?? 0.04), r = 6, c = 2 * Math.PI * r;
    return (
      <svg width="15" height="15" viewBox="0 0 16 16" aria-label="Being made" className="ring">
        <circle cx="8" cy="8" r={r} fill="none" stroke="var(--border-strong)" strokeWidth="2.2" />
        <circle cx="8" cy="8" r={r} fill="none" stroke="var(--accent)" strokeWidth="2.2" strokeDasharray={`${c * p} ${c}`} transform="rotate(-90 8 8)" strokeLinecap="round" />
      </svg>
    );
  }
  if (state === "waiting") return <Clock size={15} className="faint" aria-label="Waiting" />;
  if (state === "edited") return <PencilLine size={15} color="var(--warn)" aria-label="Edited since it was made" />;
  if (state === "gaps" || state === "partial") return <AlertTriangle size={15} color="var(--warn)" aria-label={state === "gaps" ? "Some lines were left out" : "Stopped part-way"} />;
  return <Circle size={15} className="faint" aria-label="Not made" />;
}
