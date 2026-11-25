import { useEffect, useRef, useState } from "react";
import {
  AudioLines, BookOpen, Check, Copy, Download, FileAudio, FileText,
  Loader2, Mic, RotateCcw, Trash2, Upload, Wand2, X,
} from "lucide-react";
import {
  fileUrl, projectApi, sttApi, useLoad, useTask,
  type SttModel, type SttTranscript,
} from "../api";
import { ConfirmDialog, Field, Switch, TaskProgress, clock, useToast } from "../components/ui";
import { saveFile } from "../native";

type Form = { model: string; language: string; timestamps: boolean; mic: number | null };
type AudioItem = { path: string; name: string };
const DEFAULTS: Form = { model: "whisper-large-v3-turbo", language: "auto", timestamps: true, mic: null };
const FORM_KEY = "stt-form";
const QUEUE_KEY = "stt-queue";
const ACCEPT = ".wav,.mp3,.m4a,.flac,.ogg,.webm,.mp4,.mov,.aac,.opus,audio/*,video/*";

function loadForm(): Form {
  try { return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(FORM_KEY) || "{}") }; } catch { return DEFAULTS; }
}

function loadQueueState(): { audio: AudioItem | null; queue: AudioItem[] } {
  try {
    const raw = JSON.parse(sessionStorage.getItem(QUEUE_KEY) || "{}");
    const audio = raw?.audio?.path && raw?.audio?.name ? (raw.audio as AudioItem) : null;
    const queue = Array.isArray(raw?.queue) ? raw.queue.filter((x: any) => x?.path && x?.name) : [];
    return { audio, queue };
  } catch {
    return { audio: null, queue: [] };
  }
}

function saveQueueState(audio: AudioItem | null, queue: AudioItem[]) {
  try {
    if (!audio && !queue.length) sessionStorage.removeItem(QUEUE_KEY);
    else sessionStorage.setItem(QUEUE_KEY, JSON.stringify({ audio, queue }));
  } catch { /* private window */ }
}

function fmtTime(sec: number): string {
  const s = Math.max(0, Math.floor(sec));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}` : `${m}:${String(r).padStart(2, "0")}`;
}

export function Transcribe({
  onSendToSpeech,
  onOpenBook,
}: {
  onSendToSpeech: (text: string) => void;
  onOpenBook: (id: string) => void;
}) {
  const toast = useToast();
  const [form, setForm] = useState<Form>(loadForm);
  const set = <K extends keyof Form>(k: K, v: Form[K]) => setForm((f) => ({ ...f, [k]: v }));
  useEffect(() => { try { localStorage.setItem(FORM_KEY, JSON.stringify(form)); } catch { /* private window */ } }, [form]);

  const meta = useLoad(sttApi.models);
  const mics = useLoad(sttApi.mics);
  const history = useLoad(sttApi.history);

  const [initialQ] = useState(loadQueueState);
  const [audio, setAudioState] = useState<AudioItem | null>(initialQ.audio);
  const [queue, setQueueState] = useState<AudioItem[]>(initialQ.queue);
  const audioRef = useRef<AudioItem | null>(audio);
  const queueRef = useRef<AudioItem[]>(queue);
  audioRef.current = audio;
  queueRef.current = queue;

  const setAudio = (next: AudioItem | null) => {
    audioRef.current = next;
    setAudioState(next);
    saveQueueState(next, queueRef.current);
  };
  const updateQueue = (updater: (prev: AudioItem[]) => AudioItem[]) => {
    const next = updater(queueRef.current);
    queueRef.current = next;
    setQueueState(next);
    saveQueueState(audioRef.current, next);
  };

  const [uploading, setUploading] = useState(false);
  const uploadingRef = useRef(false);
  const [dragging, setDragging] = useState(false);
  const [audioErrorPath, setAudioErrorPath] = useState<string | null>(null);
  const [active, setActive] = useState<SttTranscript | null>(null);
  const [view, setView] = useState<"plain" | "timestamps">("plain");
  const [copied, setCopied] = useState(false);
  const copyTimerRef = useRef<number | null>(null);
  const [creatingBook, setCreatingBook] = useState(false);
  const [confirmDeleteModel, setConfirmDeleteModel] = useState<SttModel | null>(null);
  const [deletingModel, setDeletingModel] = useState<string | null>(null);

  // Native macOS microphone recording state (backed by narrator/stt.py sounddevice stream)
  const [recording, setRecording] = useState(false);
  const [recSec, setRecSec] = useState(0);
  const [recLevel, setRecLevel] = useState(0);
  const [recBusy, setRecBusy] = useState(false);
  const recBusyRef = useRef(false);

  const fileIn = useRef<HTMLInputElement>(null);
  const activeAudioEl = useRef<HTMLAudioElement>(null);
  const seekAbortRef = useRef<AbortController | null>(null);
  const formRef = useRef(form);
  formRef.current = form;
  const models: SttModel[] = meta.data?.models || [];
  const selectedModel = models.find((m) => m.id === form.model);
  const selectedModelRef = useRef<SttModel | undefined>(selectedModel);
  selectedModelRef.current = selectedModel;
  const lastHandledTaskRef = useRef<string | null>(null);
  const startRef = useRef<(begin: () => Promise<any>) => Promise<void>>(async () => {});

  const startItem = (item: AudioItem) => {
    const f = formRef.current;
    const allowTs = selectedModelRef.current?.timestamps ?? true;
    startRef.current(() => sttApi.transcribe({
      path: item.path,
      model: f.model,
      language: f.language,
      timestamps: f.timestamps && allowTs,
      filename: item.name,
    }));
  };

  const { task, start, stop, running } = useTask((t) => {
    if (t.id && lastHandledTaskRef.current === t.id) return;
    if (t.id) lastHandledTaskRef.current = t.id;
    history.reload();
    meta.reload();
    if (t.state === "done" && t.result?.transcript) {
      setActive(t.result.transcript as SttTranscript);
    } else if (t.state === "failed") {
      const label = audioRef.current?.name ? `${audioRef.current.name}: ` : "";
      toast(`${label}${t.error || "Transcription failed."}`, true);
    }
    // Only auto-advance queue when the previous file succeeded (stop or failure pauses the queue)
    if (t.state === "done" && queueRef.current.length > 0) {
      const [next, ...rest] = queueRef.current;
      updateQueue(() => rest);
      setAudio(next);
      queueMicrotask(() => startItem(next));
    }
  }, "stt");
  startRef.current = start;
  const runningRef = useRef(running);
  runningRef.current = running;

  useEffect(() => () => {
    if (copyTimerRef.current) clearTimeout(copyTimerRef.current);
    seekAbortRef.current?.abort();
  }, []);

  useEffect(() => {
    if (copyTimerRef.current) {
      clearTimeout(copyTimerRef.current);
      copyTimerRef.current = null;
    }
    seekAbortRef.current?.abort();
    seekAbortRef.current = null;
    setCopied(false);
  }, [active?.id]);

  // Reconcile persisted model & mic selections against loaded backend lists
  useEffect(() => {
    const list = meta.data?.models;
    if (list?.length && !list.some((m) => m.id === form.model)) {
      set("model", list[0].id);
    }
  }, [meta.data]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const list = mics.data;
    if (list && form.mic != null && !list.some((d) => d.id === form.mic)) {
      set("mic", null);
    }
  }, [mics.data]); // eslint-disable-line react-hooks/exhaustive-deps

  // Re-sync if a mic recording was already in progress when switching tabs
  useEffect(() => {
    let live = true;
    sttApi.recordStatus().then((st) => {
      if (live && st.recording && !recBusyRef.current) {
        setRecording(true);
        setRecSec(st.seconds);
        setRecLevel(st.level);
      }
    }).catch(() => {});
    return () => { live = false; };
  }, []);

  // Chained poll for live mic level & elapsed timer while recording (no overlapping requests)
  useEffect(() => {
    if (!recording) return;
    let live = true;
    let timer: number | undefined;
    const poll = async () => {
      if (!live) return;
      if (!recBusyRef.current) {
        try {
          const st = await sttApi.recordStatus();
          if (!live) return;
          if (!recBusyRef.current) {
            setRecSec(st.seconds);
            setRecLevel(st.level);
            if (!st.recording) {
              setRecording(false);
              setRecLevel(0);
              if (st.error) toast(st.error, true);
              return;
            }
          }
        } catch { /* retry next tick */ }
      }
      if (live) timer = window.setTimeout(poll, 160);
    };
    timer = window.setTimeout(poll, 160);
    return () => { live = false; if (timer) clearTimeout(timer); };
  }, [recording]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleFiles = async (files: FileList | File[]) => {
    if (uploadingRef.current || recording || recBusyRef.current) return;
    const list = Array.from(files);
    if (!list.length) return;
    uploadingRef.current = true;
    setUploading(true);
    const uploaded: AudioItem[] = [];
    const failed: string[] = [];
    try {
      for (const file of list) {
        try {
          const r = await sttApi.upload(file);
          uploaded.push({ path: r.path, name: file.name });
        } catch {
          failed.push(file.name);
        }
      }
      if (failed.length) {
        toast(`Couldn't upload ${failed.slice(0, 2).join(", ")}${failed.length > 2 ? ` (+${failed.length - 2} more)` : ""}.`, true);
      }
      if (!uploaded.length) return;
      setAudioErrorPath(null);
      if (runningRef.current || queueRef.current.length > 0) {
        updateQueue((q) => [...q, ...uploaded]);
        toast(`Added ${uploaded.length} file${uploaded.length === 1 ? "" : "s"} to the queue.`);
      } else if (uploaded.length === 1) {
        setAudio(uploaded[0]);
        updateQueue(() => []);
        toast(`Loaded ${uploaded[0].name}.`);
      } else {
        setAudio(uploaded[0]);
        updateQueue(() => uploaded.slice(1));
        toast(`Loaded ${uploaded.length} files. Click Transcribe to process all ${uploaded.length} in order.`);
      }
    } finally {
      uploadingRef.current = false;
      setUploading(false);
    }
  };

  const toggleRecord = async () => {
    if (recBusyRef.current || runningRef.current || uploadingRef.current) return;
    recBusyRef.current = true;
    setRecBusy(true);
    try {
      if (recording) {
        const r = await sttApi.recordStop();
        setRecording(false);
        setRecLevel(0);
        setAudioErrorPath(null);
        updateQueue(() => []);
        setAudio({ path: r.path, name: r.name });
        toast(`Recorded ${clock(r.duration)} of audio.`);
      } else {
        await sttApi.recordStart(form.mic);
        setRecSec(0);
        setRecLevel(0);
        setRecording(true);
      }
    } catch (e) {
      if (recording) {
        try {
          const st = await sttApi.recordStatus();
          if (!st.recording) {
            setRecording(false);
            setRecLevel(0);
          }
        } catch {
          setRecording(false);
          setRecLevel(0);
        }
      }
      toast((e as Error).message, true);
    } finally {
      recBusyRef.current = false;
      setRecBusy(false);
    }
  };

  const go = () => {
    if (!audio) return toast("Drop or pick an audio or video file first.", true);
    if (runningRef.current || uploadingRef.current || recording || recBusyRef.current) return;
    startItem(audio);
  };

  const effectiveView = active?.segments.length ? view : "plain";

  const copyText = async () => {
    if (!active?.text?.trim()) return;
    const content = effectiveView === "timestamps" && active.segments.length
      ? active.segments.map((s) => `[${fmtTime(s.start)}] ${s.text}`).join("\n")
      : active.text;
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      if (copyTimerRef.current) clearTimeout(copyTimerRef.current);
      copyTimerRef.current = window.setTimeout(() => setCopied(false), 1600);
      toast("Copied transcript.");
    } catch {
      toast("Couldn't copy to clipboard.", true);
    }
  };

  const exportFile = async (ext: "txt" | "srt") => {
    if (!active) return;
    try {
      const { path } = await sttApi.exportFile(active.id, ext);
      const msg = await saveFile(path);
      toast(msg);
    } catch (e) {
      toast((e as Error).message, true);
    }
  };

  const sendToBook = async () => {
    if (!active?.text.trim() || creatingBook) return;
    setCreatingBook(true);
    try {
      const title = active.filename.replace(/\.[^.]+$/, "").replace(/[-_]+/g, " ").trim() || "Transcript";
      const proj = await projectApi.create({ title, story: active.text });
      onOpenBook(proj.id);
    } catch (e) {
      toast((e as Error).message, true);
    } finally {
      setCreatingBook(false);
    }
  };

  const removeHistory = async (id: string) => {
    try {
      await sttApi.remove(id);
      if (active?.id === id) setActive(null);
      history.reload();
      toast("Deleted transcript.");
    } catch (e) {
      toast((e as Error).message, true);
    }
  };

  const removeModel = async (m: SttModel) => {
    setConfirmDeleteModel(null);
    setDeletingModel(m.id);
    try {
      const r = await sttApi.deleteModel(m.id);
      meta.reload();
      toast(`Removed ${m.name}${r.freed_gb ? ` (${r.freed_gb.toFixed(1)} GB freed)` : ""}.`);
    } catch (e) {
      toast((e as Error).message, true);
    } finally {
      setDeletingModel(null);
    }
  };

  const seekActive = (sec: number) => {
    const el = activeAudioEl.current;
    if (!el) return;
    seekAbortRef.current?.abort();
    const target = Math.max(0, sec);
    if (el.readyState >= 1) {
      el.currentTime = target;
      el.play().catch(() => {});
    } else {
      const ac = new AbortController();
      seekAbortRef.current = ac;
      el.addEventListener("loadedmetadata", () => {
        el.currentTime = target;
        el.play().catch(() => {});
      }, { once: true, signal: ac.signal });
      if (el.networkState === 0 || el.networkState === 1) el.load();
    }
  };

  const languages = meta.data?.languages || [{ id: "auto", label: "Auto-detect" }];
  const micList = mics.data || [];
  const words = active?.text ? active.text.trim().split(/\s+/).filter(Boolean).length : 0;
  const hasText = Boolean(active?.text?.trim());
  const canSeek = Boolean(active?.audio_exists && active?.audio_path);

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Speech to Text</h1>
          <p>Transcribe audio or video locally on this Mac with Whisper Large or Qwen3-ASR.</p>
        </div>
      </div>

      <div className="split">
        <div className="stack">
          <div className="card card-pad stack">
            <input
              ref={fileIn}
              type="file"
              hidden
              multiple
              accept={ACCEPT}
              onChange={(e) => {
                if (e.target.files?.length) handleFiles(e.target.files);
                e.target.value = "";
              }}
            />

            <div
              className={`drop stt-drop ${dragging ? "over" : ""}`}
              onClick={() => !uploading && !recording && !recBusy && fileIn.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                if (!uploading && !recording && !recBusy) setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragging(false);
                if (!uploading && !recording && !recBusy && e.dataTransfer.files?.length) handleFiles(e.dataTransfer.files);
              }}
            >
              {uploading ? (
                <>
                  <Loader2 size={20} className="spin" />
                  <b>Uploading audio…</b>
                </>
              ) : (
                <>
                  <Upload size={20} />
                  <b>Drop audio or video file(s) here, or click to browse</b>
                  <span className="small faint">WAV, MP3, M4A, FLAC, OGG, WebM, MP4, MOV · supports multiple files</span>
                </>
              )}
            </div>

            <div className="row wrap">
              <button
                className={`btn ${recording ? "danger" : ""}`}
                disabled={uploading || running || recBusy}
                onClick={toggleRecord}
              >
                {recording ? (
                  <>
                    <span className="stt-rec-dot" />
                    Stop recording · {fmtTime(recSec)}
                  </>
                ) : (
                  <><Mic size={15} />Record mic</>
                )}
              </button>

              {recording && (
                <div className="stt-meter grow" aria-label="Microphone input level">
                  <i style={{ width: `${Math.max(4, Math.min(100, Math.round(recLevel * 100)))}%` }} />
                </div>
              )}

              {!recording && audio && (
                <div className="stt-file-pill grow">
                  <FileAudio size={14} />
                  <span className="stt-file-name" title={audio.name}>{audio.name}</span>
                  {queue.length > 0 && <span className="chip accent">+{queue.length} queued</span>}
                  <button
                    className="btn icon sm ghost"
                    disabled={running || uploading || recording}
                    onClick={() => { setAudio(null); updateQueue(() => []); }}
                    aria-label="Remove file"
                    title={running ? "Stop transcription first" : "Remove file"}
                  >
                    <X size={13} />
                  </button>
                </div>
              )}

              <span className="spacer" />
              <button
                className="btn primary"
                disabled={!audio || running || uploading || recording || recBusy}
                onClick={go}
              >
                <Wand2 size={15} />{queue.length > 0 ? `Transcribe (${queue.length + 1})` : "Transcribe"}
              </button>
            </div>

            {!recording && queue.length > 0 && (
              <div className="stt-queue">
                <div className="row" style={{ gap: 8 }}>
                  <span className="small muted">Up next ({queue.length})</span>
                  <span className="spacer" />
                  <button className="link-btn small" onClick={() => updateQueue(() => [])}>Clear queue</button>
                </div>
                {queue.map((item, idx) => (
                  <div key={`${item.path}-${idx}`} className="stt-queue-item">
                    <FileAudio size={12} className="muted" />
                    <span className="stt-file-name small grow" title={item.name}>{item.name}</span>
                    <button
                      className="btn icon sm ghost"
                      onClick={() => updateQueue((q) => q.filter((_, i) => i !== idx))}
                      aria-label={`Remove ${item.name}`}
                      title="Remove from queue"
                    >
                      <X size={12} />
                    </button>
                  </div>
                ))}
              </div>
            )}

            {!recording && audio && (
              audioErrorPath === audio.path ? (
                <div className="small muted">Audio preview isn't available for this container format — transcription still works.</div>
              ) : (
                <audio
                  key={audio.path}
                  src={fileUrl(audio.path)}
                  controls
                  onError={() => setAudioErrorPath(audio.path)}
                />
              )
            )}
          </div>

          {task && (
            <div className="card card-pad">
              <TaskProgress task={task} onStop={stop} />
            </div>
          )}

          {active && (
            <div className="card stack" style={{ gap: 0 }}>
              <div className="row wrap card-pad" style={{ paddingBottom: 12, borderBottom: "1px solid var(--border)" }}>
                <div className="grow" style={{ minWidth: 0 }}>
                  <div className="row wrap" style={{ gap: 6 }}>
                    <strong className="stt-file-name">{active.filename}</strong>
                    <span className="chip accent">{active.model_name}</span>
                    {active.language && <span className="chip">{active.language}</span>}
                  </div>
                  <div className="small muted" style={{ marginTop: 2 }}>
                    {words.toLocaleString()} words
                    {active.duration != null && ` · ${clock(active.duration)} audio`}
                    {active.elapsed > 0 && ` · transcribed in ${active.elapsed.toFixed(1)}s`}
                  </div>
                </div>
                {active.segments.length > 0 && (
                  <div className="segmented">
                    <button className={effectiveView === "plain" ? "on" : ""} onClick={() => setView("plain")}>Plain text</button>
                    <button className={effectiveView === "timestamps" ? "on" : ""} onClick={() => setView("timestamps")}>Timestamps</button>
                  </div>
                )}
              </div>

              {canSeek && (
                <div className="stt-active-audio">
                  <audio ref={activeAudioEl} key={active.audio_path} src={fileUrl(active.audio_path)} controls />
                </div>
              )}

              <div className="stt-transcript">
                {effectiveView === "timestamps" && active.segments.length > 0 ? (
                  <div className="stt-segments">
                    {active.segments.map((seg, i) => (
                      <div key={i} className="stt-seg">
                        {canSeek ? (
                          <button
                            type="button"
                            className="stt-seg-time stt-seg-seek tnum"
                            onClick={() => seekActive(seg.start)}
                            title={`Jump to ${fmtTime(seg.start)}`}
                          >
                            {fmtTime(seg.start)}
                          </button>
                        ) : (
                          <span className="stt-seg-time tnum">{fmtTime(seg.start)}</span>
                        )}
                        <span className="stt-seg-text">{seg.text}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="stt-plain">{active.text || <span className="muted">(No speech detected)</span>}</div>
                )}
              </div>

              <div className="editor-foot">
                <button className="btn sm" disabled={!hasText} onClick={copyText}>
                  {copied ? <Check size={13} /> : <Copy size={13} />}{copied ? "Copied" : "Copy"}
                </button>
                <button className="btn sm ghost" disabled={!hasText} onClick={() => exportFile("txt")}>
                  <Download size={13} />TXT
                </button>
                {active.segments.length > 0 && (
                  <button className="btn sm ghost" disabled={!hasText} onClick={() => exportFile("srt")}>
                    <Download size={13} />SRT
                  </button>
                )}
                <span className="spacer" />
                <button className="btn sm ghost" disabled={!hasText} onClick={() => onSendToSpeech(active.text)}>
                  <AudioLines size={13} />Text to Speech
                </button>
                <button className="btn sm" disabled={creatingBook || !hasText} onClick={sendToBook}>
                  {creatingBook ? <Loader2 size={13} className="spin" /> : <BookOpen size={13} />}New Audiobook
                </button>
              </div>
            </div>
          )}

          <div className="card">
            <div className="row card-pad" style={{ paddingBottom: 8 }}>
              <h3 className="card-title" style={{ margin: 0 }}>Recents</h3>
              <span className="spacer" />
              <button className="btn sm ghost" onClick={history.reload}><RotateCcw size={13} />Refresh</button>
            </div>
            <div className="list">
              {(history.data || []).slice(0, 15).map((item) => (
                <div
                  key={item.id}
                  className={`list-item stt-recent ${active?.id === item.id ? "sel" : ""}`}
                >
                  <button
                    className="stt-recent-btn grow"
                    onClick={() => {
                      setActive(item);
                      if (!item.segments.length) setView("plain");
                    }}
                  >
                    <div className="row" style={{ gap: 8 }}>
                      <FileText size={14} className="muted" style={{ flex: "none" }} />
                      <span className="title">{item.filename}</span>
                      <span className="chip">{item.model_name}</span>
                    </div>
                    <div className="small muted stt-recent-preview">
                      {item.text.slice(0, 110)}{item.text.length > 110 ? "…" : ""}
                    </div>
                    <div className="small faint">
                      {new Date(item.created).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                      {item.duration != null && ` · ${clock(item.duration)}`}
                      {item.language && ` · ${item.language}`}
                    </div>
                  </button>
                  <button
                    className="btn icon sm ghost"
                    onClick={() => removeHistory(item.id)}
                    title="Delete transcript"
                    aria-label="Delete transcript"
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              ))}
              {history.data && !history.data.length && (
                <div className="empty small">Nothing yet. Your transcripts show up here.</div>
              )}
            </div>
          </div>
        </div>

        <div className="card card-pad stack">
          <Field label="Model">
            <div className="stack" style={{ gap: 8 }}>
              {models.map((m) => {
                const on = form.model === m.id;
                return (
                  <div key={m.id} className={`stt-model-card ${on ? "on" : ""}`} onClick={() => set("model", m.id)} role="button" tabIndex={0}
                    onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); set("model", m.id); } }}>
                    <div className="row" style={{ gap: 6 }}>
                      <strong className="grow">{m.name}</strong>
                      <span className={`chip ${m.downloaded ? "ok" : ""}`}>
                        {m.downloaded ? "Downloaded" : `~${m.size_gb.toFixed(1)} GB`}
                      </span>
                      {m.downloaded && !m.core && (
                        <button
                          type="button"
                          className="btn icon sm ghost"
                          disabled={deletingModel !== null || running}
                          title={`Delete ${m.name} weights (~${m.size_gb.toFixed(1)} GB)`}
                          aria-label={`Delete ${m.name}`}
                          onClick={(e) => { e.stopPropagation(); setConfirmDeleteModel(m); }}
                        >
                          {deletingModel === m.id ? <Loader2 size={12} className="spin" /> : <Trash2 size={12} />}
                        </button>
                      )}
                    </div>
                    <div className="small faint">{m.license} · {m.languages}</div>
                    <div className="small muted stt-model-desc">{m.summary}</div>
                  </div>
                );
              })}
              {!models.length && <div className="small muted">Loading models…</div>}
            </div>
          </Field>

          {selectedModel && !selectedModel.downloaded && (
            <div className="small muted" style={{ marginTop: -4 }}>
              Downloads once (~{selectedModel.size_gb.toFixed(1)} GB) on first transcription, then runs fully offline.
            </div>
          )}

          <Field label="Language">
            <select
              className="select"
              value={form.language}
              onChange={(e) => set("language", e.target.value)}
            >
              {languages.map((l) => (
                <option key={l.id} value={l.id}>{l.label}</option>
              ))}
            </select>
          </Field>

          {micList.length > 1 && (
            <Field label="Microphone">
              <select
                className="select"
                value={form.mic == null ? "" : String(form.mic)}
                onChange={(e) => set("mic", e.target.value === "" ? null : Number(e.target.value))}
              >
                <option value="">System default</option>
                {micList.map((d) => (
                  <option key={d.id} value={d.id}>{d.name}{d.default ? " (default)" : ""}</option>
                ))}
              </select>
            </Field>
          )}

          {selectedModel?.timestamps !== false && (
            <Switch
              label="Segment timestamps"
              hint="Include start and end times for timestamped view and SRT subtitles."
              checked={form.timestamps}
              onChange={(v) => set("timestamps", v)}
            />
          )}
        </div>
      </div>

      {confirmDeleteModel && (
        <ConfirmDialog
          title={`Delete ${confirmDeleteModel.name}?`}
          confirm="Delete weights"
          danger
          onCancel={() => setConfirmDeleteModel(null)}
          onConfirm={() => removeModel(confirmDeleteModel)}
        >
          Frees about {confirmDeleteModel.size_gb.toFixed(1)} GB of disk space. You can re-download it any time.
        </ConfirmDialog>
      )}
    </div>
  );
}
