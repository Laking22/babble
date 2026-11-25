// The local API (narrator/api.py). Served by the app on this Mac; long work runs as tasks we follow live.
import { useCallback, useEffect, useRef, useState } from "react";

export const API_BASE: string = (() => {
  const env = import.meta.env.VITE_API_BASE as string | undefined;
  if (env) return env;
  // served by the app itself (http://127.0.0.1:7895/app/): the same server
  if (location.protocol.startsWith("http") && location.pathname.startsWith("/app")) return "";
  return "http://127.0.0.1:7895"; // the desktop shell and the dev server
})();

export type Engine = {
  id: string; name: string; short: string; tier: string; license: string; min_ram_gb: number;
  available: boolean; why: string; note: string; locked: boolean;
  group?: "light" | "standard" | "high"; downloaded?: boolean;
};
export type Voice = {
  value: string; name: string; label: string; gender: string; description: string; engine: string;
  tags: string[]; available: boolean; why: string;
  cloud?: boolean;  // this Mac can't run it, but Create in the cloud can make its lines (Higgs on 8 GB)
};
/** Another model's copy of a saved voice (VibeVoice, Higgs): voice lists show a saved voice once, as Qwen's. */
export const isCopy = (v: Voice) => v.tags.includes("saved") && v.engine !== "qwen";
export type Sound = {
  id: string; name: string; kind: "effect" | "ambience" | "music"; tags: string[]; file: string; path: string;
  duration: number; license?: string; source?: string; credit?: string; description?: string;
  book?: string; made_in?: string; // kept in one audiobook (its project id); made while working on one
};
export type HistoryItem = { path: string; kind: string; partial: boolean; time: string; seconds: number | null };
export type TaskState = "running" | "done" | "failed" | "stopped";
export type Task = {
  id: string; kind: string; title: string; state: TaskState; status: string; progress: number | null;
  result: Record<string, any>; error: string | null;
  eta?: number | null;  // seconds left, from how fast its progress has moved (as of etaAt)
  updated?: number;     // when it last had news (unix seconds): a long quiet stretch is shown
  stopping?: boolean;   // Stop was pressed and it's winding down
  etaAt?: number;       // (this page) when eta was heard, in ms
};

export class ApiError extends Error {}

// The engine answers only the app (narrator/guard.py): every call carries this run's key. The desktop app gets it
// from its native side; a browser tab from the launcher's link (#token=…), kept for the tab's life.
let TOKEN = "";
type TauriInvoke = { core: { invoke: <T>(cmd: string) => Promise<T> } };
const keyWatchers = new Set<() => void>();
/** Called when the engine started again with a new key (the audio and image links on screen carry the old one). */
export function onNewKey(f: () => void): () => void {
  keyWatchers.add(f);
  return () => { keyWatchers.delete(f); };
}
export async function loadToken(fresh = false): Promise<string> {
  if (TOKEN && !fresh) return TOKEN;
  const old = TOKEN;
  try { return await readToken(); } finally {
    if (old && TOKEN && TOKEN !== old && TOKEN !== announced) { announced = TOKEN; keyWatchers.forEach((f) => f()); }
  }
}
let announced = "";  // (several calls can find the new key at once: told once)
async function readToken(): Promise<string> {
  const tauri = (window as unknown as { __TAURI__?: TauriInvoke }).__TAURI__;
  if (tauri) {
    try { TOKEN = await tauri.core.invoke<string>("engine_token"); } catch { /* the engine is still starting */ }
    return TOKEN;
  }
  const m = location.hash.match(/token=([A-Za-z0-9_-]+)/);
  if (m) {
    TOKEN = m[1];
    try { sessionStorage.setItem("babble-token", TOKEN); } catch { /* private window */ }
    history.replaceState(null, "", location.pathname + location.search);
  } else if (!TOKEN) {
    try { TOKEN = sessionStorage.getItem("babble-token") || ""; } catch { /* private window */ }
  }
  return TOKEN;
}
/** Before the first page is drawn (its audio and image links carry the key): a key the engine accepts. When the app
 *  opens with the engine, the file can still hold the last run's key for a moment; this waits for the new one. */
export async function engineReady(): Promise<void> {
  for (let i = 0; i < 12; i++) {
    await loadToken(true);
    try {
      const signal = typeof AbortSignal !== "undefined" && "timeout" in AbortSignal ? AbortSignal.timeout(1200) : undefined;
      const r = await fetch(API_BASE + "/api/settings", { headers: { "X-Babble-Token": TOKEN }, signal });
      if (r.status !== 401) return;
    } catch { /* not listening yet */ }
    await new Promise((ok) => setTimeout(ok, 300));
  }
}
const authed = (url: string) => `${url}${url.includes("?") ? "&" : "?"}auth=${encodeURIComponent(TOKEN)}`;

/** fetch with this run's key; a refusal (the engine restarted with a new key) gets the new key and tries once more. */
async function call(path: string, init?: RequestInit): Promise<Response> {
  await loadToken();
  const go = () => fetch(API_BASE + path, { ...init, headers: { ...(init?.headers || {}), "X-Babble-Token": TOKEN } });
  let r = await go();
  if (r.status === 401) {
    await loadToken(true);
    r = await go();
  }
  return r;
}

/** Requests that start or change something (POST, PUT, DELETE) sent again the same within this long (a double
 *  click on Create, Send, Connect or Stop) get the first one's answer instead of starting a second run. Only when
 *  nothing else went to that address in between: on, off, on again sends all three. */
const SAME_WITHIN_MS = 1000;
const recent = new Map<string, { at: number; body: string; answer: Promise<unknown> }>();  // method + path: the last one sent

function req<T>(path: string, init?: RequestInit): Promise<T> {
  const method = (init?.method || "GET").toUpperCase();
  const body = init?.body ?? "";
  if (method === "GET" || typeof body !== "string") return send<T>(path, init);
  const key = `${method} ${path}`;
  const now = Date.now();
  for (const [k, v] of recent) if (now - v.at > SAME_WITHIN_MS) recent.delete(k);
  const last = recent.get(key);
  if (last && last.body === body) return last.answer as Promise<T>;
  const answer = send<T>(path, init);
  recent.set(key, { at: now, body, answer });
  const drop = () => { if (recent.get(key)?.answer === answer) recent.delete(key); };
  answer.then(drop, drop);
  return answer;
}

async function send<T>(path: string, init?: RequestInit): Promise<T> {
  let r: Response;
  try {
    r = await call(path, { ...init, headers: { "Content-Type": "application/json", ...(init?.headers || {}) } });
  } catch {
    throw new ApiError("Can't reach Babble. Is it running?");
  }
  if (!r.ok) {
    let msg = r.statusText;
    try {
      const j = await r.json();
      msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j);
    } catch { /* not JSON */ }
    throw new ApiError(msg);
  }
  return r.json();
}

const post = <T>(path: string, body: unknown) => req<T>(path, { method: "POST", body: JSON.stringify(body) });
const put = <T>(path: string, body: unknown) => req<T>(path, { method: "PUT", body: JSON.stringify(body) });
const del = <T>(path: string) => req<T>(path, { method: "DELETE" });
const q = (o: Record<string, string>) => new URLSearchParams(o).toString();

/** Memory and disk Babble is using, in bytes (the sidebar's meters). */
export type Usage = {
  ram: { used: number; total: number };
  storage: { used: number; data: number; models: number; free: number; total: number } | null;
};
export type StorageBreakdown = {
  books: number;
  orphans: { bytes: number; count: number; folders: string[] };
  previews: { bytes: number; count: number };
  uploads: { bytes: number; count: number };
  history: { bytes: number; count: number };
  sounds: number;
  models: number;
  free: number;
  total: number;
};
export const api = {
  health: () => req<{ ok: boolean; app: string; api: number; ram_gb: number }>("/api/health"),
  checks: () => req<{ problems: { area: string; missing: string }[] }>("/api/checks"),
  usage: () => req<Usage>("/api/usage"),
  storage: () => req<StorageBreakdown>("/api/storage"),
  cleanStorage: (target: "previews" | "uploads" | "orphans" | "history") =>
    post<{ target: string; removed: number; freed: number }>("/api/storage/clean", { target }),
  engines: () => req<Engine[]>("/api/engines"),
  voices: () => req<Voice[]>("/api/voices"),
  settings: () => req<Record<string, any>>("/api/settings"),
  saveSettings: (changes: Record<string, any>) => post<Record<string, any>>("/api/settings", changes),
  history: () => req<HistoryItem[]>("/api/history"),
  deleteHistory: (path: string) => del<{ deleted: string }>(`/api/history?${q({ path })}`),
  sounds: () => req<Sound[]>("/api/sounds"),
  /** Words every book and Text to Speech say another way. */
  pronunciations: () => req<Pronunciation[]>("/api/pronunciations"),
  savePronunciations: (rows: Pronunciation[]) => put<Pronunciation[]>("/api/pronunciations", { rows }),
  deleteSound: (id: string) => req<{ deleted: string }>(`/api/sounds/${encodeURIComponent(id)}`, { method: "DELETE" }),
  deleteSounds: (ids: string[]) => post<{ deleted: number }>("/api/sounds/delete", { ids }),
  speech: (body: SpeechRequest) => post<Task>("/api/speech", body),
  makeSound: (body: { description: string; kind: string; quality?: string; seconds?: number; book?: string; scope?: Scope }) => post<Task>("/api/sounds/make", body),
  soundScope: (id: string, book: string | null) => post<Sound>(`/api/sounds/${encodeURIComponent(id)}/scope`, { book }),
  tasks: () => req<Task[]>("/api/tasks"),
  task: (id: string) => req<Task>(`/api/tasks/${id}`),
  stopTask: (id: string) => post<Task>(`/api/tasks/${id}/stop`, {}),
};

export type SpeechRequest = {
  text: string; voice: string; style?: string; speed?: number; stability?: number; language?: string;
  seed?: number; exact?: boolean; max_chars?: number; pause?: number; batch?: number; parallel?: boolean; preview?: boolean;
};

export type SttModel = {
  id: string; name: string; repo: string; family: "whisper" | "qwen3_asr";
  size_gb: number; license: string; languages: string; summary: string;
  timestamps: boolean; downloaded: boolean; core?: boolean;
};
export type SttLanguage = { id: string; label: string };
export type SttSegment = { start: number; end: number; text: string };
export type SttMic = { id: number; name: string; default: boolean };
export type SttRecordStatus = { recording: boolean; seconds: number; level: number; error: string | null };
export type SttTranscript = {
  id: string; filename: string; audio_path: string; audio_exists?: boolean;
  model: string; model_name: string; language: string;
  duration: number | null; elapsed: number; text: string;
  segments: SttSegment[]; created: string;
};
export const sttApi = {
  models: () => req<{ models: SttModel[]; languages: SttLanguage[] }>("/api/stt/models"),
  deleteModel: (id: string) => del<{ id: string; freed_gb: number }>(`/api/stt/models/${encodeURIComponent(id)}`),
  mics: () => req<SttMic[]>("/api/stt/mics"),
  recordStart: (device?: number | null) => post<SttRecordStatus>("/api/stt/record/start", { device: device ?? null }),
  recordStatus: () => req<SttRecordStatus>("/api/stt/record/status"),
  recordStop: () => post<{ path: string; name: string; duration: number }>("/api/stt/record/stop", {}),
  history: () => req<SttTranscript[]>("/api/stt/history"),
  remove: (id: string) => del<{ deleted: string }>(`/api/stt/history/${encodeURIComponent(id)}`),
  exportFile: (id: string, fmt: "txt" | "srt") => post<{ path: string }>(`/api/stt/history/${encodeURIComponent(id)}/export`, { fmt }),
  transcribe: (body: { path: string; model: string; language: string; timestamps: boolean; filename?: string }) =>
    post<Task>("/api/stt/transcribe", body),
  upload: (file: File) => upload<{ path: string }>("/api/files/upload", file),
};

export type CastRow = { character: string; voice: string; style: string; speed: number; stability?: number | null };
export type Scope = "book" | "all";
export type BookOptions = { models: { label: string; value: string }[]; formats: string[]; languages: string[] };
export type Placement = {
  id: number; kind: "sfx" | "amb" | "music"; name: string; label: string; para: number; missing: boolean;
  sound_id: string | null; sound_path: string | null;
  /** the director's offer for a sound you don't have: the library sound that sounds most like it (CLAP) */
  match?: { id: string; name: string; tag_name: string; path: string; score: number };
  [k: string]: any;
};
export type BookRequest = {
  story: string; cast: CastRow[]; models?: string; quotes_only?: boolean; speed?: number; stability?: number;
  language?: string; seed?: number; batch?: number; parallel?: boolean; pronunciations?: Pronunciation[]; line_pause?: number; para_pause?: number; out_format?: string;
  title?: string; author?: string; cover?: string | null; sound?: boolean; bed?: number; intro?: string;
  fx_level?: number | null; amb_level?: number | null; music_level?: number | null;
  chapter_music?: ChapterMusic | null; intro_level?: number | null;
  automake?: boolean; preview?: boolean; preview_text?: string | null; preview_chapter?: number | null;
  folder?: string; name?: string; book?: string; new_sounds?: Scope;
  make_on?: MakeOnId; cloud_gpu?: string; cloud_live?: boolean; cloud_hours?: number | null; cloud_sounds?: boolean;
};
/** Each chapter's intro and outro (narrator/chapter_music.py): a library sound id each, or "" for none. */
export type ChapterMusicRow = { title: string; intro: string; outro: string; by?: "director" | "you" };
export type ChapterMusic = { chapters: ChapterMusicRow[] };
/** The director's chapter music, made on Add when "Make chapter intros and outros" is on. */
export type ChapterPlan = {
  theme: string; variants: Record<string, string>; outros: Record<string, string>;
  chapters: { title: string; variant: string; outro: string }[];
};
export type IntroChoice = { id: string; label: string; path: string; kind: string; duration: number };
export type FinishedBook = { title: string; folder: string; name: string; file: string | null; chapters: number };
export type ChapterState = "done" | "gaps" | "partial" | "edited" | "not made" | "making" | "waiting";
export type Chapter = {
  index: number; title: string; state: ChapterState; file: string | null; seconds: number | null;
  at?: number | null;  // where it starts in the story (a part of a long chapter: its own paragraphs)
};
/** Say `word` as `say` (the text keeps its spelling). */
export type Pronunciation = { word: string; say: string };
export type TrickyWord = { word: string; reason: "fantasy_name" | "homograph" | "proper_noun"; count: number; note: string };
export type StorySound = { type: "sfx" | "amb" | "music"; name: string; tag: string; sound: Sound | null };
/** A sound an audiobook can use, with the tag that plays exactly it (mine: kept in this book). */
export type PaletteSound = Sound & { tag: string; mine: boolean };

/** An audiobook on the shelf: its text, cast and settings, kept by the app. */
export type Project = {
  id: string; untitled: number; folder: string; color: string; created: number; updated: number;
  title: string; author: string; cover: string | null; story: string; original: string;
  cast: CastRow[]; models: string; quotes_only: boolean; out_format: string;
  speed: number; stability: number; language: string; seed: number; batch: number; parallel: boolean; line_pause: number; para_pause: number;
  pronunciations: Pronunciation[];
  sound: boolean; bed: number; intro: string; director_notes?: string;
  fx_level?: number | null; amb_level?: number | null; music_level?: number | null;
  chapter_music?: ChapterMusic | null; intro_level?: number | null; detect: "ai" | "quick"; new_sounds: Scope;
  // the book's own sound settings (null: the app-wide one from before they moved into each book)
  sfx_exact?: boolean | null; sfx_auto_make?: boolean | null; director_where?: string | null;
  director_models?: Record<string, string> | null; director_checks?: number | null; director_scenes?: string | null; sound_libraries?: boolean | null;
  director_intros?: boolean | null;
  make_on: MakeOnId; cloud_gpu: string; cloud_detect?: boolean; cloud_live?: boolean; cloud_hours?: number | null;
  /** with a cloud: its missing AI sounds are made there too (else on this Mac) */
  cloud_sounds?: boolean;
  file: string | null;
};
/** Before sending: how long the cloud would take and what it may cost, realistic low to worst case (the run's
 *  time limit). Kaggle is free (free: true), so only its GPU hours count. */
/** A book against its last cloud run: run_lines of its lines are here, `used` of them the book uses now; what
 *  changed since, and the settings to put back (book fields, and cast: {character: {stability, voice, style}}). */
export interface CloudMatch {
  run_lines?: number; used?: number;
  changed?: { what: string; then: unknown; now: unknown }[];
  restore?: Record<string, unknown> & { cast?: Record<string, Partial<{ stability: number | null; voice: string; style: string }>> };
}
export interface CloudEstimate {
  busy?: { state: string; sent?: number; back?: number; lines?: number; backend?: string } | null;  // already out there
  backend: string; lines: number; local: number; seconds: number; low_hours: number; high_hours: number;
  price: number | null; low_cost: number | null; high_cost: number | null; free: boolean; budget?: number; runs?: number; run_max?: number; cap?: number; gpu_note?: string;
  /** room on this Mac (bytes) for the book once it's back: what it needs, what's free */
  disk?: { need: number; free: number };
  /** every chapter is made already; why: why there's nothing to send; upload: trained voices that go up with it */
  made_all?: boolean; why?: string; upload?: { models: number; gb: number };
  /** sounds made in the cloud with the lines, or (mac_sounds) on this Mac */
  sounds?: number; mac_sounds?: number;
}

/** Check lines: a cloud line that sounds wrong (heard back as something else, cut short or said twice). */
export type CheckedLine = {
  key: string; chapters: number[]; text: string; voice: string; engine: string; path: string; seconds: number;
  heard: string; covered: number; set_aside?: number; remade?: boolean;
};
/** The last Check lines of a book (empty before the first), and how many cloud lines it has. */
export type LineCheck = {
  checked?: number; lines?: number; screened?: number; wrong?: CheckedLine[]; aside?: CheckedLine[]; cloud_lines?: number;
};
/** Two-stage Immersion QA issue & report (narrator/immersion_qa.py). */
export type QaIssue = {
  id: number; stage: 1 | 2; severity: "high" | "medium" | "low"; category: string;
  chapter: number; para: number; start?: number | null; can_regen: boolean;
  title: string; detail: string;
};
export type QaReport = {
  checked?: number; stage2?: boolean; rendered_chapters?: number; total_chapters?: number;
  issues?: QaIssue[];
};

/** A book whose lines are being made in the cloud (Create in the cloud). */
export type CloudJob = {
  state?: "sent" | "queued" | "running" | "downloading" | "ready" | "failed";
  backend?: string; gpu?: string; gpu_label?: string; max_cost?: number | null;
  url?: string; sent?: number; checked?: number; lines?: number; local?: number; seconds?: number; budget?: number;
  made?: number; speed?: number | null; stopped?: boolean; error?: string; note?: string; sound_count?: number; sounds_made?: number;
  awake?: boolean;  // this Mac is being kept awake until the lines are back
  progress?: number; eta?: number;  // lines back so far / all (0..1), and seconds left at that pace (lines brought back as they're made)
  elapsed?: number; limit?: number | null;  // seconds since it was sent, and its time limit
  live?: boolean; back?: number; live_error?: string;  // lines copied back as they're made: how many are here
  offline_since?: number;  // waiting on a lost connection since then (it carries on by itself once it's back)
  disk_full?: boolean;  // this Mac's disk is full: lines can't come back or be put together until there's room
  /** each voice's first lines from the cloud checked against that voice on this Mac (a wrong one stops the run) */
  voice_check?: { state: "checking" | "ok" | "wrong" | "unchecked"; voices: { label: string; state: string; why: string }[] } | null;
  stitch?: { id: string; status: string; progress: number | null; eta?: number | null; updated?: number };  // chapters put together as their lines come back
  // how putting it together here ended, when it didn't finish the book (it's then left for Finish)
  stitch_end?: "stopped" | "failed" | "partial"; stitch_error?: string;
};
export type ProjectSummary = {
  id: string; untitled: number; folder: string; color: string; created: number; updated: number;
  name: string; title: string; author: string; cover: string | null; words: number; chapters: number; cast: number;
  file: string | null; format: string;
  cloud?: { state: string; progress?: number; eta?: number; elapsed?: number; stitch?: string; stitch_progress?: number | null; stitch_eta?: number | null } | null;  // out in the cloud
};
export const projectName = (p: { title: string; untitled: number }) => p.title.trim() || `Untitled-${p.untitled || 1}`;
export type Imported = { text: string; title: string; author: string; cover: string | null; chapters: string[]; message: string };

async function upload<T>(path: string, file: File): Promise<T> {
  const form = new FormData();
  form.append("file", file);
  let r: Response;
  try { r = await call(path, { method: "POST", body: form }); }
  catch { throw new ApiError("Can't reach Babble. Is it running?"); }
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : msg; } catch { /* not JSON */ }
    throw new ApiError(msg);
  }
  return r.json();
}

/** A model offered by a Director AI service, with input/output pricing per million tokens. */
export type DirectorModel = {
  id: string; label: string;
  price_in?: number | null; price_out?: number | null;
  free?: boolean; free_tier?: boolean; price_label?: string;
};
/** An AI service the sound director can use with the user's own key (narrator/director.py). */
export type AiService = {
  id: string; name: string; keys: string[]; has_key: boolean; model: string; models: DirectorModel[] | null;
  cost: "paid" | "free tier" | "free models"; cost_note: string;
};
export type DirectorEstimate = {
  scenes: number; dollars: number | null; minutes: number; tokens?: number;
  low?: number | null; high?: number | null; cost?: string; basis?: string; free_tier?: boolean;  // (pricing.py)
  requests?: number;
};

/** Recorded sounds online (Freesound, Lots of Sounds) and sound packs (narrator/online_sounds.py, sfx.import_pack). */
export type SoundSource = { id: string; name: string; keys: string[]; has_key: boolean; cost: "free" | "paid"; cost_note: string };
export type OnlineSound = { source: string; id: string; name: string; tags: string[]; duration: number; credit: string; link: string; play: string };
export type SoundPack = { name: string; sounds: number; seconds: number; license: string };
/** Where the API keys are kept (narrator/vault.py): the Keychain when the app started the engine. */
export type KeychainStatus = { where: "keychain" | "file"; ready?: boolean; ok?: boolean; why?: string; keys?: number };
export const keychainApi = { get: () => req<KeychainStatus>("/api/keychain") };

/** Spending (narrator/spend.py): what paid services cost, the day limit, cloud jobs out, stop all paid work. */
export type Spending = {
  today: number; month: number; limit: number;
  agent_limit?: number; agent_today?: number;  // AI agents' own day limit (narrator/agents.py) and what they spent
  services: { service: string; name: string; today: number; month: number; requests_today: number }[];
  recent: { time: number; service: string; what: string; model: string; dollars: number; exact: boolean }[];
  alerts: { time: number; why: string; title?: string }[];
  jobs: { run: string; what: string; name: string; sent: number; price: number | null }[];
};
/** One paid job in Settings → Spending's list (spend.history): a task's charges together. */
export type SpendRow = {
  time: number; started: number; title: string; state: string; service: string; kind: string; model: string;
  dollars: number; exact: boolean; book: string; requests: number; gpu_hours: number; agent?: boolean;
};
export const spendingApi = {
  get: () => req<Spending>("/api/spending"),
  history: () => req<SpendRow[]>("/api/spending/history"),
  setLimit: (day: number) => post<Spending>("/api/spending/limit", { day }),
  setAgentLimit: (agents: number) => post<Spending>("/api/spending/limit", { agents }),
  stopAll: () => post<Spending & { stopped: string[] }>("/api/spending/stop-all", {}),
};
/** AI agents using Babble through MCP (narrator/agents.py): the switch and what an AI app needs to start it. */
export type Agents = { on: boolean; limit: number; setup: { json: unknown; claude_code: string } };
export const agentsApi = {
  get: () => req<Agents>("/api/agents"),
  set: (on: boolean) => post<Agents>("/api/agents", { on }),
};

export const soundsApi = {
  sources: () => req<SoundSource[]>("/api/sounds/online/sources"),
  saveKey: (id: string, key: string) => post<{ ok: boolean }>(`/api/sounds/online/${id}/key`, { key }),
  forget: (id: string) => del<{ ok: boolean }>(`/api/sounds/online/${id}/key`),
  search: (query: string, kind: string) => req<{ results: OnlineSound[]; errors: string[] }>(`/api/sounds/online/search?${q({ q: query, kind })}`),
  add: (sound: OnlineSound, kind: string, description = "") => post<Sound>("/api/sounds/online/add", { sound, kind, description }),
  packs: () => req<SoundPack[]>("/api/sounds/packs"),
  pick: (zip: boolean) => post<{ path: string | null }>("/api/sounds/packs/pick", { zip }),
  importPack: (path: string, name: string, license: string) => post<Task>("/api/sounds/packs", { path, name, license }),
  removePack: (name: string) => del<{ removed: number }>(`/api/sounds/packs/${encodeURIComponent(name)}`),
};

export const directorApi = {
  providers: () => req<AiService[]>("/api/director/providers"),
  models: (provider: string) => req<DirectorModel[]>(`/api/director/models?${q({ provider })}`),
  saveKey: (provider: string, key: string) => post<{ ok: boolean }>("/api/director/key", { provider, key }),
  forget: (provider: string) => post<{ ok: boolean }>("/api/director/forget", { provider }),
  estimate: (story: string, where: string, model: string, scenes?: string) =>
    post<DirectorEstimate>("/api/director/estimate", { story, where, model, scenes }),
};

export type ParaTiming = { para: number; start: number; end: number; who: string; estimated?: boolean };

export const bookApi = {
  options: () => req<BookOptions>("/api/books/options"),
  list: () => req<FinishedBook[]>("/api/books"),
  deleteFolder: (name: string) => del<{ deleted: string }>(`/api/books/${encodeURIComponent(name)}`),
  chapters: (story: string, folder: string, title: string, quotes_only: boolean) => post<Chapter[]>("/api/books/chapters", { story, folder, title, quotes_only }),
  removeChapter: (story: string, index: number, title: string, quotes_only: boolean) =>
    post<{ story: string }>("/api/books/chapters/remove", { story, index, title, quotes_only }),
  timings: (story: string, chapter: number, file: string, quotes_only: boolean) =>
    post<{ timings: ParaTiming[]; estimated: boolean }>("/api/books/timings", { story, chapter, file, quotes_only }),
  regenParagraph: (body: BookRequest & { chapter: number; para: number; seed_bump?: number }) =>
    post<Task>("/api/books/regen-paragraph", body),
  storySounds: (story: string, book: string) => post<StorySound[]>("/api/books/sounds", { story, book }),
  palette: (book: string) => req<PaletteSound[]>(`/api/books/palette?${q({ book })}`),
  stats: (story: string) => post<{ text: string }>("/api/books/stats", { story }),
  create: (body: BookRequest) => post<Task>("/api/books/create", body),
  made: (body: BookRequest) => post<{ all: boolean; chapters: number; todo: number }>("/api/books/made", body),
  remake: (body: BookRequest) => post<{ moved: number; seed: number }>("/api/books/remake", body),
  kaggle: (body: BookRequest) => post<{ path: string; message: string }>("/api/books/kaggle", body),
  cloudSend: (body: BookRequest) => post<Task>("/api/books/cloud/send", body),
  stripTags: (story: string, keep: string[]) => post<{ story: string }>("/api/books/strip-tags", { story, keep }),
  cloudEstimate: (body: BookRequest) => post<CloudEstimate>("/api/books/cloud/estimate", body),
  cloud: (folder: string) => req<CloudJob>(`/api/books/cloud?${q({ folder })}`),
  cloudCheck: (folder: string) => post<CloudJob>(`/api/books/cloud/check?${q({ folder })}`, {}),
  /** Stop the book's cloud run (or stop waiting for it). stopped: false when the service couldn't be told (Kaggle). */
  cloudForget: (folder: string) => del<{ stopped?: boolean | null; back?: number; lines?: number }>(`/api/books/cloud?${q({ folder })}`),
  /** After a stop: the run's lines and the chapters made from them go to the Trash, to be made again. */
  cloudStartAgain: (folder: string) => post<{ moved: number; chapters: number }>(`/api/books/cloud/start-again?${q({ folder })}`, {}),
  /** How the book compares with its last cloud run's settings: a change since makes its lines again. */
  cloudMatches: (body: BookRequest) => post<CloudMatch>("/api/books/cloud/matches", body),
  linesCheck: (body: BookRequest) => post<Task>("/api/books/lines/check", body),
  linesChecked: (folder: string) => req<LineCheck>(`/api/books/lines/check?${q({ folder })}`),
  linesAside: (folder: string, keys: string[]) => post<LineCheck>(`/api/books/lines/aside?${q({ folder })}`, { keys }),
  linesPutBack: (folder: string) => post<LineCheck & { put_back: number }>(`/api/books/lines/put-back?${q({ folder })}`, {}),
  qaGet: (folder: string) => req<QaReport>(`/api/books/qa?${q({ folder })}`),
  qaEstimate: (story: string, where: string, model: string) =>
    post<{ chapters: number; cost: string; free_tier?: boolean; low?: number | null; high?: number | null }>("/api/books/qa/estimate", { story, where, model }),
  qaRun: (body: BookRequest & { stage2?: boolean; where?: string; model?: string }) => post<Task>("/api/books/qa/run", body),
  trickyWords: (story: string, cast: string[], existing: Pronunciation[]) =>
    post<TrickyWord[]>("/api/books/pronunciations/suggest", { story, cast, existing }),
  detect: (story: string, cast: CastRow[], method: string, make_on = "mac", cloud_gpu = "") => post<Task>("/api/books/detect", { story, cast, method, make_on, cloud_gpu }),
  hear: (body: { story: string; character: string; voice: string; style: string; speed: number; stability: number; language: string; cast: CastRow[]; models: string; sample_text?: string; pronunciations?: Pronunciation[] }) => post<Task>("/api/books/hear", body),
  apply: (story: string, planned_on: string, placements: Placement[]) => post<{ story: string; message: string }>("/api/books/suggest/apply", { story, planned_on, placements }),
  missing: (story: string, book?: string, scope?: Scope, chapter_plan: ChapterPlan | null = null, online_only = false) =>
    post<Task>("/api/books/missing-sounds", { story, book, scope, chapter_plan, online_only }),
  direct: (story: string, where: string, model: string, checks: number, notes: string, book: string, scenes?: string) =>
    post<Task>("/api/books/direct", { story, where, model, checks, notes, book, scenes }),
  introChoices: (book?: string) => req<IntroChoice[]>(`/api/books/intro-choices${book ? `?${q({ book })}` : ""}`),
  chapterTitles: (story: string) => post<{ chapters: string[] }>("/api/books/chapter-titles", { story }),
  designMusic: (description: string, kind: string, book?: string, scope?: Scope) =>
    post<Task>("/api/books/chapter-music/design", { description, kind, book, scope }),
  detectEstimate: (story: string, make_on: string, cloud_gpu = "") =>
    post<{ cost: string; time: string; free: boolean; where: string; gpu: string }>("/api/books/detect/estimate", { story, make_on, cloud_gpu }),
  importFile: (file: File) => upload<Imported>("/api/books/import", file),
  uploadFile: (file: File) => upload<{ path: string }>("/api/files/upload", file),
};

export const projectApi = {
  list: () => req<ProjectSummary[]>("/api/projects"),
  create: (data: Partial<Project> & { legacy?: boolean }) => post<Project>("/api/projects", data),
  get: (id: string) => req<Project>(`/api/projects/${id}`),
  save: (id: string, changes: Partial<Project>) => put<ProjectSummary>(`/api/projects/${id}`, changes),
  remove: (id: string, deleteAudio = false) =>
    del<{ deleted: string; deleted_audio?: boolean }>(`/api/projects/${id}${deleteAudio ? "?delete_audio=true" : ""}`),
  preflight: (id: string, target: "acx" | "inaudio" | "apple" = "acx") =>
    post<PreflightReport>(`/api/projects/${id}/preflight`, { target }),
  snapshots: (id: string) => req<ProjectSnapshot[]>(`/api/projects/${id}/snapshots`),
  takeSnapshot: (id: string, label = "Snapshot") =>
    post<{ id: string; label: string; time: number; hash: string; deduped: boolean }>(`/api/projects/${id}/snapshots`, { label }),
  diffSnapshot: (id: string, sid: string) =>
    req<{ id: string; changes: string[] }>(`/api/projects/${id}/snapshots/${encodeURIComponent(sid)}/diff`),
  restoreSnapshot: (id: string, sid: string) =>
    post<ProjectSummary>(`/api/projects/${id}/snapshots/${encodeURIComponent(sid)}/restore`, {}),
};

export type PreflightChapter = {
  file: string; duration_s: number; sr: number; channels: number;
  rms_db: number; peak_db: number; floor_db: number; ok: boolean; issues: string[];
};
export type PreflightReport = {
  target: string; spec: string; ok: boolean; passed: number; total: number;
  policy_note?: string; book_issues: string[]; chapters: PreflightChapter[];
};
export type SeriesBible = {
  id: string; name: string; cast: unknown[]; pronunciations: Pronunciation[]; created: number; updated: number;
};
export type ProjectSnapshot = {
  id: string; label: string; time: number; hash: string; chars: number; cast: number; pronunciations: number;
};
export type AcousticPreset = { id: string; label: string; rt60: number; wet: number };
export type DatasetHealth = {
  grade: "A" | "B" | "C" | "D" | "F"; total_clips: number; usable_clips: number;
  usable_minutes: number; total_minutes: number; clipped: number; noisy: number; bad_duration: number;
  quarantined?: number; outliers: { file: string; reasons: string[]; dur?: number; peak_db?: number; floor_db?: number }[];
};

export const seriesApi = {
  list: () => req<SeriesBible[]>("/api/series"),
  create: (name: string, cast: unknown[] = [], pronunciations: Pronunciation[] = []) =>
    post<SeriesBible>("/api/series", { name, cast, pronunciations }),
  get: (sid: string) => req<SeriesBible>(`/api/series/${encodeURIComponent(sid)}`),
  save: (sid: string, changes: Partial<SeriesBible>) => put<SeriesBible>(`/api/series/${encodeURIComponent(sid)}`, changes),
  remove: (sid: string) => del<{ deleted: string }>(`/api/series/${encodeURIComponent(sid)}`),
  syncFromBook: (sid: string, pid: string) => post<SeriesBible>(`/api/series/${encodeURIComponent(sid)}/sync/${encodeURIComponent(pid)}`, {}),
  applyToBook: (sid: string, pid: string) => post<ProjectSummary>(`/api/series/${encodeURIComponent(sid)}/apply/${encodeURIComponent(pid)}`, {}),
};

export const acousticsApi = {
  presets: () => req<AcousticPreset[]>("/api/acoustics/presets"),
};

export const datasetHealthApi = {
  score: (voice: string) => req<DatasetHealth>(`/api/train/dataset/${encodeURIComponent(voice)}/health`),
  autoCure: (voice: string) => post<DatasetHealth>(`/api/train/dataset/${encodeURIComponent(voice)}/auto-cure`, {}),
};

// ---------- Voices, Library, Models, Connections, Train ----------
export type MyVoice = { name: string; gender: string; source: string; description: string; transcript: string; created: string; file: string };
export type PackVoice = { speaker?: string; name: string; gender: string; value?: string | null; preview?: string | null; has_preview?: boolean };
/** A pack's signature: verified true (the key is on that GitHub account), false (with why), null (not checked yet). */
export type PackSigned = { github: string; fingerprint: string; verified: boolean | null; why: string };
export type Pack = {
  id: string; name: string; genre: string; description: string; version: number; author: string; engine: string;
  voices: PackVoice[]; folder?: string; size_gb?: number; size?: number; where?: string; signed?: PackSigned | null;
};
export type UpdateStatus = {
  current: string; auto: boolean; available: boolean; ready: boolean; system: "mac" | "windows" | "appimage" | "deb";
  latest: { version: string; name: string; notes: string; page: string; published: string } | null;
  size: number | null; error: string; busy: string[];
};
export const updateApi = {
  status: (check = false) => req<UpdateStatus>(`/api/update${check ? "?check=true" : ""}`),
  download: () => post<Task>("/api/update/download", {}),
  auto: (on: boolean) => post<UpdateStatus>("/api/update/auto", { on }),
};
export type Signing = { github: string; public_key: string | null; fingerprint: string; on_github: boolean | null };
export const signingApi = {
  get: () => req<Signing>("/api/signing"),
  set: (github: string) => post<Signing>("/api/signing", { github }),
};
export type ModelInfo = {
  id: string; name: string; short: string; tier: string; group: "light" | "standard" | "high"; license: string;
  ok: boolean; why: string; note: string; locked: boolean; models: string; size_gb: number; missing: string[];
  pip: string[]; min_ram_gb: number; downloaded: boolean; off: boolean;
};
/** Windows and Linux: whether the voices still need PyTorch, and which build (narrator/pc_setup.py). */
export type VoicesSetup = { needed: boolean; gpu?: string; build?: string; size_gb?: number; uv?: boolean };
export type Connections = {
  ssh: { host: string; port: string; key: string; dir: string; python: string };
  orpheus: { where: "mac" | "server"; url: string; has_key: boolean; batch: boolean };
  huggingface: { has_token: boolean };
};
/** Where Create makes a book: this Mac, or the lines first on a cloud GPU (Kaggle, Modal, or a pay-as-you-go
 *  service with the user's own key: runpod, vast, thunder, lambda, lightning, beam, cerebrium, hfjobs). */
export type MakeOnId = string;
/** A cloud GPU the Create menu offers, with the service's price per hour now (where it says). */
export type CloudGpu = { id: string; label: string; gb?: number | null; price: number | null; stock?: string | null };
export type RunpodGpu = CloudGpu;
/** A pay-as-you-go GPU service (Connections, and the Create menu once its key is saved). */
export type CloudService = { id: string; name: string; pricing: string; has_key: boolean; default_gpu: string;
  sends_models: boolean; live: boolean; gpus: CloudGpu[] };
/** A trained voice model on this Mac, and whether it's on Modal (Connections → Modal). */
export type ModalModel = { folder: string; name: string; voices: string[]; mb: number; state: "uploaded" | "changed" | "not uploaded"; uploaded: number | null };
export type Check = { balance?: number; ok: boolean; detail: string; quota?: { used: number; left: number; total: number; resets: string } | null; user?: string; readiness?: string };
export type TrainedModel = {
  id: string; folder: string; title: string; engine: string; recipe: string; run: string | null;
  voices: { speaker: string; name: string; gender: string; value: string | null }[];
  max_voices: number; can_add: boolean; why: string; size_gb: number; hours: number | null;
};
export type Run = {
  key: string; model: string; recipe: string; voices: string[]; where: string; state: string; label: string;
  url: string; sent: string; hours: number | null; gpu_hours: number | null; installed: string | null;
  error: string | null; can_retry: boolean; // why it made no model; whether it can be sent again as it was
};
export type PrepareBackend = {
  id: string; name: string; pricing: string; readiness: string; default_gpu: string;
  gpus: { id: string; label: string; price?: number | null }[];
};
export type TrainOptions = {
  recipes: { id: string; name: string; ready: boolean; max_voices: number; note: string }[];
  backends: { id: string; name: string; pricing: string; readiness: string }[];
  prepare_backends?: PrepareBackend[];
  languages: string[];
};
export type PreparedVoice = { name: string; clips: number; hours: number; ready: boolean; running: boolean; stage: string; progress: number | null; where?: string; unfinished: boolean };
export type PrepareStatus = {
  voice: string; running: boolean; stage: string; progress: number | null; error: string | null; done: boolean;
  elapsed: number | null; title: string; where?: string; backend?: string; cloud_gpu?: string; log: string; unfinished: boolean;
  books: { title: string; audio_h: string; kept_h: string; clips: number | string; matched: string }[];
  dataset: { clips: number; hours: number } | null;
};
export type TrainRequest = {
  voices: { dataset: string; name: string; gender: string }[]; model?: string; recipe: string; backend: string;
  epochs?: number | null; max_clips?: number; add_to?: string | null; consent?: boolean;
};

async function uploadForm<T>(path: string, form: FormData): Promise<T> {
  let r: Response;
  try { r = await call(path, { method: "POST", body: form }); }
  catch { throw new ApiError("Can't reach Babble. Is it running?"); }
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : msg; } catch { /* not JSON */ }
    throw new ApiError(msg);
  }
  return r.json();
}

export const voiceApi = {
  mine: () => req<MyVoice[]>("/api/my-voices"),
  /** The voice reading custom text (e.g. a highlighted paragraph) or saying hello: {audio} if cached, else a task. */
  preview: (voice: string, text?: string, language?: string) =>
    post<Partial<Task> & { audio?: string }>("/api/voices/preview", { voice, ...(text ? { text } : {}), ...(language ? { language } : {}) }),
  update: (name: string, body: { gender: string; description: string; transcript?: string }) => put<MyVoice>(`/api/my-voices/${encodeURIComponent(name)}`, body),
  remove: (name: string) => del<{ deleted: string }>(`/api/my-voices/${encodeURIComponent(name)}`),
  design: (body: { description: string; text?: string; language?: string; seed?: number }) => post<Task>("/api/my-voices/design", body),
  saveTake: (body: { path: string; name: string; gender: string; description: string; transcript: string }) => post<MyVoice>("/api/my-voices/save-take", body),
  saveClone: (body: { path: string; name: string; gender: string; description: string; transcript: string }) => post<MyVoice>("/api/my-voices/clone", body),
  transcribe: (path: string) => post<Task>("/api/my-voices/transcribe", { path }),
};

export const libraryApi = {
  packs: () => req<{ installed: Pack[]; catalog: Pack[]; genres: string[] }>("/api/packs"),
  install: (id: string, version: number) => post<Task>("/api/packs/install", { id, version }),
  installFile: (file: File) => { const f = new FormData(); f.append("file", file); return uploadForm<Task>("/api/packs/install-file", f); },
  remove: (folder: string) => del<{ deleted: string }>(`/api/packs?${q({ folder })}`),
  models: () => req<ModelInfo[]>("/api/models"),
  download: (id: string) => post<Task>(`/api/models/${id}/download`, {}),
  extras: (id: string) => post<Task>(`/api/models/${id}/extras`, {}),
  setOff: (id: string, off: boolean) => post<{ id: string; off: boolean }>(`/api/models/${id}/off`, { off }),
  deleteModel: (id: string) => req<{ id: string; freed_gb: number }>(`/api/models/${id}`, { method: "DELETE" }),
  redownload: (id: string) => post<Task>(`/api/models/${id}/redownload`, {}),
  voicesSetup: () => req<VoicesSetup>("/api/voices-setup"),
  setUpVoices: () => post<Task>("/api/voices-setup", {}),
};

/** A problem a service had while Babble used it (narrator/conn_issues.py), until it works again or is dismissed. */
export type ConnIssue = { service: string; name: string; kind: "sign-in" | "limit" | "network"; error: string; where: string; when: number };
export const connectionApi = {
  get: () => req<Connections>("/api/connections"),
  check: (id: string) => post<Check>(`/api/connections/${id}/check`, {}),
  issues: () => req<ConnIssue[]>("/api/connections/issues"),
  dismissIssue: (service: string) => req<ConnIssue[]>(`/api/connections/issues/${encodeURIComponent(service)}`, { method: "DELETE" }),
  kaggleSignIn: () => post<Task>("/api/connections/kaggle/signin", {}),
  kaggleToken: (token: string) => post<{ ok: boolean; user: string }>("/api/connections/kaggle/token", { token }),
  kaggleSignOut: () => post<{ ok: boolean }>("/api/connections/kaggle/signout", {}),
  modalSignOut: () => post<{ ok: boolean }>("/api/connections/modal/signout", {}),
  forgetSsh: () => del<Connections["ssh"]>("/api/connections/ssh"),
  forgetOrpheus: () => del<Connections["orpheus"]>("/api/connections/orpheus"),
  forgetHf: () => del<{ has_token: boolean }>("/api/connections/huggingface"),
  modalSignIn: () => post<Task>("/api/connections/modal/signin", {}),
  clouds: () => req<CloudService[]>("/api/connections/clouds"),
  cloudKey: (service: string, key: string) => post<{ ok: boolean; balance?: number | null; user?: string | null }>(`/api/connections/${service}/key`, { key }),
  cloudForget: (service: string) => del<{ ok: boolean }>(`/api/connections/${service}/key`),
  cloudGpus: (service: string) => req<CloudGpu[]>(`/api/connections/${service}/gpus`),
  modalModels: () => req<ModalModel[]>("/api/connections/modal/models"),
  modalUpload: (folder: string) => post<Task>("/api/connections/modal/models/upload", { folder }),
  modalRemove: (folder: string) => del<ModalModel[]>(`/api/connections/modal/models?${q({ folder })}`),
  saveSsh: (body: Connections["ssh"]) => post<Connections["ssh"]>("/api/connections/ssh", body),
  saveOrpheus: (body: { where: string; url: string; key?: string | null; test?: boolean }) => post<Connections["orpheus"] & { detail: string }>("/api/connections/orpheus", body),
  saveHf: (token: string) => post<{ has_token: boolean; user: string }>("/api/connections/huggingface", { token }),
};

export const trainApi = {
  options: () => req<TrainOptions>("/api/train/options"),
  models: () => req<TrainedModel[]>("/api/train/models"),
  removeModel: (folder: string) => del<{ deleted: string }>(`/api/train/models?${q({ folder })}`),
  runs: () => req<Run[]>("/api/train/runs"),
  check: (key: string) => post<{ state: string; label: string; detail: string }>(`/api/train/runs/check?${q({ key })}`, {}),
  finish: (key: string) => post<Task>(`/api/train/runs/finish?${q({ key })}`, {}),
  retry: (key: string) => post<Task>(`/api/train/runs/retry?${q({ key })}`, {}),
  /** What Train again would take and cost (asked before it sends). */
  retryEstimate: (key: string) => post<{ hours: number; gpu_hours: number; backend: string; backend_id: string; text: string; cost?: string; free?: boolean; paid?: boolean }>(`/api/train/runs/retry-estimate?${q({ key })}`, {}),
  /** Training tools: files in the order given, joined into one text (.txt) or one recording. */
  mergeText: (files: File[], name: string, headings: boolean) => {
    const f = new FormData(); files.forEach((x) => f.append("files", x)); f.append("name", name); f.append("headings", String(headings));
    return uploadForm<Task>("/api/train/tools/merge-text", f);
  },
  mergeAudio: (files: File[], name: string, format: string) => {
    const f = new FormData(); files.forEach((x) => f.append("files", x)); f.append("name", name); f.append("format", format);
    return uploadForm<Task>("/api/train/tools/merge-audio", f);
  },
  forget: (key: string) => del<{ forgot: string }>(`/api/train/runs?${q({ key })}`),
  cancelRun: (key: string) => post<{ cancelled: string }>(`/api/train/runs/cancel?${q({ key })}`, {}),
  samples: (key: string) => req<{ file: string; name: string }[]>(`/api/train/runs/samples?${q({ key })}`),
  voices: () => req<PreparedVoice[]>("/api/train/voices"),
  prepareEstimate: (backend: string, cloud_gpu = "", audio_seconds = 3600, voice = "") =>
    post<{ cost: string; time: string; free: boolean; where: string; gpu: string; hours: number; budget: number; needed: boolean }>("/api/train/prepare/estimate", { backend, cloud_gpu, audio_seconds, voice }),
  prepare: (body: { name: string; title: string; language: string; options: Record<string, number>; files: File[]; backend?: string; cloud_gpu?: string }) => {
    const f = new FormData();
    f.append("name", body.name); f.append("title", body.title); f.append("language", body.language);
    f.append("options", JSON.stringify(body.options));
    if (body.backend) f.append("backend", body.backend);
    if (body.cloud_gpu) f.append("cloud_gpu", body.cloud_gpu);
    body.files.forEach((x) => f.append("files", x));
    return uploadForm<{ voice: string; title: string; resumed: boolean }>("/api/train/prepare", f);
  },
  status: (voice: string) => req<PrepareStatus>(`/api/train/prepare/${encodeURIComponent(voice)}`),
  stop: (voice: string) => post<{ stopping: boolean }>(`/api/train/prepare/${encodeURIComponent(voice)}/stop`, {}),
  clips: (voice: string) => req<{ file: string; text: string }[]>(`/api/train/prepare/${encodeURIComponent(voice)}/clips`),
  estimate: (body: TrainRequest) => post<{ hours: number; gpu_hours: number; backend: string; text: string; cost?: string; free?: boolean; paid?: boolean }>("/api/train/estimate", body),
  send: (body: TrainRequest) => post<Task>("/api/train/send", body),
  exportPack: (body: { folder: string; name: string; genre: string; version: number; description: string; author: string; rights: Record<string, { source: string; license: string; consent: string }>; previews: boolean; sign?: boolean }) => post<Task>("/api/train/export", body),
};

/** A file the engine made. v: a version (when it was made), so a file made again under the same name, like a
 *  book's finished audiobook, is fetched fresh instead of the copy the page already has. */
/** The Activity page: what the app made lately, cloud runs with their cost, and what failed. */
export type ActivityKind = "audiobook" | "speech" | "effect" | "background" | "music" | "cloud" | "other";
export type ActivityEntry = {
  id: string; kind: ActivityKind; title: string; state: "running" | "done" | "failed" | "stopped"; status: string;
  error?: string | null; started?: number; ended?: number; file?: string | null; progress?: number | null; eta?: number | null;
  backend?: string; gpu?: string; lines?: number; price?: number | null; max_cost?: number | null; gpu_hours?: number; cost?: number | null;
};
export const activityApi = {
  list: () => req<{ running: ActivityEntry[]; entries: ActivityEntry[] }>("/api/activity"),
  clear: () => del<{ cleared: boolean }>("/api/activity"),
};
/** A voice's preview from a pack that isn't installed yet (the engine fetches it from the catalog once). */
export const packPreviewUrl = (id: string, version: number, speaker: string) =>
  authed(`${API_BASE}/api/packs/preview?id=${encodeURIComponent(id)}&version=${version}&speaker=${encodeURIComponent(speaker)}`);
export const fileUrl = (path: string, v?: number) => authed(`${API_BASE}/api/files?path=${encodeURIComponent(path)}${v ? `&v=${v}` : ""}`);

/** Follow a running task: onUpdate with each change, onEnd once it's over. Returns a function that stops following. */
/** The engine restarted (it stopped, or macOS ended it when memory ran out): it no longer knows its old tasks. */
export const gone = (e: unknown) => e instanceof ApiError && e.message.startsWith("No such task");
export const lost = (t: Task): Task => ({
  ...t, state: "failed",
  error: "Babble's engine stopped while this was running (on this Mac that's usually memory running out) and was started "
    + "again. Start it again: what was already made is kept and reused.",
});

export function followTask(t: Task, onUpdate: (t: Task) => void, onEnd: (t: Task) => void): () => void {
  let cur = t, closed = false;
  const es = new EventSource(authed(`${API_BASE}/api/tasks/${t.id}/events`));
  es.onmessage = (m) => {
    const ev = JSON.parse(m.data);
    cur = {
      ...cur, status: ev.status ?? cur.status, progress: ev.progress ?? cur.progress,
      eta: ev.eta === undefined ? cur.eta : ev.eta, updated: ev.updated ?? cur.updated, etaAt: Date.now(),
      result: { ...cur.result, ...Object.fromEntries(Object.entries(ev).filter(([k]) => !["status", "n", "progress", "eta", "updated"].includes(k))) },
    };
    onUpdate(cur);
  };
  es.addEventListener("end", (m) => { es.close(); if (!closed) { closed = true; onEnd(JSON.parse((m as MessageEvent).data)); } });
  es.onerror = () => {
    es.close();
    const poll = async () => {
      if (closed) return;
      try {
        const now = await api.task(t.id);
        if (now.state === "running") { onUpdate({ ...now, etaAt: Date.now() }); setTimeout(poll, 1000); } else { closed = true; onEnd(now); }
      } catch (e) {
        if (gone(e)) { closed = true; onEnd(lost(cur)); } else setTimeout(poll, 2500);  // (the engine restarted)
      }
    };
    poll();
  };
  return () => { closed = true; es.close(); };
}

const taskKey = (k: string) => `babble-task:${k}`;
const loadTaskId = (k?: string) => { if (!k) return null; try { return sessionStorage.getItem(taskKey(k)); } catch { return null; } };
const saveTaskId = (k: string | undefined, id: string | null) => {
  if (!k) return;
  try { if (id) sessionStorage.setItem(taskKey(k), id); else sessionStorage.removeItem(taskKey(k)); } catch { /* private window */ }
};

/** Start a task and follow it live (server-sent events, falling back to polling).
 *  When `persistKey` is given, switching tabs and coming back re-attaches to the running (or just-finished) task. */
export function useTask(onDone?: (t: Task) => void, persistKey?: string) {
  const [task, setTask] = useState<Task | null>(() => {
    const id = loadTaskId(persistKey);
    return id ? { id, kind: "", title: "", state: "running", status: "Reconnecting…", progress: null, result: {}, error: null } : null;
  });
  const taskRef = useRef<Task | null>(task);
  taskRef.current = task;
  const source = useRef<EventSource | null>(null);
  const pollTimer = useRef<number | null>(null);
  const unmounted = useRef(false);
  const runIdRef = useRef(0);
  const doneRef = useRef(onDone);
  doneRef.current = onDone;

  const close = () => {
    source.current?.close();
    source.current = null;
    if (pollTimer.current !== null) {
      clearTimeout(pollTimer.current);
      pollTimer.current = null;
    }
  };

  const follow = useCallback((t: Task) => {
    close();
    const myRun = ++runIdRef.current;
    if (t.id) saveTaskId(persistKey, t.id);
    setTask({ ...t, etaAt: Date.now() });
    const es = new EventSource(authed(`${API_BASE}/api/tasks/${t.id}/events`));
    source.current = es;
    es.onmessage = (m) => {
      if (unmounted.current || runIdRef.current !== myRun) return;
      const ev = JSON.parse(m.data);
      setTask((cur) => cur && {
        ...cur,
        status: ev.status ?? cur.status,
        progress: ev.progress ?? cur.progress,
        eta: ev.eta === undefined ? cur.eta : ev.eta, updated: ev.updated ?? cur.updated, etaAt: Date.now(),
        result: { ...cur.result, ...Object.fromEntries(Object.entries(ev).filter(([k]) => !["status", "n", "progress", "eta", "updated"].includes(k))) },
      });
    };
    es.addEventListener("end", (m) => {
      if (unmounted.current || runIdRef.current !== myRun) { es.close(); return; }
      const final: Task = JSON.parse((m as MessageEvent).data);
      close();
      saveTaskId(persistKey, null);
      setTask(final);
      doneRef.current?.(final);
    });
    es.onerror = () => { // the stream dropped: poll until it's over
      if (unmounted.current || runIdRef.current !== myRun) { es.close(); return; }
      close();
      const poll = async () => {
        pollTimer.current = null;
        if (unmounted.current || runIdRef.current !== myRun) return;
        try {
          const cur = await api.task(t.id);
          if (unmounted.current || runIdRef.current !== myRun) return;
          setTask({ ...cur, etaAt: Date.now() });
          if (cur.state === "running") pollTimer.current = window.setTimeout(poll, 800);
          else { saveTaskId(persistKey, null); doneRef.current?.(cur); }
        } catch (e) {
          if (unmounted.current || runIdRef.current !== myRun) return;
          if (gone(e)) {
            saveTaskId(persistKey, null);
            const f = lost(taskRef.current?.id === t.id ? taskRef.current : t);
            setTask(f);
            doneRef.current?.(f);
          } else pollTimer.current = window.setTimeout(poll, 2000);
        }
      };
      poll();
    };
  }, [persistKey]);

  useEffect(() => {
    unmounted.current = false;
    let cancelled = false;
    const id = loadTaskId(persistKey);
    const myRun = runIdRef.current;
    if (id) {
      api.task(id).then((t) => {
        if (cancelled || unmounted.current || runIdRef.current !== myRun) return;
        if (t.state === "running") follow(t);
        else { saveTaskId(persistKey, null); setTask(t); doneRef.current?.(t); }
      }).catch(() => {
        if (cancelled || unmounted.current || runIdRef.current !== myRun) return;
        saveTaskId(persistKey, null);
        setTask(null);
      });
    }
    return () => { cancelled = true; unmounted.current = true; close(); };
  }, [persistKey, follow]);

  const start = useCallback(async (begin: () => Promise<Task>) => {
    close();
    const myRun = ++runIdRef.current;
    setTask({ id: "", kind: "", title: "", state: "running", status: "Starting…", progress: null, result: {}, error: null });
    try {
      const t = await begin();
      if (runIdRef.current !== myRun) return;
      if (unmounted.current) {
        if (t.id) saveTaskId(persistKey, t.id);
        return;
      }
      follow(t);
    } catch (e) {
      if (unmounted.current || runIdRef.current !== myRun) return;
      saveTaskId(persistKey, null);
      const failed: Task = { id: "", kind: "", title: "", state: "failed", status: "", progress: null, result: {}, error: (e as Error).message };
      setTask(failed);
      doneRef.current?.(failed);
    }
  }, [follow, persistKey]);

  const stop = useCallback(async () => { if (task?.id) await api.stopTask(task.id); }, [task?.id]);
  const clear = useCallback(() => { runIdRef.current++; close(); saveTaskId(persistKey, null); setTask(null); }, [persistKey]);
  const running = task?.state === "running";
  return { task, start, stop, running, clear, resume: follow };
}

/** Load something once (and again on reload()). */
export function useLoad<T>(load: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [n, setN] = useState(0);
  useEffect(() => {
    let live = true;
    load().then((d) => live && (setData(d), setError(null))).catch((e) => live && setError(e.message));
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, n]);
  return { data, error, reload: () => setN((x) => x + 1) };
}
