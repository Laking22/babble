import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { Copy, HardDrive, LayoutGrid, Loader2, Moon, Palette, Plug, RotateCcw, Rows3, SlidersHorizontal, Sun, Terminal, Trash2, Wallet } from "lucide-react";
import { agentsApi, api, signingApi, updateApi, useLoad, useTask, type Agents, type Signing, type StorageBreakdown, type UpdateStatus } from "../api";
import { Segmented, Slider, Starting, Switch, TaskProgress, useStarting, useToast } from "../components/ui";
import { applyUpdate, openExternal } from "../native";
import { Connections } from "./Connections";
import { SpendingTab } from "./Spending";
import { useConnIssues } from "./ConnIssues";
import { SHELF_SIZES, useShelfPrefs, useTheme, type ShelfShape, type ShelfSize, type Theme } from "../uiPrefs";

export type SettingsTab = "general" | "storage" | "appearance" | "spending" | "connections";
const TABS: SettingsTab[] = ["general", "storage", "appearance", "spending", "connections"];
const TAB_KEY = "babble-settings-tab";
// Settings → "Longest line for trained voices" (settings.longest_line; 0 = sentences never cut)
const LINE_MIN = 150, LINE_MAX = 300, LINE_WHOLE = 310;

const THEME_OPTIONS: readonly { id: Theme; title: string; icon: typeof Sun }[] = [
  { id: "light", title: "Light", icon: Sun },
  { id: "dark", title: "Dark", icon: Moon },
];

const SIZE_LABELS: Record<ShelfSize, string> = {
  small: "Small",
  medium: "Medium",
  large: "Large",
};

/** Settings → General: AI agents (Claude, Cursor, …) using Babble through MCP (narrator/agents.py, babble_mcp.py). */
async function copyText(text: string): Promise<boolean> {
  try { await navigator.clipboard.writeText(text); return true; } catch { /* the older way below */ }
  const box = document.createElement("textarea");
  box.value = text;
  box.style.position = "fixed";
  box.style.opacity = "0";
  document.body.appendChild(box);
  box.select();
  const ok = document.execCommand("copy");
  box.remove();
  return ok;
}

function AgentsCard() {
  const toast = useToast();
  const [a, setA] = useState<Agents | null>(null);
  useEffect(() => { agentsApi.get().then(setA).catch(() => {}); }, []);
  const toggle = async (on: boolean) => {
    try { setA(await agentsApi.set(on)); } catch (e) { toast((e as Error).message, true); }
  };
  const copy = async (text: string, done: string) => {
    const ok = await copyText(text);
    toast(ok ? done : "Couldn't copy.", !ok);
  };
  if (!a) return null;
  return (
    <section className="card card-pad stack">
      <h3 className="card-title" style={{ margin: 0 }}>AI agents</h3>
      <Switch label="Let AI agents use Babble (MCP)" checked={a.on} onChange={toggle} />
      {a.on && (
        <div className="row wrap" style={{ gap: 8 }}>
          <button className="btn sm" onClick={() => copy(JSON.stringify(a.setup.json, null, 2), "Copied. Paste it into your AI app's MCP settings.")}>
            <Copy size={13} />Copy setup
          </button>
          <button className="btn sm" onClick={() => copy(a.setup.claude_code, "Copied. Run it in a terminal.")}>
            <Terminal size={13} />Copy Claude Code command
          </button>
        </div>
      )}
    </section>
  );
}

/** Signing voice packs with a GitHub account (narrator/signing.py): the key is made here, its public half goes on GitHub. */
/** Updates: newer versions from Babble's releases on GitHub (narrator/updater.py; the app installs them). */
export function UpdatesCard() {
  const toast = useToast();
  const [u, setU] = useState<UpdateStatus | null>(null);
  const checking = useStarting();
  const restarting = useStarting();
  const load = (check = false) => updateApi.status(check).then(setU).catch((e) => toast((e as Error).message, true));
  useEffect(() => { load(); }, []); // eslint-disable-line react-hooks/exhaustive-deps
  const dl = useTask((t) => {
    if (t.state === "failed") toast(t.error || "Couldn't download the update.", true);
    load();
  }, "update-download");
  const restart = () => restarting.run(async () => {
    try { await applyUpdate(); } catch (e) { toast(String((e as Error).message || e), true); }
  });
  const next = u?.latest;
  const mb = u?.size ? `${Math.round(u.size / 1e6)} MB` : "";
  return (
    <section className="card card-pad stack">
      <h3 className="card-title" style={{ margin: 0 }}>Updates</h3>
      {!u && <div className="small muted row"><Loader2 size={14} className="spin" /> Loading…</div>}
      {u && (
        <>
          <div className="row wrap" style={{ gap: 10 }}>
            <div className="grow small">
              <div style={{ fontWeight: 500 }}>
                {!u.current ? "Babble (not installed from a release)"
                  : u.ready ? `Babble ${next?.version} is ready to install`
                  : u.available ? `${next?.name || `Babble ${next?.version}`} is out`
                  : `Babble ${u.current} is up to date`}
              </div>
              <div className="muted">
                {u.ready ? (u.auto ? "It's installed when you quit Babble, or restart now." : "Restart Babble to install it.")
                  : u.available ? `You have ${u.current}.${mb ? ` The download is ${mb}.` : ""}`
                  : u.current ? `Version ${u.current}` : u.error}
              </div>
              {u.error && u.current && <div className="small error-text">{u.error}</div>}
            </div>
            {next?.page && (u.available || u.ready) && <button className="btn sm ghost" onClick={() => openExternal(next.page)}>What's new</button>}
            {u.ready ? (
              <button className="btn sm primary" disabled={restarting.starting || u.busy.length > 0} onClick={restart}
                title={u.busy.length ? `Waiting for: ${u.busy.join(", ")}` : undefined}>
                {restarting.starting ? <Starting size={13} label="Restarting…" /> : "Restart to update"}</button>
            ) : u.available ? (
              <button className="btn sm primary" disabled={dl.running} onClick={() => dl.start(() => updateApi.download())}>
                {dl.running ? <Starting size={13} label="Downloading…" /> : "Download"}</button>
            ) : u.current ? (
              <button className="btn sm" disabled={checking.starting} onClick={() => checking.run(() => load(true))}>
                {checking.starting ? <Starting size={13} label="Checking…" /> : "Check now"}</button>
            ) : null}
          </div>
          {u.ready && u.busy.length > 0 && <div className="small muted">Babble restarts once {u.busy.join(", ")} {u.busy.length === 1 ? "is" : "are"} done.</div>}
          {dl.task && dl.task.state === "running" && <TaskProgress task={dl.task} onStop={dl.stop} />}
          {u.current && (
            <Switch label="Update automatically" checked={u.auto}
              hint={u.system === "deb" ? "Downloads new versions; you install them with your software installer." : undefined}
              onChange={async (v) => { try { setU(await updateApi.auto(v)); } catch (e) { toast((e as Error).message, true); } }} />
          )}
        </>
      )}
    </section>
  );
}

function SigningCard() {
  const toast = useToast();
  const [s, setS] = useState<Signing | null>(null);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => { signingApi.get().then((x) => { setS(x); setName(x.github); }).catch(() => {}); }, []);
  const run = async (f: () => Promise<Signing>) => {
    setBusy(true);
    try { const x = await f(); setS(x); setName(x.github); } catch (e) { toast((e as Error).message, true); } finally { setBusy(false); }
  };
  if (!s) return null;
  const typed = name.trim().replace(/^@/, "");
  const ready = !!(s.github && s.public_key);
  return (
    <section className="card card-pad stack">
      <h3 className="card-title" style={{ margin: 0 }}>Signing voice packs</h3>
      <div className="row" style={{ gap: 8 }}>
        <input className="input" placeholder="GitHub username" aria-label="GitHub username" value={name}
          onChange={(e) => setName(e.target.value)} style={{ maxWidth: 240 }} />
        <button className="btn sm" disabled={busy || typed === s.github} onClick={() => run(() => signingApi.set(typed))}>{s.github ? "Save" : "Set up"}</button>
        {ready && !s.on_github && <button className="btn sm ghost" disabled={busy} onClick={() => run(signingApi.get)}>{busy ? <Loader2 size={13} className="spin" /> : null}Check again</button>}
      </div>
      {ready && (s.on_github
        ? <span className="small ok-text">Your key is on github.com/{s.github}. Packs you sign show your name.</span>
        : <>
          <span className="small muted">{s.on_github === null ? "Couldn't reach GitHub to check." : "Add this key on GitHub: Settings → SSH and GPG keys → New SSH key, with Key type set to Signing Key."}</span>
          <div className="row"><button className="btn sm" onClick={async () => { const ok = await copyText(s.public_key!); toast(ok ? "Copied. Paste it on GitHub." : "Couldn't copy.", !ok); }}><Copy size={13} />Copy key</button></div>
        </>)}
    </section>
  );
}

function fmtBytes(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(n >= 10e9 ? 1 : 2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 100e6 ? 0 : 1)} MB`;
  if (n >= 1e3) return `${Math.round(n / 1e3)} KB`;
  return `${n} B`;
}

/** App-wide settings: General narration options, Storage cleanup, Appearance (audiobook shelf layout), and Connections. */
export function Settings({ initialTab, focusGroup, tabNonce }: {
  initialTab?: SettingsTab;
  focusGroup?: string;
  tabNonce?: number;
}) {
  const toast = useToast();
  const problems = useConnIssues(60000).list.length;
  const [tab, setTabState] = useState<SettingsTab>(() => {
    if (initialTab) return initialTab;
    try {
      const saved = localStorage.getItem(TAB_KEY) as SettingsTab | null;
      return saved && TABS.includes(saved) ? saved : "general";
    } catch {
      return "general";
    }
  });
  const setTab = (next: SettingsTab) => {
    setTabState(next);
    try { localStorage.setItem(TAB_KEY, next); } catch { /* private mode */ }
  };

  useEffect(() => {
    if (initialTab && tabNonce) setTab(initialTab);
  }, [initialTab, tabNonce]); // eslint-disable-line react-hooks/exhaustive-deps

  const { theme, setTheme } = useTheme();
  const themeBtnRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const onThemeKeyDown = (e: KeyboardEvent, idx: number) => {
    const len = THEME_OPTIONS.length;
    let next: number | null = null;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") next = (idx + 1) % len;
    else if (e.key === "ArrowLeft" || e.key === "ArrowUp") next = (idx - 1 + len) % len;
    if (next === null) return;
    e.preventDefault();
    setTheme(THEME_OPTIONS[next].id);
    themeBtnRefs.current[next]?.focus();
  };
  const { shape, size, setShape, setSize } = useShelfPrefs();
  const sizeIndex = SHELF_SIZES.indexOf(size);

  const [s, setS] = useState<Record<string, any> | null>(null);
  const [sErr, setSErr] = useState<string | null>(null);
  useEffect(() => {
    api.settings().then((d) => { setS(d); setSErr(null); }).catch((e) => setSErr((e as Error).message));
  }, []);
  const save = (k: string, v: any) => {
    setS((x) => ({ ...x, [k]: v }));
    api.saveSettings({ [k]: v }).catch((e) => {
      toast((e as Error).message, true);
      api.settings().then(setS).catch(() => {});
    });
  };

  return (
    <div className="page narrow">
      <div className="page-head">
        <div>
          <h1>Settings</h1>
        </div>
      </div>

      <div className="tabs page-tabs" role="tablist" aria-label="Settings sections">
        <button type="button" role="tab" aria-selected={tab === "general"} className={tab === "general" ? "on" : ""}
          onClick={() => setTab("general")}>
          <span className="settings-tab-lbl"><SlidersHorizontal size={14} />General</span>
        </button>
        <button type="button" role="tab" aria-selected={tab === "storage"} className={tab === "storage" ? "on" : ""}
          onClick={() => setTab("storage")}>
          <span className="settings-tab-lbl"><HardDrive size={14} />Storage</span>
        </button>
        <button type="button" role="tab" aria-selected={tab === "appearance"} className={tab === "appearance" ? "on" : ""}
          onClick={() => setTab("appearance")}>
          <span className="settings-tab-lbl"><Palette size={14} />Appearance</span>
        </button>
        <button type="button" role="tab" aria-selected={tab === "spending"} className={tab === "spending" ? "on" : ""}
          onClick={() => setTab("spending")}>
          <span className="settings-tab-lbl"><Wallet size={14} />Spending</span>
        </button>
        <button type="button" role="tab" aria-selected={tab === "connections"} className={tab === "connections" ? "on" : ""}
          onClick={() => setTab("connections")}>
          <span className="settings-tab-lbl">
            <Plug size={14} />Connections
            {problems > 0 && <span className="nav-badge" style={{ marginLeft: 4 }}>{problems}</span>}
          </span>
        </button>
      </div>

      {tab === "general" && (
        <div className="stack">
          <section className="card card-pad stack">
            <h3 className="card-title" style={{ margin: 0 }}>Narration &amp; audio</h3>
            {sErr && <div className="small error-text">{sErr}</div>}
            {!s && !sErr && <div className="small muted row"><Loader2 size={14} className="spin" /> Loading…</div>}
            {s && (
              <>
                <Slider
                  label="Longest line for trained voices"
                  value={s.longest_line ? Math.min(Math.max(s.longest_line, LINE_MIN), LINE_MAX) : LINE_WHOLE}
                  min={LINE_MIN} max={LINE_WHOLE} step={10}
                  onChange={(v) => save("longest_line", v >= LINE_WHOLE ? 0 : v)}
                  format={(v) => (v >= LINE_WHOLE ? "Whole sentences" : `${v} characters`)}
                  left="Fewer skipped words" right="More expression"
                />
                <Switch
                  label="Smart sound matching (CLAP)"
                  checked={s.sfx_smart !== false}
                  onChange={(v) => save("sfx_smart", v)}
                />
                <div className="row wrap" style={{ gap: 10 }}>
                  <div className="grow">
                    <div className="small" style={{ fontWeight: 500 }}>AI music quality</div>
                  </div>
                  <Segmented<string>
                    value={s.music_quality || "auto"}
                    onChange={(v) => save("music_quality", v)}
                    options={[
                      { value: "auto", label: "Auto" },
                      { value: "small", label: "Small" },
                      { value: "medium", label: "Medium" },
                    ]}
                  />
                </div>
                <div className="row wrap" style={{ gap: 10 }}>
                  <div className="grow">
                    <div className="small" style={{ fontWeight: 500 }}>Audiobook file quality</div>
                  </div>
                  <Segmented<string>
                    value={s.file_quality || "standard"}
                    onChange={(v) => save("file_quality", v)}
                    options={[
                      { value: "small", label: "Small" },
                      { value: "standard", label: "Standard" },
                      { value: "high", label: "High" },
                      { value: "highest", label: "Highest" },
                    ]}
                  />
                </div>
              </>
            )}
          </section>

          <UpdatesCard />
          <AgentsCard />
          <SigningCard />

          <section className="card card-pad stack">
            <h3 className="card-title" style={{ margin: 0 }}>Editor shortcuts</h3>
            <div className="stack" style={{ gap: 6 }}>
              {[
                ["⌘ F", "Find and replace in script"],
                ["⌘ S", "Save changes immediately"],
                ["⌘ Enter", "Preview highlighted paragraph or selection"],
                ["Esc", "Close Find bar or open popover"],
              ].map(([k, desc]) => (
                <div key={k} className="row small" style={{ justifyContent: "space-between", padding: "3px 0" }}>
                  <span className="muted">{desc}</span>
                  <code>{k}</code>
                </div>
              ))}
            </div>
          </section>
        </div>
      )}

      {tab === "storage" && <StorageTab />}

      {tab === "appearance" && (
        <div className="stack">
          <section className="card card-pad stack">
            <h3 id="settings-theme-label" className="card-title" style={{ margin: 0 }}>Theme</h3>
            <div className="shape-grid" role="radiogroup" aria-labelledby="settings-theme-label">
              {THEME_OPTIONS.map((opt, idx) => {
                const selected = theme === opt.id;
                return (
                  <button
                    key={opt.id}
                    ref={(el) => { themeBtnRefs.current[idx] = el; }}
                    type="button"
                    role="radio"
                    aria-checked={selected}
                    tabIndex={selected ? 0 : -1}
                    className={`shape-opt ${selected ? "on" : ""}`}
                    onClick={() => setTheme(opt.id)}
                    onKeyDown={(e) => onThemeKeyDown(e, idx)}
                  >
                    <span className="shape-ico" aria-hidden="true"><opt.icon size={17} /></span>
                    <span className="shape-meta">
                      <strong>{opt.title}</strong>
                    </span>
                  </button>
                );
              })}
            </div>
          </section>

          <section className="card card-pad stack">
            <h3 className="card-title" style={{ margin: 0 }}>Audiobook cards</h3>

            <div className="field">
              <span>Card shape</span>
              <div className="shape-grid" role="radiogroup" aria-label="Audiobook card shape">
                {([
                  { id: "vertical", title: "Vertical", desc: "Portrait cards in a grid", icon: LayoutGrid },
                  { id: "horizontal", title: "Horizontal", desc: "Compact horizontal tiles in a grid", icon: Rows3 },
                ] as { id: ShelfShape; title: string; desc: string; icon: typeof LayoutGrid }[]).map((opt) => (
                  <button key={opt.id} type="button" role="radio" aria-checked={shape === opt.id}
                    className={`shape-opt ${shape === opt.id ? "on" : ""}`}
                    onClick={() => setShape(opt.id)}>
                    <span className="shape-ico"><opt.icon size={17} /></span>
                    <span className="shape-meta">
                      <strong>{opt.title}</strong>
                      <span className="small muted">{opt.desc}</span>
                    </span>
                  </button>
                ))}
              </div>
            </div>

            <div className="slider">
              <div className="slider-top">
                <b>Card size</b>
                <em>{SIZE_LABELS[size]}</em>
              </div>
              <input
                type="range"
                min={0}
                max={2}
                step={1}
                value={sizeIndex < 0 ? 0 : sizeIndex}
                aria-label="Audiobook card size"
                aria-valuetext={SIZE_LABELS[size]}
                onChange={(e) => setSize(SHELF_SIZES[Number(e.target.value)] || "small")}
              />
              <div className="size-stops">
                {SHELF_SIZES.map((sz) => (
                  <button key={sz} type="button" className={`size-stop ${size === sz ? "on" : ""}`}
                    onClick={() => setSize(sz)}>
                    {SIZE_LABELS[sz]}
                  </button>
                ))}
              </div>
            </div>

            <div className="appearance-preview">
              <div className="small faint" style={{ marginBottom: 10 }}>Live preview</div>
              <div className={`shelf preview-shelf shape-${shape} size-${size}`}>
                {[
                  { title: "The Night Watch", meta: "12 chapters · 4 characters · M4B", status: "Finished", ok: true, edited: "Edited today" },
                  { title: "Glass Harbor", meta: "6 chapters · 2 characters", status: "Draft", ok: false, edited: "Edited yesterday" },
                ].map((sample) => (
                  <div key={sample.title} className="book-card" style={{ pointerEvents: "none" }}>
                    <div className="book-cover">
                      <span className="cover-title">{sample.title}</span>
                    </div>
                    <div className="book-card-body">
                      <span className="book-name">{sample.title}</span>
                      <div className="small muted">{sample.meta}</div>
                      <div className="card-foot">
                        <span className={`chip ${sample.ok ? "ok" : ""}`}>{sample.status}</span>
                        <span className="small faint">{sample.edited}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>
        </div>
      )}

      {tab === "spending" && <SpendingTab />}

      {tab === "connections" && <Connections embedded focusGroup={focusGroup} focusNonce={tabNonce} />}
    </div>
  );
}

function StorageTab() {
  const toast = useToast();
  const info = useLoad(api.storage);
  const [cleaning, setCleaning] = useState<string | null>(null);

  const clean = async (target: "previews" | "uploads" | "orphans" | "history", label: string) => {
    setCleaning(target);
    try {
      const r = await api.cleanStorage(target);
      toast(r.removed ? `Cleaned ${label}: freed ${fmtBytes(r.freed)} (${r.removed} item${r.removed === 1 ? "" : "s"}).` : `Nothing to clean in ${label}.`);
      info.reload();
    } catch (e) {
      toast((e as Error).message, true);
    } finally {
      setCleaning(null);
    }
  };

  const d: StorageBreakdown | null = info.data;
  const totalUsed = d ? d.books + d.orphans.bytes + d.previews.bytes + d.uploads.bytes + d.history.bytes + d.sounds + d.models : 0;

  return (
    <div className="stack">
      <section className="card card-pad stack">
        <div className="row">
          <div className="grow">
            <h3 className="card-title" style={{ marginBottom: 4 }}>Disk usage</h3>
            <p className="small muted" style={{ margin: 0 }}>
              {d ? `${fmtBytes(totalUsed)} used by Babble · ${fmtBytes(d.free)} free on this Mac.` : "Measuring Babble's folders…"}
            </p>
          </div>
          <button className="btn sm ghost" onClick={info.reload}><RotateCcw size={13} />Refresh</button>
        </div>

        {info.error && <div className="small error-text">{info.error}</div>}
        {!d && !info.error && <div className="small muted row"><Loader2 size={14} className="spin" /> Calculating…</div>}

        {d && (
          <div className="stack" style={{ gap: 10 }}>
            <div className="storage-row">
              <div className="grow">
                <div className="small" style={{ fontWeight: 500 }}>Voice previews cache</div>
                <div className="small muted">{d.previews.count} cached preview WAV{d.previews.count === 1 ? "" : "s"} (regenerated on demand)</div>
              </div>
              <span className="small tnum">{fmtBytes(d.previews.bytes)}</span>
              <button className="btn sm" disabled={!d.previews.count || cleaning !== null}
                onClick={() => clean("previews", "voice previews")}>
                {cleaning === "previews" ? <Loader2 size={13} className="spin" /> : <Trash2 size={13} />}Clear
              </button>
            </div>

            <div className="storage-row">
              <div className="grow">
                <div className="small" style={{ fontWeight: 500 }}>Temporary uploads</div>
                <div className="small muted">{d.uploads.count} uploaded file{d.uploads.count === 1 ? "" : "s"} (active book covers and saved transcript audio are kept)</div>
              </div>
              <span className="small tnum">{fmtBytes(d.uploads.bytes)}</span>
              <button className="btn sm" disabled={!d.uploads.count || cleaning !== null}
                onClick={() => clean("uploads", "uploads")}>
                {cleaning === "uploads" ? <Loader2 size={13} className="spin" /> : <Trash2 size={13} />}Clear
              </button>
            </div>

            <div className="storage-row">
              <div className="grow">
                <div className="small" style={{ fontWeight: 500 }}>Orphaned audiobook folders</div>
                <div className="small muted">{d.orphans.count} audio folder{d.orphans.count === 1 ? "" : "s"} from deleted books (active cloud runs are never touched)</div>
              </div>
              <span className="small tnum">{fmtBytes(d.orphans.bytes)}</span>
              <button className="btn sm" disabled={!d.orphans.count || cleaning !== null}
                onClick={() => clean("orphans", "orphaned book folders")}>
                {cleaning === "orphans" ? <Loader2 size={13} className="spin" /> : <Trash2 size={13} />}Clear
              </button>
            </div>

            <div className="storage-row">
              <div className="grow">
                <div className="small" style={{ fontWeight: 500 }}>Recent Text to Speech clips</div>
                <div className="small muted">{d.history.count} WAV file{d.history.count === 1 ? "" : "s"} in Text to Speech history</div>
              </div>
              <span className="small tnum">{fmtBytes(d.history.bytes)}</span>
              <button className="btn sm" disabled={!d.history.count || cleaning !== null}
                onClick={() => clean("history", "speech history")}>
                {cleaning === "history" ? <Loader2 size={13} className="spin" /> : <Trash2 size={13} />}Clear
              </button>
            </div>

            <div className="storage-row">
              <div className="grow">
                <div className="small" style={{ fontWeight: 500 }}>Active audiobooks</div>
                <div className="small muted">Chapters, cached lines, and exports for books on your shelf</div>
              </div>
              <span className="small tnum">{fmtBytes(d.books)}</span>
            </div>

            <div className="storage-row">
              <div className="grow">
                <div className="small" style={{ fontWeight: 500 }}>Sound library</div>
                <div className="small muted">Imported packs, online sounds, and AI-generated effects &amp; music</div>
              </div>
              <span className="small tnum">{fmtBytes(d.sounds)}</span>
            </div>

            <div className="storage-row">
              <div className="grow">
                <div className="small" style={{ fontWeight: 500 }}>Downloaded AI models</div>
                <div className="small muted">Manage individual model weights on the Models page</div>
              </div>
              <span className="small tnum">{fmtBytes(d.models)}</span>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
