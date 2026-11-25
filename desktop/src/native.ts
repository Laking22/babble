// What differs between the desktop app (Tauri) and a browser tab: saving files and opening other pages.
import { fileUrl } from "./api";

type TauriGlobal = { core: { invoke: <T>(cmd: string, args?: Record<string, unknown>) => Promise<T> } };
const tauri = (): TauriGlobal | undefined => (window as unknown as { __TAURI__?: TauriGlobal }).__TAURI__;

export const isDesktop = () => !!tauri();
export const isMac = () => /Mac/i.test(navigator.platform || navigator.userAgent);
export const isMacDesktop = () => isDesktop() && isMac();

/** Save a file the engine made: the app copies it to Downloads and shows it in Finder/Explorer; a browser downloads it. */
export async function saveFile(path: string): Promise<string> {
  const t = tauri();
  if (t) {
    const dest = await t.core.invoke<string>("save_file", { path });
    return `Saved to Downloads as ${dest.split(/[\\/]/).pop()}.`;
  }
  const a = document.createElement("a");
  a.href = fileUrl(path);
  a.download = path.split(/[\\/]/).pop() || "audio";
  document.body.appendChild(a);
  a.click();
  a.remove();
  return "Downloading…";
}

/** Open a web page: in the browser from the app, in a new tab from a browser. */
export async function openExternal(url: string) {
  const t = tauri();
  if (t) await t.core.invoke("open_external", { url });
  else window.open(url, "_blank", "noopener");
}

/** Mark the page so the styles can make room for the macOS window's traffic lights. */
export function markDesktop() {
  if (isMacDesktop()) document.documentElement.classList.add("desktop");
}

export type EngineStatus = {
  state: "starting" | "running" | "failed";
  phase?: string | null;
  error: string | null;
  log_path: string;
  log_tail: string;
};

/** Read native engine launch status (so Splash shows startup errors immediately instead of spinning forever). */
export async function engineStatus(): Promise<EngineStatus | null> {
  const t = tauri();
  if (!t) return null;
  try {
    return await t.core.invoke<EngineStatus>("engine_status");
  } catch {
    return null;
  }
}

/** Reveal Babble.log in Finder / File Explorer. */
export async function openLogFolder(): Promise<void> {
  const t = tauri();
  if (t) await t.core.invoke("open_log_folder");
}

/** Retry first-run bootstrap and engine launch after a failure (e.g. reconnecting internet). */
export async function retryEngine(): Promise<void> {
  const t = tauri();
  if (t) await t.core.invoke("retry_engine");
}

/** Restart to update: the app quits, puts the downloaded version in its place, and opens again. */
export async function applyUpdate(): Promise<void> {
  const t = tauri();
  if (!t) throw new Error("Updates are installed by the Babble app.");
  await t.core.invoke("apply_update");
}
