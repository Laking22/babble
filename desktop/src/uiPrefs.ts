import { useEffect, useState, useSyncExternalStore } from "react";

export type Theme = "dark" | "light";
export type ShelfShape = "vertical" | "horizontal";
export type ShelfSize = "small" | "medium" | "large";

export const SHELF_SIZES: ShelfSize[] = ["small", "medium", "large"];

const THEME_KEY = "babble-theme";
const SHAPE_KEY = "babble-shelf-shape";
const SIZE_KEY = "babble-shelf-size";
const EVENT = "babble-ui-prefs";

type TauriWindowGlobal = {
  __TAURI__?: {
    core?: {
      invoke?: <T>(cmd: string, args?: Record<string, unknown>) => Promise<T>;
    };
    window?: {
      getCurrentWindow?: () => { setTheme?: (t: Theme | null) => Promise<void> };
    };
  };
};

export function readTheme(): Theme {
  try {
    return localStorage.getItem(THEME_KEY) === "light" ? "light" : "dark";
  } catch {
    return "dark";
  }
}

const themeListeners = new Set<() => void>();
let currentTheme: Theme = readTheme();
let lastApplied: Theme | null = null;

export function applyTheme(theme: Theme) {
  if (typeof document === "undefined" || lastApplied === theme) return;
  lastApplied = theme;
  const root = document.documentElement;
  if (theme === "light") root.dataset.theme = "light";
  else delete root.dataset.theme;
  root.style.colorScheme = theme;
  try {
    const tauri = (window as unknown as TauriWindowGlobal).__TAURI__;
    tauri?.window?.getCurrentWindow?.()?.setTheme?.(theme)?.catch?.(() => {});
    tauri?.core?.invoke?.("set_app_icon", { theme })?.catch?.(() => {});
  } catch { /* browser mode or older Tauri global */ }
}

function notifyTheme() {
  themeListeners.forEach((l) => l());
}

export function saveTheme(theme: Theme) {
  currentTheme = theme;
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch { /* private mode / restricted storage */ }
  applyTheme(theme);
  notifyTheme();
}

function onThemeStorage(e: StorageEvent) {
  if (e.key !== null && e.key !== THEME_KEY) return;
  const next = readTheme();
  if (next === currentTheme) return;
  currentTheme = next;
  applyTheme(next);
  notifyTheme();
}

if (typeof window !== "undefined") {
  window.addEventListener("storage", onThemeStorage);
}

function subscribeTheme(listener: () => void) {
  themeListeners.add(listener);
  return () => {
    themeListeners.delete(listener);
  };
}

export function useTheme() {
  const theme = useSyncExternalStore(subscribeTheme, () => currentTheme, () => "dark");
  return { theme, setTheme: saveTheme };
}

export function readShelfShape(): ShelfShape {
  try {
    const v = localStorage.getItem(SHAPE_KEY);
    return v === "horizontal" ? "horizontal" : "vertical";
  } catch {
    return "vertical";
  }
}

export function readShelfSize(): ShelfSize {
  try {
    const v = localStorage.getItem(SIZE_KEY);
    return v === "medium" || v === "large" ? v : "small";
  } catch {
    return "small";
  }
}

export function saveShelfPrefs(patch: { shape?: ShelfShape; size?: ShelfSize }) {
  try {
    if (patch.shape) localStorage.setItem(SHAPE_KEY, patch.shape);
    if (patch.size) localStorage.setItem(SIZE_KEY, patch.size);
  } catch { /* private mode */ }
  window.dispatchEvent(new Event(EVENT));
}

export function useShelfPrefs() {
  const [shape, setShapeState] = useState<ShelfShape>(readShelfShape);
  const [size, setSizeState] = useState<ShelfSize>(readShelfSize);

  useEffect(() => {
    const sync = () => {
      setShapeState(readShelfShape());
      setSizeState(readShelfSize());
    };
    window.addEventListener(EVENT, sync);
    window.addEventListener("storage", sync);
    return () => {
      window.removeEventListener(EVENT, sync);
      window.removeEventListener("storage", sync);
    };
  }, []);

  const setShape = (next: ShelfShape) => {
    setShapeState(next);
    saveShelfPrefs({ shape: next });
  };
  const setSize = (next: ShelfSize) => {
    setSizeState(next);
    saveShelfPrefs({ size: next });
  };

  return { shape, size, setShape, setSize };
}
