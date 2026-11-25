import { useEffect, useMemo, useRef, useState } from "react";
import { BookOpen, Search } from "lucide-react";
import { projectApi, type ProjectSummary } from "../api";
import type { PageId } from "../App";

export type PaletteItem = {
  id: string;
  label: string;
  group: string;
  run: () => void;
};

type Props = {
  onPage: (page: PageId) => void;
  onOpenBook: (id: string) => void;
  onConnections: (section?: string) => void;
};

const NAV_ITEMS: { id: PageId; label: string }[] = [
  { id: "speech", label: "Text to Speech" },
  { id: "transcribe", label: "Speech to Text" },
  { id: "book", label: "Audiobooks" },
  { id: "voices", label: "Voices" },
  { id: "sounds", label: "Sounds" },
  { id: "library", label: "Library" },
  { id: "train", label: "Train a voice" },
  { id: "activity", label: "Activity" },
  { id: "models", label: "Models" },
  { id: "settings", label: "Settings" },
];

export function CommandPalette({ onPage, onOpenBook, onConnections }: Props) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [idx, setIdx] = useState(0);
  const [books, setBooks] = useState<ProjectSummary[]>([]);
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((v) => !v);
      } else if (e.key === "Escape") {
        setOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    if (!open) return;
    setQuery("");
    setIdx(0);
    projectApi.list().then(setBooks).catch(() => {});
    requestAnimationFrame(() => inputRef.current?.focus());
  }, [open]);

  const items = useMemo<PaletteItem[]>(() => {
    const list: PaletteItem[] = [
      ...NAV_ITEMS.map((p) => ({
        id: `page:${p.id}`,
        label: p.label,
        group: "Pages",
        run: () => { onPage(p.id); setOpen(false); },
      })),
      {
        id: "conn:all",
        label: "Connections",
        group: "Settings",
        run: () => { onConnections(); setOpen(false); },
      },
      ...books.map((b) => ({
        id: `book:${b.id}`,
        label: b.name,
        group: "Audiobooks",
        run: () => { onOpenBook(b.id); setOpen(false); },
      })),
    ];
    const q = query.trim().toLowerCase();
    if (!q) return list;
    return list.filter((it) => it.label.toLowerCase().includes(q) || it.group.toLowerCase().includes(q));
  }, [books, query, onPage, onOpenBook, onConnections]);

  useEffect(() => { setIdx(0); }, [query, open]);

  if (!open) return null;

  const choose = (item?: PaletteItem) => {
    if (item) item.run();
  };

  return (
    <div className="modal-backdrop" onClick={() => setOpen(false)} role="dialog" aria-label="Command palette">
      <div className="modal" style={{ width: "min(520px, 100%)", padding: 0, overflow: "hidden" }} onClick={(e) => e.stopPropagation()}>
        <div className="row" style={{ padding: "10px 14px", borderBottom: "1px solid var(--border)" }}>
          <Search size={15} className="muted" />
          <input
            ref={inputRef}
            className="input bare grow"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "ArrowDown") { e.preventDefault(); setIdx((i) => Math.min(i + 1, Math.max(items.length - 1, 0))); }
              else if (e.key === "ArrowUp") { e.preventDefault(); setIdx((i) => Math.max(i - 1, 0)); }
              else if (e.key === "Enter") { e.preventDefault(); choose(items[idx]); }
            }}
            placeholder="Jump to…"
          />
          <span className="small faint">⌘K</span>
        </div>
        <div style={{ maxHeight: 340, overflowY: "auto", padding: 6 }}>
          {items.length === 0 ? (
            <div className="empty small">No matches</div>
          ) : (
            items.map((it, i) => (
              <button
                key={it.id}
                className={`voice-opt ${i === idx ? "sel" : ""}`}
                onMouseEnter={() => setIdx(i)}
                onClick={() => choose(it)}
                style={{ justifyContent: "space-between" }}
              >
                <span className="row" style={{ gap: 8, minWidth: 0 }}>
                  {it.group === "Audiobooks" && <BookOpen size={14} className="muted" />}
                  <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{it.label}</span>
                </span>
                <span className="small faint">{it.group}</span>
              </button>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
