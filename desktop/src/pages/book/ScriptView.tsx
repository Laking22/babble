import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, Gauge, Loader2, PauseCircle, Play, RefreshCw } from "lucide-react";
import { type Chapter, type ParaTiming } from "../../api";
import { DRAG_TYPE, placeTag } from "./drag";

const PACE_OPTS = [
  { id: "", label: "Pace" },
  { id: "slow", label: "Slow" },
  { id: "fast", label: "Fast" },
];
const PAUSE_OPTS = [
  { id: "", label: "Pause" },
  { id: "0.5", label: "+0.5s" },
  { id: "1.0", label: "+1.0s" },
  { id: "1.5", label: "+1.5s" },
  { id: "2.0", label: "+2.0s" },
];

const PACE_RE = /\{\s*pace\s*:\s*([^{}]*?)\s*\}/i;
const END_PAUSE_RE = /\s*\{\s*pause\s*:\s*([^{}]*?)\s*\}\s*$/i;
const HEADING_LINE_RE = /^[ \t]*##[ \t]+(.+?)[ \t]*$/;

export type StoryBlock = {
  raw: string;
  start: number;
  end: number;
  chapter: number;
  para: number; // 0-based paragraph index inside its chapter (-1 for heading)
  heading?: string;
};

/** Split the story into heading and paragraph blocks matching backend book.split_chapters / in_parts / cast.parse.
 *  Handles paragraphs of any size (1 word to 10,000+ chars), headings with or without a blank line after them,
 *  and very long chapters split into parts via `chapters[i].at`. */
export function parseStoryBlocks(story: string, chapters?: Chapter[]): StoryBlock[] {
  const src = story || "";
  if (!src.trim()) return [];

  // First extract raw segments (## headings on their own line, and blank-line-separated paragraphs) with exact offsets
  const rawBlocks: { raw: string; start: number; end: number; heading?: string }[] = [];
  let pos = 0;
  const len = src.length;
  let paraStart = -1;
  let paraEnd = -1;

  const flushPara = () => {
    if (paraStart >= 0 && paraEnd > paraStart) {
      const raw = src.slice(paraStart, paraEnd);
      if (raw.trim()) {
        const ltrim = raw.length - raw.trimStart().length;
        const rtrim = raw.length - raw.trimEnd().length;
        rawBlocks.push({
          raw: raw.trim(),
          start: paraStart + ltrim,
          end: paraEnd - rtrim,
        });
      }
    }
    paraStart = -1;
    paraEnd = -1;
  };

  while (pos <= len) {
    const nl = src.indexOf("\n", pos);
    const lineEnd = nl < 0 ? len : nl;
    const line = src.slice(pos, lineEnd);
    const head = HEADING_LINE_RE.exec(line);
    if (head) {
      flushPara();
      const trimmed = line.trim();
      const ltrim = line.length - line.trimStart().length;
      rawBlocks.push({
        raw: trimmed,
        start: pos + ltrim,
        end: pos + ltrim + trimmed.length,
        heading: head[1].trim(),
      });
    } else if (!line.trim()) {
      flushPara();
    } else {
      if (paraStart < 0) paraStart = pos;
      paraEnd = lineEnd;
    }
    if (nl < 0) break;
    pos = nl + 1;
  }
  flushPara();

  // Map each block to its backend chapter index using `chapters[i].at` when available (handles in_parts splits too)
  const atMap = (chapters || [])
    .filter((c) => typeof c.at === "number")
    .map((c) => ({ index: c.index, at: c.at as number }))
    .sort((a, b) => a.at - b.at);

  const chapterFor = (offset: number, fallbackCh: number): number => {
    if (!atMap.length) return fallbackCh;
    let found = atMap[0].index;
    for (const c of atMap) {
      if (c.at <= offset) found = c.index;
      else break;
    }
    return found;
  };

  const blocks: StoryBlock[] = [];
  let fallbackCh = -1;
  let sawContent = false;
  const paraCountByCh = new Map<number, number>();

  for (const rb of rawBlocks) {
    if (rb.heading != null) {
      fallbackCh = fallbackCh < 0 ? 0 : fallbackCh + 1;
      sawContent = true;
      const ch = chapterFor(rb.start, fallbackCh);
      blocks.push({ ...rb, chapter: ch, para: -1 });
    } else {
      if (!sawContent) {
        fallbackCh = 0;
        sawContent = true;
      }
      const ch = chapterFor(rb.start, Math.max(0, fallbackCh));
      const pi = paraCountByCh.get(ch) ?? 0;
      paraCountByCh.set(ch, pi + 1);
      blocks.push({ ...rb, chapter: ch, para: pi });
    }
  }
  return blocks;
}

/** Update a single paragraph's {pace: ...} or trailing {pause: ...} tag cleanly without touching anything else. */
export function patchParagraphTag(story: string, block: StoryBlock, kind: "pace" | "pause", value: string): string {
  let body = block.raw;
  if (kind === "pace") {
    body = body.replace(PACE_RE, "").replace(/^[ \t]+/, "");
    if (value) body = `{pace: ${value}} ${body}`;
  } else {
    body = body.replace(END_PAUSE_RE, "").trimEnd();
    if (value) body = `${body} {pause: ${value}}`;
  }
  return story.slice(0, block.start) + body + story.slice(block.end);
}

/** Single editable paragraph/heading line: keeps the user's native text cursor intact while typing
 *  and syncs external updates (Undo, Director tags, pacing) cleanly for paragraphs of any length. */
function EditableLine({
  value,
  className,
  onCommit,
  onSplit,
  onMergePrev,
  onSelect,
  onDropTag,
  elRef,
  autoFocus,
  onBlur,
}: {
  value: string;
  className: string;
  onCommit: (next: string) => void;
  onSplit?: (before: string, after: string) => void;
  onMergePrev?: () => void;
  onSelect: () => void;
  onDropTag?: (tag: string, name: string) => void;
  elRef?: (el: HTMLDivElement | null) => void;
  autoFocus?: boolean;
  onBlur?: () => void;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  const lastCommitted = useRef(value);
  const justClicked = useRef(false);
  const [dragOver, setDragOver] = useState(false);

  // Sync external story changes (Undo, AI detect, Director, Pace/Pause) without disrupting active typing
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (value !== lastCommitted.current && el.innerText !== value) {
      el.innerText = value;
      lastCommitted.current = value;
    }
  }, [value]);

  // Initial mount text
  useEffect(() => {
    // (only when it differs: setting it again would throw the text cursor back to the start)
    if (ref.current && ref.current.innerText !== value) ref.current.innerText = value;
    if (autoFocus) ref.current?.focus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div
      ref={(el) => { ref.current = el; elRef?.(el); }}
      onBlur={onBlur}
      contentEditable
      suppressContentEditableWarning
      spellCheck
      className={`${className}${dragOver ? " el-drop" : ""}`}
      onMouseDown={() => {
        justClicked.current = true;
        onSelect();
        requestAnimationFrame(() => { justClicked.current = false; });
      }}
      onFocus={() => {
        if (!justClicked.current) onSelect();
      }}
      onInput={(e) => {
        const raw = e.currentTarget.innerText.replace(/\r\n?/g, "\n");
        const next = raw.trim() === "" ? "" : raw;
        lastCommitted.current = next;
        onCommit(next);
      }}
      onKeyDown={(e) => {
        const el = ref.current;
        const sel = window.getSelection();
        if (!el || !sel || !sel.rangeCount) return;

        if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing && onSplit) {
          e.preventDefault();
          const r = sel.getRangeAt(0);
          const pre = document.createRange();
          pre.selectNodeContents(el);
          pre.setEnd(r.startContainer, r.startOffset);
          const idx = pre.toString().length;
          const full = el.innerText;
          onSplit(full.slice(0, idx).trimEnd(), full.slice(idx).trimStart());
          return;
        }

        if (e.key === "Backspace" && onMergePrev && sel.isCollapsed) {
          const r = sel.getRangeAt(0);
          const pre = document.createRange();
          pre.selectNodeContents(el);
          pre.setEnd(r.startContainer, r.startOffset);
          if (pre.toString().length === 0) {
            e.preventDefault();
            onMergePrev();
          }
        }
      }}
      onDragOver={(e) => {
        if (e.dataTransfer.types.includes(DRAG_TYPE)) {
          e.preventDefault();
          e.dataTransfer.dropEffect = "copy";
          setDragOver(true);
        }
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        setDragOver(false);
        const raw = e.dataTransfer.getData(DRAG_TYPE);
        if (!raw || !onDropTag) return;
        e.preventDefault();
        let tag = raw;
        let name = raw;
        if (raw.startsWith("{") && raw.includes('"tag"')) {
          try {
            const parsed = JSON.parse(raw);
            tag = parsed.tag || raw;
            name = parsed.name || tag;
          } catch { /* plain tag */ }
        }
        onDropTag(tag, name);
      }}
    />
  );
}

/** Unified story editor:
 *  - Edit any paragraph directly in place (clicking places the text cursor right where you clicked).
 *  - Clicking a paragraph also seeks playback to that paragraph's start:
 *      • if currently playing -> continues playing from that paragraph
 *      • if paused -> moves playhead there without auto-playing
 *  - Rendered paragraphs show a subtle light-grey highlight; the active/playing paragraph highlights slightly deeper grey. */
export function ScriptView({
  story,
  chapters,
  activeChapter,
  scrollSignal,
  timings,
  playingPara,
  isPlaying,
  focusedBlock,
  previewingPara,
  busy,
  onFocusPara,
  onPreviewPara,
  onRegenPara,
  onChangeStory,
  onSelectChapter,
  onTagDropped,
}: {
  story: string;
  chapters: Chapter[];
  activeChapter: number;
  scrollSignal: number;
  timings: ParaTiming[];
  playingPara: number | null;
  isPlaying: boolean;
  focusedBlock: { chapter: number; para: number } | null;
  previewingPara?: { chapter: number; para: number } | null;
  busy: boolean;
  onFocusPara: (chapter: number, para: number, timing?: ParaTiming) => void;
  onPreviewPara?: (chapter: number, para: number, rawText: string) => void;
  onRegenPara: (chapter: number, para: number) => void;
  onChangeStory: (next: string) => void;
  onSelectChapter: (chapter: number) => void;
  onTagDropped?: (tag: string, name: string) => void;
}) {
  const blocks = useMemo(() => parseStoryBlocks(story, chapters), [story, chapters]);
  const timingMap = useMemo(() => {
    const map = new Map<number, ParaTiming>();
    for (const t of timings) map.set(t.para, t);
    return map;
  }, [timings]);

  // For massive books (> 300 paragraphs), scope view to the active chapter so editing stays 60fps instant
  const largeBook = blocks.length > 300;
  const visibleBlocks = useMemo(
    () => (largeBook ? blocks.filter((b) => b.chapter === activeChapter) : blocks),
    [blocks, largeBook, activeChapter],
  );

  const activeRef = useRef<HTMLDivElement | null>(null);
  const chapterRefs = useRef<Map<number, HTMLDivElement>>(new Map());

  // Enter opens a new paragraph. An empty one can't live in the story (blank paragraphs aren't paragraphs), so it's
  // a box of its own here until something is typed in it; leaving it empty takes it away again.
  const [draft, setDraft] = useState<{ key: string; at: number; side: "before" | "after" } | null>(null);
  // After a split, a merge or a new paragraph the text cursor goes where it belongs: the paragraph starting at
  // `start` in the story, `caret` letters in. Done after the paragraphs show the new story.
  const lines = useRef<Map<number, HTMLDivElement>>(new Map());
  const focusReq = useRef<{ start: number; caret: number; t: number } | null>(null);
  const focusAt = (start: number, caret: number) => { focusReq.current = { start, caret, t: Date.now() }; };
  useEffect(() => {
    const req = focusReq.current;
    if (!req) return;
    if (Date.now() - req.t > 2500) { focusReq.current = null; return; }
    const el = lines.current.get(req.start);
    if (!el) return;
    focusReq.current = null;
    el.focus();
    placeCaret(el, req.caret);
  });

  // Gently keep currently playing paragraph visible
  useEffect(() => {
    if (isPlaying && playingPara != null && activeRef.current) {
      activeRef.current.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [isPlaying, playingPara, activeChapter]);

  // Scroll to chapter when picked in the Chapters sidebar
  useEffect(() => {
    if (!scrollSignal) return;
    const node = chapterRefs.current.get(activeChapter);
    if (node) node.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [scrollSignal, activeChapter]);

  // Empty story: show a clean paste/write box
  if (!blocks.length) {
    return (
      <textarea
        className="editor"
        value={story}
        onChange={(e) => {
          const v = e.target.value;
          // the first letters typed turn this box into the paragraph editor: keep typing there
          if (v.trim() && !/\n[ \t]*\n/.test(v.trim())) focusAt(v.length - v.trimStart().length, v.trim().length);
          onChangeStory(v);
        }}
        aria-label="Story"
        placeholder={"Paste a story, or import a book.\n\n## Chapter 1\n\n“You're late,” Sarah snapped.\n\n{amb: rain} Tags in curly braces add sound; [Name] tags say who speaks."}
      />
    );
  }

  const updateBlock = (blk: StoryBlock, nextText: string) => {
    const clean = nextText.replace(/\n{3,}/g, "\n\n");
    if (clean === blk.raw) return;
    onChangeStory(story.slice(0, blk.start) + clean + story.slice(blk.end));
  };

  /** Enter: in a heading or at the end of a paragraph a new empty paragraph opens below it, at the very start one
   *  opens above it, and in the middle the paragraph is cut in two there. The cursor goes to the new paragraph. */
  const splitBlock = (blk: StoryBlock, key: string, before: string, after: string) => {
    if (blk.heading != null) return setDraft({ key, at: blk.end, side: "after" });
    if (!after.trim()) {
      if (before !== blk.raw) onChangeStory(story.slice(0, blk.start) + before + story.slice(blk.end));
      return setDraft({ key, at: blk.start + before.length, side: "after" });
    }
    if (!before.trim()) return setDraft({ key, at: blk.start, side: "before" });
    onChangeStory(story.slice(0, blk.start) + `${before}\n\n${after}` + story.slice(blk.end));
    focusAt(blk.start + before.length + 2, 0);
  };

  /** The first thing typed in a new paragraph puts it in the story. */
  const commitDraft = (text: string) => {
    const d = draft;
    const body = text.replace(/^\s+/, "");
    if (!d || !body.trim()) return;
    setDraft(null);
    if (d.side === "after") {
      onChangeStory(story.slice(0, d.at) + "\n\n" + body + story.slice(d.at));
      focusAt(d.at + 2, body.length);
    } else {
      onChangeStory(story.slice(0, d.at) + body + "\n\n" + story.slice(d.at));
      focusAt(d.at, body.length);
    }
  };

  const mergeWithPrev = (prev: StoryBlock, cur: StoryBlock) => {
    const sep = prev.raw && cur.raw ? " " : "";
    const merged = prev.raw + sep + cur.raw;
    onChangeStory(story.slice(0, prev.start) + merged + story.slice(cur.end));
    focusAt(prev.start, prev.raw.length + sep.length);
  };

  const draftRow = (blk: StoryBlock) => (
    <div className="el-row">
      <div className="el-gutter" aria-hidden="true"><span className="el-bar" /></div>
      <div className="el-body">
        <EditableLine
          value=""
          className="el-text el-new"
          autoFocus
          onSelect={() => {}}
          onCommit={commitDraft}
          onSplit={() => {}}
          onMergePrev={() => { setDraft(null); focusAt(blk.start, draft?.side === "after" ? blk.raw.length : 0); }}
          onBlur={() => setDraft(null)}
        />
      </div>
    </div>
  );

  const chHasAudio = (idx: number) => Boolean(chapters[idx]?.file);

  return (
    <div className="el-editor" role="region" aria-label="Story editor">
      {largeBook && chapters.length > 1 && (
        <div className="el-ch-pager">
          <button type="button" className="btn sm ghost" disabled={activeChapter <= 0}
            onClick={() => onSelectChapter(Math.max(0, activeChapter - 1))}>
            <ChevronLeft size={14} />Prev chapter
          </button>
          <span className="small muted">
            Chapter {activeChapter + 1} of {chapters.length}
            {chapters[activeChapter]?.title ? ` · ${chapters[activeChapter].title}` : ""}
          </span>
          <button type="button" className="btn sm ghost" disabled={activeChapter >= chapters.length - 1}
            onClick={() => onSelectChapter(Math.min(chapters.length - 1, activeChapter + 1))}>
            Next chapter<ChevronRight size={14} />
          </button>
        </div>
      )}

      {visibleBlocks.map((blk, vi) => {
        const key = `${blk.chapter}:${blk.para}`;
        const prevBlk = vi > 0 ? visibleBlocks[vi - 1] : null;
        const canMergePrev = blk.para > 0 && prevBlk != null && prevBlk.chapter === blk.chapter && prevBlk.para >= 0;
        const hasAudio = chHasAudio(blk.chapter);
        const isActiveCh = blk.chapter === activeChapter;
        const timing = isActiveCh && blk.para >= 0 ? timingMap.get(blk.para) : undefined;
        const isPlayingHere = isActiveCh && blk.para >= 0 && playingPara === blk.para;
        const isFocusedHere = focusedBlock?.chapter === blk.chapter && focusedBlock?.para === blk.para;
        // While playing, highlight follows live playback only; while paused, highlight shows focused/clicked paragraph
        const isHighlighted = isPlaying && playingPara != null ? isPlayingHere : (isFocusedHere || isPlayingHere);

        const paceVal = (PACE_RE.exec(blk.raw)?.[1] || "").trim().toLowerCase();
        const pauseVal = (END_PAUSE_RE.exec(blk.raw)?.[1] || "").trim().replace(/s$/i, "");

        return (
          <Fragment key={key}>
          {draft?.key === key && draft.side === "before" && draftRow(blk)}
          <div
            ref={(node) => {
              if (blk.heading) {
                if (node) chapterRefs.current.set(blk.chapter, node);
                else chapterRefs.current.delete(blk.chapter);
              }
              if (isPlayingHere) activeRef.current = node;
            }}
            className={`el-row ${blk.heading ? "el-heading-row" : ""} ${isHighlighted ? "active" : ""}`}
          >
            <div className="el-gutter" aria-hidden="true">
              <span className="el-bar" />
            </div>

            <div className="el-body">
              <EditableLine
                value={blk.raw}
                className={`el-text ${blk.heading ? "el-heading-text" : ""} ${hasAudio ? "has-audio" : ""} ${isHighlighted ? "active" : ""}`}
                onSelect={() => {
                  if (blk.para >= 0) onFocusPara(blk.chapter, blk.para, timing);
                  else onSelectChapter(blk.chapter);
                }}
                onCommit={(next) => updateBlock(blk, next)}
                onSplit={(a, b) => splitBlock(blk, key, a, b)}
                elRef={(el) => { if (el) lines.current.set(blk.start, el); else lines.current.delete(blk.start); }}
                onMergePrev={canMergePrev && prevBlk ? () => mergeWithPrev(prevBlk, blk) : undefined}
                onDropTag={(tag, name) => {
                  const placed = placeTag(story, blk.start, tag);
                  onChangeStory(placed.story);
                  onTagDropped?.(tag, name);
                }}
              />

              {blk.para >= 0 && (
                <div className="el-hover-tools" contentEditable={false}>
                  {onPreviewPara && (() => {
                    const isThisPrev = previewingPara?.chapter === blk.chapter && previewingPara?.para === blk.para;
                    return (
                      <button type="button" className="el-tool-btn" disabled={busy}
                        onMouseDown={(e) => { e.preventDefault(); e.stopPropagation(); }}
                        onClick={(e) => { e.stopPropagation(); onPreviewPara(blk.chapter, blk.para, blk.raw); }}
                        title="Preview this paragraph">
                        {isThisPrev ? <><Loader2 size={11} className="spin" />Previewing…</> : <><Play size={11} />Preview</>}
                      </button>
                    );
                  })()}
                  {hasAudio && (
                    <button type="button" className="el-tool-btn" disabled={busy}
                      onMouseDown={(e) => e.preventDefault()}
                      onClick={() => onRegenPara(blk.chapter, blk.para)}
                      title="Regenerate this paragraph and preview with surrounding sentences">
                      <RefreshCw size={12} />Redo
                    </button>
                  )}
                  <label className="el-tool-ctl" title="Paragraph speaking pace">
                    <Gauge size={11} />
                    <select className="el-tool-sel" value={PACE_OPTS.some((o) => o.id === paceVal) ? paceVal : ""}
                      onChange={(e) => onChangeStory(patchParagraphTag(story, blk, "pace", e.target.value))}>
                      {PACE_OPTS.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
                    </select>
                  </label>
                  <label className="el-tool-ctl" title="Pause after this paragraph">
                    <PauseCircle size={11} />
                    <select className="el-tool-sel" value={PAUSE_OPTS.some((o) => o.id === pauseVal) ? pauseVal : ""}
                      onChange={(e) => onChangeStory(patchParagraphTag(story, blk, "pause", e.target.value))}>
                      {PAUSE_OPTS.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
                    </select>
                  </label>
                </div>
              )}
            </div>
          </div>
          {draft?.key === key && draft.side === "after" && draftRow(blk)}
          </Fragment>
        );
      })}
    </div>
  );
}

/** Put the text cursor `at` letters into an editable paragraph (a line break counts as one). */
function placeCaret(el: HTMLElement, at: number) {
  const sel = window.getSelection();
  if (!sel) return;
  const range = document.createRange();
  let left = at;
  const walk = (n: Node): boolean => {
    if (n.nodeType === Node.TEXT_NODE) {
      const len = n.textContent?.length ?? 0;
      if (left <= len) { range.setStart(n, left); return true; }
      left -= len;
      return false;
    }
    if (n.nodeName === "BR") {
      if (left === 0) { range.setStartBefore(n); return true; }
      left -= 1;
      return false;
    }
    for (const c of Array.from(n.childNodes)) if (walk(c)) return true;
    return false;
  };
  if (!walk(el)) { range.selectNodeContents(el); range.collapse(false); }
  else range.collapse(true);
  sel.removeAllRanges();
  sel.addRange(range);
}
