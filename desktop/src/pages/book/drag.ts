import { useEffect, useRef, useState, type DragEvent, type RefObject } from "react";

/** Dragging a sound from a book tool onto the story: where it's dropped is where it starts.
 *  The tag goes in at the drop point, never inside a word or another tag. A tag inside a sentence leaves the
 *  sentence whole when it's spoken: the engine starts the sound on the word after the tag. */

export const DRAG_TYPE = "application/x-narrator-tag";
let dragging: { tag: string; name: string } | null = null;
const watchers = new Set<(on: boolean) => void>();
const tell = (on: boolean) => watchers.forEach((f) => f(on));

/** Props that make an element draggable as a sound: spread them on the row. */
export function soundDrag(tag: string, name: string) {
  return {
    draggable: true,
    title: "Drag into the text where it should start",
    onDragStart: (e: DragEvent) => {
      dragging = { tag, name };
      e.dataTransfer.setData(DRAG_TYPE, tag);
      e.dataTransfer.effectAllowed = "copy";
      tell(true);
    },
    onDragEnd: () => { dragging = null; tell(false); },
  };
}

const WORD = /[\p{L}\p{N}'’-]/u;

/** Where a tag may go near `at`: after a {sound} or [speaker] tag it's inside, at the start of a word it's
 *  inside, after closing punctuation it's just before, after a [Speaker] tag that starts the line, and on the
 *  line after a ## heading. */
export function snapDrop(s: string, at: number): number {
  at = Math.max(0, Math.min(s.length, at));
  for (const [open, close] of [["{", "}"], ["[", "]"]]) {
    const o = s.lastIndexOf(open, at - 1);
    if (o >= 0 && o < at) {
      const c = s.indexOf(close, o);
      const nl = s.indexOf("\n", o);
      if (c >= at && (nl < 0 || c < nl)) at = c + 1;
    }
  }
  const lineStart = s.lastIndexOf("\n", at - 1) + 1;
  if (/^[ \t]*#/.test(s.slice(lineStart, at + 1))) {
    const end = s.indexOf("\n", at);
    return end < 0 ? s.length : end + 1;
  }
  while (at > 0 && at < s.length && WORD.test(s[at - 1]) && WORD.test(s[at])) at--;
  while (at < s.length && at > 0 && !/\s/.test(s[at - 1]) && /[.,!?;:…”’"')\]]/.test(s[at])) at++;
  if (!s.slice(lineStart, at).trim()) {
    const speaker = /^[ \t]*\[[^\]\n]*\]/.exec(s.slice(at));
    if (speaker) at += speaker[0].length;
  }
  return at;
}

const FIRST_WORD = /[\p{L}\p{N}]['’\p{L}\p{N}-]*/u;
const noTags = (t: string) => t.replace(/\{[^{}]*\}|\[[^\]]*\]/g, " ");

/** When a sound put at `at` plays, in words: "starts on “door”" (a word on the same line), "starts before
 *  “Nobody”" (the next line with words), or "plays at the end of the chapter". */
export function startLabel(s: string, at: number): string {
  const lines = s.slice(at).split("\n");
  const here = FIRST_WORD.exec(noTags(lines[0]))?.[0];
  if (here) return `starts on “${here}”`;
  for (const line of lines.slice(1)) {
    if (/^[ \t]*#/.test(line)) return "plays at the end of the chapter";
    const w = FIRST_WORD.exec(noTags(line))?.[0];
    if (w) return `starts before “${w}”`;
  }
  return "plays at the very end";
}

/** The story with `tag` put in at (or snapped near) `at`, spaced like ordinary words. */
export function placeTag(s: string, at: number, tag: string) {
  at = snapDrop(s, at);
  const left = s.slice(0, at), right = s.slice(at);
  const onHeading = /^[ \t]*#/.test(left.slice(left.lastIndexOf("\n") + 1));
  const opens = /[“‘(\[]$/.test(left) || /(^|\s)["']$/.test(left); // just after an opening quote or bracket
  const lpad = onHeading ? "\n" : !left || /\s$/.test(left) || opens ? "" : " ";
  const rpad = !right || /^\s/.test(right) || /^[.,!?;:…”’")\]]/.test(right) ? "" : " ";
  const start = at + lpad.length;
  return { story: left + lpad + tag + rpad + right, start, end: start + tag.length, label: startLabel(s, at) };
}

// ---------- finding the letter under the pointer in a text box ----------
// A text box can't say which letter is at a point, so an invisible copy of it, laid out the same way, answers.

function mirrorOf(ta: HTMLTextAreaElement) {
  const cs = getComputedStyle(ta);
  const r = ta.getBoundingClientRect();
  const div = document.createElement("div");
  for (const p of ["font-family", "font-size", "font-weight", "font-style", "font-variant", "line-height", "letter-spacing",
    "word-spacing", "text-indent", "text-transform", "tab-size", "padding-top", "padding-right", "padding-bottom",
    "padding-left", "word-break"]) div.style.setProperty(p, cs.getPropertyValue(p));
  Object.assign(div.style, {
    position: "fixed", left: `${r.left + ta.clientLeft}px`, top: `${r.top + ta.clientTop}px`,
    width: `${ta.clientWidth}px`, height: `${ta.clientHeight}px`, boxSizing: "border-box", margin: "0", border: "0",
    whiteSpace: "pre-wrap", overflowWrap: "break-word", overflow: "hidden", color: "transparent", background: "none",
    visibility: "hidden", pointerEvents: "none", zIndex: "2147483647",
  });
  div.textContent = ta.value + "​";
  div.setAttribute("aria-hidden", "true");
  document.body.appendChild(div);
  return div;
}

function offsetAt(div: HTMLDivElement, ta: HTMLTextAreaElement, x: number, y: number): number | null {
  div.scrollTop = ta.scrollTop;
  div.style.visibility = "visible";
  div.style.pointerEvents = "auto";
  let node: Node | null = null, off = 0;
  const d = document as Document & {
    caretRangeFromPoint?: (x: number, y: number) => Range | null;
    caretPositionFromPoint?: (x: number, y: number) => { offsetNode: Node; offset: number } | null;
  };
  try {
    if (d.caretRangeFromPoint) { const r = d.caretRangeFromPoint(x, y); if (r) { node = r.startContainer; off = r.startOffset; } }
    else if (d.caretPositionFromPoint) { const p = d.caretPositionFromPoint(x, y); if (p) { node = p.offsetNode; off = p.offset; } }
  } finally {
    div.style.visibility = "hidden";
    div.style.pointerEvents = "none";
  }
  if (node === div) return off > 0 ? ta.value.length : 0;
  if (!node || node !== div.firstChild) return null;
  return Math.min(off, ta.value.length);
}

function caretRect(div: HTMLDivElement, at: number) {
  const node = div.firstChild as Text;
  const range = document.createRange();
  range.setStart(node, Math.min(at, node.length));
  range.collapse(true);
  return range.getClientRects()[0] ?? range.getBoundingClientRect();
}

export type DropMark = { x: number; y: number; h: number; label: string };

/** The text box's side of dragging sounds in: a marker where the sound will start, then `place` on drop. */
export function useTagDrop(ref: RefObject<HTMLTextAreaElement | null>, place: (at: number, tag: string, name: string) => void) {
  const [ready, setReady] = useState(false);
  const [mark, setMark] = useState<DropMark | null>(null);
  const mirror = useRef<HTMLDivElement | null>(null);
  const last = useRef({ x: -1, y: -1, scroll: -1, at: -1 });
  const clear = () => {
    mirror.current?.remove();
    mirror.current = null;
    last.current = { x: -1, y: -1, scroll: -1, at: -1 };
    setMark(null);
  };
  useEffect(() => {
    const on = (v: boolean) => { setReady(v); if (!v) clear(); };
    // a drag that ends anywhere ends the sound drag too, even when the row it began on is gone by then
    // (otherwise a later drag of selected text would be taken for a sound)
    const end = () => { dragging = null; on(false); };
    watchers.add(on);
    window.addEventListener("dragend", end);
    window.addEventListener("drop", end);
    return () => { watchers.delete(on); window.removeEventListener("dragend", end); window.removeEventListener("drop", end); clear(); };
  }, []);

  const over = (e: DragEvent<HTMLTextAreaElement>) => {
    const ta = ref.current;
    if (!dragging || !ta) return; // not a sound: the browser handles it (files, text)
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
    const r = ta.getBoundingClientRect();
    if (e.clientY < r.top + 36) ta.scrollTop -= 14;
    else if (e.clientY > r.bottom - 36) ta.scrollTop += 14;
    const L = last.current;
    if (Math.abs(e.clientX - L.x) + Math.abs(e.clientY - L.y) < 3 && ta.scrollTop === L.scroll) return;
    Object.assign(L, { x: e.clientX, y: e.clientY, scroll: ta.scrollTop });
    if (!mirror.current) mirror.current = mirrorOf(ta);
    const raw = offsetAt(mirror.current, ta, e.clientX, e.clientY);
    if (raw == null) { L.at = -1; setMark(null); return; }
    const at = snapDrop(ta.value, raw);
    L.at = at;
    const c = caretRect(mirror.current, at);
    const box = (ta.offsetParent as HTMLElement | null)?.getBoundingClientRect() ?? r;
    if (c.bottom < r.top || c.top > r.bottom) { setMark(null); return; }
    setMark({ x: c.left - box.left, y: c.top - box.top, h: c.height || 20, label: startLabel(ta.value, at) });
  };
  const leave = () => { last.current.at = -1; setMark(null); };
  const drop = (e: DragEvent<HTMLTextAreaElement>) => {
    const d = dragging, ta = ref.current;
    if (!d || !ta) return;
    e.preventDefault();
    let at = last.current.at;
    if (at < 0) {
      const div = mirror.current || mirrorOf(ta);
      mirror.current = div;
      const raw = offsetAt(div, ta, e.clientX, e.clientY);
      at = raw ?? ta.selectionStart ?? ta.value.length;
    }
    dragging = null;
    clear();
    tell(false);
    place(at, d.tag, d.name);
  };
  return { ready, mark, over, leave, drop };
}
