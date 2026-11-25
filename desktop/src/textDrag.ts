/** Pressing and dragging over text always selects it; text is never picked up and moved by a drag.
 *
 *  macOS lets you drag highlighted text somewhere else. In the app window that used to do nothing, because the
 *  window took every drag for itself; now the page gets drags (so sounds can be dropped into a story), and a
 *  press on highlighted text picked it up and moved it instead of starting a new selection. So: a plain press
 *  on highlighted text drops the old highlight first (a new selection starts there, as you'd expect), and the
 *  only things that can be dragged are the elements made to be dragged (the sound rows). Cut and paste still
 *  move text; double-click, triple-click and shift-click select as usual. */
export function keepTextSelectable() {
  document.addEventListener("mousedown", (e) => {
    if (e.button !== 0 || e.detail !== 1 || e.shiftKey || e.metaKey || e.ctrlKey || e.altKey) return;
    const t = e.target as HTMLElement | null;
    if (!t || t.closest?.("[draggable=true]")) return;
    if (t instanceof HTMLTextAreaElement || (t instanceof HTMLInputElement && t.selectionStart !== null)) {
      const onScrollbar = e.offsetX > t.clientWidth || e.offsetY > t.clientHeight;
      if (!onScrollbar && t.selectionStart !== t.selectionEnd) t.setSelectionRange(t.selectionStart ?? 0, t.selectionStart ?? 0);
      return;
    }
    // highlighted page text (not a field's): only a press on the highlight itself starts over
    const sel = window.getSelection();
    const field = document.activeElement instanceof HTMLTextAreaElement || document.activeElement instanceof HTMLInputElement;
    if (sel && !sel.isCollapsed && !field && !t.closest("button, a, input, textarea, select, [role=button]") && sel.containsNode(t, true)) {
      sel.removeAllRanges();
    }
  }, true);

  // a file dropped where nothing takes files does nothing (the window could otherwise open the file in place of the app)
  const files = (e: DragEvent) => !!e.dataTransfer?.types.includes("Files");
  document.addEventListener("dragover", (e) => {
    if (files(e) && !e.defaultPrevented) { e.preventDefault(); e.dataTransfer!.dropEffect = "none"; }
  });
  document.addEventListener("drop", (e) => { if (files(e) && !e.defaultPrevented) e.preventDefault(); });

  document.addEventListener("dragstart", (e) => {
    const t = e.target as Node | null;
    const el = t instanceof Element ? t : t?.parentElement;
    if (!el?.closest("[draggable=true]")) e.preventDefault();
  }, true);
}
