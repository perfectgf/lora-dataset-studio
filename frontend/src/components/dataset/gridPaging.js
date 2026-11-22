/**
 * Client-side paging for the dataset image grid.
 *
 * WHY — the grid used to render EVERY image of the filtered list at once. On a
 * 6 211-image dataset (all kept, all captioned) that is ~148 000 DOM nodes:
 * 6 211 <img>, 6 212 caption <textarea>, ~60 000 buttons, and a page 448 000 px
 * tall. Measured on that dataset, scrolling ran at ~41 ms/frame (~21 fps) while
 * the same build on a 300-image dataset ran at 6 ms/frame (~170 fps) — same
 * code, same machine, same scroll distance, images already cached. Hiding the
 * thumbnails changed nothing (41 → 41 ms) and only 42 of the 6 211 images were
 * ever requested, so `loading="lazy"` was already doing its job: the cost is the
 * NUMBER OF TILES, not the pictures in them.
 *
 * The Bank workspace — the other wall-of-images surface, built for 24 000-image
 * banks — already answers this with pages, so the dataset grid answers it the
 * same way. The difference is only where the slicing happens: the Bank asks the
 * server for a window, while the dataset workspace already holds every row in
 * the payload it polls, so paging here is pure arithmetic and needs no request.
 *
 * Selection deliberately does NOT page. It is a Set of ids owned above this
 * module, so "select all" keeps meaning "every image the current filters show,
 * all pages" (the Bank's own wording) and a selection started on page 3 is still
 * there on page 7. Same for auto-triage, the caption counters and every bulk
 * action: they read the full filtered list, never this slice.
 */

// How many tiles make a page. It was 500 — the most this machine scrolls at
// full frame rate — which settled the performance question and left the other
// one open: a 156-image dataset was a single page 10.9 screens tall (measured
// 2026-11-22), and "where am I in this set" had no answer but the scrollbar.
// A tile here is tall (the picture, ✓/✕, the caption field), so the page is
// counted in rows: 48 is eight rows of six at the default tile size on a wide
// window, about three screens, and it divides evenly by every column count the
// grid uses (2, 3, 4, 6, 8) so a full page never ends on a ragged row.
// The pager is numbered (components/common/Pager), so a long set is crossed by
// jumping, not by pressing Next a hundred times. Selection, auto-triage and
// every bulk action still read the whole filtered list (see above).
export const GRID_PAGE_SIZE = 48;

/** How many pages `total` items need (never 0 — an empty list is one empty page). */
export function pageCount(total, size = GRID_PAGE_SIZE) {
  const n = Math.max(0, Math.floor(Number(total) || 0));
  const s = Math.max(1, Math.floor(Number(size) || GRID_PAGE_SIZE));
  return Math.max(1, Math.ceil(n / s));
}

/** Bring `page` back inside [0, last] — a filter or a delete can shrink the list
 *  under the page the user is standing on, and a blank grid would read as data
 *  loss rather than as a page that no longer exists. */
export function clampPage(page, total, size = GRID_PAGE_SIZE) {
  const p = Number.isFinite(Number(page)) ? Math.floor(Number(page)) : 0;
  return Math.min(Math.max(0, p), pageCount(total, size) - 1);
}

/** Which page holds the item at `index`. The lightbox's ⟨ / ⟩ walk the whole
 *  filtered list, so the grid underneath has to be told where that landed. */
export function pageOfIndex(index, size = GRID_PAGE_SIZE) {
  const i = Number.isFinite(Number(index)) ? Math.floor(Number(index)) : -1;
  if (i < 0) return 0;
  const s = Math.max(1, Math.floor(Number(size) || GRID_PAGE_SIZE));
  return Math.floor(i / s);
}

/**
 * The slice actually rendered, plus everything the pager needs to describe it.
 * `from`/`to` are 1-based and inclusive (the "1–500 of 6211" label); `paged` is
 * false when the whole list fits on one page, which is when the pager hides
 * itself instead of showing a dead "1–12 of 12".
 */
export function pageSlice(images, page, size = GRID_PAGE_SIZE) {
  const list = Array.isArray(images) ? images : [];
  const s = Math.max(1, Math.floor(Number(size) || GRID_PAGE_SIZE));
  const current = clampPage(page, list.length, s);
  const start = current * s;
  const items = list.slice(start, start + s);
  return {
    items,
    page: current,
    pages: pageCount(list.length, s),
    size: s,
    total: list.length,
    start,
    from: items.length ? start + 1 : 0,
    to: start + items.length,
    paged: list.length > s,
  };
}
