/* 🔎 What a run card says when the board is zoomed OUT.
 *
 * The board's job is "every run you have made, at once", and the zoom that
 * actually shows that on a real library is 30-40 %. At 36 % a card is a 95-px
 * stamp whose 11-px title renders at four pixels: the board is showing you
 * everything and telling you nothing, so the only way to find a run is to zoom
 * in on each one in turn — which is the opposite of what a board is for.
 *
 * Below the threshold each card therefore gains ONE counter-scaled label: its
 * run number, at a constant size on screen, over the card it belongs to. The
 * same trick the ✕ and the ✓ box already use (utils/canvasNodeChrome), applied
 * to the one piece of information that makes a stamp identifiable.
 *
 * Deliberately ONE label and not a "simplified card":
 *   • the card itself is unchanged, so nothing about hovering, comparing,
 *     dragging or the in-card lineage graph (which passes no scale at all)
 *     behaves differently at any zoom;
 *   • a second rendering of a card would be a second thing to keep in step with
 *     the first, and the board already learned that lesson with its edges.
 *
 * The label is `pointer-events: none`. The card under it must keep every
 * gesture it has — a legibility aid that eats clicks is a regression.
 */

/** Below this board scale a card's own text is unreadable and the badge takes
 *  over. 0.55 rather than 0.5: at 0.5 exactly, an 11-px title is 5.5 px, which
 *  is already past useless — the threshold has to be above the point where it
 *  breaks, not at it. */
export const LOW_ZOOM_THRESHOLD = 0.55;

/** …and below this, even the badge cannot help: the card is a few pixels wide
 *  and a legible badge would be wider than the card it labels, so nothing is
 *  drawn rather than a row of overlapping blobs. */
export const MIN_LABEL_ZOOM = 0.08;

export function isLowZoom(boardScale) {
  const s = Number(boardScale);
  return Number.isFinite(s) && s > 0 && s < LOW_ZOOM_THRESHOLD;
}

/** Should this board draw the low-zoom labels at all? */
export function showsZoomLabels(boardScale) {
  const s = Number(boardScale);
  return isLowZoom(s) && s >= MIN_LABEL_ZOOM;
}

/**
 * The counter-scale for a low-zoom label, in BOARD units, capped so it never
 * grows wider than the card it sits on.
 *
 * `cardW` is the card's width in board units; `share` is how much of it the
 * label may claim. The cap is the whole point: 1/scale alone at 10 % zoom draws
 * a label ten times the size of its card, which is not a legibility aid, it is
 * a board covered in badges.
 */
export function zoomLabelScale(boardScale, cardW = 264, { share = 0.9, base = 64 } = {}) {
  const s = Number(boardScale);
  if (!Number.isFinite(s) || s <= 0) return 1;
  const wanted = 1 / s;
  const w = Number(cardW);
  const cap = Number.isFinite(w) && w > 0 ? Math.max(1, (w * share) / base) : Infinity;
  return Math.min(Math.max(1, wanted), cap);
}

/** What the badge says. The run number, and the dataset when the lane's own
 *  title has gone unreadable too — at 20 % a lane header is 3 px tall, so the
 *  badge is the only place left that can say WHOSE run this is. */
export function zoomLabelText(node, laneName, boardScale) {
  const rid = node?.record_id;
  const run = rid == null ? 'run' : `#${rid}`;
  const s = Number(boardScale);
  if (Number.isFinite(s) && s < 0.3 && laneName) return `${laneName} ${run}`;
  return run;
}
