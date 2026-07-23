/** 🗃️ Bank card preview strip — pure helpers (no JSX, so `node --test` can run them).
 *
 * The backend hands each bank the ids of its first few non-rejected images
 * (`preview_ids`, id order = inventory order, stable across reloads). The card
 * renders a fixed-width strip so every card is the same height whether the bank
 * holds 3 images or 3 000. */

export const PREVIEW_SLOTS = 5

/** Pad/trim `ids` to exactly `slots` entries — missing ones become null so the
 * strip keeps its shape (an empty tile, never a broken image). */
export function previewSlots(ids, slots = PREVIEW_SLOTS) {
  const src = Array.isArray(ids) ? ids : []
  return Array.from({ length: slots }, (_, i) => (i < src.length ? src[i] : null))
}

/** How many images the strip does NOT show, for the "+N" overflow badge.
 * 0 (no badge) when the bank is empty, unknown, or fully visible. */
export function hiddenCount(total, ids, slots = PREVIEW_SLOTS) {
  const shown = Math.min(Array.isArray(ids) ? ids.length : 0, slots)
  const n = Number(total)
  if (!Number.isFinite(n) || n <= shown) return 0
  return n - shown
}
