/** The size of a start-frame tile in the Video Test Studio's picker.
 *
 * One dial for the three grids (Bank, Gallery, Dataset clip): a frame is
 * chosen by eye, and at 96 px a face is a smudge. The size lives per browser,
 * under its own key — the same range and step as the concept sources' 🔍,
 * so the two dials feel like one — and is clamped on the way in, because a
 * stored value is as trustworthy as any other input.
 */
export const TILE_MIN = 72
export const TILE_MAX = 300
export const TILE_STEP = 4
export const TILE_DEFAULT = 96
export const TILE_STORAGE_KEY = 'videoStudioTileSize'

/** A tile size that is on the dial: numeric, snapped to the step, in range. */
export function clampTile(value) {
  // Number('') and Number(null) are 0, which would clamp to the smallest tile
  // rather than fall back — so only a number, or a string holding one, counts.
  const n = typeof value === 'number' ? value
    : (typeof value === 'string' && value.trim() !== '' ? Number(value) : NaN)
  if (!Number.isFinite(n)) return TILE_DEFAULT
  const snapped = Math.round(n / TILE_STEP) * TILE_STEP
  return Math.min(TILE_MAX, Math.max(TILE_MIN, snapped))
}

/** localStorage, or nothing at all: this module is imported by node tests,
 * and a browser that blocks site data throws on ACCESS, not only on read —
 * the picker reads the size while it renders, and an exception there would
 * take the whole app down to its crash screen. Owned here, as a default
 * parameter, so no component ever names the store: the tests inject one. */
function defaultStore() {
  try {
    return globalThis.localStorage || null
  } catch { return null }
}

/** The stored size, or the default when nothing (or garbage) is stored, or
 * when the store itself throws — a private window, a blocked origin. */
export function readTile(store = defaultStore()) {
  try {
    const raw = store?.getItem?.(TILE_STORAGE_KEY)
    if (raw == null || raw === '') return TILE_DEFAULT
    return clampTile(raw)
  } catch {
    return TILE_DEFAULT
  }
}

/** Remembers the size for next time; a store that refuses (quota, private
 * mode) costs nothing but the memory — the dial still works for the session,
 * which holds only if the caller sets its state from the VALUE, never by
 * reading the store back. */
export function writeTile(tile, store = defaultStore()) {
  try {
    store?.setItem?.(TILE_STORAGE_KEY, String(clampTile(tile)))
  } catch {
    /* the size lives for the session only */
  }
}

/** How tall the scrolling box is for a tile size: about two rows with their
 * captions, never below the 288 px the Gallery and clip grids had (the Bank's
 * box was 224), never past 640 px — the caller also caps it to the viewport,
 * so a phone keeps its fold. */
export function gridBoxHeight(tile) {
  const t = clampTile(tile)
  return Math.min(640, Math.max(288, 2 * (t + 44)))
}
