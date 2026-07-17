/**
 * Pure logic for the Datasets library page (DatasetListPanel): family grouping,
 * search + kind filtering, and validation of the persisted display preferences.
 * Extracted from the component so it runs under node --test without a DOM.
 */

// Model families offered at creation + section order/labels of the library.
// The library is GROUPED by this family (d.train_type): easier upkeep when
// datasets span several pipelines. Third value = the section emoji.
export const FAMILY_ORDER = [
  ['zimage', 'Z-Image', '🌀'],
  ['sdxl', 'SDXL', '🎨'],
  ['krea', 'Krea 2', '✨'],
  ['flux', 'FLUX.1', '⚡'],
  ['flux2klein', 'FLUX.2 Klein', '🌹'],
];

// A dataset whose train_type this build does not know (e.g. a family added by
// a newer server) must still be reachable — it lands in a trailing section
// instead of silently vanishing from the library.
export const OTHER_FAMILY = ['other', 'Other', '📁'];

// Tile size: 'S' compact list rows (maximum density), 'M' the historical
// photo grid, 'L' large previews. Same 3-step segmented idiom as the
// workspace image grid — no slider (mouse-fragile, no useful granularity).
export const LIBRARY_TILE_SIZES = ['S', 'M', 'L'];

/** Clamp a stored tile-size preference to a valid value (default 'M'). */
export function normalizeTileSize(v) {
  return LIBRARY_TILE_SIZES.includes(v) ? v : 'M';
}

/** Parse the persisted {family: 1} collapsed-sections map; any malformed or
 *  non-object payload (old format, manual edit) degrades to "all open". */
export function normalizeCollapsedMap(raw) {
  try {
    const m = JSON.parse(raw || '{}');
    return m && typeof m === 'object' && !Array.isArray(m) ? m : {};
  } catch {
    return {};
  }
}

/** A dataset's kind with the server default applied ('' → character). */
export function datasetKind(d) {
  return ((d?.kind || '').toLowerCase()) || 'character';
}

/** Search (name or trigger word, case-insensitive) + kind chip filter.
 *  kind 'all' disables the chip filter; query '' matches everything. */
export function datasetMatches(d, query = '', kind = 'all') {
  if (kind !== 'all' && datasetKind(d) !== kind) return false;
  const q = (query || '').trim().toLowerCase();
  if (!q) return true;
  return (d.name || '').toLowerCase().includes(q)
    || (d.trigger_word || '').toLowerCase().includes(q);
}

/** Kinds present in the library, in canonical order — the kind filter chips
 *  only render when at least two coexist (a single-kind library needs none). */
export function kindsPresent(datasets = []) {
  const present = new Set(datasets.map(datasetKind));
  return ['character', 'concept', 'style'].filter((k) => present.has(k));
}

/** Group datasets into ordered non-empty sections:
 *  [{family, label, emoji, items}] following FAMILY_ORDER, with unknown
 *  train_types collected into the trailing OTHER_FAMILY section. */
export function groupDatasets(datasets = []) {
  const known = new Set(FAMILY_ORDER.map(([fam]) => fam));
  const sections = FAMILY_ORDER.map(([family, label, emoji]) => ({
    family,
    label,
    emoji,
    items: datasets.filter((d) => (d.train_type || 'zimage') === family),
  }));
  const [family, label, emoji] = OTHER_FAMILY;
  sections.push({
    family,
    label,
    emoji,
    items: datasets.filter((d) => !known.has(d.train_type || 'zimage')),
  });
  return sections.filter((s) => s.items.length > 0);
}
