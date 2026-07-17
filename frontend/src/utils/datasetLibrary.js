/**
 * Pure logic for the Datasets library page (DatasetListPanel): family grouping,
 * search + kind filtering, and validation of the persisted display preferences.
 * Extracted from the component so it runs under node --test without a DOM.
 */

// Model families offered at creation + section order/labels of the library.
// The library is GROUPED by the family a dataset was ACTUALLY trained in (its
// primary trained family — see primaryFamily/groupDatasets), NOT by the mutable
// d.train_type scalar: train_type is only the training panel's current pick and
// gets rewritten on a mere dropdown change, so grouping on it made a dataset
// jump sections without any real run. This order also decides which trained
// family wins when a dataset has several. Third value = the section emoji.
export const FAMILY_ORDER = [
  ['zimage', 'Z-Image', '🌀'],
  ['sdxl', 'SDXL', '🎨'],
  ['krea', 'Krea 2', '✨'],
  ['flux', 'FLUX.1', '⚡'],
  ['flux2klein', 'FLUX.2 Klein', '🌹'],
];

// A dataset trained only in a family this build does not know (e.g. one added
// by a newer server) must still be reachable — it lands in a trailing section
// instead of silently vanishing from the library.
export const OTHER_FAMILY = ['other', 'Other', '📁'];

// Datasets no LoRA has ever been trained from — their own trailing section,
// AFTER every family (Other included), so the top of the library is the trained
// work. The family value below is the section's collapse key.
export const NOT_TRAINED = ['not-trained', 'Not trained yet', '🚫'];

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

/** The section a dataset belongs to: the FIRST family in FAMILY_ORDER it has
 *  actually been trained in — deterministic and independent of how the server
 *  orders trained_families (it sorts them alphabetically, which is NOT the
 *  section order). Trained only in families this build doesn't know → OTHER_FAMILY;
 *  never trained → NOT_TRAINED. Any extra trained families stay visible as the
 *  existing per-tile badges, so one dataset still maps to exactly one section. */
export function primaryFamily(d) {
  const trained = Array.isArray(d?.trained_families) ? d.trained_families : [];
  if (trained.length === 0) return NOT_TRAINED[0];
  const trainedSet = new Set(trained);
  const known = FAMILY_ORDER.find(([fam]) => trainedSet.has(fam));
  return known ? known[0] : OTHER_FAMILY[0];
}

/** Group datasets into ordered non-empty sections [{family, label, emoji,
 *  items}]: FAMILY_ORDER families first (each holding the datasets whose PRIMARY
 *  trained family is that one), then Other, then "Not trained yet" — always last.
 *  Each dataset lands in exactly one section (its primaryFamily), so the header
 *  counts sum to the library size and changing train_type never moves a tile. */
export function groupDatasets(datasets = []) {
  const byFamily = new Map();
  for (const d of datasets) {
    const fam = primaryFamily(d);
    if (!byFamily.has(fam)) byFamily.set(fam, []);
    byFamily.get(fam).push(d);
  }
  return [...FAMILY_ORDER, OTHER_FAMILY, NOT_TRAINED]
    .filter(([family]) => byFamily.has(family))
    .map(([family, label, emoji]) => ({ family, label, emoji, items: byFamily.get(family) }));
}
