/**
 * Pure, frontend-only DECISION filtering for the dataset grid — the companion of
 * `tagFilter.js` (which filters on caption content). Both compose: the grid shows
 * the images that pass the decision filter AND the tag filter.
 *
 * WHY: the header line already COUNTS the undecided images ("254 awaiting ✓/✕")
 * but nothing could ISOLATE them, so a batch of Klein improvement candidates
 * stayed buried in a 500-image grid and `select all` always took everything.
 *
 * The `undecided` predicate is deliberately the SAME expression the workspace
 * uses for its "awaiting ✓/✕" count (status pending + a file on disk + not half
 * of an unresolved Klein rescue pair), so the badge and the filter can never
 * disagree. `rejected` likewise mirrors the "unused" count (reject or failed).
 *
 * Rescue rows: an unresolved pair is only decidable in the side-by-side resolver,
 * so it is excluded from every subset (the grid list already drops the losers —
 * this is belt-and-braces, and it lets a caller pass the unfiltered list).
 */

import { isSmallImageRescueRow } from './smallImageRescue.js';

/** The selector's entries, in display order. `id` is persisted — never rename. */
export const GRID_STATUS_FILTERS = [
  { id: 'all', label: 'All', title: 'Every image in the grid' },
  { id: 'undecided', label: 'Undecided', title: 'Imported/generated images still awaiting ✓/✕' },
  { id: 'kept', label: 'Kept', title: 'Images you marked ✓ Keep' },
  { id: 'rejected', label: 'Rejected', title: 'Images you marked ✕ Reject (and failed generations)' },
  { id: 'improve', label: 'Improve candidates', title: 'Klein improvement candidates awaiting review' },
];

export const DEFAULT_GRID_STATUS_FILTER = 'all';

const IDS = new Set(GRID_STATUS_FILTERS.map((f) => f.id));

/** Is `value` one of the known filter ids? (Guards the localStorage read.) */
export function isGridStatusFilter(value) {
  return typeof value === 'string' && IDS.has(value);
}

/** Normalise anything (stale localStorage, undefined…) to a usable filter id. */
export function normalizeGridStatusFilter(value) {
  return isGridStatusFilter(value) ? value : DEFAULT_GRID_STATUS_FILTER;
}

const PREDICATES = {
  undecided: (img, blocked) => img.status === 'pending' && !!img.filename && !blocked(img),
  kept: (img) => img.status === 'keep',
  rejected: (img, blocked) => (img.status === 'reject' || img.status === 'failed') && !blocked(img),
  improve: (img, blocked) => img.derivation_kind === 'klein_image_improve' && !blocked(img),
};

/**
 * Keep only the images matching `filterId`.
 * `unresolvedRescueIds` is the workspace's set of ids belonging to a Klein rescue
 * pair that has not been resolved yet; those never belong to a decision subset.
 * Returns the same array reference for 'all' (cheap no-op, like filterImages).
 */
export function filterImagesByStatus(images, filterId, { unresolvedRescueIds } = {}) {
  const id = normalizeGridStatusFilter(filterId);
  if (id === 'all') return images || [];
  // A rescue row is undecidable here UNLESS the caller told us which ids are still
  // unresolved — then a resolved winner is an ordinary image again.
  const blocked = (img) => (unresolvedRescueIds
    ? unresolvedRescueIds.has(img.id)
    : isSmallImageRescueRow(img));
  const match = PREDICATES[id];
  return (images || []).filter((img) => match(img, blocked));
}

/** { all, undecided, kept, rejected, improve } counts, for the selector labels. */
export function gridStatusFilterCounts(images, opts) {
  const counts = {};
  for (const f of GRID_STATUS_FILTERS) {
    counts[f.id] = filterImagesByStatus(images, f.id, opts).length;
  }
  return counts;
}
