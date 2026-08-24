/* 🖼 The Gallery page's decidable half — which page of the feed a click asks
 * for, how two pages merge, and what the screen says about it. JSX-free on
 * purpose: `node --test` cannot parse JSX, and these are the decisions worth
 * pinning (the grid around them is layout).
 *
 * The feed itself is /api/gallery/images (services/cloud_training.app_gallery):
 * every image the app ever generated, newest first, cursor-paginated. The rows
 * are the SAME shape the checkpoint/run galleries publish, which is what lets
 * the page reuse GeneratedImageLightbox, the improve hook and the selection
 * utilities without a translation layer.
 */

/** One page of the feed — mirrors the backend's APP_GALLERY_PAGE. */
export const GALLERY_PAGE_LIMIT = 60;

/** The kind filter's three positions, in display order. Ids are the QUERY
 *  values the backend accepts — '' means "no filter", never a fourth kind. */
export const GALLERY_KINDS = [
  { id: '', label: 'All' },
  { id: 'renders', label: 'Renders' },
  { id: 'improved', label: '✨ Improved' },
];

/** The feed URL for one page. `beforeId` is the cursor from the previous
 *  page's `next_before_id` — null asks for the head of the feed. */
export function galleryFeedUrl({ datasetId = '', kind = '', liked = false } = {},
  { beforeId = null, limit = GALLERY_PAGE_LIMIT } = {}) {
  const p = new URLSearchParams();
  p.set('limit', String(limit));
  if (beforeId != null) p.set('before_id', String(beforeId));
  if (datasetId !== '' && datasetId != null) p.set('dataset_id', String(datasetId));
  if (kind) p.set('kind', kind);
  if (liked) p.set('liked', '1');
  return `/api/gallery/images?${p.toString()}`;
}

/** Append a page to the feed, dropping ids already on screen. The cursor makes
 *  an overlap impossible between two pages of ONE read — this guards the other
 *  case: a refresh of page 1 landing over a feed that already holds it. */
export function mergeGalleryPage(existing, incoming) {
  const seen = new Set((existing || []).map((i) => i.id));
  const fresh = (incoming || []).filter((i) => !seen.has(i.id));
  return fresh.length ? [...(existing || []), ...fresh] : (existing || []);
}

/** True when any filter narrows the feed — the empty state depends on it. */
export function galleryFiltered({ datasetId = '', kind = '', liked = false } = {}) {
  return !!(datasetId !== '' && datasetId != null) || !!kind || !!liked;
}

/** The line above the grid. States the cut whenever the grid shows fewer than
 *  the scope holds — a partial feed that looks complete is the lie every other
 *  gallery here refuses. */
export function gallerySummaryLine({ count = 0, shown = 0 } = {}) {
  if (!count) return '';
  if (shown >= count) return `${count} image${count > 1 ? 's' : ''}, newest first.`;
  return `Showing the newest ${shown} of ${count} — Load more below.`;
}

/** What an empty grid says. Two different answers on purpose: "you filtered
 *  everything away" is fixed by the filters, "nothing was ever generated" is
 *  fixed somewhere else entirely, and one wording for both misleads whoever
 *  is holding the wrong problem. */
export function galleryEmptyMessage(filters) {
  return galleryFiltered(filters)
    ? 'No image matches these filters — clear them to see the whole feed.'
    : 'Nothing generated yet — renders from the Test Studio and the ◉ Canvas '
      + 'land here as soon as you make some.';
}

/** The dataset picker's options: All first, then every dataset holding at
 *  least one generated image, each carrying its count. */
export function datasetFilterOptions(datasets, totalCount = 0) {
  const list = Array.isArray(datasets) ? datasets : [];
  return [
    { value: '', label: `All datasets (${totalCount})` },
    ...list.map((d) => ({ value: String(d.id), label: `${d.name} (${d.count})` })),
  ];
}

/** What the toast says once an improve is queued FROM the Gallery. Its own
 *  wording, because the canvas one promises "this checkpoint's gallery" — true
 *  there, and a wrong address here, where the result lands in the very feed
 *  the user is looking at (a new ✨ row at its head). */
export function galleryImproveLaunchMessage(engineLabel) {
  return `${engineLabel || 'Improve'} started — the result arrives at the top `
    + 'of this gallery as its own ✨ image, next to everything else. '
    + 'The original is left untouched.';
}
