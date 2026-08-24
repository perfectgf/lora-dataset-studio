/**
 * Which scanned items a scrape DESTINATION can actually take, and how each one
 * is drawn.
 *
 * `/api/scrape/scan` has always returned both kinds: an item carries
 * `type: 'image'` or `type: 'video'`, and video items come back from RedGifs,
 * Erome, Picazor, TikTok, X, Civitai and every gallery-dl backed source. The
 * picker used to throw the video ones away with a one-line `.filter()`, because
 * its only two destinations (a dataset, an image bank) train on and triage
 * pictures. Now that a 🎬 video bank is a destination too, that decision belongs
 * to the destination rather than to the scan.
 *
 * Filtering happens at RENDER, not at scan time, on purpose: a scan of a mixed
 * gallery is a fact about the page, and the count of what this destination
 * cannot take is worth SAYING ("18 images set aside") rather than making
 * disappear. Doing it at scan time would also lose that number across "Load
 * more", which accumulates.
 *
 * Pure logic, no JSX: `node --test` imports this file directly.
 */

/** What each destination accepts. Keys are the `destination` prop of
 * ConceptSourcesPanel; anything unknown is treated as the historical image lane
 * so a typo degrades to the old behaviour rather than to an empty grid. */
const SCRAPE_DESTINATION_MEDIA = {
  dataset: ['image'],
  bank: ['image'],
  'video-bank': ['video'],
};

/** 'video' or 'image'. Anything that is not explicitly a video reads as an
 * image — exactly what the `type === 'image'` filter did, so a source that
 * omits `type` keeps behaving as it always has. */
export function scrapeItemMediaKind(item) {
  return item && item.type === 'video' ? 'video' : 'image';
}

export function destinationMediaKinds(destination) {
  return SCRAPE_DESTINATION_MEDIA[destination] || SCRAPE_DESTINATION_MEDIA.dataset;
}

export function destinationAcceptsItem(destination, item) {
  return destinationMediaKinds(destination).includes(scrapeItemMediaKind(item));
}

/** Split one scan result into what this destination takes and what it cannot. */
export function splitScanItemsForDestination(destination, items) {
  const accepted = [];
  const setAside = [];
  for (const item of Array.isArray(items) ? items : []) {
    (destinationAcceptsItem(destination, item) ? accepted : setAside).push(item);
  }
  return { accepted, setAside };
}

/** The word for what this destination takes, for a sentence a user reads. */
export function destinationMediaLabel(destination, count = 2) {
  const kinds = destinationMediaKinds(destination);
  const one = kinds.includes('video') ? 'video' : 'image';
  return count === 1 ? one : `${one}s`;
}

/**
 * The line under the grid about what was left out — '' when nothing was.
 *
 * Said rather than silently dropped: a video bank pointed at a photo gallery
 * would otherwise show an empty grid after a scan that plainly worked, and the
 * user has no way to tell "this page has no videos" from "the scan failed".
 */
export function setAsideNotice(destination, count) {
  const n = Number(count) || 0;
  if (n <= 0) return '';
  const left = destinationMediaKinds(destination).includes('video')
    ? (n === 1 ? 'image' : 'images') : (n === 1 ? 'video' : 'videos');
  return `${n} ${left} set aside — this destination takes ${destinationMediaLabel(destination)}.`;
}

/**
 * The thumbnail URL for a tile, or null when there is nothing safe to show.
 *
 * The proxy (`/api/scrape/thumb`) only ever restreams raster content types, so
 * pointing it at a video item's own URL answers 415 — the tile would then be
 * treated as a dead link and HIDDEN, which is how a perfectly good clip
 * disappears from the picker. gallery-dl sources routinely return a video with
 * no `thumbnail` at all, so this is the common case, not the edge one: those
 * tiles get a placeholder instead of a broken image.
 */
export function scrapeTileThumbUrl(item) {
  if (!item) return null;
  const raw = item.thumbnail
    || (scrapeItemMediaKind(item) === 'image' ? item.url : '');
  return raw ? `/api/scrape/thumb?url=${encodeURIComponent(raw)}` : null;
}

/** m:ss for the badge on a video tile, '' when the source gave no duration. */
export function formatScanItemDuration(seconds) {
  const n = Number(seconds);
  if (!Number.isFinite(n) || n <= 0) return '';
  const total = Math.round(n);
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`;
}

/**
 * The label a tile announces to a screen reader and shows on hover.
 * Video items rarely carry a title (RedGifs sends an id, gallery-dl often sends
 * nothing), so the kind is named rather than left to the picture.
 */
export function scrapeItemLabel(item) {
  const kind = scrapeItemMediaKind(item);
  if (item?.title) return item.title;
  if (item?.platform === 'pexels' && item?.photographer) {
    return `Pexels photo by ${item.photographer}`;
  }
  return kind === 'video' ? 'scraped video' : 'scraped image';
}

/**
 * Which of the picker's source tabs can produce ANYTHING this destination
 * takes. Reddit, Pexels and the keyless web search are image-only by
 * construction (their backends hard-code `type: 'image'` and filter on image
 * extensions before returning), so offering them to a video bank builds a
 * guaranteed dead end — and Reddit was the DEFAULT tab, so the first thing a
 * user met was a search that can never return a video. Only the URL tab reaches
 * the sources that emit videos (RedGifs, Erome, Picazor, TikTok, X, Civitai).
 *
 * `modes` is the panel's own [key, label] list, passed in rather than imported,
 * so this stays pure and the panel stays the owner of its labels.
 */
export function sourceModesForDestination(destination, modes) {
  const list = Array.isArray(modes) ? modes : [];
  if (!destinationMediaKinds(destination).includes('video')) return list;
  return list.filter(([key]) => key === 'url');
}
