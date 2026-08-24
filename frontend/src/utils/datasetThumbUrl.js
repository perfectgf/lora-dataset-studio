/* TILE-SIZED PICTURES — one rewrite instead of twenty hand-written URLs.
 *
 * `/api/dataset/<id>/img/<name>` serves the ORIGINAL bytes: a 1-4 megapixel PNG,
 * often several megabytes, fully decoded by the browser so it can paint a 96 px
 * tile. The board, the dataset grid, the Test Studio sweep, the checkpoint
 * preview pills and the cloud-run cards all did exactly that, on every open.
 * `/api/dataset/<id>/thumb/<name>?s=<side>` is the same picture as a cached WebP.
 *
 * This is a URL REWRITE and not "build the thumb URL from an id and a filename"
 * on purpose: several of these URLs are stamped by the BACKEND (a canvas image
 * node's `url`, a checkpoint pill's `preview.url`, a run card's `preview_url`)
 * and the components that draw them never see the pieces. Rewriting means one
 * function is the only thing that knows the endpoint's shape.
 *
 * WHAT IT REFUSES TO TOUCH is the load-bearing half: anything that is not a
 * dataset `/img/` URL comes back verbatim. A bank URL, a blob:, a data:, a
 * missing value — a tile that silently 404s because a helper rewrote a URL it
 * did not understand is a worse bug than a heavy tile.
 */

/** The sizes the server will actually materialise (`dataset_thumbs.THUMB_SIDES`).
 *  Kept here so a caller asking for a rung that does not exist is a bug we can
 *  see in a test, not a silent snap-up on the server. */
export const THUMB_SIDES = [128, 192, 256, 320, 384, 512, 640, 768, 1024];

const IMG_SEGMENT = /^(\/api\/dataset\/\d+)\/img\/(.+)$/;

/** The rung that covers a tile drawn `px` wide. Unknown/garbage → 512. */
function datasetThumbSide(px) {
  const want = Number(px);
  if (!Number.isFinite(want) || want <= 0) return 512;
  return THUMB_SIDES.find((side) => side >= want) ?? THUMB_SIDES[THUMB_SIDES.length - 1];
}

/**
 * A rung that only ever goes UP, for a tile the user can resize live.
 *
 * Picking the rung from the current width alone would re-request the picture
 * every time a drag crosses 384→512→640, mid-gesture, and each of those is a
 * network round-trip on the frame the user is watching. Ratcheting means a node
 * dragged big fetches once more and then stops; a node dragged back small keeps
 * the sharper copy it already has, which costs nothing (it is already decoded)
 * and is what the user would want anyway if they enlarge it again.
 */
export function ratchetThumbSide(previous, px) {
  const want = datasetThumbSide(px);
  const held = Number(previous);
  return Number.isFinite(held) && held > want ? held : want;
}

/**
 * Rewrite a dataset image URL to its thumbnail, preserving any existing query
 * (the `?v=<nonce>` cache-buster an in-place crop appends is still needed:
 * without it the BROWSER would keep showing the pre-crop tile it already has).
 *
 * @param {string|null|undefined} url  a `/api/dataset/<id>/img/<name>` URL
 * @param {number} [side]  requested longest side; snapped by the server
 * @returns {string|null|undefined} the thumbnail URL, or `url` untouched
 */
export function datasetThumbUrl(url, side = 512) {
  if (typeof url !== 'string' || !url) return url;
  const [path, query = ''] = url.split('?');
  const m = IMG_SEGMENT.exec(path);
  if (!m) return url;
  const params = new URLSearchParams(query);
  params.set('s', String(side));
  return `${m[1]}/thumb/${m[2]}?${params.toString()}`;
}
