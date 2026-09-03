/* ⬇ GET a URL and put what comes back in the user's Downloads.
 *
 * The third caller of this shape is what made it a module. `useImageDownload`
 * saves one gallery image, `galleryDownload.downloadImagesAsFiles` saves a
 * selection in a loop, and the ⇔ comparison saves a clip the server builds on
 * demand — same three steps every time, and the same two lessons that neither
 * of the first two could pay twice:
 *
 *  1. NOT `<a href={url} download>`. That is one line and it lies: pointed at a
 *     route that answers 404 (a file swept off the disk, a render restored a
 *     second ago), an anchor cheerfully saves the error PAGE under a .mp4 name
 *     and the user finds out days later by opening it. Fetching first turns
 *     that into a sentence on screen.
 *  2. The object URL is revoked on the NEXT TICK. Revoking it synchronously
 *     races Safari's own read of the blob and the save silently produces
 *     nothing at all.
 *
 * The NAME is never rebuilt here — it is read off `Content-Disposition`, whose
 * RFC 5987 form has to win over the ASCII fallback beside it
 * (`nameFromDisposition` already knows that, so it is reused rather than
 * re-derived).
 *
 * JSX-free on purpose: `node --test` parses no JSX, and this is the half worth
 * pinning. `fetchImpl` and `saveBlob` are injectable for the same reason.
 */
// The `.js` is not decoration: Vite resolves an extensionless import, `node
// --test` does not, and this module is pinned by a test that imports it.
import { nameFromDisposition } from './galleryDownload.js';

export function saveBlobAs(blob, name) {
  const href = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = href;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(href), 0);   // see lesson 2 above
}

/**
 * Fetch `url` and save it. Resolves to the name the file landed under; throws
 * with the server's own sentence when the route refused, so the caller can put
 * it on screen instead of into a console nobody has open.
 *
 * `signal` aborts the wait. It does NOT stop work the server has already
 * started — measured on this app: a client that hangs up mid-encode leaves
 * ffmpeg running to the end — so this frees the button and the memory, and
 * nothing more. `isAbort` lets a caller tell that apart from a refusal worth
 * showing: a layer the user closed owes them no error message.
 */
export async function saveUrlAsFile(url, {
  fallbackName = 'download',
  failure = 'That file could not be downloaded.',
  fetchImpl = null,
  saveBlob = null,
  signal = null,
} = {}) {
  const doFetch = fetchImpl || ((u) => fetch(u, { credentials: 'same-origin', signal }));
  const doSave = saveBlob || saveBlobAs;
  const res = await doFetch(url);
  if (!res.ok) {
    let msg = failure;
    try { msg = (await res.json())?.error || msg; } catch { /* not JSON */ }
    throw new Error(msg);
  }
  const name = nameFromDisposition(res.headers.get('Content-Disposition'), fallbackName);
  await doSave(await res.blob(), name);
  return name;
}

/** True for the rejection an `AbortController` causes — the one failure a
 *  caller should swallow rather than display. */
export function isAbort(err) {
  return err?.name === 'AbortError';
}
