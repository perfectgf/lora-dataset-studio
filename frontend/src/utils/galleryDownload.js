/* ⬇ Taking pictures OFF the board — the decidable half.
 *
 * The ◉ Canvas is the only screen that knows an image's whole ancestry
 * (dataset → run → checkpoint → seed) and, until now, the only way to keep one
 * was a right-click "Save image as…" that threw all of it away. Two things ship:
 * a single image, from the pinned node and from the viewer; and a whole
 * gallery — a run's, or one checkpoint's — as a ZIP.
 *
 * THE NAME IS THE FEATURE and it is NOT computed here. There is no settings
 * sidecar (deliberately), so the file name is the sole carrier of the lineage,
 * and a scheme implemented twice is a scheme that will disagree with itself:
 * it lives once, in `services/gallery_download.py`, where pytest can reach
 * every hostile dataset name. The browser reads the finished name off
 * Content-Disposition or simply lets the download go.
 *
 * What lives HERE is the other half, and it is the half a JSX file would hide:
 * WHICH images a click is about, and whether the screen states the cut BEFORE
 * the archive lands. A ZIP that quietly holds 500 of 812 is exactly the kind of
 * half-truth the rest of this app refuses, so the count is on the button.
 *
 * JSX-free on purpose — `node --test` cannot parse JSX.
 */

/** Hard ceiling on one archive, mirrored from services/gallery_download.py
 *  (ZIP_IMAGE_CAP). Duplicated as a NUMBER only so the button can say the limit
 *  before the round-trip; the backend remains the one that enforces it. */
const ZIP_IMAGE_CAP = 500;

export const ZIP_SCOPE_ALL = 'all';
export const ZIP_SCOPE_SELECTION = 'selection';

function scopePath(target) {
  if (!target) return null;
  // The app-wide 🖼 Gallery: no record to scope by. Its ZIP routes are
  // selection-ONLY (the backend refuses a missing `ids`), so callers there
  // always pass one.
  if (target.kind === 'app') return '/api/gallery/images';
  if (target.recordId == null) return null;
  const rid = target.recordId;
  // Same rule as galleryScope(): no step means the whole run.
  return (target.kind === 'run' || target.step == null)
    ? `/api/train/run/${rid}/images`
    : `/api/train/checkpoint/${rid}/${target.step}/images`;
}

/** `?ids=` for an explicit selection. An EMPTY array keeps the parameter: an
 *  empty selection must stay an empty selection all the way to the backend,
 *  which refuses it — degrading it into "no parameter" would hand the whole
 *  gallery to a click that meant "these ones". */
function idsQuery(ids) {
  if (ids == null) return '';
  return `?ids=${ids.join(',')}`;
}

/** The ZIP itself. `ids` omitted → the whole scope. */
export function galleryZipUrl(target, ids = null) {
  const base = scopePath(target);
  return base ? `${base}/zip${idsQuery(ids)}` : null;
}

/** The preflight: counts, missing files and the cap, with no byte moved. */
export function galleryZipPlanUrl(target, ids = null) {
  const base = scopePath(target);
  return base ? `${base}/zip/plan${idsQuery(ids)}` : null;
}

/** One image, under its lineage name. */
export function imageDownloadUrl(imageId) {
  return imageId == null ? null : `/api/train/image/${imageId}/download`;
}

/**
 * The name the saved file lands under, read off `Content-Disposition`.
 *
 * NOT rebuilt in the browser: the lineage scheme lives once, in
 * services/gallery_download.py, where pytest can throw hostile dataset names at
 * it. Flask writes the RFC 5987 form beside an ASCII fallback whenever the name
 * needed escaping, and the escaped one has to win — matching the first
 * `filename` in the string would silently pick the lossy one.
 */
export function nameFromDisposition(header, fallback = 'image.png') {
  const value = String(header || '');
  const star = value.match(/filename\*=UTF-8''([^;]+)/i);
  if (star) {
    try { return decodeURIComponent(star[1].trim()); } catch { /* fall through */ }
  }
  const plain = value.match(/filename="?([^";]+)"?/i);
  return plain ? plain[1].trim() : fallback;
}

/**
 * ⬇ The selection as PLAIN FILES — one save per picked image, no archive.
 *
 * The ZIP next to it answers "give me these to move around as one thing"; this
 * answers "put these in my Downloads as themselves", which is what a phone's
 * photo picker, a training tool watching a folder, or someone grabbing three
 * pictures actually wants — unzipping on a phone is a chore that costs more
 * than the download. Both reuse the SAME single-image route, so every file
 * lands under its lineage name from Content-Disposition; nothing here invents
 * a second naming path.
 *
 * SEQUENTIAL on purpose. Firing N programmatic saves in one tick is how
 * browsers decide you are an attack: Chrome swallows everything past the first
 * ten or so unless the user grants "download multiple files", and the grants
 * prompt fires mid-burst so half the burst is already lost. One at a time, each
 * save waiting for its bytes, stays inside every browser's tolerance — the
 * permission prompt still appears once (that is the browser talking, not us)
 * and nothing is lost while it waits.
 *
 * A missing file SKIPS, it does not stop: the gallery legitimately lists rows
 * whose file a resume or a trash sweep took off the disk, and one such row must
 * not cost the user the other eleven. The summary names the count either way.
 *
 * `deps` exists for `node --test`: fetch and the save step are injectable, so
 * the loop's order, its skip-on-404 and its stop flag are pinned without a DOM.
 */
export async function downloadImagesAsFiles(ids, {
  onProgress = () => {},
  shouldStop = () => false,
  fetchImpl = null,
  saveBlob = null,
} = {}) {
  const doFetch = fetchImpl || ((url) => fetch(url, { credentials: 'same-origin' }));
  const doSave = saveBlob || ((blob, name) => {
    const href = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = href;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    // Next tick: revoking synchronously races Safari's own read of the blob
    // and the save silently produces nothing (same rule as useImageDownload).
    setTimeout(() => URL.revokeObjectURL(href), 0);
  });
  let saved = 0;
  let skipped = 0;
  const total = ids.length;
  for (const id of ids) {
    if (shouldStop()) break;
    onProgress({ done: saved + skipped, total });
    try {
      const res = await doFetch(imageDownloadUrl(id));
      if (!res.ok) { skipped += 1; continue; }
      const name = nameFromDisposition(res.headers.get('Content-Disposition'));
      await doSave(await res.blob(), name);
      saved += 1;
    } catch {
      skipped += 1;
    }
  }
  return { saved, skipped, stopped: shouldStop() ? total - saved - skipped : 0 };
}

/** The sentence the bar shows when the loop ends. Honest about every branch:
 *  a run someone cancelled mid-way says so instead of pretending it finished. */
export function filesDownloadSummary({ saved, skipped, stopped }) {
  const parts = [`${saved} file${saved === 1 ? '' : 's'} saved`];
  if (skipped) parts.push(`${skipped} skipped (file no longer on disk)`);
  if (stopped) parts.push(`${stopped} not started`);
  return parts.join(', ');
}

/**
 * Everything the ⬇ button in the gallery's action bar shows and does.
 *
 * ONE button, two meanings, taken from the mode already on screen: outside
 * Select mode it is the whole gallery, inside it is the picks. Two buttons both
 * saying "download" and differing by a word is how a bar becomes unreadable —
 * and Select mode is right there, already the app's answer to "these ones".
 */
export function zipButtonState({ picking = false, selectedCount = 0, totalCount = 0,
  cap = ZIP_IMAGE_CAP, busy = false } = {}) {
  const total = Math.max(0, Number(totalCount) || 0);
  const picked = Math.max(0, Number(selectedCount) || 0);
  const scope = picking ? ZIP_SCOPE_SELECTION : ZIP_SCOPE_ALL;
  if (total === 0) {
    return { shown: false, scope, disabled: true, capped: false, count: 0,
      label: '', title: '' };
  }
  const wanted = picking ? picked : total;
  const count = Math.min(wanted, cap);
  const capped = !picking && total > cap;
  let title;
  if (picking) {
    title = picked === 0
      ? 'Pick the images you want, then download them as one ZIP'
      : `Download the ${picked} selected image(s) as one ZIP`;
  } else if (capped) {
    // The cut, on the tooltip AND in the count on the face of the button, so it
    // cannot be discovered by counting files inside the archive.
    title = `Download the newest ${cap} of ${total} images as one ZIP — `
      + `one archive holds at most ${cap}. Open a single checkpoint's gallery `
      + 'for the rest.';
  } else {
    title = `Download all ${total} image(s) of this gallery as one ZIP`;
  }
  return {
    shown: true,
    scope,
    count,
    capped,
    disabled: busy || count === 0,
    label: busy ? '⬇ Zipping…' : `⬇ ZIP (${count})`,
    title,
  };
}

/**
 * The sentence the panel shows once the preflight answers — or the refusal.
 *
 * `null` means "nothing worth saying": a clean, complete, untruncated archive
 * needs no commentary. Everything else does, because the ZIP is about to be
 * shorter than the gallery it came from.
 */
export function planNotice(plan) {
  if (!plan) {
    return { blocked: true, kind: 'error',
      text: 'Could not work out what to download — the gallery could not be read.' };
  }
  if (!plan.ok || !plan.included) {
    return { blocked: true, kind: 'error',
      text: plan.note || 'None of these images can be downloaded.' };
  }
  if (plan.missing > 0 || plan.truncated) {
    return { blocked: false, kind: 'warn', text: plan.note || '' };
  }
  return null;
}
