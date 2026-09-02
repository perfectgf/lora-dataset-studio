/** 🎞 The clips of ONE video dataset — what is shown, in what order, and what
 * the counts under the grid mean.
 *
 * PURE, and it has to be: `node --test` imports this file directly, and every
 * decision worth arguing about (does a clip with no caption train on nothing? is
 * a still a clip?) is answered here rather than inside a component nobody can
 * assert against.
 *
 * The dataset's search is NOT the bank's. The bank searches embeddings server
 * side — it can rank a shot by what happens inside it — and a dataset holds a
 * few dozen to a few hundred rows already in memory, where a round trip to rank
 * text we already have would be an expensive way to be slower. So this is a
 * local, literal filter, and it says so on screen rather than borrowing the
 * bank's ranked-search vocabulary for something that does not rank.
 */

import { deleteDestination, isRecoverable } from '../../utils/deletionWording.js';
import { videoClipThumbUrl, videoDatasetClipMediaUrl } from './videoBankApi.js';

/** A stills set holds IMAGES under the same route and the same table. Wrapping
 * one in a <video> renders a dead player — found on a phone the day stills
 * shipped — so every surface that draws a clip asks this first. */
const STILL_EXTS = ['.png', '.jpg', '.jpeg', '.webp'];

export function isStillFile(name) {
  const n = String(name || '').toLowerCase();
  return STILL_EXTS.some((ext) => n.endsWith(ext));
}

/** The picture a surface shows for a clip — ONE rule for the training set's
 * grid and the Video Test Studio's start-frame picker, so a clip never has a
 * poster on one page and none on the other:
 *
 *   · a STILL is served by the media route itself — the real frame;
 *   · a clip cut from a bank borrows the bank's JPEG thumbnail through its
 *     provenance columns — already generated, a middle frame of the shot;
 *   · anything else has no picture on the server, and the tile says so with a
 *     placeholder rather than a broken image. Never a <video> per tile — Chrome
 *     caps media players at about sixty across the browser.
 *
 * Null, not '', for "no poster": tiles branch on it. A thumbnail that 404s
 * (the bank was deleted, the thumbnails pass never ran) is an ordinary state,
 * and the surface falls back to its placeholder rather than a broken glyph. */
export function datasetClipPoster(datasetId, clip) {
  if (!clip || datasetId == null) return null;
  if (isStillFile(clip.filename)) return videoDatasetClipMediaUrl(datasetId, clip.id);
  if (clip.source_bank_id && clip.source_clip_id) {
    return videoClipThumbUrl(clip.source_bank_id, clip.source_clip_id);
  }
  return null;
}

/** Seconds of source this clip was cut from, or null for a still (which has no
 * bounds at all — start and end are equal, or absent). */
export function clipDurationS(clip) {
  const start = Number(clip?.start_s);
  const end = Number(clip?.end_s);
  if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
  const d = end - start;
  return d > 0 ? d : null;
}

export function hasCaption(clip) {
  return String(clip?.caption || '').trim().length > 0;
}

export const CLIP_FILTERS = Object.freeze([
  { id: 'all', label: 'All' },
  { id: 'captioned', label: 'Captioned' },
  { id: 'uncaptioned', label: 'No caption' },
]);

export function normalizeClipFilter(id) {
  return CLIP_FILTERS.some((f) => f.id === id) ? id : 'all';
}

export function filterClipsByCaption(clips, filterId) {
  const list = Array.isArray(clips) ? clips : [];
  const id = normalizeClipFilter(filterId);
  if (id === 'captioned') return list.filter(hasCaption);
  if (id === 'uncaptioned') return list.filter((c) => !hasCaption(c));
  return list;
}

export function clipFilterCount(clips, filterId) {
  return filterClipsByCaption(clips, filterId).length;
}

/** Terms are ANDed, and a `-term` EXCLUDES — the same two rules the bank's
 * search box teaches, so the gesture transfers even though the mechanism does
 * not. Matched against the file name, the caption and the source path: the
 * source path is what lets you pull back every clip cut out of one rush. */
export function searchClips(clips, query) {
  const list = Array.isArray(clips) ? clips : [];
  const terms = String(query || '').toLowerCase().split(/\s+/).filter(Boolean);
  if (!terms.length) return list;
  const wanted = terms.filter((t) => !t.startsWith('-'));
  const unwanted = terms.filter((t) => t.startsWith('-') && t.length > 1)
    .map((t) => t.slice(1));
  return list.filter((clip) => {
    const hay = `${clip.filename || ''} ${clip.caption || ''} ${clip.src_relpath || ''}`
      .toLowerCase();
    return wanted.every((t) => hay.includes(t)) && !unwanted.some((t) => hay.includes(t));
  });
}

export const CLIP_SORTS = Object.freeze([
  // The default is the ORDER ON DISK, and that is not a fallback: clip_0001,
  // clip_0002 … is the order the trainer walks the folder in, so a set read in
  // any other order by default would be read in an order nothing else uses.
  { id: 'filename', label: 'File order' },
  { id: 'longest', label: 'Longest first' },
  { id: 'shortest', label: 'Shortest first' },
  { id: 'caption-first', label: 'Uncaptioned first' },
]);

export function normalizeClipSort(id) {
  return CLIP_SORTS.some((s) => s.id === id) ? id : 'filename';
}

const byFilename = (a, b) => String(a.filename || '').localeCompare(String(b.filename || ''));

export function sortClips(clips, sortId) {
  const list = [...(Array.isArray(clips) ? clips : [])];
  switch (normalizeClipSort(sortId)) {
    case 'longest':
      return list.sort((a, b) => (clipDurationS(b) ?? 0) - (clipDurationS(a) ?? 0)
        || byFilename(a, b));
    case 'shortest':
      return list.sort((a, b) => (clipDurationS(a) ?? 0) - (clipDurationS(b) ?? 0)
        || byFilename(a, b));
    case 'caption-first':
      // Uncaptioned first, because that is the working list: the clips still
      // owed a caption are the ones you came to this section for.
      return list.sort((a, b) => (hasCaption(a) ? 1 : 0) - (hasCaption(b) ? 1 : 0)
        || byFilename(a, b));
    default:
      return list.sort(byFilename);
  }
}

/** One pass, in the order the screen applies them: search narrows, the caption
 * filter narrows further, the sort orders what is left. */
export function visibleClips(clips, { query = '', filter = 'all', sort = 'filename' } = {}) {
  return sortClips(filterClipsByCaption(searchClips(clips, query), filter), sort);
}

export function clipCounts(clips) {
  const list = Array.isArray(clips) ? clips : [];
  const captioned = list.filter(hasCaption).length;
  const seconds = list.reduce((sum, c) => sum + (clipDurationS(c) || 0), 0);
  return {
    total: list.length,
    captioned,
    uncaptioned: list.length - captioned,
    stills: list.filter((c) => isStillFile(c.filename)).length,
    seconds: Math.round(seconds * 10) / 10,
  };
}

/** What the coverage line says, and it is deliberately not a warning.
 *
 * A clip with no caption is not skipped and does not fail: the sidecar is
 * written with the trigger word alone (video_bank_service._with_trigger), and
 * with no trigger it is written empty. That is a legitimate way to train a
 * motion LoRA, so the line REPORTS rather than nags — and it names the trigger,
 * because "trains on the trigger alone" is only reassuring if you can see what
 * the trigger is. */
export function captionCoverageNote(counts, triggerWord) {
  const { total, captioned, uncaptioned } = counts || {};
  if (!total) return 'No clip in this dataset yet.';
  if (!uncaptioned) return `All ${total} clip${total === 1 ? '' : 's'} carry a caption.`;
  const trigger = String(triggerWord || '').trim();
  const tail = trigger
    ? `train on the trigger word alone (“${trigger}”).`
    : 'train on an EMPTY .txt — this set has no trigger word either.';
  return `${captioned} of ${total} clips carry a caption; the other ${uncaptioned} ${tail}`;
}

/** Confirmation for dropping clips out of the set. Names the count and says what
 * survives, because the answer is the whole reason this is a safe thing to click.
 *
 * WHERE the files go comes from the app-wide helper, never from a sentence typed
 * here. This used to say "the app's Trash — recoverable from Settings" in three
 * places at once, and it was false: it named one of three possible destinations
 * as if it were the only one. utils/deletionWording.js exists precisely so that
 * every destructive confirmation names the destination from the MODE the server
 * reports, and its own docstring says why a second copy is a second place to
 * drift. The server sends `delete_mode` on the dataset payload for exactly this.
 *
 * The reassuring half depends on where the clips came from. A STILLS set has no
 * bank at all — its rows are written straight from an image dataset with no
 * source_bank_id — so "the bank keeps every shot" would promise something that
 * does not exist. Different behaviour must not wear the same wording. */
export function removeClipsConfirmation(names, { fromBank = true, mode } = {}) {
  const list = Array.isArray(names) ? names.filter(Boolean) : [];
  if (!list.length) return null;
  const head = list.length === 1
    ? `Remove “${list[0]}” from this dataset?`
    : `Remove ${list.length} clips from this dataset?`;
  const where = `The encoded file and its .txt caption go to ${deleteDestination(mode)}`
    + (isRecoverable(mode) ? ' — recoverable until you empty it.' : '.');
  const tail = fromBank
    ? 'The bank they were cut from keeps every shot and every decision — you can promote them again without triaging anything.'
    : 'This set was built from an image dataset, so there is no bank to promote them from again. The images are still in that dataset; the captions written here are not.';
  return `${head}\n\n${where} ${tail}`;
}

/** What the toast says after a removal — and it stops saying “removed” about a
 * clip whose FILE is still in the folder.
 *
 * `files_kept` is the server's answer for a file it could not move: an antivirus
 * scan, an open player, or a training run reading this very folder. The row is
 * kept with it, deliberately, because the folder IS the dataset and a row
 * deleted without its file takes the clip out of the app while leaving it in the
 * training set. Same register as the sidecar warning, which already settled this
 * argument for captions.
 *
 * The destination is composed from `delete_mode`, the same way the confirmation
 * did before the click — and "nothing happened" is its own sentence: a green
 * "0 clips removed" is a success toast about a no-op. */
export function removeClipsReport({ removed = 0, files_kept: kept = 0, delete_mode: mode } = {}) {
  const n = `${removed} clip${removed === 1 ? '' : 's'}`;
  if (!removed && !kept) return 'Nothing was removed — no clip matched.';
  if (!kept) return `${n} removed — sent to ${deleteDestination(mode)}.`;
  const k = `${kept} file${kept === 1 ? '' : 's'}`;
  if (!removed) {
    return `Nothing was removed: ${k} could not be moved — still open somewhere (a player, an antivirus scan, or a training run reading this folder). Those clips stay in the set.`;
  }
  return `${n} removed, but ${k} could not be moved and stay in the folder — the trainer reads the folder, so those clips are still in the set. Close whatever is holding them and try again.`;
}

/** Whether a caption draft may be dropped after its save landed.
 *
 * The posted value is read when the save STARTS; the purge happens two awaits
 * later (the POST, then the refresh). Anything typed in between lives in the
 * draft and nowhere else — and the old unconditional purge threw it away in
 * silence, with a later blur then doing nothing because there was no draft
 * left to save. The Escape fix made that window trivial to hit: Escape starts
 * the save and lands the user on the same clip's box in the Captions section.
 *
 * So the draft is dropped ONLY if it is still the value that was posted. Pure,
 * so the decision is a tested value rather than a line in a callback. */
export function purgeDraft(drafts, id, postedValue) {
  const map = drafts || {};
  if (!(id in map)) return map;
  if (map[id] !== postedValue) return map;          // the user typed on: their text wins
  const next = { ...map };
  delete next[id];
  return next;
}

/** What a key does in the player, as a VALUE. 'save-close' | 'prev' | 'next' | null.
 *
 * Escape SAVES before it closes, and it does so whether or not the caret is in
 * the caption box — that is the whole point. Closing unmounts the textarea, a
 * focused element removed from the DOM never fires blur, and blur is what owns
 * the save; so "Escape while typing" (the only case where there is a caption to
 * lose) used to drop the text in silence while the screen kept showing it. The
 * arrows, on the other hand, are exactly what the caret needs while typing, and
 * they step nowhere until it leaves the box. */
export function lightboxKeyAction(key, { typing = false, hasPrev = false, hasNext = false } = {}) {
  if (key === 'Escape') return 'save-close';
  if (typing) return null;
  if (key === 'ArrowLeft' && hasPrev) return 'prev';
  if (key === 'ArrowRight' && hasNext) return 'next';
  return null;
}

/** What the player is showing, and what it steps through — TWO different lists,
 * and the difference is the whole point.
 *
 * The clip is resolved against the FULL set; prev/next walk the FILTERED one.
 * Resolving both against the filtered list looks tidier and breaks the page's
 * own headline workflow: the filter it pushes hardest is “No caption”, the
 * reason you open a clip is to give it one, and the instant you succeed the clip
 * leaves the filter — so the player closed under the user's hands at the exact
 * moment they finished. Measured in a real browser before it was pinned here.
 *
 * `index` is -1 for a clip that has left the filtered list. The arrows do NOT
 * go dead there: that stranded the page's own workflow (filter "No caption",
 * caption a clip, and the next uncaptioned one became unreachable without
 * closing and reopening). `lastIndex` — the slot the clip occupied before it
 * left — is where the walk resumes: the clip that took its place is next, the
 * one before it is prev, both clamped to the list that remains.
 */
export function lightboxTargets(items, shown, openId, lastIndex = -1) {
  const list = Array.isArray(items) ? items : [];
  const walk = Array.isArray(shown) ? shown : [];
  const index = walk.findIndex((c) => c.id === openId);
  if (index >= 0) {
    return {
      clip: list.find((c) => c.id === openId) || null,
      index,
      prevId: index > 0 ? walk[index - 1].id : null,
      nextId: index < walk.length - 1 ? walk[index + 1].id : null,
    };
  }
  // Left the filtered list: resume from where it was. `at` is the slot that now
  // holds its successor (or the end, when it was last).
  const at = Math.max(0, Math.min(Number.isInteger(lastIndex) ? lastIndex : 0, walk.length));
  return {
    clip: list.find((c) => c.id === openId) || null,
    index: -1,
    prevId: at > 0 ? walk[at - 1].id : null,
    nextId: at < walk.length ? walk[at].id : null,
  };
}
