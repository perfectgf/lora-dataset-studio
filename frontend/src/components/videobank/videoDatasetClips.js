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

/** A stills set holds IMAGES under the same route and the same table. Wrapping
 * one in a <video> renders a dead player — found on a phone the day stills
 * shipped — so every surface that draws a clip asks this first. */
const STILL_EXTS = ['.png', '.jpg', '.jpeg', '.webp'];

export function isStillFile(name) {
  const n = String(name || '').toLowerCase();
  return STILL_EXTS.some((ext) => n.endsWith(ext));
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
 * survives, because the answer ("the bank keeps every shot") is the whole reason
 * this is a safe thing to click. */
export function removeClipsConfirmation(names) {
  const list = Array.isArray(names) ? names.filter(Boolean) : [];
  if (!list.length) return null;
  const head = list.length === 1
    ? `Remove “${list[0]}” from this dataset?`
    : `Remove ${list.length} clips from this dataset?`;
  return `${head}\n\nThe encoded file and its .txt caption are deleted from the dataset folder. The bank they were cut from keeps every shot and every decision — you can promote them again without triaging anything.`;
}
