/**
 * Pure, session-only tag filtering for the dataset grid — frontend-only: the
 * captions are already in the polled payload, so no new route is involved.
 *
 * MATCH SEMANTICS (kept deliberately honest, and surfaced in the UI help so the
 * behaviour is never surprising):
 *
 *  • Booru mode ('booru' — SDXL booru-native captions): a caption is a list of
 *    comma-separated tags. A filter tag matches when it EQUALS one whole tag,
 *    case-insensitive — the exact same tokenisation the tag-frequency panel and
 *    the tag-mode find/replace already use (split on commas, trim, lowercase).
 *    Excluding "smile" hides captions carrying the tag `smile`, but not
 *    `smiley` or `smile lines`.
 *
 *  • Prose mode ('prose' — Z-Image / natural-language captions): the caption is
 *    free text, so we match the tag as a WHOLE WORD (or exact phrase),
 *    case-insensitive, on word boundaries — boundaries are anything that is not
 *    a letter or a digit (Unicode-aware, so accents count as letters). "smile"
 *    matches "a warm smile." but NOT "smiling" — the word must appear as
 *    written. The free-text field lets you filter on any word.
 *
 * Images with no caption tokenise to [] and therefore match NO tag, so an
 * EXCLUDE filter never hides an uncaptioned / rejected / pending image — exactly
 * what a "hide what's already done" checklist wants. An INCLUDE ("only with
 * tag") filter, conversely, hides anything that doesn't carry the tag.
 */

/** Trim + lowercase a raw tag string (the canonical form used everywhere). */
export function normalizeTag(tag) {
  return (tag || '').trim().toLowerCase();
}

/** Booru tokenisation — identical to CaptionToolsBar's frequency panel. */
export function tokenizeTags(caption) {
  return (caption || '')
    .split(',')
    .map((t) => t.trim().toLowerCase())
    .filter(Boolean);
}

function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/** Whole-word / exact-phrase, case-insensitive, Unicode-aware (prose captions). */
function proseHasWord(caption, tag) {
  if (!tag) return false;
  try {
    const re = new RegExp(
      `(?:^|[^\\p{L}\\p{N}])${escapeRegExp(tag)}(?=$|[^\\p{L}\\p{N}])`, 'iu');
    return re.test(caption || '');
  } catch {
    // Ultra-defensive fallback (should never trip): plain case-insensitive contains.
    return (caption || '').toLowerCase().includes(tag);
  }
}

/** Does this caption carry `tag` under the given match mode ('booru' | 'prose')? */
export function captionHasTag(caption, tag, mode) {
  const needle = normalizeTag(tag);
  if (!needle) return false;
  if (mode === 'prose') return proseHasWord(caption, needle);
  return tokenizeTags(caption).includes(needle);   // default: booru / exact tag
}

/**
 * Apply the active exclude/include filters to a list of images.
 * An image is VISIBLE when:
 *   • it matches NONE of the exclude tags, AND
 *   • (no include tags active) OR it matches AT LEAST ONE include tag.
 * Exclude wins over include on a tie (a tag in both → hidden).
 * Returns the same array reference when no filter is active (cheap no-op).
 */
export function filterImages(images, { excludes = [], includes = [], mode = 'booru' } = {}) {
  if (!excludes.length && !includes.length) return images || [];
  return (images || []).filter((img) => {
    const cap = img.caption || '';
    if (excludes.some((t) => captionHasTag(cap, t, mode))) return false;
    if (includes.length && !includes.some((t) => captionHasTag(cap, t, mode))) return false;
    return true;
  });
}
