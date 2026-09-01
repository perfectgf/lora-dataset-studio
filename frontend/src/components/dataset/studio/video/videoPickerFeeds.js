/** What the start-frame picker's two server feeds really answer.
 *
 * Both are read here rather than inline, because both got read wrong: the
 * dataset detail was asked for `clips` and the Gallery was asked for one page
 * and shown as if it were the whole feed.
 */

/** The clips of one video training set.
 *
 * `clips` on that payload is a COUNT, not the list — the list is `items`, which
 * is what the bank's own panel reads. Handing `21` to a renderer took the whole
 * page down (a number has no .map), and `d.clips || []` never caught it because
 * 21 is truthy. So the shape is asserted, not assumed: anything that is not an
 * array answers an empty list, and the picker shows "nothing here" instead of
 * a white screen.
 */
export function datasetClips(payload) {
  const items = payload?.items
  return Array.isArray(items) ? items : []
}

/** One page of the app Gallery, and where the next one starts.
 *
 * The feed is cursor-paginated: `has_more` says another page exists and
 * `next_before_id` is where it begins. A picker that ignored both showed the
 * newest 60 pictures as though they were everything somebody had generated.
 */
export function galleryPage(payload) {
  const images = Array.isArray(payload?.images) ? payload.images : []
  return {
    images,
    // The cursor the server gives, or the oldest id on this page — a server
    // that answers has_more without a cursor still pages correctly.
    before: payload?.next_before_id ?? (images.length ? images[images.length - 1]?.id : null),
    more: Boolean(payload?.has_more) && images.length > 0,
  }
}

/** Append a page, keeping the feed's order and never showing a picture twice.
 *
 * Pages are cursor-based, so a picture generated WHILE the picker is open
 * shifts the window and can arrive in two consecutive pages. Deduping on the
 * id is what keeps React keys unique — duplicated keys are the other way this
 * list breaks.
 */
export function appendImages(existing, page) {
  const seen = new Set((existing || []).map((im) => im?.id))
  return (existing || []).concat((page || []).filter((im) => {
    if (!im || seen.has(im.id)) return false
    seen.add(im.id)
    return true
  }))
}
