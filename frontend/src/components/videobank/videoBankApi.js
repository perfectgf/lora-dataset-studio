/** 🎬 Video bank — the URLs, built in one place.
 *
 * Every path this lane talks to is assembled here rather than interpolated at
 * the call site, for one reason that has already cost this project a wave: the
 * clip list endpoint answers TWO different shapes from the same route
 * (`{clips,total}` normally, `{ids,total}` with `ids_only=1`), and a query
 * string built by hand in three components drifts into a fourth shape nobody
 * tested.
 *
 * PURE STRINGS, no fetch: `node --test` imports this file directly.
 */

/** The JPEG the grid shows. 404 until the thumbnails pass has run — that is an
 * ordinary state, not an error, and the grid draws a placeholder on it. */
export function videoClipThumbUrl(bankId, clipId) {
  return `/api/video-bank/${bankId}/clip/${clipId}/thumb`
}

/** The SOURCE file's bytes — the `base` every media fragment is built on.
 *
 * Deliberately per-SOURCE and not per-clip: a bank writes no clip files, so
 * there is nothing else to point at. The lightbox appends `#t=start,end` (see
 * videoClipFragment.clipFragmentSrc) and the browser range-requests only that
 * span out of what can be a multi-gigabyte rush. */
export function videoSourceMediaUrl(bankId, sourceId) {
  return `/api/video-bank/${bankId}/source/${sourceId}/media`
}

/** The workspace payload AND the 2 s poll. `refresh` re-walks the source folder
 * server-side; send it when the bank is OPENED, never on the poll — it costs a
 * directory walk of the whole tree every two seconds otherwise. */
export function videoBankUrl(bankId, { refresh = false } = {}) {
  return `/api/video-bank/${bankId}${refresh ? '?refresh=1' : ''}`
}

/** One page of the clip gallery, or the whole filter as bare ids.
 *
 * A falsy `status` means "every status" and MUST be left out of the query: the
 * server filters on `status in TRIAGE_STATUSES`, so `status=` or `status=all`
 * would silently return everything while the UI believed it had filtered. Same
 * for `source_id` — the server reads a bare int and treats 0 as absent. */
export function videoClipsUrl(bankId, {
  status = null, sourceId = null, offset = 0, limit = 200, idsOnly = false,
} = {}) {
  const q = new URLSearchParams()
  if (status && status !== 'all') q.set('status', status)
  if (sourceId) q.set('source_id', String(sourceId))
  if (idsOnly) {
    // offset/limit are meaningless alongside ids_only (the server answers the
    // WHOLE filter) — sending them reads like a paged answer that it is not.
    q.set('ids_only', '1')
  } else {
    if (offset) q.set('offset', String(offset))
    q.set('limit', String(limit))
  }
  const qs = q.toString()
  return `/api/video-bank/${bankId}/clips${qs ? `?${qs}` : ''}`
}

/** ✂ Retouching one shot. PATCH — it edits one field pair of an existing shot and
 * is idempotent, which matters because a nudge-and-save is easy to double-fire. */
export function videoClipBoundsUrl(bankId, clipId) {
  return `/api/video-bank/${bankId}/clip/${clipId}/bounds`
}

/** Cut one shot in two at a playhead position. */
export function videoClipSplitUrl(bankId, clipId) {
  return `/api/video-bank/${bankId}/clip/${clipId}/split`
}

/** A shot the detector missed entirely — created under the SOURCE, not the clip,
 * because there is no existing shot to hang it off. */
export function videoSourceClipsUrl(bankId, sourceId) {
  return `/api/video-bank/${bankId}/source/${sourceId}/clips`
}

/** 🎬 The threshold the detector cuts at — for the whole bank, or for ONE file.
 *
 * Two URLs and not one with an optional id, so a call site cannot accidentally
 * write a bank-wide value while meaning one file. POST rather than PATCH: the
 * body carries `null` to CLEAR the override, and a PATCH whose payload is a
 * deliberate null reads like a field somebody forgot to fill in. */
export function videoShotThresholdUrl(bankId) {
  return `/api/video-bank/${bankId}/shot-threshold`
}

/** "At threshold X you would get N shots" — read from the cached probabilities,
 * writes nothing. The same philosophy as the quality cuts' preview: a number
 * only means something once you have seen what it would do to THIS bank. */
export function videoShotDryRunUrl(bankId) {
  return `/api/video-bank/${bankId}/shot-dry-run`
}

/** Re-cut from the cache. Bank-wide spares hand-made cuts; the per-file one
 * replaces them, which is what makes it the way back from "single take". */
export function videoRecutUrl(bankId) {
  return `/api/video-bank/${bankId}/recut`
}

export function videoSourceRecutUrl(bankId, sourceId) {
  return `/api/video-bank/${bankId}/source/${sourceId}/recut`
}

/** This file is ONE take: one full-length shot, and no bulk pass may overrule
 * it afterwards. */
export function videoSourceSingleShotUrl(bankId, sourceId) {
  return `/api/video-bank/${bankId}/source/${sourceId}/single-shot`
}

/** 🔎 Rank this bank's shots against a typed phrase.
 *
 * A GET because it is a read and nothing else — re-runnable, bookmarkable, and
 * cheap to retry. Built with URLSearchParams rather than by interpolation for a
 * concrete reason: the query grammar uses a leading `-` for "push this down",
 * and ordinary phrases carry `&`, `#` and accents. Every one of those breaks a
 * hand-built query string, and the failure looks like a search that found the
 * wrong thing rather than like a bug. */
export function videoSearchUrl(bankId, { q, n = 60, status = null, pushDown = null } = {}) {
  const p = new URLSearchParams()
  p.set('q', String(q ?? ''))
  p.set('n', String(n))
  // Same rule as the clip list: a falsy status (or 'all') means every bucket and
  // MUST be left out, or the server filters on a string it does not know.
  if (status && status !== 'all') p.set('status', status)
  if (pushDown) p.set('push_down', pushDown)
  return `/api/video-bank/${bankId}/search?${p.toString()}`
}

/** 🗣 One shot's caption, edited by hand. PATCH — it replaces one field of an
 * existing shot and is idempotent, which matters because a save-on-blur is easy
 * to double-fire. */
export function videoClipCaptionUrl(bankId, clipId) {
  return `/api/video-bank/${bankId}/clip/${clipId}/caption`
}

/** Pass endpoints, so the four buttons cannot disagree about their own names. */
export function videoPassUrl(bankId, pass) {
  return `/api/video-bank/${bankId}/${pass}`
}

/** A video dataset's own page. Kept here so the library card and the promote
 * dialog's "open it" link build the same address. */
export function videoDatasetUrl(datasetId) {
  return `/api/video-dataset/${datasetId}`
}

/** ☁ The cloud lane of ONE video dataset.
 *
 * Its own URL family, and not the face lane's `/api/dataset/<id>/train/cloud`,
 * for the reason the run's `dataset_table` column exists: the two dataset tables
 * share one integer space, so the same URL shape for both would make the id
 * alone ambiguous at the outermost layer. `/api/video-dataset/...` says which
 * table it means before the server has to.
 */
export function videoDatasetCloudUrl(datasetId) {
  return `/api/video-dataset/${datasetId}/train/cloud`
}

export function videoDatasetCloudProgressUrl(datasetId) {
  return `/api/video-dataset/${datasetId}/train/cloud/progress`
}

/** Everything this dataset's cloud runs brought back, grouped by run then by
 * STEP — a Wan 2.2 save is a `_high_noise` + `_low_noise` PAIR, and a list of
 * loose files invites a button that downloads half a LoRA. */
export function videoDatasetCloudCheckpointsUrl(datasetId) {
  return `/api/video-dataset/${datasetId}/train/cloud/checkpoints`
}

/** ⬇ ONE harvested file. The filename is encoded rather than interpolated: it
 * comes back from the server carrying the dataset's own name, which may hold
 * anything a user typed. */
export function videoDatasetCheckpointUrl(datasetId, runId, filename) {
  const p = new URLSearchParams({ run_id: String(runId), filename: String(filename ?? '') })
  return `/api/video-dataset/${datasetId}/train/cloud/checkpoint?${p.toString()}`
}

export function videoDatasetCloudRetryUrl(datasetId) {
  return `/api/video-dataset/${datasetId}/train/cloud/retry`
}

export function videoDatasetCloudContinueUrl(datasetId) {
  return `/api/video-dataset/${datasetId}/train/cloud/continue`
}

export function videoDatasetCloudRunUrl(datasetId, runId) {
  return `/api/video-dataset/${datasetId}/train/cloud/run/${runId}`
}
