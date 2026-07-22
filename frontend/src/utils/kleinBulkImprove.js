import { isSmallImageRescueRow } from './smallImageRescue.js'

export function kleinImproveExclusionReason(image, allImages = []) {
  if (!image) return 'image no longer exists'
  if (!image.filename) return 'image file is not ready'
  if (isSmallImageRescueRow(image)) return 'resolve the Klein rescue pair first'
  if (image.derivation_kind === 'klein_image_improve') return 'already an improvement candidate'
  const activeChild = (allImages || []).some((candidate) => (
    candidate.parent_image_id === image.id
      && candidate.derivation_kind === 'klein_image_improve'
      && candidate.status === 'pending'
  ))
  return activeChild ? 'an improvement is already pending review' : null
}

export function partitionKleinImproveSelection(images, selectedIds) {
  const all = Array.isArray(images) ? images : []
  const byId = new Map(all.map((image) => [image.id, image]))
  const eligible = []
  const excluded = []
  for (const id of selectedIds || []) {
    const image = byId.get(id)
    const reason = kleinImproveExclusionReason(image, all)
    if (reason) excluded.push({ id, image, reason })
    else eligible.push(image)
  }
  return { eligible, excluded }
}

/* The batch itself is a SERVER job now (POST /api/dataset/<id>/improve/batch).
   The browser loop it replaces sent one request per image, which meant the run
   only existed in the tab: a selection bigger than the backend's concurrency cap
   was mostly refused, ⏹ Stop could not reach it, and closing the tab killed it.
   What stays client-side is the eligibility partition above (the grid already
   holds the rows) plus the wording below. */

/** Live progress line for the ✨ button, from the dataset's server activity.
    `null` when no improve batch is running on this dataset. */
export function kleinImproveBatchLabel(activity) {
  if (!activity || activity.kind !== 'improve') return null
  const total = Number(activity.total) || 0
  const done = Number(activity.done) || 0
  if (activity.cancelling) return '✨ Stopping…'
  return total ? `✨ Improving ${done}/${total}` : '✨ Improving…'
}

/** Toast wording for a launched batch: what the server took, what it dropped. */
export function describeKleinImproveLaunch({ queued = 0, skipped = 0 } = {}) {
  const tail = skipped ? ` · ${skipped} not eligible and skipped` : ''
  return `Improving ${queued} image(s) in the background${tail} — originals stay intact.`
    + ' You can close this tab; ⏹ Stop generation ends the batch.'
}
