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

/** Send one request at a time; the backend still owns its global fan-out cap. */
export async function runSequentialKleinImprove(images, improve, onProgress = () => {}) {
  const queue = Array.isArray(images) ? images : []
  const succeeded = []
  const failed = []
  for (let index = 0; index < queue.length; index += 1) {
    const image = queue[index]
    try {
      const result = await improve(image.id)
      if (result?.ok === true) succeeded.push({ image, result })
      else failed.push({ image, error: result?.error || 'request returned no success confirmation' })
    } catch (error) {
      failed.push({ image, error: error?.message || 'request failed' })
    }
    onProgress({ done: index + 1, total: queue.length, succeeded: succeeded.length, failed: failed.length })
  }
  return { succeeded, failed }
}
