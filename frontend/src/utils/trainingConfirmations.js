const OPTION_FOR_CONFIRM_FLAG = Object.freeze({
  allow_caption_mismatch: 'allowCaptionMismatch',
  allow_uncaptioned: 'allowUncaptioned',
  allow_caption_quality: 'allowCaptionQuality',
  allow_unverified_weights: 'allowUnverifiedWeights',
})

export function withTrainingConfirmationFlag(options, flag) {
  const option = OPTION_FOR_CONFIRM_FLAG[flag]
  return option ? { ...(options || {}), [option]: true } : options
}

/** Retry a camelCase hook call while confirmable server refusals accumulate.
 * Unknown flags stop immediately, preventing an unchanged-request loop. */
export async function runConfirmableTrainingRequest(request, initialOptions, nextFlag) {
  let options = { ...(initialOptions || {}) }
  let response = await request(options)
  while (response?.ok === false) {
    const flag = nextFlag(response.error)
    if (!flag) break
    if (flag === 'declined') return { response: null, options, declined: true }
    const nextOptions = withTrainingConfirmationFlag(options, flag)
    if (nextOptions === options) break
    options = nextOptions
    response = await request(options)
  }
  return { response, options, declined: false }
}
