/** ☁ What the cloud panel of one video dataset should SAY, as pure functions.
 *
 * Separated from the component for the reason every other `*Status.js` in this
 * folder is: `node --test` imports this file directly, and the questions worth
 * testing here — may this dataset be launched at all, is a run still on the
 * clock, may this run be continued — are the ones that cost money when they are
 * answered wrong.
 *
 * NOTHING HERE DECIDES ANYTHING THE SERVER WILL NOT RE-DECIDE. The launch guard,
 * the ownership test and the terminal-state check all live in the backend; these
 * functions exist so the UI can explain a refusal BEFORE the click rather than
 * render a 409.
 */

/** Statuses in which a pod is (or may still be) on the clock. Mirrors the
 * backend's ACTIVE_STATES; a status this list has not heard of is treated as
 * NOT active, because the alternative is a panel permanently frozen by one
 * unknown string. */
const ACTIVE = new Set([
  'preparing', 'provisioning', 'uploading', 'training', 'stopping',
])

export function isActive(status) {
  return ACTIVE.has(String(status || ''))
}

/** Why this dataset cannot be sent to a pod right now, or null when it can.
 *
 * `training_verified` is the important one and it is not pedantry: the target
 * catalogue carries it precisely because for several video models we know the
 * geometry and no LoRA trainer is known to exist. Renting an 80 GB GPU to find
 * that out is the mistake this string prevents. */
export function launchBlockedReason(dataset, run) {
  if (!dataset) return 'This dataset is no longer here.'
  if (!dataset.clips) {
    return 'This dataset has no clips on disk — rebuild it before training.'
  }
  if (dataset.training_verified === false) {
    return `No LoRA trainer is known to exist for ${dataset.target_label || 'this target'} yet.`
  }
  if (run && isActive(run.status)) {
    return 'A cloud run of this dataset is already on a pod.'
  }
  return null
}

/** One line describing where a run is. Deliberately says the COST driver
 * (the GPU) as soon as one is known — a progress panel that hides what is being
 * billed is the one users stop trusting. */
export function runSummary(run) {
  if (!run || !run.run_id) return null
  const bits = [`Run #${run.run_id}`, run.status]
  if (run.gpu) bits.push(run.gpu)
  if (run.price_per_hour) bits.push(`$${Number(run.price_per_hour).toFixed(2)}/h`)
  return bits.filter(Boolean).join(' · ')
}

/** Retry is for a run that FAILED; Continue is for a terminal run that brought
 * weights back. Both are refused while a run is active — a second pod on one
 * dataset is money spent twice on one answer. */
export function canRetry(run) {
  return Boolean(run && run.status === 'error')
}

export function canContinue(run, group) {
  return Boolean(run && !isActive(run.status) && group && group.steps?.length)
}

/** The label of one harvested step, pair-aware.
 *
 * A Wan 2.2 checkpoint is TWO files at one step. Labelling them separately is
 * how a UI ends up offering half a LoRA — so the step is the unit, and the file
 * count is stated rather than implied. */
export function stepLabel(step) {
  if (!step) return ''
  const n = step.files?.length || 0
  // A LOCAL run's final save carries no number (the lane stamps no step count
  // the listing can read): "Final", and never "Final (step null)".
  const head = step.final
    ? (step.step != null ? `Final (step ${step.step})` : 'Final')
    : `Step ${step.step}`
  return n > 1 ? `${head} — ${n} files (both experts)` : head
}
