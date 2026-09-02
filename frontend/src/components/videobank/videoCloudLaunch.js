/** ☁ What the video cloud launch window SAYS, as pure functions.
 *
 * The image lane's launch dialog has a sentence for every state it can be in
 * — the offers, an empty market, the footer that names what is about to be
 * paid for, the line under the frozen button — and the video lane used to have
 * a `<select>` with no sentence at all. These are the video lane's sentences,
 * kept out of the component so `node --test` reads the exact words a user is
 * shown before money leaves their account.
 *
 * Nothing here decides anything the server will not re-decide: the guardrails
 * (fleet limit, budget, a sibling run on a pod) live in the backend and refuse
 * on launch; the preflight below exists so the refusal is read BEFORE the GPU
 * picker opens rather than rendered as a 409 after it.
 */
import { videoDatasetCloudUrl, videoDatasetUrl } from './videoBankApi.js'

export function videoOffersUrl(datasetId, steps) {
  const n = Math.max(100, Number(steps) || 1000)
  return `${videoDatasetCloudUrl(datasetId)}/offers?steps=${n}`
}

/** `lane` rides in the query only for the cloud lane, exactly like the image
 * lane's preflightUrl: a request with no lane is the local (machine-reading)
 * report, which is what the readiness card in the workspace shows. */
export function videoPreflightUrl(datasetId, lane) {
  const base = `${videoDatasetUrl(datasetId)}/train/preflight`
  return lane === 'cloud' ? `${base}?lane=cloud` : base
}

/** Account-wide, not per dataset: the month's spend, the budget and the run
 * limit are about the account's pods and its money, which one lane cannot
 * claim. The same endpoint the image panel polls. */
export const CLOUD_STATUS_URL = '/api/dataset/train/cloud/status'

/** The line under the tiers: what is about to be paid for, in the units the
 * bill will use. Says the month's spend against the budget when there is one —
 * the image dialog does, and a cheaper dialog that hides it would be the one
 * users stop trusting. */
export function launchFooterLine(ds, data, steps, cloudStatus) {
  const parts = [`${data?.steps ?? steps ?? '—'} steps`]
  if (ds?.target_label) parts.push(ds.target_label)
  if (ds?.frames) parts.push(`${ds.frames} frames`)
  if (ds?.clips != null) parts.push(`${ds.clips} clip${ds.clips === 1 ? '' : 's'}`)
  const budget = Number(cloudStatus?.monthly_budget) || 0
  const spent = Number(cloudStatus?.month_spend) || 0
  if (budget > 0) parts.push(`this month: $${spent.toFixed(2)} of $${budget.toFixed(2)}`)
  return `${parts.join(' · ')}. Time & cost are rough (one measured run, scaled); the pod is auto-terminated when done.`
}

/** An empty market is a fact about the price cap, and the sentence names the
 * cap so the user knows which number to move. The Settings link is JSX and is
 * added by the component next to this text. */
export function offersEmptyMessage(data) {
  const cap = data?.max_price_per_hour
  return cap != null
    ? `No GPU available under $${Number(cap).toFixed(2)}/h right now. Try again shortly, or`
    : 'No GPU available under your price cap right now. Try again shortly, or'
}

/** Under the frozen button while the launch POST runs: it freezes the dataset
 * and checks the target's weights on the hub, and a motionless "Launching…"
 * was reported as a hang on the image lane. */
export function launchStatusLine() {
  return 'Reserving the run: freezing the dataset and checking the target model. '
    + 'This can take up to a minute. The GPU is rented right after, and the run then '
    + 'follows its own progress on the Runs page — you can close this window once it opens.'
}

/** What a preflight report means for the click that follows it.
 *
 *   · blocked  → {ok:false, blockers}: nothing opens, the blockers are shown;
 *   · warnings → {ok:true, confirmText}: ONE question, all warnings in it —
 *                the image lane's "turn warnings into one confirm";
 *   · ready    → {ok:true}.
 * An unreachable or malformed report never blocks — the server re-decides on
 * launch anyway, and a preflight that fails closed would make the whole lane
 * unusable on the day the probe breaks. */
export function preflightGate(report, { lane = 'cloud' } = {}) {
  const checks = Array.isArray(report?.checks) ? report.checks : null
  if (!checks) return { ok: true }
  const blockers = checks.filter((c) => c.status === 'fail').map((c) => c.detail)
  if (blockers.length) return { ok: false, blockers }
  const warnings = checks.filter((c) => c.status === 'warn').map((c) => c.detail)
  if (!warnings.length) return { ok: true }
  const head = lane === 'cloud'
    ? 'Before renting a GPU, a few things to know:'
    : 'Before training, a few things to know:'
  return {
    ok: true,
    confirmText: `${head}\n\n${warnings.map((w) => `• ${w}`).join('\n')}\n\nContinue?`,
  }
}
