/** 🎬 Find shots — thresholds, previews and transition labels, as pure values.
 *
 * The detector's per-frame probabilities are kept on disk now, so changing the
 * threshold costs a file read instead of a pass over the GPU. That makes the
 * threshold something a user is invited to argue with, and everything a user is
 * invited to argue with needs an honest way to say what would happen. This
 * module is that: no React, no fetch, `node --test` imports it directly.
 *
 * THE ONE TRAP THIS FILE EXISTS TO AVOID. An empty threshold field and a typed
 * zero mean opposite things — empty is "inherit the bank's, or the app's", zero
 * is a threshold that fires on every single frame and shatters a rush into
 * hundreds of fragments. Every function below keeps `null` and `0` apart, on
 * every path a value takes between the input box and the server.
 */

/** 0.5 is the detector's published default and was never measured — the paper
 * uses it without justifying it anywhere, and no public curve exists for the
 * amateur footage this bank was built for. It stays the default because
 * replacing it blind would only trade one unmeasured number for another; what
 * changed is that it is now free to disagree with. */
const SHOT_THRESHOLD_DEFAULT = 0.5

/** '' → null (inherit) · '0.7' → 0.7 · anything else throws with the sentence
 * to show the user. Thrown rather than clamped: a field has somebody in front
 * of it who can be told, unlike the read paths on the server, which clamp
 * because they must never abort a pass already running. */
export function parseThreshold(text) {
  const raw = String(text ?? '').trim()
  if (raw === '') return null
  const value = Number(raw)
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error('The threshold must be a number between 0 and 1.')
  }
  return value
}

/** What to print for a stored value: the number, or what it inherits. Never a
 * blank — a blank field reads as "no threshold", and there is always one. */
export function thresholdLabel(value, fallback = SHOT_THRESHOLD_DEFAULT) {
  if (value === null || value === undefined || value === '') {
    return `Default (${Number(fallback ?? SHOT_THRESHOLD_DEFAULT).toFixed(2)})`
  }
  return Number(value).toFixed(2)
}

/** File, then bank, then the app's default. `0` at any level is a real value
 * and stops the search there. */
export function effectiveThreshold(source, bankValue,
                                   fallback = SHOT_THRESHOLD_DEFAULT) {
  const own = source?.shot_threshold
  if (own !== null && own !== undefined) return own
  if (bankValue !== null && bankValue !== undefined) return bankValue
  return fallback
}

/** The dry-run rows, each carrying how it differs from the threshold in force.
 *
 * "4 shots" means nothing on its own. "8 fewer than now" is the sentence
 * somebody can actually decide on, which is the whole reason the preview reads
 * the current value as well as the ladder. */
export function sweepRows(result) {
  const rows = result?.rows || []
  const current = result?.current
  const inForce = rows.find((r) => r.threshold === current)
  return rows.map((row) => {
    const isCurrent = current !== undefined && current !== null
      && row.threshold === current
    const delta = inForce ? row.shots - inForce.shots : null
    let deltaLabel = ''
    if (isCurrent) deltaLabel = 'in force'
    else if (delta !== null && delta !== 0) {
      deltaLabel = `${Math.abs(delta)} ${delta < 0 ? 'fewer' : 'more'} than now`
    } else if (delta === 0) deltaLabel = 'same as now'
    return { ...row, current: isCurrent, delta, deltaLabel }
  })
}

/** The line under the preview. It names the files that could NOT be answered
 * for, because a count that silently covers half a bank is worse than none. */
export function dryRunSummary(result) {
  const sources = result?.sources || 0
  const skipped = result?.skipped || 0
  const single = result?.single_shot || 0
  if (!sources && !skipped && !single) {
    return 'No file in this bank has been through Find shots yet.'
  }
  const parts = [`Counted over ${sources} ${sources === 1 ? 'file' : 'files'}.`]
  if (skipped) {
    parts.push(`${skipped} ${skipped === 1 ? 'file' : 'files'} could not be `
      + 'counted — run Find shots on them once and every later change is instant.')
  }
  // Named, not folded into `skipped`: these files are not missing anything, and
  // the count leaves them out precisely because the re-cut will too.
  if (single) {
    parts.push(`${single} not counted: you marked `
      + `${single === 1 ? 'it' : 'them'} as a single take, and a bank re-cut `
      + 'leaves those alone.')
  }
  return parts.join(' ')
}

/** What a re-cut actually did, INCLUDING what it deliberately left alone. A
 * bank where two files predate the cache and one is a declared single take must
 * say so; reporting only the successes is how a user concludes the feature is
 * broken. */
export function recutSummary(result) {
  const parts = [`${result?.clips || 0} shots across `
    + `${result?.sources || 0} ${result?.sources === 1 ? 'file' : 'files'}.`]
  if (result?.skipped) {
    parts.push(`${result.skipped} could not be re-cut from cache — `
      + 'run Find shots on them once.')
  }
  if (result?.single_shot) {
    parts.push(`${result.single_shot} left alone: you marked `
      + `${result.single_shot === 1 ? 'it' : 'them'} as a single take.`)
  }
  return parts.join(' ')
}

/** Can this file be re-cut with no pass at all? Needs a cached vector AND a
 * readable probe — the conversion from frames to seconds runs on the file's own
 * measured rate. */
export function canRecut(source) {
  return Boolean(source?.has_probs) && source?.probe_state === 'ok'
}

/** The chip that names a boundary's KIND, from the detector's second head.
 *
 * ONLY dissolves get one. Hard cuts are the overwhelming majority, and a chip
 * on every tile is a chip that says nothing — the same rule the quality flags
 * follow. A shot fading at both ends is named once, by its widest edge, because
 * the number people act on is "how much of this clip is not really this shot".
 *
 * ADVISORY, and the tooltip says so. The width→kind rule is coherent with how
 * the network was trained and has never been measured on real amateur footage;
 * it is a reading, not a verdict.
 */
export function transitionChip(clip) {
  const edges = [clip?.transition?.start, clip?.transition?.end]
    .filter((e) => e && e.kind === 'dissolve')
  if (!edges.length) return null
  const widest = edges.reduce((a, b) => (b.width > a.width ? b : a))
  const where = edges.length > 1 ? 'at both ends' : 'at one end'
  return {
    label: `dissolve ${widest.width}f`,
    title: `The detector read a dissolve ${where} of this shot, `
      + `${widest.width} frames wide at the widest. Its first or last frames are `
      + 'a cross-fade of the neighbouring shot. A reading, not a verdict.',
  }
}
