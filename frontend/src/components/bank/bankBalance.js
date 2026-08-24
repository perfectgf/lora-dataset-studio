// ⚖️ Balanced selection — turning the backend's per-bucket numbers into words.
//
// The whole point of a balanced pick is that the user can SEE what they got:
// "20 face, 20 bust, 20 body" is the answer to a question no per-image score can
// ask ("does my set cover what I want to generate?"), and a selection that does
// not say its own shape is indistinguishable from an unbalanced one. So every
// readout here is NUMBERS, announceable as text — the coloured bar is decoration
// on top of a list a screen reader can read out, never the only carrier.
//
// Pure functions, no JSX: `node --test` cannot parse JSX, and the distribution
// is exactly the logic worth proving.

// Axis ids are persisted (localStorage) — NEVER rename one without an alias.
export const BALANCE_AXES = [
  { id: 'framing', label: 'Framing', hint: 'Face / bust / body / back — the 📐 Framing pass' },
  { id: 'framing+person', label: 'Framing × person',
    hint: 'Also splits per person — for a bank that really holds several subjects' },
]
export const BALANCE_DEFAULT_AXIS = 'framing'

const FRAMING_LABEL = { face: 'face', bust: 'bust', body: 'body', back: 'back' }

export function bucketLabel(b) {
  const fr = FRAMING_LABEL[b.framing] || b.framing || '?'
  return b.cluster == null ? fr : `${fr} · person #${b.cluster}`
}

/** The rows the panel renders: one per bucket, in framing order, with the numbers
 *  a reader can hear (selected / available / what an even split wanted). */
export function balanceRows(result) {
  const buckets = (result && result.buckets) || []
  return buckets.map((b) => ({
    key: b.key,
    label: bucketLabel(b),
    selected: b.selected || 0,
    available: b.available || 0,
    fairShare: b.fair_share || 0,
    short: !!b.short,
    share: (result.selected || 0) > 0 ? (b.selected || 0) / result.selected : 0,
  }))
}

/** One sentence naming the distribution obtained — the headline, and the thing a
 *  screen reader announces when the selection lands. */
export function summarizeBalance(result) {
  if (!result) return ''
  const rows = balanceRows(result)
  if (!rows.length) return 'Nothing selected.'
  const parts = rows.map((r) => `${r.selected} ${r.label}`)
  const axis = result.axis === 'framing+person' ? 'framing × person' : 'framing'
  return `Selected ${result.selected} of ${result.requested} requested, spread over ${axis}: ${parts.join(', ')}.`
}

/** The honest notes UNDER the headline: an axis that could not be satisfied, and
 *  images left out because nothing has labelled them yet. Never silent, never
 *  padded — a shortfall says what exists and what an even split wanted. */
export function balanceNotes(result) {
  if (!result) return []
  const out = []
  const rows = balanceRows(result)
  const short = rows.filter((r) => r.short)
  for (const r of short) {
    out.push({
      tone: 'warn',
      text: `Only ${r.available} ${r.label} image${r.available === 1 ? '' : 's'} exist in this filter — an even split wanted ${r.fairShare}. `
        + 'The other buckets made up the difference; add more of these for a fuller set.',
    })
  }
  if (result.shortfall > 0) {
    out.push({
      tone: 'warn',
      text: `${result.selected} images selected instead of the ${result.requested} you asked for — the labelled pool has nothing more to give.`,
    })
  }
  if (result.unlabelled > 0) {
    out.push({
      tone: 'info',
      text: `${result.unlabelled} image${result.unlabelled === 1 ? '' : 's'} in this filter have no label yet and were left out — run the 📐 Framing pass to bring them in.`,
    })
  }
  if (result.unknown > 0) {
    out.push({
      tone: 'info',
      text: `${result.unknown} image${result.unknown === 1 ? '' : 's'} came back as “unknown” framing — there is no bucket to balance them into.`,
    })
  }
  if (!out.length) {
    out.push({ tone: 'info', text: 'Every bucket got its even share.' })
  }
  return out
}

/** Whether the balanced pick can run at all, and why not when it cannot — read
 *  from the coverage payload, so the button is honest BEFORE the click. */
export function balanceReadiness({ semanticReady = false, coverage = null,
  engineLabel = 'CLIP', prerequisite = '' } = {}) {
  if (!semanticReady) {
    return { ready: false, reason: prerequisite
      || `Build the ${engineLabel} semantic index first — balanced selection reads it.` }
  }
  if (coverage && coverage.framing_available === false) {
    return { ready: false, reason: 'Run the 📐 Framing pass first — a balanced pick needs the shot type of each image.' }
  }
  return { ready: true, reason: '' }
}

