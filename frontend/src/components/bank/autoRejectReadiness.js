/* 🧹 Auto-reject — what the two surfaces that offer it are allowed to CLAIM.
 *
 * The bug this file exists to close: the popover printed "(N flagged)" from the
 * payload's `flags` map, which counts EVERY image carrying the flag whatever its
 * status. The button behind it only ever touches undecided images (apply_flags
 * never overrides a ✓/✕, by design). On a fresh bank both are the same number
 * and nothing looks wrong; on the second pass they diverge by everything the
 * first pass rejected. Measured on a real 99 000-image bank: "🌫 Blurry 5 930"
 * offered, 0 rejected on click. The pass was fine. The counter was not.
 *
 * So the number next to a checkbox now comes from `flags_actionable` — the same
 * criterion narrowed to `status='pending'`, i.e. literally what the click flips.
 * `flags` stays untouched for the filter chips, where "show me every blurry
 * image, rejected ones included" is the right question.
 *
 * Two more things the surfaces have to distinguish, both of which used to render
 * as an identical `0`:
 *   - "nothing left to reject" — good news, the bank is clean on that axis;
 *   - "this flag CANNOT catch anything here" — its pass never produced data, so
 *     the 0 is a missing prerequisite, not a result.
 * And the pile no flag can reach at all: every quality flag is gated on
 * `quality_state == 'ok'`, so an image the scan never measured is invisible to
 * all of them. That is not "it is clean": it is "we know nothing about it".
 *
 * Pure and JSX-free on purpose — the wording is the deliverable, so it has to be
 * assertable without mounting a workspace.
 */

const n = (v) => Number(v || 0).toLocaleString('en-US')

/** Which pass has to have produced data before a flag can match anything.
 *  Ids are internal to this module (never stored) — see FLAG_PREREQ_TEXT. */
export const FLAG_PREREQ = {
  blur: 'scan',
  noise: 'scan',
  uniform: 'scan',
  small: 'scan',
  unreadable: 'scan',
  // These two read a column the ORIGINAL quality pass did not write. A bank
  // scanned by an older build has quality_state='ok' and a NULL detail_ratio /
  // bars_ratio, and _flag_filter is NULL-safe on purpose — so the flag matches
  // nothing and says nothing about the images. A rescan backfills them.
  soft_detail: 'provenance',
  bars: 'provenance',
  low_aesthetic: 'score',
  nsfw: 'score',
  watermark: 'watermark',
}

/** The gesture that unblocks each prerequisite, in the app's own button names. */
const FLAG_PREREQ_TEXT = {
  scan: 'run 🔎 Scan first — no image here has been measured yet',
  provenance: 'run 🔎 Rescan first — this bank was scanned before the pass that measures it',
  score: 'run ✨ Score first — nothing here has been scored yet',
  watermark: 'run 🚩 Find watermarks first — nothing here has been checked yet',
}

/** How many images the pass behind `kind` has reached in this bank.
 *  `originMeasured` = the sum of the 🔎 Origin chips, i.e. rows the provenance
 *  half of the quality scan actually wrote (payload.origins). */
function coverage(kind, counts, originMeasured) {
  if (kind === 'scan') return counts?.scanned || 0
  if (kind === 'provenance') return originMeasured || 0
  if (kind === 'score') return counts?.scored || 0
  if (kind === 'watermark') return counts?.watermark_scanned || 0
  return 1
}

/** The missing-prerequisite sentence for one flag, or null when its pass has
 *  produced data for at least one image (in which case a 0 really does mean
 *  "nothing matches"). */
export function flagPrereq(flag, counts, originMeasured) {
  const kind = FLAG_PREREQ[flag]
  if (!kind) return null
  return coverage(kind, counts, originMeasured) > 0 ? null : (FLAG_PREREQ_TEXT[kind] || null)
}

/** The never-scanned pile: how big it is, and what the one gesture that fixes it
 *  actually reaches. Returns null when there is nothing to warn about, so the
 *  caller renders NOTHING on a fully scanned bank rather than a reassuring line
 *  nobody needs.
 *
 *  `scannable` is smaller than `unscanned` by exactly the images that were
 *  rejected before ever being measured — 🔎 Scan skips rejected rows (backend
 *  _scan_pool), so quoting the raw total next to the button would promise a
 *  scan that will not happen for them. */
export function unscannedNotice(counts) {
  const unscanned = counts?.unscanned || 0
  if (unscanned <= 0) return null
  const total = counts?.total || 0
  const scannable = counts?.unscanned_scannable ?? unscanned
  const left = unscanned - scannable
  return {
    unscanned,
    scannable,
    text: `${n(unscanned)} of ${n(total)} image(s) have never been scanned. `
      + 'Quality flags cannot see them — that is not "they are clean", it is '
      + '"nothing has been measured".',
    action: scannable > 0
      ? `🔎 Scan picks up ${n(scannable)} of them.`
      : 'They are all rejected already, so 🔎 Scan skips them — un-reject to measure them.',
    // Only voiced when the two numbers actually differ; on most banks they do not.
    caveat: left > 0 && scannable > 0
      ? `The other ${n(left)} are already rejected and stay out of the scan.`
      : null,
  }
}

/** The count to print next to one auto-reject checkbox, with its wording.
 *  `flagsActionable` is payload.flags_actionable — undecided images only. */
export function flagCandidateLabel(flag, flagsActionable) {
  const count = flagsActionable?.[flag] ?? 0
  return `${n(count)} to reject`
}

/** Sum of the ticked flags' candidates. Deliberately labelled "up to": an image
 *  carrying two ticked flags is counted once per flag, so the sum is a ceiling,
 *  never a promise. Returns null when nothing is ticked. */
export function pickedCandidates(picked, flagsActionable) {
  const flags = [...(picked || [])]
  if (!flags.length) return null
  const sum = flags.reduce((a, f) => a + (flagsActionable?.[f] ?? 0), 0)
  if (sum === 0) {
    return { sum, exact: true, text: 'Nothing to reject — no undecided image carries these flags.' }
  }
  // One flag ticked: the number is exact, so say it plainly.
  const exact = flags.length === 1
  return {
    sum,
    exact,
    text: exact
      ? `${n(sum)} undecided image(s) will be rejected.`
      : `Up to ${n(sum)} undecided image(s) — one carrying two ticked flags is counted twice here.`,
  }
}

/** The 🚀 Launch all variant. There, auto-reject runs AFTER the scan, so a count
 *  taken now is a floor, not the outcome — and saying so is worth more than a
 *  number that will silently move. `scanFirst` = the 🔎 Scan step is ticked. */
export function launchRejectNote(counts, scanFirst) {
  const unscanned = counts?.unscanned || 0
  if (unscanned <= 0) return null
  return scanFirst
    ? `Counts are what the flags catch today. 🔎 Scan runs first and measures `
      + `${n(counts?.unscanned_scannable ?? unscanned)} never-scanned image(s), so they will grow.`
    : `${n(unscanned)} image(s) have never been scanned and no flag can reach them. `
      + 'Tick 🔎 Scan quality above so auto-reject sees them.'
}
