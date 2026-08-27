/** ⚖ The licence acknowledgement gate in front of video training.
 *
 * Some video targets carry a `licence_note` — a licence whose terms reach
 * further than people expect (MiniMax H3's grants no rights at all in four
 * territories, and covers the outputs). The catalogue serves the note and the
 * cards show it, but a banner people scroll past is not consent: training is
 * the step that turns the licence question from theoretical into personal, so
 * THAT is where the app asks — once.
 *
 * Once per PROFILE, not per dataset or per launch: the licence belongs to the
 * model, and a user who has answered for MiniMax H3 has answered for every H3
 * dataset. The answer persists in localStorage — a per-browser convenience,
 * deliberately not a server-side record: the app is not collecting compliance
 * evidence, it is making sure nobody trains a territory-restricted model
 * without having read the one sentence that says so.
 *
 * Pure module: `storage` and `confirmFn` are injected by the caller (the
 * components pass `window.localStorage` / `window.confirm`), so node --test
 * exercises the whole decision without a browser.
 */

const STORE_KEY = 'videoLicenceAck.v1'

/** The acknowledged profiles, `{profile: true}`. A storage that throws or
 * holds junk reads as "nothing acknowledged" — the gate re-asks, which is the
 * safe direction. */
export function readAcks(storage) {
  try {
    return JSON.parse(storage.getItem(STORE_KEY) || '{}') || {}
  } catch {
    return {}
  }
}

/** True when this dataset needs no acknowledgement: its target carries no
 * licence note, or its profile has already been acknowledged here. */
export function hasLicenceAck(ds, storage) {
  if (!ds?.licence_note) return true
  return !!readAcks(storage)[ds.target_profile]
}

/** The one question. The note itself carries the territories and the
 * authorization route (the catalogue owns that text); this only adds what a
 * yes MEANS. */
export function licencePrompt(ds) {
  return (
    `⚖ ${ds.licence_note}\n\n` +
    'Training proceeds under rights you actually hold where you operate — ' +
    'confirm to continue.'
  )
}

/** Ask if needed, remember a yes. Returns whether the launch may proceed.
 * A refused ask persists nothing: the question is asked again next time,
 * because "no" is an answer about now, not about the profile. */
export function ensureLicenceAck(ds, { storage, confirmFn }) {
  if (hasLicenceAck(ds, storage)) return true
  if (!confirmFn(licencePrompt(ds))) return false
  const acks = readAcks(storage)
  acks[ds.target_profile] = true
  try {
    storage.setItem(STORE_KEY, JSON.stringify(acks))
  } catch {
    // Storage refused (private window, quota) — the yes still stands for this
    // launch; the question will simply be asked again next session.
  }
  return true
}
