/* The 🧽 Klein clean's three dials — values, clamps and the sentences that describe
 * them — in ONE pure module, because they are offered on BOTH surfaces (the bank's
 * Level 3 panel and the dataset's Clean bar) and hand-copied wording is how two
 * surfaces of one product drift apart (CLAUDE.md).
 *
 * WHY THEY EXIST (2026-08-31). The clean sent "remove watermark" at 2 MP and wrote the
 * result back at the file's own size — three constants in the backend source, which is
 * not a place a user can look. The maintainer asked the obvious question ("we can't see
 * what is sent? we can't choose the output MP?"), and it is the right one: a user whose
 * mark survived had no dial to turn, and 2 MP was a decision nobody could revisit on a
 * card with room to spare.
 *
 * Every clamp here MIRRORS the backend (watermark_klein.clean_prompt / clean_max_mp /
 * clean_output_mode). The backend is the authority — it re-resolves whatever it is
 * given — so these exist to keep the screen from ever showing a value the pass would
 * silently change under the user. The published `caps.watermark_clean_*` are already
 * resolved by that same backend code, which is why reading them is the honest source.
 */

export const CLEAN_PROMPT_DEFAULT = 'remove watermark'
export const CLEAN_MAX_MP_DEFAULT = 2
export const CLEAN_MAX_MP_MIN = 0.5
export const CLEAN_MAX_MP_MAX = 4
export const CLEAN_OUTPUT_DEFAULT = 'original'

/* The offered sizes. Not a free number field: the useful range is narrow, the cost is
 * superlinear, and a slider would invite 3.7 MP — a value with no meaning that still
 * snaps to the 16-px latent stride on the way in. 2 is the shipped default and stays in
 * the middle so the list reads as "less / as before / more". */
export const CLEAN_MP_CHOICES = [1, 1.5, 2, 3, 4]

export const CLEAN_OUTPUT_MODES = [
  { id: 'original', label: 'Original dimensions (resampled)' },
  { id: 'render', label: 'Render size (file dimensions change)' },
]

/** The prompt this install will really send. Blank means "the default" everywhere —
 *  in config.json, in the text box, and here — never "send nothing". */
export const cleanPromptText = (value) => {
  const text = typeof value === 'string' ? value.trim() : ''
  return text || CLEAN_PROMPT_DEFAULT
}

/** Clamp to the supported range.
 *
 *  Only a number or a numeric string is a value here; everything else is "the default",
 *  never a coercion. `Number()` is far too willing for this job — it turns `null`, `''`
 *  and `[]` into 0, which would clamp to the 0.5 MP FLOOR and hand somebody a mush
 *  render out of a missing setting, and it turns `true` into 1. Every one of those
 *  inputs is reachable: caps before the probe lands, a cleared config key, a
 *  hand-edited config.json. (Caught by kleinCleanOptions.test.js, not by review.) */
export const clampMaxMp = (value) => {
  if (typeof value !== 'number' && typeof value !== 'string') return CLEAN_MAX_MP_DEFAULT
  if (typeof value === 'string' && !value.trim()) return CLEAN_MAX_MP_DEFAULT
  const n = typeof value === 'string' ? Number(value.trim()) : value
  if (!Number.isFinite(n)) return CLEAN_MAX_MP_DEFAULT
  return Math.min(CLEAN_MAX_MP_MAX, Math.max(CLEAN_MAX_MP_MIN, n))
}

/** The choice to show as selected. A stored value the list does not offer (hand-edited
 *  config, or a future build) is kept rather than snapped, so the select never lies
 *  about what will run. */
export const maxMpChoices = (stored) => {
  const value = clampMaxMp(stored)
  return CLEAN_MP_CHOICES.includes(value)
    ? CLEAN_MP_CHOICES
    : [...CLEAN_MP_CHOICES, value].sort((a, b) => a - b)
}

export const normalizeOutput = (value) => {
  const mode = typeof value === 'string' ? value.trim().toLowerCase() : ''
  return CLEAN_OUTPUT_MODES.some((m) => m.id === mode) ? mode : CLEAN_OUTPUT_DEFAULT
}

export const formatMp = (value) => {
  const n = clampMaxMp(value)
  return Number.isInteger(n) ? `${n}` : `${n}`.replace(/\.0+$/, '')
}

/** What a larger processing size buys, and what it costs. Both halves, always: this is
 *  the dial most likely to be pushed to its maximum by someone who reads only the first
 *  half and then asks why their clean takes four times as long. */
export const mpNote = (value) => (
  `Your photo is scaled to at most ${formatMp(value)} MP before Klein re-renders it. `
  + 'Higher means finer regenerated detail — and more VRAM and more time, roughly with '
  + 'the pixel count, on a card that is probably also running ComfyUI. A photo already '
  + 'smaller than this is sent as it is: nothing is ever enlarged.'
)

/** What each write-back mode does to the FILE. The 'render' half has to say the
 *  dimensions change — that is the entire trade, and a user who discovers it after a
 *  batch has no undo but ↩ Restore original. */
export const outputNote = (value) => (normalizeOutput(value) === 'render'
  ? 'The cleaned file is written at the size Klein rendered it, with no second '
    + 'resample — so it keeps that detail, and the file CHANGES DIMENSIONS: it comes '
    + 'out smaller than your original whenever the photo was above the processing '
    + 'size above. ↩ Restore original brings the file back.'
  : 'The render is resampled back to your file’s own dimensions, so a clean never '
    + 'changes the shape of your images. This is what has always shipped.')

/** The prompt, quoted, for the engine tooltip and the panel. Short on purpose — it is
 *  appended to sentences that are already long. */
export const sentPromptLine = (value) => `Sent to Klein: “${cleanPromptText(value)}”`
