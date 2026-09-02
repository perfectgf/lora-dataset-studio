/** ✨ Neural render (DLSS 5) — the dials, their defaults and the words for them.
 *
 * PURE JS, shared by the two surfaces that offer the verb (the video dataset's
 * Clips section and the Video Test Studio's clip history) so the dialog says the
 * same thing in both places — one contract, two hosts. The backend
 * (`services/neural_render.normalize_params`) is the authority on the ranges;
 * these mirror it so the dialog refuses locally what the server would refuse.
 *
 * WHY ONLY THESE DIALS. The model has more controls than this (intensity, skin
 * structure, preset, style). Swept in both directions through the bridge this
 * app drives, they produced bit-identical output — a slider that does nothing
 * is worse than none. `tone` and `structure` are the two that act, and
 * `automask` marginally; that is the whole surface.
 */

export const NR_DEFAULTS = Object.freeze({
  tone: 1.0, structure: 1.0, automask: false, temporal: 'auto',
})

/** Optical Flow's width floor, mirrored from the backend's TEMPORAL_MIN_WIDTH.
 *  Below it temporal mode is refused by the driver; `auto` falls back to still. */
export const TEMPORAL_MIN_WIDTH = 704

export const TEMPORAL_MODES = Object.freeze([
  { id: 'auto', label: 'Auto', hint: `Keeps the frame history when the clip is at least ${TEMPORAL_MIN_WIDTH} px wide, still mode otherwise.` },
  { id: 'on', label: 'Temporal', hint: 'Motion-aware: steadier across frames. Needs a clip at least 704 px wide.' },
  { id: 'off', label: 'Still', hint: 'Every frame on its own. Works at any size; may shimmer a little on fine detail.' },
])

/** The two starting points worth naming. "Keep tones" exists because at its
 *  default the model relights flat art and greys pure whites — measured on a
 *  webtoon page, a single dial put every white back. */
export const NR_PRESETS = Object.freeze([
  { id: 'photo', label: 'Photoreal (default)', params: { tone: 1.0, structure: 1.0, automask: false } },
  { id: 'flat', label: 'Flat art / anime: keep tones', params: { tone: 0.0, structure: 1.0, automask: false } },
])

const clamp01x2 = (v, fallback) => {
  const n = Number(v)
  if (!Number.isFinite(n)) return fallback
  return Math.min(2, Math.max(0, Math.round(n * 100) / 100))
}

/** Coerce anything the dialog holds into what the server accepts. Never throws:
 *  a bad value falls back to its default, which is what the range inputs do too. */
export function normalizeNrParams(raw) {
  const r = raw || {}
  const temporal = TEMPORAL_MODES.some((m) => m.id === r.temporal) ? r.temporal : NR_DEFAULTS.temporal
  return {
    tone: clamp01x2(r.tone, NR_DEFAULTS.tone),
    structure: clamp01x2(r.structure, NR_DEFAULTS.structure),
    automask: !!r.automask,
    temporal,
  }
}

/** Which preset the current dials match, or null when they are the user's own. */
export function presetFor(params) {
  const p = normalizeNrParams(params)
  const hit = NR_PRESETS.find((pr) => pr.params.tone === p.tone
    && pr.params.structure === p.structure && pr.params.automask === p.automask)
  return hit ? hit.id : null
}

/** What `auto` will do for a clip of this width — said BEFORE the render, so
 *  nobody discovers the fallback in a log. `null` width = unknown, say so. */
export function temporalOutcome(mode, width) {
  if (mode === 'off') return 'still mode'
  if (width == null) return mode === 'on' ? 'temporal mode (refused if the clip is narrower than 704 px)' : 'auto — decided per clip'
  if (width < TEMPORAL_MIN_WIDTH) {
    return mode === 'on'
      ? `refused: this clip is ${width} px wide, temporal needs ${TEMPORAL_MIN_WIDTH}`
      : `still mode (${width} px wide, temporal needs ${TEMPORAL_MIN_WIDTH})`
  }
  return 'temporal mode'
}

/** The one sentence the buttons show when the lane is not set up, built from the
 *  capability's own list so the wording lives in one place (the backend). */
export function nrRefusal(status) {
  if (!status) return 'Neural rendering: checking what this machine has…'
  if (status.ready) return null
  const parts = Array.isArray(status.missing) ? status.missing : []
  return parts.length ? `Neural rendering needs ${parts.join('; ')}.` : 'Neural rendering is not available on this machine.'
}
