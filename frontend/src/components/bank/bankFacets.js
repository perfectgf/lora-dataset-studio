/**
 * The Bank's facet TABLES — the flag labels, the resolution/framing/origin
 * buckets and the tile status rings.
 *
 * Pulled out of BankWorkspace.jsx by the Encre redesign because the filter rail,
 * the tiles and the auto-reject sheet now live in three different files and all
 * three read the same tables. Copying them would have been the start of a drift
 * between what a chip is called in the rail and what the same flag is called on
 * a tile.
 *
 * ⚠️ Several `id`s here MUST mirror backend keys (noted per table). They are also
 * stored — in filter query strings and in localStorage — so the repo rule on
 * stored identifiers applies: never rename one without an alias path.
 */
import { ORIGIN_CHIPS, PROVENANCE_FLAG_LABEL } from './bankProvenance.js'

export const FLAG_LABEL = {
  blur: 'Blurry', noise: 'Noisy', uniform: 'Flat',
  small: 'Small', unreadable: 'Unreadable',
  // Provenance pass — same CPU scan, no extras needed.
  ...PROVENANCE_FLAG_LABEL,
  // V2 scoring flags (aesthetic · NSFW · watermark passes).
  low_aesthetic: 'Low aesthetic', nsfw: 'NSFW', watermark: 'Watermark',
}

export const FLAG_HINT = {
  soft_detail: 'The picture stops before the pixels do — usually an enlargement. '
    + 'A soft or out-of-focus shot reads the same, so check before mass-rejecting.',
  bars: 'Flat black letterbox/pillarbox bars — video screenshots and padded stills. '
    + 'A dark-themed screenshot reads the same, so check before mass-rejecting.',
}

// These two are measurements of PROVENANCE, not quality verdicts, which is why
// the overnight pipeline does not offer them (backend PIPELINE_REJECT_FLAGS) and
// why the standalone button prints their caveat instead of hiding it in a
// tooltip. Here you can see the count, undo, and look at the pile first.
// Quality flags the CPU scan produces vs the ones the ML scoring/watermark
// passes add — auto-reject only offers a flag whose pass has actually run.
export const QUALITY_REJECT_FLAGS = ['blur', 'noise', 'uniform', 'small', 'soft_detail', 'bars']
export const SCORE_REJECT_FLAGS = ['low_aesthetic', 'nsfw', 'watermark']

// Resolution tiers — ids + labels MUST mirror backend _RES_BUCKETS (order and
// megapixel bands). Rendered as a dedicated chip row so a mixed dump can be
// sliced by resolution and mass-acted one tier at a time.
export const RES_BUCKETS = [
  { id: 'res_lt_025', label: '< 0.25 MP' },
  { id: 'res_025_1', label: '0.25–1 MP' },
  { id: 'res_1_2', label: '1–2 MP' },
  { id: 'res_2_4', label: '2–4 MP' },
  { id: 'res_gt_4', label: '≥ 4 MP' },
]

// Origin states — ids MUST mirror backend ORIGINS. THREE, never two: 'unknown'
// is the honest answer for any file whose metadata was stripped (which is most
// of them) and it has to be visible and selectable as its own pile.
export const ORIGIN_BUCKETS = ORIGIN_CHIPS

// Framing buckets — ids MUST mirror backend _FRAMING_KEYS. Face/bust/body/back
// are the character composition axes; 'unknown' is a parseable-but-unclassed shot.
export const FRAMING_BUCKETS = [
  { id: 'face', label: '😀 Face' },
  { id: 'bust', label: 'Bust' },
  { id: 'body', label: '🧍 Body' },
  { id: 'back', label: '🔙 Back' },
  { id: 'unknown', label: '❔ Unknown' },
]

export const STATUS_RING = {
  keep: 'ring-2 ring-emerald-400',
  reject: 'ring-2 ring-rose-400 opacity-60',
  pending: '',
}

// Short pass names for the progress bar's pipeline readout.
export const STEP_SHORT = {
  scan: 'Scan', auto_reject: '🧹 Auto-reject', score: 'Score',
  semantic_index: 'Semantic index',
  semantic_dedup: '✂ Crops', watermark: 'Watermarks', faces: 'Person',
  framing: 'Framing', caption: 'Caption', medium: 'Medium',
  angles: '⤢ Angles',
}
