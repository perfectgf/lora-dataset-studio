/* 🗃️ Bank provenance — how the four raw signals from the quality scan are PHRASED.
 * Kept free of JSX so `node --test` can run it: the wording is the feature here,
 * and getting it wrong is how a score turns into a verdict nobody earned.
 *
 * Backend: backend/app/services/image_provenance.py (what each signal measures,
 * and its measured limits). Nothing here re-measures anything — it only reads
 * `detail_ratio`, `origin`, `origin_evidence`, `bars_ratio` and `jpeg_quality`
 * off the image payload. */

/* Measured ratio -> the enlargement it actually corresponds to. The estimator is
 * monotone but COMPRESSED: on synthetic ground truth built from real bank
 * photos, a true 1/4 enlargement reads ~0.55 and a true 1/8 reads ~0.49, not
 * 0.25 and 0.125. Rendering the raw ratio as "real pixels" would therefore
 * overstate every enlarged image by a factor of two. This table converts back,
 * log-interpolated between the anchors.
 * [measured, true fraction of the stored size] — measured descending. */
export const DETAIL_CALIBRATION = [
  [1.0, 1.0],
  [0.8, 0.5],
  [0.65, 0.33],
  [0.55, 0.25],
  [0.49, 0.13],
]

/* Below this the reading stops being distinguishable from a merely soft
 * photograph: on ground truth, one genuinely full-resolution image in ten
 * measures ~0.84. Above this bar we say nothing at all rather than cry wolf —
 * which does mean about half of exact-2x enlargements (median 0.80) go
 * unmentioned. Missing some is the right way to be wrong here. */
export const DETAIL_SPEAK_BELOW = 0.78

export function trueFraction(measured) {
  if (typeof measured !== 'number' || !Number.isFinite(measured)) return null
  const t = DETAIL_CALIBRATION
  if (measured >= t[0][0]) return 1
  if (measured <= t[t.length - 1][0]) return t[t.length - 1][1]
  for (let i = 0; i < t.length - 1; i += 1) {
    const [m1, f1] = t[i]
    const [m2, f2] = t[i + 1]
    if (measured <= m1 && measured >= m2) {
      const span = m1 - m2
      const k = span === 0 ? 0 : (m1 - measured) / span
      // Interpolate in log space: the fractions are a geometric ladder.
      return Math.exp(Math.log(f1) + k * (Math.log(f2) - Math.log(f1)))
    }
  }
  return null
}

/* Round to something a person would say out loud, not to 1123 px. */
function nicePx(px) {
  if (px >= 1000) return Math.round(px / 100) * 100
  if (px >= 200) return Math.round(px / 50) * 50
  return Math.max(16, Math.round(px / 8) * 8)
}

/* "2048 px stored · ~512 px of real detail", or null when the image is either
 * unmeasured or indistinguishable from a full-resolution one. */
export function detailSummary(img) {
  const r = img && img.detail_ratio
  if (typeof r !== 'number' || !Number.isFinite(r)) return null
  const longSide = Math.max(img.width || 0, img.height || 0)
  if (!longSide) return null
  if (r >= DETAIL_SPEAK_BELOW) {
    return { ratio: r, stored: longSide, real: null, soft: false,
             text: `${longSide} px · detail all the way up` }
  }
  const real = nicePx(longSide * trueFraction(r))
  return {
    ratio: r,
    stored: longSide,
    real,
    soft: true,
    text: `${longSide} px stored · ~${real} px of real detail`,
  }
}

/* The caveat that has to travel WITH the number. */
export const DETAIL_CAVEAT =
  'Measured, not read from the file: this is how far real detail goes, so an ' +
  'enlargement, a soft focus, motion blur and heavy denoising all read the same. ' +
  'Treat it as a score, like sharpness — not as proof the image was enlarged. ' +
  'The pixel figure is a rough order of magnitude and tends to read low.'

/* --- origin --------------------------------------------------------------- */
const ORIGIN_TEXT = {
  ai: { icon: '🤖', label: 'AI', tone: 'violet',
        why: 'The file still carries generation metadata.' },
  camera: { icon: '📷', label: 'Camera', tone: 'emerald',
            why: 'The file still carries camera EXIF (make, model or exposure).' },
  unknown: { icon: '❔', label: 'Unknown', tone: 'slate',
             why: 'No metadata left. Scrapers, chat apps and social networks strip '
                + 'it, so this is the normal answer — it is NOT evidence either way.' },
}

const EVIDENCE_TEXT = {
  'png-prompt': 'ComfyUI workflow in the PNG',
  'png-workflow': 'ComfyUI workflow in the PNG',
  'png-parameters': 'A1111-style generation parameters',
  'png-sd-metadata': 'Stable Diffusion metadata',
  'png-invokeai_metadata': 'InvokeAI metadata',
  'png-dream': 'Generator metadata',
  'xmp-ai-source': 'C2PA/XMP "generated" marker',
  'software-tag': 'A generator named in the Software tag',
  'exif-camera': 'Camera EXIF',
}

export function originLabel(img) {
  const state = (img && img.origin) || null
  if (!state || !ORIGIN_TEXT[state]) return null
  const base = ORIGIN_TEXT[state]
  const ev = img.origin_evidence ? EVIDENCE_TEXT[img.origin_evidence] : null
  return { ...base, state, evidence: ev, detail: ev ? `${ev}.` : base.why }
}

/* Generator bucket sizes (SDXL / FLUX / SD1.5). A camera essentially never
 * produces one of these exact shapes. */
const GENERATOR_SIZES = [
  [512, 512], [768, 768], [1024, 1024], [1536, 1536],
  [1152, 896], [896, 1152], [1216, 832], [832, 1216],
  [1344, 768], [768, 1344], [1536, 640], [640, 1536],
  [1216, 704], [704, 1216],
]

export function isGeneratorSize(width, height) {
  if (!width || !height) return false
  return GENERATOR_SIZES.some(([w, h]) => w === width && h === height)
}

/* A PRESUMPTION, never a state. Only offered when the metadata said nothing —
 * if the file still had a marker we already know the answer and guessing over it
 * would be worse than useless. */
export function originHint(img) {
  if (!img || img.origin !== 'unknown') return null
  if (!isGeneratorSize(img.width, img.height)) return null
  return `${img.width}×${img.height} is a standard generator size, and there is no `
       + 'camera EXIF. That is a hint, not a finding — plenty of crops and '
       + 'downloads land on round numbers.'
}

/* --- the small facts ------------------------------------------------------ */
export function barsSummary(img, barsMax = 0.04) {
  const r = img && img.bars_ratio
  if (typeof r !== 'number' || !Number.isFinite(r) || r <= 0) return null
  const pct = Math.round(r * 100)
  if (!pct) return null
  return { ratio: r, over: r > barsMax, text: `${pct}% black bars` }
}

export function jpegQualitySummary(img) {
  const q = img && img.jpeg_quality
  if (typeof q !== 'number' || !Number.isFinite(q)) return null
  return { quality: q, text: `JPEG q${Math.round(q)}` }
}

/* Labels for the two flags this pass adds, so the chip row and the lightbox
 * cannot drift apart. Ids are user-facing filter keys — never rename them. */
export const PROVENANCE_FLAG_LABEL = {
  soft_detail: '🫧 Soft detail',
  bars: '🎞 Black bars',
}

export const ORIGIN_CHIPS = [
  { id: 'ai', label: '🤖 AI' },
  { id: 'camera', label: '📷 Camera' },
  { id: 'unknown', label: '❔ Unknown' },
]
