// Pure derivation of the guided Setup wizard state from live capabilities.
// No I/O — deterministic, so it is the single source of truth for card status.

export const SETUP_STEP_IDS = ['image', 'comfyui', 'ollama', 'quality', 'training']

// Tool reachable + its extra piece present -> ready; reachable only -> partial.
function gateStatus(reachable, complete) {
  if (reachable && complete) return 'ready'
  if (reachable) return 'partial'
  return 'available'
}

function imageStep(caps) {
  const e = caps.engines || {}
  const ready = e.nanobanana || e.chatgpt || e.klein
  return {
    id: 'image', title: 'Image generation', recommended: true,
    unlocks: ['Nano Banana (Gemini)', 'ChatGPT (gpt-image-2)', 'Klein (local)'],
    status: ready ? 'ready' : 'available',
    engines: { nanobanana: !!e.nanobanana, chatgpt: !!e.chatgpt, klein: !!e.klein },
  }
}

// The Klein engine needs three weights on disk (UNET + text-encoder + VAE); the
// consistency LoRA is only recommended, so it never gates readiness. The backend
// lists the setup_installer action names still absent in comfyui.klein_missing —
// that is the source of truth, mirroring capabilities.klein_ready. Older payloads
// (pre-klein_missing) fall back to the UNET-only scan.
const KLEIN_REQUIRED_ASSETS = ['klein_model', 'klein_text_encoder', 'klein_vae']

// setup_installer action name -> the short human word used in Setup/picker hints.
export const KLEIN_ASSET_LABELS = {
  klein_model: 'model',
  klein_text_encoder: 'text encoder',
  klein_vae: 'VAE',
}

// Human names of the REQUIRED Klein weights still missing (recommended LoRA
// excluded), in a stable canonical order — so both the Setup step header and the
// picker's Klein hint can say exactly what to download. Empty => the trio is on disk.
export function kleinMissingLabels(kleinMissing) {
  const m = Array.isArray(kleinMissing) ? kleinMissing : []
  return KLEIN_REQUIRED_ASSETS.filter((a) => m.includes(a)).map((a) => KLEIN_ASSET_LABELS[a])
}

function kleinMissingRequired(c) {
  if (Array.isArray(c.klein_missing)) {
    return c.klein_missing.filter((a) => KLEIN_REQUIRED_ASSETS.includes(a))
  }
  // Legacy fallback: judge on the UNET scan alone (the old, under-strict signal).
  return (c.models && c.models.klein && c.models.klein.length) ? [] : ['klein_model']
}

function comfyuiStep(caps) {
  const c = caps.comfyui || {}
  const missingRequired = kleinMissingRequired(c)
  // hasKlein now reflects FULL readiness (all three weights), not just the UNET —
  // so the step no longer goes "nothing to do" while the TE/VAE are still missing.
  const hasKlein = missingRequired.length === 0
  // Which assets to still offer a download for (required trio + recommended LoRA),
  // so each button can grey out on its own once its file lands.
  const kleinMissing = Array.isArray(c.klein_missing)
    ? c.klein_missing
    : (hasKlein ? [] : ['klein_model'])
  const status = gateStatus(c.reachable, hasKlein)
  return {
    id: 'comfyui', title: 'ComfyUI — local generation & Test Studio', recommended: false,
    unlocks: ['Klein engine', 'Test Studio'],
    status, reachable: !!c.reachable, hasKlein, kleinMissing, apiUrl: c.api_url || '',
    // Whether comfyui.base_dir actually points at a ComfyUI install (main.py + models/):
    // a wrong/portable-wrapper path scans an empty models/ and finds no checkpoints.
    // baseDir = the path this verdict was PROBED against — the UI must not show the
    // verdict for a freshly typed (unsaved) path, it would judge the wrong string.
    dirConfigured: !!c.dir_configured, dirValid: !!c.dir_valid, resolvedDir: c.resolved_dir || '',
    baseDir: c.base_dir || '',
  }
}

function ollamaStep(caps) {
  const o = caps.ollama || {}
  const status = gateStatus(o.reachable, o.vision_model_ready)
  return {
    id: 'ollama', title: 'Ollama — captioning & auto-framing', recommended: false,
    unlocks: ['Captioning', 'Auto-classify framing', 'Auto head-crop'],
    status, reachable: !!o.reachable, visionModelReady: !!o.vision_model_ready,
    url: o.url || '', visionModel: o.vision_model || '',
    // Execution-independent install signal (binary on disk) vs `reachable` (server
    // answering): installed && !reachable -> "installed but stopped", offer a Start.
    installed: !!o.installed, binaryPath: o.binary_path || '',
  }
}

function qualityStep(caps) {
  // Three scoped ML capabilities now (face scoring, masks, watermark inpainting) —
  // each installs/repairs on its own. The step is ready only when all three are in.
  const parts = [!!caps.face_scoring, !!caps.masks, !!caps.watermark_inpaint]
  const ready = parts.every(Boolean)
  const partial = parts.some(Boolean)
  return {
    id: 'quality', title: 'Quality tools (ML extras)', recommended: false,
    unlocks: ['Face-similarity scoring', 'Person masks', 'Watermark inpainting'],
    status: ready ? 'ready' : (partial ? 'partial' : 'available'),
    faceScoring: !!caps.face_scoring, masks: !!caps.masks,
    watermarkInpaint: !!caps.watermark_inpaint,
  }
}

function trainingStep(caps) {
  const a = caps.aitoolkit || {}
  return {
    id: 'training', title: 'LoRA training — ai-toolkit', recommended: false,
    unlocks: ['LoRA training', 'JoyCaption captioning (bonus)'],
    status: a.valid ? 'ready' : 'available',
    valid: !!a.valid,
  }
}

export function deriveSetupSteps(caps) {
  const c = caps || {}
  return [imageStep(c), comfyuiStep(c), ollamaStep(c), qualityStep(c), trainingStep(c)]
}

// The user's live capability checklist (Summary card). Watermark inpainting is a
// distinct ML extra (simple-lama-inpainting) — an existing install that never ran
// it must SEE it as still missing here, not be told "everything's ready".
export function deriveCapabilitySummary(caps) {
  const c = caps || {}
  const e = c.engines || {}
  const o = c.ollama || {}
  const cap = c.captioners || {}
  return [
    { label: 'Nano Banana (Gemini)', ok: !!e.nanobanana },
    { label: 'ChatGPT (gpt-image-2)', ok: !!e.chatgpt },
    { label: 'Klein (local)', ok: !!e.klein },
    { label: 'Captioning', ok: !!(cap.joycaption || cap.ollama) },
    { label: 'Auto-framing & head-crop', ok: !!(o.reachable && o.vision_model_ready) },
    { label: 'Face-similarity scoring', ok: !!c.face_scoring },
    { label: 'Person masks', ok: !!c.masks },
    { label: 'Watermark inpainting', ok: !!c.watermark_inpaint },
    { label: 'LoRA training', ok: !!c.training_visible },
    { label: 'Test Studio', ok: !!c.studio_visible },
  ]
}

export function recommendedMet(caps) {
  const e = (caps && caps.engines) || {}
  return !!(e.nanobanana || e.chatgpt || e.klein)
}
