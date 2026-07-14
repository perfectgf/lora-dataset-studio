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

function comfyuiStep(caps) {
  const c = caps.comfyui || {}
  const hasKlein = !!(c.models && c.models.klein && c.models.klein.length)
  const status = gateStatus(c.reachable, hasKlein)
  return {
    id: 'comfyui', title: 'ComfyUI — local generation & Test Studio', recommended: false,
    unlocks: ['Klein engine', 'Test Studio'],
    status, reachable: !!c.reachable, hasKlein, apiUrl: c.api_url || '',
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
