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

// The user's choice to "continue without ComfyUI" (Setup step). The backend
// derives `comfyui.skipped` = the stored flag AND no directory configured, so it
// self-annuls the moment a path is entered. We only treat the step as skipped when
// ComfyUI is ALSO not reachable — a running ComfyUI is worth surfacing (partial/
// ready) even if the user once clicked skip.
function comfyuiStep(caps) {
  const c = caps.comfyui || {}
  const missingRequired = kleinMissingRequired(c)
  // Present-but-INVALID required assets (a licence-gate HTML page saved as
  // .safetensors, a truncated download): the file exists so it is NOT in
  // klein_missing, yet it can't load. Without this the step would go green and let
  // a doomed generate crash ComfyUI (the #help "Expecting value: line 1 column 1").
  // Only *blocking* invalids gate readiness; the advisory too_small does not.
  const kleinInvalid = Array.isArray(c.klein_invalid) ? c.klein_invalid : []
  const blockingInvalid = kleinInvalid.filter(
    (i) => i && i.blocking && KLEIN_REQUIRED_ASSETS.includes(i.asset))
  // hasKlein now reflects FULL readiness (all three weights, each a real file), not
  // just the UNET — so the step no longer goes "nothing to do" while the TE/VAE are
  // still missing or while a present asset is actually an unusable stub.
  const hasKlein = missingRequired.length === 0 && blockingInvalid.length === 0
  // Which assets to still offer a download for (required trio + recommended LoRA),
  // so each button can grey out on its own once its file lands.
  const kleinMissing = Array.isArray(c.klein_missing)
    ? c.klein_missing
    : (hasKlein ? [] : ['klein_model'])
  // Skipped is neutral, not a warning — but only when there's genuinely nothing to
  // show (unreachable). It never overrides a reachable ComfyUI's real status.
  const skipped = !!c.skipped && !c.reachable
  const status = skipped ? 'skipped' : gateStatus(c.reachable, hasKlein)
  return {
    id: 'comfyui', title: 'ComfyUI — local generation & Test Studio', recommended: false,
    unlocks: ['Klein engine', 'Test Studio'],
    status, reachable: !!c.reachable, hasKlein, kleinMissing, kleinInvalid, apiUrl: c.api_url || '',
    skipped,
    // Whether comfyui.base_dir actually points at a ComfyUI install (main.py + models/):
    // a wrong/portable-wrapper path scans an empty models/ and finds no checkpoints.
    // baseDir = the path this verdict was PROBED against — the UI must not show the
    // verdict for a freshly typed (unsaved) path, it would judge the wrong string.
    dirConfigured: !!c.dir_configured, dirValid: !!c.dir_valid, resolvedDir: c.resolved_dir || '',
    baseDir: c.base_dir || '',
  }
}

// What "continue without ComfyUI" costs vs keeps — shown in the skip-confirmation
// panel BEFORE the user commits. Sourced from the real capability gates (n'invente
// rien): studio_visible / engines.klein / watermark_klein key on ComfyUI being
// reachable with its models; the training base listers and the LoRA preset picker
// resolve from comfyui.base_dir. Everything under KEPT is independent of ComfyUI.
export const COMFYUI_SKIP_LOST = [
  'Local Klein generation, including the uncensored (NSFW) local lane',
  'Watermark cleaning with Klein (LaMa inpainting and crop still work)',
  'Test Studio (comparing checkpoints, every model family)',
  'Training on your own ComfyUI base models (built-in and cloud bases still work)',
  'Picking LoRA presets from what is on disk (free-text entry still works)',
]
export const COMFYUI_SKIP_KEPT = [
  'Scraping and dataset curation',
  'Captioning (Ollama vision model or the API engines)',
  'Nano Banana and ChatGPT image engines',
  'LoRA training — local ai-toolkit and cloud (vast.ai)',
  'Publishing datasets and LoRAs to Hugging Face',
]

// Map a /api/setup/comfyui-dir verdict to the wizard's inline feedback: a tone
// (drives the colour) and an actionable message. `suggestion` is carried through so
// the caller can render an "adopt this folder" button for the launcher-folder case.
// Pure + exhaustive so node --test can lock every branch. `checking` is the UI's own
// in-flight state; `empty` (nothing typed) renders nothing here — the skip panel owns it.
export function comfyuiDirVerdict(check) {
  const c = check || {}
  const resolved = c.resolved || ''
  const suggestion = c.suggestion || ''
  switch (c.status) {
    case 'valid':
      return { tone: 'ok', suggestion: '',
        message: resolved ? `ComfyUI found at ${resolved}.` : 'ComfyUI found.' }
    case 'nested':
      return { tone: 'warn', suggestion,
        message: `This looks like the launcher/parent folder — did you mean ${suggestion}?` }
    case 'missing':
      return { tone: 'warn', suggestion: '',
        message: "That folder doesn't exist yet — check the path." }
    case 'empty_dir':
      return { tone: 'warn', suggestion: '',
        message: 'That folder is empty — point at the folder that holds main.py and a models/ folder.' }
    case 'not_comfyui':
      return { tone: 'warn', suggestion: '',
        message: "This folder isn't a ComfyUI install — it must contain main.py and a models/ folder. "
          + 'For the portable build, point at the inner …\\ComfyUI_windows_portable\\ComfyUI.' }
    default:
      return { tone: 'muted', suggestion: '', message: '' }
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
  // Four scoped ML capabilities now (face scoring, masks, watermark inpainting,
  // bank scoring) — each installs/repairs on its own. The step is ready only when
  // all of them are in.
  const parts = [!!caps.face_scoring, !!caps.masks, !!caps.watermark_inpaint,
    !!caps.bank_scoring]
  const ready = parts.every(Boolean)
  const partial = parts.some(Boolean)
  return {
    id: 'quality', title: 'Quality tools (ML extras)', recommended: false,
    unlocks: ['Face-similarity scoring', 'Person masks', 'Watermark inpainting',
      'Bank scoring (aesthetic · NSFW · style)'],
    status: ready ? 'ready' : (partial ? 'partial' : 'available'),
    faceScoring: !!caps.face_scoring, masks: !!caps.masks,
    watermarkInpaint: !!caps.watermark_inpaint,
    bankScoring: !!caps.bank_scoring,
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
