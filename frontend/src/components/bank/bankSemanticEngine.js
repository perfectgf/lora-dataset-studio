/* Per-Bank semantic engine state and copy.
 *
 * Kept outside JSX because the important contracts are data contracts: old
 * payloads default to CLIP, readiness is never inferred from the aesthetic
 * `scored` count, and the optional pipeline step exists only for SigLIP2.
 */

export const SEMANTIC_ENGINE_OPTIONS = [
  { id: 'clip', label: 'CLIP', hint: 'Default · produced by ✨ Score' },
  { id: 'siglip2', label: 'SigLIP 2', hint: 'Optional · its own semantic index' },
]

const PIPELINE_BASE_STEPS = [
  'scan', 'auto_reject', 'score', 'semantic_dedup',
  'watermark', 'faces', 'framing', 'caption',
]

export function normalizeSemanticEngine(value) {
  const raw = typeof value === 'object' && value
    ? (value.semantic_engine ?? value.semantic?.engine ?? value.engine)
    : value
  return raw === 'siglip2' ? 'siglip2' : 'clip'
}

export function semanticEngineLabel(value) {
  return normalizeSemanticEngine(value) === 'siglip2' ? 'SigLIP 2' : 'CLIP'
}

export function semanticEnginePatchBody(engine) {
  return { engine: normalizeSemanticEngine(engine) }
}

function firstFinite(...values) {
  for (const value of values) {
    const n = Number(value)
    if (value !== null && value !== '' && Number.isFinite(n)) return n
  }
  return 0
}

function readyValue(value) {
  if (typeof value === 'boolean') return value
  const n = Number(value)
  return Number.isFinite(n) && n > 0
}

/**
 * Normalise both the rich payload introduced with engine selection and the
 * smaller additive counters used during rolling upgrades.
 *
 * Deliberately absent: `counts.scored`. A CLIP aesthetic result is not proof
 * that the currently selected semantic cache is usable (especially after a
 * switch to SigLIP2), which is the distinction this model exists to preserve.
 */
export function semanticEngineState(payload, caps = null) {
  const semantic = payload?.semantic && typeof payload.semantic === 'object'
    ? payload.semantic : {}
  const counts = payload?.counts && typeof payload.counts === 'object'
    ? payload.counts : {}
  const engine = normalizeSemanticEngine(payload)
  const readySignal = semantic.ready ?? counts.semantic_ready
  const indexed = firstFinite(
    semantic.counts?.ok,
    semantic.indexed,
    semantic.ready_count,
    counts.semantic_indexed,
    typeof counts.semantic_ready === 'number' ? counts.semantic_ready : null,
  )
  const total = firstFinite(semantic.counts?.total, semantic.total, counts.total)
  // Only the optional engine has an install gate here. A CLIP cache already
  // produced by Score remains useful for image-only operations even if the
  // scoring Python is temporarily unavailable; text availability is reported
  // separately by `semantic.text`.
  const capabilityKnown = engine === 'siglip2' && !!caps
    && Object.prototype.hasOwnProperty.call(caps, 'bank_siglip2')
  const installed = engine === 'clip' || !capabilityKnown || !!caps.bank_siglip2
  const payloadReady = readyValue(readySignal)
  const ready = installed && payloadReady
  const complete = typeof semantic.complete === 'boolean'
    ? semantic.complete
    : ready && total > 0 && indexed >= total
  const hasStatus = payload?.semantic != null
    || Object.prototype.hasOwnProperty.call(counts, 'semantic_ready')
    || Object.prototype.hasOwnProperty.call(counts, 'semantic_indexed')

  return {
    engine,
    label: semantic.model_label || semanticEngineLabel(engine),
    modelKey: semantic.model_key || null,
    source: semantic.source || (engine === 'clip' ? 'score' : 'semantic_cache'),
    installed,
    payloadReady,
    ready,
    complete,
    needsIndex: typeof semantic.needs_index === 'boolean'
      ? semantic.needs_index : !complete,
    indexed,
    total,
    stale: firstFinite(semantic.counts?.stale),
    text: semantic.text || null,
    // What the index will really run on: {requested, device, gpu}, resolved by
    // the backend against the SigLIP2 interpreter itself. null on a payload from
    // before the engine reported it — unknown, never assumed.
    device: semantic.device && typeof semantic.device === 'object'
      ? semantic.device : null,
    hasStatus,
  }
}

/** SigLIP2 measures ~105 ms/image on the CPU here — 4.3× lighter than the CLIP
 *  ViT-L/14 Score runs, but still an hour on a 30 000-image Bank. Same fix as
 *  Score's, for the same reason: a card that sits unused because the managed
 *  environment ships CPU-only torch. Null when there is nothing worth saying —
 *  CLIP selected, SigLIP2 not installed, or the index already on the GPU. */
export function semanticDeviceNote(state, gpuPresent = true) {
  if (!state || state.engine !== 'siglip2' || !state.installed) return null
  if (!state.device || state.device.gpu) return null
  if (state.device.requested === 'cpu') {
    return { tone: 'info', text: 'SigLIP 2 runs on the CPU because that is what '
      + 'this install asks for (bank_semantic.device).' }
  }
  if (!gpuPresent) {
    return { tone: 'info',
      text: 'SigLIP 2 runs on the CPU — no NVIDIA GPU detected on this machine.' }
  }
  return { tone: 'warn',
    text: 'SigLIP 2 indexes on the CPU — the app’s own environment ships CPU-only '
      + 'PyTorch so a first install stays small. If another Python here already has '
      + 'a working CUDA PyTorch (the one that trains your LoRAs, the one ComfyUI '
      + 'runs on), the index can borrow it. Nothing is downloaded and nothing is '
      + 'installed into it.' }
}

/** Is the "borrow a GPU Python" button worth showing? Only when the note above
 *  is the actionable one — offering it to a card-less machine, or one that
 *  asked for the CPU on purpose, would be noise. */
export function offersSemanticGpuPython(state, gpuPresent = true) {
  return semanticDeviceNote(state, gpuPresent)?.tone === 'warn'
}

export function semanticPrerequisite(state) {
  if (state?.ready) return ''
  if (!state?.hasStatus) {
    return `Reading ${state?.label || semanticEngineLabel(state?.engine)} semantic readiness…`
  }
  if (state?.engine === 'siglip2') {
    return state.installed
      ? 'Build or complete the SigLIP 2 semantic index first.'
      : 'Install SigLIP 2 in Setup ▸ Quality tools, then build this Bank’s index.'
  }
  return 'Run ✨ Score first — it produces this Bank’s CLIP semantic index.'
}

export function semanticIndexActionLabel(state) {
  if (!state || state.engine !== 'siglip2' || !state.installed) return ''
  if (state.complete) return 'Reindex SigLIP 2…'
  if (state.indexed > 0) return 'Complete SigLIP 2 index…'
  return 'Build SigLIP 2 index…'
}

export function semanticPurposeSentence(engine) {
  return `${semanticEngineLabel(engine)} powers semantic search, reference similarity, `
    + 'diversity, balanced sampling and crops/variants for this Bank.'
}

export const SCORE_STAYS_CLIP_SENTENCE = '✨ Score stays on CLIP for aesthetic, NSFW, '
  + 'visual style and Medium.'

export const SEMANTIC_CACHE_SENTENCE = 'Switching engines keeps both caches and both '
  + 'same-shot groupings; it starts nothing automatically and deletes nothing.'

/** The backend pipeline order, with no new step at all for legacy/default CLIP. */
export function pipelineStepKeys(engine) {
  if (normalizeSemanticEngine(engine) !== 'siglip2') return [...PIPELINE_BASE_STEPS]
  const keys = [...PIPELINE_BASE_STEPS]
  keys.splice(keys.indexOf('semantic_dedup'), 0, 'semantic_index')
  return keys
}

/** Default checked steps retain the historic caption-off behaviour. */
export function defaultPipelineStepKeys(engine, ready = {}) {
  return pipelineStepKeys(engine).filter((key) => key !== 'caption' && !!ready[key])
}

/** One request, one vector space: a payload only counts for the engine
 *  (and model) that asked for it. Shared by the workspace's coverage
 *  loader and the curation lanes (useCurationLanes.js). */
export function semanticPayloadMatches(payload, engine, modelKey = null) {
  return payload?.engine === engine
    && (!modelKey || payload?.model_key === modelKey)
}
