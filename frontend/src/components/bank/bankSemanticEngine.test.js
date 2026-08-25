import assert from 'node:assert/strict'
import test from 'node:test'

import {
  SEMANTIC_CACHE_SENTENCE, SCORE_STAYS_CLIP_SENTENCE,
  defaultPipelineStepKeys, normalizeSemanticEngine, pipelineStepKeys,
  offersSemanticGpuPython, semanticDeviceNote,
  semanticEnginePatchBody, semanticEngineState, semanticIndexActionLabel,
  semanticPrerequisite, semanticPurposeSentence,
} from './bankSemanticEngine.js'

test('old and partial payloads default to CLIP', () => {
  assert.equal(normalizeSemanticEngine(null), 'clip')
  assert.equal(normalizeSemanticEngine({ counts: { scored: 12 } }), 'clip')
  assert.equal(normalizeSemanticEngine({ semantic: { engine: 'siglip2' } }), 'siglip2')
})

test('the PATCH contract sends exactly the selected durable engine', () => {
  assert.deepEqual(semanticEnginePatchBody('clip'), { engine: 'clip' })
  assert.deepEqual(semanticEnginePatchBody('siglip2'), { engine: 'siglip2' })
  assert.deepEqual(semanticEnginePatchBody('not-an-engine'), { engine: 'clip' })
})

test('semantic readiness is distinct from scored and tolerates both payload shapes', () => {
  const scoredOnly = semanticEngineState({ counts: { total: 20, scored: 20 } },
    { bank_scoring: true })
  assert.equal(scoredOnly.ready, false, 'scored must never manufacture semantic readiness')
  assert.equal(scoredOnly.indexed, 0)

  const compact = semanticEngineState({
    semantic_engine: 'clip',
    counts: { total: 20, scored: 20, semantic_ready: true, semantic_indexed: 18 },
  }, { bank_scoring: true })
  assert.equal(compact.ready, true)
  assert.equal(compact.indexed, 18)

  const rich = semanticEngineState({
    semantic: { engine: 'siglip2', ready: true, complete: true,
      counts: { total: 20, ok: 20 } },
    counts: { total: 20, scored: 0 },
  }, { bank_siglip2: true })
  assert.equal(rich.ready, true)
  assert.equal(rich.complete, true)
  assert.equal(rich.indexed, 20)

  const cachedClipWithoutRunner = semanticEngineState({
    semantic: { engine: 'clip', ready: true, counts: { total: 20, ok: 20 } },
  }, { bank_scoring: false })
  assert.equal(cachedClipWithoutRunner.ready, true,
    'an existing CLIP cache stays usable for image-only semantic operations')
})

test('an absent SigLIP2 capability never presents a cached-looking false readiness', () => {
  const state = semanticEngineState({
    semantic_engine: 'siglip2',
    semantic: { engine: 'siglip2', ready: true, counts: { total: 5, ok: 5 } },
  }, { bank_siglip2: false })
  assert.equal(state.payloadReady, true)
  assert.equal(state.ready, false)
  assert.equal(semanticIndexActionLabel(state), '')
  assert.match(semanticPrerequisite(state), /Setup ▸ Quality tools/)
})

test('semantic gating names the selected engine prerequisite, never scored', () => {
  const clip = semanticEngineState({ semantic_engine: 'clip',
    counts: { semantic_ready: false, scored: 400 } }, { bank_scoring: true })
  const siglip = semanticEngineState({ semantic_engine: 'siglip2',
    counts: { semantic_ready: false, scored: 400 } }, { bank_siglip2: true })
  assert.match(semanticPrerequisite(clip), /✨ Score/)
  assert.match(semanticPrerequisite(siglip), /SigLIP 2 semantic index/)
  assert.match(semanticPrerequisite(semanticEngineState(null)), /Reading CLIP/)
})

test('pipeline keeps the CLIP plan and inserts SigLIP2 index between score and dedup', () => {
  const clip = pipelineStepKeys('clip')
  const siglip = pipelineStepKeys('siglip2')
  assert.equal(clip.includes('semantic_index'), false)
  assert.equal(siglip.filter((step) => step === 'semantic_index').length, 1)
  assert.equal(siglip.indexOf('semantic_index'), siglip.indexOf('score') + 1)
  assert.equal(siglip.indexOf('semantic_dedup'), siglip.indexOf('semantic_index') + 1)
  assert.deepEqual(clip, [
    'scan', 'auto_reject', 'score', 'semantic_dedup',
    'watermark', 'faces', 'framing', 'caption',
  ])
})

test('default pipeline checks semantic index only for a ready SigLIP2 install', () => {
  const ready = {
    scan: true, auto_reject: true, score: true, semantic_index: true,
    semantic_dedup: true, watermark: true, faces: true, framing: true, caption: true,
  }
  assert.equal(defaultPipelineStepKeys('clip', ready).includes('semantic_index'), false)
  assert.equal(defaultPipelineStepKeys('siglip2', ready).includes('semantic_index'), true)
  assert.equal(defaultPipelineStepKeys('siglip2', { ...ready, semantic_index: false })
    .includes('semantic_index'), false)
  assert.equal(defaultPipelineStepKeys('siglip2', ready).includes('caption'), false)
})

test('visible copy preserves Score ownership, both caches and both groupings', () => {
  assert.match(semanticPurposeSentence('clip'), /^CLIP powers semantic search/)
  assert.match(semanticPurposeSentence('siglip2'), /^SigLIP 2 powers semantic search/)
  assert.match(SCORE_STAYS_CLIP_SENTENCE, /aesthetic, NSFW, visual style and Medium/)
  assert.match(SEMANTIC_CACHE_SENTENCE, /keeps both caches/)
  assert.match(SEMANTIC_CACHE_SENTENCE, /both same-shot groupings/)
  assert.match(SEMANTIC_CACHE_SENTENCE, /starts nothing automatically/)
  assert.match(SEMANTIC_CACHE_SENTENCE, /deletes nothing/)
})


// ── Which device the index really uses, and the way out ──────────────────────

const siglip2 = (device, extra = {}) => semanticEngineState(
  { semantic: { engine: 'siglip2', ready: true, device, ...extra },
    counts: { total: 10 } },
  { bank_siglip2: true })

test('an idle card under a CPU index is named, and the borrow route offered', () => {
  const note = semanticDeviceNote(siglip2({ requested: 'auto', device: 'cpu', gpu: false }), true)
  assert.equal(note.tone, 'warn')
  assert.match(note.text, /CPU/)
  // The offer names what it will NOT do, because that is the objection.
  assert.match(note.text, /nothing is installed into it/)
  assert.ok(offersSemanticGpuPython(
    siglip2({ requested: 'auto', device: 'cpu', gpu: false }), true))
})

test('an index already on the GPU says nothing at all', () => {
  const state = siglip2({ requested: 'auto', device: 'cuda', gpu: true })
  assert.equal(semanticDeviceNote(state, true), null)
  assert.equal(offersSemanticGpuPython(state, true), false)
})

test('a card-less machine is never sold a GPU Python', () => {
  const state = siglip2({ requested: 'auto', device: 'cpu', gpu: false })
  const note = semanticDeviceNote(state, false)
  assert.equal(note.tone, 'info')
  assert.ok(!/CUDA/.test(note.text), 'no CUDA pitch to a machine with no card')
  assert.equal(offersSemanticGpuPython(state, false), false)
})

test('a deliberate CPU preference is explained, not treated as a problem', () => {
  const state = siglip2({ requested: 'cpu', device: 'cpu', gpu: false })
  assert.equal(semanticDeviceNote(state, true).tone, 'info')
  assert.equal(offersSemanticGpuPython(state, true), false)
})

test('CLIP, an uninstalled engine or an unknown device stay silent', () => {
  assert.equal(semanticDeviceNote(semanticEngineState({ semantic: { engine: 'clip' } }), true), null)
  assert.equal(semanticDeviceNote(
    semanticEngineState({ semantic: { engine: 'siglip2', ready: true } },
                        { bank_siglip2: false }), true), null)
  // No device reported (older payload) is UNKNOWN — never "it is on the CPU".
  assert.equal(semanticDeviceNote(siglip2(null), true), null)
})
