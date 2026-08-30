/**
 * The LM Studio side of the UI, exercised as BEHAVIOUR rather than as source text.
 *
 * Why this file exists: the contract test beside it asserts regexes against the
 * JSX, and a verification pass proved that is not enough — it deleted the entire
 * LM Studio ladder from both feature gates and every one of those regexes still
 * matched, 10/10 and 5/5 green. Source-as-text can say a line is WRITTEN; only
 * calling the function can say what it DOES.
 *
 * Everything here is a pure function, so `node --test` can call it for real. Each
 * test names the mutation it refuses.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

const { activeLocalLlm, localLlmLabel } = await import('../src/utils/localLlm.js')
const { enhanceBlocker } = await import('../src/components/dataset/studio/enhanceGate.js')
const { classifyBlockedReason } = await import('../src/components/dataset/classifyFramingGate.js')
const { sectionStatus } = await import('../src/components/settings/registry.js')
const { deriveSetupSteps, deriveCapabilitySummary, ollamaGateReason } =
  await import('../src/hooks/useSetupSteps.js')

// A machine that runs LM Studio and has never installed Ollama — the shape every
// one of these defects needed, and that no test was using.
const LMS = (over = {}) => ({
  local_llm: { provider: 'lmstudio' },
  ollama: { installed: false, reachable: false, vision_model_ready: false, vision_model: '' },
  lmstudio: { reachable: true, model_ready: true, vision_model: '',
              detail: 'qwen/qwen3-vl-4b loaded', ...over },
  captioners: { joycaption: false, ollama: false, local_llm: true, local_llm_vision: true },
})

// --- activeLocalLlm: the translation everything else depends on --------------

test('the active provider is normalised into the shape the gates already read', () => {
  const lms = activeLocalLlm(LMS())
  assert.equal(lms.provider, 'lmstudio')
  assert.equal(lms.reachable, true)
  assert.equal(lms.vision_model_ready, true)
  // LM Studio is never "installed but stopped" from here — there is no binary this
  // app can detect and no way for it to start one, so that branch must never fire.
  assert.equal(lms.installed, true)

  const oll = activeLocalLlm({ ollama: { installed: true, reachable: false } })
  assert.equal(oll.provider, 'ollama')
  assert.equal(oll.installed, true)
  assert.equal(oll.reachable, false)
})

test('an install predating the setting reads as Ollama, never as nothing', () => {
  assert.equal(activeLocalLlm({}).provider, 'ollama')
  assert.equal(activeLocalLlm(undefined).provider, 'ollama')
  assert.equal(localLlmLabel({}), 'Ollama')
  assert.equal(localLlmLabel(LMS()), 'LM Studio')
})

// --- the two feature gates: MUTATION = delete their LM Studio ladder ---------

test('✨ Enhance is usable on an LM Studio install with a model loaded', () => {
  // The defect: both caps.ollama flags are false there, so the button rendered
  // disabled with "install Ollama" while the ⚙️ listed LM Studio models and the
  // backend answered 200. Deleting the LM Studio branch makes this red.
  assert.equal(enhanceBlocker(activeLocalLlm(LMS())), null)
})

test('✨ Enhance names LM Studio’s own gesture when it is not ready', () => {
  const down = enhanceBlocker(activeLocalLlm(LMS({ reachable: false, model_ready: false })))
  assert.match(down, /LM Studio/)
  assert.match(down, /Start Server/)
  assert.doesNotMatch(down, /install (it|Ollama)/i)

  const noModel = enhanceBlocker(activeLocalLlm(LMS({ model_ready: false })))
  assert.match(noModel, /no usable model loaded/)
  assert.doesNotMatch(noModel, /pull/i, 'an LM Studio model is loaded, never pulled')
})

test('📐 Classify framing behaves the same way, on the same states', () => {
  assert.equal(classifyBlockedReason(activeLocalLlm(LMS())), null)
  const down = classifyBlockedReason(activeLocalLlm(LMS({ reachable: false, model_ready: false })))
  assert.match(down, /LM Studio/)
  assert.match(down, /Developer/)
  const noModel = classifyBlockedReason(activeLocalLlm(LMS({ model_ready: false })))
  assert.match(noModel, /no usable model loaded/)
})

test('the Ollama ladder is untouched — three states, three sentences', () => {
  const oll = (o) => enhanceBlocker(activeLocalLlm({ ollama: o }))
  assert.match(oll({ installed: false, reachable: false }), /install it from Settings/)
  assert.match(oll({ installed: true, reachable: false }), /installed but not running/)
  assert.match(oll({ installed: true, reachable: true, vision_model_ready: false }),
    /is not downloaded yet/)
  assert.equal(oll({ installed: true, reachable: true, vision_model_ready: true }), null)
})

// --- registry LED: MUTATION = read caps.captioners.ollama again --------------

test('the Local tools and Captioning LEDs follow the active provider', () => {
  const caps = { ...LMS(), comfyui: { reachable: true }, aitoolkit: { valid: true } }
  assert.equal(sectionStatus('local-tools', caps), 'ready')
  assert.equal(sectionStatus('captioning', caps), 'ready')
  // …and an LM Studio that is NOT reachable must not read ready either.
  const down = { ...caps, lmstudio: { ...caps.lmstudio, reachable: false, model_ready: false },
                 captioners: { joycaption: false, ollama: false, local_llm: false } }
  assert.notEqual(sectionStatus('captioning', down), 'ready')
})

test('an older caps payload without the new field still lights the Ollama way', () => {
  const legacy = { comfyui: { reachable: true }, aitoolkit: { valid: true },
                   ollama: { reachable: true }, captioners: { joycaption: false, ollama: true } }
  assert.equal(sectionStatus('local-tools', legacy), 'ready')
  assert.equal(sectionStatus('captioning', legacy), 'ready')
})

// --- the Setup step and its gate --------------------------------------------

const step = (caps) => deriveSetupSteps(caps).find((s) => s.id === 'ollama')

test('the Setup step reads LM Studio’s readiness, not Ollama’s', () => {
  const s = step(LMS())
  assert.equal(s.isLmStudio, true)
  assert.equal(s.reachable, true)
  assert.equal(s.visionModelReady, true)
  assert.match(s.title, /LM Studio/)
  assert.equal(ollamaGateReason(s), null)
})

test('the gate names LM Studio’s own two failures, and neither mentions Ollama', () => {
  const down = ollamaGateReason(step(LMS({ reachable: false, model_ready: false })))
  assert.match(down, /LM Studio is not answering/)
  assert.doesNotMatch(down, /Ollama/)

  const noModel = ollamaGateReason(step(LMS({ model_ready: false })))
  assert.match(noModel, /no usable model loaded/)
  assert.doesNotMatch(noModel, /Ollama/)
})

test('a working LM Studio install is not counted as two missing capabilities', () => {
  // The defect: both rows read caps.captioners.ollama, so the summary told a
  // perfectly working install it was short of two things — on the screen whose
  // only job is to say whether you are ready.
  const rows = deriveCapabilitySummary(LMS())
  const byLabel = Object.fromEntries(rows.map((r) => [r.label, r.ok]))
  assert.equal(byLabel.Captioning, true)
  assert.equal(byLabel['Auto-framing & head-crop'], true)
})

test('an LM Studio that cannot caption is still counted as missing', () => {
  // The other half: the rows must not become unconditionally green either.
  const rows = deriveCapabilitySummary({
    ...LMS({ reachable: false, model_ready: false }),
    captioners: { joycaption: false, ollama: false, local_llm: false, local_llm_vision: false },
  })
  const byLabel = Object.fromEntries(rows.map((r) => [r.label, r.ok]))
  assert.equal(byLabel.Captioning, false)
  assert.equal(byLabel['Auto-framing & head-crop'], false)
})
