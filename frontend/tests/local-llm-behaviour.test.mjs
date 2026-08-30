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

const { activeLocalLlm, localLlmLabel, modelPickerCopy } =
  await import('../src/utils/localLlm.js')
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
  lmstudio: { installed: true, reachable: true, model_ready: true, vision_model: '',
              detail: 'qwen/qwen3-vl-4b loaded', ...over },
  captioners: { joycaption: false, ollama: false, local_llm: true, local_llm_vision: true },
})

// --- activeLocalLlm: the translation everything else depends on --------------

test('the active provider is normalised into the shape the gates already read', () => {
  const lms = activeLocalLlm(LMS())
  assert.equal(lms.provider, 'lmstudio')
  assert.equal(lms.reachable, true)
  assert.equal(lms.vision_model_ready, true)
  // `installed` is LM Studio's OWN signal (its CLI on disk), never Ollama's and
  // never a stand-in for `reachable` — the two are what make the third rung, and
  // the third rung is what puts a ▶ Start on the screen.
  assert.equal(lms.installed, true)
  assert.equal(activeLocalLlm(LMS({ installed: false })).installed, false)

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

// --- modelPickerCopy: the words the four pickers share ----------------------

test('a picker names the provider whose list it is showing', () => {
  const oll = modelPickerCopy('ollama')
  const lms = modelPickerCopy('lmstudio')
  assert.equal(oll.label, 'Ollama')
  assert.equal(lms.label, 'LM Studio')
  assert.equal(oll.modelLabel, 'Ollama vision model')
  assert.equal(lms.modelLabel, 'LM Studio vision model')
  // A response that predates the field, or a failed fetch, means Ollama — the only
  // thing that existed before.
  assert.equal(modelPickerCopy(undefined).label, 'Ollama')
  assert.equal(modelPickerCopy('').label, 'Ollama')
})

test('only Ollama is offered a pull, because only Ollama has one', () => {
  // The defect this refuses: the ✨ Enhance popover told an LM Studio user to
  // "start it from Settings › Local tools to list your pulled models" — naming a
  // Start button this app does not have for LM Studio, and a gesture LM Studio
  // does not have at all. Models are LOADED there, inside its own window.
  assert.equal(modelPickerCopy('ollama').canPull, true)
  assert.equal(modelPickerCopy('lmstudio').canPull, false)
  assert.match(modelPickerCopy('ollama').down, /^Ollama/)
  assert.match(modelPickerCopy('lmstudio').down, /^LM Studio/)
  assert.match(modelPickerCopy('lmstudio').down, /Developer/)
  assert.doesNotMatch(modelPickerCopy('lmstudio').down, /Ollama|pull/i)
})

test('the three run-window tooltips name one provider each, and never the other', () => {
  // These moved out of the JSX because the Bank's frozen surface inventory cannot
  // see a computed label. This test is what replaced that frozen literal, and it
  // checks BOTH providers rather than one spelling.
  for (const [provider, mine, theirs] of
       [['ollama', 'Ollama', 'LM Studio'], ['lmstudio', 'LM Studio', 'Ollama']]) {
    const c = modelPickerCopy(provider)
    for (const [name, text] of Object.entries(
      { registerHint: c.registerHint, perRunHint: c.perRunHint, inertHint: c.inertHint })) {
      assert.ok(text.includes(mine), `${name} stopped naming ${mine}`)
      assert.ok(!text.includes(theirs), `${name} names ${theirs} under ${provider}`)
    }
  }
  // The register hint keeps the substance the frozen entry carried.
  assert.match(modelPickerCopy('ollama').registerHint, /uncensored \(abliterated\)/)
  assert.match(modelPickerCopy('ollama').registerHint, /make the search find more/)
  // A model is PULLED on one and LOADED on the other; the sentence says which.
  assert.match(modelPickerCopy('ollama').perRunHint, /pulled/)
  assert.match(modelPickerCopy('lmstudio').perRunHint, /loaded/)
})

// --- the two feature gates: MUTATION = delete their LM Studio ladder ---------

test('✨ Enhance is usable on an LM Studio install with a model loaded', () => {
  // The defect: both caps.ollama flags are false there, so the button rendered
  // disabled with "install Ollama" while the ⚙️ listed LM Studio models and the
  // backend answered 200. Deleting the LM Studio branch makes this red.
  assert.equal(enhanceBlocker(activeLocalLlm(LMS())), null)
})

test('✨ Enhance names LM Studio’s own gesture when it is not ready', () => {
  // Two rungs, two gestures. Installed (its CLI is on disk) means LDS can start
  // the server itself, so the sentence points at the button here; not installed
  // means the only way in is the app's own menu. Sending someone to another
  // application when a button on this page would do it is the dead end this
  // module exists to remove.
  const stopped = enhanceBlocker(activeLocalLlm(LMS({ reachable: false, model_ready: false })))
  assert.match(stopped, /LM Studio is not running/)
  assert.match(stopped, /Settings/)
  assert.doesNotMatch(stopped, /Developer/,
    'it still sends the user to the app menu although a Start button is right there')

  const down = enhanceBlocker(activeLocalLlm(
    LMS({ installed: false, reachable: false, model_ready: false })))
  assert.match(down, /LM Studio/)
  assert.match(down, /Start Server/)
  assert.doesNotMatch(down, /install (it|Ollama)/i)

  const noModel = enhanceBlocker(activeLocalLlm(LMS({ model_ready: false })))
  assert.match(noModel, /no usable model loaded/)
  assert.doesNotMatch(noModel, /pull/i, 'an LM Studio model is loaded, never pulled')
})

test('📐 Classify framing behaves the same way, on the same states', () => {
  assert.equal(classifyBlockedReason(activeLocalLlm(LMS())), null)
  const stopped = classifyBlockedReason(activeLocalLlm(LMS({ reachable: false, model_ready: false })))
  assert.match(stopped, /installed but its server is not running/)
  assert.match(stopped, /▶ Start LM Studio/)
  const down = classifyBlockedReason(activeLocalLlm(
    LMS({ installed: false, reachable: false, model_ready: false })))
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

test('the skip settles the step under EITHER provider', () => {
  // The defect: `skipped` excluded LM Studio, and the backend derived the flag on
  // Ollama's reachability — so the panel wrote a flag that read back as false and
  // the wizard asked again at every Next. The step could never be settled.
  const skipped = (over) => step({ ...LMS(over), ollama: { ...LMS().ollama, skipped: true } })
  const lms = skipped({ reachable: false, model_ready: false })
  assert.equal(lms.skipped, true, 'an LM Studio install can never close this step')
  assert.equal(lms.status, 'skipped')
  assert.equal(ollamaGateReason(lms), null)

  // …and the Ollama half is unchanged.
  const oll = deriveSetupSteps({ ollama: { reachable: false, skipped: true } })
    .find((s) => s.id === 'ollama')
  assert.equal(oll.skipped, true)
  assert.equal(ollamaGateReason(oll), null)
})

test('the installed-but-stopped rung reads LM Studio’s install, never Ollama’s', () => {
  // The defect: `installed` read caps.ollama.installed whatever the provider was.
  // On a machine holding an idle Ollama install — and someone running LM Studio
  // with its server not started — the welcome scan said "installed — not running"
  // and pointed at a ▶ Start button that would have started the OTHER product.
  // The rung is real for LM Studio now, which makes reading the RIGHT flag matter
  // more, not less: it decides which server a click starts.
  const ollamaOnDiskOnly = {
    ...LMS({ installed: false, reachable: false, model_ready: false }),
    ollama: { installed: true, binary_path: '/opt/ollama', reachable: false },
  }
  const s = step(ollamaOnDiskOnly)
  assert.equal(s.installed, false, 'the stopped-but-installed rung fired for the wrong product')
  assert.equal(s.binaryPath, '', 'a path to the other provider’s binary reached the screen')

  // LM Studio present but its server down IS the rung — that is what offers Start.
  const stopped = step(LMS({ reachable: false, model_ready: false }))
  assert.equal(stopped.installed, true)
  assert.equal(stopped.reachable, false)

  // ...and a running one still reads as installed.
  assert.equal(step(LMS()).installed, true)

  // The Ollama half is untouched — that rung is the whole point there.
  const oll = deriveSetupSteps({
    ollama: { installed: true, binary_path: '/usr/bin/ollama', reachable: false },
  }).find((x) => x.id === 'ollama')
  assert.equal(oll.installed, true)
  assert.equal(oll.binaryPath, '/usr/bin/ollama')
})

test('both Describe surfaces offer the same way out of a fence', async () => {
  // The Bank's Describe bar took an `onFenceBlocked` prop that NOTHING passed, so a
  // fenced run there printed a raw error and no remedy, while the dataset's Describe
  // modal offered "unload and retry" for the identical refusal. A dead prop reads as
  // covered ground, which is why this asserts the wiring rather than the prop.
  const { readFileSync } = await import('node:fs')
  const read = (rel) => readFileSync(new URL(rel, import.meta.url), 'utf8')
  for (const [name, src] of [
    ['bank', read('../src/components/bank/DescribeFilterBar.jsx')],
    ['dataset', read('../src/components/dataset/studio/DescribeImageModal.jsx')],
  ]) {
    assert.match(src, /useOllamaFence/, `the ${name} Describe lost its fence guard`)
    assert.match(src, /runGuarded\(/, `the ${name} Describe no longer replays a fenced call`)
    assert.match(src, /<OllamaFenceNotice fence=\{fence\} onUnload=\{unloadAndRetry\} onStop=\{stopWaiting\} \/>/,
      `the ${name} Describe renders no way out of a fence`)
  }
})

test('the skip panel names the provider it is actually about', async () => {
  // A source-order check, because `node --test` never executes JSX: a const used
  // above its own declaration passes every test here and throws on the screen.
  // This file has been bitten by that before, so the order is asserted, not hoped.
  const { readFileSync } = await import('node:fs')
  const src = readFileSync(new URL('../src/pages/SetupPage.jsx', import.meta.url), 'utf8')
  const declared = src.indexOf("const llmName = step.isLmStudio")
  const firstUse = src.indexOf('{llmName}')
  assert.ok(declared > -1 && firstUse > -1, 'the provider name is no longer computed')
  assert.ok(declared < firstUse,
    'llmName is used before it is declared — a temporal dead zone no test here can execute')
  assert.doesNotMatch(src.slice(src.indexOf('const ollamaSkipNotice'), src.indexOf('const ollamaSkipNotice') + 500),
    /without Ollama\./, 'the neutral notice still names Ollama whatever the provider')
})
