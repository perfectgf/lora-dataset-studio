/**
 * One local LLM provider, chosen once, honoured everywhere.
 *
 * Two properties, and the first is the repo's Bank/Dataset parity rule applied to
 * a plumbing change: four model pickers list what the local LLM can caption with
 * — two on the Dataset side, one on the Bank side, one in Test Studio. If any of
 * them keeps asking Ollama's own endpoint while the app is set to LM Studio, that
 * surface silently offers models it cannot use, and the user meets a behaviour
 * difference between two screens that are supposed to be one product.
 *
 * The second is that switching provider has to be REACHABLE: a card without a
 * selector is a setting nobody can change.
 */
import assert from 'node:assert/strict'
import test from 'node:test'
import { readFileSync } from 'node:fs'

const read = (rel) => readFileSync(new URL(`../src/${rel}`, import.meta.url), 'utf8')

// The four surfaces that list models, and the file each one lives in.
const PICKERS = [
  ['Dataset — Captions ⚙️ options', 'components/dataset/CaptionOptionsPopover.jsx'],
  ['Dataset — 🧪 Caption Lab', 'components/dataset/CaptionLab.jsx'],
  ['Bank — caption options', 'components/bank/useCaptionOptions.js'],
  ['Test Studio — ✨ Enhance', 'components/dataset/studio/EnhancePromptButton.jsx'],
]

test('every model picker asks the provider-routed endpoint, on both surfaces', () => {
  for (const [label, file] of PICKERS) {
    const src = read(file)
    assert.match(src, /\/api\/local-llm\/models/,
      `${label} (${file}) does not use the routed endpoint`)
    assert.doesNotMatch(src, /\/api\/ollama\/models/,
      `${label} (${file}) still asks Ollama's own endpoint — under LM Studio it would ` +
      'list models this install cannot caption with')
  }
})

test('the provider can actually be switched from Settings', () => {
  const src = read('components/settings/LocalToolsSection.jsx')
  assert.match(src, /id="local-llm-provider"/, 'the provider selector is gone')
  assert.match(src, /setField\('local_llm', 'provider'/, 'the selector does not write the setting')
  assert.match(src, /<option value="ollama"/)
  assert.match(src, /<option value="lmstudio"/)
})

test('both provider cards are configurable whichever one is active', () => {
  // Otherwise setting up the second provider requires switching to it first, and
  // switching to it before it works means a spell of broken captioning.
  const src = read('components/settings/LocalToolsSection.jsx')
  for (const id of ['ollama-url', 'ollama-vision-model',
                    'lmstudio-url', 'lmstudio-vision-model']) {
    assert.match(src, new RegExp(`id="${id}"`), `${id} is missing from Settings`)
  }
  assert.match(src, /target="lmstudio"/, 'the LM Studio card has no Test button')
})

test('the Local tools LED follows the active provider', () => {
  // Keyed on Ollama alone it read "off" on a perfectly healthy LM Studio install.
  const src = read('components/settings/registry.js')
  assert.match(src, /local_llm[\s\S]{0,200}lmstudio[\s\S]{0,120}reachable/,
    'the section status does not consider LM Studio')
})

test('LM Studio is never offered a Start button it cannot honour', () => {
  // Ollama has a real one-click start; LM Studio has no reliable way to be
  // launched from here. A dead button is worse than a sentence saying where the
  // switch is, so the status line names the Developer tab instead.
  const src = read('components/settings/LocalToolsSection.jsx')
  const lmStatus = src.slice(src.indexOf('function LmStudioStatus'),
                             src.indexOf('function OllamaStatus'))
  assert.ok(lmStatus.length > 0, 'the LM Studio status component is gone')
  assert.doesNotMatch(lmStatus, /Start Server<\/button>|onClick=\{start\}/)
  assert.match(lmStatus, /Developer/)
})

test('the Setup step describes the provider this install actually uses', () => {
  // Before this, the wizard sent an LM Studio user to download an Ollama binary,
  // start a daemon it cannot start, and pull a model into a server it does not
  // run — three instructions about the wrong product, on the screen a new install
  // trusts most.
  const step = read('hooks/useSetupSteps.js')
  assert.match(step, /isLmStudio/, 'ollamaStep does not know which provider is selected')
  assert.match(step, /LM Studio — captioning & auto-framing/, 'the step title still names only Ollama')
  assert.match(step, /has no usable model loaded/,
    'the gate does not explain LM Studio\'s own readiness question (loaded, not pulled)')

  const page = read('pages/SetupPage.jsx')
  assert.match(page, /if \(step\.isLmStudio\)/,
    'the step body still renders the Ollama install guide for an LM Studio install')
  // And it returns BEFORE the branches that offer Start/Pull, which do not apply.
  const body = page.slice(page.indexOf('const ollamaBody'))
  assert.ok(body.indexOf('if (step.isLmStudio)') < body.indexOf('if (step.installed)'),
    'the LM Studio branch must come before the Ollama ones, or Start/Pull win')
})

test('the install menu does not offer an Ollama pull under LM Studio', () => {
  const src = read('hooks/useSetupSteps.js')
  assert.match(src, /llmProvider === 'ollama' && o\.reachable && modelName/,
    'installCatalog offers the Ollama model pull regardless of provider')
  assert.match(src, /download models in the LM Studio app/i,
    'the row is turned off without saying why — the dead end this menu exists to close')
})

test('the feature gates read the ACTIVE provider, not Ollama by name', () => {
  // Measured defect: on a machine running LM Studio and no Ollama, both caps.ollama
  // flags are false, so ✨ Enhance and 📐 Classify framing rendered DISABLED with
  // "install Ollama" — while the ⚙️ beside them listed LM Studio models and the
  // backend answered 200. The button could not be clicked at all.
  for (const [label, file] of [
    ['Test Studio ✨ Enhance', 'components/dataset/studio/EnhancePromptButton.jsx'],
    ['Dataset 📐 Classify framing', 'components/dataset/DatasetWorkspace.jsx'],
  ]) {
    const src = read(file)
    assert.match(src, /activeLocalLlm\(caps\)/,
      `${label} still hands its gate caps.ollama, so it is dead under LM Studio`)
  }
  // ...and the gates themselves say the right gesture rather than a translated one.
  const enhance = read('components/dataset/studio/enhanceGate.js')
  assert.match(enhance, /provider === 'lmstudio'/)
  assert.match(enhance, /press Start Server/)
  assert.doesNotMatch(enhance.slice(enhance.indexOf("'lmstudio'"), enhance.indexOf('install it from')),
    /install Ollama/i)

  const framing = read('components/dataset/classifyFramingGate.js')
  assert.match(framing, /provider === 'lmstudio'/)
  assert.match(framing, /Developer tab/)
})
