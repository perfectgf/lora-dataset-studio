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
