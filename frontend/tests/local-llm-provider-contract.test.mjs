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

test('LM Studio gets a Start button, and only where it can honour one', () => {
  // This test used to assert the OPPOSITE, on the premise that "LM Studio has no
  // reliable way to be launched from here". Measured, that was wrong: its CLI
  // sits at a fixed per-user path and `lms server start` returns in well under a
  // second. What still has to hold is that the button appears only when the CLI
  // was actually FOUND — an install that has never run LM Studio gets the
  // sentence naming the Developer tab, not a button that cannot work.
  const src = read('components/settings/LocalToolsSection.jsx')
  const lmStatus = src.slice(src.indexOf('function LmStudioStatus'),
                             src.indexOf('function OllamaStatus'))
  assert.ok(lmStatus.length > 0, 'the LM Studio status component is gone')
  assert.match(lmStatus, /if \(l\.installed\) \{/,
    'the button is offered without checking that the CLI was found')
  assert.match(lmStatus, /▶ Start LM Studio/, 'the Start button is gone')
  assert.match(lmStatus, /onClick=\{start\}/)
  // Both providers press ONE routed path, so the two buttons cannot drift into
  // starting different servers.
  assert.match(lmStatus, /'\/api\/local-llm\/start'/,
    'the LM Studio card calls a provider-specific endpoint again')
  // ...and the fallback for an install with no CLI still says where the switch is.
  assert.match(lmStatus, /Developer/)
})

test('the Setup wizard OFFERS the choice, on the default install too', () => {
  // The defect this refuses, found by running Setup on a stock install: every LM
  // Studio sentence on that step sits behind `step.isLmStudio`, so with the
  // default provider the wizard never said the other one existed. Someone running
  // LM Studio and no Ollama was walked through installing and starting Ollama,
  // with no hint there was a choice — on the screen a new install trusts most.
  const page = read('pages/SetupPage.jsx')
  assert.match(page, /Which local LLM do you run\?/,
    'the Setup step no longer offers the provider choice')
  assert.match(page, /\[\['ollama', 'Ollama'\], \['lmstudio', 'LM Studio'\]\]/,
    'the picker no longer lists both providers')
  assert.match(page, /const pickProvider = async \(provider\)/,
    'choosing a provider no longer persists anything')

  // ORDER is the assertion that matters. The picker has to be composed ABOVE the
  // step body, outside every branch — put inside one, it would disappear on
  // exactly the installs that need it.
  const picker = page.indexOf('{llmProviderPicker}')
  const body = page.indexOf('{ollamaBody}')
  assert.ok(picker > -1 && body > -1 && picker < body,
    'the picker is not composed above the step body')

  // ...and it must not be gated on the provider it exists to change.
  const block = page.slice(page.indexOf('const llmProviderPicker'), body)
  assert.doesNotMatch(block, /step\.isLmStudio \?/,
    'the chooser is behind the very flag it is there to flip')
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
  assert.match(page, /if \(step\.isLmStudio && !step\.dockerManaged\)/,
    'the step body must take the LM Studio route on a NATIVE install — and must NOT '
    + 'take it in Docker, where the deployment cards below are the only control that '
    + 'writes ollama.deployment_mode and the launcher stalls without it')
  assert.match(page, /lmStudioNote/,
    'a Docker install set to LM Studio is told nothing about it')
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
