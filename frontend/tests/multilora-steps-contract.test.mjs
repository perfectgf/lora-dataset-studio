/**
 * Contract of 🎛 CFG/steps in the MULTI-LoRA branch of the Test Studio.
 *
 * Reported: "I cannot set the number of steps when I have two LoRAs selected, in
 * blend or in comparison." The field was not disabled — it did not exist, in the
 * one branch of the app that had no per-dataset payload to read the ladders from.
 * The end-to-end proof (non-default steps in the workflow ComfyUI receives, in
 * both modes) is backend/tests/test_studio_multilora_steps.py; the rule is
 * unit-tested in src/components/dataset/studio/studioAxes.test.js.
 *
 * What only the sources can say is asserted here.
 */
import assert from 'node:assert/strict'
import { readSource } from './support/readSource.mjs'
import test from 'node:test'

import { getHelpTopic } from '../src/help/helpRegistry.js'
import { WHATS_NEW } from '../src/whatsNew.js'
import { WHATS_NEW_ARCHIVE } from '../src/whatsNewArchive.js'

// The entry under test may have moved to the archive since it shipped
// (see whatsNew.js, rule "Keep the list tidy") — search the union.
const ALL_WHATS_NEW = [...WHATS_NEW, ...WHATS_NEW_ARCHIVE]

const read = (rel) => readSource(`src/${rel}`)
const SHELL = read('components/dataset/studio/StudioShell.jsx')
const COMPARISON = read('components/dataset/studio/ComparisonStudio.jsx')
const SETUP = read('components/dataset/studio/StudioRunSetup.jsx')
const STACK = read('components/dataset/studio/loraStack.js')

test('the ladders are fetched with the bases, in the call that already existed', () => {
  // A second round-trip for four constants would be a second thing to go wrong.
  assert.match(SHELL, /setAxes\(d\.axes \|\| null\)/)
  assert.match(SHELL, /axes=\{axes\}/)
})

test('the multi-LoRA panel really mounts the pickers — the same component', () => {
  // Not a copy: the single-LoRA studio and the canvas render this very component,
  // so the three surfaces cannot end up with three different steps ladders.
  assert.match(COMPARISON, /import AxisPickers from '\.\/AxisPickers'/)
  assert.match(COMPARISON, /axisSlot=\{axes \? \(/)
  assert.match(COMPARISON, /stepsChoices=\{axes\.steps_choices\}/)
  assert.match(COMPARISON, /cfgChoices=\{axes\.cfg_choices\}/)
  assert.match(SETUP, /\{axisSlot\}/)
})

test('the pickers survive Blend, where the strength axis does NOT', () => {
  // The reported bug covered BOTH modes. Blend drops the strength sweep because
  // each LoRA carries its weight — steps are a render setting and must stay.
  const strengthAt = SETUP.indexOf('{!combine && (')
  const slotAt = SETUP.indexOf('{axisSlot}')
  assert.ok(strengthAt > 0 && slotAt > strengthAt)
  // …and the slot is NOT wrapped in the same `!combine` guard.
  assert.doesNotMatch(SETUP, /\{!combine && \(\s*\{axisSlot\}/)
})

test('what is ticked really reaches the request body', () => {
  assert.match(COMPARISON, /\.\.\.axisPayload\(\{ cfgs: effectiveCfgs, steps: effectiveSteps, steps2: effectiveSteps2 \}\)/)
  // Last in the body: the panel's own axes win over any global setting of the
  // same name rather than being silently overwritten by the spread that follows.
  const genAt = COMPARISON.indexOf('...genSettings,')
  const axisAt = COMPARISON.indexOf('...axisPayload(')
  assert.ok(genAt > 0 && axisAt > genAt)
})

test('the cost counter carries the new axes', () => {
  // A panel announcing 6 cells while the queue receives 18 is the failure mode.
  assert.match(STACK, /axisTotal = 1 \}\)/)
  assert.match(SETUP, /axisTotal,\s*\}\);/)
  assert.match(SETUP, /axisTotal > 1 &&/)
  assert.match(COMPARISON, /axisTotal=\{axisTotal\(\{ cfgs: effectiveCfgs/)
})

test('a leftover SDXL second pass does not follow into a Z-Image run', () => {
  // These settings are persisted and the family changes with the ticked LoRAs.
  // Without the guard, a pass-2 value chosen for SDXL would be sent as an axis
  // to a family whose workflow has no second pass.
  assert.match(COMPARISON, /axes\?\.steps2_choices\s*\?\s*effectiveAxis\(selSteps2, axes\.default_steps2\) : \[\]/)
})

test('the persisted keys are NEW ones', () => {
  // Reusing an existing key would change the meaning of what is already in
  // someone's browser. These three did not exist before.
  for (const k of ['studioComp_cfgs', 'studioComp_steps', 'studioComp_steps2']) {
    assert.match(COMPARISON, new RegExp(k))
  }
})

test('the fix has a help topic and a What\'s-new entry', () => {
  const topic = getHelpTopic('studio-multilora-steps')
  assert.ok(topic, 'studio-multilora-steps must be a registered help topic')
  assert.ok(topic.keywords.includes('steps'))

  const entry = ALL_WHATS_NEW.find((e) => e.id === '2026-08-03-multilora-steps-and-cfg')
  assert.ok(entry, 'the steps fix needs a What\'s-new entry')
})
