/**
 * ☁ ONE estimate line, TWO launch dialogs.
 *
 * The image lane's CloudLaunchDialog used to hold CloudTierEstimate as a
 * private function. When the video lane grew its own launch window the choice
 * was the usual one — copy the twenty lines or share them — and the rules
 * inside them are exactly the ones a cheaper copy would quietly lose: a
 * `pending`/`unavailable` estimate says so instead of inventing a number, a run
 * over the runtime cap is warned about BEFORE the click. So the component moved
 * to components/shared and both dialogs mount it; this file fails if either
 * grows a copy of its own, or stops using the shared one.
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import { readSource } from './support/readSource.mjs'

const image = readSource('src/components/dataset/CloudLaunchDialog.jsx')
const video = readSource('src/components/videobank/VideoCloudLaunchDialog.jsx')
const shared = readSource('src/components/shared/CloudTierEstimate.jsx')

test('both launch dialogs import the shared estimate line', () => {
  for (const [name, src] of [['image', image], ['video', video]]) {
    assert.match(src, /import CloudTierEstimate from '\.\.\/shared\/CloudTierEstimate'/,
      `the ${name} dialog must import the shared CloudTierEstimate`)
    assert.match(src, /<CloudTierEstimate tier=\{t\}/,
      `the ${name} dialog must render the shared CloudTierEstimate per tier`)
  }
})

test('neither dialog carries a private copy of the estimate or the duration formatter', () => {
  for (const [name, src] of [['image', image], ['video', video]]) {
    assert.ok(!/function CloudTierEstimate\(/.test(src), `${name}: private CloudTierEstimate found`)
    assert.ok(!/function _fmtDuration\(/.test(src), `${name}: private _fmtDuration found`)
    // The dialog may still ASK the rule (the image one does, to word its footer
    // when no tier has a usable estimate); it may not RENDER the line itself.
    assert.ok(!/tabular-nums[\s\S]{0,200}dph_total/.test(src),
      `${name}: the estimate line is rendered by the shared component, not here`)
  }
})

test('the shared line owns the honesty rules it was extracted for', () => {
  assert.match(shared, /cloudTierEstimateView\(tier, \{ fullMode \}\)/)
  assert.match(shared, /duration and cost unavailable/)
  assert.match(shared, /Longer than the .* runtime cap/)
})
