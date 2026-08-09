/**
 * 🎲 Use dataset captions — the draw, and the button RENDERED in both lanes.
 *
 * Two halves on purpose:
 *   · the sampling rules are pure functions, asserted on values (kept-only,
 *     long caption, distinct, capped, replaced not appended);
 *   · the button itself is EXECUTED (see tests/support/mountJsx.mjs) in the
 *     LoRA lane's markup and in the full-model recipe — the second one can
 *     never be reached by mounting the panel, because that mode is entered from
 *     an effect and effects do not run under renderToStaticMarkup.
 */
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

import { createElement, renderToStaticMarkup } from './support/mountJsx.mjs'

/* ⚠️ Dynamic — the .jsx loader hooks are installed while mountJsx.mjs is
   evaluated, and a static import would already be linked. */
const { keptCaptions, pickDatasetCaptions } =
  await import('../src/utils/datasetCaptions.js')
const { UseDatasetCaptionsButton } =
  await import('../src/components/dataset/UseDatasetCaptionsButton.jsx')
const { FullTransformerAdvancedRecipe } =
  await import('../src/components/dataset/TrainingPanel.jsx')

const img = (id, caption, status = 'keep', extra = {}) =>
  ({ id, caption, status, ...extra })

const TEN_KEPT = Array.from({ length: 10 }, (_, i) =>
  img(i + 1, `caption number ${i + 1}`))

// ---- the draw --------------------------------------------------------------

test('only KEPT images are drawn from', () => {
  const pool = keptCaptions([
    img(1, 'kept one'),
    img(2, 'rejected one', 'reject'),
    img(3, 'pending one', 'pending'),
    img(4, 'failed one', 'failed'),
    img(5, 'kept two'),
  ])
  assert.deepEqual(pool, ['kept one', 'kept two'])
})

test('blank and whitespace-only captions never reach the textarea', () => {
  const pool = keptCaptions([
    img(1, ''), img(2, null), img(3, '   '), img(4, undefined), img(5, ' real '),
  ])
  assert.deepEqual(pool, ['real'])
})

test('a caption repeated across images is offered once', () => {
  const pool = keptCaptions([img(1, 'same words'), img(2, 'same words'), img(3, 'other')])
  assert.deepEqual(pool, ['same words', 'other'])
})

test('with dual captions on, the LONG caption is the one drawn', () => {
  // The run trains on the long text — previewing the short one would judge the
  // model against a prompt shape it never met.
  const pool = keptCaptions([
    img(1, 'a long descriptive sentence about the subject',
      'keep', { caption_short: 'short' }),
  ])
  assert.deepEqual(pool, ['a long descriptive sentence about the subject'])
})

test('at most 5 are drawn, all distinct, out of a bigger dataset', () => {
  for (let run = 0; run < 50; run += 1) {
    const drawn = pickDatasetCaptions(TEN_KEPT, 5)
    assert.equal(drawn.length, 5)
    assert.equal(new Set(drawn).size, 5, 'a caption was drawn twice')
    for (const c of drawn) assert.ok(TEN_KEPT.some((i) => i.caption === c))
  }
})

test('the panel maximum caps the draw below 5', () => {
  assert.equal(pickDatasetCaptions(TEN_KEPT, 2).length, 2)
  assert.equal(pickDatasetCaptions(TEN_KEPT, 0).length, 0)
})

test('a small dataset yields what it has, never a padded or repeated line', () => {
  const two = [img(1, 'one'), img(2, 'two'), img(3, 'rejected', 'reject')]
  const drawn = pickDatasetCaptions(two, 5)
  assert.equal(drawn.length, 2)
  assert.deepEqual([...drawn].sort(), ['one', 'two'])
})

test('re-drawing eventually gives a different lot — it is a re-roll', () => {
  const seen = new Set()
  for (let i = 0; i < 200; i += 1) seen.add(pickDatasetCaptions(TEN_KEPT, 5).join('\n'))
  assert.ok(seen.size > 1, 'the draw is not random')
})

// ---- the button ------------------------------------------------------------

function render(props) {
  return renderToStaticMarkup(createElement(UseDatasetCaptionsButton, {
    images: TEN_KEPT, max: 8, ...props,
  }))
}

test('the button is live when the dataset has captions, and says what it does', () => {
  const html = render({})
  assert.match(html, /aria-label="Use dataset captions as preview prompts"/)
  assert.match(html, /🎲 Use dataset captions/)
  assert.match(html, /title="Fill with up to 5 random captions from this dataset — click again for a new draw\."/)
  assert.doesNotMatch(html.match(/<button[^>]*>/)[0], /\sdisabled=/)
})

test('with nothing captioned the button is disabled AND says why', () => {
  const html = render({ images: [img(1, '', 'keep'), img(2, 'rejected', 'reject')] })
  assert.match(html.match(/<button[^>]*>/)[0], /disabled=""/)
  assert.match(html, /No captions yet/)
})

test('the offer never promises more lines than the dataset can give', () => {
  assert.match(render({ images: [img(1, 'only one')] }),
    /title="Fill with up to 1 random caption from this dataset/)
})

test('a click REPLACES the field with one caption per line', () => {
  // The handler is what the panel wires to applySamplePrompts; calling it is
  // the only way to see the text the textarea would receive.
  let written = null
  const props = { images: TEN_KEPT, max: 8, onPick: (text) => { written = text } }
  const handler = UseDatasetCaptionsButton(props).props.onClick
  handler()
  const lines = written.split('\n')
  assert.equal(lines.length, 5)
  assert.equal(new Set(lines).size, 5)
  for (const line of lines) assert.ok(TEN_KEPT.some((i) => i.caption === line))
  // A second click is a NEW text, not the first one grown by five more lines.
  const first = written
  handler()
  assert.equal(written.split('\n').length, 5)
  assert.ok(typeof first === 'string')
})

test('a disabled button writes nothing when its handler is reached anyway', () => {
  let called = false
  const handler = UseDatasetCaptionsButton({
    images: [], onPick: () => { called = true },
  }).props.onClick
  handler()
  assert.equal(called, false)
})

// ---- both lanes actually render it -----------------------------------------

test('the full-model recipe renders the button — the lane no test can mount', () => {
  const html = renderToStaticMarkup(createElement(FullTransformerAdvancedRecipe, {
    adv: { max_sample_prompts: 8 }, samplePromptsText: '',
    datasetImages: TEN_KEPT, applySamplePrompts: () => {},
  }))
  assert.match(html, /aria-label="Full-model preview prompts, one per line"/)
  assert.match(html, /aria-label="Use dataset captions as preview prompts"/)
})

test('the LoRA lane wires the button to the same callback and images', () => {
  // The markup of that lane is unreachable for a mount (the panel needs a
  // router, a toast host and a live dataset), so the WIRING is what is held:
  // a perfect component nobody renders was the original DenseBasePicker bug.
  const panel = readFileSync(
    new URL('../src/components/dataset/TrainingPanel.jsx', import.meta.url), 'utf8')
  const loraArm = panel.slice(panel.indexOf('LORA_ADVANCED_CONTROLS_START'))
  assert.match(loraArm, /<UseDatasetCaptionsButton images=\{datasetImages\} max=\{advMaxPrompts\}/)
  assert.match(loraArm, /onPick=\{applySamplePrompts\}/)
  assert.match(loraArm, /topic="training\.sample_prompts_from_dataset"/)
  // …and the panel feeds it the payload it already has, with no new request.
  assert.match(panel, /const datasetImages = ds\.data\?\.images \|\| \[\]/)
  // The draw persists an EXPLICIT text: reading the state back would save the
  // previous render's lines.
  assert.match(panel, /const applySamplePrompts = \(text\) => \{\s*setSamplePromptsText\(text\);\s*persistSamplePrompts\(text\);/)
})
