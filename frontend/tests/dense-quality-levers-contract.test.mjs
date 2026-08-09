/**
 * The full-model recipe card's three quality levers, rendered for real.
 *
 * Two failures this file exists to catch, neither of which a source-text test
 * can see:
 *
 *  1. the card does not execute. A settings card is JSX that nothing in the
 *     suite ever runs, so a bad reference inside it ships green and explodes on
 *     the one screen it belongs to. Mounting is the only proof.
 *  2. the card stops telling the truth about MONEY. "Images per step" is the
 *     only dense setting billed by the hour: raising it to 4 makes the run take
 *     about four times as long on a rented 80 GB GPU. If that multiplier ever
 *     silently drops out of the card, the setting still works and the user
 *     still pays — they just find out from the invoice. So the multiplier is
 *     asserted as CONTENT, at the value the server sent, not as a class name.
 *
 * The server is the authority on bounds and on the cost figure; the card must
 * render what it was handed rather than a second copy of the arithmetic, which
 * is why the props below carry deliberately non-default numbers.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { createElement, renderToStaticMarkup } from './support/mountJsx.mjs'

/* ⚠️ Dynamic — the hooks that teach Node to read .jsx are installed while
   mountJsx.mjs is evaluated, and a static import would already be linked. */
const { FullTransformerAdvancedRecipe } =
  await import('../src/components/dataset/TrainingPanel.jsx')

const ADV = {
  dense_lr: 1e-6, dense_lr_min: 1e-7, dense_lr_max: 5e-6,
  dense_resolution: 1024, dense_resolution_choices: [768, 1024],
  dense_save_every: 250, dense_save_every_min: 100, dense_save_every_max: 5000,
  dense_max_step_saves: 1, dense_max_step_saves_max: 3,
  dense_grad_accum: 1, dense_grad_accum_choices: [1, 2, 4, 8],
  dense_time_multiplier: 1, dense_images_per_step: 1,
  dense_lr_schedule: 'constant',
  dense_lr_schedule_choices: ['constant', 'constant_with_warmup', 'cosine'],
  dense_warmup: 100, dense_warmup_min: 10, dense_warmup_max: 1000,
  dense_warmup_applies: false,
  dense_timestep_type: 'linear',
  dense_timestep_type_choices: ['linear', 'sigmoid', 'weighted'],
  dense_fp8_export: true, dense_keep_bf16: true,
}

const render = (adv = {}) => renderToStaticMarkup(createElement(
  FullTransformerAdvancedRecipe,
  { stepsOverride: '', setStepsOverride: () => {}, adv: { ...ADV, ...adv },
    saveAdv: () => {}, samplePromptsDefault: ['{trigger} portrait'] },
))

test('the card renders its three quality levers at all', () => {
  const html = render()
  for (const label of ['Images per step', 'Learning-rate schedule', 'Noise schedule']) {
    assert.ok(html.includes(label), `the card never rendered "${label}"`)
  }
})

test('every choice the server allows is offered, and nothing it refuses is', () => {
  const html = render()
  for (const n of [1, 2, 4, 8]) {
    assert.ok(html.includes(`value="${n}"`), `images-per-step ${n} is missing`)
  }
  for (const value of ['constant', 'constant_with_warmup', 'cosine',
    'linear', 'sigmoid', 'weighted']) {
    assert.ok(html.includes(`value="${value}"`), `${value} is missing`)
  }
  // `shift` parses in AI Toolkit and is refused by the server for Krea 2 (its
  // shift would be computed from a four-times-too-large token count). Offering
  // it would be a control whose only outcome is a rejected save.
  assert.ok(!html.includes('value="shift"'), 'shift must not be offered')
})

test('raising images per step states the cost BEFORE the run is launched', () => {
  const cheap = render()
  assert.ok(/One image per step is the default/.test(cheap),
    'the default should still explain what the lever buys')

  const dear = render({ dense_grad_accum: 4, dense_time_multiplier: 4,
    dense_images_per_step: 4 })
  assert.ok(dear.includes('4 images instead of 1'),
    'the card must say how many images a step now learns from')
  assert.ok(/4× as long/.test(dear) && /4× as much/.test(dear),
    'the card must state BOTH the time and the money, in the same breath')
  // The figure is the server's, not a second implementation of the arithmetic:
  // a card that recomputed it could disagree with the recipe that was sent.
  const disagreeing = render({ dense_grad_accum: 2, dense_time_multiplier: 7 })
  assert.ok(/7× as long/.test(disagreeing),
    'the card must print the multiplier the server sent, not its own')
})

test('warmup is offered only on the schedule that accepts it', () => {
  assert.ok(!render().includes('Warm up over'),
    'a constant schedule must not offer warmup steps')
  assert.ok(render({ dense_lr_schedule: 'constant_with_warmup',
    dense_warmup_applies: true }).includes('Warm up over'),
    'the warmup schedule must offer its length')
  // AI Toolkit's cosine/constant schedulers RAISE on num_warmup_steps, so this
  // is not cosmetic gating — it mirrors what the server will actually send.
  assert.ok(!render({ dense_lr_schedule: 'cosine' }).includes('Warm up over'),
    'cosine must not offer warmup steps')
})

test('no control in the card can force the row wider than a 400 px screen', () => {
  // A <select> sizes itself on its WIDEST option — "Sigmoid — favour…" is
  // 303 px — and a flex item defaults to min-width:auto, so the Noise-schedule
  // row could not shrink and spilled 13.9 px off the right edge at 400 px
  // (measured in a headless browser at the real nesting). min-w-0 on the
  // control AND on the label that wraps it is what lets the row wrap instead.
  // Asserted on the rendered markup because node has no layout engine: this
  // guards the mechanism, the browser measurement guards the outcome.
  const html = render()
  const selects = html.match(/<select[^>]*>/g) || []
  assert.ok(selects.length >= 4, 'expected the card to render its selects')
  for (const tag of selects) {
    assert.ok(/class="[^"]*\bmin-w-0\b/.test(tag),
      `a select that cannot shrink will overflow a narrow screen: ${tag.slice(0, 90)}`)
  }
  const labels = html.match(/<label class="[^"]*flex[^"]*"/g) || []
  for (const tag of labels) {
    assert.ok(/\bmin-w-0\b/.test(tag) || /\bflex-wrap\b/.test(tag) || /\bflex-col\b/.test(tag),
      `a rigid flex label spills its control off-screen: ${tag.slice(0, 90)}`)
  }
})

test('the card survives a server that sent no advanced payload at all', () => {
  // Older backend, failed request, dataset still loading: the card must fall
  // back to the shipped defaults rather than crash the training screen.
  const html = renderToStaticMarkup(createElement(
    FullTransformerAdvancedRecipe,
    { stepsOverride: '', setStepsOverride: () => {}, samplePromptsDefault: [] },
  ))
  assert.ok(html.includes('Images per step') && html.includes('Noise schedule'))
  assert.ok(/One image per step is the default/.test(html))
})
