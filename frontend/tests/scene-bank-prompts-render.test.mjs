/**
 * 🎬 Scenes from a bank, RENDERED — and the two routes it calls, read from the
 * source.
 *
 * Two failure modes this holds, both already lived through:
 *
 *   · the LIST route. The bank list is `/api/banks`; `/api/bank/<id>` is one
 *     bank. Asking the wrong one 404s into an EMPTY dropdown — "Choose a bank…"
 *     offering nothing, with no error anywhere on screen. A source assertion is
 *     the only thing that catches it here, because the call lives in a callback
 *     that renderToStaticMarkup never reaches.
 *
 *   · the LOADED state. The scene list, its numbering and its thumbnails are a
 *     branch that only exists once scenes are in hand — exactly the kind of
 *     state a default-state render stays green through.
 */
import assert from 'node:assert/strict'
import test from 'node:test'
import { readFileSync } from 'node:fs'

import { renderToStaticMarkup, createElement } from './support/mountJsx.mjs'

const { default: SceneBankPrompts } = await import(
  '../src/components/dataset/studio/SceneBankPrompts.jsx')
const { ToastProvider } = await import('../src/components/common/Toast.jsx')
// The panel carries a HelpBadge, which navigates — it only ever renders inside
// the app's router, so the test gives it one rather than dropping the badge.
const { MemoryRouter } = await import('react-router')

const SRC = readFileSync(
  new URL('../src/components/dataset/studio/SceneBankPrompts.jsx', import.meta.url),
  'utf8')

const render = (value) => renderToStaticMarkup(
  createElement(MemoryRouter, null,
    createElement(ToastProvider, null,
      createElement(SceneBankPrompts, { value, onChange: () => {} }))))

const EMPTY = { source: null, scenes: [], picked: [] }
const LOADED = {
  source: { bank_id: 7, bank_name: 'Chapter 1' },
  scenes: [
    { label: 'Scene 1 — p000.jpg', framing: 'body', prompt: 'street at dawn', image_id: 41 },
    { label: 'Scene 2 — p001.jpg', framing: 'face', prompt: 'close on her face', image_id: 42 },
    { label: 'Scene 3 — p002.jpg', framing: 'body', prompt: 'rooftop chase', image_id: 43 },
  ],
  picked: [2, 0],
}

test('the bank dropdown asks the LIST route, and a scene its own bank thumb', () => {
  assert.match(SRC, /apiFetch\('\/api\/banks'\)/,
    'the list of banks is /api/banks — /api/bank/<id> is ONE bank, and its 404 '
    + 'would arrive as an empty dropdown with no error on screen')
  assert.match(SRC, /apiFetch\(`\/api\/bank\/\$\{bankId\}\/scenes`\)/)
  assert.match(SRC, /\/api\/bank\/\$\{source\.bank_id\}\/thumb\/\$\{s\.image_id\}/)
})

test('nothing loaded: the panel says what it is for, and offers no scene list', () => {
  const html = render(EMPTY)
  assert.ok(html.includes('🎬 Scenes from a bank'))
  assert.ok(html.includes('run a bank’s captions in order'))
  assert.ok(!html.includes('Select all'), 'nothing to select yet')
})

test('loaded: every scene is numbered in BANK order, whatever the tick order', () => {
  const html = render(LOADED)
  // The prompts appear in scene order — the whole point of the lane — and the
  // numbers are the reading order, not the order the user ticked them in.
  const order = ['street at dawn', 'close on her face', 'rooftop chase']
    .map((p) => html.indexOf(p))
  assert.ok(order.every((i) => i >= 0), 'a scene prompt is missing from the panel')
  assert.deepEqual([...order].sort((a, b) => a - b), order,
    'the scenes rendered out of bank order')
  assert.ok(html.includes('>1.</span>') && html.includes('>3.</span>'))
})

test('loaded: the summary counts the picks, and the ticked scenes are pressed', () => {
  const html = render(LOADED)
  assert.ok(html.includes('2 of 3 scene(s) picked from “Chapter 1”'),
    'the count of what a launch will queue must be on screen before the click')
  assert.equal((html.match(/aria-pressed="true"/g) || []).length, 2)
  assert.equal((html.match(/aria-pressed="false"/g) || []).length, 1)
})

test('loaded: each card shows the page it came from', () => {
  const html = render(LOADED)
  assert.ok(html.includes('/api/bank/7/thumb/41'))
  assert.ok(html.includes('loading="lazy"'), 'a chapter is ~80 cards — never eager')
})

test('a scene with no image_id renders anyway, without an <img> to a bad URL', () => {
  const html = render({ ...LOADED,
    scenes: [{ label: 'Scene 1', prompt: 'a scene from before thumbnails' }] })
  assert.ok(html.includes('a scene from before thumbnails'))
  assert.ok(!html.includes('/thumb/undefined'))
})
