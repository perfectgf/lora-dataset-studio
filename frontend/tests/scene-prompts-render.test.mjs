/**
 * 🎬 Scenes, RENDERED — both sources, and the routes it calls, read from the
 * source.
 *
 * Three failure modes this holds, two already lived through:
 *
 *   · the LIST routes. The bank list is `/api/banks`; `/api/bank/<id>` is one
 *     bank. The dataset list is `/api/dataset/list`. Asking the wrong one 404s
 *     into an EMPTY dropdown — "Choose a bank…" offering nothing, with no error
 *     anywhere on screen. A source assertion is the only thing that catches it
 *     here, because the call lives in a callback that renderToStaticMarkup
 *     never reaches.
 *
 *   · the LOADED state. The scene list, its numbering and its thumbnails are a
 *     branch that only exists once scenes are in hand — exactly the kind of
 *     state a default-state render stays green through.
 *
 *   · the TWO sources answering the same. A dataset's cards must render like a
 *     bank's in every respect but the thumbnail URL, which is the one thing the
 *     two surfaces genuinely address differently (row id vs file name).
 */
import assert from 'node:assert/strict'
import test from 'node:test'
import { readFileSync } from 'node:fs'

import { renderToStaticMarkup, createElement } from './support/mountJsx.mjs'

const { default: ScenePromptsPanel } = await import(
  '../src/components/dataset/studio/ScenePromptsPanel.jsx')
const { SCENE_SOURCES } = await import(
  '../src/components/dataset/studio/scenePrompts.js')
const { ToastProvider } = await import('../src/components/common/Toast.jsx')
// The panel carries a HelpBadge, which navigates — it only ever renders inside
// the app's router, so the test gives it one rather than dropping the badge.
const { MemoryRouter } = await import('react-router')

const SRC = readFileSync(
  new URL('../src/components/dataset/studio/ScenePromptsPanel.jsx', import.meta.url),
  'utf8')

const render = (value) => renderToStaticMarkup(
  createElement(MemoryRouter, null,
    createElement(ToastProvider, null,
      createElement(ScenePromptsPanel, { value, onChange: () => {} }))))

const EMPTY = { source: null, scenes: [], picked: [] }
const SCENES = [
  { label: 'Scene 1 — p000.jpg', framing: 'body', prompt: 'street at dawn', image_id: 41, filename: 'p000.jpg' },
  { label: 'Scene 2 — p001.jpg', framing: 'face', prompt: 'close on her face', image_id: 42, filename: 'p001.jpg' },
  { label: 'Scene 3 — p002.jpg', framing: 'body', prompt: 'rooftop chase', image_id: 43, filename: 'p002.jpg' },
]
const LOADED = {
  source: { kind: 'bank', id: 7, name: 'Chapter 1' },
  scenes: SCENES,
  picked: [2, 0],
}
const LOADED_DATASET = { ...LOADED, source: { kind: 'dataset', id: 4, name: 'Lola' } }

test('each source names the LIST route it asks, and the scenes route follows it', () => {
  assert.deepEqual(SCENE_SOURCES.map((s) => [s.kind, s.listUrl, s.listKey]), [
    ['bank', '/api/banks', 'banks'],
    ['dataset', '/api/dataset/list', 'datasets'],
  ], 'the list of banks is /api/banks and the list of datasets /api/dataset/list — '
    + 'a wrong one 404s into an empty dropdown with no error on screen')
  assert.match(SRC, /`\/api\/dataset\/\$\{sourceId\}\/scenes`/)
  assert.match(SRC, /`\/api\/bank\/\$\{sourceId\}\/scenes`/)
})

test('nothing loaded: the panel says what it is for, offers both sources, no list', () => {
  const html = render(EMPTY)
  assert.ok(html.includes('🎬 Scenes from a bank or dataset'))
  assert.ok(html.includes('run a bank’s or a dataset’s captions in order'))
  assert.ok(html.includes('🗃 Bank') && html.includes('📁 Dataset'),
    'both sources must be offered before anything is loaded — that IS the choice')
  assert.ok(!html.includes('Select all'), 'nothing to select yet')
})

test('loaded: every scene is numbered in SOURCE order, whatever the tick order', () => {
  const html = render(LOADED)
  // The prompts appear in scene order — the whole point of the lane — and the
  // numbers are the reading order, not the order the user ticked them in.
  const order = ['street at dawn', 'close on her face', 'rooftop chase']
    .map((p) => html.indexOf(p))
  assert.ok(order.every((i) => i >= 0), 'a scene prompt is missing from the panel')
  assert.deepEqual([...order].sort((a, b) => a - b), order,
    'the scenes rendered out of source order')
  assert.ok(html.includes('>1.</span>') && html.includes('>3.</span>'))
})

test('loaded: the summary counts the picks, and the ticked scenes are pressed', () => {
  const html = render(LOADED)
  assert.ok(html.includes('2 of 3 scene(s) picked from “Chapter 1”'),
    'the count of what a launch will queue must be on screen before the click')
  // Two ticked cards — plus the source button that is currently selected.
  assert.equal((html.match(/aria-pressed="true"/g) || []).length, 3)
  assert.equal((html.match(/aria-pressed="false"/g) || []).length, 2)
})

test('loaded: each card shows the image it came from, addressed per source', () => {
  const bank = render(LOADED)
  assert.ok(bank.includes('/api/bank/7/thumb/41'), 'a bank thumb is addressed by row id')
  assert.ok(bank.includes('loading="lazy"'), 'a chapter is ~80 cards — never eager')
  const dataset = render(LOADED_DATASET)
  assert.ok(dataset.includes('/api/dataset/4/thumb/p000.jpg'),
    'a dataset thumb is addressed by file name')
  assert.ok(!dataset.includes('/api/bank/'), 'a dataset card must not ask the bank route')
})

test('a dataset reads exactly like a bank apart from where its pictures live', () => {
  const strip = (h) => h.replace(/src="[^"]*"/g, 'src=""')
    .replace(/Chapter 1|Lola/g, 'SOURCE')
  assert.equal(strip(render(LOADED)), strip(render(LOADED_DATASET)),
    'the two sources must produce the same cards — one contract, two surfaces')
})

test('a scene with no picture renders anyway, without an <img> to a bad URL', () => {
  const bare = [{ label: 'Scene 1', prompt: 'a scene from before thumbnails' }]
  for (const value of [{ ...LOADED, scenes: bare }, { ...LOADED_DATASET, scenes: bare }]) {
    const html = render(value)
    assert.ok(html.includes('a scene from before thumbnails'))
    assert.ok(!html.includes('/thumb/undefined') && !html.includes('/thumb/null'))
    assert.ok(!html.includes('<img'), 'no picture to point at — draw no <img> at all')
  }
})
