/**
 * 📚 SAVED PROMPTS — what only a RENDER can answer.
 *
 * The sibling contract test reads these two files as text; it can tell you a
 * class is written, never what reaches the screen. The bug this whole browser
 * exists to fix was of exactly that second kind — a thumbnail that was present,
 * correct, routed through the right helper, and 32 pixels wide — so the
 * assertions that matter here are on the RENDERED markup: how many cards the
 * strip actually draws, which URL lands in `src`, and what a prompt with no
 * picture puts in the space instead.
 *
 * ⚠️ Effects never run and nothing can be typed, so the SEARCH is not exercised
 * here: it is pure and lives in savedPrompts.test.js, where it is executed on
 * the shape a real history has.
 */
import assert from 'node:assert/strict'
import test from 'node:test'

import { createElement, render } from './support/mountJsx.mjs'

const { MemoryRouter } = await import('react-router')
const { default: RecentPrompts } = await import(
  '../src/components/dataset/studio/RecentPrompts.jsx')
const { SavedPromptsPanel } = await import(
  '../src/components/dataset/studio/SavedPromptsModal.jsx')

// HelpBadge reads the router; the providers come from the harness's own render.
const draw = (Component, props) => render(
  () => createElement(MemoryRouter, null, createElement(Component, props)))

const srcsOf = (html) => [...html.matchAll(/<img[^>]*\ssrc="([^"]+)"/g)].map((m) => m[1])
const history = (n) => Array.from({ length: n }, (_, i) => ({
  prompt: `prompt number ${i}`, thumbnail: `shot-${i}.png`,
  thumb_dataset_id: 7, thumb_rating: i === 0 ? 1 : 0, count: 4,
}))

test('the strip draws a handful, and its button names the whole history', () => {
  // Le partage bande/fenêtre est le cœur du changement : si la bande dessinait
  // encore les 167, le bouton n'aurait rien à ouvrir et le mur serait intact.
  const html = draw(RecentPrompts, {
    items: history(20), datasetId: 7, selectedPrompt: null,
    onPick: () => {}, onDelete: () => {},
  })
  const drawn = [...html.matchAll(/title="prompt number (\d+)"/g)].map((m) => Number(m[1]))
  assert.ok(drawn.length > 0 && drawn.length <= 12,
    `the strip drew ${drawn.length} cards — it is meant to be a handful`)
  assert.deepEqual(drawn, [...drawn].sort((a, b) => a - b),
    'the strip keeps the API order: most recent first')
  assert.ok(html.includes('Browse all'), 'no way into the browser')
  assert.match(html, />20</, 'the button must name the real total, not what it drew')
})

test('a strip tile asks for a thumbnail big enough to see', () => {
  const html = draw(RecentPrompts, {
    items: history(1), datasetId: 7, selectedPrompt: null, onPick: () => {},
  })
  const srcs = srcsOf(html).filter((s) => s.includes('/api/dataset/'))
  assert.equal(srcs.length, 1)
  // Le barreau demandé, pas seulement l'appel à l'aide : c'est `s=` qui décide
  // du nombre de pixels reçus, et 128 était le réglage de la vignette de 32 px.
  const side = Number(/\bs=(\d+)/.exec(srcs[0])?.[1])
  assert.ok(side >= 192, `a strip tile still asks for s=${side}`)
  assert.match(srcs[0], /\/thumb\//, 'a tile must never decode the original')
})

test('the browser renders every entry, at the Civitai browser’s picture size', () => {
  const html = draw(SavedPromptsPanel, {
    open: true, onClose: () => {}, items: history(20), datasetId: 7,
    selectedPrompt: null, onPick: () => {}, onDelete: () => {},
  })
  const drawn = [...html.matchAll(/prompt number (\d+)/g)].map((m) => Number(m[1]))
  assert.equal(new Set(drawn).size, 20, 'the browser is where the WHOLE history lives')
  const srcs = srcsOf(html).filter((s) => s.includes('/api/dataset/'))
  assert.equal(srcs.length, 20)
  for (const src of srcs) {
    const side = Number(/\bs=(\d+)/.exec(src)?.[1])
    assert.ok(side >= 256, `the browser asks for s=${side} — its whole job is the picture`)
  }
  assert.ok(html.includes('w-28 sm:w-36 h-40 sm:h-48'),
    'same job as the 🌐 Civitai browser, same drawn size')
  assert.ok(html.includes('type="search"'), 'a 20-entry list without a filter is the old wall')
})

test('a prompt that never rendered says so, and spends the space on its text', () => {
  // 39 des 167 entrées mesurées n'ont aucune image : un « ? » de la taille d'une
  // vignette ne les distingue pas les unes des autres, le texte si.
  const html = draw(SavedPromptsPanel, {
    open: true, onClose: () => {}, items: [{ prompt: 'a prompt never launched', count: 0 }],
    datasetId: 7, selectedPrompt: null, onPick: () => {},
  })
  assert.equal(srcsOf(html).filter((s) => s.includes('/api/dataset/')).length, 0,
    'no thumbnail means no image request — certainly not /img/undefined')
  assert.ok(html.includes('No image yet'))
  assert.ok(html.includes('never run'), 'zero test images is a fact worth saying')
  assert.ok(html.includes('a prompt never launched'))
})

test('the batch tick appears only for a host that asked for it', () => {
  // « Generate from the board » monte la liste SANS le lot ; la case ne doit pas
  // y apparaître, ni dans la bande ni dans la fenêtre.
  const props = {
    open: true, onClose: () => {}, items: history(3), datasetId: 7,
    selectedPrompt: null, onPick: () => {}, onDelete: () => {},
  }
  assert.ok(!draw(SavedPromptsPanel, props).includes('role="checkbox"'))
  assert.ok(draw(SavedPromptsPanel, { ...props, batch: [], onToggleBatch: () => {} })
    .includes('role="checkbox"'))
  const strip = { items: history(3), datasetId: 7, selectedPrompt: null, onPick: () => {} }
  assert.ok(!draw(RecentPrompts, strip).includes('role="checkbox"'))
  assert.ok(draw(RecentPrompts, { ...strip, batch: [], onToggleBatch: () => {} })
    .includes('role="checkbox"'))
})

test('a history the API sent as bare strings still renders', () => {
  // Rétro-compat : avant un restart Flask la route répond des strings. Une
  // fenêtre qui explose sur cette forme casserait l'écran juste après un update.
  for (const Component of [RecentPrompts, SavedPromptsPanel]) {
    const html = draw(Component, {
      open: true, onClose: () => {}, items: ['plain string prompt'],
      datasetId: 7, selectedPrompt: null, onPick: () => {},
    })
    assert.ok(html.includes('plain string prompt'))
  }
})
