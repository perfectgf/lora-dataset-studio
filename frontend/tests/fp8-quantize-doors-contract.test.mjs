/**
 * The fp8 quantize tool has TWO doors and must stay ONE implementation.
 *
 * It shipped reachable from a single place: the full-model recipe card, which
 * only exists inside a dense dataset. Someone who downloaded a 26 GB model from
 * Hugging Face — the person the tool was written for — has no dataset and never
 * saw it. Settings ▸ Storage is the second door.
 *
 * The failure this file exists to prevent is not "the card is missing": it is
 * the SECOND implementation. Copying the JSX into StorageSection would look
 * identical on the day it lands and would then drift — the refusals (already
 * quantized, LoRA, overwriting the source) and the read-back verification live
 * in the component, so a copy is a second set of them. So the source assertions
 * below pin that exactly one file talks to the endpoints, and the mount
 * assertions pin that BOTH hosts render the real thing rather than a lookalike.
 *
 * Mounting matters here for a second reason: StorageSection now imports a
 * component from components/dataset/, and HfStorageCard beside it calls
 * useToast(), which THROWS outside its provider. A source-text test sees none
 * of that; rendering the whole tab does.
 */
import assert from 'node:assert/strict'
import test from 'node:test'
import { readFileSync, readdirSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

import { createElement, renderToStaticMarkup } from './support/mountJsx.mjs'

/* ⚠️ Dynamic — the hooks that teach Node to read .jsx are installed while
   mountJsx.mjs is evaluated, and a static import would already be linked. */
const { default: Fp8QuantizeTool } =
  await import('../src/components/dataset/Fp8QuantizeTool.jsx')
const { default: StorageSection } =
  await import('../src/components/settings/StorageSection.jsx')
const { ToastProvider } = await import('../src/components/common/Toast.jsx')
const { helpTopics, getHelpTopic, searchHelpTopics } =
  await import('../src/help/helpRegistry.js')

const read = (rel) => readFileSync(new URL(rel, import.meta.url), 'utf8')

// Every .js/.jsx under src/, with its path — the endpoint census needs names.
const walk = (dirUrl) => {
  const out = []
  for (const entry of readdirSync(dirUrl, { withFileTypes: true })) {
    const child = new URL(`${entry.name}${entry.isDirectory() ? '/' : ''}`, dirUrl)
    if (entry.isDirectory()) out.push(...walk(child))
    else if (/\.jsx?$/.test(entry.name) && !/\.test\.jsx?$/.test(entry.name)) {
      out.push([fileURLToPath(child), readFileSync(fileURLToPath(child), 'utf8')])
    }
  }
  return out
}
const SRC_FILES = walk(new URL('../src/', import.meta.url))

// ---- one implementation ----------------------------------------------------

test('exactly one component talks to the quantize endpoints', () => {
  // The endpoints moved: `/api/tools/fp8-deliver` is the one that can also FETCH
  // a master that is not on this machine, and place the result in ComfyUI's own
  // folder. `/api/tools/fp8-quantize` stays server-side (its guards and its
  // read-back verification are what the new lane calls) but no screen drives it
  // directly any more — two client lanes would be two sets of refusals.
  const callers = SRC_FILES
    .filter(([, src]) => src.includes('/api/tools/fp8-deliver'))
    .map(([path]) => path.replace(/\\/g, '/').split('/src/')[1])
  assert.deepEqual(callers, ['components/dataset/Fp8QuantizeTool.jsx'],
    'a second file calling the quantize endpoints means the tool was copied, not reused')
  const legacy = SRC_FILES
    .filter(([, src]) => src.includes('/api/tools/fp8-quantize'))
    .map(([path]) => path.replace(/\\/g, '/').split('/src/')[1])
  assert.deepEqual(legacy, [])
})

test('both hosts render the shared component instead of their own controls', () => {
  // The training-side host is the extracted recipe card since slice 1 —
  // the panel renders it, the recipe file imports and mounts the tool.
  for (const rel of ['../src/components/dataset/FullTransformerRecipe.jsx',
    '../src/components/settings/StorageSection.jsx']) {
    const src = read(rel)
    assert.match(src, /import Fp8QuantizeTool from '[^']*Fp8QuantizeTool'/, `${rel}: no import`)
    assert.match(src, /<Fp8QuantizeTool\b/, `${rel}: imported but never rendered`)
  }
})

// ---- the tool renders in both chromes --------------------------------------

const CONTROLS = [
  /aria-label="Path of the model file to quantize to fp8"/,
  /Quantize to fp8<\/button>/,
]

test('the recipe-card door keeps its accent frame and its own title', () => {
  const html = renderToStaticMarkup(createElement(Fp8QuantizeTool, {}))
  assert.match(html, /bg-sky-400\/10/)
  // "an EXISTING model" was dropped from the title with the words "on this
  // machine" from the blurb: the block now also reaches a master that only
  // exists in a private Hugging Face repo, which is the whole point.
  assert.match(html, /Quantize a model to fp8/)
  assert.doesNotMatch(html, /on this machine into/)
  for (const re of CONTROLS) assert.match(html, re)
})

test('the Settings door drops only the chrome — every control survives', () => {
  const html = renderToStaticMarkup(createElement(Fp8QuantizeTool, { framed: false }))
  // The Card around it already carries the title and the one-sentence what/when,
  // so saying them again inside is the only thing framed={false} removes.
  assert.doesNotMatch(html, /bg-sky-400\/10/)
  assert.doesNotMatch(html, /not the same thing as the/)
  for (const re of CONTROLS) assert.match(html, re)
})

test('a disabled tool disables both controls in either chrome', () => {
  for (const framed of [true, false]) {
    const html = renderToStaticMarkup(createElement(Fp8QuantizeTool, { framed, disabled: true }))
    assert.equal((html.match(/disabled=""/g) || []).length, 2,
      `framed=${framed}: the path field and the button must both be disabled`)
  }
})

// ---- the tab itself renders ------------------------------------------------

const storageProps = {
  config: {}, setField: () => {}, configDefaults: {},
  saveConfigPatch: async () => {}, toast: { success: () => {}, error: () => {} },
}

test('Settings ▸ Storage renders the tool, with the help-mode focus target on it', () => {
  const html = renderToStaticMarkup(
    createElement(ToastProvider, null, createElement(StorageSection, storageProps)),
  )
  assert.match(html, /id="storage-fp8-quantize"/)
  for (const re of CONTROLS) assert.match(html, re)
  // Still the whole disk tab, not a page that lost a card to the new one.
  assert.match(html, /Trash/)
  assert.match(html, /Hugging Face storage/)
})

// ---- the door is addressable ----------------------------------------------

test('the Settings door has its own help topic, pointing at that focus id', () => {
  const topic = getHelpTopic('storage.fp8_quantize')
  assert.ok(topic, 'no help topic for the Settings ▸ Storage door')
  assert.equal(topic.app.route, '/settings/storage')
  assert.equal(topic.app.focus, 'storage-fp8-quantize')
  // Two doors, two topics, two distinct titles — a search result that reads the
  // same twice cannot tell you which screen you are being sent to.
  const other = getHelpTopic('training.fp8_quantize_local')
  assert.ok(other && other.title !== topic.title)
  assert.equal(helpTopics.filter((t) => t.app.route === '/settings/storage'
    && t.app.focus === 'storage-fp8-quantize').length, 1)
})

test('the words someone with an oversized model would type reach it', () => {
  // A second door nobody can search for is not a second door. These are the
  // terms of the problem as it is felt ("this file is too big"), not the terms
  // of the solution — which is the vocabulary the person who never opened a
  // dense dataset actually has.
  for (const query of ['quantize', 'fp8', 'shrink', 'smaller', 'comfyui', 'safetensors']) {
    assert.ok(searchHelpTopics(query).some((t) => t.id === 'storage.fp8_quantize'),
      `"${query}" does not surface the Settings ▸ Storage door`)
  }
})
