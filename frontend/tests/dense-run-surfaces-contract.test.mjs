/* A full-model run must read as one on EVERY surface that draws a run.
 *
 * These read the JSX as text (node:test does not parse JSX) and assert the wiring
 * that a screenshot would otherwise be the only proof of. They exist because the
 * three surfaces are three files: the ◉ Graph card, the ☰ List card, and the
 * shared chip vocabulary. A badge added to one and forgotten in the other is the
 * exact drift lineageChrome.jsx was created to prevent. */
import test from 'node:test'
import { readSource } from './support/readSource.mjs'
import assert from 'node:assert/strict'

const read = readSource

const chrome = read('src/components/dataset/lineageChrome.jsx')
const graph = read('src/components/dataset/lineageNodes.jsx')
const list = read('src/components/dataset/RunLineageTree.jsx')
const panel = read('src/components/dataset/DenseModelsPanel.jsx')

test('the "full model" badge is declared ONCE, in the shared vocabulary', () => {
  assert.ok(chrome.includes('export function ModeChip'))
  // and neither view re-implements it
  for (const [name, src] of [['graph', graph], ['list', list]]) {
    assert.ok(src.includes('<ModeChip node={node} />'), `${name} renders the shared chip`)
    assert.ok(!src.includes("training_mode === 'full_transformer'"),
      `${name} must not re-implement the test`)
  }
})

test('both run surfaces import the chip', () => {
  for (const src of [graph, list]) {
    assert.match(src, /import \{[^}]*ModeChip[^}]*\} from '\.\/lineageChrome'/)
  }
})

test('the availability chip answers for a full model BEFORE it reads checkpoint columns', () => {
  // Order matters: `checkpoint_ready` is a LoRA-lane flag and answering from it
  // first is what produced "gone" for a model living on Hugging Face.
  const hub = chrome.indexOf("node.dense_artifact === 'hub'")
  const local = chrome.indexOf("node.dense_artifact === 'local'")
  const legacy = chrome.indexOf('node.checkpoint_ready === true')
  assert.ok(hub > 0 && local > 0 && legacy > 0)
  assert.ok(hub < legacy && local < legacy)
})

test('the full-model panel never offers the LoRA adapter verbs', () => {
  assert.match(panel, /Full models/)
  // Comments stripped: the file's own header EXPLAINS why those verbs are
  // absent, and asserting on prose would fail for the right words.
  const code = panel.replace(/\/\*[\s\S]*?\*\//g, '')
  assert.ok(!code.includes('Import →'), 'a full model never gets the adapter verb')
  assert.ok(!code.includes('Undeploy'), 'a full model is never "deployed" to undeploy')
  assert.ok(!code.includes('loras/'), 'a full model never names the adapter folder')
})

test('the Studio deep-link carries dataset, family AND base', () => {
  assert.match(panel, /\/studio\?dataset=/)
  assert.match(panel, /&family=/)
  assert.match(panel, /&base=/)
})

/* `?base=` crosses four files to reach the form. A dropped prop anywhere in that
   chain is silent — the Studio simply opens on the first base in the list, which
   is what the whole deep-link exists to avoid — so the chain is pinned link by
   link. This is wiring, not behaviour: the decision itself (URL beats
   localStorage once, clear the persisted cfg/steps, wait for the base to be in
   the payload) lives in useStudioForm and is asserted on its source below. */
test('?base= reaches the form through every file in the chain', () => {
  const page = read('src/pages/StudioPage.jsx')
  const shell = read('src/components/dataset/studio/StudioShell.jsx')
  const legacy = read('src/components/dataset/studio/LegacyDatasetStudio.jsx')
  const form = read('src/hooks/useStudioForm.js')

  assert.match(page, /sp\.get\('base'\)/)
  assert.match(page, /preselectBase=\{preselectBase\}/)
  assert.match(shell, /preselectBase = null/)
  assert.match(shell, /initialBase=\{preselectBase\}/)
  assert.match(legacy, /initialBase = null/)
  assert.match(legacy, /preselectBase: initialBase/)
  assert.match(form, /preselectBase = null/)
})

test('the base preselection wins once, re-seeds the axes, and never guesses', () => {
  const form = read('src/hooks/useStudioForm.js')
  // once
  assert.match(form, /preselectedBaseRef\.current = true/)
  assert.match(form, /if \(preselectedBaseRef\.current \|\| !preselectBase\) return/)
  // re-seed: a Turbo session's persisted cfg 1 / 8 steps must not follow an
  // undistilled base into the grid
  assert.match(form, /setSelCfgs\(null\)/)
  assert.match(form, /setSelSteps\(null\)/)
  // never guess: an unknown value would be dropped by the backend whitelist and
  // fall back to the FIRST base — generating on the wrong model, silently
  assert.match(form, /z_models \|\| \[\]\)\.some\(\(m\) => m\.value === preselectBase\)/)
})

test('the comparison screen reads per-base settings, and the base is declared first', () => {
  const comp = read('src/components/dataset/studio/ComparisonStudio.jsx')
  assert.match(comp, /modelDefaults = null/)
  assert.match(comp, /const defaultCfg = baseDefaults\?\.cfg \?\? axes\?\.default_cfg/)
  // TDZ: `baseDefaults` reads `selectedBase`, so the state must be declared
  // ABOVE it. A `const` read before its declaration is a ReferenceError at
  // render — a blank screen, not an `undefined` a `??` would rescue.
  assert.ok(comp.indexOf('const [selectedBase, setSelectedBase]') <
    comp.indexOf('const baseDefaults ='),
    'selectedBase must be declared before the axis defaults that read it')
})
