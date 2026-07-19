import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

const hook = readFileSync(
  new URL('../src/hooks/useDataset.js', import.meta.url), 'utf8')
const workspace = readFileSync(
  new URL('../src/components/dataset/DatasetWorkspace.jsx', import.meta.url), 'utf8')

test('the hook exposes a targeted recaptionImages action', () => {
  assert.match(hook, /const recaptionImages = useCallback\(async \(ids, mode\)/)
  // Returned from the hook so the workspace can call ds.recaptionImages(...).
  assert.match(hook, /recaptionImages,/)
  assert.match(hook, /recaptioningIds,/)
})

test('recaptionImages posts the subset to the caption endpoint with image_ids', () => {
  const body = hook.slice(hook.indexOf('const recaptionImages'),
    hook.indexOf('const analyzeFaces'))
  assert.match(body, /\/api\/dataset\/\$\{currentId\}\/caption/)
  assert.match(body, /image_ids: fresh/)
  // It refreshes so the server-side leak flags re-scan after the caption changes.
  assert.match(body, /await refresh\(\)/)
})

test('recaptionImages keeps its busy state per-image, not the global wrap', () => {
  const body = hook.slice(hook.indexOf('const recaptionImages'),
    hook.indexOf('const analyzeFaces'))
  // A targeted row must not disable the whole workspace like a batch does.
  assert.doesNotMatch(body, /wrap\(async/)
  assert.match(body, /recaptioningRef\.current\.add/)
  assert.match(body, /setRecaptioningIds/)
})

test('the leak panel wires a per-row targeted Re-caption button', () => {
  assert.match(workspace, /ds\.recaptionImages\(\[img\.id\], effCaptionMode\)/)
  // The clicked row shows its own spinner keyed off recaptioningIds.
  assert.match(workspace, /ds\.recaptioningIds\.has\(img\.id\)/)
  assert.match(workspace, /workspace\.captions\.recaptioning/)
})

test('the leak panel wires a "Re-caption all leaking" header button', () => {
  assert.match(workspace, /ds\.recaptionImages\(leakingImages\.map\(\(i\) => i\.id\), effCaptionMode\)/)
  assert.match(workspace, /workspace\.captions\.leak\.recaptionAll/)
})

test('targeted re-caption buttons are locked during a batch caption pass', () => {
  // recaptionLocked folds in ds.captioning (the batch/other vision pass) and any
  // in-flight targeted row, so the buttons disable exactly when a pass is running.
  assert.match(workspace,
    /const recaptionLocked = ds\.busy \|\| ds\.captioning \|\| ds\.recaptioningIds\.size > 0/)
  assert.match(workspace, /disabled=\{recaptionLocked\}/)
})
