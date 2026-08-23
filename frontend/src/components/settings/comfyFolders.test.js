import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

import {
  COMFY_FOLDER_FIELDS, comfyFolderField, folderPlaceholder, folderEffective,
  folderEffectiveNote, folderWarning, detectedSuggestion, foldersQuery, hasAnyOverride,
} from './comfyFolders.js'

const read = (p) => readFileSync(new URL(p, import.meta.url), 'utf8')

test('the four overrides are covered, in a stable order', () => {
  assert.deepEqual(COMFY_FOLDER_FIELDS.map((f) => f.key),
    ['output_dir', 'input_dir', 'models_dir', 'loras_dir'])
  for (const f of COMFY_FOLDER_FIELDS) {
    assert.ok(f.id && f.label && f.help && f.derived, `${f.key} is incomplete`)
  }
})

/* The plaint behind this feature: nothing told you which folder was in use. An empty
   field must therefore SHOW the derived path — but on its own wrapping line, NOT in the
   placeholder: an input clips a placeholder with no ellipsis and no way to scroll it,
   which at 400px hid the very path the block exists to reveal. */
test('the placeholder stays a short shape hint, readable when clipped', () => {
  assert.equal(folderPlaceholder(comfyFolderField('output_dir')), 'Empty = <ComfyUI>/output')
  assert.equal(folderPlaceholder(comfyFolderField('loras_dir')), 'Empty = <ComfyUI>/models/loras')
  for (const f of COMFY_FOLDER_FIELDS) {
    assert.ok(folderPlaceholder(f).length < 40, `${f.key} placeholder is too long to read clipped`)
  }
})

test('an empty field states the derived path it falls back to', () => {
  const info = { kind: 'output', source: 'derived', resolved: 'D:\\Comfy\\output', exists: true }
  assert.equal(folderEffective(info), 'D:\\Comfy\\output')
})

test('a filled field does not restate a path already in the input', () => {
  assert.equal(folderEffective({ source: 'override', resolved: 'X:\\in', exists: true }), null)
})

/* GitHub #25 (Geekswordsman): deploys now follow extra_model_paths.yaml, so the
   panel that promises to show "the folder the app uses" has to show that one — a
   preview still naming <ComfyUI>/models/loras would rebuild the same divergence. */
test('a folder coming from extra_model_paths.yaml is shown, and says so', () => {
  const info = { kind: 'loras', source: 'extra_paths', resolved: 'E:\\shared\\loras', exists: true }
  assert.equal(folderEffective(info), 'E:\\shared\\loras')
  assert.equal(folderEffectiveNote(info), 'from extra_model_paths.yaml')
})

test('a derived folder needs no provenance note', () => {
  assert.equal(folderEffectiveNote({ source: 'derived', resolved: 'D:\\C\\models\\loras' }), null)
  assert.equal(folderEffectiveNote(undefined), null)
})

test('with nothing to derive from, no effective path is claimed', () => {
  assert.equal(folderEffective({ source: 'unset', resolved: '', exists: false }), null)
  assert.equal(folderEffective(undefined), null)
})

/* A path that isn't there must be named. Swallowing it would re-create the original
   bug somewhere else: the app looking in the wrong place with nothing on screen. */
test('a typed path that is not on disk is reported', () => {
  const warn = folderWarning({ source: 'override', resolved: 'X:\\nope', exists: false })
  assert.match(warn, /Not found on disk: X:\\nope/)
  assert.match(warn, /used as typed/)
})

test('a derived path that is not on disk points at the install directory', () => {
  const warn = folderWarning({ source: 'derived', resolved: 'D:\\Comfy\\input', exists: false })
  assert.match(warn, /Not found on disk: D:\\Comfy\\input/)
  assert.match(warn, /install directory/)
})

test('a folder that exists, or nothing resolved, says nothing', () => {
  assert.equal(folderWarning({ source: 'override', resolved: 'X:\\in', exists: true }), null)
  assert.equal(folderWarning({ source: 'derived', resolved: 'D:\\Comfy\\input', exists: true }), null)
  assert.equal(folderWarning({ source: 'unset', resolved: '', exists: false }), null)
  assert.equal(folderWarning(undefined), null)
})

/* The folder being THERE was only ever half the contract. The app hands ComfyUI its
   source images by copying them into input/, so a ComfyUI in another container can
   have a folder that exists and is still unusable from here — the case that used to
   pass every check and then fail at the first generation with a bare 500 (reported
   on Discord by nofaceman). The backend writes the sentence (it knows the cause and
   redacts the path); the field's job is to show it rather than stay green. */
test('a folder that exists but cannot be written to is reported', () => {
  const warn = folderWarning({
    source: 'override', resolved: 'X:\\in', exists: true, usable: false,
    problem: "ComfyUI's input folder is not writable from LoRA Dataset Studio: X:\\in "
      + '(PermissionError). If ComfyUI runs in another container, in WSL or on another '
      + 'machine, this folder must be a shared volume visible to LoRA Dataset Studio '
      + 'at that exact path — pointing the app at ComfyUI’s URL is not enough.',
  })
  assert.match(warn, /not writable/)
  assert.match(warn, /shared volume/)
})

test('a usable folder, or one the backend did not probe, stays silent', () => {
  assert.equal(folderWarning({ source: 'override', resolved: 'X:\\in', exists: true, usable: true, problem: '' }), null)
  // usable=null (nothing probed) must never be read as "unusable"
  assert.equal(folderWarning({ source: 'derived', resolved: 'D:\\C\\models', exists: true, usable: null, problem: '' }), null)
  // an older backend that doesn't send the field at all: no phantom warning
  assert.equal(folderWarning({ source: 'derived', resolved: 'D:\\C\\models', exists: true }), null)
})

/* Detection is offered only when ComfyUI actually reported something. */
test('a detected folder is offered when it differs from what is typed', () => {
  const f = comfyFolderField('input_dir')
  assert.equal(detectedSuggestion(f, { input_dir: 'D:\\real-in' }, ''), 'D:\\real-in')
  assert.equal(detectedSuggestion(f, {}, ''), null)
  assert.equal(detectedSuggestion(f, undefined, ''), null)
})

test('a detected folder already in the field is not offered again', () => {
  const f = comfyFolderField('input_dir')
  assert.equal(detectedSuggestion(f, { input_dir: 'D:\\real-in' }, 'D:\\real-in'), null)
  // same folder, cosmetically different — separators, trailing slash, case
  assert.equal(detectedSuggestion(f, { input_dir: 'D:\\real-in' }, 'D:/real-in/'), null)
  assert.equal(detectedSuggestion(f, { input_dir: 'D:\\Real-In' }, 'D:\\real-in'), null)
})

test('the preview query carries every field plus the install dir', () => {
  const q = new URLSearchParams(foldersQuery({ base_dir: 'D:\\Comfy', input_dir: 'X:\\in' }))
  assert.equal(q.get('base_dir'), 'D:\\Comfy')
  assert.equal(q.get('input_dir'), 'X:\\in')
  assert.equal(q.get('output_dir'), '')
  assert.equal(q.get('detect'), null)
  assert.equal(new URLSearchParams(foldersQuery({}, { detect: true })).get('detect'), '1')
})

test('the advanced block opens by itself when an override is already set', () => {
  assert.equal(hasAnyOverride({ base_dir: 'D:\\Comfy' }), false)
  assert.equal(hasAnyOverride({ output_dir: '  ' }), false)
  assert.equal(hasAnyOverride({ output_dir: 'X:\\out' }), true)
  assert.equal(hasAnyOverride(undefined), false)
})

/* The DOM ids are written out literally in the JSX (the help-registry contract finds
   Settings ids by scanning for id="…"), so the two must not drift apart. */
test('every field id is rendered literally in LocalToolsSection and has a help topic', () => {
  const jsx = read('./LocalToolsSection.jsx')
  // Topic data lives in ./topics/* since the 2026-08-24 registry split.
  const registry = read('../../help/topics/settingsFields.js')
  for (const f of COMFY_FOLDER_FIELDS) {
    assert.ok(jsx.includes(`id="${f.id}"`), `${f.id} is not rendered literally in the JSX`)
    assert.ok(registry.includes(`'comfyui.${f.key}'`), `comfyui.${f.key} has no help topic`)
    assert.ok(registry.includes(`'${f.id}'`), `${f.id} is not the focus of its help topic`)
  }
})

test('the section really reaches the preview route', () => {
  const jsx = read('./LocalToolsSection.jsx')
  assert.match(jsx, /\/api\/setup\/comfyui-folders/)
})
