// The editor's pure parts: where the picture sits, the brush, what can run, and the descriptor's contract.
import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fit, held, onPicture, pinch, toPicture, toStage, zoomAt } from '../frontend/lib/geometry.js'
import { dirtyBox, fillOutline, togglePoint } from '../frontend/lib/strokes.js'
import { KINDS, KEEPS, blocked, missingModels, notReadyToCut, readOptions, runLabel, saveOptions } from '../frontend/lib/options.js'
import { modelRows, setupRows } from '../frontend/lib/readiness.js'
import descriptor from '../frontend/index.js'
import { parityGaps } from '../../../frontend/src/plugins/parity.js'

const manifest = JSON.parse(readFileSync(new URL('../plugin.json', import.meta.url), 'utf8'))
const picture = { width: 2000, height: 1000 }
const stage = { width: 500, height: 400 }

test('the whole picture is centred, and a point keeps its place on screen and back', () => {
  const view = fit(picture, stage)
  assert.ok(picture.width * view.scale <= stage.width && picture.height * view.scale <= stage.height)
  const back = toPicture(view, toStage(view, { x: 1234, y: 567 }))
  assert.ok(Math.abs(back.x - 1234) < 1e-6 && Math.abs(back.y - 567) < 1e-6)
  assert.equal(onPicture({ x: 0, y: 999.9 }, picture), true)
  assert.equal(onPicture({ x: 2000, y: 5 }, picture), false)
})

test('zooming keeps the point under the pointer, within limits', () => {
  const view = fit(picture, stage)
  const at = { x: 120, y: 200 }
  const before = toPicture(view, at)
  const near = zoomAt(view, 2, at, picture, stage)
  const after = toPicture(near, at)
  assert.ok(Math.abs(before.x - after.x) < 1 && Math.abs(before.y - after.y) < 1)
  let far = view
  for (let i = 0; i < 60; i += 1) far = zoomAt(far, 1.5, at, picture, stage)
  assert.ok(far.scale <= 4 + 1e-9)
  for (let i = 0; i < 90; i += 1) far = zoomAt(far, 0.5, at, picture, stage)
  assert.ok(far.scale >= view.scale * 0.5 - 1e-9)
})

test('a pan never loses the picture and two fingers move and zoom it', () => {
  const view = held({ scale: 0.2, x: 99999, y: -99999 }, picture, stage)
  assert.ok(view.x <= stage.width && view.y + picture.height * view.scale >= 0)
  const start = fit(picture, stage)
  const apart = pinch(start, [{ x: 200, y: 200 }, { x: 300, y: 200 }], [{ x: 150, y: 220 }, { x: 350, y: 220 }], picture, stage)
  assert.ok(Math.abs(apart.scale / start.scale - 2) < 1e-6)
})

test('a press on a dot takes it away, and there is an end to dots', () => {
  const one = togglePoint([], { x: 10, y: 10, keep: true }, 5)
  assert.equal(togglePoint(one, { x: 12, y: 12, keep: true }, 5).length, 0)
  const full = Array.from({ length: 32 }, (_, i) => ({ x: i * 100, y: 0, keep: true }))
  assert.equal(togglePoint(full, { x: 50, y: 500, keep: true }, 5).length, 32)
})

test('only what a stroke touched is redrawn, and a slip is not a selection', () => {
  const box = dirtyBox({ size: 20, points: [{ x: 100, y: 100 }, { x: 140, y: 100 }] }, 0.5, 1, { width: 200, height: 60 })
  assert.deepEqual([box.x, box.y, box.x + box.width <= 200, box.y + box.height <= 60], [63, 43, true, true])
  assert.equal(fillOutline(null, [{ x: 1, y: 1 }, { x: 2, y: 2 }], 1), false)
})

const status = (extra = {}) => ({
  configured: true, reachable: true,
  models: Object.fromEntries(['birefnet', 'sam', 'lucida', 'rmbg', 'rmbg_text_encoder', 'rmbg_vae'].map(slot => [slot, { state: 'ready' }])),
  kinds: { photo: { models: ['birefnet'], missing_nodes: [] }, fine: { models: ['rmbg', 'rmbg_text_encoder', 'rmbg_vae'], missing_nodes: [] },
    graphic: { models: ['lucida'], missing_nodes: [] } },
  find: { models: ['sam'], missing_nodes: [] }, ...extra,
})

test('what stops a cut-out is said, and only what it needs is installed', () => {
  assert.equal(blocked(status(), 'subject', 'photo'), '')
  assert.match(blocked(status({ reachable: false }), 'subject', 'photo'), /Start it/)
  assert.match(blocked(status({ find: { models: ['sam'], missing_nodes: ['SAM3_Detect'] } }), 'points', 'photo'), /SAM3_Detect/)
  assert.equal(blocked(status({ find: { models: ['sam'], missing_nodes: ['SAM3_Detect'] } }), 'subject', 'photo'), '')
  const bare = status()
  for (const model of Object.values(bare.models)) model.state = 'missing'
  assert.deepEqual(missingModels(bare, 'subject', 'photo'), ['birefnet'])
  assert.deepEqual(missingModels(bare, 'describe', 'fine'), ['rmbg', 'rmbg_text_encoder', 'rmbg_vae', 'sam'])
  assert.match(notReadyToCut({ keep: 'points', phrase: '', points: [{ keep: false }], hasSelection: false }), /Click/)
  assert.equal(notReadyToCut({ keep: 'subject', phrase: '', points: [], hasSelection: false }), '')
})

test('a step that follows another does not claim to wait for the GPU', () => {
  assert.equal(runLabel({ status: 'queued', stage: 'find', keep: 'describe' }), 'Waiting for the GPU…')
  assert.equal(runLabel({ status: 'queued', stage: 'cut', keep: 'describe' }), 'Cutting the edge…')
  assert.equal(runLabel({ status: 'running', stage: 'closer', keep: 'subject' }), 'Looking closer at the edge…')
})

test('the choices survive a visit and a damaged memory', () => {
  const memory = new Map()
  const storage = { getItem: key => memory.get(key) ?? null, setItem: (key, value) => memory.set(key, value) }
  saveOptions(storage, { kind: 'graphic', background: 'white', clean: false })
  assert.deepEqual(readOptions(storage), { kind: 'graphic', background: 'white', clean: false })
  assert.deepEqual(readOptions({ getItem: () => '{"kind":"gone","clean":1}' }), { kind: 'photo', background: 'transparent', clean: true })
  assert.deepEqual(readOptions({ getItem: () => { throw new Error('blocked') } }), { kind: 'photo', background: 'transparent', clean: true })
  saveOptions({ setItem: () => { throw new Error('full') } }, {})
})

test('Setup never preselects the heavy optional model', () => {
  const rows = modelRows({ comfyui: { dir_valid: true }, remove_background: { models: { birefnet: true } } })
  assert.deepEqual(rows.map(row => [row.action, row.present]), [['remove_bg_birefnet', true], ['remove_bg_sam', false], ['remove_bg_lucida', false]])
  assert.equal(modelRows({})[0].available, false)
  assert.equal(setupRows({ remove_background: { ok: true } })[0].ok, true)
})

test('the descriptor owns what the manifest declares and covers every viewer', () => {
  assert.equal(descriptor.id, manifest.id)
  assert.deepEqual(parityGaps(descriptor), [])
  assert.deepEqual(descriptor.help.map(topic => topic.id).sort(), [...manifest.owns.help_topics].sort())
  assert.deepEqual(descriptor.whatsNew.map(item => item.id), manifest.owns.whats_new_ids)
  assert.deepEqual(descriptor.guide.sections.map(item => `${item.chapter}#${item.anchor}`), manifest.guide_ownership.sections)
  const actions = modelRows({}).map(row => row.action)
  for (const action of actions) assert.ok(manifest.owns.install_actions.includes(action))
  assert.deepEqual(KINDS.map(item => item.id), ['photo', 'fine', 'graphic'])
  assert.deepEqual(KEEPS.map(item => item.id), ['subject', 'selection', 'describe', 'points'])
})
