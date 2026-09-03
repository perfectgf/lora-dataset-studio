/* The bridge between the lineage TREE and the STEP model — pinned so a pill on
   the graph and a row in the list can never offer different verbs for the same
   save. */
import test from 'node:test'
import assert from 'node:assert/strict'
import {
  EMPTY_GRAPH_NOTE, PREVIEWS_NOTE, graphSummary, nodeGroup, pillActionModel, pillKey,
  pillPreview, pillStep, samplesOfStep, videoDeployHint,
} from './videoLineage.js'
import { CONTINUE_LOCAL_REASON, HAND_PLACED_REASON, stepActionModel } from './videoCheckpoints.js'
import { runNumber, runIdentityLabel } from '../../utils/runIdentity.js'

const file = (filename, extra = {}) => ({ filename, size: 1, deployed_as: null, undeployable: false, ...extra })
const CLOUD = {
  record_id: 12, run_id: 12, source: 'cloud', status: 'done', active: false, parent_record_id: 7,
  checkpoints: [
    { step: 100, final: false, testable: true, preview_url: '/api/video-dataset/9/train/sample/poster?run_id=12&filename=a__000000100_0.mp4',
      preview_status: 'ready', preview_count: 2,
      files: [file('a_000000100.safetensors', { deployed_as: 'h3/a_000000100.safetensors', undeployable: false })] },
    { step: 2000, final: true, testable: false, preview_url: null, preview_status: null, preview_count: 0,
      files: [file('a.safetensors')] },
  ],
}
const LOCAL = {
  record_id: -9, run_id: null, source: 'local', status: null, active: false, run_name: 'video_x_ds9',
  checkpoints: [{ step: 50, final: false, testable: false, files: [file('v_000000050_high_noise.safetensors'), file('v_000000050_low_noise.safetensors')] }],
}

test('a node becomes the group the list reasons about, a pill the step', () => {
  assert.deepEqual(nodeGroup(CLOUD), { key: 'cloud-12', lane: 'cloud', run_id: 12, active: false, status: 'done',
    run_name: null, parent_run_id: 7, steps: CLOUD.checkpoints })
  assert.deepEqual(nodeGroup(LOCAL), { key: 'local', lane: 'local', run_id: null, active: false, status: null,
    run_name: 'video_x_ds9', parent_run_id: null, steps: LOCAL.checkpoints })
  assert.deepEqual(pillStep(CLOUD.checkpoints[0]), { step: 100, final: false, deployed: true, files: CLOUD.checkpoints[0].files })
  assert.equal(pillKey(CLOUD, CLOUD.checkpoints[1]), 'cloud-12:final')
  assert.equal(pillKey(LOCAL, LOCAL.checkpoints[0]), 'local:50')
})

test('the graph popover decides EXACTLY what the list row decides for the same save', () => {
  const fromGraph = pillActionModel(9, CLOUD, CLOUD.checkpoints[0], { canDeploy: true })
  const fromList = stepActionModel(9, nodeGroup(CLOUD), pillStep(CLOUD.checkpoints[0]), { canDeploy: true })
  assert.deepEqual(fromGraph, fromList)
  assert.equal(fromGraph.deployed, true)
  assert.deepEqual(fromGraph.undeploy, { reason: HAND_PLACED_REASON })
  assert.deepEqual(fromGraph.continue, { ok: true })
  assert.equal(fromGraph.files[0].url, '/api/video-dataset/9/train/cloud/checkpoint?run_id=12&filename=a_000000100.safetensors')
  const local = pillActionModel(9, LOCAL, LOCAL.checkpoints[0])
  assert.deepEqual(local.continue, { reason: CONTINUE_LOCAL_REASON })
  assert.deepEqual(local.files.map((f) => f.short), ['high noise', 'low noise'])
  assert.ok(local.files.every((f) => f.url.startsWith('/api/video-dataset/9/train/checkpoint?filename=')))
})

test('a preview is the training sample at that step, with its count; none is null', () => {
  assert.deepEqual(pillPreview(CLOUD.checkpoints[0]),
    { status: 'ready', url: CLOUD.checkpoints[0].preview_url, count: 2 })
  assert.equal(pillPreview(CLOUD.checkpoints[1]), null)
  assert.deepEqual(pillPreview({ preview_url: '/x.jpg' }), { status: null, url: '/x.jpg', count: 1 })
})

test('samples of a step come in prompt order, other steps excluded', () => {
  const all = [{ step: 100, prompt_idx: 1 }, { step: 50, prompt_idx: 0 }, { step: 100, prompt_idx: 0 }]
  assert.deepEqual(samplesOfStep(all, 100).map((s) => s.prompt_idx), [0, 1])
  assert.deepEqual(samplesOfStep(all, null), [])
  assert.deepEqual(samplesOfStep(undefined, 100), [])
})

test("the pill title's deploy sentence is true of THIS lane's verbs", () => {
  assert.equal(videoDeployHint({ testable: true }), ' — deployed to ComfyUI (the Video Studio lists it)')
  assert.equal(videoDeployHint({ testable: false }), ' — not deployed — 📦 Deploy from its actions to test it in the Studio')
  assert.equal(videoDeployHint({ present: false }), ' — this save is no longer on disk')
  assert.ok(!videoDeployHint({}).includes('Generate'))
})

test("the local node's borrowed negative id is never printed as a run number", () => {
  assert.equal(runNumber(LOCAL), 'local')
  assert.equal(runIdentityLabel(LOCAL), 'Run local')
  assert.equal(runNumber(CLOUD), '#12')
  assert.equal(runNumber({ record_id: 0 }), '#0')
})

test('the fold summary counts runs, saves and previews', () => {
  assert.equal(graphSummary({ nodes: [CLOUD, LOCAL] }), '2 runs · 3 saves · 2 previews')
  assert.equal(graphSummary({ nodes: [LOCAL] }), '1 run · 1 save · 0 previews')
  assert.equal(graphSummary(null), '0 runs · 0 saves · 0 previews')
  assert.ok(PREVIEWS_NOTE.includes('Studio') && EMPTY_GRAPH_NOTE.includes('No run'))
})
