/* The decisions of the Checkpoints & LoRAs section, pinned one by one: which
   verb a STEP offers, and — when it does not — the sentence that says why.
   Every case below is a button that used to be missing, or one that would have
   done something other than its label on the video lane. */
import test from 'node:test'
import assert from 'node:assert/strict'
import {
  ACTIVE_CLOUD_REASON, ACTIVE_LOCAL_REASON, CONTINUE_LOCAL_REASON, EMPTY_NOTE,
  HAND_PLACED_REASON, NO_LORAS_ROOT_REASON,
  checkpointGroups, continueBody, deleteReport, deployReport, describeStepDelete,
  describeUndeploy, detailsRows, downloadLinks, fileShortName, fmtSize, groupSub,
  groupTitle, runDeleteConfirmation, stepActionModel, stepKey, timeAgo, undeployReport,
} from './videoCheckpoints.js'
import { deleteDestination } from '../../utils/deletionWording.js'

const file = (filename, extra = {}) => ({ filename, size: 300 * 1024 * 1024, deployed_as: null, undeployable: false, ...extra })
const PAIR_50 = [file('v_000000050_high_noise.safetensors'), file('v_000000050_low_noise.safetensors')]
const LOCAL = {
  run_name: 'video_city_ds9', folder: 'X:/out/video_city_ds9', active: false,
  steps: [
    { step: 50, final: false, deployed: false, files: PAIR_50 },
    { step: null, final: true, deployed: true,
      files: [file('v.safetensors', { deployed_as: 'h3/lds/v.safetensors', undeployable: true })] },
  ],
}
const CLOUD_DONE = {
  run_id: 12, status: 'done', active: false, gpu: 'RTX 4090', price_per_hour: 0.5,
  parent_run_id: 7, created_at: '2026-09-01T10:00:00', finished_at: '2026-09-01T12:00:00',
  steps: [{ step: 100, final: false, deployed: false, files: [file('v_000000100.safetensors')] },
    { step: 2000, final: true, deployed: false, files: [file('v.safetensors')] }],
}
const CLOUD_ACTIVE = {
  run_id: 13, status: 'training', active: true, gpu: 'A100', price_per_hour: 1.2,
  parent_run_id: null, created_at: '2026-09-02T10:00:00', finished_at: null,
  steps: [{ step: 200, final: false, deployed: false, files: [file('v_000000200.safetensors')] }],
}
const PAYLOAD = { local: LOCAL, cloud: [CLOUD_ACTIVE, CLOUD_DONE], can_deploy: true, deploy_folder: 'h3/lds', delete_mode: 'app_trash' }
const groups = checkpointGroups(PAYLOAD)
const [local, active, done] = groups

test('groups: the local run first, then the cloud runs in the server\'s order; nothing empty', () => {
  assert.deepEqual(groups.map((g) => g.key), ['local', 'cloud-13', 'cloud-12'])
  assert.equal(local.lane, 'local')
  assert.equal(local.run_id, null)
  assert.deepEqual(checkpointGroups({ local: null, cloud: [] }), [])
  assert.deepEqual(checkpointGroups({ local: { steps: [] }, cloud: [{ run_id: 1, steps: [] }] }), [])
  assert.deepEqual(checkpointGroups(null), [])
  assert.ok(EMPTY_NOTE.includes('No checkpoints yet'))
})

test('titles and sub-lines say the lane, the genealogy and the price', () => {
  assert.equal(groupTitle(local), 'On this PC — video_city_ds9')
  assert.equal(groupTitle(done), 'Cloud run #12 — continued from #7')
  assert.equal(groupTitle(active), 'Cloud run #13')
  const now = Date.parse('2026-09-01T12:30:00Z')
  assert.equal(groupSub(done, now), 'done · RTX 4090 · $0.50/h · 30m ago')
  assert.equal(groupSub({ ...local, active: true }), 'training now — saves still being written')
  assert.equal(timeAgo('2026-09-01T12:00:00', now), '30m ago')
  assert.equal(timeAgo(null), '')
})

test('a local step: download per file, deploy, no continue (with the reason), no details', () => {
  const a = stepActionModel(9, local, local.steps[0], { canDeploy: true, deployFolder: 'h3/lds' })
  assert.equal(a.key, 'local:50')
  assert.equal(a.label, 'Step 50 — 2 files (both experts)')
  assert.deepEqual(a.files.map((f) => f.short), ['high noise', 'low noise'])
  assert.ok(a.files.every((f) => f.url.startsWith('/api/video-dataset/9/train/checkpoint?filename=')))
  assert.deepEqual(a.continue, { reason: CONTINUE_LOCAL_REASON })
  assert.deepEqual(a.deploy, { ok: true, folder: 'h3/lds' })
  assert.equal(a.undeploy, null)
  assert.equal(a.deployed, false)
  assert.equal(a.del.ok, true)
  assert.equal(a.del.label, 'Delete the training saves')
  assert.equal(a.del.title, `Move every file of this step to ${deleteDestination('app_trash')} — recoverable until you empty it`)
  assert.equal(stepActionModel(9, local, local.steps[0], { deleteMode: 'permanent' }).del.title,
    `Move every file of this step to ${deleteDestination('permanent')}`)
  assert.equal(a.details, false)
})

test('the local FINAL save is labelled "Final", never "Final (step null)"', () => {
  const a = stepActionModel(9, local, local.steps[1])
  assert.equal(a.label, 'Final')
  assert.equal(a.key, 'local:final')
  assert.equal(a.deployed, true)
  assert.deepEqual(a.undeploy, { ok: true })
  assert.equal(a.deploy, null)
  assert.equal(a.del.label, 'Delete the training save')
})

test('a terminal cloud step continues, deploys, has details; its links are the cloud route', () => {
  const a = stepActionModel(9, done, done.steps[0])
  assert.deepEqual(a.continue, { ok: true })
  assert.equal(a.details, true)
  assert.equal(a.files[0].url, '/api/video-dataset/9/train/cloud/checkpoint?run_id=12&filename=v_000000100.safetensors')
  assert.equal(stepActionModel(9, done, done.steps[1]).label, 'Final (step 2000)')
})

test('an ACTIVE cloud run offers neither continue nor delete, and says why', () => {
  const a = stepActionModel(9, active, active.steps[0])
  assert.deepEqual(a.continue, { reason: ACTIVE_CLOUD_REASON })
  assert.deepEqual(a.del, { reason: ACTIVE_CLOUD_REASON })
  assert.deepEqual(a.deploy, { ok: true, folder: 'h3/lds' })   // deploying a synced save is fine
  const busyLocal = { ...local, active: true }
  assert.deepEqual(stepActionModel(9, busyLocal, local.steps[0]).del, { reason: ACTIVE_LOCAL_REASON })
})

test('no loras root: 📦 is a stated refusal, not a button that fails', () => {
  const a = stepActionModel(9, done, done.steps[0], { canDeploy: false })
  assert.deepEqual(a.deploy, { reason: NO_LORAS_ROOT_REASON })
})

test('deployed = EVERY file of the step; a hand-placed copy is deployed but never undeployable', () => {
  const half = { step: 50, final: false, deployed: false,
    files: [file('a_high_noise.safetensors', { deployed_as: 'h3/lds/a_high_noise.safetensors', undeployable: true }),
      file('a_low_noise.safetensors')] }
  const a = stepActionModel(9, local, half)
  assert.equal(a.deployed, false)
  assert.deepEqual(a.deploy, { ok: true, folder: 'h3/lds' })
  const byHand = { step: 100, final: false, deployed: true,
    files: [file('b.safetensors', { deployed_as: 'h3/b.safetensors', undeployable: false })] }
  const b = stepActionModel(9, done, byHand)
  assert.equal(b.deployed, true)
  assert.deepEqual(b.undeploy, { reason: HAND_PLACED_REASON })
  assert.equal(b.deploy, null)
})

test('the 🗑 confirmation names every file, the step, and the destination from the app-wide wording', () => {
  const text = describeStepDelete(local, local.steps[0], 'app_trash')
  assert.match(text, /^DELETE THE TRAINING SAVES — “v_000000050_high_noise\.safetensors” \+ “v_000000050_low_noise\.safetensors” \(Step 50 — 2 files \(both experts\)\)\?/)
  assert.ok(text.includes('never half'))
  assert.ok(text.includes(`They go to ${deleteDestination('app_trash')} — recoverable until you empty it.`))
  const bin = describeStepDelete(done, done.steps[0], 'trash')
  assert.ok(bin.includes('DELETE THE TRAINING SAVE —') && bin.includes(deleteDestination('trash')))
  assert.ok(bin.includes('It goes to'))
  const gone = describeStepDelete(done, done.steps[0], 'permanent')
  assert.ok(gone.includes(deleteDestination('permanent')) && !gone.includes('recoverable'))
  // A deployed step says the ComfyUI copy is NOT what goes.
  assert.ok(describeStepDelete(local, local.steps[1], 'app_trash').includes('KEPT — use ⏏ Undeploy'))
  assert.ok(!describeStepDelete(local, local.steps[0], 'app_trash').includes('Undeploy'))
})

test('the ⏏ confirmation is about the ComfyUI copy only, and keeps the save', () => {
  const text = describeUndeploy(local.steps[1], 'app_trash')
  assert.match(text, /^UNDEPLOY — REMOVE FROM COMFYUI — “v\.safetensors” \(Final\)\?/)
  assert.ok(text.includes(deleteDestination('app_trash')))
  assert.ok(text.includes('The training save is KEPT'))
  assert.equal(undeployReport(local.steps[1]), 'Removed from ComfyUI: v.safetensors. The training save is kept.')
})

test('reports say what is on disk: moved, kept and held open, or nothing', () => {
  assert.equal(deleteReport({ removed: ['a', 'b'], files_kept: [], delete_mode: 'app_trash' }),
    `Moved 2 files to ${deleteDestination('app_trash')}.`)
  assert.equal(deleteReport({ removed: ['a'], files_kept: ['b'], delete_mode: 'app_trash' }),
    `Moved 1 file to ${deleteDestination('app_trash')}. 1 file kept — held open by another program: b.`)
  assert.equal(deleteReport({ removed: [], files_kept: ['b'], delete_mode: 'app_trash' }),
    'Nothing was moved. 1 file kept — held open by another program: b.')
  assert.equal(deleteReport({}), 'Nothing was moved.')
  assert.equal(deployReport({ deployed: ['h3\\lds\\a.safetensors', 'h3/lds/b.safetensors'], folder: 'h3/lds' }),
    'Deployed → h3/lds: a.safetensors + b.safetensors. The Video Test Studio lists it now.')
})

test('the run-level 🗑 counts every file of every step and says it is for good', () => {
  const text = runDeleteConfirmation(done)
  assert.ok(text.startsWith('Delete run #12 and its 2 LoRA file(s) from disk?'))
  assert.ok(text.includes('This cannot be undone.'))
})

test('▶ posts the harvested step it was clicked on, and at least one extra step', () => {
  assert.deepEqual(continueBody(done, done.steps[0], '1500'), { run_id: 12, extra_steps: 1500, from_step: 100 })
  // The final save is reported AT the run's step count — that number is what the server seeds from.
  assert.deepEqual(continueBody(done, done.steps[1], 0), { run_id: 12, extra_steps: 1, from_step: 2000 })
  assert.deepEqual(continueBody(done, done.steps[1], 'abc'), { run_id: 12, extra_steps: 1, from_step: 2000 })
})

test('ⓘ rows are allow-listed, in reading order, and skip what the run does not carry', () => {
  const rows = detailsRows({
    run_id: 12, status: 'done', phase_detail: 'harvested', gpu: 'RTX 4090', price_per_hour: 0.5,
    created_at: null, finished_at: null, parent_run_id: 7, saves: 2, error: null,
    params: { steps: 2000, do_i2v: false, target_profile: 'wan22_14b', frames: 81,
      resume_step: 100, sample_prompts: ['a', 'b'], resume_ckpt_paths: ['/workspace/x'] },
  })
  assert.deepEqual(rows, [
    ['Status', 'done — harvested'], ['GPU', 'RTX 4090 · $0.50/h'], ['Continued from', 'run #7'],
    ['Resumed at step', '100'], ['Steps', '2000'], ['Target', 'wan22_14b'], ['Frames per clip', '81'],
    ['Image-to-video', 'no'], ['Sample prompts', '2'], ['Saves on this machine', '2'],
  ])
  assert.ok(!JSON.stringify(rows).includes('/workspace'))
  assert.deepEqual(detailsRows({}), [])
})

test('helpers: sizes, short names, keys, links', () => {
  assert.equal(fmtSize(300 * 1024 * 1024), '300 MB')
  assert.equal(fmtSize(1.5 * 1024 ** 3), '1.5 GB')
  assert.equal(fmtSize(2048), '2 KB')
  assert.equal(fmtSize(null), '')
  assert.equal(fileShortName('a_high_noise.safetensors', 2), 'high noise')
  assert.equal(fileShortName('a_high_noise.safetensors', 1), 'a_high_noise.safetensors')
  assert.equal(fileShortName('a.safetensors', 2), 'a.safetensors')
  assert.equal(stepKey(done, done.steps[1]), 'cloud-12:final')
  assert.equal(downloadLinks(9, local, local.steps[1])[0].url,
    '/api/video-dataset/9/train/checkpoint?filename=v.safetensors')
})
