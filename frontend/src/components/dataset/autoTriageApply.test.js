import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import {
  autoTriageFailureMessage,
  autoTriageOwnershipForResult,
  createAutoTriageRunGate,
  runAutoTriageBatches,
  updateAutoTriageRuns,
} from './autoTriageApply.js'

function deferred() {
  let resolve
  const promise = new Promise((done) => { resolve = done })
  return { promise, resolve }
}

test('auto-triage succeeds only after every required batch returns a number', async () => {
  const calls = []
  const result = await runAutoTriageBatches({
    keepIds: [1], rejectIds: [2],
    onBatch: async (ids, action) => { calls.push([ids, action]); return ids.length },
  })
  assert.deepEqual(result, { ok: true, completed: ['keep', 'reject'] })
  assert.deepEqual(calls, [[[1], 'keep'], [[2], 'reject']])
})

test('a failed first batch stops before the second and cannot earn applied state', async () => {
  const calls = []
  const result = await runAutoTriageBatches({
    keepIds: [1], rejectIds: [2],
    onBatch: async (_ids, action) => { calls.push(action); return null },
  })
  assert.equal(result.ok, false)
  assert.deepEqual(result.completed, [])
  assert.deepEqual(calls, ['keep'])
  assert.match(autoTriageFailureMessage(result), /No successful batch/)
})

test('a failed second batch reports partial persistence but never overall success', async () => {
  const result = await runAutoTriageBatches({
    keepIds: [1], rejectIds: [2],
    onBatch: async (_ids, action) => (action === 'keep' ? 1 : null),
  })
  assert.deepEqual(result, { ok: false, failed: 'reject', completed: ['keep'] })
  assert.match(autoTriageFailureMessage(result), /part-way/)
})

test('a completed first batch keeps exactly its durable ownership after the second fails', () => {
  const ownership = autoTriageOwnershipForResult({
    result: { ok: false, failed: 'reject', completed: ['keep'] },
    previousOwnership: { 3: 'keep', 4: 'reject', 5: 'reject' },
    keepTargets: [
      { id: 1, status: 'pending' },
      { id: 3, status: 'keep' },
    ],
    rejectTargets: [
      { id: 2, status: 'pending' },
      // A no-op target was already owned and remains safe to replay.
      { id: 4, status: 'reject' },
      // Merely having the target status is insufficient without matching
      // previous ownership (manual choices must never become auto-owned).
      { id: 5, status: 'keep' },
      { id: 6, status: 'reject' },
    ],
  })
  assert.deepEqual(ownership, { 1: 'keep', 3: 'keep', 4: 'reject' })
  assert.equal(ownership[2], undefined, 'the failed reject write never earns ownership')
  assert.equal(ownership[5], undefined, 'a changed previous mapping is not preserved')
  assert.equal(ownership[6], undefined, 'a manual status is not invented as ownership')
})

test('a failed Re-apply flip retains old ownership only while it matches the live status', () => {
  const ownership = autoTriageOwnershipForResult({
    result: { ok: false, failed: 'keep', completed: [] },
    previousOwnership: { 7: 'reject', 8: 'reject' },
    keepTargets: [
      { id: 7, status: 'reject' },
      { id: 8, status: 'pending' },
    ],
  })
  assert.deepEqual(ownership, { 7: 'reject' })
})

test('a thrown request is a failure, not an applied run', async () => {
  const result = await runAutoTriageBatches({
    keepIds: [1],
    onBatch: async () => { throw new Error('offline') },
  })
  assert.equal(result.ok, false)
  assert.deepEqual(result.completed, [])
  assert.equal(result.error.message, 'offline')
})

test('navigation after keep prevents the stale run from launching its reject POST', async () => {
  let current = true
  const calls = []
  const result = await runAutoTriageBatches({
    keepIds: [1], rejectIds: [2],
    shouldContinue: () => current,
    onBatch: async (_ids, action) => {
      calls.push(action)
      current = false
      return 1
    },
  })
  assert.deepEqual(calls, ['keep'])
  assert.deepEqual(result, { ok: false, stale: true, completed: ['keep'] })
})

test('switching datasets invalidates every result and finally callback from the old dataset', async () => {
  const gate = createAutoTriageRunGate('dataset-a')
  assert.equal(gate.kind, 'createAutoTriageRunGate')
  const oldRequest = deferred()
  const events = []
  const oldToken = gate.begin('dataset-a')
  const oldRun = oldRequest.promise.then(() => {
    gate.commit(oldToken, () => events.push('old-result'))
    gate.finish(oldToken, () => events.push('old-idle'))
  })

  gate.syncDataset('dataset-b')
  assert.equal(gate.hasActive('dataset-a'), true,
    'navigation keeps A locked until its real request settles')
  oldRequest.resolve()
  await oldRun

  assert.deepEqual(events, [])
  assert.equal(gate.isCurrent(oldToken), false)
  assert.equal(gate.hasActive('dataset-a'), false)
  assert.equal(gate.hasActive('dataset-b'), false)
})

test('A to B to A cannot start a second write while the old A request is settling', () => {
  const gate = createAutoTriageRunGate('dataset-a')
  const oldToken = gate.begin('dataset-a')
  gate.syncDataset('dataset-b')
  gate.syncDataset('dataset-a')

  assert.equal(gate.hasActive('dataset-a'), true)
  assert.equal(gate.begin('dataset-a'), null)
  assert.equal(gate.isCurrent(oldToken), false, 'returning to A does not resurrect its old UI token')
  assert.equal(gate.finish(oldToken, () => assert.fail('stale finally touched current A')), true)
  assert.equal(gate.hasActive('dataset-a'), false)
  assert.ok(gate.begin('dataset-a'), 'A unlocks only after the old request settled')
})

test('overlapping A and B runs settle independently and A cannot release B busy', async () => {
  const gate = createAutoTriageRunGate('dataset-a')
  let busyRuns = new Map()
  const firstRequest = deferred()
  const events = []

  const firstToken = gate.begin('dataset-a')
  busyRuns = updateAutoTriageRuns(busyRuns, 'dataset-a', firstToken, true)
  const firstRun = firstRequest.promise.then(() => {
    gate.commit(firstToken, () => events.push('first-result'))
    if (gate.finish(firstToken, () => events.push('first-idle'))) {
      busyRuns = updateAutoTriageRuns(busyRuns, 'dataset-a', firstToken, false)
    }
  })

  gate.syncDataset('dataset-b')
  const secondToken = gate.begin('dataset-b')
  busyRuns = updateAutoTriageRuns(busyRuns, 'dataset-b', secondToken, true)

  firstRequest.resolve()
  await firstRun
  assert.deepEqual(events, [])
  assert.equal(busyRuns.has('dataset-a'), false)
  assert.equal(busyRuns.has('dataset-b'), true, 'settling A cannot unlock B')
  assert.equal(gate.isCurrent(secondToken), true, 'the newer run still owns busy')

  assert.equal(gate.finish(secondToken, () => events.push('second-idle')), true)
  busyRuns = updateAutoTriageRuns(busyRuns, 'dataset-b', secondToken, false)
  assert.deepEqual(events, ['second-idle'])
  assert.equal(busyRuns.has('dataset-b'), false)
})

test('a stale token cannot remove a newer busy token for the same dataset', () => {
  const stale = { runId: 1 }
  const current = { runId: 2 }
  let runs = updateAutoTriageRuns(new Map(), 'dataset-a', stale, true)
  runs = updateAutoTriageRuns(runs, 'dataset-a', current, true)
  const unchanged = updateAutoTriageRuns(runs, 'dataset-a', stale, false)
  assert.equal(unchanged, runs)
  assert.equal(unchanged.get('dataset-a'), current)
})

test('DatasetGrid preserves completed ownership on failure but records applied state only on success', () => {
  const source = readFileSync(new URL('./DatasetGrid.jsx', import.meta.url), 'utf8')
  const apply = source.slice(source.indexOf('const apply = async'),
    source.indexOf('return (', source.indexOf('const apply = async')))
  const refusal = apply.indexOf('if (!result.ok)')
  const earlyReturn = apply.indexOf('return;', refusal)
  const owns = apply.indexOf('setOwned(next)')
  const records = apply.indexOf('setLastRun(')
  assert.ok(refusal >= 0 && earlyReturn > refusal)
  assert.ok(owns > refusal && owns < earlyReturn,
    'durable partial ownership is published before the failure return')
  assert.ok(records > earlyReturn, 'applied state is committed only beyond the failure return')
  assert.match(apply, /previousOwnership: owned/)
})

test('DatasetGrid invalidates by dataset and lets only the active run publish or clear busy', () => {
  const source = readFileSync(new URL('./DatasetGrid.jsx', import.meta.url), 'utf8')
  const apply = source.slice(source.indexOf('const apply = async'),
    source.indexOf('return (', source.indexOf('const apply = async')))
  assert.match(source, /autoTriageRunGateRef\.current\.syncDataset\(datasetId\)/)
  assert.match(apply, /const token = runGate\.begin\(datasetId\)/)
  assert.match(apply, /if \(!runGate\.isCurrent\(token\)\) return;/)
  assert.match(apply, /shouldContinue: \(\) => runGate\.isCurrent\(token\)/)
  assert.match(apply, /runGate\.commit\(token, \(\) => \{/)
  assert.match(apply, /onApplyingChange\(token\.datasetId, token, false\)/)
  assert.match(source, /const bulkBusy = busy \|\| launchingImprove \|\| !!bulkAction \|\| autoTriageApplying/)
  assert.match(source, /role="status" aria-live="polite" aria-atomic="true"/)
})

test('the shared Auto-triage busy disables bulk and every mutating tile decision', () => {
  const grid = readFileSync(new URL('./DatasetGrid.jsx', import.meta.url), 'utf8')
  const tile = readFileSync(new URL('./DatasetGridItem.jsx', import.meta.url), 'utf8')
  assert.match(grid, /busy=\{bulkBusy\}/)
  assert.match(grid, /disabled=\{bulkBusy\}/)
  assert.match(tile, /onStatus\(img\.id, img\.status === 'keep'[\s\S]*?disabled=\{curationRefused\}/)
  assert.match(tile, /onStatus\(img\.id, img\.status === 'reject'[\s\S]*?disabled=\{curationRefused\}/)
  assert.match(tile, /Permanently delete this image[\s\S]*?disabled=\{curationRefused\}/)
})
