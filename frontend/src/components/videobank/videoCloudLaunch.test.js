import test from 'node:test'
import assert from 'node:assert/strict'

import {
  videoOffersUrl, videoPreflightUrl, CLOUD_STATUS_URL, launchFooterLine,
  offersEmptyMessage, launchStatusLine, preflightGate,
} from './videoCloudLaunch.js'

const DS = { id: 9, target_label: 'Wan 2.2 T2V A14B', frames: 81, clips: 6 }

test('the offers URL carries the steps and floors them where the server does', () => {
  assert.equal(videoOffersUrl(9, 2000), '/api/video-dataset/9/train/cloud/offers?steps=2000')
  // The server clamps to 100; sending less would estimate a run it will not do.
  assert.equal(videoOffersUrl(9, 10), '/api/video-dataset/9/train/cloud/offers?steps=100')
  assert.equal(videoOffersUrl(9, undefined), '/api/video-dataset/9/train/cloud/offers?steps=1000')
})

test('the preflight URL names the cloud lane and stays bare for the local one', () => {
  assert.equal(videoPreflightUrl(9, 'cloud'), '/api/video-dataset/9/train/preflight?lane=cloud')
  assert.equal(videoPreflightUrl(9, 'local'), '/api/video-dataset/9/train/preflight')
  assert.equal(videoPreflightUrl(9), '/api/video-dataset/9/train/preflight')
  assert.equal(CLOUD_STATUS_URL, '/api/dataset/train/cloud/status')
})

// ---- the footer: what is about to be paid for, in the units of the bill -------

test('the footer names the steps, the target, the frames, the clips and the budget', () => {
  const line = launchFooterLine(DS, { steps: 2000 }, 1500, { monthly_budget: 50, month_spend: 3.2 })
  assert.match(line, /^2000 steps · Wan 2\.2 T2V A14B · 81 frames · 6 clips · this month: \$3\.20 of \$50\.00\./)
  assert.match(line, /rough/)
  assert.match(line, /auto-terminated/)
})

test('the footer falls back to the dial when the offers carry no step count, and hides a zero budget', () => {
  const line = launchFooterLine(DS, null, 1500, { monthly_budget: 0, month_spend: 0 })
  assert.match(line, /^1500 steps/)
  assert.doesNotMatch(line, /this month/)
  assert.match(launchFooterLine({ id: 1, clips: 1 }, null, null, null), /^— steps · 1 clip\./)
})

test('an empty market names the cap it is empty under', () => {
  assert.equal(offersEmptyMessage({ max_price_per_hour: 0.8 }),
    'No GPU available under $0.80/h right now. Try again shortly, or')
  assert.match(offersEmptyMessage(null), /under your price cap/)
})

test('the launching line says what the frozen button is waiting on', () => {
  assert.match(launchStatusLine(), /freezing the dataset/)
  assert.match(launchStatusLine(), /Runs page/)
})

// ---- the gate: blockers stop, warnings ask once, ready passes -------------------

const row = (id, status, detail) => ({ id, status, detail })

test('a blocker stops the click and names itself', () => {
  const gate = preflightGate({ checks: [row('clips', 'fail', 'no clip on disk'), row('vast', 'ok', 'key')] })
  assert.deepEqual(gate, { ok: false, blockers: ['no clip on disk'] })
})

test('warnings become ONE question that lists them all', () => {
  const gate = preflightGate({ checks: [
    row('licence', 'warn', 'no rights in the EU'),
    row('frames', 'warn', 'off the measured grid'),
    row('clips', 'ok', '6 clips'),
  ] }, { lane: 'cloud' })
  assert.equal(gate.ok, true)
  assert.match(gate.confirmText, /^Before renting a GPU/)
  assert.match(gate.confirmText, /• no rights in the EU\n• off the measured grid/)
  assert.match(gate.confirmText, /Continue\?$/)
  assert.match(preflightGate({ checks: [row('x', 'warn', 'w')] }, { lane: 'local' }).confirmText,
    /^Before training,/)
})

test('a clean report passes without a question, and an unreachable one never blocks', () => {
  assert.deepEqual(preflightGate({ checks: [row('clips', 'ok', '6')] }), { ok: true })
  // The server re-decides on launch; a preflight that fails closed would make
  // the lane unusable on the day the probe breaks.
  assert.deepEqual(preflightGate(null), { ok: true })
  assert.deepEqual(preflightGate({ error: 'boom' }), { ok: true })
})
