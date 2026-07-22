import assert from 'node:assert/strict';
import test from 'node:test';

import { canContinueFromCheckpoint, initialResumeStep, resolveInitialLane } from './lineageContinue.js';

const cloudDone = { source: 'cloud', run_id: 42, status: 'done' };
const cloudFailed = { source: 'cloud', run_id: 43, status: 'error' };
const cloudRunning = { source: 'cloud', run_id: 44, status: 'running' };
// A local lineage node carries NO terminal status (the backend only flags a
// currently-failed local run) — the save is what says "resumable".
const localRun = { source: 'local', record_id: 7, status: null };

const save = { step: 1000, present: true, download_url: '/api/…/file?x=1' };
const goneSave = { step: 1000, present: false, download_url: null };

test('the Runs hub gate (default) stays cloud-only', () => {
  // the invariant: nothing passed → cloud runs only, exactly as before
  assert.equal(canContinueFromCheckpoint(cloudDone, save), true);
  assert.equal(canContinueFromCheckpoint(cloudFailed, save), true);
  assert.equal(canContinueFromCheckpoint(cloudFailed, goneSave), false,
    'a failed cloud run needs a downloadable pill');
  assert.equal(canContinueFromCheckpoint(cloudRunning, save), false,
    'an active run is never offered Continue');
  assert.equal(canContinueFromCheckpoint(localRun, save), false,
    'a local run gets Download/Import only on the Runs hub');
});

test('continueSource="any" opens Continue on a local run with a real save', () => {
  const any = { continueSource: 'any' };
  assert.equal(canContinueFromCheckpoint(localRun, save, any), true);
  // a save that is gone (or has no file to resume) still cannot be continued
  assert.equal(canContinueFromCheckpoint(localRun, goneSave, any), false);
  assert.equal(canContinueFromCheckpoint(localRun, { step: 500, present: true }, any), false);
});

test('continueSource="any" does not loosen the CLOUD rule', () => {
  const any = { continueSource: 'any' };
  assert.equal(canContinueFromCheckpoint(cloudRunning, save, any), false);
  assert.equal(canContinueFromCheckpoint(cloudFailed, goneSave, any), false);
  assert.equal(canContinueFromCheckpoint({ source: 'cloud', run_id: null, status: 'done' }, save, any),
    false, 'a cloud node without a run id has nothing to relaunch');
});

test('the dialog opens on the clicked step, and ignores one the run never saved', () => {
  const steps = [500, 1000, 1500];
  assert.equal(initialResumeStep(1000, steps), 1000, 'a real save is honoured');
  assert.equal(initialResumeStep(750, steps), 1500, 'an unknown step falls back to latest');
  assert.equal(initialResumeStep(null, steps), 1500, 'no request → the historical default');
  assert.equal(initialResumeStep(1000, []), 0, 'no checkpoint at all → 0 (button stays disabled)');
});

test('the lane defaults to the source run, and falls back to the one that works', () => {
  const both = { local: { available: true }, cloud: { available: true } };
  assert.equal(resolveInitialLane('local', both), 'local');
  assert.equal(resolveInitialLane('cloud', both), 'cloud');
  // a cloud run whose cloud lane is closed (no API key) opens on Local…
  assert.equal(resolveInitialLane('cloud',
    { local: { available: true }, cloud: { available: false, reason: 'no key' } }), 'local');
  // …and a local run opens on Cloud when ai-toolkit isn't set up here
  assert.equal(resolveInitialLane('local',
    { local: { available: false, reason: 'no ai-toolkit' }, cloud: { available: true } }), 'cloud');
  // neither usable → keep the source lane (the dialog shows its reason)
  assert.equal(resolveInitialLane('local',
    { local: { available: false }, cloud: { available: false } }), 'local');
  // no picker offered (the Runs hub) → simply the mount's own lane
  assert.equal(resolveInitialLane('cloud', null), 'cloud');
  assert.equal(resolveInitialLane('local', null), 'local');
});

test('no handler wired → no Continue action at all', () => {
  assert.equal(canContinueFromCheckpoint(cloudDone, save, { hasHandler: false }), false);
  assert.equal(canContinueFromCheckpoint(localRun, save, { continueSource: 'any', hasHandler: false }),
    false);
});
