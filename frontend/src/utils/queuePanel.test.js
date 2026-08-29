import test from 'node:test';
import assert from 'node:assert/strict';
import {
  elapsedLabel, hasQueue, jobLabel, jobOrigin, pausedAction, pausedReason,
  promoteBlockedReason, rowNote, summarize,
} from './queuePanel.js';

const job = (patch = {}) => ({
  job_id: 'j', title: 'Generation', surface: '📁 Dataset', engine: null,
  status: 'queued', position: 2, cancellable: true, blocked_by: null,
  dataset_name: null, ...patch,
});

test('an empty queue docks nothing over the app', () => {
  for (const listing of [null, undefined, {}, { jobs: [] }, { jobs: null }])
    assert.equal(hasQueue(listing), false);
  assert.equal(summarize({ jobs: [] }), '');
});

test('the pill names what is happening before what is waiting', () => {
  assert.equal(
    summarize({ jobs: [job(), job(), job()], generating: 1, queued: 2, stalled: 0 }),
    '1 generating · 2 queued',
  );
  assert.equal(
    summarize({ jobs: [job()], generating: 0, queued: 0, stalled: 1 }),
    '1 paused',
  );
});

// A live row in a status none of the three counters knows would otherwise make
// the pill claim an empty queue while the panel lists jobs.
test('a queue nobody counted is still reported as a queue', () => {
  assert.equal(summarize({ jobs: [job(), job()] }), '2 in the queue');
});

// Training and the vision pass hold the GPU outside this queue. A queue that
// counts a line which never advances, and says nothing, is the original
// complaint rebuilt one level up.
test('a queue held by something outside it says what is holding it', () => {
  assert.equal(
    pausedReason({ jobs: [job()], paused_reason: 'LoRA training in progress - the studio is unavailable (GPU busy).' }),
    'LoRA training in progress - the studio is unavailable (GPU busy).',
  );
  for (const listing of [{ jobs: [job()] }, { jobs: [job()], paused_reason: null },
    { jobs: [job()], paused_reason: '   ' }, null])
    assert.equal(pausedReason(listing), null);
});

// Some holds end by themselves and some never do. Only the second kind earns a
// button — and it must carry its own words, because a control the dock cannot
// explain is the failure mode this whole panel exists to remove.
test('a hold the user can answer offers the answer, with what it costs', () => {
  const action = pausedAction({
    jobs: [job()],
    paused_action: {
      kind: 'share_gpu', label: 'Run anyway', models: ['llama3:8b'],
      confirm: 'Nothing of yours is unloaded — but generation can be much slower.',
    },
  });
  assert.equal(action.label, 'Run anyway');
  assert.deepEqual(action.models, ['llama3:8b']);
  assert.match(action.confirm, /slower/);
});

test('an unknown, empty or absent offer is no button at all', () => {
  // An older backend sends nothing; a future kind is one this dock cannot drive;
  // an offer with no words on it would render as a mystery control.
  for (const paused_action of [
    undefined, null, {},
    { kind: 'reboot_the_gpu', label: 'Do it', confirm: 'Sure?' },
    { kind: 'share_gpu', label: '  ', confirm: 'Sure?' },
    { kind: 'share_gpu', label: 'Run anyway', confirm: '' },
  ]) assert.equal(pausedAction({ jobs: [job()], paused_action }), null);
  assert.equal(pausedAction(null), null);
});

test('a job names its engine only when it has one', () => {
  assert.equal(jobLabel(job({ title: 'Upscale & improve', engine: 'Klein' })),
    'Upscale & improve · Klein');
  assert.equal(jobLabel(job({ title: 'Test Studio image' })), 'Test Studio image');
});

test('a job says which dataset it is working on when two could be feeding the queue', () => {
  assert.equal(jobOrigin(job({ dataset_name: 'Faces' })), '📁 Dataset · Faces');
  assert.equal(jobOrigin(job()), '📁 Dataset');
  assert.equal(jobOrigin(job({ surface: '🧪 Test Studio', dataset_name: null })),
    '🧪 Test Studio');
});

test('elapsed time stays compact at every scale', () => {
  const now = Date.parse('2026-01-01T12:30:00Z');
  assert.equal(elapsedLabel('2026-01-01T12:29:48Z', now), '12s');
  assert.equal(elapsedLabel('2026-01-01T12:25:00Z', now), '5 min');
  assert.equal(elapsedLabel('2026-01-01T10:05:00Z', now), '2 h 25 min');
  // A clock that disagrees with the server must not print a negative age.
  assert.equal(elapsedLabel('2026-01-01T12:31:00Z', now), '0s');
  for (const bad of [null, '', 'not a date']) assert.equal(elapsedLabel(bad, now), '');
});

// The whole point of the change: a control that refuses must say why, and where
// the thing that does work lives.
test('a job the panel may not cancel says who owns it', () => {
  assert.equal(
    rowNote(job({ cancellable: false, blocked_by: 'the 🧽 Clean watermarks pass' })),
    'Owned by the 🧽 Clean watermarks pass — stop it from there.',
  );
  assert.equal(rowNote(job()), null);
});

// `status` collapses cancel_requested and stalled onto one word so the row
// renders the same; `raw_status` is what tells them apart. The dock emitted it
// and read it nowhere, so a job the user had just cancelled announced itself as
// 'ComfyUI stopped answering'.
test('a job on its way out says it is cancelling, not that ComfyUI died', () => {
  assert.equal(rowNote(job({ status: 'stalled', raw_status: 'cancel_requested' })),
    'Cancelling — waiting for ComfyUI to let go of it.');
});

test('a paused job points at the recovery banner rather than at a dead button', () => {
  assert.match(rowNote(job({ status: 'stalled', raw_status: 'stalled' })), /recovery banner/);
});

test('“already next” and “already running” are not the same refusal', () => {
  assert.equal(promoteBlockedReason(job({ position: 1 })), 'Already next in line.');
  assert.match(promoteBlockedReason(job({ status: 'generating' })), /nothing left to re-order/);
  assert.equal(promoteBlockedReason(job({ position: 3 })), null);
});
