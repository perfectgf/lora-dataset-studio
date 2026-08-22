import test from 'node:test';
import assert from 'node:assert/strict';
import { QUEUE_KINDS, SELF_EXCLUSIVE_KINDS, activityBlocks, holdsLocalGpu, laneOf } from './activityLanes.js';

// GitHub #44, the report itself: an ✨ Upscale & improve batch used to disable
// every generation control until it finished.
test('a running improve batch no longer blocks a generation', () => {
  assert.equal(activityBlocks({ kind: 'improve', done: 3, total: 40 }, 'generate'), false);
});

test('queue work lines up behind queue work, in every direction', () => {
  for (const live of QUEUE_KINDS) {
    for (const next of QUEUE_KINDS) {
      const sameAndRefused = live === next && SELF_EXCLUSIVE_KINDS.includes(next);
      assert.equal(activityBlocks({ kind: live }, next), sameAndRefused,
        `${live} -> ${next}`);
    }
  }
});

// The backend answers 409 to both of these, so the button must not offer them.
test('the two repeats the backend refuses stay disabled', () => {
  assert.equal(activityBlocks({ kind: 'improve' }, 'improve'), true);
  assert.equal(activityBlocks({ kind: 'edit_reference' }, 'edit_reference'), true);
});

// A second ⚡ Generate on top of a running one is legal (the workspace asks for
// a confirmation, the backend caps the fan-out) — it must not be greyed out.
test('a second generation batch is allowed', () => {
  assert.equal(activityBlocks({ kind: 'generate', total: 12 }, 'generate'), false);
});

test('an exclusive pass keeps blocking everything, as before', () => {
  const exclusive = ['caption', 'recaption', 'analyze_faces', 'classify',
    'watermark_detect', 'watermark_clean', 'bank_export', 'bank_import',
    'training_export', 'backup'];
  for (const kind of exclusive) {
    assert.equal(laneOf(kind), 'exclusive', kind);
    for (const next of [...QUEUE_KINDS, 'caption', 'watermark_detect'])
      assert.equal(activityBlocks({ kind }, next), true, `${kind} -> ${next}`);
  }
});

// The reverse half of the same rule: queued generations must not let a pass that
// rewrites rows (or takes the GPU vision window) start underneath them.
test('queued work still blocks an exclusive pass', () => {
  for (const live of QUEUE_KINDS)
    for (const next of ['caption', 'recaption', 'analyze_faces', 'classify',
      'watermark_detect', 'watermark_clean', 'backup'])
      assert.equal(activityBlocks({ kind: live }, next), true, `${live} -> ${next}`);
});

test('no activity blocks nothing', () => {
  for (const activity of [null, undefined, {}, { kind: null }])
    assert.equal(activityBlocks(activity, 'generate'), false);
});

// Every kind the backend can publish must be classified — a new one added to
// dataset_activity.KINDS without a lane would silently fall into 'exclusive',
// which is the safe default but should be a deliberate choice.
test('every published kind has a lane', () => {
  const published = ['watermark_detect', 'watermark_clean', 'caption', 'recaption',
    'analyze_faces', 'classify', 'generate', 'improve', 'edit_reference',
    'bank_export', 'bank_import', 'training_export', 'backup'];
  for (const kind of published)
    assert.ok(['queue', 'exclusive'].includes(laneOf(kind)), kind);
  assert.deepEqual(published.filter((k) => laneOf(k) === 'queue'), QUEUE_KINDS);
});

// Opening the import dropzone during queued work is only half an answer: the
// auto head-crop wants the exclusive GPU vision window, and local queue work
// holds ComfyUI. Widening the door without widening this shipped an import that
// opened, ran, and came back 503 'GPU busy' on the crop.
test('local queue work holds the GPU, whatever kind it is', () => {
  for (const kind of QUEUE_KINDS) {
    assert.equal(holdsLocalGpu({ kind, engine: 'klein' }), true, kind);
    assert.equal(holdsLocalGpu({ kind, engine: 'seedvr2' }), true, kind);
    // Unknown or absent engine fails SAFE — assume the GPU is taken.
    assert.equal(holdsLocalGpu({ kind }), true, kind);
    assert.equal(holdsLocalGpu({ kind, engine: 'some-future-engine' }), true, kind);
  }
});

test('the two remote engines do not hold the local GPU', () => {
  for (const engine of ['nanobanana', 'chatgpt', 'ChatGPT', 'NanoBanana'])
    assert.equal(holdsLocalGpu({ kind: 'generate', engine }), false, engine);
});

test('an exclusive pass answers for its own GPU use, not through this gate', () => {
  // caption / classify / faces take the vision window themselves and already
  // block everything; folding them in here would double-count them.
  for (const kind of ['caption', 'classify', 'analyze_faces', 'watermark_clean'])
    assert.equal(holdsLocalGpu({ kind }), false, kind);
  for (const activity of [null, undefined, {}, { kind: null }])
    assert.equal(holdsLocalGpu(activity), false);
});
