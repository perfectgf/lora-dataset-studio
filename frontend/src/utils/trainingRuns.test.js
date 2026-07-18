import test from 'node:test';
import assert from 'node:assert/strict';

import {
  canStopLocalRun,
  formatDuration,
  groupRunsByDataset,
  isTrainingRecipeReplayBlocked,
  retryRequest,
  runBaseModelLabel,
  runDurationSeconds,
  runRetryKey,
  trainingRunVariantLabel,
} from './trainingRuns.js';

test('local run is stoppable only while an identified run is in progress', () => {
  assert.equal(canStopLocalRun(null), false);
  assert.equal(canStopLocalRun({ in_progress: false, current: { dataset_id: 12 } }), false);
  assert.equal(canStopLocalRun({ in_progress: true, current: null }), false);
  assert.equal(canStopLocalRun({ in_progress: true, current: {} }), false);
  assert.equal(canStopLocalRun({ in_progress: true, current: { dataset_id: 12 } }), false);
  assert.equal(canStopLocalRun({ in_progress: true, current: { dataset_id: 12, run_token: 'run-abc' } }), true);
});

test('variant labels distinguish Z-Image Base from Krea Raw', () => {
  assert.equal(trainingRunVariantLabel('zimage', 'base'), 'Base');
  assert.equal(trainingRunVariantLabel('krea', 'base'), 'Raw');
  assert.equal(trainingRunVariantLabel('zimage', 'deturbo'), 'De-Turbo');
  assert.equal(trainingRunVariantLabel('zimage', 'turbo'), 'Turbo');
});

test('retry posts to the local endpoint for local runs and the cloud endpoint otherwise', () => {
  assert.deepEqual(
    retryRequest({ source: 'local', record_id: 42 }),
    { url: '/api/dataset/train/retry', body: { record_id: 42 } });
  assert.deepEqual(
    retryRequest({ source: 'cloud', run_id: 7 }),
    { url: '/api/dataset/train/cloud/retry', body: { run_id: 7 } });
  // record_id 0 is a real id (falsy but not null), local runs still address it
  assert.deepEqual(
    retryRequest({ source: 'local', record_id: 0 }),
    { url: '/api/dataset/train/retry', body: { record_id: 0 } });
  // no addressable id -> null so the caller can no-op
  assert.equal(retryRequest({ source: 'local' }), null);
  assert.equal(retryRequest({ source: 'cloud' }), null);
});

test('retry in-flight keys never collide across cloud and local namespaces', () => {
  assert.equal(runRetryKey({ source: 'cloud', run_id: 5 }), 'c5');
  assert.equal(runRetryKey({ source: 'local', record_id: 5 }), 'l5');
  assert.notEqual(
    runRetryKey({ source: 'cloud', run_id: 5 }),
    runRetryKey({ source: 'local', record_id: 5 }));
});

test('only incompatible recipe diagnostics block retry and continue', () => {
  assert.equal(isTrainingRecipeReplayBlocked(null), false);
  assert.equal(isTrainingRecipeReplayBlocked({ status: 'error', recipe_status: 'safe' }), false);
  assert.equal(isTrainingRecipeReplayBlocked({ status: 'done', recipe_status: 'legacy_incompatible' }), true);
  assert.equal(isTrainingRecipeReplayBlocked({ status: 'error', recipe_status: 'incompatible' }), true);
  assert.equal(isTrainingRecipeReplayBlocked({ recipe: { status: 'incompatible' } }), true);
  assert.equal(isTrainingRecipeReplayBlocked({ status: 'legacy_incompatible' }), true);
});

test('base-model label spells official bases and names custom checkpoints', () => {
  // Official base ('' = family default): the canonical family + variant name.
  assert.deepEqual(runBaseModelLabel({ base_model: '', train_type: 'zimage', variant: 'turbo' }),
    { text: 'Z-Image Turbo', title: 'Official base: Z-Image Turbo', custom: false });
  assert.deepEqual(runBaseModelLabel({ base_model: '', train_type: 'krea', variant: 'base' }),
    { text: 'Krea 2 Raw', title: 'Official base: Krea 2 Raw', custom: false });
  // No variant on the run: family alone, no trailing space.
  assert.deepEqual(runBaseModelLabel({ base_model: '', train_type: 'flux2klein' }),
    { text: 'FLUX.2 Klein', title: 'Official base: FLUX.2 Klein', custom: false });
});

test('base-model label reduces a custom base to its leaf filename/tag', () => {
  // A Windows path never surfaces its parent folders (nor a title that would).
  assert.deepEqual(
    runBaseModelLabel({ base_model: 'D:\\models\\merges\\bigLove_zt3.safetensors', train_type: 'zimage', variant: 'turbo' }),
    { text: 'bigLove_zt3.safetensors', title: 'Custom base: bigLove_zt3.safetensors', custom: true });
  // An HF repo id drops the owner, keeping the repo leaf.
  assert.deepEqual(
    runBaseModelLabel({ base_model: 'owner/lds-base-h1a2b3c', train_type: 'krea', variant: 'base' }),
    { text: 'lds-base-h1a2b3c', title: 'Custom base: lds-base-h1a2b3c', custom: true });
});

test('base-model label degrades to null when a legacy run never recorded a base', () => {
  assert.equal(runBaseModelLabel({ train_type: 'zimage', variant: 'turbo' }), null); // undefined
  assert.equal(runBaseModelLabel({ base_model: null, train_type: 'zimage' }), null);
  assert.equal(runBaseModelLabel(null), null);
});

test('runs group by dataset only when consecutive — history order is preserved', () => {
  const runs = [
    { dataset_id: 1, run_id: 9 },
    { dataset_id: 1, run_id: 8 },
    { dataset_id: 2, record_id: 3 },
    { dataset_id: 1, run_id: 5 },
  ];
  const groups = groupRunsByDataset(runs);
  assert.equal(groups.length, 3);
  assert.deepEqual(groups.map((g) => g.datasetId), [1, 2, 1]);
  assert.deepEqual(groups[0].runs.map((r) => r.run_id), [9, 8]);
  assert.equal(groups[1].runs.length, 1);
  assert.deepEqual(groupRunsByDataset([]), []);
  assert.deepEqual(groupRunsByDataset(null), []);
});

test('run duration needs both timestamps and pins naive strings to UTC', () => {
  assert.equal(runDurationSeconds({ created_at: '2026-07-17T10:00:00' }), null);
  assert.equal(runDurationSeconds({ finished_at: '2026-07-17T10:00:00' }), null);
  assert.equal(runDurationSeconds(null), null);
  // naive strings (backend utcnow.isoformat()) diff as UTC regardless of TZ
  assert.equal(runDurationSeconds({
    created_at: '2026-07-17T10:00:00', finished_at: '2026-07-17T10:42:30',
  }), 2550);
  // explicit offsets are honored as-is
  assert.equal(runDurationSeconds({
    created_at: '2026-07-17T10:00:00Z', finished_at: '2026-07-17T11:00:00+00:00',
  }), 3600);
  // a finish BEFORE the start is bogus data, not a negative badge
  assert.equal(runDurationSeconds({
    created_at: '2026-07-17T10:00:00', finished_at: '2026-07-17T09:00:00',
  }), null);
});

test('durations format compactly for the run cards', () => {
  assert.equal(formatDuration(null), null);
  assert.equal(formatDuration(-5), null);
  assert.equal(formatDuration(48), '48s');
  assert.equal(formatDuration(2550), '42m');
  assert.equal(formatDuration(3900), '1h 05m');
  assert.equal(formatDuration(2 * 86400 + 3 * 3600), '2d 3h');
});
