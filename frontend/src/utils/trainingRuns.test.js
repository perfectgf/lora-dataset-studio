import test from 'node:test';
import assert from 'node:assert/strict';

import {
  canStopLocalRun,
  isTrainingRecipeReplayBlocked,
  retryRequest,
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
