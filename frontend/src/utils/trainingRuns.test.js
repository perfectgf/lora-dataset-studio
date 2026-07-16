import test from 'node:test';
import assert from 'node:assert/strict';

import {
  canStopLocalRun,
  isTrainingRecipeReplayBlocked,
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

test('only incompatible recipe diagnostics block retry and continue', () => {
  assert.equal(isTrainingRecipeReplayBlocked(null), false);
  assert.equal(isTrainingRecipeReplayBlocked({ status: 'error', recipe_status: 'safe' }), false);
  assert.equal(isTrainingRecipeReplayBlocked({ status: 'done', recipe_status: 'legacy_incompatible' }), true);
  assert.equal(isTrainingRecipeReplayBlocked({ status: 'error', recipe_status: 'incompatible' }), true);
  assert.equal(isTrainingRecipeReplayBlocked({ recipe: { status: 'incompatible' } }), true);
  assert.equal(isTrainingRecipeReplayBlocked({ status: 'legacy_incompatible' }), true);
});
