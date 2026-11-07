import test from 'node:test';
import assert from 'node:assert/strict';
import { runOpensGraph } from './runGraph.js';

test('a local run opens its graph without a lineage or a checkpoint flag', () => {
  // The Runs list never scans a local run's disk, so it carries no checkpoint_ready.
  assert.equal(runOpensGraph({ source: 'local', record_id: 210 }), true);
});

test('a cloud run opens it once it holds saves or a lineage', () => {
  assert.equal(runOpensGraph({ source: 'cloud', record_id: 7 }), false);
  assert.equal(runOpensGraph({ source: 'cloud', record_id: 7, checkpoint_ready: true }), true);
  assert.equal(runOpensGraph({ source: 'cloud', record_id: 7, lineage: true }), true);
});

test('no graph without a record or for a full model', () => {
  assert.equal(runOpensGraph({ source: 'local', record_id: null }), false);
  assert.equal(runOpensGraph({ source: 'local', record_id: 210 }, { fullModel: true }), false);
});
