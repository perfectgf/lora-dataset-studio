import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const panel = fs.readFileSync(new URL('./TrainingPanel.jsx', import.meta.url), 'utf8');
const hook = fs.readFileSync(new URL('../../hooks/useDataset.js', import.meta.url), 'utf8');

test('continue retry uses the accumulating guarded request helper', () => {
  assert.match(panel, /runConfirmableTrainingRequest/);
  assert.match(panel, /\(continueOpts\) => ds\.continueTraining/);
  assert.match(panel, /confirmableRetryFlag\(\s*error, t\('workspace\.training\.dialogs\.continueForce'\)\)/);
});

test('continue request sends caption override flags and leaves their toast to the confirm loop', () => {
  assert.match(hook, /allow_caption_mismatch: !!opts\.allowCaptionMismatch/);
  assert.match(hook, /allow_uncaptioned: !!opts\.allowUncaptioned/);
  assert.match(hook, /allow_caption_quality: !!opts\.allowCaptionQuality/);
  assert.match(hook, /includes\('MISMATCH_CAPTION: '\)/);
  assert.match(hook, /includes\('UNCAPTIONED: '\)/);
});
