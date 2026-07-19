import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const dialog = fs.readFileSync(new URL('./ContinueDialog.jsx', import.meta.url), 'utf8');
const panel = fs.readFileSync(new URL('./TrainingPanel.jsx', import.meta.url), 'utf8');
const cloud = fs.readFileSync(new URL('../../pages/CloudRunsPage.jsx', import.meta.url), 'utf8');
const hook = fs.readFileSync(new URL('../../hooks/useDataset.js', import.meta.url), 'utf8');

test('the dialog resolves a flexible-continue payload (steps, checkpoint, overrides)', () => {
  // fromStep is null only when the newest checkpoint is chosen — the in-place resume.
  assert.match(dialog, /fromStep:\s*isEarlier\s*\?\s*fromStep\s*:\s*null/);
  assert.match(dialog, /extraSteps:\s*extraNum/);
  assert.match(dialog, /overrides:\s*Object\.keys\(overrides\)\.length/);
  // safe subset only — cadence + preview prompts, never rank/base/optimizer.
  assert.match(dialog, /overrides\.save_every/);
  assert.match(dialog, /overrides\.sample_every/);
  assert.match(dialog, /overrides\.sample_prompts/);
  // its own help topic (registered in helpRegistry)
  assert.match(dialog, /topic="continue-training"/);
});

test('both hubs open the shared ContinueDialog', () => {
  assert.match(panel, /import ContinueDialog from '\.\/ContinueDialog'/);
  assert.match(panel, /<ContinueDialog/);
  assert.match(cloud, /import ContinueDialog from '\.\.\/components\/dataset\/ContinueDialog'/);
  assert.match(cloud, /<ContinueDialog/);
});

test('local continue still routes through the guarded, accumulating request helper', () => {
  assert.match(panel, /runConfirmableTrainingRequest/);
  assert.match(panel, /\(continueOpts\) => ds\.continueTraining/);
  assert.match(panel, /fromStep:\s*payload\.fromStep,\s*overrides:\s*payload\.overrides/);
  assert.match(panel, /confirmableRetryFlag\(error, t\('workspace\.training\.dialogs\.continueForce'\)\)/);
});

test('cloud continue posts the run, extra steps, chosen checkpoint and overrides', () => {
  assert.match(cloud, /from_step:\s*payload\.fromStep/);
  assert.match(cloud, /overrides:\s*payload\.overrides/);
  assert.match(cloud, /extra_steps:\s*payload\.extraSteps/);
});

test('the continue hook forwards from_step and overrides only when present', () => {
  assert.match(hook, /opts\.fromStep\s*!=\s*null\s*\?\s*\{\s*from_step:\s*opts\.fromStep\s*\}/);
  assert.match(hook, /opts\.overrides\s*\?\s*\{\s*overrides:\s*opts\.overrides\s*\}/);
});
