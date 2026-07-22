import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const panel = fs.readFileSync(new URL('./TrainingPanel.jsx', import.meta.url), 'utf8');
const hook = fs.readFileSync(new URL('../../hooks/useDataset.js', import.meta.url), 'utf8');

test('continue retry uses the accumulating guarded request helper', () => {
  assert.match(panel, /runConfirmableTrainingRequest/);
  // the lane picker swapped the direct call for a lane-selected hook — the
  // guarded, accumulating retry wrapper is still the ONLY way either lane goes out
  assert.match(panel, /\(continueOpts\) => \(inCloud \? ds\.continueTrainingInCloud : ds\.continueTraining\)\(/);
  assert.match(panel, /confirmableRetryFlag\(error, 'Continue anyway \(force\)'\)/);
});

test('continue request sends caption override flags and leaves their toast to the confirm loop', () => {
  assert.match(hook, /allow_caption_mismatch: !!opts\.allowCaptionMismatch/);
  assert.match(hook, /allow_uncaptioned: !!opts\.allowUncaptioned/);
  assert.match(hook, /allow_caption_quality: !!opts\.allowCaptionQuality/);
  assert.match(hook, /includes\('MISMATCH_CAPTION: '\)/);
  assert.match(hook, /includes\('UNCAPTIONED: '\)/);
});

test('a disabled Continue button states its reason IN the panel, not only in a title', () => {
  // Reported as "the Continue button does nothing": it was disabled, and the only
  // explanation lived in title=, which never shows without a mouse hover.
  assert.match(panel, /Continue is off: these checkpoints come from a different LoRA family/);
  assert.match(panel, /Continue is off while a training runs on this machine/);
  // the in-panel reason must be driven by the SAME conditions as the disabled prop,
  // so the two can never disagree about why the button is off
  assert.match(panel, /\{!checkpointMatchesTraining && \(/);
  assert.match(panel, /checkpointMatchesTraining && status\.in_progress\s*\n?\s*&& !continueLanes\.cloud\.available/);
});

test('a trigger rename reports what moved on disk, and says when nothing did', () => {
  assert.match(hook, /const renamed = d\.trigger_rename;/);
  // a blocked rename must NOT read as success: the old artefacts keep the old name
  assert.match(hook, /if \(renamed && !renamed\.ok\)/);
  assert.match(hook, /toast\.warning\('Trigger word saved, but the artefacts/);
  assert.match(hook, /renamed\.files > 0/);
});
