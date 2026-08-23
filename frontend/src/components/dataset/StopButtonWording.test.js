// A destructive button must be named after what it DESTROYS, not after the
// side effect it happens to produce. The training Stop used to read
// "Finish / re-enable ComfyUI" — housekeeping wording on a button that kills an
// hours-long run — and a user reported avoiding it for days rather than
// discovering that (wannadecryptor, Discord). These are wording contracts, not
// existence checks: a test that only asserted "the button is there" would have
// passed happily on the broken label.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const panel = fs.readFileSync(new URL('./TrainingPanel.jsx', import.meta.url), 'utf8');
const workspace = fs.readFileSync(new URL('./DatasetWorkspace.jsx', import.meta.url), 'utf8');
const captionLab = fs.readFileSync(new URL('./CaptionLab.jsx', import.meta.url), 'utf8');
const hook = fs.readFileSync(new URL('../../hooks/useDataset.js', import.meta.url), 'utf8');
// Registry topic data lives in ./topics/* since the 2026-08-24 split.
const registry = ['../../help/helpRegistry.js', '../../help/topics/settingsSections.js',
  '../../help/topics/workspaceSections.js', '../../help/topics/pages.js',
  '../../help/topics/videoLane.js', '../../help/topics/settingsFields.js',
  '../../help/topics/actions.js']
  .map((p) => fs.readFileSync(new URL(p, import.meta.url), 'utf8')).join('\n');

/** The JSX block of the one button that calls stopTraining(), from the guard
 *  that renders it to its closing tag — so the label assertions below can only
 *  ever be satisfied by THAT button's own text. */
function stopTrainingBlock(source) {
  const start = source.indexOf('{status.in_progress && (');
  assert.notEqual(start, -1, 'the training Stop button block was not found');
  const end = source.indexOf('</button>', start);
  assert.notEqual(end, -1, 'the training Stop button has no closing tag');
  const block = source.slice(start, end);
  assert.match(block, /ds\.stopTraining\(\)/,
    'the located block is not the one that calls stopTraining()');
  return block;
}

test('the training Stop button is labelled with the stop verb, not the side effect', () => {
  const block = stopTrainingBlock(panel);
  // The visible label — verb first.
  assert.match(block, /⏹ Stop training/);
  // The old housekeeping wording must never come back on this button.
  assert.doesNotMatch(block, /Finish \/ re-enable ComfyUI/);
  // Nowhere in the panel either — no second copy hiding elsewhere.
  assert.doesNotMatch(panel, /Finish \/ re-enable ComfyUI/);
});

test('the training Stop button says in its title what is KEPT, ComfyUI coming second', () => {
  const block = stopTrainingBlock(panel);
  const title = block.match(/title="([^"]+)"/);
  assert.ok(title, 'the training Stop button has no title');
  assert.match(title[1], /^Stops/, 'the title must lead with the stop, not the release');
  assert.match(title[1], /Checkpoints already saved are kept/);
  assert.match(title[1], /ComfyUI/);
  assert.ok(title[1].indexOf('kept') < title[1].indexOf('ComfyUI'),
    'what is kept must come before the ComfyUI side effect');
});

test('stopping an hours-long training asks for confirmation, and says what survives', () => {
  const block = stopTrainingBlock(panel);
  assert.match(block, /window\.confirm\(/);
  assert.match(block, /Stop the training run for/);
  assert.match(block, /Checkpoints already saved remain available/);
});

test('the success toast reports the stop, not just the freed GPU', () => {
  assert.match(hook, /toast\.success\('Training stopped — checkpoints already saved are kept; ComfyUI is re-enabled\.'\)/);
  assert.doesNotMatch(hook, /toast\.success\('ComfyUI re-enabled'\)/);
});

test('the captioning Stop keeps the same convention: stop verb + what is kept', () => {
  // The banner Stop for a caption/recaption pass — same shape, already correct;
  // this guard is what stops it from drifting into side-effect wording.
  const start = workspace.indexOf("act?.kind === 'caption' || act?.kind === 'recaption'");
  assert.notEqual(start, -1, 'the captioning Stop button block was not found');
  const block = workspace.slice(start, workspace.indexOf('</button>', start));
  assert.match(block, /ds\.cancelCaption/);
  assert.match(block, /⏹ Stop/);
  assert.doesNotMatch(block, /re-enable ComfyUI/);
  const title = block.match(/title="([^"]+)"/);
  assert.ok(title, 'the captioning Stop button has no title');
  assert.match(title[1], /^Stops/);
  assert.match(title[1], /captions already written are kept/i);
  // The Caption Lab's own Stop follows suit.
  assert.match(captionLab, /■ Stop/);
  assert.doesNotMatch(captionLab, /re-enable ComfyUI/);
});

test('the training Stop is reachable from the help registry', () => {
  assert.match(registry, /action\('action-training-stop', 'Stop a training run'/);
});
