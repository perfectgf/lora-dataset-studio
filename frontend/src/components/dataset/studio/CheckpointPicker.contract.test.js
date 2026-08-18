import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const picker = readFileSync(new URL('./CheckpointPicker.jsx', import.meta.url), 'utf8');
const form = readFileSync(new URL('../../../hooks/useStudioForm.js', import.meta.url), 'utf8');
const panel = readFileSync(new URL('./RunSetupPanel.jsx', import.meta.url), 'utf8');

test('the picker has independent Mine and Theirs groups', () => {
  assert.match(picker, /Checkpoints to test/);
  assert.match(picker, /Compare with other LoRAs/);
  assert.match(picker, /<details/);
  assert.match(picker, /useKleinGenerationLoras\(family\)/);
  assert.match(picker, /HelpBadge topic="studio-guest-checkpoints"/);
  assert.match(picker, /onAddGuest/);
  assert.match(picker, /onRemoveGuest/);
  assert.match(picker, /onToggleGuest/);
});

test('the other-LoRAs accordion starts open when guests are already on the form', () => {
  assert.match(picker, /useState\(guests\.length > 0\)/);
});

test('removing a guest does not also toggle its checkbox', () => {
  assert.match(picker, /e\.stopPropagation\(\);\s*onRemoveGuest/);
});

test('the studio form persists guests separately from mine ticks', () => {
  assert.match(form, /guestCps/);
  assert.match(form, /selGuests/);
  assert.match(form, /chosenCheckpoints\(/);
  assert.match(form, /selCps still means "all of mine"/);
  assert.doesNotMatch(form, /selCps \?\? allFns\)\.filter\(\(fn\) => allFns\.includes\(fn\)\)/);
});

test('RunSetupPanel wires guests only on the epoch picker, not the canvas slot', () => {
  assert.match(panel, /guests=\{form\.guestCps\}/);
  assert.match(panel, /family=\{d\.family \|\| 'zimage'\}/);
  assert.match(panel, /checkpointSlot \?\? \(/);
});
