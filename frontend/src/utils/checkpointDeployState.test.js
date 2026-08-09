import test from 'node:test';
import assert from 'node:assert/strict';
import {
  DEPLOY_LEGEND, deployBadge, deployState, deployTally, deployTitleSuffix,
} from './checkpointDeployState.js';

test('a pill says whether you can generate from it right now', () => {
  assert.equal(deployState({ testable: true }), 'deployed');
  assert.equal(deployState({ testable: false }), 'on-disk');
  assert.equal(deployState({}), 'on-disk');
});

test('a checkpoint whose file is gone is not a deployment question', () => {
  assert.equal(deployState({ present: false, testable: true }), 'gone');
  assert.equal(deployBadge('gone').show, false);
  assert.equal(deployTitleSuffix({ present: false }), '');
});

test('each state carries a colour AND a shape AND words', () => {
  const on = deployBadge('deployed');
  const off = deployBadge('on-disk');
  assert.notEqual(on.tone, off.tone, 'colour channel');
  assert.notEqual(on.glyph, off.glyph, 'shape channel — colour alone excludes readers');
  assert.notEqual(on.text, off.text, 'words channel');
  // Filled vs hollow: the difference has to survive a greyscale screenshot.
  assert.equal(on.glyph, '●');
  assert.equal(off.glyph, '○');
});

test('a not-deployed checkpoint is never described as missing', () => {
  const off = deployBadge('on-disk');
  assert.equal(/missing|lost|not found|absent/i.test(off.text), false);
  assert.equal(off.text.includes('on disk'), true);
});

test('a not-deployed pill points at the button that already deploys it', () => {
  // The board deploys the picks that need it; nothing here may imply a second
  // mechanism the user has to find.
  assert.equal(deployBadge('on-disk').text.includes('Generate'), true);
});

test('the legend covers exactly the states that get a badge', () => {
  assert.deepEqual(DEPLOY_LEGEND.map((l) => l.tone), ['deployed', 'on-disk']);
  for (const entry of DEPLOY_LEGEND) {
    assert.equal(entry.glyph, deployBadge(entry.tone).glyph);
    // 📱 The phone-width key. It must be the SAME statement with the
    // explanation clipped, not a second wording: the board shows one or the
    // other depending on the screen, and two labels would drift.
    assert.ok(entry.short && entry.short.length <= 12, `${entry.tone} has a short key`);
    assert.ok(entry.label.startsWith(entry.short), `${entry.tone}: short is the head of label`);
  }
});

test('a lane tally counts only checkpoints that still have a file', () => {
  assert.deepEqual(deployTally([
    { testable: true }, { testable: true }, { testable: false },
    { present: false, testable: false },
  ]), { deployed: 2, onDisk: 1, total: 3 });
  assert.deepEqual(deployTally([]), { deployed: 0, onDisk: 0, total: 0 });
  assert.deepEqual(deployTally(null), { deployed: 0, onDisk: 0, total: 0 });
});
