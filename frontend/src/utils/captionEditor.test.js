import test from 'node:test';
import assert from 'node:assert/strict';
import { captionCharacterLabel, isCaptionSaveShortcut } from './captionEditor.js';

test('expanded caption editor saves with Ctrl+Enter or Command+Enter', () => {
  assert.equal(isCaptionSaveShortcut({ key: 'Enter', ctrlKey: true, metaKey: false }), true);
  assert.equal(isCaptionSaveShortcut({ key: 'Enter', ctrlKey: false, metaKey: true }), true);
  assert.equal(isCaptionSaveShortcut({ key: 'Enter', ctrlKey: false, metaKey: false }), false);
  assert.equal(isCaptionSaveShortcut({ key: 'Escape', ctrlKey: true, metaKey: false }), false);
});

test('expanded caption editor exposes a readable character count', () => {
  assert.equal(captionCharacterLabel(''), '0 characters');
  assert.equal(captionCharacterLabel('x'), '1 character');
  assert.equal(captionCharacterLabel('a caption'), '9 characters');
});
