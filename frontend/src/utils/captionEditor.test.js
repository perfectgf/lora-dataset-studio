import test from 'node:test';
import assert from 'node:assert/strict';
import { captionCharacterLabel, isCaptionSaveShortcut, isLikelyTruncatedCaption } from './captionEditor.js';

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

test('flags captions that look cut off by the legacy 800-char cap', () => {
  // Exactly 800 chars and ending on a bare word (no sentence punctuation) -> legacy
  // truncation, so the editor nudges the user to re-caption.
  const truncated = 'word '.repeat(159) + 'and a';  // 795 + 5 = 800 chars, ends on a bare word
  assert.equal(truncated.length, 800);
  assert.equal(isLikelyTruncatedCaption(truncated), true);
});

test('does not flag complete or short captions', () => {
  assert.equal(isLikelyTruncatedCaption(''), false);
  assert.equal(isLikelyTruncatedCaption('A short, finished caption.'), false);
  // Exactly 800 but ending on sentence punctuation -> deliberately complete, not flagged.
  const complete = 'x'.repeat(799) + '.';
  assert.equal(complete.length, 800);
  assert.equal(isLikelyTruncatedCaption(complete), false);
  // 801 chars -> not the legacy ceiling, never flagged.
  assert.equal(isLikelyTruncatedCaption('x'.repeat(801)), false);
});
