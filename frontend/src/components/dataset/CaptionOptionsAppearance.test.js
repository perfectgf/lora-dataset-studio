import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const popover = readFileSync(new URL('./CaptionOptionsPopover.jsx', import.meta.url), 'utf8');
const workspace = readFileSync(new URL('./DatasetWorkspace.jsx', import.meta.url), 'utf8');

test('appearance toggles are character-only and do not save until a row is flipped', () => {
  assert.match(popover, /export const APPEARANCE_FAMILIES = \[/);
  assert.match(popover, /makeup: 'describe'/);
  assert.match(popover, /names\s+what is clearly visible so it stays prompt-controllable/);
  assert.match(popover, /kind !== 'concept' && kind !== 'style'/);
  assert.match(popover, /if \(appearanceDirty\) \{/);
  assert.match(popover, /payload\.appearance = \{ \.\.\.APPEARANCE_DEFAULTS/);
  assert.match(workspace, /kind=\{d\.kind\}/);
  assert.match(workspace, /d\.caption_leak\?\.watched/);
});
