import test from 'node:test';
import assert from 'node:assert/strict';
import { appearanceKey, appearancePolicyChanged } from './captionAppearancePolicy.js';

const LOCK = null;                       // no policy: the classic identity lock
const DEFAULTS = { hair: 'omit', makeup: 'describe', facial_hair: 'omit', glasses: 'describe' };

test('activating the section is a change — the nudge fires', () => {
  // The case that bites: flipping ONE row switches the whole dataset to a policy,
  // and the 40 images already captioned keep the classic lock until re-captioned.
  assert.equal(appearancePolicyChanged(LOCK, { ...DEFAULTS, glasses: 'omit' }), true);
  assert.equal(appearancePolicyChanged(LOCK, DEFAULTS), true);
});

test('saving vocabulary or Extra instructions alone does NOT nudge', () => {
  // The popover re-sends the full four-family dict; an unchanged policy coming
  // back from the server must not read as an edit or every save would cry wolf.
  assert.equal(appearancePolicyChanged(DEFAULTS, { ...DEFAULTS }), false);
  assert.equal(appearancePolicyChanged(LOCK, {}), false);
  assert.equal(appearancePolicyChanged(LOCK, undefined), false);
});

test('moving one family inside an active policy still nudges', () => {
  assert.equal(
    appearancePolicyChanged(DEFAULTS, { ...DEFAULTS, hair: 'describe' }), true);
  assert.equal(
    appearancePolicyChanged({ ...DEFAULTS, makeup: 'omit' }, DEFAULTS), true);
});

test('key order never decides — the popover spreads defaults, the server re-serializes', () => {
  const reordered = {
    glasses: 'describe', facial_hair: 'omit', makeup: 'describe', hair: 'omit',
  };
  assert.equal(appearanceKey(reordered), appearanceKey(DEFAULTS));
  assert.equal(appearancePolicyChanged(DEFAULTS, reordered), false);
});

test('clearing a policy back to the classic lock nudges too', () => {
  assert.equal(appearancePolicyChanged(DEFAULTS, {}), true);
  assert.equal(appearanceKey(LOCK), '');
  assert.equal(appearanceKey({}), '');
});
