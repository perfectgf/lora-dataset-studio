import test from 'node:test';
import assert from 'node:assert/strict';

import {
  ENGINES_GROUPS, groupDomId, readOpenGroups, storeGroupToggle,
} from './settingsGroups.js';

/** A localStorage double that can also be told to fail — private windows and
 *  blocked site data are real states this must degrade through. */
function fakeStorage({ throwing = false } = {}) {
  const map = new Map();
  return {
    getItem: (k) => { if (throwing) throw new Error('blocked'); return map.get(k) ?? null; },
    setItem: (k, v) => { if (throwing) throw new Error('blocked'); map.set(k, v); },
    map,
  };
}

test('the Image engines groups carry stable ids — they live in localStorage and DOM anchors', () => {
  assert.deepEqual(ENGINES_GROUPS.map((g) => g.id),
    ['engines-keys', 'klein', 'krea', 'lora-presets', 'seedvr2', 'prompts']);
  for (const g of ENGINES_GROUPS) {
    assert.ok(g.title && g.blurb && g.icon, `${g.id} is missing display fields`);
  }
});

test('the DOM anchor is section-scoped, so two sections can reuse a group id', () => {
  assert.equal(groupDomId('engines', 'klein'), 'settings-group-engines-klein');
});

test('toggles round-trip through storage, per section', () => {
  const s = fakeStorage();
  storeGroupToggle(s, 'engines', 'klein', true);
  storeGroupToggle(s, 'engines', 'prompts', true);
  storeGroupToggle(s, 'engines', 'klein', false);
  assert.deepEqual([...readOpenGroups(s, 'engines')], ['prompts']);
  // Another section's memory is its own.
  assert.deepEqual([...readOpenGroups(s, 'scraping')], []);
});

test('a missing, corrupt or blocked storage means "all collapsed", never a crash', () => {
  assert.deepEqual([...readOpenGroups(null, 'engines')], []);
  const s = fakeStorage();
  s.map.set('settingsGroupsOpen.engines', '{not json');
  assert.deepEqual([...readOpenGroups(s, 'engines')], []);
  s.map.set('settingsGroupsOpen.engines', JSON.stringify([1, 'klein', null]));
  assert.deepEqual([...readOpenGroups(s, 'engines')], ['klein']);
  const blocked = fakeStorage({ throwing: true });
  assert.deepEqual([...readOpenGroups(blocked, 'engines')], []);
  storeGroupToggle(blocked, 'engines', 'klein', true);   // must not throw
});
