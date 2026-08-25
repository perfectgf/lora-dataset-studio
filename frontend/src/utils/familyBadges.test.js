import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { FAMILY_IDS, familyBadge, familyBadgeClass, familyLabel } from './familyBadges.js';

const SRC = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const read = (p) => readFileSync(resolve(SRC, p), 'utf8');

test('every family names its product, never its internal id', () => {
  assert.equal(familyLabel('flux'), 'FLUX.1');
  assert.equal(familyLabel('flux2klein'), 'FLUX.2 Klein');
  assert.equal(familyLabel('zimage'), 'Z-Image');
  // An id nobody has taught this table still renders, as itself.
  assert.deepEqual(familyBadge('brand-new'), [
    'brand-new', 'border-border bg-surface-raised text-content-muted']);
});

test('no family wears the accent colour', () => {
  // Amber IS the Safelight accent, and `indigo` is remapped onto that same ramp
  // in tailwind.config.js — a label in either reads as a control.
  for (const id of FAMILY_IDS) {
    const cls = familyBadgeClass(id);
    assert.doesNotMatch(cls, /amber|indigo/,
      `${id} is wearing the accent colour — pick a hue that is not a control`);
  }
});

test('two families never share a hue', () => {
  const hues = FAMILY_IDS.map((id) => familyBadgeClass(id).match(/text-([a-z]+)-/)[1]);
  assert.equal(new Set(hues).size, hues.length,
    `two families share a hue: ${hues.join(', ')}`);
});

test('the kind badges keep their own hues to themselves', () => {
  // Concept and Style sit on the same tile, above the family chips.
  const hues = FAMILY_IDS.map((id) => familyBadgeClass(id).match(/text-([a-z]+)-/)[1]);
  for (const taken of ['fuchsia', 'cyan']) {
    assert.ok(!hues.includes(taken),
      `${taken} already means Concept/Style on the same tile`);
  }
});

test('both surfaces read this table instead of copying it', () => {
  // The drift this module exists to stop: the library and the Studio picker each
  // kept their own copy, and Krea's colour diverged from the accent rule.
  for (const f of ['components/dataset/DatasetListPanel.jsx',
    'components/dataset/studio/LoraPicker.jsx']) {
    const src = read(f);
    assert.match(src, /from '.*familyBadges(\.js)?'/,
      `${f} must import the shared family badges`);
    // A local table is what drifted: a family id mapped to its own colours.
    for (const id of FAMILY_IDS) {
      assert.doesNotMatch(src, new RegExp(`${id}:\\s*(\\[\\s*)?'[^']*(border|text)-\\w+-\\d`),
        `${f} still maps ${id} to its own colours — read the shared table instead`);
    }
  }
});
