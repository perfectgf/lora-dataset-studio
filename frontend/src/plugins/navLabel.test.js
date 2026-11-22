import test from 'node:test';
import assert from 'node:assert/strict';
import { plainNavLabel } from './navLabel.js';

test('the menu shows the words of a plugin entry, not its pictograph', () => {
  assert.equal(plainNavLabel('📷 Camera angles'), 'Camera angles');
  assert.equal(plainNavLabel('◉ Canvas'), 'Canvas');
  assert.equal(plainNavLabel('✨ DLSS 5'), 'DLSS 5');
  // a pictograph made of several code points (variation selector, joiner)
  assert.equal(plainNavLabel('🧑‍🎨 Creative Library'), 'Creative Library');
  assert.equal(plainNavLabel('🖼️  Gallery tools'), 'Gallery tools');
});

test('a label that starts with a letter or a digit is left as written', () => {
  assert.equal(plainNavLabel('LoRA Manager'), 'LoRA Manager');
  assert.equal(plainNavLabel('3D Studio'), '3D Studio');
  assert.equal(plainNavLabel('Été & hiver ✨'), 'Été & hiver ✨');
});

test('a label with no words keeps its glyph, and a non-string is returned untouched', () => {
  assert.equal(plainNavLabel('★'), '★');
  const element = { type: 'span' };
  assert.equal(plainNavLabel(element), element);
  assert.equal(plainNavLabel(undefined), undefined);
});
