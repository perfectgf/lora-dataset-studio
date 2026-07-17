import test from 'node:test';
import assert from 'node:assert/strict';
import {
  normalizeLoraName, findLora, isKnownLora, groupLoras, filterLoras, compatBadge,
  buildVisibleOptions,
} from './kleinLoraOptions.js';

const LORAS = [
  { name: 'klein\\style_a.safetensors', arch: 'flux2klein', label: 'FLUX.2 Klein', compatible: 'yes' },
  { name: 'root_flux.safetensors', arch: 'flux', label: 'FLUX.1', compatible: 'yes' },
  { name: 'sub\\photo_sdxl.safetensors', arch: 'sdxl', label: 'SDXL', compatible: 'no' },
  { name: 'mystery.safetensors', arch: null, label: null, compatible: 'unknown' },
];

test('normalizeLoraName is separator- and case-insensitive and trims', () => {
  assert.equal(normalizeLoraName('klein\\My-LoRA.safetensors'), 'klein/my-lora.safetensors');
  assert.equal(normalizeLoraName('  klein/My-LoRA.safetensors '), 'klein/my-lora.safetensors');
  assert.equal(normalizeLoraName(null), '');
});

test('findLora matches a stored value across separators; blank/miss -> null', () => {
  // stored with forward slash, scanned with backslash -> still the same file
  assert.equal(findLora('klein/style_a.safetensors', LORAS)?.arch, 'flux2klein');
  assert.equal(findLora('klein\\style_a.safetensors', LORAS)?.arch, 'flux2klein');
  assert.equal(findLora('', LORAS), null);
  assert.equal(findLora('not/on/disk.safetensors', LORAS), null);
});

test('isKnownLora drives the "not found" badge', () => {
  assert.equal(isKnownLora('klein/style_a.safetensors', LORAS), true);
  assert.equal(isKnownLora('ghost.safetensors', LORAS), false);   // preset value with no file -> "not found"
});

test('groupLoras splits Klein-compatible from other, preserving order', () => {
  const { compatible, other } = groupLoras(LORAS);
  assert.deepEqual(compatible.map((e) => e.name),
    ['klein\\style_a.safetensors', 'root_flux.safetensors']);
  // both the positively-incompatible and the undetectable land in "other"
  assert.deepEqual(other.map((e) => e.name),
    ['sub\\photo_sdxl.safetensors', 'mystery.safetensors']);
});

test('filterLoras matches name (incl. subfolder) and arch label; empty -> all', () => {
  assert.deepEqual(filterLoras(LORAS, 'sdxl').map((e) => e.name), ['sub\\photo_sdxl.safetensors']);
  assert.deepEqual(filterLoras(LORAS, 'klein\\').map((e) => e.name), ['klein\\style_a.safetensors']);
  assert.deepEqual(filterLoras(LORAS, 'FLUX').map((e) => e.name),
    ['klein\\style_a.safetensors', 'root_flux.safetensors']);   // label match, case-insensitive
  assert.equal(filterLoras(LORAS, '').length, 4);
  assert.equal(filterLoras(LORAS, '   ').length, 4);
});

test('buildVisibleOptions caps, keeps compatible-first, and reports hidden count', () => {
  // 25 compatible + 5 incompatible; cap 20 -> only 20 shown, all compatible, 10 hidden.
  const many = [];
  for (let i = 0; i < 25; i += 1) {
    many.push({ name: `klein/c${String(i).padStart(2, '0')}.safetensors`, arch: 'flux2klein', label: 'FLUX.2 Klein', compatible: 'yes' });
  }
  for (let i = 0; i < 5; i += 1) {
    many.push({ name: `sdxl/x${i}.safetensors`, arch: 'sdxl', label: 'SDXL', compatible: 'no' });
  }
  const v = buildVisibleOptions(many, '', 20);
  assert.equal(v.options.length, 20);
  assert.equal(v.compatible.length, 20);
  assert.equal(v.other.length, 0);                 // incompatible ones fall past the cap
  assert.equal(v.hiddenCount, 10);                 // 30 filtered - 20 shown
  // options is the flat render order (compatible then other) — the keyboard index basis
  assert.deepEqual(v.options.map((e) => e.name), v.compatible.map((e) => e.name));
});

test('buildVisibleOptions filters by substring before capping', () => {
  const v = buildVisibleOptions(LORAS, 'sdxl', 20);
  assert.deepEqual(v.options.map((e) => e.name), ['sub\\photo_sdxl.safetensors']);
  assert.equal(v.hiddenCount, 0);
});

test('compatBadge maps verdicts to tone + text', () => {
  assert.equal(compatBadge('yes', 'FLUX.2 Klein').tone, 'compatible');
  assert.equal(compatBadge('yes', 'FLUX.2 Klein').text, 'FLUX.2 Klein');
  assert.equal(compatBadge('no', 'SDXL').tone, 'incompatible');
  assert.equal(compatBadge('no', 'SDXL').text, 'SDXL');
  assert.equal(compatBadge('unknown', null).tone, 'unknown');
  assert.equal(compatBadge('unknown', null).text, 'Unknown arch');
});
