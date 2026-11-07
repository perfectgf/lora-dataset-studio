import test from 'node:test';
import assert from 'node:assert/strict';
import {
  IMPORT_ENCODING_LABEL,
  IMPORT_IMAGE_ACCEPT,
  IMPORT_IMAGE_FORMATS,
  importInputLimitLine,
  importInputLimitNote,
  importPolicyLine,
  preservesOriginalFiles,
} from './importPolicy.js';

test('the picker offers HEIC and AVIF too, while the policy names only the formats kept byte-for-byte', () => {
  assert.equal(IMPORT_IMAGE_ACCEPT,
    'image/jpeg,image/png,image/webp,image/bmp,image/heic,image/heif,image/avif,.heic,.heif,.avif');
  assert.equal(IMPORT_IMAGE_FORMATS, 'JPEG, PNG, WebP and BMP');
});

test('the default import policy preserves an eligible original file rather than guessing WebP', () => {
  assert.equal(preservesOriginalFiles(), true);
  assert.equal(importInputLimitLine(), '64 Mi-pixels and 16384 px per side');
  assert.equal(importPolicyLine(),
    'stored byte-for-byte in the original file and format (input limit: 64 Mi-pixels and 16384 px per side)');
  assert.equal(importPolicyLine({ encoding: 'preserve', max_side: 1024 }),
    'stored byte-for-byte in the original file and format (input limit: 64 Mi-pixels and 16384 px per side)');
});

test('input limits use the explicit capability aliases, with the old names as a rolling-update fallback', () => {
  assert.equal(importInputLimitLine({ input_max_pixels: 8 * 1024 * 1024, input_max_side: 4096 }),
    '8 Mi-pixels and 4096 px per side');
  assert.equal(importInputLimitLine({ preserve_max_pixels: 12 * 1024 * 1024, preserve_max_side: 6144 }),
    '12 Mi-pixels and 6144 px per side');
});

test('the capability flag keeps the preserve hint correct during a rolling update', () => {
  assert.equal(preservesOriginalFiles({ encoding: 'standard', preserve: true }), true);
  assert.equal(importPolicyLine({ encoding: 'standard', preserve: true }),
    'stored byte-for-byte in the original file and format (input limit: 64 Mi-pixels and 16384 px per side)');
});

test('legacy WebP policies still describe their explicit conversion behavior', () => {
  assert.equal(IMPORT_ENCODING_LABEL.standard, 'WebP q92');
  assert.equal(preservesOriginalFiles({ encoding: 'standard', max_side: 1536 }), false);
  assert.equal(importPolicyLine({ encoding: 'standard', max_side: 1536 }),
    'stored as WebP q92, resized to 1536 px on the long side, ratio kept (input limit: 64 Mi-pixels and 16384 px per side)');
  assert.equal(importPolicyLine({ encoding: 'lossless', max_side: 0, ceiling: 8192 }),
    'stored as WebP lossless at original size (input limit: 64 Mi-pixels and 16384 px per side)');
});

/* The budget is a SETTING now, and 0 is one of its values. A fallback that
   treated 0 as "missing" would quote the shipped default at the one user who
   had just switched the limit off — the exact class of stale hint this module
   exists to prevent. */
test('0 means no limit, and is never mistaken for a missing key', () => {
  assert.equal(importInputLimitLine({ input_max_pixels: 0, input_max_side: 0 }),
    'any size (no limit)');
  assert.equal(importInputLimitLine({ input_max_pixels: 0, input_max_side: 16384 }),
    '16384 px per side');
  assert.equal(importInputLimitLine({ input_max_pixels: 64 * 1024 * 1024, input_max_side: 0 }),
    '64 Mi-pixels');
  assert.equal(importInputLimitNote({ input_max_pixels: 0, input_max_side: 0 }),
    'no input size limit');
  assert.equal(importPolicyLine({ encoding: 'preserve', input_max_pixels: 0, input_max_side: 0 }),
    'stored byte-for-byte in the original file and format (no input size limit)');
  // ...while an ABSENT key still falls back to the shipped default.
  assert.equal(importInputLimitLine({}), '64 Mi-pixels and 16384 px per side');
  assert.equal(importInputLimitLine({ input_max_side: null }),
    '64 Mi-pixels and 16384 px per side');
});
