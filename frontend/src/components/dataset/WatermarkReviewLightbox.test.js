import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const lightbox = readFileSync(new URL('./WatermarkReviewLightbox.jsx', import.meta.url), 'utf8');
const editor = readFileSync(new URL('./WatermarkRegionEditor.jsx', import.meta.url), 'utf8');

// The review lightbox stacks a fixed image cell over a controls bar. On a short
// mobile viewport a portrait image used to keep its natural height, overflow the
// flex-1 cell, and let the absolutely-positioned region box/handles paint over —
// and steal pointer events from — "Reset detection" and the rest of the bar.
// The fix: the image cell is a size-query container and the media caps its height
// to that cell (100cqh), reserving room for the resize-handle overhang.
const MEDIA_CAP = 'max-h-[min(70vh,calc(100cqh_-_1.5rem))] max-w-[min(92vw,100cqw)]';

test('image cell is a size-query container so the media can fit it', () => {
  assert.match(lightbox, /flex-1 min-h-0 flex items-center justify-center p-3 \[container-type:size\]/);
});

test('every rendered image caps its height to the cell, not just 70vh', () => {
  // plain (cleaning / terminal-outcome) image branch
  assert.ok(lightbox.includes(`block ${MEDIA_CAP} select-none`),
    'plain img must cap height to 100cqh');
  // region editor wrapper + its image
  assert.ok(editor.includes(`relative inline-block ${MEDIA_CAP} leading-none`),
    'editor container must cap height to 100cqh');
  assert.ok(editor.includes(`block ${MEDIA_CAP} select-none`),
    'editor img must cap height to 100cqh');
});

test('no image is left with the old unbounded 70vh-only cap', () => {
  assert.doesNotMatch(lightbox, /max-h-\[70vh\] max-w-\[92vw\]/);
  assert.doesNotMatch(editor, /max-h-\[70vh\] max-w-\[92vw\]/);
});
