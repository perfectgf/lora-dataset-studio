import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const lightbox = readFileSync(new URL('./WatermarkReviewLightbox.jsx', import.meta.url), 'utf8');
const editor = readFileSync(new URL('./WatermarkRegionEditor.jsx', import.meta.url), 'utf8');
const workspace = readFileSync(new URL('./DatasetWorkspace.jsx', import.meta.url), 'utf8');
const hook = readFileSync(new URL('../../hooks/useDataset.js', import.meta.url), 'utf8');

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

// --- Restore original (undo a clean) --------------------------------------

test('Restore original takes the Clean slot once a real edit ran, calling onRestore', () => {
  assert.match(lightbox, /onRestore/);                    // prop threaded through
  // Gated on a real pixel edit (cleanDetail set), not the "nothing to do" fallback.
  assert.match(lightbox,
    /const restorable = outcome === 'cleaned' && Boolean\(cleanDetail\[item\?\.id\]\)/);
  assert.match(lightbox, /\{restorable \? \(/);           // toggles the primary action button
  assert.match(lightbox, /↩ Restore original/);
  assert.match(lightbox, /await onRestore\(it\.id\)/);
});

test('restore drops the cleaned outcome + detail so the editor returns for a re-clean', () => {
  assert.match(lightbox, /setOutcomes\(\(m\) => \{ const n = \{ \.\.\.m \}; delete n\[it\.id\]; return n; \}\)/);
  assert.match(lightbox, /setCleanDetail\(\(m\) => \{ const n = \{ \.\.\.m \}; delete n\[it\.id\]; return n; \}\)/);
});

test('r triggers restore, mirroring the c/d/x shortcuts', () => {
  assert.match(lightbox, /k === 'r'[\s\S]{0,50}doRestore\(\)/);
});

test('workspace wires onRestore and the hook exposes restoreWatermarkImage', () => {
  assert.match(workspace, /onRestore=\{\(id\) => ds\.restoreWatermarkImage\(id\)\}/);
  assert.match(hook, /const restoreWatermarkImage = useCallback/);
  assert.match(hook, /image\/\$\{imageId\}\/watermark-restore/);
  // Cache-busts the touched thumbnail so the restored original actually shows.
  assert.match(hook, /if \(d\.ok\) \{\s*setNonces/);
});
