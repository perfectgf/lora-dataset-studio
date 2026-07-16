import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const hook = readFileSync(new URL('../../hooks/useDataset.js', import.meta.url), 'utf8');
const workspace = readFileSync(new URL('./DatasetWorkspace.jsx', import.meta.url), 'utf8');
const grid = readFileSync(new URL('./DatasetGrid.jsx', import.meta.url), 'utf8');
const gridItem = readFileSync(new URL('./DatasetGridItem.jsx', import.meta.url), 'utf8');
const lightbox = readFileSync(new URL('./DatasetLightbox.jsx', import.meta.url), 'utf8');

test('dataset hook mirrors once, refreshes, and cache-busts only after success', () => {
  assert.match(hook, /const \[mirroringIds, setMirroringIds\] = useState\(\(\) => new Set\(\)\)/);
  assert.match(hook, /const mirroringRef = useRef\(new Set\(\)\)/);
  assert.ok(hook.includes('`/api/dataset/image/${imageId}/mirror`'));

  const start = hook.indexOf('const mirrorImage = useCallback');
  const end = hook.indexOf('const crop = useCallback', start);
  const action = hook.slice(start, end);
  assert.match(action, /mirroringRef\.current\.has\(imageId\)/);
  assert.match(action, /if \(!d\.ok\)[\s\S]*return false;[\s\S]*await refresh\(\);[\s\S]*setNonces/);
  assert.match(action, /finally[\s\S]*mirroringRef\.current\.delete\(imageId\)/);
  assert.match(hook, /nonces, mirroringIds, refNonce/);
  assert.match(hook, /setStatus, setCaption, mirrorImage, crop/);
});

test('workspace wires mirror actions and keeps rescue previews read-only', () => {
  assert.match(workspace, /onMirror=\{ds\.mirrorImage\} mirroringIds=\{ds\.mirroringIds\}/);
  assert.match(workspace, /onMirror=\{viewImgLive\._rescueReviewPreview \? undefined : ds\.mirrorImage\}/);
  assert.match(workspace, /mirrorBusy=\{Boolean\(ds\.mirroringIds\?\.has\(viewImgLive\.id\)\)\}/);
});

test('grid exposes an accessible per-image mirror action with busy protection', () => {
  assert.match(grid, /onMirror, onRegenerate/);
  assert.match(grid, /mirrorBusy=\{Boolean\(mirroringIds\?\.has\(img\.id\)\)\} busy=\{busy\}/);
  assert.match(gridItem, /\{url && onMirror && \(/);
  assert.match(gridItem, /disabled=\{busy \|\| mirrorBusy\}/);
  assert.match(gridItem, /aria-busy=\{mirrorBusy\}/);
  assert.match(gridItem, /Mirror \$\{displayLabel\(img\.variation_label\)/);
  assert.match(gridItem, /e\.stopPropagation\(\); onMirror\(img\.id\)/);
  assert.match(gridItem, /min-h-7 min-w-7/);
  assert.match(gridItem, /flex-wrap justify-end/);
});

test('lightbox mirrors without closing and remains touch friendly on mobile', () => {
  assert.match(lightbox, /const mirror = async \(event\)/);
  assert.match(lightbox, /event\.stopPropagation\(\)/);
  assert.match(lightbox, /if \(!onMirror \|\| busy \|\| mirrorBusy\) return/);
  assert.match(lightbox, /aria-busy=\{mirrorBusy\}/);
  assert.match(lightbox, /⇆ Mirror horizontally/);
  assert.match(lightbox, /min-h-9 w-full sm:w-auto/);
});

test('crop editor uses the per-image nonce after an in-place mirror', () => {
  assert.match(workspace, /ds\.nonces\?\.\[cropImg\.id\][\s\S]*\?v=\$\{ds\.nonces\[cropImg\.id\]\}/);
});
