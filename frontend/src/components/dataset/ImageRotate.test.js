/**
 * 🔄 90° rotation, dataset AND bank (idea by 1Tomber, GitHub #17).
 *
 * `node --test` cannot parse JSX, so the surfaces are pinned as source
 * contracts: what a reviewer would otherwise have to re-check by hand every
 * time one of these files is rewritten (accessible label, busy guard, the cache
 * busting that stops a stale thumbnail from reading as "the button did
 * nothing").
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const read = (rel) => readFileSync(new URL(rel, import.meta.url), 'utf8');
const hook = read('../../hooks/useDataset.js');
const workspace = read('./DatasetWorkspace.jsx');
const lightbox = read('./DatasetLightbox.jsx');
const gridItem = read('./DatasetGridItem.jsx');
const bankWorkspace = read('../bank/BankWorkspace.jsx');
// The thumbnail cache-buster lives on the TILE now (the Encre redesign
// split the workspace); the rotate handlers stay in the workspace.
const bankTile = read('../bank/BankTile.jsx');
const bankReview = read('../bank/BankReviewLightbox.jsx');

test('the dataset hook rotates once per image and cache-busts only on success', () => {
  assert.ok(hook.includes('`/api/dataset/image/${imageId}/rotate`'));
  const start = hook.indexOf('const rotateImage = useCallback');
  assert.ok(start > 0, 'rotateImage must exist in useDataset');
  const action = hook.slice(start, hook.indexOf('const crop = useCallback'));
  // The SAME busy set as the mirror: both rewrite one file, so one must lock
  // the other out instead of racing it.
  assert.match(action, /mirroringRef\.current\.has\(imageId\)/);
  assert.match(action, /finally[\s\S]*mirroringRef\.current\.delete\(imageId\)/);
  // The nonce bump is inside the success branch, after the refresh.
  assert.match(action, /if \(!d\.ok\)[\s\S]*return false/);
  assert.match(action, /await refresh\(\)[\s\S]*setNonces/);
  assert.match(hook, /mirrorImage, rotateImage, crop/);
});

test('the dataset lightbox offers both directions, labelled and keyboard reachable', () => {
  assert.match(lightbox, /onRotate,/);
  assert.match(lightbox, /const rotate = \(degrees\) => async \(event\)/);
  // Guarded by the same "an edit is running" flag as the mirror.
  assert.match(lightbox, /if \(!onRotate \|\| pixelEditRefused \|\| mirrorBusy\) return/);
  // Real <button>s (focusable, Enter/Space) with an explicit label — the emoji
  // is decoration, never the accessible name.
  // `pixelEditReason ||` in front: while a dataset pass holds the image — or an
  // upscale of it is still rendering — the accessible name becomes the sentence
  // naming WHY, instead of a silently grey button. It is the same reason the
  // `disabled` flag reads (tests/dataset-tile-reads-stay-live.test.mjs).
  assert.match(lightbox, /aria-label=\{pixelEditReason \|\| `Rotate \$\{alt\} 90 degrees left`\}/);
  assert.match(lightbox, /aria-label=\{pixelEditReason \|\| `Rotate \$\{alt\} 90 degrees right`\}/);
  assert.match(lightbox, /onClick=\{rotate\(270\)\}/);
  assert.match(lightbox, /onClick=\{rotate\(90\)\}/);
  assert.ok(lightbox.includes('<span aria-hidden="true">↺</span> Rotate left'));
  // 400 px: the pair shares ONE row instead of stacking two full-width buttons.
  // The bar can now also live in a side rail (lightboxActionPlacement.js), but
  // that placement is unreachable below 1024 px, so the phone branch is still
  // the `w-full sm:w-auto` one and both buttons still flex to share the row.
  assert.match(lightbox, /flex items-stretch gap-2 \$\{rail \? 'w-full' : 'w-full sm:w-auto'\}/);
  assert.match(lightbox, /min-h-9 flex-1 rounded-lg[\s\S]*rail \? '' : 'sm:flex-none'/);
  assert.match(workspace, /onRotate=\{viewImgLive\._rescueReviewPreview \? undefined : ds\.rotateImage\}/);
});

test('the dataset grid tile is deliberately NOT given rotate buttons', () => {
  // The tile already carries regenerate/edit/re-improve/mirror/crop/delete; two
  // more would wrap onto a second row at 400 px and shrink the image itself.
  // Rotation lives one click away, in the lightbox, next to the mirror.
  assert.ok(!/onRotate/.test(gridItem));
});

test('the bank rotates a selection without ever writing to the user folder', () => {
  assert.ok(bankWorkspace.includes('`/api/bank/${bankId}/rotate`'));
  assert.match(bankWorkspace, /const rotateSelection = async \(degrees\)/);
  assert.match(bankWorkspace, /rotateSelection\(-90\)/);
  assert.match(bankWorkspace, /rotateSelection\(90\)/);
  assert.match(bankWorkspace, /aria-label=\{`Rotate the \$\{selected\.size\} selected image\(s\) 90 degrees left`\}/);
  // The promise made in the tooltip is the one the backend keeps.
  assert.match(bankWorkspace, /Your own files are never modified/);
  // A cached thumbnail (max-age=3600) would otherwise keep the old orientation.
  // The key itself moved into bankEdits.imageVersionQuery when the ✂ crop / ✨
  // upscale generation had to ride in it too — a re-crop changes the pixels
  // under an unchanged URL for exactly the same reason a turn does. The property
  // is unchanged and now has its own unit tests (bankEdits.test.js, "a SECOND
  // crop of the same image gets its own URL"); what this line guards is that the
  // tile still asks for a version key at all.
  assert.match(bankTile, /from '\.\/bankEdits\.js'/);
  assert.match(bankTile, /\/thumb\/\$\{img\.id\}\$\{imageVersionQuery\(img\)\}/);
});

test('the bank review lightbox rotates without deciding, and updates the tile behind it', () => {
  assert.match(bankReview, /const rotateCurrent = useCallback\(async \(degrees\)/);
  // Rotating must NOT advance the session — no decide()/skip() in that lane.
  const start = bankReview.indexOf('const rotateCurrent');
  const action = bankReview.slice(start, bankReview.indexOf('// Moving forward without judging'));
  assert.ok(!/setSession/.test(action), 'a rotation must never advance the review');
  assert.match(action, /Math\.abs\(degrees\) % 180 !== 0/);   // half turn ≠ transpose
  assert.match(bankReview, /aria-label="Rotate this image 90 degrees left"/);
  assert.match(bankReview, /aria-label="Rotate this image 90 degrees right"/);
  // Shortcuts that cannot be confused with a decision key.
  assert.match(bankReview, /e\.key === '\[' \) \{ e\.preventDefault\(\); rotateCurrent\(-90\)|e\.key === '\['/);
  assert.match(bankReview, /\[ \] rotate/);
  assert.match(bankWorkspace, /onRotated=\{onReviewRotated\}/);
  assert.match(bankWorkspace, /const onReviewRotated = \(imageId, rotation\)/);
});
