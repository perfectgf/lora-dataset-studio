/* 🖼 The run gallery's JSX contract — read as text, because `node --test` cannot
   parse JSX and every guarantee below is a rewrite away from being lost without
   anything throwing:

     • the board opens the RUN panel on a card click, through the tested
       decision helper (so the drag guard cannot be bypassed by an inline `&&`);
     • the pill click still opens its CHECKPOINT gallery — the new gesture is an
       addition, not a replacement;
     • there is exactly ONE gallery panel component. A second one would be a
       second grid over the same rows and a second delete to keep honest;
     • the run panel groups, folds, and states its notes, its settings and what
       it is not showing.
*/
import assert from 'node:assert/strict';
import { readSource } from './support/readSource.mjs'
import fs from 'node:fs';
import test from 'node:test';

const root = new URL('../src/', import.meta.url);
const read = (rel) => readSource(`src/${rel}`)

const canvas = read('components/canvas/LineageCanvas.jsx');
const panel = read('components/shared/CheckpointGalleryPanel.jsx');

test('a run card click opens the run gallery, via the tested decision helper', () => {
  assert.match(canvas, /from '\.\.\/\.\.\/utils\/canvasCardClick'/);
  assert.match(canvas, /cardClickAction\(\{\s*dragged/);
  assert.match(canvas, /setGallery\(runGalleryTarget\(/);
  // The drag guard is READ from the same ref the drop sets — not re-derived.
  assert.match(canvas, /const dragged = suppressClick\.current/);
});

test('the card gesture answers for itself — a captured pointer eats the click', () => {
  /* THE bug behind "clicking a run does nothing": onPointerDown captures the
     pointer on the FRAME (a drag leaving the frame must keep receiving moves),
     and a captured pointer retargets the following click to the capturing
     element. The card's own onClick therefore never fires on this board. Pinned
     because it is invisible in every unit test and looks like nothing at all in
     the UI — the exact failure that shipped. */
  assert.match(canvas, /const press = useRef\(null\)/);
  assert.match(canvas, /press\.current = \{ datasetId:/);
  // Travel past the slop demotes the press from a click to a drag/pan…
  assert.match(canvas, /press\.current\.moved = true/);
  // …and only an untravelled release opens anything.
  assert.match(canvas, /if \(p && !p\.moved && !d\?\.moved\)/);
  assert.match(canvas, /runCardGesture\(node, e\.shiftKey\)/);
  // A second finger is a pinch, never a click.
  assert.match(canvas, /press\.current = null;\s*\/\/ a second finger/);
  // One press, one open: the DOM click that may also arrive is ignored.
  assert.match(canvas, /Date\.now\(\) - answeredAt\.current < 400/);
});

test('clicking a checkpoint pill still opens that checkpoint, unchanged', () => {
  // Hoisted out of the JSX so the lanes' memo boundary survives a pan — the
  // wiring is the same, its declaration just moved (see LineageCanvas).
  assert.match(canvas, /const openGallery = useCallback\(\(recordId, step\) => setGallery\(\{ recordId, step \}\), \[\]\)/);
  assert.match(canvas, /onOpenGallery=\{openGallery\}/);
  // …and a pill press is still never a drag or a pan.
  assert.match(canvas, /closest\?\.\('\.lds-ckpill-wrap'\)/);
});

test('there is exactly one gallery panel, hosting both scopes', () => {
  assert.ok(!fs.existsSync(new URL('components/canvas/RunGalleryPanel.jsx', root)));
  assert.ok(!fs.existsSync(new URL('components/shared/RunGalleryPanel.jsx', root)));
  assert.match(panel, /from '\.\.\/\.\.\/utils\/runGallery'/);
  // Both endpoints come from the shared resolver — never built inline, which is
  // how a panel ends up reading one scope and deleting in another.
  assert.match(panel, /galleryEndpoints\(target\)/);
  assert.doesNotMatch(panel, /`\/api\/train\/(run|checkpoint)\//);
});

test('the run panel groups by step, folds, and says when it cut', () => {
  assert.match(panel, /data-testid="run-gallery-group"/);
  assert.match(panel, /data-testid="run-gallery-group-toggle"/);
  assert.match(panel, /aria-expanded=\{open\}/);
  assert.match(panel, /defaultOpenGroups\(/);
  assert.match(panel, /g\.truncated &&/);
  assert.match(panel, /runGallerySummary\(d\)/);
});

test('the run panel shows the notes and the training settings', () => {
  assert.match(panel, /data-testid="run-gallery-notes"/);
  assert.match(panel, /data-testid="run-gallery-settings"/);
  assert.match(panel, /configRows\(node\?\.config\)/);
  // Notes come from the PAYLOAD, so a run whose saves left the disk keeps them.
  assert.match(panel, /checkpointNotes\(d, node\)/);
  // A run that never recorded its settings says so instead of showing nothing.
  assert.match(panel, /did not record its settings/);
});

test('the honest "not traced back" footnote is still rendered', () => {
  assert.match(panel, /unlinkedNote\(d\.unlinked, scope\)/);
});

test('deletion stays unreachable without Select mode, at both scopes', () => {
  // One Select gate, one confirmation, one delete call — shared by both scopes.
  assert.equal((panel.match(/data-testid="gallery-select-toggle"/g) || []).length, 1);
  assert.equal((panel.match(/data-testid="gallery-confirm-delete"/g) || []).length, 1);
  // Counted against the REMOVE endpoint, not every postJson in the file: the
  // panel also posts non-destructive actions (📂 Open folder), and what this
  // contract protects is that deletion has exactly one path.
  assert.equal((panel.match(/postJson\(endpoints\.remove/g) || []).length, 1);
  assert.match(panel, /picking \? 'gallery-pick' : 'gallery-zoom'/);
});
