/**
 * The tile of an ✨ Upscale & improve result had NO regenerate at all: the generic
 * 🔄 is hidden on purpose (that route restarts from the dataset reference and would
 * make an unrelated image), and nothing replaced it — so re-running the pass after
 * tuning the improve settings meant deleting the tile and clicking ✨ on the parent.
 *
 * These tests pin both halves of the fix: the original guard stays closed, and the
 * correct action (re-run from the parent) is offered in its place.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { canRegenerateGeneric, improveRerunAffordance,
  REIMPROVE_TITLE, REIMPROVE_NO_PARENT_TITLE } from './improveRerun.js';

const here = dirname(fileURLToPath(import.meta.url));
const tile = readFileSync(join(here, 'DatasetGridItem.jsx'), 'utf8');

const improved = (over = {}) => ({
  id: 7, source: 'generated', status: 'pending', filename: 'improved.png',
  derivation_kind: 'klein_image_improve', parent_image_id: 3, ...over,
});

test('an improvement result gets a re-run, pointing at its parent', () => {
  const state = improveRerunAffordance(improved());
  assert.ok(state, 'the tile must offer something — it used to offer nothing');
  assert.equal(state.enabled, true);
  assert.match(state.title, /current improve settings/);
  assert.equal(state.title, REIMPROVE_TITLE);
});

test('a result improved by an OLDER version is offered it too', () => {
  // Rows written before this feature carry parent_image_id + derivation_kind and
  // nothing else — that is all the affordance needs.
  const legacy = { id: 9, source: 'generated', status: 'keep', filename: 'old.webp',
    derivation_kind: 'klein_image_improve', parent_image_id: 4 };
  assert.equal(improveRerunAffordance(legacy).enabled, true);
});

test('a dangling parent disables the button and says why (no dead affordance)', () => {
  const state = improveRerunAffordance(improved({ parent_image_id: null }));
  assert.equal(state.enabled, false);
  assert.equal(state.title, REIMPROVE_NO_PARENT_TITLE);
  assert.match(state.title, /deleted/);
});

test('nothing is offered while the pass is still generating', () => {
  assert.equal(improveRerunAffordance(improved({ filename: null, status: 'pending' })), null);
});

test('a plain tile is not an improvement — no re-run button', () => {
  assert.equal(improveRerunAffordance({ id: 1, source: 'generated', filename: 'a.png' }), null);
  assert.equal(improveRerunAffordance({ id: 2, source: 'import', filename: 'b.png',
    derivation_kind: 'klein_small_image', parent_image_id: 1 }), null);
});

test('the generic regenerate route stays closed on derived rows', () => {
  // This is the guard the feature must NOT weaken: it exists because the generic
  // route restarts from the dataset reference.
  assert.equal(canRegenerateGeneric(improved()), false);
  assert.equal(canRegenerateGeneric({ source: 'generated', filename: 'a.png', status: 'keep' },
    { isRescueDerived: true }), false);
  // ...and still opens for an ordinary generated tile, including a failed one.
  assert.equal(canRegenerateGeneric({ source: 'generated', filename: 'a.png', status: 'keep' }), true);
  assert.equal(canRegenerateGeneric({ source: 'generated', filename: null, status: 'failed' }), true);
  assert.equal(canRegenerateGeneric({ source: 'import', filename: 'a.png', status: 'keep' }), false);
});

test('the tile renders a real button with an aria-label and calls onReimprove', () => {
  // Contract by grep: node --test cannot parse JSX, and this wiring is exactly what
  // a rewrite of the component would silently drop.
  assert.match(tile, /improveRerunAffordance/);
  assert.match(tile, /onReimprove\?\.\(img\.id\)/);
  // The accessible name falls back to `rerunImprove.title` and is REPLACED, while
  // a dataset pass holds the tile, by the sentence naming that pass — a refused
  // write must never be mute (tests/dataset-tile-reads-stay-live.test.mjs).
  // Both read `queueRefused`, not `busy`: re-improving adds a row to the
  // serialized image queue, so a generation batch running elsewhere no longer
  // refuses it (GitHub #44, utils/activityLanes.js). What DOES refuse it still
  // says so out loud — the accessible name carries the sentence.
  assert.match(tile, /aria-label=\{queueRefusedReason \|\| rerunImprove\.title\}/);
  assert.match(tile, /disabled=\{queueRefused \|\| !rerunImprove\.enabled\}/);
  // The tile must never hand an improvement to the generic route.
  assert.match(tile, /canRegenerateGeneric\(img, \{ isRescueDerived \}\)/);
});
