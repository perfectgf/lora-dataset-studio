import test from 'node:test';
import assert from 'node:assert/strict';
import {
  BANK_IMPROVE_PROMISE, bankImproveEngines, cropOutcomeMessage, editBadge,
  editCounts, editSummary, imageVersionQuery, revertConfirmMessage,
  revertOutcomeMessage,
} from './bankEdits.js';

/* --- the cache key, which is the whole reason an edit is VISIBLE ----------- */

test('imageVersionQuery: an untouched image asks for no cache key at all', () => {
  assert.equal(imageVersionQuery({ id: 1 }), '');
  assert.equal(imageVersionQuery(null), '');
  // 0 is "no edit" and "no turn", not a version worth busting the cache for.
  assert.equal(imageVersionQuery({ rotation: 0, edit_generation: 0 }), '');
});

test('imageVersionQuery: a SECOND crop of the same image gets its own URL', () => {
  // The bug this closes: keying on `edit_method` alone would leave both crops at
  // ?e=crop, and the thumb route's one-hour cache would serve the first one for
  // an hour — a re-crop that reads as "the button did nothing".
  const first = imageVersionQuery({ edit_method: 'crop', edit_generation: 1 });
  const second = imageVersionQuery({ edit_method: 'crop', edit_generation: 2 });
  assert.equal(first, '?e=1');
  assert.equal(second, '?e=2');
  assert.notEqual(first, second);
});

test('imageVersionQuery: a turn made ON TOP of an edit keeps both keys', () => {
  assert.equal(imageVersionQuery({ rotation: 90, edit_generation: 3 }), '?r=90&e=3');
});

/* --- what the grid says about an edited image ------------------------------ */

test('editBadge: only an edited image gets one, and it names the edit', () => {
  assert.equal(editBadge({ id: 1 }), null);
  assert.equal(editBadge({ edit_method: 'crop' }).text, '✂');
  assert.equal(editBadge({ edit_method: 'improve' }).text, '✨');
  // The promise the whole feature rests on is in the tooltip of both.
  for (const m of ['crop', 'improve']) {
    assert.match(editBadge({ edit_method: m }).title, /not modified/i);
    assert.match(editBadge({ edit_method: m }).title, /Revert/);
  }
});

const payload = (cropped, improved) => ({ counts: { cropped, improved } });

test('editCounts / editSummary: the two edits are counted apart', () => {
  // The tallies live under payload.counts, where the server puts every per-bank
  // figure — reading them off the root is how this panel would have said
  // "nothing edited yet" on a bank full of crops (caught by
  // backend/tests/test_bank_crop_and_improve.py).
  assert.deepEqual(editCounts(payload(12, 4)), { cropped: 12, improved: 4, total: 16 });
  assert.deepEqual(editCounts({ cropped: 12, improved: 4 }),
    { cropped: 0, improved: 0, total: 0 });
  assert.deepEqual(editCounts(null), { cropped: 0, improved: 0, total: 0 });
  assert.equal(editSummary(payload(12, 4)), '12 cropped · 4 improved');
  // A hand crop must never be able to hide a run of GPU-minutes, and vice versa.
  assert.equal(editSummary(payload(12, 0)), '12 cropped');
  assert.equal(editSummary(payload(0, 4)), '4 improved');
  assert.equal(editSummary({}), 'nothing edited yet');
});

/* --- the ✨ engine buttons ------------------------------------------------- */

const READY = { comfyui: { seedvr2_ready: true } };

test('bankImproveEngines: both engines when the install has both', () => {
  const rows = bankImproveEngines(READY, { todo: 10 });
  assert.deepEqual(rows.map((r) => r.id), ['klein', 'seedvr2']);
  assert.ok(rows.every((r) => !r.disabled));
});

test('bankImproveEngines: SeedVR2 absent until it is installed, Klein always shown', () => {
  // Shared rule with the dataset lane: an engine that is not installed is a
  // SETUP task, not a choice — but Klein stays visible with its reason.
  const rows = bankImproveEngines({ comfyui: {} }, { todo: 10, engines: { klein: false } });
  assert.deepEqual(rows.map((r) => r.id), ['klein']);
  assert.ok(rows[0].disabled);
  assert.match(rows[0].reason, /not available/i);
});

test('bankImproveEngines: a running pass and an empty pool say WHICH one it is', () => {
  const live = bankImproveEngines(READY, { todo: 10, live: true });
  assert.ok(live.every((r) => r.disabled));
  assert.match(live[0].reason, /already running/i);

  const done = bankImproveEngines(READY, { todo: 0 });
  assert.ok(done.every((r) => r.disabled));
  assert.match(done[0].reason, /already been improved/i);
});

test('bankImproveEngines: "not counted yet" is not "nothing to do"', () => {
  // The payload arrives after the panel renders. Treating an absent count as
  // zero would disable the pass with a sentence claiming the bank is already
  // fully improved — a false statement, shown on first paint every time.
  const loading = bankImproveEngines(READY, {});
  assert.ok(loading.every((r) => !r.disabled));
});

test('the bank promise is NOT the dataset promise', () => {
  // A dataset improve makes a separate candidate to review; a bank improve
  // REPLACES what the bank shows. Promising a candidate here would be false
  // about the one thing worth knowing before spending GPU-minutes.
  assert.match(BANK_IMPROVE_PROMISE, /replaces/i);
  assert.doesNotMatch(BANK_IMPROVE_PROMISE, /candidate/i);
  assert.match(BANK_IMPROVE_PROMISE, /never modified/i);
  const enabled = bankImproveEngines(READY, { todo: 3 })[0];
  assert.ok(enabled.title.includes(BANK_IMPROVE_PROMISE));
});

/* --- what the actions report ---------------------------------------------- */

test('cropOutcomeMessage: quotes the new size and the re-analysis', () => {
  const msg = cropOutcomeMessage({ width: 1200, height: 800 });
  assert.match(msg, /1200×800/);
  assert.match(msg, /not modified/i);
  assert.match(msg, /re-reads/i);
  // A row whose size could not be measured still gets an honest sentence.
  assert.doesNotMatch(cropOutcomeMessage({}), /×/);
});

test('revertConfirmMessage: bank-wide and per-selection are different promises', () => {
  const all = revertConfirmMessage(payload(10, 2));
  assert.match(all, /all 12 image/);
  const some = revertConfirmMessage(payload(10, 2), [1, 2, 3]);
  assert.match(some, /3 selected/);
  for (const m of [all, some]) {
    assert.match(m, /never modified/i);
    // The passes have to run again over those rows — said before the click.
    assert.match(m, /cover those images again/i);
  }
});

test('revertOutcomeMessage: zero is reported as zero, not as a green success', () => {
  assert.equal(revertOutcomeMessage({ reverted: 0 }).type, 'info');
  assert.match(revertOutcomeMessage({ reverted: 0 }).text, /Nothing to revert/);
  const ok = revertOutcomeMessage({ reverted: 7 });
  assert.equal(ok.type, 'success');
  assert.match(ok.text, /7 image/);
});
