/* ✂ Auto-crop and 🧽 Repaint are launched from a window with a scope, like every
 * other bank pass — and, being the only two that produce new image files, they
 * say what is reversible before they run.
 *
 * The panel itself is JSX (a worktree has no node_modules, so it cannot be
 * mounted here); what is asserted is the CONTRACT the panel depends on — the two
 * specs, the reachability of the bin, and the wiring read out of the source.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { BANK_PASSES, BANK_PASS_ORDER, passScopeRows, passSelectionAvailability }
  from './bankPasses.js';
import {
  passLaunchDisabledReason, passLaunchLabel, passScopeCount, passScopeLineLabel,
  passScopeStatuses, PASS_SCOPE_OPTIONS,
} from './bankPassScope.js';
import { cropLevelState, inpaintLevelState } from './bankWatermark.js';

const PANEL = readFileSync(new URL('./BankWatermarkPanel.jsx', import.meta.url), 'utf8');
const LEVELS = ['watermark_crop', 'watermark_inpaint'];

/* A payload shaped like the server's: per pass, per pile, the pool the run
   walks. 4 kept + 3 undecided + 9 rejected croppable images. */
const payload = {
  pass_scopes: {
    watermark_crop: {
      todo: { keep: 4, pending: 3, reject: 9 },
      all: { keep: 4, pending: 3, reject: 9 },
    },
    watermark_inpaint: {
      todo: { keep: 5, pending: 3, reject: 9 },
      all: { keep: 5, pending: 3, reject: 9 },
    },
  },
};

test('both cleaning levels offer the four scopes and a selection', () => {
  for (const id of LEVELS) {
    const spec = BANK_PASSES[id];
    assert.ok(spec, `${id} has no spec`);
    assert.equal(spec.scopes, true, `${id} must offer every scope`);
    assert.equal(passSelectionAvailability(id).ok, true);
    const rows = passScopeRows(id);
    assert.equal(rows.length, PASS_SCOPE_OPTIONS.length);
    assert.ok(rows.every((r) => r.ok), `${id} disables a scope with no reason to`);
  }
});

test('the default scope sends no statuses at all', () => {
  // The byte-identical contract: leaving the window alone must post the body the
  // button posted before the window existed.
  assert.equal(passScopeStatuses(''), null);
  assert.deepEqual(passScopeStatuses('keep'), ['keep']);
});

test('each scope line quotes the pool that scope really walks', () => {
  assert.equal(passScopeCount(payload, 'watermark_crop', ''), 7);      // keep+pending
  assert.equal(passScopeCount(payload, 'watermark_crop', 'keep'), 4);
  assert.equal(passScopeCount(payload, 'watermark_crop', 'reject'), 9);
  assert.equal(passScopeCount(payload, 'watermark_crop', 'all'), 16);
  assert.match(passScopeLineLabel(payload, 'watermark_inpaint', 'pending'),
    /Undecided only — 3 images/);
  assert.equal(passLaunchLabel({
    verb: BANK_PASSES.watermark_inpaint.verb, payload, passId: 'watermark_inpaint',
    scopeId: 'keep',
  }), 'Repaint 5 images');
});

test('a scope with nothing in it refuses to launch instead of reporting success', () => {
  const empty = { pass_scopes: { watermark_crop: { todo: { keep: 0, pending: 0, reject: 9 } } } };
  assert.match(
    passLaunchDisabledReason({ payload: empty, passId: 'watermark_crop', scopeId: 'keep' }),
    /Nothing to do in this scope/);
  assert.equal(
    passLaunchDisabledReason({ payload: empty, passId: 'watermark_crop', scopeId: 'reject' }),
    '');
});

test('both levels state what is reversible, and do not overpromise', () => {
  for (const id of LEVELS) {
    const text = BANK_PASSES[id].caveats.join(' ');
    assert.match(text, /Undo cleaning/, `${id} never names the undo`);
    assert.match(text, /never written to/, `${id} never says the source is safe`);
    // The honest limits: undo deletes OUR blob, so a copy already promoted into
    // a dataset is out of its reach, and a row whose raw file moved keeps its
    // clean. Promising "fully reversible" would be a lie.
    assert.match(text, /promoted/, `${id} promises an undo wider than the one shipped`);
    assert.match(text, /changed on disk/, `${id} hides the fingerprint limit of undo`);
    assert.match(text, /bank-wide/, `${id} lets undo read as per-run`);
  }
});

test('the crop window does not pass its pool off as its result', () => {
  // Its count is the pool it walks; the router decides per image whether a crop
  // is possible. Saying so is the whole difference between a number and a promise.
  assert.match(BANK_PASSES.watermark_crop.caveats.join(' '), /not what it will\s+change/);
});

test('reaching the bin is priced, not silently free', () => {
  for (const id of LEVELS) {
    assert.ok(BANK_PASSES[id].binCost, `${id} offers the bin with no stated cost`);
  }
});

test('neither level joins the pass button row — their buttons are on the panel', () => {
  for (const id of LEVELS) assert.ok(!BANK_PASS_ORDER.includes(id));
});

/* --- the bin has to be REACHABLE, or the scope is decoration --------------- */
test('a level with work only in the bin stays clickable and says so', () => {
  const levels = { scanned: 50, flagged: 0, croppable: 0, cropped: 3 };
  const off = cropLevelState(levels, {});
  assert.equal(off.disabled, true);                 // today's behaviour, unchanged
  const on = cropLevelState(levels, { binWaiting: 9 });
  assert.equal(on.disabled, false);
  assert.match(on.label, /9 in the bin/);

  const ip = inpaintLevelState(levels, { lamaReady: true, binWaiting: 9 });
  assert.equal(ip.disabled, false);
  assert.match(ip.label, /9 in the bin/);
});

test('the line under a bin-only button does not contradict the button', () => {
  // "0 image(s) waiting" under a live "✂ Auto-crop (2 in the bin)" is the card
  // arguing with itself — the exact class of defect these counts exist to end.
  const levels = { scanned: 50, flagged: 0, croppable: 0, cropped: 7 };
  assert.match(cropLevelState(levels, { binWaiting: 2 }).note, /2 image\(s\) waiting/);
  assert.match(inpaintLevelState(levels, { lamaReady: true, binWaiting: 2 }).note,
    /Unkept only/);
  // Nothing in the bin: no note at all, so the ordinary line shows.
  assert.equal(cropLevelState(levels, {}).note, '');
  assert.equal(cropLevelState({ croppable: 4 }, { binWaiting: 2 }).note, '');
  // A disabled level must not advertise the bin under an engine refusal — the
  // reason wins, and the note would be unreachable advice.
  assert.equal(inpaintLevelState(levels, { lamaReady: false, binWaiting: 2 }).note, '');
});

test('a missing engine still wins over the bin — it is a different objection', () => {
  const ip = inpaintLevelState({ flagged: 0 }, { lamaReady: false, binWaiting: 9 });
  assert.equal(ip.disabled, true);
  assert.match(ip.reason, /LaMa/);
});

test('a running pass still wins over everything', () => {
  assert.equal(cropLevelState({ croppable: 0 }, { live: true, binWaiting: 9 }).disabled, true);
  assert.equal(inpaintLevelState({ flagged: 0 },
    { live: true, lamaReady: true, binWaiting: 9 }).disabled, true);
});

test('with work in the default scope the label stays the routed figure', () => {
  // binWaiting must never be ADDED to croppable: one is routed, the other is not.
  assert.match(cropLevelState({ croppable: 4 }, { binWaiting: 9 }).label, /\(4\)/);
  assert.match(inpaintLevelState({ flagged: 4 },
    { lamaReady: true, binWaiting: 9 }).label, /\(4\)/);
});

/* --- the panel wiring, read out of the source ------------------------------ */
test('the two level buttons open the window instead of firing a POST', () => {
  for (const id of LEVELS) {
    assert.ok(PANEL.includes(`setCleanOpen('${id}')`), `${id} button does not open a window`);
  }
  // The old straight-to-POST calls are gone.
  assert.ok(!PANEL.includes('run(`/api/bank/${bankId}/watermark/crop`'),
    'the crop button still fires without a window');
  assert.ok(!PANEL.includes('run(`/api/bank/${bankId}/watermark/inpaint`'),
    'the inpaint button still fires without a window');
});

test('the launch spreads statuses/image_ids only when set', () => {
  assert.ok(PANEL.includes('...(statuses ? { statuses } : {})'));
  assert.ok(PANEL.includes("...(imageIds === 'selection' && selectedIds.length"));
  // The engine rides along on the repaint only — the crop has none.
  assert.ok(PANEL.includes("cleanOpen === 'watermark_inpaint' ? { method } : {}"));
});

test('the bin figure comes from the payload table, not from a second predicate', () => {
  assert.ok(PANEL.includes('payload?.pass_scopes?.watermark_crop?.todo?.reject'));
  assert.ok(PANEL.includes('payload?.pass_scopes?.watermark_inpaint?.todo?.reject'));
});
